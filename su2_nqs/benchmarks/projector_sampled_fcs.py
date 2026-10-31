"""Sample block-spin FCS without dense subsystem projectors.

Two complementary readout routes are implemented.

``computational-z``
    For a Born sample ``x``, evaluate the usual VMC local estimator
    ``(P_q psi)(x) / psi(x)``.  The block-spin projector is applied as a
    polynomial in ``S_A^2`` and swaps, so neither a dense projector nor a
    full-system wavefunction is materialized by this module.

``tree-native``
    If the state is generated on a fusion tree that fuses every measured
    group first, its subtree-root labels are the FCS outcome.  A histogram is
    then sufficient and no amplitude ratios are needed.

The computational route is exact but its connectivity is the product of the
within-block fixed-magnetization orbit sizes.  It is therefore a validation
bridge, not an asymptotically scalable substitute for tree-native sampling.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from itertools import product
import math
from typing import Any

import numpy as np

from ..su2.fusion_tree import FusionTreeState
from .block_spin_fcs import FCSKey


LogAmplitudeBatch = Callable[[np.ndarray], np.ndarray]


def allowed_block_spins(n_spins: int) -> tuple[int, ...]:
    """Return allowed doubled total spins for ``n_spins`` spin-1/2 sites."""

    n_spins = int(n_spins)
    if n_spins < 1:
        raise ValueError("n_spins must be positive")
    return tuple(range(n_spins % 2, n_spins + 1, 2))


def _validated_bits(bits: Sequence[int]) -> tuple[int, ...]:
    answer = tuple(int(bit) for bit in bits)
    if not answer or any(bit not in (0, 1) for bit in answer):
        raise ValueError("a configuration must be a nonempty binary sequence")
    return answer


def _validated_groups(
    n_sites: int,
    groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    answer = tuple(tuple(int(site) for site in group) for group in groups)
    if not answer or any(not group for group in answer):
        raise ValueError("groups must be nonempty")
    flat = tuple(site for group in answer for site in group)
    if len(set(flat)) != len(flat):
        raise ValueError("measured groups must be disjoint")
    if any(not 0 <= site < n_sites for site in flat):
        raise ValueError("a group contains a site outside the system")
    return answer


def _pattern_from_bits(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _bits_from_pattern(pattern: int, n_spins: int) -> tuple[int, ...]:
    return tuple((int(pattern) >> (n_spins - index - 1)) & 1 for index in range(n_spins))


def _swap_pattern(pattern: int, n_spins: int, first: int, second: int) -> int:
    first_mask = 1 << (n_spins - first - 1)
    second_mask = 1 << (n_spins - second - 1)
    if bool(pattern & first_mask) == bool(pattern & second_mask):
        return pattern
    return pattern ^ first_mask ^ second_mask


def _apply_spin_squared(
    vector: Mapping[int, float],
    n_spins: int,
) -> dict[int, float]:
    """Apply ``S_A^2`` through its exact spin-1/2 swap identity."""

    result: dict[int, float] = {}
    constant = n_spins * (4 - n_spins) / 4.0
    for pattern, coefficient in vector.items():
        result[pattern] = result.get(pattern, 0.0) + constant * coefficient
        for first in range(n_spins):
            for second in range(first + 1, n_spins):
                swapped = _swap_pattern(pattern, n_spins, first, second)
                result[swapped] = result.get(swapped, 0.0) + coefficient
    return result


@lru_cache(maxsize=None)
def block_spin_projector_action(
    n_spins: int,
    target_q: int,
    input_pattern: int,
) -> tuple[tuple[int, float], ...]:
    """Return sparse ``P_target_q |input_pattern>`` connectivity.

    Doubled spins are used throughout.  Local sites are packed with the first
    site as the most significant bit, matching the repository convention.
    """

    n_spins = int(n_spins)
    target_q = int(target_q)
    input_pattern = int(input_pattern)
    allowed = allowed_block_spins(n_spins)
    if target_q not in allowed:
        raise ValueError("target_q is not allowed for this block size")
    if not 0 <= input_pattern < 2**n_spins:
        raise ValueError("input_pattern lies outside the block Hilbert space")

    target_eigenvalue = target_q * (target_q + 2) / 4.0
    vector: dict[int, float] = {input_pattern: 1.0}
    for other_q in allowed:
        if other_q == target_q:
            continue
        other_eigenvalue = other_q * (other_q + 2) / 4.0
        denominator = target_eigenvalue - other_eigenvalue
        applied = _apply_spin_squared(vector, n_spins)
        keys = set(applied) | set(vector)
        vector = {
            pattern: (
                applied.get(pattern, 0.0)
                - other_eigenvalue * vector.get(pattern, 0.0)
            )
            / denominator
            for pattern in keys
        }
    tolerance = 5.0e-15
    return tuple(
        sorted(
            (pattern, float(coefficient))
            for pattern, coefficient in vector.items()
            if abs(coefficient) > tolerance
        )
    )


def fcs_outcomes(groups: Sequence[Sequence[int]]) -> tuple[FCSKey, ...]:
    """Return the fixed Cartesian outcome support implied by group sizes."""

    sizes = tuple(len(tuple(group)) for group in groups)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("groups must be nonempty")
    return tuple(product(*(allowed_block_spins(size) for size in sizes)))


def joint_projector_connections(
    configuration: Sequence[int],
    groups: Sequence[Sequence[int]],
    outcome: Sequence[int],
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Apply a product of disjoint block-spin projectors to one bitstring."""

    bits = _validated_bits(configuration)
    checked = _validated_groups(len(bits), groups)
    key = tuple(int(q) for q in outcome)
    if len(key) != len(checked):
        raise ValueError("outcome arity must match the number of groups")

    local_actions = []
    for group, target_q in zip(checked, key, strict=True):
        pattern = _pattern_from_bits(tuple(bits[site] for site in group))
        local_actions.append(
            tuple(
                (_bits_from_pattern(output, len(group)), coefficient)
                for output, coefficient in block_spin_projector_action(
                    len(group), target_q, pattern
                )
            )
        )

    connections: dict[tuple[int, ...], float] = {}
    for choices in product(*local_actions):
        candidate = list(bits)
        coefficient = 1.0
        for group, (local_bits, local_coefficient) in zip(
            checked, choices, strict=True
        ):
            coefficient *= local_coefficient
            for site, bit in zip(group, local_bits, strict=True):
                candidate[site] = bit
        target = tuple(candidate)
        connections[target] = connections.get(target, 0.0) + coefficient
    return tuple(
        sorted(
            (target, float(coefficient))
            for target, coefficient in connections.items()
            if abs(coefficient) > 5.0e-14
        )
    )


def projector_local_values(
    configuration: Sequence[int],
    log_amplitude_batch: LogAmplitudeBatch,
    groups: Sequence[Sequence[int]],
    outcomes: Sequence[FCSKey] | None = None,
    *,
    complete_last_outcome: bool = False,
) -> tuple[dict[FCSKey, complex], dict[str, float | int]]:
    """Evaluate all requested joint-projector local estimators on one sample."""

    bits = _validated_bits(configuration)
    checked = _validated_groups(len(bits), groups)
    keys = tuple(fcs_outcomes(checked) if outcomes is None else outcomes)
    if not keys or any(len(key) != len(checked) for key in keys):
        raise ValueError("outcomes must be nonempty and match the group arity")
    if len(set(keys)) != len(keys):
        raise ValueError("outcomes must not contain duplicates")
    if complete_last_outcome and keys != fcs_outcomes(checked):
        raise ValueError(
            "completeness reconstruction requires the canonical exhaustive support"
        )

    evaluated_keys = keys[:-1] if complete_last_outcome else keys
    connections = {
        tuple(key): joint_projector_connections(bits, checked, key)
        for key in evaluated_keys
    }
    candidates = {target for row in connections.values() for target, _ in row}
    ordered = (bits,) + tuple(sorted(candidates - {bits}))
    logs = np.asarray(
        log_amplitude_batch(np.asarray(ordered, dtype=np.int8)),
        dtype=np.complex128,
    )
    if logs.shape != (len(ordered),):
        raise ValueError("log_amplitude_batch must return one scalar per configuration")
    if not np.isfinite(logs[0]):
        raise ValueError("the sampled configuration must have finite log amplitude")
    index = {candidate: position for position, candidate in enumerate(ordered)}
    ratios = np.exp(logs - logs[0])
    if np.any(~np.isfinite(ratios)):
        raise FloatingPointError("a connected amplitude ratio is not finite")

    values = {}
    for key, row in connections.items():
        terms = [coefficient * ratios[index[target]] for target, coefficient in row]
        values[key] = complex(
            math.fsum(term.real for term in terms),
            math.fsum(term.imag for term in terms),
        )
    if complete_last_outcome:
        # The projectors form an exact resolution of identity.  Reconstructing
        # one fixed, predeclared bin is algebraically identical and avoids a
        # catastrophic cancellation when psi(x) is extremely small.
        partial = complex(
            math.fsum(value.real for value in values.values()),
            math.fsum(value.imag for value in values.values()),
        )
        values[keys[-1]] = 1.0 - partial
    ordered_sum = complex(
        math.fsum(values[key].real for key in keys),
        math.fsum(values[key].imag for key in keys),
    )
    sum_rule = ordered_sum - 1.0
    return values, {
        "unique_amplitude_evaluations": len(ordered),
        "maximum_projector_connections": max(
            (len(row) for row in connections.values()), default=0
        ),
        "total_projector_connections": sum(len(row) for row in connections.values()),
        "completeness_reconstructed_outcome": int(complete_last_outcome),
        "local_sum_rule_absolute_error": float(abs(sum_rule)),
        "local_sum_rule_imaginary_error": float(abs(sum_rule.imag)),
    }


def weighted_projector_fcs_statistics(
    configurations: Sequence[Sequence[int]],
    weights: Sequence[float],
    log_amplitude_batch: LogAmplitudeBatch,
    groups: Sequence[Sequence[int]],
    *,
    complete_last_outcome: bool = False,
) -> dict[str, Any]:
    """Return exact weighted moments of projector-local estimators.

    This helper is intended for exact-small validation.  The production
    estimator only needs sampled configurations and the same amplitude-ratio
    callback.
    """

    samples = np.asarray(configurations, dtype=np.int8)
    probabilities = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    if probabilities.shape != (samples.shape[0],):
        raise ValueError("weights must match the configuration count")
    if np.any(~np.isfinite(probabilities)) or np.min(probabilities) < 0.0:
        raise ValueError("weights must be finite and non-negative")
    if abs(float(np.sum(probabilities)) - 1.0) > 1.0e-10:
        raise ValueError("weights must be normalized")

    checked = _validated_groups(samples.shape[1], groups)
    outcomes = fcs_outcomes(checked)
    matrix = np.empty((samples.shape[0], len(outcomes)), dtype=np.complex128)
    cost_rows = []
    for row, configuration in enumerate(samples):
        values, cost = projector_local_values(
            configuration,
            log_amplitude_batch,
            checked,
            outcomes,
            complete_last_outcome=complete_last_outcome,
        )
        matrix[row] = [values[key] for key in outcomes]
        cost_rows.append(cost)
    mean = probabilities @ matrix
    centered = matrix - mean
    covariance = (centered.conj().T * probabilities) @ centered
    real_covariance = (centered.real.T * probabilities) @ centered.real
    average_unique = float(
        probabilities
        @ np.asarray(
            [row["unique_amplitude_evaluations"] for row in cost_rows],
            dtype=np.float64,
        )
    )
    return {
        "outcomes": outcomes,
        "probabilities_complex": {key: complex(mean[i]) for i, key in enumerate(outcomes)},
        "probabilities": {key: float(mean[i].real) for i, key in enumerate(outcomes)},
        "local_values": matrix,
        "complex_covariance": covariance,
        "real_covariance": real_covariance,
        "maximum_imaginary_expectation": float(np.max(np.abs(mean.imag))),
        "normalization": float(np.sum(mean.real)),
        "maximum_local_sum_rule_error": max(
            float(row["local_sum_rule_absolute_error"]) for row in cost_rows
        ),
        "average_unique_amplitude_evaluations": average_unique,
        "maximum_unique_amplitude_evaluations": max(
            int(row["unique_amplitude_evaluations"]) for row in cost_rows
        ),
        "maximum_projector_connections": max(
            int(row["maximum_projector_connections"]) for row in cost_rows
        ),
        "average_total_projector_connections": float(
            probabilities
            @ np.asarray(
                [row["total_projector_connections"] for row in cost_rows],
                dtype=np.float64,
            )
        ),
        "completeness_reconstructed_outcome": outcomes[-1]
        if complete_last_outcome
        else None,
    }


def sampled_projector_fcs_from_local_values(
    local_values: np.ndarray,
    outcomes: Sequence[FCSKey],
    sample_indices: Sequence[int],
) -> dict[str, Any]:
    """Average precomputed local values over iid outer-sample indices."""

    values = np.asarray(local_values, dtype=np.complex128)
    keys = tuple(tuple(int(q) for q in key) for key in outcomes)
    indices = np.asarray(sample_indices, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(keys):
        raise ValueError("local_values and outcomes have inconsistent shapes")
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("at least two sample indices are required")
    if np.min(indices) < 0 or np.max(indices) >= values.shape[0]:
        raise ValueError("a sample index lies outside local_values")
    selected = values[indices]
    mean = np.mean(selected, axis=0)
    covariance_of_mean = np.cov(selected.real, rowvar=False, ddof=1) / indices.size
    covariance_of_mean = np.atleast_2d(covariance_of_mean)
    return {
        "n_samples": int(indices.size),
        "probabilities": {key: float(mean[i].real) for i, key in enumerate(keys)},
        "imaginary_means": {key: float(mean[i].imag) for i, key in enumerate(keys)},
        "maximum_imaginary_mean": float(np.max(np.abs(mean.imag))),
        "normalization": float(np.sum(mean.real)),
        "minimum_raw_probability": float(np.min(mean.real)),
        "covariance_of_mean": covariance_of_mean,
    }


def project_distribution_to_simplex(
    distribution: Mapping[FCSKey, float],
) -> dict[FCSKey, float]:
    """Return the Euclidean projection of fixed-support values onto a simplex."""

    if not distribution:
        raise ValueError("distribution must be nonempty")
    keys = tuple(sorted(distribution))
    values = np.asarray([float(distribution[key]) for key in keys], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("distribution values must be finite")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    valid = np.flatnonzero(ordered - cumulative / np.arange(1, len(values) + 1) > 0.0)
    if valid.size == 0:
        raise RuntimeError("simplex projection failed")
    rho = int(valid[-1])
    threshold = cumulative[rho] / (rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    projected /= np.sum(projected)
    return {key: float(value) for key, value in zip(keys, projected, strict=True)}


def tree_native_fcs_from_labels(
    label_samples: Sequence[Sequence[int]],
    group_label_indices: Sequence[int | None],
    *,
    support: Sequence[FCSKey] | None = None,
) -> dict[FCSKey, float]:
    """Read a block-spin FCS histogram from grouped-tree label samples."""

    labels = np.asarray(label_samples, dtype=np.int32)
    indices = tuple(group_label_indices)
    if labels.ndim != 2 or labels.shape[0] < 1:
        raise ValueError("label_samples must be a nonempty rank-two array")
    if not indices:
        raise ValueError("group_label_indices must be nonempty")
    if any(index is not None and not 0 <= index < labels.shape[1] for index in indices):
        raise ValueError("a group label index lies outside the tree state")
    counts: Counter[FCSKey] = Counter(
        tuple(1 if index is None else int(row[index]) for index in indices)
        for row in labels
    )
    if support is None:
        keys = tuple(sorted(counts))
    else:
        keys = tuple(tuple(int(q) for q in key) for key in support)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("support must be nonempty and contain no duplicates")
        if any(len(key) != len(indices) for key in keys):
            raise ValueError("support arity must match group_label_indices")
        outside = set(counts) - set(keys)
        if outside:
            raise ValueError("an observed tree outcome lies outside the fixed support")
    return {key: counts.get(key, 0) / labels.shape[0] for key in keys}


def sample_tree_native_fcs(
    amplitudes: Sequence[complex],
    states: Sequence[FusionTreeState],
    group_label_indices: Sequence[int | None],
    n_samples: int,
    seed: int,
    *,
    support: Sequence[FCSKey] | None = None,
) -> dict[FCSKey, float]:
    """Sample tree states, then read their subtree-root labels directly."""

    values = np.asarray(amplitudes, dtype=np.complex128)
    states = tuple(states)
    if values.shape != (len(states),) or not states:
        raise ValueError("amplitudes must match a nonempty tree-state sequence")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    probabilities = np.abs(values) ** 2
    normalization = float(np.sum(probabilities))
    if abs(normalization - 1.0) > 1.0e-10:
        raise ValueError("tree amplitudes must be normalized")
    rng = np.random.default_rng(int(seed))
    choices = rng.choice(len(states), size=int(n_samples), p=probabilities)
    labels = np.asarray([states[index].labels for index in choices], dtype=np.int32)
    return tree_native_fcs_from_labels(
        labels, group_label_indices, support=support
    )


__all__ = [
    "allowed_block_spins",
    "block_spin_projector_action",
    "fcs_outcomes",
    "joint_projector_connections",
    "project_distribution_to_simplex",
    "projector_local_values",
    "sample_tree_native_fcs",
    "sampled_projector_fcs_from_local_values",
    "tree_native_fcs_from_labels",
    "weighted_projector_fcs_statistics",
]
