"""Exact information diagnostics for discrete autoregressive representations.

These helpers quantify the classical mutual information (CMI) of the Born
distribution in a *specified* basis and ordering.  The result is deliberately
called amplitude information, not entanglement: it is basis and ordering
dependent.  A uniform distribution on the legal support can be supplied as a
kinematic reference for hard-mask comparisons, but subtracting that reference
does not produce a basis-invariant complexity measure.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np


def _validated_distribution(
    sequences: Sequence[Sequence[int]], probabilities: Sequence[Real]
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    rows = tuple(tuple(int(token) for token in row) for row in sequences)
    weights = np.asarray(probabilities, dtype=np.float64)
    if not rows:
        raise ValueError("at least one sequence is required")
    length = len(rows[0])
    if length < 2 or any(len(row) != length for row in rows):
        raise ValueError("sequences must have one common length of at least two")
    if len(set(rows)) != len(rows):
        raise ValueError("sequences must be unique")
    if weights.shape != (len(rows),):
        raise ValueError("probabilities must match the sequence count")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if abs(float(np.sum(weights)) - 1.0) > 1.0e-10:
        raise ValueError("probabilities must be normalized")
    if not np.any(weights > 0.0):
        raise ValueError("the distribution must have positive support")
    return rows, weights


def shannon_entropy(probabilities: Sequence[Real]) -> float:
    """Return Shannon entropy in bits, ignoring exactly zero weights."""

    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < 1:
        raise ValueError("probabilities must be a non-empty vector")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if abs(float(np.sum(weights)) - 1.0) > 1.0e-10:
        raise ValueError("probabilities must be normalized")
    active = weights[weights > 0.0]
    return float(-np.sum(active * np.log2(active)))


def cut_amplitude_information(
    sequences: Sequence[Sequence[int]],
    probabilities: Sequence[Real],
    *,
    cut: int,
    rank_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Evaluate Born CMI and joint-probability rank diagnostics at one cut.

    ``cut`` is the number of sequence tokens in the prefix.  The numerical
    rank is reported only as a joint-distribution diagnostic; unlike the CMI,
    it is tolerance dependent and should not be used as a smooth observable.
    """

    rows, weights = _validated_distribution(sequences, probabilities)
    if isinstance(cut, bool) or not isinstance(cut, int):
        raise TypeError("cut must be an integer")
    if cut <= 0 or cut >= len(rows[0]):
        raise ValueError("cut must leave non-empty prefix and suffix")
    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive and finite")

    prefixes = sorted({row[:cut] for row in rows})
    suffixes = sorted({row[cut:] for row in rows})
    prefix_index = {value: index for index, value in enumerate(prefixes)}
    suffix_index = {value: index for index, value in enumerate(suffixes)}
    joint = np.zeros((len(prefixes), len(suffixes)), dtype=np.float64)
    for row, weight in zip(rows, weights, strict=True):
        joint[prefix_index[row[:cut]], suffix_index[row[cut:]]] += weight
    p_prefix = np.sum(joint, axis=1)
    p_suffix = np.sum(joint, axis=0)
    independent = p_prefix[:, None] * p_suffix[None, :]
    active = joint > 0.0
    cmi = float(np.sum(joint[active] * np.log2(joint[active] / independent[active])))

    singular_values = np.linalg.svd(joint, compute_uv=False)
    scale = float(singular_values[0]) if len(singular_values) else 0.0
    numerical_rank = int(np.count_nonzero(singular_values > tolerance * scale))
    singular_sum = float(np.sum(singular_values))
    if singular_sum == 0.0:
        effective_rank = 0.0
    else:
        normalized = singular_values[singular_values > 0.0] / singular_sum
        effective_rank = float(2.0 ** (-np.sum(normalized * np.log2(normalized))))
    return {
        "cut": cut,
        "sequence_length": len(rows[0]),
        "prefix_support": len(prefixes),
        "suffix_support": len(suffixes),
        "cmi_bits": cmi,
        "numerical_rank": numerical_rank,
        "log2_numerical_rank": float(np.log2(numerical_rank)),
        "effective_singular_rank": effective_rank,
        "rank_tolerance": tolerance,
    }


def amplitude_information_profile(
    sequences: Sequence[Sequence[int]],
    probabilities: Sequence[Real],
    *,
    cuts: Sequence[int] | None = None,
    rank_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Return CMI over registered cuts plus support non-uniformity."""

    rows, weights = _validated_distribution(sequences, probabilities)
    length = len(rows[0])
    selected = tuple(range(1, length)) if cuts is None else tuple(cuts)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("cuts must be a non-empty sequence without duplicates")
    profile = [
        cut_amplitude_information(
            rows, weights, cut=cut, rank_tolerance=rank_tolerance
        )
        for cut in selected
    ]
    middle_cut = length // 2
    if middle_cut not in selected:
        raise ValueError("cuts must include the floor middle cut")
    middle = next(row for row in profile if row["cut"] == middle_cut)
    entropy = shannon_entropy(weights)
    support_size = int(np.count_nonzero(weights > 0.0))
    uniform_kl = float(np.log2(support_size) - entropy)
    return {
        "sequence_length": length,
        "support_size": support_size,
        "entropy_bits": entropy,
        "active_support_uniform_KL_bits": uniform_kl,
        "active_support_uniform_KL_fraction": (
            0.0 if support_size == 1 else uniform_kl / np.log2(support_size)
        ),
        # Backward-compatible aliases.  "Support" here has always meant the
        # positive-probability support, not an externally supplied legal set.
        "uniform_support_KL_bits": uniform_kl,
        "uniform_support_KL_fraction": (
            0.0 if support_size == 1 else uniform_kl / np.log2(support_size)
        ),
        "middle_cut": middle_cut,
        "middle_cmi_bits": middle["cmi_bits"],
        "maximum_cmi_bits": float(max(row["cmi_bits"] for row in profile)),
        "mean_cmi_bits": float(np.mean([row["cmi_bits"] for row in profile])),
        "cuts": profile,
    }


def compare_to_kinematic_reference(
    target: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float]:
    """Compare matched sequence profiles to a uniform legal-support reference.

    The signed differences are diagnostics, not rigorous capacity bounds.
    """

    if target["sequence_length"] != reference["sequence_length"]:
        raise ValueError("target and reference sequence lengths must match")
    target_cuts = {row["cut"]: row for row in target["cuts"]}
    reference_cuts = {row["cut"]: row for row in reference["cuts"]}
    if target_cuts.keys() != reference_cuts.keys():
        raise ValueError("target and reference cuts must match")
    excess = {
        cut: float(target_cuts[cut]["cmi_bits"] - reference_cuts[cut]["cmi_bits"])
        for cut in target_cuts
    }
    return {
        "middle_cmi_excess_bits": excess[target["middle_cut"]],
        "maximum_signed_cmi_excess_bits": max(excess.values()),
        "minimum_signed_cmi_excess_bits": min(excess.values()),
        "mean_signed_cmi_excess_bits": float(np.mean(list(excess.values()))),
        "uniform_support_KL_bits": float(target["uniform_support_KL_bits"]),
        "uniform_support_KL_fraction": float(
            target["uniform_support_KL_fraction"]
        ),
    }


def conditional_kl_to_reference(
    target_sequences: Sequence[Sequence[int]],
    target_probabilities: Sequence[Real],
    reference_sequences: Sequence[Sequence[int]],
    reference_probabilities: Sequence[Real],
    *,
    cut: int,
) -> dict[str, float | int]:
    """Split ``KL(target || reference)`` at an autoregressive cut.

    When ``reference`` is uniform over the exact legal support, the future
    term measures the target's conditional amplitude modulation beyond what
    the hard reachability mask supplies.  This quantity is non-negative and
    obeys the KL chain rule, but remains basis and ordering dependent.
    """

    target_rows, target_weights = _validated_distribution(
        target_sequences, target_probabilities
    )
    reference_rows, reference_weights = _validated_distribution(
        reference_sequences, reference_probabilities
    )
    length = len(target_rows[0])
    if len(reference_rows[0]) != length:
        raise ValueError("target and reference sequence lengths must match")
    if isinstance(cut, bool) or not isinstance(cut, int):
        raise TypeError("cut must be an integer")
    if cut <= 0 or cut >= length:
        raise ValueError("cut must leave non-empty prefix and suffix")

    target = {row: float(weight) for row, weight in zip(target_rows, target_weights, strict=True)}
    reference = {
        row: float(weight)
        for row, weight in zip(reference_rows, reference_weights, strict=True)
    }
    missing = [row for row, weight in target.items() if weight > 0.0 and reference.get(row, 0.0) <= 0.0]
    if missing:
        raise ValueError("reference must have positive weight on target support")

    target_prefix: dict[tuple[int, ...], float] = {}
    reference_prefix: dict[tuple[int, ...], float] = {}
    for row, weight in target.items():
        prefix = row[:cut]
        target_prefix[prefix] = target_prefix.get(prefix, 0.0) + weight
    for row, weight in reference.items():
        prefix = row[:cut]
        reference_prefix[prefix] = reference_prefix.get(prefix, 0.0) + weight

    total_kl = 0.0
    prefix_kl = 0.0
    future_kl = 0.0
    for row, probability in target.items():
        if probability <= 0.0:
            continue
        reference_probability = reference[row]
        prefix = row[:cut]
        p_prefix = target_prefix[prefix]
        q_prefix = reference_prefix[prefix]
        total_kl += probability * np.log2(probability / reference_probability)
        future_kl += probability * np.log2(
            (probability / p_prefix) / (reference_probability / q_prefix)
        )
    for prefix, probability in target_prefix.items():
        if probability > 0.0:
            q_probability = reference_prefix.get(prefix, 0.0)
            if q_probability <= 0.0:
                raise ValueError("reference must cover every active target prefix")
            prefix_kl += probability * np.log2(probability / q_probability)
    chain_error = abs(total_kl - prefix_kl - future_kl)
    return {
        "cut": cut,
        "total_KL_bits": float(total_kl),
        "prefix_KL_bits": float(prefix_kl),
        "conditional_future_KL_bits": float(future_kl),
        "chain_rule_error": float(chain_error),
    }


def conditioned_cut_amplitude_information(
    sequences: Sequence[Sequence[int]],
    probabilities: Sequence[Real],
    boundary_labels: Sequence[int],
    *,
    cut: int,
    rank_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Resolve cut CMI into conserved-boundary and dynamical parts.

    If the supplied boundary label is a deterministic function of both the
    prefix and suffix, then ``I(A:B) = H(C) + I(A:B|C)``.  The closure and
    both determinism conditions are returned explicitly rather than assumed.
    ``I(A:B|C)`` is the amplitude dependence left after conditioning on the
    registered conserved charge at the cut.
    """

    rows, weights = _validated_distribution(sequences, probabilities)
    labels = tuple(int(value) for value in boundary_labels)
    if len(labels) != len(rows):
        raise ValueError("boundary labels must match the sequence count")
    raw = cut_amplitude_information(
        rows, weights, cut=cut, rank_tolerance=rank_tolerance
    )
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    group_weights = np.asarray(
        [np.sum(weights[indices]) for indices in groups.values()], dtype=np.float64
    )
    boundary_entropy = shannon_entropy(group_weights)
    conditional_cmi = 0.0
    blocks = []
    for label, indices in sorted(groups.items()):
        probability = float(np.sum(weights[indices]))
        if probability <= 0.0:
            continue
        normalized = weights[indices] / probability
        block = cut_amplitude_information(
            [rows[index] for index in indices],
            normalized,
            cut=cut,
            rank_tolerance=rank_tolerance,
        )
        conditional_cmi += probability * float(block["cmi_bits"])
        blocks.append(
            {
                "boundary_label": label,
                "probability": probability,
                "cmi_bits": float(block["cmi_bits"]),
                "numerical_rank": int(block["numerical_rank"]),
                "effective_singular_rank": float(
                    block["effective_singular_rank"]
                ),
            }
        )
    prefix_to_labels: dict[tuple[int, ...], set[int]] = {}
    suffix_to_labels: dict[tuple[int, ...], set[int]] = {}
    for row, label, weight in zip(rows, labels, weights, strict=True):
        if weight <= 0.0:
            continue
        prefix_to_labels.setdefault(row[:cut], set()).add(label)
        suffix_to_labels.setdefault(row[cut:], set()).add(label)
    prefix_deterministic = all(len(values) == 1 for values in prefix_to_labels.values())
    suffix_deterministic = all(len(values) == 1 for values in suffix_to_labels.values())
    closure = float(raw["cmi_bits"] - boundary_entropy - conditional_cmi)
    return {
        "cut": cut,
        "raw_cmi_bits": float(raw["cmi_bits"]),
        "boundary_entropy_bits": boundary_entropy,
        "conditional_cmi_bits": float(conditional_cmi),
        "decomposition_error": abs(closure),
        "signed_decomposition_residual": closure,
        "boundary_is_prefix_deterministic": prefix_deterministic,
        "boundary_is_suffix_deterministic": suffix_deterministic,
        "blocks": blocks,
    }


__all__ = [
    "amplitude_information_profile",
    "compare_to_kinematic_reference",
    "conditional_kl_to_reference",
    "conditioned_cut_amplitude_information",
    "cut_amplitude_information",
    "shannon_entropy",
]
