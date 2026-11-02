"""Exact matched-rank SU(2) prefix-channel truncation diagnostics.

The truncation is a physical subspace once a leaf order is supplied.  If
``order[k]`` is the physical site at fusion leaf ``k``, constraining the
left-fusion labels after ``k`` leaves constrains the total spin of the
explicit physical prefix ``order[:k]``.  Different leaf orders therefore
give different, but exactly equal-rank, subspaces of one fixed total-spin
Hilbert space.

This module is an exact-small reference.  It does not implement a scalable
NQS truncation and it never selects a chart using a target wavefunction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from ..operators.permutation import apply_site_permutation_fusion


def nested_prefix_order(
    n_sites: int,
    inner: Sequence[int],
    outer: Sequence[int],
) -> tuple[int, ...]:
    """Return a deterministic order with ``inner`` then ``outer-inner``.

    The order inside each shell is the physical site order.  It cannot alter
    a truncation imposed only at the two registered prefix boundaries, but
    fixing it makes the coordinate convention reproducible.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError("n_sites must be an integer of at least two")
    n_sites = int(n_sites)
    checked_inner = tuple(sorted(int(site) for site in inner))
    checked_outer = tuple(sorted(int(site) for site in outer))
    if (
        len(set(checked_inner)) != len(checked_inner)
        or len(set(checked_outer)) != len(checked_outer)
        or not set(checked_inner) < set(checked_outer)
        or any(not 0 <= site < n_sites for site in checked_outer)
    ):
        raise ValueError("inner and outer must be strictly nested in-range subsets")
    remainder = tuple(site for site in range(n_sites) if site not in checked_outer)
    order = checked_inner + tuple(
        site for site in checked_outer if site not in checked_inner
    ) + remainder
    if tuple(sorted(order)) != tuple(range(n_sites)):
        raise RuntimeError("nested prefixes did not produce a site permutation")
    return order


def order_to_active_image(order: Sequence[int]) -> tuple[int, ...]:
    """Convert ``new leaf -> old physical site`` to ``old -> new`` image."""

    checked = tuple(int(site) for site in order)
    if tuple(sorted(checked)) != tuple(range(len(checked))):
        raise ValueError("order must contain every physical site exactly once")
    image = [0] * len(checked)
    for leaf, physical_site in enumerate(checked):
        image[physical_site] = leaf
    return tuple(image)


def permute_bonds_to_leaf_order(
    bonds: Iterable[Sequence[Real]], order: Sequence[int]
) -> tuple[tuple[int, int, float], ...]:
    """Express physical weighted bonds in coordinates of one leaf order."""

    image = order_to_active_image(order)
    accumulated: dict[tuple[int, int], float] = {}
    for raw_bond in bonds:
        if len(raw_bond) == 2:
            first, second = raw_bond
            coupling = 1.0
        elif len(raw_bond) == 3:
            first, second, coupling = raw_bond
        else:
            raise ValueError("bonds require two sites and an optional coupling")
        first, second = int(first), int(second)
        value = float(coupling)
        if (
            first == second
            or not 0 <= first < len(image)
            or not 0 <= second < len(image)
            or not np.isfinite(value)
        ):
            raise ValueError("bond contains invalid sites or coupling")
        pair = tuple(sorted((image[first], image[second])))
        accumulated[pair] = accumulated.get(pair, 0.0) + value
    return tuple(
        (first, second, value)
        for (first, second), value in sorted(accumulated.items())
        if value != 0.0
    )


def retained_path_indices(
    paths: Sequence[Sequence[int]],
    constraints: Mapping[int, int],
) -> tuple[int, ...]:
    """Return paths satisfying doubled-spin upper bounds at prefix sizes.

    ``constraints[k] = qmax`` means that the total doubled spin after the
    first ``k`` leaves must not exceed ``qmax``.  Labels and cutoffs are exact
    doubled integers; no floating-point half-integers are accepted.
    """

    checked_paths = tuple(tuple(path) for path in paths)
    if not checked_paths:
        raise ValueError("paths must be nonempty")
    n_sites = len(checked_paths[0])
    if any(len(path) != n_sites for path in checked_paths):
        raise ValueError("all paths must have the same length")
    checked_constraints: dict[int, int] = {}
    for raw_size, raw_qmax in constraints.items():
        if (
            isinstance(raw_size, bool)
            or not isinstance(raw_size, Integral)
            or not 1 <= int(raw_size) <= n_sites
        ):
            raise ValueError("prefix sizes must be in-range integers")
        if (
            isinstance(raw_qmax, bool)
            or not isinstance(raw_qmax, Integral)
            or int(raw_qmax) < 0
        ):
            raise ValueError("qmax values must be non-negative doubled integers")
        checked_constraints[int(raw_size)] = int(raw_qmax)
    if not checked_constraints:
        raise ValueError("at least one prefix constraint is required")
    retained = tuple(
        index
        for index, path in enumerate(checked_paths)
        if all(path[size - 1] <= qmax for size, qmax in checked_constraints.items())
    )
    if not retained:
        raise ValueError("prefix constraints retain no paths")
    return retained


def permuted_fusion_state(
    state: Sequence[complex], order: Sequence[int], *, target_q: int
) -> np.ndarray:
    """Express a fusion-basis state in the registered physical leaf order."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("state must be a vector")
    return np.asarray(
        apply_site_permutation_fusion(
            vector, order_to_active_image(order), target_q=int(target_q)
        ),
        dtype=np.complex128,
    )


def exact_truncation_metrics(
    ground_state: Sequence[complex],
    hamiltonian: np.ndarray,
    retained: Sequence[int],
    *,
    exact_energy: Real,
    n_sites: int,
    exact_singlet_gap: Real | None = None,
    exact_singlet_bandwidth: Real | None = None,
    variational_tolerance: float = 1.0e-10,
) -> dict[str, float | int]:
    """Return projection capacity and target-independent Ritz accuracy."""

    ground = np.asarray(ground_state, dtype=np.complex128)
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    indices = np.asarray(tuple(int(index) for index in retained), dtype=np.int64)
    if ground.ndim != 1 or matrix.shape != (ground.size, ground.size):
        raise ValueError("ground state and Hamiltonian shapes are incompatible")
    if len(indices) == 0 or len(set(indices.tolist())) != len(indices):
        raise ValueError("retained indices must be nonempty and unique")
    if np.min(indices) < 0 or np.max(indices) >= ground.size:
        raise ValueError("retained index lies outside the Hilbert space")
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    if variational_tolerance < 0.0:
        raise ValueError("variational_tolerance must be non-negative")
    norm = float(np.vdot(ground, ground).real)
    if abs(norm - 1.0) > 1.0e-10:
        raise ValueError("ground state must be normalized")
    projected = ground[indices]
    fidelity = float(np.vdot(projected, projected).real)
    block = matrix[np.ix_(indices, indices)]
    hermiticity_error = float(np.max(np.abs(block - block.conj().T)))
    ritz_energy = float(np.linalg.eigvalsh(block)[0])
    reference = float(exact_energy)
    raw_gap = ritz_energy - reference
    violation = max(-raw_gap, 0.0)
    positive_error = max(raw_gap, 0.0)
    gap_scale = float(exact_singlet_gap) if exact_singlet_gap is not None else 1.0
    bandwidth = (
        float(exact_singlet_bandwidth)
        if exact_singlet_bandwidth is not None
        else 1.0
    )
    if gap_scale <= 0.0 or bandwidth <= 0.0:
        raise ValueError("exact singlet gap and bandwidth must be positive")
    if fidelity > 0.0:
        normalized = projected / np.sqrt(fidelity)
        target_projected_energy = float(
            np.real(np.vdot(normalized, block @ normalized))
        )
    else:
        target_projected_energy = float("nan")
    target_projected_gap = target_projected_energy - reference
    return {
        "retained_dimension": int(len(indices)),
        "projection_fidelity": fidelity,
        "discarded_weight": float(max(1.0 - fidelity, 0.0)),
        "ritz_energy": ritz_energy,
        "ritz_energy_gap": raw_gap,
        "ritz_energy_error": positive_error,
        "ritz_energy_error_per_site": positive_error / int(n_sites),
        "ritz_energy_error_over_singlet_gap": positive_error / gap_scale,
        "ritz_energy_error_over_singlet_bandwidth": positive_error / bandwidth,
        "variational_bound_violation": violation,
        "variational_bound_pass": bool(violation <= variational_tolerance),
        "target_projected_energy": target_projected_energy,
        "target_projected_energy_gap": target_projected_gap,
        "target_projected_energy_error": max(target_projected_gap, 0.0),
        "subspace_hamiltonian_hermiticity_error": hermiticity_error,
    }


def aggregate_truncation_metrics(
    rows: Sequence[Mapping[str, Real]],
) -> dict[str, Any]:
    """Uniformly aggregate every registered chart without best-chart selection."""

    checked = tuple(rows)
    if not checked:
        raise ValueError("at least one chart metric is required")
    fields = (
        "projection_fidelity",
        "discarded_weight",
        "ritz_energy_error",
        "ritz_energy_error_per_site",
        "ritz_energy_error_over_singlet_gap",
        "ritz_energy_error_over_singlet_bandwidth",
        "target_projected_energy_error",
        "variational_bound_violation",
    )
    result: dict[str, Any] = {"chart_count": len(checked)}
    dimensions = {int(row["retained_dimension"]) for row in checked}
    if len(dimensions) != 1:
        raise ValueError("all charts must have exactly matched retained dimension")
    result["retained_dimension"] = dimensions.pop()
    for field in fields:
        values = np.asarray([float(row[field]) for row in checked], dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise ValueError(f"chart metric {field} is non-finite")
        result[field] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return result


__all__ = [
    "aggregate_truncation_metrics",
    "exact_truncation_metrics",
    "nested_prefix_order",
    "order_to_active_image",
    "permuted_fusion_state",
    "permute_bonds_to_leaf_order",
    "retained_path_indices",
]
