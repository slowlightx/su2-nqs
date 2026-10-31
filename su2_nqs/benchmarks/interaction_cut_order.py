"""Graph-only nested-cut orders and exact sparse truncation diagnostics.

An order exposes the weighted interaction cut

``B_k = sum_{i in A_k, j not in A_k} |J_ij|``

after every physical prefix ``A_k``.  The exact dynamic program below chooses
the lexicographically first order minimizing ``(max_k B_k, sum_k B_k)``.  It
uses Hamiltonian graph data only and never inspects a wavefunction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


WeightedEdge = tuple[int, int, float]


def normalized_absolute_edges(
    n_sites: int, bonds: Iterable[Sequence[Real]]
) -> tuple[WeightedEdge, ...]:
    """Combine graph bonds into finite non-negative absolute edge weights."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError("n_sites must be an integer of at least two")
    n_sites = int(n_sites)
    accumulated: dict[tuple[int, int], float] = {}
    for raw in bonds:
        if len(raw) == 2:
            first, second = raw
            coupling = 1.0
        elif len(raw) == 3:
            first, second, coupling = raw
        else:
            raise ValueError("edges require two sites and an optional weight")
        first, second = int(first), int(second)
        value = float(coupling)
        if (
            first == second
            or not 0 <= first < n_sites
            or not 0 <= second < n_sites
            or not np.isfinite(value)
        ):
            raise ValueError("edge contains invalid sites or weight")
        pair = tuple(sorted((first, second)))
        accumulated[pair] = accumulated.get(pair, 0.0) + value
    combined = {
        pair: abs(value) for pair, value in accumulated.items() if value != 0.0
    }
    if not combined:
        raise ValueError("at least one nonzero edge is required")
    return tuple(
        (first, second, value)
        for (first, second), value in sorted(combined.items())
    )


def order_cut_exposures(
    order: Sequence[int], bonds: Iterable[Sequence[Real]]
) -> tuple[float, ...]:
    """Return ``B_k`` for all nontrivial prefixes of a physical order."""

    checked = tuple(int(site) for site in order)
    if tuple(sorted(checked)) != tuple(range(len(checked))):
        raise ValueError("order must contain every site exactly once")
    edges = normalized_absolute_edges(len(checked), bonds)
    position = {site: index for index, site in enumerate(checked)}
    return tuple(
        float(
            sum(
                weight
                for first, second, weight in edges
                if (position[first] < size) != (position[second] < size)
            )
        )
        for size in range(1, len(checked))
    )


def exact_minimax_cut_order(
    n_sites: int, bonds: Iterable[Sequence[Real]], *, tolerance: float = 1.0e-12
) -> dict[str, Any]:
    """Exactly minimize ``(max B_k, sum B_k, lexicographic order)`` by DP."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    edges = normalized_absolute_edges(n_sites, bonds)
    n_sites = int(n_sites)
    count = 1 << n_sites
    full = count - 1
    cut = np.zeros(count, dtype=np.float64)
    for mask in range(1, full):
        cut[mask] = sum(
            weight
            for first, second, weight in edges
            if bool(mask & (1 << first)) != bool(mask & (1 << second))
        )
    # Phase one: exact cutwidth.  Keeping only one joint (maximum,sum) label
    # per subset would be invalid: a larger partial maximum can later be
    # hidden by the global bottleneck while its smaller partial sum wins.
    # Cutwidth alone *does* have optimal substructure.
    best_max = np.full(count, np.inf, dtype=np.float64)
    best_max[0] = 0.0
    for mask in range(1, count):
        exposure = 0.0 if mask == full else float(cut[mask])
        predecessor_width = min(
            float(best_max[mask ^ (1 << site)])
            for site in range(n_sites)
            if mask & (1 << site)
        )
        best_max[mask] = max(predecessor_width, exposure)
    global_width = float(best_max[full])

    # Phase two: among paths whose every prefix respects the globally minimal
    # width, minimize summed exposure and then the complete lexicographic
    # order.  This is a shortest path on the subset lattice.
    best_sum = np.full(count, np.inf, dtype=np.float64)
    best_order: list[tuple[int, ...] | None] = [None] * count
    best_sum[0] = 0.0
    best_order[0] = ()
    for mask in range(1, count):
        exposure = 0.0 if mask == full else float(cut[mask])
        if exposure > global_width + tolerance:
            continue
        candidates = []
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            site = bit.bit_length() - 1
            previous = mask ^ bit
            prefix = best_order[previous]
            if prefix is not None:
                candidates.append(
                    (float(best_sum[previous]) + exposure, prefix + (site,))
                )
            remaining ^= bit
        if not candidates:
            continue
        minimum_sum = min(row[0] for row in candidates)
        tied = [row for row in candidates if row[0] <= minimum_sum + tolerance]
        chosen = min(tied, key=lambda row: row[1])
        best_sum[mask], best_order[mask] = chosen
    order = best_order[full]
    if order is None:
        raise RuntimeError("dynamic program produced no complete order")
    exposures = order_cut_exposures(order, edges)
    objective = (max(exposures), sum(exposures), order)
    if objective[0] > global_width + tolerance:
        raise RuntimeError("second-stage order violates the optimal cutwidth")
    return {
        "order": order,
        "cut_exposures": exposures,
        "maximum_cut_exposure": float(objective[0]),
        "summed_cut_exposure": float(objective[1]),
        "global_minimum_cutwidth": global_width,
        "state_count": count,
        "objective_tolerance": float(tolerance),
    }


def physical_prefix_signature(
    order: Sequence[int], cuts: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Return the physical subset at every registered fusion cut."""

    checked = tuple(int(site) for site in order)
    if tuple(sorted(checked)) != tuple(range(len(checked))):
        raise ValueError("order must contain every site exactly once")
    sizes = tuple(int(size) for size in cuts)
    if (
        not sizes
        or any(not 0 < size < len(checked) for size in sizes)
        or any(right <= left for left, right in zip(sizes, sizes[1:]))
    ):
        raise ValueError("cuts must be strictly increasing nontrivial prefix sizes")
    return tuple(tuple(sorted(checked[:size])) for size in sizes)


def subset_is_connected(
    subset: Sequence[int], n_sites: int, bonds: Iterable[Sequence[Real]]
) -> bool:
    """Return graph connectivity of one nonempty physical subset."""

    members = set(int(site) for site in subset)
    if not members or any(not 0 <= site < n_sites for site in members):
        return False
    edges = normalized_absolute_edges(n_sites, bonds)
    seen = {min(members)}
    frontier = list(seen)
    while frontier:
        site = frontier.pop()
        for first, second, _ in edges:
            if first == site and second in members and second not in seen:
                seen.add(second)
                frontier.append(second)
            elif second == site and first in members and first not in seen:
                seen.add(first)
                frontier.append(first)
    return seen == members


def first_connected_random_orders(
    n_sites: int,
    bonds: Iterable[Sequence[Real]],
    *,
    seed: int,
    count: int,
    signature_cuts: Sequence[int],
    connectivity_cuts: Sequence[int],
    excluded_signatures: Iterable[Sequence[Sequence[int]]] = (),
) -> tuple[tuple[int, ...], ...]:
    """Draw the first target-blind connected orders under frozen PCG64."""

    if count < 1:
        raise ValueError("count must be positive")
    edges = normalized_absolute_edges(n_sites, bonds)
    excluded = {
        tuple(tuple(int(site) for site in subset) for subset in signature)
        for signature in excluded_signatures
    }
    seen = set(excluded)
    accepted = []
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    attempts = 0
    while len(accepted) < count:
        attempts += 1
        if attempts > 10_000_000:
            raise RuntimeError("could not construct connected random controls")
        order = tuple(int(site) for site in rng.permutation(n_sites))
        signature = physical_prefix_signature(order, signature_cuts)
        if signature in seen:
            continue
        connected = True
        for size in connectivity_cuts:
            prefix = order[: int(size)]
            complement = order[int(size) :]
            if not subset_is_connected(prefix, n_sites, edges) or not subset_is_connected(
                complement, n_sites, edges
            ):
                connected = False
                break
        if connected:
            seen.add(signature)
            accepted.append(order)
    return tuple(accepted)


def registered_exposure_summary(
    order: Sequence[int], bonds: Iterable[Sequence[Real]], cuts: Sequence[int]
) -> dict[str, Any]:
    """Summarize actual-Hamiltonian cut exposure at registered prefixes."""

    all_exposures = order_cut_exposures(order, bonds)
    sizes = tuple(int(size) for size in cuts)
    values = tuple(float(all_exposures[size - 1]) for size in sizes)
    return {
        "cuts": list(sizes),
        "values": list(values),
        "maximum": float(max(values)),
        "sum": float(sum(values)),
        "all_prefix_maximum": float(max(all_exposures)),
        "all_prefix_sum": float(sum(all_exposures)),
    }


def sparse_leakage_and_graph_metrics(
    hamiltonian: Any, retained: Sequence[int]
) -> dict[str, float | int]:
    """Return ``||(I-P)HP||_F/||H||_F`` and sparse ``PHP`` graph costs."""

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover - core exact tests require scipy
        raise RuntimeError("sparse truncation metrics require scipy") from error
    matrix = sparse.csr_matrix(hamiltonian)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    indices = np.asarray(tuple(int(index) for index in retained), dtype=np.int64)
    if len(indices) == 0 or len(set(indices.tolist())) != len(indices):
        raise ValueError("retained indices must be nonempty and unique")
    complement = np.setdiff1d(np.arange(matrix.shape[0]), indices, assume_unique=True)
    block = matrix[indices][:, indices].tocsr()
    leakage = matrix[complement][:, indices].tocsr()
    denominator = float(np.sqrt(np.sum(np.abs(matrix.data) ** 2)))
    numerator = float(np.sqrt(np.sum(np.abs(leakage.data) ** 2)))
    offdiagonal = block.copy()
    offdiagonal.setdiag(0.0)
    offdiagonal.eliminate_zeros()
    degrees = np.diff(offdiagonal.indptr)
    hermiticity = block - block.getH()
    hermiticity_error = (
        float(np.max(np.abs(hermiticity.data))) if hermiticity.nnz else 0.0
    )
    return {
        "leakage_ratio": numerator / max(denominator, 1.0e-300),
        "leakage_Frobenius_norm": numerator,
        "Hamiltonian_Frobenius_norm": denominator,
        "projected_nnz": int(block.nnz),
        "projected_offdiagonal_nnz": int(offdiagonal.nnz),
        "projected_mean_offdiagonal_degree": float(np.mean(degrees)),
        "projected_max_offdiagonal_degree": int(np.max(degrees, initial=0)),
        "projected_Hermiticity_error": hermiticity_error,
    }


def _average_ranks(values: Sequence[Real]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or np.any(~np.isfinite(array)):
        raise ValueError("rank input must be a finite vector of length at least two")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(left: Sequence[Real], right: Sequence[Real]) -> float:
    """Return tie-aware Spearman rank correlation without SciPy statistics."""

    x, y = _average_ranks(left), _average_ranks(right)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    return float(np.dot(x, y) / denominator)


def exposure_quartile_contrast(
    rows: Sequence[Mapping[str, Real | str]],
    *,
    zero_tolerance: float = 1.0e-15,
) -> dict[str, Any]:
    """Compare equal-size low/high exposure tails of a unique-projector atlas."""

    checked = tuple(rows)
    if len(checked) < 4:
        raise ValueError("quartile comparison requires at least four rows")
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")
    ordered = sorted(
        checked,
        key=lambda row: (float(row["registered_cut_exposure_sum"]), str(row["order_id"])),
    )
    tail_count = int(np.ceil(len(ordered) / 4.0))
    low, high = ordered[:tail_count], ordered[-tail_count:]

    def mean(group, field):
        return float(np.mean([float(row[field]) for row in group]))

    low_fidelity, high_fidelity = mean(low, "projection_fidelity"), mean(
        high, "projection_fidelity"
    )
    low_ritz, high_ritz = mean(low, "Ritz_error_over_bandwidth"), mean(
        high, "Ritz_error_over_bandwidth"
    )
    low_leakage, high_leakage = mean(low, "leakage_ratio"), mean(
        high, "leakage_ratio"
    )
    ritz_denominator = abs(high_ritz)
    leakage_denominator = abs(high_leakage)
    return {
        "unique_row_count": len(ordered),
        "quartile_count": tail_count,
        "low_order_ids": [str(row["order_id"]) for row in low],
        "high_order_ids": [str(row["order_id"]) for row in high],
        "fidelity_low_minus_high": low_fidelity - high_fidelity,
        "Ritz_error_reduction": (
            (high_ritz - low_ritz) / ritz_denominator
            if ritz_denominator > zero_tolerance
            else None
        ),
        "leakage_reduction": (
            (high_leakage - low_leakage) / leakage_denominator
            if leakage_denominator > zero_tolerance
            else None
        ),
        "Ritz_reduction_denominator": ritz_denominator,
        "leakage_reduction_denominator": leakage_denominator,
        "zero_tolerance": float(zero_tolerance),
        "low_means": {
            "projection_fidelity": low_fidelity,
            "Ritz_error_over_bandwidth": low_ritz,
            "leakage_ratio": low_leakage,
        },
        "high_means": {
            "projection_fidelity": high_fidelity,
            "Ritz_error_over_bandwidth": high_ritz,
            "leakage_ratio": high_leakage,
        },
    }


__all__ = [
    "exact_minimax_cut_order",
    "exposure_quartile_contrast",
    "first_connected_random_orders",
    "normalized_absolute_edges",
    "order_cut_exposures",
    "physical_prefix_signature",
    "registered_exposure_summary",
    "sparse_leakage_and_graph_metrics",
    "spearman_correlation",
    "subset_is_connected",
]
