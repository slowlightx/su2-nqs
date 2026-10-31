"""Physical multiscale total-spin FCS for nested lattice subregions.

The labels in this module are doubled spins.  A label becomes a physical
observable only after it is attached to an explicit physical-site subset.
For nested subsets the corresponding Casimirs commute, so their labels admit
a joint probability distribution.  Internal leaf order is only a coordinate
choice and is audited explicitly by callers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Integral, Real
from typing import Any

import numpy as np


SpinDistribution = dict[int, float]
NestedSpinDistribution = dict[tuple[int, ...], float]


def _edge_pairs(
    n_sites: int, edges: Sequence[Sequence[Real]]
) -> tuple[tuple[int, int], ...]:
    pairs: set[tuple[int, int]] = set()
    for edge in edges:
        if len(edge) < 2:
            raise ValueError("every edge requires two site indices")
        first, second = int(edge[0]), int(edge[1])
        if first == second or not (0 <= first < n_sites and 0 <= second < n_sites):
            raise ValueError("edge endpoints must be distinct in-range sites")
        pairs.add(tuple(sorted((first, second))))
    if not pairs:
        raise ValueError("at least one edge is required")
    return tuple(sorted(pairs))


def _connected(subset: tuple[int, ...], pairs: Sequence[tuple[int, int]]) -> bool:
    members = set(subset)
    adjacency = {site: set() for site in subset}
    for first, second in pairs:
        if first in members and second in members:
            adjacency[first].add(second)
            adjacency[second].add(first)
    seen = {subset[0]}
    frontier = [subset[0]]
    while frontier:
        site = frontier.pop()
        for neighbor in adjacency[site] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    return len(seen) == len(subset)


def compact_connected_subsets(
    n_sites: int,
    edges: Sequence[Sequence[Real]],
    size: int,
) -> dict[str, Any]:
    """Return all connected size-``k`` subsets of maximal internal J1 density."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError("n_sites must be an integer of at least two")
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise TypeError("size must be an integer")
    n_sites, size = int(n_sites), int(size)
    if not 1 < size <= n_sites // 2:
        raise ValueError("size must lie between two and half the system")
    pairs = _edge_pairs(n_sites, edges)
    connected = tuple(
        subset
        for subset in combinations(range(n_sites), size)
        if _connected(subset, pairs)
    )
    maximum = -1
    retained: list[tuple[int, ...]] = []
    for subset in connected:
        internal = sum(first in subset and second in subset for first, second in pairs)
        if internal > maximum:
            maximum = internal
            retained = [subset]
        elif internal == maximum:
            retained.append(subset)
    if not retained:
        raise RuntimeError("no connected subset was found")
    return {
        "size": size,
        "maximum_internal_edges": maximum,
        "connected_subset_count": len(connected),
        "subsets": tuple(retained),
    }


def connected_subsets(
    n_sites: int,
    edges: Sequence[Sequence[Real]],
    size: int,
) -> tuple[tuple[int, ...], ...]:
    """Return every connected physical subset of one registered size."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError("n_sites must be an integer of at least two")
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise TypeError("size must be an integer")
    n_sites, size = int(n_sites), int(size)
    if not 1 < size <= n_sites // 2:
        raise ValueError("size must lie between two and half the system")
    pairs = _edge_pairs(n_sites, edges)
    return tuple(
        subset
        for subset in combinations(range(n_sites), size)
        if _connected(subset, pairs)
    )


def nested_compact_chains(
    families: Mapping[int, Sequence[Sequence[int]]],
    *,
    scales: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate deterministic inclusion chains through compact families."""

    checked_scales = tuple(int(value) for value in scales)
    if not checked_scales or any(
        right <= left for left, right in zip(checked_scales, checked_scales[1:])
    ):
        raise ValueError("scales must be nonempty and strictly increasing")
    checked: dict[int, tuple[tuple[int, ...], ...]] = {}
    for scale in checked_scales:
        if scale not in families:
            raise KeyError(f"missing compact family for scale {scale}")
        rows = tuple(tuple(sorted(int(site) for site in row)) for row in families[scale])
        if not rows or any(len(row) != scale or len(set(row)) != scale for row in rows):
            raise ValueError("compact-family subsets have invalid sizes or duplicates")
        checked[scale] = tuple(sorted(set(rows)))

    chains: list[tuple[tuple[int, ...], ...]] = [
        (row,) for row in checked[checked_scales[0]]
    ]
    for scale in checked_scales[1:]:
        extended = []
        for chain in chains:
            previous = set(chain[-1])
            for row in checked[scale]:
                if previous.issubset(row):
                    extended.append((*chain, row))
        chains = extended
    if not chains:
        raise RuntimeError("compact families contain no nested chain")
    return tuple(chains)


def _physical_leaf_order(
    n_sites: int,
    chain: Sequence[Sequence[int]],
    *,
    reverse_shells: bool,
) -> tuple[int, ...]:
    previous: set[int] = set()
    shells = []
    for raw_subset in chain:
        subset = set(int(site) for site in raw_subset)
        if not previous < subset or any(not 0 <= site < n_sites for site in subset):
            raise ValueError("chain subsets must be strictly nested and in range")
        shell = tuple(sorted(subset - previous, reverse=reverse_shells))
        shells.append(shell)
        previous = subset
    remainder = tuple(
        sorted((site for site in range(n_sites) if site not in previous), reverse=reverse_shells)
    )
    order = tuple(site for shell in shells for site in shell) + remainder
    if len(order) != n_sites or len(set(order)) != n_sites:
        raise RuntimeError("physical leaf order is not a permutation")
    return order


def permute_computational_state(
    wavefunction: Sequence[complex], order: Sequence[int]
) -> np.ndarray:
    """Express a computational-basis state in a new physical leaf order."""

    state = np.asarray(wavefunction, dtype=np.complex128)
    if state.ndim != 1:
        raise ValueError("wavefunction must be one-dimensional")
    n_sites = int(round(np.log2(len(state))))
    if 2**n_sites != len(state):
        raise ValueError("wavefunction length must be a power of two")
    permutation = tuple(int(site) for site in order)
    if sorted(permutation) != list(range(n_sites)):
        raise ValueError("order must be a permutation of all sites")
    return state.reshape((2,) * n_sites).transpose(permutation).reshape(-1)


def nested_spin_fcs(
    wavefunction: Sequence[complex],
    chain: Sequence[Sequence[int]],
    *,
    coupled_basis: np.ndarray,
    paths: Sequence[Sequence[int]],
    reverse_shells: bool = False,
) -> NestedSpinDistribution:
    """Return the joint FCS of total spins on explicit nested subsets."""

    state = np.asarray(wavefunction, dtype=np.complex128)
    n_sites = int(round(np.log2(len(state))))
    if 2**n_sites != len(state):
        raise ValueError("wavefunction length must be a power of two")
    norm = float(np.vdot(state, state).real)
    if abs(norm - 1.0) > 1.0e-10 or np.any(~np.isfinite(state)):
        raise ValueError("wavefunction must be finite and normalized")
    checked_chain = tuple(tuple(sorted(int(site) for site in row)) for row in chain)
    scales = tuple(len(row) for row in checked_chain)
    if any(scale < 2 or scale > n_sites for scale in scales):
        raise ValueError("nested subsets have invalid sizes")
    order = _physical_leaf_order(
        n_sites, checked_chain, reverse_shells=reverse_shells
    )
    reordered = permute_computational_state(state, order)
    transform = np.asarray(coupled_basis, dtype=np.complex128)
    checked_paths = tuple(tuple(int(q) for q in path) for path in paths)
    if transform.shape != (len(state), len(checked_paths)):
        raise ValueError("coupled basis has incompatible shape")
    if any(len(path) != n_sites for path in checked_paths):
        raise ValueError("fusion paths have incompatible length")
    amplitudes = transform.conj().T @ reordered
    captured = float(np.vdot(amplitudes, amplitudes).real)
    if abs(captured - 1.0) > 1.0e-10:
        raise ValueError("wavefunction is not contained in the supplied total-spin sector")
    probabilities: NestedSpinDistribution = {}
    indices = tuple(scale - 1 for scale in scales)
    for amplitude, path in zip(amplitudes, checked_paths, strict=True):
        key = tuple(path[index] for index in indices)
        probabilities[key] = probabilities.get(key, 0.0) + float(abs(amplitude) ** 2)
    return dict(sorted(probabilities.items()))


def average_spin_distributions(
    distributions: Sequence[Mapping[int, Real]],
) -> SpinDistribution:
    """Average normalized doubled-spin distributions with equal weight."""

    rows = tuple(distributions)
    if not rows:
        raise ValueError("at least one distribution is required")
    support = sorted(set().union(*(row.keys() for row in rows)))
    answer = {
        int(q): float(np.mean([float(row.get(q, 0.0)) for row in rows]))
        for q in support
    }
    _validate_spin_distribution(answer)
    return answer


def average_nested_distributions(
    distributions: Sequence[Mapping[tuple[int, ...], Real]],
) -> NestedSpinDistribution:
    """Average normalized joint FCS distributions with equal weight."""

    rows = tuple(distributions)
    if not rows:
        raise ValueError("at least one distribution is required")
    support = sorted(set().union(*(row.keys() for row in rows)))
    arities = {len(key) for key in support}
    if len(arities) != 1:
        raise ValueError("joint FCS outcomes must have one common arity")
    answer = {
        tuple(int(q) for q in key): float(
            np.mean([float(row.get(key, 0.0)) for row in rows])
        )
        for key in support
    }
    values = np.asarray(list(answer.values()))
    if np.any(~np.isfinite(values)) or np.min(values) < -1.0e-12:
        raise ValueError("joint FCS probabilities must be finite and non-negative")
    if abs(float(np.sum(values)) - 1.0) > 1.0e-10:
        raise ValueError("joint FCS probabilities must be normalized")
    return answer


def marginal_spin_distribution(
    distribution: Mapping[tuple[int, ...], Real], index: int
) -> SpinDistribution:
    """Marginalize one doubled-spin label from a joint FCS."""

    if not distribution:
        raise ValueError("joint FCS must be nonempty")
    arity = len(next(iter(distribution)))
    if not 0 <= index < arity or any(len(key) != arity for key in distribution):
        raise ValueError("invalid joint-FCS marginal index or arity")
    answer: SpinDistribution = {}
    for key, probability in distribution.items():
        q = int(key[index])
        answer[q] = answer.get(q, 0.0) + float(probability)
    _validate_spin_distribution(answer)
    return dict(sorted(answer.items()))


def _validate_spin_distribution(distribution: Mapping[int, Real]) -> None:
    if not distribution:
        raise ValueError("spin distribution must be nonempty")
    keys = tuple(distribution)
    values = np.asarray([distribution[key] for key in keys], dtype=np.float64)
    if any(
        isinstance(q, bool) or not isinstance(q, Integral) or int(q) < 0 for q in keys
    ):
        raise ValueError("doubled-spin labels must be non-negative integers")
    if np.any(~np.isfinite(values)) or np.min(values) < -1.0e-12:
        raise ValueError("spin probabilities must be finite and non-negative")
    if abs(float(np.sum(values)) - 1.0) > 1.0e-10:
        raise ValueError("spin probabilities must be normalized")


def spin_distribution_summary(distribution: Mapping[int, Real]) -> dict[str, Any]:
    """Return physical Casimir, tail, singlet, and sector-entropy summaries."""

    _validate_spin_distribution(distribution)
    probabilities = {int(q): float(value) for q, value in distribution.items()}
    support = tuple(sorted(probabilities))
    tails = {
        str(threshold): float(
            sum(value for q, value in probabilities.items() if q >= threshold)
        )
        for threshold in support
        if threshold > 0
    }
    casimir = float(
        sum(value * q * (q + 2) / 4.0 for q, value in probabilities.items())
    )
    entropy = float(
        -sum(value * np.log(value) for value in probabilities.values() if value > 0.0)
    )
    return {
        "probabilities": {str(q): probabilities[q] for q in support},
        "singlet_weight": probabilities.get(0, 0.0),
        "casimir": casimir,
        "spin_sector_entropy": entropy,
        "large_spin_tails": tails,
    }


def endpoint_chord_screening(
    ordered: Mapping[int, Real],
    candidate: Mapping[int, Real],
    stripe: Mapping[int, Real],
    *,
    interpolation_weight: Real,
) -> dict[str, Any]:
    """Compare a candidate marginal FCS with its endpoint probability chord."""

    for row in (ordered, candidate, stripe):
        _validate_spin_distribution(row)
    weight = float(interpolation_weight)
    if not np.isfinite(weight) or not 0.0 < weight < 1.0:
        raise ValueError("interpolation_weight must lie strictly between zero and one")
    support = tuple(sorted(set(ordered) | set(candidate) | set(stripe)))
    chord = {
        q: (1.0 - weight) * float(ordered.get(q, 0.0))
        + weight * float(stripe.get(q, 0.0))
        for q in support
    }
    center = {q: float(candidate.get(q, 0.0)) for q in support}
    chord_summary = spin_distribution_summary(chord)
    center_summary = spin_distribution_summary(center)
    tail_gaps = {
        threshold: float(
            chord_summary["large_spin_tails"][threshold]
            - center_summary["large_spin_tails"][threshold]
        )
        for threshold in chord_summary["large_spin_tails"]
    }
    casimir_chord = float(chord_summary["casimir"])
    return {
        "interpolation_weight": weight,
        "chord": chord_summary,
        "candidate": center_summary,
        "singlet_excess": float(
            center_summary["singlet_weight"] - chord_summary["singlet_weight"]
        ),
        "casimir_suppression": float(
            casimir_chord - center_summary["casimir"]
        ),
        "relative_casimir_suppression": float(
            (casimir_chord - center_summary["casimir"])
            / max(abs(casimir_chord), 1.0e-15)
        ),
        "tail_suppression": tail_gaps,
        "minimum_tail_suppression": float(min(tail_gaps.values(), default=0.0)),
        "total_variation_from_chord": float(
            0.5 * sum(abs(center[q] - chord[q]) for q in support)
        ),
        "sector_entropy_chord_minus_candidate": float(
            chord_summary["spin_sector_entropy"]
            - center_summary["spin_sector_entropy"]
        ),
    }


def nested_distribution_summary(
    distribution: Mapping[tuple[int, ...], Real],
) -> dict[str, Any]:
    """Return sector entropy, adjacent mutual information, and healing flows."""

    if not distribution:
        raise ValueError("joint FCS must be nonempty")
    arity = len(next(iter(distribution)))
    averaged = average_nested_distributions((distribution,))
    entropy = float(
        -sum(value * np.log(value) for value in averaged.values() if value > 0.0)
    )
    marginals = [marginal_spin_distribution(averaged, index) for index in range(arity)]
    mutual_information = []
    healing = []
    for index in range(arity - 1):
        left, right = marginals[index], marginals[index + 1]
        pair: dict[tuple[int, int], float] = {}
        for key, probability in averaged.items():
            pair_key = (key[index], key[index + 1])
            pair[pair_key] = pair.get(pair_key, 0.0) + probability
        information = 0.0
        numerator = 0.0
        denominator = 0.0
        for (left_q, right_q), probability in pair.items():
            if probability > 0.0:
                information += probability * np.log(
                    probability / (left[left_q] * right[right_q])
                )
            if left_q > 0:
                denominator += probability
                if right_q == 0:
                    numerator += probability
        mutual_information.append(float(information))
        healing.append(float(numerator / denominator) if denominator > 0.0 else 0.0)
    return {
        "joint_spin_sector_entropy": entropy,
        "adjacent_sector_mutual_information": mutual_information,
        "conditional_nonzero_to_singlet_healing": healing,
    }


__all__ = [
    "NestedSpinDistribution",
    "SpinDistribution",
    "average_nested_distributions",
    "average_spin_distributions",
    "compact_connected_subsets",
    "connected_subsets",
    "endpoint_chord_screening",
    "marginal_spin_distribution",
    "nested_compact_chains",
    "nested_distribution_summary",
    "nested_spin_fcs",
    "permute_computational_state",
    "spin_distribution_summary",
]
