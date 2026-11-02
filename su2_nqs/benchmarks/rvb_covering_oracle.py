"""Exact-small short-range RVB covering manifold diagnostics.

The scalable proposal represents a spin state as a coherent sum of valence-
bond coverings.  Every covering is a product of Condon--Shortley singlets, so
spin ``SU(2)`` is exact before learning.  Equal coefficients on a physical
space-group orbit give an exact ``Gamma-A1`` state without a Reynolds sum in
the learned forward pass.

This module is deliberately an exact-small *manifold oracle*.  It enumerates
all perfect matchings of a registered bond graph, quotients them by the
physical space group, and asks whether their span contains a useful direction
beyond a carrier's two-Lanczos space.  It contains no neural network and no
claim of scalable covering enumeration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..lattices.spatial_symmetry import SitePermutation, validate_site_permutation


WeightedBond = tuple[int, int, float]
Dimer = tuple[int, int]
DimerMatching = tuple[Dimer, ...]


def _canonical_matching(
    matching: Sequence[Sequence[int]], *, n_sites: int | None = None
) -> DimerMatching:
    """Return a checked, lexicographically ordered perfect matching."""

    pairs: list[Dimer] = []
    for pair in matching:
        raw = tuple(pair)
        if len(raw) != 2 or any(
            isinstance(site, bool) or not isinstance(site, Integral) for site in raw
        ):
            raise TypeError("every dimer must contain two integer sites")
        first, second = sorted((int(raw[0]), int(raw[1])))
        if first == second or first < 0:
            raise ValueError("a dimer must join two distinct nonnegative sites")
        pairs.append((first, second))
    result = tuple(sorted(pairs))
    flattened = tuple(site for pair in result for site in pair)
    if len(set(flattened)) != len(flattened):
        raise ValueError("dimers in a matching must be site-disjoint")
    inferred = len(flattened) if n_sites is None else int(n_sites)
    if inferred < 2 or inferred % 2 or set(flattened) != set(range(inferred)):
        raise ValueError("a perfect matching must cover sites 0..N-1 exactly once")
    return result


def enumerate_perfect_matchings(
    n_sites: int,
    bonds: Sequence[WeightedBond | Sequence[float]],
) -> tuple[DimerMatching, ...]:
    """Enumerate every perfect matching of the undirected bond support.

    Coupling values are ignored: only nonzero graph edges define admissible
    dimers.  The least uncovered site is branched first, so output ordering is
    deterministic.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    n_sites = int(n_sites)
    if n_sites < 2 or n_sites % 2:
        raise ValueError("perfect matchings require a positive even site count")
    adjacency = [set() for _ in range(n_sites)]
    for bond in bonds:
        raw = tuple(bond)
        if len(raw) != 3:
            raise ValueError("weighted bonds must be (first,second,coupling)")
        first, second = int(raw[0]), int(raw[1])
        coupling = complex(raw[2])
        if not 0 <= first < second < n_sites:
            raise ValueError("bond sites must satisfy 0 <= first < second < N")
        if not np.isfinite(coupling.real) or not np.isfinite(coupling.imag):
            raise ValueError("bond couplings must be finite")
        if abs(coupling) > 0.0:
            adjacency[first].add(second)
            adjacency[second].add(first)

    output: list[DimerMatching] = []

    def visit(remaining: frozenset[int], chosen: tuple[Dimer, ...]) -> None:
        if not remaining:
            output.append(tuple(sorted(chosen)))
            return
        first = min(remaining)
        for second in sorted(adjacency[first] & remaining):
            visit(
                remaining - {first, second},
                chosen + ((min(first, second), max(first, second)),),
            )

    visit(frozenset(range(n_sites)), ())
    return tuple(sorted(set(output)))


def transform_matching(
    matching: Sequence[Sequence[int]], image: Sequence[int]
) -> DimerMatching:
    """Apply one active physical site permutation to a perfect matching."""

    checked = _canonical_matching(matching)
    permutation = validate_site_permutation(image, n_sites=2 * len(checked))
    return tuple(
        sorted(
            tuple(sorted((permutation[first], permutation[second])))
            for first, second in checked
        )
    )


def transform_oriented_matching(
    matching: Sequence[Sequence[int]], image: Sequence[int]
) -> tuple[DimerMatching, int]:
    """Transform an oriented singlet matching and return its gauge sign.

    Each canonical dimer ``(i,j)`` with ``i<j`` denotes the ordered singlet
    ``(|up_i down_j>-|down_i up_j>)/sqrt(2)``.  A physical permutation that
    reverses the endpoint order of a dimer therefore contributes one minus
    sign when the image is restored to canonical site order.
    """

    checked = _canonical_matching(matching)
    permutation = validate_site_permutation(image, n_sites=2 * len(checked))
    pairs: list[Dimer] = []
    sign = 1
    for first, second in checked:
        mapped_first = permutation[first]
        mapped_second = permutation[second]
        if mapped_first > mapped_second:
            sign = -sign
        pairs.append(tuple(sorted((mapped_first, mapped_second))))
    return tuple(sorted(pairs)), sign


@dataclass(frozen=True)
class DimerMatchingOrbit:
    """One exact space-group orbit of perfect matchings."""

    representative: DimerMatching
    members: tuple[DimerMatching, ...]
    member_signs: tuple[int, ...]
    stabilizer_order: int

    @property
    def orbit_size(self) -> int:
        return len(self.members)

    @property
    def has_a1_vector(self) -> bool:
        """Whether the oriented orbit has a nonzero trivial-irrep sum."""

        return len(self.member_signs) == len(self.members)


def dimer_matching_orbits(
    matchings: Sequence[Sequence[Sequence[int]]],
    group: Sequence[SitePermutation | Sequence[int]],
) -> tuple[DimerMatchingOrbit, ...]:
    """Partition a complete matching set into physical space-group orbits."""

    concrete = tuple(_canonical_matching(matching) for matching in matchings)
    if not concrete:
        raise ValueError("at least one matching is required")
    if len(set(concrete)) != len(concrete):
        raise ValueError("matchings must be duplicate free")
    n_sites = 2 * len(concrete[0])
    if any(2 * len(matching) != n_sites for matching in concrete):
        raise ValueError("all matchings must have the same site count")
    images = tuple(validate_site_permutation(image, n_sites=n_sites) for image in group)
    if not images:
        raise ValueError("space group must be nonempty")
    allowed = set(concrete)
    unseen = set(concrete)
    output: list[DimerMatchingOrbit] = []
    while unseen:
        seed = min(unseen)
        transformed = tuple(transform_oriented_matching(seed, image) for image in images)
        members = tuple(sorted({matching for matching, _sign in transformed}))
        if not set(members) <= allowed:
            raise ValueError("space group does not preserve the matching support")
        stabilizer = sum(matching == seed for matching, _sign in transformed)
        if len(members) * stabilizer != len(images):
            raise RuntimeError("matching orbit violates orbit-stabilizer")
        signs_by_member: dict[DimerMatching, set[int]] = {
            member: set() for member in members
        }
        for member, sign in transformed:
            signs_by_member[member].add(sign)
        if all(len(signs_by_member[member]) == 1 for member in members):
            member_signs = tuple(next(iter(signs_by_member[member])) for member in members)
        else:
            # A stabilizer carrying the sign representation annihilates the
            # trivial-irrep projection of this oriented covering orbit.
            member_signs = ()
        unseen.difference_update(members)
        output.append(
            DimerMatchingOrbit(min(members), members, member_signs, stabilizer)
        )
    result = tuple(sorted(output, key=lambda orbit: orbit.representative))
    if sum(orbit.orbit_size for orbit in result) != len(concrete):
        raise RuntimeError("matching orbits do not partition the support")
    return result


def fixed_m_dimer_amplitudes(
    configurations: Sequence[Sequence[int]] | np.ndarray,
    matching: Sequence[Sequence[int]],
) -> np.ndarray:
    """Evaluate one Condon--Shortley dimer product on fixed-M configurations."""

    states = np.asarray(configurations)
    checked = _canonical_matching(matching)
    n_sites = 2 * len(checked)
    if states.ndim != 2 or states.shape[1] != n_sites:
        raise ValueError(f"configurations must have shape (batch,{n_sites})")
    if np.any((states != 0) & (states != 1)):
        raise ValueError("spin configurations must be binary")
    result = np.ones(states.shape[0], dtype=np.float64)
    for first, second in checked:
        first_bits = states[:, first]
        second_bits = states[:, second]
        result *= np.where(
            first_bits == second_bits,
            0.0,
            np.where(first_bits == 0, -1.0, 1.0),
        )
    return result * 2.0 ** (-len(checked) / 2.0)


@dataclass(frozen=True)
class RVBOrbitColumns:
    """Normalized physical columns for equal-amplitude matching orbits."""

    columns: np.ndarray
    raw_norms: np.ndarray
    orbit_sizes: tuple[int, ...]


@dataclass(frozen=True)
class CoveringRelationFeatures:
    """Discrete space-group invariant features for matching bond tokens."""

    bond_classes: np.ndarray
    relation_classes: np.ndarray
    bond_class_count: int
    relation_class_count: int


def covering_relation_features(
    matchings: Sequence[Sequence[Sequence[int]]],
    bonds: Sequence[WeightedBond | Sequence[float]],
    group: Sequence[SitePermutation | Sequence[int]],
) -> CoveringRelationFeatures:
    """Compile node and ordered-pair orbit labels for covering tokens.

    The labels are invariant under the supplied physical space group.  Under a
    group action the token arrays can only be permuted, so a shared-attention
    network followed by invariant pooling is exactly ``Gamma-A1`` without
    evaluating a group average in its forward pass.
    """

    concrete = tuple(_canonical_matching(matching) for matching in matchings)
    if not concrete:
        raise ValueError("at least one matching is required")
    n_sites = 2 * len(concrete[0])
    images = tuple(validate_site_permutation(image, n_sites=n_sites) for image in group)
    if not images:
        raise ValueError("space group must be nonempty")
    edge_values: set[Dimer] = set()
    for bond in bonds:
        raw = tuple(bond)
        if len(raw) != 3:
            raise ValueError("weighted bonds must be (first,second,coupling)")
        first, second = int(raw[0]), int(raw[1])
        if not 0 <= first < second < n_sites:
            raise ValueError("bond sites must satisfy 0 <= first < second < N")
        coupling = complex(raw[2])
        if not np.isfinite(coupling.real) or not np.isfinite(coupling.imag):
            raise ValueError("bond couplings must be finite")
        if abs(coupling) > 0.0:
            edge_values.add((first, second))
    edges = tuple(sorted(edge_values))
    if not edges:
        raise ValueError("bond support must be nonempty")
    edge_set = set(edges)

    def image_edge(edge: Dimer, image: SitePermutation) -> Dimer:
        return tuple(sorted((image[edge[0]], image[edge[1]])))

    bond_representatives = {
        edge: min(image_edge(edge, image) for image in images) for edge in edges
    }
    bond_labels = {
        representative: index
        for index, representative in enumerate(sorted(set(bond_representatives.values())))
    }
    bond_class = {
        edge: bond_labels[bond_representatives[edge]] for edge in edges
    }

    pair_representatives: dict[tuple[Dimer, Dimer], tuple[Dimer, Dimer]] = {}
    for first in edges:
        for second in edges:
            pair_representatives[(first, second)] = min(
                (image_edge(first, image), image_edge(second, image))
                for image in images
            )
    pair_labels = {
        representative: index
        for index, representative in enumerate(sorted(set(pair_representatives.values())))
    }
    pair_class = {
        pair: pair_labels[representative]
        for pair, representative in pair_representatives.items()
    }

    node_rows = []
    relation_rows = []
    for matching in concrete:
        if not set(matching) <= edge_set:
            raise ValueError("a matching contains a dimer outside the bond support")
        node_rows.append([bond_class[edge] for edge in matching])
        relation_rows.append(
            [
                [pair_class[(first, second)] for second in matching]
                for first in matching
            ]
        )
    nodes = np.asarray(node_rows, dtype=np.int32)
    relations = np.asarray(relation_rows, dtype=np.int32)
    nodes.setflags(write=False)
    relations.setflags(write=False)
    return CoveringRelationFeatures(
        bond_classes=nodes,
        relation_classes=relations,
        bond_class_count=len(bond_labels),
        relation_class_count=len(pair_labels),
    )


def rvb_orbit_columns(
    configurations: Sequence[Sequence[int]] | np.ndarray,
    orbits: Sequence[DimerMatchingOrbit],
) -> RVBOrbitColumns:
    """Build normalized ``Gamma-A1`` columns, one per matching orbit."""

    if not orbits:
        raise ValueError("at least one matching orbit is required")
    columns = []
    norms = []
    for orbit in orbits:
        if not orbit.has_a1_vector:
            continue
        vector = sum(
            (
                sign * fixed_m_dimer_amplitudes(configurations, matching)
                for matching, sign in zip(
                    orbit.members, orbit.member_signs, strict=True
                )
            ),
            start=np.zeros(len(configurations), dtype=np.float64),
        )
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-14 or not np.isfinite(norm):
            raise RuntimeError("an RVB orbit column is numerically empty")
        columns.append(vector / norm)
        norms.append(norm)
    return RVBOrbitColumns(
        columns=np.stack(columns, axis=1),
        raw_norms=np.asarray(norms),
        orbit_sizes=tuple(orbit.orbit_size for orbit in orbits),
    )


@dataclass(frozen=True)
class RitzState:
    """Lowest state in a possibly nonorthogonal variational column span."""

    state: np.ndarray
    energy: float
    variance: float
    rank: int
    singular_values: np.ndarray
    coefficients: np.ndarray


def lowest_ritz_state(
    columns: np.ndarray,
    hamiltonian,
    *,
    rank_tolerance: float = 1.0e-12,
) -> RitzState:
    """Return the normalized lowest-energy state in ``span(columns)``."""

    matrix = np.asarray(columns, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("columns must be a nonempty rank-two array")
    if not np.isfinite(rank_tolerance) or not 0.0 < rank_tolerance < 1.0:
        raise ValueError("rank_tolerance must lie in (0,1)")
    left, singular, right_h = np.linalg.svd(matrix, full_matrices=False)
    if singular[0] <= 0.0:
        raise ValueError("variational columns have zero rank")
    rank = int(np.count_nonzero(singular > singular[0] * rank_tolerance))
    basis = left[:, :rank]
    projected = basis.conj().T @ (hamiltonian @ basis)
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = np.linalg.eigh(projected)
    reduced_coefficients = vectors[:, 0]
    state = basis @ reduced_coefficients
    state /= np.linalg.norm(state)
    coefficients = right_h[:rank].conj().T @ (
        reduced_coefficients / singular[:rank]
    )
    # Match the normalization applied to the physical state.  Roundoff in the
    # SVD basis is the only reason this factor is not exactly one.
    reconstructed = matrix @ coefficients
    coefficients = coefficients / np.linalg.norm(reconstructed)
    action = np.asarray(hamiltonian @ state)
    energy = float(np.vdot(state, action).real)
    residual = action - energy * state
    variance = float(np.vdot(residual, residual).real)
    return RitzState(state, energy, variance, rank, singular, coefficients)


def two_lanczos_state(carrier: np.ndarray, hamiltonian) -> RitzState:
    """Return the degree-two Hamiltonian Krylov optimum of one carrier."""

    state = np.asarray(carrier, dtype=np.complex128)
    norm = float(np.linalg.norm(state))
    if norm <= 1.0e-14 or not np.isfinite(norm):
        raise ValueError("carrier must have nonzero finite norm")
    state = state / norm
    first = np.asarray(hamiltonian @ state)
    second = np.asarray(hamiltonian @ first)
    return lowest_ritz_state(np.column_stack((state, first, second)), hamiltonian)


__all__ = [
    "DimerMatching",
    "DimerMatchingOrbit",
    "CoveringRelationFeatures",
    "RVBOrbitColumns",
    "RitzState",
    "dimer_matching_orbits",
    "covering_relation_features",
    "enumerate_perfect_matchings",
    "fixed_m_dimer_amplitudes",
    "lowest_ritz_state",
    "rvb_orbit_columns",
    "transform_matching",
    "transform_oriented_matching",
    "two_lanczos_state",
]
