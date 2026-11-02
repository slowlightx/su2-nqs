"""Hamiltonian-blind, space-group averaged two-bond carrier controls.

The functions in this module construct a control bank for symmetry-labelled
carrier studies.  A source is an unordered pair of physical Heisenberg bonds.
It is made Hermitian with an anticommutator and then averaged over a supplied
finite site-permutation group.  No Hamiltonian coefficients or energies enter
the construction.

Because every factor is a spin scalar, applying a carrier preserves total
spin.  The Reynolds average makes the resulting operator transform in the
trivial spatial irrep.  These controls are therefore suitable competitors to
analytic ``A1`` and ``E2 x E2 -> A1`` bond-pair carriers without relaxing the
exact ``SU(2)`` or lattice-sector contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


Bond = tuple[int, int]
PairAction = Callable[[np.ndarray, int, int], np.ndarray]
SitePermutation = tuple[int, ...]


def canonical_bond(bond: Sequence[int], n_sites: int) -> Bond:
    """Return a validated unoriented bond in deterministic endpoint order."""

    if len(bond) != 2:
        raise ValueError("a bond must contain exactly two endpoints")
    first, second = (int(site) for site in bond)
    if first == second or not 0 <= first < n_sites or not 0 <= second < n_sites:
        raise ValueError("bond endpoints must be distinct valid sites")
    return (first, second) if first < second else (second, first)


def hermitian_bond_pair_action(
    state: np.ndarray,
    pair_action: PairAction,
    first_bond: Sequence[int],
    second_bond: Sequence[int],
) -> np.ndarray:
    """Apply ``{B_first,B_second}/2`` to one state vector."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    first = tuple(int(site) for site in first_bond)
    second = tuple(int(site) for site in second_bond)
    if len(first) != 2 or len(second) != 2:
        raise ValueError("both bonds must contain exactly two endpoints")
    forward = np.asarray(
        pair_action(
            pair_action(vector, second[0], second[1]), first[0], first[1]
        ),
        dtype=np.complex128,
    )
    reverse = np.asarray(
        pair_action(
            pair_action(vector, first[0], first[1]), second[0], second[1]
        ),
        dtype=np.complex128,
    )
    if forward.shape != vector.shape or reverse.shape != vector.shape:
        raise ValueError("pair action returned the wrong state shape")
    return 0.5 * (forward + reverse)


def bond_pair_orbit(
    first_bond: Sequence[int],
    second_bond: Sequence[int],
    site_permutations: Sequence[Sequence[int]],
) -> tuple[tuple[Bond, Bond], ...]:
    """Return the multiplicity-free orbit of an unordered pair of bonds."""

    permutations = tuple(tuple(int(site) for site in row) for row in site_permutations)
    if not permutations:
        raise ValueError("site_permutations must be nonempty")
    n_sites = len(permutations[0])
    expected = tuple(range(n_sites))
    if any(tuple(sorted(row)) != expected for row in permutations):
        raise ValueError("every site permutation must contain each site once")
    first = canonical_bond(first_bond, n_sites)
    second = canonical_bond(second_bond, n_sites)
    images: set[tuple[Bond, Bond]] = set()
    for permutation in permutations:
        left = canonical_bond(
            (permutation[first[0]], permutation[first[1]]), n_sites
        )
        right = canonical_bond(
            (permutation[second[0]], permutation[second[1]]), n_sites
        )
        images.add(tuple(sorted((left, right))))
    return tuple(sorted(images))


def reynolds_bond_pair_action(
    state: np.ndarray,
    pair_action: PairAction,
    first_bond: Sequence[int],
    second_bond: Sequence[int],
    site_permutations: Sequence[Sequence[int]],
) -> np.ndarray:
    """Apply a normalized trivial-irrep Reynolds average of one bond pair.

    Stabilizer multiplicities cancel in the group average, so averaging the
    distinct orbit images is exactly equivalent and avoids redundant work.
    """

    vector = np.asarray(state, dtype=np.complex128)
    orbit = bond_pair_orbit(first_bond, second_bond, site_permutations)
    answer = np.zeros_like(vector)
    for first, second in orbit:
        answer += hermitian_bond_pair_action(vector, pair_action, first, second)
    return answer / float(len(orbit))


def connected_gram_schmidt(
    state: np.ndarray,
    directions: Sequence[np.ndarray],
    *,
    tolerance: float = 1.0e-11,
) -> tuple[np.ndarray, ...]:
    """Connect and sequentially whiten directions while preserving prefixes."""

    reference = np.asarray(state, dtype=np.complex128)
    if reference.ndim != 1 or np.any(~np.isfinite(reference)):
        raise ValueError("state must be a finite vector")
    norm = float(np.linalg.norm(reference))
    if norm <= tolerance:
        raise ValueError("state has zero numerical norm")
    reference = reference / norm
    basis: list[np.ndarray] = []
    for raw in directions:
        vector = np.asarray(raw, dtype=np.complex128)
        if vector.shape != reference.shape or np.any(~np.isfinite(vector)):
            raise ValueError("every direction must be finite and match the state")
        vector = vector - np.vdot(reference, vector) * reference
        for prior in basis:
            vector = vector - np.vdot(prior, vector) * prior
        residual = float(np.linalg.norm(vector))
        if residual > tolerance:
            basis.append(vector / residual)
    return tuple(basis)


__all__ = [
    "Bond",
    "bond_pair_orbit",
    "canonical_bond",
    "connected_gram_schmidt",
    "hermitian_bond_pair_action",
    "reynolds_bond_pair_action",
]
