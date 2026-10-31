"""Exact short-range reference states for triangular-lattice locality tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..su2.fusion_path import enumerate_paths
from .coupled_basis import coupled_basis_matrix


WeightedBond = tuple[int, int, float]


def nearest_neighbor_dimer_cover(
    n: int, bonds: Sequence[WeightedBond]
) -> tuple[tuple[int, int], ...]:
    """Return the lexicographically first perfect matching of a bond graph."""

    if n < 2 or n % 2:
        raise ValueError("a dimer covering requires a positive even site count")
    adjacency = {site: set() for site in range(n)}
    for first, second, _ in bonds:
        if not 0 <= first < second < n:
            raise ValueError("bond sites must satisfy 0 <= i < j < N")
        adjacency[first].add(second)
        adjacency[second].add(first)

    def visit(remaining: frozenset[int]):
        if not remaining:
            return ()
        first = min(remaining)
        for second in sorted(adjacency[first] & remaining):
            rest = visit(remaining - {first, second})
            if rest is not None:
                return ((first, second),) + rest
        return None

    matching = visit(frozenset(range(n)))
    if matching is None:
        raise ValueError("bond graph has no perfect matching")
    return matching


def physical_dimer_product(
    n: int, matching: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Return a product of Condon-Shortley singlets in the computational basis."""

    pairs = tuple(matching)
    flattened = tuple(site for pair in pairs for site in pair)
    if len(pairs) * 2 != n or sorted(flattened) != list(range(n)):
        raise ValueError("matching must contain every site exactly once")
    vector = np.zeros(1 << n, dtype=np.complex128)
    normalization = 2.0 ** (-len(pairs) / 2.0)
    for state in range(1 << n):
        sign = 1.0
        for first, second in pairs:
            first_bit = (state >> (n - first - 1)) & 1
            second_bit = (state >> (n - second - 1)) & 1
            if first_bit == second_bit:
                sign = 0.0
                break
            # |up,down> - |down,up>, with site zero the most significant bit.
            if first_bit == 0:
                sign = -sign
        vector[state] = normalization * sign
    return vector


def dimer_state_in_fusion_basis(
    n: int, matching: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, float]:
    """Project a physical dimer product into the left total-singlet basis.

    The second return value is the computational-basis projection residual and
    should be at roundoff because a product of singlets is itself a singlet.
    """

    paths = enumerate_paths(n, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    physical = physical_dimer_product(n, matching)
    fusion = transform.conj().T @ physical
    residual = float(np.linalg.norm(transform @ fusion - physical))
    fusion /= np.linalg.norm(fusion)
    return fusion, residual
