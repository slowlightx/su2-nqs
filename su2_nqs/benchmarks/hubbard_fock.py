"""Independent Fock-space references for small Hubbard clusters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from numbers import Integral

import numpy as np


FockBond = tuple[int, int] | tuple[int, int, float]


def enumerate_fock_basis(n_sites: int, n_electrons: int) -> tuple[int, ...]:
    """Return sorted bit masks for fixed particle number.

    Orbital ``2*i`` is site ``i`` spin up and orbital ``2*i+1`` is spin down.
    The occupied creation operators defining each basis ket are ordered by
    increasing orbital index.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if (
        isinstance(n_electrons, bool)
        or not isinstance(n_electrons, Integral)
        or not 0 <= n_electrons <= 2 * n_sites
    ):
        raise ValueError("n_electrons must lie between 0 and 2*n_sites")
    masks = [sum(1 << orbital for orbital in occupied) for occupied in combinations(range(2 * int(n_sites)), int(n_electrons))]
    return tuple(sorted(masks))


def apply_fermion_hop(mask: int, destination: int, source: int) -> tuple[int, int] | None:
    """Apply ``c_destination^dagger c_source`` and return ``(mask, sign)``."""

    if min(mask, destination, source) < 0:
        raise ValueError("mask and orbital indices must be non-negative")
    if not (mask >> source) & 1 or (mask >> destination) & 1:
        return None
    annihilation_sign = -1 if (mask & ((1 << source) - 1)).bit_count() % 2 else 1
    removed = mask ^ (1 << source)
    creation_sign = -1 if (removed & ((1 << destination) - 1)).bit_count() % 2 else 1
    return removed | (1 << destination), annihilation_sign * creation_sign


def _checked_bonds(n_sites: int, bonds: Iterable[FockBond]) -> tuple[tuple[int, int, float], ...]:
    checked = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            hopping = 1.0
        elif len(bond) == 3:
            first, second, hopping = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        first = int(first)
        second = int(second)
        if first == second or not (0 <= first < n_sites and 0 <= second < n_sites):
            raise ValueError(f"invalid Hubbard bond: {bond!r}")
        checked.append((first, second, float(hopping)))
    return tuple(checked)


def hubbard_fock_matrix(
    n_sites: int,
    n_electrons: int,
    bonds: Sequence[FockBond],
    *,
    interaction: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build ``-sum t_ij(c_i^dagger c_j+h.c.) + U sum n_up n_down``."""

    basis = enumerate_fock_basis(n_sites, n_electrons)
    positions = {mask: index for index, mask in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
    checked_bonds = _checked_bonds(n_sites, bonds)
    for column, mask in enumerate(basis):
        doublons = sum(
            bool((mask >> (2 * site)) & 1) and bool((mask >> (2 * site + 1)) & 1)
            for site in range(n_sites)
        )
        matrix[column, column] += float(interaction) * doublons
        for first, second, hopping in checked_bonds:
            for spin in (0, 1):
                first_orbital = 2 * first + spin
                second_orbital = 2 * second + spin
                for destination, source in (
                    (first_orbital, second_orbital),
                    (second_orbital, first_orbital),
                ):
                    result = apply_fermion_hop(mask, destination, source)
                    if result is None:
                        continue
                    new_mask, sign = result
                    matrix[positions[new_mask], column] += -hopping * sign
    return matrix, basis


__all__ = [
    "FockBond",
    "apply_fermion_hop",
    "enumerate_fock_basis",
    "hubbard_fock_matrix",
]
