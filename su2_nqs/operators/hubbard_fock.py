"""Physical-basis Hubbard connectivity in canonical interleaved modes.

Mode ``2*i`` is ``(site i, up)`` and mode ``2*i+1`` is ``(site i, down)``.
The implementation is independent of the direct-fusion Hubbard operator and
is intended as the production local-move layer for HubbardCG amplitudes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from numbers import Integral
from typing import Any

import numpy as np

from ..su2.hubbard_cg_irrep import DOUBLON, DOWN, EMPTY, UP, validate_hubbard_token


HubbardBond = tuple[int, int] | tuple[int, int, float]


def tokens_to_fock_mask(tokens: Sequence[int]) -> int:
    """Encode physical tokens as a canonical interleaved-orbital bit mask."""

    mask = 0
    for site, value in enumerate(tokens):
        token = validate_hubbard_token(value)
        if token in (UP, DOUBLON):
            mask |= 1 << (2 * site)
        if token in (DOWN, DOUBLON):
            mask |= 1 << (2 * site + 1)
    return mask


def fock_mask_to_tokens(mask: int, n_sites: int) -> tuple[int, ...]:
    """Decode a canonical bit mask into physical tokens."""

    if isinstance(mask, bool) or not isinstance(mask, Integral) or mask < 0:
        raise ValueError("mask must be a non-negative integer")
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if int(mask) >> (2 * int(n_sites)):
        raise ValueError("mask has occupied modes outside n_sites")
    tokens = []
    for site in range(int(n_sites)):
        up = (int(mask) >> (2 * site)) & 1
        down = (int(mask) >> (2 * site + 1)) & 1
        tokens.append((EMPTY, UP, DOWN, DOUBLON)[up + 2 * down])
    return tuple(tokens)


def enumerate_hubbard_tokens(
    n_sites: int,
    n_electrons: int,
    *,
    total_m2: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate one fixed ``(Ne, 2*Sz)`` physical basis deterministically."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if (
        isinstance(n_electrons, bool)
        or not isinstance(n_electrons, Integral)
        or not 0 <= n_electrons <= 2 * n_sites
    ):
        raise ValueError("n_electrons must lie between 0 and 2*n_sites")
    if isinstance(total_m2, bool) or not isinstance(total_m2, Integral):
        raise TypeError("total_m2 must be an integer doubled projection")
    if (n_electrons + total_m2) % 2:
        return ()
    n_up = (int(n_electrons) + int(total_m2)) // 2
    n_down = int(n_electrons) - n_up
    if not (0 <= n_up <= n_sites and 0 <= n_down <= n_sites):
        return ()
    masks = []
    for up_sites in combinations(range(int(n_sites)), n_up):
        up_mask = sum(1 << (2 * site) for site in up_sites)
        for down_sites in combinations(range(int(n_sites)), n_down):
            masks.append(up_mask | sum(1 << (2 * site + 1) for site in down_sites))
    return tuple(fock_mask_to_tokens(mask, int(n_sites)) for mask in sorted(masks))


def apply_canonical_hop(mask: int, destination: int, source: int) -> tuple[int, int] | None:
    """Apply ``c_destination^dagger c_source`` using interval parity."""

    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in (mask, destination, source)
    ):
        raise ValueError("mask and orbital indices must be non-negative integers")
    mask = int(mask)
    destination = int(destination)
    source = int(source)
    if destination == source or not (mask >> source) & 1 or (mask >> destination) & 1:
        return None
    lower, upper = sorted((destination, source))
    between = ((1 << upper) - 1) ^ ((1 << (lower + 1)) - 1)
    sign = -1 if (mask & between).bit_count() % 2 else 1
    return mask ^ (1 << source) ^ (1 << destination), sign


def apply_site_permutation(
    tokens: Sequence[int], image: Sequence[int]
) -> tuple[tuple[int, ...], int]:
    """Apply a site permutation and its canonical fermionic parity.

    ``image[old_site]`` is the new site.  Spin labels are unchanged.  The
    returned sign is the parity needed to reorder the mapped creation
    operators back into increasing canonical mode order.
    """

    old = tuple(validate_hubbard_token(token) for token in tokens)
    n_sites = len(old)
    permutation = tuple(int(site) for site in image)
    if len(permutation) != n_sites or sorted(permutation) != list(range(n_sites)):
        raise ValueError("image must be a permutation of the site indices")
    mask = tokens_to_fock_mask(old)
    occupied = [mode for mode in range(2 * n_sites) if (mask >> mode) & 1]
    mapped = [2 * permutation[mode // 2] + mode % 2 for mode in occupied]
    inversions = sum(
        mapped[left] > mapped[right]
        for left in range(len(mapped))
        for right in range(left + 1, len(mapped))
    )
    new_mask = sum(1 << mode for mode in mapped)
    return fock_mask_to_tokens(new_mask, n_sites), (-1 if inversions % 2 else 1)


def _checked_bonds(n_sites: int, bonds: Iterable[HubbardBond]) -> tuple[tuple[int, int, float], ...]:
    checked = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            hopping = 1.0
        elif len(bond) == 3:
            first, second, hopping = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        first, second = int(first), int(second)
        if first == second or not (0 <= first < n_sites and 0 <= second < n_sites):
            raise ValueError(f"invalid Hubbard bond: {bond!r}")
        checked.append((first, second, float(hopping)))
    return tuple(checked)


def hubbard_fock_connectivity(
    tokens: Sequence[int],
    bonds: Iterable[HubbardBond],
    *,
    interaction: float,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Return combined states and ``H[new, old]`` matrix elements."""

    old = tuple(validate_hubbard_token(token) for token in tokens)
    n_sites = len(old)
    if n_sites < 1:
        raise ValueError("tokens must contain at least one site")
    mask = tokens_to_fock_mask(old)
    elements: dict[tuple[int, ...], float] = {
        old: float(interaction) * sum(token == DOUBLON for token in old)
    }
    for first, second, hopping in _checked_bonds(n_sites, bonds):
        for spin in (0, 1):
            first_mode = 2 * first + spin
            second_mode = 2 * second + spin
            for destination, source in ((first_mode, second_mode), (second_mode, first_mode)):
                result = apply_canonical_hop(mask, destination, source)
                if result is None:
                    continue
                new_mask, sign = result
                new = fock_mask_to_tokens(new_mask, n_sites)
                elements[new] = elements.get(new, 0.0) - hopping * sign
    states = tuple(sorted(elements, key=tokens_to_fock_mask))
    return states, np.asarray([elements[state] for state in states], dtype=np.float64)


def hubbard_fock_sector_matrix(
    n_sites: int,
    n_electrons: int,
    total_m2: int,
    bonds: Iterable[HubbardBond],
    *,
    interaction: float,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Build a dense reference-sized matrix through production connectivity."""

    basis = enumerate_hubbard_tokens(n_sites, n_electrons, total_m2=total_m2)
    positions = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
    checked_bonds = _checked_bonds(n_sites, bonds)
    for column, state in enumerate(basis):
        connected, elements = hubbard_fock_connectivity(
            state, checked_bonds, interaction=interaction
        )
        for new, element in zip(connected, elements, strict=True):
            matrix[positions[new], column] += element
    return matrix, basis


def hubbard_fock_sector_sparse(
    n_sites: int,
    n_electrons: int,
    total_m2: int,
    bonds: Iterable[HubbardBond],
    *,
    interaction: float,
) -> tuple[Any, tuple[tuple[int, ...], ...]]:
    """Build a SciPy CSR matrix through production physical connectivity.

    Unlike :func:`hubbard_fock_sector_matrix`, this constructor is suitable
    for the 2x4 exact anchor and other sectors where allocating a dense
    physical-basis Hamiltonian would obscure the actual local sparsity.
    """

    from scipy import sparse

    basis = enumerate_hubbard_tokens(n_sites, n_electrons, total_m2=total_m2)
    positions = {state: index for index, state in enumerate(basis)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    checked_bonds = _checked_bonds(n_sites, bonds)
    for column, state in enumerate(basis):
        connected, elements = hubbard_fock_connectivity(
            state, checked_bonds, interaction=interaction
        )
        for new, element in zip(connected, elements, strict=True):
            if element:
                rows.append(positions[new])
                columns.append(column)
                values.append(float(element))
    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(basis), len(basis)),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix, basis


__all__ = [
    "HubbardBond",
    "apply_canonical_hop",
    "apply_site_permutation",
    "enumerate_hubbard_tokens",
    "fock_mask_to_tokens",
    "hubbard_fock_connectivity",
    "hubbard_fock_sector_matrix",
    "hubbard_fock_sector_sparse",
    "tokens_to_fock_mask",
]
