"""Dense exact-diagonalization references for small spin-1/2 systems."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def computational_basis(n: int, n_up: int | None = None) -> tuple[int, ...]:
    if n < 1:
        raise ValueError("N must be positive")
    if n_up is not None and not 0 <= n_up <= n:
        raise ValueError("n_up must lie between zero and N")
    return tuple(
        state
        for state in range(1 << n)
        if n_up is None or state.bit_count() == n_up
    )


def _bit(state: int, site: int, n: int) -> int:
    return (state >> (n - site - 1)) & 1


def heisenberg_sz_matrix(
    n: int,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]] | None = None,
    *,
    n_up: int | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Dense ``sum J_ij S_i.S_j`` in the full or fixed-magnetization basis."""

    if bonds is None:
        normalized_bonds = tuple((site, site + 1, 1.0) for site in range(n - 1))
    else:
        normalized = []
        for bond in bonds:
            if len(bond) == 2:
                left, right = bond
                coupling = 1.0
            elif len(bond) == 3:
                left, right, coupling = bond
            else:
                raise ValueError(f"invalid bond: {bond!r}")
            if not (0 <= left < right < n):
                raise ValueError(f"bond sites must satisfy 0 <= i < j < N: {bond!r}")
            normalized.append((int(left), int(right), float(coupling)))
        normalized_bonds = tuple(normalized)

    basis = computational_basis(n, n_up)
    positions = {state: position for position, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
    for column, state in enumerate(basis):
        for left, right, coupling in normalized_bonds:
            left_bit, right_bit = _bit(state, left, n), _bit(state, right, n)
            matrix[column, column] += coupling * (0.25 if left_bit == right_bit else -0.25)
            if left_bit != right_bit:
                left_mask = 1 << (n - left - 1)
                right_mask = 1 << (n - right - 1)
                flipped = state ^ left_mask ^ right_mask
                row = positions.get(flipped)
                if row is not None:
                    matrix[row, column] += 0.5 * coupling
    return matrix, basis


def spin_squared_matrix(n: int, *, n_up: int | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
    """Dense total-spin ``S^2`` in the requested computational sector."""

    all_pairs = tuple((left, right, 1.0) for left in range(n) for right in range(left + 1, n))
    pair_sum, basis = heisenberg_sz_matrix(n, all_pairs, n_up=n_up)
    return 0.75 * n * np.eye(len(basis)) + 2.0 * pair_sum, basis


def lowest_eigenpair(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Lowest Hermitian eigenpair."""

    values, vectors = np.linalg.eigh(matrix)
    return float(values[0]), vectors[:, 0]

