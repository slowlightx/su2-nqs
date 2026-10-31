"""Sparse graph Heisenberg Hamiltonian in the full computational basis."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def heisenberg_computational_graph_sparse(
    n: int,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
):
    """Build sparse ``sum J_ij S_i.S_j`` on all ``2**n`` bit strings.

    Bit strings follow the repository convention: site zero is the most
    significant bit, zero denotes ``m=-1/2``, and one denotes ``m=+1/2``.
    This is the native Hamiltonian required by a genuinely symmetry-blind
    autoregressive control; it must not be replaced by an ``Sz=0`` operator.
    """

    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("sparse computational graph operators require scipy") from error
    if n < 1:
        raise ValueError("n must be positive")
    normalized: list[tuple[int, int, float]] = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        first, second = int(first), int(second)
        if first > second:
            first, second = second, first
        if not 0 <= first < second < n:
            raise ValueError(f"bond sites must satisfy 0 <= i < j < N: {bond!r}")
        normalized.append((first, second, float(coupling)))
    basis = tuple(
        tuple((state >> (n - site - 1)) & 1 for site in range(n))
        for state in range(1 << n)
    )

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for column, configuration in enumerate(basis):
        diagonal = 0.0
        for first, second, coupling in normalized:
            first_bit, second_bit = configuration[first], configuration[second]
            diagonal += coupling * (0.25 if first_bit == second_bit else -0.25)
            if first_bit != second_bit:
                flipped = list(configuration)
                flipped[first], flipped[second] = flipped[second], flipped[first]
                row = sum(
                    bit << (n - site - 1) for site, bit in enumerate(flipped)
                )
                rows.append(row)
                columns.append(column)
                values.append(0.5 * coupling)
        rows.append(column)
        columns.append(column)
        values.append(diagonal)
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(1 << n, 1 << n), dtype=np.float64
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, basis


__all__ = ["heisenberg_computational_graph_sparse"]
