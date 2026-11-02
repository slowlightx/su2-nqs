"""Sparse graph Heisenberg Hamiltonian in a fixed-Sz configuration basis."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..su2.sz_basis import enumerate_sz_configs


def heisenberg_sz_graph_sparse(
    n: int,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    *,
    n_up: int | None = None,
):
    """Build CSR ``sum J_ij S_i.S_j`` in ``enumerate_sz_configs`` order."""

    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("sparse Sz graph operators require scipy") from error
    if n_up is None:
        n_up = n // 2
    basis = enumerate_sz_configs(n, n_up)
    positions = {configuration: index for index, configuration in enumerate(basis)}
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
                rows.append(positions[tuple(flipped)])
                columns.append(column)
                values.append(0.5 * coupling)
        rows.append(column)
        columns.append(column)
        values.append(diagonal)
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis)), dtype=np.float64
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, basis
