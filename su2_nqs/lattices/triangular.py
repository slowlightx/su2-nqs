"""Triangular lattice represented as square-grid bonds plus one diagonal."""

from __future__ import annotations

from .square import square_bonds


def triangular_bonds(
    rows: int,
    columns: int,
    *,
    periodic: bool = False,
) -> tuple[tuple[int, int], ...]:
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    bonds = set(square_bonds(rows, columns, periodic=periodic))

    def add(first: int, second: int) -> None:
        if first != second:
            bonds.add(tuple(sorted((first, second))))

    for row in range(rows):
        for column in range(columns):
            if row + 1 < rows and column + 1 < columns:
                add(row * columns + column, (row + 1) * columns + column + 1)
            elif periodic:
                add(row * columns + column, ((row + 1) % rows) * columns + (column + 1) % columns)
    return tuple(sorted(bonds))

