"""Square-lattice nearest-neighbor geometry."""

from __future__ import annotations


def square_bonds(
    rows: int,
    columns: int,
    *,
    periodic: bool = False,
) -> tuple[tuple[int, int], ...]:
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    bonds: set[tuple[int, int]] = set()

    def add(first: int, second: int) -> None:
        if first != second:
            bonds.add(tuple(sorted((first, second))))

    for row in range(rows):
        for column in range(columns):
            site = row * columns + column
            if column + 1 < columns:
                add(site, site + 1)
            elif periodic:
                add(site, row * columns)
            if row + 1 < rows:
                add(site, site + columns)
            elif periodic:
                add(site, column)
    return tuple(sorted(bonds))

