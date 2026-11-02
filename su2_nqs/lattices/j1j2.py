"""Square-lattice J1-J2 bond generation."""

from __future__ import annotations


def square_j1j2_bonds(
    rows: int,
    columns: int,
    *,
    j1: float = 1.0,
    j2: float = 0.0,
    periodic: bool = False,
) -> tuple[tuple[int, int, float], ...]:
    """Nearest and both diagonal next-nearest bonds without duplicates."""

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    bonds: dict[tuple[int, int], float] = {}

    def add(first: int, second: int, coupling: float) -> None:
        if first == second or coupling == 0:
            return
        pair = tuple(sorted((first, second)))
        bonds[pair] = bonds.get(pair, 0.0) + coupling

    for row in range(rows):
        for column in range(columns):
            site = row * columns + column
            if column + 1 < columns:
                add(site, site + 1, j1)
            elif periodic:
                add(site, row * columns, j1)
            if row + 1 < rows:
                add(site, site + columns, j1)
            elif periodic:
                add(site, column, j1)
            if row + 1 < rows and column + 1 < columns:
                add(site, site + columns + 1, j2)
                add(site + 1, site + columns, j2)
            elif periodic:
                down = (row + 1) % rows
                right = (column + 1) % columns
                add(site, down * columns + right, j2)
                add(row * columns + right, down * columns + column, j2)
    return tuple((first, second, coupling) for (first, second), coupling in sorted(bonds.items()))

