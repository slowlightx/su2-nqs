"""Triangular-lattice J1-J2 bonds in oblique integer coordinates.

The primitive vectors have unit length and enclose an angle of 120 degrees.
Consequently an integer displacement ``(dr, dc)`` has squared physical length
``dr**2 + dc**2 - dr*dc``.  The shells below therefore have squared lengths
one and three, respectively.  Stating the metric explicitly avoids silently
mixing this convention with the equally common 60-degree convention, in which
the sign of the cross term and the diagonal representative are reversed.
"""

from __future__ import annotations


def triangular_j1j2_bonds(
    rows: int,
    columns: int,
    *,
    j1: float = 1.0,
    j2: float = 0.0,
    periodic: bool = False,
) -> tuple[tuple[int, int, float], ...]:
    """Return first- and second-shell triangular bonds without duplicates.

    Coordinates use primitive directions ``(1,0)`` and ``(0,1)`` with the
    third nearest-neighbor direction ``(1,1)``.  Second-shell representatives
    are ``(1,-1)``, ``(1,2)``, and ``(2,1)``.  Under periodic wrapping,
    distinct periodic images connecting the same pair are combined by adding
    their couplings; this matters only for very small unit cells.
    """

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    bonds: dict[tuple[int, int], float] = {}

    def add_shell(displacements, coupling):
        if coupling == 0.0:
            return
        for row in range(rows):
            for column in range(columns):
                first = row * columns + column
                for dr, dc in displacements:
                    rr, cc = row + dr, column + dc
                    if periodic:
                        rr %= rows
                        cc %= columns
                    elif not (0 <= rr < rows and 0 <= cc < columns):
                        continue
                    second = rr * columns + cc
                    if first == second:
                        continue
                    edge = tuple(sorted((first, second)))
                    bonds[edge] = bonds.get(edge, 0.0) + float(coupling)

    add_shell(((1, 0), (0, 1), (1, 1)), j1)
    add_shell(((1, -1), (1, 2), (2, 1)), j2)
    return tuple((first, second, coupling) for (first, second), coupling in sorted(bonds.items()))
