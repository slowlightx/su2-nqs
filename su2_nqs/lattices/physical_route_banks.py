"""Physics-complete route banks for resonating exact-SU(2) models."""

from __future__ import annotations

from .spatial_routes import SiteOrder, rectangular_support_route_orders


def triangular_diagonal_dimer_order(rows: int, columns: int) -> SiteOrder:
    """Pair every site along the third triangular NN direction ``(1, 1)``.

    The rectangular embedding already supplies dimer routes along ``(1, 0)``
    and ``(0, 1)``.  On an even-row torus this order makes every first-level
    sibling pair of a balanced fusion tree a ``(1, 1)`` nearest-neighbour
    bond, completing the threefold dimer orientation set.
    """

    if rows < 2 or columns < 2 or rows % 2:
        raise ValueError("diagonal dimer order requires even rows and at least 2x2")
    order: list[int] = []
    for row in range(0, rows, 2):
        for column in range(columns):
            order.extend(
                (
                    row * columns + column,
                    (row + 1) * columns + (column + 1) % columns,
                )
            )
    result = tuple(order)
    if tuple(sorted(result)) != tuple(range(rows * columns)):
        raise RuntimeError("diagonal dimer route is not a complete site order")
    return result


def triangular_dimer_complete_route_orders(
    rows: int,
    columns: int,
) -> dict[str, SiteOrder]:
    """Return the standard triangular bank plus the missing dimer direction."""

    base = rectangular_support_route_orders(
        "triangular", rows, columns, drop_duplicates=True
    )
    result: dict[str, SiteOrder] = {}
    for name, order in base.items():
        result[name] = order
        if name == "column_dimers":
            result["diagonal_dimers"] = triangular_diagonal_dimer_order(
                rows, columns
            )
    if len(set(result.values())) != len(result):
        raise RuntimeError("dimer-complete route bank contains duplicate orders")
    return result


__all__ = [
    "triangular_diagonal_dimer_order",
    "triangular_dimer_complete_route_orders",
]
