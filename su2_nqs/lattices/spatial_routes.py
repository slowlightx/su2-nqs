"""Geometry-defined complete-support routes for spatial Clebsch trees.

Every route is a permutation of all physical sites.  A balanced binary tree
over that order therefore fuses disjoint supports and uses every physical
spin exactly once.  The route descriptors below measure at which tree scale
the bonds in each coupling shell first become close.  They are static
geometry data: no spin configuration or variational amplitude enters them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..su2.fusion_tree import FusionTree
from .geometry_tree import tree_leaf_distance


SiteOrder = tuple[int, ...]


def _validated_rectangle(rows: int, columns: int) -> tuple[int, int]:
    for name, value in (("rows", rows), ("columns", columns)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    rows, columns = int(rows), int(columns)
    if rows < 2 or columns < 2:
        raise ValueError("spatial routes require at least a 2x2 rectangle")
    return rows, columns


def _ordered_sites(rows: int, columns: int, key) -> SiteOrder:
    return tuple(
        sorted(
            range(rows * columns),
            key=lambda site: key(*divmod(site, columns)),
        )
    )


def _plaquette_order(rows: int, columns: int) -> SiteOrder:
    sites = []
    for block_row in range(0, rows, 2):
        for block_column in range(0, columns, 2):
            for local_row, local_column in ((0, 0), (0, 1), (1, 0), (1, 1)):
                row = block_row + local_row
                column = block_column + local_column
                if row < rows and column < columns:
                    sites.append(row * columns + column)
    return tuple(sites)


def rectangular_support_route_orders(
    lattice: str,
    rows: int,
    columns: int,
    *,
    drop_duplicates: bool = False,
) -> dict[str, SiteOrder]:
    """Return full-cover routes adapted to square or triangular order.

    The names label physical hypotheses, not independent parameter sets.  On
    narrow exact-small rectangles two hypotheses can induce the same site
    order.  ``drop_duplicates=True`` keeps the first deterministic
    representative; the strict default protects diagnostics that require
    eight distinct charts.
    """

    rows, columns = _validated_rectangle(rows, columns)
    n_sites = rows * columns
    row_major = tuple(range(n_sites))
    column_major = _ordered_sites(
        rows, columns, lambda row, column: (column, row)
    )
    plaquette = _plaquette_order(rows, columns)
    if lattice == "square":
        candidates = {
            "row_dimers": row_major,
            "column_dimers": column_major,
            "row_snake": _ordered_sites(
                rows,
                columns,
                lambda row, column: (row, column if row % 2 == 0 else -column),
            ),
            "column_snake": _ordered_sites(
                rows,
                columns,
                lambda row, column: (
                    column,
                    row if column % 2 == 0 else -row,
                ),
            ),
            "plaquette": plaquette,
            "neel": _ordered_sites(
                rows,
                columns,
                lambda row, column: ((row + column) % 2, row, column),
            ),
            "stripe_x": _ordered_sites(
                rows, columns, lambda row, column: (column % 2, row, column)
            ),
            "stripe_y": _ordered_sites(
                rows, columns, lambda row, column: (row % 2, row, column)
            ),
        }
    elif lattice == "triangular":
        candidates = {
            "row_dimers": row_major,
            "column_dimers": column_major,
            # On an even rectangular embedding this is a four-site rhombus,
            # not a three-site elementary triangle.
            "rhombus_blocks": plaquette,
            "k_012": _ordered_sites(
                rows,
                columns,
                lambda row, column: ((row + column) % 3, row, column),
            ),
            "k_021": _ordered_sites(
                rows,
                columns,
                lambda row, column: ((-(row + column)) % 3, row, column),
            ),
            "stripe_x": _ordered_sites(
                rows, columns, lambda row, column: (column % 2, row, column)
            ),
            "stripe_y": _ordered_sites(
                rows, columns, lambda row, column: (row % 2, row, column)
            ),
            "stripe_xy": _ordered_sites(
                rows,
                columns,
                lambda row, column: ((row + column) % 2, row, column),
            ),
        }
    else:
        raise ValueError("lattice must be 'square' or 'triangular'")

    expected = tuple(range(n_sites))
    if any(tuple(sorted(order)) != expected for order in candidates.values()):
        raise RuntimeError("a spatial route is not a complete site permutation")
    if not drop_duplicates and len(set(candidates.values())) != len(candidates):
        raise RuntimeError("the requested rectangle produces duplicate physical routes")
    routes: dict[str, SiteOrder] = {}
    seen: set[SiteOrder] = set()
    for name, order in candidates.items():
        if order not in seen:
            routes[name] = order
            seen.add(order)
    return routes


def globally_scrambled_support_routes(
    orders: Mapping[str, Sequence[int]],
    permutation: Sequence[int],
    *,
    preserve_name: str,
) -> dict[str, SiteOrder]:
    """Relabel nonreference routes for a geometry-blind control bank.

    The named reference is unchanged, so a zero geometry query remains the
    exact same single-tree state.  The caller is responsible for choosing a
    permutation that leaves the resulting route bank distinct; duplicates
    fail closed rather than silently reducing the control's work.
    """

    if preserve_name not in orders:
        raise ValueError("preserve_name is absent from the route bank")
    checked = {
        str(name): tuple(int(site) for site in order)
        for name, order in orders.items()
    }
    n_sites = len(next(iter(checked.values())))
    expected = tuple(range(n_sites))
    if any(tuple(sorted(order)) != expected for order in checked.values()):
        raise ValueError("every route must contain each site exactly once")
    permutation = tuple(int(site) for site in permutation)
    if tuple(sorted(permutation)) != expected:
        raise ValueError("permutation must contain every site exactly once")
    result = {
        name: (
            order
            if name == preserve_name
            else tuple(permutation[site] for site in order)
        )
        for name, order in checked.items()
    }
    if len(set(result.values())) != len(result):
        raise ValueError("scrambling produced duplicate support routes")
    return result


@dataclass(frozen=True)
class RouteLocalityFeatures:
    """Raw tree distances and standardized locality features by bond shell."""

    labels: tuple[str, ...]
    raw_mean_tree_distances: tuple[tuple[float, ...], ...]
    standardized_localities: tuple[tuple[float, ...], ...]

    @property
    def feature_count(self) -> int:
        return len(self.labels)


def coupling_shell_route_features(
    tree: FusionTree,
    orders: Mapping[str, Sequence[int]],
    bonds: Sequence[tuple[int, int, float]],
    *,
    tolerance: float = 1.0e-12,
) -> RouteLocalityFeatures:
    """Describe route locality separately for every nonzero ``|J|`` shell.

    For each route and coupling magnitude the raw descriptor is the mean tree
    distance between bonded physical sites.  Columns are converted to
    zero-mean unit-RMS *localities* (minus distance), so a positive value means
    that a shell is fused earlier than it is in the average route.  Constant
    columns are removed because they cannot organize a route mixture.
    """

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    grouped: dict[str, list[tuple[int, int, float]]] = {}
    shell_values: list[float] = []
    magnitudes: list[float] = []
    for first, second, coupling in bonds:
        first, second, coupling = int(first), int(second), float(coupling)
        if not 0 <= first < tree.n_leaves or not 0 <= second < tree.n_leaves:
            raise ValueError("bond site lies outside the fusion tree")
        if first == second or not np.isfinite(coupling):
            raise ValueError("bonds must join distinct sites with finite coupling")
        magnitude = abs(coupling)
        if magnitude <= tolerance:
            continue
        shell = next(
            (value for value in magnitudes if abs(value - magnitude) <= tolerance),
            None,
        )
        if shell is None:
            magnitudes.append(magnitude)
            shell = magnitude
        label = f"absJ={shell:.12g}"
        grouped.setdefault(label, []).append((first, second, magnitude))
    if not grouped:
        raise ValueError("at least one nonzero bond is required")
    for label in sorted(grouped, key=lambda value: -float(value.split("=")[1])):
        shell_values.append(float(label.split("=")[1]))
    ordered = {
        f"absJ={value:.12g}": grouped[f"absJ={value:.12g}"]
        for value in shell_values
    }
    return bond_shell_route_features(
        tree,
        orders,
        ordered,
        tolerance=tolerance,
    )


def bond_shell_route_features(
    tree: FusionTree,
    orders: Mapping[str, Sequence[int]],
    bond_shells: Mapping[str, Sequence[tuple[int, int, float]]],
    *,
    tolerance: float = 1.0e-12,
) -> RouteLocalityFeatures:
    """Build locality features from explicit geometry-shell edge ledgers.

    The third entry of each edge is a positive image multiplicity.  It is
    normalized within the shell, so the descriptor never contains the
    Hamiltonian ratio ``J2/J1``.  Supplying explicit ``J1`` and ``J2``
    ledgers avoids accidental shell merging on very small periodic tori.
    """

    if not orders:
        raise ValueError("orders must be nonempty")
    if not bond_shells:
        raise ValueError("bond_shells must be nonempty")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    checked_orders = {
        str(name): tuple(int(site) for site in order)
        for name, order in orders.items()
    }
    expected = tuple(range(tree.n_leaves))
    if any(tuple(sorted(order)) != expected for order in checked_orders.values()):
        raise ValueError("every route must contain each tree leaf exactly once")
    checked_shells: dict[str, tuple[tuple[int, int, float], ...]] = {}
    for label, edges in bond_shells.items():
        checked = []
        for first, second, multiplicity in edges:
            first, second = int(first), int(second)
            multiplicity = float(multiplicity)
            if not 0 <= first < tree.n_leaves or not 0 <= second < tree.n_leaves:
                raise ValueError("bond site lies outside the fusion tree")
            if first == second:
                raise ValueError("a bond shell cannot contain a self edge")
            if not np.isfinite(multiplicity) or multiplicity <= tolerance:
                raise ValueError("bond image multiplicities must be positive and finite")
            checked.append((first, second, multiplicity))
        if not checked:
            raise ValueError("every bond shell must be nonempty")
        checked_shells[str(label)] = tuple(checked)

    raw = np.empty((len(checked_orders), len(checked_shells)), dtype=np.float64)
    for route_index, order in enumerate(checked_orders.values()):
        inverse = {site: leaf for leaf, site in enumerate(order)}
        for shell_index, edges in enumerate(checked_shells.values()):
            numerator = sum(
                multiplicity
                * tree_leaf_distance(tree, inverse[first], inverse[second])
                for first, second, multiplicity in edges
            )
            denominator = sum(multiplicity for _, _, multiplicity in edges)
            raw[route_index, shell_index] = float(numerator / denominator)

    centered_locality = -(raw - raw.mean(axis=0, keepdims=True))
    rms = np.sqrt(np.mean(centered_locality**2, axis=0))
    retained = rms > tolerance
    if not np.any(retained):
        raise ValueError("the supplied routes have no distinguishable bond locality")
    standardized = centered_locality[:, retained] / rms[retained]
    labels = tuple(
        f"{label}_mean_tree_locality"
        for label, keep in zip(checked_shells, retained, strict=True)
        if keep
    )
    return RouteLocalityFeatures(
        labels=labels,
        raw_mean_tree_distances=tuple(
            tuple(float(value) for value in row) for row in raw[:, retained]
        ),
        standardized_localities=tuple(
            tuple(float(value) for value in row) for row in standardized
        ),
    )


__all__ = [
    "RouteLocalityFeatures",
    "SiteOrder",
    "bond_shell_route_features",
    "coupling_shell_route_features",
    "globally_scrambled_support_routes",
    "rectangular_support_route_orders",
]
