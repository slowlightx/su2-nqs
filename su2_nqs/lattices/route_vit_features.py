"""Multiscale physical-geometry tokens for route-conditioned SU(2) models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .geometry_tree import tree_leaf_distance
from ..su2.fusion_tree import FusionTree


@dataclass(frozen=True)
class MultiscaleRouteFeatures:
    """Standardized bond-locality histograms aligned with a route bank."""

    labels: tuple[str, ...]
    raw_histograms: tuple[tuple[float, ...], ...]
    standardized: tuple[tuple[float, ...], ...]


def multiscale_bond_route_features(
    tree: FusionTree,
    orders: Mapping[str, Sequence[int]],
    bond_shells: Mapping[str, Sequence[tuple[int, int, float]]],
    *,
    tolerance: float = 1.0e-12,
) -> MultiscaleRouteFeatures:
    """Resolve each coupling shell by the tree scale at which a bond meets.

    A mean tree distance erases physically different distributions.  Here a
    token records the weighted fraction of every bond shell at every observed
    tree distance.  Constant columns are removed and the rest are centered and
    normalized to unit RMS.  No Hamiltonian coupling ratio, route identity, or
    spin configuration is supplied.
    """

    if not orders or not bond_shells:
        raise ValueError("orders and bond_shells must be nonempty")
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
        for first, second, weight in edges:
            first, second, weight = int(first), int(second), float(weight)
            if not 0 <= first < tree.n_leaves or not 0 <= second < tree.n_leaves:
                raise ValueError("bond site lies outside the fusion tree")
            if first == second or not np.isfinite(weight) or weight <= tolerance:
                raise ValueError("bond weights must be positive and join distinct sites")
            checked.append((first, second, weight))
        if not checked:
            raise ValueError("every bond shell must be nonempty")
        checked_shells[str(label)] = tuple(checked)

    distances_by_shell: dict[str, tuple[int, ...]] = {}
    for label, edges in checked_shells.items():
        observed = set()
        for order in checked_orders.values():
            inverse = {site: leaf for leaf, site in enumerate(order)}
            observed.update(
                tree_leaf_distance(tree, inverse[first], inverse[second])
                for first, second, _weight in edges
            )
        distances_by_shell[label] = tuple(sorted(observed))

    labels = tuple(
        f"{shell}_tree_distance_{distance}_fraction"
        for shell, distances in distances_by_shell.items()
        for distance in distances
    )
    raw = np.zeros((len(checked_orders), len(labels)), dtype=np.float64)
    for route_index, order in enumerate(checked_orders.values()):
        inverse = {site: leaf for leaf, site in enumerate(order)}
        offset = 0
        for shell, edges in checked_shells.items():
            denominator = sum(weight for _first, _second, weight in edges)
            distances = distances_by_shell[shell]
            for local_index, distance in enumerate(distances):
                raw[route_index, offset + local_index] = sum(
                    weight
                    for first, second, weight in edges
                    if tree_leaf_distance(
                        tree, inverse[first], inverse[second]
                    )
                    == distance
                ) / denominator
            offset += len(distances)

    centered = raw - raw.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=0))
    retained = rms > tolerance
    if not np.any(retained):
        raise ValueError("multiscale route descriptors are all constant")
    standardized = centered[:, retained] / rms[retained]
    retained_labels = tuple(
        label for label, keep in zip(labels, retained, strict=True) if keep
    )
    return MultiscaleRouteFeatures(
        labels=retained_labels,
        raw_histograms=tuple(
            tuple(float(value) for value in row) for row in raw[:, retained]
        ),
        standardized=tuple(
            tuple(float(value) for value in row) for row in standardized
        ),
    )


__all__ = ["MultiscaleRouteFeatures", "multiscale_bond_route_features"]
