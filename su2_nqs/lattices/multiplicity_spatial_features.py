"""Static geometry descriptors for SU(2)-scalar multiplicity attention.

The descriptors depend only on fusion-node supports and lattice bonds.  They
never inspect a spin configuration or a magnetic quantum number.  A network
using them to generate reduced Clebsch--Gordan maps therefore remains an
SU(2) scalar while gaining node-resolved spatial context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..su2.fusion_tree import FusionTree, Leaf, TreeNode


@dataclass(frozen=True)
class MultiplicitySpatialFeatures:
    """Standardized node and pair descriptors for every fusion route."""

    node: tuple[tuple[tuple[float, ...], ...], ...]
    pair: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
    node_labels: tuple[str, ...]
    pair_labels: tuple[str, ...]

    @property
    def n_routes(self) -> int:
        return len(self.node)

    @property
    def n_nodes(self) -> int:
        return len(self.node[0])


def _tree_supports(tree: FusionTree) -> tuple[tuple[int, ...], ...]:
    cache: dict[TreeNode, tuple[int, ...]] = {}

    def support(node: TreeNode) -> tuple[int, ...]:
        if node not in cache:
            if isinstance(node, Leaf):
                cache[node] = (node.index,)
            else:
                cache[node] = support(node.left) + support(node.right)
        return cache[node]

    return tuple(support(node) for node in tree.internal_nodes)


def _edge_counts(
    supports: tuple[set[int], ...],
    bonds: tuple[tuple[int, int, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(supports)
    internal = np.zeros(count, dtype=np.float64)
    boundary = np.zeros(count, dtype=np.float64)
    interface = np.zeros((count, count), dtype=np.float64)
    edges = tuple((int(first), int(second)) for first, second, _ in bonds)
    for index, current in enumerate(supports):
        for first, second in edges:
            first_inside = first in current
            second_inside = second in current
            internal[index] += float(first_inside and second_inside)
            boundary[index] += float(first_inside != second_inside)
    for first_index, first_support in enumerate(supports):
        for second_index, second_support in enumerate(supports):
            if first_index == second_index:
                continue
            first_only = first_support - second_support
            second_only = second_support - first_support
            interface[first_index, second_index] = sum(
                (first in first_only and second in second_only)
                or (second in first_only and first in second_only)
                for first, second in edges
            )
    return internal, boundary, interface


def _standardize(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1, values.shape[-1])
    mean = np.mean(flattened, axis=0)
    scale = np.std(flattened, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (values - mean) / scale


def multiplicity_spatial_features(
    tree: FusionTree,
    route_orders: tuple[tuple[int, ...], ...],
    j1_bonds: tuple[tuple[int, int, float], ...],
    j2_bonds: tuple[tuple[int, int, float], ...],
) -> MultiplicitySpatialFeatures:
    """Build route-equivariant fusion-support descriptors.

    ``route_orders[r][i]`` maps tree leaf ``i`` to a physical lattice site.
    Bond weights are ignored here; the two shell registries are deliberately
    separate so that the network cannot confuse geometry with the chosen
    value of ``J2/J1``.
    """

    n_sites = tree.n_leaves
    expected = tuple(range(n_sites))
    if not route_orders or any(tuple(sorted(order)) != expected for order in route_orders):
        raise ValueError("every route must be a permutation of all physical sites")
    supports_in_tree = _tree_supports(tree)
    n_nodes = len(supports_in_tree)
    node_rows = []
    pair_rows = []
    for order in route_orders:
        supports = tuple(
            {int(order[leaf]) for leaf in support} for support in supports_in_tree
        )
        sizes = np.asarray([len(support) for support in supports], dtype=np.float64)
        j1_internal, j1_boundary, j1_interface = _edge_counts(supports, j1_bonds)
        j2_internal, j2_boundary, j2_interface = _edge_counts(supports, j2_bonds)
        route_node = np.column_stack(
            (
                np.log2(sizes) / np.log2(float(n_sites)),
                (sizes == n_sites).astype(np.float64),
                j1_internal / np.maximum(sizes, 1.0),
                j2_internal / np.maximum(sizes, 1.0),
                j1_boundary / np.sqrt(np.maximum(sizes * (n_sites - sizes), 1.0)),
                j2_boundary / np.sqrt(np.maximum(sizes * (n_sites - sizes), 1.0)),
            )
        )
        route_pair = np.empty((n_nodes, n_nodes, 5), dtype=np.float64)
        for first in range(n_nodes):
            for second in range(n_nodes):
                overlap = len(supports[first] & supports[second])
                union = len(supports[first] | supports[second])
                norm = np.sqrt(max(sizes[first] * sizes[second], 1.0))
                route_pair[first, second] = (
                    overlap / max(union, 1),
                    j1_interface[first, second] / norm,
                    j2_interface[first, second] / norm,
                    np.exp(-abs(np.log2(sizes[first] / sizes[second]))),
                    float(
                        supports[first] <= supports[second]
                        or supports[second] <= supports[first]
                    ),
                )
        node_rows.append(route_node)
        pair_rows.append(route_pair)
    node = _standardize(np.stack(node_rows, axis=0))
    pair = _standardize(np.stack(pair_rows, axis=0))
    if np.any(~np.isfinite(node)) or np.any(~np.isfinite(pair)):
        raise FloatingPointError("multiplicity spatial descriptors are not finite")
    return MultiplicitySpatialFeatures(
        node=tuple(
            tuple(tuple(map(float, row)) for row in route) for route in node
        ),
        pair=tuple(
            tuple(
                tuple(tuple(map(float, row)) for row in source)
                for source in route
            )
            for route in pair
        ),
        node_labels=(
            "log_support_size",
            "root_indicator",
            "J1_internal_density",
            "J2_internal_density",
            "J1_boundary_density",
            "J2_boundary_density",
        ),
        pair_labels=(
            "support_jaccard",
            "J1_interface_density",
            "J2_interface_density",
            "scale_compatibility",
            "nested_support",
        ),
    )


def scramble_multiplicity_spatial_features(
    features: MultiplicitySpatialFeatures,
    *,
    seed: int,
) -> MultiplicitySpatialFeatures:
    """Break the support-to-node assignment while preserving all marginals.

    Each route receives an independent node permutation, applied to both pair
    axes.  The result has identical descriptor values and parameter count but
    no longer attaches them to the correct reduced tensors, making it a
    stringent geometry-causal control.
    """

    node = np.asarray(features.node, dtype=np.float64)
    pair = np.asarray(features.pair, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    scrambled_node = np.empty_like(node)
    scrambled_pair = np.empty_like(pair)
    for route in range(features.n_routes):
        permutation = rng.permutation(features.n_nodes)
        scrambled_node[route] = node[route, permutation]
        scrambled_pair[route] = pair[route][permutation][:, permutation]
    return MultiplicitySpatialFeatures(
        node=tuple(
            tuple(tuple(map(float, row)) for row in route)
            for route in scrambled_node
        ),
        pair=tuple(
            tuple(
                tuple(tuple(map(float, row)) for row in source)
                for source in route
            )
            for route in scrambled_pair
        ),
        node_labels=features.node_labels,
        pair_labels=features.pair_labels,
    )


__all__ = [
    "MultiplicitySpatialFeatures",
    "multiplicity_spatial_features",
    "scramble_multiplicity_spatial_features",
]
