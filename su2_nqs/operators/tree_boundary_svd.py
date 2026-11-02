"""Cut- and spatial-rank-resolved Heisenberg boundary channels.

For a binary fusion-tree split ``A | B``, the crossing part of one bond shell
has coupling matrix ``J_AB`` and therefore the exact decomposition

``sum_ij J_ij S_i.S_j = sum_r sigma_r A_r^(1).B_r^(1)``.

Every spatial singular mode carries one SU(2) vector on either side of the
cut (doubled operator rank ``q_op=2``).  This compiler keeps those modes
separate while sharing trainable channels between translation-related tree
nodes at the same depth.  It contains no wavefunction or target-state data.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode


Q_OP_VECTOR = 2


class BoundaryBondLike(Protocol):
    """Structural input type for a weighted, channel-labelled graph edge."""

    first: int
    second: int
    coupling: float
    channel: int


@dataclass(frozen=True)
class TreeBoundaryBond:
    """One weighted spatial-mode edge emitted by the boundary compiler."""

    first: int
    second: int
    coupling: float
    channel: int


def _leaf_indices(node: TreeNode) -> tuple[int, ...]:
    if isinstance(node, Leaf):
        return (node.index,)
    return _leaf_indices(node.left) + _leaf_indices(node.right)


def _tree_depths(tree: FusionTree) -> dict[FusionNode, int]:
    depths: dict[FusionNode, int] = {}

    def visit(node: TreeNode, depth: int) -> None:
        if isinstance(node, Leaf):
            return
        depths[node] = depth
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(tree.root, 0)
    return depths


@dataclass(frozen=True, order=True)
class TreeBoundaryChannel:
    """One shared physical channel, using doubled tensor-rank labels."""

    channel: int
    shell: int
    tree_depth: int
    spatial_rank_index: int
    q_op: int = Q_OP_VECTOR


@dataclass(frozen=True)
class TreeBoundarySVD:
    """Target-independent decomposition consumable by the causal conditioner."""

    bonds: tuple[TreeBoundaryBond, ...]
    channels: tuple[TreeBoundaryChannel, ...]
    group_rows: tuple[dict[str, object], ...]
    maximum_reconstruction_error: float
    rank_tolerance: float

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def _normalized_bonds(
    tree: FusionTree, bonds: tuple[BoundaryBondLike, ...]
) -> tuple[TreeBoundaryBond, ...]:
    accumulated: dict[tuple[int, int, int], float] = {}
    for bond in bonds:
        if not 0 <= bond.first < tree.n_leaves or not 0 <= bond.second < tree.n_leaves:
            raise ValueError("Hamiltonian bond endpoint lies outside the tree")
        first, second = sorted((bond.first, bond.second))
        key = (first, second, bond.channel)
        accumulated[key] = accumulated.get(key, 0.0) + float(bond.coupling)
    return tuple(
        TreeBoundaryBond(first, second, coupling, shell)
        for (first, second, shell), coupling in sorted(accumulated.items())
        if coupling != 0.0
    )


def compile_tree_boundary_svd(
    tree: FusionTree,
    bonds: tuple[BoundaryBondLike, ...],
    *,
    rank_tolerance: float = 1.0e-12,
    coefficient_tolerance: float = 1.0e-14,
) -> TreeBoundarySVD:
    """Resolve every bond at its unique LCA split and spatial SVD rank.

    Output channel identities are ``(shell, tree_depth, spatial_rank_index)``.
    Thus nodes at the same coset-tree depth share parameters, while different
    physical shells, length scales, and spatial boundary modes remain
    addressable.  Summing the returned mode components reconstructs every
    input shell coupling matrix up to ``maximum_reconstruction_error``.
    """

    for name, value in (
        ("rank_tolerance", rank_tolerance),
        ("coefficient_tolerance", coefficient_tolerance),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    normalized = _normalized_bonds(tree, bonds)
    if not normalized:
        raise ValueError("at least one nonzero Hamiltonian bond is required")

    node_indices = {node: index for index, node in enumerate(tree.internal_nodes)}
    depths = _tree_depths(tree)
    split_data = []
    for node in tree.internal_nodes:
        left = _leaf_indices(node.left)
        right = _leaf_indices(node.right)
        split_data.append((node, depths[node], left, right))

    grouped: dict[tuple[FusionNode, int], list[TreeBoundaryBond]] = {}
    for bond in normalized:
        assigned = None
        for node, depth, left, right in split_data:
            left_set = set(left)
            right_set = set(right)
            if (
                bond.first in left_set
                and bond.second in right_set
                or bond.second in left_set
                and bond.first in right_set
            ):
                assigned = (node, depth, left, right)
                break
        if assigned is None:
            raise RuntimeError("failed to locate the bond's unique LCA split")
        grouped.setdefault((assigned[0], bond.channel), []).append(bond)

    decomposed_groups: list[
        tuple[int, int, tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]
    ] = []
    channel_keys: set[tuple[int, int, int]] = set()
    maximum_error = 0.0
    group_rows: list[dict[str, object]] = []
    for (node, shell), terms in sorted(
        grouped.items(),
        key=lambda item: (depths[item[0][0]], node_indices[item[0][0]], item[0][1]),
    ):
        depth = depths[node]
        left = _leaf_indices(node.left)
        right = _leaf_indices(node.right)
        left_position = {site: index for index, site in enumerate(left)}
        right_position = {site: index for index, site in enumerate(right)}
        matrix = np.zeros((len(left), len(right)), dtype=np.float64)
        for term in terms:
            if term.first in left_position:
                row, column = left_position[term.first], right_position[term.second]
            else:
                row, column = left_position[term.second], right_position[term.first]
            matrix[row, column] += term.coupling
        left_vectors, singular_values, right_vectors = np.linalg.svd(
            matrix, full_matrices=False
        )
        threshold = (
            float(rank_tolerance) * float(singular_values[0])
            if len(singular_values) and singular_values[0] > 0.0
            else float(rank_tolerance)
        )
        active = np.flatnonzero(singular_values > threshold)
        reconstructed = (
            (left_vectors[:, active] * singular_values[active])
            @ right_vectors[active]
            if len(active)
            else np.zeros_like(matrix)
        )
        error = float(np.max(np.abs(reconstructed - matrix), initial=0.0))
        maximum_error = max(maximum_error, error)
        for rank_index in range(len(active)):
            channel_keys.add((shell, depth, rank_index))
        decomposed_groups.append(
            (shell, depth, left, right, left_vectors[:, active], singular_values[active], right_vectors[active])
        )
        group_rows.append(
            {
                "node_index": node_indices[node],
                "tree_depth": depth,
                "shell": shell,
                "left_sites": list(left),
                "right_sites": list(right),
                "spatial_rank": int(len(active)),
                "singular_values": singular_values[active].tolist(),
                "reconstruction_error": error,
                "q_op": Q_OP_VECTOR,
            }
        )

    ordered_keys = tuple(sorted(channel_keys))
    key_to_channel = {key: index for index, key in enumerate(ordered_keys)}
    channels = tuple(
        TreeBoundaryChannel(index, shell, depth, rank_index)
        for index, (shell, depth, rank_index) in enumerate(ordered_keys)
    )
    output: list[TreeBoundaryBond] = []
    for shell, depth, left, right, left_vectors, singular_values, right_vectors in decomposed_groups:
        for rank_index, singular_value in enumerate(singular_values):
            channel = key_to_channel[(shell, depth, rank_index)]
            component = singular_value * np.outer(
                left_vectors[:, rank_index], right_vectors[rank_index]
            )
            for row, first in enumerate(left):
                for column, second in enumerate(right):
                    coefficient = float(component[row, column])
                    if abs(coefficient) > coefficient_tolerance:
                        output.append(
                            TreeBoundaryBond(first, second, coefficient, channel)
                        )
    return TreeBoundarySVD(
        bonds=tuple(output),
        channels=channels,
        group_rows=tuple(group_rows),
        maximum_reconstruction_error=maximum_error,
        rank_tolerance=float(rank_tolerance),
    )


def permute_tree_boundary_bonds(
    decomposition: TreeBoundarySVD,
    permutation: tuple[int, ...],
) -> tuple[TreeBoundaryBond, ...]:
    """Return a parameter-matched topology-scrambled control.

    Channel labels and coefficients are untouched; only physical endpoints are
    permuted.  The permutation is target- and wavefunction-independent.
    """

    n_sites = len(permutation)
    if tuple(sorted(permutation)) != tuple(range(n_sites)):
        raise ValueError("permutation must contain every site exactly once")
    if any(
        not 0 <= bond.first < n_sites or not 0 <= bond.second < n_sites
        for bond in decomposition.bonds
    ):
        raise ValueError("permutation size does not match the decomposition")
    return tuple(
        TreeBoundaryBond(
            permutation[bond.first],
            permutation[bond.second],
            bond.coupling,
            bond.channel,
        )
        for bond in decomposition.bonds
    )


__all__ = [
    "Q_OP_VECTOR",
    "BoundaryBondLike",
    "TreeBoundaryBond",
    "TreeBoundaryChannel",
    "TreeBoundarySVD",
    "compile_tree_boundary_svd",
    "permute_tree_boundary_bonds",
]
