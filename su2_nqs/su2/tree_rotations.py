"""Sparse local F-rotations between ordered binary fusion trees."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .fusion_tree import (
    FusionNode,
    FusionTree,
    Leaf,
    TreeNode,
    rotate_association_left,
    rotate_association_right,
)
from .wigner6j import f_move


def _sparse():
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("sparse tree rotations require scipy") from error
    return sparse


@lru_cache(maxsize=None)
def leaf_signature(node: TreeNode) -> tuple[int, ...]:
    """Ordered leaves below a tree node; stable across local rotations."""

    if isinstance(node, Leaf):
        return (node.index,)
    return leaf_signature(node.left) + leaf_signature(node.right)


def _node_at(node: TreeNode, path: tuple[int, ...]) -> TreeNode:
    for direction in path:
        if isinstance(node, Leaf):
            raise ValueError(f"path {path!r} descends through a leaf")
        node = node.left if direction == 0 else node.right
    return node


def _replace(node: TreeNode, path: tuple[int, ...], replacement: TreeNode) -> TreeNode:
    if not path:
        return replacement
    if isinstance(node, Leaf):
        raise ValueError(f"path {path!r} descends through a leaf")
    head, tail = path[0], path[1:]
    if head == 0:
        return FusionNode(_replace(node.left, tail, replacement), node.right)
    if head == 1:
        return FusionNode(node.left, _replace(node.right, tail, replacement))
    raise ValueError("tree paths contain only zero (left) or one (right)")


def rotate_structure(
    tree: FusionTree,
    path: tuple[int, ...],
    direction: str,
) -> FusionTree:
    """Rotate one internal association without assigning irrep labels."""

    node = _node_at(tree.root, path)
    if not isinstance(node, FusionNode):
        raise ValueError("rotation target must be an internal node")
    if direction == "left":
        replacement = rotate_association_left(node)
    elif direction == "right":
        replacement = rotate_association_right(node)
    else:
        raise ValueError("direction must be 'left' or 'right'")
    return FusionTree(_replace(tree.root, path, replacement))


def _node_q(
    node: TreeNode,
    labels_by_signature: dict[tuple[int, ...], int],
) -> int:
    return 1 if isinstance(node, Leaf) else labels_by_signature[leaf_signature(node)]


def rotation_matrix(
    tree: FusionTree,
    path: tuple[int, ...],
    target_q: int,
    direction: str = "left",
):
    """Return ``(new_tree, R)`` for one local F-move.

    Coordinates transform as ``psi_new = R @ psi_old``.  The matrix is real
    sparse and has at most the local number of admissible channels per column.
    """

    sparse = _sparse()
    old_node = _node_at(tree.root, path)
    if not isinstance(old_node, FusionNode):
        raise ValueError("rotation target must be internal")
    new_tree = rotate_structure(tree, path, direction)
    old_states = tree.enumerate_states(target_q)
    new_states = new_tree.enumerate_states(target_q)
    if len(old_states) != len(new_states):
        raise RuntimeError("a local F-move changed the Hilbert-space dimension")
    new_positions = {state.labels: index for index, state in enumerate(new_states)}
    old_nodes = tree.internal_nodes
    new_nodes = new_tree.internal_nodes
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    if direction == "left":
        if not isinstance(old_node.left, FusionNode):
            raise ValueError("left rotation requires ((A,B),C)")
        a, b, c = old_node.left.left, old_node.left.right, old_node.right
        old_intermediate_signature = leaf_signature(old_node.left)
        new_node = _node_at(new_tree.root, path)
        assert isinstance(new_node, FusionNode) and isinstance(new_node.right, FusionNode)
        new_intermediate_signature = leaf_signature(new_node.right)
    else:
        if not isinstance(old_node.right, FusionNode):
            raise ValueError("right rotation requires (A,(B,C))")
        a, b, c = old_node.left, old_node.right.left, old_node.right.right
        old_intermediate_signature = leaf_signature(old_node.right)
        new_node = _node_at(new_tree.root, path)
        assert isinstance(new_node, FusionNode) and isinstance(new_node.left, FusionNode)
        new_intermediate_signature = leaf_signature(new_node.left)

    outer_signature = leaf_signature(old_node)
    for column, state in enumerate(old_states):
        old_labels = {
            leaf_signature(node): q for node, q in zip(old_nodes, state.labels)
        }
        q_a, q_b, q_c = (
            _node_q(a, old_labels),
            _node_q(b, old_labels),
            _node_q(c, old_labels),
        )
        q_total = old_labels[outer_signature]
        f_matrix, q_ab_channels, q_bc_channels = f_move(
            q_a, q_b, q_c, q_total
        )
        if direction == "left":
            old_channel = old_labels[old_intermediate_signature]
            local_column = q_ab_channels.index(old_channel)
            candidates = (
                (q_bc, f_matrix[local_row, local_column])
                for local_row, q_bc in enumerate(q_bc_channels)
            )
        else:
            old_channel = old_labels[old_intermediate_signature]
            local_row = q_bc_channels.index(old_channel)
            candidates = (
                (q_ab, f_matrix[local_row, local_column])
                for local_column, q_ab in enumerate(q_ab_channels)
            )
        for new_channel, coefficient in candidates:
            if abs(coefficient) < 1e-15:
                continue
            new_labels = []
            for node in new_nodes:
                signature = leaf_signature(node)
                if signature == new_intermediate_signature:
                    new_labels.append(new_channel)
                else:
                    new_labels.append(old_labels[signature])
            row = new_positions[tuple(new_labels)]
            rows.append(row)
            columns.append(column)
            values.append(float(coefficient))
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(new_states), len(old_states))
    ).tocsr()
    return new_tree, matrix


def rotations_to_right_comb(
    tree: FusionTree,
) -> tuple[tuple[tuple[tuple[int, ...], str], ...], FusionTree]:
    """Deterministically reduce a tree to the fully right-associated comb."""

    steps: list[tuple[tuple[int, ...], str]] = []
    current = tree

    def reduce_at(path: tuple[int, ...]) -> None:
        nonlocal current
        node = _node_at(current.root, path)
        if isinstance(node, Leaf):
            return
        while isinstance(node.left, FusionNode):
            steps.append((path, "left"))
            current = rotate_structure(current, path, "left")
            node = _node_at(current.root, path)
            assert isinstance(node, FusionNode)
        if isinstance(node.right, FusionNode):
            reduce_at(path + (1,))

    reduce_at(())
    return tuple(steps), current


def recoupling_matrix_sparse(
    source: FusionTree,
    target: FusionTree,
    target_q: int,
):
    """Compose local F-moves between two trees with the same leaf order."""

    sparse = _sparse()
    if source.leaves != target.leaves:
        raise ValueError("source and target trees must have identical ordered leaves")
    source_steps, source_right = rotations_to_right_comb(source)
    target_steps, target_right = rotations_to_right_comb(target)
    if source_right != target_right:
        raise RuntimeError("right-comb canonical forms do not match")
    dimension = len(source.enumerate_states(target_q))
    total = sparse.identity(dimension, format="csr", dtype=np.float64)
    current = source
    applied: list[tuple[tuple[int, ...], str]] = []
    for path, direction in source_steps:
        current, local = rotation_matrix(current, path, target_q, direction)
        total = local @ total
        applied.append((path, direction))
    for path, _ in reversed(target_steps):
        current, local = rotation_matrix(current, path, target_q, "right")
        total = local @ total
        applied.append((path, "right"))
    if current != target:
        raise RuntimeError("F-move sequence did not reach the target tree")
    total.eliminate_zeros()
    return total.tocsr(), tuple(applied)

