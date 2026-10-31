"""Exact monomial actions of physical symmetries preserving a fusion tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..su2.fusion_tree import (
    FusionNode,
    FusionTree,
    FusionTreeState,
    Leaf,
    TreeNode,
    node_q,
)
from ..su2.quantum_numbers import coupling_range


def _validated_permutation(values: Sequence[int], n: int, *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != n or tuple(sorted(result)) != tuple(range(n)):
        raise ValueError(f"{name} must contain every site exactly once")
    return result


def _node_physical_blocks(
    tree: FusionTree, physical_order: Sequence[int]
) -> dict[TreeNode, frozenset[int]]:
    order = _validated_permutation(
        physical_order, tree.n_leaves, name="physical_order"
    )
    result: dict[TreeNode, frozenset[int]] = {}

    def visit(node: TreeNode) -> frozenset[int]:
        if isinstance(node, Leaf):
            block = frozenset((order[node.index],))
        else:
            block = visit(node.left) | visit(node.right)
        result[node] = block
        return block

    visit(tree.root)
    return result


@dataclass(frozen=True)
class TreeFrameIsomorphismPlan:
    """A physical permutation mapping one fusion-tree hierarchy to another."""

    source_tree: FusionTree
    target_tree: FusionTree
    source_physical_order: tuple[int, ...]
    target_physical_order: tuple[int, ...]
    physical_image: tuple[int, ...]
    target_node_for_source: tuple[int, ...]
    child_swaps: tuple[bool, ...]

    def apply(self, state: FusionTreeState) -> tuple[FusionTreeState, float]:
        """Map one basis state with the exact Condon-Shortley swap phase."""

        source_nodes = self.source_tree.internal_nodes
        target_nodes = self.target_tree.internal_nodes
        if len(state.labels) != len(source_nodes):
            raise ValueError("state label count does not match the source tree")
        labels = state.label_map(self.source_tree)
        output: list[int | None] = [None] * len(target_nodes)
        phase = 1.0
        for index, (node, q_parent) in enumerate(zip(source_nodes, state.labels, strict=True)):
            q_left = node_q(node.left, labels)
            q_right = node_q(node.right, labels)
            if q_parent not in coupling_range(q_left, q_right):
                raise ValueError("state contains an inadmissible doubled-spin coupling")
            output[self.target_node_for_source[index]] = int(q_parent)
            if self.child_swaps[index]:
                numerator = q_left + q_right - int(q_parent)
                if numerator % 2:
                    raise RuntimeError("exchange phase exponent is not integral")
                if (numerator // 2) % 2:
                    phase = -phase
        if any(value is None for value in output):
            raise RuntimeError("tree-frame map did not assign every target node")
        return FusionTreeState(tuple(int(value) for value in output)), phase


def compile_tree_frame_isomorphism(
    source_tree: FusionTree,
    target_tree: FusionTree,
    source_physical_order: Sequence[int],
    target_physical_order: Sequence[int],
    physical_image: Sequence[int],
) -> TreeFrameIsomorphismPlan:
    """Compile a support-one physical action between two tree frames.

    Compilation fails if any transformed internal block is absent from the
    target hierarchy.  Thus success is a structural certificate that the
    physical permutation acts monomially, rather than by an expanding sum of
    F-move histories.
    """

    if source_tree.n_leaves != target_tree.n_leaves:
        raise ValueError("source and target trees must have the same leaf count")
    n = source_tree.n_leaves
    source_order = _validated_permutation(
        source_physical_order, n, name="source_physical_order"
    )
    target_order = _validated_permutation(
        target_physical_order, n, name="target_physical_order"
    )
    image = _validated_permutation(physical_image, n, name="physical_image")
    source_blocks = _node_physical_blocks(source_tree, source_order)
    target_blocks = _node_physical_blocks(target_tree, target_order)
    target_by_block = {block: node for node, block in target_blocks.items()}
    target_positions = {
        node: index for index, node in enumerate(target_tree.internal_nodes)
    }
    mapped_positions = []
    child_swaps = []
    for source_node in source_tree.internal_nodes:
        transformed = frozenset(image[site] for site in source_blocks[source_node])
        target_node = target_by_block.get(transformed)
        if not isinstance(target_node, FusionNode):
            raise ValueError(
                "physical permutation does not preserve the target fusion hierarchy"
            )
        transformed_left = frozenset(
            image[site] for site in source_blocks[source_node.left]
        )
        if transformed_left == target_blocks[target_node.left]:
            swapped = False
        elif transformed_left == target_blocks[target_node.right]:
            swapped = True
        else:
            raise ValueError(
                "physical permutation maps a child block across the target split"
            )
        mapped_positions.append(target_positions[target_node])
        child_swaps.append(swapped)
    if tuple(sorted(mapped_positions)) != tuple(range(len(target_positions))):
        raise RuntimeError("physical permutation does not biject internal tree nodes")
    return TreeFrameIsomorphismPlan(
        source_tree=source_tree,
        target_tree=target_tree,
        source_physical_order=source_order,
        target_physical_order=target_order,
        physical_image=image,
        target_node_for_source=tuple(mapped_positions),
        child_swaps=tuple(child_swaps),
    )


def compile_tree_automorphism(
    tree: FusionTree,
    physical_order: Sequence[int],
    physical_image: Sequence[int],
) -> TreeFrameIsomorphismPlan:
    """Compile a support-one symmetry action within one fusion-tree frame."""

    return compile_tree_frame_isomorphism(
        tree, tree, physical_order, physical_order, physical_image
    )


__all__ = [
    "TreeFrameIsomorphismPlan",
    "compile_tree_automorphism",
    "compile_tree_frame_isomorphism",
]
