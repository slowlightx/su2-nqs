"""Generic binary fusion trees for physical spin-1/2 leaves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .quantum_numbers import coupling_range, validate_q


@dataclass(frozen=True, order=True)
class Leaf:
    """A physical spin-1/2 leaf identified by its chain position."""

    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("leaf indices must be non-negative integers")

    @property
    def q(self) -> int:
        return 1


@dataclass(frozen=True)
class FusionNode:
    """An ordered fusion of two child representations."""

    left: "TreeNode"
    right: "TreeNode"


TreeNode: TypeAlias = Leaf | FusionNode


@dataclass(frozen=True)
class FusionTreeState:
    """Internal doubled-spin labels in tree postorder, including the root."""

    labels: tuple[int, ...]

    def label_map(self, tree: "FusionTree") -> dict[FusionNode, int]:
        nodes = tree.internal_nodes
        if len(self.labels) != len(nodes):
            raise ValueError("state label count does not match the tree")
        return dict(zip(nodes, self.labels))

    @property
    def total_q(self) -> int:
        if not self.labels:
            raise ValueError("a one-leaf tree has no internal total-spin label")
        return self.labels[-1]


def _leaves(node: TreeNode) -> tuple[Leaf, ...]:
    if isinstance(node, Leaf):
        return (node,)
    return _leaves(node.left) + _leaves(node.right)


def _internal_postorder(node: TreeNode) -> tuple[FusionNode, ...]:
    if isinstance(node, Leaf):
        return ()
    return _internal_postorder(node.left) + _internal_postorder(node.right) + (node,)


@dataclass(frozen=True)
class FusionTree:
    """An ordered binary tree whose leaves are spins 0 through ``N-1``."""

    root: TreeNode

    def __post_init__(self) -> None:
        indices = tuple(leaf.index for leaf in _leaves(self.root))
        if indices != tuple(range(len(indices))):
            raise ValueError(
                "leaves must occur once each in physical order 0, 1, ..., N-1"
            )

    @property
    def leaves(self) -> tuple[Leaf, ...]:
        return _leaves(self.root)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def internal_nodes(self) -> tuple[FusionNode, ...]:
        return _internal_postorder(self.root)

    @classmethod
    def from_nested(cls, nested: int | tuple[object, object]) -> "FusionTree":
        """Construct from nested tuples, e.g. ``(((0, 1), 2), 3)``."""

        def build(value: int | tuple[object, object]) -> TreeNode:
            if isinstance(value, int) and not isinstance(value, bool):
                return Leaf(value)
            if isinstance(value, tuple) and len(value) == 2:
                return FusionNode(build(value[0]), build(value[1]))  # type: ignore[arg-type]
            raise TypeError("tree descriptions contain leaf integers or binary tuples")

        return cls(build(nested))

    @classmethod
    def left_associated(cls, n: int) -> "FusionTree":
        if n < 1:
            raise ValueError("N must be positive")
        node: TreeNode = Leaf(0)
        for index in range(1, n):
            node = FusionNode(node, Leaf(index))
        return cls(node)

    @classmethod
    def balanced(cls, n: int) -> "FusionTree":
        if n < 1:
            raise ValueError("N must be positive")

        def build(start: int, stop: int) -> TreeNode:
            if stop - start == 1:
                return Leaf(start)
            middle = (start + stop) // 2
            return FusionNode(build(start, middle), build(middle, stop))

        return cls(build(0, n))

    def target_is_reachable(self, target_q: int) -> bool:
        """Return whether ``N`` spin-1/2 leaves can fuse to ``target_q``.

        Reachability of the root irrep is independent of the binary tree
        topology.  Checking the triangle/parity condition directly avoids an
        exponential enumeration of all internal labels when constructing a
        scalable tree model.
        """

        validate_q(target_q, name="target_q")
        return target_q <= self.n_leaves and (self.n_leaves - target_q) % 2 == 0

    def enumerate_states(self, target_q: int) -> tuple[FusionTreeState, ...]:
        """Enumerate every admissible assignment of internal irreps."""

        validate_q(target_q, name="target_q")
        nodes = self.internal_nodes
        if not nodes:
            return (FusionTreeState(()),) if target_q == 1 else ()

        def solve(node: TreeNode) -> list[tuple[int, dict[FusionNode, int]]]:
            if isinstance(node, Leaf):
                return [(1, {})]
            answers: list[tuple[int, dict[FusionNode, int]]] = []
            for q_left, labels_left in solve(node.left):
                for q_right, labels_right in solve(node.right):
                    for q_node in coupling_range(q_left, q_right):
                        labels = {**labels_left, **labels_right, node: q_node}
                        answers.append((q_node, labels))
            return answers

        states = []
        for q_root, labels in solve(self.root):
            if q_root == target_q:
                states.append(FusionTreeState(tuple(labels[node] for node in nodes)))
        return tuple(sorted(states, key=lambda state: state.labels))


def node_q(node: TreeNode, labels: dict[FusionNode, int]) -> int:
    """Read a leaf or internal doubled-spin label."""

    return node.q if isinstance(node, Leaf) else labels[node]


def rotate_association_left(node: FusionNode) -> FusionNode:
    """Apply the structural move ``((A,B),C) -> (A,(B,C))``."""

    if not isinstance(node.left, FusionNode):
        raise ValueError("left rotation requires a left internal child")
    return FusionNode(node.left.left, FusionNode(node.left.right, node.right))


def rotate_association_right(node: FusionNode) -> FusionNode:
    """Apply the inverse structural move ``(A,(B,C)) -> ((A,B),C)``."""

    if not isinstance(node.right, FusionNode):
        raise ValueError("right rotation requires a right internal child")
    return FusionNode(FusionNode(node.left, node.right.left), node.right.right)
