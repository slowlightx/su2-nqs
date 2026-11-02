"""Spatial leaf orderings and balanced fusion trees for square lattices."""

from __future__ import annotations

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode


def _morton_code(row: int, column: int, bits: int) -> int:
    code = 0
    for bit in range(bits):
        code |= ((column >> bit) & 1) << (2 * bit)
        code |= ((row >> bit) & 1) << (2 * bit + 1)
    return code


def morton_order(rows: int, columns: int) -> tuple[int, ...]:
    """Row-major physical site ids ordered by a 2D Morton/Z curve."""

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    bits = max(rows, columns).bit_length()
    return tuple(
        sorted(
            range(rows * columns),
            key=lambda site: _morton_code(site // columns, site % columns, bits),
        )
    )


def square_geometry_tree(rows: int, columns: int) -> tuple[FusionTree, tuple[int, ...]]:
    """Balanced tree whose contiguous clusters follow Morton spatial blocks."""

    order = morton_order(rows, columns)
    tree = FusionTree.balanced(len(order))
    return tree, order


def tree_leaf_distance(tree: FusionTree, first: int, second: int) -> int:
    """Number of tree edges between two leaf positions."""

    paths: dict[int, tuple[int, ...]] = {}

    def visit(node: TreeNode, path: tuple[int, ...]) -> None:
        if isinstance(node, Leaf):
            paths[node.index] = path
            return
        visit(node.left, path + (0,))
        visit(node.right, path + (1,))

    visit(tree.root, ())
    left, right = paths[first], paths[second]
    common = 0
    while common < min(len(left), len(right)) and left[common] == right[common]:
        common += 1
    return len(left) + len(right) - 2 * common

