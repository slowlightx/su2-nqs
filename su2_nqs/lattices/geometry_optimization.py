"""Geometry-cost optimization for ordered binary fusion trees."""

from __future__ import annotations

from dataclasses import dataclass

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode
from ..su2.tree_trainability import tree_trainability_components
from ..su2.tree_rotations import rotate_structure
from .geometry_tree import tree_leaf_distance


def weighted_tree_distance(
    tree: FusionTree,
    bonds: tuple[tuple[int, int, float], ...],
) -> float:
    """Return ``sum_b |J_b| d_T(b)`` for leaf-indexed weighted bonds."""

    total = 0.0
    for first, second, coupling in bonds:
        if not 0 <= first < tree.n_leaves or not 0 <= second < tree.n_leaves:
            raise ValueError("bond leaf lies outside the fusion tree")
        total += abs(float(coupling)) * tree_leaf_distance(tree, first, second)
    return total


def leaf_depths(tree: FusionTree) -> tuple[int, ...]:
    """Return root-to-leaf depths in the tree's deterministic leaf order."""

    depths: dict[int, int] = {}

    def visit(node: TreeNode, depth: int) -> None:
        if isinstance(node, Leaf):
            depths[node.index] = depth
            return
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(tree.root, 0)
    return tuple(depths[leaf.index] for leaf in tree.leaves)


@dataclass(frozen=True)
class TreeObjectiveWeights:
    """Weights for locality and generation-depth terms in a tree objective."""

    bond_distance: float = 1.0
    maximum_depth: float = 0.0
    depth_variance: float = 0.0
    conditional_entropy_peak: float = 0.0
    entropy_weighted_depth: float = 0.0
    mean_reachable_actions: float = 0.0
    path_label_complexity: float = 0.0
    action_vocabulary: float = 0.0
    maximum_candidate_actions: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if min(values) < 0.0:
            raise ValueError("tree objective weights must be non-negative")
        if sum(values) == 0.0:
            raise ValueError("at least one tree objective weight must be positive")


def tree_objective_components(
    tree: FusionTree,
    bonds: tuple[tuple[int, int, float], ...],
) -> dict[str, float]:
    """Return the three unweighted geometry objective components."""

    depths = leaf_depths(tree)
    mean_depth = sum(depths) / len(depths)
    variance = sum((depth - mean_depth) ** 2 for depth in depths) / len(depths)
    return {
        "bond_distance": weighted_tree_distance(tree, bonds),
        "maximum_depth": float(max(depths)),
        "depth_variance": float(variance),
        **tree_trainability_components(tree),
    }


def tree_objective(
    tree: FusionTree,
    bonds: tuple[tuple[int, int, float], ...],
    weights: TreeObjectiveWeights = TreeObjectiveWeights(),
) -> float:
    """Evaluate a weighted bond-locality and generation-depth objective."""

    components = tree_objective_components(tree, bonds)
    return float(
        weights.bond_distance * components["bond_distance"]
        + weights.maximum_depth * components["maximum_depth"]
        + weights.depth_variance * components["depth_variance"]
        + weights.conditional_entropy_peak * components["conditional_entropy_peak"]
        + weights.entropy_weighted_depth * components["entropy_weighted_depth"]
        + weights.mean_reachable_actions * components["mean_reachable_actions"]
        + weights.path_label_complexity * components["path_label_complexity"]
        + weights.action_vocabulary * components["action_vocabulary"]
        + weights.maximum_candidate_actions * components["maximum_candidate_actions"]
    )


def _rotation_candidates(tree: FusionTree):
    candidates: list[tuple[tuple[int, ...], str]] = []

    def visit(node: TreeNode, path: tuple[int, ...]) -> None:
        if isinstance(node, Leaf):
            return
        if isinstance(node.left, FusionNode):
            candidates.append((path, "left"))
        if isinstance(node.right, FusionNode):
            candidates.append((path, "right"))
        visit(node.left, path + (0,))
        visit(node.right, path + (1,))

    visit(tree.root, ())
    return tuple(candidates)


def optimize_fusion_tree(
    initial: FusionTree,
    bonds: tuple[tuple[int, int, float], ...],
    *,
    max_moves: int = 256,
    weights: TreeObjectiveWeights = TreeObjectiveWeights(),
) -> tuple[FusionTree, tuple[dict[str, object], ...]]:
    """Steepest descent over local, F-move-compatible tree rotations.

    Leaf order stays fixed.  The deterministic trace contains every accepted
    rotation and its objective components.  The default weights preserve the
    original distance-only optimization.
    """

    if max_moves < 0:
        raise ValueError("max_moves must be non-negative")
    current = initial
    cost = tree_objective(current, bonds, weights)
    components = tree_objective_components(current, bonds)
    trace: list[dict[str, object]] = [
        {"move": 0, "path": None, "direction": None, "cost": cost, **components}
    ]
    for move in range(1, max_moves + 1):
        best_tree = current
        best_step: tuple[tuple[int, ...], str] | None = None
        best_cost = cost
        for path, direction in _rotation_candidates(current):
            candidate = rotate_structure(current, path, direction)
            candidate_cost = tree_objective(candidate, bonds, weights)
            if candidate_cost < best_cost - 1e-12:
                best_tree = candidate
                best_step = (path, direction)
                best_cost = candidate_cost
        if best_step is None:
            break
        current = best_tree
        cost = best_cost
        components = tree_objective_components(current, bonds)
        trace.append(
            {
                "move": move,
                "path": best_step[0],
                "direction": best_step[1],
                "cost": cost,
                **components,
            }
        )
    return current, tuple(trace)
