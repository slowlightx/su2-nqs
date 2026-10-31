"""Fusion-tree search guided by physical SU(2) block-spin fluctuations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..benchmarks.tree_physics import tree_wavefunction_metrics
from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode
from ..su2.tree_rotations import rotation_matrix
from .geometry_optimization import TreeObjectiveWeights, tree_objective, tree_objective_components


@dataclass(frozen=True)
class PhysicalTreeWeights:
    """Weights for Born-weighted SU(2) charge-fluctuation penalties."""

    mean_block_spin_entropy: float = 1.0
    maximum_block_spin_entropy: float = 0.0
    mean_block_spin_variance: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if min(values) < 0.0:
            raise ValueError("physical tree weights must be non-negative")
        if sum(values) == 0.0:
            raise ValueError("at least one physical tree weight must be positive")


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


def _physical_components(
    tree: FusionTree,
    amplitudes: tuple[np.ndarray, ...],
    target_q: int,
) -> dict[str, float]:
    metrics = [
        tree_wavefunction_metrics(tree, vector, target_q=target_q)
        for vector in amplitudes
    ]
    names = (
        "participation_entropy",
        "mean_block_spin_entropy",
        "maximum_block_spin_entropy",
        "mean_block_spin_variance",
    )
    return {
        f"physical_{name}": float(np.mean([item[name] for item in metrics]))
        for name in names
    }


def optimize_fusion_tree_physical(
    initial: FusionTree,
    bonds: tuple[tuple[int, int, float], ...],
    reference_amplitudes: tuple[np.ndarray, ...],
    *,
    target_q: int = 0,
    structural_weights: TreeObjectiveWeights = TreeObjectiveWeights(),
    physical_weights: PhysicalTreeWeights = PhysicalTreeWeights(),
    max_moves: int = 64,
) -> tuple[FusionTree, tuple[np.ndarray, ...], tuple[dict[str, object], ...]]:
    """Descend over F-moves while transforming an ensemble of physical states."""

    dimension = len(initial.enumerate_states(target_q))
    amplitudes = tuple(np.asarray(vector) for vector in reference_amplitudes)
    if not amplitudes:
        raise ValueError("at least one reference wavefunction is required")
    if any(vector.shape != (dimension,) for vector in amplitudes):
        raise ValueError(f"every reference must have shape ({dimension},)")

    def evaluate(tree: FusionTree, vectors: tuple[np.ndarray, ...]):
        structural = tree_objective_components(tree, bonds)
        physical = _physical_components(tree, vectors, target_q)
        cost = tree_objective(tree, bonds, structural_weights)
        cost += (
            physical_weights.mean_block_spin_entropy
            * physical["physical_mean_block_spin_entropy"]
            + physical_weights.maximum_block_spin_entropy
            * physical["physical_maximum_block_spin_entropy"]
            + physical_weights.mean_block_spin_variance
            * physical["physical_mean_block_spin_variance"]
        )
        return float(cost), {**structural, **physical}

    current = initial
    cost, components = evaluate(current, amplitudes)
    trace: list[dict[str, object]] = [
        {"move": 0, "path": None, "direction": None, "cost": cost, **components}
    ]
    for move in range(1, max_moves + 1):
        best = None
        for path, direction in _rotation_candidates(current):
            candidate, rotation = rotation_matrix(current, path, target_q, direction)
            candidate_amplitudes = tuple(
                np.asarray(rotation @ vector) for vector in amplitudes
            )
            candidate_cost, candidate_components = evaluate(
                candidate, candidate_amplitudes
            )
            if candidate_cost < cost - 1e-12 and (
                best is None or candidate_cost < best[0] - 1e-12
            ):
                best = (
                    candidate_cost,
                    candidate,
                    candidate_amplitudes,
                    candidate_components,
                    path,
                    direction,
                )
        if best is None:
            break
        cost, current, amplitudes, components, path, direction = best
        trace.append(
            {
                "move": move,
                "path": path,
                "direction": direction,
                "cost": cost,
                **components,
            }
        )
    return current, amplitudes, tuple(trace)
