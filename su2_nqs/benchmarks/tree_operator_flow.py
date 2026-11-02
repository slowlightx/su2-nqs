"""Physical boundary-channel and response diagnostics for fusion trees.

The fusion tree is treated as a coordinate chart.  Locality is supplied by
the physical graph Hamiltonian: every tree edge defines a real-space block,
and the rank of the coupling matrix across that block is the number of
spatial ``q_op=2`` channels required by an exact Heisenberg action.

The response optimizer in this module is an exact small-system oracle.  It
uses a supplied physical state and horizontal Hamiltonian response, rotates
both by local F-moves, and searches for a tree whose Born distribution covers
that response reliably.  It is a benchmark target for a scalable estimator,
not a production algorithm that may inspect an unknown ground state.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from ..lattices.geometry_optimization import leaf_depths, weighted_tree_distance
from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode
from ..su2.tree_rotations import leaf_signature, rotation_matrix
from .tree_physics import tree_wavefunction_metrics


GraphBond = tuple[int, int, float]


def _normalized_bonds(
    tree: FusionTree, bonds: tuple[GraphBond, ...]
) -> tuple[GraphBond, ...]:
    normalized = []
    for bond in bonds:
        if len(bond) != 3:
            raise ValueError("graph bonds must contain two sites and a coupling")
        first, second, coupling = bond
        if (
            isinstance(first, bool)
            or isinstance(second, bool)
            or not isinstance(first, int)
            or not isinstance(second, int)
            or first == second
            or not 0 <= first < tree.n_leaves
            or not 0 <= second < tree.n_leaves
        ):
            raise ValueError(f"invalid graph bond: {bond!r}")
        value = float(coupling)
        if not np.isfinite(value):
            raise ValueError("graph couplings must be finite")
        if value != 0.0:
            normalized.append((min(first, second), max(first, second), value))
    if not normalized:
        raise ValueError("at least one graph bond must have nonzero coupling")
    return tuple(normalized)


def _edge_regions(tree: FusionTree) -> tuple[tuple[TreeNode, int], ...]:
    """Return the child-side region of every edge, including leaf edges."""

    regions: list[tuple[TreeNode, int]] = []

    def visit(node: TreeNode, depth: int) -> None:
        if node != tree.root:
            regions.append((node, depth))
        if isinstance(node, Leaf):
            return
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(tree.root, 0)
    return tuple(regions)


def tree_boundary_channel_metrics(
    tree: FusionTree,
    bonds: tuple[GraphBond, ...],
    *,
    rank_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Return exact spatial ``q_op=2`` channel ranks on all tree edges.

    For a subtree ``A``, the crossing interaction is

    ``H_x = sum_alpha sigma_alpha A_alpha^(1).B_alpha^(1)``.

    The SVD rank of its coupling matrix is invariant under rotations of the
    spatial channel gauge.  SU(2) packages the three magnetic components of
    every spatial channel into one reduced vector channel.
    """

    if (
        isinstance(rank_tolerance, bool)
        or not isinstance(rank_tolerance, Real)
        or not np.isfinite(float(rank_tolerance))
        or rank_tolerance < 0.0
    ):
        raise ValueError("rank_tolerance must be finite and non-negative")
    normalized = _normalized_bonds(tree, bonds)
    all_sites = set(range(tree.n_leaves))
    rows = []
    for node, depth in _edge_regions(tree):
        region = tuple(leaf_signature(node))
        complement = tuple(sorted(all_sites - set(region)))
        coupling = np.zeros((len(region), len(complement)), dtype=np.float64)
        left_positions = {site: index for index, site in enumerate(region)}
        right_positions = {site: index for index, site in enumerate(complement)}
        crossing_absolute_weight = 0.0
        crossing_bonds = 0
        for first, second, value in normalized:
            if first in left_positions and second in right_positions:
                coupling[left_positions[first], right_positions[second]] += value
            elif second in left_positions and first in right_positions:
                coupling[left_positions[second], right_positions[first]] += value
            else:
                continue
            crossing_absolute_weight += abs(value)
            crossing_bonds += 1
        singular_values = np.linalg.svd(coupling, compute_uv=False)
        threshold = (
            float(rank_tolerance) * float(singular_values[0])
            if len(singular_values) and singular_values[0] > 0.0
            else float(rank_tolerance)
        )
        active = singular_values[singular_values > threshold]
        squared = active**2
        effective_rank = (
            float(np.sum(squared) ** 2 / np.sum(squared**2))
            if len(active) and np.sum(squared**2) > 0.0
            else 0.0
        )
        rows.append(
            {
                "region": list(region),
                "region_size": len(region),
                "tree_depth": depth,
                "crossing_bond_count": crossing_bonds,
                "crossing_absolute_weight": crossing_absolute_weight,
                "spatial_channel_rank": int(len(active)),
                "effective_spatial_channel_rank": effective_rank,
                "singular_values": active.tolist(),
            }
        )
    weighted_denominator = sum(
        float(row["crossing_absolute_weight"]) for row in rows
    )
    weighted_rank = (
        sum(
            float(row["crossing_absolute_weight"])
            * float(row["effective_spatial_channel_rank"])
            for row in rows
        )
        / weighted_denominator
        if weighted_denominator > 0.0
        else 0.0
    )
    return {
        "maximum_spatial_channel_rank": max(
            (int(row["spatial_channel_rank"]) for row in rows), default=0
        ),
        "total_spatial_channel_rank": sum(
            int(row["spatial_channel_rank"]) for row in rows
        ),
        "crossing_weighted_effective_rank": float(weighted_rank),
        "maximum_crossing_absolute_weight": max(
            (float(row["crossing_absolute_weight"]) for row in rows),
            default=0.0,
        ),
        "cuts": rows,
    }


def source_target_sampling_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    sample_count: int,
) -> dict[str, float]:
    """Measure how iid source-Born samples cover a target vector."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    source = np.asarray(source, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("source and target must be equal-length vectors")
    source_norm = float(np.real(np.vdot(source, source)))
    target_norm = float(np.real(np.vdot(target, target)))
    if source_norm <= 0.0 or target_norm <= 0.0:
        raise ValueError("source and target norms must be positive")
    probability = np.abs(source) ** 2 / source_norm
    target_weight = np.abs(target) ** 2 / target_norm
    affinity = float(np.sum(np.sqrt(probability * target_weight)))
    unseen = float(
        np.sum(target_weight * np.power(1.0 - probability, sample_count))
    )
    return {
        "source_probability_participation": float(
            1.0 / np.sum(probability**2)
        ),
        "target_weight_participation": float(
            1.0 / np.sum(target_weight**2)
        ),
        "born_target_affinity_squared": affinity**2,
        "expected_unseen_target_weight": unseen,
        "target_weight_outside_source_support": float(
            np.sum(target_weight[probability <= 1.0e-15])
        ),
    }


def tree_response_metrics(
    tree: FusionTree,
    bonds: tuple[GraphBond, ...],
    source: np.ndarray,
    response: np.ndarray,
    *,
    target_q: int = 0,
    sample_count: int = 4096,
) -> dict[str, object]:
    """Combine graph-channel, source-state, and K=1 response diagnostics."""

    source = np.asarray(source, dtype=np.complex128)
    response = np.asarray(response, dtype=np.complex128)
    dimension = len(tree.enumerate_states(target_q))
    if source.shape != (dimension,) or response.shape != (dimension,):
        raise ValueError(f"source and response must have shape ({dimension},)")
    return {
        "sampling": source_target_sampling_metrics(
            source, response, sample_count=sample_count
        ),
        "source_state": tree_wavefunction_metrics(
            tree, source, target_q=target_q
        ),
        "response_state": tree_wavefunction_metrics(
            tree, response, target_q=target_q
        ),
        "operator_flow": tree_boundary_channel_metrics(tree, bonds),
        "weighted_tree_distance": weighted_tree_distance(tree, bonds),
        "maximum_leaf_depth": max(leaf_depths(tree)),
    }


@dataclass(frozen=True)
class ResponseTreeWeights:
    """Weights for an exact K=1 response-aware tree oracle."""

    unseen_target_weight: float = 1.0
    one_minus_affinity_squared: float = 0.0
    target_participation_entropy: float = 0.0
    maximum_channel_rank: float = 0.0
    total_channel_rank: float = 0.0
    maximum_leaf_depth: float = 0.0
    bond_distance: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("response-tree weights must be finite and non-negative")
        if sum(values) == 0.0:
            raise ValueError("at least one response-tree weight must be positive")


def response_tree_objective(
    tree: FusionTree,
    bonds: tuple[GraphBond, ...],
    source: np.ndarray,
    response: np.ndarray,
    *,
    target_q: int = 0,
    sample_count: int = 4096,
    weights: ResponseTreeWeights = ResponseTreeWeights(),
) -> tuple[float, dict[str, object]]:
    """Evaluate one dimensionless response-aware tree cost."""

    metrics = tree_response_metrics(
        tree,
        bonds,
        source,
        response,
        target_q=target_q,
        sample_count=sample_count,
    )
    sampling = metrics["sampling"]
    flow = metrics["operator_flow"]
    n = tree.n_leaves
    target_entropy = np.log2(
        max(float(sampling["target_weight_participation"]), 1.0)
    ) / np.log2(max(n, 2))
    components = {
        "unseen_target_weight": float(
            sampling["expected_unseen_target_weight"]
        ),
        "one_minus_affinity_squared": 1.0
        - float(sampling["born_target_affinity_squared"]),
        "target_participation_entropy": float(target_entropy),
        "maximum_channel_rank": float(flow["maximum_spatial_channel_rank"]) / n,
        "total_channel_rank": float(flow["total_spatial_channel_rank"]) / (n * n),
        "maximum_leaf_depth": float(metrics["maximum_leaf_depth"]) / n,
        "bond_distance": float(metrics["weighted_tree_distance"])
        / (n * sum(abs(value) for _, _, value in _normalized_bonds(tree, bonds))),
    }
    cost = sum(
        float(getattr(weights, name)) * value
        for name, value in components.items()
    )
    return float(cost), {**metrics, "objective_components": components}


def _rotation_candidates(
    tree: FusionTree,
) -> tuple[tuple[tuple[int, ...], str], ...]:
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


def optimize_fusion_tree_response(
    initial: FusionTree,
    bonds: tuple[GraphBond, ...],
    source: np.ndarray,
    response: np.ndarray,
    *,
    target_q: int = 0,
    sample_count: int = 4096,
    weights: ResponseTreeWeights = ResponseTreeWeights(),
    max_moves: int = 64,
) -> tuple[FusionTree, np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
    """Greedily reduce the K=1 response cost using exact local F-moves."""

    if max_moves < 0:
        raise ValueError("max_moves must be non-negative")
    dimension = len(initial.enumerate_states(target_q))
    current_source = np.asarray(source, dtype=np.complex128)
    current_response = np.asarray(response, dtype=np.complex128)
    if current_source.shape != (dimension,) or current_response.shape != (dimension,):
        raise ValueError(f"source and response must have shape ({dimension},)")
    current = initial
    cost, metrics = response_tree_objective(
        current,
        bonds,
        current_source,
        current_response,
        target_q=target_q,
        sample_count=sample_count,
        weights=weights,
    )
    trace: list[dict[str, object]] = [
        {
            "move": 0,
            "path": None,
            "direction": None,
            "cost": cost,
            "sampling": metrics["sampling"],
            "operator_flow": {
                key: value
                for key, value in metrics["operator_flow"].items()
                if key != "cuts"
            },
            "objective_components": metrics["objective_components"],
        }
    ]
    for move in range(1, max_moves + 1):
        best = None
        for path, direction in _rotation_candidates(current):
            candidate, rotation = rotation_matrix(
                current, path, target_q, direction
            )
            candidate_source = np.asarray(rotation @ current_source)
            candidate_response = np.asarray(rotation @ current_response)
            candidate_cost, candidate_metrics = response_tree_objective(
                candidate,
                bonds,
                candidate_source,
                candidate_response,
                target_q=target_q,
                sample_count=sample_count,
                weights=weights,
            )
            if candidate_cost < cost - 1.0e-12 and (
                best is None or candidate_cost < best[0] - 1.0e-12
            ):
                best = (
                    candidate_cost,
                    candidate,
                    candidate_source,
                    candidate_response,
                    candidate_metrics,
                    path,
                    direction,
                )
        if best is None:
            break
        (
            cost,
            current,
            current_source,
            current_response,
            metrics,
            path,
            direction,
        ) = best
        trace.append(
            {
                "move": move,
                "path": path,
                "direction": direction,
                "cost": cost,
                "sampling": metrics["sampling"],
                "operator_flow": {
                    key: value
                    for key, value in metrics["operator_flow"].items()
                    if key != "cuts"
                },
                "objective_components": metrics["objective_components"],
            }
        )
    return current, current_source, current_response, tuple(trace)


__all__ = [
    "ResponseTreeWeights",
    "optimize_fusion_tree_response",
    "response_tree_objective",
    "source_target_sampling_metrics",
    "tree_boundary_channel_metrics",
    "tree_response_metrics",
]
