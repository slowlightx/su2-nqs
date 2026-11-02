"""Physics-weighted compressibility diagnostics for SU(2) fusion trees."""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode, node_q
from ..lattices.geometry_tree import tree_leaf_distance


def _decisions(tree: FusionTree) -> tuple[tuple[FusionNode, int], ...]:
    values: list[tuple[FusionNode, int]] = []

    def visit(node: TreeNode, depth: int) -> None:
        if isinstance(node, Leaf):
            return
        values.append((node, depth))
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(tree.root, 0)
    return tuple(values)


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def tree_wavefunction_metrics(
    tree: FusionTree,
    amplitudes: np.ndarray,
    *,
    target_q: int = 0,
) -> dict[str, float]:
    """Measure physical-state complexity in one fusion-tree basis.

    The amplitude ordering must match ``tree.enumerate_states(target_q)``.
    Metrics are invariant to overall wavefunction scale and use Born weights.
    """

    states = tree.enumerate_states(target_q)
    values = np.asarray(amplitudes, dtype=np.complex128)
    if values.shape != (len(states),):
        raise ValueError(f"expected amplitudes shape ({len(states)},), got {values.shape}")
    norm = float(np.real(np.vdot(values, values)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("wavefunction norm must be positive and finite")
    probabilities = np.abs(values) ** 2 / norm
    participation_entropy = _entropy(probabilities)

    nodes = tree.internal_nodes
    label_entropies = []
    label_variances = []
    for index in range(max(len(nodes) - 1, 0)):
        distribution: dict[int, float] = defaultdict(float)
        for probability, state in zip(probabilities, states):
            distribution[state.labels[index]] += float(probability)
        weights = np.asarray(tuple(distribution.values()))
        labels = np.asarray(tuple(distribution.keys()), dtype=np.float64)
        mean = float(np.sum(weights * labels))
        label_entropies.append(_entropy(weights))
        label_variances.append(float(np.sum(weights * (labels - mean) ** 2)))

    decisions = _decisions(tree)
    action_sequences = []
    for state in states:
        labels = state.label_map(tree)
        action_sequences.append(
            tuple((node_q(node.left, labels), node_q(node.right, labels)) for node, _ in decisions)
        )
    conditional_entropies = []
    for order in range(len(decisions)):
        prefix_actions: dict[
            tuple[tuple[int, int], ...], dict[tuple[int, int], float]
        ] = defaultdict(lambda: defaultdict(float))
        for probability, actions in zip(probabilities, action_sequences):
            prefix_actions[actions[:order]][actions[order]] += float(probability)
        conditional = 0.0
        for action_weights in prefix_actions.values():
            group_weight = sum(action_weights.values())
            if group_weight <= 0.0:
                continue
            local = np.asarray(tuple(action_weights.values())) / group_weight
            conditional += group_weight * _entropy(local)
        conditional_entropies.append(conditional)
    entropy_sum = sum(conditional_entropies)
    entropy_weighted_depth = (
        sum(
            (depth + 1) * entropy
            for (_, depth), entropy in zip(decisions, conditional_entropies)
        )
        / entropy_sum
        if entropy_sum > 0.0
        else 0.0
    )
    return {
        "participation_entropy": participation_entropy,
        "effective_support": float(2.0**participation_entropy),
        "inverse_participation_support": float(1.0 / np.sum(probabilities**2)),
        "born_conditional_entropy_sum": float(entropy_sum),
        "born_conditional_entropy_peak": float(max(conditional_entropies, default=0.0)),
        "born_entropy_weighted_depth": float(entropy_weighted_depth),
        "mean_block_spin_entropy": float(np.mean(label_entropies)) if label_entropies else 0.0,
        "maximum_block_spin_entropy": float(max(label_entropies, default=0.0)),
        "mean_block_spin_variance": float(np.mean(label_variances)) if label_variances else 0.0,
        "maximum_block_spin_variance": float(max(label_variances, default=0.0)),
    }


def hamiltonian_basis_metrics(matrix) -> dict[str, float]:
    """Measure operator connectivity and sign structure in a chosen basis."""

    coo = matrix.tocoo() if hasattr(matrix, "tocoo") else None
    if coo is None:
        dense = np.asarray(matrix)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError("Hamiltonian must be a square matrix")
        rows, columns = np.nonzero(dense)
        data = dense[rows, columns]
        dimension = dense.shape[0]
    else:
        if coo.shape[0] != coo.shape[1]:
            raise ValueError("Hamiltonian must be a square matrix")
        rows, columns, data = coo.row, coo.col, coo.data
        dimension = coo.shape[0]
    mask = (rows != columns) & (np.abs(data) > 1e-14)
    rows = np.asarray(rows[mask], dtype=np.int64)
    values = np.asarray(data[mask])
    absolute = np.abs(values)
    row_l1 = np.bincount(rows, weights=absolute, minlength=dimension)
    row_l2_squared = np.bincount(rows, weights=absolute**2, minlength=dimension)
    effective = np.divide(
        row_l1**2,
        row_l2_squared,
        out=np.zeros_like(row_l1),
        where=row_l2_squared > 0.0,
    )
    total_absolute = float(np.sum(absolute))
    return {
        "offdiagonal_nnz": float(len(values)),
        "offdiagonal_nnz_per_row": float(len(values) / dimension),
        "offdiagonal_density": float(
            len(values) / (dimension * max(dimension - 1, 1))
        ),
        "mean_effective_offdiagonal_degree": float(np.mean(effective)),
        "maximum_effective_offdiagonal_degree": float(np.max(effective, initial=0.0)),
        "positive_offdiagonal_fraction": float(np.mean(values > 0.0)) if len(values) else 0.0,
        "positive_offdiagonal_absolute_fraction": (
            float(np.sum(absolute[values > 0.0]) / total_absolute)
            if total_absolute > 0.0
            else 0.0
        ),
    }


def recoupling_locality_hierarchy(
    tree: FusionTree,
    bonds: tuple[tuple[int, int, float], ...] | list[tuple[int, int, float]],
) -> dict[str, object]:
    """Resolve a local Hamiltonian into fusion-tree recoupling ranges.

    ``K`` is the leaf-to-leaf tree distance of a physical bond.  The cumulative
    curve reports the fraction of absolute Hamiltonian weight representable by
    recoupling paths no longer than ``K``.  This diagnoses *representation*
    locality; it is deliberately not presented as wavefunction locality.
    """

    shell_weight: dict[int, float] = defaultdict(float)
    total = 0.0
    for first, second, coupling in bonds:
        weight = abs(float(coupling))
        if weight == 0.0:
            continue
        distance = tree_leaf_distance(tree, int(first), int(second))
        shell_weight[distance] += weight
        total += weight
    if total <= 0.0:
        raise ValueError("at least one bond must have nonzero coupling")
    cumulative = 0.0
    shells = []
    for distance in sorted(shell_weight):
        weight = shell_weight[distance]
        cumulative += weight
        shells.append(
            {
                "K": distance,
                "weight": weight,
                "fraction": weight / total,
                "cumulative_fraction": cumulative / total,
            }
        )
    mean_distance = sum(distance * weight for distance, weight in shell_weight.items()) / total
    k90 = next(item["K"] for item in shells if item["cumulative_fraction"] >= 0.9)
    return {
        "total_absolute_weight": total,
        "mean_recoupling_range": mean_distance,
        "maximum_recoupling_range": max(shell_weight),
        "range_for_90_percent_weight": k90,
        "shells": shells,
    }
