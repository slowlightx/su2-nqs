"""Exact static autoregressive-complexity metrics for fusion trees."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import log2

from .fusion_tree import FusionNode, FusionTree, Leaf, TreeNode, node_q


@dataclass(frozen=True)
class _Decision:
    node: FusionNode
    depth: int
    left_leaves: int
    right_leaves: int


def _decisions(tree: FusionTree) -> tuple[_Decision, ...]:
    decisions: list[_Decision] = []

    def leaf_count(node: TreeNode) -> int:
        if isinstance(node, Leaf):
            return 1
        return leaf_count(node.left) + leaf_count(node.right)

    def visit(node: TreeNode, depth: int) -> None:
        if isinstance(node, Leaf):
            return
        decisions.append(
            _Decision(node, depth, leaf_count(node.left), leaf_count(node.right))
        )
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(tree.root, 0)
    return tuple(decisions)


@lru_cache(maxsize=None)
def tree_trainability_components(
    tree: FusionTree,
    target_q: int = 0,
) -> dict[str, float]:
    """Return exact top-down branching metrics under uniform fusion states.

    Conditional entropies use base-two logarithms.  Their sum is the
    tree-independent ``log2(Hilbert dimension)``; the peak and entropy-weighted
    decision depth describe how difficult branching is distributed instead.
    """

    states = tree.enumerate_states(target_q)
    if not states:
        raise ValueError(f"target q={target_q} is unreachable")
    if tree.n_leaves == 1:
        return {
            "conditional_entropy_peak": 0.0,
            "conditional_entropy_sum": 0.0,
            "entropy_weighted_depth": 0.0,
            "mean_reachable_actions": 1.0,
            "maximum_reachable_actions": 1.0,
            "path_label_complexity": 1.0,
            "action_vocabulary": 1.0,
            "maximum_candidate_actions": 1.0,
        }

    decisions = _decisions(tree)
    action_sequences: list[tuple[tuple[int, int], ...]] = []
    label_sets = [set() for _ in tree.internal_nodes]
    vocabulary: set[tuple[int, int]] = set()
    for state in states:
        labels = state.label_map(tree)
        for index, q in enumerate(state.labels):
            label_sets[index].add(q)
        actions = tuple(
            (node_q(decision.node.left, labels), node_q(decision.node.right, labels))
            for decision in decisions
        )
        action_sequences.append(actions)
        vocabulary.update(actions)

    total = len(states)
    decision_entropies: list[float] = []
    decision_branching: list[float] = []
    maximum_branching = 1
    for order in range(len(decisions)):
        groups: dict[tuple[tuple[int, int], ...], Counter[tuple[int, int]]] = defaultdict(Counter)
        for actions in action_sequences:
            groups[actions[:order]][actions[order]] += 1
        entropy = 0.0
        branching = 0.0
        for counts in groups.values():
            group_size = sum(counts.values())
            weight = group_size / total
            entropy -= weight * sum(
                (count / group_size) * log2(count / group_size)
                for count in counts.values()
            )
            branching += weight * len(counts)
            maximum_branching = max(maximum_branching, len(counts))
        decision_entropies.append(entropy)
        decision_branching.append(branching)

    entropy_sum = sum(decision_entropies)
    weighted_depth = (
        sum((decision.depth + 1) * entropy for decision, entropy in zip(decisions, decision_entropies))
        / entropy_sum
        if entropy_sum > 0.0
        else 0.0
    )
    maximum_candidates = 1
    for decision in decisions:
        left_values = range(decision.left_leaves % 2, decision.left_leaves + 1, 2)
        right_values = range(decision.right_leaves % 2, decision.right_leaves + 1, 2)
        for parent_q in range(tree.n_leaves + 1):
            candidates = sum(
                abs(left_q - right_q) <= parent_q <= left_q + right_q
                and (left_q + right_q - parent_q) % 2 == 0
                for left_q in left_values
                for right_q in right_values
            )
            maximum_candidates = max(maximum_candidates, candidates)
    return {
        "conditional_entropy_peak": float(max(decision_entropies, default=0.0)),
        "conditional_entropy_sum": float(entropy_sum),
        "entropy_weighted_depth": float(weighted_depth),
        "mean_reachable_actions": float(sum(decision_branching) / len(decisions)),
        "maximum_reachable_actions": float(maximum_branching),
        "path_label_complexity": float(
            sum(len(labels) for labels in label_sets) / len(label_sets)
        ),
        "action_vocabulary": float(len(vocabulary)),
        "maximum_candidate_actions": float(maximum_candidates),
    }
