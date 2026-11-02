"""Coupled states and recoupling matrices for generic fusion trees."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .clebsch_gordan import clebsch_gordan
from .fusion_tree import FusionNode, FusionTree, FusionTreeState, Leaf, TreeNode, node_q
from .quantum_numbers import magnetic_numbers, validate_m
from .wigner6j import f_move


def tree_coupled_state(
    tree: FusionTree,
    state: FusionTreeState,
    target_m: int,
) -> np.ndarray:
    """Expand one tree-labelled state in the physical computational basis."""

    labels = state.label_map(tree)
    total_q = 1 if isinstance(tree.root, Leaf) else labels[tree.root]
    validate_m(total_q, target_m, name="target_m")

    @lru_cache(maxsize=None)
    def expand(node: TreeNode) -> tuple[tuple[int, np.ndarray], ...]:
        if isinstance(node, Leaf):
            return (
                (-1, np.asarray([1.0, 0.0], dtype=np.complex128)),
                (1, np.asarray([0.0, 1.0], dtype=np.complex128)),
            )
        q_left = node_q(node.left, labels)
        q_right = node_q(node.right, labels)
        q_node = labels[node]
        left_vectors = dict(expand(node.left))
        right_vectors = dict(expand(node.right))
        results: list[tuple[int, np.ndarray]] = []
        for m_node in magnetic_numbers(q_node):
            vector = np.zeros(
                next(iter(left_vectors.values())).size * next(iter(right_vectors.values())).size,
                dtype=np.complex128,
            )
            for m_left, left_vector in left_vectors.items():
                m_right = m_node - m_left
                right_vector = right_vectors.get(m_right)
                if right_vector is None:
                    continue
                coefficient = clebsch_gordan(
                    q_left,
                    m_left,
                    q_right,
                    m_right,
                    q_node,
                    m_node,
                )
                if coefficient:
                    vector += coefficient * np.kron(left_vector, right_vector)
            results.append((m_node, vector))
        return tuple(results)

    return dict(expand(tree.root))[target_m]


def tree_basis_matrix(
    tree: FusionTree,
    target_q: int,
    target_m: int | None = None,
) -> tuple[np.ndarray, tuple[FusionTreeState, ...]]:
    """Return computational-basis columns for all tree labelings."""

    if target_m is None:
        target_m = target_q
    states = tree.enumerate_states(target_q)
    if not states:
        return np.empty((2**tree.n_leaves, 0), dtype=np.complex128), ()
    columns = [tree_coupled_state(tree, state, target_m) for state in states]
    return np.column_stack(columns), states


def recoupling_matrix(
    source: FusionTree,
    target: FusionTree,
    target_q: int,
    target_m: int | None = None,
) -> tuple[np.ndarray, tuple[FusionTreeState, ...], tuple[FusionTreeState, ...]]:
    """Exact small-system basis change from ``source`` to ``target``.

    The construction through the computational basis is intentionally a
    reference implementation.  Local scalable changes use :func:`f_move`.
    """

    if source.n_leaves != target.n_leaves:
        raise ValueError("source and target trees must have the same leaves")
    source_basis, source_states = tree_basis_matrix(source, target_q, target_m)
    target_basis, target_states = tree_basis_matrix(target, target_q, target_m)
    return target_basis.conj().T @ source_basis, source_states, target_states


def local_f_move(
    q_a: int,
    q_b: int,
    q_c: int,
    q_total: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Local tree rotation matrix for ``((A,B),C) <-> (A,(B,C))``."""

    return f_move(q_a, q_b, q_c, q_total)

