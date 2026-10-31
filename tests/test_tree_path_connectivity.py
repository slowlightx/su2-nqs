import numpy as np
import pytest

from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.operators.tree_path_connectivity import (
    tree_heisenberg_graph_connectivity,
    tree_pair_heisenberg_connectivity,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse


def _assert_graph_columns(tree, target_q, bonds):
    left = FusionTree.left_associated(tree.n_leaves)
    recoupling, _ = recoupling_matrix_sparse(left, tree, target_q)
    left_hamiltonian = heisenberg_graph_fusion_matrix(
        tree.n_leaves, bonds, target_q=target_q
    )
    expected = (recoupling @ left_hamiltonian @ recoupling.T).tocsc()
    states = tree.enumerate_states(target_q)
    positions = {state.labels: index for index, state in enumerate(states)}
    maximum_error = 0.0
    for column, state in enumerate(states):
        connected, values = tree_heisenberg_graph_connectivity(
            tree, state, bonds, target_q=target_q
        )
        actual = np.zeros(len(states), dtype=np.float64)
        for new_state, value in zip(connected, values, strict=True):
            actual[positions[new_state.labels]] = value
        maximum_error = max(
            maximum_error,
            float(
                np.max(
                    np.abs(actual - expected.getcol(column).toarray().ravel())
                )
            ),
        )
    assert maximum_error < 8.0e-13


@pytest.mark.parametrize("target_q,expected", ((0, -0.75), (2, 0.25)))
def test_sibling_pair_has_exact_singlet_triplet_energy(target_q, expected):
    tree = FusionTree.balanced(2)
    state = tree.enumerate_states(target_q)[0]
    connected, values = tree_pair_heisenberg_connectivity(tree, state, 0, 1)
    assert connected == (state,)
    np.testing.assert_allclose(values, [expected], atol=2.0e-14)


@pytest.mark.parametrize(
    "n,target_q",
    ((4, 0), (4, 2), (6, 0), (6, 2), (8, 0), (8, 4)),
)
def test_tree_native_graph_connectivity_matches_recoupled_full_matrix(
    n, target_q
):
    tree = FusionTree.balanced(n)
    bonds = tuple(
        (site, (site + 1) % n, 1.0 + 0.07 * site) for site in range(n)
    ) + ((0, n // 2, -0.31), (1, n - 1, 0.23))
    _assert_graph_columns(tree, target_q, bonds)


def test_tree_native_connectivity_rejects_invalid_state_and_bond():
    tree = FusionTree.balanced(4)
    with pytest.raises(ValueError, match="inadmissible"):
        tree_pair_heisenberg_connectivity(tree, (1, 2, 0), 0, 3)
    state = tree.enumerate_states(0)[0]
    with pytest.raises(ValueError, match="distinct"):
        tree_pair_heisenberg_connectivity(tree, state, 1, 1)


def test_tree_native_graph_connectivity_is_not_restricted_to_balanced_trees():
    tree = FusionTree.from_nested(((0, (1, 2)), ((3, 4), (5, (6, 7)))))
    bonds = ((0, 7, 1.0), (1, 5, -0.3), (2, 3, 0.7), (4, 6, 0.2))
    _assert_graph_columns(tree, 0, bonds)
