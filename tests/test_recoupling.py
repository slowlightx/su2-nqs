import numpy as np

from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.quantum_numbers import coupling_range
from su2_nqs.su2.recoupling import recoupling_matrix, tree_basis_matrix
from su2_nqs.su2.wigner6j import f_move


def test_all_small_allowed_f_matrices_are_orthogonal():
    for q1 in range(1, 5):
        for q2 in range(1, 4):
            for q3 in range(1, 4):
                totals = {
                    total
                    for q12 in coupling_range(q1, q2)
                    for total in coupling_range(q12, q3)
                }
                for total in totals:
                    matrix, left, right = f_move(q1, q2, q3, total)
                    assert len(left) == len(right)
                    np.testing.assert_allclose(
                        matrix.T @ matrix,
                        np.eye(len(left)),
                        atol=2e-13,
                    )


def test_wigner_f_move_matches_explicit_computational_basis_overlap():
    left_tree = FusionTree.from_nested(((0, 1), 2))
    right_tree = FusionTree.from_nested((0, (1, 2)))
    explicit, _, _ = recoupling_matrix(left_tree, right_tree, 1, 1)
    wigner, left_channels, right_channels = f_move(1, 1, 1, 1)
    assert left_channels == right_channels == (0, 2)
    np.testing.assert_allclose(wigner, explicit, atol=2e-14)


def test_generic_trees_have_same_fixed_spin_space_and_unitary_change_of_basis():
    left = FusionTree.left_associated(6)
    balanced = FusionTree.balanced(6)
    left_basis, left_states = tree_basis_matrix(left, 0, 0)
    balanced_basis, balanced_states = tree_basis_matrix(balanced, 0, 0)
    assert len(left_states) == len(balanced_states) == 5
    np.testing.assert_allclose(left_basis.conj().T @ left_basis, np.eye(5), atol=2e-14)
    np.testing.assert_allclose(
        balanced_basis.conj().T @ balanced_basis, np.eye(5), atol=2e-14
    )
    transform = balanced_basis.conj().T @ left_basis
    np.testing.assert_allclose(transform.conj().T @ transform, np.eye(5), atol=2e-14)

