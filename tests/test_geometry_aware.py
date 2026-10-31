import numpy as np

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix
from su2_nqs.lattices.geometry_tree import morton_order, square_geometry_tree
from su2_nqs.operators.permutation import (
    HeisenbergGraphFusion,
    heisenberg_graph_fusion_matrix,
)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.recoupling import recoupling_matrix
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse


def test_sparse_f_move_sequence_matches_explicit_small_basis_recoupling():
    left = FusionTree.left_associated(6)
    balanced = FusionTree.balanced(6)
    sparse_matrix, steps = recoupling_matrix_sparse(left, balanced, 0)
    explicit, _, _ = recoupling_matrix(left, balanced, 0, 0)
    assert steps
    np.testing.assert_allclose(sparse_matrix.toarray(), explicit, atol=3e-14)
    np.testing.assert_allclose(
        (sparse_matrix.T @ sparse_matrix).toarray(), np.eye(5), atol=3e-14
    )


def test_arbitrary_graph_permutation_operator_matches_computational_projection():
    bonds = ((0, 5, 0.7), (1, 4, -0.2), (2, 3, 1.0))
    paths = enumerate_paths(6, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    computational, _ = heisenberg_sz_matrix(6, bonds)
    fusion = heisenberg_graph_fusion_matrix(6, bonds, target_q=0).toarray()
    np.testing.assert_allclose(
        fusion, transform.conj().T @ computational @ transform, atol=3e-14
    )


def test_graph_operator_wrappers_reproduce_every_exact_matrix_column():
    bonds = ((0, 5, 0.7), (1, 4, -0.2), (2, 3, 1.0), (0, 1, 0.4))
    fusion = HeisenbergGraphFusion(6, bonds, target_q=0)
    expected_fusion = heisenberg_graph_fusion_matrix(
        6, bonds, target_q=0
    ).toarray()
    np.testing.assert_allclose(fusion.to_dense(), expected_fusion, atol=2e-14)
    for column, path in enumerate(fusion.basis):
        connected, values = fusion.get_conn(path)
        recovered = np.zeros(len(fusion.basis))
        positions = {state: index for index, state in enumerate(fusion.basis)}
        for candidate, value in zip(connected, values):
            recovered[positions[candidate]] += value
        np.testing.assert_allclose(recovered, expected_fusion[:, column], atol=2e-14)

    sz = SzHeisenbergGraph(6, bonds, n_up=3)
    reference, integer_basis = heisenberg_sz_matrix(6, bonds, n_up=3)
    integer_positions = {state: index for index, state in enumerate(integer_basis)}
    order = [
        integer_positions[
            sum(bit << (5 - site) for site, bit in enumerate(configuration))
        ]
        for configuration in enumerate_sz_configs(6, 3)
    ]
    np.testing.assert_allclose(
        sz.to_dense(), reference[np.ix_(order, order)], atol=2e-14
    )


def test_square_geometry_tree_uses_spatial_morton_blocks():
    tree, order = square_geometry_tree(4, 4)
    assert order == morton_order(4, 4)
    assert order[:4] == (0, 1, 4, 5)
    assert len(tree.enumerate_states(0)) == 1430
