import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.benchmarks.operator_prefix_messages import (
    connected_local_energies,
    connected_operator_columns_from_sparse,
    operator_message_diagnostics,
    prefix_operator_messages,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.fusion_tree import FusionTree


def test_sparse_columns_use_h_new_old_orientation() -> None:
    # Deliberately non-Hermitian so row/column reversal cannot pass unnoticed.
    matrix = sparse.csr_matrix(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    basis = np.asarray([[0], [2]], dtype=np.int32)
    psi = np.asarray([5.0 + 0.0j, 7.0 + 0.0j])
    columns = connected_operator_columns_from_sparse(matrix, basis, psi)
    expected = np.asarray([(1.0 * 5.0 + 3.0 * 7.0) / 5.0, (2.0 * 5.0 + 4.0 * 7.0) / 7.0])
    np.testing.assert_allclose(connected_local_energies(columns), expected)


@pytest.mark.parametrize("n", (4, 6, 8, 10))
def test_exact_prefix_messages_reconstruct_heisenberg_residual_and_variance(
    n: int,
) -> None:
    bonds = tuple((site, site + 1, 1.0) for site in range(n - 1)) + (
        (0, n - 1, 0.37),
    )
    matrix = heisenberg_graph_fusion_matrix(n, bonds, target_q=0)
    paths = enumerate_paths(n, 0)
    # A left FusionTree stores path[1:] as its postorder internal labels.
    basis = np.asarray([path[1:] for path in paths], dtype=np.int32)
    tree = FusionTree.left_associated(n)
    tree_basis = np.asarray([state.labels for state in tree.enumerate_states(0)])
    np.testing.assert_array_equal(basis, tree_basis)

    rng = np.random.default_rng(900 + n)
    psi = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    psi /= np.linalg.norm(psi)
    columns = connected_operator_columns_from_sparse(matrix, basis, psi)
    actions = compact_tree_action_sequences(tree, 0, basis)
    messages = prefix_operator_messages(columns, actions, np.abs(psi) ** 2)
    diagnostics = operator_message_diagnostics(messages)

    direct_local = np.asarray(matrix @ psi) / psi
    direct_energy = np.vdot(psi, matrix @ psi)
    direct_residual_vector = matrix @ psi - direct_energy * psi
    direct_variance = float(np.vdot(direct_residual_vector, direct_residual_vector).real)
    np.testing.assert_allclose(messages.local_energies, direct_local, atol=2e-13)
    np.testing.assert_allclose(messages.prefix_messages[:, 0], direct_energy, atol=2e-13)
    np.testing.assert_allclose(
        np.sum(messages.increments, axis=1), direct_local - direct_energy, atol=2e-13
    )
    np.testing.assert_allclose(
        diagnostics["local_energy_variance"], direct_variance, atol=3e-13
    )
    np.testing.assert_allclose(
        diagnostics["explained_variance"], direct_variance, atol=3e-13
    )
    assert diagnostics["maximum_reconstruction_error"] < 3e-13
    assert diagnostics["increment_orthogonality_error"] < 3e-13

