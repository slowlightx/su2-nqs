import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.boundary_k1_oracle import (
    boundary_delta_local_contributions,
    causal_centered_features,
    compile_rank_one_boundary_channel,
    conditional_suffix_oracle,
    rank_one_fusion_operator,
)
from su2_nqs.benchmarks.boundary_operator_channels import (
    BoundaryBond,
    compile_boundary_recoupling_plan,
    partition_boundary_bonds,
)
from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.models.tree_gru import tree_decisions
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree


def _graph(n: int) -> tuple[BoundaryBond, ...]:
    return tuple(
        BoundaryBond(site, site + 1, 1.0, 0) for site in range(n - 1)
    ) + tuple(
        BoundaryBond(site, site + 2, 0.41, 1) for site in range(n - 2)
    )


@pytest.mark.parametrize("n", (4, 6, 8, 10))
def test_minimal_and_endpoint_rank_one_channels_reconstruct_every_crossing(
    n: int,
) -> None:
    bonds = _graph(n)
    for decision in range(n - 1):
        plan = compile_boundary_recoupling_plan(n, bonds, decision)
        svd_coupling = np.zeros_like(plan.coupling_matrix)
        svd_operator = None
        for channel_index in range(plan.channel_rank):
            channel = compile_rank_one_boundary_channel(
                plan, gauge="svd", channel_index=channel_index
            )
            np.testing.assert_allclose(
                channel.reconstruct_coupling_matrix(),
                channel.coupling_matrix,
                atol=2e-14,
            )
            if len(channel.left_coefficients):
                pivot = channel.left_coefficients[
                    int(np.argmax(np.abs(channel.left_coefficients)))
                ]
                assert pivot >= 0.0
            svd_coupling += channel.coupling_matrix
            operator = rank_one_fusion_operator(channel)
            svd_operator = operator if svd_operator is None else svd_operator + operator
        np.testing.assert_allclose(svd_coupling, plan.coupling_matrix, atol=3e-13)

        endpoint_coupling = np.zeros_like(plan.coupling_matrix)
        for channel_index in range(plan.endpoint_channel_count):
            channel = compile_rank_one_boundary_channel(
                plan, gauge="endpoint", channel_index=channel_index
            )
            endpoint_coupling += channel.coupling_matrix
        np.testing.assert_allclose(endpoint_coupling, plan.coupling_matrix, atol=1e-14)

        crossing = partition_boundary_bonds(n, bonds, decision)["crossing"]
        expected = heisenberg_graph_fusion_matrix(
            n,
            tuple((bond.first, bond.second, bond.coupling) for bond in crossing),
            target_q=0,
        )
        if svd_operator is None:
            np.testing.assert_allclose(expected.toarray(), 0.0, atol=2e-13)
        else:
            np.testing.assert_allclose(
                svd_operator.toarray(), expected.toarray(), atol=8e-12
            )


def test_delta_channels_use_h_new_old_and_reconstruct_a_directed_operator() -> None:
    tree = FusionTree.left_associated(4)
    basis = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    # At decision one, basis[:, 0] is the doubled spin of the hidden two-site
    # block and takes values 0 and 2.  Every entry below is q_op=2 allowed.
    matrix = sparse.csr_matrix(
        np.asarray([[1.0 + 0.2j, 2.0 - 0.3j], [3.0 + 0.7j, 4.0 - 0.1j]])
    )
    psi = np.asarray([0.7 + 0.2j, -0.3 + 0.8j])
    result = boundary_delta_local_contributions(matrix, basis, psi, 0)
    expected = np.asarray(matrix.T @ psi) / psi
    wrong_orientation = np.asarray(matrix @ psi) / psi
    np.testing.assert_allclose(result.local_energies, expected, atol=2e-14)
    np.testing.assert_allclose(np.sum(result.values, axis=1), expected, atol=2e-14)
    assert np.max(np.abs(expected - wrong_orientation)) > 0.1
    assert result.maximum_reconstruction_error < 2e-14


def test_suffix_oracle_is_prefix_invariant_and_valid_action_centered() -> None:
    n = 8
    tree = FusionTree.left_associated(n)
    basis = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    actions = compact_tree_action_sequences(tree, 0, basis)
    decision = 3
    plan = compile_boundary_recoupling_plan(n, _graph(n), decision)
    channel = compile_rank_one_boundary_channel(plan, gauge="svd")
    operator = rank_one_fusion_operator(channel)
    rng = np.random.default_rng(803)
    psi = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    psi += 0.2 + 0.1j
    psi /= np.linalg.norm(psi)
    probabilities = np.abs(psi) ** 2
    boundary_index = tree_decisions(tree)[decision].left_index
    contributions = boundary_delta_local_contributions(
        operator, basis, psi, boundary_index
    )
    oracle = conditional_suffix_oracle(
        contributions.values, actions, probabilities, decision
    )
    total_oracle = conditional_suffix_oracle(
        contributions.local_energies[:, None], actions, probabilities, decision
    )
    np.testing.assert_allclose(
        np.sum(oracle.centered_messages, axis=1),
        total_oracle.centered_messages[:, 0],
        atol=3e-13,
    )
    assert oracle.uses_hidden_suffix
    assert oracle.maximum_prefix_invariance_error < 2e-14
    assert oracle.maximum_valid_action_centering_error < 3e-14


def test_causal_feature_gate_rejects_hidden_suffix_and_centers_valid_actions() -> None:
    tree = FusionTree.left_associated(8)
    basis = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    actions = compact_tree_action_sequences(tree, 0, basis)
    probabilities = np.linspace(1.0, 2.0, len(basis))
    probabilities /= np.sum(probabilities)
    decision = 2
    causal = np.column_stack(
        (actions[:, decision], actions[:, decision] ** 2)
    ).astype(np.float64)
    centered = causal_centered_features(
        causal, actions, probabilities, decision
    )
    oracle = conditional_suffix_oracle(
        causal.astype(np.complex128), actions, probabilities, decision
    )
    np.testing.assert_allclose(centered, oracle.centered_messages.real, atol=1e-14)

    leaking = causal.copy()
    leaking[:, 0] += actions[:, decision + 1]
    with pytest.raises(ValueError, match="unrevealed suffix"):
        causal_centered_features(leaking, actions, probabilities, decision)
