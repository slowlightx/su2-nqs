import numpy as np
import pytest

from su2_nqs.benchmarks.boundary_joint_oracle import (
    boundary_sector_suffix_oracles,
    channel_increment_diagnostics,
    compile_boundary_sector_operators,
)
from su2_nqs.benchmarks.boundary_k1_oracle import (
    boundary_delta_local_contributions,
    compile_rank_one_boundary_channel,
    conditional_suffix_oracle,
    rank_one_fusion_operator,
)
from su2_nqs.benchmarks.boundary_operator_channels import (
    BoundaryBond,
    compile_boundary_recoupling_plan,
)
from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.models.tree_gru import tree_decisions
from su2_nqs.su2.fusion_tree import FusionTree


def _graph(n: int) -> tuple[BoundaryBond, ...]:
    return tuple(
        BoundaryBond(site, site + 1, 1.0, 0) for site in range(n - 1)
    ) + tuple(
        BoundaryBond(site, site + 2, 0.43, 1) for site in range(n - 2)
    )


@pytest.mark.parametrize("n", (4, 6, 8, 10))
def test_joint_sector_and_all_vector_channels_reconstruct_every_cut(n: int) -> None:
    tree = FusionTree.left_associated(n)
    basis = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    actions = compact_tree_action_sequences(tree, 0, basis)
    rng = np.random.default_rng(9100 + n)
    psi = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    psi += 0.3 + 0.2j
    psi /= np.linalg.norm(psi)
    probabilities = np.abs(psi) ** 2
    for decision in range(n - 1):
        sectors = compile_boundary_sector_operators(n, _graph(n), decision)
        np.testing.assert_allclose(
            sectors.reconstructed_operator().toarray(),
            sectors.full_operator.toarray(),
            atol=9e-12,
        )
        oracle = boundary_sector_suffix_oracles(
            sectors, psi, actions, probabilities
        )
        assert oracle.maximum_local_energy_reconstruction_error < 2e-11
        assert oracle.maximum_increment_reconstruction_error < 2e-11
        np.testing.assert_allclose(
            oracle.hidden.centered_messages
            + oracle.crossing.centered_messages
            + oracle.prefix.centered_messages,
            oracle.full.centered_messages,
            atol=2e-11,
        )

        plan = compile_boundary_recoupling_plan(n, _graph(n), decision)
        boundary_index = tree_decisions(tree)[decision].left_index
        channel_messages = []
        for channel_index in range(plan.channel_rank):
            channel = compile_rank_one_boundary_channel(
                plan, gauge="svd", channel_index=channel_index
            )
            operator = rank_one_fusion_operator(channel)
            delta = boundary_delta_local_contributions(
                operator, basis, psi, boundary_index
            )
            message = conditional_suffix_oracle(
                delta.values, actions, probabilities, decision
            )
            channel_messages.append(np.sum(message.centered_messages, axis=1))
        crossing = (
            np.sum(np.column_stack(channel_messages), axis=1)
            if channel_messages
            else np.zeros(len(basis), dtype=np.complex128)
        )
        np.testing.assert_allclose(
            crossing, oracle.crossing.centered_messages[:, 0], atol=3e-11
        )

        diagnostics = channel_increment_diagnostics(
            oracle.full.centered_messages[:, 0],
            oracle.noncrossing_centered_messages[:, 0],
            np.column_stack(channel_messages)
            if channel_messages
            else np.zeros((len(basis), 0), dtype=np.complex128),
            probabilities,
        )
        assert diagnostics["final_reconstruction_error"] < 2e-20
        assert diagnostics["maximum_pointwise_reconstruction_error"] < 3e-11
