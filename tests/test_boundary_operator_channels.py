import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.boundary_operator_channels import (
    BoundaryBond,
    advance_boundary_recoupling_plan,
    boundary_operator_dense,
    compile_boundary_recoupling_plan,
    ordinary_operator_schmidt_singular_values,
    partition_boundary_bonds,
    prefix_suffix_operator_matrix,
    reconstruct_prefix_suffix_operator,
)
from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree


def _graph(n: int) -> tuple[BoundaryBond, ...]:
    chain = tuple(BoundaryBond(site, site + 1, 1.0, 0) for site in range(n - 1))
    diagonals = tuple(
        BoundaryBond(site, site + 2, 0.37, 1) for site in range(n - 2)
    )
    return chain + diagonals


@pytest.mark.parametrize("n", (4, 6, 8, 10))
def test_boundary_plans_reconstruct_every_cut_and_advance_exactly(n: int) -> None:
    bonds = _graph(n)
    for decision in range(n - 1):
        plan = compile_boundary_recoupling_plan(n, bonds, decision)
        np.testing.assert_allclose(
            plan.reconstruct_coupling_matrix(), plan.coupling_matrix, atol=2e-13
        )
        assert plan.tensor_rank_q == 2
        assert plan.ordinary_magnetic_schmidt_rank == 3 * plan.channel_rank
        assert plan.channel_rank <= plan.endpoint_channel_count

        parts = partition_boundary_bonds(n, bonds, decision)
        assert plan.hidden_scalar_channel_count == int(bool(parts["hidden"]))
        assert plan.shell_resolved_hidden_scalar_channel_count == len(
            {bond.channel for bond in parts["hidden"]}
        )
        assert plan.forward_cost_estimate(32)["minimal_total_real_head_madds"] == (
            2 * (plan.channel_rank + plan.hidden_scalar_channel_count) * 32
        )
        reconstructed = []
        for left_index, first in enumerate(plan.hidden_sites):
            for right_index, second in enumerate(plan.prefix_sites):
                coupling = plan.reconstruct_coupling_matrix()[left_index, right_index]
                if abs(coupling) > 1e-13:
                    reconstructed.append((int(first), int(second), float(coupling)))
        expected_crossing = tuple(
            (bond.first, bond.second, bond.coupling) for bond in parts["crossing"]
        )
        expected = heisenberg_graph_fusion_matrix(
            n, expected_crossing, target_q=0
        )
        actual = heisenberg_graph_fusion_matrix(n, reconstructed, target_q=0)
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=3e-12)

        if decision < n - 2:
            advanced = advance_boundary_recoupling_plan(plan, bonds)
            direct = compile_boundary_recoupling_plan(n, bonds, decision + 1)
            np.testing.assert_array_equal(advanced.hidden_sites, direct.hidden_sites)
            np.testing.assert_array_equal(advanced.prefix_sites, direct.prefix_sites)
            np.testing.assert_allclose(
                advanced.reconstruct_coupling_matrix(),
                direct.reconstruct_coupling_matrix(),
                atol=2e-13,
            )


@pytest.mark.parametrize("n,decision", ((4, 1), (6, 2)))
def test_su2_vector_rank_is_one_third_of_magnetic_operator_schmidt_rank(
    n: int, decision: int
) -> None:
    plan = compile_boundary_recoupling_plan(n, _graph(n), decision)
    operator = boundary_operator_dense(plan)
    spectrum = ordinary_operator_schmidt_singular_values(
        operator, len(plan.hidden_sites), len(plan.prefix_sites)
    )
    tolerance = 2e-12 * max(1.0, float(spectrum[0]))
    ordinary_rank = int(np.sum(spectrum > tolerance))
    assert ordinary_rank == 3 * plan.channel_rank

    mask = plan.valid_boundary_q_transitions(n)
    assert mask[0, 2]
    assert not mask[0, 0]
    assert mask[1, 1]
    assert mask[1, 3]
    for old, new in np.argwhere(mask):
        assert int(new - old) in plan.boundary_delta_q


def test_prefix_suffix_matricization_preserves_h_new_old_orientation() -> None:
    n = 6
    tree = FusionTree.left_associated(n)
    basis = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    actions = compact_tree_action_sequences(tree, 0, basis)
    rng = np.random.default_rng(606)
    dense = rng.normal(size=(len(basis), len(basis))) + 1j * rng.normal(
        size=(len(basis), len(basis))
    )
    dense[np.abs(dense) < 0.55] = 0.0
    # Keep it deliberately non-Hermitian so H[new,old] cannot be transposed.
    matrix = sparse.csr_matrix(dense)
    for prefix_length in range(actions.shape[1] + 1):
        matricized = prefix_suffix_operator_matrix(
            matrix, actions, prefix_length
        )
        reconstructed = reconstruct_prefix_suffix_operator(
            matricized, actions
        )
        np.testing.assert_allclose(reconstructed, dense, atol=1e-14)
