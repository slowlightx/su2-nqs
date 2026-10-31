import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.benchmarks.operator_prefix_messages import (
    PrefixOperatorTerm,
    causal_prefix_recoupling_messages,
    maximum_prefix_invariance_error,
)
from su2_nqs.benchmarks.prefix_regression import (
    weighted_complex_prediction_diagnostics,
    weighted_group_ridge_oof,
)
from su2_nqs.models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    compact_tree_prefix_logs,
    init_compact_tree_gru,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree

jax.config.update("jax_enable_x64", True)


def _model_basis(n: int, seed: int = 0):
    tree = FusionTree.left_associated(n)
    config = CompactTreeGRUConfig(tree, 0, embedding_dim=3, hidden_dim=7)
    params = init_compact_tree_gru(jax.random.PRNGKey(seed), config)
    basis = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    conditional, cumulative = jax.vmap(
        lambda state: compact_tree_prefix_logs(params, state, config)
    )(jnp.asarray(basis))
    return tree, config, params, basis, np.asarray(conditional), np.asarray(cumulative)


def test_compact_prefix_logs_are_exact_and_use_no_future_actions() -> None:
    tree, config, params, basis, conditional, cumulative = _model_basis(8, 81)
    actions = compact_tree_action_sequences(tree, 0, basis)
    final = np.asarray(
        jax.vmap(lambda state: compact_tree_log_psi(params, state, config))(
            jnp.asarray(basis)
        )
    )
    np.testing.assert_allclose(cumulative, np.cumsum(conditional, axis=1), atol=1e-14)
    np.testing.assert_allclose(cumulative[:, -1], final, atol=1e-14)

    # A cumulative log is exactly a function of the revealed action prefix.
    for decision in range(actions.shape[1]):
        assert maximum_prefix_invariance_error(
            cumulative[:, decision], actions, decision
        ) < 2e-14


@pytest.mark.parametrize("n", (4, 6, 8, 10))
def test_physical_prefix_recoupling_is_causal_and_final_column_is_exact(
    n: int,
) -> None:
    tree, _, _, basis, _, cumulative = _model_basis(n, 100 + n)
    actions = compact_tree_action_sequences(tree, 0, basis)
    bonds = ((0, 1, 1.0), (max(0, n - 3), n - 1, 0.37))
    terms = tuple(
        PrefixOperatorTerm(
            f"bond_{first}_{second}",
            heisenberg_graph_fusion_matrix(
                n, ((first, second, 1.0),), target_q=0
            ),
            coupling,
            channel,
        )
        for channel, (first, second, coupling) in enumerate(bonds)
    )
    messages = causal_prefix_recoupling_messages(terms, actions, cumulative)
    psi = np.exp(cumulative[:, -1])
    full = heisenberg_graph_fusion_matrix(n, bonds, target_q=0)
    expected = np.asarray(full @ psi) / psi
    actual = messages.values[:, -1, :] @ messages.couplings
    np.testing.assert_allclose(actual, expected, atol=3e-12)
    assert np.all(messages.first_available_decisions < actions.shape[1])
    assert np.max(messages.maximum_message_invariance_error) < 2e-10

    for term, closure in enumerate(messages.first_available_decisions):
        for decision in range(int(closure), actions.shape[1]):
            assert maximum_prefix_invariance_error(
                messages.values[:, decision, term], actions, decision
            ) < 2e-10


def test_final_prefix_message_uses_h_new_old_for_directed_operator() -> None:
    tree, _, _, basis, _, cumulative = _model_basis(4, 404)
    actions = compact_tree_action_sequences(tree, 0, basis)
    # Deliberately non-Hermitian: a row/column reversal cannot pass.
    matrix = sparse.csr_matrix(
        np.asarray([[1.0 + 0.0j, 2.0j], [3.0 + 0.0j, 4.0 - 1.0j]])
    )
    result = causal_prefix_recoupling_messages(
        (PrefixOperatorTerm("directed", matrix),), actions, cumulative
    )
    psi = np.exp(cumulative[:, -1])
    # Connectivity is column-oriented even away from the Hermitian physics
    # case: source ``old`` uses sum_new H[new, old] psi(new) / psi(old).
    expected = np.asarray(matrix.T @ psi) / psi
    np.testing.assert_allclose(result.values[:, -1, 0], expected, atol=2e-13)


def test_grouped_ridge_never_splits_a_prefix_and_generalizes_linear_target() -> None:
    coordinate = np.repeat(np.linspace(-2.0, 2.0, 9), 2)
    groups = np.repeat(np.arange(9), 2)[:, None]
    features = np.column_stack((coordinate, coordinate**2))
    target = (2.5 - 0.7j) * coordinate + (0.3 + 0.2j) * coordinate**2 + 1.1j
    weights = np.linspace(0.5, 1.5, len(target))
    result = weighted_group_ridge_oof(
        features, target, weights, groups, ridge=1e-10, maximum_folds=5
    )
    for group in np.unique(groups):
        assert len(np.unique(result.fold_assignments[groups[:, 0] == group])) == 1
    diagnostics = weighted_complex_prediction_diagnostics(
        target, result.predictions, weights
    )
    assert diagnostics["explained_norm_fraction"] > 0.999999
