import numpy as np

from su2_nqs.benchmarks.causal_residual import (
    compact_tree_action_sequences,
    martingale_residual_increments,
    martingale_residual_decomposition,
    residual_concentration_summary,
)
from su2_nqs.su2.fusion_tree import FusionTree


def test_action_sequences_are_valid_and_unique_for_singlet_basis():
    tree = FusionTree.left_associated(8)
    states = np.asarray(tuple(state.labels for state in tree.enumerate_states(0)))
    actions = compact_tree_action_sequences(tree, 0, states)
    assert actions.shape == states.shape
    assert len({tuple(row) for row in actions}) == len(states)


def test_martingale_variances_and_entropies_obey_exact_chain_rules():
    tree = FusionTree.left_associated(6)
    states = np.asarray(tuple(state.labels for state in tree.enumerate_states(0)))
    actions = compact_tree_action_sequences(tree, 0, states)
    probabilities = np.asarray([0.07, 0.13, 0.19, 0.23, 0.38])
    values = np.asarray([1.0 + 0.2j, -0.4j, 2.1, -0.7 + 1.2j, 0.3 - 0.8j])
    result = martingale_residual_decomposition(probabilities, values, actions)
    expected_entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    np.testing.assert_allclose(result["explained_variance"], result["variance"], atol=1e-13)
    np.testing.assert_allclose(result["orthogonality_error"], 0.0, atol=1e-13)
    np.testing.assert_allclose(
        result["conditional_entropy_sum_bits"], expected_entropy, atol=1e-13
    )
    np.testing.assert_allclose(
        result["decisions"][-1]["cumulative_variance_fraction"], 1.0, atol=1e-13
    )
    increments, mean = martingale_residual_increments(
        probabilities, values, actions
    )
    np.testing.assert_allclose(
        np.sum(increments, axis=1), values - mean, atol=1e-13
    )
    weighted_gram = increments.conj().T @ (probabilities[:, None] * increments)
    np.testing.assert_allclose(
        weighted_gram,
        np.diag(np.diag(weighted_gram)),
        atol=1e-13,
    )


def test_constant_residual_has_zero_contributions_and_well_defined_tails():
    tree = FusionTree.left_associated(4)
    states = np.asarray(tuple(state.labels for state in tree.enumerate_states(0)))
    actions = compact_tree_action_sequences(tree, 0, states)
    result = martingale_residual_decomposition(
        np.ones(len(states)), np.full(len(states), 2.0 - 0.5j), actions
    )
    assert result["variance"] == 0.0
    assert all(row["variance_contribution"] == 0.0 for row in result["decisions"])


def test_residual_concentration_reports_small_born_mass_tail():
    summary = residual_concentration_summary(
        np.asarray([0.999, 0.001]), np.asarray([0.0, 10.0]), fractions=(0.9,)
    )
    assert summary["concentration"][0]["state_count"] == 1
    np.testing.assert_allclose(summary["concentration"][0]["born_mass"], 0.001)
