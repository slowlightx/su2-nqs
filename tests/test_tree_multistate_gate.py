from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    init_compact_tree_gru,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse
from su2_nqs.vmc.sampled_multistate import sampled_subspace_moments
from su2_nqs.vmc.tree_multistate_gate import (
    D1A_TREE_GATE_PROPOSAL,
    D1A_TREE_GATE_SCIENTIFIC_SCOPE,
    compact_tree_three_root_ritz_force_gate,
    sample_stratified_three_root_compact_tree,
)


jax.config.update("jax_enable_x64", True)


def _small_tree_problem():
    tree = FusionTree.balanced(6)
    config = CompactTreeGRUConfig(tree, 0, embedding_dim=3, hidden_dim=5)
    params = tuple(
        init_compact_tree_gru(key, config)
        for key in jax.random.split(jax.random.PRNGKey(91400), 3)
    )
    bonds = tuple((site, (site + 1) % 6, 1.0) for site in range(6)) + (
        (0, 3, 0.37),
        (1, 4, -0.19),
    )
    return config, params, bonds


def _dense_tree_hamiltonian(config, bonds):
    left = FusionTree.left_associated(config.n)
    recoupling, _ = recoupling_matrix_sparse(left, config.tree, config.target_q)
    left_hamiltonian = heisenberg_graph_fusion_matrix(
        config.n, bonds, target_q=config.target_q
    )
    return (recoupling @ left_hamiltonian @ recoupling.T).toarray()


def test_three_root_tree_gate_matches_direct_sampled_dense_moments() -> None:
    config, params, bonds = _small_tree_problem()
    result = compact_tree_three_root_ritz_force_gate(
        params,
        jax.random.PRNGKey(91401),
        8,
        config,
        bonds,
        registered_widths=(4, 8, 16, 32, 64),
        sampling_microbatch_size=4,
        action_microbatch_size=3,
        devices=(jax.local_devices()[0],),
    )

    states = config.tree.enumerate_states(config.target_q)
    labels = jnp.asarray([state.labels for state in states])
    full_logs = np.stack(
        [
            np.asarray(
                jax.vmap(
                    lambda state: compact_tree_log_psi(root, state, config)
                )(labels)
            )
            for root in params
        ],
        axis=1,
    )
    full_amplitudes = np.exp(full_logs)
    hamiltonian = _dense_tree_hamiltonian(config, bonds)
    full_actions = hamiltonian @ full_amplitudes
    positions = {state.labels: index for index, state in enumerate(states)}
    sampled_positions = np.asarray(
        [positions[tuple(map(int, row))] for row in result.samples.labels]
    )
    expected_amplitudes = full_amplitudes[sampled_positions]
    expected_actions = full_actions[sampled_positions]
    expected_proposal = np.mean(np.abs(expected_amplitudes) ** 2, axis=1)
    expected_moments = sampled_subspace_moments(
        expected_amplitudes,
        expected_actions,
        expected_proposal,
    )

    np.testing.assert_allclose(
        result.hamiltonian_actions, expected_actions, rtol=3.0e-12, atol=3.0e-12
    )
    np.testing.assert_allclose(
        result.proposal_probabilities,
        expected_proposal,
        rtol=3.0e-13,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        result.reference.moments.overlap,
        expected_moments.overlap,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        result.reference.moments.hamiltonian,
        expected_moments.hamiltonian,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        result.reference.moments.action_gram,
        expected_moments.action_gram,
        rtol=4.0e-12,
        atol=4.0e-12,
    )
    assert result.scientific_scope == D1A_TREE_GATE_SCIENTIFIC_SCOPE
    assert result.proposal_kind == D1A_TREE_GATE_PROPOSAL
    assert "no Gamma.A1" in result.scientific_scope
    assert result.samples.sample_count == 24
    np.testing.assert_array_equal(
        np.bincount(result.samples.sampled_root_ids), [8, 8, 8]
    )
    assert result.samples.next_root_sample_indices == (8, 8, 8)
    assert result.reference.ritz.rank == 3
    assert not result.reference.exact_normalized_overlap_diagonal
    assert result.reference.overlap_diagonal_policy == "sampled"
    assert np.isfinite(result.force.gradient_norm)
    assert result.maximum_root_importance_ratio <= 3.0 * (1.0 + 1.0e-12)
    assert np.all(result.root_ratio_kish_ess_fractions > 0.0)
    assert result.execution.maximum_resident_connectivity_slots == (
        3 * max(result.execution.active_bucket_widths)
    )
    assert result.execution.n_sites == 6
    assert result.execution.root_count == 3


def test_stratified_root_stream_is_replayable_from_per_root_indices() -> None:
    config, params, _ = _small_tree_problem()
    arguments = dict(
        params=params,
        key=jax.random.PRNGKey(91402),
        samples_per_root=4,
        config=config,
        starting_root_sample_indices=(5, 7, 9),
        sampling_microbatch_size=3,
        devices=(jax.local_devices()[0],),
    )
    first = sample_stratified_three_root_compact_tree(**arguments)
    replay = sample_stratified_three_root_compact_tree(**arguments)

    np.testing.assert_array_equal(first.labels, replay.labels)
    np.testing.assert_array_equal(first.root_global_indices, replay.root_global_indices)
    assert first.starting_root_sample_indices == (5, 7, 9)
    assert first.next_root_sample_indices == (9, 11, 13)
    np.testing.assert_array_equal(
        first.root_global_indices,
        np.asarray([5, 6, 7, 8, 7, 8, 9, 10, 9, 10, 11, 12], dtype=np.uint32),
    )


def test_split_root_stream_matches_uninterrupted_stream() -> None:
    config, params, _ = _small_tree_problem()
    common = dict(
        params=params,
        key=jax.random.PRNGKey(91403),
        config=config,
        sampling_microbatch_size=3,
        devices=(jax.local_devices()[0],),
    )
    whole = sample_stratified_three_root_compact_tree(
        samples_per_root=7,
        starting_root_sample_indices=(5, 7, 9),
        **common,
    )
    first = sample_stratified_three_root_compact_tree(
        samples_per_root=3,
        starting_root_sample_indices=(5, 7, 9),
        **common,
    )
    second = sample_stratified_three_root_compact_tree(
        samples_per_root=4,
        starting_root_sample_indices=first.next_root_sample_indices,
        **common,
    )
    for root in range(3):
        uninterrupted = whole.labels[whole.sampled_root_ids == root]
        resumed = np.concatenate(
            (
                first.labels[first.sampled_root_ids == root],
                second.labels[second.sampled_root_ids == root],
            ),
            axis=0,
        )
        np.testing.assert_array_equal(resumed, uninterrupted)
    assert second.next_root_sample_indices == whole.next_root_sample_indices


def test_three_root_gate_rejects_missing_root_and_non_singlet_sector() -> None:
    config, params, bonds = _small_tree_problem()
    with pytest.raises(ValueError, match="exactly three roots"):
        compact_tree_three_root_ritz_force_gate(
            params[:2], jax.random.PRNGKey(1), 2, config, bonds
        )
    triplet_config = CompactTreeGRUConfig(config.tree, 2, 3, 5)
    triplet_params = tuple(
        init_compact_tree_gru(key, triplet_config)
        for key in jax.random.split(jax.random.PRNGKey(2), 3)
    )
    with pytest.raises(ValueError, match="target_q=0"):
        compact_tree_three_root_ritz_force_gate(
            triplet_params,
            jax.random.PRNGKey(3),
            2,
            triplet_config,
            bonds,
        )
    with pytest.raises(ValueError, match="Ritz-only"):
        compact_tree_three_root_ritz_force_gate(
            params,
            jax.random.PRNGKey(4),
            2,
            config,
            bonds,
            exact_normalized_overlap_diagonal=True,
        )
