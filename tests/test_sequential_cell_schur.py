import numpy as np
import jax

from su2_nqs.benchmarks.many_cell_replay import (
    rotated_to_balanced_tree_coefficients,
)
from su2_nqs.benchmarks.sequential_cell_replay import (
    balanced_to_sequential_coefficients,
    sequential_to_rotated_coefficients,
)

from su2_nqs.models.sequential_cell_schur_vit import (
    SequentialCellSchurViTConfig,
    init_sequential_cell_schur_vit,
    sample_sequential_cell_schur_vit,
    sequential_cell_schur_log_amplitudes,
    sequential_cell_schur_spatial_distances,
)

from su2_nqs.su2.many_cell_schur import (
    rectangular_four_spin_layout,
    rectangular_sequential_four_spin_layout,
)
from su2_nqs.su2.sequential_cell_schur import (
    enumerate_sequential_cell_actions,
    labels_to_sequential_actions,
    sequential_actions_to_labels,
    sequential_cell_action_tables,
    sequential_cumulative_indices,
)


def test_cell_sequential_counts_are_complete_singlet_multiplicities():
    expected = {1: 2, 2: 14, 3: 132, 4: 1430}
    for n_cells, dimension in expected.items():
        actions = enumerate_sequential_cell_actions(n_cells, target_q=0)
        assert actions.shape == (dimension, n_cells, 3)


def test_cell_actions_and_tree_labels_are_an_exact_bijection_at_n16():
    layout = rectangular_sequential_four_spin_layout(4, 4)
    tree_labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    actions = labels_to_sequential_actions(layout, tree_labels)
    recovered = sequential_actions_to_labels(layout, actions)
    np.testing.assert_array_equal(recovered, tree_labels)
    enumerated = enumerate_sequential_cell_actions(4, target_q=0)
    encoded = sequential_actions_to_labels(layout, enumerated)
    assert {tuple(row) for row in encoded} == {tuple(row) for row in tree_labels}
    assert sequential_cumulative_indices(layout)[-1] == layout.n_sites - 2


def test_every_candidate_mask_has_an_exact_completion_to_the_singlet():
    for n_cells in (1, 2, 4, 9):
        tables = sequential_cell_action_tables(n_cells, target_q=0)
        prefixes = {0}
        for cell in range(n_cells):
            next_prefixes = set()
            for previous_q in prefixes:
                valid = np.flatnonzero(tables.valid[cell, previous_q])
                assert len(valid) > 0
                next_prefixes.update(
                    int(tables.cumulative_q[cell, previous_q, action])
                    for action in valid
                )
            prefixes = next_prefixes
        assert prefixes == {0}


def test_n36_tables_remain_small_without_enumerating_the_hilbert_space():
    tables = sequential_cell_action_tables(9, target_q=0)
    assert tables.valid.shape[:2] == (9, 37)
    assert tables.action_count <= 16
    assert np.all(~tables.valid[0, 1:])


def _parameter_shapes(params):
    return tuple(tuple(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))


def test_cell_sequential_vit_is_normalized_and_samples_only_valid_states():
    layout = rectangular_sequential_four_spin_layout(4, 2)
    config = SequentialCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    params = init_sequential_cell_schur_vit(jax.random.PRNGKey(31), config)
    states = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    logs = np.asarray(sequential_cell_schur_log_amplitudes(params, states, config))
    np.testing.assert_allclose(np.sum(np.exp(2.0 * logs.real)), 1.0, atol=2.0e-13)
    samples = np.asarray(
        sample_sequential_cell_schur_vit(
            params, jax.random.PRNGKey(32), 128, config
        )
    )
    support = {tuple(row) for row in states}
    assert all(tuple(row) in support for row in samples)


def test_cell_sequential_vit_parameter_shapes_transfer_to_n36_and_samples():
    small = SequentialCellSchurViTConfig(
        rectangular_sequential_four_spin_layout(4, 4),
        embedding_dim=8,
        head_count=2,
        depth=1,
    )
    large = SequentialCellSchurViTConfig(
        rectangular_sequential_four_spin_layout(6, 6),
        embedding_dim=8,
        head_count=2,
        depth=1,
    )
    key = jax.random.PRNGKey(33)
    small_params = init_sequential_cell_schur_vit(key, small)
    large_params = init_sequential_cell_schur_vit(key, large)
    assert _parameter_shapes(small_params) == _parameter_shapes(large_params)
    samples = np.asarray(
        sample_sequential_cell_schur_vit(
            large_params, jax.random.PRNGKey(34), 32, large
        )
    )
    assert samples.shape == (32, 35)
    assert np.all(samples[:, -1] == 0)
    labels_to_sequential_actions(large.layout, samples)


def test_q_conditioned_attention_is_normalized_and_size_transferable():
    small = SequentialCellSchurViTConfig(
        rectangular_sequential_four_spin_layout(4, 2),
        embedding_dim=8,
        head_count=2,
        depth=1,
        q_conditioned_attention=True,
    )
    large = SequentialCellSchurViTConfig(
        rectangular_sequential_four_spin_layout(6, 6),
        embedding_dim=8,
        head_count=2,
        depth=1,
        q_conditioned_attention=True,
    )
    key = jax.random.PRNGKey(35)
    small_params = init_sequential_cell_schur_vit(key, small)
    large_params = init_sequential_cell_schur_vit(key, large)
    assert _parameter_shapes(small_params) == _parameter_shapes(large_params)
    assert small_params["causal_layers"][0]["channel_pair_weight"].shape == (3, 2)
    assert small_params["global_layers"][0]["channel_pair_weight"].shape == (3, 2)

    states = np.asarray(
        [state.labels for state in small.layout.tree.enumerate_states(0)],
        dtype=np.int32,
    )
    logs = np.asarray(
        sequential_cell_schur_log_amplitudes(small_params, states, small)
    )
    np.testing.assert_allclose(np.sum(np.exp(2.0 * logs.real)), 1.0, atol=2.0e-13)

    samples = np.asarray(
        sample_sequential_cell_schur_vit(
            large_params, jax.random.PRNGKey(36), 32, large
        )
    )
    assert samples.shape == (32, 35)
    assert np.all(samples[:, -1] == 0)
    labels_to_sequential_actions(large.layout, samples)


def test_spatial_envelope_uses_size_independent_minimum_image_distances():
    n16 = SequentialCellSchurViTConfig(
        rectangular_sequential_four_spin_layout(4, 4),
        embedding_dim=8,
        head_count=2,
        depth=1,
        geometry="triangular",
    )
    n36 = SequentialCellSchurViTConfig(
        rectangular_sequential_four_spin_layout(6, 6),
        embedding_dim=8,
        head_count=2,
        depth=1,
        geometry="triangular",
    )
    distances16 = sequential_cell_schur_spatial_distances(n16)
    distances36 = sequential_cell_schur_spatial_distances(n36)
    assert distances16.shape == (4, 4)
    assert distances36.shape == (9, 9)
    np.testing.assert_allclose(np.diag(distances16), 0.0)
    np.testing.assert_allclose(np.diag(distances36), 0.0)
    assert np.min(distances16[distances16 > 0.0]) == 1.0
    assert np.min(distances36[distances36 > 0.0]) == 1.0
    # On the 3x3 triangular cell torus, the (0,0)->(1,1) separation is sqrt(3).
    np.testing.assert_allclose(distances36[0, 4], np.sqrt(3.0))


def test_spatial_decay_is_positive_and_does_not_change_exact_normalization():
    layout = rectangular_sequential_four_spin_layout(4, 2)
    enabled = SequentialCellSchurViTConfig(
        layout,
        embedding_dim=8,
        head_count=2,
        depth=1,
        initial_distance_decay=0.7,
        use_distance_envelope=True,
    )
    disabled = SequentialCellSchurViTConfig(
        layout,
        embedding_dim=8,
        head_count=2,
        depth=1,
        initial_distance_decay=0.7,
        use_distance_envelope=False,
    )
    params = init_sequential_cell_schur_vit(jax.random.PRNGKey(37), enabled)
    for stack in ("causal_layers", "global_layers"):
        decay = np.asarray(jax.nn.softplus(params[stack][0]["raw_distance_decay"]))
        np.testing.assert_allclose(decay, 0.7)
    states = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    logs_enabled = np.asarray(
        sequential_cell_schur_log_amplitudes(params, states, enabled)
    )
    logs_disabled = np.asarray(
        sequential_cell_schur_log_amplitudes(params, states, disabled)
    )
    np.testing.assert_allclose(
        np.sum(np.exp(2.0 * logs_enabled.real)), 1.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        np.sum(np.exp(2.0 * logs_disabled.real)), 1.0, atol=2.0e-13
    )
    assert np.max(np.abs(logs_enabled - logs_disabled)) > 1.0e-8


def test_n16_sequential_and_rotated_coefficient_transforms_roundtrip():
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    rng = np.random.default_rng(37)
    balanced = rng.normal(size=1430) + 1.0j * rng.normal(size=1430)
    balanced /= np.linalg.norm(balanced)
    sequential, _, forward = balanced_to_sequential_coefficients(balanced)
    rotated, inverse = sequential_to_rotated_coefficients(sequential)
    recovered, _ = rotated_to_balanced_tree_coefficients(rotated, layout)
    np.testing.assert_allclose(recovered, balanced, atol=2.0e-13)
    assert forward["coefficient_roundtrip_max_abs_error"] < 2.0e-13
    assert inverse["coefficient_roundtrip_max_abs_error"] < 2.0e-13
