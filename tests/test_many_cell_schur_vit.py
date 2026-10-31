import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    additive_log_amplitudes,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    many_cell_schur_spatial_distances,
    sample_many_cell_schur_vit,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.benchmarks.many_cell_replay import (
    balanced_tree_to_rotated_coefficients,
    rotated_to_balanced_tree_coefficients,
)


def _shapes(tree):
    return tuple(tuple(leaf.shape) for leaf in jax.tree_util.tree_leaves(tree))


def test_many_cell_layout_exposes_complete_local_and_intercell_labels():
    layout = rectangular_four_spin_layout(4, 4)
    states = layout.tree.enumerate_states(0)
    assert len(states) == 1430
    decoded = layout.decode(np.asarray([state.labels for state in states]))
    assert decoded.cell_q.shape == (1430, 4)
    assert decoded.cell_a.shape == (1430, 4)
    assert decoded.intercell_q.shape == (1430, 3)
    assert set(np.unique(decoded.cell_q)) == {0, 2, 4}
    assert np.all(decoded.intercell_q[:, -1] == 0)
    assert layout.physical_cells == (
        (0, 1, 4, 5),
        (2, 3, 6, 7),
        (8, 9, 12, 13),
        (10, 11, 14, 15),
    )


def test_many_cell_vit_is_exactly_normalized_and_samples_only_valid_states():
    layout = rectangular_four_spin_layout(2, 2)
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(7), config)
    states = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    logs = np.asarray(many_cell_schur_log_amplitudes(params, states, config))
    np.testing.assert_allclose(np.sum(np.exp(2.0 * logs.real)), 1.0, atol=2.0e-13)
    sampled = np.asarray(
        sample_many_cell_schur_vit(params, jax.random.PRNGKey(8), 128, config)
    )
    support = {tuple(row) for row in states}
    assert all(tuple(row) in support for row in sampled)


def test_many_cell_vit_parameter_shapes_transfer_from_n16_to_n36():
    small = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(4, 4),
        embedding_dim=8,
        head_count=2,
        depth=1,
    )
    large = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(6, 6),
        embedding_dim=8,
        head_count=2,
        depth=1,
    )
    key = jax.random.PRNGKey(9)
    assert _shapes(init_many_cell_schur_vit(key, small)) == _shapes(
        init_many_cell_schur_vit(key, large)
    )


def test_many_cell_spatial_envelope_is_periodic_and_size_independent():
    small = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(4, 4),
        embedding_dim=8,
        head_count=2,
        depth=1,
        geometry="triangular",
    )
    large = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(6, 6),
        embedding_dim=8,
        head_count=2,
        depth=1,
        geometry="triangular",
    )
    for config in (small, large):
        distances = many_cell_schur_spatial_distances(config)
        assert distances.shape == (config.n - 1, config.n - 1)
        np.testing.assert_allclose(distances, distances.T)
        np.testing.assert_allclose(np.diag(distances), 0.0)
        assert np.min(distances[distances > 0.0]) == 1.0


def test_many_cell_distance_decay_is_positive_and_preserves_normalization():
    layout = rectangular_four_spin_layout(2, 2)
    enabled = ManyCellSchurViTConfig(
        layout,
        embedding_dim=8,
        head_count=2,
        depth=1,
        initial_distance_decay=0.6,
        use_distance_envelope=True,
    )
    disabled = ManyCellSchurViTConfig(
        layout,
        embedding_dim=8,
        head_count=2,
        depth=1,
        initial_distance_decay=0.6,
        use_distance_envelope=False,
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(10), enabled)
    for stack in ("layers", "global_layers"):
        decay = np.asarray(jax.nn.softplus(params[stack][0]["raw_distance_decay"]))
        np.testing.assert_allclose(decay, 0.6)
    states = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    logs_enabled = np.asarray(many_cell_schur_log_amplitudes(params, states, enabled))
    logs_disabled = np.asarray(many_cell_schur_log_amplitudes(params, states, disabled))
    np.testing.assert_allclose(
        np.sum(np.exp(2.0 * logs_enabled.real)), 1.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        np.sum(np.exp(2.0 * logs_disabled.real)), 1.0, atol=2.0e-13
    )


def test_direct_phase_initializer_preserves_su2_born_probabilities():
    layout = rectangular_four_spin_layout(4, 4)
    disabled = ManyCellSchurViTConfig(
        layout,
        embedding_dim=8,
        head_count=2,
        depth=1,
        direct_phase_initialization_scale=0.0,
    )
    enabled = ManyCellSchurViTConfig(
        layout,
        embedding_dim=8,
        head_count=2,
        depth=1,
        direct_phase_initialization_scale=1.0,
    )
    key = jax.random.PRNGKey(11)
    disabled_params = init_many_cell_schur_vit(key, disabled)
    enabled_params = init_many_cell_schur_vit(key, enabled)
    states = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    disabled_logs = np.asarray(
        many_cell_schur_log_amplitudes(disabled_params, states, disabled)
    )
    enabled_logs = np.asarray(
        many_cell_schur_log_amplitudes(enabled_params, states, enabled)
    )
    np.testing.assert_allclose(enabled_logs.real, disabled_logs.real, atol=1.0e-13)
    assert float(np.std(enabled_logs.imag - disabled_logs.imag)) > 1.0e-3


def test_additive_channel_recovers_carrier_and_can_fill_a_node():
    carrier = jnp.asarray([0.0 + 0.0j, -jnp.inf + 0.0j])
    residual = jnp.asarray([-0.4 + 0.2j, -0.7 - 0.3j])
    recovered = np.asarray(additive_log_amplitudes(carrier, residual, 0.0))
    np.testing.assert_allclose(recovered[0], carrier[0], atol=1.0e-14)
    filled = np.exp(np.asarray(additive_log_amplitudes(carrier, residual, 0.2))[1])
    assert abs(filled) > 0.0


def test_local_d4_and_balanced_cell_coefficient_maps_are_inverse():
    layout = rectangular_four_spin_layout(4, 4)
    rng = np.random.default_rng(19)
    rotated = rng.normal(size=1430) + 1.0j * rng.normal(size=1430)
    balanced, labels = rotated_to_balanced_tree_coefficients(rotated, layout)
    recovered = balanced_tree_to_rotated_coefficients(balanced, layout)
    assert labels.shape == (1430, 15)
    np.testing.assert_allclose(recovered, rotated, atol=3.0e-13)
