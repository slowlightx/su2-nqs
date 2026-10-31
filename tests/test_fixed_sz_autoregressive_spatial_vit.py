import itertools

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.fixed_sz_autoregressive_spatial_vit import (
    FixedSzAutoregressiveSpatialViTConfig,
    fixed_sz_autoregressive_spatial_log_amplitude,
    fixed_sz_autoregressive_spatial_log_amplitudes,
    fixed_sz_autoregressive_spatial_parameter_count,
    init_fixed_sz_autoregressive_spatial_vit,
    sample_fixed_sz_autoregressive_spatial_vit,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_vit_parameter_count,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.spin_device import spin_device_hpsi


jax.config.update("jax_enable_x64", True)


def test_fixed_sz_shared_backbone_has_identical_parameter_shapes_and_count():
    fixed = FixedSzAutoregressiveSpatialViTConfig(
        "triangular", 4, 4, embedding_dim=16, head_count=2, depth=2
    )
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    su2 = ManyCellSchurViTConfig(
        layout, embedding_dim=16, head_count=2, depth=2, geometry="triangular"
    )
    key = jax.random.PRNGKey(61)
    fixed_params = init_fixed_sz_autoregressive_spatial_vit(key, fixed)
    su2_params = init_many_cell_schur_vit(key, su2)
    assert jax.tree_util.tree_structure(fixed_params) == jax.tree_util.tree_structure(
        su2_params
    )
    for fixed_leaf, su2_leaf in zip(
        jax.tree_util.tree_leaves(fixed_params),
        jax.tree_util.tree_leaves(su2_params),
        strict=True,
    ):
        assert fixed_leaf.shape == su2_leaf.shape
        np.testing.assert_array_equal(fixed_leaf, su2_leaf)
    assert fixed_sz_autoregressive_spatial_parameter_count(
        fixed_params
    ) == many_cell_schur_vit_parameter_count(su2_params)


def test_exhaustive_fixed_sz_wavefunction_is_normalized_and_invalid_states_masked():
    config = FixedSzAutoregressiveSpatialViTConfig(
        "square", 2, 2, embedding_dim=8, head_count=2, depth=1
    )
    params = init_fixed_sz_autoregressive_spatial_vit(
        jax.random.PRNGKey(62), config
    )
    sector = np.asarray(
        [bits for bits in itertools.product((0, 1), repeat=4) if sum(bits) == 2],
        dtype=np.int32,
    )
    logs = np.asarray(
        fixed_sz_autoregressive_spatial_log_amplitudes(params, sector, config)
    )
    assert abs(float(np.sum(np.exp(2.0 * logs.real))) - 1.0) < 2.0e-12
    invalid = fixed_sz_autoregressive_spatial_log_amplitude(
        params, jnp.asarray((0, 0, 0, 0)), config
    )
    assert np.isneginf(float(np.real(invalid)))

    draws = np.asarray(
        sample_fixed_sz_autoregressive_spatial_vit(
            params, jax.random.PRNGKey(6201), 20_000, config
        )
    )
    observed = {
        tuple(state): int(count)
        for state, count in zip(*np.unique(draws, axis=0, return_counts=True))
    }
    frequencies = np.asarray(
        [observed.get(tuple(state), 0) / len(draws) for state in sector]
    )
    probabilities = np.exp(2.0 * logs.real)
    assert float(np.max(np.abs(frequencies - probabilities))) < 0.02


def test_fixed_sz_samples_are_legal_and_device_h_action_is_finite():
    config = FixedSzAutoregressiveSpatialViTConfig(
        "triangular", 4, 4, embedding_dim=8, head_count=2, depth=1
    )
    params = init_fixed_sz_autoregressive_spatial_vit(
        jax.random.PRNGKey(63), config
    )
    samples = sample_fixed_sz_autoregressive_spatial_vit(
        params, jax.random.PRNGKey(64), 32, config
    )
    concrete = np.asarray(samples)
    assert np.all(np.sum(concrete, axis=1) == config.target_n_up)
    bonds = tuple((site, (site + 1) % 16, 1.0) for site in range(16))

    @jax.jit
    def action(candidate, states):
        return spin_device_hpsi(
            candidate,
            states,
            bonds,
            config,
            amplitude_fn=lambda values, row, model: jnp.exp(
                fixed_sz_autoregressive_spatial_log_amplitude(values, row, model)
            ),
        )

    centers, hpsi = action(params, samples[:4])
    assert np.all(np.isfinite(np.asarray(centers)))
    assert np.all(np.isfinite(np.asarray(hpsi)))


def test_symmetry_unaware_variant_normalizes_over_all_magnetizations():
    config = FixedSzAutoregressiveSpatialViTConfig(
        "square",
        2,
        2,
        target_m2=None,
        embedding_dim=8,
        head_count=2,
        depth=1,
    )
    params = init_fixed_sz_autoregressive_spatial_vit(
        jax.random.PRNGKey(65), config
    )
    support = np.asarray(
        list(itertools.product((0, 1), repeat=4)), dtype=np.int32
    )
    logs = np.asarray(
        fixed_sz_autoregressive_spatial_log_amplitudes(params, support, config)
    )
    assert np.all(np.isfinite(logs))
    assert abs(float(np.sum(np.exp(2.0 * logs.real))) - 1.0) < 2.0e-12
    draws = np.asarray(
        sample_fixed_sz_autoregressive_spatial_vit(
            params, jax.random.PRNGKey(66), 1024, config
        )
    )
    assert len(np.unique(np.sum(draws, axis=1))) > 1


def test_direct_phase_initializer_changes_only_configuration_phases():
    disabled = FixedSzAutoregressiveSpatialViTConfig(
        "square",
        2,
        2,
        embedding_dim=8,
        head_count=2,
        depth=1,
        direct_phase_initialization_scale=0.0,
    )
    enabled = FixedSzAutoregressiveSpatialViTConfig(
        "square",
        2,
        2,
        embedding_dim=8,
        head_count=2,
        depth=1,
        direct_phase_initialization_scale=1.0,
    )
    key = jax.random.PRNGKey(67)
    disabled_params = init_fixed_sz_autoregressive_spatial_vit(key, disabled)
    enabled_params = init_fixed_sz_autoregressive_spatial_vit(key, enabled)
    sector = np.asarray(
        [bits for bits in itertools.product((0, 1), repeat=4) if sum(bits) == 2],
        dtype=np.int32,
    )
    disabled_logs = np.asarray(
        fixed_sz_autoregressive_spatial_log_amplitudes(
            disabled_params, sector, disabled
        )
    )
    enabled_logs = np.asarray(
        fixed_sz_autoregressive_spatial_log_amplitudes(
            enabled_params, sector, enabled
        )
    )
    np.testing.assert_allclose(enabled_logs.real, disabled_logs.real, atol=1.0e-13)
    assert float(np.std(enabled_logs.imag - disabled_logs.imag)) > 1.0e-3


def test_marshall_checkerboard_is_a_deterministic_phase_only_carrier():
    plain = FixedSzAutoregressiveSpatialViTConfig(
        "square", 2, 2, embedding_dim=8, head_count=2, depth=1
    )
    marshall = FixedSzAutoregressiveSpatialViTConfig(
        "square",
        2,
        2,
        embedding_dim=8,
        head_count=2,
        depth=1,
        phase_convention="marshall_checkerboard",
    )
    params = init_fixed_sz_autoregressive_spatial_vit(
        jax.random.PRNGKey(68), plain
    )
    sector = np.asarray(
        [bits for bits in itertools.product((0, 1), repeat=4) if sum(bits) == 2],
        dtype=np.int32,
    )
    plain_logs = np.asarray(
        fixed_sz_autoregressive_spatial_log_amplitudes(params, sector, plain)
    )
    marshall_logs = np.asarray(
        fixed_sz_autoregressive_spatial_log_amplitudes(params, sector, marshall)
    )
    np.testing.assert_allclose(marshall_logs.real, plain_logs.real, atol=1.0e-13)
    checkerboard = np.asarray((1, 0, 0, 1), dtype=np.int32)
    expected = np.pi * (sector @ checkerboard)
    np.testing.assert_allclose(
        np.exp(1.0j * (marshall_logs.imag - plain_logs.imag)),
        np.exp(1.0j * expected),
        atol=2.0e-13,
    )
