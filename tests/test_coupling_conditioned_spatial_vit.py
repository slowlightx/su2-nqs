"""No-H small tests of the minimal shared-coupling input wrapper."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig, fixed_sz_spatial_vit_amplitudes,
    fixed_sz_spatial_vit_log_amplitudes, fixed_sz_spatial_vit_parameter_count,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.models.coupling_conditioned_spatial_vit import (
    coupling_conditioned_spatial_vit_amplitudes,
    coupling_conditioned_spatial_vit_log_amplitudes,
    coupling_conditioned_spatial_vit_parameter_count,
    effective_parameters, init_coupling_conditioned_spatial_vit,
    wrap_spatial_vit_parameters,
)


def setup(geometry="square", **updates):
    kwargs = dict(geometry=geometry, rows=4, columns=4, embedding_dim=8,
                  head_count=2, depth=1, mlp_ratio=2)
    kwargs.update(updates)
    config = FixedSzSpatialViTConfig(**kwargs)
    old = init_fixed_sz_spatial_vit(jax.random.PRNGKey(1701), config)
    params = wrap_spatial_vit_parameters(old)
    rng = np.random.default_rng(1702)
    bits = np.zeros((4, 16), dtype=np.int32)
    for row in bits:
        row[rng.choice(16, 8, replace=False)] = 1
    return config, old, params, jnp.asarray(bits)


def test_zero_input_replays_saved_backbone_exactly_without_mutating_it():
    config, old, params, bits = setup()
    previous = np.asarray(old["patch_embedding_bias"]).copy()
    for coupling in (-.1, .5, 1.2):
        np.testing.assert_array_equal(
            coupling_conditioned_spatial_vit_log_amplitudes(params, bits, coupling, config),
            fixed_sz_spatial_vit_log_amplitudes(old, bits, config))
        np.testing.assert_array_equal(
            coupling_conditioned_spatial_vit_amplitudes(params, bits, coupling, config),
            fixed_sz_spatial_vit_amplitudes(old, bits, config))
    changed = {**params, "coupling_bias": jnp.arange(8, dtype=jnp.float64)}
    effective = effective_parameters(changed, .75)
    np.testing.assert_array_equal(effective["patch_embedding_bias"], previous + .25*np.arange(8))
    np.testing.assert_array_equal(old["patch_embedding_bias"], previous)
    assert coupling_conditioned_spatial_vit_parameter_count(params) == fixed_sz_spatial_vit_parameter_count(old) + 8


def test_registered_width16_depth2_parameter_count_and_initializer_replay():
    config = FixedSzSpatialViTConfig("square", 4, 4, embedding_dim=16, head_count=4, depth=2)
    key = jax.random.PRNGKey(1703)
    old = init_fixed_sz_spatial_vit(key, config)
    params = init_coupling_conditioned_spatial_vit(key, config)
    assert fixed_sz_spatial_vit_parameter_count(old) == 3992
    assert coupling_conditioned_spatial_vit_parameter_count(params) == 4008
    for original, copied in zip(jax.tree.leaves(old), jax.tree.leaves(params["backbone"])):
        np.testing.assert_array_equal(original, copied)
    np.testing.assert_array_equal(params["coupling_bias"], np.zeros(16))


def test_nonzero_coupling_changes_physical_ray_and_full_gradient_matches_difference():
    config, _, params, bits = setup()
    params = {**params, "coupling_bias": jnp.asarray(np.linspace(-.3, .4, 8))}
    low = np.asarray(coupling_conditioned_spatial_vit_log_amplitudes(params, bits, .2, config))
    high = np.asarray(coupling_conditioned_spatial_vit_log_amplitudes(params, bits, .8, config))
    change = high-low
    assert np.linalg.norm(change-change.mean()) > 1e-8
    weights = jnp.asarray([1., -2., .3, .7])
    objective = lambda candidate: jnp.real(jnp.vdot(weights,
        coupling_conditioned_spatial_vit_log_amplitudes(candidate, bits, .2, config)))
    grad = jax.grad(objective)(params)
    assert np.linalg.norm(grad["coupling_bias"]) > 1e-8
    assert np.linalg.norm(grad["backbone"]["patch_embedding"]) > 1e-8
    direction = jax.tree.map(lambda x: jnp.cos(jnp.arange(x.size).reshape(x.shape)) / 30, params)
    derivative = sum(jnp.sum(a*b) for a,b in zip(jax.tree.leaves(grad), jax.tree.leaves(direction)))
    eps = 1e-5
    plus = jax.tree.map(lambda x,d:x+eps*d, params, direction)
    minus = jax.tree.map(lambda x,d:x-eps*d, params, direction)
    finite = (objective(plus)-objective(minus))/(2*eps)
    np.testing.assert_allclose(derivative, finite, atol=1e-9, rtol=1e-5)
    j_objective = lambda coupling: jnp.real(jnp.vdot(weights,
        coupling_conditioned_spatial_vit_log_amplitudes(params, bits, coupling, config)))
    j_derivative = jax.grad(j_objective)(.2)
    j_finite = (j_objective(.2+eps)-j_objective(.2-eps))/(2*eps)
    np.testing.assert_allclose(j_derivative, j_finite, atol=1e-9, rtol=1e-5)
    print("coupling wrapper gradient max discrepancy", float(max(abs(derivative-finite), abs(j_derivative-j_finite))))


def test_zero_initial_coupling_vector_has_live_tangent_away_from_center():
    config, _, params, bits = setup()
    weights = jnp.asarray([1., -2., .3, .7])
    def objective(candidate, coupling):
        return jnp.real(jnp.vdot(weights,
            coupling_conditioned_spatial_vit_log_amplitudes(candidate, bits, coupling, config)))
    away = jax.grad(objective)(params, .3)
    centered = jax.grad(objective)(params, .5)
    assert np.linalg.norm(away["coupling_bias"]) > 1e-8
    np.testing.assert_array_equal(centered["coupling_bias"], np.zeros(8))


@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_shared_bias_preserves_coarse_patch_translations(geometry):
    config, _, params, bits = setup(geometry)
    params = {**params, "coupling_bias": jnp.linspace(-.3, .4, 8)}
    expected = coupling_conditioned_spatial_vit_log_amplitudes(params, bits, .3, config)
    for axis in (1, 2):
        shifted = np.roll(np.asarray(bits).reshape(4, 4, 4), 2, axis=axis).reshape(4, 16)
        actual = coupling_conditioned_spatial_vit_log_amplitudes(params, shifted, .3, config)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("coupling", [np.nan, np.inf, 1j, True, [0.2]])
def test_invalid_coupling_rejected(coupling):
    _, _, params, _ = setup()
    with pytest.raises(ValueError, match="finite real scalar"):
        effective_parameters(params, coupling)


def test_invalid_structure_shape_finite_and_dynamic_inputs_fail_closed():
    _, _, params, _ = setup()
    with pytest.raises(ValueError, match="exactly"):
        effective_parameters({**params, "extra": 1}, .5)
    for invalid in (jnp.zeros(7), jnp.ones(8)*1j, jnp.full(8, jnp.nan)):
        with pytest.raises(ValueError, match="coupling_bias"):
            effective_parameters({**params, "coupling_bias": invalid}, .5)
    with pytest.raises(ValueError, match="center"):
        effective_parameters(params, .5, center=np.inf)
    with pytest.raises(ValueError, match="backbone"):
        wrap_spatial_vit_parameters({})
    evaluate = jax.jit(lambda j: effective_parameters(params, j)["patch_embedding_bias"])
    assert np.isnan(np.asarray(evaluate(jnp.nan))).all()
    assert np.isnan(np.asarray(evaluate(jnp.inf))).all()
