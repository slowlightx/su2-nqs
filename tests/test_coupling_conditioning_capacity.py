"""Small exact algebra and existing-model checks, with no H or optimization."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.coupling_conditioning_capacity import (
    affine_bias_rank_bound, conditioning_tangent_report, ray_projected_jacobian,
)
from su2_nqs.models.coupling_conditioned_spatial_vit import (
    wrap_spatial_vit_parameters, effective_parameters,
    coupling_conditioned_spatial_vit_log_amplitudes,
)
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit,
    fixed_sz_spatial_vit_log_amplitudes,
)


def _model():
    cfg = FixedSzSpatialViTConfig("square", 2, 4, embedding_dim=4, head_count=2, depth=2)
    base = init_fixed_sz_spatial_vit(jax.random.PRNGKey(1704), cfg)
    params = wrap_spatial_vit_parameters(base)
    params["coupling_bias"] = jnp.array([.1, -.3, .2, .4])
    numbers = np.asarray([n for n in range(256) if n.bit_count() == 4])
    bits = jnp.asarray((numbers[:, None] >> np.arange(7, -1, -1)) & 1)
    return cfg, params, bits


def test_affine_input_nullities_and_nearly_duplicate_couplings():
    one = affine_bias_rank_bound(16, [.5])
    assert one["gauge_free_bias_input_rank_upper_bound"] == 15
    assert one["bias_and_slope_nullity_lower_bound"] == 17
    assert one["slope_derivative_identically_zero_at_center"]
    two = affine_bias_rank_bound(16, [.2, .8])
    assert two["gauge_free_bias_input_rank_upper_bound"] == 30
    assert two["bias_and_slope_nullity_lower_bound"] == 2
    assert two["new_slope_physical_direction_upper_bound"] == 15
    near = affine_bias_rank_bound(16, [.5, .5+1e-12])
    assert near["affine_design_rank"] == 2
    assert near["design_singular_values"][-1] < 1e-11


def test_uniform_bias_and_slope_gauges_are_exact_in_existing_n8_model():
    cfg, params, bits = _model()
    shifted = {**params, "backbone":{**params["backbone"],
        "patch_embedding_bias":params["backbone"]["patch_embedding_bias"] + .321},
        "coupling_bias":params["coupling_bias"] + .789}
    largest = 0.
    for j in (.2, .5, .8):
        actual = coupling_conditioned_spatial_vit_log_amplitudes(shifted, bits, j, cfg)
        expected = coupling_conditioned_spatial_vit_log_amplitudes(params, bits, j, cfg)
        error = float(jnp.max(abs(actual-expected)));largest=max(largest,error)
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=0)
    print("uniform LayerNorm coupling gauge log-amplitude error", largest)


def test_chain_rule_and_nonlinear_curvature_not_a_linear_state_family():
    cfg, params, bits = _model(); coupling = .35
    eff = effective_parameters(params, coupling)
    def amplitude_bias(bias):
        return jnp.exp(fixed_sz_spatial_vit_log_amplitudes(
            {**eff,"patch_embedding_bias":bias}, bits, cfg))
    def amplitude_j(j):
        return jnp.exp(coupling_conditioned_spatial_vit_log_amplitudes(params, bits, j, cfg))
    state = amplitude_j(coupling)
    matrix = jax.jacfwd(amplitude_bias)(eff["patch_embedding_bias"])
    derivative = jax.jacfwd(amplitude_j)(coupling)
    report = conditioning_tangent_report(np.asarray(state), np.asarray(matrix),
        np.asarray(params["coupling_bias"]), actual_coupling_derivative=np.asarray(derivative))
    assert report["coupling_chain_rule_absolute_error"] < 1e-13
    assert report["uniform_bias_tangent_norm"] < 1e-12
    assert report["all_patch_bias_tangents"]["real_tangent_rank"] <= 3
    second_j = jax.jacfwd(jax.jacfwd(amplitude_j))(coupling)
    second_bias = jax.jacfwd(jax.jacfwd(amplitude_bias))(eff["patch_embedding_bias"])
    expected = jnp.einsum("nab,a,b->n", second_bias, params["coupling_bias"], params["coupling_bias"])
    np.testing.assert_allclose(second_j, expected, atol=3e-13, rtol=1e-10)
    assert np.linalg.norm(ray_projected_jacobian(state, np.asarray(second_j)[:,None])) > 1e-7
    print("bias-to-J chain rule error", report["coupling_chain_rule_absolute_error"],
          "bias tangent rank", report["all_patch_bias_tangents"]["real_tangent_rank"])


def test_real_tangent_capture_distinguishes_learned_bias_and_full_backbone():
    state = np.array([1.,0.,0.], dtype=complex)
    # Bias provides only REAL e1, full body also supplies imaginary e1 and e2.
    bias = np.array([[0.,0.],[1.,-1.],[0.,0.]], dtype=complex)
    body = np.column_stack((bias, np.array([0.,1j,0.]), np.array([0.,0.,1.])))
    target = np.array([0.,1j,1.])
    report = conditioning_tangent_report(state,bias,np.array([1.,0.]),
        backbone_jacobian=body,target_derivative=target)
    assert report["all_patch_bias_tangents"]["real_tangent_rank"] == 1
    assert report["all_patch_bias_tangents"]["captured_squared_fraction"] == 0
    assert report["full_backbone_tangents"]["real_tangent_rank"] == 3
    assert report["full_backbone_tangents"]["captured_squared_fraction"] == pytest.approx(1.)
    assert report["full_backbone_tangents"]["bias_tangent_outside_backbone_fraction"] < 1e-15
    assert not report["global_backbone_failure_certified"]
    # The phase and normalization components of supplied derivatives disappear.
    scaled = conditioning_tangent_report(3j*state,3j*bias+(2-4j)*state[:,None],
        np.array([1.,0.]),backbone_jacobian=3j*body,
        target_derivative=3j*target+7j*state)
    assert scaled["all_patch_bias_tangents"]["captured_squared_fraction"] == 0
    assert scaled["full_backbone_tangents"]["captured_squared_fraction"] == pytest.approx(1.)


def test_zero_directions_and_invalid_inputs_are_explicit():
    state=np.array([1.,0.]);jac=np.zeros((2,3),dtype=complex)
    report=conditioning_tangent_report(state,jac,np.zeros(3),target_derivative=np.zeros(2))
    assert report["learned_coupling_tangent"]["real_tangent_rank"] == 0
    assert report["learned_coupling_tangent"]["captured_squared_fraction"] is None
    numerical_noise=conditioning_tangent_report(state,jac+1e-15,np.ones(3))
    assert numerical_noise["learned_coupling_tangent"]["real_tangent_rank"] == 0
    with pytest.raises(ValueError):affine_bias_rank_bound(1,[.5])
    with pytest.raises(ValueError):affine_bias_rank_bound(16,[np.nan])
    with pytest.raises(ValueError):ray_projected_jacobian(np.zeros(2),jac)
    with pytest.raises(ValueError):conditioning_tangent_report(state,jac,np.zeros(2))
    with pytest.raises(ValueError):conditioning_tangent_report(state,jac,np.ones(3)*1j)
    with pytest.raises(ValueError):conditioning_tangent_report(state,jac,np.zeros(3),target_derivative=np.zeros(3))
    with pytest.raises(ValueError):conditioning_tangent_report(state,jac,np.zeros(3),absolute_rank_tolerance=-1.)
