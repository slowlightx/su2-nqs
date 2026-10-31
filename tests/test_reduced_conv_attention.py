"""Fixed local Schur/CG decoding checked before any energy optimization."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.four_cell_schur import four_cell_schur_state_vector
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.schur_tensor_decoder import four_cell_tensor_decoder_rows
from su2_nqs.models.reduced_conv_attention import (
    ReducedConvAttentionConfig, init_reduced_conv_attention, reduced_conv_attention_coefficients,
)
from su2_nqs.su2.local_schur_fusion import four_cell_schur_states

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def decoder():
    kind = 'triangular_triangle'; states = four_cell_schur_states(kind)
    c = np.random.default_rng(201).normal(size=len(states))
    v, basis = four_cell_schur_state_vector(c, kind)
    d = four_cell_tensor_decoder_rows(np.asarray(basis), kind, ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)))
    labels = np.stack((np.asarray([s.cell_q for s in states]), np.asarray([s.cell_a for s in states])), axis=-1)
    return d, np.asarray(basis), labels, np.asarray([s.pair_q for s in states]), c, v


def test_tensor_decoder_complete_orthonormal_and_independent_recoupling(decoder):
    d, basis, _, _, c, v = decoder
    assert d.shape == (924, 132)
    np.testing.assert_allclose(d.T@d, np.eye(132), atol=4e-14, rtol=0)
    np.testing.assert_allclose(d@c, v, atol=2e-14, rtol=0)
    assert spin_raising_relative_norm(d@c, basis) < 1e-13


def test_full_configuration_network_and_all_parameter_tangents_are_singlets(decoder):
    d, basis, labels, inter, _, _ = decoder
    config = ReducedConvAttentionConfig(maximum_q=6, multiplicity_count=2, embedding_dim=8, depth=1)
    params = init_reduced_conv_attention(jax.random.PRNGKey(301), config)
    fn = lambda p: reduced_conv_attention_coefficients(p, jnp.asarray(labels), jnp.asarray(inter), config)
    direction = jax.tree.map(jnp.ones_like, params)
    c, tangent = jax.jvp(fn, (params,), (direction,))
    assert np.linalg.norm(np.asarray(tangent)) > 1e-6
    assert spin_raising_relative_norm(d@np.asarray(c), basis) < 1e-13
    assert spin_raising_relative_norm(d@np.asarray(tangent), basis) < 1e-13
    epsilon = 1e-6
    plus, minus = (jax.tree.map(lambda p, v: p+sign*epsilon*v, params, direction) for sign in (1, -1))
    np.testing.assert_allclose((fn(plus)-fn(minus))/(2*epsilon), tangent, atol=2e-7, rtol=2e-6)
    # Nonlinearity sees different COMPLETE reduced configurations, not only
    # geometry. It produces complex, nonconstant coefficients.
    assert np.std(np.asarray(c)) > 1e-4
    probabilities = abs(np.asarray(c)/np.linalg.norm(np.asarray(c)))**2
    assert abs(probabilities.sum()-1) < 1e-14


def test_invalid_reduced_labels_and_configuration_contract(decoder):
    _, _, labels, inter, _, _ = decoder
    cfg = ReducedConvAttentionConfig(maximum_q=6, multiplicity_count=2, embedding_dim=4, depth=1)
    p = init_reduced_conv_attention(jax.random.PRNGKey(2), cfg)
    bad = labels.copy(); bad[0, 0, 0] = -1
    assert not np.isfinite(np.asarray(reduced_conv_attention_coefficients(p, bad, inter, cfg))[0])
    with pytest.raises(TypeError):
        reduced_conv_attention_coefficients(p, labels.astype(float), inter, cfg)
    with pytest.raises(ValueError):
        ReducedConvAttentionConfig(maximum_q=1.5)
