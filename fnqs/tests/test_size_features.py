import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fnqs.basis import all_bits
from fnqs.diagnostics.response import dense_response
from fnqs.hamiltonians import IsingCouplings
from fnqs import models
from fnqs.variants import size_features as variant


@pytest.mark.parametrize("policy", ["legacy", "no_explicit_size", "no_length"])
def test_normalization_z2_and_sampling_frequency(policy):
    cfg = variant.SizeFeatureConfig(4, size_features=policy)
    params = variant.init_model(jax.random.PRNGKey(51), cfg)
    g = jnp.array([1., .13, .05, 1.02])
    for n in (4, 6, 8):
        logs = np.asarray(variant.log_psi(params, all_bits(n), g, cfg))
        np.testing.assert_allclose(np.sum(np.exp(2*logs)), 1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(logs, variant.log_psi(params, 1-all_bits(n), g, cfg), atol=1e-14, rtol=0)
    draws = np.asarray(variant.sample(params, jax.random.PRNGKey(29), 24000, 4, g, cfg))
    p = np.exp(2*np.asarray(variant.log_psi(params, all_bits(4), g, cfg)))
    counts = np.bincount(draws @ (1 << np.arange(3, -1, -1)), minlength=16)
    assert np.max(abs(counts/len(draws)-p)/np.sqrt(p*(1-p)/len(draws))) < 5


def test_legacy_identity_including_coupling_and_parameter_gradients():
    cfg = variant.SizeFeatureConfig(4, size_features="legacy")
    old = models.GRUConfig(4)
    params = variant.init_model(jax.random.PRNGKey(51), cfg)
    g = jnp.array([1., .13, .05, 1.02])
    for n in (4, 6, 8):
        bits = all_bits(n)
        np.testing.assert_array_equal(variant.log_psi(params, bits, g, cfg), models.log_psi(params, bits, g, old))
        np.testing.assert_array_equal(variant.sample(params, jax.random.PRNGKey(2), 111, n, g, cfg),
                                      models.sample(params, jax.random.PRNGKey(2), 111, n, g, old))
    np.testing.assert_array_equal(jax.jacfwd(lambda c: variant.log_psi(params, bits, c, cfg))(g),
                                  models.coupling_scores(params, bits, g, old))
    a = jax.grad(lambda p: variant.log_psi(p, bits[:1], g, cfg).sum())(params)
    b = jax.grad(lambda p: models.log_psi(p, bits[:1], g, old).sum())(params)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_no_length_prefix_contract_and_physical_pbc_counterexample():
    cfg = variant.SizeFeatureConfig(4, size_features="no_length")
    params = variant.init_model(jax.random.PRNGKey(51), cfg)
    g = jnp.array([1., 0., 0., 1.])
    p4 = np.exp(2*np.asarray(variant.log_psi(params, all_bits(4), g, cfg)))
    p6 = np.exp(2*np.asarray(variant.log_psi(params, all_bits(6), g, cfg))).reshape(16, 4).sum(axis=1)
    np.testing.assert_allclose(p4, p6, atol=1e-14, rtol=0)
    for n in (6, 8):
        np.testing.assert_array_equal(variant.sample(params, jax.random.PRNGKey(2), 111, 4, g, cfg),
                                      variant.sample(params, jax.random.PRNGKey(2), 111, n, g, cfg)[:, :4])
    exact4 = dense_response(4, IsingCouplings()).state**2
    exact6 = (dense_response(6, IsingCouplings()).state**2).reshape(16, 4).sum(axis=1)
    # The obstruction is any nonzero gap; its independently computed value
    # is 0.01581137, much larger than the exact-reference tolerance.
    np.testing.assert_allclose(np.max(abs(exact4-exact6)), .015811370932323, atol=2e-12, rtol=0)
    # Removing explicit L scalars still leaves length-normalized geometry.
    cfg2 = variant.SizeFeatureConfig(4, size_features="no_explicit_size")
    q4 = np.exp(2*np.asarray(variant.log_psi(params, all_bits(4), g, cfg2)))
    q6 = np.exp(2*np.asarray(variant.log_psi(params, all_bits(6), g, cfg2))).reshape(16, 4).sum(axis=1)
    assert np.max(abs(q4-q6)) > 1e-7


@pytest.mark.parametrize("policy,inactive", [("legacy", 0), ("no_explicit_size", 128), ("no_length", 176)])
def test_parameter_accounting(policy, inactive):
    counts = variant.parameter_counts(variant.SizeFeatureConfig(16, size_features=policy))
    assert counts == {"nominal": 1474, "active": 1474-inactive, "inactive": inactive}


def test_variant_config_boundaries_and_legacy_trainer_incompatibility():
    for kw in ({"hidden_dim": True}, {"geometry_scale": 1.}, {"geometry_scale": np.nan}, {"size_features": "masked"}):
        with pytest.raises(ValueError):
            variant.SizeFeatureConfig(**kw)
    cfg = variant.SizeFeatureConfig(3)
    params = variant.init_model(jax.random.PRNGKey(1), cfg)
    assert np.isneginf(variant.log_psi(params, [0, 1, .5, 0], np.ones(4), cfg))
    with pytest.raises(TypeError):
        variant.log_psi(params, [0, 1, 0, 1], np.ones(4), models.GRUConfig(3))
    # Avoid silent policy omission: this config deliberately lacks size_scale.
    with pytest.raises(AttributeError):
        models.log_psi(params, [0, 1, 0, 1], np.ones(4), cfg)
