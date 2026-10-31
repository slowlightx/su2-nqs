import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit, fixed_sz_spatial_vit_log_amplitudes
from su2_nqs.benchmarks.spatial_vit_sampling import projected_log_function, make_full_heisenberg_ratios, make_log_exchange_sampler
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case, _groups


@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_projected_amplitude_and_full_H_ratio_replay(geometry):
    config = FixedSzSpatialViTConfig(geometry=geometry, rows=2, columns=2,
        patch_rows=1, patch_columns=1, embedding_dim=4, head_count=2)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(2), config)
    data = _prepare_case(dict(geometry=geometry, rows=2, columns=2,
        j2_over_j1=.125, exact_energy_anchor=-10.), "fixed_m0")
    bits = jnp.asarray(data["basis"])
    log_fn, images = projected_log_function(config, data["groups"]["gamma_a1"])
    raw = np.exp(np.asarray(fixed_sz_spatial_vit_log_amplitudes(params, bits, config)))
    exact = np.mean(raw[data["tables"]["gamma_a1"]], axis=0)
    actual = np.exp(np.asarray(jax.vmap(log_fn, in_axes=(None,0))(params,bits)))
    np.testing.assert_allclose(actual, exact, atol=1e-12, rtol=1e-12)
    ratios = make_full_heisenberg_ratios(log_fn, data["bonds"])(params, bits)
    np.testing.assert_allclose(ratios, (data["hamiltonian"]@exact)/exact, atol=1e-12, rtol=1e-12)
    assert images < len(data["groups"]["gamma_a1"])


def test_log_exchange_preserves_sector_and_born_frequencies():
    bits = jnp.asarray([[0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0],[1,1,0,0]])
    field = jnp.asarray([.1,-.2,.3,-.1])
    log_fn = lambda p, x: jnp.dot(p, x).astype(jnp.complex128)
    sampler = make_log_exchange_sampler(log_fn, 40)
    states = initialize_spin_exchange_chains(jax.random.PRNGKey(0), n_chains=4096,n_sites=4,n_up=2)
    samples, _, acceptance, moves = sampler(field,jax.random.PRNGKey(1),states)
    assert np.all(np.asarray(samples).sum(axis=1)==2)
    observed = np.mean(np.all(np.asarray(samples)[:,None,:]==np.asarray(bits)[None,:,:],axis=-1),axis=0)
    expected = np.exp(2*np.asarray(bits@field)); expected /= expected.sum()
    np.testing.assert_allclose(observed, expected, atol=.025)
    assert 0 < moves < acceptance <= 1


@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_nontrivial_4x4_cosets_equal_complete_reynolds_sum(geometry):
    config=FixedSzSpatialViTConfig(geometry=geometry,rows=4,columns=4,
        embedding_dim=6,head_count=2,depth=2)
    params=init_fixed_sz_spatial_vit(jax.random.PRNGKey(17),config)
    group=_groups(geometry,4,4)["gamma_a1"]
    log_fn,images=projected_log_function(config,group)
    samples=initialize_spin_exchange_chains(jax.random.PRNGKey(19),
        n_chains=7,n_sites=16,n_up=8)
    def complete(x):
        logs=fixed_sz_spatial_vit_log_amplitudes(params,x[jnp.asarray(group)],config)
        return jnp.mean(jnp.exp(logs))
    expected=jax.jit(jax.vmap(complete))(samples)
    actual=jax.jit(jax.vmap(lambda x:jnp.exp(log_fn(params,x))))(samples)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    assert images==(32 if geometry=="square" else 48)
