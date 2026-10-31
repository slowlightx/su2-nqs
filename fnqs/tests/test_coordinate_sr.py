"""Independent physical amplitude/derivative equivalence and SR references."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest
from fnqs.basis import all_bits
from fnqs.hamiltonians import IsingCouplings,dense_hamiltonian
from fnqs.pilots.engine import init_model,Options,sampled_block as original_block,train as original_train
from fnqs.pilots.coordinate_sr import sampled_block,train
from fnqs.variants.size_features import SizeFeatureConfig,log_psi
from fnqs.variants.coupling_units import CouplingUnits,to_units,to_raw,physical_log_psi,encoded


@pytest.mark.parametrize('policy',('legacy','no_explicit_size','no_length'))
def test_affine_map_preserves_every_state_and_physical_coupling_derivative(policy):
    cfg=SizeFeatureConfig(3,size_features=policy);units=CouplingUnits(1.1,.04)
    p=init_model(jax.random.PRNGKey(21),cfg);q=to_units(p,units);back=to_raw(q,units)
    for name in p:np.testing.assert_allclose(back[name],p[name],atol=2e-15,rtol=1e-14)
    bits=jnp.asarray(all_bits(6))
    for values in ([1.,.1,.07,.96],[1.3,-.2,.03,1.4]):
        g=jnp.asarray(values)
        np.testing.assert_allclose(log_psi(p,bits,g,cfg),physical_log_psi(q,bits,g,cfg,units),atol=2e-13,rtol=1e-13)
        old=jax.jacfwd(lambda x:log_psi(p,bits,x,cfg))(g)
        new=jax.jacfwd(lambda x:physical_log_psi(q,bits,x,cfg,units))(g)
        np.testing.assert_allclose(new,old,atol=2e-12,rtol=1e-11)
    assert abs(float(jnp.sum(jnp.exp(2*physical_log_psi(q,bits,g,cfg,units))))-1)<1e-12


def test_affine_force_and_metric_against_dense_physical_pauli():
    cfg=SizeFeatureConfig(3,size_features='legacy');units=CouplingUnits(1.,.05)
    q=to_units(init_model(jax.random.PRNGKey(3),cfg),units);bits=jnp.asarray(all_bits(4));g=jnp.array([1.,.1,.07,1.03])
    flat,unravel=ravel_pytree(q)
    scores=np.asarray(jax.jacrev(lambda f:physical_log_psi(unravel(f),bits,g,cfg,units))(flat))
    psi=np.exp(np.asarray(physical_log_psi(q,bits,g,cfg,units)))
    energy=dense_hamiltonian(4,IsingCouplings(*g))@psi/psi
    a,force,e,_=sampled_block(q,bits,g,cfg,units,Options(samples=16))
    centered=scores-scores.mean(axis=0)
    np.testing.assert_allclose(a,centered/4,atol=2e-12)
    np.testing.assert_allclose(force,centered.T@(energy-energy.mean())/64,atol=2e-12)
    np.testing.assert_allclose(e,energy.mean()/4,atol=2e-12)


def test_identity_chart_reproduces_original_training_and_prng():
    cfg=SizeFeatureConfig(2,size_features='legacy');units=CouplingUnits(0.,1.)
    p=init_model(jax.random.PRNGKey(7),cfg);key=jax.random.PRNGKey(8)
    schedule=[[[1.,0.,0.,.9],[1.,0.,0.,1.1]]]*2;options=Options(samples=8)
    a,sa,ka,ha=original_train(p,key,4,schedule,cfg,options)
    b,sb,kb,hb=train(p,key,4,schedule,cfg,units,options)
    for name in a:np.testing.assert_allclose(b[name],a[name],atol=2e-12)
    np.testing.assert_array_equal(ka,kb);assert sa['work']==sb['work']
    for x,y in zip(ha,hb):
        np.testing.assert_allclose(x['energy_per_site'],y['energy_per_site'],atol=1e-13)
        np.testing.assert_allclose(x['true_sr_residual'],y['true_sr_residual'],atol=1e-12)


@pytest.mark.parametrize('center,scale',[(0,0),(0,-1),(float('nan'),1),(0,float('inf'))])
def test_invalid_units_rejected(center,scale):
    with pytest.raises(ValueError):CouplingUnits(center,scale)
