import jax
import jax.numpy as jnp
import numpy as np
import pytest
from su2_nqs.vmc.tempered_spin_exchange import make_tempered_spin_exchange,replica_swap_log_ratio
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.frontier.artifacts import pytree_sha256

jax.config.update('jax_enable_x64',True)


def test_swap_detailed_balance_against_explicit_joint_weights():
    w=np.array([-3.7,-.2,.6,1.3])
    for a,b in ((0.,.25),(.25,.75),(.75,1.)):
        for x in w:
            for y in w:
                forward=float(replica_swap_log_ratio(a,b,x,y))
                reverse=float(replica_swap_log_ratio(a,b,y,x))
                np.testing.assert_allclose(np.exp(a*x+b*y)*min(1.,np.exp(forward)),
                    np.exp(a*y+b*x)*min(1.,np.exp(reverse)),rtol=2e-14,atol=2e-14)


def test_replica_born_frequencies_sector_lineages_and_checkpoint(tmp_path):
    basis=jnp.asarray([[0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0],[1,1,0,0]])
    field=jnp.asarray([.7,-.5,.3,-.1]);betas=np.array([0.,.1,.25,.5,.75,1.])
    log_fn=lambda p,x:jnp.dot(p,x).astype(jnp.complex128)
    states=initialize_spin_exchange_chains(jax.random.PRNGKey(1),n_chains=6*2048,n_sites=4,n_up=2).reshape(6,2048,4)
    tags=jnp.broadcast_to(jnp.arange(6)[:,None],(6,2048))
    sampler=make_tempered_spin_exchange(log_fn,betas,proposals=128)
    x,key,t,d=sampler(field,jax.random.PRNGKey(2),states,tags)
    assert np.all(np.asarray(x).sum(axis=-1)==2)
    np.testing.assert_array_equal(np.sort(np.asarray(t),axis=0),tags)
    for r,b in enumerate(betas):
        observed=np.mean(np.all(np.asarray(x[r])[:,None,:]==np.asarray(basis)[None,:,:],axis=-1),axis=0)
        expected=np.exp(2*b*np.asarray(basis@field));expected/=expected.sum()
        np.testing.assert_allclose(observed,expected,atol=.035)
    assert np.all(np.asarray(d['swap_acceptance'])>0)
    save_checkpoint(tmp_path/'chain.pkl',dict(states=x,key=key,lineages=t))
    saved=load_checkpoint(tmp_path/'chain.pkl')['params']
    assert pytree_sha256(sampler(field,key,x,t))==pytree_sha256(sampler(field,saved['key'],saved['states'],saved['lineages']))


def test_invalid_tempering_ladder_and_budget_rejected():
    f=lambda p,x:jnp.asarray(0j)
    for b in ([1.,0.],[0.,.5,.5,1.],[0.,.5],[.1,1.],[0.,np.nan,1.]):
        with pytest.raises(ValueError):make_tempered_spin_exchange(f,b,proposals=16)
    with pytest.raises(ValueError):make_tempered_spin_exchange(f,[0.,1.],proposals=15)
