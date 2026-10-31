"""Independent H replay and bounded frozen-state/seed-replication protocol."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from projects.clebsch_tree_nqs.workflows.spin_full_vit_recheck import (
    CONFIG,PRIOR_CONFIG,explicit_full_H,frozen_job_summary,square_start_pool,train_fork,
)
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from su2_nqs.frontier.artifacts import pytree_sha256


def test_explicit_H_replay_matches_independent_exact_matrix():
    case=_prepare_case(dict(geometry='square',rows=2,columns=2,j2_over_j1=.5,
        exact_energy_anchor=-10.),'fixed_m0')
    x=np.asarray(case['basis']);p=jnp.asarray([.1,.2,-.15,.3])
    fn=lambda p,x: (1+.2j)*jnp.dot(p,x)
    psi=np.exp(np.asarray(jax.vmap(fn,in_axes=(None,0))(p,x)))
    got=explicit_full_H(fn,p,x,case['bonds'])
    batch=np.asarray(make_full_heisenberg_ratios(fn,case['bonds'])(p,x))
    np.testing.assert_allclose(got,batch,atol=2e-13,rtol=2e-13)
    # Assemble the matrix independently in the physical configuration basis.
    H=np.zeros((len(x),len(x)));lookup={tuple(y):i for i,y in enumerate(x)}
    for col,y in enumerate(x):
        for i,j,c in case['bonds']:
            H[col,col]+=c*(y[i]-.5)*(y[j]-.5)
            if y[i]!=y[j]:
                z=y.copy();z[i],z[j]=y[j],y[i];H[lookup[tuple(z)],col]+=.5*c
    np.testing.assert_allclose(got,(H@psi)/psi,atol=2e-13,rtol=2e-13)


def test_summary_preserves_chain_axis_and_biased_cohorts():
    rng=np.random.default_rng(31);e=rng.normal(size=(25,24))+np.arange(24)[None,:]
    summary=frozen_job_summary(e,8,8)
    paths=np.vstack((e[-8:-4].T,e[-4:].T))
    W=np.mean(np.var(paths,axis=1,ddof=1));B=4*np.var(np.mean(paths,axis=1),ddof=1)
    assert summary['descriptive_split_rhat']==pytest.approx(np.sqrt((.75*W+.25*B)/W))
    assert summary['equilibrium_certified'] is False
    assert summary['energy_start_cohort_range']>.3
    for i,c in enumerate(summary['cohorts']):
        assert c['energy_per_site']==pytest.approx(e[-8:,8*i:8*(i+1)].mean()/36)


def test_registered_starts_reproducible_same_across_kernels_and_budget():
    spec=json.loads(CONFIG.read_text());prior=json.loads(PRIOR_CONFIG.read_text())
    cfg=spec['square'];cold,hot,key=square_start_pool(spec,cfg['seeds'][0])
    other=square_start_pool(spec,cfg['seeds'][0])
    for a,b in zip((cold,hot,key),other):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(cold,hot[-1]);assert np.all(hot.sum(axis=-1)==18)
    second=square_start_pool(spec,cfg['seeds'][1])[0]
    np.testing.assert_array_equal(cold[8:],second[8:]);assert not np.array_equal(cold[:8],second[:8])
    assert cfg['ordinary_proposals_per_block']==len(cfg['betas'])*cfg['tempered_proposals_per_block']
    assert 4*128*64+4*3*16*32+2*64+8*25*24+60==43900
    new=spec['triangular']['training_seeds']+cfg['seeds']
    new += [s for pool in spec['triangular']['audit']['seeds'].values() for values in pool.values() for s in values]
    old=list(prior['training_seeds'].values())+[s for pool in prior['audit']['seeds'].values() for v in pool.values() for s in v]
    assert len(new)==len(set(new));assert not set(new)&set(old)


def test_training_fork_checkpoint_resume_matches_continuous(tmp_path):
    # Small physical callback exercises SR counters/PRNG/checkpoint; no new ansatz.
    p=jnp.array([.1,.2,-.1,.3]);x=jnp.array([[0,0,1,1],[1,1,0,0],[0,1,0,1],[1,0,1,0]])
    fn=lambda p,x:(1+.2j)*jnp.dot(p,x)
    local=make_full_heisenberg_ratios(fn,[(0,1,1.),(1,2,1.),(2,3,1.)])
    arm=next(a for a in json.loads(PRIOR_CONFIG.read_text())['arms'] if a['id']=='ridge_sr')
    base=dict(params=dict(model=p,chains=x,key=jax.random.PRNGKey(8)))
    cfg=dict(proposals_per_update=4,steps=2,chains=4,maximum_training_seconds_per_arm=100.)
    reg=dict(result_sha256='unit-test');warm=local(p,x)
    full,fp=train_fork(tmp_path/'full',base,{},arm,cfg,reg,fn,local,warm)
    train_fork(tmp_path/'resume',base,{},arm,{**cfg,'steps':1},reg,fn,local,warm)
    # Remove only the generated test completion marker, preserving the saved checkpoint.
    (tmp_path/'resume'/'training.json').unlink()
    resumed,rp=train_fork(tmp_path/'resume',base,{},arm,cfg,reg,fn,local,warm)
    assert pytree_sha256(fp)==pytree_sha256(rp)
    assert full['training_state_sha256']==resumed['training_state_sha256']
    assert resumed['completed_steps']==2
