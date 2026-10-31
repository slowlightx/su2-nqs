"""Frozen-parent candidate construction and unbiased/biased data separation."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from projects.clebsch_tree_nqs.workflows.spin_full_vit_step_audit import CONFIG,ROOT,probe_neighborhood,probe_diagnostics,evaluate_chunks
from projects.clebsch_tree_nqs.workflows.spin_full_vit_optimization import CONFIG as PRIOR,build_update
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.frontier.artifacts import pytree_sha256


def test_fixed_probes_retain_centers_and_exact_one_bond_support():
    spec=json.loads(CONFIG.read_text());ref=json.loads((ROOT/spec['recheck_directory']/'square/reference_check.json').read_text())
    centers=np.array(ref['states']);bonds=[(0,1,1.),(1,5,.5),(5,10,0.)]
    states,indices,mask=probe_neighborhood(centers,bonds)
    np.testing.assert_array_equal(states[indices],centers)
    assert np.all(states.sum(axis=1)==18)
    assert len({tuple(x) for x in states})==len(states)
    expected={tuple(x) for x in centers}
    for x in centers:
        for a,b,c in bonds:
            if c:
                y=x.copy();y[a],y[b]=x[b],x[a];expected.add(tuple(y))
    assert {tuple(x) for x in states}==expected
    assert mask.shape==(3,len(states))
    r=np.linspace(-.3,.4,len(states))+.1j
    a=probe_diagnostics(r,indices,states,states[:3])
    b=probe_diagnostics(r+1000+2j,indices,states,states[:3])
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-12)
    assert a['training_overlap_count']==3


def test_all_candidates_are_single_same_parent_steps_and_chunk_invariant():
    spec=json.loads(CONFIG.read_text());prior=json.loads(PRIOR.read_text())
    p=jnp.asarray([.1,-.2,.3],dtype=jnp.float64);x=jnp.arange(8)[:,None]
    fn=lambda p,x:(p[0]*(x[0]-3.5)+p[1]*x[0]**2+1j*p[2]*x[0])
    e=jnp.asarray(np.cos(np.arange(8))+1j*np.sin(np.arange(8)));before=pytree_sha256(p)
    for plan in spec['arms']:
        arm={**next(a for a in prior['arms'] if a['id']==plan['base']),**plan['overrides'],'id':plan['id']}
        opt=_manual_adam_init(p) if arm['optimizer']=='adam' else dict(step=jnp.asarray(0))
        changed,_,d=build_update(arm,fn,p)(p,opt,x,e)
        assert pytree_sha256(p)==before and changed.shape==p.shape
        assert np.isfinite(changed).all() and float(d.get('relative_linear_residual',0))<1e-7
    batch=jax.jit(jax.vmap(fn,in_axes=(None,0)))
    np.testing.assert_allclose(evaluate_chunks(batch,p,x,3),batch(p,x),atol=1e-13)


def test_fixed_budget_and_seed_separation():
    spec=json.loads(CONFIG.read_text());v=spec['validation'];seeds=spec['training_seeds']+[s['seed'] for s in v['streams']]
    assert len(seeds)==len(set(seeds))==4
    assert len(spec['arms'])==4 and spec['bootstrap']['family_tests']==16
    assert v['ordinary_burn_proposals']%(len(v['betas'])*v['swap_every'])==0
    assert v['ordinary_thin_proposals']%(len(v['betas'])*v['swap_every'])==0
    val=len(v['streams'])*v['chains']*v['snapshots'];t=spec['training_chains'];seeds=len(spec['training_seeds'])
    assert seeds*t+val+seeds*len(spec['arms'])*(t+val)==9856
