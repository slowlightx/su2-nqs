"""Same-backbone optimizer policies and independent frozen-state inference."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_full_vit_optimization import (
    CONFIG,build_update,compare_audits,load_parent,prepare_initial_state,
)
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from su2_nqs.benchmarks.spatial_vit_sampling import projected_log_function,make_full_heisenberg_ratios
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig,init_fixed_sz_spatial_vit
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.frontier.artifacts import pytree_sha256


def test_common_initial_fork_is_exactly_resumable_and_fail_closed(tmp_path):
    path=tmp_path/'initial.pkl'
    p=jnp.asarray([.1,.2,.3,.4])
    log_fn=lambda p,x: jnp.dot(p,x).astype(jnp.complex128)
    args=dict(seed=73,chains=8,n_sites=4,burn=12,registration_sha256='registered')
    first=prepare_initial_state(path,p,log_fn,**args)
    resumed=prepare_initial_state(path,p,log_fn,**args)
    assert pytree_sha256(first['params'])==pytree_sha256(resumed['params'])
    with pytest.raises(ValueError,match='protocol'):
        prepare_initial_state(path,p,log_fn,**{**args,'registration_sha256':'changed'})
    with pytest.raises(ValueError,match='model/chain'):
        prepare_initial_state(path,p+.01,log_fn,**args)
    bad=first['params'];bad['chains']=jnp.zeros((8,4),dtype=jnp.int32)
    save_checkpoint(path,bad,metadata=first['metadata'])
    with pytest.raises(ValueError,match='model/chain'):
        prepare_initial_state(path,p,log_fn,**args)


def test_real_vit_optimizers_have_same_force_and_sr_checkpoint_replay(tmp_path):
    spec=json.loads(CONFIG.read_text())
    config=FixedSzSpatialViTConfig(geometry='square',rows=2,columns=2,
        patch_rows=1,patch_columns=1,embedding_dim=4,head_count=2,depth=2)
    p=init_fixed_sz_spatial_vit(jax.random.PRNGKey(31),config)
    case=_prepare_case(dict(geometry='square',rows=2,columns=2,j2_over_j1=.5,
        exact_energy_anchor=-10.),'fixed_m0')
    logs,_=projected_log_function(config,case['groups']['gamma_a1'])
    local=make_full_heisenberg_ratios(logs,case['bonds'])
    x=jnp.asarray(case['basis']);e=local(p,x)
    updates=[build_update(arm,logs,p) for arm in spec['arms']]
    adam=_manual_adam_init(p)
    _,_,a=updates[0](p,adam,x,e)
    _,_,b=updates[1](p,adam,x,e)
    changed,opt,s=updates[2](p,dict(step=jnp.asarray(0)),x,e)
    np.testing.assert_allclose([a['gradient_norm'],b['gradient_norm']],s['gradient_norm'],atol=1e-11,rtol=1e-10)
    assert float(s['relative_linear_residual'])<1e-7
    assert float(s['fisher_norm'])<=.05*(1+1e-12)
    path=tmp_path/'checkpoint.pkl'
    save_checkpoint(path,changed,optimizer_state=opt)
    energy=local(changed,x)
    continuous,co,_=updates[2](changed,opt,x,energy)
    restored=load_checkpoint(path)
    resumed,ro,_=updates[2](restored['params'],restored['optimizer_state'],x,energy)
    assert pytree_sha256((continuous,co))==pytree_sha256((resumed,ro))


def test_comparison_is_unpaired_and_resamples_whole_chains():
    rng=np.random.default_rng(19)
    right=rng.normal(size=(3,12,8))+1j*rng.normal(size=(3,12,8))
    left=right+.08
    a=compare_audits(left,right,n_sites=4,seed=72,replicates=300)
    b=compare_audits(np.repeat(left,2,axis=1),np.repeat(right,2,axis=1),n_sites=4,seed=72,replicates=300)
    assert a['energy_per_site']['difference']==pytest.approx(.02)
    assert a['variance_per_site']['difference']==pytest.approx(0.,abs=1e-14)
    # Paired resampling would falsely report zero SE for the constant shift.
    assert a['energy_per_site']['bootstrap_se']>0
    for quantity in a:
        for key in a[quantity]:
            np.testing.assert_allclose(a[quantity][key],b[quantity][key],atol=2e-14,rtol=2e-14)
        ci=a[quantity]['chain_bootstrap_95'];wide=a[quantity]['primary_family_bonferroni_98_75']
        assert wide[0]<=ci[0]<=ci[1]<=wide[1]


def test_registered_parents_arms_and_full_H_budget():
    spec=json.loads(CONFIG.read_text())
    assert spec['steps']==128 and spec['chains']==64
    assert [a['id'] for a in spec['arms']]==['adam','adam_low_lr','ridge_sr']
    for geometry in ('triangular','square'):
        p,r=load_parent(spec,geometry)
        assert p['metadata']['step']==1064
        assert r['parameter_count']==4032
    all_seeds=[s for pools in spec['audit']['seeds'].values() for stream in pools.values() for s in stream]
    assert len(all_seeds)==len(set(all_seeds))==24
    assert not set(all_seeds)&set(spec['training_seeds'].values())
    assert 2*(64+3*128*64+4*3*32*32)==73856
