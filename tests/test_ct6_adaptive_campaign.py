"""Bounded continuation, independent parent checks and restored throughput."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp

from projects.clebsch_tree_nqs.workflows import ct6_adaptive_campaign as c
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments
from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256,canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


@pytest.fixture
def plan(tmp_path):
    p=tmp_path/'seed';(p/'terminal_audit').mkdir(parents=True)
    for name in ('registration.json','checkpoint.pkl','terminal_audit/audit.json'):
        (p/name).write_text('fixture')
    roles=c.base.required_seed_roles()
    p=c.base.make_plan(.125,p,seeds=dict(zip(roles,range(801,801+len(roles)))))
    p.update(task=c.TASK,action_chunk=4,H_chunk=4,automatic_next_allocation=True,
        campaign_segments_per_coupling=4,campaign_A100_hours_upper_per_coupling=96,
        resource_horizon='next_eight_attempt_block',parent_binding='completed_frozen_audited_predecessor_only',
        control_window_boundary='inherit_values_archive_history_reset_monitor_window',segment_index=1)
    return p


def test_new_profile_is_explicit_and_old_validation_stays_closed(plan):
    c.validate_plan(plan)
    with pytest.raises(PermissionError):c.validate_plan(plan,executing=True)
    c.validate_plan(plan|dict(execution_authorized=True),executing=True)
    with pytest.raises(ValueError):c.base.validate_plan(plan)
    for change in (dict(campaign_segments_per_coupling=5),dict(action_chunk=1),dict(H_chunk=1),
                   dict(segment_index=5),dict(resource_horizon='all_remaining_updates')):
        with pytest.raises(ValueError):c.validate_plan(plan|change)


def test_controller_values_inherit_but_prior_history_is_not_forged(plan):
    controls=c.initial_controls(plan,dict(inherited_control_values=dict(fs_radius=.000625,damping=.1,fit_records=5632)))
    assert (controls.fs_radius,controls.damping,controls.fit_records)==(.000625,.1,5632)
    assert controls.looks==0 and controls.consumed_panel_ids==() and controls.good_streak==0


def test_next_block_not_unaffordable_whole_tail_controls_continuation(plan,monkeypatch):
    envelope=c.base.SRResourceEnvelope(parameter_count=1535560,maximum_device_bytes=70*1024**3,
        remaining_updates=24,remaining_fit_H_records=139776,remaining_training_seconds=51954.3715,
        seconds_per_fit_record=.26002496,other_seconds_per_step=1390.1985)
    assert not envelope.permits(5120)
    monkeypatch.setattr(c.base,'resources_for_next_block',lambda *a,**k:envelope)
    bounded=c.resources_for_next_block(plan,{},{} )
    assert bounded.remaining_updates==8 and bounded.permits(5120)
    assert bounded.remaining_training_seconds==envelope.remaining_training_seconds
    assert bounded.maximum_device_bytes==envelope.maximum_device_bytes


def test_independent_terminal_ratios_errors_and_harm_gate(tmp_path):
    rng=np.random.default_rng(102923)
    psi=rng.normal(size=(128,2))+1j*rng.normal(size=(128,2));psi[:,1]=psi[:,0]
    h=psi*np.array([-20.1,-20.1003])[None,:]
    Q=rng.random(128)+.2;labels=np.repeat(np.arange(16),8)
    m=paired_iid_moments(psi,h,Q,labels,n_sites=36,seed=901)
    path=tmp_path/'products.npz';np.savez(path,psi=psi,hpsi=h,proposal=Q)
    audit=dict(records=128,moments=m)
    assert c.independent_parent_products(path,audit)['safe_to_continue']
    bad=deepcopy(audit);bad['moments']['endpoint_values'][0][0]+=.01
    with pytest.raises(ArithmeticError):c.independent_parent_products(path,bad)
    h[:,1]=(-19.+.3*rng.normal(size=128))*psi[:,1]
    np.savez(path,psi=psi,hpsi=h,proposal=Q)
    audit['moments']=paired_iid_moments(psi,h,Q,labels,n_sites=36,seed=901)
    with pytest.raises(ValueError,match='resolves harm'):c.independent_parent_products(path,audit)


def test_completed_adaptive_parent_binds_exact_native_state_and_archives_controller(plan,tmp_path):
    src=tmp_path/'old'/'run01';(src/'terminal_audit').mkdir(parents=True)
    old=dict(plan,task=c.base.TASK,execution_authorized=True,action_chunk=1,H_chunk=1,automatic_next_allocation=False)
    c.base.validate_plan(old,executing=True)
    params={'w':jnp.asarray([1.,.25,-.4,0.])};zero=jax.tree.map(jnp.zeros_like,params)
    rng=dict(generator=np.random.default_rng(71).bit_generator.state,next_step=37,
        consumed_role_seeds=[],excluded_seeds=list(old['seeds'].values()))
    progress=c.base.initial_progress()
    for i in range(8):
        _,rng=c.base.c6.role_seeds(rng,rng['next_step'])
        progress=c.base.consume_attempt(progress,dict(attempt=i+1,accepted=True,
            work=dict(fit_H_centers=5120),predicted_decrease_per_site=0.))
    state=(params,(jnp.asarray(508),zero,zero),rng)
    controls=c.base.AdaptiveSRState(fs_radius=.000625,damping=.1,fit_records=5632,
        looks=1,last_completed_step=8,last_endpoint_sha256=pytree_sha256(params),
        consumed_panel_ids=('oldpanel',),consumed_monitor_seeds=(903,904))
    progress['monitor_history']=[dict(reason='fixture')];progress['consumed_monitor_roles']=[dict(sample=903,statistics=904)]
    reg=dict(plan=old,parent_global_step=500,native_origin=464,initial_native_cursor=37)
    atomic_json(src/'registration.json',reg)
    c.base.save_adaptive_state(src/'checkpoint.pkl',state,controls,progress,plan=old,registration=reg)
    rng2=np.random.default_rng(72);psi=np.repeat(rng2.normal(size=(8192,1)),2,axis=1)
    h=psi*np.asarray([-20.,-20.0003])[None,:];Q=np.full(8192,.1)
    np.savez(src/'terminal_audit/products.npz',psi=psi,hpsi=h,proposal=Q)
    moments=paired_iid_moments(psi,h,Q,np.repeat(np.arange(128),64),n_sites=36,seed=73)
    digest=pytree_sha256(params);checksum=file_sha256(src/'checkpoint.pkl')
    # Match the REAL c6 paired_audit schema: independence is on the final
    # driver receipt, not an invented boolean inside terminal_audit/audit.json.
    audit=dict(status='complete',records=8192,endpoint_params_sha256=digest,moments=moments,
        seed=old['seeds']['audit'],statistics_seed=old['seeds']['audit_statistics'],no_endpoint_selection=True)
    atomic_json(src/'terminal_audit/audit.json',audit)
    final=dict(status='complete_fixed_prefix_independent_audit',terminal_audit=audit,
        parameter_sha256=digest,checkpoint_sha256=checksum,independent_from_training_geometry_and_adaptation=True)
    atomic_json(src/'audit_result.json',final)
    atomic_json(src/'endpoint_frozen.json',dict(parameter_sha256=digest,checkpoint_sha256=checksum))
    p=deepcopy(plan);p['parent_directory']=str(src);p['seeds']={k:v+10000 for k,v in p['seeds'].items()}
    out=tmp_path/'new';out.mkdir();binding=c.bind_parent(p,out)
    restored,_=c.base.legacy.load_state(out/'parent.pkl')
    assert pytree_sha256(restored[:2])==pytree_sha256(state[:2])
    assert canonical_sha256(restored[2])==canonical_sha256(rng)
    assert binding['parent_global_step']==508 and binding['independent_parent_review']['safe_to_continue']
    assert binding['controller_history'][-1]['archived_full_controller']['state']['looks']==1
    fresh=c.initial_controls(p,binding)
    assert fresh.looks==0 and (fresh.fs_radius,fresh.damping,fresh.fit_records)==(.000625,.1,5632)
    assert len(binding['parent_seals'])==6
