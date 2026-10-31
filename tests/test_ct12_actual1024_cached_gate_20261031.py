"""Small independent scope, physical-step, cache and geometry safety checks."""
from copy import deepcopy
import json
from pathlib import Path

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct12_actual1024_cached_gate_20261031 as flow
from projects.clebsch_tree_nqs.explorations.ct12_actual1024_bundle_20261031 import make_plan


def test_closed_actual_record_counts_legal_widths_and_absolute_external_inputs():
    plan=make_plan();flow.validate_plan(plan,executing=True)
    assert all(Path(x['path']).is_absolute() for x in plan['inputs'].values())
    assert plan['original_parameters']==1495080 and plan['wide_parameters']==20748160
    for field,value in [('fit_records',512),('bilateral_records',128),('confirmation_records',512),
                        ('audit_records',1024),('wide_parameters',2624544),('maximum_FS',.05),
                        ('derivative_row_batch',16),('automatically_start_pilot',True)]:
        altered=deepcopy(plan);altered[field]=value
        with pytest.raises(ValueError):flow.validate_plan(altered)
    for field in ('heads','nodelist','low_rank_J','frozen_blocks'):
        altered=deepcopy(plan);altered[field]=1
        with pytest.raises(ValueError):flow.validate_plan(altered)


def test_legacy_coordinate_LR_cannot_falsely_claim_matched_physical_tangent():
    parent={'w':jnp.array([1.,2.])};proposal={'w':jnp.array([1.09,1.97])}
    info=dict(accepted=True,sampled_tangent_FS=.0009,update_scale=.03,
              applied_energy_directional_derivative=-.0001)
    changed,_,details=flow.normalize_legacy_proposal(parent,(proposal,(45,None,None),info))
    np.testing.assert_allclose(changed['w']-parent['w'],(proposal['w']-parent['w'])*(.015/.0009),atol=1e-15)
    assert details['sampled_tangent_FS']==.015
    assert details['update_scale']==pytest.approx(.5)
    with pytest.raises(ArithmeticError):flow.normalize_legacy_proposal(parent,(proposal,None,dict(info,sampled_tangent_FS=0)))


def test_cached_full_amplitude_F_exact_direction_matches_legacy_not_two_F_with_node():
    from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr
    matrix=jnp.array([[1.,0.],[0.,1.],[1.,1.],[-1.,2.]],dtype=jnp.float64)
    parent={'w':jnp.array([0.,1.])};moments=(jnp.asarray(45),jax.tree.map(jnp.zeros_like,parent),jax.tree.map(jnp.zeros_like,parent))
    rows=jnp.arange(4,dtype=jnp.int32)[:,None]
    batch=lambda p,x:(matrix@p['w'])[x[:,0]]
    psi=np.asarray(batch(parent,rows));assert psi[0]==0
    hpsi=np.asarray([.7,-.3,.4,.2]);density=np.asarray([.13,.27,.31,.29])
    rt=dict(prepare=lambda x:x,amplitude=batch)
    plan=make_plan();new,moved,info,cache=flow.cached_proposal(plan,rt,parent,moments,rows,psi,hpsi,density)
    old=make_importance_amplitude_sr(lambda p,x:batch(p,x[None])[0],parent,damping=.5,score_chunk=1,sr_tangent_fs_radius=.003)
    legacy=jax.block_until_ready(old(parent,moments,rows,hpsi,density,.03))
    expected,_,_=flow.normalize_legacy_proposal(parent,legacy)
    np.testing.assert_allclose(new['w'],expected['w'],atol=2e-12,rtol=2e-12)
    assert int(moved[0])==46 and float(info['sampled_tangent_FS'])==pytest.approx(.015,abs=1e-14)
    assert bool(info['metric_is_F_not_two_F'])
    assert np.linalg.norm(cache.jacobian[0])>0
    theta,_,_=flow.batch_function(rt,parent)
    with pytest.raises(ValueError):cache.verify(theta+.01,rows,psi,density,source_identity=flow.cache_source(plan))


def test_independent_amplitude_bracket_veto_never_reselects_after_guard_failure():
    signs=np.tile([-1.,1.],512);parent=np.ones(1024)
    phi=np.stack([np.cos(.015*f)*parent+np.sin(.015*f)*signs for f in flow.GEOMETRY_FACTORS],axis=1)
    selected,info=flow.select_geometry(phi,np.ones(1024))
    assert info['accepted'] and selected<len(flow.GEOMETRY_FACTORS)-1
    assert .0075<=info['guard']['FS_radians']<=.02
    assert info['guard_not_used_for_reselection']
    wrong=phi.copy()
    for i,f in enumerate(flow.GEOMETRY_FACTORS):wrong[512:,i]=np.cos(.06*f)+np.sin(.06*f)*signs[512:]
    selected,info=flow.select_geometry(wrong,np.ones(1024))
    assert selected==len(flow.GEOMETRY_FACTORS)-1 and not info['accepted']


def test_warm_cost_forecast_uses_all_records_not_cold_call_times():
    p=make_plan();timing=dict(H_seconds_per_center=2.,H_JVP_warm_seconds_per_center=5.,
        actual1024_amplitude_warm_seconds=10.,H8_cold_seconds=1000.,H_JVP_cold_seconds=2000.)
    report=flow.gate_forecast(p,timing)
    assert report['components_seconds']['bilateral_512']==2560
    assert report['components_seconds']['paired_confirmation_2048']==4096
    assert report['components_seconds']['full_FS_8x1024']==80
    assert report['forecast_seconds']==pytest.approx(1.5*(2560+4096+80)+1800+1800+300)
    changed=flow.gate_forecast(p,dict(timing,H8_cold_seconds=1e9,H_JVP_cold_seconds=1e9))
    assert changed['forecast_seconds']==report['forecast_seconds']


def test_peak_missing_is_uncertified_and_guard_radius_mismatch_is_not_energy_win(monkeypatch,tmp_path):
    class Device:
        def memory_stats(self):return None
    monkeypatch.setattr(flow.jax,'local_devices',lambda:[Device()])
    with pytest.raises(RuntimeError):flow.check_peak(make_plan())
    for arm,radius in [('original',.0016),('wide28',.0027)]:
        p=tmp_path/arm;p.mkdir()
        (p/'gate_result.json').write_text(json.dumps(dict(geometry=dict(guard=dict(FS_radians=radius,group_jackknife_SE=1e-5)))))
    report=flow.joint_guard_report(tmp_path)
    assert not report['matched_with_uncertainty'] and not report['comparative_energy_claim_allowed']
    assert report['each_arm_direction_certificate_remains_separate']


def test_scientific_replay_has_no_unit_scale_floor():
    assert flow.replay_error([1e-30],[1e-30])==0
    with pytest.raises(ArithmeticError):flow.replay_error([2e-30],[1e-30])
    with pytest.raises(ArithmeticError):flow.replay_error([1e-300],[0.])


def test_first_partial_pilot_attempt_preserves_state_but_consumes_reserved_native_roles(monkeypatch,tmp_path):
    params={'w':jnp.array([1.,2.])};zero=jax.tree.map(jnp.zeros_like,params)
    rng=np.random.default_rng(9831).bit_generator.state
    states={'original':dict(params=params,optimizer_state=(jnp.asarray(45),zero,zero),metadata=dict(rng=deepcopy(rng)))}
    monkeypatch.setattr(flow,'validate_pilot_release',lambda *args:{})
    monkeypatch.setattr(flow,'make_runtime',lambda *args:dict(update=lambda *args:None))
    def interrupted(*args,**kwargs):raise TimeoutError('synthetic partial work')
    monkeypatch.setattr(flow.legacy,'transition_once',interrupted)
    (tmp_path/'compare_result.json').write_text(json.dumps(dict(timings=dict(original=dict(H_seconds_per_center=.001,SR_cold_seconds=1.)))))
    result=flow.pilot_phase(make_plan(),states,{'original':()},tmp_path,'original',lambda:1e8,{},'pilot_prepare')
    assert result['status']=='first_partial_no_commit_roles_consumed'
    state,metadata=flow.legacy.restart.load_state(tmp_path/'pilot/checkpoint.pkl')
    np.testing.assert_array_equal(state[0]['w'],params['w']);assert int(state[1][0])==45
    seeds,following=flow.active._seeds(rng)
    assert state[2]==following and state[2]!=rng and metadata['completed_updates']==0
    assert json.loads((tmp_path/'pilot/first_interrupted_roles.json').read_text())['reserved_consumed_roles']==seeds


def test_pilot_microscopic_guard_is_veto_not_smaller_step_fallback(monkeypatch,tmp_path):
    params={'w':jnp.array([1.])};zero={'w':jnp.array([0.])};rng=np.random.default_rng(73).bit_generator.state
    state=(params,(jnp.asarray(45),zero,zero),rng);_,following=flow.active._seeds(rng)
    moved=({'w':jnp.array([.9])},(jnp.asarray(46),zero,zero),following)
    monkeypatch.setattr(flow.legacy,'transition_once',lambda *args,**kwargs:(moved,dict(accepted=True,calibration=dict(guard=dict(FS_radians=.003)))))
    path=tmp_path/'step';path.mkdir()
    actual,info=flow.physical_transition(make_plan(),{},state,path)
    assert not info['accepted'] and info['no_post_veto_reselection']
    np.testing.assert_array_equal(actual[0]['w'],params['w']);assert int(actual[1][0])==45
    assert actual[2]==following


def test_duplicate_gate_cannot_change_reserved_budget_epoch(tmp_path):
    path=tmp_path/'original_gate_budget.json';path.write_text('{"started_epoch": 123.0}')
    before=path.read_bytes()
    with pytest.raises(FileExistsError):flow.fresh_phase_path(tmp_path,'gate','original')
    assert path.read_bytes()==before and len(list(tmp_path.iterdir()))==1
    (tmp_path/'replay_original_result.json').write_text('{}')
    with pytest.raises(FileExistsError):flow.fresh_phase_path(tmp_path,'replay','original')
    assert path.read_bytes()==before


def test_original_ready_seal_survives_missing_wide_and_rejects_tampered_fit(tmp_path):
    p=make_plan();timings={'actual1024_amplitude_warm_seconds':1.}
    fit=tmp_path/'fit_panel.npz';fit.write_bytes(b'synthetic sealed fit only for readiness API test')
    proposal=tmp_path/'original_full_proposal.pkl';proposal.write_bytes(b'synthetic proposal not a checkpoint')
    sr=tmp_path/'original_fit_SR.json';sr.write_text('{}')
    ready=dict(status='ready',arm='original',fit_records=1024,
        plan_sha256=flow.canonical_sha256(p),source_identity=flow.cache_source(p),
        timings=timings,timings_sha256=flow.canonical_sha256(timings),native_RNG_advanced=False,
        fit_panel_sha256=flow.file_sha(fit),proposal_sha256=flow.file_sha(proposal),SR_sha256=flow.file_sha(sr))
    (tmp_path/'original_ready.json').write_text(json.dumps(ready))
    assert flow.require_arm_ready(p,tmp_path,'original')['status']=='ready'
    assert not (tmp_path/'wide28_ready.json').exists()
    with pytest.raises(FileNotFoundError):flow.require_arm_ready(p,tmp_path,'wide28')
    fit.write_bytes(b'changed')
    with pytest.raises(ValueError):flow.require_arm_ready(p,tmp_path,'original')
