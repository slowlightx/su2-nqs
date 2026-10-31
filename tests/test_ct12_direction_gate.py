"""Independent dense-H and process-contract checks for the opt-in CT12 gate."""
from copy import deepcopy
import json

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct12_direction_gate as gate
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.vmc.finite_candidate_calibration import bilateral_direction_moments
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr


def plan_fixture():
    """Actual production active profile, without loading any large state."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'configs/remote/ct12_production32_20261028/j0125.json'
    p = json.loads(path.read_text())
    for key in ('software_acceptance', 'automatic_next_allocation', 'no_requeue', 'parent_format'):
        p.pop(key, None)
    p.update(task=gate.TASK, device='A100', parent_global_step=37,
        parent_parameter_sha256='b'*64, parent_directory='/unused',
        parent_seals={name:'a'*64 for name in gate.PARENT_FILES},
        bilateral_records=512, bilateral_groups=32, confirmation_records=1024,
        confirmation_groups=32, steps=8, gate_wall_cap_seconds=21600,
        pilot_wall_cap_seconds=50400, maximum_peak_device_bytes=70*1024**3,
        seeds={name:2910296000+i for i,name in enumerate(gate.SEED_ROLES)})
    return p


def moment_fixture():
    concentration = dict(effective_contributions=[400.,390.],
        maximum_sample_fraction=[.01,.01], maximum_group_fraction=[.05,.05])
    return dict(candidate_minus_control=[-.002,0.,-.003,0.],
        paired_iid_delta_SE=[.0001,0.,.0002,0.],
        paired_independent_block_jackknife_SE=[.0002,0.,.0003,0.],
        norm_concentration=deepcopy(concentration), H2_concentration=deepcopy(concentration),
        residual_concentration=deepcopy(concentration))


def test_closed_plan_and_controls():
    p = plan_fixture(); gate.validate_plan(p, executing=True)
    assert gate.runtime_plan(p)['active_profile'] == p['active_profile']
    assert gate.runtime_plan(p)['model']['multiplicity_cap'] == 20
    for key,value in [('steps',9),('fit_records',512),('target_FS',.03),('damping',.05),
                      ('active_real_parameters',7856880),('parent_global_step',5),('device','B200')]:
        altered = deepcopy(p); altered[key]=value
        with pytest.raises(ValueError): gate.validate_plan(altered)
    altered=deepcopy(p);altered['active_profile'][-1]={'0':2}
    with pytest.raises(ValueError): gate.validate_plan(altered)
    altered=deepcopy(p);altered['seeds']['audit']=altered['seeds']['confirmation']
    with pytest.raises(ValueError): gate.validate_plan(altered)
    altered=deepcopy(p);altered['execution_authorized']=False
    with pytest.raises(PermissionError): gate.validate_plan(altered,executing=True)


def retry_plan_fixture():
    """Explicit retry changes execution budget/batching, never fixed physics."""
    p = plan_fixture()
    p.update(runtime_budget_profile=gate.BUDGET_RETRY_PROFILE,
        gate_wall_cap_seconds=36000, pilot_wall_cap_seconds=64800,
        projection_chunk=16, H_chunk=8, action_chunk=4, score_chunk=1,
        budget_retry_evidence=deepcopy(gate.BUDGET_RETRY_EVIDENCE))
    return p


def test_retry_budget_is_opt_in_and_preserves_full_scientific_contract():
    p = retry_plan_fixture(); gate.validate_plan(p, executing=True)
    original = plan_fixture()
    for key in ('fit_records', 'calibration_records', 'bilateral_records',
                'confirmation_records', 'audit_records', 'steps', 'active_profile',
                'damping', 'target_FS', 'maximum_FS', 'active_real_parameters'):
        assert p[key] == original[key]
    for key, value in [('gate_wall_cap_seconds',43200), ('pilot_wall_cap_seconds',72000),
                       ('H_chunk',2), ('projection_chunk',4), ('fit_records',512),
                       ('bilateral_records',128), ('audit_records',1024),
                       ('runtime_budget_profile','arbitrary_budget')]:
        changed=deepcopy(p);changed[key]=value
        with pytest.raises(ValueError):gate.validate_plan(changed)
    for key in ('previous_preflight_sha256', 'zero_fit_records_consumed'):
        changed=deepcopy(p);changed['budget_retry_evidence'].pop(key)
        with pytest.raises(ValueError):gate.validate_plan(changed)
    changed=deepcopy(p);changed['budget_retry_evidence']['previous_replay_sha256']='c'*64
    with pytest.raises(ValueError):gate.validate_plan(changed)
    # Larger budgets do not silently loosen the sealed historical profile.
    changed=deepcopy(p);changed.pop('runtime_budget_profile')
    with pytest.raises(ValueError):gate.validate_plan(changed)


def test_retry_forecast_distinguishes_actual_batched_H_and_single_H_JVP():
    p=retry_plan_fixture()
    replay=dict(warm_H_seconds=.8,single_H_seconds=3.4,
        warm_H_batch_records=8,full_H_centers=20)
    report=gate.gate_cost_forecast(p,replay)
    assert report['registered_record_counts']==dict(fit=1024,calibration=1024,
        bilateral=512,paired_confirmation=1024)
    assert report['charged_H_equivalent_centers']==dict(fit_full_H=1024,
        bilateral_primal_full_H=512,bilateral_tangent_full_H=512,
        bilateral_JVP_work_margin=512,confirmation_paired_full_H=2048,
        transition_product_replay_full_H=2,isolated_transition_product_replay_full_H=2)
    expected=1.5*((1024+2048)*.8+(3*512+4)*3.4)+4200
    assert report['estimated_H_plus_compilation_seconds']==pytest.approx(expected)
    assert report['records_never_reduced'] and report['one_fit_direction_only']
    assert report['already_spent_parent_replay_not_charged_twice']
    assert sum(report['explicit_nonquery_allowance_seconds'].values())==4200
    slow=gate.gate_cost_forecast(p,dict(replay,single_H_seconds=6.8))
    assert slow['estimated_H_plus_compilation_seconds']>report['estimated_H_plus_compilation_seconds']
    for bad in (dict(replay,warm_H_batch_records=2),dict(replay,single_H_seconds=0),
                dict(replay,warm_H_seconds=float('nan'))):
        with pytest.raises(ValueError):gate.gate_cost_forecast(p,bad)
    legacy=gate.gate_cost_forecast(plan_fixture(),dict(warm_H_seconds=3.4))
    assert legacy['estimated_H_plus_compilation_seconds']==pytest.approx(1.5*4096*3.4+1800)


def test_retry_saved_parent_replay_uses_H8_and_single_native_scale(monkeypatch,tmp_path):
    """Independent simple four-product kernel checks both execution shapes."""
    p=retry_plan_fixture(); rows=np.arange(12,dtype=np.int32).reshape(12,1)
    calls=[]
    def products(params,bits):
        calls.append((float(params['scale']),len(bits)))
        x=jnp.asarray(bits[:,0],float)+1
        scale=params['scale']
        return scale*x,scale*(2*x+1),scale**2*(3*x+2),scale**2*(4*x+3)
    parent=(dict(scale=jnp.asarray(3.)),None,None)
    values=np.stack([np.asarray(x) for x in products(dict(scale=1.),rows)],axis=1)
    path=tmp_path/'parent_snapshot/terminal_audit';path.mkdir(parents=True)
    np.savez_compressed(path/'untouched_products.npz',rows=rows,
        products=np.stack((values,values),axis=1))
    calls.clear()
    monkeypatch.setattr(gate,'canonical',lambda *_:(dict(scale=jnp.asarray(1.)),dict(original_raw_norm=9.)))
    monkeypatch.setattr(gate,'check_peak',lambda _:11*1024**3)
    report=gate.replay_saved_parent(p,dict(H=products),parent,tmp_path)
    assert calls==[(1.,8),(1.,8),(1.,1),(1.,1),(3.,1),(3.,1)]
    assert report['full_H_centers']==20 and report['warm_H_batch_records']==8
    assert report['single_H_seconds']>0 and report['warm_H_seconds']>0
    assert report['observed_peak_device_bytes']==11*1024**3
    assert max(report['component_relative_errors'])==0
    assert max(report['single_native_component_relative_errors'])==0
    assert report['native_product_scale_factors']==[3.,3.,9.,9.]
    assert report['batching_changes_execution_only_not_H_images_or_samples']
    # Every saved guide is checked, not just psi/Hpsi.
    damaged=values.copy();damaged[7,3]+=1
    np.savez_compressed(path/'untouched_products.npz',rows=rows,
        products=np.stack((values,damaged),axis=1))
    with pytest.raises(ArithmeticError):gate.replay_saved_parent(p,dict(H=products),parent,tmp_path)


def test_retry_actual_peak_not_replaced_by_low_Jacobian_estimate(monkeypatch):
    p=retry_plan_fixture()
    class Device:
        def __init__(self,peak):self.peak=peak
        def memory_stats(self):return dict(peak_bytes_in_use=self.peak)
    monkeypatch.setattr(gate.jax,'local_devices',lambda:[Device(69*1024**3)])
    assert gate.check_peak(p)==69*1024**3
    monkeypatch.setattr(gate.jax,'local_devices',lambda:[Device(71*1024**3)])
    with pytest.raises(MemoryError):gate.check_peak(p)


def test_gate_native_rng_is_exactly_unchanged_and_collisions_rejected():
    p=plan_fixture(); original=np.random.default_rng(173).bit_generator.state
    before=deepcopy(original)
    roles=gate.separate_role_streams(p,original,{37,38})
    assert original==before
    assert roles['native_rng_sha256']==canonical_sha256(before)
    assert not set(roles['gate_transition_seeds']) & set(roles['future_native_role_seeds'])
    assert len(roles['future_native_role_seeds'])==24
    bad=deepcopy(p);bad['seeds']['audit']=roles['future_native_role_seeds'][0]
    with pytest.raises(ValueError):gate.separate_role_streams(bad,original,{37,38})


def test_bilateral_native_H_JVP_matches_dense_quotient_derivative():
    rng=np.random.default_rng(871); matrix=rng.normal(size=(8,8));H=matrix+matrix.T
    offset=rng.normal(size=8); A=rng.normal(size=(8,3));B=rng.normal(size=(8,3))
    theta=jnp.array([.2,-.3,.1]);direction=jnp.array([.12,.2,-.09])
    def psi(p):return jnp.asarray(offset)+jnp.asarray(A)@p+.2*jnp.sin(jnp.asarray(B)@p)
    def action(p,rows):
        value=psi(p); h=jnp.asarray(H)@value
        # The two guides are placeholders here; derivatives of guide do not
        # enter either coherent slope term or the physical quotient.
        return value[rows],h[rows],jnp.ones_like(value[rows]),jnp.ones_like(value[rows])
    rows=jnp.arange(8)
    primal,tangent=gate.native_h_jvp(dict(H=action),theta,direction,rows)
    derivative=bilateral_direction_moments(np.asarray(primal[0]),np.asarray(primal[1]),
        np.asarray(tangent[0]),np.asarray(tangent[1]),np.ones(8)/8,groups=4,n_sites=1)
    def energy(p):
        v=psi(p);return v@(jnp.asarray(H)@v)/(v@v)
    exact=float(jax.jvp(energy,(theta,),(direction,))[1])
    assert abs(derivative['values'][2]-exact)<2e-13
    assert abs(derivative['values'][0]-derivative['values'][1])<2e-13
    eps=1e-5
    finite=float((energy(theta+eps*direction)-energy(theta-eps*direction))/(2*eps))
    assert abs(finite-exact)<2e-10
    # A wrong eigenparent simplification HD=E*D fails this independent check.
    energy0=float(energy(theta))
    wrong=bilateral_direction_moments(np.asarray(primal[0]),np.asarray(primal[1]),
        np.asarray(tangent[0]),energy0*np.asarray(tangent[0]),np.ones(8)/8,groups=4,n_sites=1)
    assert abs(wrong['values'][2]-exact)>1e-3


def test_SR_damping_and_tangent_radius_actually_change_update():
    theta=dict(w=jnp.array([.8,-.4])); rows=jnp.arange(4,dtype=jnp.int32)
    matrix=jnp.asarray([[1.,.2],[.1,1.],[-.7,.5],[.3,-.8]])
    def amplitude(p,x):return (matrix@p['w'])[x]
    H=jnp.diag(jnp.asarray([-.5,.2,.7,1.1])); p=matrix@theta['w']
    moments=(jnp.asarray(37),jax.tree.map(jnp.zeros_like,theta),jax.tree.map(jnp.zeros_like,theta))
    def update(damping,radius):
        f=make_importance_amplitude_sr(amplitude,theta,damping=damping,score_chunk=1,
            parameter_chunk_size=2,sr_tangent_fs_radius=radius)
        return f(theta,moments,rows,H@p,jnp.ones(4)/4,.03)
    a,ma,ia=update(.5,.003);b,mb,ib=update(.5,.001)
    assert float(ia['sampled_tangent_FS'])==pytest.approx(.003,abs=1e-13)
    assert float(ib['sampled_tangent_FS'])==pytest.approx(.001,abs=1e-13)
    assert np.linalg.norm(np.asarray(a['w']-theta['w']))>2.9*np.linalg.norm(np.asarray(b['w']-theta['w']))
    c,_,_=update(10.,1.)
    assert np.linalg.norm(np.asarray(c['w']-a['w']))>1e-6
    assert int(np.asarray(ma[0]))==38
    assert pytree_sha256(ma[1:])==pytree_sha256(moments[1:])


def test_gate_requires_every_independent_condition_and_no_reselection():
    bilateral=dict(values=[-.001,-.001,-.002,0.,0.],
        independent_group_jackknife_SE=[.0001]*5)
    moments=moment_fixture()
    confirmation=dict(moments=moments,coverage=gate.coverage_checks(moments,1024))
    passed=gate.gate_decision(bilateral,confirmation,geometry_passed=True,replay_passed=True)
    assert passed['passed']
    assert passed['no_candidate_reselection_using_confirmation']
    assert not gate.gate_decision(bilateral,confirmation,geometry_passed=True)['passed']
    for i in (0,2):
        bad=deepcopy(confirmation);bad['moments']['candidate_minus_control'][i]=.1
        assert not gate.gate_decision(bilateral,bad,geometry_passed=True,replay_passed=True)['passed']
    bad=deepcopy(bilateral);bad['values'][2]=.1
    assert not gate.gate_decision(bad,confirmation,geometry_passed=True,replay_passed=True)['passed']
    assert not gate.gate_decision(bilateral,confirmation,geometry_passed=False,replay_passed=True)['passed']
    moments['H2_concentration']['maximum_sample_fraction'][0]=.9
    assert not gate.coverage_checks(moments,1024)['passed']


def test_rejected_transition_consumes_roles_without_changing_parameters(monkeypatch,tmp_path):
    state=(dict(w=jnp.asarray([1.,2.])),(jnp.asarray(37),{},{}),np.random.default_rng(827).bit_generator.state)
    roles,next_rng=gate.active._seeds(state[2])
    def reject(*_,**__):return state,dict(accepted=False,seeds=roles)
    monkeypatch.setattr(gate.active,'transition',reject)
    candidate,info=gate.transition_once({}, {}, state, tmp_path)
    assert pytree_sha256(candidate[:2])==pytree_sha256(state[:2])
    assert canonical_sha256(candidate[2])==canonical_sha256(next_rng)
    assert canonical_sha256(candidate[2])!=canonical_sha256(state[2])
    assert info['rejected_update_roles_consumed']
    restored,_=gate.restart.load_state(tmp_path/'rejected_consumed_state.pkl')
    assert canonical_sha256(restored[2])==canonical_sha256(next_rng)


def test_failed_certificate_never_constructs_runtime(monkeypatch,tmp_path):
    p=plan_fixture();plan_path=tmp_path/'plan.json';atomic_json(plan_path,p)
    out=tmp_path/'run';out.mkdir();snap=out/'parent_snapshot';snap.mkdir()
    seals={}
    from su2_nqs.frontier.provenance import file_sha256
    for name in gate.PARENT_FILES:
        path=snap/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('sealed')
        seals[name]=file_sha256(path)
    state=(dict(w=jnp.asarray([1.])),(jnp.asarray(37),{},{}),np.random.default_rng(129).bit_generator.state)
    gate.restart.save_state(out/'parent.pkl',state,0,37)
    reg=dict(plan=p,plan_sha256=canonical_sha256(p),sources={},parent_seals=seals,
        parent_parameter_sha256=pytree_sha256(state[0]),parent_optimizer_sha256=pytree_sha256(state[1]),
        parent_rng_sha256=canonical_sha256(state[2]))
    atomic_json(out/'registration.json',reg)
    atomic_json(out/'gate_certificate.json',dict(passed=False))
    monkeypatch.setattr(gate,'_sources',lambda:{})
    def forbidden(*_,**__):raise AssertionError('failed gate invoked runtime')
    result=gate.run(plan_path,out,'train_prepare',runtime_factory=forbidden,require_gpu=False)
    assert result['status']=='gate_not_passed_no_training'
    assert result['no_update_callback_invoked']
    restored,_=gate.restart.load_state(out/'parent.pkl')
    assert pytree_sha256(restored[:2])==pytree_sha256(state[:2])
    assert canonical_sha256(restored[2])==canonical_sha256(state[2])


@pytest.mark.parametrize('missing_endpoint', [False, True])
@pytest.mark.parametrize('retry', [False, True])
def test_mock_process_cycle_preserves_original_pilot_rng(monkeypatch,tmp_path,missing_endpoint,retry):
    """Gate+replay+eight fresh steps+independent audit on a tiny native state."""
    p=retry_plan_fixture() if retry else plan_fixture()
    plan_path=tmp_path/'plan.json';atomic_json(plan_path,p)
    out=tmp_path/'run'
    parent=(dict(w=jnp.array([1.,.3])),(jnp.asarray(37),{},{}),np.random.default_rng(319).bit_generator.state)
    calls=[]
    from su2_nqs.frontier.provenance import file_sha256
    def bind(plan,where):
        snapshot=where/'parent_snapshot';snapshot.mkdir();seals={}
        for name in gate.PARENT_FILES:
            path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('sealed')
            seals[name]=file_sha256(path)
        gate.restart.save_state(where/'parent.pkl',parent,0,37)
        return dict(parent_parameter_sha256=pytree_sha256(parent[0]),
            parent_optimizer_sha256=pytree_sha256(parent[1]),parent_rng_sha256=canonical_sha256(parent[2]),
            parent_seals=seals,parameter_count=2,
            role_streams=gate.separate_role_streams(plan,parent[2],set()))
    def rt(*_,**__):return {}
    def transition(plan,runtime,state,path,replay=False):
        path.mkdir(exist_ok=True);roles,rng=gate.active._seeds(state[2])
        candidate=(dict(w=state[0]['w']-.001),(state[1][0]+1,{},{}),rng)
        info=dict(accepted=True,seeds=roles,warm_H_seconds=.0001,wall_seconds=.001)
        calls.append((str(path),replay,canonical_sha256(state[2]),roles))
        return candidate,info
    def bilateral(*_):return dict(values=[-.001,-.001,-.002,0.,0.],independent_group_jackknife_SE=[.0001]*5)
    def panel(plan,runtime,first,second,path,**kwargs):
        path.mkdir();return dict(moments=moment_fixture(),coverage=dict(passed=True),
            records=kwargs['count'],seed=kwargs['seed'])
    monkeypatch.setattr(gate,'_sources',lambda:{})
    monkeypatch.setattr(gate,'bind_parent',bind)
    monkeypatch.setattr(gate,'replay_saved_parent',lambda *_:dict(passed=True,warm_H_seconds=.0001,
        single_H_seconds=.0001,warm_H_batch_records=p['H_chunk'],full_H_centers=20))
    monkeypatch.setattr(gate.active,'transition',transition)
    monkeypatch.setattr(gate,'bilateral_panel',bilateral)
    monkeypatch.setattr(gate,'paired_panel',panel)
    for phase in gate.PHASES[:-1]:
        result=gate.run(plan_path,out,phase,runtime_factory=rt,require_gpu=False)
    if missing_endpoint:
        (out/'endpoint_frozen.json').unlink()
    result=gate.run(plan_path,out,'audit',runtime_factory=rt,require_gpu=False)
    cert=json.loads((out/'gate_certificate.json').read_text());assert cert['passed']
    final,metadata=gate.restart.load_state(out/'checkpoint.pkl')
    assert metadata['completed_updates']==8 and metadata['absolute_completed_steps']==45
    assert int(final[1][0])==45
    future=deepcopy(parent[2])
    for _ in range(8):_,future=gate.active._seeds(future)
    assert canonical_sha256(final[2])==canonical_sha256(future)
    pilot_first=next(c for c in calls if c[0].endswith('/step1') and not c[1])
    assert pilot_first[2]==canonical_sha256(parent[2])
    gate_first=next(c for c in calls if c[0].endswith('/gate_proposal') and not c[1])
    assert gate_first[2]!=pilot_first[2]
    if missing_endpoint:
        assert result['status']=='no_committed_endpoint_no_audit'
    else:
        assert result['terminal_audit']['records']==4096
        assert result['independent_from_gate_training_and_geometry']
