"""Bounded IID driver wiring, synthetic amplitudes only; no CT6 optimization."""
from copy import deepcopy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct6_importance_amplitude_continuation as w
from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256,canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import save_checkpoint


def plan():
    p=json.loads((Path(__file__).parents[1]/'configs/remote/ct6_importance_training_20261027/j0125.json').read_text())
    p['execution_authorized']=False
    return p


def test_fixed_scope_authority_seeds_and_factor_contract():
    p=plan();w.validate_plan(p)
    np.testing.assert_array_equal(w.RELATIVE_GRID,[1,.5,.25,.125,0])
    with pytest.raises(PermissionError):w.validate_plan(p,executing=True)
    for key,value in [('updates',17),('fit_records',2048),('epsilon',0.),('interim_after_updates',1),
                      ('terminal_audit_records',4096),('wall_cap_seconds',60000),('automatic_next_block',True)]:
        q=deepcopy(p);q[key]=value
        with pytest.raises(ValueError):w.validate_plan(q)
    p['seeds']['audit']=p['seeds']['training']
    with pytest.raises(ValueError):w.validate_plan(p)


def test_exact_rng_cursor_recovery_and_new_role_streams(tmp_path):
    p=plan();rng=w.initial_rng(p);params={'x':jnp.array([1.,.2])};mom=(4,{'x':jnp.ones(2)},{'x':jnp.zeros(2)})
    state=(params,mom,rng);path=tmp_path/'checkpoint.pkl';w.save_training_state(path,state,dict(completed_updates=0))
    restored,_=w.load_training_state(path)
    assert pytree_sha256(restored[:2])==pytree_sha256(state[:2]) and restored[2]==rng
    roles,next_rng=w.role_seeds(rng,1);roles2,next2=w.role_seeds(restored[2],1)
    assert roles==roles2 and next_rng==next2
    all_values=list(roles.values());rng=next_rng
    for index in range(2,17):
        values,rng=w.role_seeds(rng,index);all_values+=list(values.values())
    assert len(set(all_values))==48 and not set(all_values)&set(p['parent_registered_seeds'])
    assert not set(all_values)&set(p['seeds'].values())
    with pytest.raises(ValueError):w.role_seeds(next_rng,1)
    cp=w.load_checkpoint(path);cp['metadata']['training_rng']['next_step']=9
    w.save_checkpoint(path,cp['params'],optimizer_state=cp['optimizer_state'],metadata=cp['metadata'])
    with pytest.raises(ValueError):w.load_training_state(path)


def test_cumulative_all_process_and_campaign_deadline_budget():
    p=plan();deadline=w.datetime.fromisoformat(p['campaign_deadline_utc']).timestamp();start=deadline-60000
    assert w.remaining_seconds(p,start,start+1000,None)==49400
    assert w.remaining_seconds(p,start,start+1000,str(start+2000))==880
    assert w.remaining_seconds(p,deadline-1000,deadline-500,None)==380
    with pytest.raises(TimeoutError):w.remaining_seconds(p,start,start+50400,None)


def test_interim_unresolved_gain_is_not_called_win_and_no_guard_adaptation():
    r=dict(status='complete',records=4096,moments=dict(candidate_minus_control=[1e-6,0,1e-6,0],
        paired_iid_delta_SE=[1e-6,0,1e-6,0]))
    d=w.interim_decision(r);assert d['continue_training'] and not d['resolved_energy_improvement']
    r['moments']['candidate_minus_control'][0]=3e-6
    assert not w.interim_decision(r)['continue_training']
    r['moments']['candidate_minus_control']=[-3e-6,0,3e-6,0]
    assert not w.interim_decision(r)['continue_training']


def test_corrected_IID_metric_cap_has_nonzero_registered_candidate():
    parent=np.tile([1.,0.],64);tangent=np.tile([0.,.003],64)
    phi=np.stack([parent+factor*tangent for factor in w.FACTORS],axis=1)
    selected,report=w.select_finite_grid(phi,np.full(128,.5),w.RELATIVE_GRID,split=64)
    assert selected==3 and w.FACTORS[selected]==.25
    assert report['accepted']
    assert abs(report['guard']['FS_radians']-np.arctan(.00075))<1e-15


def fixture(monkeypatch,tmp_path,*,reject=None,interim_harm=False):
    p=plan();directory=tmp_path/'parent';directory.mkdir();p['parent']['directory']=str(directory)
    params={'x':jnp.array([1.,.2])};mom=(4,{'x':jnp.ones(2)},{'x':jnp.zeros(2)})
    parent=dict(params=params,optimizer_state=mom,metadata={})
    p['parent']['params_sha256']=pytree_sha256(params)
    save_checkpoint(directory/'checkpoint.pkl',params,optimizer_state=mom)
    for name in ('registration.json','results.json','terminal_audit.json'):atomic_json(directory/name,dict(mock=True))
    np.savez(directory/'terminal_audit.npz',bits=np.ones((2,36)),local_energies=np.ones(2))
    p['parent']['sealed_files']={name:file_sha256(directory/name) for name in p['parent']['sealed_files']}
    p['parent']['terminal_products_sha256']=file_sha256(directory/'terminal_audit.npz')
    p.update(execution_authorized=True,authorization_note='unit tests synthetic only')
    monkeypatch.setattr(w.jax,'default_backend',lambda:'gpu')
    monkeypatch.setattr(w.jax,'devices',lambda:[type('Device',(),dict(device_kind='A100 80GB'))()])
    monkeypatch.setattr(w.diagnostic,'checked_parent',lambda _:(parent,dict(native_parameter_count=2)))
    monkeypatch.setattr(w.diagnostic,'_device_peak',lambda:0)
    monkeypatch.setattr(w,'transitive_local_source_manifest',lambda *a,**k:{})
    monkeypatch.setattr(w.diagnostic.legacy,'numerical_runtime_fingerprint',lambda:dict(runtime='toy'))
    monkeypatch.setattr(w,'make_runtime',lambda *a:{})
    monkeypatch.setattr(w,'remaining_seconds',lambda *a:50000.)
    monkeypatch.setattr(w.diagnostic,'audited_parent_replay',lambda *a:dict(full_H_centers=2,maximum_relative_error=0.))
    calls=dict(transitions=[],audits=[],audit_available=[])
    def step(plan,rt,state,index,output):
        output.mkdir();roles,rng=w.role_seeds(state[2],index);calls['transitions'].append(index)
        accepted=index!=reject
        new=(jax.tree.map(lambda x:x+.0001,state[0]),(state[1][0]+1,*state[1][1:]),rng)
        report=dict(accepted=accepted,index=index,selected_index=2,actual_native_factor=1.,seeds=roles,
            rng_after_sha256=canonical_sha256(rng),calibration=dict(FS=.0009,reason='passed'),wall_seconds=1.)
        for name in ('fit_products.npz','selection_draw.npz','guard_draw.npz'):
            np.savez(output/name,rows=np.tile(np.arange(36),(2,1)))
        if accepted:w.save_training_state(output/'candidate_state.pkl',new,dict(completed_updates=index))
        atomic_json(output/'transition.json',report);return (new if accepted else state),report
    monkeypatch.setattr(w,'finite_step',step)
    def audit(plan,rt,old,state,output,count,seed,stats,remaining):
        assert pytree_sha256(old)==plan['parent']['params_sha256'];calls['audits'].append((count,seed))
        calls['audit_available'].append(remaining())
        return dict(status='complete',records=count,moments=dict(candidate_minus_control=[
            .01 if interim_harm and count==4096 else -1e-6,0,0.,0],paired_iid_delta_SE=[1e-5,0,1e-5,0]))
    monkeypatch.setattr(w,'paired_audit',audit)
    return p,parent,tmp_path/'run',calls


def test_actual_two_updates_then_interim_then16_fixed_endpoint(monkeypatch,tmp_path):
    p,parent,out,calls=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(w.os,'getpid',lambda:100);assert w.run(p,'prepare',out)['status']=='prepared_first_transition_uncommitted'
    state,meta=w.load_training_state(out/'checkpoint.pkl');assert meta['absolute_completed_steps']==476
    assert int(state[1][0])==4
    with pytest.raises(FileNotFoundError):w.run(p,'train',out)
    monkeypatch.setattr(w.os,'getpid',lambda:101);assert w.run(p,'replay',out)['status']=='replay_passed'
    assert w.load_training_state(out/'checkpoint.pkl')[1]['absolute_completed_steps']==476
    monkeypatch.setattr(w.os,'getpid',lambda:102);r=w.run(p,'train',out)
    state,meta=w.load_training_state(out/'checkpoint.pkl')
    assert r['status']=='fixed16_updates_complete_independent_audit' and meta['absolute_completed_steps']==492
    assert int(state[1][0])==20 and state[2]['next_step']==17
    assert calls['transitions']==[1,1]+list(range(2,17))
    assert calls['audits']==[(4096,p['seeds']['interim']),(16384,p['seeds']['audit'])]
    assert calls['audit_available']==[41000.,50000.]
    w.run(p,'train',out);assert len(calls['transitions'])==17


@pytest.mark.parametrize('reason',['interim','geometry'])
def test_harm_or_geometry_stops_without_retry_and_audits_committed_prefix(monkeypatch,tmp_path,reason):
    p,_,out,calls=fixture(monkeypatch,tmp_path,reject=3 if reason=='geometry' else None,interim_harm=reason=='interim')
    monkeypatch.setattr(w.os,'getpid',lambda:100);w.run(p,'prepare',out)
    monkeypatch.setattr(w.os,'getpid',lambda:101);w.run(p,'replay',out)
    monkeypatch.setattr(w.os,'getpid',lambda:102);r=w.run(p,'train',out)
    assert r['status']=='bounded_prefix_independently_audited' and r['completed_updates']==2
    assert w.load_training_state(out/'checkpoint.pkl')[1]['absolute_completed_steps']==478
    assert calls['transitions']==([1,1,2,3] if reason=='geometry' else [1,1,2])
    assert [x[0] for x in calls['audits']]==[4096,16384]


@pytest.mark.parametrize('records',[4096,5120])
def test_real_finite_step_native_scale_and_five_actual_nonlinear_candidates(monkeypatch,tmp_path,records):
    p=plan();p['fit_records']=records
    params={'x':jnp.array([1.,.1])};mom=(3,{'x':jnp.ones(2)},{'x':jnp.zeros(2)})
    state=(params,mom,w.initial_rng(p));observed={}
    def rows(count,seed):
        rng=np.random.default_rng(seed);template=np.r_[np.zeros(18,int),np.ones(18,int)]
        return np.asarray([rng.permutation(template) for _ in range(count)],np.int32)
    def amplitude(bank,bits):return bank['x'][0]+bank['x'][1]*(2*np.asarray(bits)[:,0]-1)
    def draw(banks,rt,count,seed,**kwargs):return rows(count,seed),dict(latent=np.zeros((count,1),int),Born_replay=np.zeros(count))
    monkeypatch.setattr(w.diagnostic,'draw_orbit_rows',draw)
    monkeypatch.setattr(w.diagnostic,'normalized_raw_bank',lambda p,rt:(jax.tree.map(lambda x:x/2,p),dict(original_raw_norm=4.)))
    monkeypatch.setattr(w.diagnostic,'prepare_feature_lut_bank',lambda b,cfg:b)
    monkeypatch.setattr(w.diagnostic,'native_projected_amplitude',lambda rt,p,x:amplitude(p,x))
    monkeypatch.setattr(w.diagnostic,'_device_peak',lambda:0)
    def update(parent,mom,bits,h,q,lr):
        np.testing.assert_allclose(h,.7*amplitude(parent,bits))
        assert np.all(q>0);observed.update(count=len(bits),q=q.copy())
        nextp=dict(x=parent['x']+jnp.array([0.,-.0002]))
        return nextp,(mom[0]+1,*mom[1:]),dict(accepted=jnp.array(True),sampled_tangent_FS=jnp.array(.0002))
    rt=dict(cfg=None,prepare=lambda x:x,native_update=update,
        action=lambda b,x:(amplitude(b,x),.7*amplitude(b,x)),
        bridge=lambda b,x:(amplitude(b,x),.7*amplitude(b,x),amplitude(b,x)**2/100.,amplitude(b,x)**2/100.),
        orbit_query=lambda b,x:np.stack((amplitude(b,x),amplitude(b,x)**2/100.),axis=1))
    candidate,report=w.finite_step(p,rt,state,1,tmp_path/'transition')
    assert report['accepted'] and observed['count']==records
    assert report['work']['fit_H_centers']==records and report['work']['fit_native_H_replay_centers']==2
    assert report['work']['full_native_jacobian_rows']==records
    assert report['work']['calibration_bank_centers']==20480
    assert int(candidate[1][0])==4 and candidate[2]['next_step']==2
    assert report['actual_native_factor']==2.
    np.testing.assert_allclose(candidate[0]['x'],params['x']+jnp.array([0.,-.0004]))
    a=np.load(tmp_path/'transition/fit_products.npz');np.testing.assert_allclose(a['exact_native_to_canonical_scale'],2.)

    # A failure before SR must be recoverable without redrawing/requerying H.
    broken=dict(rt,native_update=lambda *args:(_ for _ in ()).throw(MemoryError('synthetic SR allocation')))
    with pytest.raises(MemoryError):w.finite_step(p,broken,state,1,tmp_path/'failed')
    names=('fit_products.npz','fit_draw.npz','fit_draw.json','streams_frozen.json','transition.json')
    cache=dict(directory=str(tmp_path/'failed'),sealed_files={n:file_sha256(tmp_path/'failed'/n) for n in names})
    def forbidden(*a,**k):raise AssertionError('paid training fit/H must not be repeated')
    monkeypatch.setattr(w,'draw_rows',forbidden);monkeypatch.setattr(w,'_product_batches',forbidden)
    recovered,r=w.finite_step(p,rt,state,1,tmp_path/'recovery',replay_fit_from=cache)
    assert pytree_sha256(recovered[:2])==pytree_sha256(candidate[:2]) and recovered[2]==candidate[2]
    assert r['work']['fit_draws']==0 and r['work']['fit_H_centers']==0
    assert r['work']['reused_fit_records']==r['work']['reused_fit_H_centers']==records
    assert r['work']['fit_native_H_replay_centers']==2
    assert r['fit_cache_recovery']['not_independent_audit']
    base=json.loads((tmp_path/'failed/transition.json').read_text())
    for key in ('parent_params_sha256','rng_before_sha256','rng_after_sha256','index','seeds'):
        bad=deepcopy(base);bad[key]=None
        with pytest.raises(ValueError,match='stream mismatch'):
            w.restore_uncommitted_fit(cache,bad,records,2.,tmp_path/'recovery')
    with pytest.raises(ValueError,match='normalization mismatch'):
        w.restore_uncommitted_fit(cache,base,records,3.,tmp_path/'recovery')
    changed=deepcopy(cache);changed['sealed_files']['fit_products.npz']='0'*64
    with pytest.raises(ValueError,match='cache changed'):
        w.restore_uncommitted_fit(changed,base,records,2.,tmp_path/'recovery')


@pytest.mark.parametrize('comparison_parent',[None,458])
def test_paired_audit_full_support_at_exact_nodes(monkeypatch,tmp_path,comparison_parent):
    p=plan();parent={'x':jnp.array([1.])};state=(parent,(0,{},{}),w.initial_rng(p))
    p['parent']['global_step']=456
    if comparison_parent is not None:p['comparison_parent_global_step']=comparison_parent
    monkeypatch.setattr(w.diagnostic,'normalized_raw_bank',lambda p,rt:(p,dict(original_raw_norm=1.)))
    bits=np.tile(np.r_[np.zeros(18,int),np.ones(18,int)],(4096,1))
    monkeypatch.setattr(w,'draw_rows',lambda *a:(bits,dict(rng_after={'x':1})))
    def products(plan,rt,banks,rows,report,label,remaining):
        n=len(rows);a=np.zeros((n,2,4));a[:,0,0]=1;a[:,1,0]=1
        a[::2,:,0]=0;a[:,:,1]=-.3;a[:,:,2:]=.1
        report['scalar_H_centers']=2*n;return a
    monkeypatch.setattr(w,'_product_batches',products)
    r=w.paired_audit(p,{},parent,state,tmp_path/'audit',4096,1,2,lambda:1e5)
    a=np.load(tmp_path/'audit/products.npz')
    assert np.all(a['proposal']>0) and np.any(a['psi']==0) and np.all(a['hpsi']!=0)
    assert r['status']=='complete' and r['scalar_H_centers']==8192
    assert r['comparison_parent_global_step']==(456 if comparison_parent is None else comparison_parent)
    assert p['parent']['global_step']==456  # origin counts do not relabel the actual comparison parent


def test_interrupted_full_H_keeps_complete_products_and_requested_work(monkeypatch,tmp_path):
    p=plan();calls=[]
    monkeypatch.setattr(w.diagnostic,'prepare_feature_lut_bank',lambda bank,cfg:bank)
    monkeypatch.setattr(w.diagnostic,'_device_peak',lambda:0)
    def bridge(bank,rows):
        calls.append(len(rows))
        if len(calls)==2:raise TimeoutError('synthetic bounded interruption')
        return tuple(np.full(len(rows),v) for v in (1.,-.3,.1,.1))
    rt=dict(cfg=None,bridge=bridge);rows=np.zeros((12,36),int)
    report=dict(partial_artifact_prefix=str(tmp_path/'partial'))
    with pytest.raises(TimeoutError):w._product_batches(p,rt,[{}],rows,report,'synthetic')
    assert report['scalar_H_centers']==4 and report['scalar_H_centers_requested']==8
    assert report['complete_product_records']==4 and len(report['H_batch_seconds'])==1
    a=np.load(tmp_path/'partial.npz');assert a['products'].shape==(4,1,4)
    assert json.loads((tmp_path/'partial.json').read_text())['scalar_H_centers']==4


def test_nested_budget_deadline_restores_outer_timer():
    before=w.signal.getitimer(w.signal.ITIMER_REAL)
    try:
        with w.budget_timeout(10.):
            outer=w.signal.getitimer(w.signal.ITIMER_REAL)[0]
            with w.budget_timeout(3.):
                assert 0<w.signal.getitimer(w.signal.ITIMER_REAL)[0]<=3.
            restored=w.signal.getitimer(w.signal.ITIMER_REAL)[0]
            assert 0<restored<=outer and restored>3.
        assert w.signal.getitimer(w.signal.ITIMER_REAL)[0]==0
    finally:
        if before[0]>0:w.signal.setitimer(w.signal.ITIMER_REAL,*before)
