"""Finite-geometry production wiring; no new local CT6 energy optimization."""
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct6_finite_step_continuation as w
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint


def plan():
    return dict(task=w.TASK,model=dict(length=6,multiplicity_cap=28,maximum_q=10),updates=16,
        native_factors=w.FACTORS.tolist(),calibration_records=4096,calibration_split=2048,
        reused_diagnostic_confirmation_records=2048,terminal_audit_records=16384,target_FS=.001,maximum_FS=.005,
        wall_cap_seconds=50400,audit_reserve_seconds=9000,automatic_next_block=False,
        initial_verified_update_counts_toward_16=True,training_rng_policy='inherit_verified_diagnostic_fit_key_and_states',
        campaign_deadline_utc='2026-10-28T20:44:00+00:00',inherited_fit_seed=501,
        seeds=dict(audit=2001,audit_statistics=2002),step_seeds=[dict(selection=1000+2*i,guard=1001+2*i) for i in range(15)],
        parent_registered_seeds=[501,502,503,504,505,901],execution_authorized=False,release_confirmed_diagnostic_update=False,
        projection_chunk=4,action_chunk=4,score_chunk=2,H_chunk=4,
        parent=dict(directory='parent',global_step=476,params_sha256='a'*64,terminal_products_sha256='b'*64,
            sealed_files={n:'c'*64 for n in ('checkpoint.pkl','registration.json','results.json','terminal_audit.json')}),
        source_diagnostic=dict(directory='diagnostic',sealed_files={n:'d'*64 for n in (
            'registration.json','results.json','confirmation_products.npz','provisional_candidate.pkl','frozen_direction.pkl')}))


def test_fixed_scope_seed_roles_and_normalized_factor_grid():
    p=plan();w.validate_plan(p)
    assert w.diagnostic_plan(p)['seeds']['fit']==501
    assert 501 not in w.diagnostic_plan(p)['parent_registered_seeds']
    np.testing.assert_array_equal(w.RELATIVE_GRID,[1,.75,.5,.25,0])
    with pytest.raises(PermissionError):w.validate_plan(p,executing=True)
    for field,value in [('updates',17),('terminal_audit_records',2048),('native_factors',[1.,.5,0.]),
        ('automatic_next_block',True),('audit_reserve_seconds',0),('wall_cap_seconds',60000)]:
        wrong=copy.deepcopy(p);wrong[field]=value
        with pytest.raises(ValueError):w.validate_plan(wrong)
    wrong=copy.deepcopy(p);wrong['step_seeds'][3]['guard']=501
    with pytest.raises(ValueError):w.validate_plan(wrong)
    wrong=copy.deepcopy(p);wrong['step_seeds'].append(dict(selection=9001,guard=9002))
    with pytest.raises(ValueError):w.validate_plan(wrong)


def test_cumulative_budget_and_original_campaign_deadline():
    p=plan();deadline=w.datetime.fromisoformat(p['campaign_deadline_utc']).timestamp()
    start=deadline-60000
    assert w.science_seconds(p,start,start+1000,None)==49400
    assert w.science_seconds(p,start,start+1000,str(start+2000))==880
    assert w.science_seconds(p,deadline-1000,deadline-500,None)==380
    with pytest.raises((TimeoutError,ValueError)):w.science_seconds(p,start,start+50400,None)


def source_fixture(tmp_path):
    p=plan();parent_dir=tmp_path/'parent';source=tmp_path/'diagnostic';parent_dir.mkdir();source.mkdir()
    params={'x':jnp.array([.1,.2])};moments=(160,{'x':jnp.array([.3,.4])},{'x':jnp.array([.7,.8])})
    parent=dict(params=params,optimizer_state=moments,metadata={})
    delta={'x':jnp.array([.001,-.0007])};candidate=jax.tree.map(lambda a,b:a+b,params,delta)
    next_moments=(161,moments[1],moments[2]);key=jax.random.PRNGKey(72)
    chains=np.tile(np.r_[np.ones(18,int),np.zeros(18,int)],(256,1)).astype(np.int32)
    p['parent'].update(directory=str(parent_dir),params_sha256=pytree_sha256(params))
    save_checkpoint(parent_dir/'checkpoint.pkl',params,optimizer_state=moments)
    for name in ('registration.json','results.json','terminal_audit.json'):atomic_json(parent_dir/name,dict(fake_parent=True))
    np.savez(parent_dir/'terminal_audit.npz',bits=chains[:2],local_energies=np.ones(2))
    p['parent']['sealed_files']={name:file_sha256(parent_dir/name) for name in p['parent']['sealed_files']}
    p['parent']['terminal_products_sha256']=file_sha256(parent_dir/'terminal_audit.npz')
    old=w.diagnostic_plan(p);old['seeds']=dict(zip(('fit','selection','guard','confirmation','statistics'),range(501,506)))
    old['parent_registered_seeds']=[901]
    atomic_json(source/'registration.json',dict(plan=old,plan_sha256=canonical_sha256(old)))
    common=dict(parent_params_sha256=p['parent']['params_sha256'],fit_key=np.asarray(key),fit_states=chains)
    save_checkpoint(source/'frozen_direction.pkl',delta,optimizer_state=next_moments,metadata=common)
    save_checkpoint(source/'provisional_candidate.pkl',candidate,optimizer_state=moments,
        metadata=dict(common,selected_fraction=1.,proposed_optimizer_state=next_moments))
    psi=np.tile([float(params['x'].sum()),float(candidate['x'].sum())],(2048,1))
    np.savez(source/'confirmation_products.npz',rows=np.tile(chains[0],(2048,1)),psi=psi,hpsi=.7*psi)
    result=dict(status='complete_provisional_native_W_validation_no_training_release',selected_index=0,
        candidate_sha256=file_sha256(source/'provisional_candidate.pkl'),
        work=dict(confirmation_scalar_H_centers=4096,confirmation_H_JVP_centers=2048),
        source_seals_unchanged={'x':True},parent_seals_unchanged={'y':True},
        finite_step_moments=dict(candidate_minus_control=[-.001,0,.0001,0],paired_iid_delta_SE=[.0002,0,.0002,0]),
        bilateral_direction=dict(values=[-.002,-.001,-.003,0,0],independent_group_jackknife_SE=[0,0,.001,0,0]),
        resolved_ascent=False,resolved_finite_energy_increase=False,wall_seconds=1800)
    atomic_json(source/'results.json',result)
    p['source_diagnostic']=dict(directory=str(source),sealed_files={name:file_sha256(source/name) for name in p['source_diagnostic']['sealed_files']})
    p.update(execution_authorized=True,release_confirmed_diagnostic_update=True,authorization_note='test only')
    return p,parent,(candidate,next_moments,key,jnp.asarray(chains))


def test_import_commits_proposed_moments_once_and_keeps_diagnostic_fit_stream(tmp_path):
    p,parent,expected=source_fixture(tmp_path);w.validate_plan(p,executing=True)
    actual,info=w.checked_diagnostic_import(p,parent)
    assert pytree_sha256(actual)==pytree_sha256(expected)
    assert info['completed_updates']==1 and info['absolute_completed_steps']==477
    assert info['confirmation_role']=='consumed_release_not_terminal_quality'
    assert info['release_diagnostics']['variance_difference_per_site']==.0001
    original=load_checkpoint(Path(p['source_diagnostic']['directory'])/'provisional_candidate.pkl')
    assert int(original['optimizer_state'][0])==160 and int(actual[1][0])==161
    path=tmp_path/'live.pkl';w.save_training_state(path,actual,info)
    restored,_=w.load_training_state(path);assert pytree_sha256(restored)==pytree_sha256(actual)


@pytest.mark.parametrize('defect',['wrong_counter','changed_moments','old_chain','bad_energy','bad_variance','incomplete'])
def test_import_fails_closed_on_wrong_native_lineage_or_release(tmp_path,defect):
    p,parent,_=source_fixture(tmp_path);directory=Path(p['source_diagnostic']['directory'])
    if defect in ('wrong_counter','changed_moments','old_chain'):
        path=directory/'provisional_candidate.pkl';cp=load_checkpoint(path)
        if defect=='wrong_counter':cp['metadata']['proposed_optimizer_state']=(162,*parent['optimizer_state'][1:])
        if defect=='changed_moments':cp['metadata']['proposed_optimizer_state']=(161,{'x':jnp.zeros(2)},parent['optimizer_state'][2])
        if defect=='old_chain':cp['metadata']['fit_key']=np.array([0,999],np.uint32)
        save_checkpoint(path,cp['params'],optimizer_state=cp['optimizer_state'],metadata=cp['metadata'])
        result=json.loads((directory/'results.json').read_text());result['candidate_sha256']=file_sha256(path)
        atomic_json(directory/'results.json',result)
    else:
        path=directory/'results.json';result=json.loads(path.read_text())
        if defect=='bad_energy':result['finite_step_moments']['candidate_minus_control'][0]=.001
        if defect=='bad_variance':result['finite_step_moments']['candidate_minus_control'][2]=.01
        if defect=='incomplete':result['work']['confirmation_H_JVP_centers']=2044
        atomic_json(path,result)
    p['source_diagnostic']['sealed_files']={name:file_sha256(directory/name) for name in p['source_diagnostic']['sealed_files']}
    with pytest.raises(ValueError):w.checked_diagnostic_import(p,parent)


def mock_phases(monkeypatch,tmp_path,*,reject_index=None):
    p,parent,imported=source_fixture(tmp_path);out=tmp_path/'run';calls={'transition':0,'audit':0,'parent_H':0}
    monkeypatch.setattr(w.jax,'default_backend',lambda:'gpu')
    monkeypatch.setattr(w.jax,'devices',lambda:[type('Device',(),dict(device_kind='A100 80GB'))()])
    monkeypatch.setattr(w.diagnostic,'checked_parent',lambda _:(parent,dict(native_parameter_count=2)))
    monkeypatch.setattr(w.diagnostic,'_device_peak',lambda:0)
    monkeypatch.setattr(w,'transitive_local_source_manifest',lambda *a,**k:{})
    monkeypatch.setattr(w.diagnostic.legacy,'numerical_runtime_fingerprint',lambda:dict(node='compatible_gpu'))
    monkeypatch.setattr(w,'make_runtime',lambda *a:dict(prepare=lambda p:p,
        action=lambda p,rows:(np.full(len(rows),float(p['x'].sum())),np.full(len(rows),.7*float(p['x'].sum())))))
    monkeypatch.setattr(w,'science_seconds',lambda *a:50000.)
    def parent_replay(*a):calls['parent_H']+=2;return dict(full_H_centers=2,maximum_relative_error=0.)
    monkeypatch.setattr(w.diagnostic,'audited_parent_replay',parent_replay)
    monkeypatch.setattr(w.diagnostic,'normalized_raw_bank',lambda p,rt:(p,dict(original_raw_norm=1.)))
    monkeypatch.setattr(w.diagnostic,'native_projected_amplitude',lambda rt,p,rows:np.full(len(rows),float(p['x'].sum())))
    def transition(plan,rt,state,index,directory):
        calls['transition']+=1;directory.mkdir()
        new=(jax.tree.map(lambda x:x+.0001,state[0]),(state[1][0]+1,*state[1][1:]),state[2]+1,state[3])
        result=dict(accepted=index!=reject_index,selected_index=1,actual_native_factor=1.5,
            calibration=dict(FS=.0009,reason='passed'),wall_seconds=1.,work=dict(fit_H_centers=4096,score_rows=4096))
        for name in ('fit_products.npz','selection_rows.npz','guard_rows.npz'):
            np.savez(directory/name,rows=np.asarray(state[3]))
        w.save_training_state(directory/'candidate_state.pkl',new,dict(index=index))
        atomic_json(directory/'transition.json',result);return new,result
    monkeypatch.setattr(w,'finite_step',transition)
    def audit(plan,rt,parent,state,directory,remaining):
        calls['audit']+=1;assert pytree_sha256(parent)==p['parent']['params_sha256']
        return dict(status='complete',records=16384,scalar_H_centers=32768)
    monkeypatch.setattr(w,'final_audit',audit)
    return p,out,imported,calls


def test_one_import_plus15_new_updates_replay_and_fixed_fresh_audit(monkeypatch,tmp_path):
    p,out,imported,calls=mock_phases(monkeypatch,tmp_path)
    monkeypatch.setattr(w.os,'getpid',lambda:100)
    assert w.run(p,'prepare',out)['status']=='prepared_first_new_transition_uncommitted'
    state,meta=w.load_training_state(out/'checkpoint.pkl')
    assert pytree_sha256(state)==pytree_sha256(imported) and meta['absolute_completed_steps']==477
    with pytest.raises(FileNotFoundError):w.run(p,'train',out)
    monkeypatch.setattr(w.os,'getpid',lambda:101)
    assert w.run(p,'replay',out)['status']=='replay_passed'
    assert pytree_sha256(w.load_training_state(out/'checkpoint.pkl')[0])==pytree_sha256(imported)
    monkeypatch.setattr(w.os,'getpid',lambda:102)
    result=w.run(p,'train',out);state,meta=w.load_training_state(out/'checkpoint.pkl')
    assert result['status']=='fixed16_updates_complete_independent_audit'
    assert meta['completed_updates']==16 and meta['absolute_completed_steps']==492
    assert int(state[1][0])==176 and len(meta['history'])==16
    assert calls==dict(transition=16,audit=1,parent_H=6)  #15 new+one uncommitted replay, not16new.
    old=dict(calls);assert w.run(p,'train',out)==result and calls==old


def test_geometry_rejection_does_not_retry_or_commit_and_audits_prefix(monkeypatch,tmp_path):
    p,out,_,calls=mock_phases(monkeypatch,tmp_path,reject_index=3)
    monkeypatch.setattr(w.os,'getpid',lambda:100);w.run(p,'prepare',out)
    monkeypatch.setattr(w.os,'getpid',lambda:101);w.run(p,'replay',out)
    result=w.run(p,'train',out);state,meta=w.load_training_state(out/'checkpoint.pkl')
    assert result['status']=='incomplete_prefix_independently_audited'
    assert meta['absolute_completed_steps']==479 and int(state[1][0])==163
    assert calls['transition']==4 and calls['audit']==1


def test_full_transition_replay_detects_rng_change_and_uses_numeric_string_safe_metrics(tmp_path):
    p,parent,state=source_fixture(tmp_path);left=tmp_path/'left';right=tmp_path/'right';left.mkdir();right.mkdir()
    for directory in (left,right):
        for name in ('fit_products.npz','selection_rows.npz','guard_rows.npz'):np.savez(directory/name,rows=np.asarray(state[3]))
    report=dict(accepted=True,selected_index=1,actual_native_factor=1.5,
        calibration=dict(reason='passed',guard_used_to_select_smaller_fraction=False,FS=.001))
    assert w.compare_transition(state,state,report,report,left,right)['status']=='passed'
    wrong=(*state[:2],state[2]+1,state[3])
    with pytest.raises(ArithmeticError):w.compare_transition(state,wrong,report,report,left,right)


@pytest.mark.parametrize('bad_guard',[False,True])
def test_actual_finite_step_uses_all5_nonlinear_normalized_states_once(monkeypatch,tmp_path,bad_guard):
    p=plan();parent={'x':jnp.array([1.,0.])};moments=(3,{'x':jnp.ones(2)},{'x':jnp.ones(2)})
    key=jax.random.PRNGKey(7);chains=jnp.zeros((256,36),jnp.int32);state=(parent,moments,key,chains)
    rows=np.zeros((4096,36),np.int32);rows[:,0]=np.arange(4096)%2
    def amplitude(p,rows):
        factor=1+2*np.asarray(rows)[:,-1]
        return np.asarray(p['x'])[0]+np.asarray(p['x'])[1]*(2*np.asarray(rows)[:,0]-1)*factor
    calls=[]
    def draw(banks,rt,count,seed,**kwargs):
        calls.append(seed);assert len(banks)==5
        result=rows[:count].copy()
        if bad_guard and seed==p['step_seeds'][0]['guard']:result[:,-1]=1
        return result,dict(latent=np.zeros((count,4),int),Born_replay=np.zeros(count))
    def normalize(p,rt):
        norm=float(np.vdot(p['x'],p['x']).real)
        return jax.tree.map(lambda x:x/np.sqrt(norm),p),dict(original_raw_norm=norm)
    def update(p,m,x,energy,rate,**kwargs):
        assert rate==.03 and kwargs['warm'] and len(x)==4096
        return {'x':p['x']+jnp.array([0.,.00055])},(m[0]+1,*m[1:]),dict(accepted=jnp.array(True))
    rt=dict(cfg=None,prepare=lambda p:p,next_sampler=lambda p,k,c:(k+1,c,jnp.asarray(rows)),
        action=lambda p,x:(amplitude(p,x),-36*amplitude(p,x)),native_update=update,
        orbit_query=lambda p,x:np.stack((amplitude(p,x),np.ones(len(x))),axis=1))
    monkeypatch.setattr(w.diagnostic,'_device_peak',lambda:0)
    monkeypatch.setattr(w.diagnostic,'normalized_raw_bank',normalize)
    monkeypatch.setattr(w.diagnostic,'draw_orbit_rows',draw)
    monkeypatch.setattr(w.diagnostic,'prepare_feature_lut_bank',lambda b,c:b)
    monkeypatch.setattr(w.diagnostic,'native_projected_amplitude',lambda rt,p,x:amplitude(p,x))
    actual,report=w.finite_step(p,rt,state,1,tmp_path/'step')
    assert calls==[p['step_seeds'][0]['selection'],p['step_seeds'][0]['guard']]
    assert report['calibration']['proposed_index']==1
    assert not report['calibration']['guard_used_to_select_smaller_fraction']
    assert report['work']['calibration_bank_centers']==20480 and report['work']['fit_H_centers']==4096
    assert len(list((tmp_path/'step').glob('finite_native_*.pkl')))==5
    if bad_guard:
        assert not report['accepted'] and pytree_sha256(actual)==pytree_sha256(state)
    else:
        assert report['accepted'] and report['selected_index']==1 and report['actual_native_factor']==1.5
        np.testing.assert_allclose(actual[0]['x'],[1.,.000825],rtol=0,atol=1e-18)
        assert int(actual[1][0])==4 and report['no_post_IID_training_metric_shrink']
        assert max(report['native_replay_relative_errors'])<3e-16


def test_final_audit_mixture_retains_node_H_products_and_fixed16384_records(monkeypatch,tmp_path):
    p=plan();p['H_chunk']=1024
    rows=np.zeros((16384,36),np.int32);rows[:,0]=np.arange(16384)%4
    calls=[]
    def draw(banks,rt,count,seed,*,bridge):
        assert len(banks)==2 and count==16384 and seed==2001 and bridge
        return rows,dict(latent=np.zeros((count,4),int),Born_replay=np.zeros(count))
    def bridge(bank,bits):
        calls.append(len(bits));a=np.array([0.,1.,2.,1.])[bits[:,0]]
        h=np.array([.2,-1.,-1.5,-.8])[bits[:,0]]
        shift=bank['shift'];return a+shift,h+.3*shift,np.ones(len(bits))*(1+shift),np.ones(len(bits))*(2+shift)
    monkeypatch.setattr(w.diagnostic,'draw_orbit_rows',draw)
    monkeypatch.setattr(w.diagnostic,'normalized_raw_bank',lambda native,rt:(native,dict(original_raw_norm=1.)))
    monkeypatch.setattr(w.diagnostic,'prepare_feature_lut_bank',lambda bank,cfg:bank)
    monkeypatch.setattr(w.diagnostic,'_device_peak',lambda:0)
    parent=dict(shift=0.);candidate=dict(shift=.1)
    report=w.final_audit(p,dict(cfg=None,bridge=bridge),parent,(candidate,None,None,None),tmp_path/'audit',lambda:50000.)
    assert report['status']=='complete' and report['records']==16384 and sum(calls)==32768
    with np.load(tmp_path/'audit/products.npz') as data:
        np.testing.assert_allclose(data['proposal'],1.55,rtol=0,atol=2e-16)
        assert np.count_nonzero(data['psi'][:,0]==0)==4096
    assert report['moments']['exact_target_node_records'][0]==4096
    assert report['moments']['target_node_H2_contributions'][0]>0
