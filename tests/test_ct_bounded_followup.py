"""Bounded continuation controls; synthetic fixtures are not physics results."""
from copy import deepcopy
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct_bounded_followup as w
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint


def plan(kind='ct6'):
    p=dict(task='CT_bounded_followup_v1',kind=kind,execution_authorized=True,
        epsilon=.125,automatic_next_allocation=False,learning_rate=.03,damping=.05,no_requeue=True,
        j2=0.,seeds=dict(preflight=101,interim=102,interim_statistics=103,audit=104,audit_statistics=105),
        projection_chunk=4,action_chunk=4,score_chunk=1,H_chunk=4)
    if kind=='ct6':p.update(device='A100',steps=8,fit_records=5120,calibration_records=4096,
        calibration_split=2048,audit_records=8192,audit_groups=128,interim_records=2048,interim_groups=128,
        wall_cap_seconds=43200,maximum_peak_device_bytes=70*1024**3,
        model=dict(length=6,multiplicity_cap=28,maximum_q=10),arity='ternary_joint')
    else:p.update(device='B200',damping=.5,steps=4,fit_records=1024,calibration_records=1024,
        calibration_split=512,audit_records=1024,audit_groups=16,interim_records=0,interim_groups=0,
        wall_cap_seconds=10800,maximum_peak_device_bytes=136*1024**3,
        model=dict(length=12,multiplicity_cap=20,maximum_q=10))
    return p


@pytest.mark.parametrize('kind',['ct6','ct12'])
def test_only_explicit_fixed_panels_and_safe_bounds(kind):
    p=plan(kind);w.validate_plan(p)
    for name,bad in [('fit_records',32),('damping',.01),('steps',128),('no_requeue',False),
                     ('automatic_next_allocation',True),('projection_chunk',False)]:
        with pytest.raises(ValueError):w.validate_plan(dict(p,**{name:bad}))
    with pytest.raises(PermissionError):w.validate_plan(dict(p,execution_authorized=False))
    with pytest.raises(ValueError,match='silently enable adaptive'):
        w.validate_plan(dict(p,adaptive_sr=dict(enabled=True)))


def test_restore_inherits_trained_optimizer_and_exact_unused_rng(tmp_path):
    rng=np.random.default_rng(78).bit_generator.state
    s=({'w':jnp.array([1.2,.4])},(19,{'w':jnp.array([.1,.2])},{'w':jnp.array([.3,.4])}),rng)
    w.save_state(tmp_path/'ckpt.pkl',s,3,461)
    actual,meta=w.load_state(tmp_path/'ckpt.pkl')
    assert meta['completed_updates']==3 and meta['absolute_completed_steps']==461
    assert pytree_sha256(actual[:2])==pytree_sha256(s[:2])
    assert actual[2]==rng
    r1=np.random.default_rng();r1.bit_generator.state=rng
    r2=np.random.default_rng();r2.bit_generator.state=actual[2]
    np.testing.assert_array_equal(r1.integers(0,2**32,16),r2.integers(0,2**32,16))


def test_safety_never_calls_unresolved_change_a_win_or_requires_improvement():
    m=dict(candidate_minus_control=[1e-7,0,-1e-5,0],paired_iid_delta_SE=[1e-6,0,1e-6,0])
    assert w.safe_to_continue(dict(moments=m))
    assert not w.safe_to_continue(dict(m, candidate_minus_control=[3e-6,0,0,0]))
    assert not w.safe_to_continue(dict(m, candidate_minus_control=[-3e-6,0,3e-6,0]))
    with pytest.raises(ValueError):w.safe_to_continue(dict(m,paired_iid_delta_SE=[np.nan,0,0,0]))


def test_allocator_preflight_uses_live_storage_pool_not_nominal_gpu_name(monkeypatch):
    p=plan();device=type('Device',(),{'memory_stats':lambda s:dict(bytes_limit=60*1024**3)})()
    monkeypatch.setattr(w.jax,'local_devices',lambda:[device])
    with pytest.raises(MemoryError):w.check_allocator(p,1535560)
    device.memory_stats=lambda:dict(bytes_limit=72*1024**3)
    r=w.check_allocator(p,1535560)
    assert r['jacobian_bytes']==5120*1535560*8
    assert r['necessary_check_only_not_peak_guarantee']
    # A larger pool is not permission to exceed the unchanged70GiB live cap.
    with pytest.raises(MemoryError):w.check_allocator(p,2018920)


def test_recovered_fit_used_only_for_first_prepare_and_process_replay(monkeypatch,tmp_path):
    p=dict(plan(),first_fit_cache=dict(directory='sealed_failed_fit'))
    state=({},(),dict(next_step=3));calls=[]
    monkeypatch.setattr(w.c6,'finite_step',lambda *a,**k:calls.append(k['replay_fit_from']))
    for name in ('step1','step1_replay','step2'):w.transition(p,{}, {},state,tmp_path/name)
    assert calls==[p['first_fit_cache'],p['first_fit_cache'],None]


@pytest.mark.parametrize('arity',['binary_truncated','ternary_joint'])
def test_CT6_parent_replay_prepares_actual_folded_N36_bank_only_once(monkeypatch,tmp_path,arity):
    """Real N36 native tensors/features; cached toy H products, no H query.

    The bug was a second binary_to_joint conversion of the normalized bank,
    which has no ``six`` key. This test uses the real folding, canonical
    norm and LUT preparation; only the expensive H/guide products are fixed.
    """
    from su2_nqs.models.triangular_arity_clebsch import init_arity_pair,arity_clebsch_bank
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,init_mixed_clebsch_family
    from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
    cfg=MixedClebschConfig(length=6,multiplicity_cap=9,maximum_q=3)
    initial=init_mixed_clebsch_family(jax.random.PRNGKey(28801),cfg)
    family,_=init_arity_pair(initial['ternary_free'],cfg,8)
    native=family[arity]
    rt=dict(cfg=cfg,bank=lambda params:arity_clebsch_bank(params,cfg,arity,8))
    normalized,_=w.c6.diagnostic.normalized_raw_bank(native,rt)
    assert set(normalized)=={'local','top'}
    expected=prepare_feature_lut_bank(normalized,cfg)
    original_prepare=prepare_feature_lut_bank;preparations=[];queries=[]
    def prepare(bank,config):
        assert config==cfg and set(bank)=={'local','top'}
        preparations.append(True)
        return original_prepare(bank,config)
    monkeypatch.setattr(w.c6.diagnostic,'prepare_feature_lut_bank',prepare)
    def forbidden(_):raise AssertionError('normalized joint bank must not enter native/six preparation again')
    rt['prepare']=forbidden
    def fixed_products(prepared,rows):
        assert rows.shape==(2,36)
        for actual,golden in zip(jax.tree.leaves(prepared),jax.tree.leaves(expected),strict=True):
            np.testing.assert_allclose(actual,golden,rtol=2e-13,atol=2e-15)
        queries.append(True)
        return tuple(np.full(len(rows),x) for x in (1.,-.3,.05,.06))
    rt['bridge']=fixed_products
    snapshot=tmp_path/'parent_snapshot';snapshot.mkdir()
    rows=np.tile(np.r_[np.zeros(18,int),np.ones(18,int)],(2,1))
    np.savez(snapshot/'products.npz',rows=rows,psi=np.ones((2,2)),hpsi=np.full((2,2),-.3),
        guide_components=np.tile([.05,.06],(2,2,1)))
    report=w.replay_parent(dict(plan(),arity=arity),rt,(native,(),{}),tmp_path)
    assert preparations==[True] and queries==[True,True]
    assert report['component_relative_errors']==[0.,0.,0.,0.]
    assert report['complete_A1_images']==432


def test_CT12_replay_passes_canonical_bank_to_H_without_native_preparation(monkeypatch,tmp_path):
    p=plan('ct12');bank={'local':(),'top':()};calls=[]
    monkeypatch.setattr(w,'normalize',lambda *args:bank)
    def forbidden(_):raise AssertionError('CT12 H owns prepared-feature conversion; no unused native prepare')
    def fixed_H(actual,rows):
        assert actual is bank and rows.shape==(2,144);calls.append(True)
        return tuple(np.full(len(rows),x) for x in (1.,-.3,.05,.06))
    rt=dict(prepare=forbidden,H=fixed_H)
    snapshot=tmp_path/'parent_snapshot';snapshot.mkdir()
    rows=np.tile(np.r_[np.zeros(72,int),np.ones(72,int)],(2,1))
    np.savez(snapshot/'products.npz',rows=rows,products=np.tile([1.,-.3,.05,.06],(2,2,1)))
    report=w.replay_parent(p,rt,({},(),{}),tmp_path)
    assert calls==[True,True] and report['complete_A1_images']==1728


def completed_parent(tmp_path,*,parent_format='CT_bounded_followup_v1',j2=0.):
    """Actual checkpoint serializers and audits, synthetic small parameters."""
    src=tmp_path/'finished';src.mkdir();(src/'terminal_audit').mkdir()
    p=plan();p.update(j2=j2,arity='ternary_joint' if j2==0 else 'binary_truncated')
    base=deepcopy(p);base['seeds']={key:201+i for i,key in enumerate(base['seeds'])}
    rng=dict(generator=np.random.default_rng(7001).bit_generator.state,next_step=11,
        consumed_role_seeds=list(range(301,331)),excluded_seeds=list(range(401,406)))
    state=({'w':jnp.array([1.3,.25])},(29,{'w':jnp.array([.11,.22])},{'w':jnp.array([.33,.44])}),rng)
    if parent_format=='CT_bounded_followup_v1':
        w.save_state(src/'checkpoint.pkl',state,8,466)
        registration=dict(plan=base,parent_global_step=458,native_origin=456)
        result=dict(status='complete_fixed_prefix_independent_audit',completed_updates=8,
            parameter_sha256=pytree_sha256(state[0]),checkpoint_sha256=file_sha256(src/'checkpoint.pkl'))
        atomic_json(src/'endpoint_frozen.json',dict(status='fixed_prefix_frozen',completed_updates=8,
            absolute_step=466,parameter_sha256=result['parameter_sha256'],checkpoint_sha256=result['checkpoint_sha256']))
        atomic_json(src/'audit_result.json',result)
        p['parent_format']=parent_format
    else:
        rng['next_step']=3;rng['consumed_role_seeds']=list(range(301,307))
        w.c6.save_training_state(src/'checkpoint.pkl',state,dict(completed_updates=2,
            absolute_completed_steps=458,original_parent_global_step=456))
        base['j2_over_j1']=base.pop('j2')
        registration=dict(plan=base)
        result=dict(status='fixed2_updates_complete_independent_audit',completed_updates=2,
            params_sha256=pytree_sha256(state[0]),checkpoint_sha256=file_sha256(src/'checkpoint.pkl'))
        atomic_json(src/'results.json',result)
    atomic_json(src/'registration.json',registration)
    audit=dict(status='complete',endpoint_params_sha256=pytree_sha256(state[0]),
        moments=dict(candidate_minus_control=[-1e-5,0,-1e-3,0],paired_iid_delta_SE=[2e-6,0,1e-4,0]))
    atomic_json(src/'terminal_audit/audit.json',audit)
    np.savez(src/'terminal_audit/products.npz',rows=np.zeros((2,36),np.int32),
        psi=np.ones((2,2)),hpsi=np.full((2,2),-.3),guide_components=np.ones((2,2,2)))
    # A completed parent's fit is deliberately not a restart artifact.
    (src/'step1').mkdir();np.savez(src/'step1/fit_products.npz',rows=np.ones((2,36)))
    p.update(parent_directory=str(src),parent_registration_sha256=file_sha256(src/'registration.json'),
        parent_checkpoint_sha256=file_sha256(src/'checkpoint.pkl'),
        parent_audit_sha256=file_sha256(src/'terminal_audit/audit.json'))
    return p,src,state


@pytest.mark.parametrize('parent_format,j2',[('CT_bounded_followup_v1',0.),('CT_bounded_followup_v1',.125),('native',0.)])
def test_completed_native_or_followup_parent_keeps_parameters_optimizer_rng_and_origin(tmp_path,parent_format,j2):
    p,src,state=completed_parent(tmp_path,parent_format=parent_format,j2=j2)
    out=tmp_path/'next';out.mkdir();record=w.bind_parent(p,out)
    restored,metadata=w.load_state(out/'parent.pkl')
    assert pytree_sha256(restored[:2])==pytree_sha256(state[:2]) and restored[2]==state[2]
    absolute=466 if parent_format=='CT_bounded_followup_v1' else 458
    assert record['parent_global_step']==metadata['absolute_completed_steps']==absolute
    assert record['native_origin']==456 and metadata['completed_updates']==0
    assert record['original_parameter_sha256']==pytree_sha256(state[0])
    assert record['original_optimizer_sha256']==pytree_sha256(state[1])
    assert record['original_rng_sha256']==canonical_sha256(state[2])
    expected={'checkpoint.pkl','registration.json','audit.json','products.npz','training_result.json'}
    if parent_format=='CT_bounded_followup_v1':expected.add('endpoint_frozen.json')
    assert set(record['parent_seals'])==expected
    assert not (out/'parent_snapshot/step1').exists() and not p.get('first_fit_cache')
    roles,_=w.c6.role_seeds(deepcopy(restored[2]),restored[2]['next_step'])
    old=set(state[2]['consumed_role_seeds'])|set(state[2]['excluded_seeds'])
    assert not set(roles.values())&old and not set(p['seeds'].values())&old


@pytest.mark.parametrize('damage',['incomplete','registration_SHA','checkpoint_SHA','audit_SHA',
    'coupling','bad_parameter_hash','bad_optimizer_hash','bad_rng_hash','bad_rng_cursor',
    'frozen_SHA','frozen_step','reused_fit_seed','reused_audit_seed','future_fit_seed'])
def test_followup_parent_rejects_incomplete_changed_or_reused_evidence(tmp_path,damage):
    p,src,state=completed_parent(tmp_path)
    if damage=='incomplete':
        result=json.loads((src/'audit_result.json').read_text());result['status']='fixed_prefix_frozen'
        atomic_json(src/'audit_result.json',result)
    elif damage=='registration_SHA':p['parent_registration_sha256']='0'*64
    elif damage=='checkpoint_SHA':p['parent_checkpoint_sha256']='0'*64
    elif damage=='audit_SHA':p['parent_audit_sha256']='0'*64
    elif damage=='coupling':p.update(j2=.125,arity='binary_truncated')
    elif damage in ('bad_parameter_hash','bad_optimizer_hash','bad_rng_hash','bad_rng_cursor'):
        if damage!='bad_rng_cursor':
            saved=load_checkpoint(src/'checkpoint.pkl')
            if damage=='bad_parameter_hash':saved['params']['w']=saved['params']['w']+1.
            elif damage=='bad_optimizer_hash':saved['optimizer_state']=(999,*saved['optimizer_state'][1:])
            else:saved['metadata']['rng']['next_step']+=1
            save_checkpoint(src/'checkpoint.pkl',saved['params'],optimizer_state=saved['optimizer_state'],metadata=saved['metadata'])
        else:
            bad=deepcopy(state);bad[2]['next_step']+=1
            w.save_state(src/'checkpoint.pkl',bad,8,466)
        # Re-seal external byte identities: inner RNG/hash and the absolute
        # origin relation must still reject, not merely the outer file SHA.
        digest=file_sha256(src/'checkpoint.pkl');p['parent_checkpoint_sha256']=digest
        for name in ('audit_result.json','endpoint_frozen.json'):
            value=json.loads((src/name).read_text());value['checkpoint_sha256']=digest;atomic_json(src/name,value)
    elif damage in ('frozen_SHA','frozen_step'):
        value=json.loads((src/'endpoint_frozen.json').read_text())
        value['checkpoint_sha256' if damage=='frozen_SHA' else 'absolute_step']='0'*64 if damage=='frozen_SHA' else 465
        atomic_json(src/'endpoint_frozen.json',value)
    elif damage=='reused_fit_seed':p['seeds']['audit']=state[2]['consumed_role_seeds'][0]
    elif damage=='reused_audit_seed':p['seeds']['audit']=json.loads((src/'registration.json').read_text())['plan']['seeds']['audit']
    else:
        roles,_=w.c6.role_seeds(deepcopy(state[2]),state[2]['next_step']);p['seeds']['audit']=roles['fit']
    out=tmp_path/'next';out.mkdir()
    with pytest.raises(ValueError):w.bind_parent(p,out)
    assert not (out/'parent.pkl').exists()


def test_kernel_separates_comparison_parent_from_native_absolute_origin(monkeypatch):
    p=plan();params={'w':jnp.array([1.])};rng=dict(next_step=11)
    state=(params,(29,{},{}),rng);captured={}
    def runtime(kernel,template):
        captured.update(kernel)
        return dict(prepare=lambda a:a,projected=lambda a,x:a['w'][0])
    monkeypatch.setattr(w.c6,'make_runtime',runtime)
    monkeypatch.setattr(w,'make_importance_amplitude_sr',lambda *a,**k:None)
    _,kernel=w.make_runtime(p,state,dict(native_origin=456,parent_global_step=466))
    assert kernel['parent']['global_step']==456
    assert kernel['comparison_parent_global_step']==captured['comparison_parent_global_step']==466


def production_plan():
    p=plan('ct12');p.update(task='CT12_production32_v1',j2=.125,steps=32,
        audit_records=4096,audit_groups=64,wall_cap_seconds=57600,
        parent_format='CT_bounded_followup_v1',active_profile=[{'0':8},{'0':1}],
        software_acceptance=dict(directory='/sealed/software',plan_path='/sealed/software/plan.json',
            plan_sha256='a'*64,source_manifest_sha256='b'*64))
    return p


def test_production32_keeps_architecture_statistics_and_update_rule_closed():
    p=production_plan();w.validate_plan(p)
    for key,value in [('j2',0.),('kind','ct6'),('fit_records',2048),('damping',.05),
                      ('steps',16),('audit_records',1024),('wall_cap_seconds',86400)]:
        with pytest.raises(ValueError):w.validate_plan(dict(p,**{key:value}))
    with pytest.raises(ValueError):w.validate_plan(dict(p,software_acceptance={}))
    with pytest.raises(ValueError):w.validate_plan(dict(p,parent_format=None))


def completed_CT12_parent(tmp_path):
    """Actual serializers, synthetic small parameters; no physics benchmark."""
    p=production_plan();src=tmp_path/'ct12_finished';src.mkdir();(src/'terminal_audit').mkdir()
    base=dict(plan('ct12'),j2=.125,active_profile=deepcopy(p['active_profile']))
    state=({'w':jnp.array([1.3,.25])},(5,{'w':jnp.array([.11,.22])},{'w':jnp.array([.33,.44])}),
           np.random.default_rng(2810277).bit_generator.state)
    w.save_state(src/'checkpoint.pkl',state,3,5)
    atomic_json(src/'registration.json',dict(plan=base,parent_global_step=2,native_origin=0))
    audit=dict(status='complete',records=1024,independent_from_training_and_geometry=True,
        endpoint_params_sha256=pytree_sha256(state[0]),
        moments=dict(candidate_minus_control=[-1e-5,0,-1e-3,0],paired_iid_delta_SE=[2e-6,0,1e-4,0]))
    atomic_json(src/'terminal_audit/audit.json',audit)
    result=dict(status='complete_fixed_prefix_independent_audit',completed_updates=3,
        parameter_sha256=pytree_sha256(state[0]),checkpoint_sha256=file_sha256(src/'checkpoint.pkl'),terminal_audit=audit)
    atomic_json(src/'audit_result.json',result)
    atomic_json(src/'endpoint_frozen.json',dict(status='fixed_prefix_frozen',completed_updates=3,absolute_step=5,
        parameter_sha256=result['parameter_sha256'],checkpoint_sha256=result['checkpoint_sha256']))
    np.savez(src/'terminal_audit/untouched_products.npz',rows=np.zeros((2,144),np.int32),products=np.ones((2,2,4)))
    p.update(parent_directory=str(src),parent_registration_sha256=file_sha256(src/'registration.json'),
        parent_checkpoint_sha256=file_sha256(src/'checkpoint.pkl'),parent_audit_sha256=file_sha256(src/'audit_result.json'))
    return p,src,state


def test_CT12_production_restarts_latest_bounded_parent_with_exact_next_rng(tmp_path):
    p,src,state=completed_CT12_parent(tmp_path)
    # New audit streams must not be the old parent's five role seeds.
    p['seeds']={key:5101+i for i,key in enumerate(p['seeds'])}
    out=tmp_path/'new';out.mkdir();record=w.bind_parent(p,out)
    actual,meta=w.load_state(out/'parent.pkl')
    assert meta['absolute_completed_steps']==record['parent_global_step']==5
    assert meta['completed_updates']==0 and record['native_origin']==0
    assert pytree_sha256(actual[:2])==pytree_sha256(state[:2]) and actual[2]==state[2]
    a,next_a=w.c12._seeds(actual[2]);b,next_b=w.c12._seeds(state[2])
    assert a==b and next_a==next_b
    assert set(record['parent_seals'])=={'checkpoint.pkl','registration.json','audit.json','products.npz',
                                        'training_result.json','endpoint_frozen.json'}


@pytest.mark.parametrize('damage',['profile','incomplete','counter','endpoint','audit_reused_seed','future_seed'])
def test_CT12_bounded_parent_rejects_lost_history_or_wrong_support(tmp_path,damage):
    p,src,state=completed_CT12_parent(tmp_path)
    p['seeds']={key:6101+i for i,key in enumerate(p['seeds'])}
    if damage=='profile':p['active_profile'][0]['0']=16
    elif damage=='incomplete':
        r=json.loads((src/'audit_result.json').read_text());r['status']='started';atomic_json(src/'audit_result.json',r)
        p['parent_audit_sha256']=file_sha256(src/'audit_result.json')
    elif damage=='counter':
        changed=(state[0],(4,*state[1][1:]),state[2]);w.save_state(src/'checkpoint.pkl',changed,3,5)
        p['parent_checkpoint_sha256']=file_sha256(src/'checkpoint.pkl')
    elif damage=='endpoint':
        r=json.loads((src/'endpoint_frozen.json').read_text());r['absolute_step']=4
        atomic_json(src/'endpoint_frozen.json',r)
    elif damage=='audit_reused_seed':p['seeds']['audit']=101
    else:p['seeds']['audit']=w.c12._seeds(state[2])[0][0]
    out=tmp_path/'new';out.mkdir()
    with pytest.raises(ValueError):w.bind_parent(p,out)
    assert not (out/'parent.pkl').exists()


def test_production_requires_exact_passed_runtime_certificate(tmp_path):
    p=production_plan();root=tmp_path/'runtime';(root/'gpu').mkdir(parents=True)
    p['software_acceptance'].update(directory=str(root),source_manifest_sha256=canonical_sha256({'code':'sealed'}))
    path=root/'gpu/gpu_check_result.json'
    certificate=dict(status='bounded_software_acceptance_complete',passed=True,plan_sha256='a'*64,
                     source_manifest={'code':'sealed'})
    atomic_json(path,certificate)
    assert w.require_software_acceptance(p)['certificate_sha256']==file_sha256(path)
    for key,value in [('passed',False),('status','acceptance_failed_or_incomplete'),('plan_sha256','c'*64),
                      ('source_manifest',{'code':'changed'})]:
        atomic_json(path,dict(certificate,**{key:value}))
        with pytest.raises(ValueError):w.require_software_acceptance(p)


@pytest.mark.parametrize('kind,interim_timeout',[('ct6',False),('ct6',True),('ct12',False),('production',False)])
def test_four_processes_commit_first_transition_once_and_freeze_audit(monkeypatch,tmp_path,kind,interim_timeout):
    p=production_plan() if kind=='production' else plan(kind)
    path=tmp_path/'plan.json';atomic_json(path,p);out=tmp_path/'run'
    monkeypatch.setattr(w,'require_software_acceptance',lambda p:None)
    monkeypatch.setattr(w,'verify_production_runtime',lambda p:None)
    monkeypatch.setattr(w.jax,'default_backend',lambda:'gpu')
    monkeypatch.setattr(w.jax,'devices',lambda:[type('Device',(),dict(device_kind=p['device']))()])
    monkeypatch.setattr(w,'transitive_local_source_manifest',lambda *a,**k:{})
    s=({'w':jnp.array([1.,.2])},(19,{'w':jnp.array([.1,.2])},{'w':jnp.array([.3,.4])}),dict(cursor=0))
    def bind(plan,out):
        w.save_state(out/'parent.pkl',s,0,458)
        return dict(parent_global_step=458,native_origin=456,parent_seals={})
    monkeypatch.setattr(w,'bind_parent',bind)
    monkeypatch.setattr(w,'make_runtime',lambda *a:({},{}))
    monkeypatch.setattr(w,'replay_parent',lambda *a:dict(warm_H_seconds=.00001))
    calls=[]
    def step(p,k,rt,state,path,**kwargs):
        calls.append((path.name,kwargs.get('replay',False)))
        new=(jax.tree.map(lambda x:x+.0001,state[0]),(state[1][0]+1,*state[1][1:]),dict(cursor=state[2]['cursor']+1))
        return new,dict(accepted=True,wall_seconds=.001)
    monkeypatch.setattr(w,'transition',step)
    audits=[]
    def audit(p,k,rt,parent,final,path,remaining,interim=False):
        audits.append((interim,int(final[1][0]),pytree_sha256(parent[0])))
        if interim and interim_timeout:raise TimeoutError('bounded synthetic interim interruption')
        return dict(moments=dict(candidate_minus_control=[-1e-6,0,0,0],paired_iid_delta_SE=[1e-5,0,1e-5,0]))
    monkeypatch.setattr(w,'audit_pair',audit)
    w.run(path,out,'prepare');assert not (out/'checkpoint.pkl').exists()
    w.run(path,out,'replay');assert w.load_state(out/'checkpoint.pkl')[1]['completed_updates']==1
    w.run(path,out,'replay');assert len(calls)==2
    updates=2 if interim_timeout else p['steps']
    result=w.run(path,out,'train');assert result['completed_updates']==updates
    final,meta=w.load_state(out/'checkpoint.pkl')
    assert int(final[1][0])==19+updates and final[2]['cursor']==updates
    assert meta['absolute_completed_steps']==458+updates
    assert w.run(path,out,'audit')['status']=='complete_fixed_prefix_independent_audit'
    assert audits[-1][0] is False
    assert len(audits)==(2 if kind=='ct6' else 1)
    assert len(calls)==updates+1  # replay is not an additional update
    changed=deepcopy(p);changed['j2']=.125 if p['j2']==0. else 0.;atomic_json(path,changed)
    with pytest.raises(ValueError):w.run(path,out,'audit')
