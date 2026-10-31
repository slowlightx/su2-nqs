"""Native continuation seals/counts; exact SR math remains independently tested."""
import json
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_n144_continue import (
    checked_checkpoint, checked_parent, checked_training_release,
    checked_saved_pilot, phase_has_room, remaining_work_forecast,
    RESUME_PROFILE, COMPLETED_PROFILE, SAME_WIDTH_PROFILE, initialize_completed_parent, added_coordinate_mask,
    validate_committed_update, validate_plan,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

jax.config.update('jax_enable_x64', True)
PLAN = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote/continuation_20261019/ct12/j0_binary.json'


@pytest.mark.parametrize('name', ['j0_binary', 'j0_joint', 'j0125_binary', 'j0125_joint'])
def test_four_bounded_native_plans(name):
    spec = json.loads((PLAN.parent/(name+'.json')).read_text())
    validate_plan(spec)
    assert sum(spec['phase_seconds'].values()) == 8*3600


@pytest.mark.parametrize('change', ['release', 'width', 'arity', 'steps', 'records', 'budget', 'seed', 'count_bool', 'tolerance'])
def test_release_refuses_scope_or_budget_drift(change):
    spec = json.loads(PLAN.read_text())
    if change == 'release': spec['training_release'] = 1
    if change == 'width': spec['model']['multiplicity_cap'] = 20
    if change == 'arity': spec['arm'] = 'ternary_free'
    if change == 'steps': spec['training']['steps'] = 9
    if change == 'records': spec['training']['records_per_chain'] = 32
    if change == 'budget': spec['phase_seconds']['train'] += 1
    if change == 'seed': spec['audit']['seed'] = -1
    if change == 'count_bool': spec['training']['chains'] = True
    if change == 'tolerance': spec['tolerance'] = 1e-4
    with pytest.raises(ValueError): validate_plan(spec)


def fixture_parent(tmp_path):
    spec = json.loads(PLAN.read_text())
    directory = tmp_path/'native'; arm = spec['arm']
    (directory/arm).mkdir(parents=True)
    cfg = {key: spec[key] for key in ('model', 'binary_rank', 'j2_over_j1')}
    cfg['seeds'] = dict(training=1, initial_audit=2, final_audits={arm: 3})
    registration = dict(config=cfg, sources={})
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(directory/'registration.json', registration)
    params = dict(local=jnp.asarray([.1, -.2]), top=jnp.asarray([.03]))
    optimizer = dict(completed_steps=jnp.asarray(2), accepted_updates=jnp.asarray(2),
        damping=jnp.asarray(.1))
    states = jnp.tile(jnp.asarray([0, 1]*72, jnp.int32), (8, 1)); key = jax.random.PRNGKey(4)
    metadata = dict(registration_sha256=registration['sha256'], arm=arm,
        params_sha256=pytree_sha256(params), optimizer_sha256=pytree_sha256(optimizer),
        sampler_rng_sha256=pytree_sha256((states, key)), sampler_states=states, rng=key,
        history=[{}, {}])
    checkpoint = directory/arm/'checkpoint.pkl'
    save_checkpoint(checkpoint, params, optimizer_state=optimizer, metadata=metadata)
    checkpoint_sha = file_sha256(checkpoint)
    raw_path = directory/arm/'audit_samples.npz'
    np.savez_compressed(raw_path, bits=np.asarray(states), psi=np.ones(8))
    audit = dict(params_sha256=pytree_sha256(params), registration_sha256=registration['sha256'],
        independent_of_training=True, raw_samples_sha256=file_sha256(raw_path))
    atomic_json(directory/arm/'audit.json', audit)
    atomic_json(directory/'results.json', dict(status='completed', arms={arm:dict(
        status='completed', completed_steps=2, accepted_updates=2, checkpoint_sha256=checkpoint_sha,
        terminal_audit=audit)}))
    spec['parent'] = dict(directory=str(directory), registration_sha256=file_sha256(directory/'registration.json'),
        results_sha256=file_sha256(directory/'results.json'), checkpoint_sha256=checkpoint_sha,
        params_sha256=pytree_sha256(params), audited_bank_sha256=pytree_sha256(params))
    return spec, params, optimizer, states, key


def test_checked_parent_preserves_native_optimizer_sampler_and_rng(tmp_path):
    spec, params, optimizer, states, key = fixture_parent(tmp_path)
    saved, _ = checked_parent(spec)
    assert pytree_sha256(saved['params']) == pytree_sha256(params)
    assert pytree_sha256(saved['optimizer_state']) == pytree_sha256(optimizer)
    np.testing.assert_array_equal(saved['metadata']['sampler_states'], states)
    np.testing.assert_array_equal(saved['metadata']['rng'], key)


@pytest.mark.parametrize('change', ['checkpoint', 'registration', 'results', 'params', 'wrong_arity', 'model', 'seed'])
def test_parent_source_and_family_changes_fail(tmp_path, change):
    spec, *_ = fixture_parent(tmp_path)
    if change in ('checkpoint', 'registration', 'results', 'params'):
        spec['parent'][change+'_sha256'] = '0'*64
    if change == 'wrong_arity': spec['arm'] = 'ternary_joint'
    if change == 'model': spec['model']['maximum_q'] = 12
    if change == 'seed': spec['audit']['seed'] = 1
    with pytest.raises((ValueError, KeyError, FileNotFoundError)): checked_parent(spec)


def test_continuation_checkpoint_tracks_global_and_new_steps(tmp_path):
    spec, params, optimizer, states, key = fixture_parent(tmp_path)
    optimizer = {**optimizer, 'completed_steps':jnp.asarray(4), 'accepted_updates':jnp.asarray(4)}
    meta = dict(registration_sha256='continuation', parent_checkpoint_sha256=spec['parent']['checkpoint_sha256'],
        params_sha256=pytree_sha256(params), optimizer_sha256=pytree_sha256(optimizer),
        sampler_rng_sha256=pytree_sha256((states, key)), sampler_states=states, rng=key,
        completed_new_steps=2, new_history=[{}, {}])
    path = tmp_path/'continuation.pkl'
    save_checkpoint(path, params, optimizer_state=optimizer, metadata=meta)
    checked_checkpoint(path, 'continuation', spec['parent']['checkpoint_sha256'])
    meta['completed_new_steps'] = 3
    save_checkpoint(path, params, optimizer_state=optimizer, metadata=meta)
    with pytest.raises(ValueError, match='counter'): checked_checkpoint(path, 'continuation', spec['parent']['checkpoint_sha256'])


def completed_endpoint_fixture(tmp_path):
    """Independent flat completed endpoint rather than the old two-update layout."""
    spec = json.loads(PLAN.read_text()); directory=tmp_path/'completed'; directory.mkdir()
    source_model=dict(length=12,multiplicity_cap=16,maximum_q=10)
    cfg=dict(model=source_model,arm=spec['arm'],binary_rank=8,j2_over_j1=0.,audit=dict(seed=41))
    registration=dict(config=cfg,sources={});registration['sha256']=canonical_sha256(registration)
    atomic_json(directory/'registration.json',registration)
    params=dict(w=jnp.array([.1,.2]));opt=dict(completed_steps=jnp.array(10),accepted_updates=jnp.array(10),damping=jnp.array(.1))
    states=jnp.tile(jnp.array([0,1]*72),(8,1));key=jax.random.PRNGKey(77)
    metadata=dict(registration_sha256=registration['sha256'],arm=spec['arm'],
        params_sha256=pytree_sha256(params),optimizer_sha256=pytree_sha256(opt),
        sampler_rng_sha256=pytree_sha256((states,key)),sampler_states=states,rng=key,
        new_history=[dict(step=k) for k in range(8)])
    save_checkpoint(directory/'checkpoint.pkl',params,optimizer_state=opt,metadata=metadata)
    np.savez_compressed(directory/'terminal_audit_samples.npz',bits=np.asarray(states),psi=np.ones(8))
    audit=dict(untouched=True,fixed_endpoint=True,params_sha256=pytree_sha256(params),
        checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'),
        raw_samples_sha256=file_sha256(directory/'terminal_audit_samples.npz'))
    atomic_json(directory/'terminal_audit.json',audit)
    atomic_json(directory/'results.json',dict(status='completed',global_optimizer_steps=10,terminal_audit=audit))
    spec.update(release_profile=COMPLETED_PROFILE,model=dict(length=12,multiplicity_cap=20,maximum_q=10),
        expansion_seed=57,parent=dict(directory=str(directory),model=source_model,arm=spec['arm'],j2_over_j1=0.,
            global_optimizer_steps=10,params_sha256=pytree_sha256(params),**{
                k+'_sha256':file_sha256(directory/f) for k,f in (
                    ('registration','registration.json'),('results','results.json'),('checkpoint','checkpoint.pkl'),
                    ('audit','terminal_audit.json'),('raw_samples','terminal_audit_samples.npz'))}))
    spec['training']['steps']=16;spec['training']['records_per_chain']=32
    return spec


def test_completed_native_endpoint_is_checked_and_wrong_identity_rejected(tmp_path):
    spec=completed_endpoint_fixture(tmp_path);validate_plan(spec)
    saved,_=checked_parent(spec)
    assert int(saved['optimizer_state']['completed_steps'])==10
    for field in ('params_sha256','checkpoint_sha256','raw_samples_sha256'):
        changed=deepcopy(spec);changed['parent'][field]='0'*64
        with pytest.raises(ValueError):checked_parent(changed)


@pytest.mark.parametrize('field', ['parent_arity','parent_model','base_count','seed_bool','steps','records','width','budget'])
def test_completed_parent_profile_rejects_unregistered_changes(tmp_path,field):
    spec=completed_endpoint_fixture(tmp_path)
    if field=='parent_arity':spec['parent']['arm']='ternary_joint'
    if field=='parent_model':spec['parent']['model']['multiplicity_cap']=18
    if field=='base_count':spec['parent']['global_optimizer_steps']=11
    if field=='seed_bool':spec['expansion_seed']=True
    if field=='steps':spec['training']['steps']=17
    if field=='records':spec['training']['records_per_chain']=16
    if field=='width':spec['model']['multiplicity_cap']=24
    if field=='budget':spec['phase_seconds']['train']=41*3600
    with pytest.raises(ValueError):validate_plan(spec)


def test_completed_stage_preserves_counters_and_restarts_only_new_history(tmp_path):
    spec=completed_endpoint_fixture(tmp_path);spec['model']=dict(spec['parent']['model'])
    saved,_=checked_parent(spec);stage,report=initialize_completed_parent(spec,saved)
    assert report['same_width']
    assert pytree_sha256(stage['params'])==pytree_sha256(saved['params'])
    assert stage['metadata']['new_history']==[] and len(stage['metadata']['parent_history'])==8
    np.testing.assert_array_equal(stage['metadata']['rng'],saved['metadata']['rng'])
    path=tmp_path/'next.pkl';meta=stage['metadata']
    meta.update(registration_sha256='next',parent_checkpoint_sha256=spec['parent']['checkpoint_sha256'],
        params_sha256=pytree_sha256(stage['params']),optimizer_sha256=pytree_sha256(stage['optimizer_state']),
        completed_new_steps=0)
    save_checkpoint(path,stage['params'],optimizer_state=stage['optimizer_state'],metadata=meta)
    checked_checkpoint(path,'next',spec['parent']['checkpoint_sha256'],optimizer_base_step=10)
    with pytest.raises(ValueError):checked_checkpoint(path,'next',spec['parent']['checkpoint_sha256'])


def same_width_endpoint_fixture(tmp_path):
    """A global18 flat endpoint with full optimizer diagnostics and history."""
    spec = completed_endpoint_fixture(tmp_path)
    spec.update(release_profile=SAME_WIDTH_PROFILE)
    spec.pop('expansion_seed')
    spec['training']['steps'] = 8
    spec['sr']['parameter_chunk_size'] = 8192
    spec['parent'].update(model=deepcopy(spec['model']), global_optimizer_steps=18)
    directory = Path(spec['parent']['directory'])
    registration = json.loads((directory/'registration.json').read_text())
    registration['config'].update(model=deepcopy(spec['model']), sr=deepcopy(spec['sr']),
        sampler=deepcopy(spec['sampler']), projection_chunk=spec['projection_chunk'],
        action_chunk=spec['action_chunk'], training=deepcopy(spec['training']))
    registration.pop('sha256')
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(directory/'registration.json', registration)
    saved = load_checkpoint(directory/'checkpoint.pkl')
    optimizer = dict(saved['optimizer_state'], completed_steps=jnp.asarray(18),
        accepted_updates=jnp.asarray(18), last_diagnostics=dict(probe=jnp.array([.125, .25])))
    metadata = dict(saved['metadata'], registration_sha256=registration['sha256'],
        optimizer_sha256=pytree_sha256(optimizer), inherited_history=[dict(step=k) for k in range(10)])
    save_checkpoint(directory/'checkpoint.pkl', saved['params'], optimizer_state=optimizer, metadata=metadata)
    audit = json.loads((directory/'terminal_audit.json').read_text())
    audit['checkpoint_sha256'] = file_sha256(directory/'checkpoint.pkl')
    atomic_json(directory/'terminal_audit.json', audit)
    atomic_json(directory/'results.json', dict(status='completed', global_optimizer_steps=18, terminal_audit=audit))
    for key, name in (('registration', 'registration.json'), ('results', 'results.json'),
                      ('checkpoint', 'checkpoint.pkl'), ('audit', 'terminal_audit.json')):
        spec['parent'][key+'_sha256'] = file_sha256(directory/name)
    return spec


def test_same_width_global18_fresh_initialization_preserves_full_checkpoint(tmp_path):
    """Exercise the actual fresh-run dispatch, not only an accepted config."""
    spec = same_width_endpoint_fixture(tmp_path)
    validate_plan(spec)
    saved, _ = checked_parent(spec)
    stage, note = initialize_completed_parent(spec, saved)
    assert note['same_width'] and note['exact_same_physical_state']
    assert not any(note[k] for k in ('optimizer_reset', 'sampler_reset', 'rng_reset', 'expansion_performed'))
    assert pytree_sha256(stage['params']) == pytree_sha256(saved['params'])
    assert pytree_sha256(stage['optimizer_state']) == pytree_sha256(saved['optimizer_state'])
    assert 'last_diagnostics' in stage['optimizer_state']
    assert stage['metadata']['optimizer_base_step'] == 18
    assert len(stage['metadata']['parent_history']) == 18
    assert not stage['metadata']['new_history']
    np.testing.assert_array_equal(stage['metadata']['sampler_states'], saved['metadata']['sampler_states'])
    np.testing.assert_array_equal(stage['metadata']['rng'], saved['metadata']['rng'])
    meta = dict(stage['metadata'], registration_sha256='new',
        parent_checkpoint_sha256=spec['parent']['checkpoint_sha256'],
        params_sha256=pytree_sha256(stage['params']), optimizer_sha256=pytree_sha256(stage['optimizer_state']))
    path = tmp_path/'new_checkpoint.pkl'
    save_checkpoint(path, stage['params'], optimizer_state=stage['optimizer_state'], metadata=meta)
    checked_checkpoint(path, 'new', spec['parent']['checkpoint_sha256'], optimizer_base_step=18)


@pytest.mark.parametrize('change', ['base', 'width', 'steps', 'seed', 'block', 'sr', 'sampler', 'execution', 'records'])
def test_same_width_profile_rejects_unregistered_changes(tmp_path, change):
    spec = same_width_endpoint_fixture(tmp_path)
    if change == 'base': spec['parent']['global_optimizer_steps'] = 10
    if change == 'width': spec['parent']['model']['multiplicity_cap'] = 16
    if change == 'steps': spec['training']['steps'] = 16
    if change == 'seed': spec['expansion_seed'] = 42
    if change == 'block': spec['sr']['parameter_chunk_size'] = 32768
    if change == 'sr': spec['sr']['step_size'] *= 2
    if change == 'sampler': spec['sampler']['lazy_probability'] = .25
    if change == 'execution': spec['projection_chunk'] *= 2
    if change == 'records': spec['training']['proposals_between'] *= 2
    with pytest.raises(ValueError):
        validate_plan(spec)
        checked_parent(spec)


@pytest.mark.parametrize('name', ['j0_binary_w20', 'j0_joint_w20', 'j0125_binary_w20', 'j0125_joint_w20'])
def test_relaunch_templates_are_bounded_and_cannot_accidentally_submit(name):
    directory = PLAN.parents[2]/'ct12_relaunch_20261022'
    spec = json.loads((directory/(name+'.json')).read_text())
    released = 'joint' in name
    assert spec['training_release'] is released
    assert spec['release_profile'] == SAME_WIDTH_PROFILE
    assert spec['training']['steps'] == 8 and spec['pilot_steps'] == 2
    assert spec['parent']['global_optimizer_steps'] == 18
    for key in ('params', 'registration', 'results', 'checkpoint', 'audit', 'raw_samples'):
        value = spec['parent'][key+'_sha256']
        if released:
            assert len(value) == 64 and all(c in '0123456789abcdef' for c in value)
        else:
            assert value.startswith('__ROOT_MUST_SEAL_')
    if not released:
        with pytest.raises(ValueError, match='release'): validate_plan(spec)
    spec['training_release'] = True
    validate_plan(spec)


@pytest.mark.parametrize('arm',['binary_truncated','ternary_joint'])
def test_live_width_inclusion_keeps_full_physical_amplitudes_and_ward_identity(arm):
    from su2_nqs.models.triangular_mixed_clebsch import (
        MixedClebschConfig,init_mixed_clebsch_family,make_mixed_clebsch_amplitude)
    from su2_nqs.models.triangular_arity_clebsch import init_arity_pair,arity_clebsch_bank
    source=MixedClebschConfig(length=6,multiplicity_cap=9,maximum_q=3)
    target=MixedClebschConfig(length=6,multiplicity_cap=12,maximum_q=3)
    raw=init_mixed_clebsch_family(jax.random.PRNGKey(11),source)['ternary_free']
    pair,_=init_arity_pair(raw,source,8);params=pair[arm]
    states=np.asarray([np.random.default_rng(k).permutation([0]*18+[1]*18) for k in range(4)],np.int32)
    saved=dict(params=params,optimizer_state=dict(completed_steps=jnp.array(10),accepted_updates=jnp.array(10),damping=jnp.array(.1)),
        metadata=dict(new_history=[],sampler_states=states,rng=jax.random.PRNGKey(8)))
    spec=dict(parent=dict(model=source.__dict__,global_optimizer_steps=10),model=target.__dict__,
        arm=arm,binary_rank=8,expansion_seed=89,sr=dict(damping=.1))
    grown,note=initialize_completed_parent(spec,saved)
    mask=added_coordinate_mask(source,target,grown['params'])
    assert sum(int(np.asarray(v).sum()) for v in jax.tree.leaves(mask)) == (
        sum(v.size for v in jax.tree.leaves(grown['params']))-sum(v.size for v in jax.tree.leaves(params)))
    oldfn=jax.jit(jax.vmap(make_mixed_clebsch_amplitude(source),in_axes=(None,0)))
    newfn=jax.jit(jax.vmap(make_mixed_clebsch_amplitude(target),in_axes=(None,0)))
    a=oldfn(arity_clebsch_bank(params,source,arm,8),jnp.asarray(states))
    bank=arity_clebsch_bank(grown['params'],target,arm,8)
    np.testing.assert_allclose(newfn(bank,states),a,atol=2e-16,rtol=2e-11)
    bra=np.random.default_rng(67).permutation([0]*17+[1]*19)
    columns=np.repeat(bra[None],19,axis=0);columns[np.arange(19),np.flatnonzero(bra)]=0
    ward=np.asarray(newfn(bank,columns));assert abs(ward.sum())/np.linalg.norm(ward)<2e-11


@pytest.mark.parametrize('change', ['wrong_bank', 'wrong_registration', 'not_independent', 'raw_samples'])
def test_parent_audit_identity_checked_not_only_file_hashes(tmp_path, change):
    spec, *_ = fixture_parent(tmp_path)
    directory = Path(spec['parent']['directory'])
    result = json.loads((directory/'results.json').read_text())
    audit = result['arms'][spec['arm']]['terminal_audit']
    if change == 'wrong_bank': audit['params_sha256'] = '0'*64
    if change == 'wrong_registration': audit['registration_sha256'] = '0'*64
    if change == 'not_independent': audit['independent_of_training'] = False
    if change == 'raw_samples': (directory/spec['arm']/'audit_samples.npz').write_bytes(b'altered raw data')
    atomic_json(directory/'results.json', result)
    atomic_json(directory/spec['arm']/'audit.json', audit)
    spec['parent']['results_sha256'] = file_sha256(directory/'results.json')
    with pytest.raises(ValueError, match='audited physical-bank'): checked_parent(spec)


@pytest.mark.parametrize('change', ['none', 'wrong_replay_cp', 'wrong_fixture', 'source', 'cp', 'ancestor', 'nonfinite', 'discrete'])
def test_training_binds_full_replay_to_pilot_and_descendants(tmp_path, change):
    checkpoint = tmp_path/'checkpoint.pkl'; checkpoint.write_bytes(b'pilot-checkpoint')
    fixture = tmp_path/'next_transition_fixture.pkl'; fixture.write_bytes(b'full-next-native-transition')
    pilot = dict(status='passed', checkpoint_sha256=file_sha256(checkpoint),
        fixture_sha256=file_sha256(fixture), registration_sha256='registered')
    leaf = dict(passed=True, finite=True, discrete=False, relative_vector_error=1e-16)
    replay = dict(status='passed', checkpoint_sha256=pilot['checkpoint_sha256'],
        fixture_sha256=pilot['fixture_sha256'], registration_sha256='registered',
        replay_validation=dict(passed=True, protocol='exact_discrete_relative_float_v1',
            tolerance=2e-11, leaves=[leaf]))
    saved = dict(metadata=dict(completed_new_steps=2))
    if change == 'wrong_replay_cp': replay['checkpoint_sha256'] = '0'*64
    if change == 'wrong_fixture': replay['fixture_sha256'] = '0'*64
    if change == 'source': replay['registration_sha256'] = 'other'
    if change == 'cp': checkpoint.write_bytes(b'changed checkpoint')
    if change == 'ancestor': saved['metadata'] = dict(completed_new_steps=3, pilot_checkpoint_sha256='wrong')
    if change == 'nonfinite': leaf.update(finite=False, relative_vector_error=float('nan'))
    if change == 'discrete': leaf.update(discrete=True, bitwise_exact=False)
    atomic_json(tmp_path/'pilot.json', pilot); atomic_json(tmp_path/'process_replay.json', replay)
    if change == 'none':
        assert checked_training_release(tmp_path, saved, 'registered') == pilot['checkpoint_sha256']
        saved['metadata'] = dict(completed_new_steps=3, pilot_checkpoint_sha256=pilot['checkpoint_sha256'])
        checkpoint.write_bytes(b'legitimate trained descendant')
        assert checked_training_release(tmp_path, saved, 'registered') == pilot['checkpoint_sha256']
    else:
        with pytest.raises(ValueError): checked_training_release(tmp_path, saved, 'registered')


@pytest.mark.parametrize('change', ['none', 'rejected', 'zero', 'counter', 'accepted_count', 'nonfinite',
    'gauge', 'FS_nan', 'FS_large', 'ESS_small', 'ESS_nan'])
def test_rejected_or_zero_update_cannot_count_as_success(change):
    old = dict(w=jnp.asarray([.1, .2]))
    changed = [dict(w=jnp.asarray([.11, .19])), dict(completed_steps=jnp.asarray(3),
        accepted_updates=jnp.asarray(3)), None, None, None, jnp.asarray([-.3, -.2]), None,
        dict(accepted=jnp.asarray(True), sampled_fs_sine=jnp.asarray(.007),
            reweighting_ess_fraction=jnp.asarray(.999))]
    if change == 'rejected': changed[-1]['accepted'] = jnp.asarray(False)
    if change == 'zero': changed[0] = old
    if change == 'counter': changed[1]['completed_steps'] = jnp.asarray(2)
    if change == 'accepted_count': changed[1]['accepted_updates'] = jnp.asarray(2)
    if change == 'nonfinite': changed[5] = jnp.asarray([np.nan, -.2])
    if change == 'gauge': changed[-1]['sampled_fs_sine'] = jnp.asarray(0.)
    if change == 'FS_nan': changed[-1]['sampled_fs_sine'] = jnp.asarray(np.nan)
    if change == 'FS_large': changed[-1]['sampled_fs_sine'] = jnp.asarray(.02)
    if change == 'ESS_small': changed[-1]['reweighting_ess_fraction'] = jnp.asarray(.97)
    if change == 'ESS_nan': changed[-1]['reweighting_ess_fraction'] = jnp.asarray(np.nan)
    if change == 'none': validate_committed_update(old, tuple(changed), 1)
    else:
        with pytest.raises(ArithmeticError): validate_committed_update(old, tuple(changed), 1)


def saved_pilot_fixture(tmp_path):
    """Synthetic bookkeeping, not new optimized physical data."""
    spec, params, optimizer, states, key = fixture_parent(tmp_path)
    directory = tmp_path/'stopped'; directory.mkdir()
    previous = deepcopy(spec); previous['output'] = str(directory)
    registration = dict(config=previous, sources={})
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(directory/'registration.json', registration)
    history = [dict(step=i+1, seconds=1600., SR=dict(accepted=True, sampled_fs_sine=.007))
        for i in range(2)]
    optimizer = dict(optimizer, completed_steps=jnp.asarray(4), accepted_updates=jnp.asarray(4))
    metadata = dict(registration_sha256=registration['sha256'],
        parent_checkpoint_sha256=spec['parent']['checkpoint_sha256'],
        params_sha256=pytree_sha256(params), optimizer_sha256=pytree_sha256(optimizer),
        sampler_rng_sha256=pytree_sha256((states, key)), sampler_states=states, rng=key,
        completed_new_steps=2, new_history=history)
    save_checkpoint(directory/'checkpoint.pkl', params, optimizer_state=optimizer, metadata=metadata)
    atomic_json(directory/'pilot.json', dict(status='budget_failed', memory_gate_passed=True,
        registration_sha256=registration['sha256'], checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'),
        remaining_train_forecast_seconds=22000.))
    atomic_json(directory/'results.json', dict(status='failed', phase='pilot', completed_new_steps=2,
        history=history, error="RuntimeError('measured continuation forecast or memory exceeds preregistered allocation')"))
    spec.update(release_profile=RESUME_PROFILE, output=str(tmp_path/'continued'), resume_from=dict(
        directory=str(directory), **{name+'_sha256':file_sha256(directory/(name+suffix))
            for name, suffix in [('registration','.json'), ('results','.json'),
                                 ('pilot','.json'), ('checkpoint','.pkl')]}))
    spec['audit']['seed'] += 77
    spec['phase_seconds'] = dict(pilot=10800, replay=7200, train=28800)
    return spec, params, optimizer, states, key


def test_saved_budget_pilot_preserves_all_physical_and_training_state(tmp_path):
    spec, params, optimizer, states, key = saved_pilot_fixture(tmp_path)
    validate_plan(spec)
    saved = checked_saved_pilot(spec)
    assert pytree_sha256(saved['params']) == pytree_sha256(params)
    assert pytree_sha256(saved['optimizer_state']) == pytree_sha256(optimizer)
    np.testing.assert_array_equal(saved['metadata']['sampler_states'], states)
    np.testing.assert_array_equal(saved['metadata']['rng'], key)
    assert saved['metadata']['completed_new_steps'] == 2


@pytest.mark.parametrize('change', ['checkpoint', 'registration', 'results', 'pilot',
    'learning_rate', 'arity', 'H', 'batch', 'audit_reuse', 'audit_size', 'unreleased', 'budget'])
def test_saved_pilot_adoption_cannot_hide_physics_or_scope_changes(tmp_path, change):
    spec, *_ = saved_pilot_fixture(tmp_path)
    if change in ('checkpoint', 'registration', 'results', 'pilot'):
        spec['resume_from'][change+'_sha256'] = '0'*64
    if change == 'learning_rate': spec['sr']['step_size'] *= 2
    if change == 'arity': spec['arm'] = 'ternary_joint'
    if change == 'H': spec['j2_over_j1'] = .125
    if change == 'batch': spec['training']['records_per_chain'] *= 2
    if change == 'audit_reuse': spec['audit']['seed'] -= 77
    if change == 'audit_size': spec['audit']['records_per_chain'] *= 2
    if change == 'unreleased': spec.pop('release_profile')
    if change == 'budget': spec['phase_seconds']['train'] += 1
    with pytest.raises(ValueError):
        validate_plan(spec)
        checked_saved_pilot(spec)


@pytest.mark.parametrize('fault', ['memory', 'reason', 'other_error', 'unaccepted', 'zero_update'])
def test_saved_pilot_cannot_relabel_any_other_failure(tmp_path, fault):
    spec, *_ = saved_pilot_fixture(tmp_path); p = Path(spec['resume_from']['directory'])
    result = json.loads((p/'results.json').read_text())
    pilot = json.loads((p/'pilot.json').read_text())
    if fault == 'memory': pilot['memory_gate_passed'] = False
    if fault == 'reason': pilot['remaining_train_forecast_seconds'] = 1.
    if fault == 'other_error': result['error'] = 'FloatingPointError()'
    if fault in ('unaccepted', 'zero_update'):
        from su2_nqs.vmc.checkpoint import load_checkpoint
        saved = load_checkpoint(p/'checkpoint.pkl'); history = saved['metadata']['new_history']
        if fault == 'unaccepted': history[-1]['SR']['accepted'] = False
        else: history[-1]['SR']['sampled_fs_sine'] = 0.
        result['history'] = history
        save_checkpoint(p/'checkpoint.pkl', saved['params'], optimizer_state=saved['optimizer_state'], metadata=saved['metadata'])
        spec['resume_from']['checkpoint_sha256'] = file_sha256(p/'checkpoint.pkl')
        pilot['checkpoint_sha256'] = spec['resume_from']['checkpoint_sha256']
    atomic_json(p/'results.json', result); atomic_json(p/'pilot.json', pilot)
    spec['resume_from']['results_sha256'] = file_sha256(p/'results.json')
    spec['resume_from']['pilot_sha256'] = file_sha256(p/'pilot.json')
    with pytest.raises(ValueError): checked_saved_pilot(spec)


def test_cost_forecast_keeps_cold_once_and_safe_phase_reserve():
    history = [dict(seconds=10000.), dict(seconds=1600.)]
    forecast = remaining_work_forecast(history, target_steps=8, audit_centers=512, training_centers=128)
    assert forecast == pytest.approx(1600.*12*1.15)
    assert 18000 < forecast < 28800
    assert not phase_has_room(history[1:], 0., 1900.)
    assert phase_has_room(history[1:], 0., 28800.)
    assert not phase_has_room(history[1:], 21000., 28800., units=4.)
    with pytest.raises(ValueError): remaining_work_forecast([], target_steps=8, audit_centers=512, training_centers=128)
    with pytest.raises(ValueError): phase_has_room([dict(seconds=float('nan'))], 0., 3000.)
