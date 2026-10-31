"""Optional backend: physical units, identical amplitudes and full RNG replay."""
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import (
    make_driver, physical_hamiltonian, full_update, driver_payload, restore_driver,
    balanced_spin_half_initial_states,
    uniform_exchange_proposal,
)
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import model_log, initial_model, validate_spec
from projects.clebsch_tree_nqs.workflows.spin import _bonds, _groups
from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint

jax.config.update('jax_enable_x64', True)


def toy_spec():
    return dict(task='spin_spatial_vit_quality', preregistered_before_training=True,
        backend='netket_ntk', model_family='literature', spin_inversion_even=True,
        j2_over_j1=.125, initialization_seed=201, training_seed=211,
        model=dict(geometry='square', rows=2, columns=2, patch_rows=1, patch_columns=1,
                   embedding_dim=4, head_count=2, depth=1, mlp_ratio=4),
        stages=[dict(projection='none', steps=3)], quality_milestones=[0,3],
        maximum_training_seconds=120., maximum_audit_seconds_per_checkpoint=120.,
        sampling=dict(chains=4,snapshots_per_update=4,proposals_between_snapshots=4,burn_proposals=8),
        audit=dict(seeds=[221,231],chains=4,snapshots=4,proposals_between_snapshots=4,
                   burn_proposals=8,query_batch=4),
        sr=dict(damping=.01,step_size=.005),
        netket=dict(chunk_size=4,chunk_size_bwd=4,on_the_fly=True),
        momenta=dict(Gamma=[0,0]),regions=dict(compact4=dict(sites=[0,1,2,3],left=[0,1])))


def test_balanced_initial_law_has_exact_support_and_uniform_small_sector():
    from collections import Counter
    x = np.asarray(balanced_spin_half_initial_states(jax.random.PRNGKey(810), 12000, 4))
    assert np.isin(x, [-1, 1]).all() and (x.sum(-1) == 0).all()
    counts = Counter(map(tuple, x))
    assert len(counts) == 6
    assert max(abs(n/len(x)-1/6) for n in counts.values()) < .012
    np.testing.assert_array_equal(x, balanced_spin_half_initial_states(jax.random.PRNGKey(810), 12000, 4))
    for n in (36,64):
        y = np.asarray(balanced_spin_half_initial_states(jax.random.PRNGKey(811), 128, n))
        assert y.shape == (128,n) and np.isin(y,[-1,1]).all() and (y.sum(-1)==0).all()
    with pytest.raises(ValueError):
            balanced_spin_half_initial_states(jax.random.PRNGKey(812), 128, 7)


def test_zero_step_freeze_exposes_exact_initial_state_without_sampling(tmp_path, monkeypatch):
    import json
    from projects.clebsch_tree_nqs.workflows import spin_spatial_vit_netket as backend
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
    from su2_nqs.frontier.artifacts import pytree_sha256
    version = backend.importlib.metadata.version
    monkeypatch.setattr(backend.importlib.metadata, 'version',
                        lambda name: backend.NETKET_VERSION if name == 'netket' else version(name))
    def forbidden(*args, **kwargs):
        raise AssertionError('zero-step freeze must not create a sampler/optimizer')
    monkeypatch.setattr(backend, 'make_driver', forbidden)
    spec = toy_spec()
    path = tmp_path/'config.json'
    path.write_text(json.dumps(spec))
    output = tmp_path/'frozen'
    result = backend.train(path, output, until_step=0)
    saved = load_checkpoint(output/'quality'/'step00000.pkl')
    expected, _ = initial_model(spec)
    assert result['completed_steps'] == 0 and result['history'] == []
    assert result['initial_frozen_only'] and result['training_full_H_calls'] == 0
    assert pytree_sha256(saved['params']['model']) == pytree_sha256(expected)
    assert saved['metadata']['registration_sha256'] == read_sealed(output/'registration.json')['result_sha256']
    assert not (output/'checkpoint.pkl').exists()
    assert not (output/'training.json').exists()


def test_uniform_pair_exchange_preserves_support_and_has_symmetric_small_transition():
    states = np.array([[1 if i in up else -1 for i in range(4)]
                       for up in combinations(range(4), 2)], dtype=np.int8)
    # Independent enumerated proposal oracle, including two null pair choices.
    exact = np.zeros((6, 6))
    for row, x in enumerate(states):
        for i, j in combinations(range(4), 2):
            y = x.copy(); y[i], y[j] = x[j], x[i]
            col = np.flatnonzero((states == y).all(-1))[0]
            exact[row, col] += 1/6
    np.testing.assert_allclose(exact, exact.T, atol=1e-15, rtol=0)
    draws = np.repeat(states, 12000, axis=0)
    out = np.asarray(jax.jit(uniform_exchange_proposal)(jax.random.PRNGKey(551), jnp.asarray(draws)))
    assert np.isin(out, [-1, 1]).all() and (out.sum(-1) == 0).all()
    for row in range(6):
        observed = np.array([(out[row*12000:(row+1)*12000] == x).all(-1).mean() for x in states])
        np.testing.assert_allclose(observed, exact[row], atol=.012, rtol=0)
    large = balanced_spin_half_initial_states(jax.random.PRNGKey(552), 4096, 36)
    result = np.asarray(jax.jit(uniform_exchange_proposal)(jax.random.PRNGKey(553), large))
    assert np.isin(result, [-1, 1]).all() and (result.sum(-1) == 0).all()


def test_scalar_recovery_materializer_keeps_the_shell_heredoc_in_foreground():
    script = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/remote/coupling_window_recovery_20261017.sbatch'
    text = script.read_text()
    assert '"$task_python" - "$task_config" <<\'PY\'' in text
    assert 'run_python - "$task_config" <<' not in text
    assert 'run_python()' not in text


@pytest.mark.parametrize('initialization', ['balanced_spin_half', 'parent_configurations'])
def test_balanced_n36_sampler_support_and_zero_step_checkpoint_replay(tmp_path, initialization):
    pytest.importorskip('netket')
    spec=toy_spec(); spec['netket']['initialization']=initialization
    spec['model'].update(geometry='triangular',rows=6,columns=6,patch_rows=3,patch_columns=3)
    spec['sampling'].update(chains=128,snapshots_per_update=2)
    p,_=initial_model({**spec, 'netket':{**spec['netket'], 'initialization':'balanced_spin_half'}})
    if initialization == 'parent_configurations':
        from types import SimpleNamespace
        from su2_nqs.frontier.provenance import file_sha256
        source=tmp_path/'parent.pkl'
        expected=balanced_spin_half_initial_states(jax.random.PRNGKey(903),128,36)
        save_checkpoint(source,dict(model=p,netket_sampler_state=SimpleNamespace(σ=expected)),metadata={})
        spec['coupling_warm_start_from']=dict(directory=str(tmp_path),checkpoint_path='parent.pkl',checkpoint_sha256=file_sha256(source))
    driver,_,_,_=make_driver(spec,p,'gamma_a1')
    initial=np.asarray(driver.state.sampler_state.σ)
    assert initial.shape==(128,36) and np.isin(initial,[-1,1]).all() and (initial.sum(-1)==0).all()
    if initialization == 'parent_configurations':np.testing.assert_array_equal(initial,expected)
    state,opt=driver_payload(driver);path=tmp_path/'zero.pkl'
    save_checkpoint(path,state,optimizer_state=opt,metadata={'completed_steps':0})
    replay,_,_,_=make_driver(spec,p,'gamma_a1');restore_driver(replay,load_checkpoint(path))
    first=np.asarray(driver.state.samples);second=np.asarray(replay.state.samples)
    assert np.isin(first,[-1,1]).all() and (first.sum(-1)==0).all()
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(jax.random.key_data(driver.state.sampler_state.rng),
                                 jax.random.key_data(replay.state.sampler_state.rng))


def test_physical_H_and_all_projection_amplitudes():
    nk = pytest.importorskip('netket')
    spec = toy_spec(); c = validate_spec(spec); p,_ = initial_model(spec)
    basis = np.array([[int(i in s) for i in range(4)] for s in combinations(range(4),2)])
    lookup = {tuple(x):i for i,x in enumerate(basis)}
    expected = np.zeros((6,6))
    for col,x in enumerate(basis):
        for i,j,w in _bonds('square',2,2,.125):
            expected[col,col] += w*(x[i]-.5)*(x[j]-.5)
            if x[i] != x[j]:
                y=x.copy(); y[i],y[j]=y[j],y[i]
                expected[lookup[tuple(y)],col] += w/2
    hi,_,h = physical_hamiltonian(spec)
    xs,mel = h.get_conn_padded(jnp.asarray(2*basis-1))
    actual = np.zeros((6,6))
    for i,(rows,weights) in enumerate(zip(np.asarray(xs),np.asarray(mel))):
        for x,w in zip(rows,weights):
            if w: actual[i,lookup[tuple(((x+1)/2).astype(int))]] += w
    np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0.)
    for projection in ('none','translation','gamma_a1'):
        driver,_,fn,_=make_driver(spec,p,projection)
        direct=jax.vmap(fn,in_axes=(None,0))(p,jnp.asarray(basis))
        nk_values=driver.state.log_value(jnp.asarray(2*basis-1))
        np.testing.assert_allclose(nk_values,direct,atol=2e-12,rtol=0.)
        group=None if projection=='none' else _groups('square',2,2)[projection]
        oracle,_=model_log(spec,c,group)
        np.testing.assert_allclose(direct,jax.vmap(oracle,in_axes=(None,0))(p,basis),atol=2e-12,rtol=0.)


def test_completed_step_restores_optimizer_and_sampler_rng(tmp_path: Path):
    pytest.importorskip('netket')
    spec=toy_spec(); p,_=initial_model(spec)
    direct,_,_,_=make_driver(spec,p,'none')
    full_update(direct)
    state,opt=driver_payload(direct)
    path=tmp_path/'checkpoint.pkl'
    save_checkpoint(path,state,optimizer_state=opt,metadata={'completed_steps':1})
    replay,_,_,_=make_driver(spec,p,'none')
    restore_driver(replay,load_checkpoint(path))
    for a,b in zip(jax.tree_util.tree_leaves(state['model']),jax.tree_util.tree_leaves(replay.state.parameters)):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(state['netket_sampler_state'].σ,replay.state.sampler_state.σ)
    np.testing.assert_array_equal(jax.random.key_data(state['netket_sampler_state'].rng),
                                 jax.random.key_data(replay.state.sampler_state.rng))
    first=full_update(direct); second=full_update(replay)
    assert direct.state._samples is None and replay.state._samples is None
    assert first['unique_local_sample_partitions']==min(jax.device_count(),4)
    assert direct.step_count == replay.step_count == 2
    assert first['energy'] == pytest.approx(second['energy'],abs=2e-12)
    for a,b in zip(jax.tree_util.tree_leaves(direct.state.parameters),jax.tree_util.tree_leaves(replay.state.parameters)):
        np.testing.assert_allclose(a,b,atol=2e-12,rtol=0.)
    np.testing.assert_array_equal(jax.random.key_data(direct.state.sampler_state.rng),
                                 jax.random.key_data(replay.state.sampler_state.rng))


def test_bounded_netket_update_uses_stage_SR_and_replays_with_uniform_exchange(tmp_path):
    pytest.importorskip('netket')
    spec = toy_spec()
    spec['netket'].update(initialization='balanced_spin_half', exchange_kernel='uniform_all_pairs',
        replicate_initial_params=True,
        sampled_trust=dict(maximum_fs_sine=.001, minimum_ess_fraction=.95, halvings=8))
    spec['stages'] = [dict(projection='gamma_a1', steps=3, sr=dict(step_size=.02, damping=.03))]
    p, _ = initial_model(spec)
    first, _, _, _ = make_driver(spec, p, 'gamma_a1')
    row = full_update(first, spec=spec)
    assert row['sampled_fs_sine'] <= .001+1e-12
    assert row['reweighting_ess_fraction'] >= .95
    state, optimizer = driver_payload(first)
    path = tmp_path/'bounded.pkl'
    save_checkpoint(path, state, optimizer_state=optimizer, metadata={})
    replay, _, _, _ = make_driver(spec, p, 'gamma_a1', step=1)
    restore_driver(replay, load_checkpoint(path))
    left, right = full_update(first, spec=spec), full_update(replay, spec=spec)
    assert left['energy'] == pytest.approx(right['energy'], abs=2e-12)
    assert left['update_scale'] == right['update_scale']
    for a, b in zip(jax.tree_util.tree_leaves(first.state.parameters),
                    jax.tree_util.tree_leaves(replay.state.parameters)):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=0)
    np.testing.assert_array_equal(jax.random.key_data(first.state.sampler_state.rng),
                                 jax.random.key_data(replay.state.sampler_state.rng))


@pytest.mark.parametrize('on_the_fly', [True, False])
def test_NTK_direction_solves_independent_centered_complex_SR(on_the_fly):
    pytest.importorskip('netket')
    from jax.flatten_util import ravel_pytree
    from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
    spec=toy_spec(); spec['netket']['on_the_fly']=on_the_fly; p,_=initial_model(spec)
    driver,_,fn,_=make_driver(spec,p,'none')
    driver.reset_step(); _,direction=driver.compute_loss_and_update()
    samples=((np.asarray(driver.state.samples).reshape(-1,4)+1)/2).astype(int)
    flat,unravel=ravel_pytree(p)
    def components(v,x):
        z=fn(unravel(v),x)
        return jnp.array([z.real,z.imag])
    scores=np.asarray(jax.vmap(jax.jacrev(components),in_axes=(None,0))(flat,samples)).copy()
    scores-=scores.mean(axis=0,keepdims=True)
    a=np.sqrt(2/len(samples))*np.concatenate((scores[:,0],scores[:,1]))
    e=np.asarray(make_full_heisenberg_ratios(fn,_bonds('square',2,2,.125))(p,samples)).copy()
    e-=e.mean(); b=np.sqrt(2/len(samples))*np.concatenate((e.real,e.imag))
    # VMC_SR's raw direction uses twice our force/metric convention.
    # Its SGD rate is eta/2; the installed physical update is eta*d.
    d=np.asarray(ravel_pytree(direction)[0])/2
    gradient=a.T@b
    relative=np.linalg.norm(a.T@(a@d)+spec['sr']['damping']*d-gradient)/np.linalg.norm(gradient)
    print('centered complex SR residual; on_the_fly', on_the_fly, relative, flush=True)
    assert relative < 1e-9
    driver.update_parameters(direction)
    installed=np.asarray(ravel_pytree(driver.state.parameters)[0])
    np.testing.assert_allclose(installed, np.asarray(flat)-spec['sr']['step_size']*d,
                               atol=2e-13, rtol=0.)


def test_staged_training_resume_and_shared_audit_registration(tmp_path: Path):
    pytest.importorskip('netket')
    import json
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import train
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import register
    spec=toy_spec()
    spec['stages']=[dict(projection=p,steps=1) for p in ('none','translation','gamma_a1')]
    spec['quality_milestones']=[0,1,2,3]
    path=tmp_path/'config.json'; path.write_text(json.dumps(spec))
    a,b=tmp_path/'a',tmp_path/'b'
    assert train(path,a,until_step=1,session_seconds=120.)['completed_steps']==1
    resumed=train(path,a,resume=True,session_seconds=120.)
    complete=train(path,b,session_seconds=120.)
    assert resumed['completed_steps']==complete['completed_steps']==3
    # GPU reduction order across recompiled processes need not be BITWISE
    # identical. Require the pre-existing 2e-12 physical replay tolerance
    # AND identical sampler RNG/configurations; content hashes stay distinct.
    replay_state=load_checkpoint(a/'checkpoint.pkl')['params']
    direct_state=load_checkpoint(b/'checkpoint.pkl')['params']
    errors=[]
    for left,right in zip(jax.tree_util.tree_leaves(replay_state['model']),
                          jax.tree_util.tree_leaves(direct_state['model'])):
        errors.append(float(np.max(abs(np.asarray(left)-np.asarray(right)))))
        np.testing.assert_allclose(left,right,atol=2e-12,rtol=0.)
    np.testing.assert_array_equal(jax.random.key_data(replay_state['netket_sampler_state'].rng),
                                 jax.random.key_data(direct_state['netket_sampler_state'].rng))
    np.testing.assert_array_equal(replay_state['netket_sampler_state'].σ,
                                 direct_state['netket_sampler_state'].σ)
    driver,_,fn,_=make_driver(spec,replay_state['model'],'gamma_a1',step=3)
    bits=((np.asarray(replay_state['netket_sampler_state'].σ)+1)/2).astype(int)
    batch=jax.jit(jax.vmap(fn,in_axes=(None,0)))
    amplitude_error=float(np.max(abs(np.exp(np.asarray(batch(replay_state['model'],bits))
        -np.asarray(batch(direct_state['model'],bits)))-1)))
    assert amplitude_error<2e-12
    print('staged GPU/CPU replay maximum parameter and relative amplitude errors',
          max(errors),amplitude_error,flush=True)
    assert resumed['training_full_H_calls']==48
    assert [r['projection'] for r in resumed['history']]==['none','translation','gamma_a1']
    _,registered=register(path,a)  # SAME manifest used by untouched audits.
    assert registered['result_sha256']==resumed['registration_sha256']
    assert (a/'quality/step00003.pkl').is_file()


def test_sealed_runtime_recovery_preserves_parent_and_rejects_physics_change(tmp_path):
    """No optional NetKet needed to prove recovery's immutable-parent contract."""
    import copy
    import json
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import recovery_state
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import sealed_json
    from su2_nqs.frontier.provenance import file_sha256
    from su2_nqs.frontier.artifacts import pytree_sha256
    parent=tmp_path/'parent'; parent.mkdir()
    spec=toy_spec(); spec['output']='old-output'
    snapshots=parent/'source_snapshots'; snapshots.mkdir()
    (snapshots/'runtime.py').write_text('immutable oracle\n')
    reg=sealed_json(parent/'registration.json',dict(config=spec,
        source_manifest={'runtime.py':file_sha256(snapshots/'runtime.py')}))
    params={'v':np.array([.1,.2])}
    meta=dict(completed_steps=1,history=[dict(step=1)],registration_sha256=reg['result_sha256'],
              training_full_H_calls=16,active_training_seconds=3.)
    save_checkpoint(parent/'checkpoint.pkl',dict(model=params,netket_sampler_state={'rng':np.array([1,2])}),
        optimizer_state=dict(step=1,netket_version='3.22.3',
            optimizer={'momentum':np.array([3.,4.])},previous_direction=np.array([.2,.3])),metadata=meta)
    progress=sealed_json(parent/'progress.json',dict(checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'),
        final_params_sha256=pytree_sha256(params),status='paused_session_budget'))
    original={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
    new=copy.deepcopy(spec); new['output']='new-output'
    new['netket'].update(chunk_size=2,chunk_size_bwd=2,on_the_fly=False)
    new['netket_resume_from']=dict(directory=str(parent),completed_steps=1,
        checkpoint_sha256=progress['checkpoint_sha256'],registration_sha256=reg['result_sha256'],
        progress_sha256=progress['result_sha256'])
    restored=recovery_state(new,'new-registration')
    assert restored['metadata']['registration_sha256']=='new-registration'
    assert restored['metadata']['training_full_H_calls']==16
    assert restored['metadata']['active_training_seconds']==3.
    assert restored['optimizer_state']['step']==1
    assert not restored['metadata']['recovery_parent']['sampler_reinitialized']
    np.testing.assert_array_equal(restored['params']['netket_sampler_state']['rng'],[1,2])
    extended=copy.deepcopy(new)
    extended['maximum_training_seconds']+=60
    extended['netket_resume_from']['time_budget_extension_seconds']=60
    recovered=recovery_state(extended,'extended-registration')
    assert recovered['metadata']['active_training_seconds']==3.
    assert recovered['metadata']['training_full_H_calls']==16
    assert recovered['optimizer_state']['step']==1
    assert recovered['metadata']['recovery_parent']['fixed_step_budget_unchanged']
    assert recovered['metadata']['recovery_parent']['parent_training_time_cap_seconds']==120
    assert recovered['metadata']['recovery_parent']['recovery_training_time_cap_seconds']==180
    np.testing.assert_array_equal(recovered['params']['netket_sampler_state']['rng'],[1,2])
    np.testing.assert_array_equal(recovered['optimizer_state']['optimizer']['momentum'],[3.,4.])
    np.testing.assert_array_equal(recovered['optimizer_state']['previous_direction'],[.2,.3])
    undeclared=copy.deepcopy(new);undeclared['maximum_training_seconds']+=60
    with pytest.raises(ValueError,match='exact declared'):
        recovery_state(undeclared,'bad-registration')
    for extension in (-1,True,float('nan'),float('inf'),'60'):
        bad=copy.deepcopy(extended);bad['netket_resume_from']['time_budget_extension_seconds']=extension
        with pytest.raises(ValueError,match='exact declared'):
            recovery_state(bad,'bad-registration')
    altered_steps=copy.deepcopy(extended);altered_steps['stages'][0]['steps']+=1
    with pytest.raises(ValueError,match='more than runtime'):
        recovery_state(altered_steps,'bad-registration')
    for status in ('running','total_time_cap','fixed_steps_complete'):
        changed=sealed_json(parent/'progress.json',dict(checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'),
            final_params_sha256=pytree_sha256(params),status=status))
        bad=copy.deepcopy(extended);bad['netket_resume_from']['progress_sha256']=changed['result_sha256']
        with pytest.raises(ValueError,match='safely paused unfinished'):
            recovery_state(bad,'bad-registration')
    (parent/'progress.json').write_bytes(original['progress.json'])
    (parent/'training.json').write_text('{}')
    with pytest.raises(ValueError,match='safely paused unfinished'):
        recovery_state(extended,'bad-registration')
    (parent/'training.json').unlink()
    assert original=={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
    for section,key in [('sr','damping'),('sampling','chains'),('model','depth'),('audit','snapshots')]:
        bad=copy.deepcopy(new); bad[section][key]*=2
        with pytest.raises(ValueError,match='more than runtime'):
            recovery_state(bad,'new-registration')
    bad=copy.deepcopy(extended);bad['j2_over_j1']+=.025
    with pytest.raises(ValueError,match='more than runtime'):
        recovery_state(bad,'bad-registration')
    (snapshots/'runtime.py').write_text('tampered')
    with pytest.raises(ValueError,match='frozen runtime'):
        recovery_state(new,'new-registration')


@pytest.mark.parametrize('spins', [[[1,0,-1,0]], [[1,1,-1,2]], [[1,1,1,-1]]])
def test_invalid_sampler_support_is_rejected_before_any_H_call(spins):
    from types import SimpleNamespace
    def forbidden():
        raise AssertionError('invalid configurations reached H/SR')
    driver=SimpleNamespace(reset_step=lambda:None,
        state=SimpleNamespace(samples=jnp.asarray(spins)),compute_loss_and_update=forbidden)
    with pytest.raises(FloatingPointError,match='Sz=0 support'):
        full_update(driver)


def test_fixed_endpoint_continuation_keeps_immutable_parent_and_complete_state(tmp_path):
    import copy
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import continuation_state
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import sealed_json
    from su2_nqs.frontier.provenance import file_sha256
    from su2_nqs.frontier.artifacts import pytree_sha256
    parent=tmp_path/'parent'; (parent/'quality').mkdir(parents=True)
    spec=toy_spec(); spec['output']='old'
    spec['stages']=[dict(projection='gamma_a1',steps=3)]
    snapshots=parent/'source_snapshots'; snapshots.mkdir()
    (snapshots/'runtime.py').write_text('sealed parent source\n')
    registration=sealed_json(parent/'registration.json',dict(config=spec,
        source_manifest={'runtime.py':file_sha256(snapshots/'runtime.py')}))
    params={'weights':np.array([.1,.2])}
    checkpoint=parent/'quality/step00003.pkl'
    save_checkpoint(checkpoint,dict(model=params,netket_sampler_state={'rng':np.array([4,7])}),
        optimizer_state=dict(step=3,netket_version='3.22.3'),
        metadata=dict(completed_steps=3,history=[dict(step=i) for i in (1,2,3)],
            registration_sha256=registration['result_sha256'],
            training_full_H_calls=48,active_training_seconds=7.5))
    progress=sealed_json(parent/'progress.json',dict(completed_steps=3,final_params_sha256=pytree_sha256(params)))
    original={str(p.relative_to(parent)):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    new=copy.deepcopy(spec); new['output']='new'
    new['stages'].append(dict(projection='gamma_a1',steps=2))
    new['quality_milestones']=[0,3,5]; new['audit']['seeds']=[441,451]
    new['netket_continue_from']=dict(directory=str(parent),checkpoint_path='quality/step00003.pkl',
        completed_steps=3,additional_steps=2,checkpoint_sha256=file_sha256(checkpoint),
        registration_sha256=registration['result_sha256'],progress_sha256=progress['result_sha256'])
    restored=continuation_state(new,'new-registration')
    assert restored['optimizer_state']['step']==3
    assert restored['metadata']['training_full_H_calls']==48
    assert restored['metadata']['active_training_seconds']==7.5
    assert not restored['metadata']['continuation_parent']['sampler_reinitialized']
    np.testing.assert_array_equal(restored['params']['netket_sampler_state']['rng'],[4,7])
    assert original=={str(p.relative_to(parent)):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    for section,key in [('model','depth'),('sr','damping'),('sampling','chains')]:
        bad=copy.deepcopy(new); bad[section][key]*=2
        with pytest.raises(ValueError,match='changed model'):
            continuation_state(bad,'new-registration')
    bad=copy.deepcopy(new); bad['stages'][-1]['projection']='translation'
    with pytest.raises(ValueError,match='only full-A1'):
        continuation_state(bad,'new-registration')
    bad=copy.deepcopy(new); bad['audit']['seeds']=spec['audit']['seeds']
    with pytest.raises(ValueError,match='fresh independent'):
        continuation_state(bad,'new-registration')
    bad=copy.deepcopy(new); bad['netket_continue_from']['checkpoint_path']='../checkpoint.pkl'
    with pytest.raises(ValueError,match='immutable quality checkpoint'):
        continuation_state(bad,'new-registration')
    # More authorized training must not reset already consumed work/time.
    extended=copy.deepcopy(new)
    extended['maximum_training_seconds'] += 60
    extended['netket_continue_from']['time_budget_extension_seconds']=60
    continued=continuation_state(extended,'extended-registration')
    assert continued['metadata']['active_training_seconds']==7.5
    assert continued['metadata']['training_full_H_calls']==48
    assert continued['optimizer_state']['step']==3
    np.testing.assert_array_equal(continued['params']['netket_sampler_state']['rng'],[4,7])
    assert continued['metadata']['continuation_parent']['parent_training_time_cap_seconds']==120
    assert continued['metadata']['continuation_parent']['continuation_training_time_cap_seconds']==180
    assert original=={str(p.relative_to(parent)):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    undeclared=copy.deepcopy(new); undeclared['maximum_training_seconds'] += 60
    with pytest.raises(ValueError,match='declared extension'):
        continuation_state(undeclared,'new-registration')
    for extension in (-1, True, float('nan'), float('inf'), '60'):
        bad=copy.deepcopy(extended)
        bad['netket_continue_from']['time_budget_extension_seconds']=extension
        with pytest.raises(ValueError,match='declared extension'):
            continuation_state(bad,'new-registration')
    bad=copy.deepcopy(extended); bad['sr']['step_size']*=2
    with pytest.raises(ValueError,match='changed model'):
        continuation_state(bad,'new-registration')


def test_audited_paused_milestone_extension_preserves_executed_schedule(tmp_path, monkeypatch):
    """No fabricated terminal record, reset counters, or running-parent fork."""
    import copy
    import json
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import continuation_state
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import sealed_json
    from su2_nqs.frontier.provenance import file_sha256
    from su2_nqs.frontier.artifacts import pytree_sha256
    parent = tmp_path/'parent'; (parent/'quality').mkdir(parents=True)
    spec = toy_spec(); spec['output'] = 'old'
    spec['coupling_warm_start_from'] = dict(retained_as_ancestry=True)
    spec['stages'] = [dict(projection='gamma_a1', steps=4, sr=dict(damping=.02))]
    spec['quality_milestones'] = [0, 2, 4]
    registration = sealed_json(parent/'registration.json', dict(config=spec, source_manifest={}))
    params = {'weights': np.array([.1, .2])}
    checkpoint = parent/'quality/step00002.pkl'
    save_checkpoint(checkpoint, dict(model=params, netket_sampler_state={'rng': np.array([4, 7])}),
        optimizer_state=dict(step=2), metadata=dict(completed_steps=2,
            history=[dict(step=1), dict(step=2)], active_training_seconds=9.,
            training_full_H_calls=32, registration_sha256=registration['result_sha256']))
    progress = sealed_json(parent/'progress.json', dict(completed_steps=2,
        status='paused_session_budget', final_params_sha256=pytree_sha256(params)))
    audit_path = parent/'audits/step00002/results.json'
    audit = sealed_json(audit_path, dict(step=2, registration_sha256=registration['result_sha256'],
        checkpoint_sha256=file_sha256(checkpoint), params_sha256=pytree_sha256(params),
        training_projection='gamma_a1', audit_projection='gamma_a1'))
    original = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    new = copy.deepcopy(spec); new['output'] = 'new'
    new['stages'] = [dict(projection='gamma_a1', steps=2, sr=dict(damping=.02)),
                     dict(projection='gamma_a1', steps=2)]
    new['audit']['seeds'] = [441, 451]
    new['maximum_training_seconds'] += 60
    new['netket_continue_from'] = dict(directory=str(parent), completed_steps=2,
        additional_steps=2, from_registered_milestone=True, time_budget_extension_seconds=60,
        checkpoint_path='quality/step00002.pkl', checkpoint_sha256=file_sha256(checkpoint),
        registration_sha256=registration['result_sha256'], progress_sha256=progress['result_sha256'],
        audit_sha256=audit['result_sha256'])
    restored = continuation_state(new, 'child-registration')
    assert restored['optimizer_state']['step'] == 2
    assert restored['metadata']['active_training_seconds'] == 9.
    assert restored['metadata']['training_full_H_calls'] == 32
    np.testing.assert_array_equal(restored['params']['netket_sampler_state']['rng'], [4, 7])
    assert original == {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    assert not (parent/'training.json').exists()
    from projects.clebsch_tree_nqs.workflows import spin_spatial_vit_quality as quality
    def forbidden_reinitialization(*args, **kwargs):
        raise AssertionError('continuation must validate the immediate parent, not initialize ancestry')
    monkeypatch.setattr(quality, 'initial_model', forbidden_reinitialization)
    (tmp_path/'child-config.json').write_text(json.dumps(new))
    _, child_registration = quality.register(tmp_path/'child-config.json', tmp_path/'child')
    assert child_registration['config'] == new
    for mutation, error in [
        (lambda s: s['netket_continue_from'].update(from_registered_milestone='yes'), 'explicit boolean'),
        (lambda s: s['netket_continue_from'].update(completed_steps=1, checkpoint_path='quality/step00001.pkl'), 'registered milestone'),
        (lambda s: s['netket_continue_from'].update(audit_sha256='unverified'), 'completed frozen'),
        (lambda s: s['stages'][0]['sr'].update(damping=.03), 'only full-A1'),
    ]:
        bad = copy.deepcopy(new); mutation(bad)
        with pytest.raises(ValueError, match=error):
            continuation_state(bad, 'child-registration')
    for status in ('running', 'total_time_cap'):
        changed = sealed_json(parent/'progress.json', dict(completed_steps=2,
            status=status, final_params_sha256=pytree_sha256(params)))
        bad = copy.deepcopy(new); bad['netket_continue_from']['progress_sha256'] = changed['result_sha256']
        with pytest.raises(ValueError, match='safely paused'):
            continuation_state(bad, 'child-registration')
