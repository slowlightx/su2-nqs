"""Independent local-H oracle and checkpoint transition diagnostic."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_jump_probe import (
    max_tree_error, replica_diagnostics, replicate_host_candidate, run, single_device_evaluate,
    resume_replay, _numeric_payload,
)


def test_actual_replica_buffers_are_checked_and_host_repair_is_explicit():
    if jax.device_count() < 2:
        pytest.skip('requires forced two-CPU or allocated two-GPU diagnostic')
    devices = np.asarray(jax.devices()[:2])
    mesh = jax.sharding.Mesh(devices, ('S',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Deliberately violate the declared replication invariant without running
    # this malformed object through JIT. The global view hides GPU/CPU 1.
    divergent = jax.make_array_from_single_device_arrays((2,), sharding, [
        jax.device_put(np.zeros(2), devices[0]),
        jax.device_put(np.ones(2), devices[1]),
    ])
    np.testing.assert_array_equal(np.asarray(divergent), np.zeros(2))
    report = replica_diagnostics({'bad': divergent})
    assert report['maximum_replica_error'] == 1.
    assert len(report['leaves'][0]['addressable_shards']) == 2
    repaired = replicate_host_candidate({'bad': np.array([2., 3.])})
    assert replica_diagnostics(repaired)['maximum_replica_error'] == 0.
    for shard in repaired['bad'].addressable_shards:
        np.testing.assert_array_equal(shard.data, [2., 3.])


def test_numeric_checkpoint_comparison_preserves_typed_rng():
    key = jax.random.key(17)
    tree = {'key': key, 'weights': jnp.array([1.,2.])}
    numeric = _numeric_payload(tree)
    np.testing.assert_array_equal(numeric['key'], jax.random.key_data(key))
    assert max_tree_error(numeric, _numeric_payload(tree)) == 0.


def test_production_resume_replays_full_endpoint(tmp_path):
    pytest.importorskip('netket')
    from test_spatial_vit_netket import toy_spec
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import make_driver, full_update, driver_payload
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import initial_model
    from su2_nqs.vmc.checkpoint import save_checkpoint
    if jax.device_count()!=1:
        pytest.skip('single-device checkpoint replay contract')
    spec=toy_spec();spec['netket']['expected_device_count']=1
    spec['stages']=[dict(projection='gamma_a1',steps=3)]
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(spec))
    params,_=initial_model(spec)
    driver,_,_,_=make_driver(spec,params,'gamma_a1')
    state,opt=driver_payload(driver)
    start=tmp_path/'start.pkl';save_checkpoint(start,state,optimizer_state=opt)
    for _ in range(2):full_update(driver,spec=spec)
    state,opt=driver_payload(driver)
    end=tmp_path/'end.pkl';save_checkpoint(end,state,optimizer_state=opt)
    result=resume_replay(cfg,start,end,tmp_path/'replay.json')
    assert result['status']=='passed'
    assert result['maximum_absolute_errors']['sampler_and_RNG']==0.


@pytest.mark.parametrize('on_the_fly', [False, True])
def test_diagnostic_online_history_cannot_change_no_momentum_sgd_updates(tmp_path, on_the_fly):
    """A logging accumulator is not the optimizer's next-update memory.

    Deliberately reset ONLY the online-statistics history in one restored
    driver. Both must keep identical physical updates, configurations, RNG,
    counters and functional optimizer state over two real full-H updates.
    This does not change any existing replay tolerance or receipt.
    """
    pytest.importorskip('netket')
    from test_spatial_vit_netket import toy_spec
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import (
        make_driver, full_update, driver_payload, restore_driver,
    )
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import initial_model
    from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint
    if jax.device_count() != 1:
        pytest.skip('validated single-device continuation only')
    spec = toy_spec()
    spec['netket'].update(expected_device_count=1, on_the_fly=on_the_fly)
    spec['stages'] = [dict(projection='gamma_a1', steps=5)]
    spec['quality_milestones'] = [0, 5]
    params, _ = initial_model(spec)
    original, _, _, _ = make_driver(spec, params, 'gamma_a1')
    full_update(original, spec=spec)
    state, optimizer = driver_payload(original)
    assert optimizer['loss_stats_online'] is not None
    assert optimizer['old_updates'] is None
    assert not jax.tree_util.tree_leaves(optimizer['optimizer'])
    checkpoint = tmp_path/'parent.pkl'
    save_checkpoint(checkpoint, state, optimizer_state=optimizer)
    same, _, _, _ = make_driver(spec, params, 'gamma_a1')
    reset, _, _, _ = make_driver(spec, params, 'gamma_a1')
    restore_driver(same, load_checkpoint(checkpoint))
    restore_driver(reset, load_checkpoint(checkpoint))
    reset._loss_stats_online = None  # intervention on diagnostics ONLY
    for _ in range(2):
        first, second = full_update(same, spec=spec), full_update(reset, spec=spec)
        assert first['energy'] == pytest.approx(second['energy'], abs=2e-12)
        assert first['variance'] == pytest.approx(second['variance'], abs=2e-12)
        a, ao = driver_payload(same)
        b, bo = driver_payload(reset)
        assert max_tree_error(a['model'], b['model']) < 2e-12
        np.testing.assert_array_equal(a['netket_sampler_state'].σ, b['netket_sampler_state'].σ)
        np.testing.assert_array_equal(jax.random.key_data(a['netket_sampler_state'].rng),
                                      jax.random.key_data(b['netket_sampler_state'].rng))
        assert ao['step'] == bo['step'] and ao['old_updates'] is bo['old_updates'] is None
        assert not jax.tree_util.tree_leaves(ao['optimizer'])
        assert not jax.tree_util.tree_leaves(bo['optimizer'])
    ao, bo = driver_payload(same)[1], driver_payload(reset)[1]
    assert max_tree_error(ao['loss_stats_online'], bo['loss_stats_online']) > 0.
    print('online history differs; functional updates/optimizer/RNG agree', on_the_fly,
          max_tree_error(same.state.parameters, reset.state.parameters))


def test_oracle_mesh_is_independent_and_restores_production_context():
    if jax.device_count() < 2:
        pytest.skip('requires the forced two-CPU or allocated two-GPU diagnostic')
    devices = np.asarray(jax.devices()[:2])
    production_mesh = jax.sharding.Mesh(devices, ('S',))
    oracle_mesh = jax.sharding.Mesh(devices[:1], ('oracle',))
    with jax.set_mesh(production_mesh):
        # In the failed diagnostic even SingleDeviceSharding inputs were
        # incompatible with this active production mesh on JAX 0.11.
        with jax.set_mesh(oracle_mesh):
            offsets = jnp.asarray([.25, -.5])
            evaluate = jax.jit(lambda p, x: x @ p['w'] + offsets)
        observed = single_device_evaluate(evaluate, {'w': np.array([2., 3.])},
                                          np.eye(2), devices[0])
        np.testing.assert_allclose(observed, [2.25, 2.5], atol=0., rtol=0.)
        assert jax.sharding.get_abstract_mesh() == production_mesh.abstract_mesh


@pytest.mark.parametrize('invalid', [0,-1,True,1.0,'1'])
def test_registered_device_count_rejects_invalid_schema(invalid):
    from test_spatial_vit_netket import toy_spec
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import validate_spec
    spec=toy_spec(); spec['netket']['expected_device_count']=invalid
    with pytest.raises(ValueError,match='positive integer'):
        validate_spec(spec)


def test_runtime_device_contract_fails_before_H_or_optional_backend(monkeypatch):
    from test_spatial_vit_netket import toy_spec
    from projects.clebsch_tree_nqs.workflows import spin_spatial_vit_netket as backend
    spec=toy_spec(); spec['netket']['expected_device_count']=2
    monkeypatch.setattr(jax,'device_count',lambda:1)
    def forbidden(*args,**kwargs):
        pytest.fail('must refuse incompatible layout before H construction')
    monkeypatch.setattr(backend,'physical_hamiltonian',forbidden)
    with pytest.raises(RuntimeError,match='registered device count is 2, observed 1'):
        backend.make_driver(spec,{},'gamma_a1')


def test_single_gpu_repair_preserves_physical_protocol_and_only_releases_two_pilots():
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import validate_spec
    configs=Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote'
    paths=sorted((configs/'repair_20261011_v6_single_gpu').glob('*.json'))
    assert len(paths)==6
    pilots=[]
    for path in paths:
        spec=json.loads(path.read_text())
        parent=json.loads((configs/'repair_20261010_v5'/path.name).read_text())
        validate_spec(spec)
        for key in ('model','j2_over_j1','sampling','sr','stages','maximum_training_seconds',
                    'repair_parameters_from','coupling_warm_start_from'):
            assert spec.get(key)==parent.get(key),key
        assert spec['netket']['expected_device_count']==1
        assert all(spec['netket'][k]==v for k,v in parent['netket'].items())
        assert spec['output'] != parent['output']
        assert set(spec['audit']['seeds']).isdisjoint(parent['audit']['seeds'])
        gate=spec['execution_gate']
        assert (gate['pilot_endpoint'],gate['isolated_resume_endpoint'])==(20,22)
        assert 20 in spec['quality_milestones'] and 22 in spec['quality_milestones']
        assert not gate['long_training_automatically_released']
        if gate['initial_pilot_selected']: pilots.append(path.stem)
    assert sorted(pilots)==['square_j05','tri_j0']


def test_production_shape_exchange_global_chain_sharding_contract():
    """Actual 128-chain layout, not just two available devices with local arrays."""
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import (
        balanced_spin_half_initial_states, uniform_exchange_proposal,
    )
    if jax.device_count() < 2:
        pytest.skip('requires two devices; force two CPU devices or run allocated GPUs')
    devices=np.asarray(jax.devices()[:2])
    mesh=jax.sharding.Mesh(devices,('S',))
    shared=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    partitioned=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('S'))
    single=jax.sharding.SingleDeviceSharding(devices[0])
    spins=np.asarray(balanced_spin_half_initial_states(jax.random.PRNGKey(703),128,36))
    key=np.array([0,704],dtype=np.uint32)

    @jax.jit
    def repeat(rng,x):
        def body(_,carry):
            rng,x=carry; rng,k=jax.random.split(rng)
            return rng,uniform_exchange_proposal(k,x)
        return jax.lax.fori_loop(0,72,body,(rng,x))

    k0,x0=repeat(jax.device_put(key,single),jax.device_put(spins,single))
    k1,x1=repeat(jax.device_put(key,shared),jax.device_put(spins,partitioned))
    assert len({str(shard.index) for shard in x1.addressable_shards})==2
    np.testing.assert_array_equal(k0,k1)
    np.testing.assert_array_equal(x0,x1)
    assert (np.asarray(x1).sum(-1)==0).all()


def test_installation_comparison_detects_structure_nonfinite_and_wrong_update():
    left = dict(a=np.array([1.,2.]), b=(np.array(3.),))
    assert max_tree_error(left,left) == 0.
    assert max_tree_error(left,dict(a=np.array([1.,2.01]),b=(np.array(3.),))) == pytest.approx(.01)
    with pytest.raises(ValueError):
        max_tree_error(left,dict(a=np.ones(2)))
    with pytest.raises(FloatingPointError):
        max_tree_error(left,dict(a=np.array([1.,np.nan]),b=(np.array(3.),)))


def test_directional_statistics_matches_exact_energy_tangent_and_gauge():
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_jump_probe import directional_statistics
    # Uniform exact Born support; dense-H finite difference is independent of
    # the covariance formula used by the diagnostic.
    rng=np.random.default_rng(760);v=rng.normal(size=12)+1j*rng.normal(size=12)
    h=rng.normal(size=(12,12));h=(h+h.T)/2
    parent=np.ones(12)/np.sqrt(12);local=h@parent/parent
    groups=np.repeat(np.arange(4),3)
    report=directional_statistics(v,local,groups,6)
    def energy(t):
        state=parent*np.exp(t*v);state/=np.linalg.norm(state)
        return float(np.vdot(state,h@state).real/6)
    np.testing.assert_allclose(report['point']['energy_slope_per_site'],
                               (energy(1e-6)-energy(-1e-6))/2e-6,atol=2e-10)
    shifted=directional_statistics(v+7+3j,local-5,groups,6)
    for k in report['point']:
        np.testing.assert_allclose(report['point'][k],shifted['point'][k],atol=1e-14)
    deleted=np.array([directional_statistics(v[groups!=g],local[groups!=g],groups[groups!=g],6)
                      ['point']['energy_slope_per_site'] for g in np.unique(groups)])
    np.testing.assert_allclose(report['whole_chain_SE']['energy_slope_per_site'],
                               np.sqrt(3/4*np.sum((deleted-deleted.mean())**2)),atol=1e-14)
    assert not report['variance_derivative_measured']


def test_optimizer_conditions_keeps_parent_and_uses_same_batch(tmp_path):
    pytest.importorskip('netket')
    from test_spatial_vit_netket import toy_spec
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import make_driver,driver_payload
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import initial_model
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_jump_probe import optimizer_conditions
    from su2_nqs.vmc.checkpoint import save_checkpoint
    from su2_nqs.frontier.artifacts import pytree_sha256
    from su2_nqs.frontier.provenance import file_sha256
    spec=toy_spec();spec['stages']=[dict(projection='gamma_a1',steps=3)]
    spec['netket']['expected_device_count']=1
    p,_=initial_model(spec);driver,h,_,_=make_driver(spec,p,'gamma_a1')
    state,opt=driver_payload(driver);checkpoint=tmp_path/'parent.pkl'
    save_checkpoint(checkpoint,state,optimizer_state=opt,metadata={'completed_steps':0})
    streams=[]
    for seed in (11,12):
        driver.reset_step();xs=np.asarray(driver.state.samples)
        es=np.asarray(driver.state.local_estimators(h).data)
        path=tmp_path/f'stream{seed}.npz'
        np.savez_compressed(path,states=((xs+1)/2).astype(int).swapaxes(0,1),energies=es.swapaxes(0,1))
        streams.append(dict(seed=seed,raw_sha256=file_sha256(path)))
    config=tmp_path/'config.json';config.write_text(json.dumps(spec))
    audit=tmp_path/'energy.json';audit.write_text(json.dumps(dict(step=0,
        checkpoint_sha256=file_sha256(checkpoint),params_sha256=pytree_sha256(p),streams=streams)))
    original=file_sha256(checkpoint)
    result=optimizer_conditions(config,checkpoint,audit,tmp_path/'conditions.json',maximum_seconds=120.)
    assert result['status']=='complete' and len(result['rows'])==3
    assert result['full_H_calls']==6*16
    assert file_sha256(checkpoint)==original
    assert all(row['radius_matched'] for row in result['rows'])
    energies=[row['training_energy_per_site'] for row in result['rows']]
    np.testing.assert_allclose(energies,energies[0],atol=1e-12,rtol=0)


def test_probe_agrees_with_independent_H_and_sgd_on_exact_toy_support(tmp_path):
    pytest.importorskip('netket')
    # Import a shared bounded fixture rather than introduce another model.
    from test_spatial_vit_netket import toy_spec
    spec=toy_spec()
    spec['stages']=[dict(projection='gamma_a1',steps=3)]
    spec['netket'].update(initialization='balanced_spin_half',exchange_kernel='uniform_all_pairs',
        replicate_initial_params=True,
        sampled_trust=dict(maximum_fs_sine=.02,minimum_ess_fraction=.95,halvings=6))
    config=tmp_path/'config.json'; config.write_text(json.dumps(spec))
    output=tmp_path/'probe.json'
    record=run(config,None,output,steps=2,oracle_rows=6,maximum_seconds=120.)
    assert record['status']=='diagnostic_complete'
    assert len(record['rows'])==2
    assert record['extra_full_H_calls']==44
    assert len(record['devices'])==jax.device_count()
    assert not record['replicate_candidate']
    assert record['initial_parameter_replication']['maximum_replica_error']==0.
    for row in record['rows']:
        assert row['fixed_rows_native_oracle_log_error'] < 2e-12
        assert row['fixed_rows_native_oracle_H_error'] < 2e-12
        assert row['sampler_stored_log_probability_error'] < 2e-12
        assert row['parameters_mutated_by_SR_error'] == 0.
        assert row['candidate_installed_parameter_error'] < 2e-12
        assert row['candidate_installed_fixed_log_error'] < 2e-12
        assert row['parameter_replicas_before_sampling']['maximum_replica_error']==0.
        assert row['SR_direction_replication']['maximum_replica_error'] < 2e-12
        assert row['final_installed_parameter_replication']['maximum_replica_error'] < 2e-12
