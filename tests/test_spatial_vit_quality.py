"""Independent physical geometry, coherent readout, and exact-resume oracles."""
from itertools import combinations
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit,
    fixed_sz_spatial_vit_log_amplitudes, _relative_geometry, _layer_norm,
)
from su2_nqs.benchmarks.spatial_vit_physics import (
    make_physical_spatial_vit_log, physical_to_model_permutation,
    make_scalar_structure_factors, born_conditional_regional_matrices,
    scalar_structure_query_plan, scalar_structure_records,
)
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.vmc.regional_density import regional_query_plan
from su2_nqs.vmc.checkpoint import load_checkpoint
from projects.clebsch_tree_nqs.workflows.spin import _groups
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import (
    validate_spec, stage_at, train, audit, DEFAULT, CONTINUATION_DEFAULT,
    initial_model, model_log, checkpoint_projection, sealed_json,
    recovery_state, register,
    _batched_logs,
    audit_cached_physics,
)
from su2_nqs.frontier.artifacts import pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import save_checkpoint

jax.config.update('jax_enable_x64', True)


def fixed_basis(n):
    indices = [x for x in range(2**n) if x.bit_count() == n//2]
    return ((np.array(indices)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int32)


def test_triangular_input_reflection_matches_physical_120_degree_metric():
    config = FixedSzSpatialViTConfig(geometry='triangular', rows=6, columns=6,
        patch_rows=1, patch_columns=1, embedding_dim=4, head_count=2)
    mapping = physical_to_model_permutation(config)
    np.testing.assert_array_equal(mapping[mapping], np.arange(36))
    distance = _relative_geometry(config)[2]
    for r, c in ((1, 0), (0, 1), (1, 1)):
        assert distance[mapping[0], mapping[r*6+c]] == pytest.approx(1.)
    assert distance[mapping[0], mapping[1*6+5]] == pytest.approx(np.sqrt(3))


@pytest.mark.parametrize('geometry', ['triangular', 'square'])
def test_relabelled_projection_equals_independent_full_reynolds_sum(geometry):
    config = FixedSzSpatialViTConfig(geometry=geometry, rows=4, columns=4,
        embedding_dim=4, head_count=2)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(27), config)
    group = np.asarray(_groups(geometry, 4, 4)['gamma_a1'])
    mapping = physical_to_model_permutation(config)
    bits = jnp.asarray(fixed_basis(16)[[37, 99]])
    log_fn, count = make_physical_spatial_vit_log(config, group)
    actual = np.exp(np.asarray(jax.jit(jax.vmap(log_fn, in_axes=(None, 0)))(params, bits)))
    expected = []
    for x in bits:
        # Full physical group, independently applying the input coordinate map.
        logs = fixed_sz_spatial_vit_log_amplitudes(params, x[group][:, mapping], config)
        expected.append(np.mean(np.exp(np.asarray(logs))))
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
    assert count < len(group)


def test_full_scalar_structure_factor_not_three_zz_for_a_non_singlet():
    bits = fixed_basis(4)
    table = np.array([.2+.1j, -.3+.2j, .1-.4j, -.2-.1j, .4+.3j, -.1+.2j])
    integers = jnp.asarray(bits@(1 << np.arange(3, -1, -1)))
    def log_fn(p, x):
        label = x@jnp.asarray([8, 4, 2, 1])
        return p[jnp.argmax(integers == label)]
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    momenta = {'Gamma': [0, 0], 'M': [.5, 0]}
    scalar, zz = make_scalar_structure_factors(log_fn, coords, momenta)(jnp.asarray(table), jnp.asarray(bits))
    amplitudes = np.zeros(16, complex)
    amplitudes[np.asarray(integers)] = np.exp(table)
    weights = abs(amplitudes[np.asarray(integers)])**2
    weights /= weights.sum()
    for column, k in enumerate(momenta.values()):
        op = .75*np.eye(16, dtype=complex)
        for i, j in combinations(range(4), 2):
            op += .5*np.cos(2*np.pi*(coords[i]-coords[j])@k)*spin_dot_matrix(4, i, j)
        exact = np.vdot(amplitudes, op@amplitudes)/np.vdot(amplitudes, amplitudes)
        np.testing.assert_allclose(weights@np.asarray(scalar[:, column]), exact, atol=2e-13)
    assert abs(weights@np.asarray(scalar[:, 1])-3*(weights@np.asarray(zz[:, 1]))) > 1e-2


def test_coalesced_scalar_queries_match_full_pairs_and_dense_non_singlet_oracle():
    bits = fixed_basis(4)
    table = np.array([.2+.1j, -.3+.2j, .1-.4j, -.2-.1j, .4+.3j, -.1+.2j])
    integers = bits@np.array([8, 4, 2, 1])
    def log_fn(p, x):
        return p[jnp.argmax(jnp.asarray(integers) == x@jnp.asarray([8, 4, 2, 1]))]
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    momenta = {'Gamma': [0, 0], 'M': [.5, 0]}
    plan = scalar_structure_query_plan(bits)
    assert len(plan.configurations) == 6
    assert plan.opposite_spins.sum() == 6*4 < 6*6
    queries = plan.configurations@np.array([8, 4, 2, 1])
    logs = table[np.array([np.flatnonzero(integers == q)[0] for q in queries])]
    scalar, zz = scalar_structure_records(plan, logs, coords, momenta)
    full, diagonal = make_scalar_structure_factors(log_fn, coords, momenta)(jnp.asarray(table), bits)
    np.testing.assert_allclose(scalar, full, atol=3e-14, rtol=0)
    np.testing.assert_allclose(zz, diagonal, atol=3e-14, rtol=0)
    wave = np.zeros(16, complex); wave[integers] = np.exp(table)
    weights = abs(np.exp(table))**2; weights /= weights.sum()
    for column, k in enumerate(momenta.values()):
        operator = .75*np.eye(16, dtype=complex)
        for i, j in combinations(range(4), 2):
            operator += .5*np.cos(2*np.pi*(coords[i]-coords[j])@k)*spin_dot_matrix(4, i, j)
        exact = np.vdot(wave, operator@wave)/np.vdot(wave, wave)
        np.testing.assert_allclose(weights@scalar[:, column], exact, atol=2e-13, rtol=0)
    with pytest.raises(ValueError, match='coherent logs'):
        scalar_structure_records(plan, np.full(len(logs), -np.inf+0j), coords, momenta)
    assert not plan.configurations.flags.writeable


def test_physical_bit_flip_parity_excludes_odd_S_only_when_N_is_multiple_of_four():
    from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns
    for n in (2, 4, 6):
        u, labels = regional_schur_columns(n)
        inverse_bits = np.arange(2**n-1, -1, -1)
        for column, (q, m, _) in enumerate(labels):
            if m != 0:
                continue
            expected = (-1)**(n//2+q//2)
            np.testing.assert_allclose(u[inverse_bits, column], expected*u[:, column],
                                       atol=2e-14, rtol=0)
    # N=36 has N/2 even: exact bit-flip-even and Sz=0 support exclude
    # odd integer S. Hence true non-singlet weight <= <S²>/6, not /2.
    assert 36 % 4 == 0


def test_chunked_full_H_audit_is_identical_including_partial_last_batch():
    from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
    from projects.clebsch_tree_nqs.workflows.spin import _bonds
    spec=toy_spec(); config=validate_spec(spec); params,_=initial_model(spec)
    log_fn,_=model_log(spec,config,_groups('square',2,2)['gamma_a1'])
    local=make_full_heisenberg_ratios(log_fn,_bonds('square',2,2,.125))
    bits=fixed_basis(4)[:5]
    np.testing.assert_allclose(_batched_logs(local,params,bits,2),
                               np.asarray(local(params,bits)),atol=2e-13,rtol=0.)


def test_born_conditional_coherences_reproduce_exact_noninvariant_rdm_with_nodes():
    bits = fixed_basis(4)
    values = np.array([0., 1.+.3j, -.4+.2j, .6-.7j, -.2-.1j, .8+.4j])
    wave = np.zeros(16, complex)
    labels = bits@np.array([8, 4, 2, 1])
    wave[labels] = values
    wave /= np.linalg.norm(wave)
    sites = [2, 0]
    plan = regional_query_plan(bits, sites)
    queries = wave[plan.configurations@np.array([8, 4, 2, 1])]
    with np.errstate(divide='ignore'):
        logs = np.log(queries)
    matrices = born_conditional_regional_matrices(plan, logs)
    weights = abs(wave[labels])**2
    estimated = np.einsum('s,sab->ab', weights, matrices)
    grouped = wave.reshape((2,)*4).transpose((2, 0, 1, 3)).reshape((4, 4))
    np.testing.assert_allclose(estimated, grouped@grouped.conj().T, atol=2e-14)
    np.testing.assert_allclose(np.trace(matrices, axis1=1, axis2=2), 1., atol=2e-14)
    with pytest.raises(ValueError, match='zero conditional norm'):
        born_conditional_regional_matrices(plan, np.full(len(queries), -np.inf+0j))


def toy_spec():
    spec = json.loads(DEFAULT.read_text())
    spec['model'].update(geometry='square', rows=2, columns=2, patch_rows=1,
        patch_columns=1, embedding_dim=4, head_count=2, depth=1)
    spec['stages'] = [{'projection': 'none', 'steps': 2},
        {'projection': 'translation', 'steps': 1}, {'projection': 'gamma_a1', 'steps': 1}]
    spec['quality_milestones'] = [0, 1, 2, 4]
    spec['sampling'].update(chains=2, snapshots_per_update=2,
        proposals_between_snapshots=2, burn_proposals=4)
    spec['audit'].update(chains=2, snapshots=4, proposals_between_snapshots=4,
        burn_proposals=8, query_batch=2, bootstrap_replicates=20)
    spec['sr']['score_chunk_size'] = 2
    spec['momenta'] = {'Gamma': [0, 0], 'M': [.5, 0]}
    spec['regions'] = {'compact4': {'sites': [0, 1, 2, 3], 'left': [0, 1]}}
    return spec


def test_cross_coupling_warm_start_changes_H_not_backbone_or_old_checkpoint(tmp_path):
    import copy
    spec=json.loads((DEFAULT.parent.parent/'remote/spin_spatial_vit_netket_continuation_7600_n36.json').read_text())
    spec.pop('netket_continue_from')
    spec['model'].update(geometry='square',rows=2,columns=2,patch_rows=1,
        patch_columns=1,embedding_dim=4,head_count=2,depth=1)
    spec['stages']=[dict(projection='gamma_a1',steps=4)]
    spec['quality_milestones']=[0,4];spec['momenta']={'Gamma':[0,0]}
    spec['regions']={'compact4':dict(sites=[0,1,2,3],left=[0,1])}
    spec['backend']='native';spec['require_gpu']=False
    config=tmp_path/'parent.json';config.write_text(json.dumps(spec))
    parent=tmp_path/'parent';_,reg=register(config,parent)
    params,_=initial_model(spec);digest=pytree_sha256(params)
    checkpoint=parent/'quality/step00004.pkl'
    meta=dict(completed_steps=4,history=[dict(projection='gamma_a1')]*4,
              registration_sha256=reg['result_sha256'])
    save_checkpoint(checkpoint,dict(model=params),metadata=meta,optimizer_state=dict(step=4))
    before=file_sha256(checkpoint)
    sealed_json(parent/'training.json',dict(completed_steps=4,status='fixed_steps_complete',
        registration_sha256=reg['result_sha256'],final_params_sha256=digest))
    child=copy.deepcopy(spec);child['j2_over_j1']=0.;child['stages']=[dict(projection='gamma_a1',steps=2)]
    child['quality_milestones']=[0,2]
    child['coupling_warm_start_from']=dict(directory=str(parent),parent_j2_over_j1=.125,
        completed_steps=4,checkpoint_path='quality/step00004.pkl',checkpoint_sha256=before,
        registration_sha256=reg['result_sha256'],params_sha256=digest)
    loaded,record=initial_model(child)
    assert pytree_sha256(loaded)==digest and file_sha256(checkpoint)==before
    assert record['optimizer_and_work_counters_reset'] and record['sampler_reinitialized']
    assert record['new_hamiltonian_j2_over_j1']==0.
    bad=copy.deepcopy(child);bad['model']['layer_norm_epsilon']=1e-5
    with pytest.raises(ValueError,match='physical backbone'):initial_model(bad)
    bad=copy.deepcopy(child);bad['coupling_warm_start_from']['params_sha256']='0'*64
    with pytest.raises(ValueError,match='parameter/metadata'):initial_model(bad)
    bad=copy.deepcopy(child);bad['netket_continue_from']={}
    with pytest.raises(ValueError,match='not optimizer continuation'):initial_model(bad)


def test_same_H_repair_uses_immutable_quality_state_not_degraded_live_endpoint(tmp_path):
    import copy
    spec=json.loads((DEFAULT.parent.parent/'remote/spin_spatial_vit_netket_continuation_7600_n36.json').read_text())
    spec.pop('netket_continue_from')
    spec['model'].update(geometry='square',rows=2,columns=2,patch_rows=1,
        patch_columns=1,embedding_dim=4,head_count=2,depth=1)
    spec['stages']=[dict(projection='translation',steps=4)]
    spec['quality_milestones']=[0,4]; spec['momenta']={'Gamma':[0,0]}
    spec['regions']={'compact4':dict(sites=[0,1,2,3],left=[0,1])}
    spec['backend']='native'; spec['require_gpu']=False
    path=tmp_path/'parent.json'; path.write_text(json.dumps(spec))
    parent=tmp_path/'parent'; _,reg=register(path,parent)
    params,_=initial_model(spec); digest=pytree_sha256(params)
    checkpoint=parent/'quality/step00004.pkl'
    save_checkpoint(checkpoint,dict(model=params),metadata=dict(completed_steps=4,
        history=[dict(projection='translation')]*4,registration_sha256=reg['result_sha256']))
    before=file_sha256(checkpoint)
    child=copy.deepcopy(spec)
    child['stages']=[dict(projection='gamma_a1',steps=2)]
    child['quality_milestones']=[0,2]; child['sr']['damping']=.03
    child['repair_parameters_from']=dict(directory=str(parent),completed_steps=4,
        checkpoint_path='quality/step00004.pkl',checkpoint_sha256=before,
        registration_sha256=reg['result_sha256'],params_sha256=digest)
    loaded,record=initial_model(child)
    assert pytree_sha256(loaded)==digest and file_sha256(checkpoint)==before
    assert record['inherited_projection']=='translation' and record['repair_is_new_experiment']
    assert record['optimizer_and_work_counters_reset']
    bad=copy.deepcopy(child); bad['j2_over_j1']=0.
    with pytest.raises(ValueError,match='same-H repair'):initial_model(bad)
    bad=copy.deepcopy(child); bad['repair_parameters_from']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='checkpoint changed'):initial_model(bad)


def test_same_H_capacity_repair_preserves_parent_state_and_source_seals(tmp_path):
    import copy
    from su2_nqs.models.literature_spatial_vit import LiteratureSpatialViTConfig
    spec = json.loads((DEFAULT.parent.parent/'remote/spin_spatial_vit_netket_continuation_7600_n36.json').read_text())
    spec.pop('netket_continue_from')
    spec['model'].update(geometry='square', rows=2, columns=2, patch_rows=1,
        patch_columns=1, embedding_dim=4, head_count=2, depth=1, mlp_ratio=2)
    spec['stages'] = [dict(projection='gamma_a1', steps=4)]
    spec['quality_milestones'] = [0, 4]; spec['momenta'] = {'Gamma': [0, 0]}
    spec['regions'] = {'compact4': dict(sites=[0, 1, 2, 3], left=[0, 1])}
    spec['backend'] = 'native'; spec['require_gpu'] = False
    path = tmp_path/'parent.json'; path.write_text(json.dumps(spec))
    parent = tmp_path/'parent'; _, reg = register(path, parent)
    params, _ = initial_model(spec)
    checkpoint = parent/'quality/step00004.pkl'
    save_checkpoint(checkpoint, dict(model=params), metadata=dict(completed_steps=4,
        history=[dict(projection='gamma_a1')]*4, registration_sha256=reg['result_sha256']))
    before = file_sha256(checkpoint)
    child = copy.deepcopy(spec); child['model']['mlp_ratio'] = 4
    child['capacity_expansion'] = 'state_preserving_mlp_depth'
    child['repair_parameters_from'] = dict(directory=str(parent), completed_steps=4,
        checkpoint_path='quality/step00004.pkl', checkpoint_sha256=before,
        registration_sha256=reg['result_sha256'], params_sha256=pytree_sha256(params))
    loaded, info = initial_model(child)
    assert info['capacity_expanded'] and info['expanded_parameter_count'] > info['parent_parameter_count']
    assert info['initial_raw_amplitude_ratio_max_error'] < 2e-11
    assert info['sampler_reinitialized'] and info['optimizer_and_work_counters_reset']
    assert file_sha256(checkpoint) == before
    group = _groups('square', 2, 2)['gamma_a1']
    old, _ = model_log(spec, LiteratureSpatialViTConfig(**spec['model']), group)
    new, _ = model_log(child, LiteratureSpatialViTConfig(**child['model']), group)
    rows = fixed_basis(4)
    np.testing.assert_allclose(np.exp(jax.vmap(new, in_axes=(None, 0))(loaded, rows)
        -jax.vmap(old, in_axes=(None, 0))(params, rows)), 1, atol=2e-12, rtol=0)
    bad = copy.deepcopy(child); bad.pop('capacity_expansion')
    with pytest.raises(ValueError, match='same-H repair'): initial_model(bad)
    bad = copy.deepcopy(child); bad['model']['embedding_dim'] = 8
    with pytest.raises(ValueError, match='cannot change'): initial_model(bad)
    bad = copy.deepcopy(child); bad.pop('repair_parameters_from')
    with pytest.raises(ValueError, match='repair parent'): initial_model(bad)


def test_preregistered_schedule_and_validation():
    spec = toy_spec()
    validate_spec(spec)
    assert [stage_at(spec, s)[1] for s in range(4)] == ['none', 'none', 'translation', 'gamma_a1']
    spec['audit']['seeds'][0] = spec['training_seed']
    with pytest.raises(ValueError, match='disjoint'):
        validate_spec(spec)
    spec = toy_spec()
    spec['momenta']['K'] = [1/3, 1/3]
    with pytest.raises(ValueError, match='actual physical torus'):
        validate_spec(spec)


def test_full_training_resume_is_identical_and_audit_does_not_change_model(tmp_path):
    config_path = tmp_path/'protocol.json'
    config_path.write_text(json.dumps(toy_spec()))
    a, b = tmp_path/'a', tmp_path/'b'
    partial = train(config_path, a, until_step=1, session_seconds=60)
    assert partial['status'] == 'paused_session_budget'
    with pytest.raises(FileExistsError):
        train(config_path, a)
    resumed = train(config_path, a, resume=True, session_seconds=60)
    uninterrupted = train(config_path, b, session_seconds=60)
    assert resumed['completed_steps'] == uninterrupted['completed_steps'] == 4
    assert resumed['final_params_sha256'] == uninterrupted['final_params_sha256']
    for left, right in zip(jax.tree_util.tree_leaves(load_checkpoint(a/'checkpoint.pkl')['params']),
                           jax.tree_util.tree_leaves(load_checkpoint(b/'checkpoint.pkl')['params'])):
        np.testing.assert_array_equal(left, right)
    before = (a/'checkpoint.pkl').read_bytes()
    summary = audit(config_path, a, steps=[4])
    assert (a/'checkpoint.pkl').read_bytes() == before
    result = json.loads((a/'audits/step00004/results.json').read_text())
    assert not result['strict_su2']
    assert result['checkpoint_log_replay_error'] == 0
    assert result['sampled_A1_relative_error'] < 1e-11
    assert result['regions']['compact4']['strict_multiplicity_spectrum_reported'] is False
    assert not summary['competitive_accuracy_certified']
    bad = toy_spec(); bad['sr']['damping'] *= 2
    config_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='protocol/numerical source changed'):
        train(config_path, a, resume=True)


def test_energy_workers_and_readonly_aggregation_share_frozen_audit_streams(tmp_path):
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import audit_energy, read_sealed
    spec=toy_spec(); spec['stages']=[dict(projection='gamma_a1',steps=1)]
    spec['quality_milestones']=[0,1]
    path=tmp_path/'protocol.json'; path.write_text(json.dumps(spec))
    output=tmp_path/'run'
    train(path,output,session_seconds=120.)
    before=(output/'checkpoint.pkl').read_bytes()
    with pytest.raises(FileNotFoundError,match='never generates'):
        audit_energy(path,output,aggregate_only=True)
    workers=[audit_energy(path,output,stream_index=i) for i in range(2)]
    result=audit_energy(path,output,aggregate_only=True)
    assert result['full_H_calls']==16
    assert result['energy']['samples']==16
    assert [s['raw_sha256'] for s in result['streams']]==[s['raw_sha256'] for s in workers]
    assert (output/'checkpoint.pkl').read_bytes()==before
    assert not (output/'audits/step00001/results.json').exists()
    audit(path,output,steps=[1])
    full=read_sealed(output/'audits/step00001/results.json')
    assert full['energy']==result['energy']
    assert full['streams']==result['streams']


def test_energy_audit_can_read_frozen_milestone_without_claiming_terminal_completion(tmp_path):
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import audit_energy
    spec = toy_spec(); spec['stages'] = [dict(projection='gamma_a1', steps=4)]
    path = tmp_path/'milestone.json'; path.write_text(json.dumps(spec))
    output = tmp_path/'run'
    train(path, output, until_step=2, session_seconds=120.)
    before = (output/'checkpoint.pkl').read_bytes()
    with pytest.raises(ValueError, match='preregistered'):
        audit_energy(path, output, step=3)
    result = audit_energy(path, output, step=2)
    assert result['step'] == 2 and not result['fixed_terminal_endpoint']
    assert result['full_H_calls'] == 16
    assert (output/'checkpoint.pkl').read_bytes() == before
    assert not (output/'training.json').exists()


def test_cached_physics_reuses_frozen_chains_without_H_sampler_or_parent_writes(tmp_path, monkeypatch):
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import audit_energy, read_sealed
    from su2_nqs.frontier.provenance import file_sha256
    import projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality as workflow
    spec = toy_spec(); spec['stages'] = [dict(projection='gamma_a1', steps=1)]
    spec['quality_milestones'] = [0, 1]
    path = tmp_path/'training_protocol.json'; path.write_text(json.dumps(spec))
    parent = tmp_path/'run'; train(path, parent, session_seconds=120.)
    energy = audit_energy(path, parent)
    registration = read_sealed(parent/'registration.json')
    settings = dict(task='cached_spatial_vit_physics', parent=str(parent),
        output=str(tmp_path/'physics'), step=1,
        parent_registration_sha256=registration['result_sha256'],
        checkpoint_sha256=energy['checkpoint_sha256'], params_sha256=energy['params_sha256'],
        frozen_source_count=len(registration['source_manifest']),
        stream_sha256={str(row['seed']):row['raw_sha256'] for row in energy['streams']},
        snapshot_indices=[0, 2], expected_chains=4, expected_records=8,
        region=dict(sites=[2, 0],left=[0]), query_batch=4,timing_records=4,
        compute_seconds=120., bootstrap_seed=1234,bootstrap_replicates=20,require_gpu=False)
    protocol = tmp_path/'physics_protocol.json'; protocol.write_text(json.dumps(settings))
    before = {str(p):file_sha256(p) for p in parent.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('cached physics attempted new H action or sampling')
    monkeypatch.setattr(workflow, 'make_full_heisenberg_ratios', forbidden)
    monkeypatch.setattr(workflow, 'make_log_exchange_sampler', forbidden)
    monkeypatch.setattr(workflow, 'make_collector', forbidden)
    result = audit_cached_physics(protocol)
    assert {str(p):file_sha256(p) for p in parent.rglob('*') if p.is_file()} == before
    assert result['new_H_calls'] == result['new_samples'] == result['new_training'] == 0
    assert result['params_sha256'] == energy['params_sha256']
    assert result['independent_chains'] == 4 and result['records_per_chain'] == 2
    assert result['region']['sites'] == [2, 0]
    assert result['timing']['independent_scalar_replay_maximum_absolute_error'] < 2e-13
    assert not result['region']['strict_multiplicity_spectrum_reported']
    assert not result['region']['input_repaired']
    with pytest.raises(FileExistsError, match='immutable'):
        audit_cached_physics(protocol)

    # The regional-only branch reuses the same frozen chains/ALL weights,
    # without repeating all-pair scalar queries or loading optional samplers.
    from su2_nqs.frontier.parameter_artifact import save_parameter_artifact
    params = load_checkpoint(parent/'quality/step00001.pkl')['params']['model']
    portable = tmp_path/'portable/model.json'
    identity = dict(purpose='Read-only physical analysis of a frozen trained model; not initialization or a resumable optimizer checkpoint',
        step=1, checkpoint_sha256=energy['checkpoint_sha256'],
        params_sha256=energy['params_sha256'], registration_sha256=registration['result_sha256'])
    save_parameter_artifact(portable, params, identity=identity)
    portable.with_name('metadata.json').write_text(json.dumps(dict(
        checkpoint_sha256=energy['checkpoint_sha256'], metadata=dict(
            registration_sha256=registration['result_sha256'], history=[dict(projection='gamma_a1')]))))
    settings.pop('region')
    settings.update(output=str(tmp_path/'regions'), frozen_parameter_artifact=str(portable),
        regions=dict(pair=dict(sites=[2,0],left=[0]),other=dict(sites=[1,3],left=[0])),
        problem=dict(geometry='square',rows=2,columns=2,j2=spec['j2_over_j1'],
            boundary='PBC_both_physical',regions=dict(pair=[2,0],other=[1,3]),
            left_sites=dict(pair=[0],other=[0])))
    protocol2 = tmp_path/'regions_protocol.json'; protocol2.write_text(json.dumps(settings))
    monkeypatch.setattr(workflow, 'scalar_structure_query_plan', forbidden)
    monkeypatch.setattr(workflow, 'load_checkpoint', forbidden)
    regional = audit_cached_physics(protocol2)
    assert regional['new_H_calls'] == regional['new_samples'] == regional['new_training'] == 0
    assert regional['params_sha256'] == energy['params_sha256']
    assert {str(p):file_sha256(p) for p in parent.rglob('*') if p.is_file()} == before
    with np.load(tmp_path/'regions/physics.npz') as archive:
        for name, region in regional['regions'].items():
            rho = archive[name+'__rho']
            np.testing.assert_allclose(rho,archive[name+'__per_record_matrices'].mean(0),atol=2e-14)
            assert sum(v for k,v in region['scalars'].items() if k.startswith('joint_')) == pytest.approx(1.,abs=2e-14)
            assert not region['strict_multiplicity_spectrum_reported']

    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import materialize_cached_plan
    plan=dict(settings,materialized_config=str(tmp_path/'future.json'))
    materialized=materialize_cached_plan(plan)
    assert materialized['params_sha256']==settings['params_sha256']
    assert materialized['stream_sha256']==settings['stream_sha256']
    plan['step']=2
    with pytest.raises(ValueError,match='endpoint'):materialize_cached_plan(plan)
    raw=dict(settings,output=str(tmp_path/'raw'),raw_regional_only=True)
    raw_path=tmp_path/'raw.json';raw_path.write_text(json.dumps(raw))
    raw_result=audit_cached_physics(raw_path)
    assert raw_result['new_samples']==raw_result['new_H_calls']==0
    with np.load(tmp_path/'raw/physics.npz') as data, np.load(tmp_path/'regions/physics.npz') as old:
        np.testing.assert_allclose(data['pair__rho'],old['pair__rho'],atol=2e-14,rtol=0)
    assert raw_result['regions']['pair']['raw_only']


def test_cached_registered_milestone_is_independent_of_mutable_training(tmp_path, monkeypatch):
    import projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality as workflow
    from su2_nqs.frontier.provenance import file_sha256
    spec = toy_spec()
    spec['stages'] = [dict(projection='gamma_a1', steps=2)]
    spec['quality_milestones'] = [0, 1, 2]
    path = tmp_path/'training_protocol.json'; path.write_text(json.dumps(spec))
    parent = tmp_path/'run'; train(path, parent, session_seconds=120.)
    workflow.audit(path, parent, steps=[1])
    plan = dict(parent=str(parent), step=1, allow_registered_milestone=True,
        output=str(tmp_path/'physics'), materialized_config=str(tmp_path/'materialized.json'),
        problem=dict(geometry='square', rows=2, columns=2, j2=spec['j2_over_j1']),
        snapshot_indices=[0, 2], region=dict(sites=[2, 0], left=[0]),
        query_batch=4, timing_records=4, compute_seconds=120.,
        bootstrap_seed=1234, bootstrap_replicates=20, require_gpu=False)
    with pytest.raises(ValueError, match='endpoint'):
        workflow.materialize_cached_plan(dict(plan, allow_registered_milestone=False))
    settings = workflow.materialize_cached_plan(plan)
    checkpoint = parent/'quality/step00001.pkl'
    assert settings['checkpoint_sha256'] == file_sha256(checkpoint)
    # A latest-run record can change or be absent during ongoing training.
    # The frozen milestone neither reads nor writes it.
    (parent/'training.json').unlink()
    assert workflow.materialize_cached_plan(plan) == settings
    before = {str(p):file_sha256(p) for p in parent.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('milestone readout attempted new H action or sampling')
    for name in ('make_full_heisenberg_ratios', 'make_log_exchange_sampler', 'make_collector'):
        monkeypatch.setattr(workflow, name, forbidden)
    protocol = tmp_path/'physics_protocol.json'; protocol.write_text(json.dumps(settings))
    result = workflow.audit_cached_physics(protocol)
    assert result['step'] == 1
    assert result['new_H_calls'] == result['new_samples'] == result['new_training'] == 0
    assert {str(p):file_sha256(p) for p in parent.rglob('*') if p.is_file()} == before
    # Full milestone audits omit terminal-training parameter_count. The
    # multi-region branch must count the actual frozen weights, too.
    regional_settings = dict(settings, output=str(tmp_path/'milestone_regions'))
    regional_settings.pop('region')
    regional_settings['regions'] = dict(pair=dict(sites=[2,0],left=[0]))
    regional_settings['problem'] = dict(geometry='square',rows=2,columns=2,j2=spec['j2_over_j1'],
        boundary='PBC_both_physical',regions=dict(pair=[2,0]),left_sites=dict(pair=[0]))
    regional_protocol = tmp_path/'regional_protocol.json'
    regional_protocol.write_text(json.dumps(regional_settings))
    regional_result = workflow.audit_cached_physics(regional_protocol)
    frozen_params = load_checkpoint(checkpoint)['params']['model']
    assert regional_result['model_parameter_count'] == sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(frozen_params))
    assert regional_result['new_H_calls'] == regional_result['new_samples'] == regional_result['new_training'] == 0
    assert {str(p):file_sha256(p) for p in parent.rglob('*') if p.is_file()} == before
    with pytest.raises(ValueError, match='registered'):
        workflow.materialize_cached_plan(dict(plan, step=0))
    audit_path = parent/'audits/step00001/results.json'
    corrupted = workflow.read_sealed(audit_path)
    corrupted['checkpoint_sha256'] = 'wrong'
    workflow.sealed_json(audit_path, {k:v for k,v in corrupted.items() if k != 'result_sha256'})
    with pytest.raises(ValueError, match='identity'):
        workflow.materialize_cached_plan(plan)


def test_six_site_cached_protocol_preserves_joint_coherence_and_chain_errors():
    from projects.clebsch_tree_nqs.workflows.spin_su2_analysis import cached_regional_readout
    from su2_nqs.benchmarks.spin_partition_readout import _joint_projectors
    rng = np.random.default_rng(452)
    v = rng.normal(size=(16,64))+1j*rng.normal(size=(16,64))
    v /= np.linalg.norm(v,axis=1)[:,None]
    matrices = v[:,:,None]*v[:,None,:].conj()
    chains = np.repeat(np.arange(4),4)
    result, arrays = cached_regional_readout(matrices,chains,
        dict(sites=list(range(6)),left=[0,1,2],compact_triangular_2x3=True))
    p = dict(_joint_projectors(6,(0,1,2)))[(3,3,0)]
    measurements = np.einsum('cab,ba->c',matrices,p).real
    means = measurements.reshape((4,4)).mean(1)
    assert result['scalars']['joint_qL3_qR3_qT0'] == pytest.approx(means.mean(),abs=2e-14)
    assert result['scalar_whole_chain_SE']['joint_qL3_qR3_qT0'] == pytest.approx(means.std(ddof=1)/2,abs=2e-14)
    assert np.linalg.norm(arrays['sigma_q0']-np.diag(np.diag(arrays['sigma_q0']))) > .001
    assert 'triangle_conditional_A2_axis_correlation' in result['scalars']
    assert not result['input_repaired'] and not result['strict_multiplicity_spectrum_reported']


def test_archived_full_neural_parent_is_exactly_reused_without_reinitialization(monkeypatch):
    import projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality as workflow
    spec = json.loads(CONTINUATION_DEFAULT.read_text())
    config = validate_spec(spec)
    def forbidden(*args, **kwargs):
        raise AssertionError('cold initialization was attempted')
    monkeypatch.setattr(workflow, 'init_fixed_sz_spatial_vit', forbidden)
    params, record = initial_model(spec)
    assert pytree_sha256(params) == spec['initial_checkpoint']['params_sha256']
    assert not record['reinitialized_parameters']
    assert checkpoint_projection(dict(history=[], parent=record)) == 'gamma_a1'
    group = np.asarray(_groups('triangular', 6, 6)['gamma_a1'])
    log_fn, count = model_log(spec, config, group)
    saved = load_checkpoint(workflow.ROOT/spec['initial_checkpoint']['checkpoint'])
    bits = saved['params']['chains'][:3]
    actual = np.exp(np.asarray(jax.jit(jax.vmap(log_fn, in_axes=(None, 0)))(params, bits)))
    exact = []
    for x in bits:
        # Independent full sum, original physical bit order: NO input reflection.
        logs = fixed_sz_spatial_vit_log_amplitudes(params, x[group], config)
        shift = np.max(np.asarray(logs).real)
        exact.append(np.exp(shift)*np.mean(np.exp(np.asarray(logs)-shift)))
    np.testing.assert_allclose(actual, exact, rtol=3e-12, atol=0)
    assert count == 48
    bad = dict(spec, coordinate_convention='physical120')
    with pytest.raises(ValueError, match='unchanged legacy'):
        validate_spec(bad)
    bad = dict(spec, j2_over_j1=0.)
    with pytest.raises(ValueError, match='identity changed'):
        initial_model(bad)


def test_parent_fork_resume_uses_fresh_sampler_and_preserves_sealed_source(tmp_path, monkeypatch):
    from dataclasses import asdict
    import projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality as workflow
    # Isolated tiny exact oracle, never mutate or re-sign a real old experiment.
    spec = toy_spec()
    spec['stages'] = [{'projection': 'gamma_a1', 'steps': 4, 'sr': {'damping': .03}}]
    spec['coordinate_convention'] = 'legacy_checkpoint'
    config = validate_spec(spec)
    model = init_fixed_sz_spatial_vit(jax.random.PRNGKey(29), config)
    checkpoint = tmp_path/'parent.pkl'
    save_checkpoint(checkpoint, dict(model=model), optimizer_state={'step': 2})
    old_bytes = checkpoint.read_bytes()
    trained = sealed_json(tmp_path/'train.json', dict(checkpoint_sha256=file_sha256(checkpoint),
        final_params_sha256=pytree_sha256(model), completed_steps=2))
    source = sealed_json(tmp_path/'source.json', dict(model_config=asdict(config),
        j2_over_j1=spec['j2_over_j1'], parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(model))))
    spec['initial_checkpoint'] = dict(checkpoint='parent.pkl', checkpoint_sha256=file_sha256(checkpoint),
        training_result='train.json', training_result_sha256=trained['result_sha256'],
        result='source.json', result_sha256=source['result_sha256'], params_sha256=pytree_sha256(model))
    # Only the trusted-parent root changes; retain the real source-manifest root.
    original = workflow.initial_model
    def local_parent(s):
        with monkeypatch.context() as context:
            context.setattr(workflow, 'ROOT', tmp_path)
            return original(s)
    monkeypatch.setattr(workflow, 'initial_model', local_parent)
    path = tmp_path/'protocol.json'; path.write_text(json.dumps(spec))
    a, b = tmp_path/'a', tmp_path/'b'
    train(path, a, until_step=1, session_seconds=60)
    resumed = train(path, a, resume=True, session_seconds=60)
    full = train(path, b, session_seconds=60)
    assert resumed['final_params_sha256'] == full['final_params_sha256']
    assert load_checkpoint(a/'quality/step00000.pkl')['metadata']['parent']['params_sha256'] == pytree_sha256(model)
    audit(path, a, steps=[0])
    report = json.loads((a/'audits/step00000/results.json').read_text())
    assert report['training_projection'] == 'gamma_a1'
    assert not report['raw_stage_projection_after_training']
    assert checkpoint.read_bytes() == old_bytes


def test_pooling_layer_norm_order_is_not_a_uniform_wavefunction_gauge():
    hidden = jnp.asarray(np.random.default_rng(41).normal(size=(5, 9, 6)))
    norm = {'scale': jnp.ones(6), 'bias': jnp.zeros(6)}
    old = np.asarray(_layer_norm(hidden, norm).sum(axis=1))
    ref = np.asarray(_layer_norm(hidden.sum(axis=1), norm))
    # Even a configuration-specific scalar cannot generally reproduce directions.
    scale = np.sum(old*ref, axis=1)/np.sum(ref**2, axis=1)
    assert np.linalg.norm(old-scale[:, None]*ref) > .5


def test_numerical_solver_recovery_preserves_full_state_and_total_budget(tmp_path):
    spec=toy_spec(); old_path=tmp_path/'original.json'
    old_path.write_text(json.dumps(spec)); old_output=tmp_path/'original'
    original=train(old_path,old_output,until_step=1,session_seconds=60)
    old_bytes=(old_output/'checkpoint.pkl').read_bytes()
    old_reg=json.loads((old_output/'registration.json').read_text())
    stable=json.loads(json.dumps(spec)); stable['sr']['linear_solver']='svd'
    stable['continuation_from']=dict(output=str(old_output),
        checkpoint_sha256=file_sha256(old_output/'checkpoint.pkl'),
        registration_sha256=old_reg['result_sha256'],progress_sha256=original['result_sha256'],
        charged_overhead_seconds=3.,discarded_and_replay_full_H_configs=4)
    # Trusted path root includes this isolated test directory, not real artifacts.
    import projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality as workflow
    from unittest.mock import patch
    with patch.object(workflow,'ROOT',tmp_path):
        state,metadata=recovery_state(stable,{'result_sha256':'new-numerics'})
        saved=load_checkpoint(old_output/'checkpoint.pkl')
        for a,b in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(saved['params'])):
            np.testing.assert_array_equal(a,b)
        assert metadata['completed_steps']==1
        assert metadata['active_training_seconds']==original['active_training_seconds']+3.
        assert metadata['history']==original['history']
        assert metadata['registration_sha256']=='new-numerics'
        bad=json.loads(json.dumps(stable));bad['audit']['seeds'][0]+=77
        with pytest.raises(ValueError,match='preserve the physical training protocol'):
            recovery_state(bad,{'result_sha256':'new-numerics'})
    assert (old_output/'checkpoint.pkl').read_bytes()==old_bytes
