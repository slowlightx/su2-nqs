"""Prospective full-rank dictionary VMC with independent physical-step calibration.

The sealed J0 pilot remains unchanged. This controller reuses its independently
tested SR/sampler/statistics primitives, but selects the actual Hamiltonian from
the campaign configuration. Calibration evaluates amplitudes only, never a
candidate energy; final Born audits use a third, untouched random stream.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_dictionary_vmc_local import (
    make_prepared_sr_update, make_record_sampler, summarize_born_audit,
)
from projects.clebsch_tree_nqs.workflows.triangle_scale_cost import checked_scale_parent, scale_arity_pair
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_dictionary_attention import (
    TriangleDictionaryAttentionConfig, dictionary_attention_bank,
    dictionary_layout, dictionary_parameter_count, init_dictionary_attention_from_bank,
)
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.physical_step_calibration import make_fs_calibrator
from su2_nqs.vmc.replay_validation import compare_replay
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('free', 'linear', 'attention')


def campaign_bonds(spec: dict) -> tuple:
    """The actual complete periodic J1/J2 graph, not the sealed J0 graph."""
    length = spec['model']['length']
    if length not in (6, 12):
        raise ValueError('bounded optimization comparison supports only N=36 or144')
    coupling = float(spec['j2_over_j1'])
    if not math.isfinite(coupling) or coupling < 0:
        raise ValueError('finite nonnegative J2/J1 required')
    return tuple(triangular_j1j2_bonds(length, length, j2=coupling, periodic=True))


def physical_runtime(spec: dict, model: MixedClebschConfig) -> tuple:
    """Construct one complete coherent A1 amplitude and matching full-H action."""
    length = model.length
    bonds = campaign_bonds(spec)
    images = np.asarray(triangular_hnf_gamma_a1_group(((length, 0), (0, length))), np.int32)
    amplitude = make_feature_lut_projected_amplitude(model, images, chunk=spec['projection_chunk'])
    action = make_cached_projected_heisenberg_action(model, images, bonds,
        projection_chunk=spec['projection_chunk'])

    @jax.jit
    def evaluate(prepared, rows):
        def one(row):
            psi, hp = action(prepared, row[None])
            return jnp.stack((psi[0], hp[0]))
        values = jax.lax.map(one, rows, batch_size=spec['action_chunk'])
        return values[:, 0], values[:, 1]
    return images, bonds, amplitude, evaluate


def initialize_random_streams(seeds: dict, *, n_sites: int, training_chains: int,
                              calibration_chains: int) -> tuple:
    """Independent saved training/calibration streams; final audits are separate."""
    if seeds['training_seed'] == seeds['calibration_seed']:
        raise ValueError('training and calibration RNG seeds must differ')
    key, init = jax.random.split(jax.random.PRNGKey(seeds['training_seed']))
    cal_key, cal_init = jax.random.split(jax.random.PRNGKey(seeds['calibration_seed']))
    states = initialize_spin_exchange_chains(init, n_chains=training_chains,
        n_sites=n_sites, n_up=n_sites//2)
    cal_states = initialize_spin_exchange_chains(cal_init, n_chains=calibration_chains,
        n_sites=n_sites, n_up=n_sites//2)
    return states, key, cal_states, cal_key


def _plain(tree):
    return jax.tree.map(lambda value: np.asarray(value).item() if np.asarray(value).ndim == 0
                        else np.asarray(value).tolist(), tree)


def _bank_error(actual, expected) -> float:
    delta = sum(np.sum(abs(np.asarray(a)-np.asarray(b))**2)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True))
    norm = sum(np.sum(abs(np.asarray(a))**2) for a in jax.tree.leaves(expected))
    return float(np.sqrt(delta/max(norm, 1e-300)))


def _check_registration(registration: dict, config_path: Path) -> None:
    if (registration['sha256'] != canonical_sha256({key: value for key, value in registration.items()
                                                   if key != 'sha256'}) or
            registration['config_sha256'] != file_sha256(config_path) or any(
            file_sha256(ROOT/path) != digest for path, digest in registration['sources'].items())):
        raise ValueError('registered config/source changed; do not modify a running campaign')


def _arm_functions(spec, model, parent, arm, seeds, amplitude, evaluate, bonds):
    cfg = TriangleDictionaryAttentionConfig(model=model,
        architecture='linear' if arm == 'linear' else 'attention', **spec['generator'])
    if any(row['rank'] != len(row['node_ids']) for row in dictionary_layout(cfg)):
        raise ValueError('first optimization comparison requires full dictionary rank')
    if arm == 'free':
        initial, note = parent, dict(exact_same_bank_expected=True)
        generate = lambda params: params
    else:
        initial, note = init_dictionary_attention_from_bank(
            jax.random.PRNGKey(seeds['generator_seed']), parent, cfg)
        generate = lambda params: dictionary_attention_bank(params, cfg)
    error = _bank_error(generate(initial), parent)
    if not np.isfinite(error) or error > spec.get('tolerance', 2e-11):
        raise ArithmeticError('all arms must start at the unmodified same physical bank')
    prepare = lambda params: prepare_feature_lut_bank(generate(params), model)
    prepare_jit = jax.jit(prepare)
    logs = lambda bank, bits: jnp.log(amplitude(bank, bits).astype(jnp.complex128))
    update = make_prepared_sr_update(prepare, logs, initial, **spec['sr'])
    cal_options = spec['calibration']
    calibrate = make_fs_calibrator(prepare, logs, initial,
        **{key: cal_options[key] for key in ('target_fs_sine', 'relative_tolerance',
            'max_factor', 'bracket_steps', 'bisection_steps', 'minimum_ess_fraction')})
    edges = tuple((a, b) for a, b, J in bonds if J != 0)
    samplers = {}
    for name in ('training', 'calibration'):
        options = spec[name]
        for phase, burn in (('initial', options['initial_burn_proposals']),
                            ('refresh', options['refresh_proposals'])):
            samplers[name, phase] = make_record_sampler(amplitude,
                records=options['records_per_chain'], burn_proposals=burn,
                proposals_between=options['proposals_between'], exchange_edges=edges, **spec['sampler'])
    log_batch = jax.jit(jax.vmap(logs, in_axes=(None, 0)))
    n_sites = model.length**2

    def transition(params, states, key, cal_states, cal_key, step):
        bank = prepare_jit(params)
        phase = 'initial' if step == 0 else 'refresh'
        states, key, rows, sampling = jax.block_until_ready(samplers['training', phase](bank, key, states))
        cal_states, cal_key, cal_rows, cal_sampling = jax.block_until_ready(
            samplers['calibration', phase](bank, cal_key, cal_states))
        samples, cal_samples = rows.reshape(-1, n_sites), cal_rows.reshape(-1, n_sites)
        psi, hp = jax.block_until_ready(evaluate(bank, samples))
        energies = hp/psi
        if np.any(np.asarray(psi) == 0) or not np.isfinite(np.asarray(energies)).all():
            raise FloatingPointError('nonfinite/zero Born full-H sample centers')
        candidate, sr = jax.block_until_ready(update(params, samples, jax.lax.stop_gradient(energies)))
        changed, calibration = jax.block_until_ready(calibrate(params, candidate, cal_samples))
        train_info = sampled_overlap_diagnostics(log_batch(prepare_jit(changed), samples)-log_batch(bank, samples))
        # This independent check guards amplification on a badly generalized
        # training direction; it never evaluates H at the proposed endpoint.
        passed = (bool(calibration['accepted']) and bool(sr['accepted']) and
            np.isfinite(float(train_info['sampled_fs_sine'])) and
            float(train_info['sampled_fs_sine']) <= cal_options['maximum_training_fs_sine'] and
            np.isfinite(float(train_info['reweighting_ess_fraction'])) and
            float(train_info['reweighting_ess_fraction']) >= cal_options['minimum_ess_fraction'])
        if not passed:
            changed = params
        diagnostics = dict(sr=sr, calibration=calibration, accepted=passed,
            calibrated_training_fs_sine=train_info['sampled_fs_sine'],
            committed_training_fs_sine=jnp.where(passed, train_info['sampled_fs_sine'], 0.),
            committed_calibration_fs_sine=jnp.where(passed, calibration['sampled_fs_sine'], 0.),
            calibrated_training_ess_fraction=train_info['reweighting_ess_fraction'],
            training_sampler=sampling, calibration_sampler=cal_sampling,
            full_H_centers=len(samples), full_H_action_batches=1,
            calibration_H_calls=0)
        return changed, states, key, cal_states, cal_key, rows, energies, diagnostics
    return cfg, initial, note, error, generate, transition


def run(config_path: Path, seed_index: int, *, resume: bool = False,
        until_step: int | None = None, replay: bool = False) -> dict:
    """Run/resume fixed endpoints, or replay one uncommitted next transition."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('registered accelerator comparison cannot fall back to CPU')
    if isinstance(seed_index, bool) or not 0 <= seed_index < len(spec['seeds']):
        raise ValueError('seed index must identify a registered repetition')
    seeds = spec['seeds'][seed_index]
    used_random_streams = [seeds['training_seed'], seeds['calibration_seed'],
                          *[seeds['audit_seed']+i for i in range(4)]]
    if len(set(used_random_streams)) != len(used_random_streams):
        raise ValueError('training/calibration and all four final-audit seeds must be distinct')
    model = MixedClebschConfig(**spec['model'])
    bonds = campaign_bonds(spec)
    source = ROOT/spec['source']['checkpoint']
    if file_sha256(source) != spec['source']['checkpoint_sha256']:
        raise ValueError('parent checkpoint hash mismatch')
    source_checkpoint = load_checkpoint(source)
    parent = source_checkpoint['params']
    if pytree_sha256(parent) != spec['source']['params_sha256']:
        raise ValueError('parent parameter hash mismatch')
    source_registration = source.parent/'registration.json'
    original_registration = json.loads(source_registration.read_text())
    if original_registration['sha256'] != canonical_sha256({
            key: value for key, value in original_registration.items() if key != 'sha256'}):
        raise ValueError('parent registration self-seal mismatch')
    if ('registration_sha256' in spec['source'] and
            original_registration['sha256'] != spec['source']['registration_sha256']):
        raise ValueError('parent registration seal mismatch')
    source_audit = source.parent/'terminal_audit.json'
    original_audit = json.loads(source_audit.read_text())
    if ('audit_sha256' in spec['source'] and file_sha256(source_audit) != spec['source']['audit_sha256']):
        raise ValueError('parent audit file hash mismatch')
    if (original_audit['params_sha256'] != spec['source']['params_sha256'] or
            original_audit['registration_sha256'] != original_registration['sha256']):
        raise ValueError('parent audit does not identify the same state and registration')
    source_metadata = source_checkpoint.get('metadata', {})
    for field, expected in (('params_sha256', spec['source']['params_sha256']),
                            ('registration_sha256', original_registration['sha256'])):
        if field in source_metadata and source_metadata[field] != expected:
            raise ValueError('parent checkpoint metadata seal mismatch: '+field)
    transfer = None
    if spec['source'].get('mode') == 'size_initialization':
        if model.length != 12:
            raise ValueError('size-initialized dictionary trial is restricted to N144')
        source_checkpoint = checked_scale_parent(spec['source'], target_j2=spec['j2_over_j1'])
        pair, transfer = scale_arity_pair(source_checkpoint['params'],
            MixedClebschConfig(**spec['source']['model']), model,
            source_arm=spec['source']['arm'], rank=spec['source']['binary_rank'],
            key=jax.random.PRNGKey(spec['source']['initialization_seed']))
        parent = arity_clebsch_bank(pair['ternary_joint'], model, 'ternary_joint',
                                   spec['source']['binary_rank'])
    elif (original_registration['config']['model'] != spec['model'] or
              original_registration['config']['j2_over_j1'] != spec['j2_over_j1'] or
              original_registration['config']['arm'] != 'ternary_joint'):
        raise ValueError('need a same-D, same-Hamiltonian unmodified joint parent')
    output = ROOT/spec['output']/f'seed_{seed_index:02d}'
    registration_path, result_path = output/'registration.json', output/'results.json'
    if output.exists() and not (resume or replay):
        raise FileExistsError('campaign output is immutable; use explicit --resume')
    if (resume or replay) and not registration_path.exists():
        raise FileNotFoundError('cannot resume/replay an unregistered campaign')
    if registration_path.exists():
        registration = json.loads(registration_path.read_text())
        _check_registration(registration, config_path)
        if (registration['source_registration_sha256'] != file_sha256(source_registration) or
                registration['source_audit_sha256'] != file_sha256(source_audit)):
            raise ValueError('frozen parent registration/audit changed before resume')
        result = json.loads(result_path.read_text())
    else:
        output.mkdir(parents=True)
        sources = transitive_local_source_manifest([Path(__file__)], root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
        for relative in sources:
            destination = output/'source_snapshots'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, destination)
        registration = dict(config=spec, config_sha256=file_sha256(config_path),
            seed_index=seed_index, seeds=seeds, sources=sources,
            runtime=numerical_runtime_identity(), parent_checkpoint_sha256=file_sha256(source),
            source_registration_sha256=file_sha256(source_registration),
            source_audit_sha256=file_sha256(source_audit))
        registration['sha256'] = canonical_sha256(registration)
        atomic_json(registration_path, registration)
        result = dict(status='registered', seed_index=seed_index, arms={},
            registration_sha256=registration['sha256'], updates_selected_by_audit=False,
            final_audits_paired=False, calibration_is_amplitude_only=True,
            independent_physical_initializations=False, architecture_is_static_geometry_hypernetwork=True,
            source_size_initialization=transfer,
            notes=['Same physical parent; independent training repetitions, not independent physical initializations.',
                   'Matched FS is empirical and cannot certify unseen support or exact global distance.',
                   'Born local-energy variance is not an unconditional full-space H-squared certificate.',
                   'Parameter ridge damping is coordinate dependent; independent FS matches step size, not direction.'])
        atomic_json(result_path, result)
    images, bonds, amplitude, evaluate = physical_runtime(spec, model)
    result.update(full_A1_images=len(images), Hamiltonian_bonds=len(bonds),
        j2_over_j1=spec['j2_over_j1'], backend=jax.default_backend())
    final_step = spec['training']['steps']
    limit = final_step if until_step is None else until_step
    if not isinstance(limit, int) or not 0 <= limit <= final_step:
        raise ValueError('--until-step cannot change the registered total endpoint')
    if replay and result['status'] != 'completed':
        raise ValueError('process replay requires frozen completed endpoints')
    replay_path = output/'process_replay.json'
    if replay and replay_path.exists():
        raise FileExistsError('new-process replay supplement is immutable')
    start, stopped = perf_counter(), []
    previous = {}

    def stop(signum, _frame):
        stopped.append(signum)

    if not replay:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGALRM):
            previous[signum] = signal.signal(signum, stop)
        if spec.get('hard_seconds_cap'):
            signal.alarm(spec['hard_seconds_cap'])

    def save():
        result['session_elapsed_seconds'] = perf_counter()-start
        training_centers = sum(sum(row['full_H_centers'] for row in entry.get('history', []))
                               for entry in result['arms'].values())
        audit_centers = result.get('parent_audit', {}).get('full_H_centers', 0)+sum(
            entry.get('terminal_audit', {}).get('full_H_centers', 0)
            for entry in result['arms'].values())
        replay_centers = sum(entry.get('replay_full_H_centers', 0)
                             for entry in result['arms'].values())
        result['full_H_work'] = dict(training_centers=training_centers,
            untouched_audit_centers=audit_centers, uncommitted_same_process_replay_centers=replay_centers,
            total_centers=training_centers+audit_centers+replay_centers,
            calibration_H_calls=0, a_full_H_call_means_one_complete_H_action_per_center=True)
        result['wall_clock_cap_stops_at_atomic_step_boundary'] = True
        atomic_json(result_path, result)

    def audit(bank, label, audit_seed):
        directory = output/label
        path = directory/'audit.json'
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['params_sha256'] != pytree_sha256(bank):
                raise ValueError('existing immutable audit belongs to a different endpoint')
            return existing
        options = spec['audit']
        tick = perf_counter()
        sampler = make_record_sampler(amplitude, records=options['records_per_chain'],
            burn_proposals=options['burn_proposals'], proposals_between=options['proposals_between'],
            exchange_edges=tuple((a, b) for a, b, J in bonds if J != 0), **spec['sampler'])
        key, init = jax.random.split(jax.random.PRNGKey(audit_seed))
        states = initialize_spin_exchange_chains(init, n_chains=options['chains'],
            n_sites=model.length**2, n_up=model.length**2//2)
        prepared = jax.jit(lambda b: prepare_feature_lut_bank(b, model))(bank)
        states, key, rows, sampling = jax.block_until_ready(sampler(prepared, key, states))
        psi, hp = jax.block_until_ready(evaluate(prepared, rows.reshape(-1, model.length**2)))
        if np.any(np.asarray(psi) == 0) or not np.isfinite(np.asarray(hp/psi)).all():
            raise FloatingPointError('invalid untouched audit center')
        energy = np.asarray(hp/psi).reshape(rows.shape[:2])
        record = summarize_born_audit(energy, np.asarray(rows), n_sites=model.length**2)
        record.update(params_sha256=pytree_sha256(bank), seed=audit_seed,
            elapsed_seconds=perf_counter()-tick, independent_of_training=True,
            independent_of_calibration=True, endpoint_selection_used_audit=False,
            sampler=_plain(sampling), full_H_centers=int(energy.size), full_H_action_batches=1)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory/'audit_samples.npz', bits=np.asarray(rows),
            psi=np.asarray(psi).reshape(rows.shape[:2]), Hpsi=np.asarray(hp).reshape(rows.shape[:2]),
            local_energy=energy, terminal_chains=np.asarray(states), rng=np.asarray(key))
        atomic_json(path, record)
        return record

    replay_result = dict(status='running', updates_committed=0, arms={},
        original_results_sha256=file_sha256(result_path)) if replay else None
    try:
        if not replay and limit == final_step and 'parent_audit' not in result:
            result['parent_audit'] = audit(parent, 'parent', seeds['audit_seed'])
            save()
        for arm_index, arm in enumerate(ARMS):
            cfg, initial, note, error, generate, transition = _arm_functions(
                spec, model, parent, arm, seeds, amplitude, evaluate, bonds)
            path = output/arm/'checkpoint.pkl'
            if path.exists():
                loaded = load_checkpoint(path)
                metadata = loaded['metadata']
                if metadata['registration_sha256'] != registration['sha256']:
                    raise ValueError('checkpoint registration mismatch')
                params = loaded['params']
                states, key = map(jnp.asarray, (metadata['training_states'], metadata['training_rng']))
                cal_states, cal_key = map(jnp.asarray, (metadata['calibration_states'], metadata['calibration_rng']))
                history = metadata['history']
                completed = int(loaded['optimizer_state']['completed_steps'])
                if completed != len(history) or metadata['params_sha256'] != pytree_sha256(params):
                    raise ValueError('checkpoint counter/history/parameter seal mismatch')
                if float(loaded['optimizer_state']['damping']) != spec['sr']['damping']:
                    raise ValueError('saved SR damping differs from the registered optimizer')
            else:
                if replay:
                    raise FileNotFoundError('missing completed checkpoint')
                params, history, completed = initial, [], 0
                states, key, cal_states, cal_key = initialize_random_streams(seeds,
                    n_sites=model.length**2, training_chains=spec['training']['chains'],
                    calibration_chains=spec['calibration']['chains'])
            if replay:
                if (file_sha256(path) != result['arms'][arm]['checkpoint_sha256'] or
                        pytree_sha256(loaded['optimizer_state']) != result['arms'][arm]['optimizer_state_sha256']):
                    raise ValueError('frozen checkpoint/optimizer state changed before process replay')
                actual = transition(params, states, key, cal_states, cal_key, completed)
                fixture = output/arm/'next_transition_fixture.pkl'
                if file_sha256(fixture) != result['arms'][arm]['next_transition_fixture_sha256']:
                    raise ValueError('prospective transition replay fixture changed')
                report = compare_replay(load_checkpoint(fixture)['params'], actual,
                                        tolerance=spec.get('tolerance', 2e-11), diagnostic_prefixes=('[7]',))
                replay_result['arms'][arm] = dict(full_next_transition_bitwise_exact=report['bitwise_exact'],
                    replay_validation=report, next_transition_sha256=pytree_sha256(actual), checkpoint_optimizer_step=completed,
                    additional_full_H_centers=spec['training']['chains']*spec['training']['records_per_chain'])
                replay_result['full_H_work'] = dict(uncommitted_replay_centers=sum(
                    row['additional_full_H_centers'] for row in replay_result['arms'].values()),
                    calibration_H_calls=0)
                atomic_json(replay_path, replay_result)
                if not report['passed']:
                    raise ArithmeticError('new-process replay differs: '+arm)
                continue
            result['arms'].setdefault(arm, dict(status='registered',
                generator_config=asdict(cfg), initialization=note,
                initial_common_bank_relative_error=error,
                parameter_count=dictionary_parameter_count(params),
                parameter_count_by_component={name: dictionary_parameter_count(value)
                    for name, value in params.items()}))
            result['arms'][arm]['history'] = history
            for step in range(completed, limit):
                if stopped:
                    break
                tick = perf_counter()
                params, states, key, cal_states, cal_key, rows, energies, info = transition(
                    params, states, key, cal_states, cal_key, step)
                history.append(dict(step=step+1, elapsed_seconds=perf_counter()-tick,
                    training_energy_per_site=float(energies.real.mean()/model.length**2),
                    training_Born_variance_per_site=float(jnp.mean(abs(energies-energies.mean())**2)/model.length**2),
                    **_plain(info)))
                optimizer = dict(completed_steps=jnp.asarray(step+1, jnp.int64),
                    damping=jnp.asarray(spec['sr']['damping']), last_diagnostics=info)
                save_checkpoint(path, params, optimizer_state=optimizer, metadata=dict(
                    training_states=np.asarray(states), training_rng=np.asarray(key),
                    calibration_states=np.asarray(cal_states), calibration_rng=np.asarray(cal_key),
                    registration_sha256=registration['sha256'], params_sha256=pytree_sha256(params),
                    history=history, model_config=asdict(cfg), arm=arm,
                    fixed_terminal_step=final_step, optimizer='stateless original-parameter ridge SR'))
                np.savez_compressed(output/arm/'last_training_samples.npz',
                    bits=np.asarray(rows), local_energy=np.asarray(energies).reshape(rows.shape[:2]))
                result['arms'][arm].update(status='training', completed_steps=step+1,
                    full_H_training_centers=sum(row['full_H_centers'] for row in history))
                print(arm, json.dumps(history[-1]), flush=True)
                save()
            if len(history) == final_step and result['arms'][arm].get('status') != 'completed':
                endpoint_hash = pytree_sha256(params)
                terminal = audit(generate(params), arm, seeds['audit_seed']+arm_index+1)
                loaded = load_checkpoint(path)
                expected = transition(params, states, key, cal_states, cal_key, final_step)
                metadata = loaded['metadata']
                actual = transition(loaded['params'], jnp.asarray(metadata['training_states']),
                    jnp.asarray(metadata['training_rng']), jnp.asarray(metadata['calibration_states']),
                    jnp.asarray(metadata['calibration_rng']), final_step)
                report = compare_replay(expected, actual, tolerance=spec.get('tolerance', 2e-11),
                                        diagnostic_prefixes=('[7]',))
                result['arms'][arm]['same_process_replay_validation'] = report
                save()
                replay_ok = report['passed']
                if not replay_ok or pytree_sha256(params) != endpoint_hash:
                    raise ArithmeticError('frozen endpoint continuation differs')
                changes = {name: float(np.sqrt(sum(np.sum(abs(np.asarray(a)-np.asarray(b))**2)
                    for a, b in zip(jax.tree.leaves(params[name]), jax.tree.leaves(initial[name]), strict=True))))
                    for name in params}
                save_checkpoint(output/arm/'next_transition_fixture.pkl', expected,
                    metadata=dict(uncommitted=True, endpoint_params_sha256=endpoint_hash))
                result['arms'][arm].update(status='completed', terminal_audit=terminal,
                    endpoint_params_sha256=endpoint_hash, checkpoint_sha256=file_sha256(path),
                    optimizer_state_sha256=pytree_sha256(loaded['optimizer_state']),
                    next_transition_replay_sha256=pytree_sha256(expected),
                    same_process_full_next_transition_bitwise_exact=report['bitwise_exact'],
                    same_process_replay_validation=report,
                    next_transition_fixture_sha256=file_sha256(output/arm/'next_transition_fixture.pkl'),
                    replay_updates_committed=0, replay_full_H_centers=2*spec['training']['chains']*spec['training']['records_per_chain'],
                    trained_parameter_change_norm_by_component=changes)
                save()
            if stopped:
                break
        if replay:
            replay_result.update(status='passed', elapsed_seconds=perf_counter()-start)
            atomic_json(replay_path, replay_result)
            return replay_result
        result['status'] = ('completed' if all(result['arms'].get(a, {}).get('status') == 'completed'
                                              for a in ARMS) else 'stopped_at_safe_boundary')
        result['stop_signals'] = stopped
        save()
        return result
    except BaseException as error:
        if replay:
            replay_result.update(status='failed', error=repr(error))
            atomic_json(replay_path, replay_result)
        else:
            result.update(status='failed', error=repr(error))
            save()
        raise
    finally:
        if not replay:
            signal.alarm(0)
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed-index', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--until-step', type=int)
    parser.add_argument('--replay-completed', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.seed_index, resume=args.resume,
        until_step=args.until_step, replay=args.replay_completed), indent=2))
