"""Bounded native ClebschTree energy optimization after the measured N144 gate.

One controller handles either coupling and both finite-binary/joint arms. It
reuses the exact decoder/cache, native MH, original-parameter stateless SR and
atomic checkpoints. No attention, carrier, new head or new charge convention.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_dictionary_vmc_local import (
    make_prepared_sr_update, make_record_sampler, summarize_born_audit,
)
from projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate import validate_oracle_errors
from projects.clebsch_tree_nqs.workflows.triangle_scale_batch_cost import relative_vector_error
from projects.clebsch_tree_nqs.workflows.triangle_scale_cost import (
    balanced_block_probe_states, checked_scale_parent, scale_arity_pair,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_parameter_count
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.replay_validation import compare_replay
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('binary_truncated', 'ternary_joint')


def make_training_transition(
    prepare: Callable, amplitude: Callable, evaluate: Callable, template: object, *,
    n_sites: int, training: dict, sampler_options: dict, exchange_edges: tuple,
    sr_options: dict,
) -> Callable:
    """One real native sampler/full-H/original-parameter SR transition.

    All optimizer state is explicit. The optimizer is stateless ridge SR, so
    counter/damping/accepted-count/last diagnostics are saved, not invented
    Adam moments. Rejected attempts still consume a fixed registered step.
    """
    if n_sites < 2 or n_sites % 2:
        raise ValueError('an even physical spin-half system is required')
    prepared_fn = jax.jit(prepare)
    logs = lambda bank, bits: jnp.log(amplitude(bank, bits).astype(jnp.complex128))
    updater = make_prepared_sr_update(prepare, logs, template, **sr_options)
    samplers = {phase: make_record_sampler(amplitude,
        records=training['records_per_chain'], burn_proposals=training[field],
        proposals_between=training['proposals_between'], exchange_edges=exchange_edges,
        **sampler_options) for phase, field in (
            ('initial', 'initial_burn_proposals'), ('refresh', 'refresh_proposals'))}

    def transition(params, optimizer, states, key):
        step = int(optimizer['completed_steps'])
        if float(optimizer['damping']) != sr_options['damping']:
            raise ValueError('saved SR damping differs from registered optimizer')
        bank = prepared_fn(params)
        phase = 'initial' if step == 0 else 'refresh'
        states, key, rows, sampling = jax.block_until_ready(samplers[phase](bank, key, states))
        samples = rows.reshape(-1, n_sites)
        psi, hp = jax.block_until_ready(evaluate(bank, samples))
        if np.any(np.asarray(psi) == 0) or not np.isfinite(np.asarray(hp/psi)).all():
            raise FloatingPointError('invalid native Born centers; no amplitude flooring or tail clipping')
        energies = jax.lax.stop_gradient(hp/psi)
        changed, diagnostic = jax.block_until_ready(updater(params, samples, energies))
        optimizer = dict(completed_steps=jnp.asarray(step+1, jnp.int64),
            accepted_updates=jnp.asarray(int(optimizer['accepted_updates'])+int(diagnostic['accepted']), jnp.int64),
            damping=jnp.asarray(sr_options['damping']), last_diagnostics=diagnostic)
        return changed, optimizer, states, key, rows, energies, sampling, diagnostic
    return transition


def checked_training_gate(spec: dict, *, root: Path = ROOT) -> dict:
    """Require a completed, same-H native gate AND explicit measured-budget release."""
    if spec.get('main_agent_training_release') is not True:
        raise ValueError('training is unreleased until the measured cost/budget decision')
    request = spec['native_gate']
    directory = Path(request['directory'])
    for key, filename in (('results', 'results.json'), ('registration', 'registration.json'),
                          ('process_replay', 'process_replay.json'), ('accounting', 'accounting.json')):
        if file_sha256(directory/filename) != request[key+'_sha256']:
            raise ValueError('native gate artifact seal mismatch: '+key)
    gate = json.loads((directory/'results.json').read_text())
    registration = json.loads((directory/'registration.json').read_text())
    replay = json.loads((directory/'process_replay.json').read_text())
    accounting = json.loads((directory/'accounting.json').read_text())
    replay_passed = bool(replay.get('full_next_sampler_bitwise_exact'))
    if not replay_passed and replay.get('replay_validation') is not None:
        report = replay['replay_validation']
        replay_passed = (report.get('protocol') == 'exact_discrete_relative_float_v1'
            and report.get('passed') is True and 0 < report.get('tolerance', 1) <= 2e-11
            and bool(report.get('leaves')) and all(row.get('passed') is True
                and (row.get('bitwise_exact') is True if row['discrete'] else
                     row.get('finite') is True and (row['relative_vector_error'] <= report['tolerance']
                        or row.get('natural_unit_scalar_diagnostic') is True and row.get('shape') == []
                        and row['path'].startswith('[3]') and row['absolute_error'] <= report['tolerance']))
                for row in report['leaves'])
            and file_sha256(directory/'next_sampler_fixture.pkl') == gate['next_sampler_fixture_sha256'])
    # Same physical state is insufficient for a cost certificate: these pure
    # execution choices change compilation, memory and measured throughput.
    measured = registration['config']
    if (any(measured[key] != spec[key] for key in ('projection_chunk', 'action_chunk'))
            or measured['sr']['score_chunk_size'] != spec['sr']['score_chunk_size']):
        raise ValueError('training execution chunks differ from the measured native gate')
    if (any(spec['training'][key] != measured['sampling'][key]
            for key in ('chains', 'records_per_chain')) or
            spec['training']['chains']*spec['training']['records_per_chain'] != gate['native_record_count']):
        raise ValueError('training native-record batch differs from the measured gate')
    cost_request = measured['cost_parent']
    cost_registration_path = Path(cost_request['directory'])/'registration.json'
    if file_sha256(cost_registration_path) != cost_request['registration_sha256']:
        raise ValueError('measured gate cost-parent registration seal changed')
    cost_registration = json.loads(cost_registration_path.read_text())
    if cost_registration['config']['binary_rank'] != spec['binary_rank']:
        raise ValueError('training binary rank differs from the measured native gate')
    if (gate.get('status') != 'completed' or not gate.get('cost_gate_passed')
            or gate['model'] != spec['model'] or gate['j2_over_j1'] != spec['j2_over_j1']
            or gate['optimizer_updates_committed'] != 0
            or gate['registration_sha256'] != registration['sha256']
            or gate['full_A1_images'] != 1728
            or gate['Hamiltonian_bonds'] != (432 if spec['j2_over_j1'] == 0 else 864)
            or replay.get('status') != 'passed' or not replay_passed
            or replay['source_results_sha256'] != request['results_sha256']
            or replay['checkpoint_sha256'] != gate['checkpoint_sha256']
            or file_sha256(directory/'gate_checkpoint.pkl') != gate['checkpoint_sha256']
            or accounting.get('status') != 'completed'
            or registration['sha256'] != canonical_sha256({
                k: v for k, v in registration.items() if k != 'sha256'})):
        raise ValueError('completed same-H native sampler/H/gradient/replay gate required')
    for name, expected in registration['sources'].items():
        if file_sha256(root/name) != expected:
            raise ValueError('native gate numerical source changed: '+name)
    if (not 0 < spec['budget']['maximum_A100_hours'] <= 4 or
            not 0 < spec['budget']['forecast_seconds'] < spec['budget']['maximum_A100_hours']*3600 or
            not 0 < spec['seconds_cap'] <= spec['budget']['maximum_A100_hours']*3600):
        raise ValueError('a measured within-cap per-coupling budget is required')
    return gate


def _plain(tree):
    return jax.tree.map(lambda x: np.asarray(x).item() if np.asarray(x).ndim == 0
                        else np.asarray(x).tolist(), tree)


def _check_registration(registration, config_path):
    if (registration['sha256'] != canonical_sha256({
            k: v for k, v in registration.items() if k != 'sha256'}) or
            registration['config'] != json.loads(config_path.read_text()) or
            file_sha256(config_path) != registration['config_sha256'] or any(
                file_sha256(ROOT/p) != h for p, h in registration['sources'].items())):
        raise ValueError('frozen N144 training config/source changed')


def run(config_path: Path, *, resume: bool = False, until_step: int | None = None,
        replay: bool = False) -> dict:
    """Train exactly the registered endpoint or replay one uncommitted next step."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    gate = checked_training_gate(spec, root=ROOT)
    audit_and_training_seeds = [spec['seeds']['training'], spec['seeds']['initial_audit'],
                               *spec['seeds']['final_audits'].values()]
    if len(set(audit_and_training_seeds)) != len(audit_and_training_seeds):
        raise ValueError('initial/final audit RNGs must be distinct from training and one another')
    model = MixedClebschConfig(**spec['model'])
    if model.length != 12 or spec['training']['steps'] != 2:
        raise ValueError('this first native energy trial is fixed to N144/two attempts')
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('released N144 training must use the actual GPU')
    source_saved = checked_scale_parent(spec['parent'], target_j2=spec['j2_over_j1'])
    source_directory = Path(spec['parent']['checkpoint']).parent
    source_audit = json.loads((source_directory/'terminal_audit.json').read_text())
    source_results = json.loads((source_directory/'results.json').read_text())
    if (source_saved['metadata']['completed_steps'] != spec['parent']['completed_steps'] or
            len(source_saved['metadata']['history']) != spec['parent']['completed_steps'] or
            source_results['completed_steps'] != spec['parent']['completed_steps'] or
            source_results['status'] != 'fixed_steps_complete' or
            source_audit.get('independent_from_training') is not True):
        raise ValueError('source must be the independently audited completed fixed endpoint')
    if spec['parent']['arm'] != 'binary_truncated':
        raise ValueError('common binary/joint initialization requires one audited binary source')
    output = Path(spec['output'])

    def initial_result(pair, note, registration):
        return dict(status='registered', arms={}, registration_sha256=registration['sha256'],
            transfer=note, initial_pair_sha256=pytree_sha256(pair),
            actual_energy_optimization=True, equilibrium_certified=False,
            optimizer='original-parameter stateless ridge SR; explicit counter/damping/last diagnostics',
            source_N36_optimizer_sampler_RNG_reused=False, endpoint_selection_uses_audit=False,
            target_initial_state_is_not_source_complete_projected_state=True,
            shared_initial_audit_between_arity_arms=True,
            notes=['Finite-width decoder, exactS0/full coherentA1, all weighted bonds retained.',
                   '32 training records are correlated8-chain records, not32 independent draws.',
                   'Born local-energy variance does not certify unseen tails or node contributions.',
                   'Two attempts are a development trajectory, not a convergence or variational-win claim.'])
    if output.exists() and not (resume or replay):
        raise FileExistsError('N144 training outputs are immutable; explicit resume required')
    if (resume or replay) and not (output/'registration.json').exists():
        raise FileNotFoundError('cannot resume an unregistered target state')
    if not output.exists():
        pair, note = scale_arity_pair(source_saved['params'], MixedClebschConfig(**spec['parent']['model']),
            model, source_arm='binary_truncated', rank=spec['binary_rank'],
            key=jax.random.PRNGKey(spec['initialization_seed']))
        output.mkdir(parents=True)
        save_checkpoint(output/'initialization.pkl', pair, metadata=dict(trained=False,
            parent=spec['parent'], transfer=note))
        sources = transitive_local_source_manifest([Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'],
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
        registration = dict(config=spec, config_sha256=file_sha256(config_path), sources=sources,
            initial_pair_sha256=pytree_sha256(pair), initialization_checkpoint_sha256=file_sha256(output/'initialization.pkl'),
            runtime=numerical_runtime_identity(), created_utc=datetime.now(timezone.utc).isoformat())
        registration['sha256'] = canonical_sha256(registration)
        atomic_json(output/'registration.json', registration)
        for path in sources:
            destination = output/'source_snapshots'/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/path, destination)
        result = initial_result(pair, note, registration)
        atomic_json(output/'results.json', result)
    else:
        registration = json.loads((output/'registration.json').read_text())
        _check_registration(registration, config_path)
        if file_sha256(output/'initialization.pkl') != registration['initialization_checkpoint_sha256']:
            raise ValueError('common target initialization changed')
        initialized = load_checkpoint(output/'initialization.pkl')
        pair = initialized['params']
        if pytree_sha256(pair) != registration['initial_pair_sha256']:
            raise ValueError('initial target parameter pair seal mismatch')
        if (output/'results.json').exists():
            result = json.loads((output/'results.json').read_text())
        elif not any((output/arm/'checkpoint.pkl').exists() for arm in ARMS):
            # Recover an interruption between atomic registration and initial
            # result writes; this branch is forbidden once training exists.
            result = initial_result(pair, initialized['metadata']['transfer'], registration)
            atomic_json(output/'results.json', result)
        else:
            raise ValueError('training checkpoint exists without its result ledger')
        if result['registration_sha256'] != registration['sha256']:
            raise ValueError('target results identify another registration')
    final_step = spec['training']['steps']
    limit = final_step if until_step is None else until_step
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= final_step:
        raise ValueError('partial endpoint cannot alter the fixed two-attempt budget')
    if replay and result['status'] != 'completed':
        raise ValueError('new-process replay requires completed frozen trained endpoints')
    replay_path = output/'process_replay.json'
    if replay and replay_path.exists():
        raise FileExistsError('trained replay supplement is immutable')
    maps = np.asarray(triangular_hnf_gamma_a1_group(((12, 0), (0, 12))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(12, 12, j2=spec['j2_over_j1'], periodic=True))
    amplitude = make_feature_lut_projected_amplitude(model, maps, chunk=spec['projection_chunk'])
    action = make_cached_projected_heisenberg_action(model, maps, bonds, projection_chunk=spec['projection_chunk'])
    edges = tuple((a, b) for a, b, J in bonds if J != 0)

    @jax.jit
    def evaluate(prepared, rows):
        def one(row):
            psi, hp = action(prepared, row[None])
            return jnp.stack((psi[0], hp[0]))
        values = jax.lax.map(one, rows, batch_size=spec['action_chunk'])
        return values[:, 0], values[:, 1]

    if {arm: mixed_parameter_count(pair[arm]) for arm in ARMS} != gate['parameter_counts']:
        raise ValueError('actual target parameter counts differ from measured gate')
    start, stopped = perf_counter(), []
    previous_budget_seconds = result.get('budget_consumed_seconds', 0.)
    remaining_seconds = min(spec['seconds_cap'],
        spec['budget']['maximum_A100_hours']*3600-previous_budget_seconds)
    if remaining_seconds <= 0:
        raise RuntimeError('registered total training/replay budget exhausted')
    previous = {}
    def stop(signum, _):
        stopped.append(signum)
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        previous[sig] = signal.signal(sig, stop)
    signal.alarm(max(1, int(remaining_seconds)))

    def save():
        result['session_elapsed_seconds'] = perf_counter()-start
        result['budget_consumed_seconds'] = previous_budget_seconds+perf_counter()-start
        result['full_H_work'] = dict(
            training_centers=sum(sum(h['full_H_centers'] for h in a.get('history', [])) for a in result['arms'].values()),
            audit_centers=result.get('initial_audit', {}).get('full_H_centers', 0)+sum(
                a.get('terminal_audit', {}).get('full_H_centers', 0) for a in result['arms'].values()),
            uncommitted_same_process_replay_centers=sum(a.get('replay_H_centers', 0) for a in result['arms'].values()))
        result['wall_clock_stop_is_at_atomic_attempt_boundary'] = True
        atomic_json(output/'results.json', result)

    def bank_for(params, arm):
        return arity_clebsch_bank(params, model, arm, spec['binary_rank'])

    def audit(bank, name, seed):
        directory = output/name
        raw_path = directory/'audit_samples.npz'
        identity = dict(params_sha256=pytree_sha256(bank), seed=seed,
            registration_sha256=registration['sha256'], protocol=spec['audit'])
        if (directory/'audit.json').exists():
            prior = json.loads((directory/'audit.json').read_text())
            if (any(prior.get(k) != v for k, v in identity.items()) or
                    prior['raw_samples_sha256'] != file_sha256(raw_path)):
                raise ValueError('existing audit state/RNG/protocol/raw samples identity changed')
            return prior
        options = spec['audit']
        sampler = make_record_sampler(amplitude, records=options['records_per_chain'],
            burn_proposals=options['burn_proposals'], proposals_between=options['proposals_between'],
            exchange_edges=edges, **spec['sampler'])
        states = jnp.asarray(balanced_block_probe_states(model, count=options['chains'], seed=seed+1))
        prepared = jax.jit(lambda b: prepare_feature_lut_bank(b, model))(bank)
        states, key, rows, sampling = jax.block_until_ready(sampler(prepared, jax.random.PRNGKey(seed), states))
        psi, hp = jax.block_until_ready(evaluate(prepared, rows.reshape(-1, 144)))
        if np.any(np.asarray(psi) == 0) or not np.isfinite(np.asarray(hp/psi)).all():
            raise FloatingPointError('invalid independent audit center; no repair/clipping')
        energies = np.asarray(hp/psi).reshape(rows.shape[:2])
        record = summarize_born_audit(energies, np.asarray(rows), n_sites=144)
        record.update(**identity, independent_of_training=True,
            sampler=_plain(sampling), full_H_centers=int(energies.size),
            fixed_endpoint_not_selected_by_audit=True)
        directory.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(raw_path, bits=np.asarray(rows),
            psi=np.asarray(psi), Hpsi=np.asarray(hp), local_energy=energies,
            terminal_chains=np.asarray(states), rng=np.asarray(key))
        record['raw_samples_sha256'] = file_sha256(raw_path)
        atomic_json(directory/'audit.json', record)
        return record

    replay_result = dict(status='running', arms={}, optimizer_updates_committed=0,
        source_results_sha256=file_sha256(output/'results.json')) if replay else None
    try:
        common = bank_for(pair['binary_truncated'], 'binary_truncated')
        other = bank_for(pair['ternary_joint'], 'ternary_joint')
        differences = [relative_vector_error(np.asarray(a), np.asarray(b)) for a, b in
                       zip(jax.tree.leaves(common), jax.tree.leaves(other), strict=True)]
        validate_oracle_errors({'same_target_initial_bank_leaf_'+str(i): error
                                for i, error in enumerate(differences)}, spec['tolerance'])
        if not replay:
            initial = audit(common, 'initial', spec['seeds']['initial_audit'])
            if 'initial_audit' in result and result['initial_audit'] != initial:
                raise ValueError('initial audit ledger differs from sealed cached audit')
            result['initial_audit'] = initial
            save()
        for index, arm in enumerate(ARMS):
            if stopped:
                if replay:
                    raise TimeoutError('uncommitted replay stopped at atomic phase boundary')
                break
            prepare = lambda p, arm=arm: prepare_feature_lut_bank(bank_for(p, arm), model)
            transition = make_training_transition(prepare, amplitude, evaluate, pair[arm], n_sites=144,
                training=spec['training'], sampler_options=spec['sampler'], exchange_edges=edges,
                sr_options=spec['sr'])
            checkpoint = output/arm/'checkpoint.pkl'
            if checkpoint.exists():
                loaded = load_checkpoint(checkpoint)
                meta = loaded['metadata']
                if (meta['registration_sha256'] != registration['sha256'] or
                        meta['arm'] != arm or meta['fixed_terminal_step'] != final_step or
                        meta['params_sha256'] != pytree_sha256(loaded['params']) or
                        meta['optimizer_sha256'] != pytree_sha256(loaded['optimizer_state']) or
                        meta['sampler_rng_sha256'] != pytree_sha256((
                            jnp.asarray(meta['sampler_states']), jnp.asarray(meta['rng'])))):
                    raise ValueError('trained parameter/optimizer/sampler/RNG seal mismatch')
                params, optimizer = loaded['params'], loaded['optimizer_state']
                states, key = jnp.asarray(meta['sampler_states']), jnp.asarray(meta['rng'])
                history = meta['history']
                if int(optimizer['completed_steps']) != len(history):
                    raise ValueError('trained counter/history mismatch')
            else:
                if replay:
                    raise FileNotFoundError('missing trained endpoint')
                params = pair[arm]
                optimizer = dict(completed_steps=jnp.asarray(0, jnp.int64),
                    accepted_updates=jnp.asarray(0, jnp.int64), damping=jnp.asarray(spec['sr']['damping']))
                states = jnp.asarray(balanced_block_probe_states(model,
                    count=spec['training']['chains'], seed=spec['seeds']['training_initialization']))
                key = jax.random.PRNGKey(spec['seeds']['training'])
                history = []
            if replay:
                if file_sha256(checkpoint) != result['arms'][arm]['checkpoint_sha256']:
                    raise ValueError('frozen trained checkpoint file changed')
                actual = transition(params, optimizer, states, key)
                if stopped:
                    raise TimeoutError('uncommitted replay stopped after native transition')
                fixture = output/arm/'next_transition_fixture.pkl'
                if file_sha256(fixture) != result['arms'][arm]['next_transition_fixture_sha256']:
                    raise ValueError('trained replay fixture changed')
                report = compare_replay(load_checkpoint(fixture)['params'], actual, tolerance=spec['tolerance'],
                    diagnostic_prefixes=("[1]['last_diagnostics']", '[6]', '[7]'))
                passed = report['passed']
                replay_result['arms'][arm] = dict(full_next_sampler_H_SR_bitwise_exact=report['bitwise_exact'],
                    replay_validation=report,
                    trained_optimizer_step=int(optimizer['completed_steps']),
                    additional_full_H_centers=spec['training']['chains']*spec['training']['records_per_chain'])
                atomic_json(replay_path, replay_result)
                if not passed:
                    raise ArithmeticError('independent-process trained full transition differs: '+arm)
                continue
            result['arms'].setdefault(arm, dict(status='registered', parameter_count=mixed_parameter_count(params)))
            result['arms'][arm]['history'] = history
            for step in range(int(optimizer['completed_steps']), limit):
                if stopped:
                    break
                tick = perf_counter()
                params, optimizer, states, key, rows, energies, sampling, diagnostic = transition(
                    params, optimizer, states, key)
                history.append(dict(step=step+1, elapsed_seconds=perf_counter()-tick,
                    training_energy_per_site=float(energies.real.mean()/144),
                    training_Born_variance_per_site=float(jnp.mean(abs(energies-energies.mean())**2)/144),
                    sampling=_plain(sampling), SR=_plain(diagnostic), full_H_centers=int(energies.size)))
                save_checkpoint(checkpoint, params, optimizer_state=optimizer, metadata=dict(
                    registration_sha256=registration['sha256'], params_sha256=pytree_sha256(params),
                    optimizer_sha256=pytree_sha256(optimizer),
                    sampler_rng_sha256=pytree_sha256((states, key)),
                    sampler_states=np.asarray(states), rng=np.asarray(key), history=history,
                    arm=arm, trained=True, fixed_terminal_step=final_step))
                np.savez_compressed(output/arm/'last_training_samples.npz',
                    bits=np.asarray(rows), local_energy=np.asarray(energies).reshape(rows.shape[:2]))
                result['arms'][arm].update(status='training', completed_steps=step+1,
                    accepted_updates=int(optimizer['accepted_updates']))
                print(arm, json.dumps(history[-1]), flush=True)
                save()
            # A signal during the last native step must not trigger long
            # terminal-audit/replay work; the optimized checkpoint is already safe.
            if stopped:
                break
            if len(history) == final_step and result['arms'][arm].get('status') != 'completed':
                endpoint_hash = pytree_sha256(params)
                terminal = audit(bank_for(params, arm), arm, spec['seeds']['final_audits'][arm])
                if stopped:
                    save()
                    break
                expected = transition(params, optimizer, states, key)
                if stopped:
                    save()
                    break
                restored = load_checkpoint(checkpoint)
                actual = transition(restored['params'], restored['optimizer_state'],
                    jnp.asarray(restored['metadata']['sampler_states']), jnp.asarray(restored['metadata']['rng']))
                report = compare_replay(expected, actual, tolerance=spec['tolerance'],
                    diagnostic_prefixes=("[1]['last_diagnostics']", '[6]', '[7]'))
                result['arms'][arm]['same_process_replay_validation'] = report
                save()
                passed = report['passed']
                if not passed or pytree_sha256(params) != endpoint_hash:
                    raise ArithmeticError('trained same-process full transition replay differs')
                save_checkpoint(output/arm/'next_transition_fixture.pkl', expected,
                    metadata=dict(uncommitted=True, endpoint_params_sha256=endpoint_hash))
                result['arms'][arm].update(status='completed', terminal_audit=terminal,
                    endpoint_params_sha256=endpoint_hash, checkpoint_sha256=file_sha256(checkpoint),
                    trained_optimizer_sampler_full_transition_replay_bitwise_exact=report['bitwise_exact'],
                    same_process_replay_validation=report,
                    next_transition_fixture_sha256=file_sha256(output/arm/'next_transition_fixture.pkl'),
                    next_full_transition_sha256=pytree_sha256(expected),
                    replay_H_centers=2*spec['training']['chains']*spec['training']['records_per_chain'],
                    replay_updates_committed=0)
                save()
            if stopped:
                break
        if replay:
            replay_result.update(status='passed', elapsed_seconds=perf_counter()-start,
                budget_consumed_seconds_including_original=previous_budget_seconds+perf_counter()-start,
                new_full_H_centers=sum(a['additional_full_H_centers'] for a in replay_result['arms'].values()))
            atomic_json(replay_path, replay_result)
            return replay_result
        result.update(status='completed' if all(result['arms'].get(a, {}).get('status') == 'completed'
                                              for a in ARMS) else 'stopped_at_atomic_boundary',
                      stop_signals=stopped)
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
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--until-step', type=int)
    parser.add_argument('--replay-completed', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, resume=args.resume, until_step=args.until_step,
        replay=args.replay_completed), indent=2))
