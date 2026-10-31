"""Same-state N144 cache gate with real MH records and complete SR gradients.

No decoder, group table, image weight, bond or virtual width is changed.
The completed synthetic cost checkpoint is an INITIALIZATION, not a trained
state. Native Markov records now measure actual sampling/H/score costs. SR
proposals are inspected but not committed; energy optimization is a separately
budgeted next stage, never inferred from synthetic ratios or this gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from projects.clebsch_tree_nqs.workflows.triangle_scale_batch_cost import (
    checked_cost_parent, collect_accounting, make_scalar_connectivity_reference, relative_vector_error,
)
from projects.clebsch_tree_nqs.workflows.triangle_scale_cost import balanced_block_probe_states
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
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, make_mixed_projected_amplitude, mixed_parameter_count,
)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps
from su2_nqs.vmc.replay_validation import compare_replay
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('binary_truncated', 'ternary_joint')


def native_work_plan(spec: dict) -> dict:
    """Finite sample/H work; this gate commits exactly zero energy updates."""
    options = spec['sampling']
    for name in ('chains', 'records_per_chain', 'burn_proposals', 'proposals_between'):
        if isinstance(options[name], bool) or not isinstance(options[name], int) or options[name] < 1:
            raise ValueError('positive integer sampling budgets required')
    count = options['chains']*options['records_per_chain']
    if count < 2 or not 0 < spec['tolerance'] <= 2e-11:
        raise ValueError('at least two records and an exact-oracle tolerance <=2e-11 required')
    if spec['oracle_rows'] < 2 or spec['oracle_rows'] > count:
        raise ValueError('a multiple-record independent H oracle is required')
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > count
           for k in spec['H_batches']):
        raise ValueError('positive H batches bounded by the actual record pool required')
    return dict(native_records=count, complete_H_pool_centers=count,
        independent_oracle_centers=spec['oracle_rows'], optimizer_updates_committed=0,
        cached_oracle_H_centers=spec['oracle_rows'],
        uncommitted_full_network_SR_proposals=len(ARMS),
        timing_H_centers=2*sum(spec['H_batches']),
        sampler_proposals_per_chain=options['burn_proposals']+
            options['records_per_chain']*options['proposals_between'],
        all_H_bonds_and_full_A1_required=True)


def validate_oracle_errors(errors: dict[str, float], tolerance: float) -> None:
    """Fail closed on NaN/Inf as well as any excessive relative discrepancy."""
    if not errors or not all(np.isfinite(error) and error <= tolerance for error in errors.values()):
        raise ArithmeticError('nonfinite or excessive exact-oracle discrepancy')


def _plain(tree):
    return jax.tree.map(lambda x: np.asarray(x).item() if np.asarray(x).ndim == 0
                        else np.asarray(x).tolist(), tree)


def _memory() -> list:
    return [{key: int(value) for key, value in (device.memory_stats() or {}).items()
             if key in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit', 'bytes_reserved')}
            for device in jax.local_devices()]


def run(config_path: Path, *, replay: bool = False) -> dict:
    """Execute or independently replay a bounded, immutable native cost gate."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    work = native_work_plan(spec)
    output = Path(spec['output'])
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('this bounded cost gate requires an actual GPU')
    if not np.isfinite(spec['j2_over_j1']) or spec['j2_over_j1'] not in (0., .125):
        raise ValueError('this first size gate freezes J2/J1 to zero or .125')
    if output.exists() and not replay:
        raise FileExistsError('cost output is immutable; do not overwrite an incomplete gate')
    if replay and not (output/'results.json').exists():
        raise FileNotFoundError('independent replay requires a completed native gate')
    pair, parent_spec = checked_cost_parent(spec['cost_parent'], root=ROOT)
    cfg = MixedClebschConfig(**parent_spec['model'])
    if cfg.length != 12 or cfg.multiplicity_cap != 16 or cfg.maximum_q != 10:
        raise ValueError('this first gate preserves the passed N144/cap16/qmax10 model')
    rank = parent_spec['binary_rank']
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', rank)
    bank_other = arity_clebsch_bank(pair['binary_truncated'], cfg, 'binary_truncated', rank)
    common_error = relative_vector_error(
        np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(bank)]),
        np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(bank_other)]))
    validate_oracle_errors({'common_arity_bank': common_error}, spec['tolerance'])
    images = np.asarray(triangular_hnf_gamma_a1_group(((12, 0), (0, 12))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(12, 12, j2=spec['j2_over_j1'], periodic=True))
    amplitude = make_feature_lut_projected_amplitude(cfg, images, chunk=spec['projection_chunk'])
    legacy = make_mixed_projected_amplitude(cfg, images, chunk=spec['projection_chunk'])
    action = make_cached_projected_heisenberg_action(cfg, images, bonds,
        projection_chunk=spec['projection_chunk'])
    prepare = jax.jit(lambda b: prepare_feature_lut_bank(b, cfg))
    prepared = prepare(bank)
    edges = tuple((a, b) for a, b, J in bonds if J != 0)
    replay_sampler = jax.jit(lambda b, key, states: run_exchange_sweeps(
        b, key, states, amplitude_fn=amplitude, particle_kind='spin', sweeps=spec['replay_proposals'],
        exchange_edges=edges, **spec['sampler']))
    if replay:
        result = json.loads((output/'results.json').read_text())
        registration = json.loads((output/'registration.json').read_text())
        if (result['status'] != 'completed' or file_sha256(config_path) != registration['config_sha256']
                or registration['sha256'] != canonical_sha256({
                    k: v for k, v in registration.items() if k != 'sha256'})
                or any(file_sha256(ROOT/p) != h for p, h in registration['sources'].items())):
            raise ValueError('completed gate/config/source identity changed')
        destination = output/'process_replay.json'
        if destination.exists():
            raise FileExistsError('replay supplement is immutable')
        saved = load_checkpoint(output/'gate_checkpoint.pkl')
        if (file_sha256(output/'gate_checkpoint.pkl') != result['checkpoint_sha256'] or
                pytree_sha256(saved['params']) != result['parameter_pair_sha256']):
            raise ValueError('gate checkpoint changed')
        continued = jax.block_until_ready(replay_sampler(prepared,
            jnp.asarray(saved['metadata']['rng']), jnp.asarray(saved['metadata']['sampler_states'])))
        fixture = output/'next_sampler_fixture.pkl'
        if file_sha256(fixture) != result['next_sampler_fixture_sha256']:
            raise ValueError('prospective sampler replay fixture changed')
        report = compare_replay(load_checkpoint(fixture)['params'], continued, tolerance=spec['tolerance'],
                                diagnostic_prefixes=('[3]',))
        passed = report['passed']
        replay_record = dict(status='passed' if passed else 'failed',
            full_next_sampler_bitwise_exact=report['bitwise_exact'], replay_validation=report,
            optimizer_updates_committed=0,
            source_results_sha256=file_sha256(output/'results.json'),
            checkpoint_sha256=file_sha256(output/'gate_checkpoint.pkl'), new_H_centers=0)
        atomic_json(destination, replay_record)
        if not passed:
            raise ArithmeticError('independent-process N144 sampler/RNG replay differs')
        return replay_record
    output.mkdir(parents=True)
    sources = transitive_local_source_manifest([Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'],
        root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    registration = dict(config=spec, config_sha256=file_sha256(config_path),
        sources=sources, runtime=numerical_runtime_identity())
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(output/'registration.json', registration)
    for relative in sources:
        destination = output/'source_snapshots'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
    start = perf_counter()
    result = dict(status='running', backend=jax.default_backend(), work_plan=work,
        full_A1_images=len(images), Hamiltonian_bonds=len(bonds), model=parent_spec['model'],
        j2_over_j1=spec['j2_over_j1'], source_cost_H=parent_spec['j2_over_j1'],
        initial_target_state_previously_optimized=False, optimizer_updates_committed=0,
        stage='native_sampling_full_H_full_network_SR_cost_gate',
        common_arity_bank_relative_error=common_error,
        registration_sha256=registration['sha256'], parameter_pair_sha256=pytree_sha256(pair),
        parameter_counts={arm: mixed_parameter_count(params) for arm, params in pair.items()},
        same_cost_checkpoint_in_both_Hamiltonians=True,
        energy_optimization=False, energy_benchmark_claim=False, equilibrium_certified=False,
        timings={}, relative_errors={}, gradient_diagnostics={},
        created_utc=datetime.now(timezone.utc).isoformat())

    def save():
        result['elapsed_seconds'] = perf_counter()-start
        result['device_memory'] = _memory()
        atomic_json(output/'results.json', result)

    def stop(signum, _):
        result.update(status='censored', censored=True,
            reason='native kernel/signal cap '+signal.Signals(signum).name)
        save()
        raise SystemExit(2)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGUSR1)}
    signal.alarm(spec['seconds_cap'])

    def timed(name, fn, *args):
        if perf_counter()-start >= spec['seconds_cap']:
            stop(signal.SIGALRM, None)
        result['current_operation'] = name
        save()
        tick = perf_counter()
        value = jax.block_until_ready(fn(*args))
        result['timings'][name] = perf_counter()-tick
        save()
        return value

    try:
        options = spec['sampling']
        sampler = make_record_sampler(amplitude, records=options['records_per_chain'],
            burn_proposals=options['burn_proposals'], proposals_between=options['proposals_between'],
            exchange_edges=edges, **spec['sampler'])
        # These are explicitly initial chain configurations, not repaired Born
        # draws. The unconstrained bond/global MH kernel then changes them.
        initial = jnp.asarray(balanced_block_probe_states(cfg,
            count=options['chains'], seed=spec['initialization_seed']))
        key = jax.random.PRNGKey(spec['sampling_seed'])
        states, key, rows, sampling = timed('native_MH_initial_burn_and_record_pool',
            sampler, prepared, key, initial)
        samples = rows.reshape(-1, 144)
        result['sampling'] = _plain(sampling)
        result['native_rows_sha256'] = pytree_sha256(samples)
        result['native_record_count'] = len(samples)
        save_checkpoint(output/'gate_checkpoint.pkl', pair,
            optimizer_state=dict(completed_steps=jnp.asarray(0), damping=jnp.asarray(spec['sr']['damping'])),
            metadata=dict(sampler_states=np.asarray(states), rng=np.asarray(key),
                registration_sha256=registration['sha256'], trained=False,
                source_cost_checkpoint_sha256=spec['cost_parent']['checkpoint_sha256']))
        result['checkpoint_sha256'] = file_sha256(output/'gate_checkpoint.pkl')
        save()
        graph = SzHeisenbergGraph(144, bonds, n_up=72)
        reference = make_scalar_connectivity_reference(legacy, bank)
        oracle = timed('independent_legacy_Sz_connectivity_multiple_native_rows',
            lambda: np.asarray([reference(graph, np.asarray(row))
                                for row in samples[:spec['oracle_rows']]]))
        observed = timed('cached_full_H_oracle_native_rows', action,
            prepared, samples[:spec['oracle_rows']])
        result['relative_errors'].update(
            full_H=relative_vector_error(observed[1], oracle[:, 1]),
            projected_amplitude=relative_vector_error(observed[0], oracle[:, 0]))
        validate_oracle_errors(result['relative_errors'], spec['tolerance'])
        warm_batches = {}
        for batch in spec['H_batches']:
            cold = timed(f'cached_H_native_batch{batch}_cold', action, prepared, samples[:batch])
            warm = timed(f'cached_H_native_batch{batch}_warm', action, prepared, samples[:batch])
            discrepancy = relative_vector_error(np.asarray(cold), np.asarray(warm))
            validate_oracle_errors({'cold_warm_batch'+str(batch): discrepancy}, spec['tolerance'])
            if not np.isfinite(np.asarray(warm)).all():
                raise FloatingPointError('nonfinite timed H output')
            warm_batches[batch] = warm
        chunk = spec['action_chunk']

        @jax.jit
        def pool_H(prepared, samples):
            def one(row):
                psi, hp = action(prepared, row[None])
                return jnp.stack((psi[0], hp[0]))
            values = jax.lax.map(one, samples, batch_size=chunk)
            return values[:, 0], values[:, 1]

        psi, hp = timed('complete_H_entire_native_pool', pool_H, prepared, samples)
        if np.any(np.asarray(psi) == 0) or not np.isfinite(np.asarray(hp/psi)).all():
            raise FloatingPointError('exact zero or nonfinite native Born center; no flooring')
        for batch, values in warm_batches.items():
            discrepancy = relative_vector_error(np.asarray(values), np.asarray((psi[:batch], hp[:batch])))
            result['relative_errors']['batch'+str(batch)+'_versus_full_pool'] = discrepancy
        validate_oracle_errors(result['relative_errors'], spec['tolerance'])
        local_energy = hp/psi
        result['native_pool_statistics_not_benchmark'] = summarize_born_audit(
            np.asarray(local_energy).reshape(rows.shape[:2]), np.asarray(rows), n_sites=144)
        np.savez_compressed(output/'native_pool.npz', bits=np.asarray(rows),
            psi=np.asarray(psi), Hpsi=np.asarray(hp), local_energy=np.asarray(local_energy))
        for arm in ARMS:
            params = pair[arm]
            prepare_params = lambda p, arm=arm: prepare_feature_lut_bank(
                arity_clebsch_bank(p, cfg, arm, rank), cfg)
            logs = lambda p, row: jnp.log(amplitude(p, row).astype(jnp.complex128))
            updater = make_prepared_sr_update(prepare_params, logs, params, **spec['sr'])
            candidate, diagnostic = timed('full_original_parameter_SR_proposal_'+arm,
                updater, params, samples, jax.lax.stop_gradient(local_energy))
            result['gradient_diagnostics'][arm] = _plain(diagnostic)
            result['gradient_diagnostics'][arm]['uncommitted_candidate_params_sha256'] = pytree_sha256(candidate)
            result['gradient_diagnostics'][arm]['candidate_committed'] = False
            result['gradient_diagnostics'][arm]['all_parameter_count'] = mixed_parameter_count(params)
            result['gradient_diagnostics'][arm]['explicit_score_bytes'] = 2*len(samples)*mixed_parameter_count(params)*8
            if (not np.isfinite(float(diagnostic['gradient_norm'])) or
                    not np.isfinite(float(diagnostic['relative_linear_residual'])) or
                    float(diagnostic['relative_linear_residual']) > 1e-7):
                raise ArithmeticError('nonfinite/inaccurate full original-parameter SR gradient')
            save()
        expected = timed('next_native_sampler_transition', replay_sampler, prepared, key, states)
        saved = load_checkpoint(output/'gate_checkpoint.pkl')
        restored = timed('restored_next_native_sampler_transition', replay_sampler, prepared,
            jnp.asarray(saved['metadata']['rng']), jnp.asarray(saved['metadata']['sampler_states']))
        replay_report = compare_replay(expected, restored, tolerance=spec['tolerance'],
                                       diagnostic_prefixes=('[3]',))
        result['same_process_replay_validation'] = replay_report
        save()
        replay_ok = replay_report['passed']
        if not replay_ok or pytree_sha256(saved['params']) != pytree_sha256(pair):
            raise ArithmeticError('N144 parameter/sampler/RNG checkpoint replay differs')
        save_checkpoint(output/'next_sampler_fixture.pkl', expected,
                        metadata=dict(uncommitted=True, input_checkpoint_sha256=result['checkpoint_sha256']))
        result.update(status='completed', censored=False,
            same_process_sampler_checkpoint_replay_bitwise_exact=replay_report['bitwise_exact'],
            same_process_replay_validation=replay_report,
            next_sampler_fixture_sha256=file_sha256(output/'next_sampler_fixture.pkl'),
            next_sampler_replay_sha256=pytree_sha256(expected),
            optimizer_updates_committed=0,
            cost_gate_passed=True, additional_training_budget_not_released=True)
        save()
        return result
    except BaseException as error:
        if result['status'] != 'censored':
            result.update(status='failed', error=repr(error), cost_gate_passed=False)
            save()
        raise
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replay-gate', action='store_true')
    parser.add_argument('--collect-job')
    args = parser.parse_args()
    report = collect_accounting(args.config, args.collect_job) if args.collect_job else run(
        args.config, replay=args.replay_gate)
    print(json.dumps(report, indent=2))
