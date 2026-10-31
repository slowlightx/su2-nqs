"""Same-parent, fixed-width raw/QR-coordinate optimization of triangular CT.

This is an isolated continuation experiment, not a change to the production
model. QR changes only multiplicity coordinates, once at initialization; it
does not truncate, project, enlarge or re-canonicalize the state after updates.
Both arms reset Adam, use complete coherent A1 amplitudes and the same full H,
and fit a physical FS radius on calibration chains. Disjoint validation
chains may veto the step but never choose or shrink it.
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Integral
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_parameter_count
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains, run_exchange_sweeps
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
from su2_nqs.vmc.physical_step_calibration import make_fs_calibrator
from su2_nqs.vmc.replay_validation import compare_replay
from . import triangle_clebsch_continuation as continuation
from . import triangle_optimizer_comparison as shared

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('raw', 'canonical')
TASK = 'triangle_same_parent_initial_QR_coordinates_v1'


def validate_plan(plan: dict) -> None:
    """Fail closed on unregistered geometry, counts, seeds or resource changes."""
    if (plan['task'] != TASK or plan['model'] != dict(length=6, multiplicity_cap=20, maximum_q=10)
            or plan['arity'] != 'ternary_joint' or plan['binary_rank'] != 8
            or plan['j2_over_j1'] not in (0., .125) or plan['arms'] != list(ARMS)
            or plan['optimizer'] != 'sr_adam' or plan['sr_solver'] != 'dual_force'):
        raise ValueError('only same-parent w20 joint-ternary raw/QR coordinates are registered')
    development = plan['profile'] == 'local_development'
    if plan['profile'] not in ('local_development', 'gpu_comparison'):
        raise ValueError('unknown execution profile')
    if not development and (not plan['require_gpu'] or plan['pilot_steps'] != 2
            or plan['training']['steps'] != 16
            or (plan['training']['samples'], plan['training']['chains']) != (2048, 64)
            or (plan['calibration']['samples'], plan['calibration']['chains']) != (1024, 128)
            or (plan['audit']['samples'], plan['audit']['chains']) != (16384, 128)):
        raise ValueError('production needs 2-step pilot then fixed16/2048, split1024 and audit16384')
    if development and (plan['require_gpu'] or plan['training']['samples'] > 64
                         or plan['training']['steps'] > 2):
        raise ValueError('local development cannot release production work')
    warm_calibration = plan.get('calibration_initialization', 'fresh_random')
    if (warm_calibration not in ('fresh_random', 'audited_parent_calibration_chains')
            or warm_calibration != 'fresh_random' and
            (not development or plan['calibration']['chains'] != 64)):
        raise ValueError('only local arithmetic probe may warm start parent calibration chains')
    integers = [plan['pilot_steps'], plan['training']['steps'], plan['halvings'],
                plan['projection_chunk'], plan['action_chunk'], plan['score_chunk'],
                *plan['model'].values(), plan['budget']['maximum_peak_device_bytes']]
    for section in ('training', 'calibration', 'audit'):
        spec = plan[section]
        integers += [spec[k] for k in ('samples', 'chains', 'refresh_proposals', 'spacing_proposals')]
        if spec['samples'] % spec['chains'] or spec['chains'] < 2:
            raise ValueError('records must divide into independent chain labels')
    integers += [plan['calibration'][k] for k in ('initial_burn_proposals', 'bracket_steps', 'bisection_steps')]
    integers += [plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds')]
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in integers):
        raise ValueError('counts and budget limits must be positive integers')
    if plan['calibration']['chains'] % 2:
        raise ValueError('calibration and unseen validation require disjoint equal chain halves')
    if plan['pilot_steps'] > plan['training']['steps']:
        raise ValueError('pilot exceeds endpoint')
    for name in ('learning_rate', 'damping', 'tolerance', 'maximum_fs_sine', 'minimum_ess_fraction'):
        if not math.isfinite(plan[name]) or plan[name] <= 0:
            raise ValueError('invalid numerical scale')
    cal = plan['calibration']
    for name in ('target_fs_sine', 'relative_tolerance', 'validation_relative_tolerance',
                 'minimum_ess_fraction', 'max_factor'):
        if not math.isfinite(cal[name]) or cal[name] <= 0:
            raise ValueError('invalid physical step setting')
    if (plan['tolerance'] > 2e-11 or plan['learning_rate'] != 1e-4 or plan['damping'] != .05
            or plan['maximum_fs_sine'] != .005 or plan['minimum_ess_fraction'] != .98
            or cal['minimum_ess_fraction'] != .98 or cal['target_fs_sine'] != .001
            or cal['relative_tolerance'] != .02 or cal['validation_relative_tolerance'] != .25):
        raise ValueError('fixed arithmetic and physical-distance controls cannot be weakened')
    seeds = plan['seeds']
    if (set(seeds) != {'calibration', 'audit_raw', 'audit_canonical'}
            or len(set(seeds.values())) != 3
            or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < 2**32
                   for v in seeds.values())):
        raise ValueError('three named, distinct uint32 streams required')
    if sum(plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds')) > 6*3600:
        raise ValueError('one arm exceeds the preregistered six A100-hour ceiling')
    parent = plan['parent']
    if set(parent['sealed_files']) != {'checkpoint.pkl', 'registration.json', 'results.json',
                                      'terminal_audit.json', 'process_replay.json'}:
        raise ValueError('complete audited parent and replay seals required')
    if any(len(digest) != 64 for digest in parent['sealed_files'].values()):
        raise ValueError('malformed parent file seal')


def split_calibration_chains(rows: object, chains: int) -> tuple[object, object]:
    """Split record-major samples by CHAIN, never by successive record halves."""
    if chains < 4 or chains % 2 or np.shape(rows)[0] % chains:
        raise ValueError('at least four even chains and whole records required')
    shaped = jnp.asarray(rows).reshape(-1, chains, np.shape(rows)[-1])
    return (shaped[:, :chains//2].reshape(-1, shaped.shape[-1]),
            shaped[:, chains//2:].reshape(-1, shaped.shape[-1]))


def validation_decision(diagnostics: dict, calibration: dict) -> bool:
    """One frozen held-out check; this function cannot tune a proposed step."""
    fs = float(diagnostics['sampled_fs_sine'])
    ess = float(diagnostics['reweighting_ess_fraction'])
    return bool(math.isfinite(fs) and math.isfinite(ess)
        and abs(fs/calibration['target_fs_sine']-1.) <= calibration['validation_relative_tolerance']
        and ess >= calibration['minimum_ess_fraction'])


def load_parent(plan: dict) -> tuple[tuple, dict]:
    """Verify checkpoint, actual independent audit and full-state restore seals."""
    req = plan['parent']; directory = Path(req['directory'])
    for name, digest in req['sealed_files'].items():
        if file_sha256(directory/name) != digest:
            raise ValueError('audited parent changed: '+name)
    reg = json.loads((directory/'registration.json').read_text()); seal = reg['sha256']
    if canonical_sha256({k: v for k, v in reg.items() if k != 'sha256'}) != seal:
        raise ValueError('invalid parent registration')
    if any(reg.get(k) != plan[k] for k in ('model', 'arity', 'j2_over_j1', 'binary_rank')):
        raise ValueError('parent model or Hamiltonian identity differs')
    state, meta = shared.restore_state(directory/'checkpoint.pkl', seal)
    audit = json.loads((directory/'terminal_audit.json').read_text())
    result = json.loads((directory/'results.json').read_text())
    replay = json.loads((directory/'process_replay.json').read_text())
    if (audit['params_sha256'] != pytree_sha256(state[0])
            or result['params_sha256'] != pytree_sha256(state[0])
            or result['status'] != 'fixed_steps_complete' or replay['status'] != 'passed'
            or np.shape(state[3]) != (64, 36)
            or req['params_sha256'] != pytree_sha256(state[0])):
        raise ValueError('parent audit, replay or endpoint identity disagrees')
    return state, dict(params_sha256=pytree_sha256(state[0]), audit=audit,
                      checkpoint_sha256=req['sealed_files']['checkpoint.pkl'],
                      absolute_completed_steps=meta.get('absolute_completed_steps'),
                      parent_optimizer_sha256=pytree_sha256(state[1]))


def initial_parameters(parent: object, arm: str, config: MixedClebschConfig) -> tuple[object, dict]:
    """Same physical bank, same storage dimensions; one optional exact block QR."""
    if arm not in ARMS:
        raise ValueError('undeclared coordinate arm')
    if arm == 'canonical':
        bank, report = canonicalize_triangular_bank(parent, config)
        return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float64), bank), report
    return parent, dict(policy='unchanged_parent_coordinates')


def runtime(plan: dict, template: object) -> dict:
    """Reuse production H/amplitudes/SR with separate calibration and validation."""
    rt = continuation.runtime(plan, template)
    prepare, projected = rt['prepare'], rt['projected']
    update = make_physical_sr_adam(lambda p, x: projected(prepare(p), x), template,
        damping=plan['damping'], score_chunk=plan['score_chunk'],
        maximum_fs_sine=plan['maximum_fs_sine'], minimum_ess_fraction=plan['minimum_ess_fraction'],
        prepare=prepare, prepared_amplitude=projected, sr_solver=plan['sr_solver'])
    cal = plan['calibration']
    options = {k: cal[k] for k in ('target_fs_sine', 'relative_tolerance', 'max_factor',
                                  'bracket_steps', 'bisection_steps', 'minimum_ess_fraction')}
    calibrate = make_fs_calibrator(prepare,
        lambda b, x: jnp.log(projected(b, x).astype(jnp.complex128)), template,
        batch_size=plan['score_chunk'], **options)
    train_sampler = shared.make_record_sampler(projected, plan['training'], rt['edges'])
    cal_sampler = shared.make_record_sampler(projected, cal, rt['edges'])
    _, unravel = ravel_pytree(template)

    @jax.jit
    def evaluate(p, rows):
        prepared = prepare(p)
        return jax.lax.map(lambda x: projected(prepared, x), rows, batch_size=plan['score_chunk'])

    @jax.jit
    def normalize_direction(parent, proposed, rows):
        theta, _ = ravel_pytree(parent); delta = ravel_pytree(proposed)[0]-theta
        def logs(t):
            prepared = prepare(unravel(t))
            return jax.lax.map(lambda x: jnp.log(jnp.abs(projected(prepared, x))),
                               rows, batch_size=plan['score_chunk'])
        _, tangent = jax.jvp(logs, (theta,), (delta,))
        norm = jnp.std(tangent); valid = jnp.isfinite(norm) & (norm > 1e-14)
        scale = jnp.where(valid, cal['target_fs_sine']/norm, 0.)
        return unravel(theta+scale*delta), norm, valid

    def transition(state):
        p, moments, key, states, ck, cx = state
        tick = perf_counter(); prepared = prepare(p)
        key, states, rows = train_sampler(prepared, key, states)
        ck, cx, combined = cal_sampler(prepared, ck, cx)
        fit_rows, validation_rows = split_calibration_chains(combined, cal['chains'])
        jax.block_until_ready(combined); sampling_seconds = perf_counter()-tick; tick = perf_counter()
        psi, hp = rt['action'](prepared, rows); energy = jax.block_until_ready(hp/psi)
        if not np.isfinite(np.asarray(energy)).all():
            raise FloatingPointError('nonfinite complete-H ratios; no clipping or floor')
        h_seconds = perf_counter()-tick; tick = perf_counter()
        proposed, next_moments, sr = update(p, moments, rows, energy, plan['learning_rate'], warm=True)
        normalized, tangent, valid = normalize_direction(p, proposed, fit_rows)
        chosen, fitted = calibrate(p, normalized, fit_rows)
        def distance(x):
            old = evaluate(p, x); new = evaluate(chosen, x)
            return sampled_overlap_diagnostics(jnp.log((new/old).astype(jnp.complex128)))
        measured_train = distance(rows)
        # This is the first and only use of these chains in update selection.
        unseen = distance(validation_rows)
        accepted = bool(sr['accepted'] & valid & fitted['accepted']) and validation_decision(unseen, cal)
        accepted = accepted and bool(measured_train['sampled_fs_sine'] <= plan['maximum_fs_sine']
            and measured_train['reweighting_ess_fraction'] >= plan['minimum_ess_fraction'])
        as_scalars = lambda d: jax.tree.map(lambda v: np.asarray(v).item(), d)
        info = dict(accepted=bool(accepted), full_H_calls=int(len(rows)),
            training_energy_per_site=float(jnp.mean(energy))/36,
            training_variance_per_site=float(jnp.var(energy))/36,
            training_rows_sha256=pytree_sha256(rows), calibration_rows_sha256=pytree_sha256(fit_rows),
            validation_rows_sha256=pytree_sha256(validation_rows),
            raw_calibration_tangent=float(tangent), sr=as_scalars(sr), calibration=as_scalars(fitted),
            measured_training_step=as_scalars(measured_train), unseen_validation=as_scalars(unseen),
            calibration_samples=len(fit_rows), unseen_validation_samples=len(validation_rows),
            proposed_optimizer_count=int(next_moments[0]),
            proposed_params_sha256=pytree_sha256(chosen))
        if not accepted:
            chosen, next_moments = p, moments
        jax.block_until_ready(chosen)
        return (chosen, next_moments, key, states, ck, cx), info, dict(
            sampling_seconds=sampling_seconds, H_seconds=h_seconds,
            proposal_and_validation_seconds=perf_counter()-tick)

    return dict(rt, transition=transition, evaluate=evaluate)


def initialize_state(plan: dict, arm: str, parent: tuple, rt: dict) -> tuple[tuple, dict]:
    """Reset both optimizers and paired calibration streams; inherit train RNG."""
    p, gauge = initial_parameters(parent[0], arm, rt['cfg'])
    # Real projected amplitudes—not raw subtree metrics—validate same start.
    before = rt['evaluate'](parent[0], parent[3][:8]); after = rt['evaluate'](p, parent[3][:8])
    error = float(jnp.linalg.norm(before-after)/jnp.maximum(jnp.linalg.norm(before), 1e-300))
    if not math.isfinite(error) or error > plan['tolerance']:
        raise ArithmeticError('QR failed full coherent A1 same-state initialization')
    ck, init = jax.random.split(jax.random.PRNGKey(plan['seeds']['calibration']))
    if plan.get('calibration_initialization') == 'audited_parent_calibration_chains':
        cx = parent[5]
        if np.shape(cx) != (64, 36):
            raise ValueError('local warm-calibration probe requires64 parent chains')
    else:
        cx = initialize_spin_exchange_chains(init, n_chains=plan['calibration']['chains'], n_sites=36, n_up=18)
    prepared = rt['prepare'](p)
    cx, _, ck, _ = jax.jit(lambda k, x: run_exchange_sweeps(prepared, k, x,
        amplitude_fn=rt['projected'], particle_kind='spin',
        sweeps=plan['calibration']['initial_burn_proposals'], exchange_edges=rt['edges']))(ck, cx)
    state = (p, _manual_adam_init(p), parent[2], parent[3][:plan['training']['chains']], ck, cx)
    return state, dict(initial_A1_relative_error=error, gauge=gauge,
        optimizer_policy='both_arms_reset_to_zero_before_first_update',
        coordinate_policy='initial_QR_only_no_postupdate_recanonicalization',
        calibration_initialization=plan.get('calibration_initialization', 'fresh_random'),
        initial_physical_parameter_count=mixed_parameter_count(p),
        initial_training_sampler_rng_sha256=pytree_sha256(state[2:4]),
        initial_calibration_sampler_rng_sha256=pytree_sha256(state[4:]))


def run(plan: dict, arm: str, phase: str, plan_sha256: str) -> dict:
    """Measured GPU pilot -> separate replay -> bounded fixed endpoint and audit."""
    validate_plan(plan)
    if canonical_sha256(plan) != plan_sha256 or arm not in ARMS:
        raise ValueError('registered plan/arm changed')
    jax.config.update('jax_enable_x64', True)
    if plan['require_gpu'] and jax.default_backend() != 'gpu':
        raise RuntimeError('allocated GPU required')
    started = perf_counter(); parent, parent_info = load_parent(plan)
    output = Path(plan['output'])/arm; output.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_sha256=plan_sha256, arm=arm,
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    seal = canonical_sha256(identity); registered = dict(identity, sha256=seal)
    for name, data in [('registration.json', registered),
                       ('numerical_runtime.json', continuation.numerical_runtime_fingerprint())]:
        if (output/name).exists() and json.loads((output/name).read_text()) != data:
            raise ValueError('immutable '+name+' changed; independent replay must match actual pilot runtime')
        atomic_json(output/name, data)
    rt = runtime(plan, parent[0])
    if (output/'checkpoint.pkl').exists():
        state, meta = shared.restore_state(output/'checkpoint.pkl', seal)
    else:
        if phase != 'pilot':
            raise ValueError('pilot must initialize the new coordinate experiment')
        state, initialization = initialize_state(plan, arm, parent, rt)
        meta = dict(registration_sha256=seal, completed_steps=0, history=[],
                    initialization=initialization, parent=parent_info)
        shared.save_state(output/'checkpoint.pkl', state, meta)
        shared.save_state(output/'initial_checkpoint.pkl', state, meta)
    if meta['completed_steps'] != len(meta['history']) or int(state[1][0]) != meta['completed_steps']:
        raise ValueError('reset optimizer counter and accepted update ledger disagree')
    samples = plan['training']['samples']
    if phase == 'replay':
        pilot = json.loads((output/'pilot.json').read_text())
        if (pilot['status'] != 'passed' or pilot['checkpoint_sha256'] != file_sha256(output/'checkpoint.pkl')
                or pilot['fixture_sha256'] != file_sha256(output/'next_transition_fixture.pkl')):
            raise ValueError('successful immutable pilot and transition fixture required')
        expected = load_checkpoint(output/'next_transition_fixture.pkl')['params']
        new, info, _ = rt['transition'](state)
        report = compare_replay(expected, shared.transition_fixture(new, info),
            tolerance=plan['tolerance'], diagnostic_prefixes=('[1]',))
        answer = dict(status='passed' if report['passed'] else 'failed', report=report,
            optimizer_updates_committed=0, new_full_H_centers=samples,
            checkpoint_sha256=file_sha256(output/'checkpoint.pkl'), elapsed_seconds=perf_counter()-started)
        atomic_json(output/'process_replay.json', answer)
        if not report['passed']:
            raise ArithmeticError('process-isolated parameters/optimizer/sampler/RNG replay failed')
        return answer
    if phase not in ('pilot', 'train'):
        raise ValueError('unknown execution phase')
    if not (output/'preflight.json').exists():
        atomic_json(output/'preflight.json', shared.preflight(rt, state[0], state[3], plan))
    if phase == 'train':
        pilot = json.loads((output/'pilot.json').read_text())
        replay = json.loads((output/'process_replay.json').read_text())
        checkpoint = (file_sha256(output/'checkpoint.pkl') if meta['completed_steps'] == plan['pilot_steps']
                      else meta['pilot_checkpoint_sha256'])
        if (pilot['status'] != 'passed' or replay['status'] != 'passed'
                or pilot['checkpoint_sha256'] != checkpoint or replay['checkpoint_sha256'] != checkpoint):
            raise ValueError('accepted pilot and exact independent replay required before training')
        meta['pilot_checkpoint_sha256'] = checkpoint
    stop = [False]
    def request_stop(*_): stop[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM): signal.signal(sig, request_stop)
    target = plan['pilot_steps'] if phase == 'pilot' else plan['training']['steps']
    while meta['completed_steps'] < target and not stop[0]:
        previous = meta['history'][-1]['seconds'] if meta['history'] else 0
        if perf_counter()-started+previous+120 > plan['budget'][phase+'_seconds']:
            return dict(status='safe_pause', completed_steps=meta['completed_steps'])
        tick = perf_counter(); proposed, info, timings = rt['transition'](state)
        if not info['accepted']:
            atomic_json(output/'rejected_step.json', dict(info, timings=timings,
                parent_checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
                optimizer_updates_committed=0, rejection_does_not_tune_radius=True))
            return dict(status='heldout_step_rejected', completed_steps=meta['completed_steps'],
                        reason='step fit/heldout/physical guard; parent remains saved')
        state = proposed; meta['completed_steps'] += 1
        meta['history'].append(dict(step=meta['completed_steps'], seconds=perf_counter()-tick, **timings, **info))
        shared.save_state(output/'checkpoint.pkl', state, meta)
        atomic_json(output/'progress.json', dict(status='running', completed_steps=meta['completed_steps'],
                                                last=meta['history'][-1]))
        print(json.dumps(meta['history'][-1]), flush=True)
    if meta['completed_steps'] != target or stop[0]:
        return dict(status='safe_pause', completed_steps=meta['completed_steps'])
    if phase == 'pilot':
        new, info, timings = rt['transition'](state)
        save_checkpoint(output/'next_transition_fixture.pkl', shared.transition_fixture(new, info))
        warm = meta['history'][-1]['seconds']
        forecast = warm*(plan['training']['steps']-target+plan['audit']['samples']/samples)+600
        peak = max(int((d.memory_stats() or {}).get('peak_bytes_in_use', 0)) for d in jax.local_devices())
        answer = dict(status='passed' if info['accepted'] and forecast <= plan['budget']['train_seconds']
                      and peak <= plan['budget']['maximum_peak_device_bytes'] else 'failed',
            warm_step_seconds=warm, forecast_remaining_seconds=forecast, peak_device_bytes=peak,
            checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
            fixture_sha256=file_sha256(output/'next_transition_fixture.pkl'),
            preflight_sha256=file_sha256(output/'preflight.json'),
            uncommitted_fixture_full_H_centers=samples, next_step=info,
            fixture_seconds=timings, elapsed_seconds=perf_counter()-started)
        atomic_json(output/'pilot.json', answer)
        return answer
    final = shared.terminal_audit(rt, state[0], plan['audit'], plan['seeds']['audit_'+arm], output)
    answer = dict(status='fixed_steps_complete', completed_steps=target, arm=arm,
        parent=parent_info, initialization=meta['initialization'], history=meta['history'],
        terminal_audit=final, params_sha256=pytree_sha256(state[0]),
        checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
        full_H_training_centers=samples*target, full_H_fixture_and_replay_centers=2*samples,
        full_H_preflight_centers=4, full_H_terminal_centers=plan['audit']['samples'],
        calibration_H_centers=0, same_initial_physical_state_same_width=True,
        actual_gpu_time_not_forced_equal=True, no_variational_family_expansion=True,
        heldout_step_validation_is_not_a_final_independent_energy_audit=True,
        elapsed_seconds=perf_counter()-started)
    atomic_json(output/'results.json', answer)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--phase', choices=('pilot', 'replay', 'train'), required=True)
    args = parser.parse_args()
    result = run(json.loads(args.plan.read_text()), args.arm, args.phase, args.plan_sha256)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    if result['status'] not in ('passed', 'fixed_steps_complete'):
        raise SystemExit(75 if result['status'] == 'safe_pause' else 2)
