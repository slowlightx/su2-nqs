"""Native fixed-width ClebschTree optimizer comparison, with frozen endpoints.

Both arities keep their original real reduced tensors and exact S=0/A1
amplitudes. Only the optimizer recipe changes. No dictionary, attention,
carrier, magnetic-index parameter, or width extension is introduced.
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
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude, make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, make_mixed_projected_amplitude, mixed_parameter_count,
)
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains, run_exchange_sweeps
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
from su2_nqs.vmc.physical_step_calibration import make_fs_calibrator
from su2_nqs.vmc.replay_validation import compare_replay
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
from .spin_triangular_clebsch_train import checked_arity_endpoint

ROOT = Path(__file__).resolve().parents[3]
OPTIMIZERS = ('sr_adam', 'direct_sr')
ARITIES = ('binary_truncated', 'ternary_joint')


def validate_plan(plan: dict) -> None:
    """Require the registered single-H, fixed-width native-parameter trial."""
    if plan['model'] != dict(length=6, multiplicity_cap=16, maximum_q=10):
        raise ValueError('only the preregistered N36 cap16 doubled-q<=10 model is released')
    if plan['j2_over_j1'] != .125 or plan['binary_rank'] != 8:
        raise ValueError('this stage changes neither Hamiltonian nor binary rank')
    counts = [plan['pilot_steps'], plan['training']['steps'], plan['binary_rank'],
              plan['projection_chunk'], plan['action_chunk'], plan['score_chunk'],
              *plan['model'].values(), plan['budget']['maximum_peak_device_bytes']]
    for settings in (plan['training'], plan['calibration'], plan['audit']):
        counts.extend(settings[k] for k in ('samples', 'chains', 'refresh_proposals', 'spacing_proposals'))
    counts.extend(plan['calibration'][k] for k in ('initial_burn_proposals', 'bracket_steps', 'bisection_steps'))
    counts.extend(plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds'))
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in counts):
        raise ValueError('record counts, work counts and time budgets must be positive integers')
    if plan['training']['samples'] != 2048 or plan['training']['chains'] != 64:
        raise ValueError('both optimizers require 2048 records on 64 inherited chains')
    if not 3 <= plan['training']['steps'] <= 20 or plan['pilot_steps'] != 2:
        raise ValueError('two-step pilot must precede a fixed later endpoint')
    if set(plan['parents']) != set(ARITIES):
        raise ValueError('both arities must be retained without endpoint selection')
    if set(plan['optimizers']) != set(OPTIMIZERS):
        raise ValueError('exactly SR+Adam and direct SR are compared')
    if not 0 < plan['calibration']['target_fs_sine'] < plan['maximum_training_fs_sine'] < 1:
        raise ValueError('invalid physical step target/guard')
    scalars = [plan['damping'], plan['tolerance'], plan['calibration']['target_fs_sine'],
               plan['calibration']['relative_tolerance'], plan['calibration']['max_factor'],
               plan['calibration']['minimum_ess_fraction'], plan['minimum_training_ess_fraction'],
               plan['budget']['maximum_requested_A100_hours']]
    scalars += [plan['optimizers'][o]['learning_rate'] for o in OPTIMIZERS]
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in scalars):
        raise ValueError('optimizer, geometry and numerical scales must be positive finite reals')
    if (plan['tolerance'] > 2e-11 or plan['calibration']['relative_tolerance'] >= 1
            or max(plan['calibration']['minimum_ess_fraction'], plan['minimum_training_ess_fraction']) > 1):
        raise ValueError('invalid fixed mathematical/geometry/ESS tolerance')
    if any(set(row) != {'calibration', 'audit_sr_adam', 'audit_direct_sr'}
           for row in plan['seeds'].values()):
        raise ValueError('each arity requires named calibration and two untouched-audit streams')
    seeds = [v for row in plan['seeds'].values() for v in row.values()]
    if (set(plan['seeds']) != set(ARITIES) or len(seeds) != 6 or len(set(seeds)) != 6
            or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < 2**32 for v in seeds)):
        raise ValueError('distinct uint32 calibration/final-audit seeds required')
    hours = sum(plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds'))*4/3600
    if hours > plan['budget']['maximum_requested_A100_hours']:
        raise ValueError('four-case allocation exceeds the fixed total GPU budget')
    if plan['budget']['maximum_requested_A100_hours'] > 24:
        raise ValueError('this bounded stage cannot release more than24 A100 hours')
    for settings in (plan['training'], plan['calibration'], plan['audit']):
        samples, chains = settings['samples'], settings['chains']
        if samples < 2 or chains < 2 or samples % chains:
            raise ValueError('records must divide into independent chain labels')


def case_directory(plan: dict, arity: str, optimizer: str) -> Path:
    """One immutable recipe-specific output, not an energy-selected state."""
    if arity not in ARITIES or optimizer not in OPTIMIZERS:
        raise ValueError('undeclared arity or optimizer')
    return Path(plan['output']) / arity / optimizer


def load_parent(plan: dict, arity: str) -> dict:
    """Reuse the established exact endpoint/optimizer/RNG seal validation."""
    spec = dict(model=plan['model'], arm=arity, j2_over_j1=plan['j2_over_j1'],
        arity_comparison=dict(binary_rank=plan['binary_rank']),
        arity_endpoint_parent=plan['parents'][arity],
        endpoint_protocol='same_H_full_state_continuation',
        additional_steps=plan['training']['steps'],
        steps=200+plan['training']['steps'])
    _, _, endpoint = checked_arity_endpoint(MixedClebschConfig(**plan['model']), spec)
    saved = endpoint['saved']
    if saved['metadata']['completed_steps'] != 200:
        raise ValueError('only the audited200 parent is authorized')
    if np.shape(saved['metadata']['sampler_states']) != (64, 36):
        raise ValueError('parent does not have the declared inherited64 chains')
    return saved


def make_record_sampler(amplitude, settings: dict, edges: tuple):
    """Record the actual Born chain of the current immutable parameter state."""
    count = settings['samples']//settings['chains']

    @jax.jit
    def collect(prepared, key, states):
        states, values, key, _ = run_exchange_sweeps(prepared, key, states,
            amplitude_fn=amplitude, particle_kind='spin', sweeps=settings['refresh_proposals'],
            exchange_edges=edges)

        def body(carry, _):
            x, values, key = carry
            x, values, key, _ = run_exchange_sweeps(prepared, key, x,
                amplitude_fn=amplitude, particle_kind='spin', sweeps=settings['spacing_proposals'],
                amplitudes=values, exchange_edges=edges)
            return (x, values, key), x
        (states, _, key), rows = jax.lax.scan(body, (states, values, key), None, length=count)
        return key, states, rows.reshape(-1, 36)
    return collect


def runtime(plan: dict, arity: str, optimizer: str, template: object) -> dict:
    """Same real Cov Fisher convention and complete H in both recipes."""
    cfg = MixedClebschConfig(**plan['model'])
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    edges = tuple((a, b) for a, b, _ in triangular_j1j2_bonds(6, 6, j2=0., periodic=True))
    bank = lambda p: arity_clebsch_bank(p, cfg, arity, plan['binary_rank'])
    prepare = jax.jit(lambda p: prepare_feature_lut_bank(bank(p), cfg))
    projected = make_feature_lut_projected_amplitude(cfg, images, chunk=plan['projection_chunk'])
    amplitude = lambda p, row: projected(prepare(p), row)
    cached_H = make_cached_projected_heisenberg_action(cfg, images, bonds,
        projection_chunk=plan['projection_chunk'])

    @jax.jit
    def action(prepared, rows):
        def one(row):
            psi, hp = cached_H(prepared, row[None])
            return jnp.stack((psi[0], hp[0]))
        values = jax.lax.map(one, rows, batch_size=plan['action_chunk'])
        return values[:, 0], values[:, 1]

    update = make_physical_sr_adam(amplitude, template, damping=plan['damping'],
        score_chunk=plan['score_chunk'], maximum_fs_sine=plan['maximum_training_fs_sine'],
        sr_use_adam=optimizer == 'sr_adam',
        sr_tangent_fs_radius=plan['calibration']['target_fs_sine'] if optimizer == 'direct_sr' else None,
        prepare=prepare, prepared_amplitude=projected)
    settings = {k: v for k, v in plan['calibration'].items()
                if k in ('target_fs_sine', 'relative_tolerance', 'max_factor',
                         'bracket_steps', 'bisection_steps', 'minimum_ess_fraction')}
    calibrate = make_fs_calibrator(prepare,
        lambda b, row: jnp.log(projected(b, row).astype(jnp.complex128)), template, **settings)
    _, unravel = ravel_pytree(template)
    rows_amplitude = jax.jit(lambda p, rows: jax.lax.map(
        lambda row: projected(prepare(p), row), rows, batch_size=plan['score_chunk']))

    @jax.jit
    def normalize_direction(parent, proposed, rows):
        theta, _ = ravel_pytree(parent)
        delta = ravel_pytree(proposed)[0]-theta
        def logs(values):
            prepared = prepare(unravel(values))
            return jax.vmap(lambda row: jnp.log(jnp.abs(projected(prepared, row))))(rows)
        _, tangent = jax.jvp(logs, (theta,), (delta,))
        norm = jnp.std(tangent)
        valid = jnp.isfinite(norm) & (norm > 1e-14)
        scale = jnp.where(valid, plan['calibration']['target_fs_sine']/norm, 0.)
        return unravel(theta+scale*delta), norm, valid

    def transition(state):
        p, moments, key, states, ck, cx = state
        timings = {}; start = perf_counter(); prepared = prepare(p)
        key, states, rows = make_train(prepared, key, states)
        ck, cx, cal_rows = make_cal(prepared, ck, cx)
        jax.block_until_ready(rows); timings['sampling_seconds'] = perf_counter()-start
        start = perf_counter(); psi, hp = action(prepared, rows)
        energies = hp/psi; jax.block_until_ready(energies)
        if not np.isfinite(np.asarray(energies)).all():
            raise FloatingPointError('nonfinite complete-H ratios; no clipping allowed')
        timings['H_seconds'] = perf_counter()-start
        start = perf_counter()
        proposed, next_moments, sr = update(p, moments, rows, energies,
            plan['optimizers'][optimizer]['learning_rate'], warm=True)
        normalized, tangent, tangent_valid = normalize_direction(p, proposed, cal_rows)
        chosen, calibration = calibrate(p, normalized, cal_rows)
        old = rows_amplitude(p, rows); new = rows_amplitude(chosen, rows)
        measured = sampled_overlap_diagnostics(jnp.log((new/old).astype(jnp.complex128)))
        accepted = bool(sr['accepted'] & tangent_valid & calibration['accepted'] &
            (measured['sampled_fs_sine'] <= plan['maximum_training_fs_sine']) &
            (measured['reweighting_ess_fraction'] >= plan['minimum_training_ess_fraction']))
        if not accepted:
            chosen, next_moments = p, moments
        jax.block_until_ready(chosen); timings['proposal_and_calibration_seconds'] = perf_counter()-start
        info = dict(accepted=accepted, training_energy_per_site=float(jnp.mean(energies))/36,
            training_variance_per_site=float(jnp.var(energies))/36,
            full_H_calls=int(len(rows)), raw_calibration_tangent=float(tangent),
            training_rows_sha256=pytree_sha256(rows), calibration_rows_sha256=pytree_sha256(cal_rows),
            sr=jax.tree.map(lambda v: np.asarray(v).item(), sr),
            calibration=jax.tree.map(lambda v: np.asarray(v).item(), calibration),
            measured_training_step=jax.tree.map(lambda v: np.asarray(v).item(), measured))
        return (chosen, next_moments, key, states, ck, cx), info, timings

    make_train = make_record_sampler(projected, plan['training'], edges)
    make_cal = make_record_sampler(projected, plan['calibration'], edges)
    return dict(cfg=cfg, bank=bank, prepare=prepare, projected=projected, action=action,
        transition=transition, images=images, bonds=bonds, edges=edges)


def preflight(rt: dict, p: object, states: object, plan: dict) -> dict:
    """Check the cached execution against the unchanged physical/H reference."""
    legacy = make_mixed_projected_amplitude(rt['cfg'], rt['images'], chunk=plan['projection_chunk'])
    old_H = make_amplitude_heisenberg_action(legacy, rt['bonds'], n_sites=36)
    bank = rt['bank'](p); rows = jnp.asarray(states[:2]); prepared = rt['prepare'](p)
    a, h = jax.block_until_ready(rt['action'](prepared, rows))
    b, k = jax.block_until_ready(jax.jit(old_H)(bank, rows))
    def relative(x, y):
        return float(np.linalg.norm(np.asarray(x-y))/max(np.linalg.norm(np.asarray(y)), 1e-300))
    g0 = jax.grad(lambda z: legacy(rt['bank'](z), rows[0]))(p)
    g1 = jax.grad(lambda z: rt['projected'](rt['prepare'](z), rows[0]))(p)
    gradient_error = relative(ravel_pytree(g1)[0], ravel_pytree(g0)[0])
    errors = dict(amplitude_relative_error=relative(a, b), full_H_relative_error=relative(h, k),
        gradient_relative_error=gradient_error,
        A1_relative_error=relative(rt['projected'](prepared, rows[0][rt['images'][37]]), a[0]))
    bra = np.asarray(rows[0]).copy(); bra[np.flatnonzero(bra == 0)[0]] = 1
    lowered = np.tile(bra, (19, 1)); lowered[np.arange(19), np.flatnonzero(bra)] = 0
    ward = jax.jit(jax.vmap(make_feature_lut_amplitude(rt['cfg']), in_axes=(None, 0)))(
        prepared, jnp.asarray(lowered))
    errors['S_plus_relative_error'] = float(abs(ward.sum())/jnp.maximum(jnp.linalg.norm(ward), 1e-300))
    if not all(math.isfinite(v) and v <= plan['tolerance'] for v in errors.values()):
        raise ArithmeticError('cached execution changed the physical state/H/gradient')
    return dict(status='passed', errors=errors, full_H_centers=4,
        complete_A1_images=len(rt['images']), complete_H_bonds=len(rt['bonds']))


def save_state(path: Path, state: tuple, metadata: dict) -> None:
    """Atomically retain parameters, optimizer and BOTH sampler/RNG streams."""
    p, moments, key, states, ck, cx = state
    meta = dict(metadata, training_key=np.asarray(key), training_states=np.asarray(states),
        calibration_key=np.asarray(ck), calibration_states=np.asarray(cx),
        params_sha256=pytree_sha256(p), optimizer_sha256=pytree_sha256(moments),
        sampler_rng_sha256=pytree_sha256((key, states, ck, cx)))
    save_checkpoint(path, p, optimizer_state=moments, metadata=meta)


def restore_state(path: Path, registration: str) -> tuple[tuple, dict]:
    """Reject incomplete or identity-mismatched continuation states."""
    cp = load_checkpoint(path); meta = cp['metadata']
    state = (cp['params'], cp['optimizer_state'], jnp.asarray(meta['training_key']),
        jnp.asarray(meta['training_states']), jnp.asarray(meta['calibration_key']),
        jnp.asarray(meta['calibration_states']))
    if (meta['registration_sha256'] != registration or
        meta['params_sha256'] != pytree_sha256(state[0]) or
        meta['optimizer_sha256'] != pytree_sha256(state[1]) or
        meta['sampler_rng_sha256'] != pytree_sha256(state[2:])):
        raise ValueError('checkpoint parameters/optimizer/RNG seal differs')
    return state, meta


def transition_fixture(state: tuple, info: dict) -> tuple:
    """Keep replay leaves numeric; configuration hashes remain in the ledger."""
    return state, {k: v for k, v in info.items() if not k.endswith('_sha256')}


def terminal_audit(rt: dict, p: object, spec: dict, seed: int, output: Path) -> dict:
    """Fresh Born energy/variance; no SU2 repair, clipping, or endpoint choice."""
    if (output/'terminal_audit.json').exists():
        old = json.loads((output/'terminal_audit.json').read_text())
        if (old['params_sha256'] != pytree_sha256(p) or old['seed'] != seed
                or old['samples'] != spec['samples'] or old['chains'] != spec['chains']):
            raise ValueError('cached untouched audit belongs to another endpoint or protocol')
        return old
    key, init = jax.random.split(jax.random.PRNGKey(seed))
    states = initialize_spin_exchange_chains(init, n_chains=spec['chains'], n_sites=36, n_up=18)
    key, states, rows = make_record_sampler(rt['projected'], spec, rt['edges'])(rt['prepare'](p), key, states)
    psi, hp = jax.block_until_ready(rt['action'](rt['prepare'](p), rows))
    e = np.asarray(hp/psi).reshape(-1, spec['chains'])
    if not np.isfinite(e).all():
        raise FloatingPointError('nonfinite untouched Born audit')
    leaves = np.asarray([np.delete(e, c, axis=1).var()/36 for c in range(spec['chains'])])
    result = dict(energy_per_site=float(e.mean())/36,
        energy_chain_SE=float(e.mean(0).std(ddof=1))/np.sqrt(spec['chains'])/36,
        variance_per_site=float(e.var())/36,
        variance_chain_jackknife_SE=float(np.sqrt((len(leaves)-1)/len(leaves)*np.sum((leaves-leaves.mean())**2))),
        samples=int(e.size), chains=spec['chains'], seed=seed,
        params_sha256=pytree_sha256(p), independent_from_training_and_calibration=True,
        fixed_endpoint_no_selection=True, mixing_certified=False)
    np.savez_compressed(output/'terminal_audit.npz', bits=np.asarray(rows), local_energies=e,
        chain_labels=np.tile(np.arange(spec['chains']), len(e)))
    atomic_json(output/'terminal_audit.json', result)
    return result


def run(plan: dict, arity: str, optimizer: str, phase: str, plan_sha256: str) -> dict:
    """Two accepted steps, isolated full-transition replay, then fixed endpoint."""
    validate_plan(plan)
    if canonical_sha256(plan) != plan_sha256:
        raise ValueError('registered plan changed')
    jax.config.update('jax_enable_x64', True)
    if plan['require_gpu'] and jax.default_backend() != 'gpu':
        raise RuntimeError('allocated GPU required')
    parent = load_parent(plan, arity); output = case_directory(plan, arity, optimizer)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_sha256=plan_sha256, arity=arity, optimizer=optimizer,
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    reg = canonical_sha256(identity); registered = dict(identity, sha256=reg)
    if (output/'registration.json').exists():
        if json.loads((output/'registration.json').read_text()) != registered:
            raise ValueError('immutable experiment registration changed')
    else:
        atomic_json(output/'registration.json', registered)
    rt = runtime(plan, arity, optimizer, parent['params'])
    if (output/'checkpoint.pkl').exists():
        state, meta = restore_state(output/'checkpoint.pkl', reg)
    else:
        ck, ci = jax.random.split(jax.random.PRNGKey(plan['seeds'][arity]['calibration']))
        cx = initialize_spin_exchange_chains(ci, n_chains=plan['calibration']['chains'], n_sites=36, n_up=18)
        prepared = rt['prepare'](parent['params'])
        cx, _, ck, _ = jax.jit(lambda key, rows: run_exchange_sweeps(prepared, key, rows,
            amplitude_fn=rt['projected'], particle_kind='spin',
            sweeps=plan['calibration']['initial_burn_proposals'], exchange_edges=rt['edges']))(ck, cx)
        state = (parent['params'], parent['optimizer_state'], jnp.asarray(parent['metadata']['rng']),
            jnp.asarray(parent['metadata']['sampler_states']), ck, cx)
        meta = dict(registration_sha256=reg, completed_steps=0, parent_completed_steps=200, history=[])
        save_state(output/'checkpoint.pkl', state, meta)
    if (meta['completed_steps'] != len(meta['history'])
            or int(state[1][0]) != 200+meta['completed_steps']):
        raise ValueError('optimizer counter/history disagree with the registered parent and endpoint')
    if phase == 'replay':
        pilot = json.loads((output/'pilot.json').read_text())
        if (pilot['status'] != 'passed'
                or pilot['source_checkpoint_sha256'] != file_sha256(output/'checkpoint.pkl')
                or pilot['next_transition_fixture_sha256'] != file_sha256(output/'next_transition_fixture.pkl')
                or pilot['preflight_sha256'] != file_sha256(output/'preflight.json')):
            raise ValueError('successful measured pilot with unchanged checkpoint/fixture required')
        expected = load_checkpoint(output/'next_transition_fixture.pkl')['params']
        new, info, _ = rt['transition'](state)
        actual = transition_fixture(new, info)
        report = compare_replay(expected, actual, tolerance=plan['tolerance'], diagnostic_prefixes=("[1]",))
        atomic_json(output/'process_replay.json', dict(status='passed' if report['passed'] else 'failed',
            report=report, optimizer_updates_committed=0, new_full_H_centers=2048,
            checkpoint_sha256=file_sha256(output/'checkpoint.pkl')))
        if not report['passed']:
            raise ArithmeticError('complete sampler/H/optimizer replay failed')
        return dict(status='replay_passed')
    if phase not in ('pilot', 'train'):
        raise ValueError('unknown execution phase')
    if phase == 'pilot' and not (output/'preflight.json').exists():
        atomic_json(output/'preflight.json', preflight(rt, state[0], state[3], plan))
    if phase == 'train':
        pilot = json.loads((output/'pilot.json').read_text())
        replay = json.loads((output/'process_replay.json').read_text())
        original_pilot = (file_sha256(output/'checkpoint.pkl') if meta['completed_steps'] == plan['pilot_steps']
                          else meta.get('pilot_checkpoint_sha256'))
        if (pilot['status'] != 'passed'
                or pilot['preflight_sha256'] != file_sha256(output/'preflight.json')
                or pilot['source_checkpoint_sha256'] != original_pilot
                or replay['status'] != 'passed' or replay['checkpoint_sha256'] != original_pilot):
            raise ValueError('isolated pilot replay must precede continuation')
        meta['pilot_checkpoint_sha256'] = original_pilot
    stop = [False]
    def request_stop(*_): stop[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM): signal.signal(sig, request_stop)
    target = plan['pilot_steps'] if phase == 'pilot' else plan['training']['steps']
    started = perf_counter()
    while meta['completed_steps'] < target and not stop[0]:
        tick = perf_counter(); state, info, timings = rt['transition'](state)
        if not info['accepted']:
            atomic_json(output/'rejected_step.json', info)
            raise RuntimeError('unmatched/rejected update: preserve parent, do not claim equal-distance training')
        meta['completed_steps'] += 1
        meta['history'].append(dict(step=meta['completed_steps'], seconds=perf_counter()-tick, **timings, **info))
        save_state(output/'checkpoint.pkl', state, meta)
        atomic_json(output/'progress.json', dict(status='running', completed_steps=meta['completed_steps'],
            last=meta['history'][-1], params_sha256=pytree_sha256(state[0])))
        print(json.dumps(meta['history'][-1]), flush=True)
    if meta['completed_steps'] != target:
        return dict(status='safe_pause', completed_steps=meta['completed_steps'])
    if stop[0]:
        return dict(status='safe_pause', completed_steps=meta['completed_steps'])
    if phase == 'pilot':
        # Charge one uncommitted full next update to the pilot. A fresh process
        # must reproduce it; it is never committed twice to the trained state.
        new, info, _ = rt['transition'](state)
        save_checkpoint(output/'next_transition_fixture.pkl', transition_fixture(new, info))
        warm_seconds = meta['history'][-1]['seconds']
        forecast = warm_seconds*(plan['training']['steps']-target+
            plan['audit']['samples']/2048)+600.
        report = dict(status='pending_checks', completed_steps=target,
            warm_step_seconds=warm_seconds, forecast_remaining_seconds=forecast,
            maximum_remaining_seconds=plan['budget']['train_seconds'],
            source_checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
            next_transition_fixture_sha256=file_sha256(output/'next_transition_fixture.pkl'),
            preflight_sha256=file_sha256(output/'preflight.json'),
            first_training_rows_sha256=meta['history'][0]['training_rows_sha256'],
            first_calibration_rows_sha256=meta['history'][0]['calibration_rows_sha256'],
            uncommitted_fixture_full_H_centers=2048,
            peak_device_bytes=max(int((d.memory_stats() or {}).get('peak_bytes_in_use', 0)) for d in jax.local_devices()))
        passed = (forecast <= plan['budget']['train_seconds'] and info['accepted']
                  and report['peak_device_bytes'] <= plan['budget']['maximum_peak_device_bytes'])
        report['status'] = 'passed' if passed else 'failed'
        atomic_json(output/'pilot.json', report)
        if not passed:
            raise RuntimeError('pilot exceeds fixed remaining budget or cannot replay a matched next step')
        return report
    final = terminal_audit(rt, state[0], plan['audit'],
        plan['seeds'][arity]['audit_'+optimizer], output)
    result = dict(status='fixed_steps_complete', completed_steps=target,
        parent_completed_steps=200, parameter_count=mixed_parameter_count(state[0]),
        optimizer_moments_policy='inherit200; directSR retains but does not use Adam moments',
        covariance_convention='F=Cov(real logabs scores); g=2Cov(scores,E)',
        terminal_audit=final, history=meta['history'],
        params_sha256=pytree_sha256(state[0]), checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
        registration_sha256=reg, last_session_seconds=perf_counter()-started,
        full_H_training_centers=2048*target, replay_H_centers=4096,
        preflight_H_centers=4, calibration_H_centers=0,
        terminal_audit_H_centers=plan['audit']['samples'],
        equal_counts_not_identical_Born_rows_after_divergence=True,
        different_arity_parents_not_an_arity_causal_comparison=True)
    atomic_json(output/'results.json', result)
    atomic_json(output/'progress.json', dict(status='fixed_steps_complete', completed_steps=target))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--arity', choices=ARITIES, required=True)
    parser.add_argument('--optimizer', choices=OPTIMIZERS, required=True)
    parser.add_argument('--phase', choices=('pilot', 'replay', 'train'), required=True)
    args = parser.parse_args()
    answer = run(json.loads(args.plan.read_text()), args.arity, args.optimizer,
        args.phase, args.plan_sha256)
    print(json.dumps({k: v for k, v in answer.items() if k != 'history'}), flush=True)
    if answer.get('status') == 'safe_pause':
        raise SystemExit(75)
