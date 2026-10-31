"""Opt-in CT6 production with independent, bounded blockwise SR feedback.

The physical CT/A1/H kernels and native real W coordinates are unchanged.
This NEW profile never changes sealed legacy plans or running allocations.
The budget counts optimizer ATTEMPTS, with accepted updates reported
separately. Rejections consume fresh fit/calibration RNG without changing
parameters or optimizer moments. Monitor data are consumed training feedback,
never untouched terminal precision. No submission or automatic requeue occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
from time import perf_counter, time
from typing import Any, Callable

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.adaptive_sr_controls import (
    AdaptiveSRPolicy, AdaptiveSRState, SRMonitor, SRResourceEnvelope,
    advance_sr_controls, make_controlled_importance_sr, restore_sr_controls,
    select_controlled_finite_grid, sr_control_checkpoint, validate_control_state,
)
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.finite_candidate_calibration import bilateral_direction_moments
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from . import ct_bounded_followup as legacy
from . import ct6_importance_amplitude_continuation as c6
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT = Path(__file__).resolve().parents[3]
TASK = 'CT6_adaptive_continuation_v1'
FACTORS = c6.FACTORS
FRACTIONS = c6.RELATIVE_GRID


def required_seed_roles(steps: int = 32) -> tuple[str, ...]:
    """Declare disjoint monitoring/final streams; fit RNG remains inherited."""
    return ('preflight', 'audit', 'audit_statistics') + tuple(
        f'monitor{i}_{role}' for i in range(1, steps // 8)
        for role in ('sample', 'statistics', 'jvp'))


def make_plan(j2: float, parent_directory: Path, *, seeds: dict[str, int],
              steps: int = 32, wall_cap_seconds: int = 86400) -> dict[str, Any]:
    """Read immutable parent seals and create a not-yet-authorized new plan.

    Parent directories are the completed global474 J0 / global500 J.125
    bounded followups. Change a local mirror path to its identically hashed
    remote directory before sealing registration, never after execution.
    """
    parent = Path(parent_directory)
    policy = AdaptiveSRPolicy(fit_levels=(5120, 5632), initial_fs_radius=.00125,
        minimum_fs_radius=.00025, maximum_fs_radius=.0025,
        initial_damping=.05, minimum_damping=.0125, maximum_damping=.8,
        maximum_monitor_looks=4 if steps == 32 else 8, success_patience=2, noise_patience=2)
    return dict(task=TASK, kind='ct6', device='A100', j2=float(j2),
        model=dict(length=6, multiplicity_cap=28, maximum_q=10),
        arity='ternary_joint' if j2 == 0. else 'binary_truncated', binary_rank=8,
        parent_format='CT_bounded_followup_v1', parent_directory=str(parent),
        parent_registration_sha256=file_sha256(parent / 'registration.json'),
        parent_checkpoint_sha256=file_sha256(parent / 'checkpoint.pkl'),
        parent_audit_sha256=file_sha256(parent / 'terminal_audit/audit.json'),
        expected_parent_global_step=474 if j2 == 0. else 500,
        steps=steps, block_attempts=8, monitor_after=list(range(8, steps, 8)),
        fit_records=5120, calibration_records=4096, calibration_split=2048,
        monitor_records=4096, monitor_groups=128, monitor_jvp_records=256,
        monitor_jvp_groups=32, audit_records=8192, audit_groups=128,
        interim_records=0, interim_groups=0, epsilon=.125, damping=.05,
        learning_rate=.03, target_FS=.001, maximum_FS=.00125,
        native_factors=FACTORS.tolist(), adaptive_policy=asdict(policy), seeds=dict(seeds),
        maximum_fit_H_records=(steps + 1) * 5632, wall_cap_seconds=wall_cap_seconds,
        maximum_peak_device_bytes=70 * 1024**3, projection_chunk=4, action_chunk=1,
        score_chunk=2, H_chunk=1, parameter_chunk_size=8192,
        monitor_min_effective_contributions=64, monitor_maximum_fraction=.05,
        new_control_compile_reserve_seconds=600., allocator_preflight=True,
        step_counter='attempts', attempts_not_successful_updates=True, automatic_next_allocation=False, no_requeue=True,
        execution_authorized=False, training_monitor_is_not_terminal_audit=True)


def policy_from_plan(plan: dict) -> AdaptiveSRPolicy:
    """Restore immutable policy types without silently choosing new controls."""
    data = dict(plan['adaptive_policy']); data['fit_levels'] = tuple(data['fit_levels'])
    return AdaptiveSRPolicy(**data)


def validate_plan(plan: dict, *, executing: bool = False) -> None:
    """Keep this opt-in profile bounded and separate from all old profiles."""
    if plan.get('task') != TASK or plan.get('kind') != 'ct6' or plan.get('device') != 'A100':
        raise ValueError('new CT6/A100 adaptive profile required')
    j2 = plan.get('j2')
    if isinstance(j2, bool) or j2 not in (0., .125):
        raise ValueError('J2=0 or .125 required')
    fixed = dict(model=dict(length=6, multiplicity_cap=28, maximum_q=10), binary_rank=8,
        arity='ternary_joint' if j2 == 0. else 'binary_truncated',
        parent_format='CT_bounded_followup_v1', expected_parent_global_step=474 if j2 == 0. else 500,
        block_attempts=8, fit_records=5120, calibration_records=4096, calibration_split=2048,
        monitor_records=4096, monitor_groups=128, monitor_jvp_records=256, monitor_jvp_groups=32,
        audit_records=8192, audit_groups=128, interim_records=0, interim_groups=0,
        epsilon=.125, damping=.05, learning_rate=.03, target_FS=.001, maximum_FS=.00125,
        native_factors=FACTORS.tolist(), maximum_peak_device_bytes=70 * 1024**3,
        monitor_min_effective_contributions=64, monitor_maximum_fraction=.05,
        step_counter='attempts', attempts_not_successful_updates=True, automatic_next_allocation=False, no_requeue=True,
        training_monitor_is_not_terminal_audit=True, allocator_preflight=True)
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError('frozen w28, initial statistics, monitoring and device bounds changed')
    if plan.get('steps') not in (32, 64) or isinstance(plan.get('steps'), bool):
        raise ValueError('32 or 64 total optimizer attempts required')
    if plan.get('monitor_after') != list(range(8, plan['steps'], 8)):
        raise ValueError('only preregistered interior eight-attempt blocks are monitored')
    policy = policy_from_plan(plan)
    expected_policy = AdaptiveSRPolicy(fit_levels=(5120, 5632), initial_fs_radius=.00125,
        minimum_fs_radius=.00025, maximum_fs_radius=.0025, initial_damping=.05,
        minimum_damping=.0125, maximum_damping=.8, maximum_monitor_looks=4 if plan['steps'] == 32 else 8,
        success_patience=2, noise_patience=2)
    if policy != expected_policy:
        raise ValueError('adaptive policy differs from the registered physical/control envelope')
    for name in ('projection_chunk', 'action_chunk', 'score_chunk', 'H_chunk', 'parameter_chunk_size'):
        if isinstance(plan.get(name), bool) or not isinstance(plan.get(name), int) or plan[name] < 1:
            raise ValueError('positive explicit kernel chunks required')
    if (not isinstance(plan.get('wall_cap_seconds'), int) or isinstance(plan['wall_cap_seconds'], bool)
            or not 0 < plan['wall_cap_seconds'] <= 172800
            or plan.get('maximum_fit_H_records') != (plan['steps'] + 1) * 5632
            or plan.get('new_control_compile_reserve_seconds') != 600.):
        raise ValueError('finite training work/wall and recompile reserves required')
    for name in ('parent_registration_sha256', 'parent_checkpoint_sha256', 'parent_audit_sha256'):
        digest = plan.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('sealed completed parent identities required')
    roles = plan.get('seeds', {})
    if set(roles) != set(required_seed_roles(plan['steps'])):
        raise ValueError('all preregistered monitor/JVP/final stream identities required')
    if (any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**32 for v in roles.values())
            or len(set(roles.values())) != len(roles)):
        raise ValueError('fresh distinct uint32 streams required')
    if executing and plan.get('execution_authorized') is not True:
        raise PermissionError('separate explicit authorization required for numerical execution')


def save_adaptive_state(path: Path, state: tuple, controls: AdaptiveSRState, progress: dict,
                        *, plan: dict, registration: dict) -> None:
    """One atomic params/optimizer/native-RNG/controller/monitor-history bundle."""
    policy = policy_from_plan(plan); validate_control_state(policy, controls)
    if state[2]['next_step'] != registration['initial_native_cursor'] + progress['attempts']:
        raise ValueError('native RNG cursor must count every attempted optimizer step')
    if (not 0 <= progress['accepted_updates'] <= progress['attempts'] <= plan['steps']
            or len(progress['transitions']) != progress['attempts']
            or sum(bool(r['accepted']) for r in progress['transitions']) != progress['accepted_updates']):
        raise ValueError('accepted updates and total attempt budget differ')
    payload = dict(task=plan['task'], plan_sha256=canonical_sha256(plan), progress=deepcopy(progress),
        absolute_completed_steps=registration['parent_global_step'] + progress['accepted_updates'],
        native_origin=registration['native_origin'], initial_native_cursor=registration['initial_native_cursor'],
        rng=deepcopy(state[2]), adaptive_sr=sr_control_checkpoint(policy, controls),
        parameter_sha256=pytree_sha256(state[0]), optimizer_sha256=pytree_sha256(state[1]),
        rng_sha256=canonical_sha256(state[2]), registration_sha256=canonical_sha256(registration))
    save_checkpoint(path, state[0], optimizer_state=state[1], metadata=payload)


def load_adaptive_state(path: Path, *, plan: dict, registration: dict) -> tuple[tuple, AdaptiveSRState, dict]:
    """Verify physical state, exact next RNG and the adaptive decision history."""
    saved = load_checkpoint(path); meta = saved['metadata']
    state = (saved['params'], saved['optimizer_state'], meta['rng'])
    controls = restore_sr_controls(policy_from_plan(plan), meta['adaptive_sr'])
    progress = meta['progress']
    if (meta.get('task') != plan['task'] or meta['plan_sha256'] != canonical_sha256(plan)
            or meta['registration_sha256'] != canonical_sha256(registration)
            or pytree_sha256(state[0]) != meta['parameter_sha256']
            or pytree_sha256(state[1]) != meta['optimizer_sha256']
            or canonical_sha256(state[2]) != meta['rng_sha256']
            or meta['absolute_completed_steps'] != registration['parent_global_step'] + progress['accepted_updates']
            or state[2]['next_step'] != registration['initial_native_cursor'] + progress['attempts']
            or not 0 <= progress['accepted_updates'] <= progress['attempts'] <= plan['steps']
            or len(progress['transitions']) != progress['attempts']
            or sum(bool(r['accepted']) for r in progress['transitions']) != progress['accepted_updates']):
        raise ValueError('adaptive physical/optimizer/RNG/progress seal mismatch')
    if (len(progress['monitor_history']) != controls.looks
            or len(progress['consumed_monitor_roles']) != controls.looks):
        raise ValueError('adaptive monitor history and control state differ')
    return state, controls, progress


def make_runtime(plan: dict, state: tuple, registration: dict) -> tuple[dict, dict]:
    """Reuse unchanged kernels, binding only the new opt-in control wrapper."""
    shadow = dict(plan, task='CT_bounded_followup_v1')
    rt, kernel = legacy.make_runtime(shadow, state, registration)
    rt['controlled_update'] = make_controlled_importance_sr(
        lambda params, row: rt['projected'](rt['prepare'](params), row), state[0], policy_from_plan(plan),
        score_chunk=plan['score_chunk'], parameter_chunk_size=plan['parameter_chunk_size'],
        prepare=rt['prepare'], prepared_amplitude=rt['projected'])
    return rt, kernel


def initial_progress() -> dict:
    """Attempt and accepted-update clocks are intentionally different."""
    return dict(attempts=0, accepted_updates=0, fit_H_records_consumed=0,
        transitions=[], monitor_history=[], consumed_monitor_roles=[], block_predicted_decrease=0.,
        block_geometry_ok=True, stop_reason=None)


def consume_attempt(progress: dict, report: dict, *, replay_fit_records: int = 0) -> dict:
    """Record one complete attempt, including rejected proposals and their work."""
    result = deepcopy(progress)
    if report['attempt'] != result['attempts'] + 1:
        raise ValueError('attempts must commit exactly once in their native order')
    result['attempts'] += 1
    result['accepted_updates'] += int(report['accepted'])
    result['fit_H_records_consumed'] += report['work']['fit_H_centers'] + replay_fit_records
    result['transitions'].append(report)
    result['block_predicted_decrease'] += report.get('predicted_decrease_per_site', 0.)
    result['block_geometry_ok'] &= bool(report['accepted'])
    return result


def transition(plan: dict, kernel: dict, rt: dict, state: tuple,
               controls: AdaptiveSRState, path: Path, *, attempt: int) -> tuple[tuple, dict]:
    """Full native-W SR, independent finite-FS guard and transactional rejection.

    A rejected attempt STILL advances the inherited fit/calibration RNG, but
    neither parameters nor optimizer moments. This differs deliberately from
    legacy stop-on-first-rejection workflows; those modules are untouched.
    """
    validate_control_state(policy_from_plan(plan), controls)
    if path.exists():
        raise FileExistsError('fresh immutable attempt directory required')
    path.mkdir(parents=True)
    parent, moments, rng = state
    seeds, next_rng = c6.role_seeds(rng, rng['next_step'])
    report = dict(attempt=attempt, native_cursor=rng['next_step'], accepted=False,
        selected_index=4, actual_native_factor=0., predicted_decrease_per_site=0.,
        parent_params_sha256=pytree_sha256(parent), seeds=seeds,
        rng_before_sha256=canonical_sha256(rng), rng_after_sha256=canonical_sha256(next_rng),
        controls=asdict(controls), selection_target_FS=.8 * controls.fs_radius,
        timings={}, work=dict(fit_draws=0, fit_H_centers=0, fit_native_H_replay_centers=0,
            full_native_jacobian_rows=0, calibration_draws=0, calibration_bank_centers=0,
            native_amplitude_replay_centers=0))
    atomic_json(path/'streams_frozen.json', report)
    start = perf_counter()
    def timed(name: str, call: Callable):
        tick = perf_counter(); value = call(); jax.block_until_ready(value)
        report['timings'][name] = perf_counter() - tick
        if c6.diagnostic._device_peak() > plan['maximum_peak_device_bytes']:
            raise MemoryError('registered live-byte ceiling exceeded')
        atomic_json(path/'progress.json', report)
        return value
    try:
        bank, note = timed('parent_sampler_canonicalization',
            lambda: c6.diagnostic.normalized_raw_bank(parent, rt))
        scale = np.sqrt(note['original_raw_norm'])
        rows, draw = timed('fit_full_support_IID',
            lambda: c6.draw_rows([bank], rt, controls.fit_records, seeds['fit']))
        c6.save_draw(path/'fit_draw', rows, draw); report['work']['fit_draws'] = len(rows)
        product_report = dict(partial_artifact_prefix=str(path/'fit_partial'))
        report['fit_products_work'] = product_report
        products = timed('fit_full_H', lambda: c6._product_batches(kernel, rt, [bank], rows,
            product_report, 'adaptive fitH', rt.get('training_remaining')))[:, 0]
        report['work']['fit_H_centers'] = product_report['scalar_H_centers']
        density = full_support_sz0_density(products[:, 2].real, products[:, 3].real,
            n_sites=36, epsilon=plan['epsilon'])
        native_phi, native_hp = scale * products[:, 0], scale * products[:, 1]
        if any(np.max(abs(v.imag)) > 1e-13 * max(np.max(abs(v)), 1e-300)
               for v in (native_phi, native_hp)):
            raise ArithmeticError('real CT6 family acquired complex physical products')
        native_phi, native_hp = native_phi.real, native_hp.real
        np.savez(path/'fit_products.npz', rows=rows, psi=native_phi, hpsi=native_hp,
            proposal=density, canonical_products=products, exact_native_to_canonical_scale=scale)
        replay = timed('native_fit_H_scale_replay', lambda: rt['action'](rt['prepare'](parent), rows[:2]))
        expected = np.stack((native_phi[:2], native_hp[:2]))
        actual = np.stack([np.asarray(v) for v in replay])
        errors = np.max(abs(actual-expected), axis=1)/np.maximum(np.max(abs(expected), axis=1), 1e-300)
        report['native_fit_H_scale_replay_relative_errors'] = errors.tolist()
        report['work']['fit_native_H_replay_centers'] = 2
        if not np.isfinite(errors).all() or max(errors) > 2e-11:
            raise ArithmeticError('native psi/J/Hpsi scale contract replay failed')
        proposed, next_moments, sr = timed('full_native_importance_amplitude_SR',
            lambda: rt['controlled_update'](parent, moments, rows, native_hp, density,
                plan['learning_rate'], controls=controls))
        report['work']['full_native_jacobian_rows'] = len(rows)
        report['SR'] = jax.tree.map(lambda x: np.asarray(x).item(), sr)
        delta = jax.tree.map(lambda a, b: a-b, proposed, parent)
        save_checkpoint(path/'raw_SR_direction.pkl', delta, optimizer_state=next_moments,
            metadata=dict(parent_params_sha256=report['parent_params_sha256'],
                next_training_rng=next_rng, all_native_columns=True, proposed_optimizer_uncommitted=True))
        if not bool(sr['accepted']):
            report['reason'] = 'numerically_invalid_SR_attempt_consumed'
            return (parent, moments, next_rng), report
        candidates = [jax.tree.map(lambda p, d: p+factor*d, parent, delta) for factor in FACTORS]
        banks, notes = [], []
        for i, candidate in enumerate(candidates):
            b, n = c6.diagnostic.normalized_raw_bank(candidate, rt); banks.append(b); notes.append(n)
            save_checkpoint(path/f'finite_native_{i}.pkl', candidate,
                metadata=dict(native_factor=float(FACTORS[i])))
        panels = []
        for role in ('selection', 'guard'):
            bits, draw = timed(role+'_IID', lambda role=role: c6.diagnostic.draw_orbit_rows(
                banks, rt, plan['calibration_split'], seeds[role]))
            c6.save_draw(path/(role+'_draw'), bits, draw); panels.append(bits)
            report['work']['calibration_draws'] += len(bits)
        bits = np.concatenate(panels); columns = []
        for i, b in enumerate(banks):
            prepared = c6.diagnostic.prepare_feature_lut_bank(b, rt['cfg'])
            columns.append(np.asarray(timed('orbit_bank_'+str(i),
                lambda p=prepared: rt['orbit_query'](p, bits))))
            report['work']['calibration_bank_centers'] += len(bits)
        products = np.stack(columns, axis=1); phi = products[:, :, 0]
        q = products[:, :, 1].real.mean(1)
        np.savez(path/'calibration_products.npz', rows=bits, phi=phi,
            guide_densities=products[:, :, 1], proposal=q)
        errors = []
        for i, candidate in enumerate(candidates):
            value = np.asarray(timed('native_replay_'+str(i),
                lambda p=candidate: c6.diagnostic.native_projected_amplitude(rt, p, bits[:2])))
            value = value / np.sqrt(notes[i]['original_raw_norm'])
            errors.append(float(np.max(abs(value-phi[:2, i]))/max(np.max(abs(phi[:2, i])), 1e-300)))
            report['work']['native_amplitude_replay_centers'] += 2
        if not np.isfinite(errors).all() or max(errors) > 2e-11:
            raise ArithmeticError('finite native/canonical amplitude replay failed')
        selected, calibration = select_controlled_finite_grid(policy_from_plan(plan), controls,
            phi, q, FRACTIONS, split=plan['calibration_split'])
        report.update(selected_index=selected, calibration=calibration,
            native_amplitude_replay_errors=errors, actual_native_factor=float(FACTORS[selected]))
        if selected == 4 or pytree_sha256(candidates[selected]) == report['parent_params_sha256']:
            report['reason'] = 'independent_finite_geometry_rejected_attempt_consumed'
            return (parent, moments, next_rng), report
        report.update(accepted=True, reason='full_W_proposal_and_independent_finite_FS_passed',
            candidate_params_sha256=pytree_sha256(candidates[selected]),
            predicted_decrease_per_site=max(0., -float(FACTORS[selected]) *
                float(sr['raw_energy_directional_derivative']) / 36.))
        return (candidates[selected], next_moments, next_rng), report
    finally:
        report['work']['fit_H_centers'] = report.get('fit_products_work', {}).get(
            'scalar_H_centers', report['work']['fit_H_centers'])
        report['wall_seconds'] = perf_counter()-start
        report['peak_device_bytes'] = c6.diagnostic._device_peak()
        atomic_json(path/'transition.json', report)


def monitor_coverage(plan: dict, moments: dict) -> bool:
    """Operational concentration checks, NOT certification of unobserved tails."""
    for name in ('norm_concentration', 'H2_concentration', 'residual_concentration'):
        d = moments[name]
        effective, maximum = map(np.asarray, (d['effective_contributions'], d['maximum_sample_fraction']))
        if (not np.isfinite(effective).all() or not np.isfinite(maximum).all()
                or np.any(effective < plan['monitor_min_effective_contributions'])
                or np.any(maximum > plan['monitor_maximum_fraction'])):
            return False
    return True


def independent_gain_proxy(progress: dict, slope: dict, block_attempts: int) -> dict:
    """Conservative local proxy, not a forecast of the already completed block.

    The last ACTUAL finite parameter displacement has an independently measured
    two-sided slope. Multiplying it by the number of accepted steps is an
    explicitly heuristic representativeness assumption, not a sum of the
    block's unknown population derivatives. Cap by the summed in-fit linear
    prediction; never use observed paired energy gain as its own prediction.
    """
    raw = float(progress['block_predicted_decrease'])
    accepted = sum(bool(r['accepted']) for r in progress['transitions'][-block_attempts:])
    value = slope['directional_slope']
    proxy = 0. if value is None else min(raw, accepted*max(0., -float(value)))
    return dict(controller_predicted_decrease_per_site=proxy,
        raw_in_fit_linear_block_prediction_per_site=raw,
        independent_last_actual_finite_direction_slope_per_site=value,
        independent_last_actual_finite_direction_slope_SE_per_site=slope['directional_slope_se'],
        accepted_steps_in_block=accepted,
        rule='min(sum accepted in-fit first-order decreases, accepted_count * max(0,-independent last-step slope))',
        local_representativeness_heuristic_not_block_forecast=True,
        paired_observed_energy_not_used_as_prediction=True)


def frozen_direction_monitor(plan: dict, kernel: dict, rt: dict, parent: dict,
                             endpoint: dict, output: Path, seed: int, remaining: Callable) -> dict:
    """Fresh positive-Q two-sided full-H/JVP slope for one actual finite step.

    This is scheduling feedback, not a tangent-norm certificate or an estimate
    of the exact gradient norm. Q includes the exchange bridge and uniform
    Sz=0 support; records with target nodes are retained.
    """
    output.mkdir()
    delta = jax.tree.map(lambda a, b: a-b, endpoint, parent)
    if not any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(delta)):
        report = dict(status='zero_rejected_direction_no_noise_claim', seed=seed,
            directional_slope=None, directional_slope_se=None, full_H_centers=0)
        atomic_json(output/'results.json', report); return report
    start = perf_counter()
    bank, note = c6.diagnostic.normalized_raw_bank(parent, rt)
    rows, draw = c6.draw_rows([bank], rt, plan['monitor_jvp_records'], seed)
    c6.save_draw(output/'draw', rows, draw)
    products_report = dict(partial_artifact_prefix=str(output/'partial_products'))
    products = c6._product_batches(kernel, rt, [bank], rows, products_report,
        'monitor JVP proposalH', remaining)[:, 0]
    q = full_support_sz0_density(products[:, 2].real, products[:, 3].real,
        n_sites=36, epsilon=plan['epsilon'])
    prepared, tangent = jax.jvp(rt['prepare'], (parent,), (delta,))
    def one(prepared, tangent, bits):
        return jax.jvp(lambda b: rt['action'](b, bits), (prepared,), (tangent,))
    query = jax.jit(one); values = []
    for offset in range(0, len(rows), plan['H_chunk']):
        if remaining() <= 120:
            raise TimeoutError('independent directional monitor reserve exhausted')
        (psi, hp), (dpsi, hdp) = jax.block_until_ready(query(prepared, tangent,
            rows[offset:offset+plan['H_chunk']]))
        values.append(np.stack([np.asarray(v) for v in (psi, hp, dpsi, hdp)], axis=1))
        if c6.diagnostic._device_peak() > plan['maximum_peak_device_bytes']:
            raise MemoryError('full-H/JVP exceeded registered device ceiling')
    v = np.concatenate(values)
    np.savez(output/'products.npz', rows=rows, psi=v[:, 0], hpsi=v[:, 1],
        dpsi=v[:, 2], hdpsi=v[:, 3], proposal=q)
    moments = bilateral_direction_moments(*v.T, q, groups=plan['monitor_jvp_groups'], n_sites=36)
    report = dict(status='complete', seed=seed, moments=moments,
        directional_slope=moments['values'][2],
        directional_slope_se=moments['independent_group_jackknife_SE'][2],
        full_H_centers=products_report['scalar_H_centers'] + 2*len(rows),
        actual_wall_seconds=perf_counter()-start,
        both_coherent_terms=True, actual_finite_parameter_direction=True,
        native_to_canonical_scale=float(np.sqrt(note['original_raw_norm'])),
        consumed_as_training_feedback_not_final_precision=True)
    atomic_json(output/'results.json', report)
    return report


def resources_for_next_block(plan: dict, registration: dict, progress: dict,
                             *, available_seconds: float, reserve_seconds: float) -> SRResourceEnvelope:
    """Conservative measured work envelope; never call after the final attempt."""
    left = plan['steps']-progress['attempts']
    if left <= 0:
        raise ValueError('no future control/resource decision at the terminal endpoint')
    recent = progress['transitions'][-plan['block_attempts']:]
    rate = max((r['timings']['fit_full_H'] + r['timings']['full_native_importance_amplitude_SR']) /
        r['work']['fit_H_centers'] for r in recent)
    other = max(max(0., r['wall_seconds'] - r['timings']['fit_full_H'] -
        r['timings']['full_native_importance_amplitude_SR']) for r in recent)
    # One new compiled control/sample shape can occur next block. Reserving it
    # per remaining step is conservative and cannot grant extra wall time.
    other += plan['new_control_compile_reserve_seconds'] / left
    return SRResourceEnvelope(parameter_count=registration['parameter_count'],
        maximum_device_bytes=plan['maximum_peak_device_bytes'], remaining_updates=left,
        remaining_fit_H_records=max(0, plan['maximum_fit_H_records']-progress['fit_H_records_consumed']),
        remaining_training_seconds=max(0., available_seconds-reserve_seconds),
        seconds_per_fit_record=rate, other_seconds_per_step=other)


def monitor_block(plan: dict, kernel: dict, rt: dict, block_start: tuple,
                  endpoint: tuple, direction_parent: dict, controls: AdaptiveSRState,
                  progress: dict, output: Path, resources: SRResourceEnvelope,
                  remaining: Callable) -> tuple[AdaptiveSRState, dict, dict]:
    """Consume exactly one independent block panel, never the final-audit stream."""
    attempt = progress['attempts']
    if attempt not in plan['monitor_after']:
        raise ValueError('only preregistered interior frozen blocks can adapt controls')
    output.mkdir(parents=True)
    look = attempt // plan['block_attempts']
    roles = {r: plan['seeds'][f'monitor{look}_{r}'] for r in ('sample', 'statistics', 'jvp')}
    atomic_json(output/'streams_reserved.json', dict(roles,
        role='independent_training_monitor', attempt=attempt,
        not_untouched_terminal_precision=True))
    monitor_kernel = dict(kernel, interim_records=plan['monitor_records'])
    check = c6.paired_audit(monitor_kernel, rt, block_start[0], endpoint,
        output/'paired_full_H', plan['monitor_records'], roles['sample'], roles['statistics'], remaining)
    slope = frozen_direction_monitor(plan, kernel, rt, direction_parent, endpoint[0],
        output/'directional_full_H_JVP', roles['jvp'], remaining)
    moments = check['moments']; delta = moments['candidate_minus_control']
    se = moments['paired_independent_block_jackknife_SE']
    gain = independent_gain_proxy(progress, slope, plan['block_attempts'])
    panel = SRMonitor(panel_id=canonical_sha256(dict(attempt=attempt, roles=roles,
        parent=pytree_sha256(block_start[0]), endpoint=pytree_sha256(endpoint[0]),
        paired_products=file_sha256(output/'paired_full_H/products.npz'))),
        source_role='independent_training_monitor', sample_seed=roles['sample'],
        statistics_seed=roles['statistics'], completed_step=attempt,
        parent_sha256=pytree_sha256(block_start[0]), endpoint_sha256=pytree_sha256(endpoint[0]),
        delta_energy=delta[0], energy_se=se[0], delta_variance=delta[2], variance_se=se[2],
        predicted_decrease=gain['controller_predicted_decrease_per_site'],
        geometry_ok=bool(progress['block_geometry_ok']), coverage_ok=monitor_coverage(plan, moments),
        directional_slope=slope['directional_slope'], directional_slope_se=slope['directional_slope_se'])
    # The inherited native generator already excludes every planned monitor
    # and audit identity at binding. Also preview the remaining draw history.
    future = deepcopy(endpoint[2]); reserved = set(future['excluded_seeds']) | set(future['consumed_role_seeds'])
    for i in range(future['next_step'], future['next_step']+plan['steps']-attempt):
        seeds, future = c6.role_seeds(future, i); reserved.update(seeds.values())
    reserved.update(plan['seeds'].values()); reserved.difference_update((roles['sample'], roles['statistics']))
    updated, decision = advance_sr_controls(policy_from_plan(plan), controls, panel, resources,
        reserved_seeds=tuple(sorted(reserved)))
    decision.update(jvp=slope, paired_full_H=check, gain_proxy=gain,
        support_and_concentration_checks_are_operational_not_tail_certificate=True,
        step_counter='optimizer_attempts', accepted_updates=progress['accepted_updates'])
    result = deepcopy(progress)
    result['monitor_history'].append(decision)
    result['consumed_monitor_roles'].append(roles)
    result['block_predicted_decrease'] = 0.; result['block_geometry_ok'] = True
    atomic_json(output/'decision.json', decision)
    return updated, result, decision


def compare_transition(expected: tuple, actual: tuple, first: dict, second: dict,
                       left: Path, right: Path) -> dict:
    """Process replay commits once even when the first proposal was rejected."""
    comparison = c6.diagnostic.legacy.compare_replay(expected[:2], actual[:2], tolerance=2e-11)
    if not comparison['passed'] or canonical_sha256(expected[2]) != canonical_sha256(actual[2]):
        raise ArithmeticError('native parameters, optimizer or next-RNG process replay failed')
    for name in ('accepted', 'selected_index', 'actual_native_factor', 'seeds', 'rng_after_sha256', 'controls'):
        if canonical_sha256(first.get(name)) != canonical_sha256(second.get(name)):
            raise ArithmeticError('replayed control/finite-choice/stream differs: '+name)
    for name in ('fit_products.npz', 'selection_draw.npz', 'guard_draw.npz'):
        a, b = left/name, right/name
        if a.exists() != b.exists():
            raise ArithmeticError('replay changed the completed transaction stages')
        if a.exists():
            with np.load(a) as x, np.load(b) as y:
                if not np.array_equal(x['rows'], y['rows']):
                    raise ArithmeticError('replayed fit/selection/guard rows changed')
    def numeric(data):
        return jax.tree.map(lambda x: np.frombuffer(x.encode(), np.uint8) if isinstance(x, str) else x, data)
    cal = c6.diagnostic.legacy.compare_replay(numeric(first.get('calibration', {})),
        numeric(second.get('calibration', {})), tolerance=2e-11)
    if not cal['passed']:
        raise ArithmeticError('finite physical geometry replay failed')
    return dict(status='passed', native=comparison, calibration=cal,
        params_optimizer_next_RNG_controls_compared=True, physical_rows_identical=True,
        first_attempt_accepted=bool(first['accepted']), attempts_committed_by_replay=0)


def reserve_costs(plan: dict, warm_H_seconds: float) -> dict:
    """Reserve monitoring and untouched terminal products before training."""
    if not np.isfinite(warm_H_seconds) or warm_H_seconds <= 0:
        raise ValueError('positive measured complete-H throughput required')
    return dict(warm_H_seconds=warm_H_seconds,
        audit_reserve_seconds=1.5*2*plan['audit_records']*warm_H_seconds+600.,
        monitor_reserve_seconds=1.5*(2*plan['monitor_records']+
            4*plan['monitor_jvp_records'])*warm_H_seconds+600.,
        all_reserves_charged_to_same_allocation=True, no_hardware_acceleration_assumed=True)


def freeze_endpoint(plan: dict, out: Path, registration: dict, *, reason: str | None = None) -> dict:
    """Freeze the LAST committed prefix, never the best monitoring endpoint."""
    state, controls, progress = load_adaptive_state(out/'checkpoint.pkl', plan=plan, registration=registration)
    result = dict(status='fixed_attempt_prefix_frozen', attempts=progress['attempts'],
        accepted_updates=progress['accepted_updates'], requested_attempts=plan['steps'],
        fixed_attempt_budget_completed=progress['attempts'] == plan['steps'],
        absolute_completed_steps=registration['parent_global_step']+progress['accepted_updates'],
        native_next_cursor=state[2]['next_step'], parameter_sha256=pytree_sha256(state[0]),
        optimizer_sha256=pytree_sha256(state[1]), rng_sha256=canonical_sha256(state[2]),
        controller_sha256=sr_control_checkpoint(policy_from_plan(plan), controls)['sha256'],
        checkpoint_sha256=file_sha256(out/'checkpoint.pkl'), reason=reason or progress.get('stop_reason'),
        no_best_endpoint_selection=True, monitoring_not_terminal_precision=True)
    atomic_json(out/'endpoint_frozen.json', result)
    return result


def restart_boundary_reason(plan: dict, out: Path, controls: AdaptiveSRState, progress: dict) -> str | None:
    """Refuse an uncommitted feedback transaction or missing block parent."""
    past = [i for i in plan['monitor_after'] if i <= progress['attempts']]
    if (controls.looks != len(past) or len(progress['monitor_history']) != len(past)
            or (past and controls.last_completed_step != past[-1])):
        return 'uncommitted_monitor_boundary_quarantined_no_unmonitored_resume'
    last = past[-1] if past else 0
    if not (out/f'block_start_{last}.pkl').exists():
        return 'missing_committed_block_parent_quarantined_no_unmonitored_resume'
    return None


def run(plan_path: Path, out: Path, phase: str, *, profile: Any = None) -> dict:
    """Allocated-GPU-only four-process prepare/replay/train/terminal audit.

    Only atomic completed transactions resume. A partially queried step or
    monitor is quarantined and the existing committed prefix is frozen; the
    driver never silently repeats consumed data or launches another job.
    """
    plan = json.loads(plan_path.read_text())
    validator = validate_plan if profile is None else profile.validate_plan
    validator(plan, executing=True)
    if phase not in ('prepare', 'replay', 'train', 'audit'):
        raise ValueError('unknown process phase')
    if jax.default_backend() != 'gpu' or not all('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('explicitly allocated compatible A100 GPU required')
    runtime = legacy.verify_production_runtime(plan)
    stopping = [False]
    def stop(*_):
        stopping[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop)
    entries=[Path(__file__)] + ([] if profile is None else [Path(profile.__file__)])
    sources = transitive_local_source_manifest(entries, root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    if phase == 'prepare':
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('new sealed adaptive production directory required')
        out.mkdir(parents=True, exist_ok=True)
        # Compatibility shadow is ONLY for the unchanged immutable-parent
        # helper. The new strict profile was validated above; never call the
        # closed legacy.validate_plan or dispatch its production run here.
        binding = (legacy.bind_parent(dict(plan, task='CT_bounded_followup_v1'), out)
            if profile is None else profile.bind_parent(plan,out))
        parent, _ = legacy.load_state(out/'parent.pkl')
        if profile is None and binding['parent_global_step'] != plan['expected_parent_global_step']:
            raise ValueError('latest immutable parent physical step differs')
        registration = dict(binding, plan=plan, plan_sha256=canonical_sha256(plan), sources=sources,
            initial_native_cursor=parent[2]['next_step'], started_epoch=time(),
            runtime=dict(jax=jax.__version__, x64=jax.config.x64_enabled,
                checked_stack=runtime, devices=[d.device_kind for d in jax.devices()]),
            physical_FS_contract=dict(initial_controller_bound=binding.get('inherited_control_values',{}).get('fs_radius',.00125),
                initial_selection_target=.8*binding.get('inherited_control_values',{}).get('fs_radius',.00125),
                nominal_guard_upper=binding.get('inherited_control_values',{}).get('fs_radius',.00125),
                observed_distance_is_measured_independently=True),
            step_counter='optimizer_attempts_not_accepted_updates')
        atomic_json(out/'registration.json', registration)
        controls = (AdaptiveSRState.initial(policy_from_plan(plan)) if profile is None
            else profile.initial_controls(plan,binding))
        progress = initial_progress()
        for name in ('initial.pkl', 'checkpoint.pkl', 'block_start_0.pkl'):
            save_adaptive_state(out/name, parent, controls, progress, plan=plan, registration=registration)
    registration = json.loads((out/'registration.json').read_text())
    if registration['plan'] != plan or registration['sources'] != sources:
        raise ValueError('source or plan changed across allocated processes')
    for name, digest in registration['parent_seals'].items():
        if file_sha256(out/'parent_snapshot'/name) != digest:
            raise ValueError('immutable bound parent evidence changed')
    def remaining():
        end = float(os.environ.get('SLURM_JOB_END_TIME', registration['started_epoch']+plan['wall_cap_seconds']))
        return 0. if stopping[0] else min(plan['wall_cap_seconds']-(time()-registration['started_epoch']), end-time()-240.)
    parent, initial_controls, _ = load_adaptive_state(out/'initial.pkl', plan=plan, registration=registration)
    rt, kernel = make_runtime(plan, parent, registration)
    rt['remaining'] = remaining
    allocator_plan=dict(plan,fit_records=max(plan['fit_records'],initial_controls.fit_records))
    atomic_json(out/('allocator_'+phase+'.json'), legacy.check_allocator(allocator_plan, registration['parameter_count']))
    phase_path = out/(phase+'_result.json')
    if phase_path.exists():
        return json.loads(phase_path.read_text())
    report = dict(phase=phase, status='started', attempts=0, accepted_updates=0,
        step_counter='optimizer_attempts', terminal_audit_used_to_adapt=False)
    try:
        if phase == 'prepare':
            replay = legacy.replay_parent(plan, rt, parent, out)
            atomic_json(out/'device_replay.json', replay)
            cost = reserve_costs(plan, replay['warm_H_seconds']); atomic_json(out/'cost.json', cost)
            reserve = cost['audit_reserve_seconds']+len(plan['monitor_after'])*cost['monitor_reserve_seconds']
            if remaining() < reserve+2.6*(plan['fit_records']*cost['warm_H_seconds']+600.):
                report['status'] = 'cost_gate_parent_preserved'
            else:
                controls = initial_controls
                rt['training_remaining'] = lambda: remaining()-reserve
                with c6.budget_timeout(remaining()-reserve):
                    candidate, info = transition(plan, kernel, rt, parent, controls, out/'step1', attempt=1)
                progress = consume_attempt(initial_progress(), info)
                save_adaptive_state(out/'provisional.pkl', candidate, controls, progress,
                    plan=plan, registration=registration)
                report.update(status='first_attempt_provisional', transition=info,
                    first_proposal_accepted=info['accepted'])
        elif phase == 'replay':
            prepared = json.loads((out/'prepare_result.json').read_text())
            if prepared['status'] != 'first_attempt_provisional':
                report['status'] = 'no_provisional_attempt_parent_preserved'
            else:
                expected, controls, progress = load_adaptive_state(out/'provisional.pkl', plan=plan,
                    registration=registration)
                cost = json.loads((out/'cost.json').read_text())
                reserve = cost['audit_reserve_seconds']+len(plan['monitor_after'])*cost['monitor_reserve_seconds']
                rt['training_remaining'] = lambda: remaining()-reserve
                with c6.budget_timeout(remaining()-reserve):
                    actual, info = transition(plan, kernel, rt, parent, controls, out/'step1_replay', attempt=1)
                comparison = compare_transition(expected, actual, prepared['transition'], info,
                    out/'step1', out/'step1_replay')
                progress['fit_H_records_consumed'] += info['work']['fit_H_centers']
                save_adaptive_state(out/'checkpoint.pkl', expected, controls, progress,
                    plan=plan, registration=registration)
                report.update(status='first_attempt_replayed_committed_once', replay=comparison,
                    attempts=progress['attempts'], accepted_updates=progress['accepted_updates'])
        elif phase == 'train':
            replay_report = json.loads((out/'replay_result.json').read_text())
            if replay_report['status'] != 'first_attempt_replayed_committed_once':
                report.update(freeze_endpoint(plan, out, registration, reason='no_replayed_attempt'))
            else:
                state, controls, progress = load_adaptive_state(out/'checkpoint.pkl', plan=plan,
                    registration=registration)
                cost = json.loads((out/'cost.json').read_text())
                progress['stop_reason'] = restart_boundary_reason(plan, out, controls, progress)
                while progress['attempts'] < plan['steps'] and not progress['stop_reason']:
                    attempt = progress['attempts']+1
                    future_monitors = sum(i >= attempt for i in plan['monitor_after'])
                    reserve = cost['audit_reserve_seconds']+future_monitors*cost['monitor_reserve_seconds']
                    last = progress['transitions'][-1]['wall_seconds']
                    if remaining() < reserve+1.4*last:
                        progress['stop_reason'] = 'wall_gate_terminal_and_monitor_reserves_preserved'; break
                    path = out/f'step{attempt}'
                    if path.exists():
                        progress['stop_reason'] = 'uncommitted_attempt_quarantined_no_same_stream_retry'; break
                    # Last-step direction and block-start physical states are
                    # immutable checkpoint evidence, not new fit estimates.
                    if attempt in plan['monitor_after']:
                        save_checkpoint(out/f'direction_parent_{attempt}.pkl', state[0],
                            metadata=dict(parent_params_sha256=pytree_sha256(state[0])))
                    rt['training_remaining'] = lambda reserve=reserve: remaining()-reserve
                    with c6.budget_timeout(remaining()-reserve):
                        candidate, info = transition(plan, kernel, rt, state, controls, path, attempt=attempt)
                    progress = consume_attempt(progress, info); state = candidate
                    save_adaptive_state(out/'checkpoint.pkl', state, controls, progress,
                        plan=plan, registration=registration)
                    atomic_json(out/'training_progress.json', dict(attempts=progress['attempts'],
                        accepted_updates=progress['accepted_updates'], native_next_cursor=state[2]['next_step'],
                        global_step=registration['parent_global_step']+progress['accepted_updates'],
                        controls=asdict(controls)))
                    if attempt in plan['monitor_after']:
                        monitor_path = out/f'monitor{attempt//8}'
                        if monitor_path.exists():
                            progress['stop_reason'] = 'partial_monitor_quarantined_no_seed_reuse'; break
                        block_start, _, _ = load_adaptive_state(out/f'block_start_{attempt-8}.pkl',
                            plan=plan, registration=registration)
                        direction_parent = load_checkpoint(out/f'direction_parent_{attempt}.pkl')['params']
                        reserve_after = cost['audit_reserve_seconds']+sum(i > attempt for i in
                            plan['monitor_after'])*cost['monitor_reserve_seconds']
                        resource_function=resources_for_next_block if profile is None else profile.resources_for_next_block
                        envelope = resource_function(plan, registration, progress,
                            available_seconds=remaining(), reserve_seconds=reserve_after+cost['monitor_reserve_seconds'])
                        with c6.budget_timeout(remaining()-reserve_after):
                            updated, progress, decision = monitor_block(plan, kernel, rt, block_start,
                                state, direction_parent, controls, progress, monitor_path, envelope,
                                lambda: remaining()-reserve_after)
                        controls = updated
                        save_adaptive_state(out/'checkpoint.pkl', state, controls, progress,
                            plan=plan, registration=registration)
                        save_adaptive_state(out/f'block_start_{attempt}.pkl', state, controls, progress,
                            plan=plan, registration=registration)
                        if not decision['can_train_next_block']:
                            progress['stop_reason'] = 'independent_monitor_coverage_or_reserved_resource_gate'; break
                if progress['stop_reason']:
                    save_adaptive_state(out/'checkpoint.pkl', state, controls, progress,
                        plan=plan, registration=registration)
                report.update(freeze_endpoint(plan, out, registration))
        else:
            frozen = json.loads((out/'endpoint_frozen.json').read_text())
            state, controls, progress = load_adaptive_state(out/'checkpoint.pkl', plan=plan,
                registration=registration)
            if (file_sha256(out/'checkpoint.pkl') != frozen['checkpoint_sha256'] or
                    pytree_sha256(state[0]) != frozen['parameter_sha256']):
                raise ValueError('frozen terminal physical/controller checkpoint changed')
            rt['audit_reserve_seconds'] = 0.; rt['sampling_reserve_seconds'] = 120.
            audit = c6.paired_audit(kernel, rt, parent[0], state, out/'terminal_audit',
                plan['audit_records'], plan['seeds']['audit'], plan['seeds']['audit_statistics'], remaining)
            report.update(status='complete_fixed_prefix_independent_audit',
                attempts=progress['attempts'], accepted_updates=progress['accepted_updates'],
                absolute_completed_steps=registration['parent_global_step']+progress['accepted_updates'],
                terminal_audit=audit, parameter_sha256=pytree_sha256(state[0]),
                optimizer_sha256=pytree_sha256(state[1]), rng_sha256=canonical_sha256(state[2]),
                checkpoint_sha256=file_sha256(out/'checkpoint.pkl'),
                controller_sha256=sr_control_checkpoint(policy_from_plan(plan), controls)['sha256'],
                independent_from_training_geometry_and_adaptation=True)
    except (TimeoutError, MemoryError) as error:
        report.update(status='bounded_interruption_committed_prefix_preserved', error=repr(error),
            partial_queries_quarantined=True, no_same_stream_retry=True)
        # A partial prepare/replay never commits its proposal. A partial train
        # preserves only the last atomic checkpoint, then audits that prefix.
        if phase == 'train':
            report.update(freeze_endpoint(plan, out, registration, reason=repr(error)))
    atomic_json(phase_path, report)
    return report


def main() -> None:
    """CLI intentionally has no submission, requeue or allocation side effect."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'replay', 'train', 'audit'), required=True)
    args = parser.parse_args(); result = run(args.plan, args.output, args.phase)
    print(json.dumps({k: result[k] for k in ('phase', 'status', 'attempts', 'accepted_updates') if k in result}), flush=True)


if __name__ == '__main__':
    main()
