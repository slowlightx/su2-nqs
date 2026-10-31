"""Opt-in CT12 full-native-W direction confirmation, then eight fresh updates.

The gate consumes its own RNG streams and NEVER commits its candidate.  The
pilot restores the original global37 parameters, optimizer and next native
RNG.  Its terminal data are separate from all gate and finite-geometry data.
Physics comes from the unchanged active CT12 complete-A1/full-H kernels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
from time import perf_counter, time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import save_checkpoint
from su2_nqs.vmc.finite_candidate_calibration import bilateral_direction_moments
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr
from . import ct_bounded_followup as restart
from . import triangle_active_two_step_pilot as active
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT = Path(__file__).resolve().parents[3]
TASK = 'CT12_same_parent_full_W_direction_gate_v1'
BUDGET_RETRY_PROFILE = 'runtime_budget_retry_v1'
BUDGET_RETRY_EVIDENCE = dict(zero_fit_records_consumed=True,
    previous_gate_job_id=797344,
    previous_preflight_sha256='dca17a5a2b1a0429c3a71195eaee05dc00db88b28209288541b5b8d72de374fc',
    previous_gate_result_sha256='9e817bbb1a36feb41fb68ecd3abcfe8ad07cafd988d8f00398c5a2a85315a955',
    previous_replay_sha256='34b1450004bad6b3ad2651fc491a8bc7f74548c411b3791c44957b6caa218c85',
    same_unconsumed_role_streams_retained=True)
PHASES = ('gate', 'gate_replay', 'train_prepare', 'train_replay', 'train', 'audit')
PARENT_FILES = ('checkpoint.pkl', 'registration.json', 'endpoint_frozen.json',
    'audit_result.json', 'terminal_audit/audit.json',
    'terminal_audit/untouched_products.npz')
SEED_ROLES = ('gate_stream', 'bilateral', 'confirmation',
    'confirmation_statistics', 'audit', 'audit_statistics')


def validate_plan(plan: dict, *, executing: bool = False) -> None:
    """Validate the one predeclared global37, same-support A100 protocol."""
    retry = plan.get('runtime_budget_profile')
    if retry not in (None, BUDGET_RETRY_PROFILE):
        raise ValueError('unknown opt-in runtime budget profile')
    caps = (21600, 50400) if retry is None else (36000, 64800)
    fixed = dict(task=TASK, device='A100', j2=.125, parent_global_step=37,
        model=dict(length=12, multiplicity_cap=20, maximum_q=10),
        active_real_parameters=1495080, damping=.5, learning_rate=.03,
        epsilon=.125, target_FS=.003, maximum_FS=.003, fit_records=1024,
        calibration_records=1024, calibration_split=512,
        bilateral_records=512, bilateral_groups=32,
        confirmation_records=1024, confirmation_groups=32,
        audit_records=4096, audit_groups=64, steps=8,
        gate_wall_cap_seconds=caps[0], pilot_wall_cap_seconds=caps[1],
        maximum_peak_device_bytes=70*1024**3)
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError('fixed same-parent CT12 1024/1024/512/1024 gate and eight-step pilot required')
    if any(key in plan for key in ('adaptive_sr', 'software_acceptance', 'nodelist')):
        raise ValueError('no hidden adaptive profile, B200 contract or node pinning')
    for key in ('projection_chunk', 'action_chunk', 'score_chunk', 'H_chunk'):
        if isinstance(plan.get(key), bool) or not isinstance(plan.get(key), int) or plan[key] < 1:
            raise ValueError('positive execution chunks required')
    if 1728 % plan['projection_chunk']:
        raise ValueError('projection chunk must divide the FULL1728-image group')
    if retry is not None:
        chunks = dict(projection_chunk=16, H_chunk=8, action_chunk=4, score_chunk=1)
        if any(plan[key] != value for key, value in chunks.items()):
            raise ValueError('retry requires the registered projection16/H8/action4/score1 batching')
        if plan.get('budget_retry_evidence') != BUDGET_RETRY_EVIDENCE:
            raise ValueError('zero-fit failed-allocation evidence required for same-stream budget retry')
    if set(plan.get('parent_seals', {})) != set(PARENT_FILES):
        raise ValueError('all completed parent and audit seals are required')
    for value in [*plan['parent_seals'].values(), plan.get('parent_parameter_sha256', '')]:
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError('explicit parent SHA256 required')
    seeds = plan.get('seeds', {})
    if set(seeds) != set(SEED_ROLES) or len(set(seeds.values())) != len(SEED_ROLES):
        raise ValueError('six distinct external role seeds required')
    if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds.values()):
        raise ValueError('role seeds must be uint32 integers')
    # The old validator enforces the complete actual fifteen-node q profile.
    active.checked_plan(runtime_plan(plan))
    if executing and (plan.get('execution_authorized') is not True or not plan.get('authorization_note')):
        raise PermissionError('execution requires explicit user-authorized bounded protocol')


def gate_cost_forecast(plan: dict, replay: dict) -> dict:
    """Charge the complete fixed gate at its actual batch and single-row rates.

    The retry distinguishes batched full-H queries from the single-row native
    H/JVP kernel.  Its conservative JVP factor covers a primal and tangent;
    compilation, full-W SR, finite geometry and process replay have an explicit
    separate allowance.  No forecast changes any scientific panel size.
    """
    batched = float(replay['warm_H_seconds'])
    if not np.isfinite(batched) or batched <= 0:
        raise ValueError('positive measured complete-H throughput required')
    if plan.get('runtime_budget_profile') is None:
        centers = (plan['fit_records'] + 2*plan['bilateral_records']
            + 2*plan['confirmation_records'])
        estimate = 1.5*centers*batched + 1800.
        return dict(estimated_H_plus_compilation_seconds=estimate,
            full_H_equivalent_centers=centers,
            records_never_reduced=True, runtime_budget_profile='historical_6h_14h')
    single = float(replay['single_H_seconds'])
    if not np.isfinite(single) or single <= 0 or replay.get('warm_H_batch_records') != plan['H_chunk']:
        raise ValueError('retry forecast requires actual H8 and native single-row warm timings')
    # Three H-equivalents conservatively cover each complete native H/JVP;
    # the actual two-sided derivative and every one of its512 rows stay fixed.
    centers = dict(fit_full_H=plan['fit_records'],
        bilateral_primal_full_H=plan['bilateral_records'],
        bilateral_tangent_full_H=plan['bilateral_records'],
        bilateral_JVP_work_margin=plan['bilateral_records'],
        confirmation_paired_full_H=2*plan['confirmation_records'],
        transition_product_replay_full_H=2,
        isolated_transition_product_replay_full_H=2)
    seconds = dict(fit_full_H=centers['fit_full_H']*batched,
        bilateral_full_H_JVP=3*plan['bilateral_records']*single,
        confirmation_paired_full_H=centers['confirmation_paired_full_H']*batched,
        transition_and_isolated_replay_full_H=4*single)
    overhead = dict(SR_and_finite_geometry_seconds=1800.,
        native_JVP_compilation_seconds=1200., isolated_SR_geometry_replay_seconds=1200.)
    estimate = 1.5*sum(seconds.values()) + sum(overhead.values())
    return dict(estimated_H_plus_compilation_seconds=estimate,
        runtime_budget_profile=BUDGET_RETRY_PROFILE,
        measured_batched_H_seconds_per_center=batched,
        measured_single_H_seconds_per_center=single,
        measured_batch_records=plan['H_chunk'], H_work_safety_factor=1.5,
        registered_record_counts=dict(fit=plan['fit_records'],
            calibration=plan['calibration_records'], bilateral=plan['bilateral_records'],
            paired_confirmation=plan['confirmation_records']),
        full_H_equivalent_centers=sum(centers.values()),
        charged_H_equivalent_centers=centers, estimated_query_seconds=seconds,
        explicit_nonquery_allowance_seconds=overhead,
        already_spent_parent_replay_H_centers=replay['full_H_centers'],
        already_spent_parent_replay_not_charged_twice=True,
        records_never_reduced=True, one_fit_direction_only=True)


def runtime_plan(plan: dict) -> dict:
    """Reuse physics only, without enabling the sealed historical driver."""
    return dict(plan, task='CT12_fixed_active_full_A1_IID_SR_two_step_v1',
        steps=2, wall_cap_seconds=14400, automatic_production=False,
        initial_endpoint='committed_global37_same_active_support',
        seeds=dict(training=1, audit=2, statistics=3))


def make_runtime(plan: dict, params: dict, *, damping: float | None = None,
                 radius: float | None = None) -> dict:
    """Rebuild the actual SR callback with explicit controls, not metadata only."""
    rt = active.runtime(runtime_plan(plan), params)
    actual_damping = plan['damping'] if damping is None else damping
    actual_radius = plan['target_FS'] if radius is None else radius
    if not np.isfinite([actual_damping, actual_radius]).all() or actual_damping <= 0 or not 0 < actual_radius <= plan['maximum_FS']:
        raise ValueError('finite positive damping and capped physical radius required')
    rt['update'] = make_importance_amplitude_sr(
        lambda p, x: rt['amplitude'](rt['prepare'](p), x[None])[0],
        params, damping=float(actual_damping), score_chunk=plan['score_chunk'],
        parameter_chunk_size=8192, sr_tangent_fs_radius=float(actual_radius),
        prepare=rt['prepare'], prepared_amplitude=active.make_feature_lut_projected_amplitude(
            rt['cfg'], rt['images'], chunk=plan['projection_chunk'], active_top_widths=rt['profile']))
    rt['actual_controls'] = dict(damping=float(actual_damping), target_FS=float(actual_radius))
    return rt


def separate_role_streams(plan: dict, native_rng: dict, parent_seeds: set[int]) -> dict:
    """Enumerate disjoint gate and future native streams without advancing either."""
    gate_rng = np.random.default_rng(plan['seeds']['gate_stream']).bit_generator.state
    children, _ = active._seeds(gate_rng)
    future = deepcopy(native_rng); native = []
    for _ in range(plan['steps']):
        roles, future = active._seeds(future); native.extend(roles)
    external = list(plan['seeds'].values()) + children
    if (len(set(external)) != len(external) or len(set(native)) != len(native)
            or set(external) & (set(native) | parent_seeds)):
        raise ValueError('gate, pilot, terminal or parent role streams collide')
    return dict(gate_rng=deepcopy(gate_rng), gate_transition_seeds=children,
        external_role_seeds=deepcopy(plan['seeds']), future_native_role_seeds=native,
        native_rng_sha256=canonical_sha256(native_rng),
        native_rng_advanced_by_gate=False, pilot_original_next_rng_preserved=True)


def bind_parent(plan: dict, out: Path) -> dict:
    """Copy and verify the exact compact global37 checkpoint, not nominal widths."""
    source = Path(plan['parent_directory'])
    for name, digest in plan['parent_seals'].items():
        if file_sha256(source/name) != digest:
            raise ValueError('frozen parent changed: '+name)
    state, meta = restart.load_state(source/'checkpoint.pkl')
    reg = json.loads((source/'registration.json').read_text())
    frozen = json.loads((source/'endpoint_frozen.json').read_text())
    result = json.loads((source/'audit_result.json').read_text())
    audit = json.loads((source/'terminal_audit/audit.json').read_text())
    expected = plan['parent_parameter_sha256']
    if (reg['plan'].get('task') != 'CT12_production32_v1'
            or reg['plan']['active_profile'] != plan['active_profile']
            or reg['plan']['model'] != plan['model'] or reg['plan']['j2'] != plan['j2']
            or meta['absolute_completed_steps'] != 37 or int(np.asarray(state[1][0])) != 37
            or pytree_sha256(state[0]) != expected or frozen['parameter_sha256'] != expected
            or frozen['absolute_step'] != 37 or result['parameter_sha256'] != expected
            or result['status'] != 'complete_fixed_prefix_independent_audit'
            or audit != result['terminal_audit'] or audit['endpoint_params_sha256'] != expected
            or not audit['independent_from_training_and_geometry']
            or frozen['checkpoint_sha256'] != plan['parent_seals']['checkpoint.pkl']
            or result['checkpoint_sha256'] != plan['parent_seals']['checkpoint.pkl']
            or sum(np.size(x) for x in jax.tree.leaves(state[0])) != plan['active_real_parameters']):
        raise ValueError('completed compact global37 parameter/moment/RNG/profile/audit identity required')
    if not restart.safe_to_continue(audit):
        raise ValueError('resolved parent audit harm cannot be bypassed')
    historical, history = historical_role_seeds(source)
    roles = separate_role_streams(plan, state[2], historical)
    snapshot = out/'parent_snapshot'; snapshot.mkdir()
    for name in PARENT_FILES:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    restart.save_state(out/'parent.pkl', state, 0, 37)
    return dict(parent_parameter_sha256=expected,
        parent_optimizer_sha256=pytree_sha256(state[1]), parent_rng_sha256=canonical_sha256(state[2]),
        parent_seals=deepcopy(plan['parent_seals']), role_streams=roles,
        observed_parent_role_history=history,
        parameter_count=plan['active_real_parameters'],
        gate_candidate_never_committed_as_production_step=True,
        controls_fixed_after_gate_not_complete_adaptive_schedule=True)


def historical_role_seeds(source: Path) -> tuple[set[int], dict]:
    """Read available immutable ancestors and their recorded consumed role seeds.

    No old checkpoint is loaded or changed. A missing archived ancestor is
    reported explicitly rather than silently certifying its unavailable RNG
    history. The current and all readable ancestor registrations and consumed
    transitions are bound by individual file hashes in the new registration.
    """
    used = set(); evidence = {}; missing = []; visited = set()
    current = source
    for _ in range(32):
        resolved = str(current.resolve())
        if resolved in visited:
            raise ValueError('cyclic parent history')
        visited.add(resolved)
        registration = current/'registration.json'
        if not registration.exists():
            missing.append(str(registration)); break
        payload = json.loads(registration.read_text()); parent_plan = payload.get('plan', {})
        seeds = parent_plan.get('seeds', {})
        used.update(int(x) for x in seeds.values() if isinstance(x, int) and not isinstance(x, bool))
        evidence[str(registration)] = file_sha256(registration)
        for path in sorted(current.glob('step*/transition.json')):
            record = json.loads(path.read_text()); actual = record.get('seeds', [])
            actual = actual.values() if isinstance(actual, dict) else actual
            used.update(int(x) for x in actual if isinstance(x, int) and not isinstance(x, bool))
            evidence[str(path)] = file_sha256(path)
        ancestor = parent_plan.get('parent_directory')
        if not ancestor:
            break
        current = Path(ancestor)
    else:
        raise ValueError('parent history unexpectedly exceeds32 allocations')
    return used, dict(observed_seed_count=len(used), file_sha256=evidence,
        unavailable_ancestor_registration_paths=missing,
        complete_available_history_checked=True,
        unavailable_history_not_claimed_certified=bool(missing))


def check_peak(plan: dict) -> int:
    """Observe actual live device peak; a Jacobian byte estimate is not a peak."""
    peak = max(int((device.memory_stats() or {}).get('peak_bytes_in_use', 0))
        for device in jax.local_devices())
    if peak > plan['maximum_peak_device_bytes']:
        raise MemoryError('actual device peak exceeds registered70GiB ceiling')
    return peak


def coverage_checks(moments: dict, count: int) -> dict:
    """Frozen finite-panel concentration checks; not a certificate of unseen tails."""
    diagnostics = {}
    for name in ('norm_concentration', 'H2_concentration', 'residual_concentration'):
        item = moments[name]
        ess = np.asarray(item['effective_contributions'], float)
        single = np.asarray(item['maximum_sample_fraction'], float)
        group = np.asarray(item['maximum_group_fraction'], float)
        finite = np.isfinite(np.r_[ess, single, group]).all()
        diagnostics[name] = bool(finite and np.min(ess) >= max(16, .02*count)
            and np.max(single) <= .2 and np.max(group) <= .4)
    return dict(passed=all(diagnostics.values()), checks=diagnostics,
        minimum_effective_contributions=max(16, .02*count),
        maximum_single_fraction=.2, maximum_group_fraction=.4,
        finite_panel_only_not_population_tail_certificate=True)


def gate_decision(bilateral: dict, confirmation: dict, *, geometry_passed: bool,
                  replay_passed: bool = False) -> dict:
    """Use fresh slope/finite energy and variance ONCE, never reselect a candidate."""
    slope = float(bilateral['values'][2])
    slope_se = float(bilateral['independent_group_jackknife_SE'][2])
    delta = np.asarray(confirmation['moments']['candidate_minus_control'], float)
    se = np.maximum(np.asarray(confirmation['moments']['paired_iid_delta_SE'], float),
        np.asarray(confirmation['moments']['paired_independent_block_jackknife_SE'], float))
    if not np.isfinite(np.r_[slope, slope_se, delta, se]).all() or slope_se < 0 or np.any(se < 0):
        raise ValueError('finite independent errors required')
    checks = dict(geometry=bool(geometry_passed),
        bilateral_descent=slope+2*slope_se < 0,
        finite_energy_descent=bool(delta[0]+2*se[0] < 0),
        variance_not_resolved_worse=bool(delta[2]-2*se[2] <= 0),
        coverage=bool(confirmation['coverage']['passed']),
        cross_process_replay=bool(replay_passed))
    return dict(passed=all(checks.values()), checks=checks,
        bilateral_slope_upper2SE=slope+2*slope_se,
        finite_delta_energy_upper2SE=float(delta[0]+2*se[0]),
        finite_delta_variance_lower2SE=float(delta[2]-2*se[2]),
        confirmation_consumed_as_gate_not_terminal_precision=True,
        no_candidate_reselection_using_confirmation=True)


def canonical(plan: dict, rt: dict, params: dict) -> tuple[dict, dict]:
    """Sampler-only canonical bank; native optimizer coordinates stay untouched."""
    return active.canonical_active_sampling_bank(params, rt['cfg'], rt['profile'])


def paired_panel(plan: dict, rt: dict, first: tuple, second: tuple, path: Path,
                 *, count: int, groups: int, seed: int, statistics: int) -> dict:
    """Fresh full-support endpoint-mixture panel, all1728 images and864 H bonds."""
    path.mkdir()
    banks = [canonical(plan, rt, s[0])[0] for s in (first, second)]
    rows, draw = active.exact_draw(banks, rt, count, seed, bridge=True)
    save_checkpoint(path/'draw.pkl', draw, metadata=dict(role='fixed_fresh_IID_full_H', seed=seed))
    checked_runtime = dict(rt); peaks = []
    def checked_query(bank, bits):
        value = jax.block_until_ready(rt['H'](bank, bits))
        peaks.append(check_peak(plan))
        return value
    checked_runtime['H'] = checked_query
    values, warm = active.products(checked_runtime, banks, rows, path, 'paired', plan['H_chunk'])
    q = full_support_sz0_density(values[:, :, 2].real.mean(1), values[:, :, 3].real.mean(1),
        n_sites=144, epsilon=plan['epsilon'])
    moments = paired_iid_moments(values[:, :, 0], values[:, :, 1], q,
        np.repeat(np.arange(groups), count//groups), n_sites=144, seed=statistics)
    np.savez_compressed(path/'products.npz', rows=rows, products=values, proposal=q)
    report = dict(moments=moments, coverage=coverage_checks(moments, count),
        records=count, groups=groups, seed=seed, statistics_seed=statistics,
        full_H_centers=2*count, warm_H_seconds=warm,
        observed_peak_device_bytes=max(peaks, default=0),
        complete_A1_images=1728, physical_H_bonds=864,
        endpoint_parameter_sha256=[pytree_sha256(s[0]) for s in (first, second)])
    atomic_json(path/'audit.json', report)
    return report


def bilateral_panel(plan: dict, rt: dict, parent: tuple, candidate: tuple,
                    path: Path) -> dict:
    """Differentiate native Hpsi itself; do not omit the second coherent term."""
    path.mkdir()
    bank, normalization = canonical(plan, rt, parent[0])
    rows, draw = active.exact_draw([bank], rt, plan['bilateral_records'],
        plan['seeds']['bilateral'], bridge=True)
    save_checkpoint(path/'draw.pkl', draw, metadata=dict(role='independent_bilateral_H_JVP'))
    delta = jax.tree.map(lambda new, old: new-old, candidate[0], parent[0])
    # Native raw amplitudes, Hpsi, and their JVP have precisely the same scale.
    # The known positive sampling density is normalized by one frozen Zraw.
    action = jax.jit(lambda p, d, x: native_h_jvp(rt, p, d, x))
    primals = []; tangents = []; seconds = []; peak = 0
    for i, row in enumerate(rows):
        active.stage_budget(rt, 120., 'next complete native H/JVP center')
        tick = perf_counter()
        value, derivative = jax.block_until_ready(action(parent[0], delta, row[None]))
        peak = max(peak, check_peak(plan))
        primals.append(np.stack([np.asarray(x) for x in value], axis=1)[0])
        tangents.append(np.stack([np.asarray(x) for x in derivative], axis=1)[0])
        seconds.append(perf_counter()-tick)
        if i % 16 == 0:
            atomic_json(path/'progress.json', dict(completed=i+1, records=len(rows), batch_seconds=seconds,
                observed_peak_device_bytes=peak))
    value = np.asarray(primals); derivative = np.asarray(tangents)
    q = full_support_sz0_density(value[:, 2].real/normalization['original_raw_norm'],
        value[:, 3].real/normalization['original_raw_norm'], n_sites=144, epsilon=plan['epsilon'])
    report = bilateral_direction_moments(value[:, 0], value[:, 1], derivative[:, 0], derivative[:, 1],
        q, groups=plan['bilateral_groups'], n_sites=144)
    np.savez_compressed(path/'products.npz', rows=rows, products=value, derivatives=derivative,
        proposal=q, seconds=seconds)
    report.update(seed=plan['seeds']['bilateral'],
        primal_H_centers=len(rows), tangent_H_centers=len(rows),
        observed_peak_device_bytes=peak,
        same_native_scale_for_psi_Hpsi_dpsi_Hdpsi=True,
        derivative_of_complete_H_action_not_parent_eigenstate_assumption=True)
    atomic_json(path/'bilateral.json', report)
    return report


def replay_saved_parent(plan: dict, rt: dict, parent: tuple, out: Path) -> dict:
    """At least two saved full-H rows must survive B200-to-A100 replay."""
    retry = plan.get('runtime_budget_profile') == BUDGET_RETRY_PROFILE
    count = plan['H_chunk'] if retry else 2
    with np.load(out/'parent_snapshot/terminal_audit/untouched_products.npz') as f:
        rows = f['rows'][:count]; expected = f['products'][:count, 1, :]
    if len(rows) != count:
        raise ValueError('saved parent panel cannot provide the registered replay batch')
    bank, normalization = canonical(plan, rt, parent[0])
    actual = np.stack([np.asarray(x) for x in jax.block_until_ready(rt['H'](bank, rows))], axis=1)
    errors = np.max(abs(actual-expected), axis=0)/np.maximum(np.max(abs(expected), axis=0), 1e-300)
    if not np.isfinite(errors).all() or errors.max() > 2e-11:
        raise ArithmeticError('saved full coherent A1/H/guide replay failed')
    tick = perf_counter(); jax.block_until_ready(rt['H'](bank, rows)); warm = (perf_counter()-tick)/count
    report = dict(passed=True, component_relative_errors=errors.tolist(), tolerance=2e-11,
        full_H_centers=2*count, warm_H_seconds=warm, complete_A1_images=1728, physical_H_bonds=864)
    if retry:
        # Batched H8 timing is not a throughput estimate for the H/JVP path,
        # which intentionally retains its original single-center query.
        single_rows = rows[:1]
        single = np.stack([np.asarray(x) for x in jax.block_until_ready(rt['H'](bank, single_rows))], axis=1)
        single_errors = np.max(abs(single-expected[:1]), axis=0)/np.maximum(np.max(abs(expected[:1]), axis=0), 1e-300)
        tick = perf_counter(); jax.block_until_ready(rt['H'](bank, single_rows))
        canonical_seconds = perf_counter()-tick
        raw_norm = float(normalization['original_raw_norm'])
        if not np.isfinite(raw_norm) or raw_norm <= 0:
            raise ArithmeticError('positive finite native normalization required')
        expected_native = single*np.asarray([np.sqrt(raw_norm), np.sqrt(raw_norm), raw_norm, raw_norm])
        native = np.stack([np.asarray(x) for x in jax.block_until_ready(rt['H'](parent[0], single_rows))], axis=1)
        native_errors = np.max(abs(native-expected_native), axis=0)/np.maximum(np.max(abs(expected_native), axis=0), 1e-300)
        tick = perf_counter(); jax.block_until_ready(rt['H'](parent[0], single_rows))
        native_seconds = perf_counter()-tick
        if (not np.isfinite(np.r_[single_errors, native_errors]).all()
                or max(single_errors.max(), native_errors.max()) > 2e-11):
            raise ArithmeticError('single-row native/canonical full-H scale replay failed')
        report.update(full_H_centers=2*count+4, warm_H_batch_records=count,
            single_H_seconds=max(canonical_seconds, native_seconds),
            canonical_single_H_seconds=canonical_seconds, native_single_H_seconds=native_seconds,
            single_canonical_component_relative_errors=single_errors.tolist(),
            single_native_component_relative_errors=native_errors.tolist(),
            single_row_from_same_saved_batch=True,
            native_product_scale_factors=[np.sqrt(raw_norm), np.sqrt(raw_norm), raw_norm, raw_norm],
            observed_peak_device_bytes=check_peak(plan),
            batching_changes_execution_only_not_H_images_or_samples=True)
    return report


def require_gate(out: Path, registration: dict) -> dict:
    """A Slurm dependency or successful exit alone never authorizes training."""
    gate = json.loads((out/'gate_certificate.json').read_text())
    if (gate.get('passed') is not True or gate.get('plan_sha256') != registration['plan_sha256']
            or gate.get('parent_parameter_sha256') != registration['parent_parameter_sha256']
            or gate.get('parent_rng_sha256') != registration['parent_rng_sha256']
            or gate.get('gate_candidate_is_production_step') is not False
            or gate.get('controls') != dict(damping=registration['plan']['damping'], target_FS=registration['plan']['target_FS'])
            or not all(gate.get('checks', {}).values()) or len(gate.get('checks', {})) != 6
            or gate.get('gate_result_sha256') != file_sha256(out/'gate_result.json')):
        raise PermissionError('same-parent gate certificate did not pass; no pilot update permitted')
    started = out/'pilot_start.json'
    if started.exists() and json.loads(started.read_text())['gate_certificate_sha256'] != file_sha256(out/'gate_certificate.json'):
        raise ValueError('gate certificate changed after pilot began')
    return gate


def transition_once(plan: dict, rt: dict, state: tuple, path: Path, *, replay: bool = False) -> tuple[tuple, dict]:
    """Consume every used native role even when a finite-geometry veto fires.

    The sealed historical transition returns the unchanged RNG on rejection.
    This independent wrapper records its three already-used roles and advances
    only the RNG in that case; it never turns the rejected update into a step.
    """
    if not replay and path.exists() and any(path.iterdir()):
        raise FileExistsError('partial or completed transition cannot retry consumed scientific roles')
    result, info = active.transition(plan, rt, state, path, replay=replay)
    roles, following = active._seeds(state[2])
    if info['seeds'] != roles:
        raise ArithmeticError('transition role streams differ from original next RNG')
    if not info['accepted']:
        result = (state[0], state[1], following)
        info = dict(info, rejected_update_roles_consumed=True,
            next_rng_sha256=canonical_sha256(following), retry_of_consumed_roles_forbidden=True)
        if not replay:
            # The legacy replay fixture retains its own original semantics;
            # save the corrected wrapper state separately and never overwrite
            # its provisional fixture or transition report.
            restart.save_state(path/'rejected_consumed_state.pkl', result, 0, int(np.asarray(state[1][0])))
            atomic_json(path/'rejection_role_consumption.json', info)
    return result, info


def native_h_jvp(rt: dict, params: dict, direction: dict, rows: jax.Array):
    """Both coherent H derivative terms, in the unchanged native amplitude scale."""
    return jax.jvp(lambda values: rt['H'](values, rows), (params,), (direction,))


def _sources() -> dict:
    return transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))


def run(plan_path: Path, out: Path, phase: str, *, runtime_factory=None,
        require_gpu: bool = True) -> dict:
    """Run one isolated process phase; all incomplete work fails closed."""
    plan = json.loads(plan_path.read_text()); validate_plan(plan, executing=True)
    if phase not in PHASES:
        raise ValueError('unknown isolated phase')
    if require_gpu and (jax.default_backend() != 'gpu' or not all('A100' in d.device_kind for d in jax.devices())):
        raise RuntimeError('Kim A100 allocation required')
    sources = _sources(); registration_path = out/'registration.json'
    if phase == 'gate':
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('fresh sealed gate directory required')
        out.mkdir(parents=True, exist_ok=True)
        registration = dict(plan=plan, plan_sha256=canonical_sha256(plan), sources=sources,
            started_epoch=time(), **bind_parent(plan, out))
        atomic_json(registration_path, registration)
    registration = json.loads(registration_path.read_text())
    if registration['plan'] != plan or registration['sources'] != sources:
        raise ValueError('plan or numerical source changed after registration')
    for name, digest in registration['parent_seals'].items():
        if file_sha256(out/'parent_snapshot'/name) != digest:
            raise ValueError('bound parent snapshot changed')
    report_path = out/(phase+'_result.json')
    if report_path.exists():
        return json.loads(report_path.read_text())
    parent, _ = restart.load_state(out/'parent.pkl')
    if (pytree_sha256(parent[0]) != registration['parent_parameter_sha256']
            or pytree_sha256(parent[1]) != registration['parent_optimizer_sha256']
            or canonical_sha256(parent[2]) != registration['parent_rng_sha256']):
        raise ValueError('original parent checkpoint identity changed')
    stopping = [False]
    def stop(*_): stopping[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM): signal.signal(sig, stop)
    pilot = phase.startswith('train') or phase == 'audit'
    if pilot:
        try:
            gate = require_gate(out, registration)
        except (FileNotFoundError, PermissionError) as error:
            report = dict(phase=phase, status='gate_not_passed_no_training',
                committed_updates=0, error=str(error), no_update_callback_invoked=True)
            atomic_json(report_path, report)
            return report
        if phase == 'train_prepare':
            atomic_json(out/'pilot_start.json', dict(started_epoch=time(),
                original_parent_parameter_sha256=pytree_sha256(parent[0]),
                original_parent_optimizer_sha256=pytree_sha256(parent[1]),
                original_parent_rng_sha256=canonical_sha256(parent[2]),
                gate_certificate_sha256=file_sha256(out/'gate_certificate.json')))
        epoch = json.loads((out/'pilot_start.json').read_text())['started_epoch']
        controls = gate['controls']; cap = plan['pilot_wall_cap_seconds']
    else:
        epoch = registration['started_epoch']; controls = dict(damping=plan['damping'], target_FS=plan['target_FS'])
        cap = plan['gate_wall_cap_seconds']
    def remaining():
        scheduler_end = float(os.environ.get('SLURM_JOB_END_TIME', epoch+cap+120))
        return 0. if stopping[0] else min(cap-(time()-epoch), scheduler_end-time()-120.)
    rt = ((runtime_factory or make_runtime)(plan, parent[0],
        damping=controls['damping'], radius=controls['target_FS']))
    rt['remaining'] = remaining
    kernel = dict(plan, damping=controls['damping'], target_FS=controls['target_FS'])
    # This checks one Jacobian +3GiB, not a false claim of measured peak safety.
    required = plan['fit_records']*registration['parameter_count']*8+3*1024**3
    if required > plan['maximum_peak_device_bytes']:
        raise MemoryError('registered full-W Jacobian exceeds allocation ceiling')
    report = dict(phase=phase, status='started', committed_updates=0)
    try:
        if phase == 'gate':
            replay = replay_saved_parent(plan, rt, parent, out)
            atomic_json(out/'device_replay.json', replay)
            rt['warm_H_seconds'] = replay['warm_H_seconds']
            cost = gate_cost_forecast(plan, replay)
            forecast = cost['estimated_H_plus_compilation_seconds']
            atomic_json(out/'gate_cost_preflight.json', dict(cost, remaining_seconds=remaining()))
            active.stage_budget(rt, forecast, 'complete registered direction confirmation')
            gate_state = (parent[0], parent[1], registration['role_streams']['gate_rng'])
            candidate, info = transition_once(kernel, rt, gate_state, out/'gate_proposal')
            if info['seeds'] != registration['role_streams']['gate_transition_seeds']:
                raise ArithmeticError('registered external gate RNG roles changed')
            report.update(proposal=info, candidate_parameter_sha256=pytree_sha256(candidate[0]),
                gate_rng_separate_from_native=True, candidate_never_committed=True)
            if not info['accepted']:
                report.update(status='geometry_rejected_no_confirmation', preliminary_passed=False)
            else:
                restart.save_state(out/'gate_candidate.pkl', candidate, 0, 37)
                atomic_json(out/'candidate_frozen.json', dict(
                    parameter_sha256=pytree_sha256(candidate[0]),
                    checkpoint_sha256=file_sha256(out/'gate_candidate.pkl'),
                    confirmation_not_yet_evaluated=True, no_reselection_after_confirmation=True))
                bilateral = bilateral_panel(plan, rt, parent, candidate, out/'bilateral')
                confirmation = paired_panel(plan, rt, parent, candidate, out/'confirmation',
                    count=plan['confirmation_records'], groups=plan['confirmation_groups'],
                    seed=plan['seeds']['confirmation'], statistics=plan['seeds']['confirmation_statistics'])
                decision = gate_decision(bilateral, confirmation, geometry_passed=True)
                report.update(status='direction_confirmation_frozen_requires_replay',
                    bilateral=bilateral, confirmation=confirmation,
                    decision_before_replay=decision,
                    preliminary_passed=all(v for k, v in decision['checks'].items() if k != 'cross_process_replay'))
        elif phase == 'gate_replay':
            original = json.loads((out/'gate_result.json').read_text())
            accepted = original.get('proposal', {}).get('accepted', False)
            certificate = dict(passed=False, parent_parameter_sha256=registration['parent_parameter_sha256'],
                parent_rng_sha256=registration['parent_rng_sha256'],
                plan_sha256=registration['plan_sha256'], gate_candidate_is_production_step=False,
                controls=controls, gate_result_sha256=file_sha256(out/'gate_result.json'))
            if accepted and original.get('preliminary_passed'):
                gate_state = (parent[0], parent[1], registration['role_streams']['gate_rng'])
                replayed, info = transition_once(kernel, rt, gate_state, out/'gate_proposal', replay=True)
                expected, _ = restart.load_state(out/'gate_candidate.pkl')
                frozen = json.loads((out/'candidate_frozen.json').read_text())
                if (file_sha256(out/'gate_candidate.pkl') != frozen['checkpoint_sha256']
                        or pytree_sha256(expected[0]) != frozen['parameter_sha256']
                        or frozen['parameter_sha256'] != original['candidate_parameter_sha256']):
                    raise ValueError('frozen gate candidate changed before replay')
                if (pytree_sha256(replayed[0]) != pytree_sha256(expected[0])
                        or pytree_sha256(replayed[1]) != pytree_sha256(expected[1])
                        or canonical_sha256(replayed[2]) != canonical_sha256(expected[2])):
                    # transition already uses the numerical2e-11 comparison;
                    # exact hashes additionally bind the saved candidate read
                    # in this process, never a newly selected noisy endpoint.
                    differences = [np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(np.asarray(b)),1e-300)
                        for a,b in zip(jax.tree.leaves(replayed[:2]),jax.tree.leaves(expected[:2]),strict=True)]
                    if max(differences,default=0)>2e-11 or canonical_sha256(replayed[2]) != canonical_sha256(expected[2]):
                        raise ArithmeticError('gate parameter/optimizer/RNG replay failed')
                decision = gate_decision(original['bilateral'], original['confirmation'],
                    geometry_passed=True, replay_passed=True)
                certificate.update(decision, replay=info,
                    candidate_parameter_sha256=pytree_sha256(expected[0]),
                    gate_candidate_checkpoint_sha256=file_sha256(out/'gate_candidate.pkl'))
            certificate['failure_keeps_original_parent_unchanged'] = True
            atomic_json(out/'gate_certificate.json', certificate)
            report.update(status='gate_passed' if certificate['passed'] else 'gate_failed_no_training', certificate=certificate)
        elif phase == 'train_prepare':
            replay = replay_saved_parent(plan, rt, parent, out)
            atomic_json(out/'pilot_device_replay.json', replay)
            rt['warm_H_seconds'] = replay['warm_H_seconds']
            rt['audit_reserve_seconds'] = 1.5*2*plan['audit_records']*rt['warm_H_seconds']+600
            active.stage_budget(rt, rt['audit_reserve_seconds']+
                1.5*plan['fit_records']*rt['warm_H_seconds']+1200., 'first full update plus terminal audit')
            candidate, info = transition_once(kernel, rt, parent, out/'step1')
            restart.save_state(out/'pilot_provisional.pkl', candidate, int(info['accepted']), 37+int(info['accepted']))
            if not info['accepted']:
                restart.save_state(out/'checkpoint.pkl', candidate, 0, 37)
                atomic_json(out/'endpoint_frozen.json', dict(status='zero_updates_geometry_veto',
                    completed_updates=0, absolute_step=37, parameter_sha256=pytree_sha256(candidate[0]),
                    checkpoint_sha256=file_sha256(out/'checkpoint.pkl'), consumed_role_rng_preserved=True))
            report.update(status='first_update_provisional' if info['accepted'] else 'first_geometry_rejected', transition=info)
        elif phase == 'train_replay':
            first = json.loads((out/'train_prepare_result.json').read_text())
            if first['status'] != 'first_update_provisional':
                report.update(status='no_provisional_update_no_training')
            else:
                expected, _ = restart.load_state(out/'pilot_provisional.pkl')
                actual, info = transition_once(kernel, rt, parent, out/'step1', replay=True)
                if canonical_sha256(actual[2]) != canonical_sha256(expected[2]):
                    raise ArithmeticError('pilot next RNG replay mismatch')
                restart.save_state(out/'checkpoint.pkl', expected, 1, 38)
                report.update(status='replay_passed_committed_once', committed_updates=1, replay=info)
        elif phase == 'train':
            passed = json.loads((out/'train_replay_result.json').read_text())
            if passed['status'] != 'replay_passed_committed_once':
                report.update(status='no_replayed_update_no_training')
            else:
                state, meta = restart.load_state(out/'checkpoint.pkl'); done = meta['completed_updates']
                first = json.loads((out/'train_prepare_result.json').read_text())['transition']
                warm = first['warm_H_seconds']; last = first['wall_seconds']
                rt.update(warm_H_seconds=warm, audit_reserve_seconds=1.5*2*plan['audit_records']*warm+600)
                for i in range(done+1, plan['steps']+1):
                    if remaining() < rt['audit_reserve_seconds']+1.5*last:
                        report['stop_reason'] = 'terminal_audit_reserve'; break
                    candidate, info = transition_once(kernel, rt, state, out/f'step{i}')
                    last = info['wall_seconds']
                    if not info['accepted']:
                        state = candidate
                        restart.save_state(out/'checkpoint.pkl', state, done, 37+done)
                        report['stop_reason'] = 'independent_finite_geometry_veto'; break
                    state = candidate; done = i
                    restart.save_state(out/'checkpoint.pkl', state, done, 37+done)
                    atomic_json(out/'training_progress.json', dict(completed_updates=done, absolute_step=37+done,
                        actual_controls=controls, all_active_W_optimized=True))
                frozen = dict(status='fixed_prefix_frozen', completed_updates=done,
                    absolute_step=37+done, requested_updates=8, fixed_target_completed=done==8,
                    parameter_sha256=pytree_sha256(state[0]), checkpoint_sha256=file_sha256(out/'checkpoint.pkl'))
                atomic_json(out/'endpoint_frozen.json', frozen)
                report.update(frozen, committed_updates=done,
                    adaptive_monitor_not_enabled=True, gate_verified_controls_fixed_for_eight_steps=True)
        else:
            if not (out/'endpoint_frozen.json').exists() or not (out/'checkpoint.pkl').exists():
                report.update(status='no_committed_endpoint_no_audit', committed_updates=0,
                    no_training_mean_or_zero_update_promotion=True)
                atomic_json(report_path, report)
                return report
            frozen = json.loads((out/'endpoint_frozen.json').read_text())
            state, meta = restart.load_state(out/'checkpoint.pkl')
            if pytree_sha256(state[0]) != frozen['parameter_sha256'] or file_sha256(out/'checkpoint.pkl') != frozen['checkpoint_sha256']:
                raise ValueError('frozen pilot endpoint changed')
            if meta['completed_updates'] < 1:
                report.update(status='zero_updates_no_terminal_precision', committed_updates=0,
                    no_training_mean_or_zero_update_promotion=True)
                atomic_json(report_path, report)
                return report
            final = paired_panel(plan, rt, parent, state, out/'terminal_audit',
                count=plan['audit_records'], groups=plan['audit_groups'], seed=plan['seeds']['audit'],
                statistics=plan['seeds']['audit_statistics'])
            report.update(status='complete_fixed_prefix_independent_audit',
                committed_updates=meta['completed_updates'], parameter_sha256=pytree_sha256(state[0]),
                checkpoint_sha256=file_sha256(out/'checkpoint.pkl'), terminal_audit=final,
                independent_from_gate_training_and_geometry=True,
                terminal_audit_not_used_to_select_endpoint=True)
    except (TimeoutError, MemoryError) as error:
        report.update(status='bounded_interruption_parent_or_committed_prefix_preserved', error=repr(error))
        if (out/'checkpoint.pkl').exists() and not (out/'endpoint_frozen.json').exists():
            committed, meta = restart.load_state(out/'checkpoint.pkl')
            done = meta['completed_updates']
            if done > 0:
                # Only the last atomic checkpoint is frozen. A partial new
                # candidate is never promoted when a stage hits its budget.
                frozen = dict(status='bounded_committed_prefix_frozen', completed_updates=done,
                    absolute_step=meta['absolute_completed_steps'], requested_updates=8,
                    fixed_target_completed=done==8, parameter_sha256=pytree_sha256(committed[0]),
                    checkpoint_sha256=file_sha256(out/'checkpoint.pkl'), interrupted_phase=phase)
                atomic_json(out/'endpoint_frozen.json', frozen)
                report.update(committed_updates=done, committed_prefix_frozen_for_independent_audit=True)
    atomic_json(report_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output, args.phase)
    print(json.dumps({key:result[key] for key in ('status','phase','committed_updates') if key in result}), flush=True)
    if result['status'].startswith('bounded_interruption'):
        raise SystemExit(2)
