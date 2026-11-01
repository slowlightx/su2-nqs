"""Opt-in source-protected S1K all-W training with honest cold/warm costs.

Every real native W coordinate participates in complex128 amplitude AD. No
imaginary amplitude/Jacobian is discarded and no new head/scale is trained.
The fixed spin source, singlet carrier, CT reference and affine scale retain
their original parent identity. S0X is deliberately not launched by this task.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter, time
from typing import Any, Callable

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_hnf import hnf_point_group_permutations, hnf_site_permutation
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_reciprocal_little_group
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.complex_importance_amplitude_sr import make_complex_importance_amplitude_sr
from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS, select_finite_grid
from projects.clebsch_tree_nqs.explorations import ct_source_aware_direction_20261030 as old

ROOT = Path(__file__).resolve().parents[3]
TASK = 'CT6_S1K_source_protected_complex_amplitude_all_W_128x8_512_v1'
CONFIG = ROOT/'configs/remote/ct_s1k_full_network_20261101/plan.json'


def validate_plan(p: dict, *, executing: bool = False) -> None:
    """Separate one-allocation permission; no weakening of the old protocol."""
    fixed = dict(task=TASK, target_q=2, target_m2=0, momentum=[[1, 3], [1, 3]],
        little_character='all_plus_one', j2=.125,
        model=dict(length=6, multiplicity_cap=28, maximum_q=10), arity='binary_truncated', binary_rank=8,
        fit_records=128, FS_records=128, FS_split=64, bilateral_records=128, bilateral_groups=16,
        terminal_records=512, terminal_groups=32, fixed_attempts=8, maximum_updates=8,
        preflight_records=128, warm_repetitions=2, source_tilt_floor=.1,
        source_factor_squared_bound=3., base_uniform_fraction=.125,
        damping=.05, learning_rate=1., tangent_FS=.01, target_FS=.01, maximum_FS=.015,
        maximum_source_FS=.04, maximum_source_FS_upper=.06,
        fractions=[1., .5, .25, .125, 0.], minimum_ESS_fraction=.25, maximum_single_fraction=.2,
        projection_chunk=16, H_chunk=2, action_chunk=1, source_query_chunk=8,
        score_chunk=1, parameter_chunk_size=8192, real_native_parameters=1542328,
        common_normalizer_known=False, density_field='q_tilde_not_normalized_Q',
        no_statistical_record_reductions=True, no_automatic_continuation=True,
        no_successor=True, S0X_after_independent_S1_outcome_only=True,
        stop_after_first_independent_veto=True, no_new_head_or_trainable_scale=True,
        terminal_values_never_select_endpoint=True, cold_cost_multiplied_by_records=False,
        require_process_replay_before_release=True, cost_safety_factor=1.5,
        minimum_process_replay_reserve_seconds=1800., terminal_compile_shape=128,
        guided_draw_prior_seconds_per_record=.3,
        identity_center_policy='first_exactly_nonzero_amplitude_among_frozen_replay_rows_no_energy_selection')
    if any(p.get(k) != v for k, v in fixed.items()):
        raise ValueError('exact S1K8x128/512 all-W/source-protected protocol required')
    for key in ('target_q', 'target_m2', 'fit_records', 'FS_records', 'FS_split', 'bilateral_records',
                'bilateral_groups', 'terminal_records', 'terminal_groups', 'fixed_attempts',
                'maximum_updates', 'preflight_records', 'warm_repetitions', 'real_native_parameters'):
        if type(p.get(key)) is not int:
            raise TypeError('exact integer doubled labels, records and attempt clocks required')
    if any(type(v) is not int for pair in p['momentum'] for v in pair):
        raise TypeError('exact integer reciprocal momentum fractions required')
    r = p.get('resources', {})
    if r != dict(allocation_wall_seconds=43200, science_wall_seconds=41400,
                 save_reserve_seconds=600, host_memory_GiB=128, maximum_device_GiB=70,
                 gres='gpu:nvidia_a100_80gb_pcie:1', requested_node=None):
        raise ValueError('one12h A10080/128G allocation without node pin required')
    for key in ('parent_checkpoint', 'parent_result', 'parent_registration', 'fixed_source_carrier', 'fixed_CT_reference', 'runtime_replay'):
        if not Path(p.get(key, '')).is_absolute() or len(p.get(key+'_sha256', '')) != 64:
            raise ValueError('absolute external artifacts and SHA seals required')
    for key in ('parent_parameter_sha256', 'parent_optimizer_sha256', 'parent_rng_sha256'):
        if len(p.get(key, '')) != 64:
            raise ValueError('native inherited state hashes required')
    if isinstance(p.get('fixed_scale'), bool) or not np.isfinite(p.get('fixed_scale', np.nan)) or p['fixed_scale'] <= 0:
        raise ValueError('original positive affine scale required')
    if executing and (p.get('execution_authorized') is not True or p.get('exact_invariants_passed') is not True):
        raise PermissionError('separate exact-invariant and execution authorization required')


def make_persistent_H_jvp(prepare: Callable, action: Callable, scale: float) -> Callable:
    """Construct ONCE; source/ref/rows remain dynamic and preparation stays in AD."""
    def product(params, rows, source, reference):
        return source+scale*(action(prepare(params), rows)-reference)
    return jax.jit(lambda params, direction, rows, source, reference:
        jax.jvp(lambda values:product(values, rows, source, reference), (params,), (direction,)))


class Runtime(old.SourceRuntime):
    """Unchanged exact physical kernels, persistent compiled complete-H JVP."""
    def __init__(self, p: dict, parent: dict, reference: Any):
        super().__init__(p, parent, reference)
        self.counts['physical_source_amplitude_centers'] = 0
        self.compiled_H_jvp = make_persistent_H_jvp(self.raw['prepare'], self.raw['prepared_action'], p['fixed_scale'])
        def physical_batch(params, rows):
            prepared = self.raw['prepare'](params)
            return jax.lax.map(lambda row:self.amplitude(prepared, row), rows, batch_size=1)
        self.compiled_physical_amplitudes = jax.jit(physical_batch)

    def H_JVP(self, params, direction, rows, source, ref):
        value, tangent = jax.block_until_ready(self.compiled_H_jvp(
            params, direction, jnp.asarray(rows), jnp.asarray(source), jnp.asarray(ref)))
        self.counts['complete_H_JVP_centers'] += len(rows)
        return np.asarray(value), np.asarray(tangent)

    def physical_amplitudes(self, params, rows):
        pieces = []
        for first in range(0, len(rows), 128):
            block = rows[first:first+128]; count = len(block)
            padded = np.concatenate((block, np.repeat(block[-1:], 128-count, axis=0)))
            pieces.append(np.asarray(jax.block_until_ready(self.compiled_physical_amplitudes(
                params, jnp.asarray(padded))))[:count])
            self.counts['native_CT_amplitude_centers'] += 256  # current + fixed reference.
            self.counts['physical_source_amplitude_centers'] += 128  # Includes deterministic padding.
        return np.concatenate(pieces)


def measure_cold_and_warm(call: Callable, *, records: int = 128,
                          clock: Callable = perf_counter) -> tuple[Any, dict]:
    """One cold execution, TWO repeated same-shape warm executions, no fitting."""
    times = []; result = None
    for _ in range(3):
        start = clock(); result = jax.block_until_ready(call()); times.append(clock()-start)
    return result, dict(records=records, cold_compile_plus_execution_seconds=times[0],
        repeated_warm_seconds=times[1:], conservative_warm_seconds=max(times[1:]),
        warm_seconds_per_record=max(times[1:])/records,
        first_call_not_extrapolated_as_per_record_warm=True)


def phase_costs(p: dict, profile: dict, *, draw_seconds: float | None = None) -> dict:
    """Warm-only remaining work; future process cold work reserved exactly once."""
    rates = {k:v['warm_seconds_per_record'] for k, v in profile['measurements'].items()}
    s, h, a, j = (rates[k] for k in ('source_Q', 'CT_H', 'CT_amp', 'H_JVP'))
    draw = p['guided_draw_prior_seconds_per_record'] if draw_seconds is None else draw_seconds
    safety = p['cost_safety_factor']
    return dict(fit=safety*(128*(s+2*h+draw)+rates['complex_SR']*128),
        FS=safety*128*(s+6*a+draw), independent=safety*128*(s+3*h+j+draw),
        terminal=safety*512*(s+3*h+draw),
        physical_identity=safety*(1024*rates['physical_amp']+8*(s+3*h)),
        process_replay=profile['process_replay_reserve_seconds'], save=p['resources']['save_reserve_seconds'])


def remaining(p: dict, started: float) -> float:
    value = p['resources']['science_wall_seconds']-(perf_counter()-started)
    if os.environ.get('SLURM_JOB_END_TIME'):
        value = min(value, float(os.environ['SLURM_JOB_END_TIME'])-time()-p['resources']['save_reserve_seconds'])
    return value


def preflight(p: dict, rt: Runtime, state: tuple, replay: dict, output: Path) -> dict:
    """Actual128 input shape, repeated OLD replay rows; zero scientific draws."""
    params, moments, _ = state
    rows = np.resize(replay['rows'], (128, 36)).astype(np.int32)
    result = dict(records=128, reused_replay_rows_for_cost_only=True,
        pseudo_SR_proposals_discarded_not_training=True, measurements={})
    def profile(name, callback):
        value, timing = measure_cold_and_warm(callback)
        result['measurements'][name] = timing; atomic_json(output/'preflight_progress.json', result)
        rt.check_resources(); return value
    source, base, q = profile('source_Q', lambda:rt.source(rows))
    _ = profile('CT_H', lambda:rt.raw['H'](params, rows))
    rt.counts['native_CT_H_centers'] += 3*128
    _ = profile('CT_amp', lambda:rt.raw['amplitudes'](params, rows))
    rt.counts['native_CT_amplitude_centers'] += 3*128
    _ = profile('physical_amp', lambda:rt.physical_amplitudes(params, rows))
    products, ref = rt.H(params, rows, source[:, 1])
    probe = jax.tree.map(lambda x:jnp.sin(jnp.arange(x.size).reshape(x.shape)+.17)*1e-6, params)
    _ = profile('H_JVP', lambda:rt.H_JVP(params, probe, rows, source[:, 1], ref))
    rt.update = make_complex_importance_amplitude_sr(
        lambda values, row:rt.amplitude(rt.raw['prepare'](values), row), params,
        damping=p['damping'], score_chunk=p['score_chunk'], parameter_chunk_size=p['parameter_chunk_size'],
        sr_tangent_fs_radius=p['tangent_FS'], prepare=rt.raw['prepare'], prepared_amplitude=rt.amplitude)
    _ = profile('complex_SR', lambda:rt.update(params, moments, jnp.asarray(rows),
        jnp.asarray(products[:, 1]), jnp.asarray(q), p['learning_rate']))
    rt.counts['full_W_complex_J_rows'] += 3*128
    rt.counts['physical_source_amplitude_centers'] = rt.counts.get('physical_source_amplitude_centers', 0)+3*128
    cold = sum(v['cold_compile_plus_execution_seconds'] for v in result['measurements'].values())
    result.update(process_replay_reserve_seconds=max(p['minimum_process_replay_reserve_seconds'], 1.5*cold),
        cold_elapsed_charged_not_multiplied_again=True, actual_retained_complex_J_GiB=16*128*p['real_native_parameters']/1024**3)
    result['warm_cost_forecast'] = phase_costs(p, result)
    atomic_json(output/'preflight.json', result)
    return result


def clean(value: Any) -> Any:
    """JSON conversion; unknown-Z sampled mass must not be called target norm."""
    if isinstance(value, dict):
        return {({'sampled_parent_norm':'sampled_relative_mass', 'norm_estimates':'relative_mass_estimates'}.get(k, k)):clean(v)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).item() if np.asarray(value).ndim == 0 else np.asarray(value).tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def fixed_scale_replay_error(actual: Any, expected: Any, reference_scale: float) -> float:
    """No floors: exact-node batches use a separately nonzero frozen scale."""
    a, b = np.asarray(actual), np.asarray(expected)
    if (a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(reference_scale) or reference_scale <= 0):
        raise ArithmeticError('finite fixed-scale replay with nonzero reference required')
    return float(np.max(abs(a-b))/reference_scale)


def physical_checks(p: dict, rt: Runtime, params: Any, row: np.ndarray) -> dict:
    """Direct full-H swaps, S² and exact K little-character at a fixed old row."""
    edges = np.asarray([(a, b) for a, b, _ in rt.raw['bonds']], dtype=int)
    coupling = np.asarray([v for _, _, v in rt.raw['bonds']])
    changes = []
    for i, j in edges:
        x = row.copy(); x[i], x[j] = x[j], x[i]; changes.append(x)
    phi = rt.physical_amplitudes(params, np.asarray([row, *changes])); unlike = row[edges[:, 0]] != row[edges[:, 1]]
    direct_h = np.sum(coupling*np.where(unlike, -.25, .25))*phi[0]+np.sum(coupling*np.where(unlike, .5*phi[1:], 0))
    source, _, _ = rt.source(row[None]); cached, _ = rt.H(params, row[None], source[:, 1])
    scale = max(abs(phi[0]), abs(direct_h), abs(cached[0, 1]))
    if scale == 0: raise ArithmeticError('fixed physical identity row unresolved')
    h_error = float(abs(direct_h-cached[0, 1])/scale)
    swaps = []
    for i in np.flatnonzero(row):
        for j in np.flatnonzero(1-row):
            x = row.copy(); x[i], x[j] = x[j], x[i]; swaps.append(x)
    spin = rt.physical_amplitudes(params, np.asarray(swaps))
    denominator = np.linalg.norm(np.r_[phi[0], spin])
    spin_error = float(abs(16*phi[0]+spin.sum())/denominator) if denominator else float('inf')
    cell = ((6, 0), (0, 6)); little = triangular_hnf_reciprocal_little_group(cell, (12, 12))
    operations = [('T10', hnf_site_permutation(cell, translation=(1, 0)), np.exp(-2j*np.pi/3)),
                  ('T01', hnf_site_permutation(cell, translation=(0, 1)), np.exp(-2j*np.pi/3))]
    operations += [(name, image, 1.+0j) for name, _, image in hnf_point_group_permutations(cell) if name in little]
    transformed = rt.physical_amplitudes(params, np.asarray([row[np.asarray(image)] for _, image, _ in operations]))
    char_errors = {name:float(abs(value-character*phi[0])/abs(phi[0])) if phi[0] != 0 else float('inf')
                   for (name, _, character), value in zip(operations, transformed)}
    if h_error > 2e-10 or spin_error > 2e-9 or max(char_errors.values()) > 2e-9:
        raise ArithmeticError('complete physical H/S²/K.A1 identity failed')
    return dict(direct_complete_H_relative_error=h_error, physical_S2_eigen2_relative_residual=spin_error,
                character_relative_errors=char_errors, old_fixed_row_not_a_new_statistical_record=True)


def load_bound(p: dict) -> tuple:
    """Check complete native state/source/replay identities without mutation."""
    for key in ('parent_checkpoint', 'parent_result', 'parent_registration', 'fixed_source_carrier', 'fixed_CT_reference', 'runtime_replay'):
        if file_sha256(Path(p[key])) != p[key+'_sha256']:
            raise ValueError('input seal differs: '+key)
    state, meta = old.restore_state(Path(p['parent_checkpoint']))
    previous = json.loads(Path(p['parent_result']).read_text())
    registration = json.loads(Path(p['parent_registration']).read_text())
    if (previous.get('status') != 'zero_update_budget_gate_before128_fit' or previous.get('committed_updates') != 0
            or previous.get('work', {}).get('IID_rows') != 0
            or previous.get('work', {}).get('full_W_complex_J_rows') != 0
            or registration['plan_sha256'] != meta['plan_sha256']
            or meta['plan_sha256'] != canonical_sha256(meta['plan'])
            or meta['plan']['task'] != old.TASK
            or meta['plan']['fixed_source_carrier_sha256'] != p['fixed_source_carrier_sha256']
            or meta['plan']['fixed_CT_reference_sha256'] != p['fixed_CT_reference_sha256']):
        raise ValueError('current850726 zero-update endpoint/source lineage required')
    if (pytree_sha256(state[0]) != p['parent_parameter_sha256']
            or pytree_sha256(state[1]) != p['parent_optimizer_sha256']
            or canonical_sha256(state[2]) != p['parent_rng_sha256'] or meta['fixed_scale'] != p['fixed_scale']
            or sum(x.size for x in jax.tree.leaves(state[0])) != p['real_native_parameters']):
        raise ValueError('inherited W/optimizer/nextRNG/fixed scale mismatch')
    parent = load_checkpoint(Path(p['fixed_source_carrier']))
    reference = load_checkpoint(Path(p['fixed_CT_reference']))['params']
    with np.load(p['runtime_replay'], allow_pickle=False) as f: replay = dict(f)
    return state, meta, parent, reference, replay


def run(p: dict, output: Path, *, approved_plan_sha256: str) -> dict:
    """One fresh bounded allocation; no resume, requeue or successor side effect."""
    validate_plan(p, executing=True)
    if approved_plan_sha256 != canonical_sha256(p): raise PermissionError('approved sealed plan hash required')
    if jax.default_backend() != 'gpu' or not all('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('allocated compatible A100 required')
    if output.exists(): raise FileExistsError('fresh isolated output required')
    state, _, parent, reference, replay = load_bound(p)
    initial = state[0]; initial_moments = state[1]; started = perf_counter(); output.mkdir(parents=True)
    metadata = dict(plan=p, plan_sha256=canonical_sha256(p), fixed_scale=p['fixed_scale'],
        source_code_commit=os.environ.get('NQS_CODE_COMMIT'), released=False)
    old.save_state(output/'initial.pkl', state, completed_updates=0, metadata=metadata)
    old.save_state(output/'checkpoint.pkl', state, completed_updates=0, metadata=metadata)
    atomic_json(output/'registration.json', metadata)
    report = dict(task=TASK, status='started_unreleased', accepted_updates=0, attempts=[],
        no_new_head=True, all_real_native_W_in_complex_amplitude_AD=True, terminal_values_used_for_selection=False)
    rt = None; completed = 0
    def checkpoint():
        old.save_state(output/'checkpoint.pkl', state, completed_updates=completed, metadata=metadata)
        atomic_json(output/'progress.json', clean(dict(report, wall_seconds=perf_counter()-started)))
    def reserve(role):
        nonlocal state
        seed, rng = old.next_seed(state[2], role); state = (state[0], state[1], rng); checkpoint(); return seed
    def draw(role, count, directory):
        nonlocal state
        seed = reserve(role); rt.forbidden_base_seeds = set(state[2]['used_seeds']+state[2]['excluded_seeds'])
        tick = perf_counter(); rows, note = rt.draw(count, seed)
        rng = old.record_nested_seed_lineage(state[2], note['base_batch_seeds'])
        state = (state[0], state[1], rng); checkpoint()
        atomic_json(directory/(role+'_draw.json'), note)
        np.savez_compressed(directory/(role+'_rows.npz'), rows=rows)
        return rows, (perf_counter()-tick)/count
    try:
        with old.deadline(p['resources']['science_wall_seconds']):
            rt = Runtime(p, parent, reference)
            rows = replay['rows']; source, base, q = rt.source(rows); physical, _ = rt.H(initial, rows, source[:, 1])
            for name, actual, expected in (('source', source[:, 1], replay['source_products']),
                ('physical', physical, replay['physical_products']), ('baseQ', base, replay['normalized_base_density']),
                ('qtilde', q, replay['relative_density'])):
                error = fixed_scale_replay_error(actual, expected, float(np.max(abs(expected))))
                if error > 2e-11: raise ArithmeticError('fixed complex source replay failed: '+name)
            profile = preflight(p, rt, state, replay, output)
            report['preflight'] = profile; report['preflight_work'] = deepcopy(rt.counts)
            draw_rate = p['guided_draw_prior_seconds_per_record']
            for index in range(1, p['fixed_attempts']+1):
                costs = phase_costs(p, profile, draw_seconds=draw_rate)
                if remaining(p, started) < sum(costs.values()):
                    report['stop_reason'] = 'warm_next_attempt_plus_terminal_identity_freshprocess_reserve'; break
                directory = output/f'attempt{index:02d}'; directory.mkdir()
                before, moments = state[:2]; attempt = dict(index=index, accepted=False)
                report['attempts'].append(attempt)
                old.save_state(directory/'prefit.pkl', state, completed_updates=completed, metadata=metadata)
                rows, rate = draw(f'attempt{index}_fit128', 128, directory); draw_rate = max(draw_rate, rate)
                source, base, q = rt.source(rows); fit, ref = rt.H(before, rows, source[:, 1])
                rt.check_resources()
                np.savez_compressed(directory/'fit128.npz', rows=rows, relative_density=q, products=fit, source=source, reference=ref)
                attempt['fit_coverage'] = old.coverage(fit, q)
                if not attempt['fit_coverage']['passed']:
                    report['stop_reason'] = 'fit128_coverage_veto'; break
                candidate, proposed_moments, info = jax.block_until_ready(rt.update(before, moments, jnp.asarray(rows),
                    jnp.asarray(fit[:, 1]), jnp.asarray(q), p['learning_rate']))
                rt.counts['full_W_complex_J_rows'] += 128
                rt.counts['physical_source_amplitude_centers'] = rt.counts.get('physical_source_amplitude_centers', 0)+128
                attempt['SR'] = clean(info)
                rt.check_resources()
                old.save_state(directory/'proposal.pkl', (candidate, proposed_moments, state[2]), completed_updates=completed, metadata=metadata)
                sr_ok = (bool(info['numerically_valid']) and float(info['relative_SR_residual']) < 1e-7
                    and float(info['chosen_direction_tangent_square_ESS']) >= 32
                    and float(info['chosen_direction_absolute_force_projection_ESS']) >= 32
                    and float(info['chosen_direction_tangent_square_maximum_fraction']) <= .2
                    and float(info['chosen_direction_absolute_force_projection_maximum_fraction']) <= .2)
                if not sr_ok: report['stop_reason'] = 'complex_SR_chosen_direction_veto'; break
                costs = phase_costs(p, profile, draw_seconds=draw_rate)
                if remaining(p, started) < sum(v for k, v in costs.items() if k != 'fit'):
                    raise TimeoutError('FS/independent/terminal/process-replay reserve after128fit')
                rows, rate = draw(f'attempt{index}_FS128', 128, directory); draw_rate = max(draw_rate, rate)
                source, _, q = rt.source(rows); candidates = [jax.tree.map(lambda a, b:a+f*(b-a), before, candidate) for f in p['fractions']]
                phi = []; ref_amp = None
                for values in candidates:
                    amplitudes, ref_amp = rt.amplitudes(values, rows, source[:, 1, 0], ref_amp); phi.append(amplitudes)
                phi = np.column_stack(phi)
                rt.check_resources()
                chosen, fs = select_finite_grid(phi, q, np.asarray(p['fractions']), split=64, groups=8,
                                               target=p['target_FS'], maximum=p['maximum_FS'])
                source_fs = common_proposal_FS(np.column_stack((source[:, 1, 0], phi[:, chosen])), q, groups=16)
                attempt.update(finite_FS=clean(fs), cumulative_source_FS=clean(source_fs))
                np.savez_compressed(directory/'FS128.npz', rows=rows, relative_density=q, phi=phi, source=source[:, 1, 0])
                if (not fs['accepted'] or chosen == 4 or source_fs['FS_radians'] > .04 or source_fs['nominal_upper'] > .06):
                    report['stop_reason'] = 'finite_or_cumulative_source_FS_veto'; break
                candidate = candidates[chosen]
                costs = phase_costs(p, profile, draw_seconds=draw_rate)
                if remaining(p, started) < sum(v for k, v in costs.items() if k not in ('fit', 'FS')):
                    raise TimeoutError('independent/terminal/process-replay reserve after128FS')
                rows, rate = draw(f'attempt{index}_bilateral128', 128, directory); draw_rate = max(draw_rate, rate)
                source, _, q = rt.source(rows); current, ref = rt.H(before, rows, source[:, 1]); finite, _ = rt.H(candidate, rows, source[:, 1], ref)
                direction = jax.tree.map(lambda a, b:a-b, candidate, before)
                values, tangent = rt.H_JVP(before, direction, rows, source[:, 1], ref)
                rt.check_resources()
                np.testing.assert_allclose(values, current, rtol=2e-11, atol=2e-14)
                slope = old.independent_direction_statistics(values, tangent, q)
                statistic_seed = reserve(f'attempt{index}_statistics')
                finite_stats = old.paired_relative_moments(np.column_stack((current[:, 0], finite[:, 0])),
                    np.column_stack((current[:, 1], finite[:, 1])), q, np.repeat(np.arange(16), 8), n_sites=36, seed=statistic_seed)
                se = np.maximum(finite_stats['paired_iid_delta_SE'], finite_stats['paired_independent_block_jackknife_SE'])
                delta = np.asarray(finite_stats['candidate_minus_control'])
                accepted = (slope['passed'] and old.coverage(current, q)['passed'] and old.coverage(finite, q)['passed']
                            and delta[0]+2*se[0] < 0 and delta[2]-2*se[2] <= 0)
                attempt.update(independent_bilateral=slope, independent_finite=finite_stats, accepted=bool(accepted),
                    finite_energy_upper_two_maxSE=float(delta[0]+2*se[0]), selected_fraction=p['fractions'][chosen])
                np.savez_compressed(directory/'bilateral128.npz', rows=rows, relative_density=q,
                                    initial=values, candidate=finite, tangent=tangent, source=source)
                if not accepted: report['stop_reason'] = 'independent_H_JVP_or_finite_energy_veto'; break
                completed += 1; state = (candidate, proposed_moments, state[2]); report['accepted_updates'] = completed
                attempt['family_changes'] = old.family_norms(jax.tree.map(lambda a, b:a-b, candidate, before))
                checkpoint()
            # Terminal numerical values never select an earlier/better state.
            costs = phase_costs(p, profile, draw_seconds=draw_rate)
            if remaining(p, started) < sum(v for k, v in costs.items() if k not in ('fit', 'FS', 'independent')):
                raise TimeoutError('fixed512 audit/identity/process-replay reserve not available')
            fixed_amplitudes = rt.physical_amplitudes(state[0], replay['rows'])
            nonzero = np.flatnonzero(fixed_amplitudes != 0)
            if not len(nonzero): raise ArithmeticError('all frozen physical identity centers are exact nodes')
            identity_index = int(nonzero[0])
            report['physical_identities'] = physical_checks(p, rt, state[0], replay['rows'][identity_index])
            report['physical_identities'].update(frozen_replay_row_index=identity_index,
                center_policy=p['identity_center_policy'], no_amplitude_floor=True)
            old.save_state(output/'endpoint_before_terminal.pkl', state, completed_updates=completed, metadata=metadata)
            atomic_json(output/'endpoint_frozen.json', dict(parameter_sha256=pytree_sha256(state[0]),
                checkpoint_sha256=file_sha256(output/'endpoint_before_terminal.pkl'), terminal_not_drawn=True, accepted_updates=completed))
            rows, _ = draw('new_terminal512', 512, output); statistic_seed = reserve('terminal_statistics')
            first, final, sources, densities = [], [], [], []
            for start in range(0, 512, 128):
                block = rows[start:start+128]; source, _, q = rt.source(block)
                a, ref = rt.H(initial, block, source[:, 1]); b, _ = rt.H(state[0], block, source[:, 1], ref)
                first.append(a); final.append(b); sources.append(source); densities.append(q)
            first, final, source, q = map(np.concatenate, (first, final, sources, densities))
            np.savez_compressed(output/'terminal512.npz', rows=rows, initial=first, final=final, source=source, relative_density=q)
            report['terminal'] = old.paired_relative_moments(np.column_stack((first[:, 0], final[:, 0])),
                np.column_stack((first[:, 1], final[:, 1])), q, np.repeat(np.arange(32), 16), n_sites=36, seed=statistic_seed)
            source_statistic_seed = reserve('terminal_source_statistics')
            report['terminal_source_comparison'] = old.paired_relative_moments(np.column_stack((source[:, 1, 0], final[:, 0])),
                np.column_stack((source[:, 1, 1], final[:, 1])), q, np.repeat(np.arange(32), 16),
                n_sites=36, seed=source_statistic_seed)
            report['terminal_coverage'] = dict(initial=old.coverage(first, q), final=old.coverage(final, q),
                diagnostic_not_used_to_select_or_roll_back_endpoint=True)
            report.update(status=('complete_unreleased_pending_fresh_process_replay' if completed else
                'complete_unchanged_source_parent_only_pending_fresh_process_replay'), terminal_records_completed=512,
                lowest_sector_energy_or_gap_certified=False, new_optimized_state_created=bool(completed),
                unchanged_source_protected_parent_only=completed == 0)
            checkpoint()
    except (TimeoutError, MemoryError, ArithmeticError, AssertionError, RuntimeError, ValueError) as error:
        report.update(status='bounded_partial_unreleased_no_automatic_retry', error=str(error))
    finally:
        if rt is not None:
            rng = deepcopy(state[2]); rng['excluded_seeds'] = sorted(set(rng['excluded_seeds']) | set(rt.base_seeds))
            state = (state[0], state[1], rng); report['work'] = rt.counts
            report['post_preflight_work'] = {k:v-report.get('preflight_work', {}).get(k, 0) for k, v in rt.counts.items()}
        report['wall_seconds'] = perf_counter()-started
        report['accepted_updates'] = completed; report['released'] = False
        checkpoint()
        if report['status'] in ('complete_unreleased_pending_fresh_process_replay',
                'complete_unchanged_source_parent_only_pending_fresh_process_replay'):
            report['sealed_artifacts'] = {str(path.relative_to(output)):file_sha256(path)
                for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'result.json'}
        atomic_json(output/'result.json', clean(report))
    return report


def replay(p: dict, output: Path) -> dict:
    """Fresh-process exact first128 SR, final full-H/Q and next-RNG replay."""
    validate_plan(p, executing=True)
    if jax.default_backend() != 'gpu': raise RuntimeError('fresh allocated GPU replay required')
    path = output/'release_certificate.json'
    if path.exists(): raise FileExistsError('existing release certificate is immutable')
    result = json.loads((output/'result.json').read_text())
    if result['status'] not in ('complete_unreleased_pending_fresh_process_replay',
            'complete_unchanged_source_parent_only_pending_fresh_process_replay'):
        return dict(status='refused_incomplete_run_no_release')
    if (result.get('terminal_records_completed') != 512 or not result.get('sealed_artifacts')
            or any(file_sha256(output/name) != digest for name, digest in result['sealed_artifacts'].items())):
        raise ValueError('complete fixed terminal/artifact seals differ before replay')
    initial, _, parent, reference, _ = load_bound(p)
    restored, metadata = old.restore_state(output/'checkpoint.pkl')
    if (metadata['plan_sha256'] != canonical_sha256(p)
            or metadata['source_code_commit'] != os.environ.get('NQS_CODE_COMMIT')):
        raise ValueError('checkpoint plan/source commit differs')
    frozen = json.loads((output/'endpoint_frozen.json').read_text())
    if (frozen['parameter_sha256'] != pytree_sha256(restored[0])
            or frozen['checkpoint_sha256'] != file_sha256(output/'endpoint_before_terminal.pkl')
            or frozen['accepted_updates'] != result['accepted_updates']):
        raise ValueError('frozen terminal endpoint identity differs')
    rt = Runtime(p, parent, reference)
    checks = {}
    proposal_path = output/'attempt01/proposal.pkl'
    if proposal_path.exists():
        before, _ = old.restore_state(output/'attempt01/prefit.pkl')
        expected, _ = old.restore_state(proposal_path)
        with np.load(output/'attempt01/fit128.npz') as f: fit = dict(f)
        update = make_complex_importance_amplitude_sr(lambda values, row:rt.amplitude(rt.raw['prepare'](values), row),
            before[0], damping=p['damping'], score_chunk=1, parameter_chunk_size=8192,
            sr_tangent_fs_radius=.01, prepare=rt.raw['prepare'], prepared_amplitude=rt.amplitude)
        actual, moments, info = jax.block_until_ready(update(before[0], before[1], jnp.asarray(fit['rows']),
            jnp.asarray(fit['products'][:, 1]), jnp.asarray(fit['relative_density']), 1.))
        rt.counts['full_W_complex_J_rows'] += 128
        rt.counts['physical_source_amplitude_centers'] = rt.counts.get('physical_source_amplitude_centers', 0)+128
        errors = [float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected[0]))]
        if not np.isfinite(errors).all() or max(errors) > 2e-11 or pytree_sha256(moments) != pytree_sha256(expected[1]):
            raise ArithmeticError('first actual128 full-W proposal/moments process replay failed')
        checks['first128_parameter_max_absolute_error'] = max(errors)
    with np.load(output/'terminal512.npz') as f: terminal = dict(f)
    rows = terminal['rows'][:2]; source, _, q = rt.source(rows)
    products, _ = rt.H(restored[0], rows, source[:, 1])
    amplitude_scale = float(np.max(abs(terminal['final'][:, 0])))
    error = fixed_scale_replay_error(products, terminal['final'][:2], amplitude_scale)
    expected_q = terminal['relative_density'][:2]
    if not np.isfinite(q).all() or not np.isfinite(expected_q).all() or np.any(expected_q <= 0):
        raise ArithmeticError('positive finite replay density required')
    qerror = float(np.max(abs(q-expected_q)/expected_q))
    if error > 2e-11 or qerror > 2e-12: raise ArithmeticError('terminal complex H/Q process replay failed')
    seed_a, rng_a = old.next_seed(restored[2], 'next_unused_fit')
    again, _ = old.restore_state(output/'checkpoint.pkl'); seed_b, rng_b = old.next_seed(again[2], 'next_unused_fit')
    if seed_a != seed_b or rng_a != rng_b: raise ArithmeticError('exact next RNG replay failed')
    checks.update(final_physical_H_relative_error=error, final_relative_Q_error=qerror,
                  physical_replay_reference_scale=amplitude_scale,
                  next_unused_fit_seed=seed_a, exact_restored_optimizer_sha256=pytree_sha256(restored[1]))
    terminal_coverage_passed = all(result['terminal_coverage'][key]['passed'] for key in ('initial', 'final'))
    certificate = dict(status=('released_complete_fixed_terminal_and_freshprocess_replay' if result['accepted_updates'] else
        'source_parent_only_no_new_optimized_state'),
        checkpoint_sha256=file_sha256(output/'checkpoint.pkl'), parameter_sha256=pytree_sha256(restored[0]),
        optimizer_sha256=pytree_sha256(restored[1]), rng_sha256=canonical_sha256(restored[2]),
        accepted_updates=result['accepted_updates'], checks=checks, no_successor=True,
        new_optimized_state_released=bool(result['accepted_updates']),
        terminal_coverage_passed=bool(terminal_coverage_passed),
        terminal_precision_not_certified_if_failed=not terminal_coverage_passed,
        finite_coverage_is_not_unseen_tail_certificate=True,
        freshprocess_work=rt.counts,
        lowest_sector_energy_or_gap_certified=False, terminal_used_for_selection=False)
    atomic_json(path, certificate); return certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'replay'), required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approved-plan-sha256', default=''); args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    report = (run(plan, args.output, approved_plan_sha256=args.approved_plan_sha256) if args.phase == 'train'
              else replay(plan, args.output))
    print(json.dumps({k:report[k] for k in ('status', 'accepted_updates', 'wall_seconds') if k in report}), flush=True)


if __name__ == '__main__': main()
