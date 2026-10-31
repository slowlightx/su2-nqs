"""Adaptive production integration; tiny exact fixtures are not CT physics."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct6_adaptive_continuation as w
from su2_nqs.frontier.artifacts import canonical_sha256, pytree_sha256
from su2_nqs.vmc.adaptive_sr_controls import (
    AdaptiveSRPolicy, AdaptiveSRState, SRResourceEnvelope, make_controlled_importance_sr,
    sr_control_checkpoint,
)
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr


@pytest.fixture
def plan(tmp_path):
    parent = tmp_path/'parent'; (parent/'terminal_audit').mkdir(parents=True)
    for name in ('registration.json', 'checkpoint.pkl', 'terminal_audit/audit.json'):
        (parent/name).write_text('synthetic-parent-not-physical\n')
    roles = w.required_seed_roles()
    return w.make_plan(.125, parent, seeds=dict(zip(roles, range(101, 101+len(roles)))))


def native_rng():
    return dict(generator=np.random.default_rng(17).bit_generator.state, next_step=37,
        consumed_role_seeds=[9, 10, 11], excluded_seeds=[1, 2, 3])


def state():
    p = {'w': jnp.array([1., .25, -.4, 0.])}
    return p, (jnp.array(500), jax.tree.map(jnp.zeros_like, p), jax.tree.map(jnp.zeros_like, p)), native_rng()


def reg():
    return dict(parent_global_step=500, native_origin=464, initial_native_cursor=37)


def test_frozen_profile_is_new_opt_in_and_attempt_counted(plan):
    w.validate_plan(plan)
    assert plan['step_counter'] == 'attempts' and plan['monitor_after'] == [8, 16, 24]
    policy = w.policy_from_plan(plan)
    assert policy.fit_levels == (5120, 5632) and policy.success_patience == 2
    assert .8*policy.initial_fs_radius == .001
    assert policy.maximum_fs_radius == .0025
    with pytest.raises(PermissionError):
        w.validate_plan(plan, executing=True)
    w.validate_plan(dict(plan, execution_authorized=True), executing=True)
    for changes in (dict(steps=33), dict(fit_records=6144), dict(step_counter='accepted_updates'),
                    dict(audit_records=4096), dict(monitor_after=[8, 16, 24, 32]),
                    dict(automatic_next_allocation=True), dict(damping=.5)):
        with pytest.raises(ValueError):
            w.validate_plan(plan | changes)
    from projects.clebsch_tree_nqs.workflows import ct_bounded_followup as legacy
    with pytest.raises(PermissionError):
        legacy.validate_plan(plan)


def test_no_audit_seed_or_monitor_role_can_be_omitted_or_reused(plan):
    p = deepcopy(plan); p['seeds'].pop('monitor1_jvp')
    with pytest.raises(ValueError):
        w.validate_plan(p)
    p = deepcopy(plan); p['seeds']['monitor1_sample'] = p['seeds']['audit']
    with pytest.raises(ValueError):
        w.validate_plan(p)


def test_atomic_checkpoint_preserves_both_clocks_optimizer_rng_controls(plan, tmp_path):
    s = state(); controls = AdaptiveSRState.initial(w.policy_from_plan(plan))
    progress = w.initial_progress(); path = tmp_path/'adaptive.pkl'
    w.save_adaptive_state(path, s, controls, progress, plan=plan, registration=reg())
    actual, recovered, history = w.load_adaptive_state(path, plan=plan, registration=reg())
    assert pytree_sha256(actual[:2]) == pytree_sha256(s[:2])
    assert canonical_sha256(actual[2]) == canonical_sha256(s[2]) and recovered == controls
    assert history == progress
    first, r1 = w.c6.role_seeds(s[2], 37); second, r2 = w.c6.role_seeds(actual[2], 37)
    assert first == second and r1 == r2
    saved = load_checkpoint(path)
    assert saved['metadata']['absolute_completed_steps'] == 500
    saved['metadata']['progress']['attempts'] = 1
    save_checkpoint(path, saved['params'], optimizer_state=saved['optimizer_state'], metadata=saved['metadata'])
    with pytest.raises(ValueError, match='seal mismatch'):
        w.load_adaptive_state(path, plan=plan, registration=reg())


def test_rejection_is_consumed_attempt_not_accepted_update(plan, tmp_path):
    original = state(); _, next_rng = w.c6.role_seeds(original[2], 37)
    rejected = original[:2]+(next_rng,)
    info = dict(attempt=1, accepted=False, predicted_decrease_per_site=0.,
        work=dict(fit_H_centers=5120))
    progress = w.consume_attempt(w.initial_progress(), info, replay_fit_records=5120)
    assert progress['attempts'] == 1 and progress['accepted_updates'] == 0
    assert progress['fit_H_records_consumed'] == 10240 and not progress['block_geometry_ok']
    controls = AdaptiveSRState.initial(w.policy_from_plan(plan))
    w.save_adaptive_state(tmp_path/'c.pkl', rejected, controls, progress, plan=plan, registration=reg())
    _, _, restored = w.load_adaptive_state(tmp_path/'c.pkl', plan=plan, registration=reg())
    assert restored['accepted_updates'] == 0
    with pytest.raises(ValueError, match='exactly once'):
        w.consume_attempt(progress, info)


def test_checkpoint_cannot_drop_consumed_monitor_history(plan, tmp_path):
    controls = AdaptiveSRState.initial(w.policy_from_plan(plan))
    bad = replace(controls, looks=1, last_completed_step=8, last_endpoint_sha256='a'*64,
        consumed_panel_ids=('panel1',), consumed_monitor_seeds=(201, 202))
    p = w.initial_progress()
    w.save_adaptive_state(tmp_path/'bad.pkl', state(), bad, p, plan=plan, registration=reg())
    with pytest.raises(ValueError, match='history'):
        w.load_adaptive_state(tmp_path/'bad.pkl', plan=plan, registration=reg())


def test_controller_and_native_seals_reject_tampering(plan, tmp_path):
    path = tmp_path/'state.pkl'; controls = AdaptiveSRState.initial(w.policy_from_plan(plan))
    w.save_adaptive_state(path, state(), controls, w.initial_progress(), plan=plan, registration=reg())
    saved = load_checkpoint(path); saved['metadata']['adaptive_sr']['state']['damping'] = .1
    save_checkpoint(path, saved['params'], optimizer_state=saved['optimizer_state'], metadata=saved['metadata'])
    with pytest.raises(ValueError, match='identity changed'):
        w.load_adaptive_state(path, plan=plan, registration=reg())


def toy_runtime(monkeypatch, plan):
    """Independent complete four-state H, retaining one actual amplitude node."""
    p = deepcopy(plan)
    policy = AdaptiveSRPolicy(fit_levels=(64, 96), initial_fs_radius=.02,
        minimum_fs_radius=.001, maximum_fs_radius=.1)
    p.update(adaptive_policy=asdict(policy), calibration_split=64, calibration_records=128,
        H_chunk=16, monitor_jvp_records=64, monitor_jvp_groups=8)
    s = state()
    H = jnp.array([[.1, -.7, .2, .3], [-.7, .3, -.1, .15],
        [.2, -.1, -.5, -.2], [.3, .15, -.2, .6]])
    def amp(params, row):
        return params['w'][row[0]]
    def action(params, rows):
        return params['w'][rows[:, 0]], (H@params['w'])[rows[:, 0]]
    rt = dict(cfg=None, prepare=lambda x: x, projected=amp, action=action)
    rt['controlled_update'] = make_controlled_importance_sr(amp, s[0], policy,
        score_chunk=8, parameter_chunk_size=2, prepare=rt['prepare'], prepared_amplitude=amp)
    def norm(params, runtime):
        z = float(np.sum(np.asarray(params['w'])**2))
        return jax.tree.map(lambda x: x/np.sqrt(z), params), dict(original_raw_norm=z)
    def draw(banks, runtime, count, seed):
        rows = np.random.default_rng(seed).integers(0, 4, (count, 1))
        return rows, dict(seed=seed)
    def bridge(params, rows):
        a, h = action(params, rows); q = jnp.full(len(rows), .25)
        return a, h, q, q
    rt['bridge'] = bridge
    rt['orbit_query'] = lambda bank, rows: jnp.stack((amp(bank, rows.T), jnp.full(len(rows), .25)), axis=1)
    monkeypatch.setattr(w.c6.diagnostic, 'normalized_raw_bank', norm)
    monkeypatch.setattr(w.c6.diagnostic, 'prepare_feature_lut_bank', lambda bank, cfg: bank)
    monkeypatch.setattr(w.c6.diagnostic, 'native_projected_amplitude',
        lambda rt, par, rows: par['w'][rows[:, 0]])
    monkeypatch.setattr(w.c6.diagnostic, 'draw_orbit_rows', draw)
    monkeypatch.setattr(w.c6, 'draw_rows', draw)
    monkeypatch.setattr(w.c6, 'save_draw',
        lambda path, rows, report: np.savez(path.with_suffix('.npz'), rows=rows))
    monkeypatch.setattr(w.c6.diagnostic, '_device_peak', lambda: 0)
    monkeypatch.setattr(w, 'full_support_sz0_density', lambda a, b, **k: np.full(len(a), .25))
    return p, s, rt, H, policy


def test_actual_full_W_transition_process_replay_and_guard(plan, tmp_path, monkeypatch):
    p, s, rt, H, policy = toy_runtime(monkeypatch, plan)
    controls = AdaptiveSRState.initial(policy)
    a, report = w.transition(p, p, rt, s, controls, tmp_path/'first', attempt=1)
    b, again = w.transition(p, p, rt, s, controls, tmp_path/'replay', attempt=1)
    replay = w.compare_transition(a, b, report, again, tmp_path/'first', tmp_path/'replay')
    assert replay['params_optimizer_next_RNG_controls_compared']
    assert report['work']['fit_H_centers'] == 64
    assert report['work']['full_native_jacobian_rows'] == 64
    assert a[2]['next_step'] == 38 and report['controls']['fs_radius'] == .02
    assert max(report['native_fit_H_scale_replay_relative_errors']) < 2e-15
    assert max(report['native_amplitude_replay_errors']) < 2e-15
    if report['accepted']:
        assert report['calibration']['guard']['nominal_upper'] <= .02
        assert int(a[1][0]) == int(s[1][0])+1
        expected = max(0., -report['actual_native_factor']*
            report['SR']['raw_energy_directional_derivative']/36.)
        assert report['predicted_decrease_per_site'] == expected


def test_numerically_rejected_SR_keeps_params_moments_but_advances_rng(plan, tmp_path, monkeypatch):
    p, s, rt, H, policy = toy_runtime(monkeypatch, plan)
    rt['controlled_update'] = lambda par, opt, *a, **k: (par, opt, {'accepted': jnp.array(False)})
    actual, report = w.transition(p, p, rt, s, AdaptiveSRState.initial(policy), tmp_path/'rejected', attempt=1)
    assert not report['accepted'] and 'attempt_consumed' in report['reason']
    assert pytree_sha256(actual[:2]) == pytree_sha256(s[:2])
    assert actual[2]['next_step'] == 38
    assert set(report['seeds']) == {'fit', 'selection', 'guard'}


def test_bilateral_JVP_is_actual_complete_H_and_preserves_nodes(plan, tmp_path, monkeypatch):
    p, s, rt, H, policy = toy_runtime(monkeypatch, plan)
    end = {'w': s[0]['w']+jnp.array([.01, -.03, .02, .015])}
    result = w.frozen_direction_monitor(p, p, rt, s[0], end, tmp_path/'jvp', 10391, lambda: 1000.)
    with np.load(tmp_path/'jvp/products.npz') as f:
        rows = f['rows'][:, 0]; delta = end['w']-s[0]['w']
        np.testing.assert_allclose(f['hdpsi'], np.asarray(H@delta)[rows], atol=2e-16)
        np.testing.assert_allclose(f['hpsi'], np.asarray(H@s[0]['w'])[rows], atol=2e-16)
        assert np.any(f['psi'] == 0) and np.any(f['hpsi'][f['psi'] == 0] != 0)
    assert result['both_coherent_terms'] and result['actual_finite_parameter_direction']
    assert result['full_H_centers'] == 3*64


def test_zero_rejected_direction_does_not_claim_noise_plateau(plan, tmp_path, monkeypatch):
    p, s, rt, H, policy = toy_runtime(monkeypatch, plan)
    report = w.frozen_direction_monitor(p, p, rt, s[0], s[0], tmp_path/'zero', 123, lambda: 1000.)
    assert report['directional_slope'] is None and report['directional_slope_se'] is None
    assert report['full_H_centers'] == 0


def test_monitor_coverage_never_turns_positive_support_into_tail_certificate(plan):
    d = dict(effective_contributions=[200., 200.], maximum_sample_fraction=[.01, .01])
    m = {key: deepcopy(d) for key in ('norm_concentration', 'H2_concentration', 'residual_concentration')}
    assert w.monitor_coverage(plan, m)
    m['residual_concentration']['effective_contributions'][1] = 63.
    assert not w.monitor_coverage(plan, m)
    m['residual_concentration']['effective_contributions'][1] = np.nan
    assert not w.monitor_coverage(plan, m)


def test_measured_resource_envelope_and_terminal_guard(plan):
    report = dict(timings=dict(fit_full_H=1200., full_native_importance_amplitude_SR=300.),
        work=dict(fit_H_centers=5120), wall_seconds=1800.)
    progress = w.initial_progress(); progress.update(attempts=8,
        fit_H_records_consumed=9*5120, transitions=[report]*8)
    resource = w.resources_for_next_block(plan, dict(parameter_count=1535560), progress,
        available_seconds=100000., reserve_seconds=10000.)
    assert resource.permits(5632) and not resource.permits(6144)
    assert resource.remaining_updates == 24
    progress['attempts'] = 32
    with pytest.raises(ValueError, match='terminal endpoint'):
        w.resources_for_next_block(plan, dict(parameter_count=1535560), progress,
            available_seconds=1., reserve_seconds=0.)


def test_terminal_reserve_no_assumed_acceleration(plan):
    cost = w.reserve_costs(plan, .1)
    assert cost['audit_reserve_seconds'] == pytest.approx(1.5*2*8192*.1+600)
    assert cost['monitor_reserve_seconds'] > 1.5*2*4096*.1
    assert cost['no_hardware_acceleration_assumed']
    with pytest.raises(ValueError):
        w.reserve_costs(plan, 0.)


def test_independent_gain_proxy_does_not_use_observed_energy_or_unscaled_fit():
    progress = dict(block_predicted_decrease=.0002,
        transitions=[dict(accepted=True)]*6+[dict(accepted=False)]*2)
    slope = dict(directional_slope=-2.e-6, directional_slope_se=3.e-6)
    p = w.independent_gain_proxy(progress, slope, 8)
    assert p['controller_predicted_decrease_per_site'] == pytest.approx(12.e-6)
    assert p['raw_in_fit_linear_block_prediction_per_site'] == .0002
    assert p['paired_observed_energy_not_used_as_prediction']
    assert p['local_representativeness_heuristic_not_block_forecast']
    assert w.independent_gain_proxy(progress, dict(directional_slope=None,
        directional_slope_se=None), 8)['controller_predicted_decrease_per_site'] == 0.


def test_monitor_consumes_own_streams_and_changes_only_future_controls(plan, tmp_path, monkeypatch):
    old = state(); endpoint = (jax.tree.map(lambda x: x+.01, old[0]), old[1], old[2])
    controls = AdaptiveSRState.initial(w.policy_from_plan(plan)); progress = w.initial_progress()
    progress.update(attempts=8, accepted_updates=7,
        block_geometry_ok=False, block_predicted_decrease=.0002,
        transitions=[dict(accepted=True)]*7+[dict(accepted=False)])
    def paired(kernel, rt, parent, final, path, records, seed, stats, remaining):
        path.mkdir(); np.savez(path/'products.npz', rows=np.zeros((2, 1)))
        assert seed == plan['seeds']['monitor1_sample'] != plan['seeds']['audit']
        d = dict(effective_contributions=[200., 200.], maximum_sample_fraction=[.01, .01])
        return dict(moments=dict(candidate_minus_control=[-1e-6, 0., 0., 0.],
            paired_independent_block_jackknife_SE=[1e-5]*4,
            **{key: deepcopy(d) for key in ('norm_concentration', 'H2_concentration', 'residual_concentration')}))
    monkeypatch.setattr(w.c6, 'paired_audit', paired)
    monkeypatch.setattr(w, 'frozen_direction_monitor', lambda *a: dict(
        directional_slope=-2e-6, directional_slope_se=1e-5))
    envelope = SRResourceEnvelope(parameter_count=4, maximum_device_bytes=70*1024**3,
        remaining_updates=24, remaining_fit_H_records=200000, remaining_training_seconds=80000.,
        seconds_per_fit_record=.01, other_seconds_per_step=1.)
    new, history, decision = w.monitor_block(plan, {}, {}, old, endpoint, old[0], controls,
        progress, tmp_path/'monitor', envelope, lambda: 80000.)
    assert controls.looks == 0 and new.looks == 1
    assert new.fs_radius == .000625 and new.damping == .1
    assert decision['reason'] == 'geometry_failure' and decision['applies_to_future_blocks_only']
    assert not decision['terminal_audit_used']
    assert len(history['monitor_history']) == len(history['consumed_monitor_roles']) == 1
    assert decision['monitor']['predicted_decrease'] == pytest.approx(14e-6)
    assert history['block_predicted_decrease'] == 0. and history['block_geometry_ok']


@pytest.mark.parametrize('custom_profile',[False,True])
def test_first_rejected_attempt_is_process_replayed_committed_once(plan, tmp_path, monkeypatch,custom_profile):
    p = dict(plan, execution_authorized=True)
    planfile = tmp_path/'plan.json'; planfile.write_text(json.dumps(p)); out = tmp_path/'run'
    original = state()
    monkeypatch.setattr(w.jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(w.jax, 'devices', lambda: [type('D', (), {'device_kind': 'NVIDIA A100 80GB'})()])
    monkeypatch.setattr(w, 'transitive_local_source_manifest', lambda *a, **k: {'synthetic': 'source'})
    monkeypatch.setattr(w.legacy, 'verify_production_runtime', lambda p: None)
    monkeypatch.setattr(w.legacy, 'check_allocator', lambda *a: {'necessary_check_only': True})
    def bind(p, path):
        (path/'parent_snapshot').mkdir(); w.legacy.save_state(path/'parent.pkl', original, 0, 500)
        return dict(parent_global_step=500, native_origin=464, parent_seals={}, parameter_count=4)
    monkeypatch.setattr(w.legacy, 'bind_parent', bind)
    monkeypatch.setattr(w.legacy, 'replay_parent', lambda *a: dict(warm_H_seconds=.001))
    monkeypatch.setattr(w, 'make_runtime', lambda *a: ({}, {}))
    calls = []
    def proposal(plan, kernel, rt, s, controls, path, *, attempt):
        path.mkdir(); seeds, rng = w.c6.role_seeds(s[2], s[2]['next_step']); calls.append(seeds)
        return s[:2]+(rng,), dict(attempt=attempt, accepted=False, selected_index=4,
            actual_native_factor=0., seeds=seeds, rng_after_sha256=canonical_sha256(rng),
            controls=asdict(controls), work=dict(fit_H_centers=5120), wall_seconds=1.,
            predicted_decrease_per_site=0.,
            timings=dict(fit_full_H=.2, full_native_importance_amplitude_SR=.3))
    monkeypatch.setattr(w, 'transition', proposal)
    from types import SimpleNamespace
    from projects.clebsch_tree_nqs.workflows import ct6_adaptive_campaign as campaign
    profile=(SimpleNamespace(__file__=campaign.__file__,validate_plan=w.validate_plan,bind_parent=bind,
        initial_controls=lambda p,b:AdaptiveSRState.initial(w.policy_from_plan(p)),
        resources_for_next_block=w.resources_for_next_block) if custom_profile else None)
    execute=lambda phase:w.run(planfile,out,phase,profile=profile)
    first = execute('prepare'); replay = execute('replay')
    assert first['status'] == 'first_attempt_provisional' and not first['first_proposal_accepted']
    assert replay['status'] == 'first_attempt_replayed_committed_once'
    assert replay['attempts'] == 1 and replay['accepted_updates'] == 0
    assert calls[0] == calls[1]
    registration = json.loads((out/'registration.json').read_text())
    actual, controls, progress = w.load_adaptive_state(out/'checkpoint.pkl', plan=p, registration=registration)
    assert actual[2]['next_step'] == 38 and int(actual[1][0]) == 500
    assert progress['fit_H_records_consumed'] == 10240 and progress['attempts'] == 1
    # Returning a cached phase cannot replay/commit the attempt twice.
    execute('replay'); assert len(calls) == 2
    observed_attempts = []
    def monitoring(plan, kernel, rt, start, end, parent, controls, progress, output, resources, remaining):
        output.mkdir(); attempt = progress['attempts']; observed_attempts.append(attempt)
        index = attempt//8; history = deepcopy(progress)
        roles = {k: plan['seeds'][f'monitor{index}_{k}'] for k in ('sample', 'statistics', 'jvp')}
        updated = replace(controls, looks=index, last_completed_step=attempt,
            last_endpoint_sha256=pytree_sha256(end[0]),
            consumed_panel_ids=controls.consumed_panel_ids+(f'panel{index}',),
            consumed_monitor_seeds=controls.consumed_monitor_seeds+(roles['sample'], roles['statistics']))
        decision = dict(can_train_next_block=True)
        history['monitor_history'].append(decision); history['consumed_monitor_roles'].append(roles)
        history['block_geometry_ok'] = True; history['block_predicted_decrease'] = 0.
        return updated, history, decision
    monkeypatch.setattr(w, 'monitor_block', monitoring)
    training = execute('train')
    assert training['attempts'] == 32 and training['accepted_updates'] == 0
    assert observed_attempts == [8, 16, 24]
    assert len(calls) == 33  # exactly32 attempts, plus ONE first-attempt process replay
    restored, controls, history = w.load_adaptive_state(out/'checkpoint.pkl', plan=p, registration=registration)
    assert restored[2]['next_step'] == 69 and int(restored[1][0]) == 500
    assert history['fit_H_records_consumed'] == 33*5120 and controls.looks == 3
    def final_audit(kernel, rt, parent, final, path, count, seed, stats, remaining):
        assert seed == p['seeds']['audit'] and count == 8192
        assert seed not in controls.consumed_monitor_seeds
        return dict(status='complete', records=count)
    monkeypatch.setattr(w.c6, 'paired_audit', final_audit)
    final = execute('audit')
    assert final['independent_from_training_geometry_and_adaptation']
    assert final['attempts'] == 32 and final['accepted_updates'] == 0


def test_restart_cannot_skip_uncommitted_monitor_or_missing_block_parent(plan, tmp_path):
    controls = AdaptiveSRState.initial(w.policy_from_plan(plan)); progress = w.initial_progress()
    progress['attempts'] = 8
    assert w.restart_boundary_reason(plan, tmp_path, controls, progress).startswith('uncommitted_monitor')
    controls = replace(controls, looks=1, last_completed_step=8)
    progress['monitor_history'] = [{}]
    assert w.restart_boundary_reason(plan, tmp_path, controls, progress).startswith('missing_committed_block_parent')
    (tmp_path/'block_start_8.pkl').write_bytes(b'fixture-presence-only')
    assert w.restart_boundary_reason(plan, tmp_path, controls, progress) is None
    progress['attempts'] = 9
    assert w.restart_boundary_reason(plan, tmp_path, controls, progress) is None


def test_old_fixed_SR_math_matches_new_opt_in_control_binding():
    policy = AdaptiveSRPolicy(fit_levels=(8,))
    controls = AdaptiveSRState.initial(policy)
    params = {'w': jnp.array([1., .2, -.3, .4])}; rows = jnp.tile(jnp.arange(4), 2)[:, None]
    amp = lambda p, row: p['w'][row[0]]
    moments = (jnp.array(4), jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))
    hpsi = jnp.array([.4, -.2, .3, .8, .4, -.2, .3, .8]); q = jnp.ones(8)/4
    old = make_importance_amplitude_sr(amp, params, damping=.05,
        sr_tangent_fs_radius=.001, score_chunk=4, parameter_chunk_size=2)
    new = make_controlled_importance_sr(amp, params, policy, score_chunk=4, parameter_chunk_size=2)
    a = old(params, moments, rows, hpsi, q, .03)
    b = new(params, moments, rows, hpsi, q, .03, controls=controls)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, rtol=0., atol=0.)
