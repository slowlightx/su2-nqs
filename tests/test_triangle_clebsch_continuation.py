"""Bounded native-width continuation controls, without a local N36 training."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as campaign
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint

jax.config.update('jax_enable_x64', True)
CONFIG = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote/continuation_20261019/ct6'


def _plan():
    return json.loads((CONFIG/'j0_binary_truncated_w16.json').read_text())


def quality_plan(width=24):
    path=CONFIG.parents[1]/'ct6_continue_20261021/j0125_binary_truncated_w20_continue32.json'
    plan=json.loads(path.read_text())
    plan.update(task=campaign.QUALITY_TASK,optimization_control='paired_capacity_reset_direct_SR',
        learning_rate=.03,sr_use_adam=False,sr_tangent_fs_radius=.003,parameter_chunk_size=8192,
        matched_wider_arm='paired_width24')
    plan['model']['multiplicity_cap']=width
    plan['training'].update(samples=4096,steps=32)
    plan['audit'].update(samples=65536,chains=256)
    plan['budget'].update(pilot_seconds=10800,replay_seconds=3600,train_seconds=72000)
    return plan


def wide_production_plan(width=28):
    path = CONFIG.parents[1]/'ct6_relaunch_20261022/j0125_ridge005.json'
    plan = json.loads(path.read_text())
    plan.update(task=campaign.WIDE_PRODUCTION_TASK,
        optimization_control='single_wide_production_direct_SR',
        campaign_deadline_utc='2099-01-01T00:00:00+00:00')
    plan['model']['multiplicity_cap'] = width
    plan['training']['steps'] = 128
    plan['audit'].update(samples=65536, chains=256)
    plan['budget'].update(pilot_seconds=4*3600, replay_seconds=2*3600, train_seconds=64*3600)
    return plan


@pytest.mark.parametrize('width', [28, 32])
def test_wide_production_explicit_fixed128_contract_and_unchanged_physical_guards(width):
    plan = wide_production_plan(width)
    campaign.validate_plan(plan)
    assert plan['parent']['model']['multiplicity_cap'] == 20
    assert plan['training']['samples'] == 4096 and plan['pilot_steps'] == 2
    assert plan['maximum_fs_sine'] == .005 and plan['minimum_ess_fraction'] == .98
    # The128 total includes the two committed pilot updates, not128 extra.
    assert plan['training']['steps']-plan['pilot_steps'] == 126


@pytest.mark.parametrize('change', ['parent', 'width', 'steps', 'records', 'audit', 'arity',
                                  'reset', 'ridge', 'memory', 'budget', 'deadline'])
def test_wide_production_rejects_changed_scope_or_unbounded_budget(change):
    plan = wide_production_plan()
    if change == 'parent': plan['parent']['model']['multiplicity_cap'] = 24
    if change == 'width': plan['model']['multiplicity_cap'] = 40
    if change == 'steps': plan['training']['steps'] = 256
    if change == 'records': plan['training']['samples'] = 2048
    if change == 'audit': plan['audit']['samples'] = 32768
    if change == 'arity': plan['arity'] = 'ternary_joint'
    if change == 'reset': plan['reset_optimizer'] = False
    if change == 'ridge': plan['damping'] = .5
    if change == 'memory': plan['budget']['maximum_peak_device_bytes'] = 80*1024**3
    if change == 'budget': plan['budget']['train_seconds'] = 96*3600
    if change == 'deadline': plan['campaign_deadline_utc'] = '2099-01-01T00:00:00'
    with pytest.raises(ValueError): campaign.validate_plan(plan)


def test_wide_measured_pilot_cost_cannot_be_bypassed_by_passed_string():
    plan = wide_production_plan(32)
    pilot = dict(status='passed', completed_steps=2, warm_step_seconds=1000.,
        peak_device_bytes=65*1024**3, forecast_remaining_seconds=142900.,
        maximum_remaining_seconds=plan['budget']['train_seconds'])
    campaign.validate_wide_pilot_cost(plan, pilot)
    for field, bad in (('warm_step_seconds', 5000.), ('warm_step_seconds', float('nan')),
                       ('peak_device_bytes', 71*1024**3), ('peak_device_bytes', 0),
                       ('forecast_remaining_seconds', 1.), ('completed_steps', 0)):
        with pytest.raises(ValueError):
            campaign.validate_wide_pilot_cost(plan, dict(pilot, **{field: bad}))


def test_wide_time_guard_reserves_whole_next_step_and_audit_then_pauses():
    plan = wide_production_plan()
    now = datetime(2098, 12, 31, 23, 0, tzinfo=timezone.utc)
    history = [dict(seconds=800.), dict(seconds=1000.), dict(seconds=900.)]
    step = campaign.wide_production_time_guard(plan, 'train', history, 0., now=now)
    assert step['can_start'] and step['reserved_seconds'] == 1370.
    audit = campaign.wide_production_time_guard(plan, 'train', history, 0., now=now, audit=True)
    assert not audit['can_start'] and audit['reserved_seconds'] == 20120.
    budget = campaign.wide_production_time_guard(plan, 'train', history,
        plan['budget']['train_seconds']-1000., now=now)
    assert not budget['can_start'] and budget['available_seconds'] == 1000.


@pytest.mark.parametrize('width',[20,24])
def test_quality_profile_keeps_parent_and_common_physical_guards(width):
    plan=quality_plan(width);campaign.validate_plan(plan)
    assert plan['maximum_fs_sine']==.005 and plan['minimum_ess_fraction']==.98
    assert plan['sr_solver']=='dual_force' and plan['parameter_chunk_size']==8192


@pytest.mark.parametrize('change',['solver','width','parent','chunk','step','audit','guard','optimizer'])
def test_quality_profile_refuses_undeclared_physical_or_budget_changes(change):
    plan=quality_plan()
    if change=='solver':plan['sr_solver']='woodbury_gradient'
    if change=='width':plan['model']['multiplicity_cap']=28
    if change=='parent':plan['parent']['model']['multiplicity_cap']=18
    if change=='chunk':plan['parameter_chunk_size']=16384
    if change=='step':plan['training']['steps']=65
    if change=='audit':plan['audit']['samples']=16384
    if change=='guard':plan['maximum_fs_sine']=.01
    if change=='optimizer':plan['sr_use_adam']=True
    with pytest.raises(ValueError):campaign.validate_plan(plan)


def test_all_twelve_frozen_plans_have_the_right_serial_counts_and_distinct_streams():
    paths = sorted(CONFIG.glob('*.json')); assert len(paths) == 12
    seeds = []
    for path in paths:
        plan = json.loads(path.read_text()); campaign.validate_plan(plan)
        seeds += list(plan['seeds'].values())
        assert plan['parent']['absolute_completed_steps'] == (200 if plan['j2_over_j1'] == 0 else 220)+{
            16: 0, 18: 32, 20: 48}[plan['model']['multiplicity_cap']]
    assert len(seeds) == len(set(seeds))
    old = json.loads((CONFIG.parents[1]/'optimizer_comparison_20261019/plan.json').read_text())
    assert set(seeds).isdisjoint(v for row in old['seeds'].values() for v in row.values())
    assert sum(sum(json.loads(p.read_text())['budget'][k] for k in (
        'pilot_seconds', 'replay_seconds', 'train_seconds')) for p in paths) == 72*3600


@pytest.mark.parametrize('key,value', [('learning_rate', .001), ('damping', .01),
    ('maximum_fs_sine', .01), ('minimum_ess_fraction', .9), ('pilot_steps', 0),
    ('binary_rank', 9), ('tolerance', 1e-8), ('halvings', True)])
def test_reject_unregistered_optimization_contracts(key, value):
    plan = _plan(); plan[key] = value
    with pytest.raises(ValueError): campaign.validate_plan(plan)


def test_independent_physical_guard_only_shrinks_and_detects_out_of_training_step():
    parent = dict(x=jnp.asarray(0.)); proposed = dict(x=jnp.asarray(.01))
    evaluate = lambda p, rows: jnp.exp(p['x']*rows)
    chosen, report = campaign.shrink_physical_update(parent, proposed, jnp.asarray([0., 1.]),
        jnp.asarray([0., 10.]), evaluate, maximum_fs_sine=.02, minimum_ess_fraction=.98, halvings=12)
    assert report['accepted'] and 0 < report['applied_factor'] < 1
    assert chosen['x'] < proposed['x']
    assert max(report[k]['sampled_fs_sine'] for k in ('training', 'calibration')) <= .02
    # Small candidates retain their original step; no target-FS amplification.
    small = dict(x=jnp.asarray(.0001))
    chosen, report = campaign.shrink_physical_update(parent, small, jnp.asarray([0., 1.]),
        jnp.asarray([0., 10.]), evaluate, maximum_fs_sine=.02, minimum_ess_fraction=.98, halvings=12)
    assert report['applied_factor'] == 1
    np.testing.assert_array_equal(chosen['x'], small['x'])
    chosen, report = campaign.shrink_physical_update(parent, parent, jnp.asarray([0., 1.]),
        jnp.asarray([0., 10.]), evaluate, maximum_fs_sine=.02, minimum_ess_fraction=.98, halvings=12)
    assert not report['accepted'] and report['amplitude_evaluations'] == 0
    gauge = lambda p, rows: jnp.exp(p['x'])*jnp.ones_like(rows)
    chosen, report = campaign.shrink_physical_update(parent, small, jnp.asarray([0., 1.]),
        jnp.asarray([0., 10.]), gauge, maximum_fs_sine=.02, minimum_ess_fraction=.98, halvings=2)
    assert not report['accepted']
    np.testing.assert_array_equal(chosen['x'], parent['x'])


def _mock_campaign(monkeypatch, tmp_path, *, reject_at=None, nonzero_parent_moments=False,
                   full_H_calls=2048):
    plan = _plan(); plan['output'] = str(tmp_path/'output'); plan['require_gpu'] = False
    p = dict(x=jnp.asarray([.1, .2], jnp.float64)); _, a, b = _manual_adam_init(p)
    if nonzero_parent_moments:
        a = jax.tree.map(lambda value: value+.7, a)
        b = jax.tree.map(lambda value: value+.3, b)
    state = (p, (200, a, b), jax.random.PRNGKey(1), jnp.zeros((64, 36), jnp.int32),
             jax.random.PRNGKey(2), jnp.ones((64, 36), jnp.int32))
    calls = []
    lineage = dict(absolute_completed_steps=200, checkpoint_sha256='fixed_parent',
        params_sha256=pytree_sha256(p), model=plan['parent']['model'])
    monkeypatch.setattr(campaign, 'load_parent', lambda _: (state, lineage))
    monkeypatch.setattr(campaign, 'transitive_local_source_manifest', lambda *a, **k: {'test': 'frozen'})
    def transition(old):
        step = int(old[1][0])+1; calls.append(step)
        accepted = step != reject_at
        new = (jax.tree.map(lambda v: v+.001, old[0]), (step, old[1][1], old[1][2]),
               old[2]+1, old[3], old[4]+1, old[5])
        return new, dict(accepted=accepted, full_H_calls=full_H_calls, diagnostic=.001), dict(sampling_seconds=.01)
    monkeypatch.setattr(campaign, 'runtime', lambda *a: dict(transition=transition))
    monkeypatch.setattr(campaign.shared, 'preflight', lambda *a: dict(status='passed'))
    def audit(rt, p, spec, seed, out):
        return dict(params_sha256=pytree_sha256(p), samples=spec['samples'], chains=spec['chains'],
            seed=seed, independent_from_training_and_calibration=True)
    monkeypatch.setattr(campaign.shared, 'terminal_audit', audit)
    return plan, calls


def test_two_committed_pilot_updates_full_replay_and_fixed_endpoint_resume(monkeypatch, tmp_path):
    plan, calls = _mock_campaign(monkeypatch, tmp_path); seal = canonical_sha256(plan); out = Path(plan['output'])
    assert campaign.run(plan, 'pilot', seal)['status'] == 'passed'
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert cp['metadata']['completed_steps'] == 2 and cp['optimizer_state'][0] == 202
    assert calls == [201, 202, 203]  # Last update is an uncommitted replay fixture.
    before = file_sha256(out/'checkpoint.pkl')
    assert campaign.run(plan, 'replay', seal)['status'] == 'replay_passed'
    assert file_sha256(out/'checkpoint.pkl') == before
    assert calls[-1] == 203
    result = campaign.run(plan, 'train', seal)
    assert result['completed_steps'] == 32 and result['absolute_completed_steps'] == 232
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert cp['optimizer_state'][0] == 232 and len(cp['metadata']['history']) == 32
    assert result['full_H_training_centers'] == 2048*32
    assert result['fixture_and_replay_H_centers'] == 4096
    count = len(calls)
    assert campaign.run(plan, 'train', seal)['status'] == 'fixed_steps_complete'
    assert len(calls) == count


def test_new_calibrated_profile_reuses_full_checkpoint_and_process_replay(monkeypatch, tmp_path):
    original, calls = _mock_campaign(monkeypatch, tmp_path)
    path = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote/ct6_sixday_20261022/j0125_ridge050.json'
    plan = json.loads(path.read_text())
    plan.update(output=original['output'], require_gpu=False, reset_optimizer=False,
                campaign_deadline_utc='2099-01-01T00:00:00+00:00')
    # The numerical transition is mocked; retain the actual new validation,
    # counter, immutable-registration, fixture, and restore-release machinery.
    monkeypatch.setattr(campaign, 'numerical_runtime_fingerprint',
                        lambda: {'node': plan['runtime_contract']['node']})
    seal = canonical_sha256(plan)
    assert campaign.run(plan, 'pilot', seal)['status'] == 'passed'
    assert campaign.run(plan, 'replay', seal)['status'] == 'replay_passed'
    result = campaign.run(plan, 'train', seal)
    assert result['completed_steps'] == 16 and result['absolute_completed_steps'] == 216
    cp = load_checkpoint(Path(plan['output'])/'checkpoint.pkl')
    assert cp['optimizer_state'][0] == 216 and len(cp['metadata']['history']) == 16
    assert len(calls) == 18  #16committed +fixture +separate replay, not20updates.


def test_calibrated_reset_fresh_start_child_process_replay_and_fixed_endpoint(monkeypatch, tmp_path):
    original, calls = _mock_campaign(monkeypatch, tmp_path, nonzero_parent_moments=True, full_H_calls=4096)
    path = CONFIG.parents[1]/'ct6_sixday_20261022/j0125_ridge050.json'
    plan = json.loads(path.read_text())
    assert plan['reset_optimizer'] is True and 'matched_wider_arm' not in plan
    plan.update(output=original['output'], require_gpu=False,
                campaign_deadline_utc='2099-01-01T00:00:00+00:00')
    monkeypatch.setattr(campaign, 'numerical_runtime_fingerprint',
                        lambda: {'node': plan['runtime_contract']['node']})
    plan_path = tmp_path/'reset_plan.json'
    atomic_json(plan_path, plan)
    seal = canonical_sha256(plan); out = Path(plan['output'])
    assert campaign.run(plan, 'pilot', seal)['status'] == 'passed'
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert cp['metadata']['completed_steps'] == 2
    assert cp['metadata']['absolute_completed_steps'] == 202
    assert cp['metadata']['optimizer_base_step'] == 0
    assert cp['metadata']['optimizer_policy'] == 'explicit_reset_for_empirical_FS_direct_SR'
    assert int(cp['optimizer_state'][0]) == 2
    assert all(np.count_nonzero(leaf) == 0 for leaf in jax.tree.leaves(cp['optimizer_state'][1:]))
    initialization = json.loads((out/'optimizer_reset_initialization.json').read_text())
    assert initialization['same_parent_parameters_bitwise']
    assert initialization['reset_Adam_counter_and_moments']
    assert initialization['training_and_calibration_chains_and_RNG_inherited']
    assert initialization['optimization_control'] == 'matched_empirical_FS_direct_SR'
    assert 'matched_wider_arm' not in initialization
    assert not (out/'capacity_control_initialization.json').exists()
    assert calls == [1, 2, 3]
    before = file_sha256(out/'checkpoint.pkl')

    # A fresh interpreter rebuilds the deterministic numerical mock but uses
    # the real saved checkpoint, registration, fixture and replay machinery.
    child = '''
import json
from pathlib import Path
import runpy
import sys
import pytest
namespace = runpy.run_path(sys.argv[1])
campaign = namespace['campaign']
patch = pytest.MonkeyPatch()
namespace['_mock_campaign'](patch, Path(sys.argv[2]), nonzero_parent_moments=True, full_H_calls=4096)
plan = json.loads(Path(sys.argv[3]).read_text())
patch.setattr(campaign, 'numerical_runtime_fingerprint',
              lambda: {'node': plan['runtime_contract']['node']})
result = campaign.run(plan, 'replay', namespace['canonical_sha256'](plan))
print(json.dumps(result))
'''
    environment = dict(os.environ, JAX_PLATFORM_NAME='cpu', JAX_ENABLE_X64='True',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTHONPATH=str(CONFIG.parents[5]))
    replay = subprocess.run([sys.executable, '-c', child, __file__, str(tmp_path), str(plan_path)],
        cwd=CONFIG.parents[5], env=environment, capture_output=True, text=True, check=True, timeout=90)
    assert json.loads(replay.stdout.splitlines()[-1])['status'] == 'replay_passed'
    assert file_sha256(out/'checkpoint.pkl') == before
    replay_report = json.loads((out/'process_replay.json').read_text())
    assert replay_report['optimizer_updates_committed'] == 0
    assert replay_report['report']['passed']
    assert max(row['relative_vector_error'] or 0. for row in replay_report['report']['leaves']) == 0.

    result = campaign.run(plan, 'train', seal)
    assert result['completed_steps'] == 16 and result['absolute_completed_steps'] == 216
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert int(cp['optimizer_state'][0]) == 16
    assert len(cp['metadata']['history']) == 16
    np.testing.assert_array_equal(cp['metadata']['training_key'], jax.random.PRNGKey(1)+16)
    np.testing.assert_array_equal(cp['metadata']['calibration_key'], jax.random.PRNGKey(2)+16)
    assert result['full_H_training_centers'] == 16*4096
    assert result['terminal_audit']['independent_from_training_and_calibration']
    committed = file_sha256(out/'checkpoint.pkl'); count = len(calls)
    assert campaign.run(plan, 'train', seal)['status'] == 'fixed_steps_complete'
    assert file_sha256(out/'checkpoint.pkl') == committed and len(calls) == count


def test_generic_reset_provenance_cannot_require_legacy_capacity_arm():
    source = Path(campaign.__file__).read_text()
    generic = source.split("elif reset_optimizer:", 1)[1].split('if widening:', 1)[0]
    assert "output/'optimizer_reset_initialization.json'" in generic
    assert "plan['matched_wider_arm']" not in generic
    legacy = source.split('if reset_for_capacity_control:', 1)[1].split('elif reset_optimizer:', 1)[0]
    assert "output/'capacity_control_initialization.json'" in legacy
    assert "matched_wider_arm=plan['matched_wider_arm']" in legacy


def _mock_wide_campaign(monkeypatch, tmp_path):
    original, calls = _mock_campaign(monkeypatch, tmp_path,
        nonzero_parent_moments=True, full_H_calls=4096)
    plan = wide_production_plan()
    plan.update(output=original['output'], require_gpu=False)
    monkeypatch.setattr(campaign, 'numerical_runtime_fingerprint',
                        lambda: {'node': plan['runtime_contract']['node']})
    # Small numerical fixture exercises widening, real state serialization,
    # optimizer reset, counter, RNG, and replay dispatch without N36 training.
    monkeypatch.setattr(campaign, 'initialize_live_triangular_expansion',
        lambda old, p, new, key, **kwargs: (dict(x=jnp.concatenate((p['x'], jnp.zeros(2)))),
                                           dict(exact_same_physical_state=True)))
    monkeypatch.setattr(campaign, 'expansion_preflight',
                        lambda *args: dict(status='passed', full_H_centers=4))
    return plan, calls


def test_wide_fresh_process_replay_then126_remaining_updates(monkeypatch, tmp_path):
    plan, calls = _mock_wide_campaign(monkeypatch, tmp_path)
    seal = canonical_sha256(plan); out = Path(plan['output'])
    plan_path = tmp_path/'wide_plan.json'; atomic_json(plan_path, plan)
    campaign.run(plan, 'pilot', seal)
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert cp['metadata']['completed_steps'] == 2 and int(cp['optimizer_state'][0]) == 2
    assert cp['metadata']['optimizer_policy'] == 'reset_for_new_dimensions'
    assert cp['params']['x'].shape == (4,) and calls == [1, 2, 3]
    assert all(np.count_nonzero(x) == 0 for x in jax.tree.leaves(cp['optimizer_state'][1:]))
    before = file_sha256(out/'checkpoint.pkl')
    child = '''
import json
from pathlib import Path
import runpy
import sys
import pytest
namespace = runpy.run_path(sys.argv[1])
campaign = namespace['campaign']
patch = pytest.MonkeyPatch()
namespace['_mock_wide_campaign'](patch, Path(sys.argv[2]))
plan = json.loads(Path(sys.argv[3]).read_text())
print(json.dumps(campaign.run(plan, 'replay', namespace['canonical_sha256'](plan))))
'''
    environment = dict(os.environ, JAX_PLATFORM_NAME='cpu', JAX_ENABLE_X64='True',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH=str(CONFIG.parents[5]))
    replay = subprocess.run([sys.executable, '-c', child, __file__, str(tmp_path), str(plan_path)],
        cwd=CONFIG.parents[5], env=environment, capture_output=True, text=True, check=True, timeout=90)
    assert json.loads(replay.stdout.splitlines()[-1])['status'] == 'replay_passed'
    assert file_sha256(out/'checkpoint.pkl') == before
    result = campaign.run(plan, 'train', seal)
    assert result['completed_steps'] == 128 and result['absolute_completed_steps'] == 328
    assert result['full_H_training_centers'] == 128*4096
    assert result['fixture_and_replay_H_centers'] == 8192
    assert result['terminal_audit_H_centers'] == 65536 and result['preflight_H_centers'] == 8
    assert len(calls) == 129  #128committed +fixture; replay ran in another process.
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert cp['optimizer_state'][0] == 128 and len(cp['metadata']['history']) == 128
    np.testing.assert_array_equal(cp['metadata']['training_key'], jax.random.PRNGKey(1)+128)
    np.testing.assert_array_equal(cp['metadata']['calibration_key'], jax.random.PRNGKey(2)+128)
    report = json.loads((out/'process_replay.json').read_text())['report']
    assert max(row['relative_vector_error'] or 0. for row in report['leaves']) == 0.


def test_wide_time_pause_keeps_committed_checkpoint_and_never_audits(monkeypatch, tmp_path):
    plan, calls = _mock_wide_campaign(monkeypatch, tmp_path)
    seal = canonical_sha256(plan); out = Path(plan['output'])
    campaign.run(plan, 'pilot', seal); campaign.run(plan, 'replay', seal)
    before = file_sha256(out/'checkpoint.pkl'); old_calls = len(calls)
    monkeypatch.setattr(campaign, 'wide_production_time_guard', lambda *args, **kwargs:
        dict(can_start=False, available_seconds=10., reserved_seconds=120.))
    result = campaign.run(plan, 'train', seal)
    assert result['status'] == 'safe_pause' and result['completed_steps'] == 2
    assert file_sha256(out/'checkpoint.pkl') == before and len(calls) == old_calls
    assert not (out/'terminal_audit.json').exists() and not (out/'results.json').exists()
    assert json.loads((out/'progress.json').read_text())['status'] == 'safe_pause'


def test_quality_capacity_reset_keeps_original_initialization_contract(monkeypatch, tmp_path):
    original, _ = _mock_campaign(monkeypatch, tmp_path, nonzero_parent_moments=True)
    plan = quality_plan(20)
    plan.update(output=original['output'], require_gpu=False)
    monkeypatch.setattr(campaign, 'numerical_runtime_fingerprint',
                        lambda: {'node': plan['runtime_contract']['node']})
    assert campaign.run(plan, 'pilot', canonical_sha256(plan))['status'] == 'passed'
    out = Path(plan['output'])
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert cp['metadata']['optimizer_policy'] == 'explicit_reset_matching_wider_capacity_arm'
    assert cp['metadata']['optimizer_base_step'] == 0
    assert int(cp['optimizer_state'][0]) == 2
    initial = json.loads((out/'capacity_control_initialization.json').read_text())
    assert initial['matched_wider_arm'] == plan['matched_wider_arm']
    assert initial['same_parent_parameters_bitwise']
    assert not (out/'optimizer_reset_initialization.json').exists()


def test_rejected_update_keeps_previous_checkpoint_and_cannot_pass_pilot(monkeypatch, tmp_path):
    plan, calls = _mock_campaign(monkeypatch, tmp_path, reject_at=202)
    with pytest.raises(RuntimeError, match='rejected'):
        campaign.run(plan, 'pilot', canonical_sha256(plan))
    out = Path(plan['output']); cp = load_checkpoint(out/'checkpoint.pkl')
    assert cp['metadata']['completed_steps'] == 1 and cp['optimizer_state'][0] == 201
    assert not (out/'pilot.json').exists()
    assert json.loads((out/'rejected_step.json').read_text())['accepted'] is False


def test_unaccepted_fixture_does_not_release_training(monkeypatch, tmp_path):
    plan, calls = _mock_campaign(monkeypatch, tmp_path, reject_at=203)
    with pytest.raises(RuntimeError, match='pilot'):
        campaign.run(plan, 'pilot', canonical_sha256(plan))
    out = Path(plan['output'])
    assert json.loads((out/'pilot.json').read_text())['status'] == 'failed'
    assert load_checkpoint(out/'checkpoint.pkl')['metadata']['completed_steps'] == 2
    with pytest.raises(ValueError, match='pilot'):
        campaign.run(plan, 'replay', canonical_sha256(plan))


def test_training_cannot_run_before_independent_replay(monkeypatch, tmp_path):
    plan, _ = _mock_campaign(monkeypatch, tmp_path); seal = canonical_sha256(plan)
    campaign.run(plan, 'pilot', seal)
    with pytest.raises(FileNotFoundError):
        campaign.run(plan, 'train', seal)


@pytest.mark.parametrize('change', ['empty_report', 'loose_tolerance', 'wrong_fixture', 'wrong_preflight', 'wrong_reg', 'bad_leaf'])
def test_passed_replay_string_cannot_release_unsealed_or_inaccurate_training(monkeypatch, tmp_path, change):
    plan, _ = _mock_campaign(monkeypatch, tmp_path); seal = canonical_sha256(plan); out = Path(plan['output'])
    campaign.run(plan, 'pilot', seal); campaign.run(plan, 'replay', seal)
    path = out/'process_replay.json'; replay = json.loads(path.read_text())
    if change == 'empty_report': replay['report']['leaves'] = []
    if change == 'loose_tolerance': replay['report']['tolerance'] = 1e-5
    if change == 'wrong_fixture': replay['fixture_sha256'] = 'wrong'
    if change == 'wrong_preflight': replay['preflight_sha256'] = 'wrong'
    if change == 'wrong_reg': replay['registration_sha256'] = 'wrong'
    if change == 'bad_leaf':
        row = next(x for x in replay['report']['leaves'] if not x['discrete'] and not x['natural_unit_scalar_diagnostic'])
        row['relative_vector_error'] = 1e-4  # Deliberately leave the passed bit true.
    path.write_text(json.dumps(replay))
    with pytest.raises(ValueError): campaign.run(plan, 'train', seal)


def _future_parent(tmp_path, *, kind='continuation'):
    plan = json.loads((CONFIG/'j0_binary_truncated_w18.json').read_text())
    directory = tmp_path/'parent'; directory.mkdir()
    req = plan['parent']; req['directory'] = str(directory)
    source_plan = json.loads(Path(req['plan_path']).read_text())
    source_file = tmp_path/'source_plan.json'; source_file.write_text(json.dumps(source_plan))
    req['plan_path'] = str(source_file)
    identity = dict(plan_sha256=canonical_sha256(source_plan), **req['expected_identity'])
    reg = canonical_sha256(identity)
    (directory/'registration.json').write_text(json.dumps(dict(identity, sha256=reg)))
    p = dict(x=jnp.asarray([.1, .2], jnp.float64)); _, a, b = _manual_adam_init(p)
    state = (p, (232, a, b), jax.random.PRNGKey(1), jnp.zeros((64, 36), jnp.int32),
             jax.random.PRNGKey(2), jnp.ones((64, 36), jnp.int32))
    meta = dict(registration_sha256=reg, completed_steps=32, absolute_completed_steps=232,
        optimizer_base_step=200, history=[dict(step=k+1) for k in range(32)])
    campaign.shared.save_state(directory/'checkpoint.pkl', state, meta)
    audit = dict(params_sha256=pytree_sha256(p), independent_from_training_and_calibration=True)
    (directory/'terminal_audit.json').write_text(json.dumps(audit))
    result = dict(status='fixed_steps_complete', registration_sha256=reg,
        checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'), params_sha256=pytree_sha256(p),
        completed_steps=32, absolute_completed_steps=232, terminal_audit=audit)
    (directory/'results.json').write_text(json.dumps(result))
    return plan, directory


def test_future_parent_requires_actual_fixed_result_audit_optimizer_and_rng_seals(tmp_path):
    plan, directory = _future_parent(tmp_path)
    state, lineage = campaign.load_parent(plan)
    assert int(state[1][0]) == 232 and lineage['absolute_completed_steps'] == 232
    assert lineage['checkpoint_sha256'] == file_sha256(directory/'checkpoint.pkl')
    # Declaring an expected identity never bypasses the actual endpoint seal.
    result_path = directory/'results.json'; result = json.loads(result_path.read_text())
    result['checkpoint_sha256'] = 'wrong'; result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='checkpoint'):
        campaign.load_parent(plan)


def test_future_parent_cannot_change_H_case_previous_plan_or_absolute_endpoint(tmp_path):
    plan, directory = _future_parent(tmp_path)
    for mutate in ('case', 'j2', 'endpoint', 'plan'):
        changed = copy.deepcopy(plan)
        if mutate == 'case': changed['case'] = 'another_case'
        if mutate == 'j2': changed['j2_over_j1'] = .125
        if mutate == 'endpoint': changed['parent']['absolute_completed_steps'] = 231
        if mutate == 'plan':
            changed_file = tmp_path/'changed.json'; changed_file.write_text('{}')
            changed['parent']['plan_path'] = str(changed_file)
        with pytest.raises(ValueError): campaign.load_parent(changed)
def test_work_counts_follow_actual_records_and_registered_audit_not_legacy_constants():
    from projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation import continuation_work_counts
    for samples, audit in ((2048,16384), (4096,65536)):
        plan=dict(training=dict(samples=samples),audit=dict(samples=audit))
        counts=continuation_work_counts(plan,[dict(full_H_calls=samples)]*3)
        assert counts==dict(full_H_training_centers=3*samples,
            fixture_and_replay_H_centers=2*samples,
            terminal_audit_H_centers=audit,calibration_H_centers=0)
