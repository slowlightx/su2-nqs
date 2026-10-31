"""Explicit v8 policy: unresolved slopes may run one fixed audited endpoint."""
import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as campaign
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from test_ct6_direction_continuation import mock_workflow, plan_fixture


def evidence_plan(plan=None):
    plan = plan_fixture() if plan is None else plan
    plan.update(task=campaign.EVIDENCE_GUARDED_TASK, damping=.5,
                optimization_control='frozen_parent_regularized_SR_evidence_guarded_endpoint')
    plan['direction_check']['release_policy'] = 'reject_resolved_ascent_allow_unresolved'
    return plan


def moments_fixture(slope, error):
    return dict(finite=True, energy_slope_per_site=slope,
                energy_slope_chain_jackknife_SE=error, sampled_fs_sine=.001,
                reweighting_ess_fraction=.999,
                parent_energy_reference_check=dict(difference_per_site=0.,
                    combined_independent_chain_SE=.01))


@pytest.mark.parametrize('slope,error,evidence,release', [
    (-3., 1., 'confirmed_descent', True),
    (-2., 1., 'unresolved', True),
    (0., 1., 'unresolved', True),
    (1., 1., 'unresolved', True),
    (2., 1., 'unresolved', True),
    (3., 1., 'resolved_ascent', False),
    (-1., 0., 'confirmed_descent', True),
    (0., 0., 'unresolved', True),
    (1., 0., 'resolved_ascent', False),
])
def test_exact_two_standard_error_classification(slope, error, evidence, release):
    plan = evidence_plan()
    gate = campaign.registered_direction_gate(plan, moments_fixture(slope, error))
    assert gate['slope_lower_bound_per_site'] == slope-2*error
    assert gate['slope_upper_bound_per_site'] == slope+2*error
    assert gate['slope_evidence'] == evidence and gate['passed'] is release
    assert gate['confirmed_descent'] is (evidence == 'confirmed_descent')
    assert gate['endpoint_energy_gain_confirmed'] is False
    assert gate['unresolved_is_not_descent_evidence'] is True


@pytest.mark.parametrize('changes', [dict(finite=False), dict(sampled_fs_sine=.006),
    dict(sampled_fs_sine=0.), dict(reweighting_ess_fraction=.97),
    dict(parent_energy_reference_check=dict(difference_per_site=.1,
         combined_independent_chain_SE=.01))])
def test_unresolved_evidence_cannot_bypass_physical_or_parent_guard(changes):
    gate = campaign.registered_direction_gate(evidence_plan(),
        dict(moments_fixture(0., 1.), **changes))
    assert not gate['passed']


def test_nonfinite_records_are_rejected_before_slope_classification():
    arrays = [np.ones(256) for _ in range(5)]
    arrays[3][0] = np.nan
    moments = campaign.direction_check_moments(*arrays, chains=128)
    gate = campaign.registered_direction_gate(evidence_plan(), moments)
    assert gate['slope_evidence'] == 'nonfinite' and not gate['passed']
    assert gate['slope_lower_bound_per_site'] is None


@pytest.mark.parametrize('slope,error', [(np.nan, 1.), (np.inf, 1.), (-np.inf, 1.),
                                       (0., np.nan), (0., np.inf), (0., -1.)])
def test_public_gate_rejects_invalid_slope_or_error_despite_finite_flag(slope, error):
    gate = campaign.registered_direction_gate(evidence_plan(), moments_fixture(slope, error))
    assert not gate['passed'] and gate['slope_evidence'] == 'nonfinite'


@pytest.mark.parametrize('coupling,hours', [(0., 17), (.125, 19)])
def test_v8_has_explicit_compile_and_audit_headroom_with_phase_caps(coupling, hours):
    plan = evidence_plan()
    plan['j2_over_j1'] = coupling
    plan['arity'] = 'ternary_joint' if coupling == 0 else 'binary_truncated'
    plan['budget'].update(pilot_seconds=4*3600, replay_seconds=2*3600, train_seconds=hours*3600)
    campaign.validate_plan(plan)
    for phase in ('pilot', 'replay', 'train'):
        plan['budget'][phase+'_seconds'] += 1
        with pytest.raises(ValueError, match='v8 phase allocation'):
            campaign.validate_plan(plan)
        plan['budget'][phase+'_seconds'] -= 1


def test_new_ridge_and_policy_cannot_be_inserted_into_old_protocol():
    old = plan_fixture(); campaign.validate_plan(old)
    new = evidence_plan(); campaign.validate_plan(new)
    assert not campaign.registered_direction_gate(old, moments_fixture(0., 1.))['passed']
    assert campaign.registered_direction_gate(new, moments_fixture(0., 1.))['passed']
    for task, damping in ((campaign.DIRECTION_TASK, .5), (campaign.EVIDENCE_GUARDED_TASK, .05)):
        plan = plan_fixture() if task == campaign.DIRECTION_TASK else evidence_plan()
        plan['damping'] = damping
        with pytest.raises(ValueError, match='fixed direct-SR recipe'):
            campaign.validate_plan(plan)
    old['direction_check']['release_policy'] = 'reject_resolved_ascent_allow_unresolved'
    with pytest.raises(ValueError, match='evidence policy'):
        campaign.validate_plan(old)
    del new['direction_check']['release_policy']
    with pytest.raises(ValueError, match='evidence policy'):
        campaign.validate_plan(new)


def unresolved_workflow(monkeypatch, tmp_path):
    plan, parent, calls = mock_workflow(monkeypatch, tmp_path)
    evidence_plan(plan)
    runtime_factory = campaign.runtime
    runtime = runtime_factory(plan, parent[0])
    old_confirmation = runtime['direction_confirmation']

    def confirmation(old, new):
        _, raw = old_confirmation(old, new)
        # Alternate whole-chain contributions, constant in time. The exact
        # mean is zero and deleting any chain gives nonzero uncertainty.
        raw['dhpsi'] = np.tile(np.array([-.002, .002]), 1024)
        moments = campaign.direction_check_moments(*[raw[key] for key in
            ('psi', 'hpsi', 'dpsi', 'dhpsi', 'candidate_psi')], chains=128)
        return moments, raw

    runtime['direction_confirmation'] = confirmation
    monkeypatch.setattr(campaign, 'runtime', lambda *args: runtime)
    return plan, parent, calls


def test_unresolved_pilot_requires_isolated_replay_then32_updates_and_fresh_audit(monkeypatch, tmp_path):
    plan, parent, calls = unresolved_workflow(monkeypatch, tmp_path)
    digest = canonical_sha256(plan); out = Path(plan['output'])
    pilot = campaign.run(plan, 'pilot', digest)
    assert pilot['status'] == 'passed'
    assert pilot['gate']['slope_evidence'] == 'unresolved'
    assert pilot['gate']['confirmed_descent'] is False
    assert pilot['completed_steps'] == pilot['optimizer_updates_committed'] == 0
    before = file_sha256(out/'checkpoint.pkl')
    with pytest.raises(FileNotFoundError):
        campaign.run(plan, 'train', digest)
    assert campaign.run(plan, 'replay', digest)['status'] == 'replay_passed'
    assert file_sha256(out/'checkpoint.pkl') == before
    result = campaign.run(plan, 'train', digest)
    assert result['completed_steps'] == 32 and result['absolute_completed_steps'] == 476
    assert result['initial_slope_evidence'] == 'unresolved'
    assert result['initial_descent_confirmed'] is False
    assert result['terminal_audit_is_quality_decision'] is True
    assert result['terminal_audit']['independent_from_training_and_calibration'] is True
    assert result['terminal_audit']['samples'] == 16384
    assert calls == dict(transition=35, confirmation=2, audit=1, branch=1)
    checkpoint = load_checkpoint(out/'checkpoint.pkl')
    assert checkpoint['metadata']['completed_steps'] == 32


def test_resolved_ascent_remains_terminal_and_never_redraws(monkeypatch, tmp_path):
    plan, _, calls = mock_workflow(monkeypatch, tmp_path, descent=False)
    evidence_plan(plan); digest = canonical_sha256(plan); out = Path(plan['output'])
    pilot = campaign.run(plan, 'pilot', digest)
    assert pilot['status'] == 'failed'
    assert pilot['gate']['slope_evidence'] == 'resolved_ascent'
    before = file_sha256(out/'checkpoint.pkl'); counts = dict(calls)
    assert campaign.run(plan, 'pilot', digest) == pilot
    for phase in ('replay', 'train'):
        with pytest.raises(ValueError, match='direction gate'):
            campaign.run(plan, phase, digest)
    assert counts == calls and file_sha256(out/'checkpoint.pkl') == before


def test_replay_cannot_relabel_unresolved_as_confirmed_descent(monkeypatch, tmp_path):
    plan, _, _ = unresolved_workflow(monkeypatch, tmp_path)
    digest = canonical_sha256(plan); out = Path(plan['output'])
    campaign.run(plan, 'pilot', digest); campaign.run(plan, 'replay', digest)
    replay = json.loads((out/'process_replay.json').read_text())
    replay['independently_recomputed_gate']['slope_evidence'] = 'confirmed_descent'
    replay['independently_recomputed_gate']['confirmed_descent'] = True
    atomic_json(out/'process_replay.json', replay)
    before = file_sha256(out/'checkpoint.pkl')
    with pytest.raises(ValueError, match='independent full transition/direction process replay'):
        campaign.run(plan, 'train', digest)
    assert file_sha256(out/'checkpoint.pkl') == before


def test_old_failed_artifacts_cannot_be_relabelled_as_v8(monkeypatch, tmp_path):
    plan, _, calls = unresolved_workflow(monkeypatch, tmp_path)
    plan.update(task=campaign.DIRECTION_TASK, damping=.05,
                optimization_control='frozen_parent_direction_then_same_width_SR')
    del plan['direction_check']['release_policy']
    pilot = campaign.run(plan, 'pilot', canonical_sha256(plan))
    assert pilot['status'] == 'failed'
    out = Path(plan['output']); sealed = {name: file_sha256(out/name) for name in
        ('registration.json', 'pilot.json', 'direction_check.json', 'direction_check.npz', 'checkpoint.pkl')}
    evidence_plan(plan); counts = dict(calls)
    with pytest.raises(ValueError, match='immutable stage registration changed'):
        campaign.run(plan, 'pilot', canonical_sha256(plan))
    assert calls == counts
    assert all(file_sha256(out/name) == digest for name, digest in sealed.items())
