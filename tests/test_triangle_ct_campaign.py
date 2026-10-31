"""Bounded audit-gated training orchestration, never a local N36 training."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as train
from projects.clebsch_tree_nqs.workflows import triangle_ct_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]


def optimized_plan():
    plan = json.loads((ROOT/'projects/clebsch_tree_nqs/configs/remote/'
        'ct6_gcnn_quality_v2_20261021/w20.json').read_text())
    plan.update(task=train.OPTIMIZED_TASK, reset_optimizer=True,
        optimization_control='matched_empirical_FS_direct_SR',
        campaign_deadline_utc='2099-01-01T00:00:00+00:00')
    plan['calibration'].update(target_fs_sine=.001, relative_tolerance=.02, max_factor=2.,
        bracket_steps=8, bisection_steps=12, minimum_ess_fraction=.98)
    plan['training']['steps'] = 16
    plan['audit']['samples'] = 32768
    plan['budget'].update(pilot_seconds=10800, replay_seconds=3600, train_seconds=28800)
    return plan


@pytest.mark.parametrize('damping', [.05, .5])
def test_optimized_profile_keeps_exact_physical_model_and_unweakened_guards(damping):
    plan = optimized_plan(); plan['damping'] = damping
    train.validate_plan(plan)
    assert plan['maximum_fs_sine'] == .005 and plan['minimum_ess_fraction'] == .98
    assert plan['model']['multiplicity_cap'] == plan['parent']['model']['multiplicity_cap']


def test_registered_four_screens_are_same_parent_same_width_and_within_240h(monkeypatch):
    folder=ROOT/'projects/clebsch_tree_nqs/configs/remote/ct6_sixday_20261022'
    spec=json.loads((folder/'campaign.json').read_text())
    monkeypatch.setattr(campaign, 'ROOT', Path(spec['workspace']))
    campaign.validate_campaign(spec)
    for coupling in ('j0', 'j0125'):
        plans=[json.loads((folder/f'{coupling}_{name}.json').read_text())
               for name in ('ridge005', 'ridge050')]
        for plan in plans: train.validate_plan(plan)
        assert plans[0]['parent']==plans[1]['parent']
        assert plans[0]['training']==plans[1]['training']
        assert plans[0]['calibration']==plans[1]['calibration']
        assert plans[0]['seeds']['audit']!=plans[1]['seeds']['audit']
        assert plans[0]['seeds']['calibration']==plans[1]['seeds']['calibration']
        assert {p['damping'] for p in plans}=={.05,.5}
    spec['production_blocks']=4
    with pytest.raises(ValueError): campaign.validate_campaign(spec)


@pytest.mark.parametrize('fault', ['width', 'damping', 'fit_radius', 'calibration',
                                 'optimizer', 'samples', 'audit', 'deadline', 'steps', 'budget'])
def test_optimized_profile_refuses_scope_or_resource_changes(fault):
    plan = optimized_plan()
    if fault == 'width': plan['model']['multiplicity_cap'] = 24
    if fault == 'damping': plan['damping'] = .005
    if fault == 'fit_radius': plan['calibration']['target_fs_sine'] = .01
    if fault == 'calibration': plan['calibration']['chains'] = 128
    if fault == 'optimizer': plan['reset_optimizer'] = 'yes'
    if fault == 'samples': plan['training']['samples'] = 8192
    if fault == 'audit': plan['audit']['samples'] = 4096
    if fault == 'deadline': plan['campaign_deadline_utc'] = '2099-01-01'
    if fault == 'steps': plan['training']['steps'] = 65
    if fault == 'budget': plan['budget']['train_seconds'] = 36*3600
    with pytest.raises(ValueError): train.validate_plan(plan)


def test_next_block_is_same_width_full_state_continuation_with_fresh_final_audit():
    old = optimized_plan()
    spec = dict(production_blocks=3, production_steps=64, output='/remote/output')
    endpoint = dict(plan=old, plan_path='/remote/config/screen.json',
        absolute_completed_steps=332, sealed_files={name:'f'*64 for name in campaign.SEALED_FILES})
    first = campaign.next_plan(endpoint, spec, 'j0125', 1)
    last = campaign.next_plan(endpoint, spec, 'j0125', 3)
    assert first['model'] == last['model'] == old['model']
    assert first['reset_optimizer'] is False and first['training']['steps'] == 64
    assert first['parent']['sealed_files'] == endpoint['sealed_files']
    assert first['audit']['samples'] == 65536 and last['audit']['samples'] == 262144
    assert first['seeds']['audit'] != last['seeds']['audit'] != old['seeds']['audit']
    with pytest.raises(ValueError): campaign.next_plan(endpoint, spec, 'j0125', 4)


def audit(e=-.5149, se=.00004, var=.005):
    return dict(energy_per_site=e, energy_chain_SE=se, variance_per_site=var)


def test_release_allows_quality_progress_not_mechanical_extension_or_significance_claim():
    assert campaign.release_evidence(audit(e=-.5151), audit())['released']
    evidence = campaign.release_evidence(audit(var=.004), audit())
    assert evidence['released'] and evidence['reason'] == 'variance_with_compatible_energy'
    assert evidence['significance_or_variance_error_test'] is False
    assert not campaign.release_evidence(audit(e=-.5147, var=.004), audit())['released']
    assert not campaign.release_evidence(audit(var=.006), audit())['released']
    assert not campaign.release_evidence(audit(), audit())['released']
    with pytest.raises(ValueError): campaign.release_evidence(audit(se=-1), audit())


def test_screen_reports_every_arm_and_does_not_choose_a_failed_low_training_mean():
    rows = [dict(case='broken', status='failed_or_incomplete', training_mean=-1.),
            dict(case='control', status='verified', audit=audit()),
            dict(case='regularized', status='verified', audit=audit(var=.004))]
    chosen, records = campaign.choose_screen(rows, audit())
    assert chosen['case'] == 'regularized' and len(records) == 3
    assert not next(r for r in records if r['case']=='control')['released']
    assert campaign.choose_screen(rows[:2], audit())[0] is None


def test_submission_reuses_receipt_and_refuses_ambiguous_or_late_work(monkeypatch, tmp_path):
    spec = dict(deadline_utc='2099-01-01T00:00:00+00:00', output=str(tmp_path),
        campaign_path='/remote/campaign.json', runner='/remote/run.sbatch', node='kim-compute-05')
    calls = []
    def submit(args, **kwargs):
        calls.append(args)
        return type('Process', (), dict(returncode=0, stdout='12345\n', stderr=''))()
    monkeypatch.setattr(campaign.subprocess, 'run', submit)
    job = campaign._submit(spec, 'j0/pilot', phase='pilot', minutes=10)
    assert job == campaign._submit(spec, 'j0/pilot', phase='pilot', minutes=10) == '12345'
    assert len(calls) == 1 and '--gres=gpu:nvidia_a100_80gb_pcie:1' in calls[0]
    (tmp_path/'submissions'/'j0__ambiguous.json').write_text('{"status":"submitting"}')
    with pytest.raises(RuntimeError, match='ambiguous'):
        campaign._submit(spec, 'j0/ambiguous', phase='pilot', minutes=10)
    spec['deadline_utc'] = datetime.now(timezone.utc).isoformat()
    with pytest.raises(RuntimeError, match='deadline'):
        campaign._submit(spec, 'j0/late', phase='pilot', minutes=10)


def test_parent_seal_phase_is_CPU_only_and_charges_no_GPU(monkeypatch,tmp_path):
    spec=dict(deadline_utc='2099-01-01T00:00:00+00:00',output=str(tmp_path),
        campaign_path='/remote/campaign.json',runner='/remote/run.sbatch',node='kim-compute-05')
    calls=[]
    def submit(args,**kwargs):
        calls.append(args)
        return type('Process',(),dict(returncode=0,stdout='12346\n',stderr=''))()
    monkeypatch.setattr(campaign.subprocess,'run',submit)
    assert campaign._submit(spec,'parents',phase='parents',minutes=10)=='12346'
    assert not any('gres=' in a or 'nodelist=' in a for a in calls[0])
    assert json.loads((tmp_path/'submissions'/'parents.json').read_text())['requested_GPU_hours']==0
