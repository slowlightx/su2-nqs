"""Repair scope, unchanged backbones, fresh budgets and immutable inputs."""
import copy
import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows.kim_repair_campaign import (
    CASES, ROOT, nqs_spec, copy_verified, write_once, live_slurm_job, slurm_dependency_options,
)


def template():
    return json.loads((ROOT/'projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_tri_j005.json').read_text())


def test_live_controller_and_afterok_cannot_be_replaced_by_stale_accounting():
    queries = []
    def query(command):
        queries.append(command)
        return 'JobId=206901 JobName=train UserId=yx789(1) JobState=RUNNING ExitCode=0:0 WorkDir=/home/yx789/producer\n'
    state = live_slurm_job(206901, query=query)
    assert state['JobState'] == 'RUNNING'
    assert queries == [['/usr/local/slurm/current/bin/scontrol', 'show', 'job', '-o', '206901']]
    assert slurm_dependency_options([206900, 206900], afterany=[206901]) == [
        '--dependency=afterok:206900,afterany:206901', '--kill-on-invalid-dep=yes']
    assert slurm_dependency_options([206900]) == [
        '--dependency=afterok:206900', '--kill-on-invalid-dep=yes']
    with pytest.raises(ValueError, match='identity'):
        live_slurm_job(206904, query=query)
    with pytest.raises(ValueError, match='unambiguous'):
        live_slurm_job(206901, query=lambda _: '')
    with pytest.raises(ValueError, match='integer'):
        slurm_dependency_options([True])


def test_repair_cases_and_full_backbone_budget_are_explicit():
    assert {(g, j) for _, g, j in CASES} == {
        ('triangular', 0.), ('triangular', .05), ('triangular', .1),
        ('triangular', .175), ('square', 0.), ('square', .5)}
    parent = template()
    parent['j2_over_j1'] = .125
    old = copy.deepcopy(parent)
    descriptor = dict(directory='/home/yx789/nqs-wavefunctions/inputs/p', completed_steps=7600)
    seeds = set()
    for index, (name, geometry, j2) in enumerate(CASES[:4]):
        spec = nqs_spec(parent, descriptor, name=name, geometry=geometry, j2=j2, index=index)
        assert spec['model'] == old['model']
        assert spec['stages'] == [dict(projection='gamma_a1', steps=1000)]
        assert 20 in spec['quality_milestones']
        assert spec['sampling']['chains']*spec['sampling']['snapshots_per_update'] == 4096
        assert spec['maximum_training_seconds'] == 45000
        assert Path(spec['output']).is_relative_to('/home/yx789')
        assert spec['netket']['sampled_trust']['maximum_fs_sine'] == .02
        assert not (seeds & set(spec['audit']['seeds']))
        seeds.update(spec['audit']['seeds'])
    assert parent == old
    with pytest.raises(ValueError, match='authorized'):
        nqs_spec(parent, descriptor, name='square_j08', geometry='square', j2=.8, index=9)


def test_square_repair_is_not_old_protocol_or_optimizer_replay():
    parent = template()
    parent['model']['geometry'] = 'square'
    parent['j2_over_j1'] = .5
    descriptor = dict(directory='/home/yx789/nqs-wavefunctions/inputs/sq', completed_steps=5500)
    spec = nqs_spec(parent, descriptor, name='square_j05', geometry='square', j2=.5, index=5)
    assert spec['repair_parameters_from'] == descriptor
    assert 'coupling_warm_start_from' not in spec
    assert 'netket_continue_from' not in spec
    assert spec['sr']['step_size'] < parent['sr']['step_size']


def test_frustrated_condition_campaign_preserves_backbone_and_declared_factor_controls():
    """One parent, a damping control and a sample-count control; no new head."""
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import validate_spec
    folder = ROOT/'projects/clebsch_tree_nqs/configs/remote/frustrated_quality_20261012'
    plan = json.loads((folder/'campaign.json').read_text())
    assert not plan['field_reoptimization_released']
    assert not plan['head_or_backbone_changes']
    controls = [json.loads((ROOT/e['path']).read_text()) for e in plan['entries'][:3]]
    for spec in controls:
        validate_spec(spec)
        assert spec['stages'] == [dict(projection='gamma_a1', steps=200)]
        assert spec['quality_milestones'] == [0,2,4,100,200]
        assert spec['repair_parameters_from'] == controls[0]['repair_parameters_from']
        assert spec['repair_parameters_from']['completed_steps'] == 10600
        assert spec['model'] == controls[0]['model']
        assert spec['model']['embedding_dim'] == 48 and spec['model']['depth'] == 4
        assert spec['netket'] == controls[0]['netket']
        assert spec['netket']['sampled_trust']['maximum_fs_sine'] == .005
        assert spec['netket']['expected_device_count'] == 1
        assert spec['training_seed'] == controls[0]['training_seed']
        assert spec['interpretation']['new_head'] is False
        assert spec['interpretation']['strict_su2'] is False
    assert controls[0]['sampling'] == controls[1]['sampling']
    assert controls[0]['sr']['damping'] == .0002
    assert controls[1]['sr']['damping'] == controls[2]['sr']['damping'] == .002
    b, c = copy.deepcopy(controls[1]['sampling']), copy.deepcopy(controls[2]['sampling'])
    assert b.pop('snapshots_per_update') == 32
    assert c.pop('snapshots_per_update') == 64
    assert b == c
    seeds = [seed for spec in controls for seed in spec['audit']['seeds']]
    assert len(set(seeds)) == len(seeds)

    # The oversized explicit Jacobian failed before installing any update.
    # The bounded retry changes its implementation, never the state family,
    # sample law, SR equation or failed output directory.
    retry = json.loads((folder/'tri_j0125_s8192_d2e3_streamed_v2.json').read_text())
    original = controls[2]
    for key in ('model','sr','sampling','audit','repair_parameters_from',
                'initialization_seed','training_seed','stages','quality_milestones'):
        assert retry[key] == original[key]
    a, b = copy.deepcopy(original['netket']), copy.deepcopy(retry['netket'])
    assert a.pop('on_the_fly') is False and b.pop('on_the_fly') is True
    assert a == b and retry['output'] != original['output']
    assert not retry['interpretation']['new_head']


def test_frustrated_continuations_keep_executed_protocol_and_work_counters():
    from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket import validate_continuation_protocol
    folder = ROOT/'projects/clebsch_tree_nqs/configs/remote/frustrated_quality_20261012'
    plan = json.loads((folder/'campaign.json').read_text())
    cached = ROOT/'projects/clebsch_tree_nqs/results/coupling_quality_review_20261012/latest_inputs/nqs'
    for entry in plan['entries'][3:]:
        spec = json.loads((ROOT/entry['path']).read_text())
        declaration = spec['netket_continue_from']
        case = Path(declaration['directory']).parent.name
        reg = json.loads((cached/case/'seed1'/'registration.json').read_text())
        progress = json.loads((cached/case/'seed1'/'progress.json').read_text())
        validate_continuation_protocol(reg['config'], spec)
        assert declaration['progress_sha256'] == progress['result_sha256']
        assert spec['model'] == reg['config']['model']
        assert spec['sampling'] == reg['config']['sampling']
        assert spec['sr'] == reg['config']['sr']
        assert entry['pilot'] == declaration['completed_steps']+2
        assert entry['resume'] == declaration['completed_steps']+4
        assert entry['terminal'] == declaration['completed_steps']+declaration['additional_steps']
        assert spec['maximum_training_seconds'] == (
            reg['config']['maximum_training_seconds']+declaration['time_budget_extension_seconds'])
        if progress['status'] == 'paused_session_budget':
            assert declaration['from_registered_milestone']
            assert declaration['audit_sha256']
        assert not set(spec['audit']['seeds']) & set(reg['config']['audit']['seeds'])


def test_cached_readout_cap_repair_changes_no_physical_protocol():
    folder = ROOT/'projects/clebsch_tree_nqs/configs/remote/frustrated_quality_20261012'
    repaired = sorted(folder.glob('*_regional_cap1800_v2.json'))
    assert len(repaired) == 10
    for path in repaired:
        new = json.loads(path.read_text())
        old = json.loads((ROOT/new['execution_repair']['replaces_plan']).read_text())
        assert new['compute_seconds'] == 1800 and old['compute_seconds'] == 3600
        assert not new['execution_repair']['scientific_protocol_changed']
        assert new['output'] == old['output']+'_cap1800_v2'
        assert new['materialized_config'] == old['materialized_config'].replace(
            '.json', '_cap1800_v2.json')
        for key in old.keys()-{'compute_seconds','output','materialized_config'}:
            assert new[key] == old[key]


def test_verified_copy_preserves_source_and_rejects_conflicting_outputs(tmp_path):
    source = tmp_path/'old.pkl'
    source.write_bytes(b'immutable fake test boundary')
    target = tmp_path/'home'/'parent.pkl'
    copy_verified(source, target)
    copy_verified(source, target)
    assert source.read_bytes() == target.read_bytes()
    target.write_bytes(b'conflicting state')
    with pytest.raises(FileExistsError):
        copy_verified(source, target)
    path = tmp_path/'plan.json'
    write_once(path, dict(step=20))
    write_once(path, dict(step=20))
    with pytest.raises(FileExistsError):
        write_once(path, dict(step=21))


@pytest.mark.parametrize('step,status,expected', [
    (2, 'paused_session_budget', 'yes'),
    (3, 'paused_session_budget', 'no'),
    (3, 'fixed_steps_complete', 'no'),
    (2, 'running', None),
    (2, 'total_time_cap', None),
])
def test_session_completion_guard_never_repeats_or_hides_failed_work(
        tmp_path, monkeypatch, capsys, step, status, expected):
    """Execute the actual batch entry guard on independently saved test data."""
    import numpy as np
    from projects.clebsch_tree_nqs.workflows import spin_spatial_vit_quality as quality
    from su2_nqs.vmc.checkpoint import save_checkpoint
    from su2_nqs.frontier.artifacts import pytree_sha256
    from su2_nqs.frontier.provenance import file_sha256
    spec = dict(output=str(tmp_path), quality_milestones=[0,3])
    protocol = tmp_path/'config.json'; protocol.write_text(json.dumps(spec))
    params = dict(weight=np.array([.1,.2]))
    checkpoint = tmp_path/'checkpoint.pkl'
    save_checkpoint(checkpoint, dict(model=params), optimizer_state=dict(step=step),
        metadata=dict(completed_steps=step, registration_sha256='test-registration',
                      history=[dict(step=i+1) for i in range(step)]))
    quality.sealed_json(tmp_path/'progress.json', dict(completed_steps=step,
        status=status, registration_sha256='test-registration',
        checkpoint_sha256=file_sha256(checkpoint), final_params_sha256=pytree_sha256(params)))
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(quality, 'register', lambda *_: (spec,dict(result_sha256='test-registration')))
    monkeypatch.setenv('NQS_UNTIL_STEP','3')
    monkeypatch.setattr(__import__('sys'), 'argv', ['guard',str(protocol)])
    wrapper = ROOT/'projects/clebsch_tree_nqs/remote/spatial_vit_recovery_a100.sbatch'
    code = wrapper.read_text().split('task_resume_needed=')[1].split("<<'PY'\n",1)[1].split('\nPY',1)[0]
    if expected is None:
        with pytest.raises(ValueError, match='safe session pause'):
            exec(compile(code,str(wrapper),'exec'), {})
    else:
        exec(compile(code,str(wrapper),'exec'), {})
        assert capsys.readouterr().out.strip() == expected
    assert before == {p.name:p.read_bytes() for p in tmp_path.iterdir()}
