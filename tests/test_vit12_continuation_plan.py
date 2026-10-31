"""Fixed-model ViT12 continuation contracts; no new numerical architecture."""
import json
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import sealed_json
from su2_nqs.frontier.artifacts import pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import save_checkpoint

BASE = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote/continuation_20261019/vit12'


@pytest.mark.parametrize('case', ['j0', 'j0125'])
def test_existing_model_and_exact_A1_schedule_are_not_redefined(case):
    plan = json.loads((BASE/f'{case}_plan.json').read_text())
    child = plan['child_overrides']
    assert not {'model', 'model_family', 'coordinate_convention', 'spin_inversion_even', 'j2_over_j1'} & set(child)
    assert all(stage['projection'] == 'gamma_a1' for stage in child['stages'])
    assert child['output'] != plan['parent']['directory']
    assert all(0 <= seed < 2**32 for seed in child['audit']['seeds'])


def test_J0_is_explicit_parameter_repair_not_fictitious_exact_continuation():
    plan = json.loads((BASE/'j0_plan.json').read_text())
    s = plan['child_overrides']
    assert plan['mode'] == 'parameter_repair' and plan['parent']['completed_steps'] == 164
    assert s['sampling']['chains'] == 128
    assert s['sampling']['chains']*s['sampling']['snapshots_per_update'] == 4096
    assert s['netket']['initialization'] == 'balanced_spin_half'
    assert s['netket']['expected_device_count'] == 1
    assert s['sr'] == {'damping': .005, 'step_size': .00125}
    assert sum(stage['steps'] for stage in s['stages']) == 32
    assert s['quality_milestones'] == [0, 2, 4, 16, 32]


def test_J0125_keeps_full_state_and_exact_existing_physical_protocol():
    plan = json.loads((BASE/'j0125_plan.json').read_text())
    s = plan['child_overrides']
    assert plan['mode'] == 'full_state_continuation'
    assert plan['parent']['completed_steps'] == 96
    assert plan['continuation'] == {'additional_steps': 64, 'time_budget_extension_seconds': 43200}
    assert not {'netket', 'sampling', 'sr', 'initialization_seed', 'training_seed'} & set(s)
    assert sum(stage['steps'] for stage in s['stages']) == 160
    assert all(step in s['quality_milestones'] for step in [96, 98, 100, 128, 160])


@pytest.mark.parametrize('case', ['j0', 'j0125'])
def test_final_audit_has_two_fresh_streams_and_no_endpoint_selection(case):
    s = json.loads((BASE/f'{case}_plan.json').read_text())['child_overrides']
    audit = s['audit']
    assert len(audit['seeds']) == len(set(audit['seeds'])) == 2
    assert len(audit['seeds'])*audit['chains']*audit['snapshots'] == 8192
    assert audit['chains'] == 32 and audit['snapshots'] == 128


def test_runner_syntax_and_actual_update_restore_checks():
    script = BASE/'run.sbatch'
    subprocess.run(['bash', '-n', str(script)], check=True)
    source = script.read_text()
    assert "row['update_accepted'] and 0 < row['sampled_fs_sine']" in source
    assert "saved['optimizer_state']['step'] == boundary" in source
    assert "p['checkpoint_sha256'] == file_sha256(checkpoint)" in source
    assert "row['samples'] == row['full_H_calls'] == 4096" in source
    assert '--require-endpoint' in source and '--require-sample-partitions' in source
    assert "from su2_nqs.frontier.artifacts import pytree_sha256" in source


@pytest.mark.parametrize('case', ['j0', 'j0125'])
@pytest.mark.parametrize('defect', [None, 'zero_update', 'wrong_parent', 'wrong_chain_count',
                                  'nonfinite_energy', 'unprojected', 'not_sharded', 'variance_tail',
                                  'illegal_support'])
def test_actual_runner_gate_accepts_only_physical_updates_and_correct_restore(tmp_path, monkeypatch, case, defect):
    s = copy.deepcopy(json.loads((BASE/f'{case}_plan.json').read_text())['child_overrides'])
    s['output'] = str(tmp_path/'state')
    if case == 'j0125':
        s['sampling'] = dict(chains=64, snapshots_per_update=64)
        s['netket'] = dict(sampled_trust=dict(maximum_fs_sine=.025, minimum_ess_fraction=.95))
    start = 0 if case == 'j0' else 96
    devices = 1 if case == 'j0' else 2
    boundary = start+4
    rows = [dict(step=i+1, update_accepted=True, sampled_fs_sine=.002,
                 reweighting_ess_fraction=.999, samples=4096, full_H_calls=4096,
                 device_count=devices, unique_local_sample_partitions=devices,
                 projection='gamma_a1', images=216, gpu_peak_bytes=8*1024**3,
                 energy_before_update_per_site=-.5, variance_before_update_per_site=.02)
            for i in range(boundary)]
    p = dict(completed_steps=boundary, history=rows, registration_sha256='registration',
             final_params_sha256=pytree_sha256({'w': np.array([1.])}),
             training_full_H_calls=boundary*4096,
             initial_params_sha256='6b70599252dd7a08aed1c41accf6113015e31e5a275c52fe7422496534bb5b4f',
             parent=dict(completed_steps=164, repair_is_new_experiment=True,
                         sampler_reinitialized=True, optimizer_and_work_counters_reset=True),
             continuation_parent=dict(completed_steps=96, physics_and_SR_unchanged=True,
                                      parameters_reinitialized=False, sampler_reinitialized=False,
                                      inherited_work_and_time_counters=True))
    if defect == 'zero_update': rows[-1]['sampled_fs_sine'] = 0.
    if defect == 'wrong_parent':
        p['parent']['completed_steps'] = 228
        p['continuation_parent']['completed_steps'] = 32
    if defect == 'nonfinite_energy': rows[-1]['energy_before_update_per_site'] = float('nan')
    if defect == 'unprojected': rows[-1]['projection'] = 'none'
    if defect == 'not_sharded': rows[-1]['unique_local_sample_partitions'] = 0
    if defect == 'variance_tail': rows[-1]['variance_before_update_per_site'] = .295
    chains = s['sampling']['chains']-(defect == 'wrong_chain_count')
    output = Path(s['output']); output.mkdir()
    checkpoint = output/'checkpoint.pkl'
    spins = np.tile(np.array([1, -1]), (chains, 72))
    if defect == 'illegal_support': spins[0, 0] = 0
    save_checkpoint(checkpoint,
                    dict(model={'w': np.array([1.])},
                         netket_sampler_state=SimpleNamespace(σ=spins)),
                    optimizer_state={'step': boundary},
                    metadata={'registration_sha256': 'registration', 'completed_steps': boundary})
    p['checkpoint_sha256'] = file_sha256(checkpoint)
    sealed_json(output/'progress.json', p)
    config = tmp_path/'config.json'; config.write_text(json.dumps(s))
    for name, value in dict(NQS_CONFIG=str(config), NQS_CASE=case, NQS_BOUNDARY=str(boundary),
                            NQS_DEVICES=str(devices), NQS_PARENT_ENERGY='-.5').items():
        monkeypatch.setenv(name, value)
    source = (BASE/'run.sbatch').read_text().split('    "$NQS_PY" - <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
    if defect is None:
        exec(compile(source, str(BASE/'run.sbatch'), 'exec'), {})
    else:
        with pytest.raises(AssertionError):
            exec(compile(source, str(BASE/'run.sbatch'), 'exec'), {})


def test_transitive_manifest_contains_both_numerical_and_registration_imports():
    manifest = json.loads((BASE/'source_required.json').read_text())['source_files']
    for path in ['projects/clebsch_tree_nqs/workflows/spin_spatial_vit_netket.py',
                 'projects/clebsch_tree_nqs/workflows/spin_spatial_vit_quality.py',
                 'projects/clebsch_tree_nqs/workflows/spin_spatial_vit_training_plan.py',
                 'projects/clebsch_tree_nqs/workflows/spin.py',
                 'projects/clebsch_tree_nqs/workflows/spin_full_vit_pilot.py',
                 'su2_nqs/models/literature_spatial_vit.py',
                 'su2_nqs/benchmarks/spatial_vit_sampling.py']:
        assert path in manifest
