"""Scheduling has no access to audit outcomes and cannot reset work budgets."""
import copy
import json
from pathlib import Path
import os
import subprocess

import pytest

from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_schedule import materialize_stage, prepare
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import sealed_json
from su2_nqs.frontier.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def endpoint(path, spec, step, status='fixed_steps_complete'):
    path.mkdir(parents=True)
    (path/'quality').mkdir()
    cp = path/'quality'/f'step{step:05d}.pkl'
    cp.write_bytes(b'opaque checkpoint, loading validated by numerical workflow')
    reg = sealed_json(path/'registration.json', dict(config=spec))
    progress = sealed_json(path/'progress.json', dict(completed_steps=step, final_params_sha256='params'))
    sealed_json(path/'training.json', dict(completed_steps=step, final_params_sha256='params',
        registration_sha256=reg['result_sha256'], status=status))
    return dict(directory=str(path), registration_sha256=reg['result_sha256'],
                progress_sha256=progress['result_sha256'], checkpoint_sha256=file_sha256(cp))


def fixture(tmp_path):
    spec = json.loads((ROOT/'projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_netket_continuation_7600_n36.json').read_text())
    spec.pop('netket_continue_from')
    spec['stages'] = [dict(projection='gamma_a1', steps=3)]
    spec['quality_milestones'] = [0, 3]
    parent = endpoint(tmp_path/'parent', spec, 3)
    stages = [dict(output=str(tmp_path/f'stage{i}'), config=str(tmp_path/f'config{i}.json'),
                   additional_steps=2, time_extension_seconds=60,
                   audit_seeds=[1001+10*i,1002+10*i], bootstrap_seed=1003+10*i) for i in range(3)]
    return dict(parent=parent, stages=stages), spec


def test_fixed_schedule_materializes_only_completed_ancestry(tmp_path):
    plan, original = fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        materialize_stage(plan, 1)
    for i in range(3):
        spec = materialize_stage(plan, i)
        assert sum(s['steps'] for s in spec['stages']) == 5+2*i
        assert spec['maximum_training_seconds'] == original['maximum_training_seconds']+60*(i+1)
        assert spec['model'] == original['model']
        assert spec['sr'] == original['sr']
        assert spec['sampling'] == original['sampling']
        endpoint(Path(plan['stages'][i]['output']), spec, 5+2*i)
    # No audit files were created or required.
    bad = copy.deepcopy(plan); bad['stages'][2]['audit_seeds'] = bad['stages'][0]['audit_seeds']
    with pytest.raises(ValueError, match='distinct'):
        materialize_stage(bad, 2)
    bad = copy.deepcopy(plan); bad['parent']['checkpoint_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='checkpoint changed'):
        materialize_stage(bad, 0)


def test_incomplete_parent_and_modified_schedule_fail_closed(tmp_path):
    plan, _ = fixture(tmp_path)
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    digest = file_sha256(path)
    dest = prepare(path, 0, digest)
    assert prepare(path, 0, digest) == dest
    dest.write_text('{}')
    with pytest.raises(FileExistsError):
        prepare(path, 0, digest)
    with pytest.raises(ValueError, match='bytes changed'):
        prepare(path, 0, 'bad')
    parent = Path(plan['parent']['directory'])
    result = json.loads((parent/'training.json').read_text())
    result.pop('result_sha256'); result['status'] = 'safe_pause'
    sealed_json(parent/'training.json', result)
    with pytest.raises(ValueError, match='incomplete'):
        materialize_stage(plan, 0)


def test_batch_wrapper_preserves_independent_array_index(tmp_path):
    """Do not accidentally pass a literal zero to every audit-array worker."""
    python = tmp_path/'.conda-netket/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "/tmp/frozen-config.json\\n"\n')
    python.chmod(0o755)
    target = tmp_path/'projects/clebsch_tree_nqs/remote/spatial_vit_recovery_a100.sbatch'
    target.parent.mkdir(parents=True)
    target.write_text('test "$#" = 1\ntest "$1" = energy-stream\ntest "$SLURM_ARRAY_TASK_ID" = 1\n')
    wrapper = ROOT/'projects/clebsch_tree_nqs/remote/spatial_vit_schedule_a100.sbatch'
    env = dict(os.environ, NQS_PLAN='plan', NQS_STAGE='0', NQS_PLAN_SHA256='sha',
               SLURM_ARRAY_TASK_ID='1')
    subprocess.run(['bash', str(wrapper), 'energy-stream'], cwd=tmp_path, env=env, check=True)
