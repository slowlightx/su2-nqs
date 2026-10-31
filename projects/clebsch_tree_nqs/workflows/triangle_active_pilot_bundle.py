"""Build the isolated pilot's minimal import closure and frozen input list.

This is local packaging only. It never connects to a cluster or submits a
job. Root owns rsync, environment checks and sbatch after independent review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

from .triangle_active_two_step_pilot import checked_plan, validate_runtime_fixture

ROOT=Path(__file__).resolve().parents[3]
DIRECTORY=ROOT/'configs/remote/ct12_active_pilot_20261027'


def build(*, directory: Path=DIRECTORY,
        remote_workspace: str='/home/yx789/nqs-ct12-active-pilot-20261027',
        parent: Path|None=None, fixture: Path|None=None,
        manifest_environment: str='NQS_CT12_MANIFEST_SHA256') -> dict:
    """Seal only runtime/test dependencies, plan, launcher and two inputs."""
    import numpy as np
    directory=directory.resolve()
    directory.relative_to(ROOT)
    if directory==DIRECTORY.resolve():
        raise PermissionError('submitted historical package is immutable; use a NEW configuration directory')
    if (not remote_workspace.startswith('/home/yx789/') or '..' in remote_workspace
            or not manifest_environment.isidentifier() or not manifest_environment.startswith('NQS_')):
        raise ValueError('explicit isolated deployment destination and environment name required')
    entries=[ROOT/'projects/clebsch_tree_nqs/workflows/triangle_active_two_step_pilot.py',
        ROOT/'tests/test_triangular_active_train.py',
        ROOT/'tests/test_triangle_active_two_step_pilot.py',
        ROOT/'tests/test_triangle_active_pilot_scaling_review.py']
    source=transitive_local_source_manifest(entries,root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    if any('ct6_' in name or 'campaign' in name for name in source):
        raise ValueError('old CT6/campaign entry points must not enter the isolated pilot')
    plan=json.loads((directory/'plan.json').read_text());checked_plan(plan)
    parent=(parent or ROOT/'projects/clebsch_tree_nqs/results/ct_environment_center_20261027/compressed_larger/initialized_bank.pkl').resolve()
    fixture=(fixture or ROOT/'projects/clebsch_tree_nqs/results/ct12_active_pilot_20261027/runtime_fixture.npz').resolve()
    transfers=[]
    for src,destination,sha in ((parent,plan['parent_checkpoint'],plan['parent_sha256']),
            (fixture,plan['initial_replay_fixture'],plan['initial_replay_fixture_sha256'])):
        if file_sha256(src)!=sha:raise ValueError('preselected pilot input changed: '+str(src))
        transfers.append(dict(source_relative=str(src.relative_to(ROOT)),remote_destination=destination,
            sha256=sha,bytes=src.stat().st_size))
    with np.load(fixture) as data:validate_runtime_fixture(plan,data)
    atomic_json(directory/'input_transfers.json',transfers)
    files=set(source)
    # This unchanged historical plan is a focused-test fixture only. Adding
    # it to a NEW bundle never writes to the already submitted old workspace.
    files.update(str(p.relative_to(ROOT)) for p in (directory/'plan.json',directory/'run.sbatch',
        directory/'input_transfers.json',fixture,parent,DIRECTORY/'plan.json'))
    manifest={name:file_sha256(ROOT/name) for name in sorted(files)}
    atomic_json(directory/'source_manifest.json',manifest)
    manifest_name=str((directory/'source_manifest.json').relative_to(ROOT))
    config_name=str(directory.relative_to(ROOT))
    # Generated rsync artifact, not a hand-maintained source-file list.
    (directory/'sync_files.txt').write_text('\n'.join(sorted((*files,manifest_name)))+'\n')
    report=dict(j2=plan['j2'],remote_workspace='unicorn:'+remote_workspace+'/',
        source_files=len(source),deployment_files=len(files)+1,
        deployment_bytes=sum((ROOT/name).stat().st_size for name in files)+(directory/'source_manifest.json').stat().st_size,
        manifest_sha256=file_sha256(directory/'source_manifest.json'),input_transfers=transfers,
        pilot_budget_A100_GPU_hours=4,old_CT6_budget_or_jobs_changed=False,
        hostname_or_nodelist_required=False,source_manifest_relative=manifest_name,
        source_sync_command='rsync -av --files-from='+config_name+'/sync_files.txt ./ unicorn:'+remote_workspace+'/',
        pre_submission_required='Root must create remote workspace/logs before sbatch and run the three necessary tests with CPU/x64.',
        scheduler_request=dict(partition='kim',account='kim',qos='normal',gres='gpu:nvidia_a100_80gb_pcie:1',
            CPUs=4,memory='96G',time='04:00:00',requeue=False,nodelist=None,constraint=None),
        submission_command='/usr/local/slurm/current/bin/sbatch --export=ALL,'+manifest_environment+'='+
            file_sha256(directory/'source_manifest.json')+' '+config_name+'/run.sbatch',
        submitted=False)
    atomic_json(directory/'bundle_summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DIRECTORY)
    parser.add_argument('--remote-workspace',default='/home/yx789/nqs-ct12-active-pilot-20261027')
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--manifest-environment',default='NQS_CT12_MANIFEST_SHA256')
    args=parser.parse_args()
    print(json.dumps(build(directory=args.directory,remote_workspace=args.remote_workspace,
        parent=args.parent,fixture=args.fixture,manifest_environment=args.manifest_environment),indent=2))
