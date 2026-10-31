"""Freeze a completed CT writer boundary and prepare an exact backend migration.

Intended as a small CPU job with ``afterany:<old_writer>``. This workflow never
cancels, signals or edits jobs. Submission is disabled unless its registered
configuration explicitly enables it. No amplitude, H action or training is
performed here: the new GPU training runtime replays the actual frozen parent
against the independent legacy evaluator before its first new update.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint

jax.config.update('jax_enable_x64',True)


def _registration(path: Path) -> dict:
    value=json.loads(path.read_text())
    if value['sha256'] != canonical_sha256({k:v for k,v in value.items() if k != 'sha256'}):
        raise ValueError('registration/content seal mismatch: '+str(path))
    return value


def _kernel_functions(path: Path) -> dict[str,str]:
    """Only identical validated DAG math is permitted, not an arbitrary plugin."""
    tree=ast.parse(path.read_text())
    names={'_leaves','make_orbit_dag_clebsch'}
    functions={node.name:ast.dump(node) for node in tree.body
               if isinstance(node,ast.FunctionDef) and node.name in names}
    if set(functions) != names: raise ValueError('missing validated DAG implementation')
    return functions


def prepare(spec: dict, *, submit: bool=False) -> dict:
    """Prepare one immutable (same S=0/A1) migration; never mutate the old run.

    Filesystem paths and CPU/GPU requests are explicit configuration inputs.
    ``submit`` is an additional CLI gate and must agree with ``allow_submit``.
    Both old and new states use doubled ``q=2j`` labels, unchanged physical bits
    and unchanged sampling/optimizer settings.
    """
    if jax.default_backend() != 'cpu': raise RuntimeError('migration preparer must be CPU-only')
    if submit and spec.get('allow_submit') is not True:
        raise ValueError('training submission is not explicitly authorized')
    terminal_jobs={}
    writers=spec.get('writer_jobs_to_verify',[])
    if submit and not writers:
        raise ValueError('submission requires explicit old-writer scheduler checks')
    terminal={'COMPLETED','CANCELLED','FAILED','TIMEOUT','NODE_FAIL','PREEMPTED','OUT_OF_MEMORY','BOOT_FAIL','DEADLINE'}
    for job in writers:
        if not isinstance(job,int) or isinstance(job,bool) or job < 1:
            raise ValueError('writer job IDs must be explicit positive integers')
        state=subprocess.run([spec['scontrol_binary'],'show','job','-o',str(job)],
                             check=True,capture_output=True,text=True)
        match=re.search(r'\bJobState=([A-Z_]+)\b',state.stdout)
        if match is None or match[1] not in terminal:
            raise ValueError('old writer is not confirmed terminal: '+str(job))
        if job == spec.get('old_writer_job') and match[1] != 'COMPLETED':
            raise ValueError('primary old writer ended abnormally; explicit failure review is required')
        terminal_jobs[str(job)]=match[1]
    old_root=Path(spec['old_workspace']).resolve()
    new_root=Path(spec['new_workspace']).resolve()
    old_dir=Path(spec['old_training_directory']).resolve()
    new_output=Path(spec['new_output']).resolve()
    archive=Path(spec['migration_directory']).resolve()
    proof_dir=Path(spec['optimizer_replay_directory']).resolve()
    if old_root == new_root or old_dir == new_output:
        raise ValueError('migration requires independent source/output paths')
    if new_output.exists(): raise FileExistsError('new training output already exists')
    if (archive/'prepared.json').exists(): raise FileExistsError('prepared migration is immutable')
    proof_path=proof_dir/'results.json'
    if file_sha256(proof_path) != spec['optimizer_replay_results_sha256']:
        raise ValueError('full optimizer replay result changed')
    proof=json.loads(proof_path.read_text())
    proof_reg=_registration(proof_dir/'registration.json')
    if (proof['status'] != 'passed' or not all(proof['exact_replay'].values())
            or not proof['parent_unchanged']
            or proof['arms']['baseline']['cost']['samples'] != 4096
            or proof['arms']['dag']['cost']['samples'] != 4096
            or proof['arms']['baseline']['cost']['full_H_calls'] != 4096
            or proof['arms']['dag']['cost']['full_H_calls'] != 4096):
        raise ValueError('complete original-budget optimizer replay did not pass')
    validated=Path(spec['validated_kernel_source']).resolve()
    if file_sha256(validated) != proof_reg['sources']['orbit_dag.py']:
        raise ValueError('validated prototype source changed')
    promoted=new_root/'su2_nqs/models/clebsch_tree_orbit_dag.py'
    if _kernel_functions(validated) != _kernel_functions(promoted):
        raise ValueError('promoted numerical kernel differs from the full replay')
    for name,digest in spec['staged_source_manifest'].items():
        target=(new_root/name).resolve()
        if not target.is_relative_to(new_root) or file_sha256(target) != digest:
            raise ValueError('staged new numerical source changed: '+name)
    required={'su2_nqs/models/clebsch_tree_orbit_dag.py',
              'projects/clebsch_tree_nqs/workflows/spin_clebsch_tree_training.py',
              'projects/clebsch_tree_nqs/workflows/spin_clebsch_tree_migration.py'}
    if not required.issubset(spec['staged_source_manifest']):
        raise ValueError('new backend/factory/preparer must be explicitly sealed')
    registration=_registration(old_dir/'registration.json')
    if registration['sha256'] != proof_reg['parent_registration_sha256']:
        raise ValueError('current writer changed the model/protocol since the replay')
    for name,digest in registration['source_manifest'].items():
        if file_sha256(old_root/name) != digest:
            raise ValueError('old pinned numerical source changed: '+name)
        if name != 'projects/clebsch_tree_nqs/workflows/spin_clebsch_tree_training.py':
            if file_sha256(new_root/name) != digest:
                raise ValueError('migration changed old math beyond the factory: '+name)
    parent_path=old_dir/'checkpoint.pkl'
    parent_sha=file_sha256(parent_path)
    parent=load_checkpoint(parent_path)
    meta=parent['metadata']
    physical=registration['config']
    n_sites=physical['rows']*physical['columns']
    chains=np.asarray(meta['sampler_states'])
    training=json.loads((old_dir/'training.json').read_text())
    if (meta['registration_sha256'] != registration['sha256']
            or meta['completed_steps'] != len(meta['history'])
            or meta['params_sha256'] != pytree_sha256(parent['params'])
            or int(parent['optimizer_state'][0]) != meta['optimizer_step']
            or chains.shape != (physical['chains'],n_sites)
            or not np.isin(chains,[0,1]).all()
            or not np.all(chains.sum(axis=1) == n_sites//2)
            or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(parent['params']))
            or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(parent['optimizer_state']))
            or not np.isfinite(meta['active_seconds'])
            or training['status'] not in ('safe_pause','fixed_steps_complete')
            or training['registration_sha256'] != registration['sha256']
            or training['completed_steps'] != meta['completed_steps']
            or training['checkpoint_sha256'] != parent_sha
            or training['params_sha256'] != meta['params_sha256']):
        raise ValueError('latest atomic optimizer/chains/RNG boundary is invalid')
    if file_sha256(parent_path) != parent_sha:
        raise ValueError('old writer is still changing checkpoints; do not race it')
    target=int(spec['until_step'])
    remaining=float(physical['maximum_training_seconds'])-float(meta['active_seconds'])
    if target <= meta['completed_steps'] or remaining <= 0:
        result=dict(status='target_already_reached' if target<=meta['completed_steps'] else 'scientific_cap_reached',
            latest_step=meta['completed_steps'],target_step=target,remaining_scientific_seconds=remaining,
            submitted_training_job=None,old_checkpoint_unchanged=True)
        archive.mkdir(parents=True,exist_ok=True)
        atomic_json(archive/'prepared.json',result)
        return result
    if (not isinstance(spec['session_seconds'],int) or spec['session_seconds'] <= 0
            or spec['session_seconds'] > 43200):
        raise ValueError('bounded migration session may not exceed the registered 12 hours')
    session_seconds=min(spec['session_seconds'],int(remaining))
    if session_seconds < 1: raise ValueError('insufficient remaining scientific budget')
    snapshot=archive/'parents'/f'step{meta["completed_steps"]:05d}_{parent_sha[:12]}'
    snapshot.mkdir(parents=True,exist_ok=False)
    shutil.copy2(parent_path,snapshot/'checkpoint.pkl')
    shutil.copy2(old_dir/'registration.json',snapshot/'registration.json')
    if (file_sha256(snapshot/'checkpoint.pkl') != parent_sha
            or file_sha256(parent_path) != parent_sha):
        raise ValueError('parent changed during atomic-boundary snapshot')
    config=dict(physical)
    config['execution']=dict(config.get('execution',{}),kernel='orbit_dag')
    config['output']=str(new_output)
    config['training_resume_from']=dict(directory=str(snapshot),checkpoint_sha256=parent_sha,
        registration_sha256=registration['sha256'],source_workspace=str(old_root),
        same_projected_physical_state_proved=True)
    config['interpretation']='Execution-only orbit-DAG migration, full4096 optimizer proof passed; actual-parent legacy amplitude/H/JVP/gradient replay precedes every first update. Unchanged SU2/A1/model/samples/steps/optimizer/FS and cumulative scientific cap.'
    config_path=archive/'actual_parent_training_config.json'
    atomic_json(config_path,config)
    measured=float(proof['arms']['dag']['cost']['synchronized_complete_step_seconds'])
    estimate=(target-meta['completed_steps'])*measured
    result=dict(status='prepared',source_job_dependency=spec.get('old_writer_job'),
        preparer_config_sha256=spec.get('sha256'),
        verified_terminal_writer_states=terminal_jobs,
        preparer_job_id=os.environ.get('SLURM_JOB_ID'),parent_step=meta['completed_steps'],
        parent_checkpoint_sha256=parent_sha,parent_registration_sha256=registration['sha256'],
        parent_params_sha256=pytree_sha256(parent['params']),parent_optimizer_step=int(parent['optimizer_state'][0]),
        snapshot=str(snapshot),config_path=str(config_path),new_workspace=str(new_root),new_output=str(new_output),
        target_step=target,session_seconds=session_seconds,
        remaining_scientific_seconds=remaining,measured_full_step_seconds=measured,
        estimated_remaining_seconds_without_margin=estimate,endpoint_completion_not_guaranteed=True,
        audit_requires_real_endpoint_checkpoint=True,old_checkpoint_unchanged=True,
        staged_source_manifest=spec['staged_source_manifest'],
        optimizer_replay_results_sha256=spec['optimizer_replay_results_sha256'],
        validated_promoted_kernel_AST_identical=True,submitted_training_job=None)
    atomic_json(archive/'prepared.json',result)
    if submit:
        if spec['wall_limit'] != '13:00:00': raise ValueError('unregistered training wall request')
        for path in (str(config_path),str(new_root)):
            if ',' in path: raise ValueError('Slurm export paths may not contain commas')
        command=[spec['sbatch_binary'],'--parsable','--chdir='+str(new_root),
                 '--job-name=CT8-DAG-to250',
                 '--time='+spec['wall_limit'],
                 '--export=ALL,NQS_CT_CONFIG='+str(config_path)+',NQS_CT_UNTIL_STEP='+str(target)
                    +',NQS_CT_SESSION_SECONDS='+str(session_seconds),str(new_root/spec['training_sbatch'])]
        completed=subprocess.run(command,check=True,capture_output=True,text=True)
        job=completed.stdout.strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError('unrecognized sbatch response')
        result['submitted_training_job']=int(job)
        result['submission_command']=command
        atomic_json(archive/'submitted.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    print(json.dumps(prepare(_registration(args.config),submit=args.submit),indent=2),flush=True)
