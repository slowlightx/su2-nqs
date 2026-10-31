"""Bounded paper-capacity square ClebschTree optimizer reentry.

Reuse the existing physical CT trainer and independent audit, not a second H,
sampler or neural model. This explicitly resets imported Adam moments once;
it is not disguised as strict continuation of the old direct-SR optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.models.literature_clebsch_tree import clebsch_layers
from su2_nqs.vmc.checkpoint import load_checkpoint
from .spin_clebsch_tree_audit import audit
from .spin_clebsch_tree_sampled import ROOT, components
from .spin_clebsch_tree_training import train


def validate_plan(plan: dict, spec: dict) -> None:
    """Restrict this release to exact paper widths and 128 new same-H steps."""
    start=plan['parent']['completed_steps']
    expected=dict(geometry='square',rows=8,columns=8,multiplicity_cap=16,
                  high_spin_cap=8,maximum_q=8,j2_over_j1=plan['j2_over_j1'])
    if (plan['j2_over_j1'] not in (0.,.5) or start!=(32 if plan['j2_over_j1']==0 else 450)
            or any(spec.get(k)!=v for k,v in expected.items())
            or plan['additional_updates']!=128 or spec['steps']!=start+128
            or spec['sr_warm_steps']!=start+16 or spec['warm_samples']!=256
            or spec['adam_samples']!=4096 or spec['chains']!=16
            or spec['sr_damping']!=.01 or spec['warm_lr']!=.005 or spec['adam_lr']!=.001
            or spec.get('sr_use_adam') is not True or spec.get('strict_nonzero_updates') is not True
            or spec['maximum_FS_sine']!=.05 or spec.get('sr_tangent_fs_radius') is not None):
        raise ValueError('exact paper-capacity/registered paper-like optimizer reentry required')
    if (not spec['training_resume_from']['reset_optimizer_moments']
            or not spec['training_resume_from']['optimizer_reset_reason']
            or plan['phase_seconds']!=dict(pilot=3600,resume=3600,train=50400,audit=3600)
            or sum(plan['phase_seconds'].values())!=17*3600
            or not 0<plan['maximum_peak_device_bytes']<=70*1024**3):
        raise ValueError('explicit moment reset and fixed17-A100-hour per-case allocation required')
    a=plan['audit']
    if (a['endpoint_step']!=start+128 or a['streams']!=2 or a['chains_per_stream']!=32
            or a['snapshots_per_stream']!=128 or len(set(a['audit_seeds']))!=2
            or set(a['audit_seeds'])&{spec['training_seed'],spec['sampling_seed'],spec['seed']}
            or a['training_directory']!=spec['output']):
        raise ValueError('fresh two-stream8192-record final endpoint audit required')


def checked_parent(plan: dict, spec: dict) -> dict:
    """Bind native parameters, historical work, optimizer and physical chains."""
    request=plan['parent']; path=Path(request['directory'])
    reg=json.loads((path/'registration.json').read_text())
    saved=load_checkpoint(path/'checkpoint.pkl'); meta=saved['metadata']
    if (reg['sha256']!=canonical_sha256({k:v for k,v in reg.items() if k!='sha256'})
            or reg['sha256']!=request['registration_sha256']
            or file_sha256(path/'checkpoint.pkl')!=request['checkpoint_sha256']
            or meta['registration_sha256']!=reg['sha256']
            or meta['completed_steps']!=request['completed_steps']
            or len(meta['history'])!=request['completed_steps']
            or pytree_sha256(saved['params'])!=request['params_sha256']
            or meta['params_sha256']!=request['params_sha256']
            or int(saved['optimizer_state'][0])!=meta['optimizer_step']
            or pytree_sha256(saved['optimizer_state'])!=request['optimizer_sha256']
            or pytree_sha256((jnp.asarray(meta['rng']),jnp.asarray(meta['sampler_states'])))
                !=request['sampler_rng_sha256']):
        raise ValueError('immutable same-H native parent/optimizer/sampler/RNG changed')
    for name in ('geometry','rows','columns','j2_over_j1','multiplicity_cap','high_spin_cap','maximum_q'):
        if reg['config'][name]!=spec[name]:
            raise ValueError('parent physical family changed: '+name)
    states=np.asarray(meta['sampler_states'])
    if (states.shape!=(16,64) or not np.isin(states,[0,1]).all()
            or not np.all(states.sum(-1)==32)
            or sum(x.size for x in jax.tree.leaves(saved['params']))!=227958):
        raise ValueError('parent must have legal16 physical M0 chains and paper227958 parameters')
    return saved


def checked_progress(plan: dict, spec: dict, step: int) -> tuple[dict,list[dict]]:
    """Check each NEW update, not inherited historical accepted flags."""
    out=Path(spec['output']); cp=load_checkpoint(out/'checkpoint.pkl'); meta=cp['metadata']
    reg=json.loads((out/'registration.json').read_text()); start=plan['parent']['completed_steps']
    rows=[row for row in meta['history'] if row['step']>start]
    if (meta['completed_steps']!=step or len(meta['history'])!=step or len(rows)!=step-start
            or meta['registration_sha256']!=reg['sha256']
            or meta['params_sha256']!=pytree_sha256(cp['params'])
            or meta['optimizer_sha256']!=pytree_sha256(cp['optimizer_state'])
            or meta['sampler_rng_sha256']!=pytree_sha256((jnp.asarray(meta['rng']),
                jnp.asarray(meta['sampler_states'])))):
        raise ValueError('new checkpoint counters/full-state seals differ')
    expected_counter=step-start if step<=start+16 else step-start-16
    if int(cp['optimizer_state'][0])!=expected_counter:
        raise ValueError('optimizer reset was repeated or missing at registered phase switch')
    for row in rows:
        warm=row['step']<=start+16
        if (row['samples']!=(256 if warm else 4096)
                or row['full_H_calls']!=(256 if warm else 4096)
                or row['optimizer_phase']!=('SR_then_Adam' if warm else 'clipped_Adam')
                or row['accepted'] is not True or not 1e-10<row['FS_sine']<=.05
                or not np.isfinite(row['energy_before_update_per_site'])
                or not np.isfinite(row['variance_before_update_per_site'])
                or row['variance_before_update_per_site']<0
                or not np.isfinite(row['ESS_fraction']) or not .95<=row['ESS_fraction']<=1+1e-12):
            raise ValueError('zero/rejected/nonfinite or wrong-recipe update cannot release production')
    return cp,rows


def checked_boundaries(out: Path, registration: str) -> tuple[dict,dict]:
    """Require the actual saved pilot and independently restored checkpoint."""
    pilot=json.loads((out/'pilot.json').read_text()); resumed=json.loads((out/'resume.json').read_text())
    if (pilot['status']!='passed' or resumed['status']!='passed'
            or pilot['campaign_sha256']!=registration or resumed['campaign_sha256']!=registration
            or file_sha256(out/'quality'/f"step{pilot['endpoint']:05d}.pkl")!=pilot['checkpoint_sha256']
            or file_sha256(out/'quality'/f"step{resumed['endpoint']:05d}.pkl")!=resumed['checkpoint_sha256']
            or resumed['pilot_checkpoint_sha256']!=pilot['checkpoint_sha256']):
        raise ValueError('sealed real pilot and separate-process resume required')
    return pilot,resumed


def run(plan_path: Path, phase: str) -> dict:
    """Bounded actual training/restart then untouched audit; never cold-retry."""
    jax.config.update('jax_enable_x64',True)
    plan=json.loads(plan_path.read_text()); config_path=ROOT/plan['training_config']
    if file_sha256(config_path)!=plan['training_config_sha256']:
        raise ValueError('actual isolated training config missing or changed')
    spec=json.loads(config_path.read_text()); validate_plan(plan,spec); parent=checked_parent(plan,spec)
    if phase=='preflight':
        cfg,_,_,_,images,_=components(spec)
        return dict(status='passed',parameter_count=sum(x.size for x in jax.tree.leaves(parent['params'])),
            full_A1_images=len(images),layer_widths=[list(x.widths) for x in clebsch_layers(cfg)],
            parent=plan['parent'],new_optimizer_reset_is_explicit=True,new_parameter_updates=0)
    if spec['require_gpu'] and jax.default_backend()!='gpu':
        raise RuntimeError('actual allocated GPU required')
    out=Path(spec['output']); out.mkdir(parents=True,exist_ok=True)
    identity=dict(plan_sha256=file_sha256(plan_path),config_sha256=file_sha256(config_path),
        source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    seal=canonical_sha256(identity); path=out/'campaign_registration.json'
    if path.exists() and json.loads(path.read_text())!=dict(identity,sha256=seal):
        raise ValueError('isolated campaign source/config changed')
    atomic_json(path,dict(identity,sha256=seal))
    start=plan['parent']['completed_steps']
    if phase=='audit':
        checked_boundaries(out,seal);checked_progress(plan,spec,start+128)
        return audit(plan['audit'],Path(plan['audit']['output']))
    if phase not in ('pilot','resume','train'):
        raise ValueError('unknown phase')
    if phase=='resume':
        pilot=json.loads((out/'pilot.json').read_text())
        if pilot['status']!='passed' or pilot['campaign_sha256']!=seal or file_sha256(out/'checkpoint.pkl')!=pilot['checkpoint_sha256']:
            raise ValueError('only the exact completed pilot may resume')
    if phase=='train':
        _,resumed=checked_boundaries(out,seal)
        if not (out/'production_started.json').exists() and file_sha256(out/'checkpoint.pkl')!=resumed['checkpoint_sha256']:
            raise ValueError('production must start at the checked independent resume boundary')
        atomic_json(out/'production_started.json',dict(campaign_sha256=seal,resume_checkpoint_sha256=resumed['checkpoint_sha256']))
    target=start+{'pilot':2,'resume':4,'train':128}[phase]
    result=train(spec,out,target,plan['phase_seconds'][phase]-300)
    if result['completed_steps']!=target:
        return dict(status='safe_pause',completed_steps=result['completed_steps'])
    cp,rows=checked_progress(plan,spec,target)
    # Verify actual imported parameters/chains/RNG at the archived step-zero
    # boundary. Only the registered optimizer moments are deliberately reset.
    initial=load_checkpoint(out/'quality'/f'step{start:05d}.pkl')
    if (pytree_sha256(initial['params'])!=plan['parent']['params_sha256']
            or int(initial['optimizer_state'][0])!=0
            or not all(np.array_equal(initial['metadata'][k],parent['metadata'][k]) for k in ('rng','sampler_states'))):
        raise ValueError('warm start did not preserve actual parent physical state/chains/RNG')
    if phase=='train':
        return dict(status='fixed_steps_complete',completed_steps=target,new_updates=128,
            checkpoint_sha256=file_sha256(out/'checkpoint.pkl'),optimizer_counter=int(cp['optimizer_state'][0]))
    stats=jax.local_devices()[0].memory_stats() or {}
    peak=int(stats.get('peak_bytes_in_use',0)); warm=rows[-1]['seconds']
    expected_adam=max(plan['parent_measured_4096_step_seconds'],16*warm)
    forecast=max(0,16-len(rows))*warm+112*expected_adam+2*expected_adam+1800
    passed=0<peak<=plan['maximum_peak_device_bytes'] and forecast<plan['phase_seconds']['train']
    record=dict(status='passed' if passed else 'cost_gate_failed',campaign_sha256=seal,endpoint=target,
        checkpoint_sha256=file_sha256(out/'checkpoint.pkl'),actual_new_updates=len(rows),
        measured_last_warm_step_seconds=warm,forecast_remaining_with_audit_seconds=forecast,
        peak_device_bytes=peak,restored_optimizer_step=int(cp['optimizer_state'][0]),
        deterministic_transition_replay_claim=False)
    if phase=='resume':record['pilot_checkpoint_sha256']=pilot['checkpoint_sha256']
    atomic_json(out/(phase+'.json'),record)
    if not passed:raise RuntimeError('measured time/memory gate failed; keep actual checkpoint, no cold retry')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--phase',choices=('preflight','pilot','resume','train','audit'),required=True)
    args=parser.parse_args();answer=run(args.plan,args.phase);print(json.dumps(answer,indent=2))
    if answer['status']=='safe_pause':raise SystemExit(75)
