"""Explicitly bounded automatic CT6 continuation, never a live-job hot patch.

Four serial24h allocations per coupling are authorized. Each binds only a
completed independently audited predecessor. Native W/moments/next RNG and
learned control VALUES continue exactly. Monitoring windows/streaks reset at
segment boundaries; the entire old controller/history is retained, not erased
or falsely described as one uninterrupted feedback window.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import shutil
import sys

import jax
import numpy as np

from su2_nqs.frontier.artifacts import canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.adaptive_sr_controls import AdaptiveSRState,restore_sr_controls
from . import ct6_adaptive_continuation as base

TASK='CT6_adaptive_campaign_segment_v1'


def validate_plan(plan:dict,*,executing:bool=False)->None:
    """Independent new profile; never disguise it as the old closed protocol."""
    if plan.get('task')!=TASK or plan.get('j2') not in (0.,.125):
        raise ValueError('explicit CT6 automatic campaign profile required')
    fixed=dict(kind='ct6',device='A100',model=dict(length=6,multiplicity_cap=28,maximum_q=10),
        arity='ternary_joint' if plan['j2']==0 else 'binary_truncated',binary_rank=8,
        steps=32,block_attempts=8,monitor_after=[8,16,24],fit_records=5120,
        calibration_records=4096,calibration_split=2048,monitor_records=4096,monitor_groups=128,
        monitor_jvp_records=256,monitor_jvp_groups=32,audit_records=8192,audit_groups=128,
        interim_records=0,interim_groups=0,epsilon=.125,damping=.05,learning_rate=.03,
        target_FS=.001,maximum_FS=.00125,projection_chunk=4,action_chunk=4,H_chunk=4,
        score_chunk=2,parameter_chunk_size=8192,maximum_peak_device_bytes=70*1024**3,
        maximum_fit_H_records=33*5632,wall_cap_seconds=86400,
        monitor_min_effective_contributions=64,monitor_maximum_fraction=.05,
        new_control_compile_reserve_seconds=600.,allocator_preflight=True,
        no_requeue=True,attempts_not_successful_updates=True,step_counter='attempts',
        training_monitor_is_not_terminal_audit=True,native_factors=base.FACTORS.tolist(),
        automatic_next_allocation=True,campaign_segments_per_coupling=4,
        campaign_A100_hours_upper_per_coupling=96,resource_horizon='next_eight_attempt_block',
        parent_binding='completed_frozen_audited_predecessor_only',
        control_window_boundary='inherit_values_archive_history_reset_monitor_window')
    if any(plan.get(k)!=v for k,v in fixed.items()):
        raise ValueError('fixed scientific, four-segment and next-block resource bounds changed')
    if plan.get('segment_index') not in (1,2,3,4) or isinstance(plan['segment_index'],bool):
        raise ValueError('only four serial allocations per coupling are authorized')
    if base.policy_from_plan(plan)!=base.AdaptiveSRPolicy(fit_levels=(5120,5632),
            initial_fs_radius=.00125,minimum_fs_radius=.00025,maximum_fs_radius=.0025,
            initial_damping=.05,minimum_damping=.0125,maximum_damping=.8,
            maximum_monitor_looks=4,success_patience=2,noise_patience=2):
        raise ValueError('registered per-segment adaptive policy required')
    seeds=plan.get('seeds',{})
    if (set(seeds)!=set(base.required_seed_roles()) or len(set(seeds.values()))!=len(seeds)
            or any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<2**32 for v in seeds.values())):
        raise ValueError('fresh disjoint uint32 role streams required')
    if executing and plan.get('execution_authorized') is not True:
        raise PermissionError('bounded campaign execution not authorized')


def independent_parent_products(path:Path,audit:dict)->dict:
    """Independent NumPy ratios/delete-group errors of immutable full-H data."""
    with np.load(path) as raw:
        p,h,Q=[raw[k] for k in ('psi','hpsi','proposal')]
    if (p.shape!=h.shape or p.ndim!=2 or p.shape[1]!=2 or Q.shape!=(len(p),)
            or not all(np.isfinite(x).all() for x in (p,h,Q)) or np.any(Q<=0)):
        raise ValueError('finite node-faithful paired terminal products required')
    groups=audit['moments']['independent_groups']
    if groups<4 or len(p)%groups or len(p)!=audit['records']:
        raise ValueError('registered independent terminal grouping required')
    def estimate(ids):
        z=(abs(p[ids])**2/Q[ids,None]).mean(0)
        e=(p[ids].conj()*h[ids]/Q[ids,None]).mean(0)/z
        v=(abs(h[ids]-e*p[ids])**2/Q[ids,None]).mean(0)/z
        return np.stack((e.real/36.,e.imag/36.,v/36.,z),axis=1)
    ids=np.arange(len(p));point=estimate(ids);labels=np.repeat(np.arange(groups),len(p)//groups)
    leave=np.asarray([estimate(ids[labels!=g]) for g in range(groups)])
    diff=leave[:,1]-leave[:,0]
    se=np.sqrt((groups-1)/groups*np.sum((diff-diff.mean(0))**2,axis=0))
    delta=point[1]-point[0];m=audit['moments']
    errors=[np.max(abs(point-np.asarray(m['endpoint_values']))),
        np.max(abs(delta-np.asarray(m['candidate_minus_control']))),
        np.max(abs(se-np.asarray(m['paired_independent_block_jackknife_SE'])))]
    if not np.isfinite(errors).all() or max(errors)>2e-10:
        raise ArithmeticError('independent terminal product recomputation disagrees')
    if delta[0]>2*se[0] or delta[2]>2*se[2]:
        raise ValueError('parent terminal audit resolves harm; automatic continuation vetoed')
    return dict(maximum_recompute_difference=float(max(errors)),delta=delta.tolist(),
        paired_independent_SE=se.tolist(),safe_to_continue=True)


def bind_parent(plan:dict,out:Path)->dict:
    """Seal a COMPLETED frozen predecessor; never read a moving checkpoint."""
    src=Path(plan['parent_directory'])
    reg=json.loads((src/'registration.json').read_text());old=reg['plan']
    if (old.get('task') not in (base.TASK,TASK) or old['model']!=plan['model']
            or old['j2']!=plan['j2'] or old['arity']!=plan['arity']):
        raise ValueError('same-model/coupling audited adaptive predecessor required')
    (base.validate_plan if old['task']==base.TASK else validate_plan)(old,executing=True)
    state,controls,progress=base.load_adaptive_state(src/'checkpoint.pkl',plan=old,registration=reg)
    final=json.loads((src/'audit_result.json').read_text())
    frozen=json.loads((src/'endpoint_frozen.json').read_text())
    audit=json.loads((src/'terminal_audit/audit.json').read_text())
    digest=pytree_sha256(state[0]);checkpoint=file_sha256(src/'checkpoint.pkl')
    if (final.get('status')!='complete_fixed_prefix_independent_audit'
            or progress['accepted_updates']<1 or audit.get('status')!='complete'
            or final.get('terminal_audit')!=audit or final.get('parameter_sha256')!=digest
            or frozen.get('parameter_sha256')!=digest or audit.get('endpoint_params_sha256')!=digest
            or final.get('checkpoint_sha256')!=checkpoint or frozen.get('checkpoint_sha256')!=checkpoint
            or final.get('independent_from_training_geometry_and_adaptation') is not True
            or audit.get('seed')!=old['seeds']['audit']
            or audit.get('statistics_seed')!=old['seeds']['audit_statistics']
            or audit.get('no_endpoint_selection') is not True
            or audit.get('records')!=old['audit_records']):
        raise ValueError('completed successful terminal audit/checkpoint binding required')
    verified=independent_parent_products(src/'terminal_audit/products.npz',audit)
    future=deepcopy(state[2]);historical=set(reg.get('campaign_seed_history',[]))|set(old['seeds'].values())
    used=set(future['consumed_role_seeds'])|set(future['excluded_seeds'])|historical
    for i in range(future['next_step'],future['next_step']+plan['steps']):
        roles,future=base.c6.role_seeds(future,i)
        if set(roles.values())&historical:
            raise ValueError('future native RNG collides with historical monitoring/audit')
        used.update(roles.values())
    if used&set(plan['seeds'].values()):raise ValueError('new campaign streams collide with parent/future draws')
    snapshot=out/'parent_snapshot';snapshot.mkdir()
    for name,path in (('checkpoint.pkl',src/'checkpoint.pkl'),('registration.json',src/'registration.json'),
            ('audit.json',src/'terminal_audit/audit.json'),('products.npz',src/'terminal_audit/products.npz'),
            ('audit_result.json',src/'audit_result.json'),('endpoint_frozen.json',src/'endpoint_frozen.json')):
        shutil.copy2(path,snapshot/name)
    absolute=reg['parent_global_step']+progress['accepted_updates']
    base.legacy.save_state(out/'parent.pkl',state,0,absolute)
    history=reg.get('controller_history',[])+[dict(source_checkpoint_sha256=checkpoint,
        archived_full_controller=base.sr_control_checkpoint(base.policy_from_plan(old),controls),
        parent_native_cursor=state[2]['next_step'])]
    return dict(parent_global_step=absolute,native_origin=reg['native_origin'],
        parent_seals={p.name:file_sha256(p) for p in snapshot.iterdir()},
        parameter_count=sum(np.size(x) for x in jax.tree.leaves(state[0])),
        original_parameter_sha256=digest,original_optimizer_sha256=pytree_sha256(state[1]),
        original_rng_sha256=canonical_sha256(state[2]),independent_parent_review=verified,
        inherited_control_values={k:getattr(controls,k) for k in ('fs_radius','damping','fit_records')},
        controller_history=history,campaign_seed_history=sorted(historical|set(plan['seeds'].values())),
        control_monitor_window_reset_not_continuous_streak_claim=True,
        parent_audit_consumed_only_for_continuation_safety=True)


def initial_controls(plan:dict,binding:dict)->AdaptiveSRState:
    """Inherit learned physical controls, explicitly open a fresh monitor window."""
    state=AdaptiveSRState(**binding['inherited_control_values'])
    base.validate_control_state(base.policy_from_plan(plan),state)
    return state


def resources_for_next_block(plan:dict,registration:dict,progress:dict,**kwargs):
    """Require enough budget for NEXT8 attempts, not all remaining24 at once."""
    envelope=base.resources_for_next_block(plan,registration,progress,**kwargs)
    return replace(envelope,remaining_updates=min(plan['block_attempts'],envelope.remaining_updates))


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','replay','train','audit'),required=True)
    args=parser.parse_args()
    result=base.run(args.plan,args.output,args.phase,profile=sys.modules[__name__])
    print(json.dumps({k:result[k] for k in ('phase','status','attempts','accepted_updates') if k in result}),flush=True)


if __name__=='__main__':main()
