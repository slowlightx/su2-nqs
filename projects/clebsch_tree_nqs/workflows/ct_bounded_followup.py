"""Short, sealed CT6/A100 or CT12/B200 continuation of real optimized states.

Reuses the established physical kernels, native-amplitude SR, finite-state
guards and IID estimators. Keeps parameters, moments and unused RNG across
allocations. Four process phases replay the first complete update before
committing it once. No architecture, charge cutoff or spatial group changes.
No automatic requeue: a preemption preserves the last atomic committed prefix
for an explicitly authorized later continuation, not a zero-cost restart.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
from time import time, perf_counter

import jax
jax.config.update('jax_enable_x64',True)
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from . import ct6_importance_amplitude_continuation as c6
from . import triangle_active_two_step_pilot as c12
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT=Path(__file__).resolve().parents[3]


def validate_plan(p:dict) -> None:
    """Keep legacy pilots closed; explicitly register one CT12 production arm."""
    if 'adaptive_sr' in p:
        raise ValueError('frozen followup/production32 profiles cannot silently enable adaptive controls')
    production=p.get('task')=='CT12_production32_v1'
    if p.get('task') not in ('CT_bounded_followup_v1','CT12_production32_v1') or not p.get('execution_authorized'):
        raise PermissionError('explicit bounded followup authority required')
    common=dict(epsilon=.125,automatic_next_allocation=False,learning_rate=.03,no_requeue=True)
    if any(p.get(k)!=v for k,v in common.items()):raise ValueError('closed followup controls changed')
    profiles={
        'ct6':dict(device='A100',damping=.05,steps=8,fit_records=5120,calibration_records=4096,
            calibration_split=2048,audit_records=8192,audit_groups=128,
            interim_records=2048,interim_groups=128,wall_cap_seconds=43200,
            maximum_peak_device_bytes=70*1024**3),
        'ct12':dict(device='B200',damping=.5,steps=4,fit_records=1024,calibration_records=1024,
            calibration_split=512,audit_records=1024,audit_groups=16,
            interim_records=0,interim_groups=0,wall_cap_seconds=10800,
            maximum_peak_device_bytes=136*1024**3)}
    profile=profiles.get(p.get('kind'))
    if production:
        if p.get('kind')!='ct12' or p.get('j2')!=.125 or p.get('parent_format')!='CT_bounded_followup_v1':
            raise ValueError('production32 is only the frozen J.125 CT12 continuation')
        profile=dict(profiles['ct12'],steps=32,audit_records=4096,audit_groups=64,
                     wall_cap_seconds=57600)
        acceptance=p.get('software_acceptance',{})
        if (not isinstance(acceptance.get('directory'),str)
                or not isinstance(acceptance.get('plan_path'),str)
                or len(acceptance.get('plan_sha256',''))!=64
                or len(acceptance.get('source_manifest_sha256',''))!=64):
            raise ValueError('sealed successful B200 software acceptance required')
    if profile is None or any(p.get(k)!=v for k,v in profile.items()):
        raise ValueError('complete fixed panels and allocation envelope required')
    if p.get('j2') not in (0.,.125) or isinstance(p.get('j2'),bool):raise ValueError('registered coupling required')
    seeds=p.get('seeds',{})
    if set(seeds)!={'preflight','interim','interim_statistics','audit','audit_statistics'}:
        raise ValueError('five explicit independent role streams required')
    if any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<2**32 for x in seeds.values()) or len(set(seeds.values()))!=5:
        raise ValueError('fresh distinct uint32 seeds required')
    if p.get('kind')=='ct6' and (p.get('model')!=dict(length=6,multiplicity_cap=28,maximum_q=10)
            or p.get('arity')!=('ternary_joint' if p['j2']==0 else 'binary_truncated')):
        raise ValueError('unchanged w28/q10 CT6 production structure required')
    if p.get('kind')=='ct12' and p.get('model')!=dict(length=12,multiplicity_cap=20,maximum_q=10):
        raise ValueError('unchanged active CT12 support required')
    if p.get('parent_format') not in (None,'CT_bounded_followup_v1'):
        raise ValueError('unknown parent checkpoint format')
    if p.get('parent_format')=='CT_bounded_followup_v1' and p['kind']!='ct6' and not production:
        raise ValueError('bounded checkpoint continuation currently validated for CT6 only')
    for name in ('projection_chunk','action_chunk','score_chunk','H_chunk'):
        if isinstance(p.get(name),bool) or not isinstance(p.get(name),int) or p[name]<1:
            raise ValueError('positive explicit kernel chunks required')


def safe_to_continue(audit:dict) -> bool:
    """One frozen safety rule; unresolved gains are not called a win."""
    m=audit.get('moments',audit)
    delta=np.asarray(m['candidate_minus_control']);se=np.asarray(m['paired_iid_delta_SE'])
    if not np.isfinite(np.r_[delta,se]).all() or np.any(se<0):raise ValueError('finite audited moments required')
    return bool(delta[0]<=2*se[0] and delta[2]<=2*se[2])


def save_state(path:Path,state:tuple,completed:int,absolute_step:int) -> None:
    """Atomic original parameters, inherited moments and exact next RNG."""
    save_checkpoint(path,state[0],optimizer_state=state[1],metadata=dict(
        completed_updates=completed,absolute_completed_steps=absolute_step,rng=state[2],
        parameter_sha256=pytree_sha256(state[0]),optimizer_sha256=pytree_sha256(state[1]),
        rng_sha256=canonical_sha256(state[2])))


def load_state(path:Path) -> tuple[tuple,dict]:
    """Fail closed on lost moments or RNG, not only parameter restoration."""
    v=load_checkpoint(path);m=v['metadata'];s=(v['params'],v['optimizer_state'],m['rng'])
    if (pytree_sha256(s[0])!=m['parameter_sha256'] or pytree_sha256(s[1])!=m['optimizer_sha256']
            or canonical_sha256(s[2])!=m['rng_sha256']):raise ValueError('continuation checkpoint hash mismatch')
    return s,m


def require_software_acceptance(p:dict) -> dict|None:
    """A queued production allocation cannot bypass failed/incomplete software."""
    if p['task']!='CT12_production32_v1':return None
    expected=p['software_acceptance'];root=Path(expected['directory'])
    certificate=root/'gpu/gpu_check_result.json'
    result=json.loads(certificate.read_text())
    if (result.get('passed') is not True or result.get('status')!='bounded_software_acceptance_complete'
            or result.get('plan_sha256')!=expected['plan_sha256']
            or canonical_sha256(result.get('source_manifest'))!=expected['source_manifest_sha256']):
        raise ValueError('production requires the exact passed software acceptance, not job exit alone')
    return dict(certificate_sha256=file_sha256(certificate),directory=str(root),
                plan_sha256=result['plan_sha256'],no_check_bypass=True)


def verify_production_runtime(p:dict) -> dict|None:
    """Check actual libraries inside each training process, not just a wrapper."""
    if p['task']!='CT12_production32_v1':return None
    from . import accelerator_runtime_validation as acceptance
    expected=p['software_acceptance']
    plan=json.loads(Path(expected['plan_path']).read_text());acceptance.validate_plan(plan)
    if (canonical_sha256(plan)!=expected['plan_sha256']
            or canonical_sha256(acceptance._source_manifest())!=expected['source_manifest_sha256']):
        raise ValueError('accepted numerical kernel/runtime contract changed before production')
    stack,_,_=acceptance.loaded_cuda_stack(plan)
    return stack


def bind_parent(p:dict,out:Path) -> dict:
    """Bind only a completed, independently audited previous allocation.

    A queued CT6/.125 continuation may wait for its running parent. The
    predeclared path/registration are fixed now; the completed endpoint seals
    are captured once at execution, never from a moving checkpoint.
    """
    src=Path(p['parent_directory']);registration=json.loads((src/'registration.json').read_text())
    if file_sha256(src/'registration.json')!=p['parent_registration_sha256']:
        raise ValueError('preselected parent registration identity changed')
    if p.get('parent_checkpoint_sha256') and file_sha256(src/'checkpoint.pkl')!=p['parent_checkpoint_sha256']:
        raise ValueError('preselected committed checkpoint identity changed')
    base=registration['plan'];saved=load_checkpoint(src/'checkpoint.pkl')
    if p['kind']=='ct6':
        audit=json.loads((src/'terminal_audit/audit.json').read_text())
        if p.get('parent_format')=='CT_bounded_followup_v1':
            result_path=src/'audit_result.json';result=json.loads(result_path.read_text())
            frozen=json.loads((src/'endpoint_frozen.json').read_text())
            state,meta=load_state(src/'checkpoint.pkl')
            endpoint=result.get('parameter_sha256');absolute=meta['absolute_completed_steps']
            origin=registration['native_origin']
            if (base.get('task')!='CT_bounded_followup_v1' or base.get('kind')!='ct6'
                    or result.get('status')!='complete_fixed_prefix_independent_audit'
                    or result.get('completed_updates',0)<1
                    or meta['completed_updates']!=result['completed_updates']
                    or frozen.get('status')!='fixed_prefix_frozen'
                    or frozen.get('completed_updates')!=result['completed_updates']
                    or frozen.get('parameter_sha256')!=endpoint
                    or frozen.get('checkpoint_sha256')!=result.get('checkpoint_sha256')
                    or frozen.get('absolute_step')!=absolute
                    or registration['parent_global_step']+result['completed_updates']!=absolute):
                raise ValueError('completed bounded CT6 endpoint/audit identity required')
            parent_j2=base['j2']
        else:
            result_path=src/'results.json';result=json.loads(result_path.read_text())
            if result.get('status') not in ('fixed16_updates_complete_independent_audit',
                    'fixed2_updates_complete_independent_audit','bounded_prefix_independently_audited'):
                raise ValueError('completed native CT6 endpoint/audit identity required')
            state,meta=c6.load_training_state(src/'checkpoint.pkl')
            endpoint=result.get('params_sha256');absolute=meta['absolute_completed_steps']
            origin=meta['original_parent_global_step'];parent_j2=base['j2_over_j1']
        if (audit.get('status')!='complete' or result.get('completed_updates',0)<1
                or endpoint!=pytree_sha256(saved['params']) or audit['endpoint_params_sha256']!=endpoint
                or result['checkpoint_sha256']!=file_sha256(src/'checkpoint.pkl')
                or absolute!=origin+state[2]['next_step']-1):
            raise ValueError('completed CT6 checkpoint/audit/native RNG identity required')
        if parent_j2!=p['j2'] or base['model']!=p['model'] or base['arity']!=p['arity']:
            raise ValueError('parent model/coupling differs')
        product=src/'terminal_audit/products.npz'
    elif p.get('parent_format')=='CT_bounded_followup_v1':
        result_path=src/'audit_result.json';result=json.loads(result_path.read_text())
        audit=json.loads((src/'terminal_audit/audit.json').read_text())
        frozen=json.loads((src/'endpoint_frozen.json').read_text())
        state,meta=load_state(src/'checkpoint.pkl');absolute=meta['absolute_completed_steps']
        origin=registration['native_origin'];done=result.get('completed_updates',0)
        if (base.get('task') not in ('CT_bounded_followup_v1','CT12_production32_v1')
                or base.get('kind')!='ct12' or base.get('j2')!=p['j2']
                or base.get('model')!=p['model'] or base.get('active_profile')!=p['active_profile']
                or result.get('status')!='complete_fixed_prefix_independent_audit' or done<1
                or meta['completed_updates']!=done or frozen.get('completed_updates')!=done
                or frozen.get('status')!='fixed_prefix_frozen'
                or frozen.get('absolute_step')!=absolute
                or registration['parent_global_step']+done!=absolute
                or int(np.asarray(state[1][0]))!=absolute
                or result.get('parameter_sha256')!=pytree_sha256(state[0])
                or frozen.get('parameter_sha256')!=result['parameter_sha256']
                or audit.get('endpoint_params_sha256')!=result['parameter_sha256']
                or not audit.get('independent_from_training_and_geometry')
                or audit.get('records')!=base['audit_records']
                or result.get('terminal_audit')!=audit
                or frozen.get('checkpoint_sha256')!=file_sha256(src/'checkpoint.pkl')
                or result.get('checkpoint_sha256')!=file_sha256(src/'checkpoint.pkl')):
            raise ValueError('completed same-profile CT12 endpoint, moments and original optimizer required')
        product=src/'terminal_audit/untouched_products.npz'
    else:
        audit=json.loads((src/'audit_result.json').read_text())
        result=json.loads((src/'replay_train_result.json').read_text())
        # The sealed original J.125 pilot predates the redundant audit.j2
        # field. Its exact audit SHA plus registration/bond identity are
        # required; never infer a new coupling from an unsealed old report.
        audit_j2=audit.get('j2')
        if audit_j2 is None and p.get('parent_audit_sha256')==file_sha256(src/'audit_result.json'):
            audit_j2=base['j2']
        if (result.get('completed_updates')!=2 or audit.get('status')!='complete_fixed_prefix_independent_audit'
                or not audit.get('independent_from_training_and_geometry') or audit_j2!=p['j2']
                or base['j2']!=p['j2'] or base['active_profile']!=p['active_profile']
                or audit.get('physical_bonds')!=len(c12.registered_bonds(p['j2'])) or audit.get('images')!=1728):
            raise ValueError('actual completed same-profile CT12 state required')
        state=c12._state_load(src/'checkpoint.pkl');absolute=2;origin=0
        product=src/'audit/untouched_products.npz'
    audit_path=src/('terminal_audit/audit.json' if p['kind']=='ct6' else 'audit_result.json')
    if p.get('parent_audit_sha256') and p['parent_audit_sha256']!=file_sha256(audit_path):
        raise ValueError('preselected completed independent audit identity changed')
    if not safe_to_continue(audit):raise ValueError('parent final audit resolved harm; do not blindly extend')
    used=set(base.get('seeds',{}).values());future=deepcopy(state[2])
    if p['kind']=='ct6':
        used.update(future['consumed_role_seeds']);used.update(future['excluded_seeds'])
        for i in range(future['next_step'],future['next_step']+p['steps']):
            roles,future=c6.role_seeds(future,i);used.update(roles.values())
    else:
        for _ in range(p['steps']):
            roles,future=c12._seeds(future);used.update(roles)
    if used & set(p['seeds'].values()):raise ValueError('reserved new audit stream collides with training/history')
    snapshot=out/'parent_snapshot';snapshot.mkdir()
    # Copy only necessary immutable restart/audit evidence, never old logs/caches.
    import shutil
    for name,path in [('checkpoint.pkl',src/'checkpoint.pkl'),('registration.json',src/'registration.json'),
                      ('audit.json',src/('terminal_audit/audit.json' if p['kind']=='ct6' else 'audit_result.json')),
                      ('products.npz',product)]:
        shutil.copy2(path,snapshot/name)
    if p['kind']=='ct6' or p.get('parent_format')=='CT_bounded_followup_v1':
        shutil.copy2(result_path,snapshot/'training_result.json')
        if p.get('parent_format')=='CT_bounded_followup_v1':
            shutil.copy2(src/'endpoint_frozen.json',snapshot/'endpoint_frozen.json')
    seals={x.name:file_sha256(x) for x in snapshot.iterdir()}
    save_state(out/'parent.pkl',state,0,absolute)
    return dict(parent_global_step=absolute,native_origin=origin,parent_seals=seals,
        parameter_count=sum(np.size(x) for x in jax.tree.leaves(state[0])),
        original_parameter_sha256=pytree_sha256(state[0]),original_optimizer_sha256=pytree_sha256(state[1]),
        original_rng_sha256=canonical_sha256(state[2]),parent_audit_consumed_only_for_continuation_safety=True)


def make_runtime(p:dict,state:tuple,reg:dict) -> tuple[dict,dict]:
    """Re-use unchanged complete-A1/full-H physics and all native columns."""
    if p['kind']=='ct12':
        kernel=dict(p,task='CT12_fixed_active_full_A1_IID_SR_two_step_v1',steps=2,damping=.5,
            target_FS=.003,maximum_FS=.003,wall_cap_seconds=14400,maximum_peak_device_bytes=70*1024**3,
            automatic_production=False,initial_endpoint='committed_A100_two_step_parent',
            seeds=dict(training=1,audit=2,statistics=3))
        rt=c12.runtime(kernel,state[0])
        # Preserve the A100-tested CT12 optimizer exactly on first B200 use.
        # checked_plan's .5 damping matches this closed followup profile.
        return rt,p
    kernel=dict(p,j2_over_j1=p['j2'],binary_rank=8,
        parent=dict(global_step=reg['native_origin']),updates=state[2]['next_step']+p['steps'],
        comparison_parent_global_step=reg['parent_global_step'],
        terminal_audit_records=p['audit_records'],interim_records=p['interim_records'])
    rt=c6.make_runtime(kernel,state[0])
    rt['native_update']=make_importance_amplitude_sr(lambda par,row:rt['projected'](rt['prepare'](par),row),
        state[0],damping=p['damping'],score_chunk=p['score_chunk'],parameter_chunk_size=8192,
        sr_tangent_fs_radius=.003,prepare=rt['prepare'],prepared_amplitude=rt['projected'])
    return rt,kernel


def normalize(p:dict,rt:dict,params:dict) -> dict:
    if p['kind']=='ct6':return c6.diagnostic.normalized_raw_bank(params,rt)[0]
    return c12.canonical_active_sampling_bank(params,rt['cfg'],rt['profile'])[0]


def replay_parent(p:dict,rt:dict,state:tuple,out:Path) -> dict:
    """Recompile on the allocated device and compare actual full-H products."""
    with np.load(out/'parent_snapshot/products.npz') as f:
        rows=f['rows'][:2]
        golden=(np.column_stack((f['psi'][:2,1],f['hpsi'][:2,1],f['guide_components'][:2,1,:]))
                if p['kind']=='ct6' else f['products'][:2,1,:])
    bank=normalize(p,rt,state[0])
    if p['kind']=='ct6':
        # normalize already folds a binary native bank into joint physical
        # tensors. Native rt.prepare would fold it AGAIN and require 'six'.
        prepared=c6.diagnostic.prepare_feature_lut_bank(bank,rt['cfg'])
        query=lambda x:rt['bridge'](prepared,x)
    else:query=lambda x:rt['H'](bank,x)
    tick=perf_counter();actual=np.stack([np.asarray(v) for v in jax.block_until_ready(query(rows))],axis=1)
    error=np.max(abs(actual-golden),axis=0)/np.maximum(np.max(abs(golden),axis=0),1e-300)
    if not np.isfinite(error).all() or error.max()>2e-11:raise ArithmeticError('device complete-A1/H parent replay failed')
    tick=perf_counter();jax.block_until_ready(query(rows));warm=(perf_counter()-tick)/len(rows)
    return dict(component_relative_errors=error.tolist(),warm_H_seconds=warm,full_H_centers=4,
        complete_A1_images=432 if p['kind']=='ct6' else 1728)


def transition(p:dict,kernel:dict,rt:dict,state:tuple,path:Path,*,replay:bool=False) -> tuple[tuple,dict]:
    if p['kind']=='ct12':return c12.transition(p,rt,state,path,replay=replay)
    cache=p.get('first_fit_cache') if path.name in ('step1','step1_replay') else None
    return c6.finite_step(kernel,rt,state,state[2]['next_step'],path,replay_fit_from=cache)


def check_allocator(p:dict,parameter_count:int) -> dict:
    """Fail before paid H work if the A100 pool cannot hold the full Jacobian.

    This is a necessary storage check, NOT a measured peak-memory guarantee.
    Reserving a contiguous pool does not change the existing live-byte cap.
    """
    required=p['fit_records']*parameter_count*8+3*1024**3
    pools=[int((d.memory_stats() or {}).get('bytes_limit',0)) for d in jax.local_devices()]
    if (required>p['maximum_peak_device_bytes'] or not pools or min(pools)<required):
        raise MemoryError('full native SR requires Jacobian plus3GiB inside both pool and live-byte ceiling')
    return dict(jacobian_bytes=p['fit_records']*parameter_count*8,required_pool_bytes=required,
        device_pool_limits=pools,live_byte_ceiling=p['maximum_peak_device_bytes'],
        preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        memory_fraction=os.environ.get('XLA_CLIENT_MEM_FRACTION',os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')),
        necessary_check_only_not_peak_guarantee=True)


def audit_pair(p:dict,kernel:dict,rt:dict,parent:tuple,final:tuple,path:Path,
               remaining,*,interim:bool=False) -> dict:
    count=p['interim_records'] if interim else p['audit_records']
    seed=p['seeds']['interim' if interim else 'audit'];stats=p['seeds']['interim_statistics' if interim else 'audit_statistics']
    if p['kind']=='ct6':return c6.paired_audit(kernel,rt,parent[0],final,path,count,seed,stats,remaining)
    banks=[normalize(p,rt,s[0]) for s in (parent,final)];path.mkdir()
    rows,draw=c12.exact_draw(banks,rt,count,seed,bridge=True)
    save_checkpoint(path/'draw.pkl',draw,metadata=dict(role='untouched_terminal'))
    products,warm=c12.products(rt,banks,rows,path,'untouched',p['H_chunk'])
    q=full_support_sz0_density(products[:,:,2].real.mean(1),products[:,:,3].real.mean(1),n_sites=144,epsilon=p['epsilon'])
    moments=paired_iid_moments(products[:,:,0],products[:,:,1],q,
        np.repeat(np.arange(p['audit_groups']),count//p['audit_groups']),n_sites=144,seed=stats)
    report=dict(status='complete',moments=moments,records=count,independent_from_training_and_geometry=True,
        endpoint_params_sha256=pytree_sha256(final[0]),warm_H_seconds=warm)
    atomic_json(path/'audit.json',report);return report


def run(plan_path:Path,out:Path,phase:str) -> dict:
    p=json.loads(plan_path.read_text());validate_plan(p)
    software_acceptance=require_software_acceptance(p)
    if phase not in ('prepare','replay','train','audit'):raise ValueError('unknown process phase')
    if jax.default_backend()!='gpu' or not all(p['device'] in d.device_kind for d in jax.devices()):
        raise RuntimeError('explicitly allocated '+p['device']+' GPU required')
    actual_stack=verify_production_runtime(p)
    stopping=[False]
    def stop(*_):stopping[0]=True
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop)
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    if phase=='prepare':
        if out.exists() and any(out.iterdir()):raise FileExistsError('fresh sealed output required')
        out.mkdir(parents=True,exist_ok=True)
        reg=dict(plan=p,plan_sha256=canonical_sha256(p),started_epoch=time(),sources=sources,
            software_acceptance=software_acceptance,
            runtime=dict(jax=jax.__version__,x64=jax.config.x64_enabled,
                validated_B200_stack=actual_stack,
                devices=[dict(kind=d.device_kind,compute_capability=str(getattr(d,'compute_capability','not_exposed')))
                         for d in jax.devices()]),**bind_parent(p,out))
        atomic_json(out/'registration.json',reg)
    reg=json.loads((out/'registration.json').read_text())
    if reg['plan']!=p or reg['sources']!=sources:raise ValueError('source or plan changed between processes')
    if reg.get('software_acceptance')!=software_acceptance:
        raise ValueError('accepted runtime certificate changed between training processes')
    for name,digest in reg['parent_seals'].items():
        if file_sha256(out/'parent_snapshot'/name)!=digest:raise ValueError('bound parent snapshot changed')
    def remaining():
        end=float(os.environ.get('SLURM_JOB_END_TIME',reg['started_epoch']+p['wall_cap_seconds']))
        return 0. if stopping[0] else min(p['wall_cap_seconds']-(time()-reg['started_epoch']),end-time()-240.)
    parent,_=load_state(out/'parent.pkl');rt,kernel=make_runtime(p,parent,reg);rt['remaining']=remaining
    if p.get('allocator_preflight'):
        atomic_json(out/('allocator_'+phase+'.json'),check_allocator(p,reg['parameter_count']))
    report=dict(phase=phase,status='started',completed_updates=0)
    phase_path=out/(phase+'_result.json')
    if phase_path.exists():return json.loads(phase_path.read_text())
    try:
        if phase=='prepare':
            replay=replay_parent(p,rt,parent,out);atomic_json(out/'device_replay.json',replay)
            warm=replay['warm_H_seconds'];reserve=1.5*2*p['audit_records']*warm+600
            atomic_json(out/'cost.json',dict(warm_H_seconds=warm,audit_reserve_seconds=reserve))
            rt.update(warm_H_seconds=warm,audit_reserve_seconds=reserve)
            if remaining()<reserve+2.6*(p['fit_records']*warm+600):
                report.update(status='cost_gate_parent_preserved',remaining=remaining(),audit_reserve=reserve)
            else:
                with c6.budget_timeout(remaining()-reserve):candidate,info=transition(p,kernel,rt,parent,out/'step1')
                save_state(out/'provisional.pkl',candidate,int(info['accepted']),reg['parent_global_step']+int(info['accepted']))
                report.update(status='first_update_provisional' if info['accepted'] else 'first_geometry_rejected',transition=info)
        else:
            prepared=json.loads((out/'prepare_result.json').read_text())
            if prepared['status']!='first_update_provisional':
                report.update(status='no_provisional_update_no_training');atomic_json(phase_path,report);return report
            cost=json.loads((out/'cost.json').read_text());rt.update(cost)
            if phase=='replay':
                expected,_=load_state(out/'provisional.pkl')
                with c6.budget_timeout(remaining()-cost['audit_reserve_seconds']):
                    actual,info=transition(p,kernel,rt,parent,out/('step1' if p['kind']=='ct12' else 'step1_replay'),replay=p['kind']=='ct12')
                comparison=c6.diagnostic.legacy.compare_replay(expected[:2],actual[:2],tolerance=2e-11)
                if not comparison['passed'] or canonical_sha256(expected[2])!=canonical_sha256(actual[2]):
                    raise ArithmeticError('full first-update parameter/optimizer/RNG replay failed')
                save_state(out/'checkpoint.pkl',expected,1,reg['parent_global_step']+1)
                report.update(status='replay_passed_committed_once',completed_updates=1,replay=comparison)
            elif phase=='train':
                replay=json.loads((out/'replay_result.json').read_text())
                if replay['status']!='replay_passed_committed_once':raise ValueError('process replay required')
                state,meta=load_state(out/'checkpoint.pkl');done=meta['completed_updates']
                last=prepared['transition']['wall_seconds'];reserve=cost['audit_reserve_seconds']
                warm_updates=[prepared['transition'].get('warm_H_seconds',cost['warm_H_seconds'])]
                for i in range(done+1,p['steps']+1):
                    if p['task']=='CT12_production32_v1':
                        # The actual first full-H panels, not GPU marketing or
                        # a cold two-row probe, calibrate the new-stack reserve.
                        warm=max(warm_updates[:3]);reserve=1.5*2*p['audit_records']*warm+600
                        rt.update(warm_H_seconds=warm,audit_reserve_seconds=reserve)
                        atomic_json(out/'throughput_calibration.json',dict(
                            observed_full_H_warm_seconds=warm_updates[:3],
                            audit_reserve_seconds=reserve,last_update_wall_seconds=last,
                            fixed_target_updates=p['steps'],no_acceleration_assumed=True))
                    extra=1.5*2*p['interim_records']*cost['warm_H_seconds']+300 if p['interim_records'] and i<=2 else 0.
                    if remaining()<reserve+extra+1.4*last:break
                    rt['audit_reserve_seconds']=reserve+extra
                    try:
                        with c6.budget_timeout(remaining()-reserve-extra):candidate,info=transition(p,kernel,rt,state,out/f'step{i}')
                    except (TimeoutError,MemoryError) as err:
                        report['interrupted_uncommitted_step']=repr(err);break
                    last=info['wall_seconds']
                    if p['task']=='CT12_production32_v1' and len(warm_updates)<3:
                        warm_updates.append(info.get('warm_H_seconds',rt['warm_H_seconds']))
                    if not info['accepted']:report['stop_reason']='finite_geometry_rejection';break
                    state=candidate;done=i;save_state(out/'checkpoint.pkl',state,done,reg['parent_global_step']+done)
                    atomic_json(out/'training_progress.json',dict(completed_updates=done,absolute_step=reg['parent_global_step']+done))
                    if i==2 and p['interim_records']:
                        try:
                            with c6.budget_timeout(remaining()-reserve):
                                check=audit_pair(p,kernel,rt,parent,state,out/'interim_audit',lambda:remaining()-reserve,interim=True)
                        except TimeoutError as err:
                            report['stop_reason']='interim_incomplete_terminal_reserve_preserved';break
                        safe=safe_to_continue(check);atomic_json(out/'interim_decision.json',dict(continue_training=safe,
                            consumed_as_safety_not_terminal_precision=True))
                        if not safe:report['stop_reason']='interim_resolved_harm';break
                report.update(status='fixed_prefix_frozen',completed_updates=done,absolute_step=reg['parent_global_step']+done,
                              requested_updates=p['steps'],fixed_target_completed=done==p['steps'])
                atomic_json(out/'endpoint_frozen.json',dict(report,parameter_sha256=pytree_sha256(state[0]),
                    checkpoint_sha256=file_sha256(out/'checkpoint.pkl')))
            else:
                frozen=json.loads((out/'endpoint_frozen.json').read_text());state,meta=load_state(out/'checkpoint.pkl')
                if pytree_sha256(state[0])!=frozen['parameter_sha256']:raise ValueError('frozen endpoint changed')
                rt['audit_reserve_seconds']=0.;rt['sampling_reserve_seconds']=120.
                final=audit_pair(p,kernel,rt,parent,state,out/'terminal_audit',remaining)
                report.update(status='complete_fixed_prefix_independent_audit',completed_updates=meta['completed_updates'],
                    terminal_audit=final,parameter_sha256=pytree_sha256(state[0]),checkpoint_sha256=file_sha256(out/'checkpoint.pkl'))
    except (TimeoutError,MemoryError) as err:
        report.update(status='bounded_interruption_committed_prefix_preserved',error=repr(err))
    atomic_json(phase_path,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','replay','train','audit'),required=True)
    a=parser.parse_args();r=run(a.plan,a.output,a.phase)
    print(json.dumps({k:r[k] for k in ('status','phase','completed_updates') if k in r}),flush=True)
    if r['status']=='bounded_interruption_committed_prefix_preserved':raise SystemExit(2)
