"""Isolated, zero-update CT12 w28 GPU cost/replay preflight.

Four immutable physical rows diagnose execution only. No scientific fit,
direction decision, energy estimate, optimizer step, or RNG consumption occurs.
Each expensive phase is a child process with an enforced wall deadline.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter,time
from typing import Any,Callable

ROOT=Path(__file__).resolve().parents[3]
TASK='CT12_w28_GPU_cost_replay_preflight_v1'
REQUIRED=('amplitude','full_H','derivatives','fisher')


def digest(path:Path) -> str:
    """Return the SHA256 of one immutable input."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def validate_plan(plan:dict,*,executing:bool=False) -> None:
    """Validate fixed doubled labels, true active width, and zero-update scope."""
    expected=dict(task=TASK,length=12,j2=.125,target_q=0,target_m2=0,maximum_q=10,
        multiplicity_cap=28,active_real_parameters=20748160,complete_A1_images=1728,
        physical_H_bonds=864,rows=4,row_indices=[0,1,2,3],maximum_updates=0,
        scientific_fit_records_consumed=0,scientific_RNG_consumed=False,
        wall_cap_seconds=14400,maximum_peak_device_bytes=70*1024**3,
        nodelist=None,full_dense_Jacobian_allowed=False)
    if any(plan.get(key)!=value for key,value in expected.items()):
        raise ValueError('fixed true-w28 zero-update replay/cost contract changed')
    if plan.get('required_phases')!=list(REQUIRED):raise ValueError('all four required phases must remain registered')
    if set(plan.get('phase_caps_seconds',{}))!=set((*REQUIRED,'H_JVP','full_H_four')):
        raise ValueError('explicit separate H/JVP and phase caps required')
    if any(not isinstance(v,int) or isinstance(v,bool) or v<60 for v in plan['phase_caps_seconds'].values()):
        raise ValueError('positive bounded phase caps required')
    if plan.get('projection_chunk') not in (4,8,16) or 1728%plan['projection_chunk']:
        raise ValueError('complete dividing A1 projection chunk required')
    if executing and plan.get('execution_authorized') is not True:
        raise PermissionError('root execution release is required')


def relative_error(actual:Any,expected:Any) -> float:
    """Natural vector-relative comparison without a unit-scale floor."""
    import numpy as np
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ArithmeticError('nonfinite or shape-mismatched replay vectors')
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))


def load_inputs(plan:dict) -> tuple:
    """Verify original/expanded native state identities before GPU queries."""
    import jax
    jax.config.update('jax_enable_x64',True)
    import numpy as np
    from su2_nqs.frontier.artifacts import canonical_sha256,pytree_sha256
    from su2_nqs.vmc.checkpoint import load_checkpoint
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,mixed_clebsch_layout
    from .ct12_live_w28_20261030 import lift_inherited_tree
    for name,spec in plan['inputs'].items():
        if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('input SHA mismatch: '+name)
    old=load_checkpoint(ROOT/plan['inputs']['parent']['path'])
    wide=load_checkpoint(ROOT/plan['inputs']['expanded']['path'])
    reg=json.loads((ROOT/plan['inputs']['parent_registration']['path']).read_text())
    profile=tuple({int(q):int(d) for q,d in node.items()} for node in reg['plan']['active_profile'])
    oldcfg=MixedClebschConfig(**reg['plan']['model']);newcfg=MixedClebschConfig(12,28,10)
    for name,state in (('parent',old),('expanded',wide)):
        m=state['metadata']
        if (pytree_sha256(state['params'])!=m['parameter_sha256']
                or pytree_sha256(state['optimizer_state'])!=m['optimizer_sha256']
                or canonical_sha256(m['rng'])!=m['rng_sha256']
                or m['absolute_completed_steps']!=45):raise ValueError('native state seal mismatch: '+name)
    if old['metadata']['rng']!=wide['metadata']['rng']:raise ValueError('expansion consumed scientific RNG')
    if int(old['optimizer_state'][0])!=int(wide['optimizer_state'][0]):raise ValueError('optimizer counter changed')
    for state in (old,wide):
        generator=np.random.default_rng();generator.bit_generator.state=deepcopy(state['metadata']['rng'])
        probe=generator.integers(2**32,size=16,dtype=np.uint32)
        if state is old:first=probe
        elif not np.array_equal(first,probe):raise ValueError('next scientific RNG replay differs')
    if sum(x.size for x in jax.tree.leaves(wide['params']))!=plan['active_real_parameters']:
        raise ValueError('true active w28 parameter count differs')
    if [row['widths'] for row in mixed_clebsch_layout(newcfg)]!=[{q:28 for q in range(0,11,2)}]*14+[{0:1}]:
        raise ValueError('nonroot doubled-q28 blocks or singlet root differ')
    for index in (1,2):
        lifted=lift_inherited_tree(old['optimizer_state'][index],oldcfg,profile,wide['params'],newcfg)
        if any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(lifted),jax.tree.leaves(wide['optimizer_state'][index]),strict=True)):
            raise ValueError('old moment axes or new zero moments changed')
    with np.load(ROOT/plan['inputs']['rows']['path']) as f:rows=f['rows'][plan['row_indices']]
    if rows.shape!=(4,144) or not np.all((rows==0)|(rows==1)) or not np.all(rows.sum(1)==72):
        raise ValueError('four fixed physical Sz0 rows required')
    return old,wide,oldcfg,newcfg,profile,rows


def make_runtime(cfg:Any,profile:Any,plan:dict) -> dict:
    """Complete coherent A1 and full physical H, with no training callback."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank,make_feature_lut_projected_amplitude
    from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
    from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products
    images=np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32)
    bonds=triangular_j1j2_bonds(12,12,j2=.125,periodic=True)
    if len(images)!=1728 or len(bonds)!=864:raise ArithmeticError('full physical projector/H identity differs')
    phi=make_feature_lut_projected_amplitude(cfg,images,chunk=plan['projection_chunk'],active_top_widths=profile)
    prepare=lambda p:prepare_feature_lut_bank(p,cfg)
    amplitude=jax.jit(lambda p,rows:jax.lax.map(lambda row:phi(prepare(p),row),rows))
    probabilities=np.abs([w for _,_,w in bonds]);probabilities/=probabilities.sum()
    bridge,metadata=make_orbit_exchange_bridge_products(
        lambda edges:make_feature_lut_exchange_queries(cfg,edges,active_top_widths=profile),
        images,bonds,probabilities,n_sites=144,projection_chunk=plan['projection_chunk'])
    H=jax.jit(lambda p,rows:bridge(prepare(p),rows))
    return dict(amplitude=amplitude,H=H,metadata=metadata)


def timed(call:Callable) -> tuple[Any,float]:
    """Measure completed device work, including compilation on the first call."""
    import jax
    start=perf_counter();value=jax.block_until_ready(call())
    return value,perf_counter()-start


def root_direction(params:dict,cfg:Any,*,new_copy:int|None,profile:Any=None) -> dict:
    """A fixed unit native root coordinate in legal doubled-q2 or q0."""
    import jax
    import jax.numpy as jnp
    from su2_nqs.models.triangular_mixed_clebsch import mixed_clebsch_layout
    from su2_nqs.models.triangular_clebsch_active_layout import mixed_active_layout
    target=(2,2,0) if new_copy is not None else (0,0,0)
    row=(mixed_clebsch_layout(cfg) if profile is None else mixed_active_layout(cfg,profile))[-1]
    gi,ki=next((gi,ki) for gi,g in enumerate(row['groups']) for ki,key in enumerate(g['keys']) if key==target)
    result=jax.tree.map(jnp.zeros_like,params);top=list(result['top']);root=list(top[-1])
    root[gi]=root[gi].at[ki,0,0 if new_copy is None else new_copy,0].set(1.)
    top[-1]=tuple(root);result['top']=tuple(top)
    return result


def inherited_local_direction(params:dict) -> dict:
    """Unit first-node doubled-q1 inherited matrix coordinate, not a frozen LUT."""
    import jax
    import jax.numpy as jnp
    result=jax.tree.map(jnp.zeros_like,params)
    nodes=list(result['local']);first=list(nodes[0]);first[0]=first[0].at[0,0].set(1.)
    nodes[0]=tuple(first);result['local']=tuple(nodes)
    return result


def phase(plan:dict,name:str,output:Path) -> dict:
    """Run one independently timed GPU phase, with no scientific state writes."""
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree
    from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256
    from .ct12_live_w28_20261030 import projected_streaming_operators
    if jax.default_backend()!='gpu' or not any('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('an allocated Kim A100 is required; no CPU execution fallback')
    old,wide,oldcfg,cfg,profile,rows=load_inputs(plan)
    rows=jnp.asarray(rows);native=wide['params']
    oldrt=make_runtime(oldcfg,profile,plan);rt=make_runtime(cfg,None,plan)
    report=dict(phase=name,updates=0,scientific_fit_records_consumed=0,
        fixed_diagnostic_rows=4,not_a_population_Fisher_or_direction_audit=True,
        parent_and_expanded_next_rng_sha256=canonical_sha256(wide['metadata']['rng']),
        device=[d.device_kind for d in jax.devices()],results={})
    report['runtime_versions']={key:importlib.metadata.version(key) for key in ('jax','jaxlib','numpy','scipy')}
    def record(label:str,fn:Callable) -> Any:
        value,seconds=timed(fn)
        stats=[d.memory_stats() for d in jax.devices()]
        if any(not item or 'peak_bytes_in_use' not in item or int(item['peak_bytes_in_use'])<=0 for item in stats):
            raise RuntimeError('CUDA allocator peak counter unavailable: memory gate uncertified')
        peak=max(int(item['peak_bytes_in_use']) for item in stats)
        report['results'][label]=dict(seconds=seconds,peak_device_bytes=peak)
        atomic_json(output/(name+'_progress.json'),report)
        if peak>plan['maximum_peak_device_bytes']:raise MemoryError('measured device allocation exceeds70GiB ceiling')
        return value
    if name=='amplitude':
        before=record('parent_full_A1_cold',lambda:oldrt['amplitude'](old['params'],rows))
        record('parent_full_A1_warm',lambda:oldrt['amplitude'](old['params'],rows))
        after=record('wide_full_A1_cold',lambda:rt['amplitude'](native,rows))
        record('wide_full_A1_warm',lambda:rt['amplitude'](native,rows))
        error=relative_error(after,before)
        if error>plan['replay_tolerance']:raise ArithmeticError('full1728-image preserved-state replay failed')
        report.update(relative_error=error,parent=np.asarray(before).tolist(),wide=np.asarray(after).tolist())
        np.savez_compressed(output/'amplitude_fixture.npz',rows=np.asarray(rows),psi=np.asarray(after),parent=np.asarray(before))
    elif name in ('full_H','full_H_four'):
        use=rows[:1] if name=='full_H' else rows
        before=record('parent_complete_H_cold',lambda:oldrt['H'](old['params'],use))
        record('parent_complete_H_warm',lambda:oldrt['H'](old['params'],use))
        after=record('wide_complete_H_cold',lambda:rt['H'](native,use))
        record('wide_complete_H_warm',lambda:rt['H'](native,use))
        errors=[relative_error(a,b) for a,b in zip(after,before,strict=True)]
        if max(errors)>plan['replay_tolerance']:raise ArithmeticError('complete physical H four-product replay failed')
        report.update(component_relative_errors=errors,centers=len(use),complete_A1_images=1728,physical_H_bonds=864,
            component_names=['coherent_psi','complete_Hpsi','incoherent_Qorbit','exchange_Qpush'],
            H_centers_counting_cold_and_warm_both_states=4*len(use),energy_not_estimated=True)
        np.savez_compressed(output/(name+'_products.npz'),rows=np.asarray(use),parent=np.asarray(before),wide=np.asarray(after))
    elif name in ('derivatives','fisher'):
        theta,unravel=ravel_pytree(native)
        batch=lambda flat,x:rt['amplitude'](unravel(flat),x)
        rootrow=__import__('su2_nqs.models.triangular_mixed_clebsch',fromlist=['mixed_clebsch_layout']).mixed_clebsch_layout(cfg)[-1]
        newcopy=profile[rootrow['left']-16][2]
        vector,_=ravel_pytree(root_direction(native,cfg,new_copy=newcopy))
        if name=='derivatives':
            f=lambda x:batch(x,rows[:1])[0]
            jvp=jax.jit(lambda x,v:jax.jvp(f,(x,),(v,))[1])
            value=record('single_direction_full_A1_JVP_cold',lambda:jvp(theta,vector))
            record('single_direction_full_A1_JVP_warm',lambda:jvp(theta,vector))
            vjp=jax.jit(jax.grad(f))
            gradient=record('full_native_VJP_cold',lambda:vjp(theta))
            record('full_native_VJP_warm',lambda:vjp(theta))
            dual=float(jnp.vdot(vector,gradient).real)
            error=relative_error(np.asarray([value]),np.asarray([dual]))
            if error>plan['derivative_tolerance']:raise ArithmeticError('full native JVP/VJP duality failed')
            localvector,_=ravel_pytree(inherited_local_direction(native))
            newlocal=record('inherited_local_full_A1_JVP_wide',lambda:jvp(theta,localvector))
            oldlocaldir=inherited_local_direction(old['params'])
            oldlocal=jax.jit(lambda p,d:jax.jvp(lambda x:oldrt['amplitude'](x,rows[:1])[0],(p,),(d,))[1])
            beforelocal=record('inherited_local_full_A1_JVP_parent',lambda:oldlocal(old['params'],oldlocaldir))
            localerror=relative_error(np.asarray([newlocal]),np.asarray([beforelocal]))
            localdual=relative_error(np.asarray([newlocal]),np.asarray([jnp.vdot(localvector,gradient).real]))
            if max(localerror,localdual)>plan['derivative_tolerance']:
                raise ArithmeticError('inherited local derivative or native preparation pullback lost')
            report.update(JVP=float(value),VJP_inner_product=dual,duality_relative_error=error,
                full_native_gradient_elements=int(gradient.size),gradient_norm=float(jnp.linalg.norm(gradient)),
                registered_new_q2_root_copy=int(newcopy),explicit_N_by_P_Jacobian_allocated=False,
                inherited_local_JVP_wide=float(newlocal),inherited_local_JVP_parent=float(beforelocal),
                inherited_local_relative_error=localerror,inherited_local_VJP_duality_relative_error=localdual,
                prepare_feature_LUT_inside_native_AD=True)
        else:
            psi=record('four_full_A1_values',lambda:batch(theta,rows))
            Q=jnp.ones(4,jnp.float64)
            def operator(x,v):
                forward,adjoint=projected_streaming_operators(batch,x,rows,psi,Q)
                Av=forward(v)
                return adjoint(Av),Av
            apply=jax.jit(operator)
            Fv,Av=record('streamed_Fisher_Fv_cold',lambda:apply(theta,vector))
            record('streamed_Fisher_Fv_warm',lambda:apply(theta,vector))
            lhs=float(jnp.vdot(vector,Fv).real);rhs=float(jnp.vdot(Av,Av).real)
            error=relative_error(np.asarray([lhs]),np.asarray([rhs]))
            if error>plan['derivative_tolerance'] or lhs<0:raise ArithmeticError('finite-panel F=A* A identity failed')
            report.update(vFv=lhs,norm_Av_squared=rhs,quadratic_identity_relative_error=error,
                Fv_norm=float(jnp.linalg.norm(Fv)),F_not_two_F=True,full_dense_Jacobian_allocated=False,
                constant_uniform_Q_on_four_registered_rows_not_population_sampling=True,
                full_1024_or_2048_SR_solve_not_performed=True)
    elif name=='H_JVP':
        olddir=root_direction(old['params'],oldcfg,new_copy=None,profile=profile)
        newdir=root_direction(native,cfg,new_copy=None)
        query0=jax.jit(lambda p,d:jax.jvp(lambda x:oldrt['H'](x,rows[:1]),(p,),(d,))[1])
        query1=jax.jit(lambda p,d:jax.jvp(lambda x:rt['H'](x,rows[:1]),(p,),(d,))[1])
        before=record('parent_inherited_complete_H_JVP_cold',lambda:query0(old['params'],olddir))
        after=record('wide_inherited_complete_H_JVP_cold',lambda:query1(native,newdir))
        record('wide_inherited_complete_H_JVP_warm',lambda:query1(native,newdir))
        errors=[relative_error(a,b) for a,b in zip(after,before,strict=True)]
        if max(errors)>plan['derivative_tolerance']:raise ArithmeticError('inherited complete-H JVP replay failed')
        report.update(component_relative_errors=errors,primal_physical_centers=1,
            tangent_H_queries_separately_charged=True)
    else:raise ValueError('unknown bounded phase')
    if wide['metadata']['rng']!=old['metadata']['rng']:raise ArithmeticError('scientific RNG mutated')
    report['status']='complete_zero_update_phase'
    atomic_json(output/(name+'_result.json'),report)
    return report


def run(plan:dict,plan_path:Path,output:Path) -> dict:
    """Enforce the four-hour wall cap across child GPU compilation/execution."""
    validate_plan(plan,executing=True)
    if output.exists():raise FileExistsError('new immutable preflight output required')
    output.mkdir(parents=True)
    start=time();deadline=min(start+plan['wall_cap_seconds'],float(os.environ.get('SLURM_JOB_END_TIME',start+plan['wall_cap_seconds'])))
    stop=[False]
    def requested_stop(*_:Any) -> None:stop[0]=True
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,requested_stop)
    result=dict(task=TASK,started_epoch=start,deadline_epoch=deadline,updates=0,
        required_phases=list(REQUIRED),phases={},scientific_fit_records_consumed=0,
        status='started_zero_update_cost_preflight',checkpoint_reference=plan['inputs']['expanded'])
    def save() -> None:
        result['wall_seconds']=time()-start
        temporary=output/'results.json.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output/'results.json')
    (output/'registration.json').write_text(json.dumps(dict(plan=plan,plan_sha256=digest(plan_path)),indent=2)+'\n')
    (output/'checkpoint_reference.json').write_text(json.dumps(dict(
        unchanged_global_step=45,scientific_rng_unconsumed=True,optimizer_unmodified=True,
        immutable_expanded_checkpoint=plan['inputs']['expanded'],immutable_parent=plan['inputs']['parent']),indent=2)+'\n')
    save()
    names=list(REQUIRED)+(['full_H_four'] if plan['optional_full_H_four'] else [])+(['H_JVP'] if plan['optional_H_JVP'] else [])
    for name in names:
        available=deadline-time()-plan['save_reserve_seconds']
        if stop[0] or available<plan['minimum_phase_launch_seconds']:
            result.update(status='bounded_partial_preflight_stopped_no_updates',stop_reason='wall_or_signal_before_'+name);break
        timeout=min(available,plan['phase_caps_seconds'][name])
        command=[sys.executable,'-u','-m',__name__ if __name__!='__main__' else
            'projects.clebsch_tree_nqs.explorations.ct12_w28_gpu_preflight_20261030',
            '--plan',str(plan_path),'--output',str(output),'--phase',name]
        phase_start=time()
        with (output/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            while child.poll() is None:
                if stop[0] or time()-phase_start>timeout:
                    timed_out=True;os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                    break
                try:child.wait(timeout=min(2.,max(.05,timeout-(time()-phase_start))))
                except subprocess.TimeoutExpired:pass
        result['phases'][name]=dict(returncode=child.returncode,wall_seconds=time()-phase_start,timed_out=timed_out)
        if timed_out:
            result.update(status='bounded_partial_preflight_stopped_no_updates',stop_reason='phase_deadline_'+name);save();break
        if child.returncode:
            text=(output/(name+'.log')).read_text(errors='replace')[-12000:]
            result.update(status='failed_preflight_no_updates',stop_reason=name,
                failure_class='device_OOM' if any(s in text for s in ('RESOURCE_EXHAUSTED','out of memory','CUDA_ERROR_OUT_OF_MEMORY')) else 'phase_error',
                no_bottom_width_or_dense_storage_cause_inferred_without_evidence=True)
            save();return result
        result['phases'][name]['report']=json.loads((output/(name+'_result.json')).read_text());save()
    else:result['status']='complete_bounded_GPU_cost_preflight_zero_updates'
    result['full_training_and_scientific_direction_still_uncertified']=True
    result['scientific_RNG_consumed']=False
    save();return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=(*REQUIRED,'full_H_four','H_JVP'))
    args=parser.parse_args();plan=json.loads(args.plan.read_text());validate_plan(plan,executing=bool(args.phase))
    if args.phase:result=phase(plan,args.phase,args.output)
    else:result=run(plan,args.plan,args.output)
    print(json.dumps(result,indent=2))
    if result['status']=='failed_preflight_no_updates':raise SystemExit(2)


if __name__=='__main__':main()
