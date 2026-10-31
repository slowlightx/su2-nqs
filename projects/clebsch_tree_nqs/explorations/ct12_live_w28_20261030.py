"""Exact live CT12 active-bank inclusion to physical width28, no training.

Only legal doubled-q blocks are allocated. Inherited outputs ignore every
new child input; new child/output features are nonzero orthogonal completions.
The old state, moments and scientific RNG are preserved. This is an isolated
prepared-state artifact, not a production submission or an energy improvement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter,process_time

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.models.triangular_clebsch_active_layout import mixed_active_layout
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,mixed_clebsch_layout
from su2_nqs.su2.triangle_blocks import nine_site_output
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint

ROOT=Path(__file__).resolve().parents[3]


def orthogonal_additions(old:np.ndarray,additional:int,rng:np.random.Generator,scale:float=.2) -> np.ndarray:
    """Nonzero same-q columns perpendicular to inherited columns; old is untouched."""
    a=np.asarray(old)
    if (a.ndim!=2 or additional<0 or a.shape[1]+additional>a.shape[0]
            or not np.isfinite(a).all() or scale<=0):raise ValueError('finite tall matrix and legal new-copy count required')
    if additional==0:return np.zeros((len(a),0),dtype=a.dtype)
    q=np.linalg.qr(a,mode='reduced')[0]
    candidate=rng.normal(size=(len(a),additional))
    for _ in range(2):candidate-=q@(q.T@candidate)
    fresh,r=np.linalg.qr(candidate,mode='reduced')
    if np.min(abs(np.diag(r)))<1e-10:raise ArithmeticError('random orthogonal completion lost rank')
    return scale*fresh


def _unpack(top:tuple,layout:tuple) -> list[dict]:
    if len(top)!=len(layout):raise ValueError('one bank per upper merge required')
    result=[]
    for bank,row in zip(top,layout,strict=True):
        node={}
        for values,group in zip(bank,row['groups'],strict=True):
            if values.shape!=group['shape']:raise ValueError('source active shape does not match profile')
            node.update({key:np.asarray(value) for key,value in zip(group['keys'],values,strict=True)})
        result.append(node)
    return result


def expand_active_bank(source:dict,old_cfg:MixedClebschConfig,old_profile:tuple,
                       new_cfg:MixedClebschConfig,*,seed:int,scale:float=.2) -> tuple[dict,dict]:
    """Keep source axes exactly and open physical nonzero same-q feature columns."""
    if (old_cfg.length!=new_cfg.length or old_cfg.maximum_q!=new_cfg.maximum_q
            or new_cfg.multiplicity_cap<=old_cfg.multiplicity_cap or set(source)!={'local','top'}):
        raise ValueError('free folded bank, same tree/q cutoff and strictly wider cap required')
    old_layout=mixed_active_layout(old_cfg,old_profile);new_layout=mixed_clebsch_layout(new_cfg)
    old_top=_unpack(source['top'],old_layout);rng=np.random.default_rng(seed)
    local=[];local_checks=[]
    for b,node in enumerate(source['local']):
        values=[]
        for q,a in zip(old_cfg.local_q,node,strict=True):
            a=np.asarray(a);target=nine_site_output(q,new_cfg.multiplicity_cap).width
            fresh=orthogonal_additions(a,target-a.shape[1],rng,scale)
            values.append(np.concatenate((a,fresh),axis=1))
            local_checks.append(dict(node=b,q=q,old=a.shape[1],new=target,
                old_new_orthogonality_max=float(np.max(abs(a.T@fresh))) if fresh.size else 0.))
        local.append(tuple(values))
    top=[];rows=[]
    for i,(lookup,row,oldrow) in enumerate(zip(old_top,new_layout,old_layout,strict=True)):
        blocks={key:np.zeros(g['shape'][1:],np.float64) for g in row['groups'] for key in g['keys']}
        for key,a in lookup.items():
            if key not in blocks or any(s>t for s,t in zip(a.shape,blocks[key].shape,strict=True)):
                raise ValueError('target cannot include an old active block')
            blocks[key][tuple(slice(n) for n in a.shape)]=a
        for qo,newwidth in row['widths'].items():
            oldwidth=oldrow['widths'].get(qo,0);keys=sorted(key for key in blocks if key[2]==qo)
            inherited=np.concatenate([blocks[key][:oldwidth].reshape(oldwidth,-1).T for key in keys],axis=0)
            fresh=orthogonal_additions(inherited,newwidth-oldwidth,rng,scale)
            offset=0
            for key in keys:
                count=blocks[key].shape[1]*blocks[key].shape[2]
                blocks[key][oldwidth:]=fresh[offset:offset+count].T.reshape(newwidth-oldwidth,*blocks[key].shape[1:])
                offset+=count
            rows.append(dict(node=i,q=qo,old=oldwidth,new=newwidth,child_product_rows=len(inherited),
                             old_new_orthogonality_max=float(np.max(abs(inherited.T@fresh))) if fresh.size else 0.))
        top.append(tuple(np.stack([blocks[key] for key in group['keys']]) for group in row['groups']))
    target=dict(local=tuple(local),top=tuple(top))
    return target,dict(local=local_checks,upper=rows,
        same_state_by_old_output_new_child_zero_closure=True,
        new_outputs_are_nonzero_not_all_new_coordinates_first_order_live=True,
        root_q0_width1=True,maximum_q=new_cfg.maximum_q)


def lift_inherited_tree(old:dict,old_cfg:MixedClebschConfig,old_profile:tuple,
                         target_template:dict,new_cfg:MixedClebschConfig) -> dict:
    """Exact old coordinate/moment embedding; all genuinely new entries are zero."""
    old_layout=mixed_active_layout(old_cfg,old_profile);new_layout=mixed_clebsch_layout(new_cfg)
    topold=_unpack(old['top'],old_layout);local=[]
    for node,template in zip(old['local'],target_template['local'],strict=True):
        row=[]
        for a,t in zip(node,template,strict=True):
            value=np.zeros(t.shape,np.asarray(a).dtype);value[:a.shape[0],:a.shape[1]]=a;row.append(value)
        local.append(tuple(row))
    top=[]
    for lookup,row in zip(topold,new_layout,strict=True):
        groups=[]
        for group in row['groups']:
            array=np.zeros(group['shape'],np.float64)
            for i,key in enumerate(group['keys']):
                if key in lookup:
                    a=lookup[key];array[(i,*[slice(n) for n in a.shape])]=a
            groups.append(array)
        top.append(tuple(groups))
    return dict(local=tuple(local),top=tuple(top))


def projected_streaming_operators(batch_amplitude,theta:jax.Array,rows:jax.Array,
                                   psi:jax.Array,density:jax.Array):
    """Weighted amplitude JVP/VJP for existing LSQR, using F rather than2F.

    JVP produces only N complex tangents; a second sequential row scan
    accumulates a real VJP. The whole scientific record set is retained.
    Repeated SR matvecs are expensive and require a separate GPU timing gate.
    """
    if (theta.ndim!=1 or theta.dtype!=jnp.float64 or rows.ndim!=2
            or psi.shape!=(len(rows),) or density.shape!=psi.shape or density.dtype!=jnp.float64):
        raise ValueError('real flat coordinates, physical rows and aligned float64 positive Q required')
    for value in (psi,density):
        if not isinstance(value,jax.core.Tracer) and not np.isfinite(np.asarray(value)).all():raise ValueError('finite amplitudes/Q required')
    if not isinstance(density,jax.core.Tracer) and np.any(np.asarray(density)<=0):raise ValueError('strict full-support Q required')
    norm=jnp.sum(abs(psi)**2/density);scale=jax.lax.rsqrt(density*norm);u=psi*scale
    if not isinstance(norm,jax.core.Tracer) and float(norm)<=0:raise ValueError('positive sampled norm required')
    def forward(vector):
        jv=jax.lax.map(lambda row:jax.jvp(lambda x:batch_amplitude(x,row[None,:])[0],
                                        (theta,),(vector,))[1],rows)
        z=jv*scale;z=z-u*jnp.vdot(u,z)
        return jnp.concatenate((z.real,z.imag))
    def adjoint(real_weights):
        z=real_weights[:len(psi)]+1j*real_weights[len(psi):]
        z=z-u*jnp.vdot(u,z);weights=z*scale
        def pull(carry,items):
            row,w=items
            value=jax.grad(lambda x:jnp.real(jnp.conj(w)*batch_amplitude(x,row[None,:])[0]))(theta)
            return carry+value,None
        result,_=jax.lax.scan(pull,jnp.zeros_like(theta),(rows,weights))
        return result
    return forward,adjoint


def streamed_metric_matvec(batch_amplitude,theta:jax.Array,rows:jax.Array,psi:jax.Array,
                            density:jax.Array,vector:jax.Array,damping:float) -> jax.Array:
    """Executable Fv closure; use existing damped_lsqr on its forward/adjoint."""
    forward,adjoint=projected_streaming_operators(batch_amplitude,theta,rows,psi,density)
    return adjoint(forward(vector))+damping*vector


def run(parent:Path,registration:Path,output:Path) -> dict:
    """Create one preserved-state artifact and bounded144-spin amplitude/JVP checks."""
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('use a new immutable output directory')
    start=perf_counter();cpu=process_time()
    def stop(signum,frame):raise TimeoutError('fixed300-second local preflight limit')
    signal.signal(signal.SIGALRM,stop);signal.alarm(300)
    signal.signal(signal.SIGXCPU,stop);resource.setrlimit(resource.RLIMIT_CPU,(280,300))
    source=load_checkpoint(parent);plan=json.loads(registration.read_text())['plan']
    old_cfg=MixedClebschConfig(**plan['model']);new_cfg=MixedClebschConfig(12,28,10)
    profile=tuple({int(q):int(d) for q,d in row.items()} for row in plan['active_profile'])
    meta=source['metadata'];step=int(meta['absolute_completed_steps']);initial_seed=26103071
    reg=dict(task='CT12_same_state_actual_active_width28_preflight_no_optimization',parent=str(parent),
        parent_checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),parent_global_step=step,
        parent_parameter_sha256=pytree_sha256(source['params']),initialization_seed=initial_seed,
        scientific_RNG_not_consumed_for_expansion=True,local_target={1:28,3:28,5:27,7:8,9:1},
        upper_target='all14 nonroot nodes each legal q=0,2,4,6,8,10 width28',root={0:1},maximum_q=10,
        no_full_H_queries=True,no_updates=True,CPU_cap=300,wall_cap=300,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/'registration.json',reg);result=dict(registration=reg,status='started')
    def progress(stage):
        result['stage']=stage;result['wall_seconds']=perf_counter()-start;result['CPU_seconds']=process_time()-cpu
        atomic_json(output/'progress.json',result)
        if result['wall_seconds']>270 or result['CPU_seconds']>260:raise TimeoutError('save margin within300-second budget')
    try:
        target,note=expand_active_bank(source['params'],old_cfg,profile,new_cfg,seed=initial_seed)
        count=sum(a.size for a in jax.tree.leaves(target));old_count=sum(a.size for a in jax.tree.leaves(source['params']))
        result.update(expansion=note,old_parameters=old_count,target_parameters=count)
        progress('same_state_live_expansion_ready')
        opt=source['optimizer_state']
        moved=(deepcopy(opt[0]),lift_inherited_tree(opt[1],old_cfg,profile,target,new_cfg),
               lift_inherited_tree(opt[2],old_cfg,profile,target,new_cfg))
        rngcopy=deepcopy(meta['rng']);a=np.random.default_rng();a.bit_generator.state=deepcopy(meta['rng'])
        b=np.random.default_rng();b.bit_generator.state=deepcopy(rngcopy)
        rngmatch=bool(np.array_equal(a.integers(2**31,size=16),b.integers(2**31,size=16)))
        updated=dict(meta,parameter_sha256=pytree_sha256(target),optimizer_sha256=pytree_sha256(moved),
            rng=rngcopy,model=vars(new_cfg),active_profile=[dict(row['widths']) for row in mixed_clebsch_layout(new_cfg)],
            expansion_parent_sha256=reg['parent_checkpoint_sha256'],expansion_seed=initial_seed,
            optimizer_migration=dict(old_moments_copied_by_qkey_and_axis=True,new_moments_zero=True,
                inherited_counter=int(opt[0]),new_coordinate_moment_age=0,
                new_coordinate_birth_step=step,inherited_coordinate_birth_steps='not_inferred',
                current_SR_does_not_use_Adam_moments=True,
                future_Adam_requires_coordinate_age_bias_correction=True))
        save_checkpoint(output/'expanded_checkpoint.pkl',target,optimizer_state=moved,metadata=updated)
        result['migration']=dict(optimizer_counter=int(moved[0]),scientific_RNG_preserved=rngmatch,
            checkpoint_sha256=hashlib.sha256((output/'expanded_checkpoint.pkl').read_bytes()).hexdigest(),
            optimizer_sha256=updated['optimizer_sha256'],new_coordinate_moment_age=0)
        result['memory_plan']=dict(parameter_bytes=count*8,parameters_plus_two_moments_bytes=count*24,
            full_real_J_1024_GiB=1024*count*8/1024**3,full_complex_J_1024_GiB=1024*count*16/1024**3,
            full_real_J_2048_GiB=2048*count*8/1024**3,
            streamed_parameter_vectors_6_GiB=6*count*8/1024**3,
            no_dense_J_or_parameter_Fisher_permitted=True,
            row_streamed_JVP_VJP_available='streamed_metric_matvec',
            production_fit_records_not_silently_reduced=2048,
            matrix_free_runtime_and_A100_peak_not_yet_certified=True)
        progress('checkpoint_optimizer_RNG_migrated')
        del moved
        from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank,make_feature_lut_amplitude,make_feature_lut_projected_amplitude
        from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
        oldraw=make_feature_lut_amplitude(old_cfg,active_top_widths=profile);newraw=make_feature_lut_amplitude(new_cfg)
        oldprepared=prepare_feature_lut_bank(source['params'],old_cfg);newprepared=prepare_feature_lut_bank(target,new_cfg)
        rng=np.random.default_rng(26103072);rows=np.asarray([rng.permutation(np.r_[np.zeros(72,np.int32),np.ones(72,np.int32)]) for _ in range(4)])
        oldeval=jax.jit(jax.vmap(oldraw,in_axes=(None,0)));neweval=jax.jit(jax.vmap(newraw,in_axes=(None,0)))
        tick=perf_counter();before=np.asarray(oldeval(oldprepared,jnp.asarray(rows)));oldtime=perf_counter()-tick
        tick=perf_counter();after=np.asarray(neweval(newprepared,jnp.asarray(rows)));newtime=perf_counter()-tick
        relative=float(np.linalg.norm(after-before)/max(np.linalg.norm(before),np.finfo(float).tiny))
        if relative>2e-11:raise ArithmeticError('actual144-spin raw state was not preserved')
        result['raw_check']=dict(relative_error=relative,old_four_rows_seconds=oldtime,new_four_rows_seconds=newtime,
            old_amplitudes=before.tolist(),new_amplitudes=after.tolist())
        progress('four_actual_raw_amplitudes_preserved')
        # A root gate is first-order live because its new child features exist.
        root=target['top'][-1];layout=mixed_clebsch_layout(new_cfg);rootrow=layout[-1]
        chosen=next((gi,ki,key) for gi,g in enumerate(rootrow['groups']) for ki,key in enumerate(g['keys']) if key==(2,2,0))
        gi,ki,key=chosen;leftold=mixed_active_layout(old_cfg,profile)[rootrow['left']-16]['widths'][2]
        gate=np.zeros_like(root[gi]);gate[ki,0,leftold,0]=1.
        def root_line(t,x):
            groups=tuple(jnp.asarray(w)+(t*jnp.asarray(gate) if i==gi else 0.) for i,w in enumerate(root))
            bank=dict(newprepared,top=(*newprepared['top'][:-1],groups));return newraw(bank,x)
        live=np.asarray(jax.jit(jax.vmap(lambda x:jax.jvp(lambda t:root_line(t,x),(jnp.asarray(0.),),(jnp.asarray(1.),))[1]))(jnp.asarray(rows)))
        result['new_root_gate_JVP']=dict(q=2,new_left_copy=leftold,old_right_copy=0,
            raw_derivatives=live.tolist(),raw_norm=float(np.linalg.norm(live)),nonzero=bool(np.any(live!=0)))
        if not np.any(live!=0):raise ArithmeticError('registered new downstream gate has zero sampled tangent')
        progress('new_gate_first_order_live_on_actual_rows')
        images=np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32)
        oldproj=jax.jit(make_feature_lut_projected_amplitude(old_cfg,images,active_top_widths=profile))
        newproj=jax.jit(make_feature_lut_projected_amplitude(new_cfg,images))
        tick=perf_counter();p0=float(oldproj(oldprepared,jnp.asarray(rows[0])));t0=perf_counter()-tick
        tick=perf_counter();p1=float(newproj(newprepared,jnp.asarray(rows[0])));t1=perf_counter()-tick
        error=abs(p1-p0)/max(abs(p0),np.finfo(float).tiny)
        if error>3e-10:raise ArithmeticError('complete1728-image amplitude inclusion failed')
        result['complete_A1_check']=dict(images=len(images),old_amplitude=p0,new_amplitude=p1,
            relative_error=error,old_first_row_seconds=t0,new_first_row_seconds=t1,full_H_computed=False)
        def projected_gate(amplitude,prepared,layout,key,leftcopy,rightcopy):
            row=layout[-1];gi,ki=next((gi,ki) for gi,g in enumerate(row['groups'])
                                    for ki,k in enumerate(g['keys']) if k==key)
            gate=np.zeros_like(prepared['top'][-1][gi]);gate[ki,0,leftcopy,rightcopy]=1.
            def line(bank,x,t):
                root=tuple(w+(t*jnp.asarray(gate) if i==gi else 0.) for i,w in enumerate(bank['top'][-1]))
                return amplitude(dict(bank,top=(*bank['top'][:-1],root)),x)
            derivative=jax.jit(lambda bank,x:jax.jvp(lambda t:line(bank,x,t),
                (jnp.asarray(0.),),(jnp.asarray(1.),))[1])
            tick=perf_counter();value=float(derivative(prepared,jnp.asarray(rows[0])))
            return value,perf_counter()-tick
        newgate,newgatetime=projected_gate(newproj,newprepared,mixed_clebsch_layout(new_cfg),(2,2,0),leftold,0)
        inherited0,tangenttime0=projected_gate(oldproj,oldprepared,mixed_active_layout(old_cfg,profile),(0,0,0),0,0)
        inherited1,tangenttime1=projected_gate(newproj,newprepared,mixed_clebsch_layout(new_cfg),(0,0,0),0,0)
        tangenterror=abs(inherited1-inherited0)/max(abs(inherited0),np.finfo(float).tiny)
        result['projected_JVP_check']=dict(new_q2_gate_derivative=newgate,new_gate_nonzero=newgate!=0,
            new_gate_first_seconds=newgatetime,inherited_q0_old_derivative=inherited0,
            inherited_q0_new_derivative=inherited1,inherited_relative_error=tangenterror,
            inherited_old_first_seconds=tangenttime0,inherited_new_first_seconds=tangenttime1)
        if newgate==0 or tangenterror>3e-10:raise ArithmeticError('complete_A1 live/inherited tangent gate failed')
        np.savez_compressed(output/'actual_amplitude_checks.npz',rows=rows,old=before,new=after,new_gate_JVP=live)
        result['status']='complete_same_state_w28_checkpoint_and_actual_amplitude_preflight_no_updates'
    except TimeoutError as error:
        result['status']='bounded_partial_no_training';result['error']=str(error)
    finally:
        signal.alarm(0);result['wall_seconds']=perf_counter()-start;result['CPU_seconds']=process_time()-cpu
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_RSS_GiB']=rss/(1024**3 if sys.platform=='darwin' else 1024**2)
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.parent,a.registration,a.output)
