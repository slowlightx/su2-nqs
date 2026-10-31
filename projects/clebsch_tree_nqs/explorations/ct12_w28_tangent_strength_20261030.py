"""Fixed projected-gate strength and H-independent new-feature Gram matching.

Both unoccupied-feature gauges are retained. No old coefficient, moment,
scientific RNG, physical state or energy is used to choose a gauge.
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
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,mixed_clebsch_layout
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint


def match_new_Gram_RMS(bank:dict,config:MixedClebschConfig,old_local_widths:dict,
                       old_profile:tuple) -> tuple[dict,dict]:
    """Scale only unoccupied new columns by exact one-magnetic-copy q-Grams."""
    local=[];grams=[];records=[]
    for b,node in enumerate(bank['local']):
        values=[];gram={}
        for q,w in zip(config.local_q,node,strict=True):
            a=np.array(w,copy=True);k=old_local_widths[q];g=a.T@a
            oldrms=float(np.sqrt(np.diag(g)[:k].mean()));factor=1.
            if k<a.shape[1]:
                newrms=float(np.sqrt(np.diag(g)[k:].mean()))
                if newrms<=0:raise ArithmeticError('zero physical new feature cannot be repaired by scaling')
                factor=oldrms/newrms;a[:,k:]*=factor;g=a.T@a
            values.append(a);gram[q]=g
            records.append(dict(layer='local',node=b,q=q,old=k,new=a.shape[1],factor=factor,
                old_RMS=oldrms,new_RMS_after=float(np.sqrt(np.diag(g)[k:].mean())) if k<a.shape[1] else None))
        local.append(tuple(values));grams.append(gram)
    layout=mixed_clebsch_layout(config);top=[]
    for i,(row,bankrow) in enumerate(zip(layout,bank['top'],strict=True)):
        values=[np.array(w,copy=True) for w in bankrow]
        lookup={key:a for group,array in zip(row['groups'],values,strict=True)
                for key,a in zip(group['keys'],array,strict=True)}
        out={}
        for q,d in row['widths'].items():
            g=np.zeros((d,d))
            for (ql,qr,qo),w in lookup.items():
                if qo==q:g+=np.einsum('alr,buv,lu,rv->ab',w,w,grams[row['left']][ql],grams[row['right']][qr],optimize=True)
            k=old_profile[i][q];oldrms=float(np.sqrt(np.diag(g)[:k].mean()));factor=1.
            if k<d:
                newrms=float(np.sqrt(np.diag(g)[k:].mean()))
                if newrms<=0:raise ArithmeticError('zero physical upper new feature cannot be repaired by scaling')
                factor=oldrms/newrms
                for key,w in lookup.items():
                    if key[2]==q:w[k:]*=factor
                scale=np.r_[np.ones(k),np.full(d-k,factor)];g*=scale[:,None]*scale[None,:]
            out[q]=g
            records.append(dict(layer='upper',node=i,q=q,old=k,new=d,factor=factor,old_RMS=oldrms,
                new_RMS_after=float(np.sqrt(np.diag(g)[k:].mean())) if k<d else None,
                new_feature_minimum_Gram_diagonal=float(np.min(np.diag(g)[k:])) if k<d else None))
        top.append(tuple(values));grams.append(out)
    return dict(local=tuple(local),top=tuple(top)),dict(records=records,root_raw_norm=float(grams[-1][0][0,0]),
        no_old_W_changed=True,no_H_labels_or_state_quality_selection=True)


def ray_strength(psi:np.ndarray,tangent:np.ndarray) -> dict:
    """Four-row, constant-Q direct-amplitude metric; not population certification."""
    p,d=np.asarray(psi),np.asarray(tangent);norm=float(np.vdot(p,p).real)
    relative=d/np.sqrt(norm);unit=p/np.sqrt(norm);horizontal=relative-unit*np.vdot(unit,relative)
    return dict(JVP_norm_over_parent_norm=float(np.linalg.norm(relative)),
        horizontal_Fisher=float(np.vdot(horizontal,horizontal).real),
        horizontal_FS_per_unit_coordinate=float(np.linalg.norm(horizontal)),
        parent_norm_ESS=float(norm**2/np.sum(abs(p)**4)),
        maximum_parent_norm_fraction=float(np.max(abs(p)**2)/norm),
        tangent=tangent.tolist(),constant_uniform_Q_cancels=True,
        samples=len(p),physical_population_Fisher_not_certified=True)


def run(source:Path,old_registration:Path,output:Path) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('new directory required')
    start=perf_counter();cpu=process_time()
    def stop(signum,frame):raise TimeoutError('fixed140-second completion budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(140)
    signal.signal(signal.SIGXCPU,stop);resource.setrlimit(resource.RLIMIT_CPU,(120,140))
    reg=dict(task='fixed_A1_gate_strength_and_new_only_physical_Gram_scale_branch',
        source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        branches=['original_scale_0p2','new_only_qwise_physical_Gram_RMS_matched'],
        scale_chosen_from_old_new_Gram_only=True,energy_queries=0,optimization_updates=0,
        gate_coordinates='root (qleft,qright,qroot)=(2,2,0), old_left0 vs first_new_left8, same right0; unit Frobenius',
        row_source='same4 registered uniform_M0 rows from base expansion probe; no selection',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/'registration.json',reg);result=dict(registration=reg,status='started')
    try:
        saved=load_checkpoint(source);cfg=MixedClebschConfig(12,28,10)
        plan=json.loads(old_registration.read_text())['plan'];oldlocal={1:20,3:20,5:20,7:8,9:1}
        profile=tuple({int(q):int(d) for q,d in row.items()} for row in plan['active_profile'])
        rows=np.load(source.parent/'actual_amplitude_checks.npz')['rows']
        matched,note=match_new_Gram_RMS(saved['params'],cfg,oldlocal,profile)
        result['Gram_matching']=note
        meta=deepcopy(saved['metadata']);meta['new_feature_initialization']='qwise_true_physical_Gram_RMS_match_to_old_prefix'
        meta['parameter_sha256']=pytree_sha256(matched)
        save_checkpoint(output/'Gram_matched_checkpoint.pkl',matched,optimizer_state=saved['optimizer_state'],metadata=meta)
        result['checkpoint']=dict(path=str(output/'Gram_matched_checkpoint.pkl'),
            sha256=hashlib.sha256((output/'Gram_matched_checkpoint.pkl').read_bytes()).hexdigest(),
            parameter_sha256=meta['parameter_sha256'],optimizer_sha256=pytree_sha256(saved['optimizer_state']),
            optimizer_RNG_unchanged=True,global_step=meta['absolute_completed_steps'])
        atomic_json(output/'progress.json',result)
        from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank,make_feature_lut_projected_amplitude
        from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
        images=np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32)
        projected=make_feature_lut_projected_amplitude(cfg,images)
        layout=mixed_clebsch_layout(cfg);root=layout[-1];gi,ki=next((i,k) for i,g in enumerate(root['groups'])
            for k,key in enumerate(g['keys']) if key==(2,2,0))
        oldwidth=profile[root['left']-16][2]
        gates=[]
        for leftcopy in (0,oldwidth):
            gate=np.zeros(root['groups'][gi]['shape']);gate[ki,0,leftcopy,0]=1.;gates.append(jnp.asarray(gate))
        def one(bank,x):
            psi=projected(bank,x)
            def derivative(gate):
                def line(t):
                    r=tuple(w+(t*gate if i==gi else 0.) for i,w in enumerate(bank['top'][-1]))
                    return projected(dict(bank,top=(*bank['top'][:-1],r)),x)
                return jax.jvp(line,(jnp.asarray(0.),),(jnp.asarray(1.),))[1]
            return psi,derivative(gates[0]),derivative(gates[1])
        evaluate=jax.jit(lambda bank,x:jax.lax.map(lambda row:one(bank,row),x))
        result['branches']={};outputs={}
        for name,params in (('original_scale_0p2',saved['params']),('Gram_RMS_matched',matched)):
            tick=perf_counter();prepared=prepare_feature_lut_bank(params,cfg)
            p,old,new=(np.asarray(a) for a in jax.block_until_ready(evaluate(prepared,jnp.asarray(rows))))
            result['branches'][name]=dict(seconds=perf_counter()-tick,old_gate=ray_strength(p,old),
                new_gate=ray_strength(p,new),new_over_old_Fisher=ray_strength(p,new)['horizontal_Fisher']/ray_strength(p,old)['horizontal_Fisher'])
            outputs[name]=p;np.savez_compressed(output/(name+'_projected_gates.npz'),rows=rows,psi=p,old_gate=old,new_gate=new)
        result['state_preservation_relative_error']=float(np.linalg.norm(outputs['original_scale_0p2']-outputs['Gram_RMS_matched'])/np.linalg.norm(outputs['original_scale_0p2']))
        if result['state_preservation_relative_error']>3e-10:raise ArithmeticError('new-only scale branch changed the parent state')
        result['status']='complete_fixed_projected_gate_comparison_no_energy_selection'
    except TimeoutError as error:
        result['status']='bounded_partial_no_extension';result['error']=str(error)
    finally:
        signal.alarm(0);result['wall_seconds']=perf_counter()-start;result['CPU_seconds']=process_time()-cpu
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_RSS_GiB']=rss/(1024**3 if sys.platform=='darwin' else 1024**2)
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--old-registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.old_registration,a.output)
