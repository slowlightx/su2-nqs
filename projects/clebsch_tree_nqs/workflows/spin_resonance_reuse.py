"""Fixed-probe exact cost reduction, then multienvironment scalar attention.

No optimization, ED or energy estimator. All images/H terms remain complete.
Original slow and new fast evaluations use identical saved carrier/parameters.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.a1_carrier_queries import compile_a1_overlapping_carrier_queries
from su2_nqs.benchmarks.overlapping_carrier_queries import evaluate_overlapping_carrier_queries, overlap_filter_rows
from su2_nqs.benchmarks.multicontext_resonance_queries import multicontext_overlap_filter_rows
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit, triangular_pi_zero_flux_hopping, BOUNDARY_ORBIT,
)
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig, projected_singlet_agp_amplitudes
from su2_nqs.models.shared_resonance_attention import ResonanceAttentionConfig, shared_resonance_generators
from su2_nqs.models.multicontext_resonance_attention import (
    MulticontextResonanceAttentionConfig, init_multicontext_resonance_attention,
    make_multicontext_reference, multicontext_resonance_generators,
    free_multicontext_generators, interaction_contrast,
)
from su2_nqs.models.slater_minor_updates import compile_slater_minor_plan, slater_minor_updates, closed_shell_amplitudes
from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries, gather_physical_h_action
from .spin_overlap_attention_cost import _timed, _relative, _save_npz, peak_rss_gib, profile_chunk, ARMS

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_resonance_reuse.json'


def _stream_kernel(kernel, gradient, params, rows, terminal, weights, chunk_size, repeats):
    """One static-shape JIT is reused across all H-output chunks and centers."""
    output=[]; grad_sum=jax.tree.map(jnp.zeros_like,params)
    records=[]
    for i in range(0,len(rows),chunk_size):
        staging_start=perf_counter()
        size=min(chunk_size,len(rows)-i)
        r=np.zeros((chunk_size,rows.shape[1]),dtype=np.int32);r[:size]=rows[i:i+size]
        v=np.zeros((chunk_size,*terminal.shape[1:]),dtype=terminal.dtype);v[:size]=terminal[i:i+size]
        w=np.zeros(chunk_size);w[:size]=weights[i:i+size]
        rr,vv,ww=map(jnp.asarray,(r,v,w))
        jax.block_until_ready((rr,vv,ww))
        staging_seconds=perf_counter()-staging_start
        amp,first=_timed(kernel,params,rr,vv)
        grad,ad_first=_timed(gradient,params,rr,vv,ww)
        warm=[_timed(kernel,params,rr,vv)[1] for _ in range(repeats)]
        ad_warm=[_timed(gradient,params,rr,vv,ww)[1] for _ in range(repeats)]
        output.append(np.asarray(amp)[:size])
        grad_sum=jax.tree.map(lambda a,b:a+b,grad_sum,grad)
        records.append(dict(host_staging_seconds=staging_seconds,forward_first_seconds=first,reverse_first_seconds=ad_first,
                            forward_warm_seconds=warm,reverse_warm_seconds=ad_warm))
    return np.concatenate(output),jax.tree.map(np.asarray,grad_sum),records


def run(output: Path,config_path: Path=CONFIG)->dict:
    cfg=json.loads(config_path.read_text())
    if cfg['training_steps']!=0 or cfg['remote_allowed'] or cfg['task']!='spin_resonance_reuse':
        raise ValueError('fixed local cost and interface experiment only')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing to overwrite registered experiment')
    source=ROOT/cfg['source_run']
    if file_sha256(source/'results.json')!=cfg['source_results_sha256']:
        raise ValueError('source result seal mismatch')
    old=json.loads((source/'results.json').read_text()); oldcfg=old['config']
    for name,key in [('checkpoint.npz','checkpoint_sha256'),('full_h_replay.npz','replay_sha256')]:
        if file_sha256(source/name)!=old[key]:raise ValueError('saved input changed')
    output.mkdir(parents=True,exist_ok=True)
    registration=dict(config=cfg,config_sha256=file_sha256(config_path),runtime=numerical_runtime_identity(),
        source_manifest=transitive_local_source_manifest([Path(__file__)],root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    atomic_json(output/'registration.json',registration)
    result=dict(status='running',registration_sha256=file_sha256(output/'registration.json'),
                geometries={},training_steps=0,energy_estimate=None,variance_estimate=None,remote_released=False)
    start=perf_counter()
    def budget(stage):
        print(stage,flush=True)
        atomic_json(output/'progress.json',dict(stage=stage,seconds=perf_counter()-start,peak_rss_gib=peak_rss_gib()))
        if peak_rss_gib()>cfg['peak_rss_gib_cap']:raise MemoryError('registered RSS cap reached')
    def timeout(*_):raise TimeoutError('registered wall cap reached')
    handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(cfg['wall_seconds_cap'])
    saved={}
    try:
        with np.load(source/'checkpoint.npz') as data:
            params={k[10:]:jnp.asarray(data[k]) for k in data.files if k.startswith('parameter_')}
            initial={k[8:]:jnp.asarray(data[k]) for k in data.files if k.startswith('initial_')}
            reference=data['reference'].copy();offsets=jnp.asarray(data['offsets']);centers=data['centers'].copy()
        direction={k:(params[k]-initial[k])/oldcfg['parameter_perturbation_norm'] for k in params}
        model_cfg=ResonanceAttentionConfig(oldcfg['width'],oldcfg['hidden_width'])
        group=triangular_hnf_gamma_a1_group(((6,0),(0,6)))
        hplan=compile_physical_h_queries(centers,triangular_hnf_j1j2_bonds(((6,0),(0,6)),j2=.125))
        if len(centers)!=cfg['H_centers'] or len(hplan.configurations)!=cfg['full_H_outputs']:
            raise ValueError('fixed-probe support mismatch')
        result['h_stencil']=hplan.diagnostics
        orbit=build_n36_pi_zero_flux_orbit(); agp=ProjectedSingletAGPConfig(36,(tuple(range(36)),))
        pairings=[jnp.asarray(p) for p in orbit.pairings]
        @jax.jit
        def dense_kernel(bits):
            values=jnp.stack([projected_singlet_agp_amplitudes({'pairing_real':p},bits,agp).amplitudes for p in pairings],axis=1)
            return values@jnp.asarray([1.,-1.,-1.])/3
        batch=cfg['carrier_batch_size']
        def carrier(bits):
            n=len(bits)
            return dense_kernel(jnp.asarray(np.pad(bits,((0,batch-n),(0,0)),mode='edge')))[:n]
        cvalues=np.asarray(carrier(hplan.configurations))
        cpsi,chpsi=map(np.asarray,gather_physical_h_action(hplan,jnp.asarray(cvalues)))
        scale=max(float(np.linalg.norm(cpsi)),1e-300);hscale=max(float(np.linalg.norm(chpsi)),1e-300)
        weights=np.zeros(len(cvalues));np.add.at(weights,hplan.center_indices,1/scale)
        np.add.at(weights,hplan.connected_indices[hplan.valid],hplan.elements[hplan.valid]/hscale)
        # Establish the A1 precondition on the same centers and ALL432 images.
        image_bits=centers[:,np.asarray(group)].reshape(-1,36)
        image_values=np.concatenate([np.asarray(carrier(image_bits[i:i+batch])) for i in range(0,len(image_bits),batch)]).reshape(2,-1)
        result['carrier_all_images_relative_error']=_relative(image_values,np.broadcast_to(cpsi[:,None],image_values.shape))
        if result['carrier_all_images_relative_error']>cfg['amplitude_relative_tolerance']:
            raise FloatingPointError('carrier is not certified A1 on the registered images')
        # Small one-body occupied-orbital construction, NOT many-body ED.
        occupied=np.stack([np.linalg.eigh(triangular_pi_zero_flux_hopping(6,boundary_x=x,boundary_y=y))[1][:,:18]
                           for x,y in BOUNDARY_ORBIT])
        def single_amplitude(p,rows,values):
            tables=shared_resonance_generators(p,reference,offsets,model_cfg)
            local=overlap_filter_rows(tables[0],tables[1],jnp.arange(4096))
            return jnp.mean(jnp.sum(local[rows]*values,axis=-1),axis=1)
        fast=jax.jit(single_amplitude)
        fast_gradient=jax.jit(jax.grad(lambda p,r,v,w:jnp.real(jnp.sum(w*single_amplitude(p,r,v)))))
        for geometry in cfg['supports']:
            profile=dict(baseline_chunks=[]);result['geometries'][geometry]=profile
            budget(geometry+': paired uncompressed full-H baseline')
            baseline={k:[] for k in ARMS};gbase=jax.tree.map(lambda p:np.zeros(p.shape),params)
            chunk_size=cfg['H_output_chunk_size']
            for i in range(0,len(cvalues),chunk_size):
                record,arrays,grad=profile_chunk(hplan.configurations[i:i+chunk_size],group,
                    oldcfg[geometry+'_cells'],carrier,oldcfg,model_cfg,offsets,reference,initial,params,
                    direction,params,reference,weights[i:i+chunk_size])
                profile['baseline_chunks'].append(record)
                for arm in ARMS:baseline[arm].append(arrays[arm])
                gbase=jax.tree.map(lambda a,b:a+b,gbase,grad)
                jax.clear_caches();gc.collect()
            baseline={k:np.concatenate(v) for k,v in baseline.items()}
            saved.update({geometry+'_baseline_'+k:v for k,v in baseline.items()})
            profile['baseline_fresh_H_seconds']=sum(c['planning_seconds']+c['carrier_query_inclusive_seconds']+
                c['forward_compile_inclusive_seconds'] for c in profile['baseline_chunks'])
            profile['baseline_fresh_H_AD_seconds']=profile['baseline_fresh_H_seconds']+sum(
                c['gradient_compile_inclusive_seconds'] for c in profile['baseline_chunks'])
            budget(geometry+': A1 pullback query union, every H output at once')
            plan,seconds=_timed(lambda:compile_a1_overlapping_carrier_queries(hplan.configurations,group,
                cells=oldcfg[geometry+'_cells'],carrier_sector='Gamma-A1',max_unique_queries=cfg['max_unique_queries'],
                max_terminal_requests=cfg['max_terminal_requests'],max_plan_bytes=cfg['max_plan_bytes']))
            profile.update(pullback_plan=plan.diagnostics,pullback_planning_seconds=seconds)
            dense_values,seconds=_timed(lambda:evaluate_overlapping_carrier_queries(plan,carrier,batch_size=batch))
            profile['pullback_dense_carrier_seconds']=seconds
            budget(geometry+': exact rank-seven carrier update comparison')
            # Each pulled-back H-connected target is within7 exchanges of one saved center.
            # Dynamic plan arrays, fixed query shape: one JIT instance for the
            # entire corpus rather than vmap over millions of minor matrices.
            # The host retains only the current padded plan. Padding repeats
            # a real query and is excluded from all physical/work diagnostics.
            minor_batch=cfg['minor_batch_size']
            if minor_batch!=4096:
                raise ValueError('this bounded reuse experiment fixes minor_batch_size=4096')
            minor_values=np.empty(len(plan.configurations),dtype=np.complex128)
            minor_carrier=None;minor_records=[];fallback_count=0;maximum_condition=0.
            fallback_by_component_spin=np.zeros((3,2),dtype=np.int64)
            padded_fallback_by_component_spin=np.zeros((3,2),dtype=np.int64)
            rank_counts={str(k):0 for k in range(cfg['minor_max_rank']+1)}
            planning_seconds=0.;input_seconds=0.;output_seconds=0.;first_pass_seconds=0.
            warm_seconds=np.zeros(cfg['warm_repeats']);minor_numeric_plan_bytes=0
            minor_peak=peak_rss_gib()
            tick=perf_counter()
            occupied_device=jnp.asarray(occupied)
            jax.block_until_ready(occupied_device)
            input_seconds+=perf_counter()-tick
            for begin in range(0,len(plan.configurations),minor_batch):
                count=min(minor_batch,len(plan.configurations)-begin)
                tick=perf_counter()
                targets=np.pad(plan.configurations[begin:begin+count],
                    ((0,minor_batch-count),(0,0)),mode='edge')
                refs=np.argmin(np.sum(targets[:,None]!=centers[None],axis=-1),axis=1)
                minor_plan=compile_slater_minor_plan(centers,targets,refs,max_rank=cfg['minor_max_rank'])
                elapsed=perf_counter()-tick;planning_seconds+=elapsed
                minor_numeric_plan_bytes=max(minor_numeric_plan_bytes,minor_plan.numeric_plan_bytes)
                if minor_carrier is None:
                    # Only these two tiny reference tables are static. All
                    # query-dependent arrays are arguments, so changing a
                    # query batch does not retrace or alter the physical state.
                    plan_type=type(minor_plan)
                    reference_rows=minor_plan.reference_rows
                    @jax.jit
                    def minor_carrier(orb,target_bits,reference_indices,added_sites,
                                      replacement_positions,active,sorting_signs,ranks,physical_signs):
                        dynamic_plan=plan_type(centers,target_bits,reference_indices,reference_rows,
                            added_sites,replacement_positions,active,sorting_signs,ranks,physical_signs,
                            cfg['minor_max_rank'])
                        # Do not vmap over carrier components: that would
                        # batch a scalar fallback cond into a select and can
                        # execute dense determinants for every component.
                        components=[slater_minor_updates(orb[k],dynamic_plan,
                            condition_limit=cfg['minor_condition_limit']) for k in range(3)]
                        amplitudes=jnp.stack([closed_shell_amplitudes(res,dynamic_plan)
                                              for res in components])
                        return (jnp.asarray([1.,-1.,-1.])@amplitudes/3,
                                jnp.stack([res.dense_fallback for res in components]),
                                jnp.stack([res.reference_conditions for res in components]))
                tick=perf_counter()
                query_args=tuple(jnp.asarray(getattr(minor_plan,name)) for name in (
                    'target_bits','reference_indices','added_sites','replacement_positions',
                    'active','sorting_signs','ranks','physical_signs'))
                jax.block_until_ready(query_args)
                transfer=perf_counter()-tick;input_seconds+=transfer
                (values,fallback,conditions),first=_timed(minor_carrier,occupied_device,*query_args)
                first_pass_seconds+=first
                warms=[_timed(minor_carrier,occupied_device,*query_args)[1]
                       for _ in range(cfg['warm_repeats'])]
                warm_seconds+=np.asarray(warms)
                tick=perf_counter()
                minor_values[begin:begin+count]=np.asarray(values)[:count]
                padded_fallback=np.asarray(fallback)
                padded_fallback_by_component_spin+=np.count_nonzero(padded_fallback,axis=1)
                host_fallback=padded_fallback[:,:count]
                fallback_count+=int(np.count_nonzero(host_fallback))
                fallback_by_component_spin+=np.count_nonzero(host_fallback,axis=1)
                maximum_condition=max(maximum_condition,float(np.max(np.asarray(conditions))))
                back_transfer=perf_counter()-tick;output_seconds+=back_transfer
                for k in rank_counts:
                    rank_counts[k]+=int(np.count_nonzero(minor_plan.ranks[:count,0]==int(k)))
                minor_peak=max(minor_peak,peak_rss_gib())
                minor_records.append(dict(query_start=begin,valid_queries=count,padded_queries=minor_batch,
                    planning_seconds=elapsed,input_transfer_seconds=transfer,
                    first_execution_seconds=first,warm_execution_seconds=warms,
                    output_transfer_seconds=back_transfer))
                if minor_peak>cfg['peak_rss_gib_cap']:
                    raise MemoryError('registered RSS cap reached during bounded Slater updates')
                del query_args,values,fallback,conditions,targets,padded_fallback,host_fallback
            profile.update(minor_planning_seconds=planning_seconds,
                minor_first_seconds=first_pass_seconds,
                minor_first_batch_seconds=minor_records[0]['first_execution_seconds'],
                minor_remaining_first_pass_seconds=first_pass_seconds-minor_records[0]['first_execution_seconds'],
                minor_input_transfer_seconds=input_seconds,minor_output_transfer_seconds=output_seconds,
                minor_warm_seconds=warm_seconds.tolist(),minor_fixed_batch_size=minor_batch,
                minor_batches=minor_records,minor_peak_rss_gib=minor_peak,
                minor_peak_rss_scope='process high-water mark, not incremental allocation',
                minor_maximum_host_plan_bytes=minor_numeric_plan_bytes,
                minor_valid_queries=len(plan.configurations),
                minor_padded_queries=len(minor_records)*minor_batch,
                minor_valid_query_visits=len(plan.configurations)*(1+cfg['warm_repeats']),
                minor_padded_query_visits=len(minor_records)*minor_batch*(1+cfg['warm_repeats']),
                minor_fallback_components=fallback_count,
                minor_fallback_component_spin_counts=fallback_by_component_spin.tolist(),
                minor_executed_fallback_component_spin_visits=(
                    padded_fallback_by_component_spin*(1+cfg['warm_repeats'])).tolist(),
                minor_maximum_reference_condition=maximum_condition,
                minor_dense_relative_error=_relative(minor_values,dense_values),
                minor_rank_counts=rank_counts)
            if profile['minor_dense_relative_error']>cfg['amplitude_relative_tolerance']:
                raise FloatingPointError('minor update changed physical carrier')
            minor_cost=sum(profile[k] for k in ('minor_planning_seconds','minor_first_seconds',
                'minor_input_transfer_seconds','minor_output_transfer_seconds'))
            profile['minor_fresh_carrier_seconds']=minor_cost
            use_minor=minor_cost<profile['pullback_dense_carrier_seconds']
            profile['selected_carrier_evaluator']='row_minor' if use_minor else 'dense'
            chosen_values=minor_values if use_minor else dense_values
            chosen_cost=minor_cost if use_minor else profile['pullback_dense_carrier_seconds']
            budget(geometry+': static-shape full-H forward and all-parameter reverse AD')
            tick=perf_counter()
            rows=plan.unique_local_rows[plan.local_row_indices]
            terminal=np.asarray(chosen_values,dtype=np.complex128)[plan.connection_indices]
            profile['terminal_gather_seconds']=perf_counter()-tick
            amps,grad,records=_stream_kernel(fast,fast_gradient,params,rows,terminal,weights,chunk_size,cfg['warm_repeats'])
            profile['reused_kernel_chunks']=records
            profile['amplitude_relative_error']=_relative(amps,baseline['cascade'])
            hp=gather_physical_h_action(hplan,jnp.asarray(amps))[1]
            hbase=gather_physical_h_action(hplan,jnp.asarray(baseline['cascade']))[1]
            profile['full_H_relative_error']=_relative(hp,hbase)
            profile['reverse_gradient_relative_error']=_relative(np.concatenate([grad[k].ravel() for k in params]),
                np.concatenate([gbase[k].ravel() for k in params]))
            saved[geometry+'_fast_all_H_outputs']=amps
            query_cost=profile['pullback_planning_seconds']+chosen_cost+profile['terminal_gather_seconds']
            profile['fast_fresh_H_seconds']=query_cost+sum(r['host_staging_seconds']+r['forward_first_seconds'] for r in records)
            profile['fast_fresh_H_AD_seconds']=profile['fast_fresh_H_seconds']+sum(r['reverse_first_seconds'] for r in records)
            profile['H_speedup']=profile['baseline_fresh_H_seconds']/profile['fast_fresh_H_seconds']
            profile['H_AD_speedup']=profile['baseline_fresh_H_AD_seconds']/profile['fast_fresh_H_AD_seconds']
            if (profile['amplitude_relative_error']>cfg['amplitude_relative_tolerance'] or
                profile['full_H_relative_error']>cfg['H_relative_tolerance'] or
                profile['reverse_gradient_relative_error']>cfg['gradient_relative_tolerance']):
                raise FloatingPointError('reused action changed state or gradient')
            atomic_json(output/'partial_results.json',result)
            # Multi-environment physical action uses the SAME216-input query graph.
            budget(geometry+': multienvironment attention and free-table controls')
            mcfg=MulticontextResonanceAttentionConfig()
            pos=jnp.asarray(cfg['multicontext_positions'])
            m0=init_multicontext_resonance_attention(jax.random.PRNGKey(cfg['multicontext_initialization_seed']),mcfg)
            anchor=make_multicontext_reference(m0,pos,mcfg)
            rng=np.random.default_rng(cfg['multicontext_perturbation_seed'])
            noise={k:rng.normal(size=v.shape) for k,v in m0.items()};norm=np.sqrt(sum(np.sum(x*x) for x in noise.values()))
            mp={k:v+cfg['multicontext_parameter_distance']*noise[k]/norm for k,v in m0.items()}
            tables=multicontext_resonance_generators(mp,anchor,pos,mcfg)
            profile['generator_interaction_fraction']=float(jnp.linalg.norm(interaction_contrast(tables))/jnp.linalg.norm(tables))
            profile['multicontext']={}
            for control in cfg['multicontext_controls']:
                pp=pos[::-1] if control=='position_scrambled' else pos
                ref=make_multicontext_reference(m0,pp,mcfg) if control=='position_scrambled' else anchor
                if control=='free_table':
                    parameter=tables.reshape(-1)
                    generator=lambda t:free_multicontext_generators(t,mcfg)
                else:
                    parameter=mp
                    restriction=control if control in ('one_environment','additive') else 'full'
                    generator=lambda p,ref=ref,pp=pp,restriction=restriction:multicontext_resonance_generators(
                        p,ref,pp,mcfg,control=restriction)
                def context_amp(p,r,v):
                    table=generator(p)
                    local=multicontext_overlap_filter_rows(table,table,jnp.arange(4096))
                    return jnp.mean(jnp.sum(local[r]*v,axis=-1),axis=1)
                mkernel=jax.jit(context_amp)
                mgrad=jax.jit(jax.grad(lambda p,r,v,w:jnp.real(jnp.sum(w*context_amp(p,r,v)))))
                endpoint,mg,mtimes=_stream_kernel(mkernel,mgrad,parameter,rows,terminal,weights,chunk_size,1)
                if not np.isfinite(endpoint).all() or any(not np.isfinite(v).all() for v in jax.tree.leaves(mg)):
                    raise FloatingPointError('nonfinite multi-environment action/gradient')
                saved[geometry+'_multicontext_'+control]=endpoint
                profile['multicontext'][control]=dict(kernel_chunks=mtimes,
                    gradient_norm=float(np.sqrt(sum(np.sum(v*v) for v in jax.tree.leaves(mg)))),
                    parameter_count=sum(v.size for v in jax.tree.leaves(parameter)))
                del mkernel,mgrad;gc.collect()
            profile['multicontext_free_table_replay_error']=_relative(saved[geometry+'_multicontext_full'],saved[geometry+'_multicontext_free_table'])
            profile['multicontext_vs_additive_point_difference']=_relative(saved[geometry+'_multicontext_full'],saved[geometry+'_multicontext_additive'])
            if profile['multicontext_free_table_replay_error']>cfg['amplitude_relative_tolerance']:
                raise FloatingPointError('free tensor failed exact neural table replay')
            saved.update({geometry+'_gradient_'+k:v for k,v in grad.items()})
            budget(geometry+': complete')
            atomic_json(output/'partial_results.json',result)
            del plan,minor_plan,terminal,dense_values,minor_values,minor_carrier,chosen_values
            jax.clear_caches();gc.collect()
        _save_npz(output/'replay.npz',centers=centers,h_configurations=hplan.configurations,h_indices=hplan.connected_indices,
            h_elements=hplan.elements,h_valid=hplan.valid,**saved)
        _save_npz(output/'multicontext_checkpoint.npz',reference=anchor,positions=pos,**{'parameter_'+k:v for k,v in mp.items()},
                  **{'initial_'+k:v for k,v in m0.items()})
        result.update(status='completed',replay_sha256=file_sha256(output/'replay.npz'),
            checkpoint_sha256=file_sha256(output/'multicontext_checkpoint.npz'),
            interpretation='Exact evaluation speedup and joint-context interface only. Uniform fixed probes are not energy/variance/entanglement evidence. Free360 table contains neural outputs; no trained null or variational win.')
    except (MemoryError,TimeoutError) as error:
        result.update(status='resource_cap_reached',reason=str(error))
    except Exception as error:
        result.update(status='failed',reason=repr(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,handler)
        result.update(total_seconds=perf_counter()-start,peak_rss_gib=peak_rss_gib())
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();r=run(args.output,args.config)
    print(json.dumps({k:r[k] for k in ('status','total_seconds','peak_rss_gib')},indent=2))
