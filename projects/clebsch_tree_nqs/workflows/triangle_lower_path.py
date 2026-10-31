"""Bounded 9->18 enrichment with the actual 126-site raw complement.

The paired center step refines the36 node in its108-site complement and
coherently refactors72/root. It is an exact restricted raw variational step,
not unrestricted simultaneous optimization of every ancestor, full bath
response leakage, physical H variance, or complete coherent A1 quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import signal
from time import perf_counter, process_time

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_complement import (
    descend_singlet_complement, restore_singlet_parent, make_fixed_complement_cut_response,
)
from su2_nqs.operators.su2_conditional_response import root_coefficient_matrices, nested_response_expansion
from su2_nqs.operators.su2_sector_merge import project_sector_maps
from su2_nqs.operators.su2_three_block_center import ThreeBlockSingletOperator, parent_anchored_center_ritz
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_boundary_multisector import hierarchy_boundary_diagnostic
from .triangle_boundary_response_enrichment import TREE, _independent_node_map_copy
from .triangle_hierarchy_initialization import native_bank_from_state, source_channels, split_bond_ledger
from .triangle_sector_transfer import load_source
from .triangle_upper_sweep import raw_layer_energy

ROOT=Path(__file__).resolve().parents[3]


def cross_bonds(left, right, bonds):
    """Every target-H bond crossing two declared disjoint ordered supports."""
    lp={s:i for i,s in enumerate(left)};rp={s:i for i,s in enumerate(right)}
    if set(lp)&set(rp):raise ValueError('disjoint physical supports required')
    result=[]
    for i,j,g in bonds:
        if i in lp and j in rp:result.append((lp[i],rp[j],g))
        elif j in lp and i in rp:result.append((lp[j],rp[i],g))
    return result


def real_maps(maps):
    """Keep the existing real native family, refusing complex-state changes."""
    if max(float(np.max(abs(w.imag),initial=0.)) for w in maps.values())>2e-11:
        raise ArithmeticError('real native exporter cannot discard a complex update')
    return {q:np.asarray(w.real).copy() for q,w in maps.items()}


def reproject(state,node,joins):
    left,right,parent=TREE.binary_merges[node]
    maps=state['maps'][node]
    state['blocks'][parent]=project_sector_maps(state['blocks'][left],state['blocks'][right],
        joins[node],maps=maps,compact_boundary=True,
        max_output_dimension=sum((q+1)*w.shape[1] for q,w in maps.items()),
        memory_limit_bytes=768*1024**2)
    state['channels'][node]={q:source_channels(state['blocks'][left].q_widths,
        state['blocks'][right].q_widths,q) for q in maps}


def clone(state):
    out=deepcopy(state);out['maps']=_independent_node_map_copy(state['maps'])
    out['cache']={};out['leaf_cache']={};out['records']=[]
    return out


def run(output: Path, *, seconds_cap: float=900.) -> dict:
    """Registered matched comparison; numerical cap excludes code/tests/imports."""
    if not 30<=seconds_cap<=900:raise ValueError('30..900 second local numerical cap required')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('new immutable output directory required')
    parent_path=ROOT/'projects/clebsch_tree_nqs/results/ct_boundary_multisector_20261026/boundary_q024_plus4/state.pkl'
    source_files=[Path(__file__),ROOT/'su2_nqs/operators/su2_complement.py',
        ROOT/'tests/test_su2_complement.py']
    registration=dict(task='CT12_actual_complement_lower18_path_v2_independent_root_H',
        parent=str(parent_path),parent_sha256=file_sha256(parent_path),j2=.125,
        changed18_node=0,output_q=[0,2,4],additional_columns_per_q=2,
        exact_whole_complement_sizes=[108,126],
        paired_arms=['old18','nested18'],center_node=8,refactored_ancestor_nodes=[12,14],
        all_source_q_coherent=True,old18_columns_protected=True,
        maximum_wall_seconds=seconds_cap,maximum_CPU_seconds=seconds_cap,
        Krylov_vectors_cost_rule='3 each if preflight 1.3*2*(sum H seconds)+120 fits; otherwise2 each if1.3*(sum H seconds)+120 fits; otherwise stop',
        actual_whole_complement_projected_gradient=True,physical_bath_leakage_included=False,
        unrestricted_all_ancestor_variation=False,complete_A1_quality_unknown=True,
        no_MC_or_complete_A1_H_queries=True,
        source_sha256={str(p):file_sha256(p) for p in source_files})
    atomic_json(output/'registration.json',registration)
    for p in source_files:copyfile(p,output/('registered_'+p.name))
    result=dict(registration=registration,status='running',stages=[],arms={})
    started=perf_counter();cpu=process_time();operators={}
    def stage(name,**values):
        result['stages'].append(dict(name=name,wall_seconds=perf_counter()-started,**values))
        atomic_json(output/'progress.json',result);print(name,round(perf_counter()-started,3),flush=True)
    def timeout(*_):raise TimeoutError('registered local numerical budget exhausted')
    previous=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,seconds_cap)
    try:
        source,provenance=load_source(.125);result['source']=provenance
        parent=load_checkpoint(parent_path)['params'];_,joins,bonds=split_bond_ledger(12,.125)
        supports=TREE.coarse_supports
        result['before']=raw_layer_energy(parent,source,.125)
        roots=root_coefficient_matrices(parent['maps'][14][0],parent['channels'][14][0])
        env36=descend_singlet_complement(parent['blocks'][24],parent['blocks'][25],parent['blocks'][29],
            parent['maps'][12],roots,cross_bonds(supports[25],supports[29],bonds))
        support108=supports[25]+supports[29]
        stage('actual108site_complement_ready',diagnostics=dict(env36.diagnostics))
        env18=descend_singlet_complement(parent['blocks'][16],parent['blocks'][17],env36.complement,
            parent['maps'][8],env36.coefficients,cross_bonds(supports[17],support108,bonds))
        support126=supports[17]+support108
        stage('actual126site_complement_ready',diagnostics=dict(env18.diagnostics))
        expanded=clone(parent);details={}
        for q in (0,2,4):
            response=make_fixed_complement_cut_response(a_children=(parent['blocks'][0],parent['blocks'][1]),
                a_maps=parent['maps'][0],environment=env18.complement,coefficients=env18.coefficients,
                cross_bonds=cross_bonds(supports[16],support126,bonds),target_q=q)
            new=nested_response_expansion(q,parent['maps'][0][q],response,additional=2,seed=2610269200+q)
            expanded['maps'][0][q]=real_maps({q:new.frame})[q]
            details[q]=dict(response=dict(response.diagnostics),expansion=dict(new.diagnostics),
                singular_values=new.added_singular_values.tolist())
        reproject(expanded,0,joins)
        for q,old in parent['maps'][8].items():
            labels=source_channels(expanded['blocks'][16].q_widths,expanded['blocks'][17].q_widths,q)
            lookup={label:i for i,label in enumerate(labels)}
            padded=np.zeros((len(labels),old.shape[1]),old.dtype)
            for i,label in enumerate(parent['channels'][8][q]):padded[lookup[label]]=old[i]
            expanded['maps'][8][q]=padded
        for node in (8,12):reproject(expanded,node,joins)
        # The zero extension is state preserving; root expectation stays the same.
        cfg=MixedClebschConfig(12,20,10)
        oldbank,_=native_bank_from_state(source,parent);newbank,_=native_bank_from_state(source,expanded)
        overlap=triangular_raw_overlap(oldbank,newbank,cfg);norm=triangular_raw_overlap(newbank,newbank,cfg)
        if max(abs(overlap-1),abs(norm-1))>2e-10:raise ArithmeticError('lower expansion did not preserve physical parent')
        result['enrichment']=dict(sectors=details,native_parent_overlap_error=float(abs(overlap-1)),
            native_expanded_norm_error=float(abs(norm-1)),ancestor_zero_extension=True)
        save_checkpoint(output/'preserved_expanded.pkl',expanded,metadata=dict(registration=registration,raw_only=True))
        stage('protected9to18_parent_embedded')
        states={'old18':parent,'nested18':expanded};prepared={};costs={}
        for name,state in states.items():
            op=ThreeBlockSingletOperator(state['blocks'][16],state['blocks'][17],env36.complement,
                joins[8],cross_bonds(supports[16],support108,bonds),cross_bonds(supports[17],support108,bonds),
                memory_limit_bytes=768*1024**2,max_dimension=100_000)
            operators[name]=op;center=op.encode(state['maps'][8],env36.coefficients)
            tick=perf_counter();hp=op.matvec(center);costs[name]=perf_counter()-tick
            error=abs(np.vdot(center,hp)-result['before']['total'])
            if error>2e-8:raise ArithmeticError('whole-complement H disagrees with physical parent')
            prepared[name]=(center,hp)
            result['arms'][name]=dict(preflight_parent_energy_error=float(error),preflight_H_seconds=costs[name],
                center_dimension=op.dimension,physical_bonds=864,whole_complement_sites=108)
            stage('preflight_'+name)
        remaining=seconds_cap-(perf_counter()-started)
        vectors=3 if 1.3*2*sum(costs.values())+120<remaining else 2
        if 1.3*(vectors-1)*sum(costs.values())+120>remaining:
            result['status']='preserved_expansion_only_cost_gate';return result
        result['matched_Krylov_vectors']=vectors
        for name,state in states.items():
            center,hp=prepared[name];op=operators[name]
            optimized,quality=parent_anchored_center_ritz(op,center,krylov_vectors=vectors,initial_product=hp)
            maps36,coeff36,factor=op.factor(optimized,widths={q:w.shape[1] for q,w in state['maps'][8].items()})
            maps72,newroots,restoration=restore_singlet_parent(env36,coeff36,
                {q:w.shape[1] for q,w in maps36.items()},
                parent_widths={q:w.shape[1] for q,w in state['maps'][12].items()})
            candidate=clone(state);candidate['maps'][8]=real_maps(maps36);candidate['maps'][12]=real_maps(maps72)
            rootvalues=np.asarray([newroots[q][a,b] for q,a,qr,b in state['channels'][14][0]])[:,None]
            candidate['maps'][14][0]=real_maps({0:rootvalues})[0]
            for node in (8,12):reproject(candidate,node,joins)
            # Independently contract the restored root. A Ritz-assigned cache
            # would make the subsequent telescoping energy check circular.
            reproject(candidate,14,joins)
            root_error=abs(candidate['blocks'][30].h_by_q[0][0,0]-quality['raw_energy_after'])
            if root_error>2e-8:raise ArithmeticError('independent restored root H disagrees with center Ritz')
            after=raw_layer_energy(candidate,source,.125)
            bank,embedding=native_bank_from_state(source,candidate)
            rawnorm=triangular_raw_overlap(bank,bank,cfg)
            if abs(rawnorm-1)>2e-10:raise ArithmeticError('exported endpoint is not normalized')
            save_checkpoint(output/name/'state.pkl',candidate,metadata=dict(registration=registration,raw_only=True))
            save_checkpoint(output/name/'initialized_bank.pkl',bank,metadata=dict(registration=registration,
                model=dict(length=12,multiplicity_cap=20,maximum_q=10),j2=.125,embedding=embedding,
                optimizer_reset_required=True,complete_A1_quality_unknown=True))
            result['arms'][name].update(optimization=quality,factorization=factor,ancestor_restoration=restoration,
                independent_restored_root_H_error=float(root_error),
                after=after,raw_bank_norm=float(rawnorm.real),bank_sha256=file_sha256(output/name/'initialized_bank.pkl'),
                hierarchy_after=hierarchy_boundary_diagnostic(candidate,source,nodes=(0,8)))
            stage('completed_'+name)
        result['status']='complete_matched_raw_lower_path_comparison'
    except TimeoutError as error:
        result.update(status='bounded_partial',error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result.update(wall_seconds=perf_counter()-started,CPU_seconds=process_time()-cpu,
            center_H_matvecs_actual={name:op.matvec_count for name,op in operators.items()},
            complete_A1_H_queries=0,new_MC_records=0)
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds-cap',type=float,default=900.)
    args=parser.parse_args();run(args.output,seconds_cap=args.seconds_cap)
