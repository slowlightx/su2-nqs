"""Bounded physical boundary enrichment of the18->36 CT12 interface.

This is raw SU(2) candidate construction. Complete coherent A1 energy,
variance and support reliability must be tested separately before attention.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import signal
from time import perf_counter

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_conditional_response import (
    make_conditional_cut_response,nested_response_expansion,root_coefficient_matrices,
)
from su2_nqs.operators.su2_hierarchy_response import hierarchy_local_spin_response
from su2_nqs.operators.su2_sector_merge import project_sector_maps
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from .triangle_boundary_response_enrichment import (
    TREE,active_environment,_independent_node_map_copy,make_joint_upper_root_operator,joint_upper_root_probe,
)
from .triangle_hierarchy_initialization import native_bank_from_state,source_channels,split_bond_ledger
from .triangle_sector_transfer import load_source
from .triangle_upper_sweep import raw_layer_energy

ROOT=Path(__file__).resolve().parents[3]


def hierarchy_boundary_diagnostic(state: dict,source: dict, *, nodes: tuple[int,...]=(0,1,8,9)) -> dict:
    """Use the same target-H boundaries and actual raw-state density at each scale."""
    environment,labels,_=active_environment(state,source)
    _,_,bonds=split_bond_ledger(12,.125)
    result={}
    for node in nodes:
        left,right,parent=TREE.binary_merges[node]
        support=TREE.coarse_supports[parent];inside=set(support)
        boundary=sorted({i if i in inside else j for i,j,g in bonds
                         if g!=0 and (i in inside)!=(j in inside)})
        positions={s:i for i,s in enumerate(support)}
        sigma={q:(q+1)*e/environment.norm_squared for q,e in environment.downward[labels[parent]].items()}
        details=hierarchy_local_spin_response(state['blocks'][left],state['blocks'][right],
            state['blocks'][parent],state['maps'][node],sigma,sites=[positions[s] for s in boundary])
        details['physical_boundary_sites']=boundary;details['region_sites']=list(support)
        result[node]=details
    return result


def enriched_36_parent(state: dict, *, additional: int=4) -> tuple[dict,dict]:
    """Nest four boundary-H response columns in each q=0,2,4 sector, preserve the parent."""
    if additional!=4:raise ValueError('this registered candidate uses four additions per frozen q')
    al,ar,a=TREE.binary_merges[8];_,b,c=TREE.binary_merges[12];_,d,root=TREE.binary_merges[14]
    _,joins,bonds=split_bond_ledger(12,.125)
    positions=[{s:i for i,s in enumerate(TREE.coarse_supports[n])} for n in (a,b,d)]
    ab=[];ad=[]
    for i,j,g in bonds:
        if (i in positions[0])==(j in positions[0]):continue
        u,v=(i,j) if i in positions[0] else (j,i)
        if v in positions[1]:ab.append((positions[0][u],positions[1][v],g))
        elif v in positions[2]:ad.append((positions[0][u],positions[2][v],g))
        else:raise ArithmeticError('the two retained bath blocks do not cover the complement')
    roots=root_coefficient_matrices(state['maps'][14][0],state['channels'][14][0])
    candidate=deepcopy(state);candidate['maps']=_independent_node_map_copy(state['maps'])
    candidate['cache']={};candidate['leaf_cache']={}
    candidate['parent_initialization_records']=candidate.pop('records',[]);candidate['records']=[]
    details={'cross_AB_bonds':len(ab),'cross_AD_bonds':len(ad),'sectors':{}}
    for q in (0,2,4):
        tick=perf_counter()
        response=make_conditional_cut_response(a_children=(state['blocks'][al],state['blocks'][ar]),
            a_source=state['blocks'][a],a_maps=state['maps'][8],b=state['blocks'][b],d=state['blocks'][d],
            parent_maps=state['maps'][12],root_coefficients=roots,cross_bonds_ab=ab,cross_bonds_ad=ad,
            target_q=q,center=True)
        expanded=nested_response_expansion(q,state['maps'][8][q],response,
            additional=additional,seed=2610261200+q)
        if np.max(abs(expanded.frame.imag))>2e-12:
            raise ArithmeticError('real native exporter cannot discard complex response columns')
        candidate['maps'][8][q]=expanded.frame.real
        if not np.array_equal(candidate['maps'][8][q][:,:state['maps'][8][q].shape[1]],state['maps'][8][q]):
            raise ArithmeticError('the original occupied subspace was altered')
        details['sectors'][q]=dict(response=response.diagnostics,expansion=expanded.diagnostics,
            new_singular_values=expanded.added_singular_values.tolist(),seconds=perf_counter()-tick)
    candidate['blocks'][a]=project_sector_maps(candidate['blocks'][al],candidate['blocks'][ar],joins[8],
        maps=candidate['maps'][8],compact_boundary=True,max_output_dimension=400)
    for q,old in state['maps'][12].items():
        channels=source_channels(candidate['blocks'][a].q_widths,candidate['blocks'][b].q_widths,q)
        index={label:i for i,label in enumerate(channels)}
        padded=np.zeros((len(channels),old.shape[1]),dtype=old.dtype)
        for i,label in enumerate(state['channels'][12][q]):padded[index[label]]=old[i]
        candidate['maps'][12][q]=padded;candidate['channels'][12][q]=channels
    candidate['blocks'][c]=project_sector_maps(candidate['blocks'][a],candidate['blocks'][b],joins[12],
        maps=candidate['maps'][12],compact_boundary=True,max_output_dimension=400)
    candidate['blocks'][root]=project_sector_maps(candidate['blocks'][c],candidate['blocks'][d],joins[14],
        maps=candidate['maps'][14],compact_boundary=True,max_output_dimension=400)
    oldenergy=float(state['blocks'][30].h_by_q[0][0,0].real)
    newenergy=float(candidate['blocks'][30].h_by_q[0][0,0].real)
    if abs(oldenergy-newenergy)>2e-8:raise ArithmeticError('nested parent energy was not preserved')
    if any(not np.array_equal(w,candidate['maps'][n][q]) for n,rows in state['maps'].items()
           if n not in (8,12) for q,w in rows.items()):raise ArithmeticError('unrelated tree tensors changed')
    details.update(raw_parent_energy_error=abs(oldenergy-newenergy),
        old_columns_preserved=True,ancestor_exact_zero_extension=True,
        extra_coarse_H_columns=sum(w.shape[1] for w in candidate['maps'][8].values())+49,
        original_parent_feasible=True,complete_A1_quality_unknown=True)
    return candidate,details


def run(output: Path) -> dict:
    """One registered 900-second parent-vs-enriched exact effective-H comparison."""
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('choose a new immutable result folder')
    oldbase=ROOT/'projects/clebsch_tree_nqs/results/ct_upper_sweep_20261025'
    parents={name:oldbase/name/'after_node13.pkl' for name in ('old8_joint','nested10_joint')}
    files=[Path(__file__),ROOT/'su2_nqs/operators/su2_hierarchy_response.py',
           ROOT/'su2_nqs/operators/su2_conditional_response.py',
           Path(__file__).with_name('triangle_boundary_response_enrichment.py'),
           ROOT/'tests/test_su2_hierarchy_response.py']
    registration=dict(task='CT12_multisector_boundary_18_to36_then_left72_root_v1',
        parent={name:dict(path=str(p),sha256=file_sha256(p)) for name,p in parents.items()},
        optimization_parent='nested10_joint_after_node13',changed36_node=8,
        output_q=[0,2,4],extra_columns_per_q=4,all_source_q_and_m_included=True,
        optimization_arms=['unchanged_36','boundary_q024_plus4'],Krylov_vectors=4,
        maximum_wall_seconds=900.,diagnostic_nodes=[0,1,8,9],no_A1_or_MC_H_queries=True,
        source_sha256={str(p):file_sha256(p) for p in files},
        complete_A1_energy_variance_required_before_attention=True)
    atomic_json(output/'registration.json',registration)
    for p in files:copyfile(p,output/('registered_'+p.name))
    started=perf_counter();result=dict(registration=registration,status='running',arms={},stages=[])
    def stage(name,**data):
        result['stages'].append(dict(name=name,seconds=perf_counter()-started,**data))
        atomic_json(output/'progress.json',result);print(name,round(perf_counter()-started,2),flush=True)
    previous=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('registered900-second total local candidate budget')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,900.)
    prepared={};states={};completed={}
    try:
        source,provenance=load_source(.125);result['source']=provenance
        parent=load_checkpoint(parents['nested10_joint'])['params']
        result['before']=raw_layer_energy(parent,source,.125)
        result['hierarchy_before']=hierarchy_boundary_diagnostic(parent,source)
        stage('hierarchy_losses_measured')
        expanded,details=enriched_36_parent(parent)
        result['enrichment']=details
        # Zero-H exact physical overlap establishes that no parent quality was lost.
        cfg=MixedClebschConfig(12,20,10)
        parentbank,_=native_bank_from_state(source,parent);expandedbank,embedding=native_bank_from_state(source,expanded)
        overlap=triangular_raw_overlap(parentbank,expandedbank,cfg)
        norm=triangular_raw_overlap(expandedbank,expandedbank,cfg)
        if max(abs(overlap-1),abs(norm-1))>2e-10:raise ArithmeticError('native full-state parent preservation failed')
        result['enrichment'].update(native_overlap_real=float(overlap.real),native_norm_real=float(norm.real))
        save_checkpoint(output/'preserved_expanded.pkl',expanded,metadata=dict(registration=registration,raw_only=True))
        result['hierarchy_preserved_expanded']=hierarchy_boundary_diagnostic(expanded,source,nodes=(8,))
        stage('multisector_parent_embedded',native_overlap_error=float(abs(overlap-1)))
        states={'unchanged_36':parent,'boundary_q024_plus4':expanded}
        costs={}
        for arm,state in states.items():
            op,center,notes=make_joint_upper_root_operator(state,.125,upper_node=12)
            tick=perf_counter();hp=op.matvec(center);costs[arm]=perf_counter()-tick
            if abs(np.vdot(center,hp)-notes['raw_parent_energy_expected'])>2e-8:
                raise ArithmeticError('full target-H preflight disagrees with frozen parent')
            prepared[arm]=(op,center,hp);result['arms'][arm]=dict(preflight=notes,H_seconds=costs[arm])
            stage('preflight_'+arm)
        forecast=1.4*3*sum(costs.values())+120.
        result['paired_cost_gate']=dict(forecast_seconds=forecast,remaining_seconds=900.-(perf_counter()-started))
        if forecast>900.-(perf_counter()-started):
            result['status']='preserved_candidate_only_cost_gate';return result
        for arm,state in states.items():
            op,center,hp=prepared[arm]
            candidate,notes=joint_upper_root_probe(state,source,.125,operator=op,center=center,
                initial_product=hp,krylov_vectors=4,upper_node=12)
            completed[arm]=candidate
            result['arms'][arm].update(optimization=notes,after=raw_layer_energy(candidate,source,.125))
            save_checkpoint(output/arm/'state.pkl',candidate,metadata=dict(registration=registration,raw_only=True))
            bank,embedding=native_bank_from_state(source,candidate)
            norm=triangular_raw_overlap(bank,bank,cfg)
            if abs(norm-1)>2e-10:raise ArithmeticError('candidate export raw normalization failed')
            save_checkpoint(output/arm/'initialized_bank.pkl',bank,metadata=dict(registration=registration,
                model=dict(length=12,multiplicity_cap=20,maximum_q=10),j2=.125,embedding=embedding,
                optimizer_reset_required=True,complete_A1_quality_unknown=True))
            result['arms'][arm]['bank_sha256']=file_sha256(output/arm/'initialized_bank.pkl')
            result['arms'][arm]['raw_bank_norm']=float(norm.real)
            result['arms'][arm]['hierarchy_after']=hierarchy_boundary_diagnostic(candidate,source,nodes=(8,))
            stage('completed_'+arm)
        result['status']='complete_raw_matched_multisector_candidate_comparison'
    except TimeoutError as error:
        result.update(status='bounded_partial',error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result['wall_seconds']=perf_counter()-started
        result['center_H_matvecs_actual']={arm:op.matvec_count for arm,(op,_,_) in prepared.items()}
        result['MC_full_H_queries']=0;result['complete_A1_H_queries']=0
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
