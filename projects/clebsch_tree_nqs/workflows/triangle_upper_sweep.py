"""Bounded alternating72/root refinement of frozen CT12 raw initializers.

This reuses the existing three-block exact H and never changes local36
subspaces, q cutoffs, widths, MC proposals or complete-A1 production code.
It is a candidate-construction experiment, not an A1 energy/variance audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfile
import signal
from time import perf_counter

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_boundary_response_enrichment import (
    TREE, active_environment, joint_upper_root_probe,
)
from .triangle_hierarchy_initialization import (
    active_amplitude, native_bank_from_state, split_bond_ledger,
)
from .triangle_sector_transfer import load_source

ROOT = Path(__file__).resolve().parents[3]


def center_stage_fits_budget(completed_seconds: list[float], *, remaining_seconds: float,
                            remaining_arms_in_stage: int, export_reserve: float = 120.) -> tuple[bool,float]:
    """Conservative measured-cost guard, independent of observed energies."""
    if (remaining_seconds<0 or export_reserve<0 or isinstance(remaining_arms_in_stage,bool)
            or remaining_arms_in_stage not in (1,2)
            or any(not np.isfinite(x) or x<=0 for x in completed_seconds)):
        raise ValueError('finite positive stage costs and valid remaining budget required')
    predicted=(1.15*max(completed_seconds)*remaining_arms_in_stage+export_reserve
               if completed_seconds else export_reserve)
    return remaining_seconds>=predicted,predicted


def raw_layer_energy(state: dict, source: dict, j2: float) -> dict:
    """Telescope cached projected H into actual once-only LCA bond energies.

    The downward block obeys rho=I_(q+1) tensor E_q for canonical maps.
    This exact raw-tree decomposition is not a projected-state readout.
    """
    environment, nodes, defect = active_environment(state, source)
    leaves, joins, bonds = split_bond_ledger(12, j2)
    def expectation(block_index: int, h_by_q: dict) -> float:
        return sum((q+1)*float(np.trace(h @ environment.downward[nodes[block_index]][q]).real)
                   for q, h in h_by_q.items()) / environment.norm_squared
    leaf_energy = sum(expectation(i, b.h_by_q) for i, b in enumerate(state['blocks'][:16]))
    rows=[]
    for node,(left,right,parent) in enumerate(TREE.binary_merges):
        l,r=state['blocks'][left],state['blocks'][right]
        inside={}
        for q,w in state['maps'][node].items():
            h=np.zeros((w.shape[1],w.shape[1]),dtype=np.result_type(w,complex))
            channels=state['channels'][node][q]
            for ql,qr in sorted({(x[0],x[2]) for x in channels}):
                ids=[i for i,x in enumerate(channels) if (x[0],x[2])==(ql,qr)]
                tensor=w[ids].reshape(l.q_widths[ql],r.q_widths[qr],-1)
                acted=(np.einsum('ab,bct->act',l.h_by_q[ql],tensor,optimize=True)
                       +np.einsum('ab,cbt->cat',r.h_by_q[qr],tensor,optimize=True))
                h+=tensor.reshape(-1,w.shape[1]).conj().T @ acted.reshape(-1,w.shape[1])
            inside[q]=h
        crossing={q:state['blocks'][parent].h_by_q[q]-inside[q] for q in inside}
        rows.append(dict(node=node,sites=state['blocks'][parent].n_sites,
            cross_bonds=len(joins[node]),cross_energy=expectation(parent,crossing)))
    total=leaf_energy+sum(x['cross_energy'] for x in rows)
    cached=float(state['blocks'][-1].h_by_q[0][0,0].real)
    if abs(total-cached)>2e-8:
        raise ArithmeticError('cached once-only bond telescoping disagrees with raw energy')
    return dict(leaf_internal=leaf_energy,by_node=rows,
        layer_cross_energy={n:sum(x['cross_energy'] for x in rows if x['sites']==n)
                            for n in (18,36,72,144)},
        total=total,total_per_site=total/144,energy_replay_error=abs(total-cached),
        weighted_physical_bonds=len(bonds),norm=environment.norm_squared,Gram_error=defect)


def run(output: Path, *, seconds_cap: float = 600.) -> dict:
    """Freeze right→left, four Ritz vectors, for both old8/nested10 parents."""
    if not 60 <= seconds_cap <= 600:
        raise ValueError('bounded development cap must be60..600 seconds')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('choose a fresh immutable output folder')
    parent_base=ROOT/'projects/clebsch_tree_nqs/results/ct_conditional_nested_initialization_20261023/step2'
    names=('old8_joint','nested10_joint')
    parents={name:parent_base/name/'state.pkl' for name in names}
    cfg=MixedClebschConfig(12,20,10)
    files=(Path(__file__),Path(__file__).with_name('triangle_boundary_response_enrichment.py'),
           ROOT/'su2_nqs/operators/su2_three_block_center.py',
           ROOT/'tests/test_su2_three_block_center.py')
    registration=dict(task='alternating_both72_root_frozen_width_v1',j2=.125,
        parents={name:dict(path=str(p),sha256=file_sha256(p)) for name,p in parents.items()},
        node_sequence=[13,12],Krylov_vectors_per_center=4,arms=list(names),
        maximum_center_H_matvecs=16,maximum_wall_seconds=seconds_cap,
        no_attention=True,no_remote_jobs=True,no_A1_or_MC_H_queries=True,
        budget_guard='before EACH center,1.15*max completed wall*remaining arms of that side+120s export reserve',
        source_sha256={str(p):file_sha256(p) for p in files},
        fixed_all_nodes_below72=True,all_target864_bonds=True,
        parent_retained_in_each_trial_space=True,raw_not_projected_A1=True)
    atomic_json(output/'registration.json',registration)
    for path in files:copyfile(path,output/('registered_'+path.name))
    start=perf_counter();result=dict(registration=registration,status='running',arms={},
        actual_center_H_matvecs=0,actual_extra_coarse_H_columns=0,A1_H_queries=0,MC_records=0)
    previous=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('registered600-second upper sweep cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds_cap)
    try:
        source,source_provenance=load_source(.125)
        result['source_provenance']=source_provenance
        states={name:load_checkpoint(path)['params'] for name,path in parents.items()}
        original={name:load_checkpoint(path)['params'] for name,path in parents.items()}
        for name,state in states.items():
            result['arms'][name]=dict(before=raw_layer_energy(state,source,.125),stages=[])
        # Interleave arms at each side, rather than finishing one arm if time expires.
        for node in (13,12):
            for arm_index,name in enumerate(names):
                observed=[row['elapsed_seconds'] for arm in result['arms'].values() for row in arm['stages']]
                allowed,forecast=center_stage_fits_budget(observed,
                    remaining_seconds=max(0.,seconds_cap-(perf_counter()-start)),
                    remaining_arms_in_stage=len(names)-arm_index)
                if not allowed:
                    result['status']='budget_partial_before_next_center'
                    result['unstarted_center']=dict(node=node,arm=name,
                        predicted_stage_and_export_seconds=forecast,
                        remaining_seconds=seconds_cap-(perf_counter()-start))
                    # All completed stage checkpoints remain available. Export
                    # only the common completed prefix in a separate zero-H job.
                    return result
                tick=perf_counter();state=states[name]
                candidate,quality=joint_upper_root_probe(state,source,.125,
                                                         upper_node=node,krylov_vectors=4)
                if any(not np.array_equal(w,candidate['maps'][i][q])
                       for i,maps in original[name]['maps'].items() if i<12 for q,w in maps.items()):
                    raise ArithmeticError('a frozen lower subspace changed')
                before=float(state['blocks'][-1].h_by_q[0][0,0].real)
                after=float(candidate['blocks'][-1].h_by_q[0][0,0].real)
                if after>before+2e-8:raise ArithmeticError('parent feasible raw energy increased')
                states[name]=candidate
                folder=output/name;folder.mkdir(exist_ok=True)
                save_checkpoint(folder/f'after_node{node}.pkl',candidate,optimizer_state=None,
                    metadata=dict(registration=registration,node=node,raw_only=True))
                product=quality['joint_Ritz']
                result['actual_center_H_matvecs']+=product['center_H_matvecs']+product['previously_evaluated_parent_H_products']
                result['actual_extra_coarse_H_columns']+=quality['additional_upper72_coarse_H_columns']
                row=dict(node=node,quality=quality,after=raw_layer_energy(candidate,source,.125),
                         elapsed_seconds=perf_counter()-tick)
                result['arms'][name]['stages'].append(row)
                atomic_json(output/'progress.json',result)
                print(name,node,after/144,row['elapsed_seconds'],flush=True)
        sample_path=ROOT/'projects/clebsch_tree_nqs/results/ct_conditional_nested_initialization_20261023/support_stage_2048/support_prefix.npz'
        sample=np.load(sample_path)
        # Frozen legal rich-support configurations; no new Monte Carlo.
        bits=np.asarray(sample['terminal_states'])[:4]
        raw=jax.jit(jax.vmap(make_feature_lut_amplitude(cfg),in_axes=(None,0)))
        for name,state in states.items():
            folder=output/name
            bank,embedding=native_bank_from_state(source,state)
            reference=np.asarray([active_amplitude(source,state,row) for row in bits])
            actual=np.asarray(raw(prepare_feature_lut_bank(bank,cfg),bits))
            scale=max(float(np.max(abs(reference))),1e-300)
            replay=float(np.max(abs(actual-reference))/scale)
            if replay>2e-10:raise ArithmeticError('native-bank amplitude replay failed')
            norm=triangular_raw_overlap(bank,bank,cfg)
            if abs(norm-1)>2e-10:raise ArithmeticError('exported raw norm failed')
            metadata=dict(model=dict(length=12,multiplicity_cap=20,maximum_q=10),j2=.125,
                          arm=name+'_both72',embedding=embedding,no_A1_energy=True,
                          optimizer_reset_required=True)
            save_checkpoint(folder/'state.pkl',state,metadata=metadata)
            save_checkpoint(folder/'initialized_bank.pkl',bank,metadata=metadata)
            result['arms'][name]['export']=dict(norm_real=float(norm.real),norm_imag=float(norm.imag),
                raw_amplitude_relative_replay_error=replay,embedding=embedding,
                state_sha256=file_sha256(folder/'state.pkl'),bank_sha256=file_sha256(folder/'initialized_bank.pkl'))
        result['status']='completed_two_sided_raw_candidate_construction'
    except TimeoutError as error:
        result['status']='bounded_incomplete';result['error']=str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result['seconds']=perf_counter()-start;atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
