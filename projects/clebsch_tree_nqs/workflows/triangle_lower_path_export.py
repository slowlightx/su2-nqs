"""Common-layout export, independent root-H and endpoint amplitude checks."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter, process_time

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude,prepare_feature_lut_bank
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_sector_merge import project_sector_maps
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from .triangle_boundary_response_enrichment import active_environment
from .triangle_hierarchy_initialization import active_amplitude,native_bank_from_state,source_channels,split_bond_ledger
from .triangle_boundary_multisector import hierarchy_boundary_diagnostic
from .triangle_sector_transfer import load_source
from .triangle_lower_path import ROOT,clone


def run(directory: Path) -> dict:
    """Export parent/control/enriched states with identical active widths."""
    output=directory/'common_layout'
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('new immutable common-layout directory required')
    report=json.loads((directory/'results.json').read_text())
    if report['status']!='complete_matched_raw_lower_path_comparison':raise ValueError('both complete endpoints required')
    registration=dict(task='common_layout_export_independent_root_H',source_result_sha256=file_sha256(directory/'results.json'),
        source_sha256=file_sha256(Path(__file__)),full_physical_H_queries=0,new_MC_records=0,
        additional_coarse_root_H_projection_columns=3,
        replay_configurations_not_audit=True)
    atomic_json(output/'registration.json',registration)
    started=perf_counter();cpu=process_time()
    source,_=load_source(.125)
    _,joins,_=split_bond_ledger(12,.125)
    states={'parent':load_checkpoint(report['registration']['parent'])['params'],
        **{n:load_checkpoint(directory/n/'state.pkl')['params'] for n in ('old18','nested18')}}
    expanded=load_checkpoint(directory/'preserved_expanded.pkl')['params']
    bitsfile=ROOT/'projects/clebsch_tree_nqs/results/ct_conditional_nested_initialization_20261023/support_stage_2048/support_prefix.npz'
    bits=np.load(bitsfile)['terminal_states'][:4]
    cfg=MixedClebschConfig(12,20,10)
    amplitude=jax.jit(jax.vmap(make_feature_lut_amplitude(cfg),in_axes=(None,0)))
    result=dict(registration=registration,seed_configuration_file_sha256=file_sha256(bitsfile),states={})
    for name,state in states.items():
        # Independent root projection: do not reuse the cached Ritz value.
        projected_root=project_sector_maps(state['blocks'][28],state['blocks'][29],joins[14],
            maps=state['maps'][14],compact_boundary=True,max_output_dimension=1,
            memory_limit_bytes=768*1024**2)
        independent_energy=float(projected_root.h_by_q[0][0,0].real)
        reference_energy=report['before']['total'] if name=='parent' else report['arms'][name]['optimization']['raw_energy_after']
        root_H_error=abs(independent_energy-reference_energy)
        if root_H_error>2e-8:raise ArithmeticError('independent physical root-H disagrees with assigned Ritz result')
        padded=clone(state)
        padded['blocks'][30]=projected_root
        padded['maps'][0]=deepcopy(expanded['maps'][0]);padded['blocks'][16]=expanded['blocks'][16]
        for q,old in state['maps'][8].items():
            labels=source_channels(padded['blocks'][16].q_widths,padded['blocks'][17].q_widths,q)
            lookup={label:i for i,label in enumerate(labels)}
            new=np.zeros((len(labels),old.shape[1]),old.dtype)
            for i,label in enumerate(state['channels'][8][q]):new[lookup[label]]=old[i]
            padded['maps'][8][q]=new;padded['channels'][8][q]=labels
        bank,embedding=native_bank_from_state(source,padded)
        oldbank,_=native_bank_from_state(source,state)
        overlap=triangular_raw_overlap(oldbank,bank,cfg);norm=triangular_raw_overlap(bank,bank,cfg)
        expected=np.asarray([active_amplitude(source,state,row) for row in bits])
        actual=np.asarray(amplitude(prepare_feature_lut_bank(bank,cfg),bits))
        error=float(np.max(abs(actual-expected))/max(np.max(abs(expected)),1e-300))
        if max(abs(overlap-1),abs(norm-1),error)>2e-10:raise ArithmeticError('common layout lost state or amplitude')
        env,labels,gram=active_environment(padded,source)
        probabilities={q:float((q+1)*np.trace(env.downward[labels[16]][q][8:,8:]).real/env.norm_squared)
                       for q in (0,2,4)}
        save_checkpoint(output/(name+'_state.pkl'),padded,metadata=dict(registration=registration,raw_only=True))
        save_checkpoint(output/(name+'_bank.pkl'),bank,metadata=dict(registration=registration,
            model=dict(length=12,multiplicity_cap=20,maximum_q=10),j2=.125,embedding=embedding,
            optimizer_reset_required=True,complete_A1_quality_unknown=True,original_endpoint=name))
        result['states'][name]=dict(checkpoint=str((output/(name+'_bank.pkl')).resolve()),
            sha256=file_sha256(output/(name+'_bank.pkl')),raw_norm=float(norm.real),
            old_padded_overlap_error=float(abs(overlap-1)),active_native_relative_amplitude_error=error,
            maximum_upward_Gram_defect=gram,added_q_probability=probabilities,
            total_added_probability=sum(probabilities.values()),
            independent_root_H_total_energy=independent_energy,independent_root_H_vs_Ritz_error=root_H_error,
            hierarchy_response=hierarchy_boundary_diagnostic(padded,source,nodes=(0,8)),
            raw_energy_per_site=independent_energy/144)
    result['common_active_top_widths']=[{q:w.shape[1] for q,w in expanded['maps'][n].items()} for n in range(15)]
    columns=lambda state,node:sum(w.shape[1] for w in state['maps'][node].values())
    projections=dict(complement108=columns(states['parent'],8),complement126=columns(states['parent'],0),
        protected_embedding=sum(columns(expanded,n) for n in (0,8,12)),
        candidate_reprojection=sum(columns(states[name],n) for name in ('old18','nested18') for n in (8,12)),
        independent_root_verification=3)
    result['work_accounting']=dict(center_H_matvecs=report['center_H_matvecs_actual'],
        coarse_H_projection_columns=projections,total_coarse_H_projection_columns=sum(projections.values()),
        center_H_matvec_and_projection_column_are_different_work_units=True,
        original_numerical_wall_seconds=report['wall_seconds'],original_numerical_CPU_seconds=report['CPU_seconds'])
    result.update(wall_seconds=perf_counter()-started,CPU_seconds=process_time()-cpu,complete_A1_queries=0,
        full_physical_H_queries=0,additional_coarse_root_H_projection_columns=3,new_MC_records=0)
    atomic_json(output/'verification.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    run(parser.parse_args().directory)
