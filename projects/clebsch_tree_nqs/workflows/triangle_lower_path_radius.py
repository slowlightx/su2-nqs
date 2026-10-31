"""One posthoc fixed-radius fairness check; no new Krylov optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copyfile
import signal
from time import perf_counter,process_time

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_complement import descend_singlet_complement,restore_singlet_parent
from su2_nqs.operators.su2_conditional_response import _cut_coefficients,root_coefficient_matrices
from su2_nqs.operators.su2_three_block_center import ThreeBlockSingletOperator
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from .triangle_boundary_response_enrichment import active_environment
from .triangle_hierarchy_initialization import native_bank_from_state,split_bond_ledger
from .triangle_lower_path import TREE,clone,cross_bonds,real_maps,reproject
from .triangle_sector_transfer import load_source


def run(directory: Path) -> dict:
    """Return enriched candidate to the control's exact parent raw-FS radius."""
    output=directory/'matched_radius'
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('immutable matched-radius output')
    verification=json.loads((directory/'common_layout/verification.json').read_text())
    distances=json.loads((directory/'raw_distance_verification.json').read_text())
    target=distances['arms']['old18']['Fubini_Study_radians']
    registration=dict(task='one_posthoc_exact_control_raw_FS_interpolation',
        target_raw_FS_radians=target,posthoc_fairness_check=True,additional_optimization=False,
        radius_selection='exact saved old18 control distance; no scan or energy selection',
        maximum_wall_seconds=180.,maximum_CPU_seconds=180.,
        endpoints={n:verification['states'][n]['sha256'] for n in ('parent','old18','nested18')},
        source_sha256=file_sha256(Path(__file__)),new_center_H_matvecs=0,
        independent_root_H_projection_required=True,complete_A1_quality_unknown=True)
    atomic_json(output/'registration.json',registration)
    copyfile(Path(__file__),output/'registered_triangle_lower_path_radius.py')
    started=perf_counter();cpu=process_time();result=dict(registration=registration,status='running')
    previous=signal.getsignal(signal.SIGALRM)
    previous_cpu=signal.getsignal(signal.SIGPROF)
    def stop(*_):raise TimeoutError('fixed180-second interpolation budget')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGPROF,stop)
    signal.setitimer(signal.ITIMER_REAL,180.);signal.setitimer(signal.ITIMER_PROF,180.)
    try:
        source,_=load_source(.125)
        parent=load_checkpoint(directory/'common_layout/parent_state.pkl')['params']
        optimized=load_checkpoint(directory/'common_layout/nested18_state.pkl')['params']
        _,joins,bonds=split_bond_ledger(12,.125);supports=TREE.coarse_supports
        roots=root_coefficient_matrices(parent['maps'][14][0],parent['channels'][14][0])
        env=descend_singlet_complement(parent['blocks'][24],parent['blocks'][25],parent['blocks'][29],
            parent['maps'][12],roots,cross_bonds(supports[25],supports[29],bonds))
        support108=supports[25]+supports[29]
        newroots=root_coefficient_matrices(optimized['maps'][14][0],optimized['channels'][14][0])
        physical_cuts=_cut_coefficients(optimized['blocks'][24],optimized['blocks'][25],optimized['blocks'][29],
            optimized['maps'][12],newroots,2e-10)
        coefficients={q:c@env.complement_maps[q].conj() for q,c in physical_cuts.items()}
        reconstruction=max(float(np.max(abs(coefficients[q]@env.complement_maps[q].T-c)))
                           for q,c in physical_cuts.items())
        if reconstruction>2e-10:raise ArithmeticError('candidate leaves frozen108 complement span')
        op=ThreeBlockSingletOperator(parent['blocks'][16],parent['blocks'][17],env.complement,
            joins[8],cross_bonds(supports[16],support108,bonds),cross_bonds(supports[17],support108,bonds),
            memory_limit_bytes=768*1024**2,max_dimension=100_000)
        c0=op.encode(parent['maps'][8],env.coefficients)
        c1=op.encode(optimized['maps'][8],coefficients)
        c0=c0/np.linalg.norm(c0);c1=c1/np.linalg.norm(c1)
        overlap=np.vdot(c0,c1);c1=c1*np.conj(overlap)/abs(overlap);s=float(abs(overlap))
        angle=float(np.arccos(np.clip(s,0,1)))
        if not 0<target<angle:raise ValueError('control radius must lie strictly inside enriched radius')
        beta=float(np.sin(target)/np.sqrt(1-s*s));alpha=float(np.cos(target)-beta*s)
        center=alpha*c0+beta*c1
        if abs(np.vdot(center,center)-1)>2e-10:raise ArithmeticError('geodesic center lost normalization')
        maps36,coeff36,factor=op.factor(center,widths={q:w.shape[1] for q,w in parent['maps'][8].items()})
        maps72,root,restore=restore_singlet_parent(env,coeff36,{q:w.shape[1] for q,w in maps36.items()},
            parent_widths={q:w.shape[1] for q,w in parent['maps'][12].items()})
        candidate=clone(parent);candidate['maps'][8]=real_maps(maps36);candidate['maps'][12]=real_maps(maps72)
        candidate['maps'][14][0]=real_maps({0:np.asarray([root[q][a,b]
            for q,a,qr,b in parent['channels'][14][0]])[:,None]})[0]
        for node in (8,12,14):reproject(candidate,node,joins)
        energy=float(candidate['blocks'][30].h_by_q[0][0,0].real)
        e0=144*verification['states']['parent']['raw_energy_per_site']
        e1=144*verification['states']['nested18']['raw_energy_per_site']
        analytic=alpha*alpha*e0+(beta*beta+2*alpha*beta*s)*e1
        analytic_error=abs(analytic-energy)
        if analytic_error>2e-8:raise ArithmeticError('independent root H disagrees with Ritz-subspace interpolation')
        cfg=MixedClebschConfig(12,20,10)
        bank,embedding=native_bank_from_state(source,candidate)
        parentbank=load_checkpoint(verification['states']['parent']['checkpoint'])['params']
        native_overlap=triangular_raw_overlap(parentbank,bank,cfg)
        norm=triangular_raw_overlap(bank,bank,cfg)
        observed=float(np.arccos(np.clip(abs(native_overlap),0,1)))
        if max(abs(norm-1),abs(observed-target))>2e-10:raise ArithmeticError('native endpoint lost exact matched radius')
        density,labels,gram=active_environment(candidate,source)
        occupation={q:float((q+1)*np.trace(density.downward[labels[16]][q][8:,8:]).real/density.norm_squared)
                    for q in (0,2,4)}
        save_checkpoint(output/'state.pkl',candidate,metadata=dict(registration=registration,raw_only=True))
        save_checkpoint(output/'initialized_bank.pkl',bank,metadata=dict(registration=registration,
            model=dict(length=12,multiplicity_cap=20,maximum_q=10),j2=.125,embedding=embedding,
            complete_A1_quality_unknown=True,optimizer_reset_required=True))
        if op.matvec_count!=0:raise ArithmeticError('interpolation unexpectedly evaluated a center H product')
        result.update(status='complete_fixed_radius_raw_comparison',raw_energy_per_site=energy/144,
            control_raw_energy_per_site=verification['states']['old18']['raw_energy_per_site'],
            matched_minus_control_per_site=energy/144-verification['states']['old18']['raw_energy_per_site'],
            target_raw_FS_radians=target,observed_native_raw_FS_radians=observed,
            maximum_complement_reconstruction_error=reconstruction,analytic_root_H_error=analytic_error,
            native_norm=float(norm.real),maximum_upward_Gram_defect=gram,
            added_q_probability=occupation,total_added_probability=sum(occupation.values()),
            geodesic_alpha=alpha,geodesic_beta=beta,original_enriched_FS_radians=angle,
            factorization=factor,restoration=restore,
            checkpoint=str((output/'initialized_bank.pkl').resolve()),
            checkpoint_sha256=file_sha256(output/'initialized_bank.pkl'),
            new_center_H_matvecs=0,new_coarse_H_projection_columns=62+62+48+1,
            new_MC_records=0,complete_A1_queries=0)
    except TimeoutError as error:
        result.update(status='bounded_incomplete',error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.setitimer(signal.ITIMER_PROF,0)
        signal.signal(signal.SIGALRM,previous);signal.signal(signal.SIGPROF,previous_cpu)
        result.update(wall_seconds=perf_counter()-started,CPU_seconds=process_time()-cpu)
        atomic_json(output/'results.json',result)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    run(parser.parse_args().directory)
