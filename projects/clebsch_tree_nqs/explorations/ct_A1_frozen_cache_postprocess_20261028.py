"""Zero-query frozen-scalar/paired-dephasing analysis of complete-A1 CT caches.

Only reads completed caches. Does not import or modify the running physical
amplitude workflow, prepare a wavefunction, query H, fit or repair an RDM.
The local nonselective measurement is an observable intervention, not a new
globally A1-projected optimized variational state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.su2_multiplicity_intervention import pair_spin_projectors
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028'
SITES = np.asarray([12,13,14,18,19,20])
GROUPS = 32


def paired_statistics(numerators: np.ndarray, mass: np.ndarray, names: tuple[str,...],
                      ratios: dict[str,tuple[str,str]]) -> dict:
    """Same-record SNIS means and delete-group numerator/denominator covariance.

    Every original record is retained, including exact target nodes. The
    conditional ratios are ratios of two physical observable means; their
    uncertainty uses both entries from the SAME deletion, not independent SE.
    """
    if len(mass)!=512 or numerators.shape!=(512,len(names)):
        raise ValueError('fixed512-record/32-group cached calculation required')
    if np.any(mass<0) or not np.isfinite(mass).all() or mass.sum()<=0:
        raise ValueError('finite nonnegative original importance mass required')
    total_n = numerators.sum(axis=0); total_d = mass.sum()
    group_n = numerators.reshape(GROUPS,-1,len(names)).sum(axis=1)
    group_d = mass.reshape(GROUPS,-1).sum(axis=1)
    if np.any(total_d-group_d<=0):
        raise ValueError('a delete-group target normalization vanished; no clipping')
    point = total_n/total_d
    leaves = (total_n-group_n)/(total_d-group_d)[:,None]
    real = leaves.real; centered = real-real.mean(axis=0)
    covariance = (GROUPS-1)/GROUPS*(centered.T@centered)
    report = dict(observable_order=list(names),linear_same_group_covariance=covariance.tolist(),
        observables={name:dict(real=float(point[i].real),imaginary=float(point[i].imag),
            real_group_jackknife_SE=float(np.sqrt(covariance[i,i])),
            imaginary_group_jackknife_SE=float(np.sqrt((GROUPS-1)/GROUPS*
                np.sum((leaves[:,i].imag-leaves[:,i].imag.mean())**2))))
            for i,name in enumerate(names)},conditional_ratios={})
    for label,(num,den) in ratios.items():
        a,b=names.index(num),names.index(den)
        if point[b].real<=0 or np.any(leaves[:,b].real<=0):
            report['conditional_ratios'][label]=dict(withheld=True,reason='nonpositive estimated sector weight; no floor/repair')
            continue
        ratio=point[a].real/point[b].real
        deleted=leaves[:,a].real/leaves[:,b].real
        se=float(np.sqrt((GROUPS-1)/GROUPS*np.sum((deleted-deleted.mean())**2)))
        report['conditional_ratios'][label]=dict(value=float(ratio),paired_group_jackknife_SE=se,
            numerator=num,denominator=den,
            numerator_denominator_covariance=covariance[np.ix_([a,b],[a,b])].tolist(),
            warning='Conditional ratio has finite-sample ratio bias; JK SE does not remove it')
    return report


def run(output: Path) -> dict:
    start=perf_counter()
    if output.exists():
        raise FileExistsError('write a new immutable postprocessing output')
    sources=dict(six_body=CAMPAIGN/'interventions/frozen_six_body.npz',
        supported=CAMPAIGN/'interventions/frozen_operators.npz',
        reference=CAMPAIGN/'regions/frozen_projectors.npz')
    with np.load(sources['six_body']) as f:
        ops={k:f[k] for k in ('T','T_qtotal0','T_qtotal2','chi_A_chi_B')}
    with np.load(sources['supported']) as f:
        for k in ('vector_chirality_A_dot_B','vector_chirality_qtotal0','vector_chirality_qtotal2','P_1_1_0','P_1_1_2'):
            ops[k]=f[k]
    with np.load(sources['reference']) as f:
        if not np.array_equal(f['compact6_sites'],SITES):
            raise ValueError('reference projector region order mismatch')
        ops['Pstar']=f['compact6_q0_reference_main_projector']
        ops['Pq0']=f['compact6_q0_projector']
    projectors=pair_spin_projectors(6,(0,3))
    expanded={}
    for name,O in ops.items():
        after=sum(p@O@p for p in projectors.values())
        expanded.update({name:O,name+'__after_pair03':after,name+'__delta_pair03':after-O})
    names=tuple(expanded)
    ratios={}
    for suffix in ('','__after_pair03','__delta_pair03'):
        for label,num,den in [('T0_given_P110','T_qtotal0','P_1_1_0'),
                              ('T2_given_P112','T_qtotal2','P_1_1_2'),
                              ('Pstar_given_q0','Pstar','Pq0')]:
            # Denominators commute with the scalar pair-spin measurement.
            ratios[label+suffix]=(num+suffix,den)
    cases={};missing=[]
    for tag,directory in [('ct6_j0',CAMPAIGN/'ct_A1_compact6'),
                          ('ct6_j0125',CAMPAIGN/'ct_A1_compact6_j0125')]:
        path=directory/(tag+'_physical_region.npz')
        result=directory/(tag+'_results.json')
        if not path.exists() or not result.exists():
            missing.append(dict(tag=tag,expected=str(path.relative_to(ROOT))))
            continue
        original=json.loads(result.read_text())
        if original['status']!='complete_exploratory_full_A1_readout' or original['complete_A1_images']!=432:
            raise ValueError('complete fullA1 amplitude cache required')
        with np.load(path) as f:
            rows,sites,phi,q=[f[k] for k in ('rows','sites','region_amplitudes','proposal')]
        if not np.array_equal(sites,SITES):
            raise ValueError('physical cache sites disagree with frozen operators')
        products=ROOT/'projects/clebsch_tree_nqs/results/ct_status_20261028/verified_next_parents'/tag/'parent_snapshot/products.npz'
        if file_sha256(products)!=original['source_products_sha256']:
            raise ValueError('original full-support IID source products changed')
        packed=rows[:,sites]@(2**np.arange(5,-1,-1))
        center=phi[np.arange(len(rows)),packed]
        with np.load(products) as f:
            if not np.array_equal(rows,f['rows'][:512]) or not np.array_equal(q,f['proposal'][:512]):
                raise ValueError('cache does not preserve original IID records and Q')
            replay=float(np.max(abs(center-f['psi'][:512,1]))/max(np.max(abs(f['psi'][:512,1])),1e-300))
        if replay>2e-11:
            raise ValueError('cached center amplitudes do not replay frozen endpoint')
        mass=abs(center)**2/q
        numerators=np.stack([center.conj()*np.sum(O[packed]*phi,axis=1)/q for O in expanded.values()],axis=1)
        report=paired_statistics(numerators,mass,names,ratios)
        report.update(source=str(path.relative_to(ROOT)),source_sha256=file_sha256(path),
            source_result_sha256=file_sha256(result),endpoint_parameter_sha256=original['parameter_sha256'],
            original_source_products_sha256=original['source_products_sha256'],center_replay_relative_error=replay,
            global_step=original['global_step'],records=512,independent_IID_groups=32,
            norm_effective_contributions=float(mass.sum()**2/(mass@mass)),
            maximum_norm_record_fraction=float(mass.max()/mass.sum()),exact_target_nodes=int(np.count_nonzero(center==0)))
        cases[tag]=report
    if not cases:
        raise FileNotFoundError('no completed physical amplitude cache is available')
    output.mkdir(parents=True)
    report=dict(task='CT_fullA1_frozen_scalar_dephasing_cache_postprocess_v1',
        cases=cases,not_yet_available=missing,physical_sites=SITES.tolist(),pair_spin_dephasing=[0,3],
        source_operators={k:dict(path=str(v.relative_to(ROOT)),sha256=file_sha256(v)) for k,v in sources.items()},
        source_script_sha256=file_sha256(Path(__file__)),
        maximum_sector_denominator_dephasing_error=max(float(np.linalg.norm(expanded[k+'__after_pair03']-ops[k])) for k in ('P_1_1_0','P_1_1_2','Pq0')),
        dephased_operator_frobenius_norms={k:float(np.linalg.norm(expanded[k+'__after_pair03'])) for k in ('T','T_qtotal0','T_qtotal2','chi_A_chi_B','vector_chirality_A_dot_B')},
        actual_new_amplitude_queries=0,actual_new_H_queries=0,actual_new_sampler_records=0,
        RDM_PSD_repair=False,RDM_SU2_twirl=False,cached_rows_not_a_new_untouched_audit=True,
        intervention_scope='All-outcome local SU2-scalar pair-spin measurement, evaluated as adjoint-channel operators on the same parent records. Not a newly optimized fullA1 global state; no full energy claim.',
        kinematic_warning='T and kappa-dot vanish after this pair-spin dephasing by operator algebra for every state; that zero is not a QSL fingerprint. State-dependent magnitudes and conditional channels require controls.',
        wall_seconds=perf_counter()-start)
    atomic_json(output/'results.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.output)
    print(json.dumps(dict(output=str(args.output),cases=list(report['cases']),
        not_yet_available=report['not_yet_available'],wall_seconds=report['wall_seconds'])),flush=True)
