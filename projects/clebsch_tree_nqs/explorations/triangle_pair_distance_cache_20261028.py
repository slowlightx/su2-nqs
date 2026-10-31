"""Frozen chirality channels at genuinely different distances from one8-RDM.

Zero new amplitudes, samples, H actions, training, PSD repair or SU2 twirl.
All spatial contrasts and conditional ratios share the original128 chain
deletions. Only two opposite-face distances are available: no fitted length,
thermodynamic phase, or QSL-exclusive identification is inferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.multipartite_geometry import partial_trace_sites
from su2_nqs.benchmarks.singlet_physics_readout import scalar_chirality_matrix
from su2_nqs.benchmarks.su2_multiplicity_intervention import pair_spin_projectors,triangle_vector_chirality
from su2_nqs.benchmarks.triangle_pair_geometry import disjoint_elementary_triangle_pairs,frozen_six_site_order
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256

ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'projects/clebsch_tree_nqs/results'
OUTPUT=RESULTS/'su2_spatial_frozen_channels_local_20261028'
SOURCE=RESULTS/'spin_spatial_vit_gpu_continuation_7600_v1/analysis/regions78_raw_v2/physics.npz'


def se(values: np.ndarray) -> float:
    """Delete-one-chain SE; first row is point, remaining rows are deletions."""
    x=np.asarray(values)[1:]
    return float(np.sqrt((len(x)-1)/len(x)*np.sum((x-x.mean())**2)))


def linear_summary(values: np.ndarray,names: tuple[str,...]) -> dict:
    return dict(means={k:dict(value=float(values[0,i]),chain_jackknife_SE=se(values[:,i])) for i,k in enumerate(names)})


def summarize(values: np.ndarray,names: tuple[str,...]) -> dict:
    """Same-chain connected and doublet-sector-conditional physical channels."""
    report=linear_summary(values,names)
    col=lambda k:values[:,names.index(k)]
    connected_T=(16/3)*(col('chi_A_chi_B')-col('chi_A')*col('chi_B'))
    connected_k=col('kappa_A_dot_B')-sum(col('kappaA_'+str(i))*col('kappaB_'+str(i)) for i in range(3))
    report['connected']=dict(T=dict(value=float(connected_T[0]),chain_jackknife_SE=se(connected_T)),
        kappa=dict(value=float(connected_k[0]),chain_jackknife_SE=se(connected_k)),
        T_minus_raw=float(connected_T[0]-col('T')[0]),kappa_minus_raw=float(connected_k[0]-col('kappa_A_dot_B')[0]))
    report['conditional']={}
    for key,num,den in [('c0','T_qtotal0','P_1_1_0'),('c1','T_qtotal2','P_1_1_2')]:
        a,b=col(num),col(den)
        if np.any(b<=0):
            report['conditional'][key]=dict(withheld=True,reason='nonpositive sector estimate; no floor')
            continue
        ratio=a/b
        pair=values[1:,[names.index(num),names.index(den)]]
        centered=pair-pair.mean(0)
        cov=(len(pair)-1)/len(pair)*(centered.T@centered)
        report['conditional'][key]=dict(value=float(ratio[0]),paired_chain_jackknife_SE=se(ratio),
            numerator=num,denominator=den,numerator_denominator_covariance=cov.tolist(),
            warning='Ratio bias is not removed by jackknife SE')
    t0,t1=col('T_qtotal0'),col('T_qtotal2')
    report['algebra']=dict(T_decomposition_max_error=float(np.max(abs(col('T')-t0-t1))),
        kappa_decomposition_max_error=float(np.max(abs(col('kappa_A_dot_B')+9/4*t0-3/4*t1))),
        warning='T0=T/4-kappa/3 and T2=3T/4+kappa/3: the two projections are not two independent signals beyond T andkappa')
    return report


def main() -> None:
    start=perf_counter()
    if OUTPUT.exists():raise FileExistsError('new immutable spatial artifact required')
    OUTPUT.mkdir(parents=True)
    meta=json.loads((SOURCE.parent/'results.json').read_text())
    if file_sha256(SOURCE)!=meta['raw_sha256'] or meta['step']!=7600:
        raise ValueError('same frozen7600 regional source identity required')
    with np.load(SOURCE) as f:
        sites=f['compact8__region_sites'].tolist()
        rhos=np.concatenate([f['compact8__rho'][None],f['compact8__jackknife_rhos']],axis=0)
        ids,counts=np.unique(f['compact8__chain_ids'],return_counts=True)
        if len(ids)!=128 or not np.all(counts==4) or len(rhos)!=129:
            raise ValueError('same128-chain/four-record original protocol required')
    frozen=RESULTS/'su2_mechanism_local_20261028/interventions'
    with np.load(frozen/'frozen_six_body.npz') as f:ops={k:f[k] for k in f.files}
    with np.load(frozen/'frozen_operators.npz') as f:
        for key in ('P_1_1_0','P_1_1_2'):ops[key]=f[key]
    for label,triangle in [('A',(0,3,4)),('B',(1,5,2))]:
        ops['chi_'+label]=scalar_chirality_matrix(6,triangle)
        for i,O in enumerate(triangle_vector_chirality(6,triangle)):ops['kappa'+label+'_'+str(i)]=O
    ps=pair_spin_projectors(6,(0,3))
    for k in ('T','T_qtotal0','T_qtotal2','kappa_A_dot_B'):
        after=sum(p@ops[k]@p for p in ps.values())
        ops[k+'__pair03_delta']=after-ops[k]
    names=tuple(ops);operator_array=np.stack(list(ops.values()))
    geometries=disjoint_elementary_triangle_pairs(sites)
    data=[];reports=[];keys=[]
    for i,g in enumerate(geometries):
        a,b=g['A']['sites'],g['B']['sites']
        order=frozen_six_site_order(a,b)
        keep=tuple(sites.index(s) for s in order)
        reduced=partial_trace_sites(rhos,keep)
        values=np.einsum('bij,kji->bk',reduced,operator_array)
        if abs(values.imag).max()>2e-10:raise ArithmeticError('nonreal physical Hermitian scalar estimator')
        values=values.real;data.append(values)
        key='pair'+str(i);keys.append(key)
        reports.append(dict(key=key,geometry=g,frozen_operator_site_order=list(order),
            dephased_physical_pair=[order[0],order[3]],readout=summarize(values,names)))
    data=np.stack(data,axis=1)
    near=next(i for i,g in enumerate(geometries) if set(g['A']['sites'])=={12,18,19} and set(g['B']['sites'])=={13,14,20})
    far=next(i for i,g in enumerate(geometries) if set(g['A']['sites'])=={12,18,19} and set(g['B']['sites'])=={14,15,21})
    delta=data[:,far]-data[:,near]
    comparison=dict(near=keys[near],far=keys[far],same_anchor=[12,18,19],
        same_up_down_flavors=True,paired_far_minus_near=linear_summary(delta,names),
        conditional_far_minus_near={})
    # Conditional contrast is difference of ratios, not ratio of differences.
    for label,num,den in [('c0','T_qtotal0','P_1_1_0'),('c1','T_qtotal2','P_1_1_2')]:
        ni,di=names.index(num),names.index(den)
        d=data[:,far,ni]/data[:,far,di]-data[:,near,ni]/data[:,near,di]
        comparison['conditional_far_minus_near'][label]=dict(value=float(d[0]),paired_chain_jackknife_SE=se(d))
    flattened=data[1:].reshape(128,-1);centered=flattened-flattened.mean(0)
    covariance=127/128*(centered.T@centered)
    np.savez_compressed(OUTPUT/'same_chain_channels.npz',values=data,spatial_pair_keys=np.asarray(keys),
        observable_names=np.asarray(names),same_chain_covariance=covariance)
    inventory=[]
    for label,region in [('compact6',[12,13,14,18,19,20]),('winding6',[12,13,14,15,16,17]),
                         ('local_hexagon6',[20,21,15,8,7,13]),('local_hexagon7',[20,21,15,8,7,13,14]),('compact8',sites)]:
        pairs=disjoint_elementary_triangle_pairs(region)
        inventory.append(dict(region=label,sites=region,disjoint_J1_triangle_pair_count=len(pairs),
            distances_squared=sorted(set(p['shortest_center_distance_squared'] for p in pairs))))
    j0=[]
    for p in (RESULTS/'coupling_quality_review_20261012').rglob('physics.npz'):
        if 'tri_j0' not in str(p):continue
        with np.load(p) as f:
            for key in f.files:
                if key.endswith('__region_sites'):
                    region=f[key].tolist()
                    j0.append(dict(source=str(p.relative_to(ROOT)),key=key,sites=region,
                        same_eight_sites=set(region)==set(sites)))
    report=dict(task='frozen_six_body_spatial_cache_v1',source=str(SOURCE.relative_to(ROOT)),source_sha256=file_sha256(SOURCE),
        checkpoint_sha256=meta['checkpoint_sha256'],parameter_sha256=meta['params_sha256'],step=7600,j2=.125,
        parent_region_sites=sites,physical_metric='dr²+dc²-dr*dc',state_records=512,independent_chains=128,
        operator_source_sha256={p.name:file_sha256(p) for p in (frozen/'frozen_six_body.npz',frozen/'frozen_operators.npz')},
        observable_order=list(names),pairs=reports,primary_same_anchor_distance_comparison=comparison,
        regional_geometry_inventory=inventory,J0_physical_cache_inventory=j0,
        J0_same_compact8_available=any(r['same_eight_sites'] for r in j0),
        minimum_new_readout_plan=dict(ordered_reference='same J0 physical7600-style compact8 or these exacttwo six-site supports',
            near_sites=[12,13,14,18,19,20],far_sites=[12,14,15,18,19,21],
            required_channels=['T','kappa_A_dot_B','P_1_1_0','P_1_1_2','chi_A','chi_B','kappaA','kappaB'],
            reused_channel_relations='T0=T/4-kappa/3;T2=3T/4+kappa/3',
            no_new_Schur_basis_or_full8_RDM_required=True),
        warnings=['Only two opposite-flavor distances in one6x6 state: no decay-length fit, critical exponent or long-range order claim.',
                  'Same-flavor r²4 pairs are separate face channels, not an intermediate point on the opposite-flavor curve.',
                  'Pair dephasing kills T andkappa algebraically for allstates; this zero is not a QSL signal.',
                  'Actual point RDMs are not repaired/twirled and no strict SU2 entanglement spectrum is inferred.'],
        actual_new_H_queries=0,actual_new_amplitude_queries=0,actual_new_samples=0,
        source_script_sha256=file_sha256(Path(__file__)),wall_seconds=perf_counter()-start)
    atomic_json(OUTPUT/'results.json',report)
    print(json.dumps(dict(output=str(OUTPUT),pairs=len(reports),J0_same_compact8_available=report['J0_same_compact8_available'],seconds=report['wall_seconds'])))


if __name__=='__main__':main()
