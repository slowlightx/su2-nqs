"""Zero-query physical interpretation and red-team audit of frozen channels.

No training, Born samples, amplitude queries, Hamiltonian queries or ED.
Statistics are algebraic functions of existing scalar means, with complete
same-chain/group jackknife propagation. They are not new independent data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.multipartite_geometry import partial_trace_sites
from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.benchmarks.su2_multiplicity_intervention import (
    compact6_intervention_operators, pair_spin_dephase,
)
from su2_nqs.benchmarks.su2_six_body_chirality import six_body_chirality_operators
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.recoupling import tree_basis_matrix
from projects.clebsch_tree_nqs.workflows.su2_multiplicity_intervention_local import covering_state

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028'
NAMES=('T','kappa','p0','p1','c0','c1','c0_minus_c1','p_doublet_doublet',
       'conditional_T','conditional_U','conditional_TU','cov_T_U','corr_T_U')
FOUR=('T','kappa_A_dot_B','P_1_1_0','P_1_1_2')
ops=six_body_chirality_operators()
supported=compact6_intervention_operators()
MATRICES=np.stack([ops['T'],ops['kappa_A_dot_B'],supported['P_1_1_0'],supported['P_1_1_2']])


def statistics(x: np.ndarray) -> np.ndarray:
    """Conditional commuting-variable statistics; no new independent observable."""
    t,k,p0,p1=np.asarray(x).T
    pd=p0+p1
    t0=t/4-k/3; t1=3*t/4+k/3
    with np.errstate(divide='ignore',invalid='ignore'):
        c0=np.where(p0>1e-12,t0/p0,np.nan)
        c1=np.where(p1>1e-12,t1/p1,np.nan)
        ex=np.where(pd>1e-12,t/pd,np.nan)
        ey=np.where(pd>1e-12,(-.75*p0+.25*p1)/pd,np.nan)
        exy=np.where(pd>1e-12,k/(3*pd),np.nan)
        cov=exy-ex*ey
        varx=1-ex**2;vary=p0*p1/pd**2
        corr=np.where((varx>1e-12)&(vary>1e-12),cov/np.sqrt(varx*vary),np.nan)
    return np.stack((t,k,p0,p1,c0,c1,c0-c1,pd,ex,ey,exy,cov,corr),axis=-1)


def jk_cov(leaves: np.ndarray) -> np.ndarray:
    """Standard complete delete-chain/group covariance, retaining pairing."""
    centered=leaves-leaves.mean(axis=0)
    return (len(leaves)-1)/len(leaves)*(centered.T@centered)


def report(x: np.ndarray, leaves: np.ndarray | None) -> dict:
    values=statistics(x);deleted=statistics(leaves) if leaves is not None else None
    result={}
    for i,name in enumerate(NAMES):
        point=values[i]
        se=(np.sqrt(jk_cov(deleted[:,i:i+1])[0,0]) if deleted is not None and
            np.isfinite(deleted[:,i]).all() else None)
        result[name]=dict(value=float(point) if np.isfinite(point) else None,
            same_chain_or_group_jackknife_SE=float(se) if se is not None else None)
    return dict(statistics=result,delete_units=len(leaves) if leaves is not None else 0,
        covariance=jk_cov(deleted).tolist() if deleted is not None else None,
        ratio_bias_not_removed=True,not_an_entanglement_witness=True)


def moment(rhos: np.ndarray) -> np.ndarray:
    return np.einsum('...ab,kba->...k',rhos,MATRICES).real


def shapley(a: np.ndarray,b: np.ndarray) -> np.ndarray:
    """Exact symmetric decomposition of delta kappa; not a causal attribution."""
    sa,sb=statistics(a),statistics(b)
    pa,pb=sa[...,2:4],sb[...,2:4]
    ca,cb=sa[...,4:6],sb[...,4:6]
    factors=np.asarray([-9/4,3/4])
    occupation=np.sum(factors*(pb-pa)*(ca+cb)/2,axis=-1)
    conditional=np.sum(factors*(cb-ca)*(pa+pb)/2,axis=-1)
    total=sb[...,1]-sa[...,1]
    return np.stack((total,occupation,conditional,conditional/total),axis=-1)


def run(output: Path) -> dict:
    started=perf_counter()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refuse overwriting red-team evidence')
    original=json.loads((BASE/'interventions/six_body_results.json').read_text())
    cache={};cases={}
    for name,source in original['cases'].items():
        if 'source_npz' not in source:continue
        path=ROOT/source['source_npz']
        with np.load(path) as f:
            x=moment(f[source['RDM_key']])
            leaves=moment(f['compact6__jackknife_rhos']) if name.startswith('ViT') else None
        cache[name]=(x,leaves)
        cases[name]=dict(report(x,leaves),source=source['source_npz'],source_sha256=file_sha256(path))
    # Complete-A1 CT caches preserve each original IID importance record.
    for name,directory in [('CT_J0_466','ct_A1_compact6'),('CT_J0125_492','ct_A1_compact6_j0125')]:
        tag='ct6_j0' if name=='CT_J0_466' else 'ct6_j0125'
        path=BASE/directory/(tag+'_physical_region.npz')
        with np.load(path) as f:
            rows,sites,phi,q=[f[k] for k in ('rows','sites','region_amplitudes','proposal')]
        packed=rows[:,sites]@(1<<np.arange(5,-1,-1))
        center=phi[np.arange(len(rows)),packed]
        mass=abs(center)**2/q
        numerator=np.stack([center.conj()*np.sum(o[packed]*phi,axis=1)/q for o in MATRICES],axis=1).real
        x=numerator.sum(axis=0)/mass.sum()
        ng=numerator.reshape(32,-1,4).sum(axis=1);mg=mass.reshape(32,-1).sum(axis=1)
        leaves=(numerator.sum(axis=0)-ng)/(mass.sum()-mg)[:,None]
        cache[name]=(x,leaves)
        cases[name]=dict(report(x,leaves),source=str(path.relative_to(ROOT)),source_sha256=file_sha256(path),
            old_IID_records_reused=True,new_untouched_audit=False)
    # Both distances are partial traces of the SAME compact8 RDM and deletions.
    path=ROOT/'projects/clebsch_tree_nqs/results/spin_spatial_vit_gpu_continuation_7600_v1/analysis/regions78_raw_v2/physics.npz'
    with np.load(path) as f:
        rho=f['compact8__rho'];deleted=f['compact8__jackknife_rhos'];sites=f['compact8__region_sites'].tolist()
    for name,physical in [('near',(12,13,14,18,19,20)),('far',(12,14,15,18,19,21))]:
        keep=[sites.index(i) for i in physical]
        x=moment(partial_trace_sites(rho,keep));leaves=moment(partial_trace_sites(deleted,keep))
        cache['ViT7600_'+name]=(x,leaves)
        cases['ViT7600_'+name]=dict(report(x,leaves),source=str(path.relative_to(ROOT)),
            source_sha256=file_sha256(path),sites=physical)
    contrasts={};decompositions={}
    pairs=[('coupling','ViT_J0_step1000','ViT_J0125_step7600',False),
        ('J0_quality','ViT_J0_step750','ViT_J0_step1000',False),
        ('J0125_quality','ViT_J0125_step6600','ViT_J0125_step7600',False),
        ('DMRG_quality','DMRG_J0125_chi4096','DMRG_J0125_chi10240',False),
        ('CT_coupling','CT_J0_466','CT_J0125_492',False),
        ('same_chain_distance','ViT7600_near','ViT7600_far',True)]
    for label,an,bn,paired in pairs:
        a,al=cache[an];b,bl=cache[bn]
        diff=statistics(b)-statistics(a)
        variance=(jk_cov(statistics(bl)-statistics(al)) if paired else
            jk_cov(statistics(al))+jk_cov(statistics(bl)) if al is not None and bl is not None else None)
        contrasts[label]=dict(source_a=an,source_b=bn,same_deletions=paired,
            statistics={n:dict(delta=float(diff[i]),SE=float(np.sqrt(variance[i,i])) if variance is not None else None)
                for i,n in enumerate(NAMES)})
        attribution=shapley(a,b)
        avar=(jk_cov(shapley(al,bl)) if paired else
            jk_cov(shapley(al,np.broadcast_to(b,al.shape)))+jk_cov(shapley(np.broadcast_to(a,bl.shape),bl))
            if al is not None and bl is not None else None)
        decompositions[label]=dict(source_a=an,source_b=bn,
            values={n:dict(value=float(attribution[i]),SE=float(np.sqrt(avar[i,i])) if avar is not None else None)
                for i,n in enumerate(('delta_kappa','occupation','conditional_internal','conditional_fraction'))},
            covariance=avar.tolist() if avar is not None else None,
            exact_sum_error=float(abs(attribution[0]-attribution[1]-attribution[2])),
            causal_interpretation=False)
        if avar is not None and abs(attribution[0])<2*np.sqrt(avar[0,0]):
            decompositions[label]['values']['conditional_fraction']=dict(value=None,SE=None,
                reason='total change unresolved; fraction unstable, no clipping or biased percent claim')
    # Explicit false-positive controls: a classical coplanar120 product and
    # a genuine12-spin singlet projection of two copies of that ordered motif.
    coords=np.asarray([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])
    ordered=np.ones(1,complex)
    for angle in 2*np.pi*coords.sum(axis=1)/3:
        ordered=np.kron(ordered,np.asarray([1,np.exp(-1j*angle)])/np.sqrt(2))
    cover=covering_state(((0,3),(1,5),(4,2)));other=covering_state(((3,4),(5,2),(0,1)))
    controls={'single_dimer':cover,'RVB_plus':(cover+other)/np.linalg.norm(cover+other),
        'RVB_minus':(cover-other)/np.linalg.norm(cover-other),
        'quartet_pair_singlet':_three_plus_three_singlet_columns((0,3,4))[:,-1],
        'coplanar120_product':ordered}
    projectors,_=tree_basis_matrix(FusionTree.balanced(12),0,0)
    projected=projectors@(projectors.conj().T@np.kron(ordered,ordered))
    projected/=np.linalg.norm(projected)
    matrix=projected.reshape(64,64)
    controls_rho={k:np.outer(v,v.conj()) for k,v in controls.items()}
    controls_rho['coplanar120_global12_singlet_region6']=matrix@matrix.conj().T
    reference=np.load(BASE/'regions/frozen_projectors.npz')['compact6_q0_reference_main_projector']
    control_report={}
    for name,r in controls_rho.items():
        x=moment(r);after,_=pair_spin_dephase(r,(0,3))
        control_report[name]=dict(report(x,None),frozen_Pstar_before=float(np.trace(r@reference).real),
            frozen_Pstar_delta=float(np.trace((after-r)@reference).real),
            is_an_optimized_Hamiltonian_candidate=False,
            undefined_if_screening_channel_has_no_variance=True)
    # Full energy response is two-body and includes EVERY environment bond.
    energy={};perm=np.arange(36);perm[[12,18]]=perm[[18,12]]
    for label,source in [('chi5120','spin_tensor_network_continuation_v2/global_readouts/chi5120_spin/regions/spin/rdm.npz'),
        ('chi10240','coupling_quality_review_20261012/latest_inputs/dmrg/readouts/chi10240_sweep32_spin_env_v1/regions/spin/rdm.npz')]:
        path=ROOT/'projects/clebsch_tree_nqs/results'/source
        c=np.load(path)['spin_correlations'];parts=[];es=[];inner=[]
        region={12,13,14,18,19,20}
        for j1,j2 in [(1.,0.),(0.,1.)]:
            bonds=triangular_j1j2_bonds(6,6,j1=j1,j2=j2,periodic=True)
            parts.append(.5*sum(g*(c[perm[i],perm[j]]-c[i,j]) for i,j,g in bonds))
            inner.append(.5*sum(g*(c[perm[i],perm[j]]-c[i,j]) for i,j,g in bonds if i in region and j in region))
            es.append(sum(g*c[i,j] for i,j,g in bonds))
        source_case=original['cases']['DMRG_J0125_'+label]
        local_rho=np.load(ROOT/source_case['source_npz'])[source_case['RDM_key']]
        sites=(12,13,14,18,19,20)
        mismatch=max(abs(np.trace(local_rho@spin_dot_matrix(6,i,j)).real-c[sites[i],sites[j]])
            for i in range(6) for j in range(i+1,6))
        global_meta=json.loads(path.with_name('readout.json').read_text())
        expected_energy=(-.5145360655301436 if label=='chi5120' else -.5152498958883492)
        energy_error=abs((es[0]+.125*es[1])/36-expected_energy)
        admitted=bool(mismatch<1e-10 and energy_error<1e-9)
        energy[label]=dict(delta_total_energy_J1=float(parts[0]),delta_unweighted_J2=float(parts[1]),
            delta_total_energy_at_J0125=float(parts[0]+.125*parts[1]),
            supported_inside_compact6_delta_energy=float(inner[0]+.125*inner[1]),
            environment_incident_delta_energy=float(parts[0]+.125*parts[1]-inner[0]-.125*inner[1]),
            parent_energy_per_site=float((es[0]+.125*es[1])/36),source=str(path.relative_to(ROOT)),
            checkpoint_sha256=global_meta['checkpoint_sha256'],
            source_sha256=file_sha256(path),statistical_SE=None,variational_quality_error_not_zero=True,
            raw_spin_correlations_not_connected=True,
            same_parent_local_RDM_pair_max_error=float(mismatch),
            recomputed_vs_recorded_parent_energy_per_site_error=float(energy_error),
            admitted_as_physical_energy_control=admitted,
            warning=None if admitted else 'Legacy global Cij cache fails same-checkpoint RDM/energy closure; quarantined, not used as chi drift')
    output.mkdir(parents=True,exist_ok=True)
    result=dict(task='handedness_spin_interpretation_redteam_v2',cases=cases,contrasts=contrasts,
        symmetric_product_decompositions=decompositions,physical_controls=control_report,
        full_pair_dephasing_energy=energy,seconds=perf_counter()-started,
        definitions=dict(D='P(q_A=1,q_B=1)',T='(16/3) chi_A chi_B',U='S_A dot S_B',
            covariance='E[T U|D]-E[T|D]E[U|D]',T0='T/4-kappa/3',T1='3T/4+kappa/3',
            energy='Delta E=(1/2)<Swap_(12,18) H Swap_(12,18)-H>; total local cost, not per site'),
        warnings=['All statistics are algebraic functions of the same existing T,kappa,p0,p1; no new independent information.',
            'Two distances do not determine a correlation length or exponent.',
            'Near and far displacement vectors differ in BOTH length and direction: (-1,4)/3 and (-1,7)/3 in physical120degree coordinates; not a radial decay fit.',
            'Conditional ratios and correlations have finite-sample ratio bias; JK propagates uncertainty but does not remove bias.',
            'Classical coplanar120 and projected magnetic-singlet controls must be checked before any QSL interpretation.',
            'Spatially averaged dephasing preserves lattice-invariance of an invariant density matrix, not necessarily support in the Hilbert Gamma-A1 irrep.',
            'Only six momenta cached for matching ViT7600, not enough to invert all Cij; no ViT full energy cost invented.'],
        actual_new_samples=0,actual_new_amplitude_queries=0,actual_new_H_queries=0,new_ED=0,
        script_sha256=file_sha256(Path(__file__)))
    # Deliberately replace undefined conditional quantities by null, not zero.
    for control in result['physical_controls'].values():control['covariance']=None
    atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'handedness_redteam_v2')
    args=parser.parse_args()
    run(args.output)
