"""Bounded cached-RDM physical multiplicity interventions; no sampling/training."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from su2_nqs.benchmarks.su2_multiplicity_intervention import (
    compact6_intervention_operators, pair_spin_dephase, pair_spin_projectors,
    chirality_unitary, intervention_observables, triangle_multiplicity_identity)
from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028/interventions'
BASE=Path('projects/clebsch_tree_nqs/results')


def covering_state(pairs):
    """Explicit down/up computational-basis product of oriented singlets."""
    state=np.zeros(64,complex)
    for packed in range(64):
        bits=[(packed>>(5-i))&1 for i in range(6)]
        value=1.
        for i,j in pairs:
            value*=0 if bits[i]==bits[j] else (1 if bits[i]==0 else -1)/np.sqrt(2)
        state[packed]=value
    return state


def summarize(rho, jackknifes=None):
    operators=compact6_intervention_operators()
    before=intervention_observables(rho)
    reports={}
    for pair in ((0,3),(3,4),(0,4)):
        changed, probabilities=pair_spin_dephase(rho,pair)
        after=intervention_observables(changed)
        delta={k:after[k]-before[k] for k in before}
        row=dict(pair=list(pair),outcome_probabilities={str(k):v for k,v in probabilities.items()},
            all_outcomes_retained=True,after=after,delta=delta,
            trace_error=float(abs(np.trace(changed)-np.trace(rho))),
            joint_spin_weight_max_drift=max(abs(v) for k,v in delta.items() if k.startswith('P_')),
            RDM_change_frobenius=float(np.linalg.norm(changed-rho)))
        if jackknifes is not None:
            p=pair_spin_projectors(6,pair)
            delta_ops=np.stack([sum(x@op@x for x in p.values())-op for op in operators.values()])
            leaves=np.einsum('nij,kji->nk',jackknifes,delta_ops).real
            se=np.sqrt((len(leaves)-1)/len(leaves)*np.sum((leaves-leaves.mean(0))**2,axis=0))
            row['paired_delta_chain_jackknife_SE']=dict(zip(operators,se.tolist()))
            row['independent_cached_chain_deletions']=len(leaves)
        reports['_'.join(map(str,pair))]=row
    unitary_reports={}
    for angle in (-np.pi/4,np.pi/4):
        unitary=chirality_unitary(6,(0,3,4),angle)
        after=intervention_observables(unitary@rho@unitary.conj().T)
        unitary_reports[str(angle)]={k:after[k]-before[k] for k in before}
    raising=np.zeros((64,64),complex)
    for packed in range(64):
        for site in range(6):
            mask=1<<(5-site)
            if not packed&mask:raising[packed|mask,packed]+=1
    decomposition={}
    for doubled_q,spin_dot_coefficient in ((0,-.75),(2,.25)):
        contribution=before['vector_chirality_qtotal'+str(doubled_q)]
        weight=before['P_1_1_'+str(doubled_q)]
        tau_weight=contribution/(3*spin_dot_coefficient)
        decomposition[str(doubled_q)]=dict(qA=1,qB=1,qtotal=doubled_q,
            joint_probability=weight,kappa_dot_contribution=contribution,
            spin_dot_eigenvalue=spin_dot_coefficient,
            tau_y_A_tau_y_B_weight=tau_weight,
            conditional_tau_y_A_tau_y_B=(tau_weight/weight if weight>1e-14 else None))
        if jackknifes is not None:
            cop=operators['vector_chirality_qtotal'+str(doubled_q)]
            pop=operators['P_1_1_'+str(doubled_q)]
            cleave=np.einsum('nij,ji->n',jackknifes,cop).real
            pleave=np.einsum('nij,ji->n',jackknifes,pop).real
            for name,values in [('kappa_dot_contribution',cleave),('joint_probability',pleave),
                    ('conditional_tau_y_A_tau_y_B',cleave/(3*spin_dot_coefficient*pleave))]:
                decomposition[str(doubled_q)][name+'_chain_jackknife_SE']=float(np.sqrt(
                    (len(values)-1)/len(values)*np.sum((values-values.mean())**2)))
    return dict(before=before,dephasing=reports,chirality_unitary_delta=unitary_reports,
        doublet_multiplicity_decomposition=decomposition,
        actual_RDM_diagnostics=dict(trace_error=float(abs(np.trace(rho)-1)),
            hermiticity_frobenius=float(np.linalg.norm(rho-rho.conj().T)),
            minimum_eigenvalue=float(np.linalg.eigvalsh(rho).min()),
            SU2_raising_commutator=float(np.linalg.norm(rho@raising-raising@rho))),
        strict_SU2_entanglement_spectrum_claim=False)


def main():
    start=time.monotonic();OUTPUT.mkdir(parents=True,exist_ok=True)
    cases=[
        ('ViT_J0_step1000',0.,BASE/'coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step01000_regional/physics.npz','compact6__rho','compact6__jackknife_rhos'),
        ('ViT_J0125_step6600',.125,BASE/'spin_spatial_vit_gpu_continuation_6600_v1/analysis/six_site_readout/physics.npz','compact6__rho','compact6__jackknife_rhos'),
        ('ViT_J0125_step7600',.125,BASE/'spin_spatial_vit_gpu_continuation_7600_v1/analysis/witness_regions_v1/physics.npz','compact6__rho','compact6__jackknife_rhos'),
        ('DMRG_J0125_chi4096',.125,BASE/'spin_tensor_network_continuation_v2/chi4096/rdms.npz','chi4096_compact6',None),
        ('DMRG_J0125_chi5120',.125,BASE/'spin_tensor_network_continuation_v2/chi5120_probe/rdms.npz','chi5120_compact6',None),
        ('DMRG_J0125_chi10240',.125,BASE/'coupling_quality_review_20261012/latest_inputs/dmrg/work/chi10240_from8192_v1/rdms.npz','chi10240_compact6',None)]
    reports={};missing=[]
    for label,j2,path,key,jack in cases:
        path=ROOT/path
        with np.load(path) as data:
            if key not in data.files:
                missing.append(dict(label=label,path=str(path.relative_to(ROOT)),missing_key=key));continue
            rho=data[key];leaves=data[jack] if jack and jack in data.files else None
            if 'compact6__region_sites' in data.files:
                if data['compact6__region_sites'].tolist()!=[12,13,14,18,19,20]:
                    raise ValueError('physical compact6 order changed')
            report=summarize(rho,leaves)
        report.update(j2=j2,source_npz=str(path.relative_to(ROOT)),source_sha256=file_sha256(path),RDM_key=key,
            supported_energy_before=report['before']['H_J1_supported']+j2*report['before']['H_J2_supported'])
        for value in report['dephasing'].values():
            value['supported_energy_delta']=value['delta']['H_J1_supported']+j2*value['delta']['H_J2_supported']
        reports[label]=report
    cover=covering_state(((0,3),(1,5),(4,2)))
    cover2=covering_state(((3,4),(5,2),(0,1)))
    rvb=cover+cover2;rvb/=np.linalg.norm(rvb)
    rvb_odd=cover-cover2;rvb_odd/=np.linalg.norm(rvb_odd)
    magnetic=_three_plus_three_singlet_columns((0,3,4))[:,-1]
    for label,state in [('single_dimer_cover',cover),('two_cover_resonance',rvb),
            ('two_cover_opposite_phase_resonance',rvb_odd),('quartet_pair_magnetic_singlet',magnetic)]:
        reports[label]=summarize(np.outer(state,state.conj()))
    frozen=compact6_intervention_operators()
    frozen_path=OUTPUT/'frozen_operators.npz'
    if frozen_path.exists():
        with np.load(frozen_path) as existing:
            if set(existing.files)!=set(frozen) or any(not np.array_equal(existing[k],v) for k,v in frozen.items()):
                raise ValueError('existing frozen physical operators changed')
    else: np.savez_compressed(frozen_path,**frozen)
    atomic_json(OUTPUT/'results.json',dict(task='physical_SU2_multiplicity_intervention_v1',
        region_sites=[12,13,14,18,19,20],A=[12,18,19],B=[13,20,14],
        definition='D_pair(rho)=sum_q=0,2 P_pair,q rho P_pair,q=(rho+Swap_pair rho Swap_pair)/2; no outcome discarded',
        hypotheses='Does a physical multiplicity measurement alter vector-chirality organization while preserving measured block-spin screening? Not a QSL-specific witness.',
        scope='Only compact6-supported observables. Neither full-H energy nor a new globally A1 projected state/response is inferred. No SU2 twirl/PSD repair.',
        no_new_samples=True,no_new_H_queries=True,no_new_training=True,
        cases=reports,missing_inputs=missing,wall_seconds=time.monotonic()-start,
        frozen_operators_sha256=file_sha256(OUTPUT/'frozen_operators.npz'),
        exact_three_spin_multiplicity_identity=triangle_multiplicity_identity(),
        source_manifest=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))))
    # This projector is selected/frozen independently by the regional route.
    # Do not refit its eigenvector on a ViT state or dephased RDM.
    reference=OUTPUT.parent/'regions/frozen_projectors.npz'
    if reference.exists():
        with np.load(reference) as data:
            projector=data['compact6_q0_reference_main_projector']
        ps=pair_spin_projectors(6,(0,3))
        delta_op=sum(p@projector@p for p in ps.values())-projector
        effects={}
        for label,j2,path,key,jack in cases:
            with np.load(ROOT/path) as data:
                if key not in data.files:continue
                rho=data[key]
                row=dict(before=float(np.trace(rho@projector).real),
                    delta=float(np.trace(rho@delta_op).real),RDM_key=key,
                    source_npz=str(path),source_sha256=file_sha256(ROOT/path))
                row['after']=row['before']+row['delta']
                if jack and jack in data.files:
                    values=np.einsum('nij,ji->n',data[jack],delta_op).real
                    row['paired_delta_chain_jackknife_SE']=float(np.sqrt(
                        (len(values)-1)/len(values)*np.sum((values-values.mean())**2)))
                effects[label]=row
        atomic_json(OUTPUT/'frozen_projector_dephasing.json',dict(
            reference_projector=str(reference.relative_to(ROOT)),
            reference_projector_sha256=file_sha256(reference),
            key='compact6_q0_reference_main_projector',pair=[0,3],cases=effects,
            discovery='Projector selected from chi10240, so that source is not a held-out validation.',
            scope='Linear supported physical observable after all-outcome scalar dephasing; not a GME/QSL witness or global energy.'))
    for label,r in reports.items():
        print(label,'kappa=',r['before']['vector_chirality_A_dot_B'],
            'dephase_delta=',r['dephasing']['0_3']['delta']['vector_chirality_A_dot_B'],
            'joint_p_drift=',r['dephasing']['0_3']['joint_spin_weight_max_drift'])


if __name__=='__main__':main()
