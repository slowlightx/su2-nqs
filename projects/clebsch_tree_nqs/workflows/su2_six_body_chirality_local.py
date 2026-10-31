"""Measure a frozen pure-six-body scalar on cached RDMs, with paired JK."""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np

from su2_nqs.benchmarks.su2_six_body_chirality import six_body_chirality_operators,six_body_operator_diagnostics
from su2_nqs.benchmarks.singlet_physics_readout import scalar_chirality_matrix
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from .su2_multiplicity_intervention_local import covering_state
from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028/interventions'


def measurements(rho: np.ndarray,leaves: np.ndarray|None=None) -> dict:
    """All channels on the same physical RDM and the SAME chain deletions."""
    bank=six_body_chirality_operators();ops=np.stack(list(bank.values()))
    values=np.einsum('ij,kji->k',rho,ops)
    if abs(values.imag).max()>1e-9:raise ValueError('nonreal scalar expectation')
    values=values.real
    one_ops=np.stack([scalar_chirality_matrix(6,(0,3,4)),scalar_chirality_matrix(6,(1,5,2))])
    onepoints=np.einsum('ij,kji->k',rho,one_ops).real
    connected=(16/3)*(values[1]-np.prod(onepoints))
    report=dict(means=dict(zip(bank,values.tolist())),
        scalar_chirality_onepoints=dict(chi_A=float(onepoints[0]),chi_B=float(onepoints[1])),
        T_connected=float(connected),T_connected_minus_T=float(connected-values[0]),
        no_time_reversal_assumed=True,
        T_equals_sum_error=float(abs(values[0]-values[2]-values[3])),
        kappa_weighted_parts_error=float(abs(values[4]-(-9/4*values[2]+3/4*values[3]))))
    if leaves is not None:
        rows=np.einsum('nij,kji->nk',leaves,ops).real
        centered=rows-rows.mean(0)
        covariance=(len(rows)-1)/len(rows)*(centered.T@centered)
        one_leaves=np.einsum('nij,kji->nk',leaves,one_ops).real
        connected_leaves=(16/3)*(rows[:,1]-np.prod(one_leaves,axis=1))
        report.update(observable_order=list(bank),chain_jackknife_SE=dict(zip(bank,np.sqrt(np.diag(covariance)).tolist())),
            chain_jackknife_covariance=covariance.tolist(),chain_deletions=len(rows),
            T_variance_from_same_chain_parts=float(covariance[2,2]+covariance[3,3]+2*covariance[2,3]),
            T_variance_direct=float(covariance[0,0]),
            T_connected_chain_jackknife_SE=float(np.sqrt((len(rows)-1)/len(rows)*
                np.sum((connected_leaves-connected_leaves.mean())**2))),
            warning='Linear means and covariance share all chain deletions; T_q0/T_q2 errors are not treated as independent.')
    return report


def main():
    start=time.monotonic();OUTPUT.mkdir(parents=True,exist_ok=True)
    source=OUTPUT/'results.json';previous=json.loads(source.read_text())
    reports={}
    for label,case in previous['cases'].items():
        if 'source_npz' not in case:continue
        path=ROOT/case['source_npz']
        if file_sha256(path)!=case['source_sha256']:raise ValueError('cached input changed')
        with np.load(path) as data:
            report=measurements(data[case['RDM_key']],data['compact6__jackknife_rhos']
                if 'compact6__jackknife_rhos' in data.files else None)
        report.update(j2=case['j2'],source_npz=case['source_npz'],source_sha256=case['source_sha256'],RDM_key=case['RDM_key'])
        reports[label]=report
    path=ROOT/'projects/clebsch_tree_nqs/results/coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step00750_regional/physics.npz'
    if path.is_file():
        with np.load(path) as data:
            report=measurements(data['compact6__rho'],data['compact6__jackknife_rhos'])
        report.update(j2=0.,source_npz=str(path.relative_to(ROOT)),source_sha256=file_sha256(path),RDM_key='compact6__rho')
        reports['ViT_J0_step750']=report
    cover=covering_state(((0,3),(1,5),(4,2)))
    other=covering_state(((3,4),(5,2),(0,1)))
    for label,state in [('single_dimer_cover',cover),('two_cover_resonance',cover+other),
            ('two_cover_opposite_phase_resonance',cover-other),
            ('quartet_pair_magnetic_singlet',_three_plus_three_singlet_columns((0,3,4))[:,-1])]:
        state=state/np.linalg.norm(state)
        reports[label]=measurements(np.outer(state,state.conj()))
    frozen=OUTPUT/'frozen_six_body.npz';bank=six_body_chirality_operators()
    if frozen.exists():
        with np.load(frozen) as saved:
            if set(saved.files)!=set(bank) or any(not np.array_equal(saved[k],v) for k,v in bank.items()):
                raise ValueError('six-body frozen operators changed')
    else:np.savez_compressed(frozen,**bank)
    atomic_json(OUTPUT/'six_body_results.json',dict(task='frozen_SU2_six_body_chirality_v1',
        region_sites=[12,13,14,18,19,20],A=[12,18,19],B=[13,20,14],
        definition='T=(16/3)chi_Achi_B=T_qtotal0+T_qtotal2; kappaA.kappaB=(-9/4)T_qtotal0+(3/4)T_qtotal2',
        algebra=six_body_operator_diagnostics(),cases=reports,
        previous_intervention_results_sha256=file_sha256(source),
        frozen_six_body_sha256=file_sha256(frozen),wall_seconds=time.monotonic()-start,
        no_new_H_queries=True,no_new_samples=True,no_new_training=True,
        interpretation='T is an operator-linearly independent six-body scalar, not automatically a QSL diagnostic. Constancy or differences require shared error and own-quality-drift controls. kappa dot alone is four-spin and not beyond the full two-bond operator space.',
        source_manifest=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))))
    for name,r in reports.items():print(name,r['means'],'T_SE',r.get('chain_jackknife_SE',{}).get('T'))


if __name__=='__main__':main()
