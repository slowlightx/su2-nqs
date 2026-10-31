"""Frozen exact controls and static sum rules for symmetry-readout calibration.

Known NN-RVB family evidence, finite physical representatives and an analytic
virtual-boundary mechanism are kept distinct. This workflow does not optimize,
sample, query a neural amplitude/H, or identify the triangular model's phase.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from su2_nqs.benchmarks.spin_rotation_calibration import (
    aklt_local_tensors, boundary_even_charge_rotation, kagome_rvb_local_tensors,
    kagome_rvb_reference, lieb_mattis_singlet, manifold_region_algebra,
    pure_state_rdm, singlet_covering_state,
)
from su2_nqs.benchmarks.spin_rotation_disorder import regional_spin_rotation
from su2_nqs.benchmarks.spin_oscillator_strength import scalar_oscillator_strength
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds

ROOT = Path(__file__).resolve().parents[3]


def analyze(config_path: Path) -> dict:
    """Run a fixed exact reference/negative control/static-correlation protocol."""
    settings=json.loads(config_path.read_text())
    output=ROOT/settings['output']
    if output.exists():
        raise FileExistsError('preserve frozen output; use a distinct version')
    inputs={str(config_path.resolve().relative_to(ROOT)):file_sha256(config_path)}
    for entry in settings.get('spin_references',[]):
        for key in ('arrays','report'):
            path=ROOT/entry[key]
            digest=file_sha256(path)
            if digest!=entry[key+'_sha256']:
                raise ValueError('frozen spin-reference input changed')
            inputs[entry[key]]=digest
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration=dict(config=settings,inputs_sha256=inputs,sources=sources,
        known_bulk_family_vs_finite_calibration_kept_separate=True)
    output.mkdir(parents=True)
    atomic_json(output/'registration.json',dict(registration,
        result_sha256=canonical_sha256(registration)))
    started=perf_counter()
    angles=np.linspace(0,2*np.pi,settings['angle_points'])
    saved={'angles':angles}; references=[]; rotations=[]; mechanisms=[]; static=[]
    def guard():
        if perf_counter()-started>settings['seconds_cap']:
            raise TimeoutError('registered exact offline-analysis budget')
    # Verify genuine published local tensor identities, not an inferred edge
    # charge from a fixed-parity physical region.
    t=kagome_rvb_local_tensors(); p=t['site']; epsilon=t['epsilon']
    sx=np.array([[0.,.5],[.5,0.]]); sy=np.array([[0.,.5j],[-.5j,0.]])
    sz=np.diag([-.5,.5]); errors=[]
    for axis in (sx,sy,sz,.2*sx-.3*sy+.4*sz):
        u=expm(.73j*axis); w=np.eye(3,dtype=complex); w[:2,:2]=u
        errors.extend([np.max(abs(u@p.reshape(2,9)-p.reshape(2,9)@np.kron(w,w))),
            np.max(abs(np.einsum('ia,jb,kc,abc->ijk',w,w,w,epsilon)-epsilon))])
    virtual_center=np.eye(3,dtype=complex); virtual_center[:2,:2]=expm(2j*np.pi*sz)
    identity_report=dict(maximum_SU2_pullthrough_error=float(max(errors)),
        virtual_center_error=float(np.max(abs(virtual_center-t['J']))),
        virtual_q=t['virtual_q'].tolist(),virtual_m=t['virtual_m'].tolist(),
        ordinary_SU2_representation=True,relative_SO3_projective_sign_requires_topological_context=True,
        AKLT_same_virtual_half_spin_sign_is_non_QSL_counterexample=True)
    aklt=aklt_local_tensors()
    saved['AKLT_CG_tensor']=aklt['site']; saved['AKLT_singlet_bond']=aklt['singlet_bond']
    for rows,columns in settings['kagome_geometries']:
        guard(); rvb=kagome_rvb_reference(rows,columns); n=rvb['n_sites']
        prefix=f'kagome{n}'
        negative=singlet_covering_state(n,[rvb['edges'][i][:2] for i in rvb['covers'][0]])
        ordered=lieb_mattis_singlet(n,list(range(0,n,2)))
        saved[prefix+'__gram']=rvb['gram']
        algebra={}
        for sites in settings['local_algebra_regions']:
            gate=manifold_region_algebra(rvb['states'],sites)
            key=prefix+'__algebra'+str(len(sites))
            for name in ('transition_rdms','average_rdm','gram_eigenvalues'):
                saved[key+'__'+name]=gate[name]
            algebra[str(sites)]=dict(gram_eigenvalues=gate['gram_eigenvalues'].tolist(),
                orthonormality_error=gate['orthonormality_error'],
                local_algebra_defect_frobenius=gate['local_algebra_defect_frobenius'],
                metric='sqrt(sum_ab Frobenius^2 of all16 whitened transition defects); not sup operator norm',
                complete_local_algebra=True,diagonal_only=False)
        references.append(dict(n_sites=n,rows=rows,columns=columns,
            covering_counts=rvb['covering_counts'].tolist(),coverings=len(rvb['covers']),
            maximum_Gram_offdiagonal=float(np.max(abs(rvb['gram']-np.eye(4)))),
            local_algebra=algebra,known_family='published kagome NN-RVB SET family',
            finite_representative_topology_certified=False,geometry_scaling_certified=False))
        for name,state,known in [('known_family_RVB',rvb['total_state'],True),
                ('single_covering',negative,False),('SU2_restored_order',ordered,False)]:
            for sites in settings['physical_regions']:
                guard(); rho=pure_state_rdm(state,sites)
                report=regional_spin_rotation(rho,angles)
                key=prefix+'__'+name+'__A'+str(len(sites))
                saved[key+'__rho']=rho
                for axis in ('x','z','scalar'):
                    saved[key+'__'+axis]=report[axis]
                rotations.append(dict(array_prefix=key,n_sites=n,sites=sites,
                    state=name,known_positive_family=known,
                    pi_real=float(report['z'][len(angles)//2].real),
                    factorization_frobenius=report['factorization_frobenius'],
                    center_error=float(abs(report['z'][-1]-(-1)**len(sites))),
                    QSL_classifier=False))
    for length in settings['boundary_lengths']:
        guard(); report=boundary_even_charge_rotation(length,settings['vacuum_weight'],angles)
        key='boundary_L'+str(length)
        for name in ('z','rate','asymptotic_rate','lambda_plus','lambda_minus'):
            saved[key+'__'+name]=report[name]
        mechanisms.append(dict(length=length,pi_rate=float(report['rate'][len(angles)//2]),
            pi_asymptotic_rate=float(report['asymptotic_rate'][len(angles)//2]),
            finite_height_limit=True,physical_spin_region=False,known_bulk_positive=False))
    for entry in settings.get('spin_references',[]):
        guard(); report=json.loads((ROOT/entry['report']).read_text())
        if (report['status']!='complete' or report['checkpoint_sha256']!=entry['checkpoint_sha256']
                or report.get('input_repaired') or report.get('checkpoint_unchanged') is not True):
            raise ValueError('spin reference parent/completion proof failed')
        with np.load(ROOT/entry['arrays']) as a:
            correlations=a['spin_correlations']
        bonds=triangular_j1j2_bonds(6,6,j2=.125,periodic=True)
        coords=np.array([divmod(i,6) for i in range(36)])
        values=[]
        for kr in range(6):
            for kc in range(6):
                phase=np.exp(2j*np.pi*(coords@np.array([kr,kc]))/6)
                q=scalar_oscillator_strength(correlations,bonds,phase)
                values.append(dict(momentum_grid=[kr,kc],**q))
        static.append(dict(label=entry['label'],chi=entry['chi'],energy_per_site=entry['energy_per_site'],
            checkpoint_sha256=entry['checkpoint_sha256'],momenta=values,
            interpretation='static sum rule from two-body correlations only; no spectral or gap certificate',
            H_variance_available=False,SU2_exact=False))
    np.savez_compressed(output/'readouts.npz',**saved)
    result=dict(references=references,physical_rotations=rotations,local_tensor_identities=identity_report,
        conditional_boundary_mechanism=mechanisms,static_oscillator_references=static,
        seconds=perf_counter()-started,raw_sha256=file_sha256(output/'readouts.npz'),
        new_neural_H_calls=0,new_neural_amplitude_queries=0,new_samples=0,new_training=0,
        cusp_detected_in_triangular_data=False,QSL_claim=False,
        stopping_threshold='No finite partial-rotation value, covering label or virtual sign alone certifies intrinsic QSL. Further physical inference needs matched quality and topology/spectral context.')
    atomic_json(output/'results.json',dict(result,result_sha256=canonical_sha256(result)))
    completion=dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'))
    atomic_json(output/'completion.json',dict(completion,result_sha256=canonical_sha256(completion)))
    if settings.get('render_figure'):
        render_figure(result,saved,output)
    return result


def render_figure(result: dict, arrays: dict, output: Path) -> None:
    """Show raw exact reference curves and conditional mechanism, without bars."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.7),constrained_layout=True)
    angles=arrays['angles']/np.pi
    for ax,n in zip(axes[:2],(12,18)):
        for row in result['physical_rotations']:
            if row['n_sites']==n and len(row['sites'])==6:
                ax.plot(angles,arrays[row['array_prefix']+'__z'].real,label=row['state'].replace('_',' '))
        ax.axhline(0,color='.65',lw=.7); ax.set(title=f'Kagome {n}: actual A6 moment',xlabel=r'$\theta/\pi$',ylabel=r'$Z_A(\theta)$')
        ax.legend(fontsize=7)
    for length in (4,16,64,256):
        axes[2].plot(angles,arrays[f'boundary_L{length}__rate'],label=f'L={length}')
    axes[2].plot(angles,arrays['boundary_L256__asymptotic_rate'],'k--',label='conditional limit')
    axes[2].set(title='Virtual-boundary mechanism ONLY',xlabel=r'$\theta/\pi$',ylabel='exact finite rate')
    axes[2].legend(fontsize=7)
    fig.suptitle('Known-family control / false positives / conditional cusp — not triangular QSL certification',fontsize=10)
    fig.savefig(output/'calibration_controls.png',dpi=160);fig.savefig(output/'calibration_controls.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    r=analyze(parser.parse_args().config)
    print(json.dumps(dict(seconds=r['seconds'],known_family_representatives=len(r['references']),QSL_claim=r['QSL_claim'])))
