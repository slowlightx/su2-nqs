"""Magnetic-rotor and dimer controls for actual U(1) Schmidt eigenvalues.

This analysis does not assign q labels to a DMRG parent. Its three-sublattice
reference checks physical SU(2), while the actual finite-MPS plot is labeled
by m=2Sz. Degeneracies and rotor-like levels are not themselves phase proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.coplanar_entanglement_control import (
    coplanar_collective_singlet_entanglement,
)
from projects.clebsch_tree_nqs.workflows.dmrg_region_probe import _canonical


def run(output: Path, actual_inputs: list[Path]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    reference = coplanar_collective_singlet_entanglement(12,6)
    serial_reference = {k:v for k,v in reference.items() if not isinstance(v,np.ndarray)}
    dimer = dict(n_region=18, crossing_dimers=6, internal_region_dimers=6,
        entanglement_energies=[6*np.log(2)]*64,
        von_neumann_entropy_nats=float(6*np.log(2)),
        q_sectors={str(q):dict(q=q, multiplicity_dimension=comb(6,(6-q)//2)
            -(comb(6,(6-q)//2-1) if (6-q)//2>0 else 0), magnetic_degeneracy=q+1)
                   for q in range(0,7,2)},
        strict_su2_verified=True,
        warning='Product of six crossing singlets and internal singlets. Spin multiplets occur without a magnetic rotor or a QSL.')
    actual = []
    for path in actual_inputs:
        record = json.loads(path.read_text())
        worker = record.get('readout') if 'supervisor' in record else record
        if worker['status'] != 'complete' or 'readout' not in worker:
            raise ValueError('complete sealed Schmidt worker result required')
        if worker.get('record_sha256') != _canonical({k:v for k,v in worker.items() if k!='record_sha256'}):
            raise ValueError('Schmidt worker result seal differs from its content')
        data = worker['readout']
        actual.append(dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_sha256=worker['checkpoint_sha256'], chi=worker['actual_chi'],
            cut=data['cut'], von_neumann_entropy_nats=data['von_neumann_entropy_nats'],
            renyi2_entropy_nats=data['renyi2_entropy_nats'],
            magnetic_spin_inversion_spectrum_l1=data['magnetic_spin_inversion_spectrum_l1'],
            leading_sector_energies={m:s['entanglement_energies'][:16]
                                     for m,s in data['sectors'].items() if s['weight']>1e-8},
            q_labels_assigned=data['q_labels_assigned'],
            strict_su2_verified=data['strict_su2_verified']))
    report = dict(reference=serial_reference, crossing_dimer_control=dimer, actual=actual,
        interpretation=dict(
            coplanar_rotor_low_spin_counting='2S+1 SU(2) multiplets per S, total (2S+1)^2 states; body-axis anisotropy can split multiplets',
            canonical_collinear_rotor_counting='one SU(2) multiplet per S, total 2S+1 states',
            actual_magnetic_degeneracies_are_insufficient=True,
            missing_rotor_does_not_prove_QSL=True,
            required_to_answer_J2_question=['verified rotation covariance or physical spin-labelled vectors',
                'J2=0 ordered baseline at matching region/boundary/quality',
                'low-energy family and separation from non-rotor levels',
                'region/system-size dependence']),
        new_training=0, new_samples=0, new_Hamiltonian_ED=0, phase_claim=False)
    (output/'controls_and_actual_spectra.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for q,s in reference['q_sectors'].items():
        if int(q)<=8:
            levels=np.asarray(s['entanglement_energies'])
            axes[0].plot(np.repeat(int(q)/2,len(levels)),levels-reference['q_sectors']['0']['entanglement_energies'][0],
                         'o',ms=5,color='#2469b1')
    axes[0].axhline(0., color='#b85b38',ls='--',
                    label='Six crossing dimers (flat, own minimum subtracted)')
    axes[0].set(xlabel=r'Physical subsystem spin $S_A$ (verified)',ylabel=r'$\xi-\xi_{\min}$',
                title='Three-sublattice magnetic singlet: N=36, A=18',ylim=(-.1,8))
    axes[0].legend(fontsize=8)
    for index,parent in enumerate(actual):
        color=plt.get_cmap('tab10')(index)
        baseline=min(x for levels in parent['leading_sector_energies'].values() for x in levels if x is not None)
        for m,levels in parent['leading_sector_energies'].items():
            if 0<=int(m)<=6:
                data=[x-baseline for x in levels[:12] if x is not None]
                axes[1].plot(np.repeat(int(m)/2+.035*index,len(data)),data,'o',ms=3.5,alpha=.75,color=color,
                             label=f"DMRG chi={parent['chi']}" if int(m)==0 else None)
    axes[1].set(xlabel=r'Physical subsystem magnetic charge $m_A/2$',
                ylabel=r'$\xi-\xi_{\min}$', title='Actual J2=0.125 MPS: A=18',ylim=(-.1,8))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(output/'magnetic_rotor_and_actual_schmidt.png',dpi=180)
    fig.savefig(output/'magnetic_rotor_and_actual_schmidt.pdf')
    plt.close(fig)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--actual',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    result=run(args.output,args.actual)
    print(json.dumps(dict(actual_parents=len(result['actual']),phase_claim=False)))


if __name__=='__main__':
    main()
