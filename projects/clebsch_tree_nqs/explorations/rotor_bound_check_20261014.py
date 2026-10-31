"""Independently calibrate the general positive-rotor bound and actual spectra."""
from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.workflows.dmrg_region_probe import _canonical
from su2_nqs.benchmarks.coplanar_entanglement_control import coplanar_collective_singlet_entanglement
from su2_nqs.benchmarks.rotor_spectrum_template import (
    conditional_magnetic_rotor_test, positive_rotor_spin2_bound,
)


def rotor_boundary_counterexample(epsilon: float, rotor_triplet_gap: float) -> dict:
    """A strict SU(2) rotor plus two boundary spin halves violates the low ratio.

    H_E=(Delta/2) J_rotor^2 + epsilon P_edge,triplet. The first term remains
    a complete positive SO(3) rotor. The second is the positive scalar
    epsilon*(s_left dot s_right+3/4). The physical spin is the sum of all
    three spins. Low spectra need not satisfy the single-rotor inequality.
    q<=4 rotor sectors suffice to audit the minima; higher rotor sectors
    have larger energy and cannot change them.
    """
    if not np.isfinite([epsilon,rotor_triplet_gap]).all() or min(epsilon,rotor_triplet_gap)<=0:
        raise ValueError('positive finite boundary and rotor gaps required')
    levels={}
    for qr in (0,2,4):
        rotor_energy=rotor_triplet_gap*qr*(qr+2)/8
        rotor_multiplicity=qr+1
        for qb,boundary_energy in ((0,0.),(2,epsilon)):
            for physical_q in range(abs(qr-qb),qr+qb+1,2):
                levels.setdefault(str(physical_q),[]).append(dict(
                    rotor_q=qr,boundary_q=qb,energy=rotor_energy+boundary_energy,
                    independent_physical_multiplets=rotor_multiplicity,
                    magnetic_degeneracy=physical_q+1))
    for family in levels.values():
        family.sort(key=lambda x:x['energy'])
    minima={q:min(x['energy'] for x in family) for q,family in levels.items()}
    partition=sum((int(q)+1)*sum(x['independent_physical_multiplets']*np.exp(-x['energy'])
                  for x in family) for q,family in levels.items())
    return dict(model='H_E=(Delta/2) J_rotor^2+epsilon P_triplet(edge spin-half pair)',
        epsilon=float(epsilon),rotor_triplet_gap=float(rotor_triplet_gap),
        coefficients=[rotor_triplet_gap/2]*3,
        physical_total_spin='J_total=J_rotor+s_left+s_right',
        q_sectors=levels,physical_spin_minima=minima,
        physical_spin2_to_spin1_ratio=minima['4']/minima['2'],
        isolated_rotor_spin2_to_spin1_ratio=3.,
        strict_su2_symmetry=True,positive_so3_rotor_is_present=True,
        lowest_positive_rotor_template_fails=bool(minima['4']>3*minima['2']),
        finite_partition_function_up_to_rotor_S2=float(partition),
        next_spatial_probe=dict(
            observable='three balanced row magnetizations in the lowest physical m=+1 Schmidt vector; sum=1',
            ideal_three_sublattice_collective_control=[1/3,1/3,1/3],
            pure_boundary_pair_control=[.5,0.,.5],
            caution='Uniform row weights are a property of this ideal control, not a theorem for all ordered states; ordinary crossing dimers can also produce the boundary profile.'),
        warning='The low rotor-ratio violation can coexist with a higher rotor family and trivial boundary spins. It does not exclude bulk magnetic order or certify fractionalization/QSL.')


def run(output: Path, actual_inputs: list[Path], covariance_inputs: list[Path] | None=None) -> dict:
    if output.exists():
        raise FileExistsError('use a fresh rotor-template sidecar')
    rng=np.random.default_rng(20261014)
    coefficients=np.concatenate([rng.lognormal(0,3,size=(5000,3)),
        np.asarray(list(permutations([0.,1.,100.]))),
        np.asarray([[0,0,0],[0,0,1],[0,1,1],[1,1,1],[1,1,1e6]])])
    values=[positive_rotor_spin2_bound(a) for a in coefficients]
    normalized_excess=np.asarray([(v['spin2_minimum']-v['bound'])/max(1.,max(v['coefficients']))
                                 for v in values])
    formula_error=np.asarray([v['spin1_formula_error']/max(1.,max(v['coefficients'])) for v in values])
    if normalized_excess.max()>128*np.finfo(float).eps:
        raise RuntimeError('independent positive-rotor validation violated the analytic bound')
    reference=coplanar_collective_singlet_entanglement(12,6)
    baseline=reference['q_sectors']['0']['entanglement_energies'][0]
    e1=min(reference['q_sectors']['2']['entanglement_energies'])-baseline
    e2=min(reference['q_sectors']['4']['entanglement_energies'])-baseline
    covariance_records={}
    for path in covariance_inputs or []:
        loaded=json.loads(path.read_text())
        worker=loaded['readout'] if 'supervisor' in loaded else loaded
        if (worker.get('status')!='complete'
                or worker.get('record_sha256')!=_canonical({k:v for k,v in worker.items() if k!='record_sha256'})):
            raise ValueError('complete sealed physical covariance worker required')
        covariance_records[worker['checkpoint_sha256']]=(path,worker)
    actual=[]
    for path in actual_inputs:
        loaded=json.loads(path.read_text())
        worker=loaded['readout'] if 'supervisor' in loaded else loaded
        if (worker.get('status')!='complete'
                or worker.get('record_sha256')!=_canonical({k:v for k,v in worker.items() if k!='record_sha256'})):
            raise ValueError('complete sealed physical Schmidt worker required')
        spectra={m:s['entanglement_energies'] for m,s in worker['readout']['sectors'].items()}
        parent=dict(chi=worker['actual_chi'],source=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_sha256=worker['checkpoint_sha256'],
            conditional_test=conditional_magnetic_rotor_test(spectra))
        if worker['checkpoint_sha256'] in covariance_records:
            covariance_path,cov=covariance_records[worker['checkpoint_sha256']]
            content=cov['readout']['leading_spin_content']['sectors']
            assignments={}
            for charge in ('0','2','4'):
                sector=content[charge]
                first=sector['leading_levels'][0]
                if abs(first['entanglement_energy']-parent['conditional_test']['magnetic_sector_minima'][charge])>1e-9:
                    raise ValueError('covariance and spectrum workers disagree on their physical leading vector')
                assignments[charge]=dict(physical_casimir_expectation=first['physical_casimir_expectation'],
                    minimal_allowed_q=first['minimal_allowed_q'],
                    higher_spin_weight_upper_bound_nominal=first['higher_spin_weight_upper_bound_nominal'],
                    retained_gram_condition_number=sector['retained_gram_condition_number'],
                    omitted_physical_density_weight=sector['omitted_physical_density_weight'],
                    numerical_bound_not_certified_interval=True,q_eigenvector_verified=False)
            parent['physical_spin_premises']=dict(source=str(covariance_path),
                source_sha256=hashlib.sha256(covariance_path.read_bytes()).hexdigest(),
                purity_normalized_commutator_squared=cov['readout']['purity_normalized_commutator_squared'],
                leading_magnetic_branches=assignments,
                strict_su2_labels_claimed=False,
                message='The actual leading m=0 and m=2 Schmidt vectors are tested by physical Casimir positivity, not inferred from spectral coincidences.')
        actual.append(parent)
    result=dict(
        proof=dict(model='H_E=A_x J_x^2+A_y J_y^2+A_z J_z^2; all A>=0',
            ordered_coefficients='a<=b<=c',
            exact_spin1_minimum='E1_min=a+b (Cartesian eigenvalues b+c, a+c, a+b)',
            spin2_trial='m_body=0 along c axis: <Ja^2>=<Jb^2>=3, <Jc^2>=0',
            rayleigh_bound='E2_min <= 3(a+b) = 3 E1_min',
            anisotropy_assumed_isotropic=False),
        numerical_validation=dict(seed=20261014,sample_count=len(values),
            maximum_scale_normalized_bound_excess=float(normalized_excess.max()),
            maximum_scale_normalized_spin1_formula_error=float(formula_error.max()),
            passing_absolute_roundoff_scale=128*np.finfo(float).eps),
        collective_singlet_reference=dict(n_sites=36,n_region=18,strict_su2_verified=True,
            spin1_gap=e1,spin2_gap=e2,ratio=e2/e1,bound_margin=3*e1-e2,
            pure_quadratic_rotor_assumed=False,
            note='Exact magnetic reference obeys the necessary rotor ratio but contains finite-spin corrections.'),
        dimer_control=dict(crossing_dimers=6,spin1_gap=0.,spin2_gap=0.,bound_satisfied=True,
            note='Flat dimer spectrum also obeys the inequality: satisfying it does not identify magnetic order.'),
        actual=actual,
        scope='Conditional finite-region positive-quadratic-rotor template test, not a bulk order or QSL certificate.',
        missing_premises=['physical rotation covariance',
            'lowest m=0 branch assigned spin zero',
            'lowest m=2 branch assigned spin one',
            'low-spectrum family accurately described by a complete positive quadratic rotor'],
        new_training=0,new_samples=0,new_Hamiltonian_ED=0,phase_claim=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--actual',type=Path,nargs='*',default=[])
    parser.add_argument('--covariance',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    result=run(args.output,args.actual,args.covariance)
    print(json.dumps(result['numerical_validation']))


if __name__=='__main__':
    main()
