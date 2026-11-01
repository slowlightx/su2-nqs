"""Exact postprocessing of archived N16 states; no ED, training or twirling.

This finite-state diagnostic distinguishes scalar accuracy from total-spin
purity and the spin content of S_0^z|psi>. It is not a GCNN reproduction,
not a current CT6 comparison, and not a thermodynamic phase classifier.
All total-spin labels in data/interfaces are doubled integers q=2S.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter, process_time

import numpy as np

from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'projects/clebsch_tree_nqs/results/spin_qsl_symmetry_diagnostics_v1'
EXPECTED_BUNDLE_SHA = '512cf84e0377263e823317f063e821e1cc9f1abb25220f0e8f9e692d3b9d306c'
EXPECTED_RESULT_SHA = '246a7318b77a6e7d4c430555162b6867fe917ed08ac2affb30fb96769bee0359'
EXPECTED_RESULT_CONTENT_SHA = '342cbc2459ae996c212fd3af4c7189ce174c25e5bbd0cec5fe6394d6e926543e'
OUTPUT = ROOT/'projects/clebsch_tree_nqs/results/spin_contamination_actual_n16_20261101'


def sector_components(state: np.ndarray, spin_squared: object, *, n_sites: int) -> dict[int, np.ndarray]:
    """Apply exact M=0 spectral polynomials P_q, q=0,2,...,N.

    This is a component decomposition of the original vector, not an SU(2)
    twirl or a claim that the original state has become spin invariant.
    """
    vector = np.asarray(state, dtype=np.complex128)
    if (isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 2
            or n_sites % 2 or vector.ndim != 1 or not np.isfinite(vector).all()
            or spin_squared.shape != (len(vector), len(vector))):
        raise ValueError('finite even-N fixed-M0 state and matching S² operator required')
    norm = np.linalg.norm(vector)
    if not norm > 0:
        raise ValueError('nonzero state required')
    vector = vector/norm
    eigenvalues = {q:q*(q+2)/4 for q in range(0, n_sites+1, 2)}
    components = {}
    for q, target in eigenvalues.items():
        part = vector.copy()
        for other_q, value in eigenvalues.items():
            if other_q != q:
                part = (np.asarray(spin_squared@part)-value*part)/(target-value)
        components[q] = part
    reconstruction = np.linalg.norm(sum(components.values())-vector)
    residual = max(np.linalg.norm(spin_squared@part-eigenvalues[q]*part)
                   for q, part in components.items())
    if reconstruction > 1e-9 or residual > 1e-8:
        raise ArithmeticError('spectral polynomial numerical accuracy gate failed')
    return components


def pair_readout(state: np.ndarray, basis: np.ndarray) -> dict[str, np.ndarray]:
    """Actual <Sx_i Sx_j>, <Sz_i Sz_j> and <Si.Sj> in fixed doubled M=0.

    Site zero is MSB; same-direction flips leave M=0 and thus contribute zero
    to Sx_iSx_j. No isotropy constraint is imposed on the measured state.
    """
    bits = np.asarray(basis)
    vector = np.asarray(state, dtype=np.complex128)
    if (bits.ndim != 2 or bits.dtype.kind not in 'iu' or len(bits) != len(vector)
            or not np.isin(bits, (0, 1)).all() or not np.all(bits.sum(1) == bits.shape[1]//2)
            or len(np.unique(bits, axis=0)) != len(bits)):
        raise ValueError('distinct integer-bit fixed-M0 basis required')
    vector = vector/np.linalg.norm(vector)
    n = bits.shape[1]
    packed = bits@(1 << np.arange(n-1, -1, -1))
    position = np.full(2**n, -1, dtype=np.int64); position[packed] = np.arange(len(bits))
    zz = (bits-.5).T@(abs(vector[:, None])**2*(bits-.5))
    xx = np.eye(n, dtype=np.complex128)/4
    for i in range(n):
        for j in range(i+1, n):
            selected = bits[:, i] != bits[:, j]
            targets = position[packed[selected] ^ (1 << (n-1-i)) ^ (1 << (n-1-j))]
            if np.any(targets < 0):
                raise ValueError('complete fixed-M0 basis required for pair readout')
            xx[i, j] = xx[j, i] = np.vdot(vector[targets], vector[selected])/4
    return dict(xx=xx, zz=zz, scalar=2*xx+zz,
                local_z=(abs(vector)**2)@(bits-.5))


def audit_state(state: np.ndarray, basis: np.ndarray, hamiltonian: object,
                spin_squared: object) -> dict:
    """Exact archived-state diagnostics, including sector content of a source."""
    vector = np.asarray(state, dtype=np.complex128); vector /= np.linalg.norm(vector)
    n = basis.shape[1]
    components = sector_components(vector, spin_squared, n_sites=n)
    weights = {q:float(np.vdot(v, v).real) for q, v in components.items()}
    hstate = hamiltonian@vector
    energy = complex(np.vdot(vector, hstate))
    variance = float(np.linalg.norm(hstate-energy*vector)**2)
    s2 = float(np.vdot(vector, spin_squared@vector).real)
    packed = basis@(1 << np.arange(n-1, -1, -1))
    position = np.full(2**n, -1, dtype=np.int64); position[packed] = np.arange(len(basis))
    rotated = (-1)**(n//2)*vector[position[packed ^ (2**n-1)]]
    even_residual = float(np.linalg.norm(rotated-vector))
    pair = pair_readout(vector, basis)
    source = (basis[:, 0]-.5)*vector
    source_norm = float(np.vdot(source, source).real)
    source_parts = sector_components(source, spin_squared, n_sites=n)
    source_weights = {q:float(np.vdot(v, v).real) for q, v in source_parts.items()}
    cross_energy = sum(complex(np.vdot(v, hamiltonian@v)) for v in components.values())
    p0 = components[0]/np.linalg.norm(components[0])
    singlet_energy = float(np.vdot(p0, hamiltonian@p0).real)
    singlet_pairs = pair_readout(p0, basis)
    singlet_source = sector_components((basis[:, 0]-.5)*p0, spin_squared, n_sites=n)
    return dict(energy_total=energy.real, energy_imaginary=energy.imag,
        variance_total=variance, energy_per_site=energy.real/n, variance_per_site=variance/n,
        spin_squared_expectation=s2, sector_weights={str(q):w for q, w in weights.items()},
        non_singlet_weight=sum(w for q, w in weights.items() if q),
        exact_M0_pi_x_even_residual=even_residual,
        pi_x_even_bound_numerically_applicable=bool(even_residual < 1e-10),
        odd_integer_spin_weight=sum(w for q, w in weights.items() if q % 4 == 2),
        spin_squared_only_leakage_upper_bound=s2/2,
        even_spin_only_leakage_upper_bound_if_pi_x_even=s2/6,
        spin_squared_only_leakage_lower_bound=s2/(n*(n+2)/4),
        local_z_maximum=float(max(abs(pair['local_z']))),
        pair_maximum_abs_xx_minus_zz=float(np.max(abs(pair['xx']-pair['zz']))),
        source=dict(operator='S_0^z', norm=source_norm,
            normalized_sector_weights={str(q):w for q, w in source_weights.items()},
            non_q2_weight=sum(w for q, w in source_weights.items() if q != 2),
            strict_singlet_source_is_only_q2_m0=True,
            selection_rule_not_asserted_for_other_source_components=True),
        normalized_singlet_component_counterfactual_energy_per_site=singlet_energy/n,
        singlet_component_counterfactual=dict(
            label='normalized_P0psi_different_state_not_original_or_twirled',
            energy_per_site=singlet_energy/n,
            variance_per_site=float(np.linalg.norm(hamiltonian@p0-singlet_energy*p0)**2)/n,
            source_non_q2_weight=sum(float(np.vdot(v, v).real) for q, v in singlet_source.items() if q != 2),
            scalar_correlations=singlet_pairs['scalar'].real.tolist(),
            pair_maximum_abs_xx_minus_zz=float(np.max(abs(singlet_pairs['xx']-singlet_pairs['zz'])))),
        energy_sector_sum_replay_error=float(abs(cross_energy-energy)),
        sector_weight_sum_error=abs(sum(weights.values())-1),
        s2_sector_sum_replay_error=abs(sum(q*(q+2)/4*w for q, w in weights.items())-s2),
        maximum_projector_eigen_residual=max(float(np.linalg.norm(spin_squared@v-q*(q+2)/4*v))
                                             for q, v in components.items()),
        projector_reconstruction_error=float(np.linalg.norm(sum(components.values())-vector)),
        scalar_correlations=pair['scalar'].real.tolist(),
        anisotropy_matrix=(pair['xx']-pair['zz']).real.tolist())


def run(output: Path = OUTPUT) -> dict:
    """Replay only trusted saved amplitudes; fail rather than solve/retrain."""
    tick, cpu = perf_counter(), process_time()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('new isolated report directory required')
    if (file_sha256(SOURCE/'states.npz') != EXPECTED_BUNDLE_SHA
            or file_sha256(SOURCE/'results.json') != EXPECTED_RESULT_SHA):
        raise ValueError('archived saved-state provenance mismatch')
    old = json.loads((SOURCE/'results.json').read_text())
    if (old['config']['j2_over_j1'] != .125 or old.get('result_sha256') != EXPECTED_RESULT_CONTENT_SHA
            or canonical_sha256({k:v for k, v in old.items() if k != 'result_sha256'}) != EXPECTED_RESULT_CONTENT_SHA):
        raise ValueError('archived geometry/coupling identity differs')
    with np.load(SOURCE/'states.npz', allow_pickle=False) as archive:
        basis = archive['basis']; states = {k:archive[k].copy() for k in archive.files if k != 'basis'}
    bonds = triangular_hnf_j1j2_bonds(((4, 0), (0, 4)), j2=.125)
    h, hbasis = heisenberg_sz_graph_sparse(16, bonds, n_up=8)
    s2, sbasis = spin_squared_sz_sparse(16, n_up=8)
    positions = {tuple(row):i for i, row in enumerate(basis)}
    if set(positions) != set(hbasis) or hbasis != sbasis:
        raise ValueError('saved/independent exact computational basis mismatch')
    order = np.asarray([positions[row] for row in hbasis])
    basis = np.asarray(hbasis, dtype=np.int8)
    states = {k:v[order] for k, v in states.items()}
    result = {name:audit_state(state, basis, h, s2) for name, state in states.items()}
    exact = result['exact_ground']
    if (abs(exact['energy_total']-old['exact_ground']['energy_total']) > 2e-11
            or exact['variance_total'] > 1e-20 or exact['non_singlet_weight'] > 1e-18):
        raise ArithmeticError('archived exact ground energy/spin reference failed')
    for item in result.values():
        item['energy_error_per_site_vs_exact'] = item['energy_per_site']-exact['energy_per_site']
        item['scalar_pair_RMS_error_vs_exact'] = float(np.sqrt(np.mean((
            np.asarray(item['scalar_correlations'])-np.asarray(exact['scalar_correlations']))**2)))
        component = item['singlet_component_counterfactual']
        component['energy_error_per_site_vs_exact'] = component['energy_per_site']-exact['energy_per_site']
        component['scalar_pair_RMS_error_vs_exact'] = float(np.sqrt(np.mean((
            np.asarray(component['scalar_correlations'])-np.asarray(exact['scalar_correlations']))**2)))
    report = dict(task='actual_saved_N16_spin_contamination_20261101', status='complete',
        geometry='triangular_4x4_periodic', j2_over_j1=.125, n_sites=16,
        source_bundle=str(SOURCE.relative_to(ROOT)/'states.npz'),
        source_bundle_sha256=EXPECTED_BUNDLE_SHA, source_result_sha256=EXPECTED_RESULT_SHA,
        source_result_content_sha256=EXPECTED_RESULT_CONTENT_SHA,
        source_script_sha256=file_sha256(Path(__file__)), states=result,
        cpu_seconds=process_time()-cpu, wall_seconds=perf_counter()-tick,
        new_training_steps=0, new_ED_solves=0, GPU_queries=0,
        claim_boundary=dict(actual_archived_states_not_Roth_GCNN=True,
            no_current_CT6_quality_comparison=True, no_thermodynamic_QSL_claim=True,
            no_state_twirl_or_repair=True, normalized_singlet_component_is_different_state=True,
            symmetry_of_model_family_does_not_determine_actual_leakage=True,
            scalar_accuracy_and_spin_selection_are_separate_checks=True))
    output.mkdir(parents=True, exist_ok=True); atomic_json(output/'report.json', report)
    lines = ['# Actual saved N16 spin-contamination diagnostic', '',
        'Same archived triangular4x4, J2=.125, M=0 amplitudes; no training, ED, or twirl.', '',
        '| Archived state | non-S0 weight | <S²> | energy error/site | max abs(xx−zz) | S0z source non-S1 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, item in result.items():
        lines.append(f"| {name} | {item['non_singlet_weight']:.6g} | {item['spin_squared_expectation']:.6g} | "
            f"{item['energy_error_per_site_vs_exact']:.6g} | {item['pair_maximum_abs_xx_minus_zz']:.6g} | "
            f"{item['source']['non_q2_weight']:.6g} |")
    lines.extend(['', '## Exact singlet-component counterfactual (a different state)', '',
        '| Archived state | E/N before | E/N after P0 normalization | scalar-pair RMS error before | after | source non-S1 after |',
        '|---|---:|---:|---:|---:|---:|'])
    for name in ('optimized_pi_flux_jastrow_two_lanczos', 'pi_flux_fixed_sz_gamma_a1_spatial_vit'):
        item = result[name]; component = item['singlet_component_counterfactual']
        lines.append(f"| {name} | {item['energy_per_site']:.12g} | {component['energy_per_site']:.12g} | "
            f"{item['scalar_pair_RMS_error_vs_exact']:.9g} | {component['scalar_pair_RMS_error_vs_exact']:.9g} | "
            f"{component['source_non_q2_weight']:.6g} |")
    lines.extend(['', 'These are historical saved-state diagnostics, not a GCNN reproduction or an architecture win.',
        'Nonzero leakage does not logically invalidate every scalar observable; low energy does not certify sector purity.',
        'For exact even-spin/pi-x parity at M=0 the first leaked sector is S=2, so p(non-S0)<=<S²>/6, not /2.',
        'This archived ordinary Spatial-ViT is NOT pi-x even (residual0.0246): that tighter bound does not apply to it; it is not Roth GCNN.',
        'A strict singlet gives a pure S=1,M=0 source under S0z; actual source weights quantify the archived contamination.',
        'The normalized singlet component is reported only as a counterfactual different state, never as a repaired original.',
        f"CPU {report['cpu_seconds']:.3f}s; wall {report['wall_seconds']:.3f}s. Full precision/provenance: report.json."])
    # Generated report only; code/file edits are through apply_patch.
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    def expired(*_):
        raise TimeoutError('bounded local diagnostic exceeded300 wall seconds')
    signal.signal(signal.SIGALRM, expired); signal.alarm(300)
    report = run(args.output)
    print(json.dumps(dict(status=report['status'], cpu_seconds=report['cpu_seconds'],
                          wall_seconds=report['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
