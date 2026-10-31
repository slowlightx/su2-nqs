"""Fixed-region SU(2) readouts from sealed physical states, with no eigensolve.

Only small reduced density matrices are diagonalized. No many-body Hamiltonian,
ED, new neural amplitudes, or Monte Carlo samples enter this workflow.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import signal
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.local_multiplicity_observables import local_multiplicity_density
from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix, subsystem_total_spin_squared
from su2_nqs.frontier.artifacts import array_sha256, atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.su2.local_schur import cell_schur_tensor
from su2_nqs.su2.sz_basis import enumerate_sz_configs

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'projects/clebsch_tree_nqs/configs/validation/spin_symmetry_readout.json'


def schur_columns() -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    """Complete four-spin columns, labelled by doubled q,m and multiplicity a."""
    table = cell_schur_tensor('square_plaquette')
    labels = tuple((q, m, a) for q, d in ((0, 2), (2, 3), (4, 1))
                   for m in range(-q, q + 1, 2) for a in range(d))
    return np.column_stack([table[:, q, m + 4, a] for q, m, a in labels]), labels


def _entropy(values: np.ndarray) -> float:
    positive = np.asarray(values)[np.asarray(values) > 1e-14]
    return float(-np.sum(positive * np.log(positive)))


def _matrix_json(values: np.ndarray) -> list:
    return np.stack((values.real, values.imag), axis=-1).tolist()


def density_readout(rho: np.ndarray, *, tolerance: float = 1e-9) -> tuple[dict, dict]:
    """Read an *unprojected* physical rho and reject non-SU(2)-invariance.

    R_q is the magnetic trace, tr R_q=p_q; sigma_q=R_q/p_q. All spectra
    include zeros. Computing a magnetic trace never licenses replacing a
    noninvariant rho by its twirl: reconstruction and S+ are checked first.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (16, 16) or not np.isfinite(rho).all():
        raise ValueError('readout requires a finite four-spin density matrix')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive')
    hermitian = float(np.linalg.norm(rho - rho.conj().T))
    values = np.linalg.eigvalsh(rho)
    raising = np.zeros((16, 16), dtype=np.complex128)
    for packed in range(16):
        for site in range(4):
            mask = 1 << site
            if not packed & mask:
                raising[packed | mask, packed] += 1
    s2 = subsystem_total_spin_squared(4)
    u, labels = schur_columns()
    transformed = u.conj().T @ rho @ u
    reconstructed = np.zeros_like(transformed)
    q_diagonal = np.zeros_like(transformed)
    sectors, arrays = {}, {'rho': rho, 'schur_rho': transformed}
    spin_entropy = magnetic_entropy = multiplicity_entropy = 0.
    for q, d in ((0, 2), (2, 3), (4, 1)):
        magnetic_indices = [[labels.index((q, m, a)) for a in range(d)] for m in range(-q, q + 1, 2)]
        indices = [i for block in magnetic_indices for i in block]
        q_diagonal[np.ix_(indices, indices)] = transformed[np.ix_(indices, indices)]
        reduced = sum(transformed[np.ix_(i, i)] for i in magnetic_indices)
        p = float(np.trace(reduced).real)
        conditional = reduced / p if p > 1e-14 else np.zeros_like(reduced)
        conditional_eigenvalues = np.linalg.eigvalsh(conditional)[::-1]
        reduced_eigenvalues = np.linalg.eigvalsh(reduced)[::-1]
        for i in magnetic_indices:
            reconstructed[np.ix_(i, i)] = reduced / (q + 1)
        sector_spectrum = np.linalg.eigvalsh(transformed[np.ix_(indices, indices)])[::-1]
        sectors[str(q)] = dict(q=q, p=p, multiplicity=d,
            rho_multiplicity=_matrix_json(reduced), conditional_matrix=_matrix_json(conditional),
            unnormalized_multiplicity_spectrum=reduced_eigenvalues.tolist(),
            conditional_multiplicity_spectrum=conditional_eigenvalues.tolist(),
            magnetic_repeated_physical_spectrum=sector_spectrum.tolist(),
            conditional_entropy=_entropy(conditional_eigenvalues))
        arrays[f'mult_q{q}'] = reduced
        if p > 1e-14:
            spin_entropy -= p * np.log(p)
            magnetic_entropy += p * np.log(q + 1)
            multiplicity_entropy += p * _entropy(conditional_eigenvalues)
    diagnostics = dict(trace_error=float(abs(np.trace(rho) - 1)), hermiticity_frobenius=hermitian,
        minimum_eigenvalue=float(values.min()), schur_unitarity_frobenius=float(np.linalg.norm(u.conj().T @ u - np.eye(16))),
        raising_commutator_frobenius=float(np.linalg.norm(rho @ raising - raising @ rho)),
        spin_squared_commutator_frobenius=float(np.linalg.norm(rho @ s2 - s2 @ rho)),
        off_q_block_frobenius=float(np.linalg.norm(transformed - q_diagonal)),
        magnetic_factorization_frobenius=float(np.linalg.norm(transformed - reconstructed)),
        physical_reconstruction_frobenius=float(np.linalg.norm(rho - u @ reconstructed @ u.conj().T)))
    total_entropy = _entropy(values)
    diagnostics['entropy_sum_error'] = abs(total_entropy - spin_entropy - magnetic_entropy - multiplicity_entropy)
    if (values.min() < -tolerance or max(v for k, v in diagnostics.items() if k != 'minimum_eigenvalue') > tolerance):
        raise ValueError('input rho fails positivity/trace/SU2/reconstruction: no projection repair allowed')
    return dict(sectors=sectors, entropy_nats=total_entropy, spin_weight_entropy=float(spin_entropy),
                magnetic_entropy=float(magnetic_entropy), multiplicity_entropy=float(multiplicity_entropy),
                physical_spectrum=values[::-1].tolist(), diagnostics=diagnostics), arrays


def dimer_state(basis: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Physical product of oriented singlets; bit 0=down, bit 1=up."""
    n = basis.shape[1]
    if sorted(site for pair in pairs for site in pair) != list(range(n)):
        raise ValueError('pairs must cover every physical site exactly once')
    values = np.ones(len(basis))
    for i, j in pairs:
        values *= np.where(basis[:, i] != basis[:, j], np.where(basis[:, i] == 0, 1., -1.) / np.sqrt(2), 0.)
    return values


def multiplicity_bell_state(basis: np.ndarray) -> np.ndarray:
    """Two four-spin singlet multiplicities entangled, with remaining dimers."""
    table = cell_schur_tensor('square_plaquette')[:, 0, 4, :2]
    a, b = (0, 1, 4, 5), (2, 3, 6, 7)
    packed_a, packed_b = basis[:, a] @ [8, 4, 2, 1], basis[:, b] @ [8, 4, 2, 1]
    values = np.sum(table[packed_a] * table[packed_b], axis=1) / np.sqrt(2)
    for i, j in ((8, 9), (10, 11), (12, 13), (14, 15)):
        values *= np.where(basis[:, i] != basis[:, j], np.where(basis[:, i] == 0, 1., -1.) / np.sqrt(2), 0.)
    return values


def _sealed_result(path: Path) -> dict:
    result = json.loads(path.read_text())
    unsigned = dict(result)
    sha = unsigned.pop('result_sha256')
    if canonical_sha256(unsigned) != sha:
        raise ValueError('source result canonical seal changed')
    return result


def load_states(spec: dict) -> tuple[np.ndarray, dict, dict]:
    """Load physical reference vectors; never invoke a Hamiltonian eigensolver."""
    for path, sha in spec['source_files'].items():
        if file_sha256(ROOT / path) != sha:
            raise ValueError('source hash changed: ' + path)
    basis = np.asarray(enumerate_sz_configs(16, 8), dtype=np.int8)
    base = ROOT / 'projects/clebsch_tree_nqs/results'
    states, metadata = {}, {}
    source = base / 'spin_exchange_environment_v1'
    result = _sealed_result(source / 'results.json')
    with np.load(source / 'states.npz') as z, np.load(base / 'spin_carrier_cell_scalar_circuit_v1/states.npz') as old:
        phi = old['physical_a1_basis']
        if phi.shape != (12870, 14) or np.linalg.norm(phi.conj().T @ phi - np.eye(14)) > 1e-10:
            raise ValueError('sealed physical A1 basis mismatch')
        for coupling in (0., .125, .175):
            name = f'triangular_j{coupling:g}'
            states[name] = phi @ z[f'j{coupling}_exact']
            metadata[name] = dict(geometry='triangular_4x4_PBC', coupling=coupling,
                kind='saved exact Gamma-A1 singlet reference, reduced dimension 14',
                source=str(source.relative_to(ROOT) / 'states.npz'), key=f'j{coupling}_exact',
                energy_per_site=result['cases'][str(coupling)]['exact_energy_per_site'],
                warning='4x4 does not contain K; no phase identification')
    cached = ROOT / 'projects/clebsch_tree_nqs/references/exact_cache/22bc977a758e8bae385c06d088a2a2f64852c4ed98fc00ca81402c0a57150ec8.npz'
    manifest = json.loads(cached.with_suffix('.manifest.json').read_text())
    if manifest['identity']['basis_sha256'] != array_sha256(basis) or manifest['payload']['sha256'] != file_sha256(cached):
        raise ValueError('cached exact state basis/payload provenance mismatch')
    with np.load(cached) as z:
        states['triangular_j0.075'] = z['state']
        metadata['triangular_j0.075'] = dict(geometry='triangular_4x4_PBC', coupling=.075,
            kind='saved full fixed-M0 exact reference', source=str(cached.relative_to(ROOT)), key='state',
            energy_per_site=float(z['energy']) / 16, saved_H_residual_norm=float(z['residual_norm']))
    for seed in spec['square_seeds']:
        source = base / f'spin_full_vit_vap_v1/square_d16_l2_s{seed}_vap'
        result = _sealed_result(source / 'results.json')
        name = f'square_j0.5_vap_seed{seed}'
        with np.load(source / 'states.npz') as z:
            if not np.array_equal(z['basis'], basis):
                raise ValueError('saved physical basis ordering mismatch')
            states[name] = z['projected_terminal']
        metadata[name] = dict(geometry='square_4x4_PBC', coupling=.5, kind='saved variational VAP state; NOT exact',
            source=str(source.relative_to(ROOT) / 'states.npz'), key='projected_terminal',
            energy_per_site=result['final']['energy_per_site'], saved_variance_per_site=result['final']['variance_per_site'])
    internal = tuple((r * 4 + c, r * 4 + c + 1) for r in range(4) for c in (0, 2))
    crossed = tuple((r * 4 + c, r * 4 + (c + 1) % 4) for r in range(4) for c in (1, 3))
    states['dimer_internal'] = dimer_state(basis, internal)
    states['dimer_four_crossing'] = dimer_state(basis, crossed)
    states['multiplicity_bell_q0'] = multiplicity_bell_state(basis)
    metadata.update(dimer_internal=dict(kind='analytic product of singlets; zero cut-crossing pairs', pairs=internal),
        dimer_four_crossing=dict(kind='analytic product of singlets; four cut-crossing pairs', pairs=crossed),
        multiplicity_bell_q0=dict(kind='analytic Bell pair of local singlet multiplicities; p(q=0)=1, S_mult=ln2'))
    return basis, states, metadata


def run(config: Path = CONFIG) -> dict:
    """Analyze only saved states under a fixed 120-second numerical cap."""
    spec = json.loads(config.read_text())
    if spec['task'] != 'spin_symmetry_readout' or spec['region'] != [0, 1, 4, 5] or spec['n_sites'] != 16:
        raise ValueError('fixed physical region protocol changed')
    if any(spec[k] for k in ('new_ed', 'new_samples', 'new_hamiltonian_actions')) or not 0 < spec['wall_seconds_cap'] <= 120:
        raise ValueError('saved-state readout cannot authorize new physics calculations')
    output = ROOT / spec['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('readout output already exists')
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    def timeout(*_):
        raise TimeoutError('fixed readout budget exhausted')
    prior_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(int(spec['wall_seconds_cap']))
    try:
        basis, states, metadata = load_states(spec)
        reports, saved, raw = {}, {'basis': basis}, {}
        for name, state in states.items():
            state = np.asarray(state, dtype=np.complex128)
            norm = float(np.vdot(state, state).real)
            state = state / np.sqrt(norm)
            rho = reduced_density_matrix(state, basis, spec['region'])
            report, arrays = density_readout(rho, tolerance=spec['tolerance'])
            local = local_multiplicity_density(state, basis, cell=spec['region'], kind='square_plaquette')
            report['diagnostics']['existing_local_multiplicity_replay_max'] = max(float(np.max(abs(arrays[f'mult_q{block.q}'] - block.matrix))) for block in local.blocks)
            report['diagnostics']['global_spin_raising_relative_norm'] = float(spin_raising_relative_norm(state, basis))
            report['source_norm_squared'] = norm
            report['provenance'] = metadata[name]
            if report['diagnostics']['global_spin_raising_relative_norm'] > spec['tolerance']:
                raise ValueError('source is not a singlet')
            reports[name], raw[name] = report, arrays
            saved[name + '_state'] = state
            saved.update({name + '_' + key: value for key, value in arrays.items()})
        comparisons = {}
        for left, right in combinations(states, 2):
            a, b = raw[left], raw[right]
            sector_distance = {str(q): .5 * float(np.sum(abs(np.linalg.eigvalsh(a[f'mult_q{q}'] - b[f'mult_q{q}'])))) for q in (0, 2, 4)}
            comparisons[left + '-versus-' + right] = dict(
                physical_trace_distance=.5 * float(np.sum(abs(np.linalg.eigvalsh(a['rho'] - b['rho'])))),
                p_q_total_variation=.5 * sum(abs(reports[left]['sectors'][str(q)]['p'] - reports[right]['sectors'][str(q)]['p']) for q in (0, 2, 4)),
                unnormalized_multiplicity_trace_distances=sector_distance,
                sum_multiplicity_trace_distance=sum(sector_distance.values()),
                conditional_spectrum_L1={str(q): float(np.sum(abs(np.asarray(reports[left]['sectors'][str(q)]['conditional_multiplicity_spectrum']) - reports[right]['sectors'][str(q)]['conditional_multiplicity_spectrum']))) for q in (0, 2, 4)})
        temporary = output / 'readout.tmp'
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **saved)
        temporary.replace(output / 'readout.npz')
        result = dict(task=spec['task'], status='complete', config=spec, states=reports, comparisons=comparisons,
            sources=transitive_local_source_manifest((Path(__file__),), root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')),
            input_sha256=spec['source_files'], arrays_sha256=file_sha256(output / 'readout.npz'),
            seconds=perf_counter() - started, new_ED=0, new_H_actions=0, new_samples=0,
            caveat='Coupling-dependent finite-region data are not a QSL or phase-boundary diagnosis. Dimer/Bell controls are singlets but need not be A1. Square VAP vectors are not exact references.')
        result['result_sha256'] = canonical_sha256(result)
        atomic_json(output / 'results.json', result)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    print(json.dumps({k: v for k, v in run(parser.parse_args().config).items() if k in ('status', 'seconds', 'new_ED', 'new_H_actions', 'new_samples')}))
