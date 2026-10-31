"""Independent bit/Pauli oracles: no training, new ED or symmetry repair."""
import itertools

import numpy as np
import pytest

from su2_nqs.benchmarks.singlet_physics_readout import (
    singlet_structure_factor_records, triangular_torus_structure_records,
    importance_jackknife_observables, scalar_chirality_matrix,
    compact6_channel_operators, compact6_channel_readout,
)


def _bits(n):
    return (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1


def _spin(n, site, component):
    small = (np.array([[0, .5], [.5, 0]]),
        np.array([[0, .5j], [-.5j, 0]]), np.diag([-.5, .5]))[component]
    result = np.ones((1, 1))
    for i in range(n):
        result = np.kron(result, small if i == site else np.eye(2))
    return result


def _dimer(n, pairs):
    bits = _bits(n)
    vector = np.ones(2**n)
    for i, j in pairs:
        vector *= (bits[:, i]-bits[:, j])/np.sqrt(2)
    return vector/np.linalg.norm(vector)


def test_diagonal_singlet_estimator_matches_all_three_pauli_components():
    n = 4
    bits = _bits(n)
    keep = bits.sum(axis=1) == 2
    psi = _dimer(4, ((0, 1), (2, 3)))
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    momenta = {'Gamma': (0, 0), 'M1': (.5, 0), 'M2': (0, .5), 'M3': (.5, .5)}
    records = singlet_structure_factor_records(bits[keep], coords, momenta, strict_singlet=True)
    for key, k in momenta.items():
        phase = np.exp(2j*np.pi*(coords@k))
        direct = 0.
        for component in range(3):
            op = sum(p*_spin(n, i, component) for i, p in enumerate(phase))
            direct += np.vdot(op@psi, op@psi).real/n
        assert abs(psi[keep])**2@records[key] == pytest.approx(direct, abs=2e-14, rel=0)
    with pytest.raises(ValueError):
        singlet_structure_factor_records(bits[keep], coords, momenta, strict_singlet=False)


def test_commensurate_k_and_parseval_sum_rule_are_not_sampling_certificates():
    rng = np.random.default_rng(2026101001)
    bits = np.zeros((20, 36), dtype=int)
    for row in bits:
        row[rng.choice(36, size=18, replace=False)] = 1
    data = triangular_torus_structure_records(bits, 6, 6, strict_singlet=True)
    assert np.max(abs(data['K']-data['minus_K'])) < 1e-14
    assert np.max(abs(data['Gamma'])) == 0
    coords = np.array([divmod(i, 6) for i in range(36)])
    momenta = {f'{i}_{j}': (i/6, j/6) for i in range(6) for j in range(6)}
    full = singlet_structure_factor_records(bits, coords, momenta, strict_singlet=True)
    np.testing.assert_allclose(sum(full.values()), np.full(20, 27.), atol=3e-14, rtol=0)
    with pytest.raises(ValueError):
        triangular_torus_structure_records(np.zeros((4, 16)), 4, 4, strict_singlet=True)


def test_importance_chain_jackknife_matches_independent_ratio_and_scaling():
    rng = np.random.default_rng(2026101002)
    q = rng.uniform(.2, 2, 24)
    psi = rng.normal(size=24)
    psi[2] = 0
    values = {'x': rng.normal(size=24)}
    groups = np.arange(24) % 4
    got = importance_jackknife_observables(values, psi, q, groups)
    weights = psi**2/q
    assert got['observables']['x']['value'] == pytest.approx(np.sum(weights*values['x'])/weights.sum(), abs=1e-15)
    expected = np.array([np.sum(weights[groups!=c]*values['x'][groups!=c])/weights[groups!=c].sum() for c in range(4)])
    np.testing.assert_allclose(got['observables']['x']['jackknife_values'], expected, atol=2e-15, rtol=0)
    other = importance_jackknife_observables(values, -7j*psi, 3*q, groups)
    assert other['observables']['x']['chain_SE'] == pytest.approx(got['observables']['x']['chain_SE'], abs=1e-15)


def test_oriented_chirality_matches_epsilon_pauli_oracle_and_scalar_commutator():
    n = 3
    chi = np.zeros((8, 8), complex)
    for permutation in itertools.permutations(range(3)):
        inversions = sum(permutation[i] > permutation[j] for i in range(3) for j in range(i+1, 3))
        chi += (-1)**inversions*(_spin(n, 0, permutation[0])@_spin(n, 1, permutation[1])@_spin(n, 2, permutation[2]))
    np.testing.assert_allclose(scalar_chirality_matrix(3, (0, 1, 2)), chi, atol=1e-15, rtol=0)
    np.testing.assert_allclose(scalar_chirality_matrix(3, (0, 2, 1)), -chi, atol=1e-15, rtol=0)
    for component in range(3):
        total = sum(_spin(n, i, component) for i in range(n))
        assert np.linalg.norm(total@chi-chi@total) < 1e-14


def test_local_e2_and_chirality_remain_scalar_and_read_coherent_density():
    bank = compact6_channel_operators()
    for component in range(3):
        total = sum(_spin(6, i, component) for i in range(6))
        assert max(np.linalg.norm(total@op-op@total) for op in bank.values()) < 3e-14
    v = _dimer(6, ((0, 3), (1, 4), (2, 5)))
    rho = np.outer(v, v)
    readout = compact6_channel_readout(rho)
    assert abs(readout['chi_triangle0']) < 1e-15
    assert abs(readout['chi_triangle1']) < 1e-15
    assert readout['E2_triangle0_connected_norm_squared'] >= -1e-14
    chi0 = bank['chi_triangle0']
    chi1 = bank['chi_triangle1']
    assert readout['chi_intertriangle_connected'] == pytest.approx(np.vdot(v, chi0@chi1@v).real, abs=1e-15)
    diagonal_only = np.diag(np.diag(rho))
    assert abs(compact6_channel_readout(diagonal_only)['bond_triangle0_direction0']-readout['bond_triangle0_direction0']) > 1e-4


def test_spin_half_local_e2_and_chirality_norms_are_only_two_body_information():
    bank = compact6_channel_operators()
    for side in range(2):
        bond_sum = sum(bank[f'bond_triangle{side}_direction{a}'] for a in range(3))
        np.testing.assert_allclose(bank[f'chi_triangle{side}_squared'],
            3*np.eye(64)/32-bond_sum/8, atol=1e-15, rtol=0)
        np.testing.assert_allclose(bank[f'E2_triangle{side}_norm_squared'],
            3*np.eye(64)/8-bond_sum/2, atol=1e-15, rtol=0)


def test_disjoint_chirality_is_signed_bridge_determinant_and_pure_six_body():
    from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
    left, right = (0, 3, 4), (1, 5, 2)
    bank = compact6_channel_operators()
    exact = bank['chi_intertriangle_product']
    determinant = np.zeros((64,64))
    unsigned = np.zeros((64,64))
    for permutation in itertools.permutations(range(3)):
        inversions = sum(permutation[i] > permutation[j] for i in range(3) for j in range(i+1,3))
        product = np.eye(64)
        for i in range(3):
            product = product@spin_dot_matrix(6,left[i],right[permutation[i]])
        determinant += (-1)**inversions*product
        unsigned += product
    np.testing.assert_allclose(exact, determinant, atol=1e-15, rtol=0)
    assert np.linalg.norm(exact-unsigned) > 1e-2
    # Every single-site partial trace vanishes: no Pauli weight <=5 component.
    tensor = exact.reshape((2,)*12)
    for site in range(6):
        assert np.linalg.norm(np.trace(tensor,axis1=site,axis2=6+site)) < 1e-15


def test_triangle_e2_chi_are_three_multiplicity_pseudospin_axes():
    bank = compact6_channel_operators()
    for side in range(2):
        axes = [bank[f'E2_triangle{side}_component{i}']/np.sqrt(3/8) for i in range(2)]
        axes += [bank[f'chi_triangle{side}']/(np.sqrt(3)/4)]
        support = axes[2]@axes[2]
        np.testing.assert_allclose(support@support,support,atol=2e-15,rtol=0)
        for axis in axes:
            np.testing.assert_allclose(axis@axis,support,atol=2e-15,rtol=0)
        for i,j in itertools.combinations(range(3),2):
            assert np.linalg.norm(axes[i]@axes[j]+axes[j]@axes[i]) < 2e-14


def test_saved_physics_analysis_seal_and_no_spectrum_repair():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_saved_state_physics_v1'
    if not directory.exists():
        pytest.skip('optional saved-state analysis artifact absent')
    registration = json.loads((directory/'registration.json').read_text())
    results = json.loads((directory/'results.json').read_text())
    completion = json.loads((directory/'completion.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path, digest in registration['inputs'].items():
        assert sha(root/path) == digest
    for path, digest in registration['sources'].items():
        source = root/path
        # The reusable workflow gained --pair-source after this immutable
        # analysis. Verify the EXACT original source snapshot, not a new
        # digest or a regenerated result. Mathematical dependencies still
        # require their registered live hashes; no generic mismatch bypass.
        if (path == 'projects/clebsch_tree_nqs/workflows/spin_saved_state_physics.py'
                and sha(source) != digest):
            source = directory.parent/'source_snapshots'/f'spin_saved_state_physics_{digest}.py'
        assert source.is_file() and sha(source) == digest
    assert sha(directory/'registration.json') == completion['registration_sha256']
    assert sha(directory/'results.json') == completion['results_sha256']
    assert all(results[name] == 0 for name in ('new_training','new_samples','new_ED','new_full_H_calls','new_amplitude_queries'))
    assert not results['phase_claim'] and not results['strict_multiplicity_spectrum_reported']
    audits = results['pair_quality_sequence']['audits']
    assert all(not d['strict_su2_eligible'] for audit in audits for region in audit['regions'].values()
        for d in region['diagnostics'].values())
    assert max(state['full_three_component_comparison_maximum_error']
        for state in results['exact_N16_calibrations'].values()) < 1e-12
