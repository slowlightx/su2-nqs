"""Independent small-system validation of exact postprocessing, no data needed."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.spin_contamination_actual_n16_20261101 import (
    audit_state, pair_readout, sector_components,
)
from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix, spin_squared_matrix


def test_all_doubled_spin_polynomials_match_dense_spectral_projectors():
    s2, packed = spin_squared_matrix(6, n_up=3)
    eigenvalues, eigenvectors = np.linalg.eigh(s2)
    rng = np.random.default_rng(20261101)
    state = rng.normal(size=len(packed))+1j*rng.normal(size=len(packed)); state /= np.linalg.norm(state)
    actual = sector_components(state, s2, n_sites=6)
    for q, component in actual.items():
        columns = eigenvectors[:, abs(eigenvalues-q*(q+2)/4) < 1e-10]
        np.testing.assert_allclose(component, columns@(columns.conj().T@state), atol=3e-13, rtol=0)
    np.testing.assert_allclose(sum(actual.values()), state, atol=3e-13, rtol=0)


def test_pair_anisotropy_matches_independent_full_Pauli_operators():
    n = 4; _, packed = spin_squared_matrix(n, n_up=2)
    basis = np.asarray([[(k >> (n-1-site)) & 1 for site in range(n)] for k in packed], dtype=np.int8)
    rng = np.random.default_rng(91)
    vector = rng.normal(size=len(packed))+1j*rng.normal(size=len(packed)); vector /= np.linalg.norm(vector)
    full = np.zeros(2**n, dtype=complex); full[np.asarray(packed)] = vector
    sx = np.array([[0, 1], [1, 0]])/2
    sz = np.diag([-.5, .5])
    measured = pair_readout(vector, basis)
    for i in range(n):
        for j in range(i+1, n):
            for name, pauli in (('xx', sx), ('zz', sz)):
                operator = np.array([[1.]])
                for site in range(n):
                    operator = np.kron(operator, pauli if site in (i, j) else np.eye(2))
                np.testing.assert_allclose(measured[name][i, j], np.vdot(full, operator@full), atol=2e-14)


def test_single_spin_source_of_singlet_is_only_q2_and_even_source_parity():
    s2, packed = spin_squared_matrix(4, n_up=2)
    values, columns = np.linalg.eigh(s2)
    g = columns[:, abs(values) < 1e-12][:, 0]
    bits = np.asarray([[(k >> (3-site)) & 1 for site in range(4)] for k in packed], np.int8)
    source = (bits[:, 0]-.5)*g
    weights = {q:np.linalg.norm(v)**2 for q, v in sector_components(source, s2, n_sites=4).items()}
    np.testing.assert_allclose(weights[2], 1., atol=2e-14)
    assert weights[0]+weights[4] < 1e-27
    with pytest.raises(ValueError): sector_components(g, s2, n_sites=3)


def test_even_spin_contamination_and_singlet_component_are_distinct_states():
    s2, packed = spin_squared_matrix(4, n_up=2)
    h, hbasis = heisenberg_sz_matrix(4, n_up=2)
    assert packed == hbasis
    energies, eigenvectors = np.linalg.eigh(h)
    ground = eigenvectors[:, 0]
    highest_spin = np.ones(len(packed))/np.sqrt(len(packed))
    mixed = np.sqrt(.99)*ground+np.sqrt(.01)*highest_spin
    basis = np.asarray([[(k >> (3-site)) & 1 for site in range(4)] for k in packed], np.int8)
    result = audit_state(mixed, basis, h, s2)
    np.testing.assert_allclose(result['non_singlet_weight'], .01, atol=2e-14)
    np.testing.assert_allclose(result['spin_squared_expectation'], .06, atol=2e-14)
    assert result['pi_x_even_bound_numerically_applicable']
    component = result['singlet_component_counterfactual']
    np.testing.assert_allclose(component['energy_per_site'], energies[0]/4, atol=2e-14)
    assert component['variance_per_site'] < 1e-27
    assert result['energy_per_site'] > component['energy_per_site']
