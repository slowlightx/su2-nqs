import numpy as np
import pytest

from fnqs.exact import TFIMReference
from fnqs.hamiltonians import IsingCouplings
from fnqs.observables import sampled_correlations
from fnqs.sectors import lift_parity_state, parity_hamiltonian, parity_reference


def _pauli(n, sites, op):
    result = np.ones((1, 1))
    for i in range(n):
        result = np.kron(result, op if i in sites else np.eye(2))
    return result


def _independent_hamiltonian(n, g):
    x, z = np.array([[0, 1], [1, 0]]), np.diag([-1, 1])
    matrix = np.zeros((1 << n, 1 << n))
    for i in range(n):
        matrix -= g.j1 * _pauli(n, {i, (i + 1) % n}, z)
        matrix -= g.j2 * _pauli(n, {i, (i + 2) % n}, z)
        matrix -= g.jx * _pauli(n, {i, (i + 1) % n}, x)
        matrix -= g.h * _pauli(n, {i}, x)
    return matrix


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("parity", [-1, 1])
def test_sector_operator_orientation_and_lift_against_independent_pauli(n, parity):
    half = 1 << (n - 1)
    embedding = np.zeros((1 << n, half))
    embedding[np.arange(half), np.arange(half)] = 1 / np.sqrt(2)
    embedding[(1 << n) - 1 - np.arange(half), np.arange(half)] = parity / np.sqrt(2)
    g = IsingCouplings(j2=.2, jx=.1, h=.9)
    expected = embedding.T @ _independent_hamiltonian(n, g) @ embedding
    np.testing.assert_allclose(parity_hamiltonian(n, g, parity).toarray(), expected, atol=2e-14, rtol=0)
    state = np.random.default_rng(31).normal(size=half)
    lifted = lift_parity_state(state, n, parity)
    np.testing.assert_allclose(lifted, embedding @ state, atol=2e-14, rtol=0)
    np.testing.assert_allclose(lifted[::-1], parity * lifted, atol=2e-14, rtol=0)


@pytest.mark.parametrize("g", [IsingCouplings(h=.9), IsingCouplings(j2=.2, h=1.2),
                               IsingCouplings(jx=.2, h=.7)])
def test_sector_response_and_gaps_against_full_spectral_sum(g):
    n = 6; hamiltonian = _independent_hamiltonian(n, g)
    values, vectors = np.linalg.eigh(hamiltonian)
    x = np.array([[0, 1], [1, 0]])
    derivative = -sum(_pauli(n, {i}, x) for i in range(n))
    overlaps = vectors[:, 1:].T @ derivative @ vectors[:, 0]
    chi = np.sum(overlaps**2 / (values[1:] - values[0])**2)
    reference = parity_reference(n, g)
    np.testing.assert_allclose(reference.energy, values[0], atol=5e-11, rtol=0)
    np.testing.assert_allclose(reference.chi_total, chi, atol=5e-10, rtol=0)
    np.testing.assert_allclose(min(reference.same_parity_gap, reference.opposite_parity_gap),
                               values[1] - values[0], atol=5e-11, rtol=0)
    assert reference.state_residual < 5e-11
    assert reference.response_relative_residual < 1e-8
    assert reference.response_gauge_error < 1e-11


@pytest.mark.parametrize("n", [4, 8, 10, 14])
def test_large_sector_response_matches_independent_jordan_wigner(n):
    ref = parity_reference(n, IsingCouplings(h=1.))
    np.testing.assert_allclose(ref.energy, TFIMReference(n, 1.).energy(), atol=5e-10, rtol=0)
    np.testing.assert_allclose(ref.chi_total, n * (n - 1) / 32, atol=5e-9, rtol=0)


def test_unsigned_sampled_correlations_match_independent_pauli_diagonals():
    n = 4
    labels = np.arange(1 << n)
    bits = ((labels[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8)
    z = np.diag([-1, 1]); expected = []
    for r in range(n):
        if r == 0:
            expected.append(np.ones(1 << n))
        else:
            expected.append(np.diag(sum(_pauli(n, {i, (i + r) % n}, z) for i in range(n))) / n)
    records = np.stack(expected, axis=1)
    mean, se = sampled_correlations(bits)
    np.testing.assert_allclose(mean, records.mean(axis=0), atol=2e-14, rtol=0)
    np.testing.assert_allclose(se, records.std(axis=0, ddof=1) / np.sqrt(len(bits)), atol=2e-14, rtol=0)


def test_sector_boundary_validation():
    for n, p in [(16, 1), (4, 0), (4, True), (4, 1.)]:
        with pytest.raises(ValueError): parity_hamiltonian(n, IsingCouplings(), p)
    with pytest.raises(ValueError): parity_reference(4, IsingCouplings(h=0))
