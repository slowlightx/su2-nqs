"""Independent physical controls for SU(2) multipartite entanglement witnesses.

The controls distinguish entanglement of coarse spatial parties from genuine
entanglement of all individual spins.  No model training or sampled RDM is used.
"""

from itertools import product

import numpy as np
import pytest

from su2_nqs.benchmarks.su2_multipartite_witness import (
    canonical_bipartitions,
    partial_transpose,
    repair_fully_decomposable_witness,
    solve_su2_multipartite_witness,
)


def _partial_transpose_oracle(matrix, sites):
    """Explicit bit-index definition, independent of tensor-axis reshaping."""
    dimension = matrix.shape[0]
    n_sites = dimension.bit_length() - 1
    mask = sum(1 << (n_sites - 1 - site) for site in sites)
    result = np.empty_like(matrix)
    for row, column in product(range(dimension), repeat=2):
        exchanged = (row ^ column) & mask
        result[row, column] = matrix[row ^ exchanged, column ^ exchanged]
    return result


def _three_singlets():
    """Three independent singlets on physical pairs (0,1),(2,3),(4,5)."""
    singlet = np.array([0., 1., -1., 0.]) / np.sqrt(2)
    vector = np.kron(np.kron(singlet, singlet), singlet)
    return np.outer(vector, vector)


def _dimer_vector(pairs):
    """Construct six-site valence bonds directly from down/up bit strings."""
    vector = np.zeros(64)
    for configuration in range(64):
        bits = [(configuration >> (5-site)) & 1 for site in range(6)]
        amplitude = 1.
        for left, right in pairs:
            amplitude *= (0. if bits[left] == bits[right]
                          else (1. if bits[left] == 0 else -1.) / np.sqrt(2))
        vector[configuration] = amplitude
    return vector


def _global_spin(n_sites, direction):
    local = (
        np.array([[0., 1.], [1., 0.]]),
        np.array([[0., 1.j], [-1.j, 0.]]),
        np.diag([-1., 1.]),
    )[direction] / 2
    result = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for site in range(n_sites):
        result += np.kron(np.kron(np.eye(2**site), local),
                          np.eye(2**(n_sites - site - 1)))
    return result


def _check_dual_certificate(certificate, *, tolerance=2e-11):
    """Check the returned matrices directly, without trusting solver flags."""
    witness = certificate.witness
    identity = np.eye(len(witness))
    assert np.linalg.norm(witness - witness.conj().T) < tolerance
    for cut, q_operator in zip(certificate.cuts, certificate.q_operators):
        assert np.linalg.norm(q_operator - q_operator.conj().T) < tolerance
        assert np.linalg.eigvalsh(q_operator).min() >= -tolerance
        assert np.linalg.eigvalsh(identity - q_operator).min() >= -tolerance
        p_operator = witness - _partial_transpose_oracle(q_operator, cut)
        assert np.linalg.eigvalsh(p_operator).min() >= -tolerance


def test_partial_transpose_matches_independent_bit_indices():
    rng = np.random.default_rng(310271)
    raw = rng.normal(size=(8, 8)) + 1.j * rng.normal(size=(8, 8))
    matrix = raw + raw.conj().T
    for cut in ((0,), (1,), (2,), (0, 2)):
        expected = _partial_transpose_oracle(matrix, cut)
        np.testing.assert_allclose(partial_transpose(matrix, cut), expected,
                                   rtol=0., atol=0.)
        np.testing.assert_allclose(partial_transpose(expected, cut), matrix,
                                   rtol=0., atol=0.)
    # The public witness API deliberately accepts only nontrivial cuts.
    with pytest.raises(ValueError, match='proper subset'):
        partial_transpose(matrix, (0, 1, 2))


def test_independent_dimers_can_maximize_coarse_three_party_gmn():
    rho = _three_singlets()
    # Each coarse party owns the ends of two different singlets.  Across
    # any party cut the four nonzero Schmidt probabilities are all 1/4.
    parties = ((0, 5), (1, 2), (3, 4))
    cuts = canonical_bipartitions(parties, 6)
    assert len(cuts) == 3
    witness = np.eye(64) / 2 - 2 * rho
    q_operators = [_partial_transpose_oracle(witness, cut) for cut in cuts]
    for cut in cuts:
        eigenvalues = np.linalg.eigvalsh(_partial_transpose_oracle(rho, cut))
        assert -eigenvalues[eigenvalues < 0].sum() == pytest.approx(1.5, abs=2e-14)
    for direction in range(3):
        spin = _global_spin(6, direction)
        assert np.linalg.norm(spin @ rho) < 2e-14
        assert np.linalg.norm(spin @ witness - witness @ spin) < 2e-14
    certificate = repair_fully_decomposable_witness(
        witness, q_operators, cuts, safety_margin=1e-12)
    _check_dual_certificate(certificate)
    # Explicit feasible witness saturates the pure-state minimum-negativity
    # upper bound; this proves the exact coarse-party GMN without an SDP.
    assert -np.trace(rho @ certificate.witness).real == pytest.approx(1.5, abs=2e-10)


def test_same_dimers_have_zero_six_spin_gmn():
    rho = _three_singlets()
    cuts = canonical_bipartitions(tuple((i,) for i in range(6)), 6)
    assert len(cuts) == 31
    # A whole singlet pair is unentangled from the other two pairs.
    cut = (0, 1)
    assert np.linalg.eigvalsh(_partial_transpose_oracle(rho, cut)).min() > -2e-14
    witness = np.eye(64) / 2 - 2 * rho
    # Reusing the coarse-party witness for all six spins is invalid.  The
    # feasibility repair must eliminate its apparently positive lower bound.
    q_operators = [_partial_transpose_oracle(witness, part) for part in cuts]
    certificate = repair_fully_decomposable_witness(
        witness, q_operators, cuts, safety_margin=1e-12)
    _check_dual_certificate(certificate)
    assert -np.trace(rho @ certificate.witness).real <= 2e-11


def test_feasibility_repair_preserves_su2_and_checks_every_cut():
    rho = _three_singlets()
    cuts = canonical_bipartitions(((0, 5), (1, 2), (3, 4)), 6)
    witness = np.eye(64) / 2 - 2 * rho
    # Intentionally violate both ends of 0 <= Q <= I.  This is an audit of
    # the returned certificate, independent of a particular SDP solver.
    q_operators = [1.17 * _partial_transpose_oracle(witness, cut) - .031 * np.eye(64)
                   for cut in cuts]
    certificate = repair_fully_decomposable_witness(
        witness, q_operators, cuts, safety_margin=1e-10)
    _check_dual_certificate(certificate)
    for direction in range(3):
        spin = _global_spin(6, direction)
        assert np.linalg.norm(spin @ certificate.witness - certificate.witness @ spin) < 3e-12


def test_coherent_singlet_loop_has_genuine_six_spin_entanglement():
    # These independent product-dimer states are biseparable as six spins.
    # Their coherent sum has genuine six-spin entanglement; the incoherent
    # mixture does not.  This is a positive SU(2) control, not a QSL claim.
    first = _dimer_vector(((0, 1), (2, 3), (4, 5)))
    second = _dimer_vector(((1, 2), (3, 4), (5, 0)))
    psi = first + second
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi)
    incoherent = (np.outer(first, first) + np.outer(second, second)) / 2
    cuts = canonical_bipartitions(tuple((i,) for i in range(6)), 6)
    largest_schmidt_weights = []
    for cut in cuts:
        order = cut + tuple(i for i in range(6) if i not in cut)
        theta = psi.reshape((2,) * 6).transpose(order).reshape(2**len(cut), -1)
        largest_schmidt_weights.append(np.linalg.svd(theta, compute_uv=False)[0]**2)
    alpha = max(largest_schmidt_weights)
    assert alpha == pytest.approx(.5, abs=2e-14)
    # PT(rho) has largest eigenvalue <= alpha and smallest >= -alpha.
    # Therefore Q=PT(W) for W=(alpha I-rho)/(2alpha) lies in [0,I]
    # on ALL cuts, giving an explicit fully-decomposable witness with P=0.
    witness = (alpha*np.eye(64) - rho) / (2*alpha)
    q_operators = [_partial_transpose_oracle(witness, cut) for cut in cuts]
    certificate = repair_fully_decomposable_witness(
        witness, q_operators, cuts, safety_margin=1e-10)
    _check_dual_certificate(certificate)
    assert -np.trace(rho @ certificate.witness).real == pytest.approx(.5, abs=2e-8)
    assert np.trace(incoherent @ certificate.witness).real >= 0
    for direction in range(3):
        assert np.linalg.norm(_global_spin(6, direction) @ psi) < 2e-14


def test_su2_biseparable_mixture_is_npt_across_every_cut():
    # Equal mixture of singlet_AB x I_C/2 and its two permutations is
    # explicitly biseparable, yet every single-party negativity equals 1/6.
    singlet = np.array([0., 1., -1., 0.]) / np.sqrt(2)
    base = np.kron(np.outer(singlet, singlet), np.eye(2) / 2).reshape((2,) * 6)
    rho = np.zeros((8, 8))
    for pair in ((0, 1), (0, 2), (1, 2)):
        order = list(pair) + [next(i for i in range(3) if i not in pair)]
        axes = np.argsort(order)
        rho += base.transpose(list(axes) + list(axes + 3)).reshape(8, 8) / 3
    for direction in range(3):
        spin = _global_spin(3, direction)
        assert np.linalg.norm(spin @ rho - rho @ spin) < 2e-14
    for cut in ((0,), (1,), (2,)):
        eigenvalues = np.linalg.eigvalsh(_partial_transpose_oracle(rho, cut))
        assert -eigenvalues[eigenvalues < 0].sum() == pytest.approx(1/6, abs=2e-14)

    cvxpy = pytest.importorskip('cvxpy')
    if 'SCS' not in cvxpy.installed_solvers():
        pytest.skip('optional SDP control requires SCS')
    _, certificate = solve_su2_multipartite_witness(
        rho, ((0,), (1,), (2,)), solver='SCS',
        solver_options=dict(eps=2e-7, max_iters=10000), safety_margin=1e-9)
    _check_dual_certificate(certificate)
    assert -np.trace(rho @ certificate.witness).real <= 2e-10
