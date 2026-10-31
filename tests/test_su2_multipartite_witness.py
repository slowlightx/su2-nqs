"""Independent bounded witness oracles; no neural optimization involved."""
import numpy as np
import pytest

from su2_nqs.benchmarks.su2_multipartite_witness import (
    partial_transpose, canonical_bipartitions, su2_invariant_witness_basis,
    repair_fully_decomposable_witness, evaluate_frozen_witness,
    frozen_witness_chain_audit, solve_su2_multipartite_witness,
    WitnessCertificate, validate_witness_certificate,
)


def _pt_oracle(matrix, sites):
    d = len(matrix); n = d.bit_length()-1
    result = np.empty_like(matrix)
    mask = sum(1 << (n-1-site) for site in sites)
    for a in range(d):
        for b in range(d):
            ra = (a & ~mask) | (b & mask)
            rb = (b & ~mask) | (a & mask)
            result[a, b] = matrix[ra, rb]
    return result


def _total_spin(n, axis):
    spins = (np.array([[0, 1], [1, 0]])/2,
             np.array([[0, 1j], [-1j, 0]])/2,
             np.diag([-.5, .5]))
    result = np.zeros((2**n, 2**n), complex)
    for site in range(n):
        term = np.ones((1, 1), complex)
        for k in range(n):
            term = np.kron(term, spins[axis] if k == site else np.eye(2))
        result += term
    return result


def test_physical_partial_transpose_and_party_cut_conventions():
    rng = np.random.default_rng(2026101001)
    a = rng.normal(size=(16, 16))+1j*rng.normal(size=(16, 16))
    a = a+a.conj().T
    for cut in [(0,), (1, 3), (0, 1, 2)]:
        actual = partial_transpose(a, cut)
        np.testing.assert_array_equal(actual, _pt_oracle(a, cut))
        np.testing.assert_array_equal(partial_transpose(actual, cut), a)
        assert np.trace(actual) == pytest.approx(np.trace(a))
    cuts = canonical_bipartitions(((0, 5), (1, 2), (3, 4)), 6)
    assert cuts == ((0, 5), (0, 1, 2, 5), (0, 3, 4, 5))
    assert len(canonical_bipartitions(tuple((i,) for i in range(6)), 6)) == 31
    for parties in [((0,), (1, 2)), ((0, 1), (1, 2, 3)), ((0,),)]:
        with pytest.raises(ValueError):
            canonical_bipartitions(parties, 4)
    with pytest.raises(ValueError): partial_transpose(a, (0, 0))
    with pytest.raises(ValueError): partial_transpose(a, (True,))


@pytest.mark.parametrize('n, real_dim, full_dim', [(2, 2, 2), (4, 10, 14), (6, 76, 132)])
def test_invariant_witness_basis_has_correct_dimension_and_all_spin_commutators(n, real_dim, full_dim):
    for real, dimension in [(True, real_dim), (False, full_dim)]:
        basis = su2_invariant_witness_basis(n, real)
        assert len(basis) == dimension
        flat = basis.reshape(dimension, -1)
        np.testing.assert_allclose(flat.conj()@flat.T, np.eye(dimension), atol=8e-14, rtol=0)
        np.testing.assert_allclose(basis, basis.conj().swapaxes(1, 2), atol=1e-15, rtol=0)
        if real: assert np.max(abs(basis.imag)) == 0
        for axis in range(3):
            s = _total_spin(n, axis)
            assert np.max(abs(basis@s-s@basis)) < 2e-14


def test_repaired_two_spin_singlet_witness_has_explicit_decomposition_and_cost():
    singlet = np.array([0., 1., -1., 0.])/np.sqrt(2)
    phi = np.array([1., 0., 0., 1.])/np.sqrt(2)
    rho = np.outer(singlet, singlet)
    q = np.outer(phi, phi)
    w = .5*np.eye(4)-rho
    np.testing.assert_allclose(_pt_oracle(q, (0,)), w, atol=3e-16, rtol=0)
    certificate = repair_fully_decomposable_witness(w, [q], [(0,)])
    result = evaluate_frozen_witness(certificate, rho)
    assert result['entanglement_lower_bound'] == pytest.approx(.5, abs=2e-8)
    delta = certificate.diagnostics['identity_shift']
    np.testing.assert_allclose(certificate.witness-w, delta*np.eye(4), atol=1e-16, rtol=0)
    assert result['expectation'] == pytest.approx(np.trace(rho@w).real+delta, abs=2e-16)
    for cut, safe_q in zip(certificate.cuts, certificate.q_operators):
        p = certificate.witness-_pt_oracle(safe_q, cut)
        assert np.linalg.eigvalsh(p).min() > 0
        assert np.linalg.eigvalsh(safe_q).min() > 0
        assert np.linalg.eigvalsh(safe_q).max() < 1
    assert not certificate.diagnostics['P_upper_bound_imposed']
    assert not certificate.diagnostics['formal_interval_certificate']


def test_repair_does_not_impose_P_upper_bound_and_handles_bad_auxiliary():
    w = 3*np.eye(4)
    certificate = repair_fully_decomposable_witness(w, [.2*np.eye(4)], [(0,)])
    p = certificate.witness-partial_transpose(certificate.q_operators[0], (0,))
    assert np.linalg.eigvalsh(p).min() > 2
    assert certificate.diagnostics['identity_shift'] == 0
    broken_q = np.diag([-.1, .2, .7, 1.2])
    certificate = repair_fully_decomposable_witness(-.2*np.eye(4), [broken_q], [(0,)])
    assert certificate.diagnostics['identity_shift'] > .2
    assert certificate.diagnostics['cuts'][0]['original_minimum_eigenvalue'] == -.1
    assert certificate.diagnostics['cuts'][0]['original_maximum_eigenvalue'] == 1.2
    with pytest.raises(ValueError): repair_fully_decomposable_witness(w, [w], [(0,)], safety_margin=0)
    with pytest.raises(ValueError): evaluate_frozen_witness(certificate, np.eye(4))


def test_frozen_whole_chain_linear_audit_retains_signed_negative_estimate():
    certificate = repair_fully_decomposable_witness(.2*np.eye(4), [np.zeros((4, 4))], [(0,)])
    rows = np.stack([np.eye(4)/4 for _ in range(12)])
    result = frozen_witness_chain_audit(certificate, rows, np.repeat(np.arange(4), 3))
    assert result['expectation'] == pytest.approx(.2)
    assert result['signed_entanglement_lower_estimate'] == pytest.approx(-.2)
    assert result['two_SE_lower_bound'] == 0
    assert result['whole_chain_SE'] == 0
    assert result['negative_estimates_retained']
    with pytest.raises(ValueError): frozen_witness_chain_audit(certificate, rows, np.repeat(np.arange(3), 4)[:11])


def test_frozen_validation_rejects_forged_flags_and_incomplete_party_cuts():
    fake = WitnessCertificate(-np.eye(4), (np.zeros((4, 4)),), ((0,),),
                              {'feasible_to_numerical_margin': True})
    with pytest.raises(ValueError, match='feasible decomposition'):
        evaluate_frozen_witness(fake, np.eye(4)/4)
    # Two of the three bipartitions are insufficient for genuine tripartite
    # entanglement, even when each submitted decomposition is positive.
    with pytest.raises(ValueError, match='complete nonduplicated'):
        repair_fully_decomposable_witness(np.eye(8), [np.zeros((8, 8))]*2,
                                          [(0,), (1,)])
    good = repair_fully_decomposable_witness(np.eye(8), [np.zeros((8, 8))]*3,
                                            [(0,), (1,), (2,)])
    audit = validate_witness_certificate(good)
    assert audit['inferred_parties'] == [[0], [1], [2]]


def test_optional_SDP_singlet_and_product_oracles():
    cp = pytest.importorskip('cvxpy')
    if 'SCS' not in cp.installed_solvers(): pytest.skip('optional SCS SDP solver absent')
    singlet = np.array([0., 1., -1., 0.])/np.sqrt(2)
    options = dict(eps=1e-7, max_iters=5000, time_limit_secs=20.)
    result, certificate = solve_su2_multipartite_witness(
        np.outer(singlet, singlet), ((0,), (1,)), solver_options=options)
    assert result['entanglement_lower_bound'] == pytest.approx(.5, abs=3e-6)
    assert result['invariant_parameters'] == 2
    assert not result['physical_state_twirl_performed']
    assert certificate.diagnostics['feasible_to_numerical_margin']
    product = np.zeros(4); product[0] = 1
    result, _ = solve_su2_multipartite_witness(np.outer(product, product), ((0,), (1,)),
                                             time_reversal_even=True, solver_options=options)
    assert result['entanglement_lower_bound'] < 1e-8
