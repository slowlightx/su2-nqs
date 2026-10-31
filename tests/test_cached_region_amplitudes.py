"""Exhaustive complex fixed-M oracle independent of cached-region estimators."""
import numpy as np
import pytest

from su2_nqs.benchmarks.cached_region_amplitudes import (
    cached_region_readout, reconstruct_region_queries, region_completion_rows,
)


def test_complex_exhaustive_RDM_and_operator_expectations_with_exact_nodes():
    all_bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    rows = all_bits[all_bits.sum(1) == 3]
    rng = np.random.default_rng(28102841)
    state = rng.normal(size=20) + 1j*rng.normal(size=20); state[2] = 0
    state /= np.linalg.norm(state)
    full = np.zeros(64, complex); packed = rows @ (1 << np.arange(5, -1, -1)); full[packed] = state
    sites = (4, 0, 2)
    query, ids = region_completion_rows(rows, sites)
    query_values = full[query @ (1 << np.arange(5, -1, -1))]
    phi = reconstruct_region_queries(query_values, ids)
    hermitian = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8)); hermitian += hermitian.conj().T
    outside = tuple(i for i in range(6) if i not in sites)
    matrix = np.transpose(full.reshape((2,)*6), sites+outside).reshape(8, 8)
    expected = matrix @ matrix.conj().T
    report, actual = cached_region_readout(rows, sites, phi, np.ones(20)/20,
        dict(identity=np.eye(8), observable=hermitian), groups=4)
    np.testing.assert_allclose(actual, expected, atol=6e-16, rtol=0)
    assert report['exact_target_nodes'] == 1
    assert report['observables']['identity']['real'] == pytest.approx(1., abs=3e-16)
    assert report['observables']['observable']['real'] == pytest.approx(np.trace(expected @ hermitian).real, abs=2e-15)
    assert abs(report['observables']['observable']['imaginary']) < 1e-15
    # An arbitrary complex operator also checks row/column orientation.
    complex_op = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8))
    report, _ = cached_region_readout(rows, sites, phi, np.ones(20)/20, dict(o=complex_op), groups=4)
    value = report['observables']['o']
    np.testing.assert_allclose(value['real']+1j*value['imaginary'], np.trace(expected @ complex_op), atol=2e-15)


def test_missing_M_sectors_are_structural_zeros_and_sampling_is_not_frequencies():
    rows = np.array([[0, 0, 1, 1], [1, 1, 0, 0]] * 4)
    unique, ids = region_completion_rows(rows, (0, 1))
    assert np.all(unique.sum(1) == 2)
    assert np.all(ids[:1, 1:4] == -1)
    phi = reconstruct_region_queries(np.ones(len(unique)), ids)
    assert np.all(phi[ids == -1] == 0)
    with pytest.raises(ValueError): region_completion_rows(rows, (0, 0))
    with pytest.raises(ValueError): region_completion_rows(np.array([[0, 0], [1, 1]]), (0,))
    with pytest.raises(ValueError): cached_region_readout(rows, (0, 1), phi, np.zeros(8), dict(i=np.eye(4)), groups=4)


def test_zero_delete_group_norm_is_rejected_not_reported_as_precision():
    rows = np.tile([0, 1], (8, 1))
    phi = np.zeros((8, 2)); phi[:2, 0] = 1.
    with pytest.raises(FloatingPointError, match='delete-group target norm'):
        cached_region_readout(rows, (0,), phi, np.ones(8), dict(i=np.eye(2)), groups=4)


def test_nonuniform_importance_quadrature_is_invariant_under_complex_scale():
    # Deterministic quadrature oracle, not a claim that these repeats are IID.
    all_bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    support = all_bits[all_bits.sum(1) == 3]
    counts = np.tile(np.arange(1, 6), 4)
    rows = np.repeat(support, counts, axis=0)
    proposal = np.repeat(counts/float(counts.sum()), counts)
    rng = np.random.default_rng(28102842)
    state = rng.normal(size=20) + 1j*rng.normal(size=20)
    state[[2, 11]] = 0
    state /= np.linalg.norm(state)
    full = np.zeros(64, complex)
    full[support @ (1 << np.arange(5, -1, -1))] = state
    sites = (4, 0, 2)
    query, lookup = region_completion_rows(rows, sites)
    phi = reconstruct_region_queries((2+3j)*full[query @ (1 << np.arange(5, -1, -1))], lookup)
    outside = tuple(i for i in range(6) if i not in sites)
    matrix = np.transpose(full.reshape((2,)*6), sites+outside).reshape(8, 8)
    expected = matrix @ matrix.conj().T
    op = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8))
    report, actual = cached_region_readout(rows, sites, phi, proposal, dict(o=op), groups=4)
    np.testing.assert_allclose(actual, expected, atol=7e-16, rtol=0)
    value = report['observables']['o']
    np.testing.assert_allclose(value['real']+1j*value['imaginary'], np.trace(expected @ op), atol=3e-15)
