"""Exhaustive nonuniform-Q coherence and exact-node tests of regional sampling."""
import itertools
import numpy as np
import pytest
from su2_nqs.vmc.regional_density import (regional_query_plan, conditional_regional_matrices,
    regional_density_summary, regional_observable_summary)


def test_exhaustive_positive_proposal_with_nodes_and_unsorted_sites():
    n = 6; x = np.array([b for b in itertools.product((0, 1), repeat=n) if sum(b) == 3], np.int8)
    rng = np.random.default_rng(2026100603)
    psi = rng.normal(size=len(x))+1j*rng.normal(size=len(x)); psi[::3] = 0
    psi /= np.linalg.norm(psi)
    q = .1+rng.random(len(x))
    lookup = {tuple(b):i for i, b in enumerate(x)}
    plan = regional_query_plan(x, (4, 0, 2))
    ids = [lookup[tuple(b)] for b in plan.configurations]
    a = conditional_regional_matrices(plan, psi[ids], q[ids])
    measured = np.einsum('c,cab->ab', q/q.sum(), a)
    measured /= np.trace(measured)
    full = np.zeros(2**n, complex)
    integers = x@(1 << np.arange(n-1, -1, -1)); full[integers] = psi
    # Independent direct partial trace in requested, non-sorted site order.
    order = (4, 0, 2, 1, 3, 5)
    m = full.reshape((2,)*n).transpose(order).reshape(8, 8)
    expected = m@m.conj().T
    np.testing.assert_allclose(measured, expected, atol=3e-15, rtol=0)
    assert np.linalg.norm(expected-np.diag(np.diag(expected))) > .1
    # A positive empirical matrix is NOT an SU2-twirled matrix.
    summary = regional_density_summary(a, np.arange(len(a))%4)
    assert summary['minimum_eigenvalue'] > -1e-14
    op = np.diag(np.arange(8))
    r = regional_observable_summary(summary, op)
    assert r['value'] == pytest.approx(np.trace(summary['rho']@op).real, abs=1e-14, rel=0)
    assert r['chain_jackknife_se'] >= 0


def test_invalid_proposal_and_sector_are_not_repaired():
    x = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], np.int8)
    plan = regional_query_plan(x, (0, 2))
    with pytest.raises(ValueError): conditional_regional_matrices(plan, np.ones(len(plan.configurations)), np.zeros(len(plan.configurations)))
    with pytest.raises(ValueError): regional_query_plan(np.array([[0, 1], [0, 0]]), (0,))
    with pytest.raises(ValueError): regional_query_plan(x, (0, 0))
    with pytest.raises(ValueError): regional_density_summary(np.zeros((2, 4, 4)), np.array([0, 1]))


def test_disjoint_conditional_quadrature_matches_full_complex_wavefunction():
    from su2_nqs.vmc.regional_density import disjoint_regional_query_plan, disjoint_born_operator_moments
    # Independent dense embedding, unsorted physical blocks, complex operators
    # and wavefunction: this tests both conjugation and MSB conventions.
    n = 8
    x = np.array([b for b in itertools.product((0, 1), repeat=n) if sum(b) == 4], np.int8)
    rng = np.random.default_rng(2026101401)
    psi = rng.normal(size=len(x))+1j*rng.normal(size=len(x))
    psi[::7] = 0
    psi /= np.linalg.norm(psi)
    regions = ((5, 0, 2), (7, 3, 1))
    op = np.zeros((8, 8), complex)
    for count in range(4):
        ids = [i for i in range(8) if i.bit_count() == count]
        a = rng.normal(size=(len(ids), len(ids)))+1j*rng.normal(size=(len(ids), len(ids)))
        op[np.ix_(ids, ids)] = (a+a.conj().T)/2
    lookup = {tuple(b): i for i, b in enumerate(x)}
    for cuts in (regions[:1], regions):
        plan = disjoint_regional_query_plan(x, cuts)
        queried = psi[[lookup[tuple(b)] for b in plan.configurations]]
        with np.errstate(divide='ignore'):
            moments = disjoint_born_operator_moments(plan, np.log(queried.astype(complex)), {'O':op})
        full = np.zeros(2**n, complex)
        full[x@(1 << np.arange(n-1, -1, -1))] = psi
        order = tuple(i for cut in cuts for i in cut)
        order += tuple(i for i in range(n) if i not in order)
        vec = full.reshape((2,)*n).transpose(order).reshape(-1)
        operators = {'A':np.kron(op, np.eye(2**(n-3)))}
        if len(cuts) == 2:
            operators.update(B=np.kron(np.eye(8), np.kron(op, np.eye(4))),
                             AB=np.kron(op, np.kron(op, np.eye(4))))
        for key, exact in operators.items():
            estimate = abs(psi)**2@moments['O__'+key]
            np.testing.assert_allclose(estimate, np.vdot(vec, exact@vec).real, atol=3e-14, rtol=0)
    with pytest.raises(ValueError):
        disjoint_regional_query_plan(x, ((0, 1), (1, 2)))
    bad = op.copy(); bad[0, 1] = bad[1, 0] = 1
    with pytest.raises(ValueError):
        disjoint_born_operator_moments(plan, np.log(np.ones(len(plan.configurations))), {'O':bad})
