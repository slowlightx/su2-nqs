"""Independent checks of the cached-state charge-process analysis."""
import numpy as np
from scipy.linalg import expm

from projects.clebsch_tree_nqs.workflows.su2_structure_mechanism_analysis import (
    center_resolve, charge_process_operators, multiplicity_state, validate_operators,
    dimer_operator_moments, dimer_pauli_expectations,
)
from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix


def test_charge_blocks_and_sum_on_independent_bit_swaps():
    # Random invariant scalar polynomial exercises all four charge blocks.
    a = spin_dot_matrix(5, 0, 2)
    b = spin_dot_matrix(5, 1, 3)
    w = a+.31*b+.27*(a@b+b@a)
    ops = charge_process_operators(w, (0, 3), (2, 4))
    report = validate_operators(ops, (0, 3), (2, 4))
    assert report['projector_swap_error'] == 0
    assert report['sum_closure_max'] < 2e-15
    p = ops['charge_projectors']
    for i in range(4):
        np.testing.assert_allclose(p[i]@p[i], p[i], atol=2e-15)
        for j in range(i):
            np.testing.assert_allclose(p[i]@p[j], 0, atol=2e-15)
    # Transfer cannot connect the (0,0) or (2,2) charge sectors.
    np.testing.assert_allclose((p[0]+p[3])@ops['score_transfer'], 0, atol=2e-15)


def test_center_split_independent_pauli_strings():
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., 1j], [-1j, 0.]])
    sz = np.diag([-1., 1.]); eye = np.eye(2)
    def kron(xs):
        out = np.array([[1.]])
        for x in xs: out = np.kron(out, x)
        return out
    spectator = kron([sx, eye, sy, sz, sx])
    correlated = kron([sy, sz, sx, eye, sz])
    a, b = center_resolve(spectator+.47*correlated, 1)
    np.testing.assert_allclose(a, spectator, atol=2e-15)
    np.testing.assert_allclose(b, .47*correlated, atol=2e-15)
    assert abs(np.vdot(a,b)) < 2e-15


def test_locally_encoded_multiplicity_bell_pair_and_product():
    u, labels = regional_schur_columns(3)
    def col(m, a): return u[:, labels.index((1, m, a))]
    a = np.kron(col(-1, 0), col(1, 0))
    b = np.kron(col(-1, 1), col(1, 1))
    bell = (a+b)/np.sqrt(2)
    reduced = multiplicity_state(np.outer(bell, bell.conj()), (0,1,2))
    target = np.array([1,0,0,1])/np.sqrt(2)
    np.testing.assert_allclose(reduced, np.outer(target,target), atol=3e-15)
    pt = reduced.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
    np.testing.assert_allclose(np.linalg.eigvalsh(pt), [-.5,.5,.5,.5], atol=3e-15)
    product = multiplicity_state(np.outer(a,a.conj()), (0,1,2))
    np.testing.assert_allclose(product, np.diag([1,0,0,0]), atol=3e-15)


def test_multiplicity_reduction_invariant_under_independent_block_rotations():
    rng = np.random.default_rng(20261013)
    v = rng.normal(size=64)+1j*rng.normal(size=64); v /= np.linalg.norm(v)
    rho = np.outer(v,v.conj())
    sx = np.array([[0., .5], [.5, 0.]])
    sz = np.diag([-.5,.5])
    l, r = expm(.43j*sx), expm(-.27j*sz)
    rotation = np.kron(np.kron(np.kron(l,l),l), np.kron(np.kron(r,r),r))
    np.testing.assert_allclose(multiplicity_state(rotation@rho@rotation.conj().T, (0,1,2)),
                               multiplicity_state(rho, (0,1,2)), atol=3e-15)


def test_dimer_joint_moments_against_full_six_spin_wavefunction():
    pairs = ((0,3),(1,2),(4,5))
    bits = (np.arange(64)[:,None] >> np.arange(5,-1,-1)) & 1
    state = np.ones(64)
    for a,b in pairs:
        state *= np.where(bits[:,a] != bits[:,b], np.where(bits[:,a] == 0, 1., -1.)/np.sqrt(2), 0.)
    np.testing.assert_allclose(state@state,1.,atol=2e-15)
    rng = np.random.default_rng(171013)
    words = rng.integers(0,4,size=(13,3)); coefficients = rng.normal(size=13)
    sites_a,sites_b = (0,1,4),(2,3,5)
    pauli = (np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,1j],[-1j,0]]),np.diag([-1,1]))
    def full_operator(sites):
        total = np.zeros((64,64),complex)
        for word,c in zip(words,coefficients):
            labels = dict(zip(sites,word)); op = np.array([[1.]])
            for site in range(6): op = np.kron(op,pauli[labels.get(site,0)])
            total += c*op
        return total
    a,b = full_operator(sites_a),full_operator(sites_b)
    result = dimer_operator_moments(pairs,sites_a,sites_b,words,coefficients)
    np.testing.assert_allclose(result['mean_A'],np.vdot(state,a@state),atol=3e-14)
    np.testing.assert_allclose(result['mean_B'],np.vdot(state,b@state),atol=3e-14)
    np.testing.assert_allclose(result['joint'],np.vdot(state,a@b@state),atol=3e-14)


def test_dimer_strings_with_external_partners():
    pairs = ((0,1),(2,3))
    # 0's partner lies outside the requested region; any Pauli at 0 gives zero.
    words = np.array([[0,1,1],[0,2,2],[0,3,3],[1,1,1],[0,1,2],[0,0,0]])
    values = dimer_pauli_expectations(pairs,(0,2,3),words)
    np.testing.assert_array_equal(values,[-1,-1,-1,0,0,1])
