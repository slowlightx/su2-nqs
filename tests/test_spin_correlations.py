"""Independent Pauli-matrix and complex/nodal physical-spin oracles."""
import numpy as np
import pytest

from su2_nqs.benchmarks.spin_correlations import (
    spin_dot_connections, dimer_product_connections, chirality_dot_connections,
)
from su2_nqs.benchmarks.replica_spin_response import operator_action_over_sample


def dense_dot(n, bond):
    spin = [np.array([[0, 1], [1, 0]], complex)/2,
            np.array([[0, 1j], [-1j, 0]], complex)/2, np.diag([-.5, .5])]
    result = np.zeros((2**n, 2**n), complex)
    for s in spin:
        term = np.ones((1, 1))
        for k in range(n):
            term = np.kron(term, s if k in bond else np.eye(2))
        result += term
    return result


@pytest.mark.parametrize('left,right', [((0, 1), (0, 1)), ((0, 1), (1, 2)), ((0, 1), (2, 3))])
def test_composed_dimer_rows_match_independent_dense_matrices(left, right):
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int8)
    a, b = dense_dot(4, left), dense_dot(4, right)
    for connector, oracle in ((spin_dot_connections(bits, left), a),
                              (dimer_product_connections(bits, left, right), (a@b+b@a)/2)):
        t, s, c = connector
        actual = np.zeros((16, 16), complex)
        np.add.at(actual, (s, t@(1 << np.arange(3, -1, -1))), c)
        np.testing.assert_allclose(actual, oracle, atol=1e-15, rtol=0)
    rng = np.random.default_rng(671008)
    psi = rng.normal(size=16)+1j*rng.normal(size=16)
    psi[[1, 5, 8]] = 0; psi /= np.linalg.norm(psi)
    ids = np.flatnonzero(psi)
    t, s, c = dimer_product_connections(bits[ids], left, right)
    with np.errstate(divide='ignore'):
        logs = np.log(psi[t@(1 << np.arange(3, -1, -1))])
    local = operator_action_over_sample(logs, s, c, np.log(psi[ids]))
    np.testing.assert_allclose(abs(psi[ids])**2@local, np.vdot(psi, (a@b+b@a)@psi)/2,
                               atol=1e-15, rtol=0)


def test_spin_connector_rejects_invalid_sites_and_keeps_spin_conservation():
    x = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8)
    t, s, _ = spin_dot_connections(x, (0, 1))
    np.testing.assert_array_equal(t.sum(1), x[s].sum(1))
    for bond in ((0, 0), (True, 1), (0, 3)):
        with pytest.raises(ValueError):
            spin_dot_connections(x, bond)


def test_small_even_spin_leakage_can_have_first_order_rank_two_anisotropy():
    # Analytic (triplet x triplet) S=0 and S=2,m=0; no ED or neural fit.
    # For N=4k, fixed Sz=0 and even bit flip exclude odd total S, NOT S=2.
    down=np.array([1.,0,0,0]);up=down[::-1]
    zero=np.array([0.,1,1,0])/np.sqrt(2)
    singlet=(np.kron(up,down)+np.kron(down,up)-np.kron(zero,zero))/np.sqrt(3)
    quintet=(np.kron(up,down)+np.kron(down,up)+2*np.kron(zero,zero))/np.sqrt(6)
    epsilon=.1;psi=np.sqrt(1-epsilon**2)*singlet+epsilon*quintet
    total=3*np.eye(16)+2*sum(dense_dot(4,(i,j))for i in range(4)for j in range(i+1,4))
    np.testing.assert_allclose(total@singlet,0,atol=1e-15,rtol=0)
    np.testing.assert_allclose(total@quintet,6*quintet,atol=1e-15,rtol=0)
    np.testing.assert_allclose(psi,psi[::-1],atol=0,rtol=0)
    np.testing.assert_allclose(np.vdot(psi,total@psi),6*epsilon**2,atol=1e-15,rtol=0)
    bits=(np.arange(16)[:,None]>>np.arange(3,-1,-1))&1
    assert np.all(bits[np.flatnonzero(psi)].sum(-1)==2)
    zz=np.diag((bits[:,0]-.5)*(bits[:,1]-.5))
    difference=np.vdot(psi,(3*zz-dense_dot(4,(0,1)))@psi)
    np.testing.assert_allclose(difference,np.sqrt(2)*epsilon*np.sqrt(1-epsilon**2)-epsilon**2/2,
                               atol=1e-15,rtol=0)
    assert difference>.13  # one-percent S=2 weight is not one-percent isotropy error


def test_chirality_dot_matches_independent_pauli_cross_product_and_is_scalar():
    spin = [np.array([[0, 1], [1, 0]], complex)/2,
            np.array([[0, 1j], [-1j, 0]], complex)/2, np.diag([-.5, .5])]
    def embed(s, site):
        a = np.ones((1, 1))
        for k in range(4):a = np.kron(a, s if k == site else np.eye(2))
        return a
    local = [[embed(s, i)for s in spin]for i in range(4)]
    cross = lambda i,j: [local[i][a]@local[j][b]-local[i][b]@local[j][a]
                         for a,b in [(1,2),(2,0),(0,1)]]
    left, right = cross(0,1), cross(2,3)
    oracle = sum(a@b for a,b in zip(left,right))
    bits = ((np.arange(16)[:,None] >> np.arange(3,-1,-1))&1).astype(np.int8)
    t,s,c = chirality_dot_connections(bits,(0,1),(2,3))
    actual = np.zeros((16,16),complex)
    np.add.at(actual,(s,t@(1<<np.arange(3,-1,-1))),c)
    np.testing.assert_allclose(actual,oracle,atol=2e-15,rtol=0)
    for axis in range(3):
        total=sum(local[i][axis] for i in range(4))
        np.testing.assert_allclose(actual@total,total@actual,atol=2e-15,rtol=0)
    reverse=chirality_dot_connections(bits,(1,0),(2,3))
    rev=np.zeros_like(actual);np.add.at(rev,(reverse[1],reverse[0]@(1<<np.arange(3,-1,-1))),reverse[2])
    np.testing.assert_allclose(rev,-actual,atol=2e-15,rtol=0)
    rng=np.random.default_rng(671016)
    psi=rng.normal(size=16)+1j*rng.normal(size=16)
    psi[bits.sum(1)!=2]=0;psi[5]=0;psi/=np.linalg.norm(psi)
    keep=np.flatnonzero(psi)
    t,s,c=chirality_dot_connections(bits[keep],(0,1),(2,3))
    with np.errstate(divide='ignore'):
        logs=np.log(psi[t@(1<<np.arange(3,-1,-1))])
    estimate=abs(psi[keep])**2@operator_action_over_sample(logs,s,c,np.log(psi[keep]))
    np.testing.assert_allclose(estimate,np.vdot(psi,oracle@psi),atol=2e-15,rtol=0)
    # Fixed Sz is not isotropy. A three-times-kappa_z shortcut fails here.
    assert abs(np.vdot(psi,(oracle-3*left[2]@right[2])@psi))>.01
    for pair in [((0,1),(1,2)),((0,0),(2,3)),((0,1),(2,4))]:
        with pytest.raises(ValueError):chirality_dot_connections(bits,*pair)
