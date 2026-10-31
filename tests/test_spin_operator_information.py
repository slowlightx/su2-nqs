"""Independent Casimir/pure-six-body checks; not a training-derived oracle."""
import itertools
import numpy as np
import pytest

from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split, spin_dot_matrix


def _pair(n, i, j):
    # Kronecker Pauli oracle, independent of bit-swap production construction.
    sx = np.array([[0, 1], [1, 0]])/2
    sy = np.array([[0, 1j], [-1j, 0]])/2
    sz = np.diag([-.5, .5])
    result = np.zeros((2**n, 2**n), complex)
    for s in (sx, sy, sz):
        term = np.ones((1, 1))
        for k in range(n): term = np.kron(term, s if k in (i, j) else np.eye(2))
        result += term
    return result


def _projector(n, sites, q):
    c = .75*len(sites)*np.eye(2**n)+2*sum(_pair(n, i, j) for i, j in itertools.combinations(sites, 2))
    p = np.eye(2**n, dtype=complex)
    for k in range(len(sites)%2, len(sites)+1, 2):
        if k != q: p = p@(c-k*(k+2)*np.eye(2**n)/4)/((q*(q+2)-k*(k+2))/4)
    return p


@pytest.mark.parametrize('q', [1, 3])
def test_casimir_projector_and_genuine_six_body_remainder(q):
    n = 6; left = (0, 2, 4); right = (1, 3, 5)
    split = joint_spin_operator_split(n, left, (q, q, 0))
    oracle = _projector(n, left, q)@_projector(n, right, q)@_projector(n, tuple(range(n)), 0)
    np.testing.assert_allclose(split.full, oracle, atol=1e-13, rtol=0)
    assert split.diagnostics['low_order_rank'] == 61
    assert split.diagnostics['orthogonality_max'] < 1e-12
    assert split.diagnostics['high_hermiticity_error'] < 1e-12
    np.testing.assert_allclose(split.full, split.low+split.high, atol=1e-14, rtol=0)
    # Tracing ANY one spin gives zero: Pauli strings in high have weight six,
    # hence cannot be any sum of operators on five or fewer physical spins.
    for k in range(n):
        traced = np.trace(split.high.reshape((2,)*(2*n)), axis1=k, axis2=n+k)
        assert np.linalg.norm(traced) < 2e-13
    assert np.linalg.norm(split.high) > .1
    pairs = [_pair(n, i, j) for i, j in itertools.combinations(range(n), 2)]
    for a, b in itertools.combinations_with_replacement(pairs, 2):
        assert abs(np.trace(split.high@(a@b+b@a)/2)) < 1e-12


def test_four_spin_negative_control_and_input_validation():
    split = joint_spin_operator_split(4, (0, 2), (2, 2, 0))
    assert np.linalg.norm(split.high) < 1e-13
    np.testing.assert_allclose(spin_dot_matrix(4, 0, 2), _pair(4, 0, 2), atol=1e-15, rtol=0)
    for args in [(6, (0, 2, 4), (1.5, 1, 0)), (6, (0, 2, 4), (2, 2, 0)),
                 (6, (0, 0), (0, 0, 0)), (True, (0,), (1, 1, 0))]:
        with pytest.raises(ValueError): joint_spin_operator_split(*args)


def test_six_body_readout_is_two_explicit_three_bond_orbits():
    """Interpretation via independent Pauli products, not fitted state data."""
    from functools import reduce
    def matchings(sites):
        if not sites:return [()]
        a=sites[0]
        return [((a,b),)+tail for b in sites[1:]
                for tail in matchings(tuple(x for x in sites[1:] if x!=b))]
    covers=matchings(tuple(range(6)))
    dots={ij:_pair(6,*ij) for ij in itertools.combinations(range(6),2)}
    products=[reduce(np.matmul,[dots[bond] for bond in cover]) for cover in covers]
    assert len(covers)==15
    for left in ((0,2,4),(0,1,2)):
        cross=[sum((a in left)!=(b in left) for a,b in cover) for cover in covers]
        assert cross.count(1)==9 and cross.count(3)==6
        x=sum((op for op,c in zip(products,cross) if c==1),start=np.zeros((64,64),complex))
        y=sum((op for op,c in zip(products,cross) if c==3),start=np.zeros((64,64),complex))
        np.testing.assert_allclose(joint_spin_operator_split(6,left,(1,1,0)).high,-4*x/9,atol=3e-13,rtol=0)
        np.testing.assert_allclose(joint_spin_operator_split(6,left,(3,3,0)).high,x/9-y/3,atol=3e-13,rtol=0)


def test_three_bond_connected_readout_product_state_and_projector_identity():
    from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
    # Six independent down spins: every disjoint product factorizes exactly.
    psi=np.zeros(64);psi[0]=1
    result=three_bond_readout(np.outer(psi,psi),(0,2,4))
    assert result['orbits']['X']['count']==9
    assert result['orbits']['Y']['count']==6
    assert max(abs(r['connected']) for r in result['matchings'])<1e-14
    assert result['orbits']['X']['raw']==9/64
    assert result['orbits']['Y']['raw']==6/64
    for q in (1,3):
        value=joint_spin_operator_split(6,(0,2,4),(q,q,0)).measure(np.outer(psi,psi))['high']
        assert abs(value-result[f'high_q{q}'])<1e-13
    # The maximally mixed state has no disjoint bond product or cumulant.
    mixed=three_bond_readout(np.eye(64)/64,(0,1,2))
    assert all(abs(v[k])<1e-15 for v in mixed['orbits'].values() for k in ('raw','connected'))
    with pytest.raises(ValueError):three_bond_readout(np.eye(64),(0,2,4))
    with pytest.raises(ValueError):three_bond_readout(np.eye(64)/64,(0,2,2))
