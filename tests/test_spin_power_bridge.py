"""Independent dense Pauli oracle, including nodes and ordered products."""
import itertools

import numpy as np
import pytest

from su2_nqs.benchmarks.operator_word_jax import evaluate_scalar_word_action_plan_numpy
from su2_nqs.benchmarks.spin_power_bridge import build_bounded_spin_power_plan, standardized_krylov_polynomial


@pytest.mark.parametrize('n',[4,6])
def test_powers_through_three_against_dense_pauli(n):
    states = np.array(list(itertools.product((0,1),repeat=n)),dtype=np.int8)
    bonds = [(i,(i+1)%n,(-1)**i*.7) for i in range(n)] + [(0,1,.3),(1,0,.2)]
    sx=np.array([[0,1],[1,0]],complex)/2
    sy=np.array([[0,1j],[-1j,0]],complex)/2
    sz=np.diag([-.5,.5])
    h=np.zeros((2**n,2**n),complex)
    for a,b,c in bonds:
        for s in (sx,sy,sz):
            term=np.ones((1,1),complex)
            for k in range(n):
                term=np.kron(term,s if k in (a,b) else np.eye(2))
            h += c*term
    rng=np.random.default_rng(1902)
    p=rng.normal(size=2**n)+1j*rng.normal(size=2**n); p[5]=0
    plan, work = build_bounded_spin_power_plan(states,bonds,maximum_power=3)
    got=evaluate_scalar_word_action_plan_numpy(plan,
        lambda z:p[z@(1<<np.arange(n-1,-1,-1))],amplitude_batch_size=7)
    expected=[p]
    for _ in range(3): expected.append(h@expected[-1])
    np.testing.assert_allclose(got,np.column_stack(expected),atol=8e-14,rtol=2e-14)
    assert not work['small_coefficient_clipping'] and work['full_H_applications_per_center']==3


def test_tiny_coefficients_not_clipped_and_zero_hamiltonian():
    x=np.array([[1,0]],np.int8)
    plan,_=build_bounded_spin_power_plan(x,[(0,1,1e-16)],maximum_power=3)
    assert np.any((abs(plan.entry_coefficients)>0)&(abs(plan.entry_coefficients)<1e-40))
    plan,_=build_bounded_spin_power_plan(x,[(0,1,1.),(1,0,-1.)],maximum_power=3)
    got=evaluate_scalar_word_action_plan_numpy(plan,lambda z:np.ones(len(z)),amplitude_batch_size=2)
    np.testing.assert_array_equal(got,[[1,0,0,0]])


@pytest.mark.parametrize('bond',[(0,0),(0.,1),(0,4),(0,1,np.inf)])
def test_bad_bonds_rejected(bond):
    with pytest.raises(ValueError): build_bounded_spin_power_plan(np.zeros((1,4),np.int8),[bond],maximum_power=3)


def test_resource_bounds_fail_closed():
    with pytest.raises(MemoryError):
        build_bounded_spin_power_plan(np.array([[1,0,1,0]],np.int8),[(0,1),(1,2),(2,3)],maximum_power=3,maximum_expansion_entries=2)
    with pytest.raises(ValueError):
        build_bounded_spin_power_plan(np.zeros((1,4)),[(0,1)],maximum_power=3)


def test_algebraic_polynomial_conversion_preserves_noncommuting_order():
    rng=np.random.default_rng(721)
    a=rng.normal(size=(5,5)); a=(a+a.T)/2
    b=rng.normal(size=(5,5)); b=(b+b.T)/2
    m,s,w=np.array([.2,-.8]),np.array([1.7,.4]),np.array([1.,.125])
    poly=np.array([.9,-.3,.02]); mu=w@m
    # H=mu I+sum_i w_i s_i Z_i; coefficients of an EXACT polynomial.
    linear=(poly[1]+2*poly[2]*mu)*w*s
    quad=poly[2]*np.outer(w*s,w*s)
    coefficients=np.r_[poly[0]+poly[1]*mu+poly[2]*mu**2,linear,quad.ravel()]
    result, info=standardized_krylov_polynomial(coefficients,m,s,w)
    np.testing.assert_allclose(result,poly,atol=2e-16)
    z=[(op-mi*np.eye(5))/si for op,mi,si in zip((a,b),m,s)]
    original=coefficients[0]*np.eye(5)+sum(c*op for c,op in zip(linear,z))
    original += sum(quad[i,j]*z[i]@z[j] for i in range(2) for j in range(2))
    h=a+.125*b
    np.testing.assert_allclose(original,result[0]*np.eye(5)+result[1]*h+result[2]*h@h,atol=3e-15)
    assert info['algebraic_conversion_not_variational_refit']
    coefficients[-2]+=.01 # Never silently symmetrize/project a non-Krylov family.
    with pytest.raises(ValueError): standardized_krylov_polynomial(coefficients,m,s,w)
