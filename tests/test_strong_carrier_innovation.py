"""Independent finite-dimensional tests, before using cached spin states."""

import numpy as np
import pytest

from su2_nqs.benchmarks.strong_carrier_innovation import operator_innovation
from su2_nqs.benchmarks.spectral_error_tomography import lowest_subspace_state


def test_innovation_is_covariant_and_preserves_enlarged_ritz_space():
    rng=np.random.default_rng(2026092840)
    a=rng.normal(size=(11,11))+1j*rng.normal(size=(11,11));h=a+a.conj().T
    a=rng.normal(size=(11,11))+1j*rng.normal(size=(11,11));op=a+a.conj().T
    b=rng.normal(size=(11,3))+1j*rng.normal(size=(11,3))
    r=operator_innovation(h,b,op)
    assert r['diagnostics']['direct_rank']==4 and r['diagnostics']['conditional_rank']==5
    np.testing.assert_allclose(r['h_basis'].conj().T@r['innovation'],0,atol=3e-14)
    assert abs(r['diagnostics']['innovation_first_derivative_total'])<2e-13
    original,_=lowest_subspace_state(np.column_stack((r['h_basis'],op@r['carrier'])),h)
    assert abs(np.vdot(original,r['conditional']))>1-2e-13
    u,_=np.linalg.qr(rng.normal(size=(11,11))+1j*rng.normal(size=(11,11)))
    transformed=operator_innovation(u@h@u.conj().T,u@b,u@op@u.conj().T)
    for key in ('carrier','direct','conditional','h_once','h_twice'):
        assert abs(np.vdot(u@r[key],transformed[key]))>1-3e-13
    for key in ('outside_anchor_fraction','outside_anchor_and_h_fraction','next_gradient_capture_fraction'):
        np.testing.assert_allclose(r['diagnostics'][key],transformed['diagnostics'][key],atol=3e-13)


def test_hamiltonian_polynomial_has_no_innovation_beyond_one_h_step():
    h=np.diag(np.arange(1.,10.));p=np.ones(9)/3
    r=operator_innovation(h,np.column_stack((p,h@p)),2*h+np.eye(9))
    assert np.linalg.norm(r['innovation'])<2e-13
    assert r['diagnostics']['conditional_rank']==r['diagnostics']['h_once_rank']==3


def test_nonzero_independence_does_not_imply_ground_state_improvement():
    h=np.diag(np.arange(7.));b=np.eye(7)[:,:2]
    op=np.eye(7);op[0,5]=op[5,0]=1.
    r=operator_innovation(h,b,op)
    assert r['diagnostics']['outside_anchor_and_h_fraction']>.49
    for key in ('carrier','direct','conditional'):
        assert abs(np.vdot(r[key],h@r[key]))<1e-14
    with pytest.raises(ValueError,match='Hermitian'):
        operator_innovation(h,b,op+np.triu(np.ones((7,7)),1))
