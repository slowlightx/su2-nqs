"""Complete finite-Q gradients, nodes, and strict joint actual-step gates."""
import numpy as np
import pytest

from su2_nqs.vmc.carrier_protection import physical_moments
from su2_nqs.vmc.joint_linear_step import linear_moment_slopes,joint_linear_coefficients,joint_step_gate


def test_both_coherences_and_second_moment_slopes_match_complex_nonnormal_finite_quadrature():
    rng=np.random.default_rng(1721);p=rng.normal(size=13)+1j*rng.normal(size=13);p[2]=0
    h=rng.normal(size=13)+1j*rng.normal(size=13)
    t=rng.normal(size=(13,3))+1j*rng.normal(size=(13,3))
    ht=rng.normal(size=(13,3))+1j*rng.normal(size=(13,3));q=rng.uniform(.2,2.,13)
    actual=linear_moment_slopes(p,h,t,ht,q,12);epsilon=2e-6
    for col in range(3):
        a=physical_moments(p+epsilon*t[:,col],h+epsilon*ht[:,col],q,12)
        b=physical_moments(p-epsilon*t[:,col],h-epsilon*ht[:,col],q,12)
        np.testing.assert_allclose(actual['energy'][col],(a['energy_per_site']-b['energy_per_site'])/(2*epsilon),rtol=2e-8,atol=2e-10)
        np.testing.assert_allclose(actual['variance'][col],(a['variance_per_site']-b['variance_per_site'])/(2*epsilon),rtol=2e-8,atol=2e-10)


def test_joint_existing_columns_descend_both_moments_on_each_exact_block():
    p=np.tile(np.array([1.,.25,.1]),2).astype(complex)
    eigen=np.tile([-1.,1.,2.],2);h=eigen*p
    t=np.tile(np.array([[0.],[1.],[.5]]),(2,1)).astype(complex);ht=eigen[:,None]*t
    blocks=[np.arange(6)<3,np.arange(6)>=3]
    c,info=joint_linear_coefficients(p,h,t,ht,np.ones(6),blocks,n_sites=3)
    assert info['common_descent'] and info['no_new_H_calls']
    for m in blocks:
        s=linear_moment_slopes(p[m],h[m],t[m]@c[:,None],ht[m]@c[:,None],np.ones(3),3)
        assert s['energy'][0]<0 and s['variance'][0]<0
    beta=np.vdot(p,t@c)/np.vdot(p,p)
    np.testing.assert_allclose(np.linalg.norm(t@c-beta*p)/np.linalg.norm(p),1.,rtol=3e-13)


def test_joint_gate_rejects_energy_only_improvement_and_accepts_smaller_quality_safe_step():
    before=[dict(energy_per_site=-.5,variance_per_site=.02)]*2
    large=[dict(energy_per_site=-.51,variance_per_site=.021)]*2
    small=[dict(energy_per_site=-.5001,variance_per_site=.0199)]*2
    args=dict(energy_tolerance=2e-5,variance_ratio=1.25)
    assert not joint_step_gate(before,large,before,**args)['accepted']
    assert joint_step_gate(before,small,before,**args)['accepted']
    with pytest.raises(ValueError):linear_moment_slopes(np.ones(3),np.ones(3),np.ones((3,1)),np.ones((3,1)),np.zeros(3),3)
