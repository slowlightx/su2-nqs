from itertools import combinations,product

import numpy as np

from su2_nqs.benchmarks.stratified_finite_audit import (select_strata,carrier_moments,
    correction_moments,stratified_correction,physical_values_and_jacobian,summarize_independent_pools)
from su2_nqs.benchmarks.finite_scalar_dressing import finite_step_moments


def test_stratified_mean_and_design_covariance_are_exact_in_exhaustive_draws():
    values=np.array([[1.,2.],[3.,-4.],[8.,5.],[20.,7.],[30.,-3.],[80.,25.],[1000.,-90.]])
    labels=np.array([0,0,0,1,1,1,2]);counts=np.bincount(labels)
    means,covariances=[],[]
    for a,b in product(combinations(range(3),2),combinations(range(3,6),2)):
        ix=np.array([*a,*b,6]);r=stratified_correction(values[ix],labels[ix],counts)
        means.append(r['mean']);covariances.append(r['covariance'])
    np.testing.assert_allclose(np.mean(means,axis=0),values.mean(axis=0),atol=1e-13)
    np.testing.assert_allclose(np.mean(covariances,axis=0),np.cov(means,rowvar=False,ddof=0),atol=3e-13)
    selection=select_strata(labels,[2,2,None],seed=42)
    assert len(np.unique(selection['indices']))==5
    assert selection['indices'][-1]==6 and selection['inclusion_probability'][-1]==1


def test_centered_quadratic_moments_and_jacobian_include_carrier_nodes():
    rng=np.random.default_rng(2026092830);p=rng.normal(size=12)+1j*rng.normal(size=12);p[0]=0
    d=rng.normal(size=12)+1j*rng.normal(size=12)
    h=rng.normal(size=(12,12));h=(h+h.T)/2
    q=abs(p)**2+.2;epsilon=.5;eref=-2.3
    a=carrier_moments(p,h@p,q,eref);delta=correction_moments(p,d,h@p,h@d,q,epsilon,eref)
    np.testing.assert_allclose(a+delta,carrier_moments(p+epsilon*d,h@(p+epsilon*d),q,eref),atol=2e-13)
    t=np.r_[a.mean(axis=0),delta.mean(axis=0)];v,jac=physical_values_and_jacobian(t,eref)
    ref=finite_step_moments(p,d,h@p,h@d,q,epsilon)
    np.testing.assert_allclose(v[[0,2,4,5,6]],[ref[k] for k in ('norm_ratio','energy_total','variance_total','energy_change_total','variance_change_total')],atol=2e-13)
    for i in range(6):
        shift=np.zeros(6);shift[i]=1e-5
        numeric=(physical_values_and_jacobian(t+shift,eref)[0]-physical_values_and_jacobian(t-shift,eref)[0])/2e-5
        np.testing.assert_allclose(jac[:,i],numeric,atol=2e-7,rtol=2e-8)
    repeated=t+rng.normal(size=(8,6))*1e-4
    r=summarize_independent_pools(repeated,np.zeros((8,6,6)),eref)
    assert r['degrees_of_freedom']==7 and 2.36<r['t_critical_95']<2.37
