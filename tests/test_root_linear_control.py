import numpy as np
import pytest

from su2_nqs.vmc.root_linear_control import (
    projected_root_residual_direction,root_line_diagnostics,match_root_line_radius,native_root_matrix,
)


def test_complex_projected_metric_direction_matches_independent_dense_ridge():
    rng=np.random.default_rng(61101)
    f=rng.normal(size=(11,7))+1j*rng.normal(size=(11,7))
    a=rng.normal(size=(11,11))+1j*rng.normal(size=(11,11));h=(a+a.conj().T)/2
    c=rng.normal(size=7);c/=np.linalg.norm(c);hp=h@(f@c)
    q=np.ones(11)
    d,report=projected_root_residual_direction(f,c,hp,q,maximum_rank=7)
    p=f@c;norm=np.vdot(p,p).real;e=np.vdot(p,hp).real/norm
    centered=(f-p[:,None]*(p.conj()@f/norm)[None,:])/np.sqrt(norm)
    metric=(centered.conj().T@centered).real
    vals,vecs=np.linalg.eigh(metric);keep=vals>vals.max()*1e-10
    force=2*np.real(f.conj().T@(hp-e*p))/norm
    expected=-vecs[:,keep]@((vecs[:,keep].T@force)/(vals[keep]+.1*vals.max()))
    expected-=c*(c@expected);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(d,expected,atol=2e-13,rtol=0)
    assert abs(c@d)<2e-14 and abs(d@d-1)<2e-14
    assert report['root_only_not_all_native_W']
    rotation=np.linalg.qr(rng.normal(size=(7,7)))[0]
    rotated,_=projected_root_residual_direction(f@rotation,rotation.T@c,hp,q,maximum_rank=7)
    np.testing.assert_allclose(rotation@rotated,d,atol=3e-13,rtol=0)


def test_finite_bilateral_curvature_and_arclength_are_coordinate_covariant():
    rng=np.random.default_rng(61102)
    f=rng.normal(size=(17,2))+1j*rng.normal(size=(17,2))
    h=rng.normal(size=f.shape)+1j*rng.normal(size=f.shape) # intentionally non-Hermitian sampled cross action
    q=rng.uniform(.1,1.,17);t=.08;step=1e-4
    row=root_line_diagnostics(f,h,q,t)
    plus=root_line_diagnostics(f,h,q,t+step);minus=root_line_diagnostics(f,h,q,t-step)
    assert row['energy_first']==pytest.approx((plus['energy_real']-minus['energy_real'])/(2*step),abs=2e-8)
    assert row['energy_second']==pytest.approx((plus['energy_real']-2*row['energy_real']+minus['energy_real'])/step**2,abs=3e-7)
    assert row['FS_metric_first']==pytest.approx((plus['FS_metric']-minus['FS_metric'])/(2*step),abs=3e-8)
    factor=3.7
    other=root_line_diagnostics(f*np.array([1.,factor]),h*np.array([1.,factor]),q,t/factor)
    for name in ('energy_real','FS_radians','arclength_slope','arclength_curvature'):
        assert other[name]==pytest.approx(row[name],abs=3e-13)
    finite_arc=(plus['arclength_slope']-minus['arclength_slope'])/(2*step*np.sqrt(row['FS_metric']))
    assert finite_arc==pytest.approx(row['arclength_curvature'],abs=2e-7)
    matched,diagnostic=match_root_line_radius(f,q,.0135,bracket=.5,iterations=16)
    assert root_line_diagnostics(f,h,q,matched)['FS_radians']==pytest.approx(.0135,abs=6e-6)
    assert diagnostic['amplitude_only']


def test_root_node_action_is_not_discarded_and_invalid_control_fails():
    f=np.array([[1.,0.],[0.,1.],[0.,0.]])
    h=np.array([[0.,1.],[1.,0.],[2.,3.]])
    result=root_line_diagnostics(f,h,np.ones(3),0.)
    assert result['residual_variance']==pytest.approx(5.)
    with pytest.raises(ValueError):
        match_root_line_radius(f,np.ones(3),.1,bracket=.0001)
    with pytest.raises(ValueError):
        projected_root_residual_direction(f,np.array([2.,0.]),h[:,0],np.ones(3))


def test_complex_native_root_transport_matches_independent_physical_ket_matrix():
    rng=np.random.default_rng(61103)
    left=np.linalg.qr(rng.normal(size=(8,3))+1j*rng.normal(size=(8,3)))[0]
    right=np.linalg.qr(rng.normal(size=(8,2))+1j*rng.normal(size=(8,2)))[0]
    a=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))+3*np.eye(3)
    b=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))+3*np.eye(2)
    c=rng.normal(size=(3,2))+1j*rng.normal(size=(3,2))
    native=native_root_matrix(c,left.conj().T@(left@a),right.conj().T@(right@b))
    np.testing.assert_allclose(left@c@right.T,(left@a)@native@(right@b).T,atol=4e-14,rtol=0)


def test_bilateral_root_force_matches_complex_nonhermitian_finite_quotient():
    rng=np.random.default_rng(61104)
    f=rng.normal(size=(13,7))+1j*rng.normal(size=(13,7))
    hf=rng.normal(size=f.shape)+1j*rng.normal(size=f.shape)
    q=rng.uniform(.2,1.,len(f));c=rng.normal(size=7);c/=np.linalg.norm(c)
    d,report=projected_root_residual_direction(f,c,hf@c,q,hfeatures=hf,maximum_rank=7)
    x=f/np.sqrt(q[:,None]);y=hf/np.sqrt(q[:,None]);p=x@c;hp=y@c
    norm=np.vdot(p,p).real;e=np.vdot(p,hp).real/norm
    force=np.real(x.conj().T@hp+y.conj().T@p-2*e*x.conj().T@p)/norm
    def energy(v):
        return np.vdot(x@v,y@v).real/np.vdot(x@v,x@v).real
    eps=1e-6
    finite=np.array([(energy(c+eps*v)-energy(c-eps*v))/(2*eps) for v in np.eye(7)])
    np.testing.assert_allclose(force,finite,atol=2e-10,rtol=0)
    center=(x-p[:,None]*(p.conj()@x/norm)[None,:])/np.sqrt(norm)
    metric=(center.conj().T@center).real;vals,vecs=np.linalg.eigh(metric)
    keep=vals>vals.max()*1e-10
    expected=-vecs[:,keep]@((vecs[:,keep].T@force)/(vals[keep]+.1*vals.max()))
    expected-=c*(c@expected);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(d,expected,atol=3e-13,rtol=0)
    line=root_line_diagnostics(np.column_stack((f@c,f@d)),np.column_stack((hf@c,hf@d)),q,0.)
    assert report['selected_force_dot_direction']==pytest.approx(line['energy_first'],abs=2e-14)
    assert report['selected_force_is_exact_finite_sample_derivative']
    assert np.linalg.norm(force-2*np.real(x.conj().T@(hp-e*p))/norm)>.1
    with pytest.raises(ValueError):
        projected_root_residual_direction(f,c,hf@c,q,hfeatures=hf+.1)
