"""Transport geometry, non-smooth rejection, and real-coordinate E/R slopes."""
import numpy as np
import pytest

from su2_nqs.vmc.importance_tangent import tangent_system
from su2_nqs.vmc.joint_linear_step import linear_moment_slopes
from su2_nqs.vmc.retraction_transport import retracted_real_columns,joint_real_coefficients


def inputs():
    rng=np.random.default_rng(1901)
    f=rng.normal(size=(15,4))+1j*rng.normal(size=(15,4))
    z=rng.normal(size=4)+1j*rng.normal(size=4)
    d=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    p=f@z;p[0]=0
    return z,d,f,p,rng.uniform(.5,2,15)


def test_linear_projection_transport_is_node_safe_and_gauge_invariant():
    z,d,f,p,q=inputs()
    projection=np.diag([1.,1.,0.,0.]);z=projection@z
    a,info=retracted_real_columns(z,d,lambda v:projection@v,f,p,q,amplitude_scale=.7)
    b,other=retracted_real_columns(z,d,lambda v:projection@v,f,p*3j,q*9,amplitude_scale=2.1)
    assert info['retained_real_columns']==4 and info['no_H_calls']
    np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-10)
    for col,source in zip(a.T,np.c_[d,1j*d].T,strict=True):
        _,_,v,_=tangent_system(p,np.zeros_like(p),(.7*f@(projection@source))[:,None],q)
        np.testing.assert_allclose(col,projection@source/np.linalg.norm(v),rtol=2e-10,atol=2e-10)
    assert max(v['half_step_relative_defect'] for v in other['columns'])<2e-10


def test_rank_one_svd_derivative_agrees_with_analytic_tangent_projection():
    z=np.array([2.,0.,0.,0.],complex)
    d=np.array([[.2],[.7j],[.3],[1.2]],complex)
    f=np.eye(4,dtype=complex);p=np.array([2.,.1,.2,.3],complex);q=np.ones(4)
    def retract(v):
        u,s,vh=np.linalg.svd(v.reshape(2,2),full_matrices=False)
        return (u[:,:1]*s[:1])@vh[:1]
    columns,info=retracted_real_columns(z,d,lambda v:retract(v).reshape(-1),f,p,q,amplitude_scale=1.)
    assert info['retained_real_columns']==2
    for col,source in zip(columns.T,np.c_[d,1j*d].T,strict=True):
        source=source.copy();source[-1]=0
        _,_,v,_=tangent_system(p,np.zeros_like(p),source[:,None],q)
        np.testing.assert_allclose(col,source/np.linalg.norm(v),rtol=3e-8,atol=3e-8)


def test_non_smooth_abs_direction_is_rejected_not_called_a_tangent():
    z=np.zeros(2,complex);d=np.array([[1.],[0.]],complex)
    f=np.eye(2,dtype=complex);p=np.ones(2,complex);q=np.ones(2)
    columns,info=retracted_real_columns(z,d,lambda v:np.abs(v).astype(complex),f,p,q,amplitude_scale=1.)
    assert columns.shape==(2,0) and info['retained_real_columns']==0


def test_real_common_coefficients_do_not_introduce_imaginary_parameters():
    p=np.tile(np.array([1.,.25,.1]),2).astype(complex)
    e=np.tile([-1.,1.,2.],2);h=e*p
    t=np.tile(np.array([[0.],[1.],[.5]]),(2,1)).astype(complex);ht=e[:,None]*t
    blocks=[np.arange(6)<3,np.arange(6)>=3]
    c,info=joint_real_coefficients(p,h,t,ht,np.ones(6),blocks,n_sites=3)
    assert c.dtype.kind=='f' and info['common_descent'] and not info['complex_doubling']
    for m in blocks:
        slopes=linear_moment_slopes(p[m],h[m],t[m]@c[:,None],ht[m]@c[:,None],np.ones(3),3)
        assert slopes['energy'][0]<0 and slopes['variance'][0]<0


def test_invalid_probe_and_q_are_rejected():
    z,d,f,p,q=inputs()
    for radius in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):retracted_real_columns(z,d,lambda v:v,f,p,q,amplitude_scale=1.,probe_fs=radius)
    with pytest.raises(ValueError):retracted_real_columns(z,d,lambda v:v,f,p,q*0,amplitude_scale=1.)
