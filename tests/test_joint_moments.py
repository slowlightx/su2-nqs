"""Independent exact H/FD oracles, complex finite-Q E, nodes and common cones."""
import numpy as np
import pytest
from su2_nqs.vmc.joint_moments import residual_moment_cotangents,normalized_common_moment_direction


@pytest.mark.parametrize('complete',[False,True])
def test_residual_gradient_is_complete_at_nodes_and_complex_energy(complete):
    rng=np.random.default_rng(76);a=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7));h=a+a.conj().T
    p=rng.normal(size=7)+1j*rng.normal(size=7);p[0]=0
    j=rng.normal(size=(7,3))+1j*rng.normal(size=(7,3));j=np.column_stack((j,p,1j*p))
    hp,hj=h@p,h@j;keep=np.arange(7) if complete else np.array([0,2,4,6])
    q=np.ones(len(keep)) if complete else np.array([.3,2.,.7,1.1])
    wp,wh,info=residual_moment_cotangents(p[keep],hp[keep],q,7)
    gradient=np.real(wp@j[keep]+wh@hj[keep])
    def variance(x):
        z=p+j@x;hz=h@z;n=np.sum(abs(z[keep])**2/q)
        e=np.sum(z[keep].conj()*hz[keep]/q)/n
        return np.sum(abs(hz[keep]-e*z[keep])**2/q)/n/7
    observed=[]
    for c in range(j.shape[1]):
        step=np.eye(j.shape[1])[c]*1e-6
        observed.append((variance(step)-variance(-step))/2e-6)
    np.testing.assert_allclose(gradient,observed,atol=3e-9,rtol=3e-9)
    np.testing.assert_allclose(gradient[-2:],0,atol=3e-14)
    assert info['exact_parent_node_records']==1 and info['orthogonality_scaled']<1e-13
    if not complete:assert abs(info['energy_imaginary'])>.01
    assert info['residual_moment_matches_variance']
    for scale in (.03,7.3):
        a,b,_=residual_moment_cotangents(p[keep],hp[keep],scale*q,7)
        np.testing.assert_allclose(a,wp,atol=2e-14);np.testing.assert_allclose(b,wh,atol=2e-14)


def test_six_objective_common_direction_is_unit_invariant_and_FS_normalized():
    p,hp,j,q=np.array([1.,0.,0.]),np.zeros(3),np.array([[0.,0.],[1.,0.],[0.,1.]]),np.ones(3)
    gs=np.array([[1.,0.],[-.2,3.],[1.,2.],[2.,.4],[-.1,2.],[.1,1.]])
    names=[str(i) for i in range(6)]
    d,info=normalized_common_moment_direction(p,hp,j,q,gs,names)
    assert info['common_descent'] and np.all(gs@d>0)
    assert np.linalg.norm(d)==pytest.approx(1.)
    assert info['simplex_kkt_scaled_violation']<1e-12
    # Changing energy/variance units must not change the physical direction.
    ds,other=normalized_common_moment_direction(p,hp,j,q,gs*np.array([.01,7,2,.3,10,100])[:,None],names)
    np.testing.assert_allclose(ds,d,atol=2e-13)
    assert other['common_descent']
    zero,stop=normalized_common_moment_direction(p,hp,j,q,np.array([[1.,0.],[-1.,0.]]),['E','R'])
    np.testing.assert_allclose(zero,0,atol=1e-14);assert not stop['common_descent']
    stationary,result=normalized_common_moment_direction(p,hp,j,q,np.array([[1.,0.],[0.,0.]]),['E','R'])
    np.testing.assert_array_equal(stationary,0);assert not result['common_descent']


def test_invalid_residual_contracts_and_objective_counts_fail():
    p,hp,j,q=np.ones(2),np.ones(2),np.ones((2,1)),np.ones(2)
    with pytest.raises(ValueError):residual_moment_cotangents(p,hp,q*0,2)
    with pytest.raises(ValueError):residual_moment_cotangents(p,hp,q,True)
    with pytest.raises(ValueError):normalized_common_moment_direction(p,hp,j,q,np.ones((7,1)),list('abcdefg'))
