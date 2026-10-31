"""Independent dense-H population variance oracle and incomplete-Q countercase."""
import numpy as np
import pytest
from su2_nqs.vmc.population_moment_forces import population_moment_forces,imaginary_time_energy_certificate
from su2_nqs.vmc.joint_moments import residual_moment_cotangents


@pytest.mark.parametrize('complete',[True,False])
def test_H2_population_variance_force_oracle_gauge_shift_nodes(complete):
    rng=np.random.default_rng(3031);a=rng.normal(size=(10,10))+1j*rng.normal(size=(10,10));h=a+a.conj().T
    p=rng.normal(size=10)+1j*rng.normal(size=10);p[0]=0
    j=np.column_stack((rng.normal(size=(10,3))+1j*rng.normal(size=(10,3)),p,1j*p))
    ids=np.arange(10) if complete else np.array([0,1,3,7,9]);q=np.ones(len(ids)) if complete else np.linspace(.4,2,len(ids))
    x,hp,h2,y=p[ids],(h@p)[ids],(h@h@p)[ids],j[ids]
    ge,gr,info=population_moment_forces(x,hp,h2,y,q,10)
    z=np.sum(abs(x)**2/q);e=np.real(np.sum(x.conj()*hp/q)/z)
    centered=y-x[:,None]*np.sum(x.conj()[:,None]*y/q[:,None],axis=0)/z
    oracle=2*np.real(centered.conj().T@((h2-2*e*hp)/q))/z/10
    np.testing.assert_allclose(gr,oracle,atol=4e-13)
    wp,wh,_=residual_moment_cotangents(x,hp,q,10)
    finite=np.real(wp@y+wh@(h@j)[ids])
    if complete:
        np.testing.assert_allclose(gr,finite,atol=4e-13)
        assert info['second_moment_quadrature_defect']<1e-12
        eps=1e-6
        def variance(v):
            e=np.vdot(v,h@v)/np.vdot(v,v)
            return (np.vdot(h@v,h@v)/np.vdot(v,v)-abs(e)**2).real/10
        fd=np.array([(variance(p+eps*j[:,k])-variance(p-eps*j[:,k]))/(2*eps) for k in range(j.shape[1])])
        np.testing.assert_allclose(gr,fd,atol=2e-8)
    else:
        assert np.linalg.norm(gr-finite)>.01
        assert info['second_moment_quadrature_defect']>1
    np.testing.assert_allclose(gr[-2:],0,atol=3e-13)
    shifted=population_moment_forces(3j*x,3j*(hp+2.3*x),3j*(h2+4.6*hp+2.3**2*x),3j*y,7*q,10)
    np.testing.assert_allclose(shifted[0],ge,atol=4e-13);np.testing.assert_allclose(shifted[1],gr,atol=4e-13)
    with pytest.raises(ValueError):population_moment_forces(x,hp,h2[:-1],y,q,10)


def test_spectral_filter_bound_for_arbitrary_probability_and_empirical_anchor():
    rng=np.random.default_rng(3051);js=np.array([1.,.125,-.3]);bound=.75*abs(js).sum()
    energies=np.linspace(-bound,bound,12);weights=rng.random(12);weights/=weights.sum()
    for anchor in (-10.,-.3,0.,2.,10.):
        cert=imaginary_time_energy_certificate(js,anchor);beta=cert['maximum_beta']
        factor=1+beta*(anchor-energies)
        assert factor.min()>=.5 and np.all(np.diff(factor**2)<0)
        after=np.sum(energies*weights*factor**2)/np.sum(weights*factor**2)
        assert after<np.sum(energies*weights)
        assert not cert['variance_nonincrease_proved']
