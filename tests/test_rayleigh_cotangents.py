"""Independent finite differences for complex, incomplete-Q physical states."""
import numpy as np
import pytest

from su2_nqs.vmc.rayleigh_cotangents import energy_moment_cotangents


@pytest.mark.parametrize('complete',[False,True])
def test_full_energy_cotangents_at_nodes_gauges_and_incomplete_support(complete):
    rng=np.random.default_rng(1056)
    a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8));h=a+a.conj().T
    p=rng.normal(size=8)+1j*rng.normal(size=8);p[0]=0
    j=np.column_stack((rng.normal(size=(8,4))+1j*rng.normal(size=(8,4)),p,1j*p))
    hp,hj=h@p,h@j
    ids=np.arange(8) if complete else np.array([0,2,4,7])
    q=np.ones(len(ids)) if complete else np.array([.4,1.9,.8,.3])
    wp,wh,info=energy_moment_cotangents(p[ids],hp[ids],q,8)
    observed=np.real(wp@j[ids]+wh@hj[ids])
    def value(x):
        z=p+j@x;hz=h@z
        return (np.sum(z[ids].conj()*hz[ids]/q)/np.sum(abs(z[ids])**2/q)).real/8
    fd=np.array([(value(np.eye(j.shape[1])[k]*1e-6)-value(-np.eye(j.shape[1])[k]*1e-6))/2e-6 for k in range(j.shape[1])])
    np.testing.assert_allclose(observed,fd,atol=2e-10,rtol=2e-9)
    np.testing.assert_allclose(observed[-2:],0,atol=2e-14)
    assert info['exact_node_records']==1 and info['complete_two_sided_derivative']
    for scale in (.1,7.):
        x,y,_=energy_moment_cotangents(p[ids],hp[ids],scale*q,8)
        np.testing.assert_allclose(x,wp,atol=2e-15);np.testing.assert_allclose(y,wh,atol=2e-15)
    if not complete:
        assert abs(info['imaginary_energy_per_site'])>.01
        # The population 2-forward replacement is NOT this finite-Q derivative.
        norm=np.sum(abs(p[ids])**2/q)
        e=np.sum(p[ids].conj()*hp[ids]/q)/norm
        population=2*np.real(j[ids].conj().T@((hp[ids]-e*p[ids])/q))/norm/8
        assert np.linalg.norm(population-observed)>1e-3


def test_bad_density_and_per_site_contracts_rejected():
    with pytest.raises(ValueError):energy_moment_cotangents(np.ones(3),np.ones(3),np.zeros(3),3)
    with pytest.raises(ValueError):energy_moment_cotangents(np.ones(3),np.ones(3),np.ones(3),True)
