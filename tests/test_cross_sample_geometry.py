"""Dense exact oracles distinguish a population force from an empirical ratio."""
import numpy as np
import pytest

from su2_nqs.vmc.cross_sample_geometry import hermitian_energy_force,equal_mass_geometry,observable_common_direction
from su2_nqs.vmc.full_rayleigh import full_rayleigh_gradient,reverse_rayleigh_weights
from su2_nqs.vmc.importance_tangent import tangent_system
from su2_nqs.vmc.joint_moments import normalized_common_moment_direction


@pytest.mark.parametrize('complete',[True,False])
def test_population_force_exact_hermitian_oracle_nodes_gauge_and_Q(complete):
    rng=np.random.default_rng(301);a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8));h=a+a.conj().T
    p=rng.normal(size=8)+1j*rng.normal(size=8);p[0]=0
    j=rng.normal(size=(8,3))+1j*rng.normal(size=(8,3));j=np.column_stack((j,p,1j*p))
    ids=np.arange(8) if complete else np.array([0,2,4,7])
    q=np.ones(len(ids)) if complete else np.array([.5,1.2,.7,2.])
    hp,hj=(h@p)[ids],(h@j)[ids];x,y=p[ids],j[ids]
    force=hermitian_energy_force(x,hp,y,q,8)
    norm=np.sum(abs(x)**2/q);e=np.sum(x.conj()*hp/q)/norm
    beta=np.sum(x.conj()[:,None]*y/q[:,None],axis=0)/norm
    oracle=2*np.real((y-x[:,None]*beta).conj().T@((hp-e*x)/q))/norm/8
    np.testing.assert_allclose(force,oracle,atol=3e-14)
    finite=full_rayleigh_gradient(x,hp,y,q,np.real(reverse_rayleigh_weights(x,q)@hj))['gradient']/8
    if complete:np.testing.assert_allclose(force,finite,atol=3e-14)
    else:assert np.linalg.norm(force-finite)>.01 # NEVER assert the wrong incomplete-Q identity.
    np.testing.assert_allclose(force[-2:],0,atol=3e-14)
    np.testing.assert_allclose(hermitian_energy_force(x,hp+3.1*x,y,11*q,8),force,atol=3e-14)


def test_balanced_geometry_keeps_between_block_phase_and_unknown_normalizers():
    p=[np.ones(3),np.ones(5)];j=[1j*np.ones((3,1)),-1j*np.ones((5,1))];q=[np.ones(3),2*np.ones(5)]
    x,y,density=equal_mass_geometry(p,j,q)
    a,_,_,_=tangent_system(x,np.zeros_like(x),y,density)
    assert np.sum(abs(x)**2/density)==pytest.approx(1.)
    assert (a.T@a)[0,0]==pytest.approx(2.) # Separate gauges would incorrectly erase this direction.
    for factor in ([.03,13.],[9.,.2]):
        x2,y2,q2=equal_mass_geometry(p,j,[v*f for v,f in zip(q,factor)])
        np.testing.assert_allclose(q2,density,atol=2e-14)


def test_observable_update_removes_unmeasured_axis_explicitly_and_matches_dense_metric():
    p=np.array([1.,0,0]);j=np.array([[0,0,0],[1.,0,0],[0,1e-4,0]])
    g=np.array([[1.,.2,20.],[2.,.1,30.]])
    d,info=observable_common_direction(p,j,np.ones(3),g,['E','R'],relative_cutoff=1e-6)
    assert info['common_descent'] and info['resolved_rank']==1 and info['optimizer_range_only']
    np.testing.assert_allclose(d,[1.,0,0],atol=2e-13)
    assert min(info['gradient_discarded_fractions'])>.99
    # With a full-rank metric, the resolved solver reduces to the original one.
    j=np.array([[0,0],[1.,0],[0,2.]])
    g=np.array([[1.,.3],[.5,1.]])
    d,r=observable_common_direction(p,j,np.ones(3),g,['E','R'],relative_cutoff=1e-9)
    original,_=normalized_common_moment_direction(p,np.zeros(3),j,np.ones(3),g,['E','R'])
    np.testing.assert_allclose(d,original,atol=3e-13)
    assert r['mode_orthogonality_error']<1e-12
    rescaled,_=observable_common_direction(p,j,np.ones(3),g*np.array([.04,18.])[:,None],['E','R'])
    np.testing.assert_allclose(rescaled,d,atol=3e-13)


def test_observable_geometry_is_independent_of_forces_and_gauges_and_cones_stop():
    rng=np.random.default_rng(311);p=rng.normal(size=8)+1j*rng.normal(size=8)
    j=rng.normal(size=(8,3))+1j*rng.normal(size=(8,3));q=np.linspace(.4,1.8,8)
    g=np.array([[1.,.4,.2],[.2,1.,.3]])
    d,info=observable_common_direction(p,j,q,g,['E','R'])
    gauges=np.array([.2+.3j,-1j,.4])
    d2,r=observable_common_direction(p,j+p[:,None]*gauges,7*q,g,['E','R'])
    np.testing.assert_allclose(d2,d,atol=3e-12)
    assert r['resolved_rank']==info['resolved_rank']
    zero,stop=observable_common_direction(p,j,q,np.array([[1.,0,0],[-1.,0,0]]),['E','R'])
    assert not stop['common_descent'];np.testing.assert_allclose(zero,0,atol=2e-13)


def test_invalid_force_geometry_and_cutoff_contracts():
    p=np.ones(3);j=np.ones((3,2));q=np.ones(3)
    with pytest.raises(ValueError):hermitian_energy_force(p,p,j,q,True)
    with pytest.raises(ValueError):equal_mass_geometry([],[],[])
    with pytest.raises(ValueError):equal_mass_geometry([p,p],[j,j[:,:1]],[q,q])
    with pytest.raises(ValueError):observable_common_direction(p,j,q,np.ones((2,2)),['E','R'],relative_cutoff=0)
    with pytest.raises(ValueError):observable_common_direction(p,j,q*0,np.ones((2,2)),['E','R'])
