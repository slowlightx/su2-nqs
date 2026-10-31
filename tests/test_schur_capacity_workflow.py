"""Deterministic complete-block geometry and exact finite-Q energy checks."""
import numpy as np
from projects.clebsch_tree_nqs.workflows.spin_schur_capacity import search_directions,moment


def test_whole_block_directions_have_independent_unit_physical_norm_and_descend_exact_target():
    rng=np.random.default_rng(9481);m,k=40,9
    f=rng.normal(size=(m,k))+1j*rng.normal(size=(m,k))
    p=rng.normal(size=m)+1j*rng.normal(size=m)
    h=rng.normal(size=(m,m))+1j*rng.normal(size=(m,m));h=(h+h.conj().T)/2
    hp=h@p;q=np.ones(m)
    directions,info=search_directions(f,p,hp,q,.7,dict(metric_cutoff=1e-6,metric_damping=.1))
    t=.7*f@directions;norm=np.vdot(p,p).real
    projected=t-p[:,None]*(p.conj()@t/norm)[None]
    np.testing.assert_allclose(projected.conj().T@projected/norm,np.eye(2),atol=3e-14)
    energy=np.vdot(p,hp).real/norm
    slope=2*np.vdot(projected[:,0],hp-energy*p).real/norm
    assert slope < 0 and info['rank']==k
    assert not info['stationarity_certified']


def test_zero_projective_block_is_not_a_stationarity_certificate():
    p=np.ones(8,dtype=complex);f=np.tile(p[:,None],(1,5))
    d,info=search_directions(f,p,p,np.ones(8),1.,dict(metric_cutoff=1e-6,metric_damping=.1))
    assert d.shape==(5,0) and info['rank']==0


def test_node_safe_moments_include_h_at_nodes_and_keep_imaginary_energy():
    p=np.array([1,0,2j]);hp=np.array([2+.1j,3,5j]);q=np.array([1,2,3])
    row=moment(p,hp,q);norm=np.sum(abs(p)**2/q);e=np.sum(p.conj()*hp/q)/norm
    np.testing.assert_allclose(row['energy_per_site'],e.real/36)
    assert row['imaginary_energy_per_site']!=0
    np.testing.assert_allclose(row['variance_per_site'],np.sum(abs(hp-e*p)**2/q)/norm/36)
