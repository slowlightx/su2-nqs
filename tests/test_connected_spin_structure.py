"""Independent quantum and ordered product-state checks of connected S(k)."""
import numpy as np
import pytest
from su2_nqs.benchmarks.connected_spin_structure import (
    connected_spin_structure_factors,connected_longitudinal_structure_factors,
)


def test_broken_120_product_removes_elastic_order_retains_quantum_self_variance():
    angles=2*np.pi*np.arange(3)/3
    m=.5*np.column_stack((np.cos(angles),np.sin(angles),np.zeros(3)))
    c=m@m.T;np.fill_diagonal(c,.75)
    coords=np.column_stack((np.arange(3),np.zeros(3)))
    ks={'Gamma':(0,0),'K':(1/3,0)}
    phase=np.exp(2j*np.pi*coords@np.array(list(ks.values())).T)
    raw=np.array([np.vdot(p,c@p).real/3 for p in phase.T])
    read=connected_spin_structure_factors(raw,m,coords,ks)
    np.testing.assert_allclose(read['connected'],.5,atol=1e-14)
    np.testing.assert_allclose(read['connected_offsite'],0,atol=1e-14)
    assert read['elastic'][1]>.3
    assert read['onsite_quantum_variance']==pytest.approx(.5)


def test_singlet_is_connected_and_offsite_can_be_negative():
    # Exact two-spin singlet <Si.Sj>=-3/4, <Si>=0.
    raw=np.array([0.,1.5]);m=np.zeros((2,3));coords=np.array([[0,0],[1,0]])
    ks={'Gamma':(0,0),'pi':(.5,0)}
    read=connected_spin_structure_factors(raw,m,coords,ks)
    np.testing.assert_array_equal(read['connected'],raw)
    np.testing.assert_array_equal(read['connected_offsite'],[-.75,.75])
    zz=connected_longitudinal_structure_factors(raw/3,m,coords,ks)
    np.testing.assert_array_equal(zz['connected_offsite'],[-.25,.25])


def test_general_complex_state_matches_direct_centered_spin_operator_norm():
    sx=np.array([[0,.5],[.5,0]]);sy=np.array([[0,.5j],[-.5j,0]]);sz=np.diag([-.5,.5])
    rng=np.random.default_rng(140026);psi=rng.normal(size=8)+1j*rng.normal(size=8);psi/=np.linalg.norm(psi)
    ops=[]
    for i in range(3):
        site=[]
        for op in (sx,sy,sz):
            a=np.ones((1,1))
            for j in range(3):a=np.kron(a,op if i==j else np.eye(2))
            site.append(a)
        ops.append(site)
    m=np.array([[np.vdot(psi,a@psi).real for a in site] for site in ops])
    coords=np.array([[0,0],[1,0],[0,1]]);ks={'a':(.25,.125),'b':(0,0)}
    raw=[];expected=[];zz=[];expected_z=[]
    for k in ks.values():
        phase=np.exp(2j*np.pi*coords@np.array(k));norm=[];centered=[]
        for alpha in range(3):
            o=sum(phase[i]*ops[i][alpha] for i in range(3))/np.sqrt(3)
            action=o@psi;mean=np.vdot(psi,action)
            norm.append(np.vdot(action,action).real)
            centered.append(np.linalg.norm(action-mean*psi)**2)
        raw.append(sum(norm));expected.append(sum(centered));zz.append(norm[2]);expected_z.append(centered[2])
    read=connected_spin_structure_factors(np.array(raw),m,coords,ks)
    zread=connected_longitudinal_structure_factors(np.array(zz),m,coords,ks)
    np.testing.assert_allclose(read['connected'],expected,atol=2e-14,rtol=0)
    np.testing.assert_allclose(zread['connected'],expected_z,atol=2e-14,rtol=0)
    with pytest.raises(ValueError):connected_spin_structure_factors(np.array(raw),m+1,coords,ks)
