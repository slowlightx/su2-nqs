"""Independent finite references and explicit non-QSL false positives."""
from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from su2_nqs.benchmarks.spin_rotation_calibration import (
    aklt_local_tensors, boundary_even_charge_rotation, kagome_clockwise_edges, kagome_rvb_local_tensors,
    kagome_rvb_reference, lieb_mattis_singlet, manifold_region_algebra,
    pure_state_rdm, singlet_covering_state,
)
from su2_nqs.benchmarks.spin_rotation_disorder import regional_spin_rotation


def _raising(state):
    n = len(state).bit_length()-1
    out = np.zeros_like(state)
    for packed in range(len(state)):
        for i in range(n):
            if not (packed >> i) & 1:
                out[packed | (1 << i)] += state[packed]
    return out


def _literal_simplex_amplitude(rows, columns, bits):
    # An independent contraction: choose one of the two incident triangles
    # for each site, rather than enumerate singlet pairs/matchings.
    n = 3*rows*columns
    triangles = []
    index = lambda r,c,s: 3*((r % rows)*columns+c % columns)+s
    for r,c in product(range(rows),range(columns)):
        triangles.extend([(index(r,c,0),index(r,c,1),index(r,c,2)),
                          (index(r,c,0),index(r,c-1,1),index(r-1,c,2))])
    incident = [[] for _ in range(n)]
    for t, tri in enumerate(triangles):
        for leg, site in enumerate(tri):
            incident[site].append((t,leg))
    epsilon = np.zeros((3,3,3))
    epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1
    epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1
    epsilon[2,2,2] = 1
    assignments = ((np.arange(2**n)[:,None] >> np.arange(n)) & 1)
    virtual = np.full((2**n,len(triangles),3),2,dtype=int)
    for i in range(n):
        for k,(t,leg) in enumerate(incident[i]):
            virtual[assignments[:,i]==k,t,leg] = bits[i]
    values = epsilon[virtual[:,:,0],virtual[:,:,1],virtual[:,:,2]]
    return values.prod(axis=1).sum()


def test_simplex_full_SU2_pullthrough_and_charge_center():
    data = kagome_rvb_local_tensors()
    p, epsilon, J = data['site'], data['epsilon'], data['J']
    sx = np.array([[0,.5],[.5,0]])
    sy = np.array([[0,.5j],[-.5j,0]])  # down/up magnetic ordering
    sz = np.diag([-.5,.5])
    for axis in (sx,sy,sz,.2*sx-.3*sy+.4*sz):
        u = expm(1j*.73*axis)
        w = np.eye(3,dtype=complex); w[:2,:2] = u
        np.testing.assert_allclose(u@p.reshape(2,9),p.reshape(2,9)@np.kron(w,w),atol=2e-14,rtol=0)
        moved = np.einsum('ia,jb,kc,abc->ijk',w,w,w,epsilon)
        np.testing.assert_allclose(moved,epsilon,atol=2e-14,rtol=0)
    w_2pi = np.eye(3,dtype=complex); w_2pi[:2,:2] = expm(2j*np.pi*sz)
    np.testing.assert_allclose(w_2pi,J,atol=2e-14,rtol=0)
    rx = np.eye(3,dtype=complex); rz = np.eye(3,dtype=complex)
    rx[:2,:2] = expm(1j*np.pi*sx); rz[:2,:2] = expm(1j*np.pi*sz)
    np.testing.assert_allclose(rx@rz@rx.conj().T@rz.conj().T,J,atol=2e-14,rtol=0)
    # AKLT has exactly the same half-spin virtual sign but is an SPT, not
    # an intrinsic 2D topologically ordered spin liquid.
    np.testing.assert_allclose((rx@rz@rx.conj().T@rz.conj().T)[:2,:2],-np.eye(2),atol=2e-14,rtol=0)


def test_clockwise_covering_matches_CS_and_is_short_range_entangled():
    state = singlet_covering_state(4,[(0,2),(3,1)])
    assert np.vdot(state,state).real == pytest.approx(1,abs=2e-14)
    assert np.linalg.norm(_raising(state)) < 2e-14
    angles = np.array([0,.31,np.pi,2*np.pi])
    crossing = regional_spin_rotation(pure_state_rdm(state,[0,1]),angles)
    np.testing.assert_allclose(crossing['z'],np.cos(angles/2)**2,atol=2e-14,rtol=0)
    internal = regional_spin_rotation(pure_state_rdm(state,[0,2]),angles)
    np.testing.assert_allclose(internal['z'],1,atol=2e-14,rtol=0)


def test_actual_AKLT_intertwiner_shares_half_spin_virtual_sign_without_QSL():
    data=aklt_local_tensors()
    p=data['site'].reshape(3,4)
    raising=np.diag([np.sqrt(2),np.sqrt(2)],k=-1)
    s1=[(raising+raising.T)/2,(raising-raising.T)/(2j),np.diag([-1.,0.,1.])]
    raising_half=np.array([[0.,0.],[1.,0.]])
    sh=[(raising_half+raising_half.T)/2,(raising_half-raising_half.T)/(2j),np.diag([-.5,.5])]
    for axis in (np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([.2,-.3,.4])):
        u=expm(.71j*sum(a*s for a,s in zip(axis,s1)))
        v=expm(.71j*sum(a*s for a,s in zip(axis,sh)))
        np.testing.assert_allclose(u@p,p@np.kron(v,v),atol=2e-14,rtol=0)
        bond=data['singlet_bond']
        np.testing.assert_allclose(v@bond@v.T,bond,atol=2e-14,rtol=0)
    rx=expm(1j*np.pi*sh[0]); rz=expm(1j*np.pi*sh[2])
    np.testing.assert_allclose(rx@rz@rx.conj().T@rz.conj().T,-np.eye(2),atol=2e-14,rtol=0)
    np.testing.assert_allclose(expm(2j*np.pi*s1[2]),np.eye(3),atol=2e-14,rtol=0)


def test_twelve_site_RVB_counts_SU2_and_literal_PEPS_contraction():
    data = kagome_rvb_reference()
    assert len(data['covers']) == 32
    np.testing.assert_array_equal(data['covering_counts'],[8,8,8,8])
    degrees = np.zeros(12,dtype=int)
    for i,j,_,_ in data['edges']:
        degrees[i] += 1; degrees[j] += 1
    np.testing.assert_array_equal(degrees,4)
    assert np.linalg.matrix_rank(data['gram'],tol=1e-10) == 4
    for state in data['states'].T:
        assert np.linalg.norm(_raising(state)) < 2e-13
    rng = np.random.default_rng(2026101201)
    packed = rng.choice(np.arange(4096),size=19,replace=False)
    for k in packed:
        bits = (int(k) >> np.arange(11,-1,-1)) & 1
        oracle = _literal_simplex_amplitude(2,2,bits)
        np.testing.assert_allclose(data['total_raw_state'][k]*np.sqrt(2)**6,oracle,atol=2e-13,rtol=0)
    assert not data['intrinsic_topology_certified'] and not data['cusp_claim']


def test_ordered_singlet_is_invariant_but_has_long_range_spin_correlations():
    state = lieb_mattis_singlet(8,[0,2,4,6])
    assert np.vdot(state,state).real == pytest.approx(1,abs=2e-14)
    assert np.linalg.norm(_raising(state)) < 2e-14
    rho = pure_state_rdm(state,[0,6])
    sz = np.diag([-.5,.5]); sx = np.array([[0,.5],[.5,0]])
    sy = np.array([[0,.5j],[-.5j,0]])
    correlator = sum(np.kron(s,s) for s in (sx,sy,sz))
    assert np.trace(rho@correlator).real == pytest.approx(.25,abs=2e-14)
    readout = regional_spin_rotation(rho,np.linspace(0,2*np.pi,9))
    assert readout['factorization_frobenius'] < 2e-14


def test_local_algebra_keeps_offdiagonal_transitions_and_actual_Gram():
    # Bell +/- have identical one-site diagonal RDMs; offdiagonal local
    # transitions expose distinguishability of their coherent state span.
    states = np.array([[1,1],[0,0],[0,0],[1,-1]],complex)/np.sqrt(2)
    report = manifold_region_algebra(states,[0])
    np.testing.assert_allclose(report['transition_rdms'][0,0],report['average_rdm'],atol=2e-14,rtol=0)
    np.testing.assert_allclose(report['transition_rdms'][1,1],report['average_rdm'],atol=2e-14,rtol=0)
    assert report['local_algebra_defect_frobenius'] == pytest.approx(1.,abs=2e-14)
    mixed = states@np.array([[1,.3],[.2,1.1]])
    transformed = manifold_region_algebra(mixed,[0])
    assert transformed['orthonormality_error'] < 2e-14
    assert transformed['local_algebra_defect_frobenius'] == pytest.approx(1.,abs=2e-14)


@pytest.mark.parametrize('length',[1,2,4,5])
def test_boundary_formula_against_literal_virtual_charge_distribution(length):
    p = .65
    angles = np.array([0,.27,np.pi,1.37*np.pi,2*np.pi])
    labels = np.array(list(product((0,1,2),repeat=length)),dtype=int)
    local_prob = np.array([(1-p)/2,(1-p)/2,p])
    weights = local_prob[labels].prod(axis=1)
    charge = (labels != 2).sum(axis=1)
    weights[charge % 2 == 1] = 0
    weights /= weights.sum()
    doubled_m = np.array([-1,1,0])[labels].sum(axis=1)
    exact = weights@np.exp(.5j*doubled_m[:,None]*angles)
    report = boundary_even_charge_rotation(length,p,angles)
    np.testing.assert_allclose(report['z'],exact,atol=3e-14,rtol=0)
    assert report['z'][-1] == pytest.approx(1,abs=3e-14)
    assert not report['known_bulk_positive'] and not report['physical_spin_region']


def test_finite_boundary_smoothness_vs_conditional_asymptotic_cusp():
    angles = np.array([np.pi-.001,np.pi,np.pi+.001])
    report = boundary_even_charge_rotation(64,.65,angles)
    np.testing.assert_allclose(report['rate'][0],report['rate'][2],atol=2e-14,rtol=0)
    assert report['rate'][1] < report['asymptotic_rate'][1]
    assert report['asymptotic_rate'][1] == pytest.approx(-np.log(.65),abs=2e-14)
    # Finite-L analytic curves cannot have a first-derivative cusp at pi.
    near = boundary_even_charge_rotation(64,.65,np.array([np.pi-1e-5,np.pi,np.pi+1e-5]))
    assert (near['rate'][1]-near['rate'][0])/1e-5 < .001


@pytest.mark.parametrize('bad',[True,1,4])
def test_bad_kagome_geometry_rejected(bad):
    with pytest.raises(ValueError):
        kagome_clockwise_edges(bad,2)


def test_bad_states_and_boundary_labels_rejected():
    with pytest.raises(ValueError):
        singlet_covering_state(4,[(0,1),(1,2)])
    with pytest.raises(ValueError):
        pure_state_rdm(np.ones(4),[0])
    with pytest.raises(ValueError):
        lieb_mattis_singlet(4,[0,0])
    with pytest.raises(ValueError):
        boundary_even_charge_rotation(4,.65,np.array([1j]))
