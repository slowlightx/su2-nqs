"""Independent entangled/separable controls, local-charge versus singlet cuts."""
import numpy as np
import pytest
from scipy.linalg import expm

from su2_nqs.benchmarks.spin_sector_entanglement import (
    bipartite_negativity, spin_screening_witness, spin_sector_entanglement,
)
from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns
from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns


def pure(v):
    return np.outer(v, v.conj())


def dimer(pairs):
    """Independent computational construction, zero bit down, MSB site0."""
    v = np.zeros(64, complex)
    for packed in range(64):
        bits = [(packed >> (5-i)) & 1 for i in range(6)]
        if all(bits[a] != bits[b] for a, b in pairs):
            v[packed] = np.prod([1 if bits[a] == 1 else -1 for a, b in pairs])/np.sqrt(8)
    return v


def generators():
    sm = np.array([[0, 1], [0, 0]], complex)
    sx, sy, sz = (sm+sm.T)/2, (sm.T-sm)/(2j), np.diag([-.5, .5])
    result = []
    for op in (sx, sy, sz):
        total = np.zeros((64, 64), complex)
        for site in range(6):
            term = np.array([[1]], complex)
            for i in range(6):
                term = np.kron(term, op if i == site else np.eye(2))
            total += term
        result.append(total)
    return result


def test_partial_transpose_independent_bell_and_product():
    bell = np.array([1., 0., 0., 1.])/np.sqrt(2)
    assert bipartite_negativity(pure(bell), (2, 2)) == pytest.approx(.5, abs=1e-14)
    assert bipartite_negativity(np.eye(4)/4, (2, 2)) == 0


def test_same_joint_spin_weights_different_multiplicity_entanglement():
    c = _three_plus_three_singlet_columns((0, 1, 2))
    a, b = c[:, 0], (c[:, 0]+c[:, 3])/np.sqrt(2)
    ra, rb = [spin_sector_entanglement(pure(v), (0, 1, 2)) for v in (a, b)]
    for report in (ra, rb):
        assert report['sectors']['1,1']['probability'] == pytest.approx(1, abs=2e-14)
        assert report['sectors']['1,1']['singlet_probability'] == pytest.approx(1, abs=2e-14)
        assert report['sectors']['1,1']['screening_witness'] == pytest.approx(.5, abs=2e-14)
    assert ra['sectors']['1,1']['multiplicity_negativity'] == pytest.approx(0, abs=2e-14)
    assert rb['sectors']['1,1']['multiplicity_negativity'] == pytest.approx(.5, abs=2e-14)
    assert ra['raw_negativity'] == pytest.approx(.5, abs=2e-14)
    assert rb['raw_negativity'] == pytest.approx(1.5, abs=2e-14)
    for g in generators():
        assert np.linalg.norm(g@a) < 2e-14
        assert np.linalg.norm(g@b) < 2e-14


def test_quartet_screening_has_no_multiplicity_entanglement():
    v = _three_plus_three_singlet_columns((0, 1, 2))[:, 4]
    r = spin_sector_entanglement(pure(v), (0, 1, 2))
    assert r['raw_negativity'] == pytest.approx(1.5, abs=2e-14)
    assert r['sectors']['3,3']['screening_witness'] == pytest.approx(.75, abs=2e-14)
    assert r['weighted_multiplicity_negativity'] == pytest.approx(0, abs=2e-14)


def test_quartet_pair_probability_is_already_two_and_four_body_information():
    # P(qL=3) = 1/2 + (2/3) sum_{left pairs} Si.Sj. Therefore the
    # witness's subtraction cannot create a new six-body remainder.
    from su2_nqs.benchmarks.spin_partition_readout import _joint_projectors
    def pair(i, j):
        result = np.zeros((64,64))
        for packed in range(64):
            a,b=(packed>>(5-i))&1,(packed>>(5-j))&1
            result[packed,packed]=.25 if a==b else -.25
            if a!=b:result[packed ^ (1<<(5-i)) ^ (1<<(5-j)),packed]=.5
        return result
    left=sum(pair(i,j) for i,j in ((0,1),(0,2),(1,2)))
    right=sum(pair(i,j) for i,j in ((3,4),(3,5),(4,5)))
    low=.25*np.eye(64)+(left+right)/3+4*left@right/9
    exact=sum(p for (ql,qr,qt),p in _joint_projectors(6,(0,1,2)) if ql==qr==3)
    assert np.max(abs(exact-low)) < 2e-14


@pytest.mark.parametrize('pairs,n,mult', [(((0,3),(1,4),(2,5)),3.5,.5),
                                          (((0,1),(2,3),(4,5)),.5,0.)])
def test_dimer_controls_are_not_spin_liquid_witnesses(pairs,n,mult):
    r = spin_sector_entanglement(pure(dimer(pairs)), (0, 1, 2))
    assert r['raw_negativity'] == pytest.approx(n, abs=3e-14)
    assert r['sectors']['1,1']['multiplicity_negativity'] == pytest.approx(mult, abs=3e-14)
    assert r['weighted_local_charge_negativity'] <= r['raw_negativity']+1e-14


def test_separable_states_never_violate_linear_screening_bound():
    rng = np.random.default_rng(82104)
    witnesses = [spin_screening_witness(q, (0,1,2)) for q in (1,3)]
    for _ in range(100):
        a, b = [rng.normal(size=8)+1j*rng.normal(size=8) for _ in range(2)]
        v = np.kron(a/np.linalg.norm(a), b/np.linalg.norm(b))
        assert max(np.vdot(v,w@v).real for w in witnesses) <= 1e-13
    for w in witnesses:
        assert np.linalg.norm(w-w.conj().T) < 2e-14
        for g in generators():
            assert np.linalg.norm(w@g-g@w) < 2e-14


def test_local_schur_gauge_and_site_order_preserve_entanglement():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(64,11))+1j*rng.normal(size=(64,11))
    rho=x@x.conj().T; rho/=np.trace(rho)
    original=spin_sector_entanglement(rho,(0,1,2))
    u,labels=regional_schur_columns(3)
    small=expm(1j*np.array([[.2,.3],[.3,-.1]]))
    gauge=np.eye(8,dtype=complex)
    for m in (-1,1):
        ids=[i for i,l in enumerate(labels) if l[:2]==(1,m)]
        gauge[np.ix_(ids,ids)]=small
    physical=u@gauge@u.conj().T
    action=np.kron(physical,np.eye(8))
    changed=spin_sector_entanglement(action@rho@action.conj().T,(0,1,2))
    for k in ('raw_negativity','weighted_multiplicity_negativity'):
        assert original[k] == pytest.approx(changed[k],abs=2e-14)
    order=(2,0,1,5,3,4)
    bits=(np.arange(64)[:,None] >> np.arange(5,-1,-1))&1
    grouped=bits[:,order]@(2**np.arange(5,-1,-1))
    alternate=rho[np.ix_(np.argsort(grouped),np.argsort(grouped))]
    r=spin_sector_entanglement(alternate,tuple(order.index(i) for i in range(3)))
    assert r['weighted_multiplicity_negativity'] == pytest.approx(
        original['weighted_multiplicity_negativity'],abs=2e-14)


def test_nonlocal_singlet_postselection_is_not_used():
    # This separable state can acquire magnetic entanglement after nonlocal
    # singlet projection; local-charge conditioning must keep negativity zero.
    v=np.zeros(64); v[int('111000',2)]=1
    r=spin_sector_entanglement(pure(v),(0,1,2))
    assert r['raw_negativity'] == 0
    assert r['weighted_multiplicity_negativity'] == 0
    assert r['sectors']['3,3']['screening_witness'] == pytest.approx(0,abs=2e-14)


def test_no_entanglement_swapping_from_nonlocal_singlet_postselection():
    # Each block has a local magnetic--multiplicity Bell pair. The two
    # physical blocks themselves are separable. A NONLOCAL magnetic-singlet
    # measurement would swap entanglement into multiplicity; do not do this.
    u, labels = regional_schur_columns(3)
    i = labels.index((1, -1, 0))
    j = labels.index((1, 1, 1))
    local = (u[:, i]+u[:, j])/np.sqrt(2)
    product = np.kron(local, local)
    original = spin_sector_entanglement(pure(product), (0,1,2))
    assert original['raw_negativity'] == pytest.approx(0, abs=2e-14)
    assert original['weighted_multiplicity_negativity'] == pytest.approx(0, abs=2e-14)
    # Independent S^2 diagonalization, not the readout's joint projector.
    spin_squared = sum(g@g for g in generators())
    values, vectors = np.linalg.eigh(spin_squared)
    singlets = vectors[:, abs(values)<1e-12]
    projected = singlets@(singlets.conj().T@product)
    success = np.vdot(projected, projected).real
    assert success == pytest.approx(.25, abs=2e-14)
    after = spin_sector_entanglement(pure(projected/np.sqrt(success)), (0,1,2))
    assert after['weighted_multiplicity_negativity'] == pytest.approx(.5, abs=2e-14)


@pytest.mark.parametrize('left',[(0,0,1),(0,1),(0,1,6),(False,1,2)])
def test_invalid_cuts(left):
    with pytest.raises(ValueError):
        spin_sector_entanglement(np.eye(64)/64,left)


def test_invalid_input_not_repaired():
    bad=np.eye(64)/64; bad[0,0]=-.1; bad[1,1]+=.1+1/64
    with pytest.raises(ValueError,match='no repair'):
        spin_sector_entanglement(bad,(0,1,2))
    with pytest.raises(ValueError):spin_screening_witness(2,(0,1,2))
    with pytest.raises(ValueError):bipartite_negativity(np.eye(4)/4,(True,4))
