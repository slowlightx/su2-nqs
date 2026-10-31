"""Independent bit-basis rotations and trivial-state false-positive controls."""
import numpy as np
import pytest
from scipy.linalg import expm

from su2_nqs.benchmarks.spin_rotation_disorder import (
    regional_spin_rotation, rotation_characteristic, sampled_spin_rotation,
    sampled_translation_rotation,
)


def generator(n, single):
    result = np.zeros((2**n, 2**n), complex)
    for site in range(n):
        term = np.array([[1]], complex)
        for i in range(n):
            term = np.kron(term, single if i == site else np.eye(2))
        result += term
    return result


@pytest.mark.parametrize('n', [1, 2, 4])
def test_actual_axes_against_independent_exponentials(n):
    rng = np.random.default_rng(260612132+n)
    v = rng.normal(size=(2**n, 5))+1j*rng.normal(size=(2**n, 5))
    rho = v@v.conj().T; rho /= np.trace(rho)
    angles = np.array([0., .31, np.pi, 2*np.pi])
    report = regional_spin_rotation(rho, angles)
    for axis, single in [('z', np.diag([-.5, .5])), ('x', np.array([[0., .5], [.5, 0.]]))]:
        g = generator(n, single)
        oracle = [np.trace(rho@expm(1j*t*g)) for t in angles]
        np.testing.assert_allclose(report[axis], oracle, atol=2e-13, rtol=0)
    assert report['z'][-1] == pytest.approx((-1)**n, abs=2e-13)
    assert report['scalar'][-1] == pytest.approx((-1)**n, abs=2e-13)
    assert not report['input_repaired'] and not report['QSL_claim']


def test_singlet_character_identity_and_non_SU2_difference():
    singlet = np.array([0., -1., 1., 0.])/np.sqrt(2)
    rho = .7*np.outer(singlet, singlet)+.3*np.eye(4)/4
    angles = np.linspace(0, 2*np.pi, 65)
    report = regional_spin_rotation(rho, angles)
    np.testing.assert_allclose(report['z'], report['scalar'], atol=2e-13, rtol=0)
    assert report['factorization_frobenius'] < 2e-13
    product = np.diag([0., 1., 0., 0.])
    broken = regional_spin_rotation(product, angles)
    assert broken['axis_scalar_max_difference'] > .5
    assert broken['factorization_frobenius'] > .1


def test_axis_check_does_not_hide_offdiagonal_symmetry_defect():
    rho = np.eye(4)/4
    rho[0, 3] = rho[3, 0] = .1
    report = regional_spin_rotation(rho, np.array([0., np.pi]))
    np.testing.assert_allclose(report['z'], report['scalar'], atol=2e-14, rtol=0)
    assert report['axis_scalar_max_difference'] == pytest.approx(.2, abs=2e-14)


def test_invariant_odd_region_has_a_forced_pi_zero():
    # Any rotationally invariant odd region contains only half-integer q/2.
    rng = np.random.default_rng(260612132)
    rho = np.eye(8)/8
    report = regional_spin_rotation(rho, np.array([np.pi]))
    assert abs(report['scalar'][0]) < 2e-14
    # An arbitrary mixture of q=1 and q=3 characters is also identically zero.
    probabilities = rng.dirichlet([1., 1.])
    value = sum(p*rotation_characteristic(np.arange(-q, q+1, 2),
                np.full(q+1, 1/(q+1)), np.array([np.pi]))[0]
                for p, q in zip(probabilities, (1, 3)))
    assert abs(value) < 2e-14


def test_crossing_dimers_have_pi_zeros_without_topology():
    # Three A spins, each paired with an exterior spin: rho_A=I/8.
    angles = np.array([0., .7, np.pi, 2*np.pi])
    report = regional_spin_rotation(np.eye(8)/8, angles)
    np.testing.assert_allclose(report['z'], np.cos(angles/2)**3, atol=2e-13, rtol=0)
    assert abs(report['z'][2]) < 2e-13
    assert not report['QSL_claim']


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_physical_pi_rotation_commutator_is_kinematic_not_spinon(n):
    ux = expm(1j*np.pi*generator(n, np.array([[0., .5], [.5, 0.]])))
    uz = expm(1j*np.pi*generator(n, np.diag([-.5, .5])))
    commutator = ux@uz@ux.conj().T@uz.conj().T
    np.testing.assert_allclose(commutator, (-1)**n*np.eye(2**n), atol=2e-13, rtol=0)


def test_sampled_means_keep_chain_covariance_and_bit_convention():
    bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    groups = np.array([2, 2, 7, 7])
    angles = np.array([0., .5, np.pi, 2*np.pi])
    report = sampled_spin_rotation(bits, groups, [0, 1], angles)
    explicit = np.exp(1j*((bits-.5).sum(1))[:, None]*angles[None, :])
    chains = np.array([explicit[:2].mean(0), explicit[2:].mean(0)])
    np.testing.assert_allclose(report['chain_means'], chains, atol=2e-14, rtol=0)
    np.testing.assert_allclose(report['value'], chains.mean(0), atol=2e-14, rtol=0)
    np.testing.assert_allclose(report['real_chain_SE'], chains.real.std(0, ddof=1)/np.sqrt(2), atol=2e-14, rtol=0)
    assert report['new_samples'] == report['new_H_calls'] == 0


@pytest.mark.parametrize('bad', [np.diag([-1., 2.]), np.eye(3)/3, np.eye(2)])
def test_invalid_density_not_repaired(bad):
    with pytest.raises(ValueError):
        regional_spin_rotation(bad, np.array([0., np.pi]))


def test_invalid_labels_and_regions():
    with pytest.raises(ValueError):
        rotation_characteristic(np.array([-.5, .5]), np.array([.5, .5]), np.array([1.]))
    with pytest.raises(ValueError):
        sampled_spin_rotation(np.zeros((3, 4)), np.arange(3), [True, 2], np.array([1.]))


@pytest.mark.parametrize('angles', [np.array([.3j]), np.array([]), np.array([np.inf])])
def test_rotation_requires_nonempty_finite_real_angles(angles):
    with pytest.raises(ValueError):
        rotation_characteristic(np.array([-1, 1]), np.array([.5, .5]), angles)


def test_translation_average_matches_explicit_records_not_extra_samples():
    rng = np.random.default_rng(20261012)
    bits = rng.integers(0,2,size=(12,6))
    groups = np.repeat([0,1,2],4)
    angles = np.array([0.,.2,np.pi])
    report = sampled_translation_rotation(bits,groups,[0,1],angles,rows=2,columns=3)
    explicit=[]
    for x in bits:
        regions=[[((i//3+dr)%2)*3+(i%3+dc)%3 for i in (0,1)]
                 for dr in range(2) for dc in range(3)]
        explicit.append(np.mean([np.exp(1j*((x[r]-.5).sum())*angles) for r in regions],axis=0))
    means=np.array([np.array(explicit)[groups==c].mean(0) for c in (0,1,2)])
    np.testing.assert_allclose(report['chain_means'],means,atol=2e-14,rtol=0)
    np.testing.assert_allclose(report['real_chain_SE'],means.real.std(0,ddof=1)/np.sqrt(3),atol=2e-14,rtol=0)
    assert report['records']==12 and report['chains']==3 and report['translation_images']==6
    assert not report['images_are_independent']
    flip=sampled_translation_rotation(bits,groups,[0,1],angles,rows=2,columns=3,spin_inversion_average=True)
    np.testing.assert_allclose(flip['chain_means'],means.real,atol=2e-14,rtol=0)


def test_translation_has_same_expectation_on_invariant_distribution():
    bits=((np.arange(16)[:,None] >> np.arange(3,-1,-1))&1)
    # A nonuniform translation/spin-flip-invariant distribution is represented
    # by repeated exhaustive records, not an optimizer-produced state.
    weights=1+(bits.sum(1)==2).astype(int)
    records=np.repeat(bits,weights,axis=0)
    samples=np.concatenate([records,records])
    groups=np.repeat([0,1],len(records))
    angles=np.array([0.,.43,np.pi,2*np.pi])
    direct=sampled_spin_rotation(samples,groups,[0,1],angles)
    orbit=sampled_translation_rotation(samples,groups,[0,1],angles,rows=2,columns=2,spin_inversion_average=True)
    np.testing.assert_allclose(orbit['value'],direct['value'],atol=2e-14,rtol=0)


def test_orbit_is_not_a_symmetry_repair_for_noninvariant_distribution():
    samples=np.tile([0,0,1,0,1,1],(4,1))
    groups=np.array([0,0,1,1])
    angles=np.array([np.pi])
    direct=sampled_spin_rotation(samples,groups,[0,1],angles)
    orbit=sampled_translation_rotation(samples,groups,[0,1],angles,rows=2,columns=3)
    assert abs(orbit['value'][0]-direct['value'][0]) > .5
    assert orbit['same_expectation_requires_state_symmetry']


def test_record_orbits_centers_complements_and_pi_flip_redundancy():
    rng=np.random.default_rng(2026101201)
    bits=np.array([rng.permutation([0]*3+[1]*3) for _ in range(12)])
    groups=np.repeat([0,1,2],4)
    angles=np.array([0.,.31,np.pi,2*np.pi])
    region=[0,1]
    report=sampled_translation_rotation(bits,groups,region,angles,rows=2,columns=3)
    shuffled=np.roll(bits.reshape(-1,2,3),1,axis=2).reshape(-1,6)
    shifted=sampled_translation_rotation(shuffled,groups,region,angles,rows=2,columns=3)
    np.testing.assert_allclose(report['chain_means'],shifted['chain_means'],atol=2e-14,rtol=0)
    flip=sampled_translation_rotation(bits,groups,region,angles,rows=2,columns=3,spin_inversion_average=True)
    np.testing.assert_allclose(report['chain_means'][:,2].real,flip['chain_means'][:,2].real,atol=2e-14,rtol=0)
    np.testing.assert_allclose(report['chain_means'][:,[0,3]],1,atol=2e-14,rtol=0)
    other=sampled_translation_rotation(bits,groups,[2,3,4,5],-angles,rows=2,columns=3)
    np.testing.assert_allclose(report['chain_means'],other['chain_means'],atol=2e-14,rtol=0)
    # A full-width strip has a translation stabilizer; repeated group images
    # must keep their multiplicity and cannot create additional chain units.
    strip=sampled_translation_rotation(bits,groups,[0,1,2],angles,rows=2,columns=3)
    assert strip['translation_images']==6 and strip['chains']==3
