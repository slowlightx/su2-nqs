"""Independent analytic oracles for the bounded regional reporting adapter."""
import numpy as np

from projects.clebsch_tree_nqs.explorations.su2_mechanism_regions_local_20261028 import regional


def test_two_spin_singlet_has_one_strict_spin_zero_multiplet():
    psi = np.asarray([0, 1, -1, 0], complex)/np.sqrt(2)
    report = regional(np.outer(psi, psi.conj()))
    assert report['strict_SU2_spectrum_accepted']
    np.testing.assert_allclose(report['spin_sector_weights']['0']['p'], 1, atol=2e-14)
    np.testing.assert_allclose(report['spin_sector_weights']['2']['p'], 0, atol=2e-14)
    assert abs(report['strict_report']['entropy_nats']) < 2e-14


def test_fixed_m_triplet_is_not_falsely_called_invariant():
    rho = np.diag([0., 0., 0., 1.]).astype(complex)
    report = regional(rho)
    assert not report['strict_SU2_spectrum_accepted']
    np.testing.assert_allclose(report['spin_sector_weights']['2']['p'], 1, atol=2e-14)
    np.testing.assert_allclose(report['spin_sector_weights']['2']['conditional_multiplicity_reduced_spectrum'], [1], atol=2e-14)


def test_nonpositive_estimated_rdm_does_not_get_conditional_spectra():
    report = regional(np.diag([1.1, -.1, 0., 0.]).astype(complex))
    assert not report['strict_SU2_spectrum_accepted']
    assert all(row['conditional_multiplicity_reduced_spectrum'] is None
               for row in report['spin_sector_weights'].values())


def test_jackknife_projector_error_matches_analytic_weights():
    singlet = np.asarray([0, 1, -1, 0], complex)/np.sqrt(2)
    p0 = np.outer(singlet, singlet.conj())
    triplet = np.diag([0., 0., 0., 1.]).astype(complex)
    weights = np.asarray([.2, .4, .6, .8])
    samples = np.asarray([w*p0+(1-w)*triplet for w in weights])
    report = regional(.5*p0+.5*triplet, samples)
    expected = np.sqrt(3/4*np.sum((weights-weights.mean())**2))
    np.testing.assert_allclose(report['spin_sector_weights']['0']['independent_chain_jackknife_se'], expected, atol=2e-14)
