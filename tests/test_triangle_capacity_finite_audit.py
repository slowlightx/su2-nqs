"""Independent finite full-product estimators; parent H cannot substitute."""
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_capacity_finite_audit import paired_product_statistics


def test_actual_products_recover_exact_weighted_energy_and_variance():
    # Four equally weighted parent Born configurations, one in each chain.
    p = np.ones(4)/2
    h = np.diag([-2., -1., 1., 3.])
    trial = np.array([.8, .6, .4, .3])
    result = paired_product_statistics(p, h@p, trial, h@trial, np.arange(4), n_sites=4)
    energy = np.vdot(trial, h@trial).real/np.vdot(trial, trial).real
    variance = np.linalg.norm(h@trial-energy*trial)**2/np.vdot(trial, trial).real
    np.testing.assert_allclose(result['candidate']['energy_per_site'], energy/4, atol=2e-15)
    np.testing.assert_allclose(result['candidate']['variance_per_site'], variance/4, atol=2e-15)


def test_identical_state_has_zero_paired_change_and_scale_invariant_variance():
    p = np.array([1., 2., 3., 4., 1., 2., 3., 4.])
    hp = np.arange(8)*p
    result = paired_product_statistics(p, hp, 3j*p, 3j*hp, np.tile(np.arange(4), 2))
    for key in ('energy_change_per_site', 'variance_change_per_site',
                'paired_energy_change_SE', 'paired_variance_change_SE'):
        np.testing.assert_allclose(result[key], 0, atol=2e-14)
