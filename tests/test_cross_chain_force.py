"""Independent linear-state checks for the once-only fixed-parent diagnostic."""
import numpy as np
import pytest

from su2_nqs.vmc.cross_chain_force import cross_chain_calibration
from su2_nqs.vmc.carrier_protection import physical_moments
from su2_nqs.vmc.retraction_transport import joint_real_coefficients


def data():
    rng = np.random.default_rng(417)
    p = rng.normal(size=16)+1j*rng.normal(size=16)
    p[0] = 0  # A node with nonzero Hpsi must remain in all moments.
    h = rng.normal(size=16)+1j*rng.normal(size=16)
    t = rng.normal(size=(16, 3))+1j*rng.normal(size=(16, 3))
    ht = rng.normal(size=(16, 3))+1j*rng.normal(size=(16, 3))
    q = np.exp(rng.normal(size=16))
    g = np.arange(16)//2
    return p, h, t, ht, q, g, g % 2


def test_complete_slopes_and_real_fit_use_training_only():
    p, h, t, ht, q, g, f = data()
    result = cross_chain_calibration(p, h, t, ht, q, g, f, n_sites=4)
    for row in result['folds']:
        train = f == row['training_fold']; held = ~train
        expected, _ = joint_real_coefficients(p[train], h[train], t[train], ht[train], q[train],
            [np.ones(train.sum(), bool)], n_sites=4)
        np.testing.assert_array_equal(row['coefficients'], expected)
        assert len(expected) == 3
        delta, hd = t[held] @ expected, ht[held] @ expected
        eps = 2e-5
        a = physical_moments(p[held]+eps*delta, h[held]+eps*hd, q[held], 4)
        b = physical_moments(p[held]-eps*delta, h[held]-eps*hd, q[held], 4)
        for k in ('energy', 'variance'):
            assert abs((a[k+'_per_site']-b[k+'_per_site'])/(2*eps)-row['held'][k]) < 2e-8
            assert abs(row['held_contribution_concentration'][k]['signed_sum']-row['held'][k]) < 1e-12
        assert max(row['linear_state_finite_difference_error'].values()) < 2e-8
    altered = h.copy(); altered[f == 1] *= -23
    other = cross_chain_calibration(p, altered, t, ht, q, g, f, n_sites=4)
    np.testing.assert_array_equal(result['folds'][0]['coefficients'], other['folds'][0]['coefficients'])
    assert result['no_parameter_update'] and result['not_an_untouched_audit']


def test_physical_gauge_and_Q_normalizer_cancel():
    p, h, t, ht, q, g, f = data()
    original = cross_chain_calibration(p, h, t, ht, q, g, f, n_sites=4)
    gauge = np.array([.2+.4j, -.8j, 2.])
    z = .7*np.exp(.83j)
    changed = cross_chain_calibration(z*p, z*h, z*(t+p[:, None]*gauge),
        z*(ht+h[:, None]*gauge), 17*q, g, f, n_sites=4)
    for a, b in zip(original['folds'], changed['folds']):
        np.testing.assert_allclose(a['coefficients'], b['coefficients'], atol=2e-12)
        for k in ('energy', 'variance', 'fs_speed'):
            assert abs(a['held'][k]-b['held'][k]) < 2e-12


def test_zero_physical_force_does_not_create_a_step():
    p, _, t, _, q, g, f = data()
    h, ht = -2*p, -2*t
    report = cross_chain_calibration(p, h, t, ht, q, g, f, n_sites=4)
    for row in report['folds']:
        assert np.linalg.norm(row['coefficients']) == 0
        assert row['fs_speed_transfer'] is None
    assert not report['held_joint_point_descent_both']


@pytest.mark.parametrize('fault', ['straddled', 'one_fold', 'too_few', 'bad_Q'])
def test_invalid_chain_partition_and_density(fault):
    p, h, t, ht, q, g, f = data()
    if fault == 'straddled': f[0] = 1-f[0]
    if fault == 'one_fold': f[:] = 0
    if fault == 'too_few': g[:] = np.arange(16)%2; f = g.copy()
    if fault == 'bad_Q': q[0] = 0
    with pytest.raises(ValueError):
        cross_chain_calibration(p, h, t, ht, q, g, f, n_sites=4)
