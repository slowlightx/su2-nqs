"""Independent product-ray tangent references (no neural model/eigensolver)."""
import numpy as np
import pytest
from su2_nqs.vmc.product_tangent_diagnostic import diagnose_product_tangent


def test_each_ray_has_its_own_phase_and_real_parameter_span():
    p = np.array([[1, 0], [1j, 0]], complex)
    h = np.array([[0, 1+2j], [0, -2+1j]], complex)
    t = np.array([[[1j, 3], [1, 1j]], [[7, -2j], [1j, -1]]], complex)
    report = diagnose_product_tangent(p, h, t, n_sites=2)
    # Both rays have the same REAL horizontal response; exactly capture the
    # 1+i*2 residual using the two real columns, not their complex span.
    np.testing.assert_allclose(report.projected_target, report.target, atol=2e-14)
    assert report.metrics['spectra']['column_equilibrated']['cutoffs']['1e-10']['rank'] == 2
    phase = np.array([np.exp(.3j), np.exp(-1.2j)])
    changed = diagnose_product_tangent(p*phase[:, None], h*phase[:, None],
        (t+p[:, :, None]*np.array([[6+2j, -4j], [1j, 3]] )[:, None, :])*phase[:, None, None], n_sites=2)
    assert changed.metrics['mean_variance_per_site'] == pytest.approx(report.metrics['mean_variance_per_site'])
    assert changed.metrics['spectra']['column_equilibrated']['cutoffs']['1e-10']['projected_residual_fraction'] == pytest.approx(1)
    assert changed.metrics['maximum_horizontal_gauge_defect'] < 1e-14


def test_full_pointwise_span_can_conflict_in_shared_span():
    p = np.array([[1, 0], [1, 0]], complex)
    h = np.array([[0, 1], [0, -1]], complex)
    t = np.array([[[0], [1]], [[0], [1]]], complex)
    report = diagnose_product_tangent(p, h, t, n_sites=2)
    entry = report.metrics['spectra']['column_equilibrated']['cutoffs']['1e-10']
    assert entry['projected_residual_fraction'] < 1e-28
    for k in range(2):
        point = diagnose_product_tangent(p[k:k+1], h[k:k+1], t[k:k+1], n_sites=2)
        assert point.metrics['spectra']['column_equilibrated']['cutoffs']['1e-10']['projected_residual_fraction'] == pytest.approx(1)
    assert report.metrics['spectra']['parameter_coordinates']['gradient_norm_per_site'] < 1e-14


def test_damping_suppresses_reachable_weak_directions():
    p = np.array([[1, 0, 0]], complex); h = np.array([[0, 0, 1]], complex)
    t = np.array([[[0, 0], [1, 0], [0, 1e-4]]], complex)
    r = diagnose_product_tangent(p, h, t, n_sites=3, damping=1e-3)
    raw = r.metrics['spectra']['parameter_coordinates']
    assert raw['cutoffs']['1e-10']['projected_residual_fraction'] == pytest.approx(1)
    assert raw['damped_target_norm_fraction'] < 1e-9
    np.testing.assert_allclose(r.design.T@r.projected_target, r.design.T@r.target, atol=1e-14)


def test_input_validation_and_null_span():
    p = np.array([[1, 0]], complex); h = np.array([[0, 1]], complex)
    r = diagnose_product_tangent(p, h, 1j*p[:, :, None], n_sites=2)
    assert r.metrics['spectra']['column_equilibrated']['cutoffs']['1e-10']['rank'] == 0
    for kwargs in ({'weights': np.array([0.])}, {'n_sites': True}, {'relative_cutoffs': (1e-10, 1e-6)}, {'damping': 0.}):
        with pytest.raises(ValueError):
            diagnose_product_tangent(p, h, p[:, :, None], **({'n_sites': 2}|kwargs))
