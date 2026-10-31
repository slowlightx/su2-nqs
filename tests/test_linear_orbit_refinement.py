"""Independent finite physical-vector checks for orbit/span calibration helpers."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.linear_orbit_refinement import (
    make_orbit_amplitude_products, protected_center_directions,
    raw_normalized_line_coefficients, calibrate_shared_projected_radius,
)


def test_complex_orbit_amplitude_and_positive_guide_match_explicit_physical_vectors():
    rng = np.random.default_rng(2610269301)
    state = rng.normal(size=16)+1j*rng.normal(size=16)
    state /= np.linalg.norm(state)
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int32)
    images = np.asarray([[0, 1, 2, 3], [1, 2, 3, 0], [3, 2, 1, 0], [2, 0, 3, 1]])
    raw = lambda bank, row: bank[row@jnp.asarray([8, 4, 2, 1])]
    for chunk in (1, 2, 4):
        phi, guide = make_orbit_amplitude_products(raw, images, projection_chunk=chunk)(state, bits)
        values = np.asarray([[state[row[image]@np.asarray([8, 4, 2, 1])] for image in images] for row in bits])
        np.testing.assert_allclose(phi, values.mean(1), atol=3e-16)
        np.testing.assert_allclose(guide, (abs(values)**2).mean(1), atol=3e-16)
        assert np.max(abs(phi)**2-np.asarray(guide)) < 2e-16
        assert np.sum(guide) == pytest.approx(1., abs=4e-16)


def test_protected_residual_split_precedes_orthogonalization_and_preserves_nested_span():
    rng = np.random.default_rng(2610269302)
    c = rng.normal(size=12)+1j*rng.normal(size=12); c /= np.linalg.norm(c)
    matrix = rng.normal(size=(12, 12))+1j*rng.normal(size=(12, 12))
    h = (matrix+matrix.conj().T)/2
    mask = np.arange(12) < 7
    basis, report = protected_center_directions(c, h@c, mask)
    np.testing.assert_allclose(basis.conj().T@basis, np.eye(3), atol=5e-16)
    old = -(mask*(h@c-np.vdot(c, h@c)*c))
    np.testing.assert_allclose(basis[:, :2]@(basis[:, :2].conj().T@old), old, atol=8e-16)
    assert report['old_control_is_direction_about_same_enriched_parent']
    with pytest.raises(ValueError):
        protected_center_directions(c, c, mask)


def test_amplitude_only_shared_radius_matches_exact_complex_physical_FS():
    rng = np.random.default_rng(2610269303)
    physical = rng.normal(size=(8, 3))+1j*rng.normal(size=(8, 3))
    q = np.arange(1., 9.); q /= q.sum()
    one = np.repeat(np.arange(8), np.arange(1, 9))
    rows = np.tile(one, 8); groups = np.repeat(np.arange(8), len(one))
    fitted = np.asarray([[np.cos(.12), np.sin(.12), 0.], [np.cos(.08), 0., np.sin(.08)]])
    selected, report = calibrate_shared_projected_radius(physical[rows], q[rows], groups, fitted, seed=21)
    expected = []
    for c in selected:
        overlap = abs(np.vdot(physical[:, 0], physical@c))/np.linalg.norm(physical[:, 0])/np.linalg.norm(physical@c)
        expected.append(np.arccos(overlap))
    np.testing.assert_allclose(report['selected_FS_radians'], expected, atol=3e-14)
    assert max(expected) <= .03
    assert 0 < report['shared_fraction'] < 1
    assert report['energy_or_H_used'] is False
    scaled, _ = calibrate_shared_projected_radius(physical[rows]*1e-100, q[rows]*1e-80, groups, fitted, seed=21)
    np.testing.assert_allclose(scaled, selected, atol=2e-15)


def test_raw_normalization_and_cap_and_null_fraction_are_exact():
    c = np.asarray([-2., 1., 3.])
    capped = raw_normalized_line_coefficients(c, raw_angle_cap=.08)
    assert np.linalg.norm(capped) == pytest.approx(1., abs=2e-16)
    assert np.arccos(capped[0]) == pytest.approx(.08, abs=2e-15)
    np.testing.assert_array_equal(raw_normalized_line_coefficients(c, 0.), [1., 0., 0.])
    with pytest.raises(ValueError):
        raw_normalized_line_coefficients(c.astype(complex))


def test_integer_center_inputs_and_invalid_calibration_resamples_fail_closed():
    basis, _ = protected_center_directions(np.asarray([1, 0, 0]), np.asarray([2, 3, 4]),
                                          np.asarray([True, True, False]))
    np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=1e-15)
    p = np.eye(4, 3, dtype=complex)
    selected, report = calibrate_shared_projected_radius(p, np.ones(4), np.arange(4),
        np.asarray([[.99, .1, 0.], [.99, 0., .1]]), seed=7)
    np.testing.assert_array_equal(selected, [[1., 0., 0.], [1., 0., 0.]])
    assert report['invalid_bootstrap_values'] > 0 and report['null_parent_fallback']
    assert report['selected_FS_bootstrap_95'] == [[None, None], [None, None]]
