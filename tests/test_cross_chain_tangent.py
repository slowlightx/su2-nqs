"""Independent exact/numerical checks for a physical small tangent quotient."""
from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.cross_chain_tangent import (
    chain_fold_metrics, consensus_tangent_direction, fixed_direction_report,
    tangent_metric,
)


def test_complex_fs_centering_removes_global_amplitude_and_phase_not_chain_offsets():
    scores = np.array([[1+2j, 0], [1+2j, 1], [1+2j, 4], [1+2j, 5]])
    metric = tangent_metric(scores)
    np.testing.assert_allclose(metric, [[0., 0.], [0., 4.25]], atol=1e-15)
    np.testing.assert_allclose(tangent_metric(scores+np.array([3-8j, 4+2j])), metric, atol=1e-15)
    # Two chains each at a different constant score are NOT individually centered.
    assert tangent_metric(np.array([[0.], [0.], [2.], [2.]]))[0, 0] == 1.


def test_chain_folds_cannot_leak_and_must_have_same_statistics():
    labels = np.tile(np.arange(4), 3)
    scores = np.arange(24).reshape(12, 2)
    folds = ((0, 2), (1, 3))
    centered = scores-scores.mean(axis=0)
    expected = [centered[np.isin(labels, fold)].T @ centered[np.isin(labels, fold)]/6 for fold in folds]
    np.testing.assert_allclose(chain_fold_metrics(scores, labels, folds), expected)
    np.testing.assert_allclose(chain_fold_metrics(scores, labels, folds).mean(0), tangent_metric(scores))
    # Do not silently erase an extreme between-basin/fold mean difference.
    basin_scores = np.array([[0.], [0.], [2.], [2.]])
    basin_metrics = chain_fold_metrics(basin_scores, np.arange(4), ((0, 1), (2, 3)))
    np.testing.assert_allclose(basin_metrics, np.ones((2, 1, 1)))
    with pytest.raises(ValueError, match='partition'):
        chain_fold_metrics(scores, labels, ((0, 1), (1, 2, 3)))
    with pytest.raises(ValueError, match='partition'):
        chain_fold_metrics(scores, labels, ((0, 1), (2,)))
    with pytest.raises(ValueError, match='equal'):
        chain_fold_metrics(scores, labels, ((0,), (1, 2, 3)))


def test_zero_penalty_matches_exact_natural_gradient():
    g0 = np.array([[2., .3], [.3, 1.]])
    force = np.array([.5, -.7])
    fit = consensus_tangent_direction(np.stack([g0, g0]), force,
        disagreement_strength=0., target_tangent_fs=.003)
    expected = -np.linalg.solve(g0, force)
    expected *= .003/np.sqrt(expected @ g0 @ expected)
    np.testing.assert_allclose(fit.coefficients, expected, rtol=3e-15, atol=1e-17)
    assert fit.directional_slope < 0 and fit.physical_rank == 2


def test_complete_state_finite_overlap_has_the_same_infinitesimal_metric():
    # An independently normalized full state, with amplitude AND phase changes.
    scores = np.array([[1+1j, .3], [-1+.2j, -.7j], [.2-1j, .8], [-.2-.2j, -.4]])
    c = np.array([.6, -.3])
    exact_metric_speed = np.sqrt(c @ tangent_metric(scores) @ c)
    amount = 1e-4
    parent = np.ones(4, complex)/2
    changed = np.exp(amount*(scores @ c))
    changed /= np.linalg.norm(changed)
    finite_fs = np.sqrt(max(0., 1-abs(np.vdot(parent, changed))**2))
    assert finite_fs/amount == pytest.approx(exact_metric_speed, rel=1e-4)


@pytest.mark.parametrize('strength', [0., 1., 4.])
def test_coordinate_change_and_exact_gauge_addition_preserve_physical_direction(strength):
    metrics = np.array([[[1., .1], [.1, .4]], [[2., -.1], [-.1, 1.4]]])
    force = np.array([.4, -.5])
    base = consensus_tangent_direction(metrics, force, disagreement_strength=strength,
                                      target_tangent_fs=.001)
    change = np.array([[2., .4], [-.3, .7]])
    mapped = consensus_tangent_direction(np.einsum('ki,fkl,lj->fij', change, metrics, change),
        change.T @ force, disagreement_strength=strength, target_tangent_fs=.001)
    np.testing.assert_allclose(change @ mapped.coefficients, base.coefficients, rtol=2e-13, atol=1e-16)
    extended = np.zeros((2, 3, 3)); extended[:, :2, :2] = metrics
    gauged = consensus_tangent_direction(extended, np.r_[force, 0.],
        disagreement_strength=strength, target_tangent_fs=.001)
    assert gauged.physical_rank == 2
    np.testing.assert_allclose(gauged.coefficients, np.r_[base.coefficients, 0.], atol=1e-16)
    mixed_gauge = np.array([[1.4, .2, .7], [-.3, .8, -.1], [.4, -.6, 1.1]])
    mixed = consensus_tangent_direction(
        np.einsum('ki,fkl,lj->fij', mixed_gauge, extended, mixed_gauge),
        mixed_gauge.T @ np.r_[force, 0.], disagreement_strength=strength,
        target_tangent_fs=.001)
    # Euclidean representative may differ by gauge; physical first2 must agree.
    np.testing.assert_allclose((mixed_gauge @ mixed.coefficients)[:2],
                               base.coefficients, rtol=3e-13, atol=1e-16)


def test_disagreement_changes_direction_not_only_radius_and_holds_test_fixed():
    # First direction is consistent, second has strongly fold-dependent support.
    metrics = np.array([np.diag([1., .1]), np.diag([1., 1.9])])
    force = np.array([-1., -1.])
    old = consensus_tangent_direction(metrics, force, disagreement_strength=0., target_tangent_fs=.001)
    new = consensus_tangent_direction(metrics, force, disagreement_strength=4., target_tangent_fs=.001)
    assert abs(new.coefficients[1]/new.coefficients[0]) < abs(old.coefficients[1]/old.coefficients[0])
    assert np.ptp(new.predicted_fold_fs) < np.ptp(old.predicted_fold_fs)
    assert old.predicted_common_fs == pytest.approx(new.predicted_common_fs, abs=1e-18)
    scores = np.array([[-1., -np.sqrt(1.9)], [-1., np.sqrt(1.9)],
                       [1., -np.sqrt(1.9)], [1., np.sqrt(1.9)]])
    frozen = new.coefficients.copy()
    report = fixed_direction_report(scores, new.coefficients, reference_fs=.001)
    assert report['relative_to_reference'] < fixed_direction_report(scores, old.coefficients,
        reference_fs=.001)['relative_to_reference']
    np.testing.assert_array_equal(new.coefficients, frozen)
    # Shrinking the old direction does not change held-out / common ratio.
    for scale in (.25, .5, 1.):
        assert fixed_direction_report(scores, scale*old.coefficients,
            reference_fs=scale*.001)['relative_to_reference'] == pytest.approx(
                fixed_direction_report(scores, old.coefficients, reference_fs=.001)['relative_to_reference'])


def test_fail_closed_on_null_force_negative_metric_and_nonfinite_inputs():
    metrics = np.array([np.diag([1., 0.])]*2)
    with pytest.raises(ValueError, match='metric-null'):
        consensus_tangent_direction(metrics, np.array([1., .1]),
            disagreement_strength=1., target_tangent_fs=.001)
    with pytest.raises(ValueError, match='negative'):
        consensus_tangent_direction(np.array([np.diag([1., -.1])]*2), np.ones(2),
            disagreement_strength=1., target_tangent_fs=.001)
    with pytest.raises(ValueError, match='finite'):
        tangent_metric(np.full((2, 1), np.nan))
