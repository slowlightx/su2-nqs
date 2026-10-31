"""Independent linear-algebra checks of attention output-head limitations."""
import numpy as np
import pytest

from su2_nqs.benchmarks.resonance_attention_capacity import (
    head_rank_certificate, jacobian_rank_report, scalar_generator_hs_weights,
)
from su2_nqs.models.conditional_carrier_resonance import local_scalar_components


def test_center_correct_raw_target_not_anchored_generator_alone():
    rng = np.random.default_rng(2026101601)
    reference = rng.normal(size=(13, 2))@rng.normal(size=(2, 5))+rng.normal(size=5)
    reachable = rng.normal(size=(13, 2))@rng.normal(size=(2, 5))+rng.normal(size=5)
    teacher = reachable-reference
    assert np.linalg.matrix_rank(teacher-teacher.mean(0)) == 4
    report = head_rank_certificate(reference, teacher, head_width=2)
    assert report['weighted_frobenius_error_lower_bound'] < 1e-13
    assert report['context_centered_raw_rank_limit'] == 2
    assert report['physical_state_replay_excluded'] is False


def test_uniform_svd_tail_is_best_matrix_relaxation_error_not_neural_achievability():
    rng = np.random.default_rng(2026101602)
    reference = rng.normal(size=(6, 6, 10))
    teacher = rng.normal(size=(6, 6, 10))
    report = head_rank_certificate(reference, teacher, include_best_table=True)
    relaxed = np.asarray(report['best_rank_limited_generator_table'])
    raw = (reference+relaxed).reshape(36, 10)
    assert np.linalg.matrix_rank(raw-raw.mean(0), tol=1e-10) == 8
    np.testing.assert_allclose(np.linalg.norm(relaxed-teacher),
                               report['weighted_frobenius_error_lower_bound'], rtol=3e-15)
    np.testing.assert_allclose(np.mean((relaxed-teacher)**2),
                               report['weighted_mean_square_error_lower_bound'], rtol=5e-15)
    assert report['table_jacobian_rank_upper_bound'] == 306
    assert report['nonlinear_trunk_achievability_certified'] is False
    assert head_rank_certificate(reference, teacher, head_width=10)['weighted_frobenius_error_lower_bound'] == 0


def test_global_ray_gauge_cannot_evade_uniform_or_weighted_rank_condition():
    rng = np.random.default_rng(2026101603)
    reference = rng.normal(size=(6, 6, 10)); teacher = rng.normal(size=(6, 6, 10))
    identity = np.asarray([a == b for _, a, b in local_scalar_components().generator_labels])
    for weights in ({}, dict(zip(('row_weights', 'column_weights'), scalar_generator_hs_weights()))):
        report = head_rank_certificate(reference, teacher, **weights)
        shifted = head_rank_certificate(reference, teacher+3.19*identity, **weights)
        np.testing.assert_allclose(report['weighted_singular_values'], shifted['weighted_singular_values'], atol=3e-14)
        np.testing.assert_allclose(report['weighted_frobenius_error_lower_bound'],
                                   shifted['weighted_frobenius_error_lower_bound'], atol=3e-14)
        np.testing.assert_allclose(report['relative_to_centered_teacher_norm'],
                                   shifted['relative_to_centered_teacher_norm'], atol=2e-15)


def test_HS_weights_include_every_m_and_weighted_svd_error_is_exact():
    parts = local_scalar_components(); wr, wc = scalar_generator_hs_weights()
    projector_norms = np.einsum('zij,zij->z', parts.spectator_projectors, parts.spectator_projectors)
    active_norms = np.einsum('rij,rij->r', parts.active_basis, parts.active_basis)
    np.testing.assert_allclose(wr, np.outer(projector_norms, projector_norms).reshape(36), atol=2e-14)
    np.testing.assert_allclose(wc, active_norms, atol=2e-15)
    rng = np.random.default_rng(2026101604)
    ref = rng.normal(size=(6, 6, 10)); teacher = rng.normal(size=(6, 6, 10))
    report = head_rank_certificate(ref, teacher, row_weights=wr, column_weights=wc, include_best_table=True)
    delta = (np.asarray(report['best_rank_limited_generator_table'])-teacher).reshape(36, 10)
    norm = np.sqrt(np.sum(wr[:, None]*wc[None, :]*delta**2))
    np.testing.assert_allclose(norm, report['weighted_frobenius_error_lower_bound'], rtol=4e-15)


def test_jacobian_thresholds_are_explicit_and_not_global_no_go():
    jacobian = np.diag([2., 1., 1e-5, 0.])
    report = jacobian_rank_report(jacobian, relative_tolerances=(1e-4, 1e-6),
        target_deltas={'missing':np.array([0., 0., 0., 1.]), 'present':np.array([1., 0., 0., 0.])})
    assert report['ranks_by_relative_tolerance'] == {'0.0001':2, '1e-06':3}
    assert report['target_projections']['missing']['linearized_unreachable_fraction'] == 1
    assert report['target_projections']['present']['linearized_unreachable_fraction'] == 0
    zero = jacobian_rank_report(np.zeros((3, 5)), target_deltas={'zero':np.zeros(3)})
    assert zero['projection_rank'] == 0
    assert zero['target_projections']['zero']['linearized_unreachable_fraction'] is None


def test_invalid_capacity_inputs_fail_closed():
    ref = np.zeros((6, 6, 10))
    with pytest.raises(ValueError): head_rank_certificate(ref, ref[:1])
    with pytest.raises(ValueError): head_rank_certificate(ref, ref.astype(complex))
    with pytest.raises(ValueError): head_rank_certificate(ref, ref, head_width=0)
    with pytest.raises(ValueError): head_rank_certificate(ref, ref, row_weights=np.zeros(36))
    with pytest.raises(ValueError): head_rank_certificate(ref, ref, column_weights=np.ones(9))
    with pytest.raises(ValueError): jacobian_rank_report(np.ones((3, 5)), relative_tolerances=(0,))
    with pytest.raises(ValueError): jacobian_rank_report(np.ones((3, 5)), target_deltas={'bad':np.ones(4)})
