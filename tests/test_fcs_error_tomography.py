from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.fcs_error_tomography import (
    block_casimir_features,
    casimir_error_tomography,
)


def test_casimir_features_use_doubled_spin_labels() -> None:
    actual = block_casimir_features(((0, 2), (4, 6)))
    np.testing.assert_allclose(actual, ((0.0, 2.0), (6.0, 12.0)))


def test_pure_first_moment_error_has_zero_shape_direction() -> None:
    outcomes = ((0,), (2,), (4,))
    reference = {(0,): 0.25, (2,): 0.50, (4,): 0.25}
    features = block_casimir_features(outcomes)[:, 0]
    centered = features - sum(reference[key] * features[i] for i, key in enumerate(outcomes))
    epsilon = 0.01
    candidate = {
        key: reference[key] * (1.0 + epsilon * centered[index])
        for index, key in enumerate(outcomes)
    }
    result = casimir_error_tomography(reference, candidate, outcomes=outcomes)
    assert result["casimir_orthogonal_shape_information"] < 1.0e-28
    assert result["shape_total_variation_tangent"] < 1.0e-14
    assert result["first_casimir_moment_information"] == pytest.approx(
        result["total_pearson_information"]
    )


def test_shape_error_preserves_normalization_and_first_moment() -> None:
    outcomes = ((0,), (2,), (4,))
    reference = {(0,): 0.25, (2,): 0.50, (4,): 0.25}
    # This direction has zero sum and zero Casimir moment:
    # 4*delta_2 + 12*delta_4 = 0.
    candidate = {(0,): 0.27, (2,): 0.47, (4,): 0.26}
    result = casimir_error_tomography(reference, candidate, outcomes=outcomes)
    assert result["first_casimir_moment_information"] < 1.0e-28
    assert result["casimir_orthogonal_shape_information"] == pytest.approx(
        result["total_pearson_information"]
    )
    assert result["shape_normalization_error"] < 1.0e-14
    assert result["maximum_shape_casimir_moment_error"] < 1.0e-14


def test_redundant_complement_features_do_not_break_projection() -> None:
    outcomes = ((0, 0), (2, 2), (4, 4))
    reference = {outcomes[0]: 0.2, outcomes[1]: 0.5, outcomes[2]: 0.3}
    candidate = {outcomes[0]: 0.22, outcomes[1]: 0.47, outcomes[2]: 0.31}
    result = casimir_error_tomography(reference, candidate, outcomes=outcomes)
    assert result["feature_rank"] == 1
    assert result["fisher_pythagorean_error"] < 1.0e-14
    assert abs(result["fisher_cross_term"]) < 1.0e-14


def test_material_candidate_mass_outside_reference_support_fails_closed() -> None:
    reference = {(0,): 1.0, (2,): 0.0}
    candidate = {(0,): 0.9, (2,): 0.1}
    with pytest.raises(ValueError, match="outside"):
        casimir_error_tomography(reference, candidate)


def test_invalid_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="normalized"):
        casimir_error_tomography({(0,): 0.7}, {(0,): 1.0})

