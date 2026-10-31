import math

import numpy as np
import pytest

from su2_nqs.benchmarks.sampled_block_spin_fcs import (
    fcs_distance_metrics,
    multinomial_frequency_covariance,
    replicated_sampling_summary,
    sample_block_spin_fcs,
    sanitize_fcs_distribution,
    unbiased_frequency_covariance_from_counts,
)


def test_probability_sanitation_reports_tiny_clip_and_rejects_repairs():
    cleaned, audit = sanitize_fcs_distribution(
        {(0, 0): 0.7, (0, 2): -1.0e-14, (2, 2): 0.3 + 1.0e-14}
    )
    assert cleaned[(0, 2)] == 0.0
    assert audit["clipped_negative_probability_magnitude"] == 1.0e-14
    assert np.isclose(sum(cleaned.values()), 1.0)
    with pytest.raises(ValueError, match="more negative"):
        sanitize_fcs_distribution({(0, 0): 1.01, (0, 2): -0.01})
    with pytest.raises(ValueError, match="not normalized"):
        sanitize_fcs_distribution({(0, 0): 0.8, (2, 2): 0.1})


def test_count_covariance_estimator_is_exactly_unbiased_for_binomial_case():
    probability = 0.3
    n_samples = 6
    exact = multinomial_frequency_covariance(
        np.asarray([probability, 1.0 - probability]), n_samples
    )
    expectation = np.zeros((2, 2), dtype=np.float64)
    for first_count in range(n_samples + 1):
        weight = (
            math.comb(n_samples, first_count)
            * probability**first_count
            * (1.0 - probability) ** (n_samples - first_count)
        )
        expectation += weight * unbiased_frequency_covariance_from_counts(
            np.asarray([first_count, n_samples - first_count], dtype=int)
        )
    np.testing.assert_allclose(expectation, exact, atol=1.0e-15)
    np.testing.assert_allclose(np.sum(exact, axis=1), 0.0, atol=1.0e-15)


def test_sampled_fcs_reports_covariance_casimir_and_forbidden_mass():
    distribution = {(0, 0): 0.5, (0, 2): 0.2, (2, 2): 0.3}
    row = sample_block_spin_fcs(distribution, 20_000, 31)
    assert sum(row["counts"]) == 20_000
    assert row["frequency_normalization"] == 1.0
    assert row["minimum_frequency"] >= 0.0
    assert row["maximum_exact_covariance_row_sum"] < 1.0e-15
    assert row["maximum_estimated_covariance_row_sum"] < 1.0e-15
    assert row["forbidden_singlet_closure_mass"] == row["frequencies"][1]
    assert abs(row["forbidden_singlet_closure_mass"] - 0.2) < 0.01
    eigenvalues = np.linalg.eigvalsh(row["frequency_covariance_estimated_unbiased"])
    assert np.min(eigenvalues) > -1.0e-15


def test_repeated_oracle_sampling_closes_on_analytic_variance():
    distribution = {
        (0, 0): 0.15,
        (2, 2): 0.55,
        (4, 4): 0.30,
    }
    summary = replicated_sampling_summary(
        distribution,
        256,
        range(1_000, 1_400),
        keep_raw_counts=False,
    )
    assert summary["n_repeats"] == 400
    assert summary["maximum_absolute_standardized_frequency_bias"] < 4.0
    assert 0.75 < summary["frequency_mse_to_prediction_ratio"] < 1.25
    assert 0.65 < summary["casimir_mse_to_prediction_ratio"] < 1.35
    assert summary["empirical_covariance_relative_frobenius_error"] < 0.15
    assert summary["mean_estimated_covariance_relative_frobenius_error"] < 0.03
    assert summary["minimum_empirical_covariance_eigenvalue"] > -1.0e-15
    assert summary["forbidden_singlet_closure_mass"]["maximum"] == 0.0
    assert summary["total_variation"]["mean"] > 0.0


def test_distance_metrics_keep_full_fixed_support_without_rare_bin_merging():
    reference = {(0, 0): 0.5, (0, 2): 0.0, (2, 2): 0.5}
    estimate = {(0, 0): 0.4, (0, 2): 0.2, (2, 2): 0.4}
    metrics = fcs_distance_metrics(reference, estimate)
    assert np.isclose(metrics["total_variation"], 0.2)
    assert np.isclose(metrics["reference_forbidden_singlet_closure_mass"], 0.0)
    assert np.isclose(metrics["estimated_forbidden_singlet_closure_mass"], 0.2)
