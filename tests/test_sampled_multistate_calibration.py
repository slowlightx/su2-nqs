from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.sampled_multistate_calibration import (
    EqualStratumSample,
    calibrate_fit_and_held_out_multistate,
    compare_flat_forces,
    draw_equal_stratum_sample,
    summarize_scalar_metric,
)


def _uniform_phase_problem(
    dimension: int = 17, root_count: int = 3, seed: int = 91500
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, size=(dimension, root_count))
    states = np.exp(1.0j * phases) / np.sqrt(dimension)
    raw = rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(
        size=(dimension, dimension)
    )
    hamiltonian = 0.5 * (raw + raw.conj().T)
    return states, hamiltonian @ states


def _exhaustive_equal_strata(dimension: int, root_count: int) -> EqualStratumSample:
    return EqualStratumSample(
        basis_indices=np.tile(np.arange(dimension, dtype=np.int64), root_count),
        root_ids=np.repeat(np.arange(root_count, dtype=np.int32), dimension),
        samples_per_root=dimension,
        root_count=root_count,
    )


def test_equal_stratum_draw_has_registered_counts_and_replays() -> None:
    states, _ = _uniform_phase_problem()
    probabilities = np.abs(states) ** 2
    first = draw_equal_stratum_sample(
        probabilities, 23, np.random.default_rng(91501)
    )
    replay = draw_equal_stratum_sample(
        probabilities, 23, np.random.default_rng(91501)
    )

    np.testing.assert_array_equal(first.basis_indices, replay.basis_indices)
    np.testing.assert_array_equal(np.bincount(first.root_ids), [23, 23, 23])
    assert first.sample_count == 69
    assert first.samples_per_root == 23
    assert first.root_count == 3


def test_exhaustive_uniform_strata_recovers_every_held_out_reference() -> None:
    states, actions = _uniform_phase_problem()
    sample = _exhaustive_equal_strata(len(states), states.shape[1])
    result = calibrate_fit_and_held_out_multistate(
        states,
        actions,
        sample,
        sample,
        n_sites=8,
        maximum_condition_number=1.0e8,
    )

    assert result.maximum_fit_ritz_energy_error_per_site < 2.0e-15
    assert result.maximum_held_out_energy_error_per_site < 2.0e-15
    assert result.maximum_held_out_residual_absolute_error_per_sqrt_site < 2.0e-15
    assert result.maximum_held_out_residual_relative_error < 3.0e-14
    assert result.held_out_hamiltonian_antihermiticity_relative < 4.0e-15
    assert result.maximum_root_importance_ratio == pytest.approx(1.0)
    assert result.minimum_root_ratio_kish_ess_fraction == pytest.approx(1.0)


def test_fit_and_held_out_samples_must_have_matching_strata() -> None:
    states, actions = _uniform_phase_problem()
    fit = draw_equal_stratum_sample(
        np.abs(states) ** 2, 8, np.random.default_rng(91502)
    )
    held = draw_equal_stratum_sample(
        np.abs(states) ** 2, 9, np.random.default_rng(91503)
    )
    with pytest.raises(ValueError, match="equal stratum sizes"):
        calibrate_fit_and_held_out_multistate(
            states, actions, fit, held, n_sites=8
        )


def test_force_comparison_reports_direction_and_zero_reference_honestly() -> None:
    exact = np.asarray([1.0, -2.0, 0.5])
    sampled = 1.2 * exact
    result = compare_flat_forces(exact, sampled)
    assert result.cosine_similarity == pytest.approx(1.0)
    assert result.relative_l2_error == pytest.approx(0.2)

    zero = compare_flat_forces(np.zeros(3), np.ones(3))
    assert np.isnan(zero.cosine_similarity)
    assert zero.relative_l2_error > 1.0e12


def test_scalar_summary_is_deterministic_and_contains_the_mean() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.5, 0.8])
    first = summarize_scalar_metric(
        values, bootstrap_seed=91504, bootstrap_resamples=500
    )
    second = summarize_scalar_metric(
        values, bootstrap_seed=91504, bootstrap_resamples=500
    )
    assert first == second
    assert first.count == 5
    assert first.mean_ci95_lower <= first.mean <= first.mean_ci95_upper
    assert first.minimum == pytest.approx(0.1)
    assert first.maximum == pytest.approx(0.8)
    assert first.q05 < first.median < first.q95
