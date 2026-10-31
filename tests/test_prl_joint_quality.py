"""Exact-small contracts for the P8 joint target and sampled V-score."""

from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.prl_joint_quality import (
    complete_joint_quality,
    confirmed_quality_crossing,
    draw_sampled_v_score,
    draw_uniform_mixture_v_score,
    exact_v_score,
    importance_sampled_v_score,
    sampled_v_score,
    v_score_closure,
)


def test_physical_and_certified_targets_are_kept_separate() -> None:
    result = complete_joint_quality(
        energy_error_per_site=0.01,
        v_score=0.1,
        spectral_normalized_scalar_errors={"K": 0.01, "stripe": 0.03},
        fcs_total_variation=0.01,
        fcs_hellinger=0.01,
        spectral_normalized_fcs_casimir_error=0.01,
        native_non_singlet_weight=0.2,
        native_wrong_spatial_sector_weight=0.1,
    )
    assert result["Q_phys"] == pytest.approx(1.5)
    assert result["Q_cert"] == pytest.approx(4.0)
    assert result["worst_physical_terms"] == ["scalar:stripe"]
    assert result["worst_certified_terms"] == ["non_singlet_weight"]


def test_joint_target_is_noncompensating_and_rejects_bad_inputs() -> None:
    common = dict(
        energy_error_per_site=0.001,
        v_score=0.01,
        spectral_normalized_scalar_errors={"N": 0.001},
        fcs_total_variation=0.001,
        fcs_hellinger=0.001,
        spectral_normalized_fcs_casimir_error=0.001,
        native_non_singlet_weight=0.0,
    )
    assert complete_joint_quality(**common)["Q_phys"] < 1.0
    with pytest.raises(ValueError, match="at least one scalar"):
        complete_joint_quality(**{**common, "spectral_normalized_scalar_errors": {}})
    with pytest.raises(ValueError, match="non-negative"):
        complete_joint_quality(**{**common, "v_score": -1.0})


def test_crossing_is_the_confirmation_point_of_first_consecutive_run() -> None:
    trajectory = [
        {"step": step, "Q_phys": value}
        for step, value in enumerate((1.2, 0.9, 1.1, 0.8, 0.7, 0.6, 0.5))
    ]
    crossing = confirmed_quality_crossing(
        trajectory, score_key="Q_phys", confirmation_points=3
    )
    assert crossing == {
        "first_qualifying_index": 3,
        "confirmation_index": 5,
        "first_qualifying_step": 3.0,
        "confirmation_step": 5.0,
    }


def test_v_score_exact_definition_and_large_sample_closure() -> None:
    probabilities = np.asarray([0.2, 0.3, 0.5])
    local_energies = np.asarray([-2.0, -1.0, -3.0])
    exact = exact_v_score(probabilities, local_energies, n_sites=12)
    energy = float(probabilities @ local_energies)
    variance = float(probabilities @ (local_energies - energy) ** 2)
    assert exact["energy"] == pytest.approx(energy)
    assert exact["variance"] == pytest.approx(variance)
    assert exact["v_score"] == pytest.approx(12 * variance / energy**2)

    estimate = draw_sampled_v_score(
        probabilities,
        local_energies,
        n_sites=12,
        n_samples=500_000,
        seed=701,
    )
    closure = v_score_closure(
        exact=exact["v_score"],
        estimate=estimate.v_score,
        standard_error=estimate.standard_error,
    )
    assert closure["pass"]


def test_v_score_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="normalized"):
        exact_v_score([0.2, 0.2], [-1.0, -2.0], n_sites=2)
    with pytest.raises(ValueError, match="zero energy"):
        exact_v_score([0.5, 0.5], [-1.0, 1.0], n_sites=2)
    with pytest.raises(ValueError, match="at least three"):
        draw_sampled_v_score(
            [0.5, 0.5], [-1.0, -2.0], n_sites=2, n_samples=2, seed=0
        )


def test_complex_local_energy_sample_keeps_imaginary_mean_as_an_audit() -> None:
    samples = np.asarray([-2.0 + 0.4j, -1.0 - 0.1j, -3.0 + 0.3j, -2.0])
    estimate = sampled_v_score(samples, n_sites=4)
    energy = float(np.mean(samples).real)
    expected_variance = float(np.mean(np.abs(samples - energy) ** 2))
    assert estimate.energy == pytest.approx(energy)
    assert estimate.imaginary_energy == pytest.approx(float(np.mean(samples).imag))
    assert estimate.variance == pytest.approx(expected_variance)
    assert estimate.v_score == pytest.approx(4 * expected_variance / energy**2)
    assert estimate.as_dict()["imaginary_energy"] == pytest.approx(0.15)


def test_importance_estimator_reduces_to_iid_for_unit_weights() -> None:
    samples = np.asarray([-2.0 + 0.2j, -1.0 - 0.1j, -3.0, -2.5 + 0.1j])
    iid = sampled_v_score(samples, n_sites=6)
    importance = importance_sampled_v_score(
        samples, np.ones(len(samples)), n_sites=6
    )
    assert importance.energy == pytest.approx(iid.energy)
    assert importance.variance == pytest.approx(iid.variance)
    assert importance.v_score == pytest.approx(iid.v_score)
    assert importance.standard_error == pytest.approx(iid.standard_error)
    assert importance.effective_sample_size == pytest.approx(len(samples))


def test_uniform_mixture_closes_a_rare_tail_that_plain_born_sampling_misses() -> None:
    probabilities = np.asarray([0.999_999, 0.000_001])
    local_energies = np.asarray([-2.0, 998.0])
    exact = exact_v_score(probabilities, local_energies, n_sites=12)
    estimate = draw_uniform_mixture_v_score(
        probabilities,
        local_energies,
        n_sites=12,
        n_samples=200_000,
        seed=937,
        uniform_fraction=0.01,
    )
    closure = v_score_closure(
        exact=exact["v_score"],
        estimate=estimate.v_score,
        standard_error=estimate.standard_error,
    )
    assert closure["pass"]
    assert estimate.effective_sample_size > 190_000


def test_importance_estimator_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="importance weights"):
        importance_sampled_v_score([-1.0, -2.0, -3.0], [1.0, -1.0, 1.0], n_sites=3)
    with pytest.raises(ValueError, match="uniform_fraction"):
        draw_uniform_mixture_v_score(
            [0.5, 0.5],
            [-1.0, -2.0],
            n_sites=2,
            n_samples=10,
            seed=0,
            uniform_fraction=1.1,
        )
    with pytest.raises(ValueError, match="weight sums"):
        importance_sampled_v_score(
            [-1.0, -2.0, -3.0], [1.0e308, 1.0e308, 1.0e308], n_sites=3
        )
