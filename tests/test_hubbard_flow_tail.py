"""Exact checks for the enumerable Hubbard flow-tail diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from su2_nqs.benchmarks.hubbard_flow_tail import (
    coherent_importance_gradient_statistics,
    coherent_importance_sr_direction_statistics,
    hubbard_flow_tail_diagnostics,
)


jax.config.update("jax_enable_x64", True)


def test_flow_tail_uses_residual_weight_and_exact_hit_probabilities() -> None:
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    amplitudes = np.asarray([1.0, 0.0])
    report = hubbard_flow_tail_diagnostics(
        matrix,
        amplitudes,
        (1, 4),
        target_fraction=0.99,
        rare_expected_hit_thresholds=(1.0,),
    )
    assert report["energy"] == pytest.approx(0.0)
    assert report["variance"] == pytest.approx(1.0)
    assert report["flow_target_state_count"] == 1
    assert report["flow_target_probability_mass"] == pytest.approx(0.0)
    for row in report["rows"]:
        assert row["expected_flow_weight_covered"] == pytest.approx(0.0)
        rare = row["rare_expected_hit_sectors"]["1"]
        assert rare["state_count"] == 1
        assert rare["probability_mass"] == pytest.approx(0.0)
        assert rare["flow_weight"] == pytest.approx(1.0)


def test_flow_tail_expected_coverage_increases_with_batch_size() -> None:
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.5]])
    amplitudes = np.asarray([1.0, 2.0])
    report = hubbard_flow_tail_diagnostics(matrix, amplitudes, (1, 8, 64))
    coverage = [row["expected_flow_weight_covered"] for row in report["rows"]]
    assert coverage == sorted(coverage)
    assert 0.0 <= coverage[0] <= coverage[-1] <= 1.0


def test_coherent_importance_gradient_matches_autodiff_identity() -> None:
    basis = ((0,), (1,))
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.5]])
    params = {"theta": jnp.asarray(0.3)}

    def log_fn(candidate, state):
        return candidate["theta"] * state[0] + 0.0j

    report = coherent_importance_gradient_statistics(
        params,
        basis,
        matrix,
        log_fn,
        np.asarray([0.5, 0.5]),
    )
    assert report["exact_gradient_identity_error"] < 1.0e-12
    assert report["single_draw_trace_variance"] >= 0.0
    assert report["maximum_born_importance_weight"] < 2.0
    assert 0.0 < report["born_importance_ess_fraction"] <= 1.0


def test_exhaustive_importance_sr_direction_matches_exact_born_reference() -> None:
    basis = ((0,), (1,))
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.5]])
    params = {"theta": jnp.asarray(0.3)}

    def log_fn(candidate, state):
        return candidate["theta"] * state[0] + 0.0j

    unnormalized = np.asarray([1.0, np.exp(0.6)])
    probabilities = unnormalized / np.sum(unnormalized)
    report = coherent_importance_sr_direction_statistics(
        params,
        basis,
        matrix,
        log_fn,
        basis,
        2.0 * probabilities,
        damping=1.0e-3,
    )
    assert report["gradient_relative_error"] < 1.0e-12
    assert report["fisher_relative_frobenius_error"] < 1.0e-12
    assert report["direction_relative_error"] < 1.0e-12
    assert report["gradient_cosine"] == pytest.approx(1.0)
    assert report["direction_cosine"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("batch_sizes", "target_fraction"),
    [((0,), 0.99), ((1,), 0.0), ((1,), 1.01)],
)
def test_flow_tail_rejects_invalid_controls(batch_sizes, target_fraction) -> None:
    with pytest.raises(ValueError):
        hubbard_flow_tail_diagnostics(
            np.eye(2), np.ones(2), batch_sizes, target_fraction=target_fraction
        )
