"""Shot-noise calibration for collective-spin full counting statistics.

This module starts from a probability distribution over *already measured*
collective-spin outcomes and draws multinomial counts.  It therefore answers a
limited but useful question: how many ideal block-spin measurement shots are
needed to estimate a fixed FCS?  It does not turn computational-basis spin
snapshots into outcomes of the non-diagonal block-spin projectors and it does
not implement a scalable NQS estimator for those projectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .block_spin_fcs import FCSKey, doubled_spins_can_close_to_singlet


def sanitize_fcs_distribution(
    probabilities: Mapping[FCSKey, float],
    *,
    negative_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-10,
) -> tuple[dict[FCSKey, float], dict[str, float | int]]:
    """Validate and minimally normalize a floating-point FCS distribution.

    Negative values within ``negative_tolerance`` are explicitly clipped and
    reported.  Larger negative values or a normalization discrepancy larger
    than ``normalization_tolerance`` are rejected rather than silently fixed.
    """

    if negative_tolerance < 0.0 or normalization_tolerance < 0.0:
        raise ValueError("probability tolerances must be non-negative")
    if not probabilities:
        raise ValueError("probabilities must be nonempty")
    keys = tuple(sorted(probabilities))
    arity = len(keys[0])
    if arity < 1 or any(len(key) != arity for key in keys):
        raise ValueError("all FCS outcomes must have the same positive arity")
    if any(any(not isinstance(q, (int, np.integer)) or int(q) < 0 for q in key) for key in keys):
        raise ValueError("FCS outcomes must contain non-negative doubled spins")
    values = np.asarray([float(probabilities[key]) for key in keys], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    minimum = float(np.min(values))
    if minimum < -negative_tolerance:
        raise ValueError("a probability is more negative than negative_tolerance")
    input_normalization = float(np.sum(values))
    if abs(input_normalization - 1.0) > normalization_tolerance:
        raise ValueError("probabilities are not normalized within tolerance")
    clipped_magnitude = float(np.sum(np.maximum(-values, 0.0)))
    clipped = np.maximum(values, 0.0)
    clipped_normalization = float(np.sum(clipped))
    if clipped_normalization <= 0.0:
        raise ValueError("probability normalization must be positive")
    normalized = clipped / clipped_normalization
    result = {key: float(value) for key, value in zip(keys, normalized, strict=True)}
    return result, {
        "support_size": len(keys),
        "input_normalization": input_normalization,
        "input_normalization_error": abs(input_normalization - 1.0),
        "minimum_input_probability": minimum,
        "clipped_negative_probability_magnitude": clipped_magnitude,
        "renormalization_factor": 1.0 / clipped_normalization,
        "output_normalization": float(np.sum(normalized)),
    }


def casimir_features(outcomes: Sequence[FCSKey]) -> np.ndarray:
    """Return one ``S_g(S_g+1)`` feature vector per doubled-spin outcome."""

    keys = tuple(outcomes)
    if not keys:
        raise ValueError("outcomes must be nonempty")
    arity = len(keys[0])
    if any(len(key) != arity for key in keys):
        raise ValueError("all FCS outcomes must have the same arity")
    return np.asarray(
        [[int(q) * (int(q) + 2) / 4.0 for q in key] for key in keys],
        dtype=np.float64,
    )


def multinomial_frequency_covariance(
    probabilities: Sequence[float],
    n_samples: int,
) -> np.ndarray:
    """Return the exact covariance of multinomial frequencies ``counts/M``."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("probabilities must be a nonempty vector")
    if np.any(~np.isfinite(values)) or np.min(values) < 0.0:
        raise ValueError("probabilities must be finite and non-negative")
    if abs(float(np.sum(values)) - 1.0) > 1.0e-12:
        raise ValueError("probabilities must be normalized")
    return (np.diag(values) - np.outer(values, values)) / n_samples


def unbiased_frequency_covariance_from_counts(counts: Sequence[int]) -> np.ndarray:
    """Estimate ``Cov(counts/M)`` without bias for multinomial samples.

    The returned matrix estimates the covariance of the *frequency estimator*,
    not the covariance of a single categorical draw.  At least two shots are
    required.
    """

    values = np.asarray(counts)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("counts must be a nonempty vector")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("counts must be non-negative integers")
    n_samples = int(np.sum(values))
    if n_samples < 2:
        raise ValueError("at least two samples are required")
    frequencies = values.astype(np.float64) / n_samples
    return (np.diag(frequencies) - np.outer(frequencies, frequencies)) / (
        n_samples - 1
    )


def _forbidden_indicators(keys: Sequence[FCSKey]) -> np.ndarray | None:
    if len(keys[0]) not in (2, 3):
        return None
    return np.asarray(
        [not doubled_spins_can_close_to_singlet(key) for key in keys],
        dtype=np.float64,
    )


def fcs_distance_metrics(
    reference: Mapping[FCSKey, float],
    estimate: Mapping[FCSKey, float],
) -> dict[str, Any]:
    """Compare two normalized FCS distributions on their fixed union support."""

    truth, _ = sanitize_fcs_distribution(reference)
    trial, _ = sanitize_fcs_distribution(estimate)
    keys = tuple(sorted(set(truth) | set(trial)))
    p = np.asarray([truth.get(key, 0.0) for key in keys], dtype=np.float64)
    q = np.asarray([trial.get(key, 0.0) for key in keys], dtype=np.float64)
    features = casimir_features(keys)
    reference_casimir = p @ features
    estimated_casimir = q @ features
    bhattacharyya = float(np.sum(np.sqrt(p * q)))
    forbidden = _forbidden_indicators(keys)
    return {
        "total_variation": float(0.5 * np.sum(np.abs(p - q))),
        "hellinger": float(np.sqrt(max(0.0, 1.0 - min(bhattacharyya, 1.0)))),
        "casimir_moments_reference": reference_casimir.tolist(),
        "casimir_moments_estimate": estimated_casimir.tolist(),
        "casimir_error_vector": (estimated_casimir - reference_casimir).tolist(),
        "casimir_error_l2": float(np.linalg.norm(estimated_casimir - reference_casimir)),
        "total_casimir_absolute_error": float(
            abs(np.sum(estimated_casimir) - np.sum(reference_casimir))
        ),
        "reference_forbidden_singlet_closure_mass": (
            None if forbidden is None else float(p @ forbidden)
        ),
        "estimated_forbidden_singlet_closure_mass": (
            None if forbidden is None else float(q @ forbidden)
        ),
    }


def sample_block_spin_fcs(
    probabilities: Mapping[FCSKey, float],
    n_samples: int,
    seed: int,
    *,
    negative_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Draw ideal block-spin outcomes and report estimator uncertainty.

    The frequency and all linear moments are unbiased.  TV, Hellinger, and
    absolute-error summaries computed downstream are nonlinear and generally
    have a positive finite-shot noise floor.
    """

    if n_samples < 2:
        raise ValueError("at least two samples are required")
    cleaned, sanitation = sanitize_fcs_distribution(
        probabilities,
        negative_tolerance=negative_tolerance,
        normalization_tolerance=normalization_tolerance,
    )
    keys = tuple(cleaned)
    p = np.asarray([cleaned[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(int(n_samples), p)
    frequencies = counts.astype(np.float64) / n_samples
    covariance_exact = multinomial_frequency_covariance(p, n_samples)
    covariance_estimated = unbiased_frequency_covariance_from_counts(counts)
    features = casimir_features(keys)
    casimir_mean = frequencies @ features
    casimir_covariance_exact = features.T @ covariance_exact @ features
    casimir_covariance_estimated = features.T @ covariance_estimated @ features
    forbidden = _forbidden_indicators(keys)
    distribution = {
        key: float(value) for key, value in zip(keys, frequencies, strict=True)
    }
    return {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "outcomes": [list(key) for key in keys],
        "counts": counts.astype(int).tolist(),
        "frequencies": frequencies.tolist(),
        "frequency_normalization": float(np.sum(frequencies)),
        "minimum_frequency": float(np.min(frequencies)),
        "frequency_covariance_exact": covariance_exact.tolist(),
        "frequency_covariance_estimated_unbiased": covariance_estimated.tolist(),
        "maximum_exact_covariance_row_sum": float(
            np.max(np.abs(np.sum(covariance_exact, axis=1)))
        ),
        "maximum_estimated_covariance_row_sum": float(
            np.max(np.abs(np.sum(covariance_estimated, axis=1)))
        ),
        "casimir_moments": casimir_mean.tolist(),
        "casimir_covariance_exact": casimir_covariance_exact.tolist(),
        "casimir_covariance_estimated_unbiased": casimir_covariance_estimated.tolist(),
        "forbidden_singlet_closure_mass": (
            None if forbidden is None else float(frequencies @ forbidden)
        ),
        "forbidden_singlet_closure_standard_error_estimated": (
            None
            if forbidden is None
            else float(
                np.sqrt(max(0.0, forbidden @ covariance_estimated @ forbidden))
            )
        ),
        "distances_from_generating_distribution": fcs_distance_metrics(
            cleaned, distribution
        ),
        "sanitation": sanitation,
    }


def replicated_sampling_summary(
    probabilities: Mapping[FCSKey, float],
    n_samples: int,
    seeds: Sequence[int],
    *,
    keep_raw_counts: bool = True,
) -> dict[str, Any]:
    """Summarize independent multinomial repeats against analytic covariance."""

    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("at least two distinct repeat seeds are required")
    cleaned, sanitation = sanitize_fcs_distribution(probabilities)
    keys = tuple(cleaned)
    p = np.asarray([cleaned[key] for key in keys], dtype=np.float64)
    samples = [
        sample_block_spin_fcs(cleaned, n_samples, seed) for seed in seed_values
    ]
    frequencies = np.asarray([row["frequencies"] for row in samples])
    exact_covariance = multinomial_frequency_covariance(p, n_samples)
    centered_frequencies = frequencies - np.mean(frequencies, axis=0)
    empirical_covariance = (
        centered_frequencies.T @ centered_frequencies / (len(samples) - 1)
    )
    estimated_covariance_mean = np.mean(
        np.asarray(
            [row["frequency_covariance_estimated_unbiased"] for row in samples]
        ),
        axis=0,
    )
    mean_frequency = np.mean(frequencies, axis=0)
    bias = mean_frequency - p
    standard_error_of_mean = np.sqrt(
        np.maximum(np.diag(exact_covariance), 0.0) / len(samples)
    )
    standardized = np.zeros_like(bias)
    nonzero = standard_error_of_mean > 0.0
    standardized[nonzero] = bias[nonzero] / standard_error_of_mean[nonzero]
    metrics = [row["distances_from_generating_distribution"] for row in samples]
    squared_frequency_errors = np.sum((frequencies - p) ** 2, axis=1)
    expected_squared_frequency_error = float(np.trace(exact_covariance))
    features = casimir_features(keys)
    casimir_errors = (frequencies - p) @ features
    squared_casimir_errors = np.sum(casimir_errors**2, axis=1)
    expected_squared_casimir_error = float(
        np.trace(features.T @ exact_covariance @ features)
    )

    def metric_summary(name: str) -> dict[str, float]:
        values = np.asarray([float(row[name]) for row in metrics])
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "quantile_025": float(np.quantile(values, 0.025)),
            "quantile_95": float(np.quantile(values, 0.95)),
            "quantile_975": float(np.quantile(values, 0.975)),
        }

    denominator = float(np.linalg.norm(exact_covariance))
    covariance_relative_error = (
        lambda matrix: float(np.linalg.norm(matrix - exact_covariance) / denominator)
        if denominator > 0.0
        else 0.0
    )
    forbidden_values = [row["forbidden_singlet_closure_mass"] for row in samples]
    payload: dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_repeats": len(samples),
        "seeds": list(seed_values),
        "outcomes": [list(key) for key in keys],
        "generating_probabilities": p.tolist(),
        "sanitation": sanitation,
        "mean_frequencies": mean_frequency.tolist(),
        "frequency_bias": bias.tolist(),
        "maximum_absolute_frequency_bias": float(np.max(np.abs(bias))),
        "maximum_absolute_standardized_frequency_bias": float(
            np.max(np.abs(standardized))
        ),
        "frequency_covariance_exact": exact_covariance.tolist(),
        "frequency_covariance_across_repeats": empirical_covariance.tolist(),
        "mean_unbiased_frequency_covariance_estimate": estimated_covariance_mean.tolist(),
        "empirical_covariance_relative_frobenius_error": covariance_relative_error(
            empirical_covariance
        ),
        "mean_estimated_covariance_relative_frobenius_error": covariance_relative_error(
            estimated_covariance_mean
        ),
        "maximum_empirical_covariance_row_sum": float(
            np.max(np.abs(np.sum(empirical_covariance, axis=1)))
        ),
        "minimum_empirical_covariance_eigenvalue": float(
            np.min(np.linalg.eigvalsh(empirical_covariance))
        ),
        "mean_squared_frequency_error": float(np.mean(squared_frequency_errors)),
        "expected_squared_frequency_error": expected_squared_frequency_error,
        "frequency_mse_to_prediction_ratio": float(
            np.mean(squared_frequency_errors) / expected_squared_frequency_error
        ) if expected_squared_frequency_error > 0.0 else 1.0,
        "mean_squared_casimir_error": float(np.mean(squared_casimir_errors)),
        "expected_squared_casimir_error": expected_squared_casimir_error,
        "casimir_mse_to_prediction_ratio": float(
            np.mean(squared_casimir_errors) / expected_squared_casimir_error
        ) if expected_squared_casimir_error > 0.0 else 1.0,
        "total_variation": metric_summary("total_variation"),
        "hellinger": metric_summary("hellinger"),
        "casimir_error_l2": metric_summary("casimir_error_l2"),
        "forbidden_singlet_closure_mass": (
            None
            if any(value is None for value in forbidden_values)
            else {
                "mean": float(np.mean(forbidden_values)),
                "maximum": float(np.max(forbidden_values)),
            }
        ),
    }
    if keep_raw_counts:
        payload["raw_counts"] = [row["counts"] for row in samples]
    return payload
