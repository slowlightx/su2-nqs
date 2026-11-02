"""Invariant summaries of a Hamiltonian perturbation along a fusion filtration.

For a normalized state and scalar observable, the exact Doob increments of
the local estimator are orthogonal.  Their squared norms therefore define a
positive, additive distribution over autoregressive cuts.  This module turns
that distribution into small coordinate diagnostics without introducing a
learned model or a tunable channel weight.

The primary quantity is the participation ratio

``P_flow = (sum_k w_k)**2 / sum_k w_k**2``.

It counts the effective number of fusion cuts carrying the observable
response.  It is a property of the state, observable, and chosen physical
site ordering; it is not a thermodynamic order parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FusionFlowProfile:
    """Positive cut distribution and its effective support diagnostics."""

    cut_variances: np.ndarray
    normalized_cut_weights: np.ndarray
    total_variance: float
    participation_ratio: float
    normalized_participation_ratio: float
    entropy_dimension: float
    normalized_entropy_dimension: float
    peak_cut: int
    peak_fraction: float
    weighted_cut: float
    normalized_weighted_cut: float
    early_middle_late_fractions: tuple[float, float, float]


@dataclass(frozen=True)
class DoobCrossCovarianceProfile:
    """Cut-resolved covariance and imaginary-time response of two observables."""

    cut_covariances: np.ndarray
    cut_responses: np.ndarray
    normalized_absolute_response: np.ndarray
    first_cut_variances: np.ndarray
    second_cut_variances: np.ndarray
    direct_covariance: complex
    explained_covariance: complex
    total_response: float
    absolute_response: float
    signed_coherence: float
    response_participation_ratio: float
    flow_response_bhattacharyya: float
    h2_majority_cut_count: int
    h2_majority_response_capture: float
    maximum_reconstruction_error: float
    maximum_cross_cut_covariance: float
    covariance_additivity_error: float
    variance_additivity_error: float


def fusion_flow_profile(
    cut_variances: Sequence[float] | np.ndarray,
    *,
    total_variance: float | None = None,
    tolerance: float = 2.0e-12,
) -> FusionFlowProfile:
    """Summarize non-negative orthogonal flow weights over fusion cuts.

    ``cut_variances[k]`` is ``||Delta_k||**2`` for the exact conditional
    martingale increment at cut ``k``.  Tiny negative roundoff is clipped;
    a materially negative value or a mismatch with ``total_variance`` is an
    invariant failure.
    """

    values = np.asarray(cut_variances, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("cut_variances must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(values)):
        raise ValueError("cut_variances must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    if float(np.min(values)) < -tolerance * scale:
        raise ValueError("cut variances must be non-negative")
    values = np.maximum(values, 0.0)
    summed = float(np.sum(values))
    if total_variance is None:
        total = summed
    else:
        total = float(total_variance)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("total_variance must be positive and finite")
        if abs(summed - total) > tolerance * max(1.0, total):
            raise ValueError("cut weights do not add to total_variance")
    if total <= 0.0:
        raise ValueError("at least one cut must carry positive variance")

    probabilities = values / total
    inverse_participation = float(np.sum(probabilities**2))
    participation = 1.0 / inverse_participation
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    entropy_dimension = float(np.exp(entropy))
    active_cut_count = max(int(np.count_nonzero(positive)), 1)
    indices = np.arange(len(values), dtype=np.float64)
    thirds = np.array_split(np.arange(len(values)), 3)
    fractions = tuple(float(np.sum(probabilities[index])) for index in thirds)
    weighted = float(np.dot(indices, probabilities))
    return FusionFlowProfile(
        cut_variances=values,
        normalized_cut_weights=probabilities,
        total_variance=total,
        participation_ratio=participation,
        normalized_participation_ratio=participation / active_cut_count,
        entropy_dimension=entropy_dimension,
        normalized_entropy_dimension=entropy_dimension / active_cut_count,
        peak_cut=int(np.argmax(values)),
        peak_fraction=float(np.max(probabilities)),
        weighted_cut=weighted,
        normalized_weighted_cut=(
            weighted / (len(values) - 1) if len(values) > 1 else 0.0
        ),
        early_middle_late_fractions=fractions,
    )


def jensen_shannon_divergence(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float:
    """Return the natural-log Jensen--Shannon divergence of two cut profiles."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or len(left) < 1:
        raise ValueError("profiles must be nonempty one-dimensional arrays of equal shape")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("profiles must be finite")
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("profiles must be non-negative")
    left_total = float(np.sum(left))
    right_total = float(np.sum(right))
    if left_total <= 0.0 or right_total <= 0.0:
        raise ValueError("profiles must have positive total weight")
    left = left / left_total
    right = right / right_total
    midpoint = 0.5 * (left + right)

    def relative_entropy(probability: np.ndarray) -> float:
        support = probability > 0.0
        return float(
            np.sum(probability[support] * np.log(probability[support] / midpoint[support]))
        )

    return 0.5 * (relative_entropy(left) + relative_entropy(right))


def _normalized_weights(values: np.ndarray, size: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (size,) or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite, non-negative, and match values")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("probabilities must have positive total")
    return weights / total


def _conditional_values(
    values: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray,
    prefix_length: int,
) -> np.ndarray:
    totals: dict[tuple[int, ...], tuple[float, complex]] = {}
    for action, value, probability in zip(actions, values, probabilities, strict=True):
        key = tuple(int(item) for item in action[:prefix_length])
        mass, numerator = totals.get(key, (0.0, 0.0j))
        totals[key] = (mass + float(probability), numerator + probability * value)
    means = {key: numerator / mass for key, (mass, numerator) in totals.items()}
    return np.asarray(
        [means[tuple(int(item) for item in action[:prefix_length])] for action in actions],
        dtype=np.complex128,
    )


def doob_cross_covariance_profile(
    probabilities: Sequence[float] | np.ndarray,
    first_local_values: Sequence[complex] | np.ndarray,
    second_local_values: Sequence[complex] | np.ndarray,
    action_sequences: np.ndarray,
    *,
    tolerance: float = 2.0e-12,
) -> DoobCrossCovarianceProfile:
    """Resolve ``Cov(first,second)`` into orthogonal fusion-cut increments.

    For local estimators of Hermitian operators ``O`` and ``H2``, setting
    ``first=O`` and ``second=H2`` gives the normalized imaginary-time response
    ``d<O>/d tau = -2 Re Cov(O,H2)``.  The response can have either sign at a
    cut, so both signed coherence and an absolute response distribution are
    reported.  ``H2`` in the field names denotes the second argument.
    """

    first = np.asarray(first_local_values, dtype=np.complex128)
    second = np.asarray(second_local_values, dtype=np.complex128)
    actions = np.asarray(action_sequences)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("local estimators must be equal one-dimensional arrays")
    if actions.ndim != 2 or len(actions) != len(first) or actions.shape[1] < 1:
        raise ValueError("action_sequences must have one nonempty row per value")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("action sequences must contain integers")
    if len({tuple(int(item) for item in row) for row in actions}) != len(actions):
        raise ValueError("complete action sequences must identify states uniquely")
    weights = _normalized_weights(np.asarray(probabilities), len(first))
    active = weights > 0.0
    if np.any(~np.isfinite(first[active])) or np.any(~np.isfinite(second[active])):
        raise ValueError("local estimators must be finite on positive-weight states")
    first = np.where(active, first, 0.0j)
    second = np.where(active, second, 0.0j)

    cuts = actions.shape[1]
    first_messages = np.column_stack(
        [_conditional_values(first, actions, weights, prefix) for prefix in range(cuts + 1)]
    )
    second_messages = np.column_stack(
        [_conditional_values(second, actions, weights, prefix) for prefix in range(cuts + 1)]
    )
    first_increments = np.diff(first_messages, axis=1)
    second_increments = np.diff(second_messages, axis=1)
    cross = np.einsum(
        "s,sk,sl->kl", weights, np.conjugate(first_increments), second_increments
    )
    cut_covariances = np.diag(cross)
    first_variances = np.einsum(
        "s,sk,sk->k", weights, np.conjugate(first_increments), first_increments
    ).real
    second_variances = np.einsum(
        "s,sk,sk->k", weights, np.conjugate(second_increments), second_increments
    ).real
    centered_first = first - weights @ first
    centered_second = second - weights @ second
    direct_covariance = complex(
        np.sum(weights * np.conjugate(centered_first) * centered_second)
    )
    explained_covariance = complex(np.sum(cut_covariances))
    cut_responses = -2.0 * np.real(cut_covariances)
    absolute = np.abs(cut_responses)
    absolute_total = float(np.sum(absolute))
    if absolute_total <= tolerance:
        raise ValueError("the two observables have no resolved real response")
    normalized_response = absolute / absolute_total
    response_participation = float(1.0 / np.sum(normalized_response**2))
    second_total = float(np.sum(second_variances))
    if second_total <= tolerance:
        raise ValueError("the second observable has zero variance")
    second_weights = second_variances / second_total
    overlap = float(np.sum(np.sqrt(second_weights * normalized_response)))
    order = np.argsort(-second_weights, kind="stable")
    cumulative = np.cumsum(second_weights[order])
    count = int(np.searchsorted(cumulative, 0.5, side="left") + 1)
    dominant = order[:count]
    capture = float(np.sum(normalized_response[dominant]))

    off_diagonal = cross - np.diag(cut_covariances)
    reconstruction = max(
        float(np.max(np.abs(first_messages[:, -1] - first), initial=0.0)),
        float(np.max(np.abs(second_messages[:, -1] - second), initial=0.0)),
        float(
            np.max(
                np.abs(np.sum(first_increments, axis=1) - centered_first),
                initial=0.0,
            )
        ),
        float(
            np.max(
                np.abs(np.sum(second_increments, axis=1) - centered_second),
                initial=0.0,
            )
        ),
    )
    variance_error = max(
        abs(float(np.sum(weights * np.abs(centered_first) ** 2)) - float(np.sum(first_variances))),
        abs(float(np.sum(weights * np.abs(centered_second) ** 2)) - second_total),
    )
    total_response = float(np.sum(cut_responses))
    return DoobCrossCovarianceProfile(
        cut_covariances=np.asarray(cut_covariances, dtype=np.complex128),
        cut_responses=np.asarray(cut_responses, dtype=np.float64),
        normalized_absolute_response=np.asarray(normalized_response, dtype=np.float64),
        first_cut_variances=np.asarray(first_variances, dtype=np.float64),
        second_cut_variances=np.asarray(second_variances, dtype=np.float64),
        direct_covariance=direct_covariance,
        explained_covariance=explained_covariance,
        total_response=total_response,
        absolute_response=absolute_total,
        signed_coherence=float(abs(total_response) / absolute_total),
        response_participation_ratio=response_participation,
        flow_response_bhattacharyya=overlap,
        h2_majority_cut_count=count,
        h2_majority_response_capture=capture,
        maximum_reconstruction_error=reconstruction,
        maximum_cross_cut_covariance=float(np.max(np.abs(off_diagonal), initial=0.0)),
        covariance_additivity_error=float(abs(direct_covariance - explained_covariance)),
        variance_additivity_error=float(variance_error),
    )


__all__ = [
    "DoobCrossCovarianceProfile",
    "FusionFlowProfile",
    "doob_cross_covariance_profile",
    "fusion_flow_profile",
    "jensen_shannon_divergence",
]
