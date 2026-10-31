"""Bounded-weight bridge estimators between two normalized Born states."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class BornMixtureBridgeObservations(NamedTuple):
    """Per-sample observations under ``q = (p_candidate + p_current) / 2``."""

    energy_difference: np.ndarray
    forward_kl: np.ndarray
    reverse_kl: np.ndarray
    candidate_weight: np.ndarray
    current_weight: np.ndarray


def born_mixture_bridge_observations(
    candidate_local_energies: object,
    current_local_energies: object,
    candidate_log_probabilities: object,
    current_log_probabilities: object,
) -> BornMixtureBridgeObservations:
    """Return unbiased current/candidate comparisons on a common mixture.

    Samples must be drawn from
    ``q(x) = 0.5 * (p_candidate(x) + p_current(x))``.  The likelihood ratios
    ``p_candidate/q`` and ``p_current/q`` are pointwise bounded by two, unlike
    a fixed-proposal importance ratio after an outer optimization step.  The
    real-energy difference and both directional Born KL divergences are then
    expectations of the returned observations under ``q``.
    """

    candidate_local = np.asarray(candidate_local_energies, dtype=np.complex128)
    current_local = np.asarray(current_local_energies, dtype=np.complex128)
    candidate_log = np.asarray(candidate_log_probabilities, dtype=np.float64)
    current_log = np.asarray(current_log_probabilities, dtype=np.float64)
    shape = candidate_local.shape
    values = (current_local, candidate_log, current_log)
    if candidate_local.ndim != 1 or any(value.shape != shape for value in values):
        raise ValueError("Born-mixture bridge inputs must be matching vectors")
    if len(candidate_local) < 1 or any(np.any(~np.isfinite(value)) for value in values):
        raise ValueError("Born-mixture bridge inputs must be finite and nonempty")

    log_mixture = np.logaddexp(candidate_log, current_log) - np.log(2.0)
    candidate_weight = np.exp(candidate_log - log_mixture)
    current_weight = np.exp(current_log - log_mixture)
    if (
        np.any(candidate_weight < 0.0)
        or np.any(current_weight < 0.0)
        or np.any(candidate_weight > 2.0 * (1.0 + 1.0e-12))
        or np.any(current_weight > 2.0 * (1.0 + 1.0e-12))
    ):
        raise FloatingPointError("Born-mixture bridge weights escaped [0,2]")
    log_ratio = candidate_log - current_log
    return BornMixtureBridgeObservations(
        energy_difference=(
            candidate_weight * np.real(candidate_local)
            - current_weight * np.real(current_local)
        ),
        forward_kl=candidate_weight * log_ratio,
        reverse_kl=-current_weight * log_ratio,
        candidate_weight=candidate_weight,
        current_weight=current_weight,
    )


__all__ = [
    "BornMixtureBridgeObservations",
    "born_mixture_bridge_observations",
]
