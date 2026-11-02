"""Zero-variance overlap diagonals for normalized multistate estimators.

Normalized autoregressive states satisfy ``<psi_a|psi_a> = 1`` exactly, so
estimating the diagonal of ``S`` adds noise without information.  Off-diagonal
overlaps remain sampled.  This module applies only that honest control
variate and separately verifies the elementary identity

``mean(0.5 * (h_x + h_x^dagger)) == Hermitize(mean(h_x))``.

The identity makes the matrix supplied to Ritz Hermitian sample by sample,
but it does *not* reduce the final estimator variance relative to the existing
post-mean Hermitization.  The raw anti-Hermitian component is retained as a
sampling diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vmc.sampled_multistate import (
    SampledSubspaceMoments,
    sampled_subspace_moments,
)


@dataclass(frozen=True)
class KnownNormalizationMoments:
    """Sampled moments plus exact-diagonal and Hermitian-pair audits."""

    moments: SampledSubspaceMoments
    overlap_diagonal_before: np.ndarray
    overlap_diagonal_after: np.ndarray
    paired_hamiltonian: np.ndarray
    paired_vs_post_mean_maximum_error: float


def _weights(sample_count: int, observation_weights: np.ndarray | None) -> np.ndarray:
    if observation_weights is None:
        return np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
    values = np.asarray(observation_weights, dtype=np.float64)
    if values.shape != (sample_count,) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("observation_weights must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("observation_weights must have positive mass")
    return values / total


def per_observation_hermitian_pair_hamiltonian(
    amplitudes: np.ndarray,
    hamiltonian_actions: np.ndarray,
    proposal_probabilities: np.ndarray,
    *,
    observation_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Average the explicitly Hermitian Hamiltonian integrand.

    This exists as an audit, not a claimed variance reduction: linearity
    makes it equal to Hermitizing the raw sample mean up to roundoff.
    """

    states = np.asarray(amplitudes, dtype=np.complex128)
    actions = np.asarray(hamiltonian_actions, dtype=np.complex128)
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if states.ndim != 2 or actions.shape != states.shape:
        raise ValueError("states and actions must share shape (samples, roots)")
    if proposal.shape != (len(states),) or np.any(proposal <= 0.0):
        raise ValueError("proposal must be positive with one value per sample")
    weights = _weights(len(states), observation_weights)
    raw_rows = (
        states.conj()[:, :, None]
        * actions[:, None, :]
        / proposal[:, None, None]
    )
    paired_rows = 0.5 * (raw_rows + raw_rows.conj().transpose(0, 2, 1))
    return np.einsum("x,xab->ab", weights, paired_rows, optimize=True)


def sampled_moments_with_known_normalization(
    amplitudes: np.ndarray,
    hamiltonian_actions: np.ndarray,
    proposal_probabilities: np.ndarray,
    *,
    exact_norms_squared: np.ndarray | None = None,
    observation_weights: np.ndarray | None = None,
) -> KnownNormalizationMoments:
    """Replace only the known overlap diagonal and audit paired ``H``.

    ``exact_norms_squared`` defaults to ones, which is appropriate only for
    exactly normalized autoregressive roots.  Supplying another positive
    vector supports independent unit tests and normalized non-neural inputs.
    """

    states = np.asarray(amplitudes)
    if states.ndim != 2:
        raise ValueError("amplitudes must have shape (samples, roots)")
    if exact_norms_squared is None:
        norms = np.ones(states.shape[1], dtype=np.float64)
    else:
        norms = np.asarray(exact_norms_squared, dtype=np.float64)
    if (
        norms.shape != (states.shape[1],)
        or np.any(~np.isfinite(norms))
        or np.any(norms <= 0.0)
    ):
        raise ValueError("exact_norms_squared must be finite and positive per root")
    corrected = sampled_subspace_moments(
        amplitudes,
        hamiltonian_actions,
        proposal_probabilities,
        observation_weights=observation_weights,
        exact_overlap_diagonal=norms,
    )
    paired = per_observation_hermitian_pair_hamiltonian(
        amplitudes,
        hamiltonian_actions,
        proposal_probabilities,
        observation_weights=observation_weights,
    )
    pair_error = float(
        np.max(np.abs(paired - corrected.hamiltonian), initial=0.0)
    )
    return KnownNormalizationMoments(
        moments=corrected,
        overlap_diagonal_before=corrected.sampled_overlap_diagonal.copy(),
        overlap_diagonal_after=np.real(np.diag(corrected.overlap)).copy(),
        paired_hamiltonian=np.asarray(paired, dtype=np.complex128),
        paired_vs_post_mean_maximum_error=pair_error,
    )


__all__ = [
    "KnownNormalizationMoments",
    "per_observation_hermitian_pair_hamiltonian",
    "sampled_moments_with_known_normalization",
]
