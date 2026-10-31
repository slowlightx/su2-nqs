"""Horizontal Ky--Fan force for a fixed sampled multistate reference.

The noisy generalized eigensystem is never differentiated.  A reference
sample first fixes its proposal, full-rank Ritz coefficients, projected raw
Hamiltonian, and horizontal residual.  Automatic differentiation then sees
only the trial amplitudes at the observed configurations.  In exhaustive
quadrature this is the Hellmann--Feynman derivative of the mean Ky--Fan
energy; for one normalized state it reduces to the usual centered-local-
energy VMC gradient.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .sampled_multistate import (
    FailClosedGeneralizedRitz,
    SAMPLED_OVERLAP_DIAGONAL,
    SampledPhysicalResiduals,
    SampledSubspaceMoments,
    _scaled_observations,
    _weighted_gram,
    fail_closed_generalized_ritz,
    sampled_physical_residuals,
    sampled_subspace_moments,
)


@dataclass(frozen=True)
class SampledKyFanForceReference:
    """Frozen empirical geometry used by one horizontal-force VJP."""

    mean_ritz_energy: float
    moments: SampledSubspaceMoments
    ritz: FailClosedGeneralizedRitz
    physical_residuals: SampledPhysicalResiduals
    inverse_sqrt_proposal: np.ndarray
    observation_weights: np.ndarray
    coefficients: np.ndarray
    projected_raw_hamiltonian: np.ndarray
    horizontal_residual_scaled: np.ndarray
    maximum_horizontality_error: float
    overlap_diagonal_policy: str
    exact_normalized_overlap_diagonal: bool
    sample_count: int
    state_count: int


@dataclass(frozen=True)
class SampledKyFanHorizontalForce:
    """Energy-valued surrogate result and its real-parameter gradient."""

    energy: float
    gradient: object
    gradient_norm: float
    surrogate_value_at_reference: float
    reference: SampledKyFanForceReference


def _readonly(array: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def prepare_sampled_ky_fan_force_reference(
    reference_log_amplitudes: np.ndarray,
    hamiltonian_actions: np.ndarray,
    proposal_probabilities: np.ndarray,
    *,
    observation_weights: np.ndarray | None = None,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float | None = None,
    maximum_orthonormality_error: float = 1.0e-8,
    maximum_horizontality_error: float = 1.0e-10,
    exact_normalized_overlap_diagonal: bool = False,
) -> SampledKyFanForceReference:
    """Freeze a full-rank sampled Ritz geometry and its horizontal residual.

    ``reference_log_amplitudes`` and ``hamiltonian_actions`` have shape
    ``(samples, states)``.  The latter contains the amplitudes ``H psi_a`` and
    not local-energy ratios.  Rank and conditioning failures propagate from
    :func:`fail_closed_generalized_ritz` before any gradient can be formed.

    The generic default keeps the historical fully sampled overlap matrix.
    Exact normalized overlap diagonals remain available to the standalone
    moment/Ritz layer, but are deliberately unavailable in this force helper.
    Replacing only ``S_aa`` changes the differential of the overlap estimator:
    the historical horizontal residual is centered in the raw sampled metric,
    not in that corrected metric.  Combining the two produces a nonzero force
    under independent global phase changes of the roots.  Until a separately
    calibrated corrected-gradient estimator is available, requesting that
    combination therefore fails closed before any force reference is built.
    """

    logs = np.asarray(reference_log_amplitudes, dtype=np.complex128)
    if logs.ndim != 2 or logs.shape[0] < 1 or logs.shape[1] < 1:
        raise ValueError(
            "reference_log_amplitudes must be a nonempty sample-by-state matrix"
        )
    if np.any(~np.isfinite(logs)):
        raise ValueError("reference_log_amplitudes must be finite")
    if not isinstance(exact_normalized_overlap_diagonal, (bool, np.bool_)):
        raise TypeError("exact_normalized_overlap_diagonal must be boolean")
    if exact_normalized_overlap_diagonal:
        raise ValueError(
            "exact normalized overlap diagonals are available for sampled "
            "moments/Ritz only, not the Ky-Fan force: the corrected force "
            "estimator has not passed independent-root phase-gauge validation"
        )
    if (
        not np.isfinite(maximum_horizontality_error)
        or maximum_horizontality_error <= 0.0
    ):
        raise ValueError("maximum_horizontality_error must be finite and positive")
    amplitudes = np.exp(logs)
    moments = sampled_subspace_moments(
        amplitudes,
        hamiltonian_actions,
        proposal_probabilities,
        observation_weights=observation_weights,
        exact_overlap_diagonal=None,
    )
    ritz = fail_closed_generalized_ritz(
        moments,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number,
        maximum_orthonormality_error=maximum_orthonormality_error,
    )
    physical_residuals = sampled_physical_residuals(
        amplitudes,
        hamiltonian_actions,
        proposal_probabilities,
        ritz,
        observation_weights=observation_weights,
    )
    scaled_states, scaled_actions, weights = _scaled_observations(
        amplitudes,
        hamiltonian_actions,
        proposal_probabilities,
        observation_weights,
    )
    ritz_states = scaled_states @ ritz.coefficients
    ritz_actions = scaled_actions @ ritz.coefficients
    projected_raw = _weighted_gram(ritz_states, ritz_actions, weights)
    horizontal_residual = ritz_actions - ritz_states @ projected_raw
    projected_horizontal = _weighted_gram(
        ritz_states, horizontal_residual, weights
    )
    horizontality_error = float(
        np.max(np.abs(projected_horizontal), initial=0.0)
    )
    if (
        not np.all(np.isfinite(projected_raw))
        or not np.all(np.isfinite(horizontal_residual))
        or not np.isfinite(horizontality_error)
        or horizontality_error > maximum_horizontality_error
    ):
        raise FloatingPointError(
            "sampled horizontal projection failed: "
            f"error={horizontality_error:.6e}"
        )
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    return SampledKyFanForceReference(
        mean_ritz_energy=float(np.mean(ritz.energies)),
        moments=moments,
        ritz=ritz,
        physical_residuals=physical_residuals,
        inverse_sqrt_proposal=_readonly(
            1.0 / np.sqrt(proposal), dtype=np.dtype(np.float64)
        ),
        observation_weights=_readonly(weights, dtype=np.dtype(np.float64)),
        coefficients=_readonly(
            ritz.coefficients, dtype=np.dtype(np.complex128)
        ),
        projected_raw_hamiltonian=_readonly(
            projected_raw, dtype=np.dtype(np.complex128)
        ),
        horizontal_residual_scaled=_readonly(
            horizontal_residual, dtype=np.dtype(np.complex128)
        ),
        maximum_horizontality_error=horizontality_error,
        overlap_diagonal_policy=moments.overlap_diagonal_policy,
        exact_normalized_overlap_diagonal=bool(
            exact_normalized_overlap_diagonal
        ),
        sample_count=int(logs.shape[0]),
        state_count=int(logs.shape[1]),
    )


def sampled_ky_fan_horizontal_force(
    params: object,
    reference: SampledKyFanForceReference,
    log_amplitudes_fn: Callable[[object], jax.Array],
) -> SampledKyFanHorizontalForce:
    """Differentiate the fixed-reference horizontal Ky--Fan surrogate.

    The callback returns all observed log amplitudes with shape
    ``(reference.sample_count, reference.state_count)``.  Parameters are
    required to be real because this is the VMC convention used throughout
    the project.  Every frozen reference array is wrapped in
    :func:`jax.lax.stop_gradient` explicitly.
    """

    if (
        reference.exact_normalized_overlap_diagonal
        or reference.overlap_diagonal_policy != SAMPLED_OVERLAP_DIAGONAL
    ):
        raise ValueError(
            "sampled Ky-Fan force requires the raw sampled overlap metric; "
            "exact-diagonal references are Ritz-only"
        )
    if not jax.config.x64_enabled:
        raise RuntimeError("sampled Ky-Fan force requires JAX x64 mode")
    probe = jnp.asarray(log_amplitudes_fn(params))
    expected = (reference.sample_count, reference.state_count)
    if probe.shape != expected:
        raise ValueError(
            f"log_amplitudes_fn must return shape {expected}, got {probe.shape}"
        )
    if probe.dtype != jnp.complex128:
        raise TypeError("sampled Ky-Fan force requires complex128 log amplitudes")
    for leaf in jax.tree_util.tree_leaves(params):
        value = jnp.asarray(leaf)
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("sampled Ky-Fan force requires real-valued parameters")

    inverse_sqrt = jax.lax.stop_gradient(
        jnp.asarray(reference.inverse_sqrt_proposal)
    )
    weights = jax.lax.stop_gradient(jnp.asarray(reference.observation_weights))
    coefficients = jax.lax.stop_gradient(jnp.asarray(reference.coefficients))
    horizontal_residual = jax.lax.stop_gradient(
        jnp.asarray(reference.horizontal_residual_scaled)
    )
    root_count = reference.state_count

    def surrogate(candidate: object) -> jax.Array:
        logs = jnp.asarray(log_amplitudes_fn(candidate))
        if logs.shape != expected:
            raise ValueError(
                f"log_amplitudes_fn must return shape {expected}, got {logs.shape}"
            )
        scaled_states = jnp.exp(logs) * inverse_sqrt[:, None]
        ritz_states = scaled_states @ coefficients
        return (2.0 / root_count) * jnp.real(
            jnp.sum(
                weights[:, None]
                * jnp.conj(ritz_states)
                * horizontal_residual
            )
        )

    def energy_valued_surrogate(candidate: object) -> jax.Array:
        force_value = surrogate(candidate)
        return (
            jax.lax.stop_gradient(
                jnp.asarray(reference.mean_ritz_energy, dtype=jnp.float64)
            )
            + force_value
            - jax.lax.stop_gradient(force_value)
        )

    energy, gradient = jax.value_and_grad(energy_valued_surrogate)(params)
    surrogate_value = surrogate(params)
    jax.block_until_ready((energy, gradient, surrogate_value))
    leaves = jax.tree_util.tree_leaves(gradient)
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise FloatingPointError("sampled Ky-Fan force produced a non-finite gradient")
    squared_norm = sum(
        float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2)) for leaf in leaves
    )
    return SampledKyFanHorizontalForce(
        energy=float(energy),
        gradient=gradient,
        gradient_norm=float(np.sqrt(squared_norm)),
        surrogate_value_at_reference=float(surrogate_value),
        reference=reference,
    )


__all__ = [
    "SampledKyFanForceReference",
    "SampledKyFanHorizontalForce",
    "prepare_sampled_ky_fan_force_reference",
    "sampled_ky_fan_horizontal_force",
]
