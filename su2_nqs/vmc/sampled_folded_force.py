"""Fixed-proposal force for a folded-spectrum objective.

For an exactly normalized autoregressive state ``psi(theta)`` and a real
shift ``sigma``, the folded objective is

``F_sigma(theta) = ||(H - sigma) psi(theta)||^2``.

The implementation differentiates the *residual action* evaluated on a fixed
proposal sample.  Consequently it never forms ``H**2``: the callback may
evaluate ``(H-sigma) psi`` by the same one-hop Hamiltonian connectivity used
for an ordinary local energy.  With exhaustive quadrature the force is the
exact derivative of ``<(H-sigma)^2>``.  With samples from a frozen positive
proposal it is its standard importance-sampled stochastic derivative.

This narrow routine deliberately assumes exact normalization for every
parameter value, as guaranteed by the fusion autoregressive models in this
repository.  It must not be used for an unnormalized ansatz without adding
the derivative of the norm denominator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .sampled_multistate import _normalized_observation_weights


@dataclass(frozen=True)
class SampledFoldedForceReference:
    """Frozen proposal geometry and reference folded expectation."""

    shift: float
    folded_expectation: float
    residual_amplitudes: np.ndarray
    inverse_sqrt_proposal: np.ndarray
    observation_weights: np.ndarray
    sample_count: int
    exact_normalization_required: bool


@dataclass(frozen=True)
class SampledFoldedForce:
    """Folded expectation and its gradient with real model parameters."""

    folded_expectation: float
    gradient: object
    gradient_norm: float
    surrogate_value_at_reference: float
    maximum_reference_action_error: float
    reference: SampledFoldedForceReference


def _readonly(array: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def prepare_sampled_folded_force_reference(
    reference_residual_amplitudes: np.ndarray,
    proposal_probabilities: np.ndarray,
    *,
    shift: float,
    observation_weights: np.ndarray | None = None,
) -> SampledFoldedForceReference:
    """Freeze one importance-sampled ``||(H-sigma) psi||^2`` estimate.

    Ordinary IID draws from ``proposal_probabilities`` use the default
    uniform observation weights.  Exhaustive quadrature passes the normalized
    proposal itself as ``observation_weights``.  Stratified callers pass their
    actual stratum weights, just as in :mod:`sampled_multistate`.
    """

    residual = np.asarray(reference_residual_amplitudes, dtype=np.complex128)
    if residual.ndim != 1 or residual.size < 1:
        raise ValueError("reference_residual_amplitudes must be a nonempty vector")
    if np.any(~np.isfinite(residual)):
        raise ValueError("reference_residual_amplitudes must be finite")
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if proposal.shape != residual.shape:
        raise ValueError("proposal_probabilities must match the residual vector")
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal_probabilities must be finite and positive")
    resolved_shift = float(shift)
    if not np.isfinite(resolved_shift):
        raise ValueError("shift must be finite")
    weights = _normalized_observation_weights(
        residual.size, observation_weights
    )
    scaled = residual / np.sqrt(proposal)
    folded = float(np.sum(weights * np.abs(scaled) ** 2))
    if not np.isfinite(folded) or folded < 0.0:
        raise FloatingPointError("folded expectation is not finite and nonnegative")
    return SampledFoldedForceReference(
        shift=resolved_shift,
        folded_expectation=folded,
        residual_amplitudes=_readonly(
            residual, dtype=np.dtype(np.complex128)
        ),
        inverse_sqrt_proposal=_readonly(
            1.0 / np.sqrt(proposal), dtype=np.dtype(np.float64)
        ),
        observation_weights=_readonly(weights, dtype=np.dtype(np.float64)),
        sample_count=int(residual.size),
        exact_normalization_required=True,
    )


def sampled_folded_spectrum_force(
    params: object,
    reference: SampledFoldedForceReference,
    residual_amplitudes_fn: Callable[[object], jax.Array],
    *,
    maximum_reference_action_error: float = 1.0e-10,
) -> SampledFoldedForce:
    """Differentiate the fixed-proposal folded residual norm.

    ``residual_amplitudes_fn`` returns ``(H-sigma) psi`` only at the observed
    configurations.  It must use the same real ``shift`` stored in
    ``reference``.  The callback is differentiated, while the proposal and
    observation weights are explicitly stopped.  No matrix representing
    ``H**2`` is accepted or constructed by this API.
    """

    if not jax.config.x64_enabled:
        raise RuntimeError("sampled folded-spectrum force requires JAX x64 mode")
    if (
        not np.isfinite(maximum_reference_action_error)
        or maximum_reference_action_error <= 0.0
    ):
        raise ValueError(
            "maximum_reference_action_error must be finite and positive"
        )
    probe = jnp.asarray(residual_amplitudes_fn(params))
    expected = (reference.sample_count,)
    if probe.shape != expected:
        raise ValueError(
            f"residual_amplitudes_fn must return shape {expected}, got {probe.shape}"
        )
    if probe.dtype != jnp.complex128:
        raise TypeError(
            "sampled folded-spectrum force requires complex128 residual amplitudes"
        )
    for leaf in jax.tree_util.tree_leaves(params):
        value = jnp.asarray(leaf)
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(
                "sampled folded-spectrum force requires real-valued parameters"
            )

    observed_probe = np.asarray(probe)
    reference_residual = np.asarray(reference.residual_amplitudes)
    reference_error = float(
        np.max(np.abs(observed_probe - reference_residual), initial=0.0)
    )
    reference_scale = max(
        1.0, float(np.max(np.abs(reference_residual), initial=0.0))
    )
    if (
        not np.isfinite(reference_error)
        or reference_error > maximum_reference_action_error * reference_scale
    ):
        raise ValueError(
            "residual_amplitudes_fn does not reproduce the frozen reference: "
            f"maximum_error={reference_error:.6e}"
        )

    inverse_sqrt = jax.lax.stop_gradient(
        jnp.asarray(reference.inverse_sqrt_proposal)
    )
    weights = jax.lax.stop_gradient(jnp.asarray(reference.observation_weights))

    def surrogate(candidate: object) -> jax.Array:
        residual = jnp.asarray(residual_amplitudes_fn(candidate))
        if residual.shape != expected:
            raise ValueError(
                f"residual_amplitudes_fn must return shape {expected}, got {residual.shape}"
            )
        scaled = residual * inverse_sqrt
        return jnp.sum(weights * jnp.real(jnp.conj(scaled) * scaled))

    def expectation_valued_surrogate(candidate: object) -> jax.Array:
        value = surrogate(candidate)
        return (
            jax.lax.stop_gradient(
                jnp.asarray(reference.folded_expectation, dtype=jnp.float64)
            )
            + value
            - jax.lax.stop_gradient(value)
        )

    folded, gradient = jax.value_and_grad(expectation_valued_surrogate)(params)
    surrogate_value = surrogate(params)
    jax.block_until_ready((folded, gradient, surrogate_value))
    leaves = jax.tree_util.tree_leaves(gradient)
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise FloatingPointError(
            "sampled folded-spectrum force produced a non-finite gradient"
        )
    squared_norm = sum(
        float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
        for leaf in leaves
    )
    return SampledFoldedForce(
        folded_expectation=float(folded),
        gradient=gradient,
        gradient_norm=float(np.sqrt(squared_norm)),
        surrogate_value_at_reference=float(surrogate_value),
        maximum_reference_action_error=reference_error,
        reference=reference,
    )


__all__ = [
    "SampledFoldedForce",
    "SampledFoldedForceReference",
    "prepare_sampled_folded_force_reference",
    "sampled_folded_spectrum_force",
]
