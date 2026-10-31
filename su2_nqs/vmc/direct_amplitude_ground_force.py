"""Node-safe ground-state force for a sampled direct-amplitude carrier bank.

The carrier bank may contain an immutable reference column and any number of
trainable exact-sector columns.  A generalized Ritz solve chooses the lowest
linear combination.  The nonlinear parameters are then updated with the
Hellmann--Feynman residual while the Ritz coefficients are held fixed.  No
``log(psi)`` or local-energy division is formed, so projected nodes are
admissible.

At finite sample count the raw Hamiltonian moment is only approximately
Hermitian.  As in the multistate force, the residual is projected away from
the complete sampled bank.  This removes pure column mixing and scale gauges
from the nonlinear force.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .resonating_projected_agp import (
    ResonatingProjectedAGPAction,
    solve_three_root_resonating_subspace,
)


@dataclass(frozen=True)
class DirectAmplitudeGroundForceReference:
    """Frozen sampled geometry for one horizontal ground-state force."""

    energy: float
    variance: float
    amplitudes: np.ndarray
    measure: np.ndarray
    column_rescales: np.ndarray
    scaled_coefficients: np.ndarray
    root_amplitude: np.ndarray
    root_action: np.ndarray
    horizontal_residual: np.ndarray
    maximum_horizontality_error: float
    state_weight_ess_fraction: float
    overlap_condition_number: float
    sample_count: int
    bank_size: int


@dataclass(frozen=True)
class DirectAmplitudeGroundForce:
    """Energy-valued surrogate and gradient over real nonlinear parameters."""

    energy: float
    gradient: object
    gradient_norm: float
    surrogate_value_at_reference: float
    reference: DirectAmplitudeGroundForceReference


def _observation_weights(
    sample_count: int,
    observation_weights: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    if observation_weights is None:
        return np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
    values = np.asarray(observation_weights, dtype=np.float64)
    if values.shape != (sample_count,):
        raise ValueError("observation_weights must have one entry per sample")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("observation_weights must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("observation_weights must have positive finite total")
    return values / total


def _importance_measure(proposal: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if proposal.shape != weights.shape or np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal_probabilities must be finite and positive")
    with np.errstate(divide="ignore"):
        log_measure = np.where(weights > 0.0, np.log(weights) - np.log(proposal), -np.inf)
    maximum = float(np.max(log_measure))
    if not np.isfinite(maximum):
        raise ValueError("the sampled quadrature has no positive row")
    return np.where(np.isfinite(log_measure), np.exp(log_measure - maximum), 0.0)


def prepare_direct_amplitude_ground_force_reference(
    action: ResonatingProjectedAGPAction,
    proposal_probabilities: np.ndarray | Sequence[float],
    *,
    observation_weights: np.ndarray | Sequence[float] | None = None,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float = 1.0e8,
    maximum_orthonormality_error: float = 1.0e-8,
    maximum_horizontality_error: float = 1.0e-10,
) -> DirectAmplitudeGroundForceReference:
    """Solve the sampled lowest Ritz root and freeze its horizontal residual."""

    amplitudes = np.asarray(action.amplitudes, dtype=np.complex128)
    actions = np.asarray(action.hamiltonian_actions, dtype=np.complex128)
    if amplitudes.ndim != 2 or amplitudes.shape[0] < 1 or amplitudes.shape[1] < 3:
        raise ValueError("the ground-force bank must have at least three columns")
    if actions.shape != amplitudes.shape:
        raise ValueError("Hamiltonian actions must match bank amplitudes")
    sample_count, bank_size = amplitudes.shape
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    weights = _observation_weights(sample_count, observation_weights)
    measure = _importance_measure(proposal, weights)
    ritz = solve_three_root_resonating_subspace(
        action,
        proposal,
        observation_weights=weights,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number,
        maximum_orthonormality_error=maximum_orthonormality_error,
    )
    root = np.asarray(ritz.root_amplitudes[:, 0], dtype=np.complex128)
    root_action = np.asarray(ritz.root_actions[:, 0], dtype=np.complex128)
    scaled_amplitudes = amplitudes / ritz.column_rescales[None, :]
    carrier_to_action = np.einsum(
        "x,xa,x->a", measure, scaled_amplitudes.conj(), root_action, optimize=True
    )
    try:
        projection = np.linalg.solve(ritz.overlap, carrier_to_action)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("ground-force carrier projection failed") from error
    horizontal = root_action - scaled_amplitudes @ projection
    projected = np.einsum(
        "x,xa,x->a", measure, scaled_amplitudes.conj(), horizontal, optimize=True
    )
    horizontal_error = float(np.max(np.abs(projected), initial=0.0))
    if not np.isfinite(horizontal_error) or horizontal_error > maximum_horizontality_error:
        raise FloatingPointError(
            "ground-force horizontal projection failed: "
            f"error={horizontal_error:.6e}"
        )
    root_norm = float(np.sum(measure * np.abs(root) ** 2))
    energy = complex(np.sum(measure * np.conj(root) * root_action) / root_norm)
    variance = float(
        np.sum(measure * np.abs(root_action - energy * root) ** 2) / root_norm
    )
    state_weights = measure * np.abs(root) ** 2
    ess = float(np.sum(state_weights) ** 2 / np.sum(state_weights**2))
    return DirectAmplitudeGroundForceReference(
        energy=float(energy.real),
        variance=variance,
        amplitudes=np.array(amplitudes, copy=True),
        measure=np.array(measure, copy=True),
        column_rescales=np.array(ritz.column_rescales, copy=True),
        scaled_coefficients=np.array(ritz.scaled_coefficients[:, 0], copy=True),
        root_amplitude=np.array(root, copy=True),
        root_action=np.array(root_action, copy=True),
        horizontal_residual=np.array(horizontal, copy=True),
        maximum_horizontality_error=horizontal_error,
        state_weight_ess_fraction=ess / sample_count,
        overlap_condition_number=ritz.overlap_condition_number,
        sample_count=sample_count,
        bank_size=bank_size,
    )


def direct_amplitude_ground_horizontal_force(
    parameters: object,
    reference: DirectAmplitudeGroundForceReference,
    amplitudes_fn: Callable[[object], jax.Array],
) -> DirectAmplitudeGroundForce:
    """Differentiate the frozen lowest-root horizontal-force surrogate."""

    expected = (reference.sample_count, reference.bank_size)
    probe = jnp.asarray(amplitudes_fn(parameters))
    if probe.shape != expected:
        raise ValueError(f"amplitudes_fn must return shape {expected}")
    probe_host = np.asarray(probe, dtype=np.complex128)
    scale = max(float(np.max(np.abs(reference.amplitudes), initial=0.0)), 1.0)
    if float(np.max(np.abs(probe_host - reference.amplitudes), initial=0.0)) / scale > 1.0e-11:
        raise ValueError("amplitudes_fn(parameters) does not reproduce the reference")
    for leaf in jax.tree_util.tree_leaves(parameters):
        if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating):
            raise TypeError("ground force requires real nonlinear parameters")

    measure = jax.lax.stop_gradient(jnp.asarray(reference.measure))
    rescales = jax.lax.stop_gradient(jnp.asarray(reference.column_rescales))
    coefficients = jax.lax.stop_gradient(jnp.asarray(reference.scaled_coefficients))
    horizontal = jax.lax.stop_gradient(jnp.asarray(reference.horizontal_residual))

    def surrogate(candidate: object) -> jax.Array:
        bank = jnp.asarray(amplitudes_fn(candidate))
        root = (bank / rescales[None, :]) @ coefficients
        return 2.0 * jnp.real(jnp.sum(measure * jnp.conj(root) * horizontal))

    def energy_valued(candidate: object) -> jax.Array:
        value = surrogate(candidate)
        return jax.lax.stop_gradient(jnp.asarray(reference.energy)) + value - jax.lax.stop_gradient(value)

    energy, gradient = jax.value_and_grad(energy_valued)(parameters)
    surrogate_value = surrogate(parameters)
    jax.block_until_ready((energy, gradient, surrogate_value))
    leaves = jax.tree_util.tree_leaves(gradient)
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise FloatingPointError("direct-amplitude ground force is non-finite")
    norm = np.sqrt(
        sum(float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2)) for leaf in leaves)
    )
    return DirectAmplitudeGroundForce(
        energy=float(energy),
        gradient=gradient,
        gradient_norm=float(norm),
        surrogate_value_at_reference=float(surrogate_value),
        reference=reference,
    )


__all__ = [
    "DirectAmplitudeGroundForce",
    "DirectAmplitudeGroundForceReference",
    "direct_amplitude_ground_horizontal_force",
    "prepare_direct_amplitude_ground_force_reference",
]
