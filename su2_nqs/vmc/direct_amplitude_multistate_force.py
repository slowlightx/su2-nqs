"""Node-safe horizontal Ky--Fan force for direct-amplitude carrier banks.

Strict symmetry projections can create exact wavefunction nodes.  This module
therefore never forms ``log(psi)`` or a local-energy ratio.  A frozen sampled
reference supplies direct carrier amplitudes ``Phi`` and actions ``H Phi``;
automatic differentiation sees only the candidate direct amplitudes.  In
exhaustive quadrature the resulting vector-Jacobian product is the exact
Hellmann--Feynman derivative of the mean of the lowest three Ritz values.

Finite-sample Hamiltonian moments need not be exactly Hermitian.  The residual
is therefore projected away from the complete sampled carrier span, not only
the three occupied Ritz roots.  This keeps independent carrier basis/scale
directions from becoming spurious optimization forces when the bank contains
more than three columns.
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
    ThreeRootResonatingResult,
    solve_three_root_resonating_subspace,
)


@dataclass(frozen=True)
class DirectAmplitudeKyFanForceReference:
    """Frozen empirical geometry for one node-safe horizontal-force VJP."""

    mean_ritz_energy: float
    ritz: ThreeRootResonatingResult
    amplitudes: np.ndarray
    measure: np.ndarray
    column_rescales: np.ndarray
    scaled_coefficients: np.ndarray
    projected_raw_hamiltonian: np.ndarray
    carrier_projection_coefficients: np.ndarray
    horizontal_residual: np.ndarray
    maximum_horizontality_error: float
    relative_ritz_boundary_gap: float | None
    sample_count: int
    bank_size: int
    root_count: int = 3


@dataclass(frozen=True)
class DirectAmplitudeKyFanForce:
    """Energy-valued surrogate and its gradient over real carrier parameters."""

    energy: float
    gradient: object
    gradient_norm: float
    surrogate_value_at_reference: float
    reference: DirectAmplitudeKyFanForceReference


def _readonly(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _normalized_observation_weights(
    sample_count: int,
    observation_weights: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    if observation_weights is None:
        return np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
    values = np.asarray(observation_weights, dtype=np.float64)
    if values.shape != (sample_count,):
        raise ValueError("observation_weights must have one entry per row")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("observation_weights must be finite and nonnegative")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("observation_weights must have positive total")
    return values / total


def _rescaled_importance_measure(
    proposal_probabilities: np.ndarray | Sequence[float],
    observation_weights: np.ndarray,
) -> np.ndarray:
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if proposal.shape != observation_weights.shape:
        raise ValueError("proposal_probabilities must have one entry per row")
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal probabilities must be finite and positive")
    with np.errstate(divide="ignore"):
        log_measure = np.where(
            observation_weights > 0.0,
            np.log(observation_weights) - np.log(proposal),
            -np.inf,
        )
    maximum = float(np.max(log_measure))
    if not np.isfinite(maximum):
        raise ValueError("the supplied quadrature has no positive observation row")
    return np.where(np.isfinite(log_measure), np.exp(log_measure - maximum), 0.0)


def prepare_direct_amplitude_ky_fan_force_reference(
    action: ResonatingProjectedAGPAction,
    proposal_probabilities: np.ndarray | Sequence[float],
    *,
    observation_weights: np.ndarray | Sequence[float] | None = None,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float = 1.0e8,
    maximum_orthonormality_error: float = 1.0e-8,
    maximum_horizontality_error: float = 1.0e-10,
    minimum_relative_ritz_boundary_gap: float = 1.0e-10,
) -> DirectAmplitudeKyFanForceReference:
    """Solve and freeze a lowest-three-root direct-amplitude reference.

    ``observation_weights=None`` is the IID Monte Carlo contract.  Exhaustive
    quadrature should pass the normalized proposal itself as the observation
    weights.  Exact zeros in ``action.amplitudes`` are explicitly admissible.
    """

    amplitudes = np.asarray(action.amplitudes, dtype=np.complex128)
    actions = np.asarray(action.hamiltonian_actions, dtype=np.complex128)
    if amplitudes.ndim != 2 or amplitudes.shape[0] < 1:
        raise ValueError("action amplitudes must be a nonempty sample-by-bank matrix")
    if actions.shape != amplitudes.shape:
        raise ValueError("Hamiltonian actions must match carrier amplitudes")
    if not np.isfinite(maximum_horizontality_error) or maximum_horizontality_error <= 0.0:
        raise ValueError("maximum_horizontality_error must be finite and positive")
    if (
        not np.isfinite(minimum_relative_ritz_boundary_gap)
        or minimum_relative_ritz_boundary_gap <= 0.0
    ):
        raise ValueError(
            "minimum_relative_ritz_boundary_gap must be finite and positive"
        )

    sample_count, bank_size = amplitudes.shape
    weights = _normalized_observation_weights(sample_count, observation_weights)
    measure = _rescaled_importance_measure(proposal_probabilities, weights)
    ritz = solve_three_root_resonating_subspace(
        action,
        proposal_probabilities,
        observation_weights=weights,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number,
        maximum_orthonormality_error=maximum_orthonormality_error,
    )
    root_states = np.asarray(ritz.root_amplitudes, dtype=np.complex128)
    root_actions = np.asarray(ritz.root_actions, dtype=np.complex128)
    scaled_amplitudes = amplitudes / ritz.column_rescales[None, :]
    projected_raw = np.einsum(
        "x,xa,xb->ab",
        measure,
        root_states.conj(),
        root_actions,
        optimize=True,
    )
    # At finite sample count, <Phi|H Phi> need not be Hermitian.  Removing
    # only the occupied three-root component would leave a spurious force
    # along column rescalings/mixings through the unused bank directions when
    # bank_size > 3.  Project against the *entire* frozen carrier span.
    carrier_to_root_action = np.einsum(
        "x,xa,xb->ab",
        measure,
        scaled_amplitudes.conj(),
        root_actions,
        optimize=True,
    )
    try:
        carrier_projection = np.linalg.solve(
            ritz.overlap, carrier_to_root_action
        )
    except np.linalg.LinAlgError as error:
        raise FloatingPointError(
            "direct-amplitude carrier projection solve failed"
        ) from error
    horizontal = root_actions - scaled_amplitudes @ carrier_projection
    projected_horizontal = np.einsum(
        "x,xa,xb->ab",
        measure,
        scaled_amplitudes.conj(),
        horizontal,
        optimize=True,
    )
    error = float(np.max(np.abs(projected_horizontal), initial=0.0))
    if (
        not np.all(np.isfinite(projected_raw))
        or not np.all(np.isfinite(horizontal))
        or not np.isfinite(error)
        or error > float(maximum_horizontality_error)
    ):
        raise FloatingPointError(
            "direct-amplitude horizontal projection failed: "
            f"error={error:.6e}"
        )

    relative_boundary_gap: float | None = None
    if bank_size > 3:
        overlap_values, overlap_vectors = np.linalg.eigh(ritz.overlap)
        whitening = overlap_vectors / np.sqrt(overlap_values)[None, :]
        effective = whitening.conj().T @ ritz.hamiltonian @ whitening
        effective = 0.5 * (effective + effective.conj().T)
        all_energies = np.linalg.eigvalsh(effective)
        boundary_gap = float(all_energies[3] - all_energies[2])
        energy_scale = max(float(np.max(np.abs(all_energies[:4]))), 1.0)
        relative_boundary_gap = boundary_gap / energy_scale
        if (
            not np.isfinite(relative_boundary_gap)
            or relative_boundary_gap < minimum_relative_ritz_boundary_gap
        ):
            raise FloatingPointError(
                "the lowest-three Ky-Fan boundary is unresolved: "
                f"relative_gap={relative_boundary_gap:.6e} < "
                f"{minimum_relative_ritz_boundary_gap:.6e}"
            )
    return DirectAmplitudeKyFanForceReference(
        mean_ritz_energy=float(np.mean(ritz.energies)),
        ritz=ritz,
        amplitudes=_readonly(amplitudes, np.dtype(np.complex128)),
        measure=_readonly(measure, np.dtype(np.float64)),
        column_rescales=_readonly(
            ritz.column_rescales, np.dtype(np.float64)
        ),
        scaled_coefficients=_readonly(
            ritz.scaled_coefficients, np.dtype(np.complex128)
        ),
        projected_raw_hamiltonian=_readonly(
            projected_raw, np.dtype(np.complex128)
        ),
        carrier_projection_coefficients=_readonly(
            carrier_projection, np.dtype(np.complex128)
        ),
        horizontal_residual=_readonly(horizontal, np.dtype(np.complex128)),
        maximum_horizontality_error=error,
        relative_ritz_boundary_gap=relative_boundary_gap,
        sample_count=sample_count,
        bank_size=bank_size,
    )


def direct_amplitude_ky_fan_horizontal_force(
    params: object,
    reference: DirectAmplitudeKyFanForceReference,
    amplitudes_fn: Callable[[object], jax.Array],
) -> DirectAmplitudeKyFanForce:
    """Differentiate the frozen node-safe mean-Ky--Fan surrogate."""

    if not jax.config.x64_enabled:
        raise RuntimeError("direct-amplitude Ky-Fan force requires JAX x64 mode")
    expected = (reference.sample_count, reference.bank_size)
    probe = jnp.asarray(amplitudes_fn(params))
    if probe.shape != expected:
        raise ValueError(
            f"amplitudes_fn must return shape {expected}, got {probe.shape}"
        )
    if probe.dtype not in (jnp.float64, jnp.complex128):
        raise TypeError("direct amplitudes must use float64 or complex128")
    probe_host = np.asarray(probe, dtype=np.complex128)
    if np.any(~np.isfinite(probe_host)):
        raise FloatingPointError("direct amplitudes must be finite")
    reference_scale = max(
        float(np.max(np.abs(reference.amplitudes), initial=0.0)), 1.0
    )
    reference_error = float(
        np.max(np.abs(probe_host - reference.amplitudes), initial=0.0)
        / reference_scale
    )
    if reference_error > 1.0e-11:
        raise ValueError(
            "amplitudes_fn(params) does not reproduce the frozen reference: "
            f"relative_error={reference_error:.6e}"
        )
    for leaf in jax.tree_util.tree_leaves(params):
        if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating):
            raise TypeError("direct-amplitude Ky-Fan force requires real parameters")

    measure = jax.lax.stop_gradient(jnp.asarray(reference.measure))
    rescales = jax.lax.stop_gradient(jnp.asarray(reference.column_rescales))
    coefficients = jax.lax.stop_gradient(
        jnp.asarray(reference.scaled_coefficients)
    )
    horizontal = jax.lax.stop_gradient(
        jnp.asarray(reference.horizontal_residual)
    )

    def surrogate(candidate: object) -> jax.Array:
        amplitudes = jnp.asarray(amplitudes_fn(candidate))
        if amplitudes.shape != expected:
            raise ValueError(
                f"amplitudes_fn must return shape {expected}, got {amplitudes.shape}"
            )
        candidate_roots = (amplitudes / rescales[None, :]) @ coefficients
        return (2.0 / reference.root_count) * jnp.real(
            jnp.sum(measure[:, None] * jnp.conj(candidate_roots) * horizontal)
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
        raise FloatingPointError("direct-amplitude Ky-Fan force is non-finite")
    squared_norm = sum(
        float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2)) for leaf in leaves
    )
    return DirectAmplitudeKyFanForce(
        energy=float(energy),
        gradient=gradient,
        gradient_norm=float(np.sqrt(squared_norm)),
        surrogate_value_at_reference=float(surrogate_value),
        reference=reference,
    )


__all__ = [
    "DirectAmplitudeKyFanForce",
    "DirectAmplitudeKyFanForceReference",
    "direct_amplitude_ky_fan_horizontal_force",
    "prepare_direct_amplitude_ky_fan_force_reference",
]
