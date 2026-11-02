"""Exact Hamiltonian-flow tail diagnostics for enumerable Hubbard sectors."""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def hubbard_flow_tail_diagnostics(
    matrix: Any,
    amplitudes: np.ndarray,
    batch_sizes: Sequence[int],
    *,
    target_fraction: float = 0.99,
    rare_expected_hit_thresholds: Sequence[float] = (1.0, 0.1),
) -> dict[str, object]:
    """Resolve Schrödinger-residual weight and its Born-sampling visibility.

    The normalized per-state flow weight is

    ``|(H-E) psi|^2 / ||(H-E) psi||^2``.

    Unlike a local-energy quotient, this definition remains finite when an
    amplitude vanishes. For ``M`` independent Born draws, the probability of
    observing state ``i`` at least once is ``1-(1-p_i)^M``. The returned
    expected coverage is therefore an exact finite-support quantity rather
    than a Monte Carlo estimate.
    """

    vector = np.asarray(amplitudes, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("amplitudes must be a nonempty vector")
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("matrix and amplitudes must have matching dimensions")
    sizes = tuple(int(value) for value in batch_sizes)
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must lie in (0, 1]")
    thresholds = tuple(float(value) for value in rare_expected_hit_thresholds)
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("rare expected-hit thresholds must be finite and positive")

    normalization = float(np.vdot(vector, vector).real)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("amplitudes must have finite nonzero norm")
    vector = vector / np.sqrt(normalization)
    probabilities = np.abs(vector) ** 2
    h_vector = np.asarray(matrix @ vector, dtype=np.complex128)
    energy = float(np.real(np.vdot(vector, h_vector)))
    residual_weight = np.abs(h_vector - energy * vector) ** 2
    variance = float(np.sum(residual_weight))
    if variance > 0.0:
        flow = residual_weight / variance
        order = np.argsort(flow)[::-1]
        cumulative = np.cumsum(flow[order])
        n_target = int(np.searchsorted(cumulative, target_fraction, side="left") + 1)
        target_indices = order[:n_target]
        target_probability_mass = float(np.sum(probabilities[target_indices]))
    else:
        flow = np.zeros_like(probabilities)
        n_target = 0
        target_probability_mass = 0.0

    rows = []
    for batch_size in sizes:
        safe_probabilities = np.clip(
            probabilities, 0.0, 1.0 - np.finfo(np.float64).eps
        )
        observed_probability = -np.expm1(
            batch_size * np.log1p(-safe_probabilities)
        )
        row: dict[str, object] = {
            "batch_size": batch_size,
            "expected_unique_states": float(np.sum(observed_probability)),
            "expected_probability_mass_covered": float(
                probabilities @ observed_probability
            ),
            "expected_flow_weight_covered": float(flow @ observed_probability),
            "rare_expected_hit_sectors": {},
        }
        rare = row["rare_expected_hit_sectors"]
        assert isinstance(rare, dict)
        for threshold in thresholds:
            mask = batch_size * probabilities < threshold
            rare[f"{threshold:g}"] = {
                "state_count": int(np.count_nonzero(mask)),
                "probability_mass": float(np.sum(probabilities[mask])),
                "flow_weight": float(np.sum(flow[mask])),
            }
        rows.append(row)

    return {
        "normalization_before_rescaling": normalization,
        "energy": energy,
        "variance": variance,
        "flow_target_fraction": float(target_fraction),
        "flow_target_state_count": n_target,
        "flow_target_probability_mass": target_probability_mass,
        "rows": rows,
    }


def coherent_importance_gradient_statistics(
    params: object,
    basis: Sequence[Sequence[int]],
    matrix: Any,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    proposal_probabilities: np.ndarray,
) -> dict[str, float | int]:
    """Return exact trace noise for a phase-preserving importance gradient.

    For real neural parameters, state ``i`` contributes

    ``z_i = 2 Re[conj(O_i) conj(psi_i) ((H-E)psi)_i]``

    to the exact energy gradient. Sampling ``i`` from proposal ``q`` and using
    ``z_i/q_i`` is unbiased and retains the complex phase before taking the
    real part. This is distinct from using ``|(H-E)psi|^2`` as a Fisher metric.
    """

    basis_array = jnp.asarray(basis, dtype=jnp.int32)
    if basis_array.ndim != 2 or basis_array.shape[0] == 0:
        raise ValueError("basis must be a nonempty rectangular state array")
    dimension = int(basis_array.shape[0])
    if matrix.shape != (dimension, dimension):
        raise ValueError("matrix and basis dimensions differ")
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if proposal.shape != (dimension,) or np.any(~np.isfinite(proposal)) or np.any(proposal < 0.0):
        raise ValueError("proposal probabilities are invalid")
    proposal_total = float(np.sum(proposal))
    if proposal_total <= 0.0:
        raise ValueError("proposal must have positive mass")
    proposal = proposal / proposal_total

    flat, unravel = ravel_pytree(params)

    def logs_from_flat(values):
        candidate = unravel(values)
        return jax.vmap(lambda state: log_psi_fn(candidate, state))(basis_array)

    logs = logs_from_flat(flat)
    amplitudes = jnp.exp(logs)
    norm = jnp.real(jnp.vdot(amplitudes, amplitudes))
    normalized = amplitudes / jnp.sqrt(norm)
    dense_matrix = jnp.asarray(
        matrix.toarray() if hasattr(matrix, "toarray") else matrix,
        dtype=jnp.complex128,
    )
    h_vector = dense_matrix @ normalized
    energy = jnp.real(jnp.vdot(normalized, h_vector))
    residual = h_vector - energy * normalized

    def real_imag_logs(values):
        candidate_logs = logs_from_flat(values)
        return jnp.concatenate((jnp.real(candidate_logs), jnp.imag(candidate_logs)))

    jacobian = jax.jacrev(real_imag_logs)(flat)
    log_derivatives = jacobian[:dimension] + 1j * jacobian[dimension:]
    contributions = 2.0 * jnp.real(
        jnp.conj(log_derivatives)
        * (jnp.conj(normalized) * residual)[:, None]
    )
    exact_gradient = jnp.sum(contributions, axis=0)

    def exact_energy(values):
        candidate = jnp.exp(logs_from_flat(values))
        candidate_h = dense_matrix @ candidate
        return jnp.real(jnp.vdot(candidate, candidate_h)) / jnp.real(
            jnp.vdot(candidate, candidate)
        )

    autodiff_gradient = jax.grad(exact_energy)(flat)
    identity_error = jnp.linalg.norm(exact_gradient - autodiff_gradient)
    probabilities = np.asarray(jnp.abs(normalized) ** 2, dtype=np.float64)
    concrete_contributions = np.asarray(contributions, dtype=np.float64)
    concrete_gradient = np.asarray(exact_gradient, dtype=np.float64)
    active = proposal > 0.0
    outside = float(
        np.sum(np.linalg.norm(concrete_contributions[~active], axis=1))
    )
    if outside > 1.0e-14:
        raise ValueError("proposal misses a nonzero energy-gradient contribution")
    sample_vectors = np.zeros_like(concrete_contributions)
    sample_vectors[active] = concrete_contributions[active] / proposal[active, None]
    centered = sample_vectors - concrete_gradient[None, :]
    trace_variance = float(np.sum(proposal * np.sum(centered**2, axis=1)))
    importance = np.divide(
        probabilities,
        proposal,
        out=np.zeros_like(probabilities),
        where=proposal > 0.0,
    )
    second_moment = float(np.sum(proposal * importance**2))
    positive_target = probabilities > 0.0
    return {
        "dimension": dimension,
        "parameter_count": int(flat.size),
        "normalization": float(norm),
        "energy": float(energy),
        "exact_gradient_norm": float(jnp.linalg.norm(exact_gradient)),
        "exact_gradient_identity_error": float(identity_error),
        "single_draw_trace_variance": trace_variance,
        "single_draw_rms_gradient_error": float(np.sqrt(trace_variance)),
        "relative_single_draw_trace_variance": float(
            trace_variance / max(float(np.sum(concrete_gradient**2)), np.finfo(float).tiny)
        ),
        "maximum_born_importance_weight": float(
            np.max(importance[positive_target], initial=0.0)
        ),
        "born_importance_ess_fraction": float(1.0 / second_moment),
        "gradient_contribution_outside_proposal": outside,
    }


def coherent_importance_sr_direction_statistics(
    params: object,
    basis: Sequence[Sequence[int]],
    matrix: Any,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    samples: Sequence[Sequence[int]],
    importance_weights: np.ndarray,
    *,
    damping: float,
    learning_rate: float | None = None,
    maximum_predicted_feature_step: float | None = None,
) -> dict[str, float | int]:
    """Compare one weighted sampled-SR system with its exact Born reference.

    The sampled definitions exactly match
    :func:`su2_nqs.vmc.padded_sparse.train_padded_sparse_importance_sr`:
    target-over-proposal weights are self-normalized, the complex local-energy
    phase is retained, and real parameters use ``2 Re Cov(O)``.  Materializing
    both Fisher matrices is intentionally restricted to enumerable diagnostic
    systems.
    """

    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    if learning_rate is not None and (
        not np.isfinite(learning_rate) or learning_rate <= 0.0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if maximum_predicted_feature_step is not None and (
        learning_rate is None
        or not np.isfinite(maximum_predicted_feature_step)
        or maximum_predicted_feature_step <= 0.0
    ):
        raise ValueError(
            "maximum_predicted_feature_step requires a positive learning_rate"
        )
    basis_tuple = tuple(tuple(int(value) for value in row) for row in basis)
    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    dimension = len(basis_tuple)
    if dimension == 0 or matrix.shape != (dimension, dimension):
        raise ValueError("matrix and basis dimensions differ")
    positions = {state: index for index, state in enumerate(basis_tuple)}
    concrete_samples = np.asarray(samples, dtype=np.int32)
    if concrete_samples.ndim != 2 or concrete_samples.shape[0] == 0:
        raise ValueError("samples must be a nonempty state array")
    try:
        sample_indices = np.asarray(
            [positions[tuple(row)] for row in concrete_samples], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("sample lies outside the supplied basis") from error
    raw_weights = np.asarray(importance_weights, dtype=np.float64)
    if (
        raw_weights.shape != (len(sample_indices),)
        or np.any(~np.isfinite(raw_weights))
        or np.any(raw_weights <= 0.0)
    ):
        raise ValueError("importance weights must be finite and positive")
    sample_weights = raw_weights / np.sum(raw_weights)

    flat, unravel = ravel_pytree(params)

    def logs_from_flat(values):
        candidate = unravel(values)
        return jax.vmap(lambda state: log_psi_fn(candidate, state))(basis_array)

    logs = logs_from_flat(flat)
    amplitudes = jnp.exp(logs)
    normalized = amplitudes / jnp.sqrt(jnp.real(jnp.vdot(amplitudes, amplitudes)))
    dense_matrix = jnp.asarray(
        matrix.toarray() if hasattr(matrix, "toarray") else matrix,
        dtype=jnp.complex128,
    )
    local_energies = (dense_matrix @ normalized) / normalized
    probabilities = np.asarray(jnp.abs(normalized) ** 2, dtype=np.float64)

    def real_imag_logs(values):
        candidate_logs = logs_from_flat(values)
        return jnp.concatenate((jnp.real(candidate_logs), jnp.imag(candidate_logs)))

    jacobian = np.asarray(jax.jacrev(real_imag_logs)(flat), dtype=np.float64)
    log_derivatives = jacobian[:dimension] + 1j * jacobian[dimension:]
    local = np.asarray(local_energies, dtype=np.complex128)

    def gradient(weights, indices):
        selected_local = local[indices]
        selected_derivatives = log_derivatives[indices]
        mean_energy = np.sum(weights * selected_local)
        return 2.0 * np.real(
            np.sum(
                weights[:, None]
                * np.conj(selected_derivatives)
                * (selected_local - mean_energy)[:, None],
                axis=0,
            )
        )

    def fisher(weights, indices):
        selected = log_derivatives[indices]
        centered = selected - np.sum(weights[:, None] * selected, axis=0)
        return 2.0 * np.real(
            np.conj(centered).T @ (weights[:, None] * centered)
        )

    all_indices = np.arange(dimension, dtype=np.int64)
    exact_gradient = gradient(probabilities, all_indices)
    sampled_gradient = gradient(sample_weights, sample_indices)
    exact_fisher = fisher(probabilities, all_indices)
    sampled_fisher = fisher(sample_weights, sample_indices)
    identity = np.eye(flat.size, dtype=np.float64)
    exact_system = exact_fisher + damping * identity
    sampled_system = sampled_fisher + damping * identity
    exact_direction = np.linalg.solve(exact_system, exact_gradient)
    sampled_direction = np.linalg.solve(sampled_system, sampled_gradient)

    def norm(vector):
        return float(np.linalg.norm(vector))

    def cosine(first, second):
        denominator = norm(first) * norm(second)
        return float(np.dot(first, second) / denominator) if denominator > 0.0 else 0.0

    exact_gradient_norm = norm(exact_gradient)
    exact_direction_norm = norm(exact_direction)
    exact_fisher_norm = float(np.linalg.norm(exact_fisher))
    exact_eigenvalues = np.linalg.eigvalsh(exact_system)
    sampled_eigenvalues = np.linalg.eigvalsh(sampled_system)
    report: dict[str, float | int] = {
        "dimension": dimension,
        "parameter_count": int(flat.size),
        "sample_count": int(len(sample_indices)),
        "unique_sample_count": int(len(np.unique(sample_indices))),
        "exact_gradient_norm": exact_gradient_norm,
        "sampled_gradient_norm": norm(sampled_gradient),
        "gradient_relative_error": norm(sampled_gradient - exact_gradient)
        / max(exact_gradient_norm, np.finfo(float).tiny),
        "gradient_cosine": cosine(sampled_gradient, exact_gradient),
        "exact_direction_norm": exact_direction_norm,
        "sampled_direction_norm": norm(sampled_direction),
        "direction_relative_error": norm(sampled_direction - exact_direction)
        / max(exact_direction_norm, np.finfo(float).tiny),
        "direction_cosine": cosine(sampled_direction, exact_direction),
        "sampled_direction_exact_gradient_cosine": cosine(
            sampled_direction, exact_gradient
        ),
        "exact_descent_slope_for_sampled_direction": float(
            np.dot(exact_gradient, sampled_direction)
        ),
        "sampled_descent_slope_for_sampled_direction": float(
            np.dot(sampled_gradient, sampled_direction)
        ),
        "fisher_relative_frobenius_error": float(
            np.linalg.norm(sampled_fisher - exact_fisher)
            / max(exact_fisher_norm, np.finfo(float).tiny)
        ),
        "exact_damped_fisher_condition_number": float(
            exact_eigenvalues[-1] / exact_eigenvalues[0]
        ),
        "sampled_damped_fisher_condition_number": float(
            sampled_eigenvalues[-1] / sampled_eigenvalues[0]
        ),
    }
    if learning_rate is not None:
        sampled_fisher_direction_norm = float(
            np.sqrt(max(np.dot(sampled_direction, sampled_fisher @ sampled_direction), 0.0))
        )
        unclipped_feature_step = learning_rate * sampled_fisher_direction_norm
        step_scale = (
            1.0
            if maximum_predicted_feature_step is None
            else min(
                1.0,
                maximum_predicted_feature_step
                / max(unclipped_feature_step, np.finfo(float).tiny),
            )
        )
        parameter_step = learning_rate * step_scale
        candidate_flat = np.asarray(flat) - parameter_step * sampled_direction
        candidate_logs = np.asarray(logs_from_flat(jnp.asarray(candidate_flat)))
        current_logs = np.asarray(logs)
        log_displacement = candidate_logs - current_logs
        sampled_displacement = log_displacement[sample_indices]
        sampled_displacement -= np.sum(
            sample_weights * sampled_displacement
        )
        exact_displacement = log_displacement - np.sum(
            probabilities * log_displacement
        )
        candidate_amplitudes = np.exp(candidate_logs)
        candidate_amplitudes /= np.linalg.norm(candidate_amplitudes)
        current_amplitudes = np.asarray(normalized)
        state_fidelity = float(
            abs(np.vdot(current_amplitudes, candidate_amplitudes)) ** 2
        )
        candidate_energy = float(
            np.real(np.vdot(candidate_amplitudes, np.asarray(dense_matrix) @ candidate_amplitudes))
        )
        current_energy = float(
            np.real(np.vdot(current_amplitudes, np.asarray(dense_matrix) @ current_amplitudes))
        )
        report.update(
            {
                "applied_parameter_step_size": parameter_step,
                "predicted_sampled_feature_step_unclipped": unclipped_feature_step,
                "predicted_sampled_feature_step": step_scale
                * unclipped_feature_step,
                "sampled_actual_centered_log_feature_step": float(
                    np.sqrt(
                        2.0
                        * np.sum(sample_weights * np.abs(sampled_displacement) ** 2)
                    )
                ),
                "exact_born_actual_centered_log_feature_step": float(
                    np.sqrt(
                        2.0
                        * np.sum(probabilities * np.abs(exact_displacement) ** 2)
                    )
                ),
                "exact_state_fidelity_after_step": state_fidelity,
                "exact_fubini_study_distance_after_step": float(
                    np.arccos(np.sqrt(np.clip(state_fidelity, 0.0, 1.0)))
                ),
                "exact_energy_before_step": current_energy,
                "exact_energy_after_step": candidate_energy,
                "exact_energy_change_after_step": candidate_energy - current_energy,
            }
        )
    return report


__all__ = [
    "coherent_importance_gradient_statistics",
    "coherent_importance_sr_direction_statistics",
    "hubbard_flow_tail_diagnostics",
]
