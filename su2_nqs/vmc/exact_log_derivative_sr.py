"""Exact-support natural-gradient ingredients from log derivatives.

This module is a small-system calibration instrument.  It materializes the
log-derivative matrix only after the caller has selected the active parameter
subspace, which avoids paying for frozen carrier or backbone parameters.
Production calculations must use sampled, matrix-free estimators instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


LogAmplitudeFunction = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class ExactLogDerivativeNaturalGradient:
    """One exact-support energy and natural-gradient evaluation."""

    energy: float
    variance: float
    gradient: np.ndarray
    metric: np.ndarray
    descent_direction: np.ndarray
    numerical_rank: int
    relative_linear_residual: float
    minimum_metric_eigenvalue: float
    maximum_metric_eigenvalue: float


def exact_log_derivative_natural_gradient(
    parameters: object,
    *,
    log_amplitude_fn: LogAmplitudeFunction,
    hamiltonian: object,
    damping: float,
    relative_rank_cutoff: float = 1.0e-10,
) -> ExactLogDerivativeNaturalGradient:
    """Return the exact pure-state natural-gradient descent direction.

    ``parameters`` must be a one-dimensional real array containing only the
    active variables.  ``log_amplitude_fn`` returns finite complex logarithms
    for the complete valid support in Hamiltonian basis order.
    """

    values = jnp.asarray(parameters)
    if values.ndim != 1 or not np.issubdtype(np.asarray(values).dtype, np.floating):
        raise TypeError("parameters must be a one-dimensional real array")
    if values.size == 0:
        raise ValueError("parameters must not be empty")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite")
    if not np.isfinite(relative_rank_cutoff) or not 0.0 < relative_rank_cutoff < 1.0:
        raise ValueError("relative_rank_cutoff must lie strictly between zero and one")
    matrix = jnp.asarray(hamiltonian)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")

    def checked_logs(candidate: jax.Array) -> jax.Array:
        logs = jnp.ravel(jnp.asarray(log_amplitude_fn(candidate)))
        if logs.shape[0] != matrix.shape[0]:
            raise ValueError("log amplitudes and Hamiltonian dimensions disagree")
        return logs

    logs = checked_logs(values)
    derivatives = jax.jacfwd(checked_logs)(values)
    return exact_log_derivative_natural_gradient_from_data(
        logs,
        derivatives,
        hamiltonian=matrix,
        damping=damping,
        relative_rank_cutoff=relative_rank_cutoff,
    )


def exact_log_derivative_natural_gradient_from_data(
    log_amplitudes: object,
    log_derivatives: object,
    *,
    hamiltonian: object,
    damping: float,
    relative_rank_cutoff: float = 1.0e-10,
) -> ExactLogDerivativeNaturalGradient:
    """Evaluate the natural gradient from cached amplitudes and derivatives."""

    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite")
    if not np.isfinite(relative_rank_cutoff) or not 0.0 < relative_rank_cutoff < 1.0:
        raise ValueError("relative_rank_cutoff must lie strictly between zero and one")
    matrix = jnp.asarray(hamiltonian)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    logs_np = np.asarray(jax.device_get(log_amplitudes), dtype=np.complex128).ravel()
    derivatives_np = np.asarray(jax.device_get(log_derivatives), dtype=np.complex128)
    if derivatives_np.ndim != 2 or derivatives_np.shape[0] != logs_np.size:
        raise ValueError("log_derivatives must have shape (support, parameters)")
    if logs_np.size != matrix.shape[0] or derivatives_np.shape[1] == 0:
        raise ValueError("cached amplitudes, derivatives, and Hamiltonian disagree")
    if not np.all(np.isfinite(logs_np)) or not np.all(np.isfinite(derivatives_np)):
        raise FloatingPointError("log amplitudes or derivatives contain non-finite values")

    state = np.exp(logs_np - np.max(np.real(logs_np)))
    state /= np.linalg.norm(state)
    h_state = np.asarray(jax.device_get(matrix @ jnp.asarray(state)))
    energy = float(np.vdot(state, h_state).real)
    residual_state = h_state - energy * state
    variance = float(np.vdot(residual_state, residual_state).real)
    probabilities = np.abs(state) ** 2
    centered = derivatives_np - np.sum(
        probabilities[:, None] * derivatives_np, axis=0, keepdims=True
    )
    gradient = 2.0 * np.real(
        np.sum(
            np.conjugate(centered)
            * (np.conjugate(state) * residual_state)[:, None],
            axis=0,
        )
    )
    metric = np.real(
        np.einsum(
            "s,si,sj->ij", probabilities, np.conjugate(centered), centered
        )
    )
    metric = 0.5 * (metric + metric.T)
    eigenvalues = np.linalg.eigvalsh(metric)
    maximum = float(max(eigenvalues[-1], 0.0))
    cutoff = max(
        relative_rank_cutoff * max(maximum, np.finfo(float).tiny),
        100.0 * np.finfo(float).eps,
    )
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    regularized = metric + float(damping) * np.eye(derivatives_np.shape[1])
    solved = np.linalg.solve(regularized, gradient)
    residual = regularized @ solved - gradient
    relative_residual = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(gradient), np.finfo(float).tiny)
    )
    return ExactLogDerivativeNaturalGradient(
        energy=energy,
        variance=variance,
        gradient=gradient,
        metric=metric,
        descent_direction=-solved,
        numerical_rank=rank,
        relative_linear_residual=relative_residual,
        minimum_metric_eigenvalue=float(eigenvalues[0]),
        maximum_metric_eigenvalue=maximum,
    )


__all__ = [
    "ExactLogDerivativeNaturalGradient",
    "LogAmplitudeFunction",
    "exact_log_derivative_natural_gradient",
    "exact_log_derivative_natural_gradient_from_data",
]
