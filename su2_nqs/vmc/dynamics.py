"""Exact-support real-time TDVP for small normalized neural states."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)


LogPsi = Callable[[object, jax.Array], jax.Array]


@dataclass(frozen=True)
class TDVPDiagnostics:
    """Gauge-centered least-squares diagnostics for one TDVP direction."""

    residual_norm: float
    target_norm: float
    direction_norm: float
    tangent_rank: int
    tangent_condition_number: float
    integrator_iterations: int = 1
    integrator_residual: float = 0.0


@dataclass(frozen=True)
class SampledTDVPDiagnostics(TDVPDiagnostics):
    """TDVP diagnostics augmented by empirical-support information."""

    sample_count: int = 0
    unique_sample_count: int = 0


def normalized_wavefunction(
    params: object,
    basis: jax.Array,
    log_psi_fn: LogPsi,
) -> jax.Array:
    """Return the explicitly normalized wavefunction on a complete basis."""

    states = jnp.asarray(basis)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("basis must be a non-empty rank-two array")
    logs = jax.vmap(lambda state: log_psi_fn(params, state))(states)
    amplitudes = jnp.exp(logs)
    return amplitudes / jnp.linalg.norm(amplitudes)


def exact_tdvp_direction(
    params: object,
    basis: jax.Array,
    hamiltonian: jax.Array,
    log_psi_fn: LogPsi,
    *,
    damping: float = 1.0e-8,
    singular_value_tolerance: float = 1.0e-10,
    compute_spectrum: bool = True,
) -> tuple[object, TDVPDiagnostics]:
    """Project ``-i(H-E)|psi>`` onto a real-parameter neural tangent space.

    The overall normalization and phase directions are removed by centering
    log derivatives with the exact Born probabilities.  The returned pytree is
    ``d params / dt`` in the McLachlan real-time convention.
    """

    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if singular_value_tolerance <= 0.0:
        raise ValueError("singular_value_tolerance must be positive")
    states = jnp.asarray(basis)
    matrix = jnp.asarray(hamiltonian, dtype=jnp.complex128)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("basis must be a non-empty rank-two array")
    if matrix.shape != (states.shape[0], states.shape[0]):
        raise ValueError("Hamiltonian dimension must match the complete basis")
    flat, unravel = ravel_pytree(params)
    logs = jax.vmap(lambda state: log_psi_fn(params, state))(states)
    amplitudes = jnp.exp(logs)
    amplitudes = amplitudes / jnp.linalg.norm(amplitudes)
    probabilities = jax.lax.stop_gradient(jnp.abs(amplitudes) ** 2)
    local_energies = (matrix @ amplitudes) / amplitudes
    energy = jnp.sum(probabilities * local_energies)
    target_logs = -1j * (local_energies - energy)
    target_logs = target_logs - jnp.sum(probabilities * target_logs)
    square_root_weights = jnp.sqrt(probabilities)

    def centered_features(values: jax.Array) -> jax.Array:
        candidate = unravel(values)
        candidate_logs = jax.vmap(lambda state: log_psi_fn(candidate, state))(
            states
        )
        centered = candidate_logs - jnp.sum(probabilities * candidate_logs)
        return jnp.concatenate(
            (
                square_root_weights * jnp.real(centered),
                square_root_weights * jnp.imag(centered),
            )
        )

    target = jnp.concatenate(
        (
            square_root_weights * jnp.real(target_logs),
            square_root_weights * jnp.imag(target_logs),
        )
    )
    # Exact-support problems have far fewer realified wavefunction rows than
    # network parameters, so reverse-mode materialization is the narrow side.
    jacobian = jax.jacrev(centered_features)(flat)
    if damping == 0.0:
        direction = jnp.linalg.lstsq(jacobian, target, rcond=None)[0]
    elif jacobian.shape[0] < jacobian.shape[1]:
        # Dual ridge solve avoids a parameter-square matrix for overcomplete
        # neural tangents: J^T (J J^T + lambda I)^-1 target.
        dual = jnp.linalg.solve(
            jacobian @ jacobian.T
            + float(damping)
            * jnp.eye(jacobian.shape[0], dtype=jacobian.dtype),
            target,
        )
        direction = jacobian.T @ dual
    else:
        gram = jacobian.T @ jacobian
        rhs = jacobian.T @ target
        direction = jnp.linalg.solve(
            gram + float(damping) * jnp.eye(flat.size, dtype=flat.dtype), rhs
        )
    residual = jacobian @ direction - target
    if compute_spectrum:
        singular_values = np.linalg.svd(np.asarray(jacobian), compute_uv=False)
        if len(singular_values) and singular_values[0] > 0.0:
            rank = int(
                np.sum(
                    singular_values
                    > float(singular_value_tolerance) * singular_values[0]
                )
            )
            condition = (
                float(singular_values[0] / singular_values[rank - 1])
                if rank
                else float("inf")
            )
        else:
            rank = 0
            condition = float("inf")
    else:
        rank = -1
        condition = float("nan")
    diagnostics = TDVPDiagnostics(
        residual_norm=float(jnp.linalg.norm(residual)),
        target_norm=float(jnp.linalg.norm(target)),
        direction_norm=float(jnp.linalg.norm(direction)),
        tangent_rank=rank,
        tangent_condition_number=condition,
    )
    return unravel(direction), diagnostics


def sampled_tdvp_direction(
    params: object,
    samples: jax.Array,
    log_psi_fn: LogPsi,
    get_conn_fn: Callable,
    *,
    damping: float = 1.0e-8,
    singular_value_tolerance: float = 1.0e-10,
    importance_weights: jax.Array | np.ndarray | None = None,
) -> tuple[object, SampledTDVPDiagnostics]:
    """Estimate a real-time TDVP direction from iid autoregressive samples.

    Duplicate configurations are compressed into exact empirical weights, so
    the matrix dimension is controlled by unique support rather than raw batch
    size. Local energies retain the generic ``H[new,old]`` connectivity API.
    """

    from .local_energy import local_energies

    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if singular_value_tolerance <= 0.0:
        raise ValueError("singular_value_tolerance must be positive")
    concrete = np.asarray(samples, dtype=np.int32)
    if concrete.ndim != 2 or concrete.shape[0] == 0:
        raise ValueError("samples must be a non-empty rank-two array")
    unique, inverse, counts = np.unique(
        concrete, axis=0, return_inverse=True, return_counts=True
    )
    states = jnp.asarray(unique, dtype=jnp.int32)
    if importance_weights is None:
        empirical_weights = counts.astype(np.float64)
    else:
        raw_weights = np.asarray(importance_weights, dtype=np.float64)
        if raw_weights.shape != (concrete.shape[0],):
            raise ValueError("importance_weights must match the sample count")
        if np.any(~np.isfinite(raw_weights)) or np.any(raw_weights <= 0.0):
            raise ValueError("importance_weights must be positive and finite")
        empirical_weights = np.zeros(len(unique), dtype=np.float64)
        np.add.at(empirical_weights, inverse, raw_weights)
    probabilities = jax.lax.stop_gradient(
        jnp.asarray(empirical_weights / np.sum(empirical_weights), dtype=jnp.float64)
    )
    flat, unravel = ravel_pytree(params)
    energies = local_energies(params, unique, log_psi_fn, get_conn_fn)
    mean_energy = jnp.sum(probabilities * energies)
    target_logs = -1j * (energies - mean_energy)
    target_logs = target_logs - jnp.sum(probabilities * target_logs)
    square_root_weights = jnp.sqrt(probabilities)

    def centered_features(values: jax.Array) -> jax.Array:
        candidate = unravel(values)
        candidate_logs = jax.vmap(lambda state: log_psi_fn(candidate, state))(
            states
        )
        centered = candidate_logs - jnp.sum(probabilities * candidate_logs)
        return jnp.concatenate(
            (
                square_root_weights * jnp.real(centered),
                square_root_weights * jnp.imag(centered),
            )
        )

    target = jnp.concatenate(
        (
            square_root_weights * jnp.real(target_logs),
            square_root_weights * jnp.imag(target_logs),
        )
    )
    jacobian = jax.jacrev(centered_features)(flat)
    if damping == 0.0:
        direction = jnp.linalg.lstsq(jacobian, target, rcond=None)[0]
    elif jacobian.shape[0] < jacobian.shape[1]:
        dual = jnp.linalg.solve(
            jacobian @ jacobian.T
            + float(damping)
            * jnp.eye(jacobian.shape[0], dtype=jacobian.dtype),
            target,
        )
        direction = jacobian.T @ dual
    else:
        gram = jacobian.T @ jacobian
        rhs = jacobian.T @ target
        direction = jnp.linalg.solve(
            gram + float(damping) * jnp.eye(flat.size, dtype=flat.dtype), rhs
        )
    residual = jacobian @ direction - target
    singular_values = np.linalg.svd(np.asarray(jacobian), compute_uv=False)
    if len(singular_values) and singular_values[0] > 0.0:
        rank = int(
            np.sum(
                singular_values
                > float(singular_value_tolerance) * singular_values[0]
            )
        )
        condition = (
            float(singular_values[0] / singular_values[rank - 1])
            if rank
            else float("inf")
        )
    else:
        rank = 0
        condition = float("inf")
    diagnostics = SampledTDVPDiagnostics(
        residual_norm=float(jnp.linalg.norm(residual)),
        target_norm=float(jnp.linalg.norm(target)),
        direction_norm=float(jnp.linalg.norm(direction)),
        tangent_rank=rank,
        tangent_condition_number=condition,
        sample_count=concrete.shape[0],
        unique_sample_count=len(unique),
    )
    return unravel(direction), diagnostics


def _tree_add_scaled(first: object, second: object, scale: float) -> object:
    return jax.tree.map(lambda x, y: x + scale * y, first, second)


def rk4_tdvp_step(
    params: object,
    basis: jax.Array,
    hamiltonian: jax.Array,
    log_psi_fn: LogPsi,
    *,
    time_step: float,
    damping: float = 1.0e-8,
) -> tuple[object, TDVPDiagnostics]:
    """Advance one explicit fourth-order Runge--Kutta TDVP step."""

    if time_step <= 0.0 or not np.isfinite(time_step):
        raise ValueError("time_step must be positive and finite")
    k1, diagnostics = exact_tdvp_direction(
        params, basis, hamiltonian, log_psi_fn, damping=damping
    )
    k2, _ = exact_tdvp_direction(
        _tree_add_scaled(params, k1, 0.5 * time_step),
        basis,
        hamiltonian,
        log_psi_fn,
        damping=damping,
        compute_spectrum=False,
    )
    k3, _ = exact_tdvp_direction(
        _tree_add_scaled(params, k2, 0.5 * time_step),
        basis,
        hamiltonian,
        log_psi_fn,
        damping=damping,
        compute_spectrum=False,
    )
    k4, _ = exact_tdvp_direction(
        _tree_add_scaled(params, k3, time_step),
        basis,
        hamiltonian,
        log_psi_fn,
        damping=damping,
        compute_spectrum=False,
    )
    updated = jax.tree.map(
        lambda value, a, b, c, d: value
        + (time_step / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        params,
        k1,
        k2,
        k3,
        k4,
    )
    return updated, diagnostics


def implicit_midpoint_tdvp_step(
    params: object,
    basis: jax.Array,
    hamiltonian: jax.Array,
    log_psi_fn: LogPsi,
    *,
    time_step: float,
    damping: float = 1.0e-8,
    maximum_iterations: int = 8,
    tolerance: float = 1.0e-9,
) -> tuple[object, TDVPDiagnostics]:
    """Advance one TDVP step by fixed-point implicit midpoint iteration."""

    if time_step <= 0.0 or not np.isfinite(time_step):
        raise ValueError("time_step must be positive and finite")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, (int, np.integer))
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be positive")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    initial_direction, diagnostics = exact_tdvp_direction(
        params, basis, hamiltonian, log_psi_fn, damping=damping
    )
    candidate = _tree_add_scaled(params, initial_direction, time_step)
    residual = float("inf")
    used_iterations = 0
    for used_iterations in range(1, int(maximum_iterations) + 1):
        midpoint = jax.tree.map(lambda first, second: 0.5 * (first + second), params, candidate)
        midpoint_direction, _ = exact_tdvp_direction(
            midpoint,
            basis,
            hamiltonian,
            log_psi_fn,
            damping=damping,
            compute_spectrum=False,
        )
        updated = _tree_add_scaled(params, midpoint_direction, time_step)
        squared_residual = sum(
            jnp.sum(jnp.abs(first - second) ** 2)
            for first, second in zip(
                jax.tree.leaves(updated), jax.tree.leaves(candidate)
            )
        )
        residual = float(jnp.sqrt(squared_residual))
        candidate = updated
        if residual <= tolerance:
            break
    diagnostics = TDVPDiagnostics(
        residual_norm=diagnostics.residual_norm,
        target_norm=diagnostics.target_norm,
        direction_norm=diagnostics.direction_norm,
        tangent_rank=diagnostics.tangent_rank,
        tangent_condition_number=diagnostics.tangent_condition_number,
        integrator_iterations=used_iterations,
        integrator_residual=residual,
    )
    return candidate, diagnostics


def evolve_exact_tdvp(
    params: object,
    basis: jax.Array,
    hamiltonian: jax.Array,
    log_psi_fn: LogPsi,
    *,
    time_step: float,
    steps: int,
    damping: float = 1.0e-8,
    integrator: str = "rk4",
    implicit_maximum_iterations: int = 8,
    implicit_tolerance: float = 1.0e-9,
) -> tuple[object, np.ndarray, tuple[TDVPDiagnostics, ...]]:
    """Return final parameters and the exact-support TDVP trajectory."""

    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    current = params
    trajectory = [
        np.asarray(normalized_wavefunction(current, basis, log_psi_fn))
    ]
    diagnostics = []
    for _ in range(int(steps)):
        if integrator == "rk4":
            current, row = rk4_tdvp_step(
                current,
                basis,
                hamiltonian,
                log_psi_fn,
                time_step=time_step,
                damping=damping,
            )
        elif integrator == "implicit_midpoint":
            current, row = implicit_midpoint_tdvp_step(
                current,
                basis,
                hamiltonian,
                log_psi_fn,
                time_step=time_step,
                damping=damping,
                maximum_iterations=implicit_maximum_iterations,
                tolerance=implicit_tolerance,
            )
        else:
            raise ValueError(f"unknown TDVP integrator: {integrator!r}")
        diagnostics.append(row)
        trajectory.append(
            np.asarray(normalized_wavefunction(current, basis, log_psi_fn))
        )
    return current, np.asarray(trajectory), tuple(diagnostics)


__all__ = [
    "TDVPDiagnostics",
    "SampledTDVPDiagnostics",
    "evolve_exact_tdvp",
    "exact_tdvp_direction",
    "implicit_midpoint_tdvp_step",
    "normalized_wavefunction",
    "rk4_tdvp_step",
    "sampled_tdvp_direction",
]
