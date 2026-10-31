"""Matrix-free sampled stochastic reconfiguration on physical states.

This module owns neither a sampler nor a Hamiltonian action.  Callers provide
an already drawn physical sample batch, a real VMC energy gradient, and either
a batched center-amplitude callback or a batched log-amplitude callback.  The
empirical quantum geometric tensor is the centered real-parameter metric

``S = 2 Re <(O - <O>)^* (O - <O>)>``,

where ``O = d log(psi) / d theta``.  Its product with a vector is evaluated by
one JVP followed by one VJP.  Neither the sample-by-parameter score matrix nor
a parameter-square matrix is materialized.

For the center-amplitude path, amplitudes are cast to ``complex128`` *before*
taking their logarithm.  This preserves the derivative of a negative real
amplitude instead of producing a real-log NaN.  Exact zero centers fail
closed, because their log derivative and hence the sampled QGT are undefined.

The returned ``raw_direction`` solves ``(S + damping I) d = gradient``.  The
returned ``direction`` additionally includes ``step_size`` and any requested
Euclidean-relative or Fisher-radius trust scaling; it is the object to
subtract from the parameters.  ``linear_residual`` always belongs to the raw
linear solve, so trust scaling cannot masquerade as CG nonconvergence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


PhysicalBatchCenter = Callable[[object, Any], jax.Array]
PhysicalBatchLogAmplitude = Callable[[object, Any], jax.Array]


@dataclass(frozen=True)
class SampledPhysicalSRDiagnostics:
    """Scalar safety and work ledger for one matrix-free SR solve."""

    sample_count: int
    parameter_count: int
    amplitude_source: str
    damping: float
    cg_tolerance: float
    cg_max_iterations: int
    cg_iterations: int
    qgt_matvec_count: int
    gradient_norm: float
    parameter_norm: float
    raw_direction_norm: float
    raw_fisher_norm: float
    raw_damped_fisher_norm: float
    linear_residual_norm: float
    relative_linear_residual: float
    linear_solve_converged: bool
    requested_step_size: float
    trust_scale: float
    relative_trust_scale: float
    fisher_trust_scale: float
    direction_norm: float
    relative_parameter_step: float
    fisher_norm: float
    damped_fisher_norm: float
    directional_derivative: float
    maximum_relative_parameter_step: float | None
    fisher_trust_radius: float | None
    minimum_center_magnitude: float | None


@dataclass(frozen=True)
class SampledPhysicalSRDirection:
    """One sampled natural-gradient direction and its true linear residual.

    ``direction`` is a parameter pytree and is intended for the update
    ``new_params = params - direction``.  ``raw_direction`` is the unscaled
    solution of the damped SR equation.  ``linear_residual`` has the same
    pytree structure and equals
    ``(S + damping I) raw_direction - energy_gradient``.
    """

    direction: object
    raw_direction: object
    linear_residual: object
    fisher_norm: float
    diagnostics: SampledPhysicalSRDiagnostics


def _positive_finite(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    answer = float(value)
    if not math.isfinite(answer) or answer <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return answer


def _optional_positive_finite(value: Real | None, name: str) -> float | None:
    if value is None:
        return None
    return _positive_finite(value, name)


def _real_finite_pytree(tree: object, *, name: str) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one array leaf")
    for leaf in leaves:
        concrete = np.asarray(leaf)
        if not np.issubdtype(concrete.dtype, np.floating):
            raise TypeError(f"{name} must contain real floating-point leaves")
        if not np.all(np.isfinite(concrete)):
            raise FloatingPointError(f"{name} contains non-finite values")


def _checked_reference_values(
    params: object,
    samples: Any,
    *,
    center_fn: PhysicalBatchCenter | None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None,
) -> tuple[int, str, float | None]:
    """Validate the current sampled centers/logs before differentiating."""

    if (center_fn is None) == (log_amplitude_fn is None):
        raise ValueError("supply exactly one of center_fn or log_amplitude_fn")
    if center_fn is not None:
        values = np.asarray(center_fn(params, samples))
        source = "center_amplitude_complex_log"
        if values.ndim != 1 or values.shape[0] < 2:
            raise ValueError("center_fn must return at least two sampled amplitudes")
        if not np.issubdtype(values.dtype, np.inexact):
            raise TypeError("center_fn must return real or complex amplitudes")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("center_fn returned non-finite amplitudes")
        magnitudes = np.abs(values)
        if np.any(magnitudes == 0.0):
            raise FloatingPointError(
                "sampled SR encountered a zero center amplitude"
            )
        return int(values.shape[0]), source, float(np.min(magnitudes))

    values = np.asarray(log_amplitude_fn(params, samples))
    if values.ndim != 1 or values.shape[0] < 2:
        raise ValueError(
            "log_amplitude_fn must return at least two sampled log amplitudes"
        )
    if not np.issubdtype(values.dtype, np.inexact):
        raise TypeError("log_amplitude_fn must return real or complex values")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("log_amplitude_fn returned non-finite values")
    return int(values.shape[0]), "direct_log_amplitude", None


def _cg_with_true_residual(
    matrix_vector: Callable[[jax.Array], jax.Array],
    right_hand_side: jax.Array,
    *,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve an SPD system and return the independently recomputed residual.

    Returns ``(solution, true_residual, iterations, matvec_count)``.  The work
    count includes one matrix-vector product per CG iteration and the final
    independent residual product.
    """

    solution = jnp.zeros_like(right_hand_side)
    residual = right_hand_side
    search = residual
    squared_residual = jnp.real(jnp.vdot(residual, residual))
    rhs_norm = jnp.linalg.norm(right_hand_side)
    tiny = jnp.finfo(right_hand_side.dtype).tiny
    threshold_squared = (tolerance * jnp.maximum(rhs_norm, tiny)) ** 2

    def condition(state):
        iteration, _, _, _, value = state
        return (iteration < maximum_iterations) & (value > threshold_squared)

    def body(state):
        iteration, current, carried_residual, direction, value = state
        image = matrix_vector(direction)
        denominator = jnp.real(jnp.vdot(direction, image))
        safe_denominator = jnp.maximum(denominator, tiny)
        alpha = value / safe_denominator
        changed = current + alpha * direction
        next_residual = carried_residual - alpha * image
        next_value = jnp.real(jnp.vdot(next_residual, next_residual))
        beta = next_value / jnp.maximum(value, tiny)
        next_direction = next_residual + beta * direction
        return (
            iteration + 1,
            changed,
            next_residual,
            next_direction,
            next_value,
        )

    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        solution,
        residual,
        search,
        squared_residual,
    )
    iterations, solution, _, _, _ = jax.lax.while_loop(
        condition, body, initial
    )
    true_residual = matrix_vector(solution) - right_hand_side
    return solution, true_residual, iterations, iterations + 1


def sampled_physical_sr_direction(
    params: object,
    samples: Any,
    *,
    energy_gradient: object,
    center_fn: PhysicalBatchCenter | None = None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None = None,
    damping: float = 1.0e-3,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 50,
    step_size: float = 1.0,
    maximum_relative_parameter_step: float | None = None,
    fisher_trust_radius: float | None = None,
    raise_on_nonconvergence: bool = True,
) -> SampledPhysicalSRDirection:
    """Return a matrix-free sampled physical-state SR update direction.

    Parameters are restricted to real floating arrays.  ``energy_gradient``
    must be the real gradient of the energy with the same pytree structure as
    ``params``.  The empirical metric gives every supplied sample equal
    weight, as appropriate for a Born-distributed physical sample batch.

    ``center_fn(params, samples)`` must return one real or complex amplitude
    per sample.  Negative real amplitudes are supported.  Alternatively,
    ``log_amplitude_fn(params, samples)`` may directly return finite complex
    logs.  Exactly one callback is required.

    The unconstrained natural direction is multiplied by ``step_size`` and
    then uniformly shortened, if necessary, to obey both requested trust
    bounds.  The Euclidean relative bound uses ``||params||`` as its scale and
    therefore requires a nonzero parameter norm.  The Fisher trust radius is
    ``sqrt(direction.T @ S @ direction)`` and excludes diagonal damping.
    """

    if not jax.config.x64_enabled:
        raise RuntimeError("sampled physical SR requires JAX x64 mode")
    _real_finite_pytree(params, name="params")
    _real_finite_pytree(energy_gradient, name="energy_gradient")
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(
        energy_gradient
    ):
        raise ValueError("params and energy_gradient must share a pytree structure")
    parameter_shapes = tuple(
        np.asarray(leaf).shape for leaf in jax.tree_util.tree_leaves(params)
    )
    gradient_shapes = tuple(
        np.asarray(leaf).shape
        for leaf in jax.tree_util.tree_leaves(energy_gradient)
    )
    if gradient_shapes != parameter_shapes:
        raise ValueError("energy_gradient leaf shapes must match params")

    damping = _positive_finite(damping, "damping")
    cg_tolerance = _positive_finite(cg_tolerance, "cg_tolerance")
    step_size = _positive_finite(step_size, "step_size")
    maximum_relative_parameter_step = _optional_positive_finite(
        maximum_relative_parameter_step, "maximum_relative_parameter_step"
    )
    fisher_trust_radius = _optional_positive_finite(
        fisher_trust_radius, "fisher_trust_radius"
    )
    if (
        isinstance(cg_max_iterations, (bool, np.bool_))
        or not isinstance(cg_max_iterations, Integral)
        or int(cg_max_iterations) < 1
    ):
        raise ValueError("cg_max_iterations must be a positive integer")
    if not isinstance(raise_on_nonconvergence, (bool, np.bool_)):
        raise TypeError("raise_on_nonconvergence must be boolean")
    cg_max_iterations = int(cg_max_iterations)

    sample_count, amplitude_source, minimum_center_magnitude = (
        _checked_reference_values(
            params,
            samples,
            center_fn=center_fn,
            log_amplitude_fn=log_amplitude_fn,
        )
    )
    flat, unravel = ravel_pytree(params)
    flat_gradient, _ = ravel_pytree(energy_gradient)
    # Mixed real leaf dtypes are legal pytrees, but SR uses the promoted flat
    # parameter dtype consistently for the solve and returned residual.
    flat_gradient = flat_gradient.astype(flat.dtype)

    expected_shape = (sample_count,)

    def sampled_logs(values: jax.Array) -> jax.Array:
        candidate = unravel(values)
        if center_fn is not None:
            centers = jnp.asarray(center_fn(candidate, samples))
            if centers.shape != expected_shape:
                raise ValueError("center_fn changed its sampled output shape")
            return jnp.log(centers.astype(jnp.complex128))
        logs = jnp.asarray(log_amplitude_fn(candidate, samples))
        if logs.shape != expected_shape:
            raise ValueError("log_amplitude_fn changed its sampled output shape")
        return logs.astype(jnp.complex128)

    scale = jnp.sqrt(jnp.asarray(2.0 / sample_count, dtype=flat.dtype))

    def centered_real_features(values: jax.Array) -> jax.Array:
        logs = sampled_logs(values)
        centered = logs - jnp.mean(logs)
        return scale * jnp.concatenate((jnp.real(centered), jnp.imag(centered)))

    _, pullback = jax.vjp(centered_real_features, flat)

    def fisher_product(vector: jax.Array) -> jax.Array:
        _, tangent = jax.jvp(centered_real_features, (flat,), (vector,))
        return pullback(tangent)[0]

    def damped_product(vector: jax.Array) -> jax.Array:
        return fisher_product(vector) + damping * vector

    raw_direction, flat_residual, iterations, matvecs = _cg_with_true_residual(
        damped_product,
        flat_gradient,
        tolerance=cg_tolerance,
        maximum_iterations=cg_max_iterations,
    )
    gradient_norm_array = jnp.linalg.norm(flat_gradient)
    residual_norm_array = jnp.linalg.norm(flat_residual)
    tiny = jnp.finfo(flat.dtype).tiny
    relative_residual_array = residual_norm_array / jnp.maximum(
        gradient_norm_array, tiny
    )
    convergence_threshold = max(
        10.0 * cg_tolerance, math.sqrt(np.finfo(np.dtype(flat.dtype)).eps)
    )

    raw_fisher_image = fisher_product(raw_direction)
    raw_damped_image = raw_fisher_image + damping * raw_direction
    raw_fisher_squared = jnp.maximum(
        jnp.real(jnp.vdot(raw_direction, raw_fisher_image)), 0.0
    )
    raw_damped_squared = jnp.maximum(
        jnp.real(jnp.vdot(raw_direction, raw_damped_image)), 0.0
    )

    concrete = {
        "gradient_norm": float(gradient_norm_array),
        "residual_norm": float(residual_norm_array),
        "relative_residual": float(relative_residual_array),
        "raw_direction_norm": float(jnp.linalg.norm(raw_direction)),
        "raw_fisher_norm": float(jnp.sqrt(raw_fisher_squared)),
        "raw_damped_fisher_norm": float(jnp.sqrt(raw_damped_squared)),
        "parameter_norm": float(jnp.linalg.norm(flat)),
        "iterations": int(iterations),
        "matvecs": int(matvecs) + 1,
    }
    if not all(
        math.isfinite(value)
        for name, value in concrete.items()
        if name not in {"iterations", "matvecs"}
    ) or not np.all(np.isfinite(np.asarray(raw_direction))):
        raise FloatingPointError("sampled SR produced non-finite linear algebra")
    converged = concrete["relative_residual"] <= convergence_threshold
    if raise_on_nonconvergence and not converged:
        raise FloatingPointError(
            "sampled SR CG did not meet its true-residual gate: "
            f"{concrete['relative_residual']:.6g} > {convergence_threshold:.6g}"
        )

    base_scale = step_size
    base_direction_norm = step_size * concrete["raw_direction_norm"]
    relative_scale = 1.0
    if maximum_relative_parameter_step is not None:
        if concrete["parameter_norm"] == 0.0:
            raise ValueError(
                "maximum_relative_parameter_step requires nonzero parameter norm"
            )
        allowed = maximum_relative_parameter_step * concrete["parameter_norm"]
        if base_direction_norm > 0.0:
            relative_scale = min(1.0, allowed / base_direction_norm)

    base_fisher_norm = step_size * concrete["raw_fisher_norm"]
    fisher_scale = 1.0
    if fisher_trust_radius is not None and base_fisher_norm > 0.0:
        fisher_scale = min(1.0, fisher_trust_radius / base_fisher_norm)
    trust_scale = min(relative_scale, fisher_scale)
    total_scale = step_size * trust_scale
    applied = total_scale * raw_direction
    applied_fisher_norm = total_scale * concrete["raw_fisher_norm"]
    applied_damped_norm = total_scale * concrete["raw_damped_fisher_norm"]
    applied_norm = float(jnp.linalg.norm(applied))
    relative_parameter_step = applied_norm / max(concrete["parameter_norm"], tiny)
    directional_derivative = float(jnp.real(jnp.vdot(flat_gradient, applied)))
    if not all(
        math.isfinite(value)
        for value in (
            trust_scale,
            total_scale,
            applied_fisher_norm,
            applied_damped_norm,
            applied_norm,
            relative_parameter_step,
            directional_derivative,
        )
    ):
        raise FloatingPointError("sampled SR trust scaling produced non-finite data")

    diagnostics = SampledPhysicalSRDiagnostics(
        sample_count=sample_count,
        parameter_count=int(flat.size),
        amplitude_source=amplitude_source,
        damping=damping,
        cg_tolerance=cg_tolerance,
        cg_max_iterations=cg_max_iterations,
        cg_iterations=concrete["iterations"],
        qgt_matvec_count=concrete["matvecs"],
        gradient_norm=concrete["gradient_norm"],
        parameter_norm=concrete["parameter_norm"],
        raw_direction_norm=concrete["raw_direction_norm"],
        raw_fisher_norm=concrete["raw_fisher_norm"],
        raw_damped_fisher_norm=concrete["raw_damped_fisher_norm"],
        linear_residual_norm=concrete["residual_norm"],
        relative_linear_residual=concrete["relative_residual"],
        linear_solve_converged=converged,
        requested_step_size=step_size,
        trust_scale=trust_scale,
        relative_trust_scale=relative_scale,
        fisher_trust_scale=fisher_scale,
        direction_norm=applied_norm,
        relative_parameter_step=relative_parameter_step,
        fisher_norm=applied_fisher_norm,
        damped_fisher_norm=applied_damped_norm,
        directional_derivative=directional_derivative,
        maximum_relative_parameter_step=maximum_relative_parameter_step,
        fisher_trust_radius=fisher_trust_radius,
        minimum_center_magnitude=minimum_center_magnitude,
    )
    return SampledPhysicalSRDirection(
        direction=unravel(applied),
        raw_direction=unravel(raw_direction),
        linear_residual=unravel(flat_residual),
        fisher_norm=applied_fisher_norm,
        diagnostics=diagnostics,
    )


__all__ = [
    "PhysicalBatchCenter",
    "PhysicalBatchLogAmplitude",
    "SampledPhysicalSRDiagnostics",
    "SampledPhysicalSRDirection",
    "sampled_physical_sr_direction",
]
