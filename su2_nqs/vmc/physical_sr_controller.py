"""Persistent active-subspace SR and held-out physical update control.

The historical :mod:`su2_nqs.vmc.physical_sampled_sr` entry point accepts
arbitrary parameter pytrees and remains the reference implementation.  This
module adds two operational pieces needed by staged channel release:

* a solver whose parameters are an explicit one-dimensional active vector and
  whose shape-specialized XLA executable persists across updates; and
* a correlated, reference-Born held-out line search that either accepts a
  predeclared step scale or returns the reference vector unchanged.

The held-out sample must be independent of the force/QGT sample.  These tools
do not own a sampler or Hamiltonian action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Integral
from statistics import NormalDist
import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .physical_sampled_sr import (
    PhysicalBatchCenter,
    PhysicalBatchLogAmplitude,
    SampledPhysicalSRDiagnostics,
    _cg_with_true_residual,
    _optional_positive_finite,
    _positive_finite,
)


PhysicalBatchLocalEnergy = Callable[[object, Any], jax.Array]
PhysicalStandardErrorMethod = Literal["delta", "batch", "max"]


@dataclass(frozen=True)
class PersistentCompilationDiagnostics:
    """Compile-versus-execute ledger for one persistent active-SR call."""

    compile_cache_hit: bool
    compiled_signature_count: int
    compilation_seconds: float
    execution_seconds: float


@dataclass(frozen=True)
class PersistentActiveSampledPhysicalSRDirection:
    """An active-vector SR direction plus persistent-compilation metadata."""

    direction: jax.Array
    raw_direction: jax.Array
    linear_residual: jax.Array
    fisher_norm: float
    diagnostics: SampledPhysicalSRDiagnostics
    compilation: PersistentCompilationDiagnostics


@dataclass(frozen=True)
class CorrelatedPhysicalEnergyStatistics:
    """One self-normalized physical-energy estimate on held-out samples."""

    energy: float
    mean_imaginary_local_energy: float
    variance: float
    effective_sample_size: float
    effective_sample_size_fraction: float
    delta_method_standard_error: float
    batch_standard_error: float | None
    standard_error: float
    standard_error_method: str
    sample_count: int
    batch_count: int | None


@dataclass(frozen=True)
class CorrelatedPhysicalEnergyDifference:
    """Candidate-minus-reference comparison using the same held-out sample."""

    candidate: CorrelatedPhysicalEnergyStatistics
    reference: CorrelatedPhysicalEnergyStatistics
    energy_difference: float
    delta_method_standard_error: float
    batch_standard_error: float | None
    standard_error: float
    standard_error_method: str
    confidence_level: float
    upper_confidence_bound: float
    variance_ratio: float


@dataclass(frozen=True)
class HeldOutLineSearchAttempt:
    """One predeclared step-scale decision in a held-out line search."""

    step_scale: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    comparison: CorrelatedPhysicalEnergyDifference


@dataclass(frozen=True)
class HeldOutLineSearchDecision:
    """Accepted active vector, or the bitwise unchanged reference rollback."""

    active_params: jax.Array
    accepted: bool
    accepted_step_scale: float | None
    rollback_performed: bool
    attempts: tuple[HeldOutLineSearchAttempt, ...]


def _shape_dtype_signature(
    tree: object,
) -> tuple[object, tuple[tuple[tuple[int, ...], str], ...]]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    signature = tuple(
        (tuple(np.shape(leaf)), np.dtype(jnp.asarray(leaf).dtype).str)
        for leaf in leaves
    )
    return structure, signature


class PersistentActiveSampledPhysicalSRSolver:
    """Compile-once matrix-free SR for a one-dimensional active vector.

    The amplitude callback receives ``(active_params, samples)``.  Frozen
    backbone parameters may be closed over by the callback, while the QGT is
    differentiated only in the active subspace.  Sample values may change
    without recompilation when their pytree, shapes, and dtypes are fixed.
    """

    def __init__(
        self,
        *,
        center_fn: PhysicalBatchCenter | None = None,
        log_amplitude_fn: PhysicalBatchLogAmplitude | None = None,
        damping: float = 1.0e-3,
        cg_tolerance: float = 1.0e-6,
        cg_max_iterations: int = 50,
        step_size: float = 1.0,
        maximum_relative_parameter_step: float | None = None,
        fisher_trust_radius: float | None = None,
        raise_on_nonconvergence: bool = True,
    ) -> None:
        if (center_fn is None) == (log_amplitude_fn is None):
            raise ValueError("supply exactly one of center_fn or log_amplitude_fn")
        self._center_fn = center_fn
        self._log_amplitude_fn = log_amplitude_fn
        self._amplitude_source = (
            "center_amplitude_complex_log"
            if center_fn is not None
            else "direct_log_amplitude"
        )
        self._damping = _positive_finite(damping, "damping")
        self._cg_tolerance = _positive_finite(cg_tolerance, "cg_tolerance")
        self._step_size = _positive_finite(step_size, "step_size")
        self._maximum_relative_parameter_step = _optional_positive_finite(
            maximum_relative_parameter_step, "maximum_relative_parameter_step"
        )
        self._fisher_trust_radius = _optional_positive_finite(
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
        self._cg_max_iterations = int(cg_max_iterations)
        self._raise_on_nonconvergence = bool(raise_on_nonconvergence)
        self._compiled: dict[object, object] = {}
        self._jitted_kernel = jax.jit(self._build_kernel())

    @property
    def compiled_signature_count(self) -> int:
        """Number of shape/dtype signatures compiled by this instance."""

        return len(self._compiled)

    def clear_compiled_executables(self) -> None:
        """Drop this instance's executable references without global effects."""

        self._compiled.clear()

    def _build_kernel(self):
        center_fn = self._center_fn
        log_amplitude_fn = self._log_amplitude_fn
        damping = self._damping
        tolerance = self._cg_tolerance
        maximum_iterations = self._cg_max_iterations
        step_size = self._step_size
        maximum_relative = self._maximum_relative_parameter_step
        fisher_radius = self._fisher_trust_radius

        def kernel(active_params, samples, energy_gradient):
            def sampled_logs(values):
                if center_fn is not None:
                    centers = jnp.asarray(center_fn(values, samples))
                    if centers.ndim != 1 or centers.shape[0] < 2:
                        raise ValueError(
                            "center_fn must return at least two sampled amplitudes"
                        )
                    return jnp.log(centers.astype(jnp.complex128))
                logs = jnp.asarray(log_amplitude_fn(values, samples))
                if logs.ndim != 1 or logs.shape[0] < 2:
                    raise ValueError(
                        "log_amplitude_fn must return at least two sampled logs"
                    )
                return logs.astype(jnp.complex128)

            if center_fn is not None:
                reference_values = jnp.asarray(center_fn(active_params, samples))
                if reference_values.ndim != 1 or reference_values.shape[0] < 2:
                    raise ValueError(
                        "center_fn must return at least two sampled amplitudes"
                    )
                reference_finite = jnp.all(jnp.isfinite(reference_values))
                minimum_center_magnitude = jnp.min(jnp.abs(reference_values))
                sample_count = reference_values.shape[0]
            else:
                reference_values = jnp.asarray(
                    log_amplitude_fn(active_params, samples)
                )
                if reference_values.ndim != 1 or reference_values.shape[0] < 2:
                    raise ValueError(
                        "log_amplitude_fn must return at least two sampled logs"
                    )
                reference_finite = jnp.all(jnp.isfinite(reference_values))
                minimum_center_magnitude = jnp.asarray(
                    0.0, dtype=active_params.dtype
                )
                sample_count = reference_values.shape[0]

            scale = jnp.sqrt(
                jnp.asarray(2.0 / sample_count, dtype=active_params.dtype)
            )

            def centered_real_features(values):
                logs = sampled_logs(values)
                centered = logs - jnp.mean(logs)
                return scale * jnp.concatenate(
                    (jnp.real(centered), jnp.imag(centered))
                )

            _, pullback = jax.vjp(centered_real_features, active_params)

            def fisher_product(vector):
                _, tangent = jax.jvp(
                    centered_real_features, (active_params,), (vector,)
                )
                return pullback(tangent)[0]

            def damped_product(vector):
                return fisher_product(vector) + damping * vector

            raw_direction, residual, iterations, matvecs = _cg_with_true_residual(
                damped_product,
                energy_gradient,
                tolerance=tolerance,
                maximum_iterations=maximum_iterations,
            )
            gradient_norm = jnp.linalg.norm(energy_gradient)
            residual_norm = jnp.linalg.norm(residual)
            parameter_norm = jnp.linalg.norm(active_params)
            tiny = jnp.finfo(active_params.dtype).tiny
            relative_residual = residual_norm / jnp.maximum(gradient_norm, tiny)

            fisher_image = fisher_product(raw_direction)
            damped_image = fisher_image + damping * raw_direction
            raw_fisher_norm = jnp.sqrt(
                jnp.maximum(jnp.real(jnp.vdot(raw_direction, fisher_image)), 0.0)
            )
            raw_damped_fisher_norm = jnp.sqrt(
                jnp.maximum(jnp.real(jnp.vdot(raw_direction, damped_image)), 0.0)
            )
            raw_direction_norm = jnp.linalg.norm(raw_direction)

            relative_scale = jnp.asarray(1.0, dtype=active_params.dtype)
            if maximum_relative is not None:
                allowed = maximum_relative * parameter_norm
                proposed_norm = step_size * raw_direction_norm
                relative_scale = jnp.where(
                    proposed_norm > 0.0,
                    jnp.minimum(1.0, allowed / proposed_norm),
                    1.0,
                )
            fisher_scale = jnp.asarray(1.0, dtype=active_params.dtype)
            if fisher_radius is not None:
                proposed_fisher = step_size * raw_fisher_norm
                fisher_scale = jnp.where(
                    proposed_fisher > 0.0,
                    jnp.minimum(1.0, fisher_radius / proposed_fisher),
                    1.0,
                )
            trust_scale = jnp.minimum(relative_scale, fisher_scale)
            total_scale = step_size * trust_scale
            direction = total_scale * raw_direction
            direction_norm = jnp.linalg.norm(direction)
            relative_parameter_step = direction_norm / jnp.maximum(
                parameter_norm, tiny
            )
            fisher_norm = total_scale * raw_fisher_norm
            damped_fisher_norm = total_scale * raw_damped_fisher_norm
            directional_derivative = jnp.real(jnp.vdot(energy_gradient, direction))
            return (
                direction,
                raw_direction,
                residual,
                fisher_norm,
                gradient_norm,
                parameter_norm,
                raw_direction_norm,
                raw_fisher_norm,
                raw_damped_fisher_norm,
                residual_norm,
                relative_residual,
                iterations,
                matvecs + 1,
                trust_scale,
                relative_scale,
                fisher_scale,
                direction_norm,
                relative_parameter_step,
                damped_fisher_norm,
                directional_derivative,
                reference_finite,
                minimum_center_magnitude,
                jnp.asarray(sample_count, dtype=jnp.int32),
            )

        return kernel

    def solve(
        self,
        active_params: jax.Array,
        samples: Any,
        *,
        energy_gradient: jax.Array,
    ) -> PersistentActiveSampledPhysicalSRDirection:
        """Solve in the active vector and explicitly reuse a compiled kernel."""

        if not jax.config.x64_enabled:
            raise RuntimeError("sampled physical SR requires JAX x64 mode")
        active = jnp.asarray(active_params)
        gradient = jnp.asarray(energy_gradient)
        if active.ndim != 1 or active.size < 1:
            raise ValueError("active_params must be a nonempty one-dimensional vector")
        if gradient.shape != active.shape:
            raise ValueError("energy_gradient must match the active parameter vector")
        if not np.issubdtype(np.asarray(active).dtype, np.floating):
            raise TypeError("active_params must be a real floating-point vector")
        if not np.issubdtype(np.asarray(gradient).dtype, np.floating):
            raise TypeError("energy_gradient must be a real floating-point vector")
        if not np.all(np.isfinite(np.asarray(active))):
            raise FloatingPointError("active_params contains non-finite values")
        if not np.all(np.isfinite(np.asarray(gradient))):
            raise FloatingPointError("energy_gradient contains non-finite values")
        gradient = gradient.astype(active.dtype)
        concrete_samples = jax.tree_util.tree_map(jnp.asarray, samples)
        parameter_norm = float(jnp.linalg.norm(active))
        if (
            self._maximum_relative_parameter_step is not None
            and parameter_norm == 0.0
        ):
            raise ValueError(
                "maximum_relative_parameter_step requires nonzero parameter norm"
            )

        signature = (
            _shape_dtype_signature(active),
            _shape_dtype_signature(concrete_samples),
            _shape_dtype_signature(gradient),
        )
        executable = self._compiled.get(signature)
        cache_hit = executable is not None
        compilation_seconds = 0.0
        if executable is None:
            compilation_started = time.perf_counter()
            executable = self._jitted_kernel.lower(
                active, concrete_samples, gradient
            ).compile()
            compilation_seconds = time.perf_counter() - compilation_started
            self._compiled[signature] = executable

        execution_started = time.perf_counter()
        outputs = executable(active, concrete_samples, gradient)
        jax.block_until_ready(outputs)
        execution_seconds = time.perf_counter() - execution_started
        (
            direction,
            raw_direction,
            residual,
            fisher_norm,
            gradient_norm,
            output_parameter_norm,
            raw_direction_norm,
            raw_fisher_norm,
            raw_damped_fisher_norm,
            residual_norm,
            relative_residual,
            iterations,
            matvecs,
            trust_scale,
            relative_scale,
            fisher_scale,
            direction_norm,
            relative_parameter_step,
            damped_fisher_norm,
            directional_derivative,
            reference_finite,
            minimum_center_magnitude,
            sample_count,
        ) = outputs

        if not bool(reference_finite):
            callback_name = (
                "center_fn" if self._center_fn is not None else "log_amplitude_fn"
            )
            raise FloatingPointError(f"{callback_name} returned non-finite values")
        minimum_center = (
            float(minimum_center_magnitude)
            if self._center_fn is not None
            else None
        )
        if minimum_center is not None and minimum_center == 0.0:
            raise FloatingPointError("sampled SR encountered a zero center amplitude")
        finite_arrays = (
            direction,
            raw_direction,
            residual,
            fisher_norm,
            gradient_norm,
            output_parameter_norm,
            raw_direction_norm,
            raw_fisher_norm,
            raw_damped_fisher_norm,
            residual_norm,
            relative_residual,
            trust_scale,
            relative_scale,
            fisher_scale,
            direction_norm,
            relative_parameter_step,
            damped_fisher_norm,
            directional_derivative,
        )
        if not all(np.all(np.isfinite(np.asarray(value))) for value in finite_arrays):
            raise FloatingPointError("sampled SR produced non-finite linear algebra")

        relative_residual_value = float(relative_residual)
        convergence_threshold = max(
            10.0 * self._cg_tolerance,
            math.sqrt(np.finfo(np.dtype(active.dtype)).eps),
        )
        converged = relative_residual_value <= convergence_threshold
        if self._raise_on_nonconvergence and not converged:
            raise FloatingPointError(
                "sampled SR CG did not meet its true-residual gate: "
                f"{relative_residual_value:.6g} > {convergence_threshold:.6g}"
            )

        diagnostics = SampledPhysicalSRDiagnostics(
            sample_count=int(sample_count),
            parameter_count=int(active.size),
            amplitude_source=self._amplitude_source,
            damping=self._damping,
            cg_tolerance=self._cg_tolerance,
            cg_max_iterations=self._cg_max_iterations,
            cg_iterations=int(iterations),
            qgt_matvec_count=int(matvecs),
            gradient_norm=float(gradient_norm),
            parameter_norm=float(output_parameter_norm),
            raw_direction_norm=float(raw_direction_norm),
            raw_fisher_norm=float(raw_fisher_norm),
            raw_damped_fisher_norm=float(raw_damped_fisher_norm),
            linear_residual_norm=float(residual_norm),
            relative_linear_residual=relative_residual_value,
            linear_solve_converged=converged,
            requested_step_size=self._step_size,
            trust_scale=float(trust_scale),
            relative_trust_scale=float(relative_scale),
            fisher_trust_scale=float(fisher_scale),
            direction_norm=float(direction_norm),
            relative_parameter_step=float(relative_parameter_step),
            fisher_norm=float(fisher_norm),
            damped_fisher_norm=float(damped_fisher_norm),
            directional_derivative=float(directional_derivative),
            maximum_relative_parameter_step=self._maximum_relative_parameter_step,
            fisher_trust_radius=self._fisher_trust_radius,
            minimum_center_magnitude=minimum_center,
        )
        return PersistentActiveSampledPhysicalSRDirection(
            direction=direction,
            raw_direction=raw_direction,
            linear_residual=residual,
            fisher_norm=float(fisher_norm),
            diagnostics=diagnostics,
            compilation=PersistentCompilationDiagnostics(
                compile_cache_hit=cache_hit,
                compiled_signature_count=len(self._compiled),
                compilation_seconds=compilation_seconds,
                execution_seconds=execution_seconds,
            ),
        )


def _validate_statistical_options(
    sample_count: int,
    *,
    batch_count: int | None,
    standard_error_method: PhysicalStandardErrorMethod,
) -> None:
    if sample_count < 2:
        raise ValueError("held-out comparison requires at least two samples")
    if standard_error_method not in {"delta", "batch", "max"}:
        raise ValueError("standard_error_method must be delta, batch, or max")
    if batch_count is None:
        if standard_error_method == "batch":
            raise ValueError("batch standard errors require batch_count")
        return
    if (
        isinstance(batch_count, (bool, np.bool_))
        or not isinstance(batch_count, Integral)
        or int(batch_count) < 2
        or int(batch_count) > sample_count
        or sample_count % int(batch_count)
    ):
        raise ValueError(
            "batch_count must divide the sample count and lie between 2 and it"
        )


def _influence_standard_errors(
    influence: np.ndarray,
    *,
    batch_count: int | None,
    standard_error_method: PhysicalStandardErrorMethod,
) -> tuple[float, float | None, float]:
    sample_count = len(influence)
    delta = float(np.std(influence, ddof=1) / math.sqrt(sample_count))
    batch = None
    if batch_count is not None:
        means = np.mean(
            influence.reshape(int(batch_count), sample_count // int(batch_count)),
            axis=1,
        )
        batch = float(np.std(means, ddof=1) / math.sqrt(int(batch_count)))
    if standard_error_method == "delta":
        selected = delta
    elif standard_error_method == "batch":
        assert batch is not None
        selected = batch
    else:
        selected = max(delta, batch) if batch is not None else delta
    return delta, batch, selected


def _summarize_energy(
    local_energies: np.ndarray,
    normalized_weights: np.ndarray,
    *,
    batch_count: int | None,
    standard_error_method: PhysicalStandardErrorMethod,
) -> tuple[CorrelatedPhysicalEnergyStatistics, np.ndarray]:
    sample_count = len(local_energies)
    complex_mean = np.sum(normalized_weights * local_energies)
    real_mean = float(np.real(complex_mean))
    variance = float(
        np.sum(normalized_weights * np.abs(local_energies - complex_mean) ** 2)
    )
    effective_sample_size = float(1.0 / np.sum(normalized_weights**2))
    influence = (
        sample_count
        * normalized_weights
        * (np.real(local_energies) - real_mean)
    )
    delta, batch, selected = _influence_standard_errors(
        influence,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    return (
        CorrelatedPhysicalEnergyStatistics(
            energy=real_mean,
            mean_imaginary_local_energy=float(np.imag(complex_mean)),
            variance=variance,
            effective_sample_size=effective_sample_size,
            effective_sample_size_fraction=effective_sample_size / sample_count,
            delta_method_standard_error=delta,
            batch_standard_error=batch,
            standard_error=selected,
            standard_error_method=standard_error_method,
            sample_count=sample_count,
            batch_count=batch_count,
        ),
        influence,
    )


def _logs_and_local_energies(
    params: object,
    samples: Any,
    *,
    center_fn: PhysicalBatchCenter | None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None,
    local_energy_fn: PhysicalBatchLocalEnergy,
) -> tuple[np.ndarray, np.ndarray]:
    if (center_fn is None) == (log_amplitude_fn is None):
        raise ValueError("supply exactly one of center_fn or log_amplitude_fn")
    if center_fn is not None:
        centers = np.asarray(center_fn(params, samples))
        if not np.issubdtype(centers.dtype, np.inexact):
            raise TypeError("center_fn must return real or complex amplitudes")
        if not np.all(np.isfinite(centers)):
            raise FloatingPointError("held-out center amplitudes are non-finite")
        if np.any(np.abs(centers) == 0.0):
            raise FloatingPointError("held-out comparison encountered a zero center")
        logs = np.log(centers.astype(np.complex128))
    else:
        logs = np.asarray(
            log_amplitude_fn(params, samples), dtype=np.complex128
        )
    local = np.asarray(local_energy_fn(params, samples), dtype=np.complex128)
    if logs.ndim != 1 or local.shape != logs.shape or logs.size < 2:
        raise ValueError(
            "held-out log amplitudes and local energies must be matching "
            "one-dimensional arrays with at least two entries"
        )
    if not np.all(np.isfinite(logs)):
        raise FloatingPointError("held-out log amplitudes contain non-finite values")
    if not np.all(np.isfinite(local)):
        raise FloatingPointError("held-out local energies contain non-finite values")
    return logs, local


def _comparison_from_observations(
    reference_logs: np.ndarray,
    reference_local: np.ndarray,
    candidate_logs: np.ndarray,
    candidate_local: np.ndarray,
    *,
    batch_count: int | None,
    standard_error_method: PhysicalStandardErrorMethod,
    confidence_level: float,
) -> CorrelatedPhysicalEnergyDifference:
    sample_count = len(reference_logs)
    if candidate_logs.shape != reference_logs.shape:
        raise ValueError("candidate and reference held-out samples differ in shape")
    _validate_statistical_options(
        sample_count,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    log_weight = 2.0 * np.real(candidate_logs - reference_logs)
    log_weight -= np.max(log_weight)
    candidate_weights = np.exp(log_weight)
    candidate_weights /= np.sum(candidate_weights)
    reference_weights = np.full(sample_count, 1.0 / sample_count)
    candidate, candidate_influence = _summarize_energy(
        candidate_local,
        candidate_weights,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    reference, reference_influence = _summarize_energy(
        reference_local,
        reference_weights,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    delta_error, batch_error, selected_error = _influence_standard_errors(
        candidate_influence - reference_influence,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    difference = candidate.energy - reference.energy
    upper = difference + NormalDist().inv_cdf(confidence_level) * selected_error
    if reference.variance == 0.0:
        variance_ratio = 1.0 if candidate.variance == 0.0 else math.inf
    else:
        variance_ratio = candidate.variance / reference.variance
    return CorrelatedPhysicalEnergyDifference(
        candidate=candidate,
        reference=reference,
        energy_difference=difference,
        delta_method_standard_error=delta_error,
        batch_standard_error=batch_error,
        standard_error=selected_error,
        standard_error_method=standard_error_method,
        confidence_level=confidence_level,
        upper_confidence_bound=upper,
        variance_ratio=variance_ratio,
    )


def correlated_physical_energy_difference(
    reference_params: object,
    candidate_params: object,
    samples: Any,
    *,
    local_energy_fn: PhysicalBatchLocalEnergy,
    center_fn: PhysicalBatchCenter | None = None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None = None,
    batch_count: int | None = None,
    standard_error_method: PhysicalStandardErrorMethod = "max",
    confidence_level: float = 0.95,
) -> CorrelatedPhysicalEnergyDifference:
    """Compare two states on an independent reference-Born held-out sample."""

    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0.5 and 1")
    reference_logs, reference_local = _logs_and_local_energies(
        reference_params,
        samples,
        center_fn=center_fn,
        log_amplitude_fn=log_amplitude_fn,
        local_energy_fn=local_energy_fn,
    )
    candidate_logs, candidate_local = _logs_and_local_energies(
        candidate_params,
        samples,
        center_fn=center_fn,
        log_amplitude_fn=log_amplitude_fn,
        local_energy_fn=local_energy_fn,
    )
    return _comparison_from_observations(
        reference_logs,
        reference_local,
        candidate_logs,
        candidate_local,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
        confidence_level=confidence,
    )


def held_out_correlated_line_search(
    reference_active_params: jax.Array,
    direction: jax.Array,
    samples: Any,
    *,
    local_energy_fn: PhysicalBatchLocalEnergy,
    center_fn: PhysicalBatchCenter | None = None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None = None,
    step_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
    maximum_energy_difference: float = 0.0,
    minimum_effective_sample_size_fraction: float = 0.25,
    maximum_variance_ratio: float | None = None,
    batch_count: int | None = None,
    standard_error_method: PhysicalStandardErrorMethod = "max",
    confidence_level: float = 0.95,
) -> HeldOutLineSearchDecision:
    """Accept the first safe predeclared scale, otherwise exactly roll back.

    ``confidence_level`` is family-wise across ``step_scales``.  Each attempt
    uses the corresponding Bonferroni one-sided confidence level.  Because
    this sample selects a scale, a final scientific audit should still use a
    third, untouched sample stream.
    """

    reference = jnp.asarray(reference_active_params)
    proposed_direction = jnp.asarray(direction)
    if reference.ndim != 1 or reference.size < 1:
        raise ValueError("reference_active_params must be a nonempty vector")
    if proposed_direction.shape != reference.shape:
        raise ValueError("direction must match reference_active_params")
    if not np.all(np.isfinite(np.asarray(reference))) or not np.all(
        np.isfinite(np.asarray(proposed_direction))
    ):
        raise FloatingPointError("reference parameters and direction must be finite")
    if not step_scales:
        raise ValueError("step_scales must be nonempty")
    concrete_scales = tuple(float(scale) for scale in step_scales)
    if not all(math.isfinite(scale) and scale > 0.0 for scale in concrete_scales):
        raise ValueError("step_scales must be positive and finite")
    if any(
        later >= earlier
        for earlier, later in zip(concrete_scales, concrete_scales[1:])
    ):
        raise ValueError("step_scales must be strictly decreasing")
    maximum_delta = float(maximum_energy_difference)
    if not math.isfinite(maximum_delta):
        raise ValueError("maximum_energy_difference must be finite")
    minimum_ess = float(minimum_effective_sample_size_fraction)
    if not math.isfinite(minimum_ess) or not 0.0 < minimum_ess <= 1.0:
        raise ValueError(
            "minimum_effective_sample_size_fraction must lie in (0,1]"
        )
    maximum_variance = _optional_positive_finite(
        maximum_variance_ratio, "maximum_variance_ratio"
    )
    familywise_confidence = float(confidence_level)
    if (
        not math.isfinite(familywise_confidence)
        or not 0.5 < familywise_confidence < 1.0
    ):
        raise ValueError("confidence_level must lie strictly between 0.5 and 1")
    attempt_confidence = 1.0 - (
        1.0 - familywise_confidence
    ) / len(concrete_scales)
    # Reference Hpsi/log evaluations can dominate the held-out cost.  They
    # are immutable across the predeclared scales and are therefore evaluated
    # exactly once.
    reference_logs, reference_local = _logs_and_local_energies(
        reference,
        samples,
        center_fn=center_fn,
        log_amplitude_fn=log_amplitude_fn,
        local_energy_fn=local_energy_fn,
    )

    attempts: list[HeldOutLineSearchAttempt] = []
    for scale in concrete_scales:
        candidate = reference - scale * proposed_direction
        candidate_logs, candidate_local = _logs_and_local_energies(
            candidate,
            samples,
            center_fn=center_fn,
            log_amplitude_fn=log_amplitude_fn,
            local_energy_fn=local_energy_fn,
        )
        comparison = _comparison_from_observations(
            reference_logs,
            reference_local,
            candidate_logs,
            candidate_local,
            batch_count=batch_count,
            standard_error_method=standard_error_method,
            confidence_level=attempt_confidence,
        )
        reasons: list[str] = []
        if comparison.candidate.effective_sample_size_fraction < minimum_ess:
            reasons.append("importance_ess")
        if comparison.upper_confidence_bound > maximum_delta:
            reasons.append("correlated_energy_bound")
        if (
            maximum_variance is not None
            and comparison.variance_ratio > maximum_variance
        ):
            reasons.append("physical_variance_ratio")
        accepted = not reasons
        attempts.append(
            HeldOutLineSearchAttempt(
                step_scale=scale,
                accepted=accepted,
                rejection_reasons=tuple(reasons),
                comparison=comparison,
            )
        )
        if accepted:
            return HeldOutLineSearchDecision(
                active_params=candidate,
                accepted=True,
                accepted_step_scale=scale,
                rollback_performed=False,
                attempts=tuple(attempts),
            )
    return HeldOutLineSearchDecision(
        active_params=reference,
        accepted=False,
        accepted_step_scale=None,
        rollback_performed=True,
        attempts=tuple(attempts),
    )


__all__ = [
    "CorrelatedPhysicalEnergyDifference",
    "CorrelatedPhysicalEnergyStatistics",
    "HeldOutLineSearchAttempt",
    "HeldOutLineSearchDecision",
    "PersistentActiveSampledPhysicalSRDirection",
    "PersistentActiveSampledPhysicalSRSolver",
    "PersistentCompilationDiagnostics",
    "PhysicalBatchLocalEnergy",
    "PhysicalStandardErrorMethod",
    "correlated_physical_energy_difference",
    "held_out_correlated_line_search",
]
