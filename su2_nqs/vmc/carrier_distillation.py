"""Representation-aware distillation of one normalized physical carrier.

The production comparison uses different exact supports: a reduced SU(2)
singlet basis, the physical ``M=0`` basis, and the full computational basis.
Sharing raw parameter arrays does not make their represented wavefunctions
equal.  This module supplies a narrow teacher-distribution fit so that each
backbone can instead be initialized from the same physical small-system
state.  No Hamiltonian is used during this fit.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp
import numpy as np


BatchLogAmplitude = Callable[[object, jax.Array], jax.Array]


def train_normalized_carrier_distillation(
    params: object,
    *,
    basis: np.ndarray,
    target: np.ndarray,
    log_amplitude_fn: BatchLogAmplitude,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    phase_weight: float = 1.0,
    record_every: int = 50,
    exact_teacher_distribution: bool = False,
) -> tuple[object, list[dict[str, float]]]:
    """Fit a normalized autoregressive model to one normalized carrier.

    Mini-batches are sampled exactly from ``abs(target)**2``.  The Born
    cross-entropy therefore retains the normalization of the student's full
    native support.  This matters for the symmetry-unaware arm: probability
    outside ``M=0`` is penalized even though the teacher has no amplitudes
    there.  A circular phase loss is invariant under the global phase.
    """

    states = np.asarray(basis)
    desired = np.asarray(target, dtype=np.complex128)
    if states.ndim != 2 or desired.shape != (len(states),) or len(states) < 1:
        raise ValueError("basis and target must have aligned nonempty leading axes")
    norm = float(np.vdot(desired, desired).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("target must have positive finite norm")
    desired = desired / np.sqrt(norm)
    probabilities = np.abs(desired) ** 2
    active = probabilities > 0.0
    if not np.any(active):
        raise ValueError("target has empty support")
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("record_every", record_every),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rate = float(learning_rate)
    phase_scale = float(phase_weight)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(phase_scale) or phase_scale < 0.0:
        raise ValueError("phase_weight must be nonnegative and finite")
    if not isinstance(exact_teacher_distribution, bool):
        raise TypeError("exact_teacher_distribution must be boolean")

    active_states = jnp.asarray(states[active])
    active_probabilities = jnp.asarray(probabilities[active])
    active_probabilities = active_probabilities / jnp.sum(active_probabilities)
    active_phases = jnp.asarray(np.angle(desired[active]))
    log_teacher_probabilities = jnp.log(active_probabilities)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    def objective(candidate, indices, batch_weights):
        selected_states = active_states[indices]
        selected_phases = active_phases[indices]
        selected_teacher_log_p = log_teacher_probabilities[indices]
        weights = batch_weights / jnp.sum(batch_weights)
        logs = log_amplitude_fn(candidate, selected_states)
        student_log_p = 2.0 * jnp.real(logs)
        magnitude = -jnp.sum(weights * student_log_p)
        phase_alignment = jnp.sum(
            weights * jnp.exp(1.0j * (jnp.imag(logs) - selected_phases))
        )
        phase = 1.0 - jnp.abs(phase_alignment)
        # This diagnostic differs from the optimized cross entropy only by
        # the fixed teacher entropy and is zero at the exact Born target.
        born_kl_estimate = jnp.sum(
            weights * (selected_teacher_log_p - student_log_p)
        )
        return magnitude + phase_scale * phase, (
            magnitude,
            phase,
            jnp.abs(phase_alignment),
            born_kl_estimate,
        )

    value_gradient = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def update(candidate, first_moment, second_moment, indices, weights, step):
        (loss, metrics), gradient = value_gradient(candidate, indices, weights)
        first_moment = jax.tree.map(
            lambda moment, value: 0.9 * moment + 0.1 * value,
            first_moment,
            gradient,
        )
        second_moment = jax.tree.map(
            lambda moment, value: 0.999 * moment + 0.001 * value * value,
            second_moment,
            gradient,
        )
        first_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.9**step), first_moment
        )
        second_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.999**step), second_moment
        )
        candidate = jax.tree.map(
            lambda value, mean, square: value
            - rate * mean / (jnp.sqrt(square) + 1.0e-8),
            candidate,
            first_hat,
            second_hat,
        )
        return candidate, first_moment, second_moment, loss, metrics

    history: list[dict[str, float]] = []
    key = jax.random.PRNGKey(int(seed))
    for step in range(1, steps + 1):
        if exact_teacher_distribution:
            indices = jnp.arange(active_states.shape[0], dtype=jnp.int32)
            weights = active_probabilities
        else:
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(
                subkey,
                active_states.shape[0],
                shape=(batch_size,),
                replace=True,
                p=active_probabilities,
            )
            weights = jnp.ones((batch_size,), dtype=jnp.float64)
        params, first, second, loss, metrics = update(
            params,
            first,
            second,
            indices,
            weights,
            jnp.asarray(step, dtype=jnp.float64),
        )
        if step == 1 or step % record_every == 0 or step == steps:
            values = jax.device_get((loss, metrics))
            history.append(
                {
                    "step": float(step),
                    "loss": float(values[0]),
                    "magnitude_cross_entropy": float(values[1][0]),
                    "phase_loss": float(values[1][1]),
                    "phase_coherence": float(values[1][2]),
                    "born_kl_batch_estimate": float(values[1][3]),
                }
            )
    return params, history


def evaluate_log_amplitudes_in_batches(
    params: object,
    basis: np.ndarray,
    log_amplitude_fn: BatchLogAmplitude,
    *,
    batch_size: int,
) -> np.ndarray:
    """Evaluate a complete finite support without changing model semantics."""

    states = np.asarray(basis)
    if states.ndim != 2 or len(states) < 1:
        raise ValueError("basis must be a nonempty rank-two array")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    compiled = jax.jit(log_amplitude_fn)
    rows = []
    for start in range(0, len(states), batch_size):
        values = compiled(params, jnp.asarray(states[start : start + batch_size]))
        rows.append(np.asarray(jax.device_get(values), dtype=np.complex128))
    return np.concatenate(rows)


def train_sampled_carrier_pool_distillation(
    params: object,
    *,
    sample_pool: np.ndarray,
    target_phases: np.ndarray,
    log_amplitude_fn: BatchLogAmplitude,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    phase_weight: float = 1.0,
    record_every: int = 50,
) -> tuple[object, list[dict[str, float]]]:
    """Fit to IID samples drawn analytically from a carrier Born law.

    Every pool row already follows the normalized teacher distribution, so
    uniform pool resampling gives an unbiased Born cross-entropy estimator.
    This is the scalable counterpart of exhaustive N=16 distillation and
    makes no Hamiltonian call.
    """

    states = np.asarray(sample_pool)
    phases = np.asarray(target_phases, dtype=np.float64)
    if states.ndim != 2 or len(states) < 2 or phases.shape != (len(states),):
        raise ValueError("sample_pool and target_phases must be aligned and nonempty")
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("record_every", record_every),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rate = float(learning_rate)
    phase_scale = float(phase_weight)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(phase_scale) or phase_scale < 0.0:
        raise ValueError("phase_weight must be nonnegative and finite")
    states_jax = jnp.asarray(states)
    phases_jax = jnp.asarray(phases)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    def objective(candidate, indices):
        logs = log_amplitude_fn(candidate, states_jax[indices])
        magnitude = -jnp.mean(2.0 * jnp.real(logs))
        phase_alignment = jnp.mean(
            jnp.exp(1.0j * (jnp.imag(logs) - phases_jax[indices]))
        )
        phase = 1.0 - jnp.abs(phase_alignment)
        return magnitude + phase_scale * phase, (
            magnitude,
            phase,
            jnp.abs(phase_alignment),
        )

    value_gradient = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def update(candidate, first_moment, second_moment, indices, step):
        (loss, metrics), gradient = value_gradient(candidate, indices)
        first_moment = jax.tree.map(
            lambda moment, value: 0.9 * moment + 0.1 * value,
            first_moment,
            gradient,
        )
        second_moment = jax.tree.map(
            lambda moment, value: 0.999 * moment + 0.001 * value * value,
            second_moment,
            gradient,
        )
        first_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.9**step), first_moment
        )
        second_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.999**step), second_moment
        )
        candidate = jax.tree.map(
            lambda value, mean, square: value
            - rate * mean / (jnp.sqrt(square) + 1.0e-8),
            candidate,
            first_hat,
            second_hat,
        )
        return candidate, first_moment, second_moment, loss, metrics

    key = jax.random.PRNGKey(int(seed))
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        indices = jax.random.randint(
            subkey, (batch_size,), 0, states_jax.shape[0]
        )
        params, first, second, loss, metrics = update(
            params,
            first,
            second,
            indices,
            jnp.asarray(step, dtype=jnp.float64),
        )
        if step == 1 or step % record_every == 0 or step == steps:
            loss_host, metrics_host = jax.device_get((loss, metrics))
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss_host),
                    "magnitude_cross_entropy": float(metrics_host[0]),
                    "phase_loss": float(metrics_host[1]),
                    "phase_coherence": float(metrics_host[2]),
                }
            )
    return params, history


def normalized_carrier_audit(
    amplitudes_on_teacher_support: np.ndarray,
    target: np.ndarray,
    *,
    native_norm: float,
) -> dict[str, float]:
    """Return fidelity and normalization diagnostics for one student state."""

    student = np.asarray(amplitudes_on_teacher_support, dtype=np.complex128)
    desired = np.asarray(target, dtype=np.complex128)
    if student.shape != desired.shape or student.ndim != 1:
        raise ValueError("student and target must be aligned vectors")
    target_norm = float(np.vdot(desired, desired).real)
    total_norm = float(native_norm)
    if target_norm <= 0.0 or total_norm <= 0.0:
        raise ValueError("state norms must be positive")
    overlap = np.vdot(desired, student)
    fidelity = float(abs(overlap) ** 2 / (target_norm * total_norm))
    return {
        "fidelity": fidelity,
        "infidelity": 1.0 - fidelity,
        "native_norm": total_norm,
        "teacher_sector_weight": float(np.vdot(student, student).real / total_norm),
    }


__all__ = [
    "evaluate_log_amplitudes_in_batches",
    "normalized_carrier_audit",
    "train_sampled_carrier_pool_distillation",
    "train_normalized_carrier_distillation",
]
