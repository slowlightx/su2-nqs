"""Full-support neural projection with matrix-free damped LSQR.

This module is deliberately separate from :mod:`su2_nqs.vmc.sampled_sr` so
that solver comparisons do not change the provenance of existing CG
benchmarks.  It uses exactly the same weighted, centered, realified complex
log-wavefunction features, but solves the rectangular linearized problem
directly with Golub--Kahan LSQR instead of applying CG to normal equations.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .matrix_free_lsqr import DampedLSQRResult, damped_lsqr


def centered_real_log_features(
    params: object,
    samples: jax.Array,
    importance_weights: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> jax.Array:
    """Return the weighted, gauge-centered realification of complex logs.

    The returned vector is

    ``concat(sqrt(2 w) Re(logpsi-<logpsi>_w),
             sqrt(2 w) Im(logpsi-<logpsi>_w))``.

    Consequently the Jacobian is the same rectangular tangent operator whose
    Gram matrix is used by the existing sampled-SR implementation.  Overall
    normalization and global phase directions are removed before both the
    target and tangent are realified.
    """

    weights = jnp.asarray(importance_weights)
    if weights.ndim != 1:
        raise ValueError("importance_weights must be one-dimensional")
    if samples.shape[0] != weights.shape[0]:
        raise ValueError("samples and importance_weights must have equal length")
    normalized_weights = weights / jnp.maximum(
        jnp.sum(weights), jnp.finfo(weights.dtype).tiny
    )
    logs = jax.vmap(lambda sample: log_psi_fn(params, sample))(samples)
    centered = logs - jnp.sum(normalized_weights * logs)
    square_root_weights = jnp.sqrt(2.0 * normalized_weights)
    return jnp.concatenate(
        (
            square_root_weights * jnp.real(centered),
            square_root_weights * jnp.imag(centered),
        )
    )


def solve_linearized_projection_direction(
    values: jax.Array,
    feature_fn: Callable[[jax.Array], jax.Array],
    target_features: jax.Array,
    *,
    damping: float,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    max_iterations: int = 100,
) -> DampedLSQRResult:
    """Solve one rectangular linearized projection problem using JVP/VJP.

    ``feature_fn`` maps a flat real parameter vector to realified centered log
    features.  Only its JVP and VJP are exposed to :func:`damped_lsqr`; an
    explicit Jacobian is never formed by this routine.
    """

    values_array = jnp.asarray(values)
    if values_array.ndim != 1:
        raise ValueError("values must be a flat one-dimensional array")
    target_array = jnp.asarray(target_features)
    current_features, pullback = jax.vjp(feature_fn, values_array)
    if target_array.shape != current_features.shape:
        raise ValueError("target_features must match feature_fn(values)")
    residual = target_array - current_features

    def matvec(vector):
        _, tangent = jax.jvp(feature_fn, (values_array,), (vector,))
        return tangent

    def rmatvec(vector):
        return pullback(vector)[0]

    return damped_lsqr(
        matvec,
        rmatvec,
        residual,
        damping=damping,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        max_iterations=max_iterations,
    )


def train_lsqr_projection(
    params: object,
    key: jax.Array,
    *,
    sample_pool: jax.Array,
    pool_importance_weights: jax.Array,
    pool_base_logs: jax.Array,
    pool_target_log_ratios: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    batch_size: int,
    damping: float = 1e-3,
    step_size: float = 1.0,
    lsqr_tolerance: float = 1e-8,
    lsqr_absolute_tolerance: float = 0.0,
    lsqr_max_iterations: int = 100,
    backtracking_steps: int = 6,
    acceptance_score_fn: Callable[[object], float] | None = None,
    random_batches: bool = True,
) -> tuple[object, list[dict[str, object]]]:
    """Project a sampled complex teacher displacement with matrix-free LSQR.

    Sampling, target construction, feature centering, candidate step schedule,
    and optional energy acceptance have the same semantics as
    ``train_sampled_sr_projection``.  The difference is solely the Krylov
    solver: this function acts on the rectangular centered log-Jacobian rather
    than using CG on its normal matrix.

    Each outer-step history row contains exact primal, augmented, and normal
    residual histories for the linear solve.  The native LSQR residual
    estimate is kept under a separate key.
    """

    if steps < 1 or batch_size < 1 or len(sample_pool) < 1:
        raise ValueError("steps, batch size, and sample pool must be positive")
    if damping <= 0.0 or step_size <= 0.0:
        raise ValueError("damping and step size must be positive")
    if lsqr_tolerance < 0.0 or lsqr_absolute_tolerance < 0.0:
        raise ValueError("LSQR tolerances must be non-negative")
    if lsqr_max_iterations < 1:
        raise ValueError("lsqr_max_iterations must be positive")
    if backtracking_steps < 1:
        raise ValueError("backtracking_steps must be positive")
    flat, unravel = ravel_pytree(params)

    def update(values, samples, importance, base_logs, target_log_ratios):
        weights = importance / jnp.maximum(
            jnp.sum(importance), jnp.finfo(importance.dtype).tiny
        )
        square_root_weights = jnp.sqrt(2.0 * weights)

        def centered_features(candidate):
            return centered_real_log_features(
                unravel(candidate), samples, importance, log_psi_fn
            )

        target_logs = base_logs + target_log_ratios
        target_centered = target_logs - jnp.sum(weights * target_logs)
        target_features = jnp.concatenate(
            (
                square_root_weights * jnp.real(target_centered),
                square_root_weights * jnp.imag(target_centered),
            )
        )
        current_features = centered_features(values)
        residual = target_features - current_features
        solve = solve_linearized_projection_direction(
            values,
            centered_features,
            target_features,
            damping=damping,
            relative_tolerance=lsqr_tolerance,
            absolute_tolerance=lsqr_absolute_tolerance,
            max_iterations=lsqr_max_iterations,
        )
        direction = solve.solution
        candidate_steps = step_size * 0.5 ** jnp.arange(backtracking_steps)
        candidate_steps = jnp.concatenate((jnp.zeros((1,)), candidate_steps))
        candidate_values = (
            values[None, :] + candidate_steps[:, None] * direction[None, :]
        )
        candidate_losses = jax.vmap(
            lambda candidate: jnp.mean(
                (target_features - centered_features(candidate)) ** 2
            )
        )(candidate_values)
        return (
            candidate_values,
            candidate_losses,
            jnp.mean(residual**2),
            jnp.linalg.norm(direction),
            solve,
        )

    compiled_update = jax.jit(update)
    history: list[dict[str, object]] = []
    pool_size = len(sample_pool)
    accepted_score = (
        float(acceptance_score_fn(unravel(flat)))
        if acceptance_score_fn is not None
        else None
    )
    for step in range(1, steps + 1):
        if random_batches:
            key, subkey = jax.random.split(key)
            indices = jax.random.randint(subkey, (batch_size,), 0, pool_size)
        else:
            if batch_size != pool_size:
                raise ValueError("deterministic batches require batch_size == pool size")
            indices = jnp.arange(pool_size)
        previous_flat = flat
        (
            candidate_flats,
            candidate_losses,
            loss_before,
            direction_norm,
            solve,
        ) = compiled_update(
            flat,
            sample_pool[indices],
            pool_importance_weights[indices],
            pool_base_logs[indices],
            pool_target_log_ratios[indices],
        )
        accepted = True
        candidate_score = None
        if acceptance_score_fn is not None:
            accepted = False
            selected = 0
            evaluations = 0
            for candidate_index in range(1, len(candidate_flats)):
                if float(candidate_losses[candidate_index]) > float(loss_before):
                    continue
                evaluations += 1
                score = float(
                    acceptance_score_fn(unravel(candidate_flats[candidate_index]))
                )
                candidate_score = (
                    score if candidate_score is None else min(candidate_score, score)
                )
                if score <= accepted_score:
                    accepted = True
                    selected = candidate_index
                    accepted_score = score
                    candidate_score = score
                    break
            flat = candidate_flats[selected] if accepted else previous_flat
        else:
            selected = int(jnp.argmin(candidate_losses))
            flat = candidate_flats[selected]
            evaluations = 0
        effective_step = (
            step_size * 0.5 ** (selected - 1) if selected > 0 else 0.0
        )
        loss_after = candidate_losses[selected] if accepted else loss_before
        iterations = int(solve.iterations)
        active_history = slice(0, iterations + 1)
        history.append(
            {
                "step": float(step),
                "feature_loss_before": float(loss_before),
                "feature_loss_after": float(loss_after),
                "direction_norm": float(direction_norm),
                "linear_solver": "matrix_free_golub_kahan_lsqr",
                "lsqr_iterations": iterations,
                "lsqr_converged": bool(solve.converged),
                "lsqr_breakdown": bool(solve.breakdown),
                "linear_solver_warm_start": False,
                "primal_residual_norm": float(solve.primal_residual_norm),
                "augmented_residual_norm": float(solve.augmented_residual_norm),
                "normal_residual_norm": float(solve.normal_residual_norm),
                "normal_residual_scale": float(solve.normal_residual_scale),
                "normal_residual_threshold": float(
                    solve.normal_residual_threshold
                ),
                "relative_normal_residual": float(
                    solve.relative_normal_residual
                ),
                "lsqr_residual_estimate": float(
                    solve.lsqr_residual_estimate
                ),
                "primal_residual_history": np.asarray(
                    solve.primal_residual_history[active_history]
                ).tolist(),
                "augmented_residual_history": np.asarray(
                    solve.augmented_residual_history[active_history]
                ).tolist(),
                "normal_residual_history": np.asarray(
                    solve.normal_residual_history[active_history]
                ).tolist(),
                "relative_normal_residual_history": np.asarray(
                    solve.relative_normal_residual_history[active_history]
                ).tolist(),
                "lsqr_residual_estimate_history": np.asarray(
                    solve.lsqr_residual_estimate_history[active_history]
                ).tolist(),
                "effective_step_size": float(effective_step),
                "accepted": bool(accepted),
                "energy_backtracking_evaluations": int(evaluations),
                **(
                    {"acceptance_score": float(candidate_score)}
                    if candidate_score is not None
                    else {}
                ),
            }
        )
    return unravel(flat), history


__all__ = [
    "centered_real_log_features",
    "solve_linearized_projection_direction",
    "train_lsqr_projection",
]
