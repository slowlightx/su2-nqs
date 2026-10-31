"""Sampled matrix-free stochastic-reconfiguration teacher projection."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def train_sampled_sr_projection(
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
    cg_tolerance: float = 1e-5,
    cg_max_iterations: int = 30,
    backtracking_steps: int = 6,
    acceptance_score_fn: Callable[[object], float] | None = None,
    random_batches: bool = True,
    fisher_preconditioner: str = "identity",
    fisher_block_specs: tuple[object, ...] | None = None,
    fisher_score_chunk_size: int = 32,
    maximum_feature_distance: float | None = None,
) -> tuple[object, list[dict[str, object]]]:
    """Project a sampled complex teacher displacement with matrix-free SR.

    The Fisher/Gauss--Newton product is evaluated as ``J.T @ J`` using JVP and
    VJP; no parameter-square matrix or explicit log-derivative matrix is
    materialized.  Overall normalization and global phase are removed by a
    weighted centering operation. ``fisher_preconditioner='physical_block'``
    is an opt-in bounded-memory carrier-block PCG path.  A non-``None``
    ``maximum_feature_distance`` opts into the same projection backend and
    bounds the actual centered-complex-log displacement of every accepted
    candidate.  With both options at their defaults, the established identity
    path below is retained bitwise.
    """

    if steps < 1 or batch_size < 1 or len(sample_pool) < 1:
        raise ValueError("steps, batch size, and sample pool must be positive")
    if damping <= 0.0 or step_size <= 0.0:
        raise ValueError("damping and step size must be positive")
    if backtracking_steps < 1:
        raise ValueError("backtracking_steps must be positive")
    if fisher_preconditioner not in ("identity", "physical_block"):
        raise ValueError(
            "fisher_preconditioner must be 'identity' or 'physical_block'"
        )
    if fisher_score_chunk_size < 1:
        raise ValueError("fisher_score_chunk_size must be positive")
    if maximum_feature_distance is not None and maximum_feature_distance <= 0.0:
        raise ValueError("maximum_feature_distance must be positive")
    if fisher_preconditioner == "physical_block" or maximum_feature_distance is not None:
        # Opt-in provenance seam. The established identity-CG implementation
        # below remains untouched and is still the default. Import lazily to
        # avoid changing its compilation/import surface.
        from .preconditioned_sr_projection import (
            train_preconditioned_sr_projection,
        )

        return train_preconditioned_sr_projection(
            params,
            key,
            sample_pool=sample_pool,
            pool_importance_weights=pool_importance_weights,
            pool_base_logs=pool_base_logs,
            pool_target_log_ratios=pool_target_log_ratios,
            log_psi_fn=log_psi_fn,
            steps=steps,
            batch_size=batch_size,
            solver_control=fisher_preconditioner,
            block_specs=fisher_block_specs,
            score_chunk_size=fisher_score_chunk_size,
            damping=damping,
            step_size=step_size,
            cg_tolerance=cg_tolerance,
            cg_max_iterations=cg_max_iterations,
            backtracking_steps=backtracking_steps,
            maximum_feature_distance=maximum_feature_distance,
            acceptance_score_fn=acceptance_score_fn,
            random_batches=random_batches,
        )
    flat, unravel = ravel_pytree(params)

    def update(values, samples, importance, base_logs, target_log_ratios):
        weights = importance / jnp.maximum(
            jnp.sum(importance), jnp.finfo(importance.dtype).tiny
        )
        square_root_weights = jnp.sqrt(2.0 * weights)

        def centered_features(candidate):
            logs = jax.vmap(
                lambda sample: log_psi_fn(unravel(candidate), sample)
            )(samples)
            centered = logs - jnp.sum(weights * logs)
            return jnp.concatenate(
                (
                    square_root_weights * jnp.real(centered),
                    square_root_weights * jnp.imag(centered),
                )
            )

        target_logs = base_logs + target_log_ratios
        target_centered = target_logs - jnp.sum(weights * target_logs)
        target_features = jnp.concatenate(
            (
                square_root_weights * jnp.real(target_centered),
                square_root_weights * jnp.imag(target_centered),
            )
        )
        current_features, pullback = jax.vjp(centered_features, values)
        residual = target_features - current_features
        rhs = pullback(residual)[0]

        def fisher_product(vector):
            _, tangent = jax.jvp(centered_features, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        direction, _ = jax.scipy.sparse.linalg.cg(
            fisher_product,
            rhs,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        normal_equation_residual = fisher_product(direction) - rhs
        cg_relative_residual = jnp.linalg.norm(normal_equation_residual) / jnp.maximum(
            jnp.linalg.norm(rhs), jnp.finfo(values.dtype).tiny
        )
        candidate_steps = step_size * 0.5 ** jnp.arange(backtracking_steps)
        candidate_steps = jnp.concatenate((jnp.zeros((1,)), candidate_steps))
        candidate_values = values[None, :] + candidate_steps[:, None] * direction[None, :]
        candidate_losses = jax.vmap(
            lambda candidate: jnp.mean(
                (target_features - centered_features(candidate)) ** 2
            )
        )(candidate_values)
        return (
            candidate_values,
            candidate_losses,
            jnp.mean(residual**2),
            jnp.linalg.norm(rhs),
            jnp.linalg.norm(direction),
            cg_relative_residual,
        )

    compiled_update = jax.jit(update)
    history = []
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
            rhs_norm,
            direction_norm,
            cg_relative_residual,
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
            # Index zero is the unchanged state.  The remaining candidates are
            # ordered from the largest step to successive halvings.  A genuine
            # trust-region search must test the smaller candidates after an
            # energy rejection instead of rolling back the whole SR solve.
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
        effective_step = step_size * 0.5 ** (selected - 1) if selected > 0 else 0.0
        loss_after = candidate_losses[selected] if accepted else loss_before
        history.append(
            {
                "step": float(step),
                "feature_loss_before": float(loss_before),
                "feature_loss_after": float(loss_after),
                "rhs_norm": float(rhs_norm),
                "direction_norm": float(direction_norm),
                "cg_relative_residual": float(cg_relative_residual),
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
