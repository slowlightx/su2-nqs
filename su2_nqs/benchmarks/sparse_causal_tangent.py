"""Exact full-support tangent diagnostics for sparse causal message heads."""

from __future__ import annotations

from collections.abc import Mapping
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.sparse_causal_tree import (
    SparseCausalTreeGRUConfig,
    sparse_causal_tree_prefix_logs,
)


HEAD_GROUPS: Mapping[str, tuple[str, ...]] = {
    "vector": ("probability_endpoint_head", "phase_endpoint_head"),
    "closed": ("probability_closed_head", "phase_closed_head"),
    "hidden": ("probability_hidden_head", "phase_hidden_head"),
    "probability": (
        "probability_endpoint_head",
        "probability_closed_head",
        "probability_hidden_head",
    ),
    "phase": (
        "phase_endpoint_head",
        "phase_closed_head",
        "phase_hidden_head",
    ),
}
HEAD_NAMES = tuple(
    name
    for name in (
        "probability_endpoint_head",
        "phase_endpoint_head",
        "probability_closed_head",
        "phase_closed_head",
        "probability_hidden_head",
        "phase_hidden_head",
    )
)


def _squared_norm(values: Mapping[str, jax.Array], names) -> jax.Array:
    return sum(jnp.sum(values[name] ** 2) for name in names)


def exact_message_head_tangent_diagnostics(
    params: dict[str, object],
    config: SparseCausalTreeGRUConfig,
    states: jax.Array,
    born_weights: jax.Array,
    target_log_ratios: jax.Array,
) -> dict[str, object]:
    """Decompose the zero-init message-head RHS by component and AR cut.

    The base TreeGRU and recurrent encoder are frozen.  The derivative of each
    conditional is Born-centered separately; summing these centered cut
    derivatives equals the centered derivative of the complete log amplitude.
    ``target_log_ratios`` is the complex teacher displacement relative to the
    base state, as used by the exact/full-support SR gate.
    """

    states = jnp.asarray(states, dtype=jnp.int32)
    born_weights = jnp.asarray(born_weights, dtype=jnp.float64)
    target_log_ratios = jnp.asarray(target_log_ratios, dtype=jnp.complex128)
    if states.ndim != 2 or states.shape[1] != config.n - 1:
        raise ValueError("states have the wrong fusion-label shape")
    if born_weights.shape != (len(states),) or target_log_ratios.shape != (
        len(states),
    ):
        raise ValueError("weights and target ratios require one value per state")
    if np.any(np.asarray(born_weights) < 0.0):
        raise ValueError("Born weights must be non-negative")
    weight_sum = jnp.sum(born_weights)
    weights = born_weights / jnp.maximum(
        weight_sum, jnp.finfo(jnp.float64).tiny
    )
    square_root_weights = jnp.sqrt(2.0 * weights)
    centered_target = target_log_ratios - jnp.sum(weights * target_log_ratios)
    target_feature = jnp.concatenate(
        (
            square_root_weights * jnp.real(centered_target),
            square_root_weights * jnp.imag(centered_target),
        )
    )

    fixed_message = {
        name: value
        for name, value in params["message"].items()
        if name not in HEAD_NAMES
    }
    heads = {name: params["message"][name] for name in HEAD_NAMES}

    def cut_features(candidate_heads):
        message = dict(fixed_message)
        message.update(candidate_heads)
        candidate = {"base": params["base"], "message": message}
        conditionals = jax.vmap(
            lambda state: sparse_causal_tree_prefix_logs(
                candidate, state, config
            )[0]
        )(states)
        centered = conditionals - jnp.sum(
            weights[:, None] * conditionals, axis=0, keepdims=True
        )
        real = (square_root_weights[:, None] * jnp.real(centered)).T
        imaginary = (square_root_weights[:, None] * jnp.imag(centered)).T
        return jnp.concatenate((real, imaginary), axis=1)

    def calculate(candidate_heads):
        features, pullback = jax.vjp(cut_features, candidate_heads)
        n_decisions = features.shape[0]
        cotangents = (
            jnp.eye(n_decisions, dtype=jnp.float64)[:, :, None]
            * target_feature[None, None, :]
        )
        by_cut = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
            cotangents
        )
        aggregate = pullback(
            jnp.broadcast_to(target_feature, features.shape)
        )[0]
        return features, by_cut, aggregate

    started = time.perf_counter()
    features, by_cut, aggregate = jax.jit(calculate)(heads)
    jax.block_until_ready(features)
    elapsed = time.perf_counter() - started

    cut_rows = []
    for decision in range(config.n - 1):
        values = {name: by_cut[name][decision] for name in HEAD_NAMES}
        groups = {
            name: float(jnp.sqrt(_squared_norm(values, members)))
            for name, members in HEAD_GROUPS.items()
        }
        groups["total"] = float(
            jnp.sqrt(_squared_norm(values, HEAD_NAMES))
        )
        cut_rows.append({"decision": decision, **groups})

    aggregate_groups = {
        name: float(jnp.sqrt(_squared_norm(aggregate, members)))
        for name, members in HEAD_GROUPS.items()
    }
    aggregate_groups["total"] = float(
        jnp.sqrt(_squared_norm(aggregate, HEAD_NAMES))
    )
    summed = {
        name: jnp.sum(by_cut[name], axis=0) for name in HEAD_NAMES
    }
    maximum_sum_error = max(
        (
            float(jnp.max(jnp.abs(summed[name] - aggregate[name])))
            for name in HEAD_NAMES
        ),
        default=0.0,
    )
    return {
        "n_states": len(states),
        "n_decisions": config.n - 1,
        "elapsed_seconds": elapsed,
        "target_feature_norm": float(jnp.linalg.norm(target_feature)),
        "aggregate_rhs_norms": aggregate_groups,
        "cut_rhs_norms": cut_rows,
        "maximum_cut_sum_error": maximum_sum_error,
        "all_finite": bool(
            all(
                np.all(np.isfinite(np.asarray(value)))
                for value in jax.tree_util.tree_leaves((by_cut, aggregate))
            )
        ),
    }


__all__ = ["HEAD_GROUPS", "HEAD_NAMES", "exact_message_head_tangent_diagnostics"]
