"""Analytic zero-head tangent of the exact-carrier causal decoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..models._core import embed, gru_step, heads
from ..models.compact_tree_gru import split_tables
from ..models.exact_carrier_tree import (
    DECODER_HEAD_NAMES,
    ExactCarrierTreeGRUConfig,
    _candidate_states_and_scores,
    _plan_value,
    _zero_carrier,
)
from ..models.tree_gru import tree_decisions


def _candidate_role_feature(
    state: tuple[jax.Array, jax.Array],
    base_hidden: jax.Array,
    decision: int,
    config: ExactCarrierTreeGRUConfig,
) -> jax.Array:
    """Flatten the derivative of one scalar decoder score for one action."""

    endpoint, closed = state
    plan = config.recurrence_plan
    valid = np.asarray(_plan_value(plan, "event_valid"), dtype=bool)
    last = int(np.flatnonzero(valid[decision])[-1])
    sites = jnp.asarray(_plan_value(plan, "new_endpoint_sites"), dtype=jnp.int32)[
        decision, last
    ]
    active = jnp.asarray(_plan_value(plan, "new_endpoint_mask"), dtype=jnp.float64)[
        decision, last
    ]
    safe_sites = jnp.clip(sites, 0, config.n - 1)
    context = jnp.concatenate((base_hidden, jnp.ones((1,), dtype=jnp.float64)))
    site_selector = jax.nn.one_hot(safe_sites, config.n, dtype=jnp.float64)
    endpoint_feature = jnp.einsum(
        "wi,wd,wh,w->idh",
        site_selector,
        endpoint,
        jnp.broadcast_to(context, (config.endpoint_slots, len(context))),
        active,
    )
    endpoint_feature /= jnp.sqrt(jnp.maximum(1.0, 3.0 * jnp.sum(active)))
    closed_feature = jnp.zeros(
        (config.n - 1, config.hidden_dim + 1), dtype=jnp.float64
    ).at[decision].set(closed * context)
    return jnp.concatenate((endpoint_feature.reshape(-1), closed_feature.reshape(-1)))


def exact_carrier_decoder_log_derivatives(
    params: dict[str, object],
    labels: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return analytic probability/phase head derivatives for one valid state.

    Decoder heads must be exactly zero.  If ``f_a`` is the candidate carrier x
    prefix-environment feature, autoregressive normalization gives

    ``d log(psi) / d theta_probability = 0.5 (f_chosen - E_p[f_a])``

    at each cut, whereas the phase derivative is ``i f_chosen``.  The returned
    arrays omit the explicit factor ``i``: the first is the real probability
    block and the second is the real coefficient of the imaginary phase block.
    """

    if any(
        bool(np.any(np.asarray(params["decoder"][name]) != 0.0))
        for name in DECODER_HEAD_NAMES
    ):
        raise ValueError("analytic exact-carrier tangent requires exactly zero heads")
    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},)")
    base = params["base"]
    decoder = params["decoder"]
    left, right, table_valid = (jnp.asarray(value) for value in split_tables(config.tree))
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    endpoint, closed = _zero_carrier(config)
    role_size = (
        config.n * 3 * (config.hidden_dim + 1)
        + (config.n - 1) * (config.hidden_dim + 1)
    )
    probability_derivative = jnp.zeros((role_size,), dtype=jnp.float64)
    phase_derivative = jnp.zeros((role_size,), dtype=jnp.float64)
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        safe_parent = jnp.clip(parent_q, 0, config.n)
        desired_left = (
            jnp.asarray(1, dtype=jnp.int32)
            if decision.left_index is None
            else labels[decision.left_index]
        )
        desired_right = (
            jnp.asarray(1, dtype=jnp.int32)
            if decision.right_index is None
            else labels[decision.right_index]
        )
        candidates = left[decision.order, safe_parent]
        candidate_right = right[decision.order, safe_parent]
        mask = table_valid[decision.order, safe_parent]
        matches = mask & (candidates == desired_left) & (candidate_right == desired_right)
        action = jnp.argmax(matches).astype(jnp.int32)
        inputs = embed(
            base,
            (
                previous,
                safe_parent,
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(base["gru"], hidden, inputs)
        base_logits, _ = heads(base, hidden)
        candidate_states, _, _ = _candidate_states_and_scores(
            decoder,
            endpoint,
            closed,
            hidden,
            decision.order,
            safe_parent,
            candidates,
            mask,
            config,
        )
        features = jax.vmap(
            lambda z, c: _candidate_role_feature(
                (z, c), hidden, decision.order, config
            )
        )(candidate_states[0], candidate_states[1])
        masked_logits = jnp.where(mask, base_logits, -1.0e30)
        conditional_probability = jax.nn.softmax(masked_logits)
        probability_derivative += 0.5 * (
            features[action]
            - jnp.sum(conditional_probability[:, None] * features, axis=0)
        )
        phase_derivative += features[action]
        endpoint = candidate_states[0][action]
        closed = candidate_states[1][action]
        previous = action
    return probability_derivative, phase_derivative


__all__ = ["exact_carrier_decoder_log_derivatives"]
