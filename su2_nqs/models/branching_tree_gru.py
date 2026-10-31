"""Normalized TreeGRU with hidden states propagated along fusion-tree edges."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._core import embed, gru_step, heads
from .compact_tree_gru import (
    CompactTreeGRUConfig,
    init_compact_tree_gru,
    split_tables,
)
from .tree_gru import tree_decisions


init_branching_tree_gru = init_compact_tree_gru


def branching_tree_log_psi(params, labels, config: CompactTreeGRUConfig):
    """Normalized log amplitude with parent-to-child hidden propagation."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},), got {labels.shape}")
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    hidden_by_node = [params["h0"] for _ in range(config.n - 1)]
    previous_by_node = [
        jnp.asarray(config.action_dim, dtype=jnp.int32)
        for _ in range(config.n - 1)
    ]
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid_state = labels[-1] == config.target_q
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        parent_in_range = (parent_q >= 0) & (parent_q <= config.n)
        safe_parent = jnp.clip(parent_q, 0, config.n)
        desired_left = jnp.asarray(1, dtype=jnp.int32) if decision.left_index is None else labels[decision.left_index]
        desired_right = jnp.asarray(1, dtype=jnp.int32) if decision.right_index is None else labels[decision.right_index]
        candidates_left = table_left[decision.order, safe_parent]
        candidates_right = table_right[decision.order, safe_parent]
        mask = table_valid[decision.order, safe_parent]
        matches = mask & (candidates_left == desired_left) & (candidates_right == desired_right)
        action = jnp.argmax(matches).astype(jnp.int32)
        inputs = embed(
            params,
            (
                previous_by_node[decision.node_index],
                safe_parent,
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(params["gru"], hidden_by_node[decision.node_index], inputs)
        logits, phases = heads(params, hidden)
        logits = jnp.where(mask, logits, -1.0e30)
        valid_state = valid_state & parent_in_range & jnp.any(matches)
        log_probability += jax.nn.log_softmax(logits)[action]
        phase += phases[action]
        for child_index in (decision.left_index, decision.right_index):
            if child_index is not None:
                hidden_by_node[child_index] = hidden
                previous_by_node[child_index] = action
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid_state, value, -jnp.inf + 0j)


def branching_tree_probabilities(params, states, config: CompactTreeGRUConfig):
    logs = jax.vmap(lambda state: branching_tree_log_psi(params, state, config))(states)
    return jnp.exp(2.0 * jnp.real(logs))


def sample_branching_tree(params, key, n_samples: int, config: CompactTreeGRUConfig):
    """Sample valid labels while propagating batched hidden states by branch."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(
        params["h0"], (n_samples, config.n - 1, config.hidden_dim)
    )
    previous = jnp.full(
        (n_samples, config.n - 1), config.action_dim, dtype=jnp.int32
    )
    for decision in tree_decisions(config.tree):
        key, subkey = jax.random.split(key)
        parent_q = labels[:, decision.node_index]
        candidates_left = table_left[decision.order, parent_q]
        candidates_right = table_right[decision.order, parent_q]
        mask = table_valid[decision.order, parent_q]
        inputs = embed(
            params,
            (
                previous[:, decision.node_index],
                parent_q,
                jnp.full((n_samples,), decision.left_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.right_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.depth, dtype=jnp.int32),
                jnp.full((n_samples,), decision.order, dtype=jnp.int32),
            ),
        )
        node_hidden = gru_step(
            params["gru"], hidden[:, decision.node_index], inputs
        )
        logits, _ = heads(params, node_hidden)
        action = jax.random.categorical(
            subkey, jnp.where(mask, logits, -1.0e30), axis=-1
        ).astype(jnp.int32)
        chosen_left = jnp.take_along_axis(candidates_left, action[:, None], axis=1)[:, 0]
        chosen_right = jnp.take_along_axis(candidates_right, action[:, None], axis=1)[:, 0]
        for child_index, chosen in (
            (decision.left_index, chosen_left),
            (decision.right_index, chosen_right),
        ):
            if child_index is not None:
                labels = labels.at[:, child_index].set(chosen)
                hidden = hidden.at[:, child_index].set(node_hidden)
                previous = previous.at[:, child_index].set(action)
    return labels
