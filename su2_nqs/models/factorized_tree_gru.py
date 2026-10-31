"""Normalized TreeGRU with factorized binary-fusion decisions.

At an internal node the compact TreeGRU treats the admissible pair
``(q_left, q_right)`` as one categorical action.  On branching trees that
joint alphabet grows much faster than either child alphabet.  This module
instead samples ``q_left`` first and ``q_right`` conditionally, applying an
exact reachability mask before both softmaxes.  The factorization preserves
the same fusion-tree Hilbert space and exact normalization while exposing the
two physical legs of a binary fusion as separate causal events.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import FusionTree
from ._core import embed, gru_step, heads, init_gru_core
from .tree_gru import tree_decisions


def _reachable_q(leaves: int) -> tuple[int, ...]:
    return tuple(range(leaves % 2, leaves + 1, 2))


def _couples(left_q: int, right_q: int, parent_q: int) -> bool:
    return (
        abs(left_q - right_q) <= parent_q <= left_q + right_q
        and (left_q + right_q - parent_q) % 2 == 0
    )


@lru_cache(maxsize=None)
def factorized_split_tables(
    tree: FusionTree,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return padded left and conditional-right doubled-spin tables.

    The left tables have indices ``[decision, parent_q, slot]``.  The right
    tables have indices ``[decision, parent_q, left_q, slot]``.  Values are
    doubled integer spins and the accompanying boolean arrays are the masks
    that must be applied before each softmax.
    """

    decisions = tree_decisions(tree)
    n = tree.n_leaves
    left_rows: list[list[tuple[int, ...]]] = []
    right_rows: list[list[list[tuple[int, ...]]]] = []
    maximum = 1
    for decision in decisions:
        by_parent_left = []
        by_parent_right = []
        for parent_q in range(n + 1):
            right_by_left = []
            left_candidates = []
            for left_q in range(n + 1):
                right_candidates = tuple(
                    right_q
                    for right_q in _reachable_q(decision.right_leaves)
                    if left_q in _reachable_q(decision.left_leaves)
                    and _couples(left_q, right_q, parent_q)
                )
                right_by_left.append(right_candidates)
                if right_candidates:
                    left_candidates.append(left_q)
                    maximum = max(maximum, len(right_candidates))
            maximum = max(maximum, len(left_candidates))
            by_parent_left.append(tuple(left_candidates))
            by_parent_right.append(right_by_left)
        left_rows.append(by_parent_left)
        right_rows.append(by_parent_right)

    left = np.zeros((len(decisions), n + 1, maximum), dtype=np.int32)
    left_valid = np.zeros_like(left, dtype=bool)
    right = np.zeros(
        (len(decisions), n + 1, n + 1, maximum), dtype=np.int32
    )
    right_valid = np.zeros_like(right, dtype=bool)
    for decision, by_parent in enumerate(left_rows):
        for parent_q, candidates in enumerate(by_parent):
            left[decision, parent_q, : len(candidates)] = candidates
            left_valid[decision, parent_q, : len(candidates)] = True
            for left_q, right_candidates in enumerate(
                right_rows[decision][parent_q]
            ):
                right[
                    decision, parent_q, left_q, : len(right_candidates)
                ] = right_candidates
                right_valid[
                    decision, parent_q, left_q, : len(right_candidates)
                ] = True
    return left, left_valid, right, right_valid


@dataclass(frozen=True)
class FactorizedTreeGRUConfig:
    """Configuration for a two-stage binary-fusion decoder."""

    tree: FusionTree
    target_q: int = 0
    embedding_dim: int = 4
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if not self.tree.target_is_reachable(self.target_q):
            raise ValueError(f"target q={self.target_q} is unreachable")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")

    @property
    def n(self) -> int:
        return self.tree.n_leaves

    @property
    def action_dim(self) -> int:
        return int(factorized_split_tables(self.tree)[0].shape[-1])


def init_factorized_tree_gru(
    key: jax.Array, config: FactorizedTreeGRUConfig
) -> dict[str, object]:
    """Initialize the shared two-stage TreeGRU core."""

    return init_gru_core(
        key,
        embedding_sizes=(
            config.action_dim + 1,
            config.n + 1,
            config.n + 1,
            config.n + 1,
            config.n + 1,
            2 * max(config.n - 1, 1),
        ),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.action_dim,
    )


def _step(
    params: dict[str, object],
    hidden: jax.Array,
    previous: jax.Array,
    parent_q: jax.Array,
    context_q: jax.Array,
    decision,
    stage: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    index_shape = parent_q.shape
    inputs = embed(
        params,
        (
            previous,
            parent_q,
            context_q,
            jnp.broadcast_to(
                jnp.asarray(decision.left_leaves, dtype=jnp.int32), index_shape
            ),
            jnp.broadcast_to(
                jnp.asarray(decision.right_leaves, dtype=jnp.int32), index_shape
            ),
            jnp.broadcast_to(
                jnp.asarray(2 * decision.order + stage, dtype=jnp.int32),
                index_shape,
            ),
        ),
    )
    hidden = gru_step(params["gru"], hidden, inputs)
    logits, phases = heads(params, hidden)
    return hidden, logits, phases


def factorized_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: FactorizedTreeGRUConfig,
) -> jax.Array:
    """Return the normalized complex log amplitude of one tree state."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(
            f"expected label shape ({config.n - 1},), got {labels.shape}"
        )
    left, left_valid, right, right_valid = (
        jnp.asarray(value) for value in factorized_split_tables(config.tree)
    )
    hidden = params["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid_state = labels[-1] == config.target_q
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        parent_in_range = (parent_q >= 0) & (parent_q <= config.n)
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

        hidden, logits, phases = _step(
            params,
            hidden,
            previous,
            safe_parent,
            jnp.asarray(0, dtype=jnp.int32),
            decision,
            0,
        )
        candidates = left[decision.order, safe_parent]
        mask = left_valid[decision.order, safe_parent]
        left_matches = mask & (candidates == desired_left)
        left_action = jnp.argmax(left_matches).astype(jnp.int32)
        log_probability += jax.nn.log_softmax(
            jnp.where(mask, logits, -1.0e30)
        )[left_action]
        phase += phases[left_action]
        valid_state = valid_state & parent_in_range & jnp.any(left_matches)
        previous = left_action

        left_in_range = (desired_left >= 0) & (desired_left <= config.n)
        safe_left = jnp.clip(desired_left, 0, config.n)
        hidden, logits, phases = _step(
            params,
            hidden,
            previous,
            safe_parent,
            safe_left,
            decision,
            1,
        )
        candidates = right[decision.order, safe_parent, safe_left]
        mask = right_valid[decision.order, safe_parent, safe_left]
        right_matches = mask & (candidates == desired_right)
        right_action = jnp.argmax(right_matches).astype(jnp.int32)
        log_probability += jax.nn.log_softmax(
            jnp.where(mask, logits, -1.0e30)
        )[right_action]
        phase += phases[right_action]
        valid_state = (
            valid_state & left_in_range & jnp.any(right_matches)
        )
        previous = right_action
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid_state, value, -jnp.inf + 0j)


def factorized_tree_probabilities(
    params: dict[str, object],
    states: jax.Array,
    config: FactorizedTreeGRUConfig,
) -> jax.Array:
    """Evaluate exact Born probabilities for a batch of tree states."""

    logs = jax.vmap(
        lambda state: factorized_tree_log_psi(params, state, config)
    )(states)
    return jnp.exp(2.0 * jnp.real(logs))


def sample_factorized_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: FactorizedTreeGRUConfig,
) -> jax.Array:
    """Draw valid tree states with both exact conditional masks."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    left, left_valid, right, right_valid = (
        jnp.asarray(value) for value in factorized_split_tables(config.tree)
    )
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    for decision in tree_decisions(config.tree):
        parent_q = labels[:, decision.node_index]
        hidden, logits, _ = _step(
            params,
            hidden,
            previous,
            parent_q,
            jnp.zeros((n_samples,), dtype=jnp.int32),
            decision,
            0,
        )
        mask = left_valid[decision.order, parent_q]
        key, subkey = jax.random.split(key)
        left_action = jax.random.categorical(
            subkey, jnp.where(mask, logits, -1.0e30), axis=-1
        ).astype(jnp.int32)
        candidates = left[decision.order, parent_q]
        left_q = jnp.take_along_axis(
            candidates, left_action[:, None], axis=1
        )[:, 0]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(left_q)
        previous = left_action

        hidden, logits, _ = _step(
            params,
            hidden,
            previous,
            parent_q,
            left_q,
            decision,
            1,
        )
        mask = right_valid[decision.order, parent_q, left_q]
        key, subkey = jax.random.split(key)
        right_action = jax.random.categorical(
            subkey, jnp.where(mask, logits, -1.0e30), axis=-1
        ).astype(jnp.int32)
        candidates = right[decision.order, parent_q, left_q]
        right_q = jnp.take_along_axis(
            candidates, right_action[:, None], axis=1
        )[:, 0]
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(right_q)
        previous = right_action
    return labels


__all__ = [
    "FactorizedTreeGRUConfig",
    "factorized_split_tables",
    "factorized_tree_log_psi",
    "factorized_tree_probabilities",
    "init_factorized_tree_gru",
    "sample_factorized_tree",
]
