"""Compact normalized TreeGRU using local admissible split indices."""

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
def split_tables(tree: FusionTree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padded ``left_q, right_q, valid`` lookup by decision and parent q."""

    n = tree.n_leaves
    all_candidates: list[list[tuple[tuple[int, int], ...]]] = []
    maximum = 1
    for decision in tree_decisions(tree):
        by_parent = []
        for parent_q in range(n + 1):
            candidates = tuple(
                (left_q, right_q)
                for left_q in _reachable_q(decision.left_leaves)
                for right_q in _reachable_q(decision.right_leaves)
                if _couples(left_q, right_q, parent_q)
            )
            by_parent.append(candidates)
            maximum = max(maximum, len(candidates))
        all_candidates.append(by_parent)
    shape = (len(all_candidates), n + 1, maximum)
    left = np.zeros(shape, dtype=np.int32)
    right = np.zeros(shape, dtype=np.int32)
    valid = np.zeros(shape, dtype=bool)
    for decision_index, by_parent in enumerate(all_candidates):
        for parent_q, candidates in enumerate(by_parent):
            for slot, (left_q, right_q) in enumerate(candidates):
                left[decision_index, parent_q, slot] = left_q
                right[decision_index, parent_q, slot] = right_q
                valid[decision_index, parent_q, slot] = True
    return left, right, valid


@dataclass(frozen=True)
class CompactTreeGRUConfig:
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
        return int(split_tables(self.tree)[0].shape[-1])


def init_compact_tree_gru(
    key: jax.Array, config: CompactTreeGRUConfig
) -> dict[str, object]:
    """Initialize the compact TreeGRU parameter pytree."""

    return init_gru_core(
        key,
        embedding_sizes=(
            config.action_dim + 1,
            config.n + 1,
            config.n + 1,
            config.n + 1,
            config.n,
            max(config.n - 1, 1),
        ),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.action_dim,
    )


def compact_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: CompactTreeGRUConfig,
) -> jax.Array:
    """Normalized complex log amplitude for postorder internal labels."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    expected = (config.n - 1,)
    if labels.shape != expected:
        raise ValueError(f"expected label shape {expected}, got {labels.shape}")
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
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
                previous,
                safe_parent,
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        logits = jnp.where(mask, logits, -1.0e30)
        valid_state = valid_state & parent_in_range & jnp.any(matches)
        log_probability += jax.nn.log_softmax(logits)[action]
        phase += phases[action]
        previous = action
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid_state, value, -jnp.inf + 0j)


def fixed_compact_tree_path_data(
    params: dict[str, object],
    labels: jax.Array,
    config: CompactTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return fixed logits, phases, masks, and selected actions per decision.

    This exact-small helper exposes the same normalized conditionals as
    :func:`compact_tree_log_psi` without exposing future labels to an earlier
    conditional.  It is intended for fitting a small frozen-base residual.
    """

    labels = jnp.asarray(labels, dtype=jnp.int32)
    expected = (config.n - 1,)
    if labels.shape != expected:
        raise ValueError(f"expected label shape {expected}, got {labels.shape}")
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    hidden = params["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    all_logits, all_phases, all_masks, all_actions = [], [], [], []
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        candidates_left = table_left[decision.order, parent_q]
        candidates_right = table_right[decision.order, parent_q]
        mask = table_valid[decision.order, parent_q]
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
        matches = (
            mask
            & (candidates_left == desired_left)
            & (candidates_right == desired_right)
        )
        action = jnp.argmax(matches).astype(jnp.int32)
        inputs = embed(
            params,
            (
                previous,
                parent_q,
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        all_logits.append(logits)
        all_phases.append(phases)
        all_masks.append(mask)
        all_actions.append(action)
        previous = action
    return (
        jnp.stack(all_logits),
        jnp.stack(all_phases),
        jnp.stack(all_masks),
        jnp.stack(all_actions),
    )


def compact_tree_prefix_logs(
    params: dict[str, object],
    labels: jax.Array,
    config: CompactTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return conditional and cumulative complex log amplitudes.

    This is a diagnostic counterpart of :func:`compact_tree_log_psi` for a
    *complete* postorder label assignment.  Entry ``d`` of the first returned
    array is

    ``0.5 * log p(action_d | action_<d) + 1j * phase_d``;

    entry ``d`` of the second is its cumulative sum through decision ``d``.
    Thus the last cumulative entry is exactly ``compact_tree_log_psi``.  The
    computation follows the same root-first filtration and applies the same
    exact split mask before every softmax.  It does not expose hidden states or
    inspect labels that have not yet been revealed by the current decision.

    Invalid complete assignments return ``-inf + 0j`` at every entry.  The
    public model evaluator remains the appropriate API when only the final
    amplitude is needed.
    """

    labels = jnp.asarray(labels, dtype=jnp.int32)
    expected = (config.n - 1,)
    if labels.shape != expected:
        raise ValueError(f"expected label shape {expected}, got {labels.shape}")
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    hidden = params["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    valid_state = labels[-1] == config.target_q
    conditionals = []
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
        candidates_left = table_left[decision.order, safe_parent]
        candidates_right = table_right[decision.order, safe_parent]
        mask = table_valid[decision.order, safe_parent]
        matches = (
            mask
            & (candidates_left == desired_left)
            & (candidates_right == desired_right)
        )
        action = jnp.argmax(matches).astype(jnp.int32)
        inputs = embed(
            params,
            (
                previous,
                safe_parent,
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        logits = jnp.where(mask, logits, -1.0e30)
        valid_state = valid_state & parent_in_range & jnp.any(matches)
        conditionals.append(0.5 * jax.nn.log_softmax(logits)[action] + 1j * phases[action])
        previous = action
    conditional = jnp.stack(conditionals)
    cumulative = jnp.cumsum(conditional)
    invalid = jnp.full_like(conditional, -jnp.inf + 0j)
    return (
        jnp.where(valid_state, conditional, invalid),
        jnp.where(valid_state, cumulative, invalid),
    )


def sample_compact_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: CompactTreeGRUConfig,
) -> jax.Array:
    """Draw valid postorder label assignments from the compact model."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    for decision in tree_decisions(config.tree):
        key, subkey = jax.random.split(key)
        parent_q = labels[:, decision.node_index]
        candidates_left = table_left[decision.order, parent_q]
        candidates_right = table_right[decision.order, parent_q]
        mask = table_valid[decision.order, parent_q]
        inputs = embed(
            params,
            (
                previous,
                parent_q,
                jnp.full((n_samples,), decision.left_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.right_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.depth, dtype=jnp.int32),
                jnp.full((n_samples,), decision.order, dtype=jnp.int32),
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        logits = jnp.where(mask, logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        chosen_left = jnp.take_along_axis(candidates_left, action[:, None], axis=1)[:, 0]
        chosen_right = jnp.take_along_axis(candidates_right, action[:, None], axis=1)[:, 0]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(chosen_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(chosen_right)
        previous = action
    return labels


def sample_compact_tree_indexed(
    params: dict[str, object],
    key: jax.Array,
    global_indices: jax.Array,
    config: CompactTreeGRUConfig,
) -> jax.Array:
    """Draw samples from deterministic per-index random streams.

    Each row starts from ``jax.random.fold_in(key, global_index)`` and then
    advances its own key at every autoregressive decision.  Consequently, the
    sample associated with an index is independent of batch position, batch
    size, and how the indices are partitioned across devices.  This makes the
    sampler suitable for sample-axis data parallelism while retaining the
    exact reachability mask before every categorical draw.

    ``global_indices`` must be a non-empty rank-one integer array.  Repeated
    indices intentionally produce repeated samples.
    """

    global_indices = jnp.asarray(global_indices)
    if global_indices.ndim != 1:
        raise ValueError("global_indices must be a rank-one array")
    if global_indices.shape[0] < 1:
        raise ValueError("global_indices must be non-empty")
    if not jnp.issubdtype(global_indices.dtype, jnp.integer):
        raise TypeError("global_indices must have an integer dtype")

    n_samples = global_indices.shape[0]
    sample_keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(
        global_indices
    )
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    for decision in tree_decisions(config.tree):
        split_keys = jax.vmap(jax.random.split)(sample_keys)
        sample_keys = split_keys[:, 0]
        decision_keys = split_keys[:, 1]
        parent_q = labels[:, decision.node_index]
        candidates_left = table_left[decision.order, parent_q]
        candidates_right = table_right[decision.order, parent_q]
        mask = table_valid[decision.order, parent_q]
        inputs = embed(
            params,
            (
                previous,
                parent_q,
                jnp.full((n_samples,), decision.left_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.right_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.depth, dtype=jnp.int32),
                jnp.full((n_samples,), decision.order, dtype=jnp.int32),
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        logits = jnp.where(mask, logits, -1.0e30)
        action = jax.vmap(
            lambda draw_key, row: jax.random.categorical(
                draw_key, row, axis=-1
            )
        )(decision_keys, logits).astype(jnp.int32)
        chosen_left = jnp.take_along_axis(
            candidates_left, action[:, None], axis=1
        )[:, 0]
        chosen_right = jnp.take_along_axis(
            candidates_right, action[:, None], axis=1
        )[:, 0]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(chosen_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(chosen_right)
        previous = action
    return labels


def compact_tree_probabilities(
    params: dict[str, object],
    states: jax.Array,
    config: CompactTreeGRUConfig,
) -> jax.Array:
    logs = jax.vmap(lambda state: compact_tree_log_psi(params, state, config))(states)
    return jnp.exp(2.0 * jnp.real(logs))
