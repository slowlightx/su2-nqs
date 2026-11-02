"""Normalized top-down autoregressive model on a binary SU(2) fusion tree."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp

from ..su2.fusion_tree import FusionTree, Leaf, TreeNode
from ._core import embed, gru_step, heads, init_gru_core


@dataclass(frozen=True)
class TreeDecision:
    node_index: int
    left_index: int | None
    right_index: int | None
    left_leaves: int
    right_leaves: int
    depth: int
    order: int


def _leaf_count(node: TreeNode) -> int:
    if isinstance(node, Leaf):
        return 1
    return _leaf_count(node.left) + _leaf_count(node.right)


@lru_cache(maxsize=None)
def tree_decisions(tree: FusionTree) -> tuple[TreeDecision, ...]:
    """Internal nodes in root-first order with postorder label indices."""

    indices = {node: index for index, node in enumerate(tree.internal_nodes)}
    decisions: list[TreeDecision] = []

    def visit(node: TreeNode, depth: int) -> None:
        if isinstance(node, Leaf):
            return
        decisions.append(
            TreeDecision(
                node_index=indices[node],
                left_index=None if isinstance(node.left, Leaf) else indices[node.left],
                right_index=None if isinstance(node.right, Leaf) else indices[node.right],
                left_leaves=_leaf_count(node.left),
                right_leaves=_leaf_count(node.right),
                depth=depth,
                order=len(decisions),
            )
        )
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(tree.root, 0)
    return tuple(decisions)


@dataclass(frozen=True)
class TreeGRUConfig:
    tree: FusionTree
    target_q: int = 0
    embedding_dim: int = 2
    hidden_dim: int = 4

    def __post_init__(self) -> None:
        if not self.tree.target_is_reachable(self.target_q):
            raise ValueError(f"target q={self.target_q} is unreachable")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")

    @property
    def n(self) -> int:
        return self.tree.n_leaves

    @property
    def q_size(self) -> int:
        return self.n + 1

    @property
    def action_dim(self) -> int:
        return self.q_size * self.q_size


def init_tree_gru(key: jax.Array, config: TreeGRUConfig) -> dict[str, object]:
    """Initialize a TreeGRU parameter pytree."""

    return init_gru_core(
        key,
        embedding_sizes=(
            config.action_dim + 1,
            config.q_size,
            config.q_size,
            config.q_size,
            config.n,
            max(config.n - 1, 1),
        ),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.action_dim,
    )


def _reachable(q: jax.Array, leaves: int) -> jax.Array:
    return (q <= leaves) & ((leaves - q) % 2 == 0)


def _split_mask(
    parent_q: jax.Array,
    decision: TreeDecision,
    config: TreeGRUConfig,
) -> jax.Array:
    actions = jnp.arange(config.action_dim, dtype=jnp.int32)
    left_q = actions // config.q_size
    right_q = actions % config.q_size
    left_valid = _reachable(left_q, decision.left_leaves)
    right_valid = _reachable(right_q, decision.right_leaves)
    coupling_valid = (
        (jnp.abs(left_q - right_q) <= parent_q)
        & (parent_q <= left_q + right_q)
        & ((left_q + right_q - parent_q) % 2 == 0)
    )
    return left_valid & right_valid & coupling_valid


def tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: TreeGRUConfig,
) -> jax.Array:
    """Return normalized complex log amplitude for postorder tree labels."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    expected = (config.n - 1,)
    if labels.shape != expected:
        raise ValueError(f"expected label shape {expected}, got {labels.shape}")
    hidden = params["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid = labels[-1] == config.target_q
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        left_q = jnp.asarray(1, dtype=jnp.int32) if decision.left_index is None else labels[decision.left_index]
        right_q = jnp.asarray(1, dtype=jnp.int32) if decision.right_index is None else labels[decision.right_index]
        inputs = embed(
            params,
            (
                previous,
                jnp.clip(parent_q, 0, config.n),
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        mask = _split_mask(parent_q, decision, config)
        logits = jnp.where(mask, logits, -1.0e30)
        in_range = (left_q >= 0) & (left_q <= config.n) & (right_q >= 0) & (right_q <= config.n)
        safe_left = jnp.clip(left_q, 0, config.n)
        safe_right = jnp.clip(right_q, 0, config.n)
        action = safe_left * config.q_size + safe_right
        valid = valid & in_range & mask[action]
        log_probability += jax.nn.log_softmax(logits)[action]
        phase += phases[action]
        previous = action
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid, value, -jnp.inf + 0j)


def sample_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: TreeGRUConfig,
) -> jax.Array:
    """Sample valid postorder label assignments with fixed root spin."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    for decision in tree_decisions(config.tree):
        key, subkey = jax.random.split(key)
        parent_q = labels[:, decision.node_index]
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
        mask = _split_mask(parent_q[:, None], decision, config)
        logits = jnp.where(mask, logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        left_q = action // config.q_size
        right_q = action % config.q_size
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(left_q)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(right_q)
        previous = action
    return labels


def tree_state_probabilities(
    params: dict[str, object],
    states: jax.Array,
    config: TreeGRUConfig,
) -> jax.Array:
    values = jax.vmap(lambda state: tree_log_psi(params, state, config))(states)
    return jnp.exp(2.0 * jnp.real(values))


@dataclass(frozen=True)
class TreeGRU:
    config: TreeGRUConfig

    def init(self, key: jax.Array) -> dict[str, object]:
        return init_tree_gru(key, self.config)

    def log_psi(self, params: dict[str, object], labels: jax.Array) -> jax.Array:
        return tree_log_psi(params, labels, self.config)

    def sample(self, params: dict[str, object], key: jax.Array, n_samples: int) -> jax.Array:
        return sample_tree(params, key, n_samples, self.config)
