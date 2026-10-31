"""Normalized autoregressive GRU on left-associated SU(2) fusion paths."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..su2.fusion_path import count_paths
from ._core import embed, gru_step, heads, init_gru_core

ACTIONS = jnp.asarray([-1, 1], dtype=jnp.int32)
BOS_TOKEN = 2


@dataclass(frozen=True)
class FusionGRUConfig:
    n: int
    target_q: int = 0
    embedding_dim: int = 16
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if count_paths(self.n, self.target_q) == 0:
            raise ValueError(f"target q={self.target_q} is unreachable for N={self.n}")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")


def init_fusion_gru(key: jax.Array, config: FusionGRUConfig) -> dict[str, object]:
    """Initialize a pure-JAX parameter pytree."""

    # previous token (minus, plus, BOS), current q, step, spins remaining
    return init_gru_core(
        key,
        embedding_sizes=(3, config.n + 1, max(config.n, 1), config.n + 1),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )


def _valid_action_mask(current_q: jax.Array, remaining_after: int, target_q: int) -> jax.Array:
    next_q = current_q[..., None] + ACTIONS
    distance = jnp.abs(next_q - target_q)
    return (
        (next_q >= 0)
        & (distance <= remaining_after)
        & ((remaining_after - distance) % 2 == 0)
    )


def log_psi(
    params: dict[str, object],
    path: jax.Array,
    config: FusionGRUConfig,
) -> jax.Array:
    """Return ``0.5 log p(path) + i phase(path)``.

    Exact reachability masks make ``p`` normalized over the fixed-total-spin
    fusion paths for every choice of network parameters.
    """

    path = jnp.asarray(path, dtype=jnp.int32)
    hidden = params["h0"]
    previous_token = jnp.asarray(BOS_TOKEN, dtype=jnp.int32)
    current_q = jnp.asarray(1, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    structurally_valid = path.shape == (config.n,)
    if path.shape != (config.n,):
        raise ValueError(f"expected path shape ({config.n},), got {path.shape}")
    structurally_valid = structurally_valid & (path[0] == 1)

    for decision in range(config.n - 1):
        remaining_after = config.n - decision - 2
        inputs = embed(
            params,
            (previous_token, current_q, decision, remaining_after + 1),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phase_values = heads(params, hidden)
        mask = _valid_action_mask(current_q, remaining_after, config.target_q)
        masked_logits = jnp.where(mask, logits, -1.0e30)
        action = path[decision + 1] - current_q
        action_index = jnp.where(action == 1, 1, 0)
        selected_valid = mask[action_index] & ((action == -1) | (action == 1))
        structurally_valid = structurally_valid & selected_valid
        log_probability += jax.nn.log_softmax(masked_logits)[action_index]
        phase += phase_values[action_index]
        current_q = path[decision + 1]
        previous_token = action_index

    structurally_valid = structurally_valid & (current_q == config.target_q)
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(structurally_valid, value, -jnp.inf + 0j)


def sample(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: FusionGRUConfig,
) -> jax.Array:
    """Draw independent exact autoregressive fusion paths."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous_token = jnp.full((n_samples,), BOS_TOKEN, dtype=jnp.int32)
    current_q = jnp.ones((n_samples,), dtype=jnp.int32)
    labels = [current_q]
    for decision in range(config.n - 1):
        key, subkey = jax.random.split(key)
        remaining_after = config.n - decision - 2
        step = jnp.full((n_samples,), decision, dtype=jnp.int32)
        remaining = jnp.full((n_samples,), remaining_after + 1, dtype=jnp.int32)
        inputs = embed(params, (previous_token, current_q, step, remaining))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        mask = _valid_action_mask(current_q, remaining_after, config.target_q)
        logits = jnp.where(mask, logits, -1.0e30)
        action_index = jax.random.categorical(subkey, logits, axis=-1)
        current_q = current_q + ACTIONS[action_index]
        previous_token = action_index
        labels.append(current_q)
    return jnp.stack(labels, axis=1)


def path_probabilities(
    params: dict[str, object],
    paths: jax.Array,
    config: FusionGRUConfig,
) -> jax.Array:
    """Evaluate normalized probabilities for a batch of paths."""

    values = jax.vmap(lambda path: log_psi(params, path, config))(paths)
    return jnp.exp(2.0 * jnp.real(values))


@dataclass(frozen=True)
class FusionGRU:
    """Object-oriented wrapper around the functional fusion-GRU API."""

    config: FusionGRUConfig

    def init(self, key: jax.Array) -> dict[str, object]:
        return init_fusion_gru(key, self.config)

    def log_psi(self, params: dict[str, object], path: jax.Array) -> jax.Array:
        return log_psi(params, path, self.config)

    def sample(self, params: dict[str, object], key: jax.Array, n_samples: int) -> jax.Array:
        return sample(params, key, n_samples, self.config)

