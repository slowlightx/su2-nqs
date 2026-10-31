"""SU(2) fusion GRU conditioned by a persistent low-dimensional state token.

Unlike a conventional shared trunk with separate output heads, all recurrent
and output weights are common.  A head identity enters as a boundary token at
every fusion decision.  This tests whether several low-energy states can share
one local fusion rule while differing through a compact state-sector label.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..su2.fusion_path import count_paths
from ._core import embed, gru_step, heads, init_gru_core
from .fusion_gru import ACTIONS, BOS_TOKEN, _valid_action_mask


@dataclass(frozen=True)
class PersistentHeadFusionGRUConfig:
    n: int
    n_heads: int
    target_q: int = 0
    embedding_dim: int = 4
    hidden_dim: int = 11
    head_embedding_scale: float = 0.2

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError("N must be a positive integer")
        if count_paths(self.n, int(self.target_q)) == 0:
            raise ValueError(
                f"target q={self.target_q} is unreachable for N={self.n}"
            )
        if (
            isinstance(self.n_heads, bool)
            or not isinstance(self.n_heads, int)
            or self.n_heads < 1
        ):
            raise ValueError("n_heads must be a positive integer")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if not 0.0 < self.head_embedding_scale <= 1.0:
            raise ValueError("head_embedding_scale must lie in (0, 1]")


def init_persistent_head_fusion_gru(
    key: jax.Array, config: PersistentHeadFusionGRUConfig
) -> dict[str, object]:
    """Initialize one shared conditional rule and its persistent head tokens."""

    core_key, token_key = jax.random.split(key)
    params = init_gru_core(
        core_key,
        embedding_sizes=(
            3,
            config.n + 1,
            max(config.n, 1),
            config.n + 1,
            config.n_heads,
        ),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    embeddings = list(params["embeddings"])
    embeddings[-1] = config.head_embedding_scale * jax.random.normal(
        token_key,
        (config.n_heads, config.embedding_dim),
        dtype=jnp.float64,
    )
    params["embeddings"] = tuple(embeddings)
    return params


def persistent_head_log_psi(
    params: dict[str, object],
    path: jax.Array,
    config: PersistentHeadFusionGRUConfig,
) -> jax.Array:
    """Return one normalized log amplitude per persistent state token."""

    path = jnp.asarray(path, dtype=jnp.int32)
    if path.shape != (config.n,):
        raise ValueError(f"expected path shape ({config.n},), got {path.shape}")
    head_ids = jnp.arange(config.n_heads, dtype=jnp.int32)
    hidden = jnp.broadcast_to(
        params["h0"], (config.n_heads, config.hidden_dim)
    )
    previous_token = jnp.full((config.n_heads,), BOS_TOKEN, dtype=jnp.int32)
    current_q = jnp.ones((config.n_heads,), dtype=jnp.int32)
    log_probability = jnp.zeros((config.n_heads,), dtype=jnp.float64)
    phase = jnp.zeros((config.n_heads,), dtype=jnp.float64)
    structurally_valid = path[0] == 1

    for decision in range(config.n - 1):
        remaining_after = config.n - decision - 2
        inputs = embed(
            params,
            (
                previous_token,
                current_q,
                jnp.full((config.n_heads,), decision, dtype=jnp.int32),
                jnp.full(
                    (config.n_heads,), remaining_after + 1, dtype=jnp.int32
                ),
                head_ids,
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phase_values = heads(params, hidden)
        mask = _valid_action_mask(current_q, remaining_after, config.target_q)
        logits = jnp.where(mask, logits, -1.0e30)
        action = path[decision + 1] - current_q
        action_index = jnp.where(action == 1, 1, 0)
        selected_valid = mask[jnp.arange(config.n_heads), action_index] & (
            (action == -1) | (action == 1)
        )
        structurally_valid = structurally_valid & jnp.all(selected_valid)
        log_probability += jax.nn.log_softmax(logits, axis=-1)[
            jnp.arange(config.n_heads), action_index
        ]
        phase += phase_values[jnp.arange(config.n_heads), action_index]
        current_q = jnp.full(
            (config.n_heads,), path[decision + 1], dtype=jnp.int32
        )
        previous_token = action_index

    structurally_valid = structurally_valid & jnp.all(
        current_q == config.target_q
    )
    values = 0.5 * log_probability + 1j * phase
    return jnp.where(structurally_valid, values, -jnp.inf + 0j)


def persistent_head_probabilities(
    params: dict[str, object],
    paths: jax.Array,
    config: PersistentHeadFusionGRUConfig,
) -> jax.Array:
    """Return ``[path, head]`` probabilities on an exhaustive batch."""

    values = jax.vmap(
        lambda path: persistent_head_log_psi(params, path, config)
    )(paths)
    return jnp.exp(2.0 * jnp.real(values))


__all__ = [
    "PersistentHeadFusionGRUConfig",
    "init_persistent_head_fusion_gru",
    "persistent_head_log_psi",
    "persistent_head_probabilities",
]
