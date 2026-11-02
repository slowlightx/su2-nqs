"""Shared-core normalized autoregressive states on SU(2) fusion paths.

Every head defines its own exactly normalized conditional distribution and
phase while reusing the embeddings and GRU transition.  The object represented
by several heads is their *span*: downstream multistate objectives must not
attach physical meaning to the raw head labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..su2.fusion_path import count_paths
from ._core import embed, gru_step, init_gru_core
from .fusion_gru import ACTIONS, BOS_TOKEN, _valid_action_mask


@dataclass(frozen=True)
class MultiHeadFusionGRUConfig:
    """Configuration for a shared recurrent core with several wavefunctions."""

    n: int
    n_heads: int
    target_q: int = 0
    embedding_dim: int = 16
    hidden_dim: int = 32

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


def init_multihead_fusion_gru(
    key: jax.Array, config: MultiHeadFusionGRUConfig
) -> dict[str, object]:
    """Initialize shared embeddings/GRU and independent normalized heads."""

    core_key, probability_key, phase_key, hidden_key = jax.random.split(key, 4)
    params = init_gru_core(
        core_key,
        embedding_sizes=(3, config.n + 1, max(config.n, 1), config.n + 1),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    scale = (2.0 / (config.hidden_dim + 2)) ** 0.5
    params["probability"] = {
        "w": scale
        * jax.random.normal(
            probability_key,
            (config.n_heads, config.hidden_dim, 2),
            dtype=jnp.float64,
        ),
        "b": jnp.zeros((config.n_heads, 2), dtype=jnp.float64),
    }
    params["phase"] = {
        "w": scale
        * jax.random.normal(
            phase_key,
            (config.n_heads, config.hidden_dim, 2),
            dtype=jnp.float64,
        ),
        "b": jnp.zeros((config.n_heads, 2), dtype=jnp.float64),
    }
    # A small head-dependent boundary condition breaks the otherwise exact
    # permutation symmetry among heads without changing normalization.
    params["h0"] = 0.05 * jax.random.normal(
        hidden_key, (config.n_heads, config.hidden_dim), dtype=jnp.float64
    )
    return params


def _multihead_outputs(
    params: dict[str, object], hidden: jax.Array
) -> tuple[jax.Array, jax.Array]:
    probability = params["probability"]
    phase = params["phase"]
    logits = jnp.einsum("rh,rha->ra", hidden, probability["w"]) + probability["b"]
    phases = jnp.pi * jnp.tanh(
        jnp.einsum("rh,rha->ra", hidden, phase["w"]) + phase["b"]
    )
    return logits, phases


def multihead_log_psi(
    params: dict[str, object],
    path: jax.Array,
    config: MultiHeadFusionGRUConfig,
) -> jax.Array:
    """Return one normalized log amplitude per head for a fusion path."""

    path = jnp.asarray(path, dtype=jnp.int32)
    if path.shape != (config.n,):
        raise ValueError(f"expected path shape ({config.n},), got {path.shape}")
    hidden = params["h0"]
    previous_token = jnp.asarray(BOS_TOKEN, dtype=jnp.int32)
    current_q = jnp.asarray(1, dtype=jnp.int32)
    log_probability = jnp.zeros((config.n_heads,), dtype=jnp.float64)
    phase = jnp.zeros((config.n_heads,), dtype=jnp.float64)
    structurally_valid = path[0] == 1

    for decision in range(config.n - 1):
        remaining_after = config.n - decision - 2
        inputs = embed(
            params,
            (previous_token, current_q, decision, remaining_after + 1),
        )
        inputs = jnp.broadcast_to(inputs, (config.n_heads, inputs.shape[-1]))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phase_values = _multihead_outputs(params, hidden)
        mask = _valid_action_mask(current_q, remaining_after, config.target_q)
        masked_logits = jnp.where(mask[None, :], logits, -1.0e30)
        action = path[decision + 1] - current_q
        action_index = jnp.where(action == 1, 1, 0)
        selected_valid = mask[action_index] & ((action == -1) | (action == 1))
        structurally_valid = structurally_valid & selected_valid
        log_probability += jax.nn.log_softmax(masked_logits, axis=-1)[:, action_index]
        phase += phase_values[:, action_index]
        current_q = path[decision + 1]
        previous_token = action_index

    structurally_valid = structurally_valid & (current_q == config.target_q)
    values = 0.5 * log_probability + 1j * phase
    return jnp.where(structurally_valid, values, -jnp.inf + 0j)


def multihead_path_probabilities(
    params: dict[str, object],
    paths: jax.Array,
    config: MultiHeadFusionGRUConfig,
) -> jax.Array:
    """Return ``[path, head]`` probabilities on a batch of paths."""

    values = jax.vmap(lambda path: multihead_log_psi(params, path, config))(paths)
    return jnp.exp(2.0 * jnp.real(values))


def sample_multihead(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    head: int,
    config: MultiHeadFusionGRUConfig,
) -> jax.Array:
    """Draw independent fusion paths from one selected head."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head < config.n_heads:
        raise ValueError("head index is out of range")
    hidden = jnp.broadcast_to(params["h0"][head], (n_samples, config.hidden_dim))
    previous_token = jnp.full((n_samples,), BOS_TOKEN, dtype=jnp.int32)
    current_q = jnp.ones((n_samples,), dtype=jnp.int32)
    labels = [current_q]
    probability = params["probability"]
    for decision in range(config.n - 1):
        key, subkey = jax.random.split(key)
        remaining_after = config.n - decision - 2
        step = jnp.full((n_samples,), decision, dtype=jnp.int32)
        remaining = jnp.full((n_samples,), remaining_after + 1, dtype=jnp.int32)
        inputs = embed(params, (previous_token, current_q, step, remaining))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits = hidden @ probability["w"][head] + probability["b"][head]
        mask = _valid_action_mask(current_q, remaining_after, config.target_q)
        logits = jnp.where(mask, logits, -1.0e30)
        action_index = jax.random.categorical(subkey, logits, axis=-1)
        current_q = current_q + ACTIONS[action_index]
        previous_token = action_index
        labels.append(current_q)
    return jnp.stack(labels, axis=1)


__all__ = [
    "MultiHeadFusionGRUConfig",
    "init_multihead_fusion_gru",
    "multihead_log_psi",
    "multihead_path_probabilities",
    "sample_multihead",
]
