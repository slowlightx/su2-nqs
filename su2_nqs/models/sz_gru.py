"""Comparable normalized autoregressive GRU in a fixed-Sz basis."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..su2.sz_basis import enumerate_sz_configs
from ._core import embed, gru_step, heads, init_gru_core

BOS_TOKEN = 2


@dataclass(frozen=True)
class SzGRUConfig:
    n: int
    n_up: int | None = None
    embedding_dim: int = 16
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if self.n_up is None:
            object.__setattr__(self, "n_up", self.n // 2)
        if not 0 <= self.n_up <= self.n:
            raise ValueError("n_up must lie between zero and N")


def init_sz_gru(key: jax.Array, config: SzGRUConfig) -> dict[str, object]:
    # previous bit (down, up, BOS), number up so far, site, sites remaining
    return init_gru_core(
        key,
        embedding_sizes=(3, config.n + 1, config.n, config.n + 1),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )


def _valid_bit_mask(count_up: jax.Array, remaining_after: int, target_up: int) -> jax.Array:
    next_counts = count_up[..., None] + jnp.asarray([0, 1], dtype=jnp.int32)
    return (next_counts <= target_up) & (next_counts + remaining_after >= target_up)


def sz_log_psi(
    params: dict[str, object],
    configuration: jax.Array,
    config: SzGRUConfig,
) -> jax.Array:
    """Normalized complex log-amplitude in the fixed-magnetization sector."""

    configuration = jnp.asarray(configuration, dtype=jnp.int32)
    if configuration.shape != (config.n,):
        raise ValueError(f"expected configuration shape ({config.n},), got {configuration.shape}")
    hidden = params["h0"]
    previous = jnp.asarray(BOS_TOKEN, dtype=jnp.int32)
    count_up = jnp.asarray(0, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid = jnp.asarray(True)
    for site in range(config.n):
        remaining_after = config.n - site - 1
        inputs = embed(params, (previous, count_up, site, remaining_after + 1))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        mask = _valid_bit_mask(count_up, remaining_after, config.n_up)
        logits = jnp.where(mask, logits, -1.0e30)
        bit = configuration[site]
        safe_bit = jnp.where(bit == 1, 1, 0)
        valid = valid & ((bit == 0) | (bit == 1)) & mask[safe_bit]
        log_probability += jax.nn.log_softmax(logits)[safe_bit]
        phase += phases[safe_bit]
        count_up += safe_bit
        previous = safe_bit
    valid = valid & (count_up == config.n_up)
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid, value, -jnp.inf + 0j)


def sample_sz(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: SzGRUConfig,
) -> jax.Array:
    """Draw independent configurations with exactly ``n_up`` spins up."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), BOS_TOKEN, dtype=jnp.int32)
    count_up = jnp.zeros((n_samples,), dtype=jnp.int32)
    bits = []
    for site in range(config.n):
        key, subkey = jax.random.split(key)
        remaining_after = config.n - site - 1
        sites = jnp.full((n_samples,), site, dtype=jnp.int32)
        remaining = jnp.full((n_samples,), remaining_after + 1, dtype=jnp.int32)
        inputs = embed(params, (previous, count_up, sites, remaining))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        mask = _valid_bit_mask(count_up, remaining_after, config.n_up)
        logits = jnp.where(mask, logits, -1.0e30)
        bit = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        bits.append(bit)
        count_up += bit
        previous = bit
    return jnp.stack(bits, axis=1)


@dataclass(frozen=True)
class SzGRU:
    """Object-oriented wrapper matching :class:`FusionGRU`."""

    config: SzGRUConfig

    def init(self, key: jax.Array) -> dict[str, object]:
        return init_sz_gru(key, self.config)

    def log_psi(self, params: dict[str, object], configuration: jax.Array) -> jax.Array:
        return sz_log_psi(params, configuration, self.config)

    def sample(self, params: dict[str, object], key: jax.Array, n_samples: int) -> jax.Array:
        return sample_sz(params, key, n_samples, self.config)
