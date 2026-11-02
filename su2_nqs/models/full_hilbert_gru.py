"""Normalized autoregressive GRU on the full spin-1/2 Hilbert space.

This model is the no-hard-symmetry control for the fusion and fixed-``Sz``
GRUs.  It uses the same recurrent core and the same four deterministic prefix
features as :class:`~su2_nqs.models.sz_gru.SzGRU`, but it never applies a
magnetization or total-spin reachability mask.  Consequently every binary
configuration is in the support and normalization is over all ``2**N``
states.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._core import embed, gru_step, heads, init_gru_core

BOS_TOKEN = 2


@dataclass(frozen=True)
class FullHilbertGRUConfig:
    """Configuration for an unrestricted binary spin autoregressive GRU."""

    n: int
    embedding_dim: int = 16
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")


def init_full_hilbert_gru(
    key: jax.Array,
    config: FullHilbertGRUConfig,
) -> dict[str, object]:
    """Initialize the parameter-matched unrestricted recurrent core."""

    # Previous bit, prefix up-spin count, site, and sites remaining.  Prefix
    # count is an ordinary deterministic sequence feature here: it never masks
    # an action or restricts the final magnetization.
    return init_gru_core(
        key,
        embedding_sizes=(3, config.n + 1, config.n, config.n + 1),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )


def full_hilbert_log_psi(
    params: dict[str, object],
    configuration: jax.Array,
    config: FullHilbertGRUConfig,
) -> jax.Array:
    """Return ``0.5 log p + i phase`` for any binary configuration."""

    configuration = jnp.asarray(configuration, dtype=jnp.int32)
    if configuration.shape != (config.n,):
        raise ValueError(
            f"expected configuration shape ({config.n},), got {configuration.shape}"
        )
    hidden = params["h0"]
    previous = jnp.asarray(BOS_TOKEN, dtype=jnp.int32)
    count_up = jnp.asarray(0, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid = jnp.asarray(True)
    for site in range(config.n):
        remaining = config.n - site
        inputs = embed(params, (previous, count_up, site, remaining))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        bit = configuration[site]
        safe_bit = jnp.where(bit == 1, 1, 0)
        valid = valid & ((bit == 0) | (bit == 1))
        log_probability += jax.nn.log_softmax(logits)[safe_bit]
        phase += phases[safe_bit]
        count_up += safe_bit
        previous = safe_bit
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid, value, -jnp.inf + 0j)


def sample_full_hilbert(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: FullHilbertGRUConfig,
) -> jax.Array:
    """Draw independent samples without a global spin-sector constraint."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), BOS_TOKEN, dtype=jnp.int32)
    count_up = jnp.zeros((n_samples,), dtype=jnp.int32)
    bits = []
    for site in range(config.n):
        key, subkey = jax.random.split(key)
        sites = jnp.full((n_samples,), site, dtype=jnp.int32)
        remaining = jnp.full((n_samples,), config.n - site, dtype=jnp.int32)
        inputs = embed(params, (previous, count_up, sites, remaining))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        bit = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        bits.append(bit)
        count_up += bit
        previous = bit
    return jnp.stack(bits, axis=1)


@dataclass(frozen=True)
class FullHilbertGRU:
    """Object-oriented wrapper for the unrestricted binary spin GRU."""

    config: FullHilbertGRUConfig

    def init(self, key: jax.Array) -> dict[str, object]:
        return init_full_hilbert_gru(key, self.config)

    def log_psi(
        self,
        params: dict[str, object],
        configuration: jax.Array,
    ) -> jax.Array:
        return full_hilbert_log_psi(params, configuration, self.config)

    def sample(
        self,
        params: dict[str, object],
        key: jax.Array,
        n_samples: int,
    ) -> jax.Array:
        return sample_full_hilbert(params, key, n_samples, self.config)
