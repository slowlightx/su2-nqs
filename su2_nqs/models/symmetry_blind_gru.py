"""Budget-matched autoregressive GRU without a spin-symmetry input or mask."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._core import embed, gru_step, heads, init_gru_core

BOS_TOKEN = 2


@dataclass(frozen=True)
class SymmetryBlindGRUConfig:
    """Configuration for the full-Hilbert, symmetry-blind control.

    The fourth embedding is a second absolute-position channel.  It keeps the
    scalar parameter budget identical to the fusion and fixed-``Sz`` GRUs at
    equal ``embedding_dim`` and ``hidden_dim`` without exposing a prefix
    magnetization or total-spin label.
    """

    n: int
    embedding_dim: int = 16
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")


def init_symmetry_blind_gru(
    key: jax.Array,
    config: SymmetryBlindGRUConfig,
) -> dict[str, object]:
    """Initialize a recurrent core with the common benchmark budget."""

    return init_gru_core(
        key,
        embedding_sizes=(3, config.n + 1, config.n, config.n + 1),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )


def symmetry_blind_log_psi(
    params: dict[str, object],
    configuration: jax.Array,
    config: SymmetryBlindGRUConfig,
) -> jax.Array:
    """Return a normalized log-amplitude on the full binary Hilbert space."""

    configuration = jnp.asarray(configuration, dtype=jnp.int32)
    if configuration.shape != (config.n,):
        raise ValueError(
            f"expected configuration shape ({config.n},), got {configuration.shape}"
        )
    hidden = params["h0"]
    previous = jnp.asarray(BOS_TOKEN, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid = jnp.asarray(True)
    for site in range(config.n):
        remaining = config.n - site
        # No spin quantum number is supplied.  The repeated positional channel
        # is deliberately distinct in parameter space and only budget-matches
        # the symmetry-aware controls.
        inputs = embed(params, (previous, site + 1, site, remaining))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        bit = configuration[site]
        safe_bit = jnp.where(bit == 1, 1, 0)
        valid = valid & ((bit == 0) | (bit == 1))
        log_probability += jax.nn.log_softmax(logits)[safe_bit]
        phase += phases[safe_bit]
        previous = safe_bit
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid, value, -jnp.inf + 0j)


def sample_symmetry_blind(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: SymmetryBlindGRUConfig,
) -> jax.Array:
    """Draw independent samples from all magnetization sectors."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), BOS_TOKEN, dtype=jnp.int32)
    bits = []
    for site in range(config.n):
        key, subkey = jax.random.split(key)
        sites = jnp.full((n_samples,), site, dtype=jnp.int32)
        position_copy = jnp.full((n_samples,), site + 1, dtype=jnp.int32)
        remaining = jnp.full((n_samples,), config.n - site, dtype=jnp.int32)
        inputs = embed(params, (previous, position_copy, sites, remaining))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        bit = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        bits.append(bit)
        previous = bit
    return jnp.stack(bits, axis=1)


@dataclass(frozen=True)
class SymmetryBlindGRU:
    """Object-oriented wrapper for the symmetry-blind full-Hilbert GRU."""

    config: SymmetryBlindGRUConfig

    def init(self, key: jax.Array) -> dict[str, object]:
        return init_symmetry_blind_gru(key, self.config)

    def log_psi(
        self,
        params: dict[str, object],
        configuration: jax.Array,
    ) -> jax.Array:
        return symmetry_blind_log_psi(params, configuration, self.config)

    def sample(
        self,
        params: dict[str, object],
        key: jax.Array,
        n_samples: int,
    ) -> jax.Array:
        return sample_symmetry_blind(params, key, n_samples, self.config)
