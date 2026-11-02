"""Standard full-Hilbert autoregressive GRU without conserved-charge inputs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._core import embed, gru_step, heads, init_gru_core

BOS_TOKEN = 2


@dataclass(frozen=True)
class TokenPositionGRUConfig:
    """A binary GRU conditioned only on the previous token and site index.

    Unlike the fixed-``Sz`` and fusion models, this control receives no prefix
    charge, total-spin label, reachability mask, or target sector.  Its support
    is the full ``2**N`` computational basis.  Benchmark configurations may
    widen its hidden state to approximately match a symmetry-aware raw scalar
    parameter budget without introducing redundant deterministic channels.
    """

    n: int
    embedding_dim: int = 16
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")


def init_token_position_gru(
    key: jax.Array,
    config: TokenPositionGRUConfig,
) -> dict[str, object]:
    """Initialize a token-plus-position recurrent model."""

    return init_gru_core(
        key,
        embedding_sizes=(3, config.n),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )


def token_position_log_psi(
    params: dict[str, object],
    configuration: jax.Array,
    config: TokenPositionGRUConfig,
) -> jax.Array:
    """Return an exactly normalized log-amplitude on the full Hilbert space."""

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
        inputs = embed(params, (previous, site))
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


def sample_token_position(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: TokenPositionGRUConfig,
) -> jax.Array:
    """Draw independent samples without a global symmetry constraint."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), BOS_TOKEN, dtype=jnp.int32)
    bits = []
    for site in range(config.n):
        positions = jnp.full((n_samples,), site, dtype=jnp.int32)
        inputs = embed(params, (previous, positions))
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        key, subkey = jax.random.split(key)
        bit = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        bits.append(bit)
        previous = bit
    return jnp.stack(bits, axis=1)


@dataclass(frozen=True)
class TokenPositionGRU:
    """Object-oriented wrapper for :class:`TokenPositionGRUConfig`."""

    config: TokenPositionGRUConfig

    def init(self, key: jax.Array) -> dict[str, object]:
        return init_token_position_gru(key, self.config)

    def log_psi(
        self,
        params: dict[str, object],
        configuration: jax.Array,
    ) -> jax.Array:
        return token_position_log_psi(params, configuration, self.config)

    def sample(
        self,
        params: dict[str, object],
        key: jax.Array,
        n_samples: int,
    ) -> jax.Array:
        return sample_token_position(params, key, n_samples, self.config)
