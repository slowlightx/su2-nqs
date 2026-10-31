"""Pure-JAX positive normalized multi-size coupling-conditioned GRU baseline.

No Hamiltonian logic lives here. Parameters are shared at every n: continuous
position/size features replace size-specific embedding tables. An exact Z2
probability mixture gives psi(b)=sqrt((p_AR(b)+p_AR(1-b))/2). This baseline
targets stoquastic Ising cuts and does not implement the papers' ViT.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .basis import validate_size


@dataclass(frozen=True)
class GRUConfig:
    """Shared recurrent width; size is supplied by the configuration shape."""

    hidden_dim: int = 16
    size_scale: float = 64.0

    def __post_init__(self) -> None:
        if isinstance(self.hidden_dim, bool) or not isinstance(self.hidden_dim, int) or self.hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not math.isfinite(self.size_scale) or not self.size_scale > 1:
            raise ValueError("size_scale must exceed one")


def init_model(key: jax.Array, config: GRUConfig = GRUConfig()) -> dict[str, jax.Array]:
    """Initialize float64 parameters without any per-size parameter blocks."""
    keys = jax.random.split(key, 4)
    width = config.hidden_dim
    return {
        "w": jax.random.normal(keys[0], (11, 3 * width), dtype=jnp.float64) / jnp.sqrt(11.0),
        "u": jax.random.normal(keys[1], (width, 3 * width), dtype=jnp.float64) / jnp.sqrt(float(width)),
        "b": jnp.zeros(3 * width, dtype=jnp.float64),
        "context": 0.1 * jax.random.normal(keys[2], (6, width), dtype=jnp.float64),
        "head": 0.1 * jax.random.normal(keys[3], (width, 2), dtype=jnp.float64),
        "head_bias": jnp.zeros(2, dtype=jnp.float64),
    }


def _initial(params: dict, couplings: jax.Array, n: int, config: GRUConfig, batch: int) -> jax.Array:
    context = jnp.concatenate((couplings, jnp.asarray([jnp.log(n) / jnp.log(config.size_scale), 1 / n])))
    hidden = jnp.tanh(context @ params["context"])
    return jnp.broadcast_to(hidden, (batch, config.hidden_dim))


def _step(params: dict, hidden: jax.Array, previous: jax.Array, total: jax.Array,
          first: jax.Array, site: jax.Array, n: int, couplings: jax.Array,
          config: GRUConfig) -> tuple[jax.Array, jax.Array]:
    batch = len(previous)
    global_features = jnp.concatenate((jnp.asarray([site / n, (n - site) / n,
                                      jnp.log(n) / jnp.log(config.size_scale), 1 / n]), couplings))
    inputs = jnp.concatenate((jnp.stack((previous, total / n, first), axis=-1),
                              jnp.broadcast_to(global_features, (batch, 8))), axis=-1)
    xz, xr, xc = jnp.split(inputs @ params["w"] + params["b"], 3, axis=-1)
    hz, hr, hc = jnp.split(hidden @ params["u"], 3, axis=-1)
    update, reset = jax.nn.sigmoid(xz + hz), jax.nn.sigmoid(xr + hr)
    hidden = (1 - update) * hidden + update * jnp.tanh(xc + reset * hc)
    return hidden, hidden @ params["head"] + params["head_bias"]


def _log_probability(params: dict, bits: jax.Array, couplings: jax.Array,
                     config: GRUConfig) -> jax.Array:
    batch, n = bits.shape
    initial = (_initial(params, couplings, n, config, batch),
               jnp.zeros(batch), jnp.zeros(batch), jnp.zeros(batch))

    def body(carry, item):
        hidden, previous, total, first = carry
        site, bit = item
        hidden, logits = _step(params, hidden, previous, total, first, site, n, couplings, config)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), bit[:, None], axis=-1)[:, 0]
        spin = (2 * bit - 1).astype(jnp.float64)
        return (hidden, spin, total + spin, jnp.where(site == 0, spin, first)), logp

    _, logps = jax.lax.scan(body, initial, (jnp.arange(n), bits.T))
    return logps.sum(axis=0)


def log_psi(params: dict, bits: jax.Array, couplings: jax.Array,
            config: GRUConfig = GRUConfig()) -> jax.Array:
    """Return real log psi = 0.5 log p; phase is exactly zero.

    Accept one configuration or a batch. Invalid bit labels get -inf without
    repairing them. Couplings must have shape (4,) in (j1,j2,jx,h) order.
    """
    bits = jnp.asarray(bits)
    couplings = jnp.asarray(couplings, dtype=jnp.float64)
    single = bits.ndim == 1
    if single:
        bits = bits[None, :]
    if bits.ndim != 2 or couplings.shape != (4,):
        raise ValueError("expected bits shape (n,) or (batch,n), couplings shape (4,)")
    validate_size(bits.shape[1])
    valid = jnp.all((bits == 0) | (bits == 1), axis=-1) & jnp.all(jnp.isfinite(couplings))
    safe_bits = jnp.where(bits == 1, 1, 0).astype(jnp.int32)
    logp = jnp.logaddexp(_log_probability(params, safe_bits, couplings, config),
                        _log_probability(params, 1 - safe_bits, couplings, config)) - jnp.log(2.0)
    result = jnp.where(valid, 0.5 * logp, -jnp.inf)
    return result[0] if single else result


def sample(params: dict, key: jax.Array, n_samples: int, n: int, couplings: jax.Array,
           config: GRUConfig = GRUConfig()) -> jax.Array:
    """Draw independent Born configurations, then sample the Z2 mixture coin."""
    validate_size(n)
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be positive")
    couplings = jnp.asarray(couplings, dtype=jnp.float64)
    if couplings.shape != (4,):
        raise ValueError("couplings must have shape (4,)")
    key, coin_key = jax.random.split(key)
    initial = (_initial(params, couplings, n, config, n_samples), jnp.zeros(n_samples),
               jnp.zeros(n_samples), jnp.zeros(n_samples), key)

    def body(carry, site):
        hidden, previous, total, first, key = carry
        key, subkey = jax.random.split(key)
        hidden, logits = _step(params, hidden, previous, total, first, site, n, couplings, config)
        bit = jax.random.categorical(subkey, logits).astype(jnp.int32)
        spin = (2 * bit - 1).astype(jnp.float64)
        return (hidden, spin, total + spin, jnp.where(site == 0, spin, first), key), bit

    _, bits = jax.lax.scan(body, initial, jnp.arange(n))
    flip = jax.random.bernoulli(coin_key, shape=(n_samples, 1))
    return jnp.where(flip, 1 - bits.T, bits.T)


def coupling_scores(params: dict, bits: jax.Array, couplings: jax.Array,
                    config: GRUConfig = GRUConfig()) -> jax.Array:
    """Differentiate log psi with respect to inputs at FIXED trained parameters."""
    # Four inputs and potentially thousands of outputs: forward AD avoids an
    # output-by-output reverse Jacobian and its quadratic sample memory cost.
    return jax.jacfwd(lambda g: log_psi(params, bits, g, config))(couplings)
