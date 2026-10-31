"""Pure-JAX size-feature ablations with identical nominal GRU shapes.

Policies: legacy; no_explicit_size (remove log L and 1/L but retain
length-normalized geometry); no_length (all features independent of total L).
The last policy enforces consistent prefix marginals across lengths. For finite
PBC ground states this is an additional restriction, not a free simplification.
Use cold initialization and this module's API; legacy training is incompatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fnqs.basis import validate_size
from fnqs.models import GRUConfig, init_model as legacy_init

SizePolicy = Literal["legacy", "no_explicit_size", "no_length"]


@dataclass(frozen=True)
class SizeFeatureConfig:
    """Keep geometry_scale constant across sizes; it is not a learned weight."""
    hidden_dim: int = 16
    geometry_scale: float = 64.
    size_features: SizePolicy = "no_explicit_size"

    def __post_init__(self) -> None:
        GRUConfig(self.hidden_dim, self.geometry_scale)
        if self.size_features not in ("legacy", "no_explicit_size", "no_length"):
            raise ValueError("unknown size feature policy")


def init_model(key: jax.Array, config: SizeFeatureConfig = SizeFeatureConfig()) -> dict:
    """Initialize the same nominal parameter arrays for every policy."""
    return legacy_init(key, GRUConfig(config.hidden_dim, config.geometry_scale))


def parameter_counts(config: SizeFeatureConfig) -> dict[str, int]:
    """Count nominal weights and structurally active weights after masking."""
    w = config.hidden_dim
    nominal = 3*w*w+44*w+2
    inactive = {"legacy": 0, "no_explicit_size": 8*w, "no_length": 11*w}[config.size_features]
    return {"nominal": nominal, "active": nominal-inactive, "inactive": inactive}


def _initial(params: dict, couplings: jax.Array, n: int,
             config: SizeFeatureConfig, batch: int) -> jax.Array:
    sizes = (jnp.asarray([jnp.log(n)/jnp.log(config.geometry_scale), 1/n])
             if config.size_features == "legacy" else jnp.zeros(2))
    hidden = jnp.tanh(jnp.concatenate((couplings, sizes)) @ params["context"])
    return jnp.broadcast_to(hidden, (batch, config.hidden_dim))


def _step(params: dict, hidden: jax.Array, previous: jax.Array, total: jax.Array,
          first: jax.Array, site: jax.Array, n: int, couplings: jax.Array,
          config: SizeFeatureConfig) -> tuple[jax.Array, jax.Array]:
    if config.size_features == "no_length":
        geometry = jnp.asarray([site/config.geometry_scale, 0., 0., 0.])
        total_feature = total/config.geometry_scale
    else:
        sizes = (jnp.asarray([jnp.log(n)/jnp.log(config.geometry_scale), 1/n])
                 if config.size_features == "legacy" else jnp.zeros(2))
        geometry = jnp.concatenate((jnp.asarray([site/n, (n-site)/n]), sizes))
        total_feature = total/n
    globals_ = jnp.concatenate((geometry, couplings))
    inputs = jnp.concatenate((jnp.stack((previous, total_feature, first), axis=-1),
                              jnp.broadcast_to(globals_, (len(previous), 8))), axis=-1)
    xz, xr, xc = jnp.split(inputs@params["w"]+params["b"], 3, axis=-1)
    hz, hr, hc = jnp.split(hidden@params["u"], 3, axis=-1)
    update, reset = jax.nn.sigmoid(xz+hz), jax.nn.sigmoid(xr+hr)
    hidden = (1-update)*hidden+update*jnp.tanh(xc+reset*hc)
    return hidden, hidden@params["head"]+params["head_bias"]


def _log_probability(params: dict, bits: jax.Array, couplings: jax.Array,
                     config: SizeFeatureConfig) -> jax.Array:
    batch, n = bits.shape
    initial = (_initial(params, couplings, n, config, batch),
               jnp.zeros(batch), jnp.zeros(batch), jnp.zeros(batch))

    def body(carry, item):
        hidden, previous, total, first = carry
        site, bit = item
        hidden, logits = _step(params, hidden, previous, total, first, site, n, couplings, config)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), bit[:, None], axis=-1)[:, 0]
        spin = (2*bit-1).astype(jnp.float64)
        return (hidden, spin, total+spin, jnp.where(site == 0, spin, first)), logp

    _, logps = jax.lax.scan(body, initial, (jnp.arange(n), bits.T))
    return logps.sum(axis=0)


def log_psi(params: dict, bits: jax.Array, couplings: jax.Array,
            config: SizeFeatureConfig = SizeFeatureConfig()) -> jax.Array:
    """Normalized positive Z2 mixture; coupling order is (j1,j2,jx,h)."""
    if not isinstance(config, SizeFeatureConfig):
        raise TypeError("use SizeFeatureConfig and this variant's API")
    bits, couplings = jnp.asarray(bits), jnp.asarray(couplings, dtype=jnp.float64)
    single = bits.ndim == 1
    if single:
        bits = bits[None, :]
    if bits.ndim != 2 or couplings.shape != (4,):
        raise ValueError("expected (n,) or (batch,n) bits and four couplings")
    validate_size(bits.shape[1])
    valid = jnp.all((bits == 0) | (bits == 1), axis=1) & jnp.all(jnp.isfinite(couplings))
    safe = jnp.where(bits == 1, 1, 0).astype(jnp.int32)
    logp = jnp.logaddexp(_log_probability(params, safe, couplings, config),
                        _log_probability(params, 1-safe, couplings, config))-jnp.log(2.)
    result = jnp.where(valid, .5*logp, -jnp.inf)
    return result[0] if single else result


def sample(params: dict, key: jax.Array, n_samples: int, n: int, couplings: jax.Array,
           config: SizeFeatureConfig = SizeFeatureConfig()) -> jax.Array:
    """Independent Born sampling with the selected size-feature policy."""
    if not isinstance(config, SizeFeatureConfig):
        raise TypeError("use SizeFeatureConfig and this variant's API")
    validate_size(n)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be positive")
    g = jnp.asarray(couplings, dtype=jnp.float64)
    if g.shape != (4,):
        raise ValueError("expected four couplings")
    key, coin_key = jax.random.split(key)
    initial = (_initial(params, g, n, config, n_samples), jnp.zeros(n_samples),
               jnp.zeros(n_samples), jnp.zeros(n_samples), key)

    def body(carry, site):
        hidden, previous, total, first, key = carry
        key, subkey = jax.random.split(key)
        hidden, logits = _step(params, hidden, previous, total, first, site, n, g, config)
        bit = jax.random.categorical(subkey, logits).astype(jnp.int32)
        spin = (2*bit-1).astype(jnp.float64)
        return (hidden, spin, total+spin, jnp.where(site == 0, spin, first), key), bit

    _, bits = jax.lax.scan(body, initial, jnp.arange(n))
    flip = jax.random.bernoulli(coin_key, shape=(n_samples, 1))
    return jnp.where(flip, 1-bits.T, bits.T)
