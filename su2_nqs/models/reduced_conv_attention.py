"""Local convolution and global attention on COMPLETE reduced configurations.

This exact-small coefficient generator has no magnetic or physical-spin input.
For a validated fixed singlet tensor decoder D, D@f_theta is a singlet for
every theta and every parameter tangent. It is NOT yet a many-cell TN
contraction or an autoregressive sampler: enumerating all configurations is
exponential, and this boundary must not be relabelled as N36 production.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import jax
import jax.numpy as jnp
import numpy as np

from .su2_schur_config_vit import _layer_norm, _matrix


@dataclass(frozen=True)
class ReducedConvAttentionConfig:
    """Reduced tokens on a periodic rectangular cell grid; q=2j throughout."""
    rows: int = 2
    columns: int = 2
    maximum_q: int = 8
    multiplicity_count: int = 3
    embedding_dim: int = 16
    depth: int = 2
    initialization_scale: float = .2

    def __post_init__(self) -> None:
        for name in ('rows', 'columns', 'multiplicity_count', 'embedding_dim', 'depth'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Integral) or v < 1:
                raise ValueError(name+' must be a positive integer')
        if isinstance(self.maximum_q, bool) or not isinstance(self.maximum_q, Integral) or self.maximum_q < 0:
            raise ValueError('maximum_q must be a nonnegative doubled integer')
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0:
            raise ValueError('positive finite initialization scale required')


def init_reduced_conv_attention(key: jax.Array, config: ReducedConvAttentionConfig) -> dict:
    """All convolution, attention, embeddings and complex coefficients train."""
    keys = iter(jax.random.split(key, 4+8*config.depth))
    d, scale = config.embedding_dim, config.initialization_scale
    layers = []
    for _ in range(config.depth):
        layers.append({
            'conv': scale*jax.random.normal(next(keys), (3, d, d), dtype=jnp.float64)/np.sqrt(3*d),
            **{k: _matrix(next(keys), d, d, scale) for k in ('query', 'key', 'value', 'output')},
            'distance_bias': scale*jax.random.normal(next(keys), (), dtype=jnp.float64),
            'mlp_in': _matrix(next(keys), d, 2*d, scale),
            'mlp_out': _matrix(next(keys), 2*d, d, scale),
        })
    return {
        'q_embedding': scale*jax.random.normal(next(keys), (config.maximum_q+1, d), dtype=jnp.float64),
        'a_embedding': scale*jax.random.normal(next(keys), (config.multiplicity_count, d), dtype=jnp.float64),
        'internal_q_embedding': scale*jax.random.normal(next(keys), (config.maximum_q+1, d), dtype=jnp.float64),
        'readout': _matrix(next(keys), d, 2, scale), 'layers': tuple(layers),
    }


def reduced_conv_attention_coefficients(
    params: dict, labels: jax.Array, internal_q: jax.Array,
    config: ReducedConvAttentionConfig,
) -> jax.Array:
    """Complex f[configuration] from local (q,a) and inter-cell doubled q.

    ``labels[K,rows*columns,2]`` and ``internal_q[K]`` are finite numeric
    reduced configurations supplied by an independently validated basis.
    Invalid dynamic labels yield NaN, not silently repaired states. No physical
    leg enters a nonlinearity. Local axis kernels are directional; exact A1
    must be supplied by the FIXED decoder, not claimed from positional biases.
    """
    z, inter = jnp.asarray(labels), jnp.asarray(internal_q)
    n, d = config.rows*config.columns, config.embedding_dim
    if z.ndim != 3 or z.shape[1:] != (n, 2) or inter.shape != (z.shape[0],):
        raise ValueError('expected [K,cells,2] reduced labels and [K] internal q')
    if not jnp.issubdtype(z.dtype, jnp.integer) or not jnp.issubdtype(inter.dtype, jnp.integer):
        raise TypeError('q,a labels must be integers; q is doubled')
    valid = (jnp.all((z[..., 0] >= 0) & (z[..., 0] <= config.maximum_q), axis=1)
             & jnp.all((z[..., 1] >= 0) & (z[..., 1] < config.multiplicity_count), axis=1)
             & (inter >= 0) & (inter <= config.maximum_q))
    q = jnp.clip(z[..., 0], 0, config.maximum_q)
    a = jnp.clip(z[..., 1], 0, config.multiplicity_count-1)
    t = params['q_embedding'][q]+params['a_embedding'][a]
    t += params['internal_q_embedding'][jnp.clip(inter, 0, config.maximum_q)][:, None]
    rr, cc = np.indices((config.rows, config.columns))
    coords = np.stack((rr.ravel(), cc.ravel()), axis=1)
    delta = abs(coords[:, None]-coords[None, :])
    delta = np.minimum(delta, np.asarray((config.rows, config.columns))-delta)
    distance2 = jnp.asarray(np.sum(delta**2, axis=-1), dtype=jnp.float64)
    for layer in params['layers']:
        grid = _layer_norm(t).reshape(-1, config.rows, config.columns, d)
        local = (grid@layer['conv'][0]
                 + .5*(jnp.roll(grid, 1, axis=1)+jnp.roll(grid, -1, axis=1))@layer['conv'][1]
                 + .5*(jnp.roll(grid, 1, axis=2)+jnp.roll(grid, -1, axis=2))@layer['conv'][2])
        t += jax.nn.gelu(local.reshape(-1, n, d))
        norm = _layer_norm(t)
        query, key, value = (norm@layer[k] for k in ('query', 'key', 'value'))
        logits = jnp.einsum('kid,kjd->kij', query, key)/np.sqrt(d)
        logits += layer['distance_bias']*distance2[None]
        t += jnp.einsum('kij,kjd->kid', jax.nn.softmax(logits, axis=-1), value)@layer['output']
        t += jax.nn.gelu(_layer_norm(t)@layer['mlp_in'])@layer['mlp_out']
    output = _layer_norm(t).mean(axis=1)@params['readout']
    return jnp.where(valid, output[:, 0]+1j*output[:, 1], jnp.nan+1j*jnp.nan)


__all__ = ['ReducedConvAttentionConfig', 'init_reduced_conv_attention', 'reduced_conv_attention_coefficients']
