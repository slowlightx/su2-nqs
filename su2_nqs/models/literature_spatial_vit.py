"""Author-aligned factored Spatial-ViT, without modifying legacy checkpoints.

Independent pure-JAX implementation of the NetKet ViT tutorial architecture:
Xavier-uniform Dense layers (including biases), pre-LN residual encoders,
4d feed-forward layers, LN after token summation, and separate real/imaginary
output LNs. The optional normalized distance envelope follows arXiv:2602.02665.
This is fixed doubled magnetization, NOT an exact-SU(2) state. Projection,
Hamiltonians and training live outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig, _patch_values, _complex_log_cosh,
)


@dataclass(frozen=True)
class LiteratureSpatialViTConfig(FixedSzSpatialViTConfig):
    """Physical 120-degree triangular coordinates; labels remain doubled ints.

    ``pooling='tokenwise_control'`` changes ONLY pooling-LN order, retaining
    the same parameters and both output LNs. It is not the archived model.
    ``use_distance_envelope=False`` gives raw relative factors, as in the
    tutorial, rather than factors divided by the number of tokens.
    """
    mlp_ratio: int = 4
    initialization_scale: float = 1.0
    initial_distance_decay: float = 3.0
    pooling: str = 'authors'
    layer_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pooling not in {'authors', 'tokenwise_control'}:
            raise ValueError('unknown pooling control')
        if not np.isfinite(self.layer_norm_epsilon) or self.layer_norm_epsilon <= 0:
            raise ValueError('positive finite LayerNorm epsilon required')


def init_literature_spatial_vit(key: jax.Array,
                              config: LiteratureSpatialViTConfig) -> dict:
    """Xavier-uniform weights, zero Dense biases, unit LN scales.

    RNG ordering is local, not claimed to match Flax RNG ordering. Identical
    explicit arrays, not an identical seed, define implementation parity.
    """
    keys = iter(jax.random.split(key, 3 + 5*config.depth))
    d, f = config.embedding_dim, config.mlp_dim

    def matrix(rows: int, columns: int) -> jax.Array:
        bound = config.initialization_scale*np.sqrt(6/(rows+columns))
        return jax.random.uniform(next(keys), (rows, columns), dtype=jnp.float64,
                                  minval=-bound, maxval=bound)

    def norm() -> dict:
        return dict(scale=jnp.ones(d, jnp.float64), bias=jnp.zeros(d, jnp.float64))

    patch = matrix(config.patch_site_count, d)
    layers = []
    for _ in range(config.depth):
        layer = dict(norm_spatial=norm(), norm_mlp=norm(),
                     relative_factor=matrix(config.head_count, config.token_count),
                     value=matrix(d, d), value_bias=jnp.zeros(d, jnp.float64),
                     output=matrix(d, d), output_bias=jnp.zeros(d, jnp.float64),
                     mlp_in=matrix(d, f), mlp_in_bias=jnp.zeros(f, jnp.float64),
                     mlp_out=matrix(f, d), mlp_out_bias=jnp.zeros(d, jnp.float64))
        if config.use_distance_envelope:
            layer['raw_distance_decay'] = jnp.full(
                config.head_count, np.log(np.expm1(config.initial_distance_decay)), jnp.float64)
        layers.append(layer)
    return dict(patch_embedding=patch, patch_embedding_bias=jnp.zeros(d, jnp.float64),
                layers=tuple(layers), final_norm=norm(),
                log_cosh_head=dict(weight_real=matrix(d, d), weight_imag=matrix(d, d),
                    bias_real=jnp.zeros(d, jnp.float64), bias_imag=jnp.zeros(d, jnp.float64),
                    norm_real=norm(), norm_imag=norm()))


def literature_layer_norm(values: jax.Array, params: dict,
                          epsilon: float = 1e-6) -> jax.Array:
    """Flax default fast-variance LN over the embedding axis."""
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.maximum(0., jnp.mean(values**2, axis=-1, keepdims=True)-mean**2)
    return (values-mean)*jax.lax.rsqrt(variance+epsilon)*params['scale']+params['bias']


@lru_cache(maxsize=64)
def literature_relative_geometry(config: LiteratureSpatialViTConfig) -> tuple[np.ndarray, np.ndarray]:
    """Relative-factor indices and shortest distances in physical-site units.

    Triangular primitive vectors have 120 degrees: (1,0),(0,1),(1,1) are
    nearest neighbours. Torus minimization is done offline, not during JIT.
    """
    coords = np.array([(r, c) for r in range(config.patch_grid_rows)
                       for c in range(config.patch_grid_columns)], dtype=int)
    delta = coords[None, :, :]-coords[:, None, :]
    relative = ((delta[..., 0] % config.patch_grid_rows)*config.patch_grid_columns
                + delta[..., 1] % config.patch_grid_columns)
    squared = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            r = delta[..., 0]*config.patch_rows+a*config.rows
            c = delta[..., 1]*config.patch_columns+b*config.columns
            squared.append(r*r+c*c-(r*c if config.geometry == 'triangular' else 0))
    return relative, np.sqrt(np.min(squared, axis=0))


def literature_spatial_vit_log_amplitude(params: dict, configuration: jax.Array,
                                         config: LiteratureSpatialViTConfig) -> jax.Array:
    """Unnormalized complex log amplitude; invalid bits/target m give -inf."""
    bits = jnp.asarray(configuration, jnp.int32)
    if bits.shape != (config.n_sites,):
        raise ValueError('configuration shape disagrees with finite lattice')
    valid = jnp.all((bits == 0) | (bits == 1)) & (jnp.sum(bits) == config.target_n_up)
    hidden = _patch_values(bits, config)@params['patch_embedding']+params['patch_embedding_bias']
    relative, distances = literature_relative_geometry(config)
    norm = lambda x, p: literature_layer_norm(x, p, config.layer_norm_epsilon)
    for layer in params['layers']:
        values = (norm(hidden, layer['norm_spatial'])@layer['value']+layer['value_bias'])
        values = values.reshape(config.token_count, config.head_count, config.head_dim)
        weights = layer['relative_factor'][:, jnp.asarray(relative)]
        if config.use_distance_envelope:
            weights = weights*jax.nn.softmax(-jax.nn.softplus(layer['raw_distance_decay'])[:, None, None]
                                             *jnp.asarray(distances)[None], axis=-1)
        mixed = jnp.einsum('hij,jhd->ihd', weights, values).reshape(config.token_count, config.embedding_dim)
        hidden = hidden+mixed@layer['output']+layer['output_bias']
        hidden = hidden+jax.nn.gelu(norm(hidden, layer['norm_mlp'])@layer['mlp_in']
                                    +layer['mlp_in_bias'])@layer['mlp_out']+layer['mlp_out_bias']
    pooled = (norm(jnp.sum(hidden, axis=0), params['final_norm']) if config.pooling == 'authors'
              else jnp.sum(norm(hidden, params['final_norm']), axis=0))
    head = params['log_cosh_head']
    real = norm(pooled@head['weight_real']+head['bias_real'], head['norm_real'])
    imag = norm(pooled@head['weight_imag']+head['bias_imag'], head['norm_imag'])
    return jnp.where(valid, jnp.sum(_complex_log_cosh(real+1j*imag)), -jnp.inf+0j)
