"""Conventional position-aware ViT control with no imposed lattice symmetry."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .fixed_sz_spatial_vit import _complex_log_cosh, _layer_norm, _matrix


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FixedSzGenericViTConfig:
    """A site-token ViT restricted only to one doubled-magnetization sector."""

    n_sites: int
    embedding_dim: int = 8
    head_count: int = 2
    depth: int = 1
    mlp_ratio: int = 2
    target_m2: int = 0
    initialization_scale: float = 0.15

    def __post_init__(self) -> None:
        for name in ("n_sites", "embedding_dim", "head_count", "depth", "mlp_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if abs(int(self.target_m2)) > self.n_sites or (
            self.n_sites + int(self.target_m2)
        ) % 2:
            raise ValueError("target_m2 is unreachable")
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive and finite")

    @property
    def target_n_up(self) -> int:
        return (self.n_sites + int(self.target_m2)) // 2

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio


def init_fixed_sz_generic_vit(
    key: jax.Array, config: FixedSzGenericViTConfig
) -> dict[str, object]:
    """Initialize learned spin, absolute-position, and ordinary-attention tokens."""

    keys = iter(jax.random.split(key, 8 + 8 * config.depth))
    scale = float(config.initialization_scale)
    spin_embedding = scale * jax.random.normal(
        next(keys), (2, config.embedding_dim), dtype=jnp.float64
    )
    position_embedding = scale * jax.random.normal(
        next(keys), (config.n_sites + 1, config.embedding_dim), dtype=jnp.float64
    )
    cls_token = scale * jax.random.normal(
        next(keys), (config.embedding_dim,), dtype=jnp.float64
    )
    layers = []
    for _ in range(config.depth):
        layers.append(
            {
                "norm_attention": {
                    "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                    "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
                },
                "query": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "key": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "value": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "output": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "norm_mlp": {
                    "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                    "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
                },
                "mlp_in": _matrix(next(keys), config.embedding_dim, config.mlp_dim, scale),
                "mlp_out": _matrix(next(keys), config.mlp_dim, config.embedding_dim, scale),
            }
        )
    return {
        "spin_embedding": spin_embedding,
        "position_embedding": position_embedding,
        "cls_token": cls_token,
        "layers": tuple(layers),
        "final_norm": {
            "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
            "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
        },
        "log_cosh_head": {
            "weight_real": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
            "weight_imag": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
            "bias_real": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
            "bias_imag": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
        },
    }


def fixed_sz_generic_vit_log_amplitude(
    params: dict[str, object],
    configuration: jax.Array,
    config: FixedSzGenericViTConfig,
) -> jax.Array:
    """Evaluate one unprojected complex log amplitude."""

    bits = jnp.asarray(configuration, dtype=jnp.int32)
    if bits.shape != (config.n_sites,):
        raise ValueError(f"expected configuration shape ({config.n_sites},)")
    valid_bits = jnp.all((bits == 0) | (bits == 1))
    valid_sector = jnp.sum(bits) == config.target_n_up
    safe = jnp.clip(bits, 0, 1)
    sites = params["spin_embedding"][safe]
    tokens = jnp.concatenate((params["cls_token"][None, :], sites), axis=0)
    tokens = tokens + params["position_embedding"]
    token_count = config.n_sites + 1
    for layer in params["layers"]:
        normalized = _layer_norm(tokens, layer["norm_attention"])
        query = (normalized @ layer["query"]).reshape(
            token_count, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            token_count, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            token_count, config.head_count, config.head_dim
        )
        logits = jnp.einsum("ihd,jhd->hij", query, key) / np.sqrt(config.head_dim)
        attention = jax.nn.softmax(logits, axis=-1)
        mixed = jnp.einsum("hij,jhd->ihd", attention, value).reshape(
            token_count, config.embedding_dim
        )
        tokens = tokens + mixed @ layer["output"]
        normalized = _layer_norm(tokens, layer["norm_mlp"])
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer["mlp_out"]
    pooled = _layer_norm(tokens, params["final_norm"])[0]
    head = params["log_cosh_head"]
    weight = head["weight_real"] + 1j * head["weight_imag"]
    bias = head["bias_real"] + 1j * head["bias_imag"]
    result = jnp.sum(_complex_log_cosh(pooled @ weight + bias))
    return jnp.where(valid_bits & valid_sector, result, -jnp.inf + 0.0j)


def fixed_sz_generic_vit_amplitude(
    params: dict[str, object],
    configuration: jax.Array,
    config: FixedSzGenericViTConfig,
) -> jax.Array:
    """Exponentiate a single generic-ViT log amplitude."""

    value = fixed_sz_generic_vit_log_amplitude(params, configuration, config)
    return jnp.where(jnp.isfinite(jnp.real(value)), jnp.exp(value), 0.0 + 0.0j)


__all__ = [
    "FixedSzGenericViTConfig",
    "fixed_sz_generic_vit_amplitude",
    "fixed_sz_generic_vit_log_amplitude",
    "init_fixed_sz_generic_vit",
]
