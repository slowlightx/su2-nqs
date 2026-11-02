"""Space-group-invariant Transformer on exact-singlet RVB coverings.

The network never sees magnetic indices.  Its tokens are dimers in a perfect
matching; node labels identify physical bond orbits and pair labels identify
ordered bond-pair orbits under the lattice space group.  A group operation can
therefore only permute tokens and their pair-label matrix.  Shared self-
attention followed by mean pooling gives an exactly invariant scalar
coefficient without a Reynolds sum in the forward pass.

This module maps a batch of *coverings* to complex reduced coefficients.  The
physical wavefunction, nonorthogonal valence-bond overlaps, Hamiltonian, and
sampling remain outside the model layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class RVBCoveringTransformerConfig:
    """Static capacity contract for invariant covering attention."""

    token_count: int
    bond_class_count: int
    relation_class_count: int
    embedding_dim: int = 4
    head_count: int = 1
    depth: int = 1
    mlp_ratio: int = 2
    initialization_scale: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "token_count",
            "bond_class_count",
            "relation_class_count",
            "embedding_dim",
            "head_count",
            "depth",
            "mlp_ratio",
        ):
            object.__setattr__(self, name, _positive_integer(name, getattr(self, name)))
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if (
            not np.isfinite(self.initialization_scale)
            or self.initialization_scale <= 0.0
        ):
            raise ValueError("initialization_scale must be positive and finite")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio


def _matrix(
    key: jax.Array, rows: int, columns: int, scale: float
) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(
        key, (rows, columns), dtype=jnp.float64
    )


def init_rvb_covering_transformer(
    key: jax.Array,
    config: RVBCoveringTransformerConfig,
) -> dict[str, object]:
    """Initialize one invariant relation-attention coefficient model."""

    # Each attention block consumes eight independent keys and the shared
    # embedding/readout consumes three more.  The former ``4 + 7 * depth``
    # expression happened to be correct only for ``depth == 1`` and raised
    # ``StopIteration`` for every deeper model.
    keys = iter(jax.random.split(key, 3 + 8 * config.depth))
    scale = float(config.initialization_scale)
    layers = []
    for _ in range(config.depth):
        layers.append(
            {
                "query": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim, scale
                ),
                "key": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim, scale
                ),
                "value": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim, scale
                ),
                "output": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim, scale
                ),
                "relation_bias": scale
                * jax.random.normal(
                    next(keys),
                    (config.relation_class_count, config.head_count),
                    dtype=jnp.float64,
                ),
                "relation_value": scale
                * jax.random.normal(
                    next(keys),
                    (
                        config.relation_class_count,
                        config.head_count,
                        config.head_dim,
                    ),
                    dtype=jnp.float64,
                ),
                "mlp_in": _matrix(
                    next(keys), config.embedding_dim, config.mlp_dim, scale
                ),
                "mlp_out": _matrix(
                    next(keys), config.mlp_dim, config.embedding_dim, scale
                ),
            }
        )
    return {
        "bond_embedding": scale
        * jax.random.normal(
            next(keys),
            (config.bond_class_count, config.embedding_dim),
            dtype=jnp.float64,
        ),
        "layers": tuple(layers),
        "readout_real": _matrix(
            next(keys), config.embedding_dim, 1, scale
        )[:, 0],
        "readout_imag": _matrix(
            next(keys), config.embedding_dim, 1, scale
        )[:, 0],
        "bias_real": jnp.asarray(0.0, dtype=jnp.float64),
        "bias_imag": jnp.asarray(0.0, dtype=jnp.float64),
    }


def _layer_norm(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)


def _checked_features(
    bond_classes: jax.Array,
    relation_classes: jax.Array,
    config: RVBCoveringTransformerConfig,
) -> tuple[jax.Array, jax.Array]:
    nodes = jnp.asarray(bond_classes)
    relations = jnp.asarray(relation_classes)
    if nodes.ndim != 2 or nodes.shape[1] != config.token_count:
        raise ValueError(
            f"bond_classes must have shape (batch,{config.token_count})"
        )
    if relations.shape != (nodes.shape[0], config.token_count, config.token_count):
        raise ValueError(
            "relation_classes must have shape (batch,token_count,token_count)"
        )
    if nodes.dtype == jnp.bool_ or not jnp.issubdtype(nodes.dtype, jnp.integer):
        raise TypeError("bond_classes must use an integer dtype")
    if relations.dtype == jnp.bool_ or not jnp.issubdtype(
        relations.dtype, jnp.integer
    ):
        raise TypeError("relation_classes must use an integer dtype")
    return nodes.astype(jnp.int32), relations.astype(jnp.int32)


def rvb_covering_transformer_coefficients(
    params: dict[str, object],
    bond_classes: jax.Array,
    relation_classes: jax.Array,
    config: RVBCoveringTransformerConfig,
) -> jax.Array:
    """Return one complex scalar coefficient per exact-singlet covering."""

    nodes, relations = _checked_features(
        bond_classes, relation_classes, config
    )
    tokens = params["bond_embedding"][nodes]
    scale = np.sqrt(float(config.head_dim))
    for layer in params["layers"]:
        normalized = _layer_norm(tokens)
        query = (normalized @ layer["query"]).reshape(
            nodes.shape[0], config.token_count, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            nodes.shape[0], config.token_count, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            nodes.shape[0], config.token_count, config.head_count, config.head_dim
        )
        scores = jnp.einsum("bihd,bjhd->bhij", query, key) / scale
        pair_bias = layer["relation_bias"][relations].transpose((0, 3, 1, 2))
        weights = jax.nn.softmax(scores + pair_bias, axis=-1)
        pair_values = (
            value[:, None, :, :, :] + layer["relation_value"][relations]
        )
        context = jnp.einsum("bhij,bijhd->bihd", weights, pair_values)
        context = context.reshape(
            nodes.shape[0], config.token_count, config.embedding_dim
        )
        tokens = tokens + context @ layer["output"]
        mlp_input = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(mlp_input @ layer["mlp_in"]) @ layer[
            "mlp_out"
        ]
    pooled = jnp.mean(_layer_norm(tokens), axis=1)
    real = pooled @ params["readout_real"] + params["bias_real"]
    imag = pooled @ params["readout_imag"] + params["bias_imag"]
    return real + 1.0j * imag


def rvb_covering_transformer_parameter_count(params: object) -> int:
    """Return the number of trainable real scalars."""

    return int(
        sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params))
    )


__all__ = [
    "RVBCoveringTransformerConfig",
    "init_rvb_covering_transformer",
    "rvb_covering_transformer_coefficients",
    "rvb_covering_transformer_parameter_count",
]
