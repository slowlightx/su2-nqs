"""Validation-scale fixed-``Sz`` spatial Vision Transformer wavefunction.

This module provides the conventional physical-spin baseline required before
attempting an SU(2)-equivariant Transformer.  It deliberately implements the
*factored spatial attention* mechanism used by modern frustrated-spin NQS:
each head applies a translation-shared learned relative-displacement factor,
modulated by a normalized exponential distance kernel.  The mixing weights
are static model parameters rather than configuration-dependent Q/K scores.

The model consumes physical bits (``0=down``, ``1=up``), linearly embeds
nonoverlapping rectangular patches, and returns an unnormalised complex log
amplitude.  Fixed magnetisation is enforced at the API boundary.  Lattice
character projection, Hamiltonians, and optimisation remain outside the
model.  The local ``2x2`` patch used by the 4x4 validation workload is not a
claim to reproduce the production-size patching of the literature.
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
class FixedSzSpatialViTConfig:
    """Static geometry and capacity contract for the fixed-``Sz`` baseline."""

    geometry: str
    rows: int
    columns: int
    patch_rows: int = 2
    patch_columns: int = 2
    embedding_dim: int = 16
    head_count: int = 2
    depth: int = 1
    mlp_ratio: int = 2
    target_m2: int = 0
    initialization_scale: float = 0.2
    initial_distance_decay: float = 1.0
    use_distance_envelope: bool = True

    def __post_init__(self) -> None:
        if self.geometry not in {"square", "triangular"}:
            raise ValueError("geometry must be 'square' or 'triangular'")
        for name in (
            "rows",
            "columns",
            "patch_rows",
            "patch_columns",
            "embedding_dim",
            "head_count",
            "depth",
            "mlp_ratio",
        ):
            object.__setattr__(self, name, _positive_integer(name, getattr(self, name)))
        if self.rows % self.patch_rows or self.columns % self.patch_columns:
            raise ValueError("patch dimensions must divide the finite lattice")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if isinstance(self.target_m2, bool) or not isinstance(self.target_m2, Integral):
            raise TypeError("target_m2 must be a doubled integer")
        if abs(int(self.target_m2)) > self.n_sites:
            raise ValueError("target_m2 is outside the physical spin range")
        if (self.n_sites + int(self.target_m2)) % 2:
            raise ValueError("target_m2 is unreachable for this lattice")
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive and finite")
        if not np.isfinite(self.initial_distance_decay) or self.initial_distance_decay <= 0:
            raise ValueError("initial_distance_decay must be positive and finite")
        if not isinstance(self.use_distance_envelope, bool):
            raise TypeError("use_distance_envelope must be boolean")

    @property
    def n_sites(self) -> int:
        return self.rows * self.columns

    @property
    def target_n_up(self) -> int:
        return (self.n_sites + int(self.target_m2)) // 2

    @property
    def patch_site_count(self) -> int:
        return self.patch_rows * self.patch_columns

    @property
    def patch_grid_rows(self) -> int:
        return self.rows // self.patch_rows

    @property
    def patch_grid_columns(self) -> int:
        return self.columns // self.patch_columns

    @property
    def token_count(self) -> int:
        return self.patch_grid_rows * self.patch_grid_columns

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio


def _matrix(
    key: jax.Array,
    rows: int,
    columns: int,
    scale: float,
) -> jax.Array:
    standard = np.sqrt(2.0 / float(rows + columns)) * scale
    return standard * jax.random.normal(
        key, (rows, columns), dtype=jnp.float64
    )


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def init_fixed_sz_spatial_vit(
    key: jax.Array,
    config: FixedSzSpatialViTConfig,
) -> dict[str, object]:
    """Initialise a deterministic factored-spatial Transformer."""

    keys = iter(jax.random.split(key, 8 + 8 * config.depth))
    patch_embedding = _matrix(
        next(keys),
        config.patch_site_count,
        config.embedding_dim,
        config.initialization_scale,
    )
    layers = []
    for _ in range(config.depth):
        alpha = jnp.ones(
            (
                config.head_count,
                config.patch_grid_rows,
                config.patch_grid_columns,
            ),
            dtype=jnp.float64,
        )
        alpha = alpha + 0.01 * config.initialization_scale * jax.random.normal(
            next(keys), alpha.shape, dtype=jnp.float64
        )
        layers.append(
            {
                "norm_spatial": {
                    "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                    "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
                },
                "value": _matrix(
                    next(keys),
                    config.embedding_dim,
                    config.embedding_dim,
                    config.initialization_scale,
                ),
                "output": _matrix(
                    next(keys),
                    config.embedding_dim,
                    config.embedding_dim,
                    config.initialization_scale,
                ),
                "relative_factor": alpha,
                "raw_distance_decay": jnp.full(
                    (config.head_count,),
                    _inverse_softplus(config.initial_distance_decay),
                    dtype=jnp.float64,
                ),
                "norm_mlp": {
                    "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                    "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
                },
                "mlp_in": _matrix(
                    next(keys),
                    config.embedding_dim,
                    config.mlp_dim,
                    config.initialization_scale,
                ),
                "mlp_in_bias": jnp.zeros((config.mlp_dim,), dtype=jnp.float64),
                "mlp_out": _matrix(
                    next(keys),
                    config.mlp_dim,
                    config.embedding_dim,
                    config.initialization_scale,
                ),
                "mlp_out_bias": jnp.zeros(
                    (config.embedding_dim,), dtype=jnp.float64
                ),
            }
        )
    return {
        "patch_embedding": patch_embedding,
        "patch_embedding_bias": jnp.zeros(
            (config.embedding_dim,), dtype=jnp.float64
        ),
        "layers": tuple(layers),
        "final_norm": {
            "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
            "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
        },
        "log_cosh_head": {
            "weight_real": _matrix(
                next(keys),
                config.embedding_dim,
                config.embedding_dim,
                config.initialization_scale,
            ),
            "weight_imag": _matrix(
                next(keys),
                config.embedding_dim,
                config.embedding_dim,
                config.initialization_scale,
            ),
            "bias_real": jnp.zeros(
                (config.embedding_dim,), dtype=jnp.float64
            ),
            "bias_imag": jnp.zeros(
                (config.embedding_dim,), dtype=jnp.float64
            ),
        },
    }


def _layer_norm(
    values: jax.Array,
    params: dict[str, jax.Array],
) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    normalized = (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)
    return normalized * params["scale"] + params["bias"]


def _complex_log_cosh(values: jax.Array) -> jax.Array:
    """Stable principal-branch ``log(cosh(z))`` away from its zeros."""

    signs = jnp.where(jnp.real(values) >= 0.0, 1.0, -1.0)
    reflected = signs * values
    return reflected + jnp.log1p(jnp.exp(-2.0 * reflected)) - jnp.log(2.0)


def _patch_values(
    configuration: jax.Array,
    config: FixedSzSpatialViTConfig,
) -> jax.Array:
    grid = configuration.reshape((config.rows, config.columns))
    patches = grid.reshape(
        (
            config.patch_grid_rows,
            config.patch_rows,
            config.patch_grid_columns,
            config.patch_columns,
        )
    ).transpose((0, 2, 1, 3))
    patches = patches.reshape((config.token_count, config.patch_site_count))
    return 2.0 * patches.astype(jnp.float64) - 1.0


def _patch_coordinates(config: FixedSzSpatialViTConfig) -> np.ndarray:
    return np.asarray(
        [
            (row, column)
            for row in range(config.patch_grid_rows)
            for column in range(config.patch_grid_columns)
        ],
        dtype=np.int32,
    )


def _relative_geometry(
    config: FixedSzSpatialViTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return displacement lookup tables and minimal torus distances."""

    coordinates = _patch_coordinates(config)
    first = coordinates[:, None, :]
    second = coordinates[None, :, :]
    displacement = second - first
    row_index = np.mod(displacement[..., 0], config.patch_grid_rows)
    column_index = np.mod(displacement[..., 1], config.patch_grid_columns)
    distances = np.empty((config.token_count, config.token_count), dtype=np.float64)
    for source in range(config.token_count):
        for target in range(config.token_count):
            delta = coordinates[target] - coordinates[source]
            candidates = []
            for row_shift in (-1, 0, 1):
                for column_shift in (-1, 0, 1):
                    dy = float(
                        delta[0] + row_shift * config.patch_grid_rows
                    )
                    dx = float(
                        delta[1] + column_shift * config.patch_grid_columns
                    )
                    squared = dy * dy + dx * dx
                    if config.geometry == "triangular":
                        squared += dx * dy
                    candidates.append(max(squared, 0.0))
            distances[source, target] = np.sqrt(min(candidates))
    return row_index, column_index, distances


def fixed_sz_spatial_vit_spatial_weights(
    params: dict[str, object],
    config: FixedSzSpatialViTConfig,
    *,
    layer_index: int,
) -> jax.Array:
    """Return ``(head, source, target)`` static factored spatial weights."""

    if not 0 <= layer_index < config.depth:
        raise ValueError("layer_index is outside the Transformer depth")
    row_index, column_index, distances = _relative_geometry(config)
    layer = params["layers"][layer_index]
    decay = jax.nn.softplus(layer["raw_distance_decay"])
    if not config.use_distance_envelope:
        decay = jnp.zeros_like(decay)
    log_envelope = -decay[:, None, None] * jnp.asarray(distances)[None, :, :]
    envelope = jax.nn.softmax(log_envelope, axis=-1)
    relative = layer["relative_factor"][
        :,
        jnp.asarray(row_index, dtype=jnp.int32),
        jnp.asarray(column_index, dtype=jnp.int32),
    ]
    return envelope * relative


def fixed_sz_spatial_vit_log_amplitude(
    params: dict[str, object],
    configuration: jax.Array,
    config: FixedSzSpatialViTConfig,
) -> jax.Array:
    """Evaluate one complex log amplitude, masked to the target ``Sz``."""

    bits = jnp.asarray(configuration, dtype=jnp.int32)
    if bits.shape != (config.n_sites,):
        raise ValueError(
            f"expected configuration shape ({config.n_sites},), got {bits.shape}"
        )
    valid_bits = jnp.all((bits == 0) | (bits == 1))
    valid_sector = jnp.sum(bits) == config.target_n_up
    patches = _patch_values(jnp.where(valid_bits, bits, 0), config)
    hidden = patches @ params["patch_embedding"] + params["patch_embedding_bias"]
    for layer_index, layer in enumerate(params["layers"]):
        normalized = _layer_norm(hidden, layer["norm_spatial"])
        values = normalized @ layer["value"]
        values = values.reshape(
            (config.token_count, config.head_count, config.head_dim)
        )
        weights = fixed_sz_spatial_vit_spatial_weights(
            params, config, layer_index=layer_index
        )
        mixed = jnp.einsum("hij,jhd->ihd", weights, values)
        mixed = mixed.reshape((config.token_count, config.embedding_dim))
        hidden = hidden + mixed @ layer["output"]
        normalized = _layer_norm(hidden, layer["norm_mlp"])
        mlp = jax.nn.gelu(
            normalized @ layer["mlp_in"] + layer["mlp_in_bias"]
        )
        hidden = hidden + mlp @ layer["mlp_out"] + layer["mlp_out_bias"]
    pooled = jnp.sum(_layer_norm(hidden, params["final_norm"]), axis=0)
    head = params["log_cosh_head"]
    weight = head["weight_real"] + 1j * head["weight_imag"]
    bias = head["bias_real"] + 1j * head["bias_imag"]
    value = jnp.sum(_complex_log_cosh(pooled @ weight + bias))
    return jnp.where(valid_bits & valid_sector, value, -jnp.inf + 0.0j)


def fixed_sz_spatial_vit_log_amplitudes(
    params: dict[str, object],
    configurations: jax.Array,
    config: FixedSzSpatialViTConfig,
) -> jax.Array:
    """Vectorise complex log amplitudes over physical configurations."""

    array = jnp.asarray(configurations, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected configuration batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: fixed_sz_spatial_vit_log_amplitude(params, row, config)
    )(array)


def fixed_sz_spatial_vit_amplitudes(
    params: dict[str, object],
    configurations: jax.Array,
    config: FixedSzSpatialViTConfig,
) -> jax.Array:
    """Return a batch of exponentiated amplitudes with a stable common shift."""

    logs = fixed_sz_spatial_vit_log_amplitudes(params, configurations, config)
    finite = jnp.isfinite(jnp.real(logs))
    maximum = jnp.max(jnp.where(finite, jnp.real(logs), -jnp.inf))
    shifted = jnp.where(finite, logs - jax.lax.stop_gradient(maximum), -jnp.inf)
    return jnp.where(finite, jnp.exp(shifted), 0.0 + 0.0j)


def fixed_sz_spatial_vit_parameter_count(params: dict[str, object]) -> int:
    """Count real trainable scalars in the Transformer pytree."""

    return int(
        sum(np.prod(np.asarray(leaf).shape) for leaf in jax.tree_util.tree_leaves(params))
    )


__all__ = [
    "FixedSzSpatialViTConfig",
    "fixed_sz_spatial_vit_amplitudes",
    "fixed_sz_spatial_vit_log_amplitude",
    "fixed_sz_spatial_vit_log_amplitudes",
    "fixed_sz_spatial_vit_parameter_count",
    "fixed_sz_spatial_vit_spatial_weights",
    "init_fixed_sz_spatial_vit",
]
