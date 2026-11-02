"""Configuration-dependent spatial attention in a complete local Schur basis.

The neural configuration is a collection of *reduced* labels ``(q, a)``:
``q`` is a doubled cell spin and ``a`` resolves its local multiplicity.  No
magnetic label is exposed to the Transformer.  The resulting scalar is used
as a coefficient of a fixed Condon--Shortley singlet basis vector, so every
parameter value and every parameter tangent remains an exact total-spin
singlet.

This module is deliberately the exact-small implementation boundary.  It
supports the complete one- and two-cell :class:`StrictSchurSpatialPlan`
oracles.  Unlike ``StaticSchurGenerator``, its attention tokens change with
the Schur configuration.  Scaling beyond two cells requires autoregressive
sampling of the same reduced labels and local recoupling kernels; it must not
fall back to applying nonlinearities to computational-basis spinors.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.local_schur import cell_schur_spec
from .strict_schur_spatial import (
    StrictSchurSpatialPlan,
    strict_schur_amplitudes,
)


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SU2SchurConfigViTConfig:
    """Architecture for an exact-singlet Schur-configuration Transformer."""

    plan: StrictSchurSpatialPlan
    embedding_dim: int = 16
    head_count: int = 4
    depth: int = 2
    mlp_ratio: int = 2
    initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive and finite")

    @property
    def n_cells(self) -> int:
        return len(self.plan.cells)

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio

    @property
    def maximum_q(self) -> int:
        return max(cell.size for cell in self.plan.cells)

    @property
    def maximum_multiplicity(self) -> int:
        return max(
            cell_schur_spec(cell.kind).maximum_multiplicity
            for cell in self.plan.cells
        )


def schur_configuration_labels(plan: StrictSchurSpatialPlan) -> np.ndarray:
    """Return immutable ``(configuration, cell, [q,a])`` doubled labels.

    Row order is exactly the flattened reduced-block order used by
    :func:`strict_schur_amplitudes`.
    """

    rows: list[tuple[tuple[int, int], ...]] = []
    if len(plan.cells) == 1:
        for a in range(plan.reduced_shapes[0][0]):
            rows.append(((0, a),))
    else:
        for q, shape in zip(plan.active_q, plan.reduced_shapes, strict=True):
            for left_a in range(shape[0]):
                for right_a in range(shape[1]):
                    rows.append(((q, left_a), (q, right_a)))
    result = np.asarray(rows, dtype=np.int32)
    expected = (plan.reduced_parameter_count, len(plan.cells), 2)
    if result.shape != expected:
        raise RuntimeError(
            f"Schur configuration packing produced {result.shape}, expected {expected}"
        )
    result.setflags(write=False)
    return result


def schur_spatial_pair_features(plan: StrictSchurSpatialPlan) -> np.ndarray:
    """Return immutable relative ``dx,dy,distance`` features between cells."""

    coordinates = np.asarray(plan.cell_coordinates, dtype=np.float64)
    delta = coordinates[None, :, :] - coordinates[:, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = max(float(np.max(distance, initial=0.0)), 1.0)
    result = np.concatenate((delta / scale, distance / scale), axis=-1)
    result.setflags(write=False)
    return result


def _matrix(key: jax.Array, rows: int, columns: int, scale: float) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(key, (rows, columns), dtype=jnp.float64)


def init_su2_schur_config_vit(
    key: jax.Array,
    config: SU2SchurConfigViTConfig,
) -> dict[str, object]:
    """Initialize the reduced-label Transformer with a nonzero wavefunction."""

    keys = iter(jax.random.split(key, 7 + 7 * config.depth))
    scale = float(config.initialization_scale)
    layers = []
    for _ in range(config.depth):
        layers.append(
            {
                "query": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "key": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "value": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "output": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "pair_weight": _matrix(next(keys), 3, config.head_count, scale),
                "mlp_in": _matrix(next(keys), config.embedding_dim, config.mlp_dim, scale),
                "mlp_out": _matrix(next(keys), config.mlp_dim, config.embedding_dim, scale),
            }
        )
    return {
        "q_embedding": scale
        * jax.random.normal(
            next(keys), (config.maximum_q + 1, config.embedding_dim), dtype=jnp.float64
        ),
        "multiplicity_embedding": scale
        * jax.random.normal(
            next(keys),
            (config.maximum_multiplicity, config.embedding_dim),
            dtype=jnp.float64,
        ),
        "position_projection": _matrix(next(keys), 2, config.embedding_dim, scale),
        "cell_size_projection": _matrix(next(keys), 1, config.embedding_dim, scale),
        "layers": tuple(layers),
        "coefficient_head": _matrix(next(keys), config.embedding_dim, 1, scale)[:, 0],
        "coefficient_bias": jnp.asarray(scale, dtype=jnp.float64),
    }


def _layer_norm(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)


def su2_schur_config_vit_tokens(
    params: dict[str, object],
    config: SU2SchurConfigViTConfig,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Return contextual tokens and per-layer attention probabilities.

    The leading axis enumerates multiplicity configurations.  This explicit
    axis is the essential distinction from the earlier geometry-only heads.
    """

    labels = jnp.asarray(schur_configuration_labels(config.plan))
    q = labels[..., 0]
    multiplicity = labels[..., 1]
    coordinates = jnp.asarray(config.plan.cell_coordinates, dtype=jnp.float64)
    sizes = jnp.asarray(
        [[cell.size / config.plan.n_sites] for cell in config.plan.cells],
        dtype=jnp.float64,
    )
    tokens = (
        params["q_embedding"][q]
        + params["multiplicity_embedding"][multiplicity]
        + coordinates @ params["position_projection"]
        + sizes @ params["cell_size_projection"]
    )
    pair = jnp.asarray(schur_spatial_pair_features(config.plan))
    histories = []
    for layer in params["layers"]:
        normalized = _layer_norm(tokens)
        query = (normalized @ layer["query"]).reshape(
            labels.shape[0], config.n_cells, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            labels.shape[0], config.n_cells, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            labels.shape[0], config.n_cells, config.head_count, config.head_dim
        )
        logits = jnp.einsum("bihd,bjhd->bhij", query, key) / np.sqrt(
            config.head_dim
        )
        logits = logits + jnp.einsum("ijf,fh->hij", pair, layer["pair_weight"])[
            None, ...
        ]
        attention = jax.nn.softmax(logits, axis=-1)
        histories.append(attention)
        attended = jnp.einsum("bhij,bjhd->bihd", attention, value).reshape(
            labels.shape[0], config.n_cells, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer[
            "mlp_out"
        ]
    return _layer_norm(tokens), tuple(histories)


def su2_schur_config_vit_coefficients(
    params: dict[str, object],
    config: SU2SchurConfigViTConfig,
) -> jax.Array:
    """Evaluate one real coefficient for every complete Schur configuration."""

    tokens, _ = su2_schur_config_vit_tokens(params, config)
    pooled = jnp.mean(tokens, axis=1)
    return pooled @ params["coefficient_head"] + params["coefficient_bias"]


def su2_schur_config_vit_reduced_blocks(
    params: dict[str, object],
    config: SU2SchurConfigViTConfig,
) -> tuple[jax.Array, ...]:
    """Pack generated coefficients into strict-Schur reduced blocks."""

    coefficients = su2_schur_config_vit_coefficients(params, config)
    blocks = []
    start = 0
    for shape in config.plan.reduced_shapes:
        stop = start + int(np.prod(shape))
        blocks.append(coefficients[start:stop].reshape(shape))
        start = stop
    return tuple(blocks)


def su2_schur_config_vit_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SU2SchurConfigViTConfig,
) -> jax.Array:
    """Evaluate exact-singlet physical amplitudes for a batch of bitstrings."""

    blocks = su2_schur_config_vit_reduced_blocks(params, config)
    return strict_schur_amplitudes(blocks, spins, config.plan)


def projected_su2_schur_config_vit_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SU2SchurConfigViTConfig,
    permutations: Iterable[Iterable[int]],
) -> jax.Array:
    """Apply an exact trivial-irrep spatial Reynolds projection.

    Each permutation is an old-site index list, so ``spins[:, permutation]``
    is one group-transformed configuration.  Projection preserves exact SU(2)
    because physical permutations commute with global spin rotations.
    """

    batch = jnp.asarray(spins)
    if batch.ndim != 2 or batch.shape[1] != config.plan.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.plan.n_sites})"
        )
    group = np.asarray(tuple(tuple(row) for row in permutations), dtype=np.int32)
    if group.ndim != 2 or group.shape[1] != config.plan.n_sites or len(group) == 0:
        raise ValueError("permutations must be a nonempty (group,N) integer array")
    expected = np.arange(config.plan.n_sites)
    if any(not np.array_equal(np.sort(row), expected) for row in group):
        raise ValueError("every spatial action must be a permutation of 0..N-1")
    values = [
        su2_schur_config_vit_amplitudes(
            params, batch[:, jnp.asarray(row)], config
        )
        for row in group
    ]
    return jnp.mean(jnp.stack(values, axis=0), axis=0)


def su2_schur_config_vit_parameter_count(params: object) -> int:
    """Count real trainable scalars."""

    return int(
        sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))
    )


__all__ = [
    "SU2SchurConfigViTConfig",
    "init_su2_schur_config_vit",
    "projected_su2_schur_config_vit_amplitudes",
    "schur_configuration_labels",
    "schur_spatial_pair_features",
    "su2_schur_config_vit_amplitudes",
    "su2_schur_config_vit_coefficients",
    "su2_schur_config_vit_parameter_count",
    "su2_schur_config_vit_reduced_blocks",
    "su2_schur_config_vit_tokens",
]
