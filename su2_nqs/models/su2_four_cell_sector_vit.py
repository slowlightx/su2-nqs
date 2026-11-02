"""Shared autoregressive Spatial-ViT for four-cell SU(2) sectors.

The six reduced decisions are ``cell0, cell1, q01, cell2, cell3, q23``.
The root doubled spin is fixed by :class:`SU2FourCellSectorViTConfig` and is
enforced by an exact completion mask before every softmax.  A sector embedding
allows one parameter pytree to serve several spin/lattice heads; exact lattice
projection remains outside this pure coefficient model.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.local_schur import CellKind, cell_schur_spec
from ..su2.local_schur_fusion import (
    four_cell_maximum_action_count,
    four_cell_schur_sector_states,
    four_cell_sector_action_table,
    local_schur_channels,
)
from ..su2.quantum_numbers import coupling_range


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SU2FourCellSectorViTConfig:
    """One strict spin-sector view of a shared reduced-label backbone."""

    kind: CellKind
    cell_coordinates: tuple[tuple[float, float], ...]
    target_q: int
    sector_index: int
    sector_count: int = 5
    embedding_dim: int = 24
    head_count: int = 4
    depth: int = 2
    mlp_ratio: int = 2
    phase_scale: float = 0.25
    initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        _ = cell_schur_spec(self.kind)
        coordinates = np.asarray(self.cell_coordinates, dtype=np.float64)
        if coordinates.shape != (4, 2) or np.any(~np.isfinite(coordinates)):
            raise ValueError("four finite two-dimensional cell coordinates are required")
        if len({tuple(row) for row in coordinates}) != 4:
            raise ValueError("cell coordinates must be distinct")
        object.__setattr__(
            self,
            "cell_coordinates",
            tuple(tuple(float(value) for value in row) for row in coordinates),
        )
        for name in (
            "target_q",
            "sector_index",
            "sector_count",
            "embedding_dim",
            "head_count",
            "depth",
            "mlp_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.target_q < 0 or not four_cell_schur_sector_states(
            self.kind, int(self.target_q)
        ):
            raise ValueError("target_q is unreachable")
        if self.sector_count < 1 or not 0 <= self.sector_index < self.sector_count:
            raise ValueError("sector_index must lie in [0,sector_count)")
        if min(self.embedding_dim, self.head_count, self.depth, self.mlp_ratio) < 1:
            raise ValueError("architecture dimensions must be positive")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        for name in ("phase_scale", "initialization_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def n_channels(self) -> int:
        return len(local_schur_channels(self.kind))

    @property
    def action_count(self) -> int:
        return four_cell_maximum_action_count(self.kind)

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio

    @property
    def decision_coordinates(self) -> tuple[tuple[float, float], ...]:
        cells = np.asarray(self.cell_coordinates, dtype=np.float64)
        left_pair = 0.5 * (cells[0] + cells[1])
        right_pair = 0.5 * (cells[2] + cells[3])
        return tuple(
            tuple(float(value) for value in row)
            for row in (
                cells[0],
                cells[1],
                left_pair,
                cells[2],
                cells[3],
                right_pair,
            )
        )


def triangular_sector_config(
    target_q: int,
    sector_index: int,
    **kwargs,
) -> SU2FourCellSectorViTConfig:
    """Return the registered four-triangle N=12 sector view."""

    return SU2FourCellSectorViTConfig(
        "triangular_triangle",
        ((0.5, 0.25), (2.5, 0.25), (4.5, 0.75), (5.5, 0.75)),
        target_q,
        sector_index,
        **kwargs,
    )


def square_sector_config(
    target_q: int,
    sector_index: int,
    **kwargs,
) -> SU2FourCellSectorViTConfig:
    """Return the registered four-plaquette square-4x4 sector view."""

    return SU2FourCellSectorViTConfig(
        "square_plaquette",
        ((0.5, 0.5), (2.5, 0.5), (0.5, 2.5), (2.5, 2.5)),
        target_q,
        sector_index,
        **kwargs,
    )


def _matrix(key: jax.Array, rows: int, columns: int, scale: float) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(key, (rows, columns), dtype=jnp.float64)


def init_su2_four_cell_sector_vit(
    key: jax.Array,
    config: SU2FourCellSectorViTConfig,
) -> dict[str, object]:
    """Initialize a parameter pytree shared by every compatible sector view."""

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
        "channel_embedding": scale
        * jax.random.normal(
            next(keys), (config.n_channels, config.embedding_dim), dtype=jnp.float64
        ),
        "q_embedding": scale
        * jax.random.normal(
            next(keys), (config.action_count, config.embedding_dim), dtype=jnp.float64
        ),
        "position_projection": _matrix(next(keys), 2, config.embedding_dim, scale),
        "decision_embedding": scale
        * jax.random.normal(next(keys), (6, config.embedding_dim), dtype=jnp.float64),
        "sector_embedding": scale
        * jax.random.normal(
            next(keys), (config.sector_count, config.embedding_dim), dtype=jnp.float64
        ),
        "bos": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
        "layers": tuple(layers),
        "logit_head": _matrix(next(keys), config.embedding_dim, config.action_count, scale),
        "phase_head": _matrix(next(keys), config.embedding_dim, config.action_count, scale),
        "logit_bias": jnp.zeros((6, config.action_count), dtype=jnp.float64),
        "phase_bias": jnp.zeros((6, config.action_count), dtype=jnp.float64),
    }


def _layer_norm(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)


def _safe_actions(
    actions: jax.Array,
    config: SU2FourCellSectorViTConfig,
) -> jax.Array:
    values = jnp.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("actions must have shape (batch,6)")
    if values.dtype == jnp.bool_ or not jnp.issubdtype(values.dtype, jnp.integer):
        raise TypeError("actions must use an integer dtype")
    return jnp.clip(values.astype(jnp.int32), 0, config.action_count - 1)


def _exact_masks(
    safe: jax.Array,
    config: SU2FourCellSectorViTConfig,
) -> jax.Array:
    channels = local_schur_channels(config.kind)
    channel_q = jnp.asarray([q for q, _a in channels], dtype=jnp.int32)
    width = config.action_count
    action_axis = jnp.arange(width, dtype=jnp.int32)
    channel_static = action_axis < config.n_channels
    q0 = channel_q[jnp.minimum(safe[:, 0], config.n_channels - 1)]
    q1 = channel_q[jnp.minimum(safe[:, 1], config.n_channels - 1)]
    candidate = action_axis[None, :]
    left_mask = (
        (candidate >= jnp.abs(q0[:, None] - q1[:, None]))
        & (candidate <= q0[:, None] + q1[:, None])
        & (((candidate - q0[:, None] - q1[:, None]) % 2) == 0)
    )
    right_values = {
        q
        for q2, _a2 in channels
        for q3, _a3 in channels
        for q in coupling_range(q2, q3)
    }
    left_completable = np.asarray(
        [
            any(
                config.target_q in coupling_range(left_q, right_q)
                for right_q in right_values
            )
            for left_q in range(width)
        ],
        dtype=np.bool_,
    )
    left_mask &= jnp.asarray(left_completable)[None, :]
    cell2_table = np.zeros((width, config.n_channels), dtype=np.bool_)
    cell3_table = np.zeros(
        (width, config.n_channels, config.n_channels), dtype=np.bool_
    )
    right_table = np.zeros(
        (width, config.n_channels, config.n_channels, width), dtype=np.bool_
    )
    for left_q in range(width):
        for channel2, (q2, _a2) in enumerate(channels):
            for channel3, (q3, _a3) in enumerate(channels):
                for right_q in coupling_range(q2, q3):
                    allowed = config.target_q in coupling_range(left_q, right_q)
                    if allowed:
                        cell2_table[left_q, channel2] = True
                        cell3_table[left_q, channel2, channel3] = True
                        right_table[left_q, channel2, channel3, right_q] = True
    left_q = jnp.clip(safe[:, 2], 0, width - 1)
    cell2 = jnp.minimum(safe[:, 3], config.n_channels - 1)
    cell3 = jnp.minimum(safe[:, 4], config.n_channels - 1)
    cell2_mask = jnp.asarray(cell2_table)[left_q]
    cell3_mask = jnp.asarray(cell3_table)[left_q, cell2]
    right_mask = jnp.asarray(right_table)[left_q, cell2, cell3]
    padding = ((0, 0), (0, width - config.n_channels))
    cell2_mask = jnp.pad(cell2_mask, padding, constant_values=False)
    cell3_mask = jnp.pad(cell3_mask, padding, constant_values=False)
    static = jnp.broadcast_to(channel_static, (safe.shape[0], width))
    return jnp.stack(
        (static, static, left_mask, cell2_mask, cell3_mask, right_mask), axis=1
    )


def _pair_features(config: SU2FourCellSectorViTConfig) -> jax.Array:
    coordinates = np.asarray(config.decision_coordinates, dtype=np.float64)
    delta = coordinates[None, :, :] - coordinates[:, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = max(float(np.max(distance, initial=0.0)), 1.0)
    return jnp.asarray(np.concatenate((delta / scale, distance / scale), axis=-1))


def su2_four_cell_sector_vit_tokens(
    params: dict[str, object],
    actions: jax.Array,
    config: SU2FourCellSectorViTConfig,
) -> jax.Array:
    """Return causal contextual tokens for one strict sector head."""

    safe = _safe_actions(actions, config)
    batch = safe.shape[0]
    previous = jnp.zeros((batch, 6, config.embedding_dim), dtype=jnp.float64)
    previous = previous.at[:, 0].set(params["bos"])
    previous = previous.at[:, 1].set(
        params["channel_embedding"][jnp.minimum(safe[:, 0], config.n_channels - 1)]
    )
    previous = previous.at[:, 2].set(
        params["channel_embedding"][jnp.minimum(safe[:, 1], config.n_channels - 1)]
    )
    previous = previous.at[:, 3].set(params["q_embedding"][safe[:, 2]])
    previous = previous.at[:, 4].set(
        params["channel_embedding"][jnp.minimum(safe[:, 3], config.n_channels - 1)]
    )
    previous = previous.at[:, 5].set(
        params["channel_embedding"][jnp.minimum(safe[:, 4], config.n_channels - 1)]
    )
    coordinates = jnp.asarray(config.decision_coordinates, dtype=jnp.float64)
    tokens = (
        previous
        + coordinates @ params["position_projection"]
        + params["decision_embedding"]
        + params["sector_embedding"][config.sector_index]
    )
    pair = _pair_features(config)
    causal = jnp.tril(jnp.ones((6, 6), dtype=jnp.bool_))
    for layer in params["layers"]:
        normalized = _layer_norm(tokens)
        query = (normalized @ layer["query"]).reshape(
            batch, 6, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            batch, 6, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            batch, 6, config.head_count, config.head_dim
        )
        logits = jnp.einsum("bihd,bjhd->bhij", query, key) / np.sqrt(
            config.head_dim
        )
        logits += jnp.einsum("ijf,fh->hij", pair, layer["pair_weight"])[None, ...]
        logits = jnp.where(causal[None, None, :, :], logits, -jnp.inf)
        attention = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhij,bjhd->bihd", attention, value).reshape(
            batch, 6, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer[
            "mlp_out"
        ]
    return _layer_norm(tokens)


def su2_four_cell_sector_log_amplitudes(
    params: dict[str, object],
    actions: jax.Array,
    config: SU2FourCellSectorViTConfig,
) -> jax.Array:
    """Return exactly normalized autoregressive coefficients."""

    raw = jnp.asarray(actions)
    safe = _safe_actions(raw, config)
    tokens = su2_four_cell_sector_vit_tokens(params, safe, config)
    masks = _exact_masks(safe, config)
    logits = tokens @ params["logit_head"] + params["logit_bias"]
    log_probabilities = jax.nn.log_softmax(
        jnp.where(masks, logits, -jnp.inf), axis=-1
    )
    selected_log_p = jnp.take_along_axis(
        log_probabilities, safe[..., None], axis=-1
    )[..., 0]
    phases = float(config.phase_scale) * np.pi * jnp.tanh(
        tokens @ params["phase_head"] + params["phase_bias"]
    )
    selected_phase = jnp.take_along_axis(
        phases, safe[..., None], axis=-1
    )[..., 0]
    in_range = jnp.all((raw >= 0) & (raw < config.action_count), axis=-1)
    selected_allowed = jnp.take_along_axis(
        masks, safe[..., None], axis=-1
    )[..., 0]
    valid = in_range & jnp.all(selected_allowed, axis=-1)
    result = 0.5 * jnp.sum(selected_log_p, axis=-1) + 1.0j * jnp.sum(
        selected_phase, axis=-1
    )
    return jnp.where(valid, result, -jnp.inf + 0.0j)


def sample_su2_four_cell_sector_vit(
    params: dict[str, object],
    key: jax.Array,
    sample_count: int,
    config: SU2FourCellSectorViTConfig,
) -> jax.Array:
    """Draw exact reduced configurations without post-generation repair."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    actions = jnp.zeros((int(sample_count), 6), dtype=jnp.int32)
    for decision, subkey in enumerate(jax.random.split(key, 6)):
        tokens = su2_four_cell_sector_vit_tokens(params, actions, config)
        masks = _exact_masks(actions, config)
        logits = tokens @ params["logit_head"] + params["logit_bias"]
        logits = jnp.where(masks, logits, -jnp.inf)
        selected = jax.random.categorical(
            subkey, logits[:, decision], axis=-1
        ).astype(jnp.int32)
        actions = actions.at[:, decision].set(selected)
    return actions


def exhaustive_four_cell_sector_actions(
    config: SU2FourCellSectorViTConfig,
) -> jax.Array:
    """Return every valid reduced configuration in deterministic order."""

    return jnp.asarray(
        four_cell_sector_action_table(config.kind, int(config.target_q)),
        dtype=jnp.int32,
    )


def su2_four_cell_sector_vit_parameter_count(params: object) -> int:
    """Count real trainable scalars in the shared multi-sector backbone."""

    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


__all__ = [
    "SU2FourCellSectorViTConfig",
    "exhaustive_four_cell_sector_actions",
    "init_su2_four_cell_sector_vit",
    "sample_su2_four_cell_sector_vit",
    "square_sector_config",
    "su2_four_cell_sector_log_amplitudes",
    "su2_four_cell_sector_vit_parameter_count",
    "su2_four_cell_sector_vit_tokens",
    "triangular_sector_config",
]
