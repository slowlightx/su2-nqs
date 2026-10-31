"""Spin-bit autoregressive controls matched to the Schur Spatial-ViT.

This model deliberately reuses the *same parameter pytree* as
``ManyCellSchurViT``.  It changes only the physical representation and exact
mask: row-major spin bits replace reduced Schur labels, and a fixed-particle-
number reachability mask replaces SU(2) triangle reachability.  Causal Q/K
attention, the bidirectional phase network, the periodic distance envelope,
depth, head count, and hidden dimensions are otherwise identical.

The resulting comparison isolates the value of the non-Abelian reduced
representation much more cleanly than the older factored-attention baseline.
Setting ``target_m2=None`` removes the magnetization mask and supplies the
fully spin-symmetry-unaware arm without changing the backbone or parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
)
from ..su2.many_cell_schur import rectangular_four_spin_layout


jax.config.update("jax_enable_x64", True)


_FEATURE_COUNT = 8


@dataclass(frozen=True)
class FixedSzAutoregressiveSpatialViTConfig:
    """Geometry and matched-backbone contract for a fixed-``Sz`` NQS."""

    geometry: str
    rows: int
    columns: int
    target_m2: int | None = 0
    embedding_dim: int = 32
    head_count: int = 4
    depth: int = 2
    mlp_ratio: int = 2
    phase_scale: float = 0.25
    initialization_scale: float = 0.15
    direct_phase_initialization_scale: float = 0.0
    initial_distance_decay: float = 1.0
    use_distance_envelope: bool = True
    phase_convention: str = "none"

    def __post_init__(self) -> None:
        if self.geometry not in {"square", "triangular"}:
            raise ValueError("geometry must be 'square' or 'triangular'")
        for name in (
            "rows",
            "columns",
            "embedding_dim",
            "head_count",
            "depth",
            "mlp_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.rows % 2 or self.columns % 2:
            raise ValueError("matched 2x2-cell comparisons require even dimensions")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if self.target_m2 is not None:
            if isinstance(self.target_m2, bool) or not isinstance(
                self.target_m2, Integral
            ):
                raise TypeError("target_m2 must be a doubled integer or None")
            if abs(int(self.target_m2)) > self.n_sites or (
                self.n_sites + int(self.target_m2)
            ) % 2:
                raise ValueError("target_m2 is unreachable")
        if not isinstance(self.use_distance_envelope, bool):
            raise TypeError("use_distance_envelope must be boolean")
        if self.phase_convention not in {"none", "marshall_checkerboard"}:
            raise ValueError(
                "phase_convention must be 'none' or 'marshall_checkerboard'"
            )
        for name in (
            "phase_scale",
            "initialization_scale",
            "initial_distance_decay",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        direct_phase_scale = float(self.direct_phase_initialization_scale)
        if not np.isfinite(direct_phase_scale) or direct_phase_scale < 0.0:
            raise ValueError(
                "direct_phase_initialization_scale must be nonnegative and finite"
            )

    @property
    def n_sites(self) -> int:
        return self.rows * self.columns

    @property
    def target_n_up(self) -> int:
        """Reference up count, or half filling when no sector is imposed."""

        if self.target_m2 is None:
            return self.n_sites // 2
        return (self.n_sites + int(self.target_m2)) // 2

    @property
    def fixes_magnetization(self) -> bool:
        return self.target_m2 is not None

    @property
    def decision_count(self) -> int:
        return self.n_sites

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio


def init_fixed_sz_autoregressive_spatial_vit(
    key: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> dict[str, object]:
    """Initialize exactly the same parameter shapes as the SU(2) arm."""

    layout = rectangular_four_spin_layout(
        config.rows,
        config.columns,
        triangular_coordinates=config.geometry == "triangular",
    )
    matched = ManyCellSchurViTConfig(
        layout,
        target_q=0,
        embedding_dim=config.embedding_dim,
        head_count=config.head_count,
        depth=config.depth,
        mlp_ratio=config.mlp_ratio,
        phase_scale=config.phase_scale,
        initialization_scale=config.initialization_scale,
        direct_phase_initialization_scale=(
            config.direct_phase_initialization_scale
        ),
        geometry=config.geometry,
        initial_distance_decay=config.initial_distance_decay,
        use_distance_envelope=config.use_distance_envelope,
    )
    return init_many_cell_schur_vit(key, matched)


@lru_cache(maxsize=None)
def _geometry_tables(
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [
            (row, column)
            for row in range(config.rows)
            for column in range(config.columns)
        ],
        dtype=np.float64,
    )
    decision = np.empty((config.n_sites, _FEATURE_COUNT), dtype=np.float64)
    for order, (row, column) in enumerate(coordinates):
        decision[order] = (
            row / max(config.rows - 1, 1),
            column / max(config.columns - 1, 1),
            (order + 1) / config.n_sites,
            (config.n_sites - order) / config.n_sites,
            float(order == 0),
            float(order + 1 == config.n_sites),
            config.target_n_up / config.n_sites,
            1.0,
        )
    pair = np.empty((config.n_sites, config.n_sites, 3), dtype=np.float64)
    distances = np.empty((config.n_sites, config.n_sites), dtype=np.float64)
    for source, (source_row, source_column) in enumerate(coordinates):
        for target, (target_row, target_column) in enumerate(coordinates):
            raw_row = int(target_row - source_row)
            raw_column = int(target_column - source_column)
            candidates = []
            for row_shift in (-1, 0, 1):
                for column_shift in (-1, 0, 1):
                    dy = float(raw_row + row_shift * config.rows)
                    dx = float(raw_column + column_shift * config.columns)
                    squared = dy * dy + dx * dx
                    if config.geometry == "triangular":
                        squared += dx * dy
                    candidates.append((max(squared, 0.0), dy, dx))
            squared, dy, dx = min(candidates, key=lambda item: item[0])
            distance = np.sqrt(squared)
            pair[source, target] = (
                dy / max(config.rows, 1),
                dx / max(config.columns, 1),
                distance / max(config.rows, config.columns, 1),
            )
            distances[source, target] = distance
    for value in (decision, pair, distances):
        value.setflags(write=False)
    return decision, pair, distances


def _layer_norm(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)


def _candidate_features(
    prefix_up: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return dynamic two-action features and the exact reachability mask."""

    batch = prefix_up.shape[0]
    order = jnp.arange(config.n_sites, dtype=jnp.int32)[None, :, None]
    actions = jnp.asarray((0, 1), dtype=jnp.int32)[None, None, :]
    before = prefix_up[:, :, None]
    after = before + actions
    remaining_sites = config.n_sites - order - 1
    remaining_up = config.target_n_up - after
    if config.fixes_magnetization:
        valid = (remaining_up >= 0) & (remaining_up <= remaining_sites)
    else:
        valid = jnp.ones_like(remaining_up, dtype=jnp.bool_)
    scale = float(config.n_sites)
    features = jnp.stack(
        (
            jnp.broadcast_to(actions, (batch, config.n_sites, 2)),
            1 - jnp.broadcast_to(actions, (batch, config.n_sites, 2)),
            jnp.broadcast_to(before, (batch, config.n_sites, 2)) / scale,
            after / scale,
            remaining_up / scale,
            jnp.broadcast_to(remaining_sites, (batch, config.n_sites, 2)) / scale,
            jnp.broadcast_to(order, (batch, config.n_sites, 2)) / scale,
            jnp.ones((batch, config.n_sites, 2), dtype=jnp.float64),
        ),
        axis=-1,
    ).astype(jnp.float64)
    return features, valid


def _attention_layers(
    tokens: jax.Array,
    layers,
    pair: jax.Array,
    distances: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
    *,
    causal: bool,
) -> jax.Array:
    batch = tokens.shape[0]
    causal_mask = jnp.tril(
        jnp.ones((config.n_sites, config.n_sites), dtype=jnp.bool_)
    )
    for layer in layers:
        normalized = _layer_norm(tokens)
        query = (normalized @ layer["query"]).reshape(
            batch, config.n_sites, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            batch, config.n_sites, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            batch, config.n_sites, config.head_count, config.head_dim
        )
        logits = jnp.einsum("bihd,bjhd->bhij", query, key) / np.sqrt(
            config.head_dim
        )
        logits = logits + jnp.einsum(
            "ijf,fh->hij", pair, layer["pair_weight"]
        )[None]
        decay = jax.nn.softplus(layer["raw_distance_decay"])
        if not config.use_distance_envelope:
            decay = jnp.zeros_like(decay)
        logits = logits - decay[None, :, None, None] * distances[None, None]
        if causal:
            logits = jnp.where(causal_mask[None, None], logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        mixed = jnp.einsum("bhij,bjhd->bihd", weights, value).reshape(
            batch, config.n_sites, config.embedding_dim
        )
        tokens = tokens + mixed @ layer["output"]
        normalized = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer[
            "mlp_out"
        ]
    return _layer_norm(tokens)


def _context_tokens(
    params: dict[str, object],
    chosen_features: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> jax.Array:
    decision, pair, distances = _geometry_tables(config)
    shifted = jnp.zeros_like(chosen_features)
    shifted = shifted.at[:, 1:].set(chosen_features[:, :-1])
    tokens = (
        jnp.asarray(decision)[None] @ params["decision_projection"]
        + shifted @ params["previous_projection"]
    )
    tokens = tokens.at[:, 0].add(params["bos"])
    return _attention_layers(
        tokens,
        params["layers"],
        jnp.asarray(pair),
        jnp.asarray(distances),
        config,
        causal=True,
    )


def _candidate_scores(
    params: dict[str, object],
    tokens: jax.Array,
    features: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> jax.Array:
    candidate = features @ params["candidate_projection"]
    context = tokens @ params["logit_context"]
    return (
        jnp.einsum("bde,bdae->bda", context, candidate)
        / np.sqrt(config.embedding_dim)
        + jnp.einsum("bdaf,f->bda", features, params["logit_feature"])
        + params["logit_bias"]
    )


def _global_phase(
    params: dict[str, object],
    chosen_features: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> jax.Array:
    decision, pair, distances = _geometry_tables(config)
    tokens = (
        jnp.asarray(decision)[None] @ params["decision_projection"]
        + chosen_features @ params["configuration_projection"]
    )
    tokens = _attention_layers(
        tokens,
        params["global_layers"],
        jnp.asarray(pair),
        jnp.asarray(distances),
        config,
        causal=False,
    )
    pooled = jnp.mean(tokens, axis=1)
    static = jnp.asarray(decision, dtype=jnp.float64)
    centered_static = static - jnp.mean(static, axis=0, keepdims=True)
    static_rms = jnp.sqrt(
        jnp.mean(centered_static**2, axis=0, keepdims=True) + 1.0e-12
    )
    standardized_static = centered_static / static_rms
    direct = jnp.einsum(
        "df,fg,bdg->b",
        standardized_static,
        params["direct_phase_bilinear"],
        chosen_features,
    ) / np.sqrt(config.decision_count)
    return float(config.phase_scale) * np.pi * (
        pooled @ params["global_phase_head"]
        + direct
        + params["global_phase_bias"]
    )


def fixed_sz_autoregressive_spatial_log_amplitudes(
    params: dict[str, object],
    configurations: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> jax.Array:
    """Return exactly normalized fixed-sector complex log amplitudes."""

    bits = jnp.asarray(configurations, dtype=jnp.int32)
    if bits.ndim != 2 or bits.shape[1] != config.n_sites:
        raise ValueError(
            f"configurations must have shape (batch,{config.n_sites})"
        )
    safe = jnp.clip(bits, 0, 1)
    prefix = jnp.cumsum(safe, axis=1) - safe
    candidates, mask = _candidate_features(prefix, config)
    chosen = jnp.take_along_axis(
        candidates, safe[:, :, None, None], axis=2
    )[:, :, 0, :]
    tokens = _context_tokens(params, chosen, config)
    logits = _candidate_scores(params, tokens, candidates, config)
    log_probabilities = jax.nn.log_softmax(
        jnp.where(mask, logits, -jnp.inf), axis=-1
    )
    selected = jnp.take_along_axis(
        log_probabilities, safe[:, :, None], axis=-1
    )[:, :, 0]
    valid = jnp.all((bits == 0) | (bits == 1), axis=1)
    if config.fixes_magnetization:
        valid = valid & (jnp.sum(bits, axis=1) == config.target_n_up)
    phase = _global_phase(params, chosen, config)
    if config.phase_convention == "marshall_checkerboard":
        checkerboard = jnp.asarray(
            [
                (row + column) % 2 == 0
                for row in range(config.rows)
                for column in range(config.columns)
            ],
            dtype=jnp.int32,
        )
        phase = phase + np.pi * jnp.sum(safe * checkerboard[None, :], axis=1)
    result = 0.5 * jnp.sum(selected, axis=1) + 1.0j * phase
    return jnp.where(valid, result, -jnp.inf + 0.0j)


def fixed_sz_autoregressive_spatial_log_amplitude(
    params: dict[str, object],
    configuration: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> jax.Array:
    """Single-configuration log amplitude."""

    bits = jnp.asarray(configuration, dtype=jnp.int32)
    if bits.shape != (config.n_sites,):
        raise ValueError(f"configuration must have shape ({config.n_sites},)")
    return fixed_sz_autoregressive_spatial_log_amplitudes(
        params, bits[None], config
    )[0]


def sample_fixed_sz_autoregressive_spatial_vit(
    params: dict[str, object],
    key: jax.Array,
    sample_count: int,
    config: FixedSzAutoregressiveSpatialViTConfig,
) -> jax.Array:
    """Sample fixed-``Sz`` bits with the exact reachability mask at each site."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    count = int(sample_count)
    configurations = jnp.zeros((count, config.n_sites), dtype=jnp.int32)
    chosen = jnp.zeros(
        (count, config.n_sites, _FEATURE_COUNT), dtype=jnp.float64
    )
    prefix_up = jnp.zeros((count,), dtype=jnp.int32)
    batch = jnp.arange(count)
    for order, subkey in enumerate(jax.random.split(key, config.n_sites)):
        prefix_table = jnp.repeat(prefix_up[:, None], config.n_sites, axis=1)
        candidates, masks = _candidate_features(prefix_table, config)
        tokens = _context_tokens(params, chosen, config)
        features = candidates[:, order]
        logits = _candidate_scores(
            params,
            tokens[:, order : order + 1],
            features[:, None],
            config,
        )[:, 0]
        action = jax.random.categorical(
            subkey, jnp.where(masks[:, order], logits, -jnp.inf), axis=-1
        ).astype(jnp.int32)
        configurations = configurations.at[:, order].set(action)
        chosen = chosen.at[:, order].set(features[batch, action])
        prefix_up = prefix_up + action
    return configurations


def fixed_sz_autoregressive_spatial_parameter_count(params: object) -> int:
    """Count real trainable scalars in the shared parameter pytree."""

    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


__all__ = [
    "FixedSzAutoregressiveSpatialViTConfig",
    "fixed_sz_autoregressive_spatial_log_amplitude",
    "fixed_sz_autoregressive_spatial_log_amplitudes",
    "fixed_sz_autoregressive_spatial_parameter_count",
    "init_fixed_sz_autoregressive_spatial_vit",
    "sample_fixed_sz_autoregressive_spatial_vit",
]
