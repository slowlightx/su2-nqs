"""Cell-sequential SU(2)-scalar Spatial-ViT.

The normalized autoregressive probability is factorized into one physical
``(q,a,Q)`` decision per four-spin cell.  Exact dynamic-program reachability
masks are applied before every softmax.  Causal and bidirectional attention
share a positive learned minimum-image distance envelope whose lattice-unit
scale does not shrink when the torus grows.  The model sees only reduced Schur
labels and cell coordinates, never magnetic indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.many_cell_schur import (
    ManyCellSchurLayout,
    four_spin_multiplicity_channels,
)
from ..su2.sequential_cell_schur import (
    sequential_cell_action_tables,
    sequential_cell_coordinates,
    sequential_cumulative_indices,
)


jax.config.update("jax_enable_x64", True)


_STATIC_FEATURES = 8
_ACTION_FEATURES = 12
_CHANNEL_PAIR_FEATURES = 3


@dataclass(frozen=True)
class SequentialCellSchurViTConfig:
    """Architecture and immutable sequential-cell Schur layout."""

    layout: ManyCellSchurLayout
    target_q: int = 0
    embedding_dim: int = 32
    head_count: int = 4
    depth: int = 2
    mlp_ratio: int = 2
    phase_scale: float = 1.0
    initialization_scale: float = 0.15
    q_conditioned_attention: bool = False
    geometry: str = "triangular"
    initial_distance_decay: float = 1.0
    use_distance_envelope: bool = True

    def __post_init__(self) -> None:
        sequential_cumulative_indices(self.layout)
        if not self.layout.tree.target_is_reachable(self.target_q):
            raise ValueError(f"target q={self.target_q} is unreachable")
        for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if not isinstance(self.q_conditioned_attention, bool):
            raise TypeError("q_conditioned_attention must be boolean")
        if self.geometry not in {"square", "triangular"}:
            raise ValueError("geometry must be 'square' or 'triangular'")
        if not isinstance(self.use_distance_envelope, bool):
            raise TypeError("use_distance_envelope must be boolean")
        for name in ("phase_scale", "initialization_scale", "initial_distance_decay"):
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def n(self) -> int:
        return self.layout.n_sites

    @property
    def decision_count(self) -> int:
        return self.layout.n_cells

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio

    @property
    def action_count(self) -> int:
        return sequential_cell_action_tables(
            self.layout.n_cells, self.target_q
        ).action_count


def _matrix(key: jax.Array, rows: int, columns: int, scale: float) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(key, (rows, columns), dtype=jnp.float64)


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def init_sequential_cell_schur_vit(
    key: jax.Array,
    config: SequentialCellSchurViTConfig,
) -> dict[str, object]:
    """Initialize a parameter pytree whose shapes are independent of size."""

    layer_key_count = 8 if config.q_conditioned_attention else 7
    keys = iter(jax.random.split(key, 12 + 2 * layer_key_count * config.depth))
    scale = float(config.initialization_scale)

    def layers() -> tuple[dict[str, jax.Array], ...]:
        result = []
        for _ in range(config.depth):
            layer = {
                    "query": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                    "key": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                    "value": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                    "output": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                    "pair_weight": _matrix(next(keys), 4, config.head_count, scale),
                    "raw_distance_decay": jnp.full(
                        (config.head_count,),
                        _inverse_softplus(config.initial_distance_decay),
                        dtype=jnp.float64,
                    ),
                    "mlp_in": _matrix(next(keys), config.embedding_dim, config.mlp_dim, scale),
                    "mlp_out": _matrix(next(keys), config.mlp_dim, config.embedding_dim, scale),
                }
            if config.q_conditioned_attention:
                layer["channel_pair_weight"] = _matrix(
                    next(keys), _CHANNEL_PAIR_FEATURES, config.head_count, scale
                )
            result.append(layer)
        return tuple(result)

    causal_layers = layers()
    global_layers = layers()
    return {
        "static_projection": _matrix(next(keys), _STATIC_FEATURES, config.embedding_dim, scale),
        "previous_projection": _matrix(next(keys), _ACTION_FEATURES, config.embedding_dim, scale),
        "bos": scale * jax.random.normal(next(keys), (config.embedding_dim,), dtype=jnp.float64),
        "causal_layers": causal_layers,
        "candidate_projection": _matrix(next(keys), _ACTION_FEATURES, config.embedding_dim, scale),
        "logit_context": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
        "logit_feature": scale * jax.random.normal(next(keys), (_ACTION_FEATURES,), dtype=jnp.float64),
        "logit_bias": jnp.asarray(0.0, dtype=jnp.float64),
        "phase_context": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
        "phase_feature": scale * jax.random.normal(next(keys), (_ACTION_FEATURES,), dtype=jnp.float64),
        "phase_bias": jnp.asarray(0.0, dtype=jnp.float64),
        "global_static_projection": _matrix(next(keys), _STATIC_FEATURES, config.embedding_dim, scale),
        "configuration_projection": _matrix(next(keys), _ACTION_FEATURES, config.embedding_dim, scale),
        "global_layers": global_layers,
        "global_phase_head": scale * jax.random.normal(next(keys), (config.embedding_dim,), dtype=jnp.float64),
        "global_phase_bias": jnp.asarray(0.0, dtype=jnp.float64),
    }


@lru_cache(maxsize=None)
def _static_features(config: SequentialCellSchurViTConfig) -> np.ndarray:
    coordinates = np.asarray(sequential_cell_coordinates(config.layout), dtype=np.float64)
    count = config.decision_count
    rows = []
    for cell, (x, y) in enumerate(coordinates):
        denominator = max(count - 1, 1)
        rows.append(
            (
                x,
                y,
                cell / denominator,
                (count - cell - 1) / denominator,
                float(cell == 0),
                float(cell == count - 1),
                np.sin(2.0 * np.pi * cell / max(count, 1)),
                np.cos(2.0 * np.pi * cell / max(count, 1)),
            )
        )
    result = np.asarray(rows, dtype=np.float64)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _pair_features(config: SequentialCellSchurViTConfig) -> np.ndarray:
    coordinates = np.asarray(sequential_cell_coordinates(config.layout), dtype=np.float64)
    delta = coordinates[None, :, :] - coordinates[:, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = max(float(np.max(distance, initial=0.0)), 1.0)
    count = config.decision_count
    order = np.arange(count, dtype=np.float64)
    order_delta = (order[None, :] - order[:, None]) / max(count - 1, 1)
    result = np.concatenate((delta / scale, distance / scale, order_delta[..., None]), axis=-1)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _torus_cell_distances(config: SequentialCellSchurViTConfig) -> np.ndarray:
    """Return minimum-image cell distances in size-independent lattice units.

    ``rectangular_sequential_four_spin_layout`` orders complete ``2x2`` cells
    row by row.  Distances are measured on that cell lattice, so nearest-cell
    separation stays one when the physical torus grows.  This is deliberately
    distinct from the centered plotting coordinates used by
    :func:`_pair_features`, whose span normalization would make a local bond
    shrink as ``1/L`` and spoil a scaling comparison.
    """

    cells = config.layout.physical_cells
    if not cells:
        raise ValueError("the sequential layout must contain physical cells")
    first = cells[0]
    columns = int(first[2] - first[0])
    if columns < 2 or columns % 2 or config.n % columns:
        raise ValueError("cannot infer an even rectangular torus from the layout")
    rows = config.n // columns
    if rows < 2 or rows % 2:
        raise ValueError("cannot infer an even rectangular torus from the layout")
    cell_rows = rows // 2
    cell_columns = columns // 2
    expected = tuple(
        (
            (2 * row) * columns + 2 * column,
            (2 * row) * columns + 2 * column + 1,
            (2 * row + 1) * columns + 2 * column,
            (2 * row + 1) * columns + 2 * column + 1,
        )
        for row in range(cell_rows)
        for column in range(cell_columns)
    )
    if tuple(cells) != expected:
        raise ValueError(
            "spatial envelope requires the canonical rectangular four-spin tiling"
        )
    coordinates = np.asarray(
        [
            (row, column)
            for row in range(cell_rows)
            for column in range(cell_columns)
        ],
        dtype=np.float64,
    )
    distances = np.empty((config.decision_count, config.decision_count), dtype=np.float64)
    for source in range(config.decision_count):
        for target in range(config.decision_count):
            delta = coordinates[target] - coordinates[source]
            candidates = []
            for row_shift in (-1, 0, 1):
                for column_shift in (-1, 0, 1):
                    dy = float(delta[0] + row_shift * cell_rows)
                    dx = float(delta[1] + column_shift * cell_columns)
                    squared = dy * dy + dx * dx
                    if config.geometry == "triangular":
                        squared += dx * dy
                    candidates.append(max(squared, 0.0))
            distances[source, target] = np.sqrt(min(candidates))
    distances.setflags(write=False)
    return distances


def sequential_cell_schur_spatial_distances(
    config: SequentialCellSchurViTConfig,
) -> np.ndarray:
    """Return a read-only view of the minimum-image cell-distance table."""

    return _torus_cell_distances(config)


@lru_cache(maxsize=None)
def _candidate_features(config: SequentialCellSchurViTConfig) -> np.ndarray:
    tables = sequential_cell_action_tables(config.decision_count, config.target_q)
    shape = (*tables.valid.shape, _ACTION_FEATURES)
    result = np.zeros(shape, dtype=np.float64)
    for cell in range(config.decision_count):
        support = 4.0 * (cell + 1)
        previous_support = max(4.0 * cell, 1.0)
        for previous_q in range(tables.valid.shape[1]):
            for action in np.flatnonzero(tables.valid[cell, previous_q]):
                q = int(tables.cell_q[cell, previous_q, action])
                a = int(tables.cell_a[cell, previous_q, action])
                total = int(tables.cumulative_q[cell, previous_q, action])
                result[cell, previous_q, action] = (
                    q / 4.0,
                    a / 2.0,
                    (a * a) / 4.0,
                    previous_q / previous_support,
                    total / support,
                    q * (q + 2.0) / 24.0,
                    total * (total + 2.0) / (support * (support + 2.0)),
                    abs(total - previous_q) / 4.0,
                    float(q == 0),
                    float(q == 2),
                    float(q == 4),
                    1.0,
                )
    result.setflags(write=False)
    return result


def _layer_norm(values: jax.Array) -> jax.Array:
    centered = values - jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + 1.0e-8)


def _attention_stack(
    tokens: jax.Array,
    layers: tuple[dict[str, jax.Array], ...],
    config: SequentialCellSchurViTConfig,
    *,
    causal: bool,
    channel_features: jax.Array | None = None,
) -> jax.Array:
    batch = tokens.shape[0]
    pair = jnp.asarray(_pair_features(config), dtype=jnp.float64)
    distances = jnp.asarray(_torus_cell_distances(config), dtype=jnp.float64)
    channel_pair = None
    if config.q_conditioned_attention:
        if channel_features is None:
            raise ValueError("q-conditioned attention requires reduced channel features")
        local_q = channel_features[..., 0]
        cumulative_q = channel_features[..., 4]
        channel_pair = jnp.stack(
            (
                jnp.isclose(local_q[:, :, None], local_q[:, None, :]).astype(jnp.float64),
                jnp.abs(local_q[:, :, None] - local_q[:, None, :]),
                jnp.abs(cumulative_q[:, :, None] - cumulative_q[:, None, :]),
            ),
            axis=-1,
        )
    mask = jnp.tril(jnp.ones((config.decision_count, config.decision_count), dtype=jnp.bool_))
    for layer in layers:
        normalized = _layer_norm(tokens)
        query = (normalized @ layer["query"]).reshape(
            batch, config.decision_count, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            batch, config.decision_count, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            batch, config.decision_count, config.head_count, config.head_dim
        )
        logits = jnp.einsum("bihd,bjhd->bhij", query, key) / np.sqrt(config.head_dim)
        logits = logits + jnp.einsum("ijf,fh->hij", pair, layer["pair_weight"])[None]
        decay = jax.nn.softplus(layer["raw_distance_decay"])
        if not config.use_distance_envelope:
            decay = jnp.zeros_like(decay)
        logits = logits - decay[None, :, None, None] * distances[None, None]
        if channel_pair is not None:
            logits = logits + jnp.einsum(
                "bijf,fh->bhij", channel_pair, layer["channel_pair_weight"]
            )
        if causal:
            logits = jnp.where(mask[None, None], logits, -jnp.inf)
        attention = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhij,bjhd->bihd", attention, value).reshape(
            batch, config.decision_count, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer["mlp_out"]
    return _layer_norm(tokens)


def _context_tokens(params, chosen_features, config):
    shifted = jnp.zeros_like(chosen_features).at[:, 1:].set(chosen_features[:, :-1])
    static = jnp.asarray(_static_features(config), dtype=jnp.float64)
    tokens = static[None] @ params["static_projection"] + shifted @ params["previous_projection"]
    tokens = tokens.at[:, 0].add(params["bos"])
    return _attention_stack(
        tokens,
        params["causal_layers"],
        config,
        causal=True,
        channel_features=shifted,
    )


def _candidate_scores(params, tokens, features, config):
    candidates = features @ params["candidate_projection"]
    context = tokens @ params["logit_context"]
    return (
        jnp.einsum("bde,bdae->bda", context, candidates) / np.sqrt(config.embedding_dim)
        + jnp.einsum("bdaf,f->bda", features, params["logit_feature"])
        + params["logit_bias"]
    )


def _candidate_phases(params, tokens, features, config):
    """Gauge phases attached to selected physical fusion actions."""

    candidates = features @ params["candidate_projection"]
    context = tokens @ params["phase_context"]
    return (
        jnp.einsum("bde,bdae->bda", context, candidates) / np.sqrt(config.embedding_dim)
        + jnp.einsum("bdaf,f->bda", features, params["phase_feature"])
        + params["phase_bias"]
    )


def _global_phase(params, chosen_features, config):
    static = jnp.asarray(_static_features(config), dtype=jnp.float64)
    tokens = static[None] @ params["global_static_projection"] + chosen_features @ params["configuration_projection"]
    tokens = _attention_stack(
        tokens,
        params["global_layers"],
        config,
        causal=False,
        channel_features=chosen_features,
    )
    pooled = jnp.mean(tokens, axis=1)
    return float(config.phase_scale) * np.pi * (
        pooled @ params["global_phase_head"] + params["global_phase_bias"]
    )


def _multiplicity_from_pairs(q, left, right):
    valid = (
        ((q == 0) & (((left == 0) & (right == 0)) | ((left == 2) & (right == 2))))
        | ((q == 2) & (((left == 0) & (right == 2)) | ((left == 2) & (right == 0)) | ((left == 2) & (right == 2))))
        | ((q == 4) & (left == 2) & (right == 2))
    )
    multiplicity = jnp.where(
        q == 0,
        jnp.where((left == 0) & (right == 0), 0, 1),
        jnp.where(q == 2, jnp.where(left == 0, 0, jnp.where(right == 0, 1, 2)), 0),
    )
    return multiplicity.astype(jnp.int32), valid


def _selected_actions(labels: jax.Array, config: SequentialCellSchurViTConfig):
    tables = sequential_cell_action_tables(config.decision_count, config.target_q)
    tq = jnp.asarray(tables.cell_q)
    ta = jnp.asarray(tables.cell_a)
    tt = jnp.asarray(tables.cumulative_q)
    tv = jnp.asarray(tables.valid)
    features = jnp.asarray(_candidate_features(config), dtype=jnp.float64)
    cumulative_indices = sequential_cumulative_indices(config.layout)
    actions = []
    masks = []
    selected = []
    state_valid = labels[:, -1] == config.target_q
    previous = jnp.zeros(labels.shape[0], dtype=jnp.int32)
    batch = jnp.arange(labels.shape[0])
    for cell in range(config.decision_count):
        local_q = labels[:, config.layout.cell_root_indices[cell]]
        left_index, right_index = config.layout.cell_pair_indices[cell]
        local_a, local_valid = _multiplicity_from_pairs(
            local_q, labels[:, left_index], labels[:, right_index]
        )
        total = labels[:, cumulative_indices[cell]]
        safe_previous = jnp.clip(previous, 0, tv.shape[1] - 1)
        mask = tv[cell, safe_previous]
        matches = mask & (tq[cell, safe_previous] == local_q[:, None]) & (ta[cell, safe_previous] == local_a[:, None]) & (tt[cell, safe_previous] == total[:, None])
        action = jnp.argmax(matches, axis=-1).astype(jnp.int32)
        candidate_features = features[cell, safe_previous]
        actions.append(action)
        masks.append(mask)
        selected.append(candidate_features[batch, action])
        state_valid = state_valid & local_valid & jnp.any(matches, axis=-1)
        previous = total
    return (
        jnp.stack(actions, axis=1),
        jnp.stack(masks, axis=1),
        jnp.stack(selected, axis=1),
        state_valid,
    )


def sequential_cell_schur_log_amplitudes(
    params: dict[str, object],
    labels: jax.Array,
    config: SequentialCellSchurViTConfig,
) -> jax.Array:
    """Return exactly normalized complex log amplitudes for label rows."""

    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.ndim != 2 or values.shape[1] != config.n - 1:
        raise ValueError(f"labels must have shape (batch,{config.n - 1})")
    actions, masks, chosen, valid = _selected_actions(values, config)
    tokens = _context_tokens(params, chosen, config)
    tables = sequential_cell_action_tables(config.decision_count, config.target_q)
    # Recover previous-Q table indices from integer labels rather than from
    # the normalized floating-point action features.
    cumulative_indices = sequential_cumulative_indices(config.layout)
    totals = values[:, jnp.asarray(cumulative_indices)]
    previous = jnp.concatenate((jnp.zeros((len(values), 1), dtype=jnp.int32), totals[:, :-1]), axis=1)
    all_features = jnp.asarray(_candidate_features(config), dtype=jnp.float64)
    order = jnp.arange(config.decision_count)[None]
    candidates = all_features[order, previous]
    logits = _candidate_scores(params, tokens, candidates, config)
    log_probabilities = jax.nn.log_softmax(jnp.where(masks, logits, -jnp.inf), axis=-1)
    selected_log_p = jnp.take_along_axis(log_probabilities, actions[..., None], axis=-1)[..., 0]
    candidate_phases = _candidate_phases(params, tokens, candidates, config)
    selected_phases = jnp.take_along_axis(
        candidate_phases, actions[..., None], axis=-1
    )[..., 0]
    conditional_phase = float(config.phase_scale) * np.pi * jnp.sum(
        jnp.tanh(selected_phases), axis=-1
    ) / np.sqrt(config.decision_count)
    phase = conditional_phase + 0.25 * _global_phase(params, chosen, config)
    result = 0.5 * jnp.sum(selected_log_p, axis=-1) + 1.0j * phase
    return jnp.where(valid, result, -jnp.inf + 0.0j)


def sequential_cell_schur_log_psi(params, labels, config):
    """Single-state counterpart of :func:`sequential_cell_schur_log_amplitudes`."""

    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.shape != (config.n - 1,):
        raise ValueError(f"labels must have shape ({config.n - 1},)")
    return sequential_cell_schur_log_amplitudes(params, values[None], config)[0]


def sample_sequential_cell_schur_vit(
    params: dict[str, object],
    key: jax.Array,
    sample_count: int,
    config: SequentialCellSchurViTConfig,
) -> jax.Array:
    """Sample valid complete tree labels with masks applied before softmax."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    count = int(sample_count)
    tables = sequential_cell_action_tables(config.decision_count, config.target_q)
    tq, ta, tt, tv = map(jnp.asarray, (tables.cell_q, tables.cell_a, tables.cumulative_q, tables.valid))
    all_features = jnp.asarray(_candidate_features(config), dtype=jnp.float64)
    labels = jnp.zeros((count, config.n - 1), dtype=jnp.int32)
    chosen = jnp.zeros((count, config.decision_count, _ACTION_FEATURES), dtype=jnp.float64)
    previous = jnp.zeros(count, dtype=jnp.int32)
    batch = jnp.arange(count)
    channels = four_spin_multiplicity_channels()
    cumulative_indices = sequential_cumulative_indices(config.layout)
    for cell, subkey in enumerate(jax.random.split(key, config.decision_count)):
        tokens = _context_tokens(params, chosen, config)
        candidate_features = all_features[cell, previous]
        logits = _candidate_scores(
            params,
            tokens[:, cell : cell + 1],
            candidate_features[:, None],
            config,
        )[:, 0]
        mask = tv[cell, previous]
        action = jax.random.categorical(subkey, jnp.where(mask, logits, -jnp.inf), axis=-1).astype(jnp.int32)
        local_q = tq[cell, previous, action]
        local_a = ta[cell, previous, action]
        total = tt[cell, previous, action]
        left = jnp.zeros(count, dtype=jnp.int32)
        right = jnp.zeros(count, dtype=jnp.int32)
        for q_value, multiplicities in channels.items():
            for a_value, (left_value, right_value) in enumerate(multiplicities):
                chosen_channel = (local_q == q_value) & (local_a == a_value)
                left = jnp.where(chosen_channel, left_value, left)
                right = jnp.where(chosen_channel, right_value, right)
        left_index, right_index = config.layout.cell_pair_indices[cell]
        labels = labels.at[:, left_index].set(left)
        labels = labels.at[:, right_index].set(right)
        labels = labels.at[:, config.layout.cell_root_indices[cell]].set(local_q)
        labels = labels.at[:, cumulative_indices[cell]].set(total)
        chosen = chosen.at[:, cell].set(candidate_features[batch, action])
        previous = total
    return labels


def sequential_cell_schur_vit_parameter_count(params: object) -> int:
    """Count real trainable scalars."""

    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


__all__ = [
    "SequentialCellSchurViTConfig",
    "init_sequential_cell_schur_vit",
    "sample_sequential_cell_schur_vit",
    "sequential_cell_schur_log_amplitudes",
    "sequential_cell_schur_log_psi",
    "sequential_cell_schur_spatial_distances",
    "sequential_cell_schur_vit_parameter_count",
]
