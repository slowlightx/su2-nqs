"""Normalized many-cell SU(2) Spatial-ViT with local simplex F-moves.

The physical fusion tree groups nine complete four-spin Schur cells into
three three-cell simplexes.  At the decision
``((q0,q1)q12,q2)Q -> (q0,q1)`` the model augments the shared Spatial-ViT
conditional with features obtained by contracting the exact local
``F[q23,q12]`` column.  Consequently both magnitude and phase depend on a
configuration-dependent virtual-intertwiner resonance while every softmax
still receives the exact reachability mask.

Magnetic indices never enter a learned tensor.  The small head may be trained
with the backbone frozen for a bounded physical-versus-scramble gate, or the
entire pytree may be optimized in a later production calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.many_cell_schur import ManyCellSimplexLayout
from ..su2.wigner6j import f_move
from .compact_tree_gru import split_tables
from .many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    _candidate_features,
    _candidate_scores,
    _context_tokens,
    _global_configuration_phase,
    _selected_tree_actions,
    init_many_cell_schur_vit,
)
from .tree_gru import tree_decisions


jax.config.update("jax_enable_x64", True)

_F_FEATURES = 4
_GEOMETRY_FEATURES = 4
_SIMPLEX_FEATURES = _F_FEATURES * _GEOMETRY_FEATURES


@dataclass(frozen=True)
class ManyCellSimplexViTConfig:
    """A complete Schur backbone plus three local virtual intertwiners."""

    simplex_layout: ManyCellSimplexLayout
    embedding_dim: int = 16
    head_count: int = 2
    depth: int = 1
    mlp_ratio: int = 2
    phase_scale: float = 0.25
    initialization_scale: float = 0.15
    direct_phase_initialization_scale: float = 0.0
    initial_distance_decay: float = 1.0
    use_distance_envelope: bool = True
    simplex_head_initialization_scale: float = 0.0
    initial_pair_resonance_bias: float = 0.0
    initial_root_resonance_bias: float = 0.0
    initial_inter_simplex_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.simplex_layout.n_sites != 36:
            raise ValueError("the present local simplex gate is defined on N=36")
        for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        scale = float(self.simplex_head_initialization_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("simplex_head_initialization_scale must be nonnegative")
        bias = float(self.initial_root_resonance_bias)
        if not np.isfinite(bias) or bias < 0.0:
            raise ValueError("initial_root_resonance_bias must be nonnegative")
        pair_bias = float(self.initial_pair_resonance_bias)
        if not np.isfinite(pair_bias) or pair_bias < 0.0:
            raise ValueError("initial_pair_resonance_bias must be nonnegative")
        inter_bias = float(self.initial_inter_simplex_bias)
        if not np.isfinite(inter_bias) or inter_bias < 0.0:
            raise ValueError("initial_inter_simplex_bias must be nonnegative")
        # Reuse all exact validation performed by the backbone config.
        self.base_config

    @property
    def base_config(self) -> ManyCellSchurViTConfig:
        return ManyCellSchurViTConfig(
            self.simplex_layout.schur,
            target_q=0,
            embedding_dim=int(self.embedding_dim),
            head_count=int(self.head_count),
            depth=int(self.depth),
            mlp_ratio=int(self.mlp_ratio),
            phase_scale=float(self.phase_scale),
            initialization_scale=float(self.initialization_scale),
            direct_phase_initialization_scale=float(
                self.direct_phase_initialization_scale
            ),
            geometry="triangular",
            initial_distance_decay=float(self.initial_distance_decay),
            use_distance_envelope=bool(self.use_distance_envelope),
        )

    @property
    def n(self) -> int:
        return self.simplex_layout.n_sites

    @property
    def decision_count(self) -> int:
        return self.n - 1

    @property
    def action_count(self) -> int:
        return self.base_config.action_count


def init_many_cell_simplex_vit(
    key: jax.Array,
    config: ManyCellSimplexViTConfig,
) -> dict[str, object]:
    """Initialize a shared backbone and a size-independent local F head."""

    backbone_key, logit_key, root_key, inter_key, phase_key = jax.random.split(key, 5)
    scale = float(config.simplex_head_initialization_scale)
    root = scale * jax.random.normal(
        root_key, (_SIMPLEX_FEATURES,), dtype=jnp.float64
    )
    root = root.at[0].add(float(config.initial_root_resonance_bias))
    inter = scale * jax.random.normal(inter_key, (_F_FEATURES,), dtype=jnp.float64)
    inter = inter.at[0].add(float(config.initial_inter_simplex_bias))
    logit = scale * jax.random.normal(
        logit_key, (_SIMPLEX_FEATURES,), dtype=jnp.float64
    )
    logit = logit.at[4].add(float(config.initial_pair_resonance_bias))
    return {
        "backbone": init_many_cell_schur_vit(backbone_key, config.base_config),
        "simplex_logit": logit,
        "simplex_root_logit": root,
        "inter_simplex_logit": inter,
        "simplex_phase": scale
        * jax.random.normal(phase_key, (_SIMPLEX_FEATURES,), dtype=jnp.float64),
    }


def init_many_cell_simplex_head(
    key: jax.Array,
    config: ManyCellSimplexViTConfig,
) -> dict[str, jax.Array]:
    """Initialize only the bounded F-move head for frozen-backbone tests."""

    logit_key, root_key, inter_key, phase_key = jax.random.split(key, 4)
    scale = float(config.simplex_head_initialization_scale)
    root = scale * jax.random.normal(
        root_key, (_SIMPLEX_FEATURES,), dtype=jnp.float64
    )
    root = root.at[0].add(float(config.initial_root_resonance_bias))
    inter = scale * jax.random.normal(inter_key, (_F_FEATURES,), dtype=jnp.float64)
    inter = inter.at[0].add(float(config.initial_inter_simplex_bias))
    logit = scale * jax.random.normal(
        logit_key, (_SIMPLEX_FEATURES,), dtype=jnp.float64
    )
    logit = logit.at[4].add(float(config.initial_pair_resonance_bias))
    return {
        "simplex_logit": logit,
        "simplex_root_logit": root,
        "inter_simplex_logit": inter,
        "simplex_phase": scale
        * jax.random.normal(phase_key, (_SIMPLEX_FEATURES,), dtype=jnp.float64),
    }


@lru_cache(maxsize=None)
def _f_move_feature_table() -> np.ndarray:
    """Exact local recoupling features indexed by doubled irreps."""

    # Four physical spins per cell imply q_i <= 4.  Three cells imply
    # q12 <= 8 and Q <= 12.
    table = np.zeros((13, 5, 5, 5, 9, _F_FEATURES), dtype=np.float64)
    for q0 in (0, 2, 4):
        for q1 in (0, 2, 4):
            for q2 in (0, 2, 4):
                for total in range(0, 13, 2):
                    try:
                        matrix, left, right = f_move(q0, q1, q2, total)
                    except ValueError:
                        continue
                    for column, q12 in enumerate(left):
                        probabilities = np.square(matrix[:, column])
                        norm = float(np.sum(probabilities))
                        if norm <= 0.0:
                            continue
                        probabilities /= norm
                        count = len(probabilities)
                        signed_uniform = float(
                            np.sum(matrix[:, column]) / np.sqrt(float(count))
                        )
                        participation = 0.0
                        entropy = 0.0
                        if count > 1:
                            participation = float(
                                (1.0 / np.sum(probabilities**2) - 1.0)
                                / (count - 1.0)
                            )
                            nonzero = probabilities > 0.0
                            entropy = float(
                                -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
                                / np.log(float(count))
                            )
                        mean_channel = float(
                            np.dot(probabilities, np.asarray(right, dtype=np.float64))
                            / 8.0
                        )
                        table[total, q0, q1, q2, q12] = (
                            signed_uniform,
                            participation,
                            entropy,
                            mean_channel,
                        )
    table.setflags(write=False)
    return table


def local_f_move_feature_table() -> np.ndarray:
    """Return the immutable exact feature table for independent tests."""

    return _f_move_feature_table()


@lru_cache(maxsize=None)
def _root_resonance_feature_table() -> np.ndarray:
    """Causal future-marginal of exact F resonance at a simplex root."""

    table = np.zeros((13, 9, 5, _F_FEATURES), dtype=np.float64)
    for total in range(0, 13, 2):
        for q12 in range(0, 9, 2):
            for q2 in (0, 2, 4):
                rows = []
                for q0 in (0, 2, 4):
                    for q1 in (0, 2, 4):
                        try:
                            matrix, left, _ = f_move(q0, q1, q2, total)
                        except ValueError:
                            continue
                        if q12 not in left:
                            continue
                        probabilities = np.square(matrix[:, left.index(q12)])
                        count = len(probabilities)
                        participation = 0.0
                        if count > 1:
                            participation = float(
                                (1.0 / np.sum(probabilities**2) - 1.0)
                                / (count - 1.0)
                            )
                        rows.append((float(count > 1), participation))
                if not rows:
                    continue
                values = np.asarray(rows)
                table[total, q12, q2] = (
                    float(np.max(values[:, 0])),
                    float(np.mean(values[:, 0])),
                    float(np.max(values[:, 1])),
                    float(np.mean(values[:, 1])),
                )
    table.setflags(write=False)
    return table


def _minimum_torus_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    lx: int,
    ly: int,
) -> float:
    lattice_x = np.asarray((float(lx), 0.0))
    lattice_y = np.asarray((0.5 * float(ly), 0.5 * np.sqrt(3.0) * float(ly)))
    delta = second - first
    return min(
        float(np.linalg.norm(delta + sx * lattice_x + sy * lattice_y))
        for sx in (-1, 0, 1)
        for sy in (-1, 0, 1)
    )


@lru_cache(maxsize=None)
def _simplex_geometry_features(config: ManyCellSimplexViTConfig) -> np.ndarray:
    layout = config.simplex_layout
    lx, ly = layout.lattice_shape
    coordinates = np.asarray(layout.schur.site_coordinates, dtype=np.float64)
    cell_centers = np.asarray(
        [np.mean(coordinates[4 * cell : 4 * cell + 4], axis=0) for cell in range(9)]
    )
    rows = []
    for group in layout.cell_groups:
        points = cell_centers[np.asarray(group)]
        distances = np.asarray(
            (
                _minimum_torus_distance(points[0], points[1], lx=lx, ly=ly),
                _minimum_torus_distance(points[1], points[2], lx=lx, ly=ly),
                _minimum_torus_distance(points[2], points[0], lx=lx, ly=ly),
            )
        ) / 2.0
        compactness = float(np.exp(-np.sum((distances - 1.0) ** 2)))
        rows.append((1.0, compactness, float(np.mean(distances)), float(np.std(distances))))
    result = np.asarray(rows, dtype=np.float64)
    result.setflags(write=False)
    return result


def many_cell_simplex_geometry_features(
    config: ManyCellSimplexViTConfig,
) -> np.ndarray:
    """Return per-simplex geometry features used by the matched control."""

    return _simplex_geometry_features(config)


@lru_cache(maxsize=None)
def _simplex_decision_orders(
    config: ManyCellSimplexViTConfig,
) -> tuple[int, ...]:
    by_node = {
        decision.node_index: decision.order
        for decision in tree_decisions(config.base_config.layout.tree)
    }
    return tuple(by_node[index] for index in config.simplex_layout.pair_indices)


@lru_cache(maxsize=None)
def _simplex_root_decision_orders(
    config: ManyCellSimplexViTConfig,
) -> tuple[int, ...]:
    by_node = {
        decision.node_index: decision.order
        for decision in tree_decisions(config.base_config.layout.tree)
    }
    return tuple(by_node[index] for index in config.simplex_layout.root_indices)


@lru_cache(maxsize=None)
def _inter_simplex_decision_orders(
    config: ManyCellSimplexViTConfig,
) -> tuple[int, ...]:
    layout = config.simplex_layout
    excluded = set(layout.pair_indices) | set(layout.root_indices)
    global_nodes = tuple(
        index for index in layout.schur.intercell_indices if index not in excluded
    )
    by_node = {
        decision.node_index: decision.order
        for decision in tree_decisions(config.base_config.layout.tree)
    }
    return tuple(by_node[index] for index in global_nodes)


def _candidate_inter_simplex_features(
    labels: jax.Array,
    decision_order: int,
    config: ManyCellSimplexViTConfig,
) -> jax.Array:
    """Causal virtual-loop support features above the simplex roots."""

    base = config.base_config
    decisions = tree_decisions(base.layout.tree)
    decision = decisions[decision_order]
    left, right, _ = (jnp.asarray(value) for value in split_tables(base.layout.tree))
    parent = jnp.clip(labels[:, decision.node_index], 0, config.n)
    q_left = left[decision_order, parent]
    q_right = right[decision_order, parent]
    maximum = jnp.maximum(q_left, q_right)
    minimum = jnp.minimum(q_left, q_right)
    return jnp.stack(
        (
            (maximum > 0).astype(jnp.float64),
            (minimum > 0).astype(jnp.float64),
            (q_left * (q_left + 2) + q_right * (q_right + 2)) / 224.0,
            minimum / 12.0,
        ),
        axis=-1,
    )


def _candidate_simplex_features(
    labels: jax.Array,
    simplex: int,
    config: ManyCellSimplexViTConfig,
) -> jax.Array:
    """Return exact F/geometry features for every candidate of one simplex."""

    base = config.base_config
    decision_order = _simplex_decision_orders(config)[simplex]
    left, right, _ = (jnp.asarray(value) for value in split_tables(base.layout.tree))
    group = config.simplex_layout.cell_groups[simplex]
    cell_roots = config.simplex_layout.schur.cell_root_indices
    pair_index = config.simplex_layout.pair_indices[simplex]
    root_index = config.simplex_layout.root_indices[simplex]
    parent = jnp.clip(labels[:, pair_index], 0, 8)
    q0 = jnp.clip(left[decision_order, parent], 0, 4)
    q1 = jnp.clip(right[decision_order, parent], 0, 4)
    q2 = jnp.clip(labels[:, cell_roots[group[2]]], 0, 4)
    total = jnp.clip(labels[:, root_index], 0, 12)
    table = jnp.asarray(_f_move_feature_table(), dtype=jnp.float64)
    features = table[
        total[:, None],
        q0,
        q1,
        q2[:, None],
        parent[:, None],
    ]
    geometry = jnp.asarray(_simplex_geometry_features(config)[simplex])
    return (features[..., :, None] * geometry[None, None, None, :]).reshape(
        (labels.shape[0], base.action_count, _SIMPLEX_FEATURES)
    )


def _candidate_root_resonance_features(
    labels: jax.Array,
    simplex: int,
    config: ManyCellSimplexViTConfig,
) -> jax.Array:
    """Return causal exact-F support potential at a simplex-root split."""

    base = config.base_config
    decision_order = _simplex_root_decision_orders(config)[simplex]
    left, right, _ = (jnp.asarray(value) for value in split_tables(base.layout.tree))
    root_index = config.simplex_layout.root_indices[simplex]
    total = jnp.clip(labels[:, root_index], 0, 12)
    q12 = jnp.clip(left[decision_order, total], 0, 8)
    q2 = jnp.clip(right[decision_order, total], 0, 4)
    table = jnp.asarray(_root_resonance_feature_table(), dtype=jnp.float64)
    features = table[total[:, None], q12, q2]
    geometry = jnp.asarray(_simplex_geometry_features(config)[simplex])
    return (features[..., :, None] * geometry[None, None, None, :]).reshape(
        (labels.shape[0], base.action_count, _SIMPLEX_FEATURES)
    )


def _log_amplitudes_core(
    backbone: dict[str, object],
    head: dict[str, jax.Array],
    labels: jax.Array,
    config: ManyCellSimplexViTConfig,
) -> jax.Array:
    base = config.base_config
    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.ndim != 2 or values.shape[1] != config.n - 1:
        raise ValueError(f"labels must have shape (batch,{config.n - 1})")
    actions, masks, chosen_features, valid = _selected_tree_actions(
        values, base
    )
    tokens = _context_tokens(backbone, chosen_features, base)
    decisions = tree_decisions(base.layout.tree)
    parent = values[:, [decision.node_index for decision in decisions]]
    safe_parent = jnp.clip(parent, 0, config.n)
    all_candidates = jnp.asarray(_candidate_features(base), dtype=jnp.float64)
    order = jnp.arange(base.decision_count)[None, :]
    candidate_features = all_candidates[order, safe_parent]
    logits = _candidate_scores(backbone, tokens, candidate_features, base)
    phase = _global_configuration_phase(backbone, chosen_features, base)
    for decision_order in _inter_simplex_decision_orders(config):
        inter_features = _candidate_inter_simplex_features(
            values, decision_order, config
        )
        logits = logits.at[:, decision_order, :].add(
            jnp.einsum(
                "baf,f->ba", inter_features, head["inter_simplex_logit"]
            )
        )
    for simplex, decision_order in enumerate(
        _simplex_root_decision_orders(config)
    ):
        root_features = _candidate_root_resonance_features(values, simplex, config)
        logits = logits.at[:, decision_order, :].add(
            jnp.einsum(
                "baf,f->ba", root_features, head["simplex_root_logit"]
            )
        )
    for simplex, decision_order in enumerate(_simplex_decision_orders(config)):
        exact_features = _candidate_simplex_features(values, simplex, config)
        correction = jnp.einsum(
            "baf,f->ba", exact_features, head["simplex_logit"]
        )
        logits = logits.at[:, decision_order, :].add(correction)
        selected_indices = jnp.broadcast_to(
            actions[:, decision_order, None, None],
            (values.shape[0], 1, _SIMPLEX_FEATURES),
        )
        selected = jnp.take_along_axis(
            exact_features,
            selected_indices,
            axis=1,
        )[:, 0, :]
        phase = phase + jnp.einsum(
            "bf,f->b", selected, head["simplex_phase"]
        ) / np.sqrt(config.simplex_layout.n_simplexes)
    log_probabilities = jax.nn.log_softmax(
        jnp.where(masks, logits, -jnp.inf), axis=-1
    )
    selected_log_p = jnp.take_along_axis(
        log_probabilities, actions[..., None], axis=-1
    )[..., 0]
    result = 0.5 * jnp.sum(selected_log_p, axis=-1) + 1.0j * phase
    return jnp.where(valid, result, -jnp.inf + 0.0j)


def many_cell_simplex_log_amplitudes(
    params: dict[str, object],
    labels: jax.Array,
    config: ManyCellSimplexViTConfig,
) -> jax.Array:
    """Evaluate normalized complex amplitudes of the full trainable model."""

    head = {
        "simplex_logit": params["simplex_logit"],
        "simplex_root_logit": params["simplex_root_logit"],
        "inter_simplex_logit": params["inter_simplex_logit"],
        "simplex_phase": params["simplex_phase"],
    }
    return _log_amplitudes_core(params["backbone"], head, labels, config)


def frozen_backbone_simplex_log_amplitudes(
    head: dict[str, jax.Array],
    labels: jax.Array,
    config: ManyCellSimplexViTConfig,
    backbone: dict[str, object],
) -> jax.Array:
    """Evaluate a bounded local head on one immutable shared backbone."""

    return _log_amplitudes_core(backbone, head, labels, config)


def _sample_core(
    backbone: dict[str, object],
    head: dict[str, jax.Array],
    key: jax.Array,
    sample_count: int,
    config: ManyCellSimplexViTConfig,
) -> jax.Array:
    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    base = config.base_config
    count = int(sample_count)
    labels = jnp.zeros((count, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(0)
    chosen_features = jnp.zeros(
        (count, base.decision_count, 8), dtype=jnp.float64
    )
    left, right, valid = (jnp.asarray(value) for value in split_tables(base.layout.tree))
    features = jnp.asarray(_candidate_features(base), dtype=jnp.float64)
    batch_indices = jnp.arange(count)
    simplex_by_decision = {
        order: simplex
        for simplex, order in enumerate(_simplex_decision_orders(config))
    }
    root_by_decision = {
        order: simplex
        for simplex, order in enumerate(_simplex_root_decision_orders(config))
    }
    inter_decisions = set(_inter_simplex_decision_orders(config))
    for decision, subkey in zip(
        tree_decisions(base.layout.tree),
        jax.random.split(key, base.decision_count),
        strict=True,
    ):
        tokens = _context_tokens(backbone, chosen_features, base)
        parent = labels[:, decision.node_index]
        candidate_features = features[decision.order, parent]
        logits = _candidate_scores(
            backbone,
            tokens[:, decision.order : decision.order + 1],
            candidate_features[:, None, :, :],
            base,
        )[:, 0]
        if decision.order in inter_decisions:
            inter_features = _candidate_inter_simplex_features(
                labels, decision.order, config
            )
            logits = logits + jnp.einsum(
                "baf,f->ba", inter_features, head["inter_simplex_logit"]
            )
        root_simplex = root_by_decision.get(decision.order)
        if root_simplex is not None:
            root_features = _candidate_root_resonance_features(
                labels, root_simplex, config
            )
            logits = logits + jnp.einsum(
                "baf,f->ba", root_features, head["simplex_root_logit"]
            )
        simplex = simplex_by_decision.get(decision.order)
        if simplex is not None:
            exact_features = _candidate_simplex_features(labels, simplex, config)
            logits = logits + jnp.einsum(
                "baf,f->ba", exact_features, head["simplex_logit"]
            )
        mask = valid[decision.order, parent]
        logits = jnp.where(mask, logits, -jnp.inf)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        chosen_left = left[decision.order, parent, action]
        chosen_right = right[decision.order, parent, action]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(chosen_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(chosen_right)
        chosen_features = chosen_features.at[:, decision.order].set(
            candidate_features[batch_indices, action]
        )
    return labels


def sample_many_cell_simplex_vit(
    params: dict[str, object],
    key: jax.Array,
    sample_count: int,
    config: ManyCellSimplexViTConfig,
) -> jax.Array:
    """Sample the full model with exact masks before every softmax."""

    head = {
        "simplex_logit": params["simplex_logit"],
        "simplex_root_logit": params["simplex_root_logit"],
        "inter_simplex_logit": params["inter_simplex_logit"],
        "simplex_phase": params["simplex_phase"],
    }
    return _sample_core(params["backbone"], head, key, sample_count, config)


def sample_frozen_backbone_simplex_vit(
    head: dict[str, jax.Array],
    key: jax.Array,
    sample_count: int,
    config: ManyCellSimplexViTConfig,
    backbone: dict[str, object],
) -> jax.Array:
    """Sample a bounded F-move head on a frozen shared backbone."""

    return _sample_core(backbone, head, key, sample_count, config)


def many_cell_simplex_parameter_count(params: object) -> int:
    """Count real trainable scalars."""

    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


__all__ = [
    "ManyCellSimplexViTConfig",
    "frozen_backbone_simplex_log_amplitudes",
    "init_many_cell_simplex_head",
    "init_many_cell_simplex_vit",
    "local_f_move_feature_table",
    "many_cell_simplex_geometry_features",
    "many_cell_simplex_log_amplitudes",
    "many_cell_simplex_parameter_count",
    "sample_frozen_backbone_simplex_vit",
    "sample_many_cell_simplex_vit",
]
