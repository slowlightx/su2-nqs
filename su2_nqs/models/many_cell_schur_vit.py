"""Size-transferable autoregressive Spatial-ViT in SU(2) Schur space.

The model generates every internal doubled-spin label of a physical fusion
tree in root-first order.  For a four-spin-cell layout this is exactly an
autoregressive distribution over local ``(q,a)`` and inter-cell ``q``: the
local multiplicity ``a`` is the deterministic channel index of the two pair
labels inside a cell.  Candidate split scores are functions of normalized
quantum-number features rather than a size-dependent categorical head, so a
single parameter pytree can be evaluated on both 4x4 and 6x6 layouts.

All exact reachability masks are applied before softmax.  The Transformer sees
only reduced labels and support coordinates; magnetic indices remain in fixed
Clebsch--Gordan tensors implicit in the tree basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.many_cell_schur import ManyCellSchurLayout
from ..su2.tree_rotations import leaf_signature
from .compact_tree_gru import split_tables
from .tree_gru import tree_decisions


jax.config.update("jax_enable_x64", True)


_DECISION_FEATURES = 8
_CANDIDATE_FEATURES = 8


@dataclass(frozen=True)
class ManyCellSchurViTConfig:
    """Architecture and one immutable many-cell Schur layout."""

    layout: ManyCellSchurLayout
    target_q: int = 0
    embedding_dim: int = 32
    head_count: int = 4
    depth: int = 2
    mlp_ratio: int = 2
    phase_scale: float = 0.25
    initialization_scale: float = 0.15
    direct_phase_initialization_scale: float = 0.0
    geometry: str = "triangular"
    initial_distance_decay: float = 1.0
    use_distance_envelope: bool = True
    conditional_mode: str = 'legacy'

    def __post_init__(self) -> None:
        if not self.layout.tree.target_is_reachable(self.target_q):
            raise ValueError(f"target q={self.target_q} is unreachable")
        for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if self.geometry not in {"square", "triangular"}:
            raise ValueError("geometry must be 'square' or 'triangular'")
        if not isinstance(self.use_distance_envelope, bool):
            raise TypeError("use_distance_envelope must be boolean")
        if self.conditional_mode not in ('legacy','aligned_joint'):
            raise ValueError('conditional_mode must be legacy or aligned_joint')
        for name in ("phase_scale", "initialization_scale", "initial_distance_decay"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        direct_phase_scale = float(self.direct_phase_initialization_scale)
        if not np.isfinite(direct_phase_scale) or direct_phase_scale < 0.0:
            raise ValueError(
                "direct_phase_initialization_scale must be nonnegative and finite"
            )

    @property
    def n(self) -> int:
        return self.layout.n_sites

    @property
    def decision_count(self) -> int:
        return self.n - 1

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio

    @property
    def action_count(self) -> int:
        return int(split_tables(self.layout.tree)[0].shape[-1])


def _matrix(key: jax.Array, rows: int, columns: int, scale: float) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(key, (rows, columns), dtype=jnp.float64)


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def init_many_cell_schur_vit(
    key: jax.Array,
    config: ManyCellSchurViTConfig,
) -> dict[str, object]:
    """Initialize parameters whose shapes do not depend on system size."""

    keys = iter(jax.random.split(key, 12 + 14 * config.depth))
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
                "raw_distance_decay": jnp.full(
                    (config.head_count,),
                    _inverse_softplus(config.initial_distance_decay),
                    dtype=jnp.float64,
                ),
                "mlp_in": _matrix(next(keys), config.embedding_dim, config.mlp_dim, scale),
                "mlp_out": _matrix(next(keys), config.mlp_dim, config.embedding_dim, scale),
            }
        )
    global_layers = []
    for _ in range(config.depth):
        global_layers.append(
            {
                "query": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "key": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "value": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "output": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "pair_weight": _matrix(next(keys), 3, config.head_count, scale),
                "raw_distance_decay": jnp.full(
                    (config.head_count,),
                    _inverse_softplus(config.initial_distance_decay),
                    dtype=jnp.float64,
                ),
                "mlp_in": _matrix(next(keys), config.embedding_dim, config.mlp_dim, scale),
                "mlp_out": _matrix(next(keys), config.mlp_dim, config.embedding_dim, scale),
            }
        )
    return {
        "decision_projection": _matrix(
            next(keys), _DECISION_FEATURES, config.embedding_dim, scale
        ),
        "previous_projection": _matrix(
            next(keys), _CANDIDATE_FEATURES, config.embedding_dim, scale
        ),
        "bos": scale
        * jax.random.normal(next(keys), (config.embedding_dim,), dtype=jnp.float64),
        "layers": tuple(layers),
        "candidate_projection": _matrix(
            next(keys), _CANDIDATE_FEATURES, config.embedding_dim, scale
        ),
        "logit_context": _matrix(
            next(keys), config.embedding_dim, config.embedding_dim, scale
        ),
        "logit_feature": scale
        * jax.random.normal(next(keys),
            (_CANDIDATE_FEATURES if config.conditional_mode=='legacy' else config.embedding_dim,),dtype=jnp.float64),
        "logit_bias": jnp.asarray(0.0, dtype=jnp.float64),
        "configuration_projection": _matrix(
            next(keys), _CANDIDATE_FEATURES, config.embedding_dim, scale
        ),
        "direct_phase_bilinear": _matrix(
            next(keys),
            _DECISION_FEATURES,
            _CANDIDATE_FEATURES,
            float(config.direct_phase_initialization_scale),
        ),
        "global_layers": tuple(global_layers),
        "global_phase_head": scale
        * jax.random.normal(next(keys), (config.embedding_dim,), dtype=jnp.float64),
        "global_phase_bias": jnp.asarray(0.0, dtype=jnp.float64),
    }


@lru_cache(maxsize=None)
def _decision_features(config: ManyCellSchurViTConfig) -> np.ndarray:
    decisions = tree_decisions(config.layout.tree)
    coordinates = np.asarray(config.layout.decision_coordinates(), dtype=np.float64)
    maximum_depth = max((decision.depth for decision in decisions), default=1)
    rows = []
    for decision, coordinate in zip(decisions, coordinates, strict=True):
        size = decision.left_leaves + decision.right_leaves
        rows.append(
            (
                coordinate[0],
                coordinate[1],
                decision.left_leaves / size,
                decision.right_leaves / size,
                size / config.n,
                decision.depth / max(maximum_depth, 1),
                float(decision.left_index is None),
                float(decision.right_index is None),
            )
        )
    result = np.asarray(rows, dtype=np.float64)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _pair_features(config: ManyCellSchurViTConfig) -> np.ndarray:
    coordinates = np.asarray(config.layout.decision_coordinates(), dtype=np.float64)
    delta = coordinates[None, :, :] - coordinates[:, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = max(float(np.max(distance, initial=0.0)), 1.0)
    result = np.concatenate((delta / scale, distance / scale), axis=-1)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _support_torus_distances(config: ManyCellSchurViTConfig) -> np.ndarray:
    """Minimum site distance between decision supports on the periodic lattice.

    The balanced tree contains both local-cell and hierarchical decisions, so
    a centroid distance is ambiguous for supports that wrap around the torus.
    The minimum physical-site separation is periodic, size independent, and
    exactly zero for nested/overlapping supports.  It therefore supplies a
    stable locality envelope without changing the fusion-tree convention.
    """

    cells = config.layout.physical_cells
    if not cells:
        raise ValueError("the many-cell layout must contain physical cells")
    columns = int(cells[0][2] - cells[0][0])
    if columns < 2 or columns % 2 or config.n % columns:
        raise ValueError("cannot infer an even rectangular torus from the layout")
    rows = config.n // columns
    if rows < 2 or rows % 2:
        raise ValueError("cannot infer an even rectangular torus from the layout")

    site_distances = np.empty((config.n, config.n), dtype=np.float64)
    for first in range(config.n):
        first_row, first_column = divmod(first, columns)
        for second in range(config.n):
            second_row, second_column = divmod(second, columns)
            delta_row = second_row - first_row
            delta_column = second_column - first_column
            candidates = []
            for row_shift in (-1, 0, 1):
                for column_shift in (-1, 0, 1):
                    dy = float(delta_row + row_shift * rows)
                    dx = float(delta_column + column_shift * columns)
                    squared = dy * dy + dx * dx
                    if config.geometry == "triangular":
                        squared += dx * dy
                    candidates.append(max(squared, 0.0))
            site_distances[first, second] = np.sqrt(min(candidates))

    supports = []
    for decision in tree_decisions(config.layout.tree):
        node = config.layout.tree.internal_nodes[decision.node_index]
        supports.append(
            np.asarray(
                [config.layout.site_order[index] for index in leaf_signature(node)],
                dtype=np.int64,
            )
        )
    distances = np.empty((config.decision_count, config.decision_count), dtype=np.float64)
    support_sets = [set(int(value) for value in support) for support in supports]
    for first, left in enumerate(supports):
        for second, right in enumerate(supports):
            if support_sets[first].intersection(support_sets[second]):
                distances[first, second] = 0.0
            else:
                distances[first, second] = float(
                    np.min(site_distances[np.ix_(left, right)])
                )
    distances.setflags(write=False)
    return distances


def many_cell_schur_spatial_distances(config: ManyCellSchurViTConfig) -> np.ndarray:
    """Return a read-only view of the support-distance attention table."""

    return _support_torus_distances(config)


@lru_cache(maxsize=None)
def _candidate_features(config: ManyCellSchurViTConfig) -> np.ndarray:
    left, right, valid = split_tables(config.layout.tree)
    decisions = tree_decisions(config.layout.tree)
    result = np.zeros((*left.shape, _CANDIDATE_FEATURES), dtype=np.float64)
    for order, decision in enumerate(decisions):
        node_size = decision.left_leaves + decision.right_leaves
        for parent_q in range(config.n + 1):
            for action in range(left.shape[-1]):
                if not valid[order, parent_q, action]:
                    continue
                q_left = int(left[order, parent_q, action])
                q_right = int(right[order, parent_q, action])
                left_scale = max(decision.left_leaves, 1)
                right_scale = max(decision.right_leaves, 1)
                result[order, parent_q, action] = (
                    q_left / left_scale,
                    q_right / right_scale,
                    parent_q / max(node_size, 1),
                    q_left * (q_left + 2.0) / (left_scale * (left_scale + 2.0)),
                    q_right * (q_right + 2.0) / (right_scale * (right_scale + 2.0)),
                    abs(q_left - q_right) / max(node_size, 1),
                    (q_left + q_right) / max(node_size, 1),
                    1.0,
                )
    result.setflags(write=False)
    return result


def _layer_norm(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)


@lru_cache(maxsize=None)
def causal_history_geometry(config: ManyCellSchurViTConfig) -> tuple[np.ndarray,np.ndarray]:
    """Query i predicts decision i; key j>0 carries action/support j-1.

    The BOS key has no physical support and receives zero distance/pair bias.
    This changes an inductive bias, not reachability or the CG convention.
    Legacy geometry is returned verbatim for historical checkpoint replay.
    """
    pair=_pair_features(config);distance=_support_torus_distances(config)
    if config.conditional_mode=='legacy':return pair,distance
    aligned_pair=np.zeros_like(pair);aligned_distance=np.zeros_like(distance)
    aligned_pair[:,1:]=pair[:,:-1];aligned_distance[:,1:]=distance[:,:-1]
    aligned_pair.setflags(write=False);aligned_distance.setflags(write=False)
    return aligned_pair,aligned_distance


def _context_tokens(
    params: dict[str, object],
    chosen_features: jax.Array,
    config: ManyCellSchurViTConfig,
) -> jax.Array:
    batch = chosen_features.shape[0]
    shifted = jnp.zeros_like(chosen_features)
    shifted = shifted.at[:, 1:].set(chosen_features[:, :-1])
    static = jnp.asarray(_decision_features(config), dtype=jnp.float64)
    history_static=(static if config.conditional_mode=='legacy' else
        jnp.concatenate((jnp.zeros_like(static[:1]),static[:-1]),axis=0))
    tokens = (
        history_static[None, ...] @ params["decision_projection"]
        + shifted @ params["previous_projection"]
    )
    tokens = tokens.at[:, 0].add(params["bos"])
    pair,distances=map(jnp.asarray,causal_history_geometry(config))
    target=static@params['decision_projection']
    causal = jnp.tril(
        jnp.ones((config.decision_count, config.decision_count), dtype=jnp.bool_)
    )
    for layer in params["layers"]:
        normalized = _layer_norm(tokens)
        query_input=(normalized if config.conditional_mode=='legacy' else normalized+target[None])
        query = (query_input @ layer["query"]).reshape(
            batch,
            config.decision_count,
            config.head_count,
            config.head_dim,
        )
        key = (normalized @ layer["key"]).reshape(
            batch,
            config.decision_count,
            config.head_count,
            config.head_dim,
        )
        value = (normalized @ layer["value"]).reshape(
            batch,
            config.decision_count,
            config.head_count,
            config.head_dim,
        )
        logits = jnp.einsum("bihd,bjhd->bhij", query, key) / np.sqrt(
            config.head_dim
        )
        logits = logits + jnp.einsum("ijf,fh->hij", pair, layer["pair_weight"])[
            None, ...
        ]
        decay = jax.nn.softplus(layer["raw_distance_decay"])
        if not config.use_distance_envelope:
            decay = jnp.zeros_like(decay)
        logits = logits - decay[None, :, None, None] * distances[None, None]
        logits = jnp.where(causal[None, None, :, :], logits, -jnp.inf)
        attention = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhij,bjhd->bihd", attention, value).reshape(
            batch, config.decision_count, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer[
            "mlp_out"
        ]
    return _layer_norm(tokens if config.conditional_mode=='legacy' else tokens+target[None])


def _candidate_scores(
    params: dict[str, object],
    tokens: jax.Array,
    candidate_features: jax.Array,
    config: ManyCellSchurViTConfig,
) -> jax.Array:
    candidate = candidate_features @ params["candidate_projection"]
    context = tokens @ params["logit_context"]
    if config.conditional_mode=='aligned_joint':
        # A nonlinear candidate/context score replaces, rather than stacks
        # onto, the old linear-in-eight-features conditional readout. At the
        # root it is no longer confined to {1,q,q²} contrasts.
        hidden=jnp.tanh(context[:,:,None,:]+candidate)
        return jnp.einsum('bdae,e->bda',hidden,params['logit_feature'])+params['logit_bias']
    logits = jnp.einsum("bde,bdae->bda", context, candidate) / np.sqrt(
        config.embedding_dim
    )
    logits = (
        logits
        + jnp.einsum("bdaf,f->bda", candidate_features, params["logit_feature"])
        + params["logit_bias"]
    )
    return logits


def _global_configuration_phase(
    params: dict[str, object],
    chosen_features: jax.Array,
    config: ManyCellSchurViTConfig,
    shared_tokens: jax.Array | None = None,
) -> jax.Array:
    """Configuration-dependent scalar phase after normalized AR sampling.

    Phase does not alter the Born probability, so it need not obey the causal
    sampling filtration.  A separate bidirectional Spatial-ViT can therefore
    see every already generated reduced label while remaining a strict SU(2)
    scalar.  This is the scalable nodal/correlator channel.
    """

    batch = chosen_features.shape[0]
    static = jnp.asarray(_decision_features(config), dtype=jnp.float64)
    tokens = (
        static[None, ...] @ params["decision_projection"]
        + chosen_features @ params["configuration_projection"]
    )
    if config.conditional_mode=='aligned_joint':
        # The phase force now reaches every causal-backbone layer, not only
        # the shared static projection. Reuse the already computed context
        # in a forward pass; phase may still see the full generated row.
        tokens=tokens+(_context_tokens(params,chosen_features,config)
            if shared_tokens is None else shared_tokens)
    pair = jnp.asarray(_pair_features(config), dtype=jnp.float64)
    distances = jnp.asarray(_support_torus_distances(config), dtype=jnp.float64)
    for layer in params["global_layers"]:
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
        logits = jnp.einsum("bihd,bjhd->bhij", query, key) / np.sqrt(
            config.head_dim
        )
        logits = logits + jnp.einsum("ijf,fh->hij", pair, layer["pair_weight"])[
            None, ...
        ]
        decay = jax.nn.softplus(layer["raw_distance_decay"])
        if not config.use_distance_envelope:
            decay = jnp.zeros_like(decay)
        logits = logits - decay[None, :, None, None] * distances[None, None]
        attention = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhij,bjhd->bihd", attention, value).reshape(
            batch, config.decision_count, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer[
            "mlp_out"
        ]
    pooled = jnp.mean(_layer_norm(tokens), axis=1)
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


def _selected_tree_actions(
    labels: jax.Array,
    config: ManyCellSchurViTConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    left, right, valid = (jnp.asarray(value) for value in split_tables(config.layout.tree))
    candidate = jnp.asarray(_candidate_features(config), dtype=jnp.float64)
    actions = []
    masks = []
    features = []
    state_valid = labels[:, -1] == config.target_q
    batch_indices = jnp.arange(labels.shape[0])
    for decision in tree_decisions(config.layout.tree):
        parent = labels[:, decision.node_index]
        in_range = (parent >= 0) & (parent <= config.n)
        safe_parent = jnp.clip(parent, 0, config.n)
        candidate_left = left[decision.order, safe_parent]
        candidate_right = right[decision.order, safe_parent]
        mask = valid[decision.order, safe_parent]
        desired_left = (
            jnp.ones_like(parent)
            if decision.left_index is None
            else labels[:, decision.left_index]
        )
        desired_right = (
            jnp.ones_like(parent)
            if decision.right_index is None
            else labels[:, decision.right_index]
        )
        matches = (
            mask
            & (candidate_left == desired_left[:, None])
            & (candidate_right == desired_right[:, None])
        )
        action = jnp.argmax(matches, axis=-1).astype(jnp.int32)
        selected = candidate[decision.order, safe_parent, action]
        actions.append(action)
        masks.append(mask)
        features.append(selected)
        state_valid = state_valid & in_range & jnp.any(matches, axis=-1)
    return (
        jnp.stack(actions, axis=1),
        jnp.stack(masks, axis=1),
        jnp.stack(features, axis=1),
        state_valid,
    )


def many_cell_schur_log_amplitudes(
    params: dict[str, object],
    labels: jax.Array,
    config: ManyCellSchurViTConfig,
) -> jax.Array:
    """Return normalized complex log amplitudes for complete label rows."""

    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.ndim != 2 or values.shape[1] != config.n - 1:
        raise ValueError(f"labels must have shape (batch,{config.n - 1})")
    actions, masks, chosen_features, valid = _selected_tree_actions(values, config)
    tokens = _context_tokens(params, chosen_features, config)
    parent = values[:, [decision.node_index for decision in tree_decisions(config.layout.tree)]]
    safe_parent = jnp.clip(parent, 0, config.n)
    all_candidates = jnp.asarray(_candidate_features(config), dtype=jnp.float64)
    order = jnp.arange(config.decision_count)[None, :]
    candidate_features = all_candidates[order, safe_parent]
    logits = _candidate_scores(params, tokens, candidate_features, config)
    log_probabilities = jax.nn.log_softmax(
        jnp.where(masks, logits, -jnp.inf), axis=-1
    )
    selected_log_p = jnp.take_along_axis(
        log_probabilities, actions[..., None], axis=-1
    )[..., 0]
    phase = _global_configuration_phase(params, chosen_features, config,shared_tokens=tokens)
    result = 0.5 * jnp.sum(selected_log_p, axis=-1) + 1.0j * phase
    return jnp.where(valid, result, -jnp.inf + 0.0j)


def many_cell_schur_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: ManyCellSchurViTConfig,
) -> jax.Array:
    """Single-state counterpart of :func:`many_cell_schur_log_amplitudes`."""

    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.shape != (config.n - 1,):
        raise ValueError(f"labels must have shape ({config.n - 1},)")
    return many_cell_schur_log_amplitudes(params, values[None, :], config)[0]


def sample_many_cell_schur_vit(
    params: dict[str, object],
    key: jax.Array,
    sample_count: int,
    config: ManyCellSchurViTConfig,
) -> jax.Array:
    """Sample complete valid tree labels without post-generation repair."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    count = int(sample_count)
    labels = jnp.zeros((count, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    chosen_features = jnp.zeros(
        (count, config.decision_count, _CANDIDATE_FEATURES), dtype=jnp.float64
    )
    left, right, valid = (jnp.asarray(value) for value in split_tables(config.layout.tree))
    features = jnp.asarray(_candidate_features(config), dtype=jnp.float64)
    batch_indices = jnp.arange(count)
    for decision, subkey in zip(
        tree_decisions(config.layout.tree),
        jax.random.split(key, config.decision_count),
        strict=True,
    ):
        tokens = _context_tokens(params, chosen_features, config)
        parent = labels[:, decision.node_index]
        candidate_features = features[decision.order, parent]
        logits = _candidate_scores(
            params,
            tokens[:, decision.order : decision.order + 1],
            candidate_features[:, None, :, :],
            config,
        )
        mask = valid[decision.order, parent]
        logits = jnp.where(mask, logits[:, 0], -jnp.inf)
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


def additive_log_amplitudes(
    carrier_logs: jax.Array,
    residual_logs: jax.Array,
    residual_scale: jax.Array | complex | float,
) -> jax.Array:
    """Stable coherent additive carrier channel in reduced-label space.

    Unlike multiplicative dressing, a nonzero residual can fill a carrier
    node.  The result is intentionally not assumed normalized; VMC sampling
    must use the explicit carrier/residual mixture proposal implemented by the
    caller.
    """

    carrier = jnp.asarray(carrier_logs)
    residual = jnp.asarray(residual_logs)
    if carrier.shape != residual.shape:
        raise ValueError("carrier and residual logs must have identical shapes")
    scale = jnp.asarray(residual_scale)
    if scale.ndim != 0:
        raise ValueError("residual_scale must be scalar")
    log_abs_scale = jnp.where(jnp.abs(scale) > 0.0, jnp.log(jnp.abs(scale)), -jnp.inf)
    shift = jnp.maximum(jnp.real(carrier), jnp.real(residual) + log_abs_scale)
    value = jnp.exp(carrier - shift) + scale * jnp.exp(residual - shift)
    combined = shift + jnp.log(value)
    return jnp.where(jnp.abs(scale) == 0.0, carrier, combined)


def many_cell_schur_vit_parameter_count(params: object) -> int:
    """Count real trainable scalars."""

    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


__all__ = [
    "ManyCellSchurViTConfig",
    "additive_log_amplitudes",
    "init_many_cell_schur_vit",
    "many_cell_schur_log_amplitudes",
    "many_cell_schur_log_psi",
    "many_cell_schur_spatial_distances",
    "many_cell_schur_vit_parameter_count",
    "sample_many_cell_schur_vit",
]
