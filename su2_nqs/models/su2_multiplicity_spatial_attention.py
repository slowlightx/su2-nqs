"""Node-resolved spatial attention acting only on SU(2) multiplicity data.

The attention network reads immutable fusion-support geometry.  Its outputs
modulate reduced ClebschTree maps separately at every internal node; fixed CG
coefficients continue to carry all magnetic indices.  Consequently the state,
every route component, and every parameter tangent remain in the requested
total-spin irrep.

This is the minimal scalable successor to route-only carriers.  It can express
where and at which fusion scale virtual-spin fluctuations should be enhanced,
without applying a nonlinear map to physical spin or magnetic indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.multiplicity_spatial_features import MultiplicitySpatialFeatures
from ..su2.fusion_tree import TreeNode
from .spin_clebsch_tree import (
    spin_cg_tensor,
    spin_clebsch_tree_layouts,
)
from .su2_route_vit import (
    SU2RouteViTConfig,
    _conditioned_backbone,
    su2_route_vit_controls,
)


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SU2MultiplicitySpatialAttentionConfig:
    """Frozen Route-ViT plus a geometry-aware reduced-tensor attention head."""

    base: SU2RouteViTConfig
    features: MultiplicitySpatialFeatures
    embedding_dim: int = 8
    head_count: int = 2
    modulation_scale: float = 0.12
    initialization_scale: float = 0.15

    def __post_init__(self) -> None:
        for name in ("embedding_dim", "head_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if self.features.n_routes != self.base.n_routes:
            raise ValueError("feature route count does not match Route-ViT")
        expected_nodes = len(self.base.spatial.backbone.tree.internal_nodes)
        if self.features.n_nodes != expected_nodes:
            raise ValueError("feature node count does not match the fusion tree")
        node = np.asarray(self.features.node, dtype=np.float64)
        pair = np.asarray(self.features.pair, dtype=np.float64)
        if node.shape != (
            self.base.n_routes,
            expected_nodes,
            len(self.features.node_labels),
        ):
            raise ValueError("node feature tensor has the wrong shape")
        if pair.shape != (
            self.base.n_routes,
            expected_nodes,
            expected_nodes,
            len(self.features.pair_labels),
        ):
            raise ValueError("pair feature tensor has the wrong shape")
        if np.any(~np.isfinite(node)) or np.any(~np.isfinite(pair)):
            raise ValueError("spatial features must be finite")
        for name in ("modulation_scale", "initialization_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def n_sites(self) -> int:
        return self.base.n_sites

    @property
    def n_routes(self) -> int:
        return self.base.n_routes

    @property
    def n_nodes(self) -> int:
        return self.features.n_nodes

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count


def _matrix(key: jax.Array, rows: int, columns: int, scale: float) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(
        key, (rows, columns), dtype=jnp.float64
    )


def _layer_norm(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)


def init_su2_multiplicity_spatial_attention(
    key: jax.Array,
    frozen_params: dict[str, object],
    config: SU2MultiplicitySpatialAttentionConfig,
) -> dict[str, object]:
    """Initialize an exact zero-residual nest of the frozen SU(2) base."""

    keys = iter(jax.random.split(key, 10 + len(frozen_params["backbone"]["merges"])))
    scale = float(config.initialization_scale)
    node_dim = len(config.features.node_labels)
    pair_dim = len(config.features.pair_labels)
    directions = []
    for merge in frozen_params["backbone"]["merges"]:
        directions.append(
            (scale / np.sqrt(max(int(np.prod(merge.shape)), 1)))
            * jax.random.normal(next(keys), merge.shape, dtype=jnp.float64)
        )
    return {
        "input": _matrix(next(keys), node_dim, config.embedding_dim, scale),
        "input_bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
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
        "pair_weight": _matrix(next(keys), pair_dim, config.head_count, scale),
        "control_head": jnp.zeros(
            (config.embedding_dim, 2), dtype=jnp.float64
        ),
        "directions": tuple(directions),
    }


def su2_multiplicity_spatial_tokens(
    params: dict[str, object],
    config: SU2MultiplicitySpatialAttentionConfig,
) -> jax.Array:
    """Return ``(route,node,feature)`` contextual geometry tokens."""

    node = jnp.asarray(config.features.node, dtype=jnp.float64)
    pair = jnp.asarray(config.features.pair, dtype=jnp.float64)
    tokens = node @ params["input"] + params["input_bias"]
    normalized = _layer_norm(tokens)
    query = (normalized @ params["query"]).reshape(
        config.n_routes, config.n_nodes, config.head_count, config.head_dim
    )
    key = (normalized @ params["key"]).reshape(
        config.n_routes, config.n_nodes, config.head_count, config.head_dim
    )
    value = (normalized @ params["value"]).reshape(
        config.n_routes, config.n_nodes, config.head_count, config.head_dim
    )
    logits = jnp.einsum("rnhd,rmhd->rhnm", query, key) / np.sqrt(
        config.head_dim
    )
    logits = logits + jnp.einsum(
        "rnmf,fh->rhnm", pair, params["pair_weight"]
    )
    attention = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("rhnm,rmhd->rnhd", attention, value).reshape(
        config.n_routes, config.n_nodes, config.embedding_dim
    )
    return _layer_norm(tokens + attended @ params["output"])


def su2_multiplicity_spatial_controls(
    params: dict[str, object],
    config: SU2MultiplicitySpatialAttentionConfig,
) -> jax.Array:
    """Return bounded additive/Casimir controls for every route and node."""

    tokens = su2_multiplicity_spatial_tokens(params, config)
    return float(config.modulation_scale) * jnp.tanh(
        tokens @ params["control_head"]
    )


def _node_merges(
    params: dict[str, object],
    frozen_params: dict[str, object],
    base_control: jax.Array,
    node_controls: jax.Array,
    config: SU2MultiplicitySpatialAttentionConfig,
) -> tuple[jax.Array, ...]:
    backbone = _conditioned_backbone(frozen_params, base_control, config.base)
    layouts = spin_clebsch_tree_layouts(config.base.spatial.backbone)
    result = []
    for node_index, layout in enumerate(layouts):
        merge = backbone["merges"][layout.parameter_index]
        direction = params["directions"][layout.parameter_index]
        direction_rms = jnp.sqrt(jnp.mean(direction * direction))
        merge_rms = jnp.sqrt(jnp.mean(merge * merge))
        normalized = direction * merge_rms / jnp.maximum(
            direction_rms, jnp.asarray(1.0e-14, dtype=jnp.float64)
        )
        additive, fugacity = node_controls[node_index]
        candidate = merge + additive * normalized
        q_total = np.asarray([key[2] for key in layout.keys], dtype=np.float64)
        casimir = q_total * (q_total + 2.0) / 4.0
        centered = casimir - np.mean(casimir)
        rms = float(np.sqrt(np.mean(centered**2)))
        if rms > 1.0e-14:
            score = jnp.asarray(centered / rms, dtype=jnp.float64)
            candidate = candidate * jnp.exp(fugacity * score[:, None, None, None])
        result.append(candidate)
    return tuple(result)


def _amplitude_from_node_merges(
    node_merges: tuple[jax.Array, ...],
    readout: jax.Array,
    spins: jax.Array,
    config: SU2MultiplicitySpatialAttentionConfig,
) -> jax.Array:
    """Contract node-resolved reduced maps with immutable CG tensors.

    This implementation intentionally lives in the new correlator module:
    the established shared-map ClebschTree remains a frozen, provenance-sealed
    reference rather than acquiring a second public contraction interface.
    """

    backbone_config = config.base.spatial.backbone
    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (backbone_config.n_sites,):
        raise ValueError(
            f"expected spin shape ({backbone_config.n_sites},), got {spins.shape}"
        )
    valid_spins = jnp.all((spins == 0) | (spins == 1))
    safe_spins = jnp.clip(spins, 0, 1)
    width = int(backbone_config.feature_dim)
    node_values: dict[TreeNode, tuple[jax.Array, jax.Array]] = {}
    for leaf in backbone_config.tree.leaves:
        m2 = 2 * safe_spins[leaf.index] - 1
        by_q = jnp.zeros((2, 1), dtype=jnp.float64).at[1, 0].set(1.0)
        node_values[leaf] = (m2, by_q)

    layouts = spin_clebsch_tree_layouts(backbone_config)
    if len(node_merges) != len(layouts):
        raise ValueError("one reduced merge is required per internal node")
    for node, layout, weights in zip(
        backbone_config.tree.internal_nodes,
        layouts,
        node_merges,
        strict=True,
    ):
        left_m2, left_by_q = node_values[node.left]
        right_m2, right_by_q = node_values[node.right]
        expected = (
            len(layout.keys),
            width,
            1 if layout.left_size == 1 else width,
            1 if layout.right_size == 1 else width,
        )
        if tuple(weights.shape) != expected:
            raise ValueError(
                f"node reduced merge has shape {tuple(weights.shape)}, expected {expected}"
            )
        channels = jnp.asarray(layout.keys, dtype=jnp.int32)
        q_left = channels[:, 0]
        q_right = channels[:, 1]
        q_total = channels[:, 2]
        reduced = jnp.einsum(
            "ca,cb,ckab->ck",
            left_by_q[q_left],
            right_by_q[q_right],
            weights,
        )
        cg = jnp.asarray(spin_cg_tensor(layout.left_size, layout.right_size))
        coefficient = cg[
            q_left,
            left_m2 + layout.left_size,
            q_right,
            right_m2 + layout.right_size,
            q_total,
        ]
        result = jnp.zeros((layout.size + 1, width), dtype=jnp.float64)
        result = result.at[q_total].add(coefficient[:, None] * reduced)
        node_values[node] = (left_m2 + right_m2, result)

    total_m2, root_by_q = node_values[backbone_config.tree.root]
    readout = jnp.asarray(readout, dtype=jnp.float64)
    if readout.shape != (width,):
        raise ValueError(f"readout must have shape ({width},)")
    value = jnp.dot(root_by_q[backbone_config.target_q], readout)
    return jnp.where(
        valid_spins & (total_m2 == backbone_config.target_m2),
        value,
        jnp.asarray(0.0, dtype=value.dtype),
    )


def _amplitudes_from_node_merges(
    node_merges: tuple[jax.Array, ...],
    readout: jax.Array,
    spins: jax.Array,
    config: SU2MultiplicitySpatialAttentionConfig,
) -> jax.Array:
    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    return jax.vmap(
        lambda row: _amplitude_from_node_merges(
            node_merges, readout, row, config
        )
    )(array)


def su2_multiplicity_spatial_component_matrix(
    params: dict[str, object],
    frozen_params: dict[str, object],
    spins: jax.Array,
    config: SU2MultiplicitySpatialAttentionConfig,
) -> jax.Array:
    """Return exact-SU(2) route components with node-resolved reduced maps."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    _, base_controls = su2_route_vit_controls(frozen_params, config.base)
    controls = su2_multiplicity_spatial_controls(params, config)
    orders = jnp.asarray(config.base.spatial.route_orders, dtype=jnp.int32)

    def one_route(base_control, node_control, order):
        merges = _node_merges(
            params, frozen_params, base_control, node_control, config
        )
        return _amplitudes_from_node_merges(
            merges,
            frozen_params["backbone"]["readout"],
            array[:, order],
            config,
        )

    by_route = jax.vmap(one_route)(base_controls, controls, orders)
    return jnp.swapaxes(by_route, 0, 1)


def su2_multiplicity_spatial_attention_amplitudes(
    params: dict[str, object],
    frozen_params: dict[str, object],
    spins: jax.Array,
    config: SU2MultiplicitySpatialAttentionConfig,
) -> jax.Array:
    """Evaluate the coherent exact-SU(2) spatial-attention state."""

    coefficients, _ = su2_route_vit_controls(frozen_params, config.base)
    components = su2_multiplicity_spatial_component_matrix(
        params, frozen_params, spins, config
    )
    return components @ coefficients


def su2_multiplicity_spatial_attention_parameter_count(params: object) -> int:
    """Count trainable real scalars in the spatial-attention head."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


__all__ = [
    "SU2MultiplicitySpatialAttentionConfig",
    "init_su2_multiplicity_spatial_attention",
    "su2_multiplicity_spatial_attention_amplitudes",
    "su2_multiplicity_spatial_attention_parameter_count",
    "su2_multiplicity_spatial_component_matrix",
    "su2_multiplicity_spatial_controls",
    "su2_multiplicity_spatial_tokens",
]
