"""Exact-SU(2) physical-spin amplitudes on a reduced fusion DAG.

This is the Phase-1a amplitude engine for
:class:`~su2_nqs.lattices.fusion_catalog.ReducedFusionCatalog`.  Every
non-leaf support stores one feature for each registered disjoint merge route.
Linear maps may mix those route features at fixed doubled spin ``q``.  The
mixed features are then used by a later disjoint Clebsch--Gordan merge, so an
expanded term contains every physical spin exactly once.

Phase-1a is exact plumbing, not yet the proposed expressivity result.  The
current parent merge uses a fixed normalized sum over each child's route
axis.  Consequently a dense support mixer is observable only through its
column sums and can be absorbed into sufficiently general route-diagonal
maps.  A prospective energy comparison must add an independently validated
route-pair interaction between two disjoint multi-route children; it must not
claim a win from ``off_diagonal`` below alone.

The unrestricted ``route_diagonal`` and ``off_diagonal`` modes deliberately
do *not* claim finite-space-group equivariance.  The explicitly named
``equivariant_*`` modes add a permutation-commuting tying rule; only those
modes are exact A1 scalars for the catalog group.

``single_route``
    Every composite support has exactly one incoming route.  No route-mixing
    parameters exist, and parameters from :mod:`spin_clebsch_tree` can be
    reused exactly.
``route_diagonal``
    Each route has an independent reduced-feature map, but distinct routes
    of the same support never mix.
``off_diagonal``
    A dense route/reduced-feature map acts at fixed support and fixed ``q``.
``equivariant_route_diagonal`` / ``equivariant_off_diagonal``
    A1-equivariant variants.  Parameters are shared by support size and route
    count.  The off-diagonal map has the permutation-commutant form
    ``A F_r + B sum_(s != r) F_s / sqrt(R-1)``.  Combined with group closure,
    size-shared merge maps, and the invariant root route sum, this gives an
    exact A1 lattice scalar for the registered site-permutation group.

Like ``SpinClebschTree``, this is an unnormalised real amplitude model.  All
angular momenta and magnetic projections are doubled integers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from numbers import Integral
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.fusion_catalog import (
    DisjointMergeRoute,
    FusionSupport,
    ReducedFusionCatalog,
)
from ..su2.quantum_numbers import coupling_range, validate_m, validate_q
from .spin_clebsch_tree import _cg_tensor


SpinDAGRouteMode = Literal[
    "single_route",
    "route_diagonal",
    "off_diagonal",
    "equivariant_route_diagonal",
    "equivariant_off_diagonal",
]


@dataclass(frozen=True)
class SpinReducedFusionDAGConfig:
    """Static exact-spin and route-mixing contract.

    ``maximum_intermediate_q`` is a cutoff in doubled-spin units.  It applies
    only away from the root; the requested root irrep is never silently
    truncated.

    Group closure of ``catalog`` is validated by ``ReducedFusionCatalog``.
    Group closure alone must not be reported as lattice equivariance: only
    the explicitly named ``equivariant_*`` modes tie every trainable map in a
    way that commutes with the catalog action.
    """

    catalog: ReducedFusionCatalog
    target_q: int = 0
    target_m2: int = 0
    feature_dim: int = 4
    maximum_intermediate_q: int | None = None
    initialization_scale: float = 0.2
    route_mode: SpinDAGRouteMode = "off_diagonal"

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, ReducedFusionCatalog):
            raise TypeError("catalog must be a ReducedFusionCatalog")
        target_q = validate_q(self.target_q, name="target_q")
        target_m2 = validate_m(target_q, self.target_m2, name="target_m2")
        n_sites = self.catalog.n_sites
        if target_q > n_sites or (n_sites - target_q) % 2:
            raise ValueError("target_q is unreachable for the catalog leaves")
        if (
            isinstance(self.feature_dim, bool)
            or not isinstance(self.feature_dim, Integral)
            or self.feature_dim < 1
        ):
            raise ValueError("feature_dim must be a positive integer")
        if self.maximum_intermediate_q is not None:
            validate_q(
                self.maximum_intermediate_q,
                name="maximum_intermediate_q",
            )
        if (
            not np.isfinite(self.initialization_scale)
            or self.initialization_scale <= 0.0
        ):
            raise ValueError("initialization_scale must be positive and finite")
        allowed = {
            "single_route",
            "route_diagonal",
            "off_diagonal",
            "equivariant_route_diagonal",
            "equivariant_off_diagonal",
        }
        if self.route_mode not in allowed:
            raise ValueError(
                "route_mode must be single_route, route_diagonal, or "
                "off_diagonal, equivariant_route_diagonal, or "
                "equivariant_off_diagonal"
            )
        incoming = _incoming_routes(self.catalog)
        composite = [
            support for support in self.catalog.supports if len(support) > 1
        ]
        if self.route_mode == "single_route" and any(
            len(incoming[support]) != 1 for support in composite
        ):
            raise ValueError(
                "single_route mode requires exactly one incoming route for "
                "every composite support"
            )
        if self.route_mode in {
            "off_diagonal",
            "equivariant_off_diagonal",
        } and not any(
            len(support) < n_sites and len(incoming[support]) > 1
            for support in composite
        ):
            raise ValueError(
                "an off-diagonal mode requires multiple routes on a proper "
                "sub-support that can enter a later merge"
            )
        # Compile eagerly so a cutoff that deletes an entire required merge
        # fails at the config boundary rather than during a JIT trace.
        spin_reduced_fusion_dag_merge_layouts(self)
        object.__setattr__(self, "target_q", target_q)
        object.__setattr__(self, "target_m2", target_m2)
        object.__setattr__(self, "feature_dim", int(self.feature_dim))

    @property
    def n_sites(self) -> int:
        return self.catalog.n_sites


@dataclass(frozen=True)
class SpinDAGMergeLayout:
    """Packed doubled-spin channels for one oriented catalog route."""

    route: DisjointMergeRoute
    parameter_index: int
    keys: tuple[tuple[int, int, int], ...]

    @property
    def left_size(self) -> int:
        return len(self.route.left)

    @property
    def right_size(self) -> int:
        return len(self.route.right)

    @property
    def size(self) -> int:
        return len(self.route.parent)


@dataclass(frozen=True)
class SpinDAGSupportLayout:
    """Incoming route indices and optional mixer index for one support."""

    support: FusionSupport
    incoming_route_indices: tuple[int, ...]
    mixer_index: int

    @property
    def route_count(self) -> int:
        return max(len(self.incoming_route_indices), 1)


def _reachable_q(size: int, maximum_q: int | None) -> tuple[int, ...]:
    upper = size if maximum_q is None else min(size, int(maximum_q))
    return tuple(range(size % 2, upper + 1, 2))


@lru_cache(maxsize=None)
def _incoming_routes(
    catalog: ReducedFusionCatalog,
) -> dict[FusionSupport, tuple[DisjointMergeRoute, ...]]:
    grouped: dict[FusionSupport, list[DisjointMergeRoute]] = {
        support: [] for support in catalog.supports
    }
    for route in catalog.routes:
        grouped[route.parent].append(route)
    return {support: tuple(routes) for support, routes in grouped.items()}


@lru_cache(maxsize=None)
def spin_reduced_fusion_dag_merge_layouts(
    config: SpinReducedFusionDAGConfig,
) -> tuple[SpinDAGMergeLayout, ...]:
    """Compile deterministic per-route channels and shared merge indices."""

    root = FusionSupport(tuple(range(config.n_sites)))
    # Assign shared parameter groups in a root-recursive postorder.  This is
    # intentionally not the canonical catalog-route order: for a single tree
    # it exactly reproduces SpinClebschTree's DFS postorder even when unequal
    # subtree shapes would otherwise reorder equal-size parameter groups.
    group_indices: dict[tuple[int, int, bool], int] = {}
    route_parameter_indices: dict[DisjointMergeRoute, int] = {}
    visited_supports: set[FusionSupport] = set()
    incoming = _incoming_routes(config.catalog)

    def assign_support(support: FusionSupport) -> None:
        if support in visited_supports:
            return
        for route in incoming[support]:
            assign_support(route.left)
            assign_support(route.right)
            group_key = (
                len(route.left),
                len(route.right),
                route.parent == root,
            )
            if group_key not in group_indices:
                group_indices[group_key] = len(group_indices)
            route_parameter_indices[route] = group_indices[group_key]
        visited_supports.add(support)

    assign_support(root)
    if len(route_parameter_indices) != len(config.catalog.routes):
        raise RuntimeError("catalog traversal did not visit every merge route")
    layouts: list[SpinDAGMergeLayout] = []
    for route in config.catalog.routes:
        left_size = len(route.left)
        right_size = len(route.right)
        is_root = route.parent == root
        keys: list[tuple[int, int, int]] = []
        for q_left in _reachable_q(
            left_size, config.maximum_intermediate_q
        ):
            for q_right in _reachable_q(
                right_size, config.maximum_intermediate_q
            ):
                for q_total in coupling_range(q_left, q_right):
                    if is_root:
                        if q_total != config.target_q:
                            continue
                    elif (
                        config.maximum_intermediate_q is not None
                        and q_total > config.maximum_intermediate_q
                    ):
                        continue
                    keys.append((q_left, q_right, q_total))
        if not keys:
            raise ValueError(
                "angular-momentum cutoff removes required route "
                f"{route!r}"
            )
        layouts.append(
            SpinDAGMergeLayout(
                route=route,
                parameter_index=route_parameter_indices[route],
                keys=tuple(keys),
            )
        )
    return tuple(layouts)


@lru_cache(maxsize=None)
def spin_reduced_fusion_dag_support_layouts(
    config: SpinReducedFusionDAGConfig,
) -> tuple[SpinDAGSupportLayout, ...]:
    """Compile supports in dependency order (strictly increasing size)."""

    route_positions = {
        route: index for index, route in enumerate(config.catalog.routes)
    }
    incoming = _incoming_routes(config.catalog)
    layouts: list[SpinDAGSupportLayout] = []
    mixer_index = 0
    shared_mixer_indices: dict[tuple[int, int], int] = {}
    for support in config.catalog.supports:
        indices = tuple(route_positions[route] for route in incoming[support])
        if len(support) == 1:
            layouts.append(SpinDAGSupportLayout(support, (), -1))
        else:
            if config.route_mode == "single_route":
                current = -1
            elif config.route_mode.startswith("equivariant_"):
                key = (len(support), len(indices))
                if key not in shared_mixer_indices:
                    shared_mixer_indices[key] = mixer_index
                    mixer_index += 1
                current = shared_mixer_indices[key]
            else:
                current = mixer_index
                mixer_index += 1
            layouts.append(SpinDAGSupportLayout(support, indices, current))
    return tuple(layouts)


def _unique_merge_layouts(
    config: SpinReducedFusionDAGConfig,
) -> tuple[SpinDAGMergeLayout, ...]:
    unique: dict[int, SpinDAGMergeLayout] = {}
    for layout in spin_reduced_fusion_dag_merge_layouts(config):
        unique.setdefault(layout.parameter_index, layout)
    return tuple(unique[index] for index in range(len(unique)))


def _unique_mixer_layouts(
    config: SpinReducedFusionDAGConfig,
) -> tuple[SpinDAGSupportLayout, ...]:
    unique: dict[int, SpinDAGSupportLayout] = {}
    for layout in spin_reduced_fusion_dag_support_layouts(config):
        if layout.mixer_index >= 0:
            unique.setdefault(layout.mixer_index, layout)
    return tuple(unique[index] for index in range(len(unique)))


def init_spin_reduced_fusion_dag(
    key: jax.Array, config: SpinReducedFusionDAGConfig
) -> dict[str, object]:
    """Initialise real merge, route-mixing, and root-readout tensors."""

    unique = _unique_merge_layouts(config)
    mixed_supports = _unique_mixer_layouts(config)
    keys = jax.random.split(key, len(unique) + len(mixed_supports) + 1)
    width = config.feature_dim
    scale = float(config.initialization_scale)
    merges: list[jax.Array] = []
    for parameter_index, layout in enumerate(unique):
        left_width = 1 if layout.left_size == 1 else width
        right_width = 1 if layout.right_size == 1 else width
        fan_in = max(left_width * right_width, 1)
        merges.append(
            (scale / np.sqrt(fan_in))
            * jax.random.normal(
                keys[parameter_index],
                (len(layout.keys), width, left_width, right_width),
                dtype=jnp.float64,
            )
        )

    mixers: list[jax.Array] = []
    offset = len(unique)
    for local_index, layout in enumerate(mixed_supports):
        size = len(layout.support)
        count = len(layout.incoming_route_indices)
        noise_key = keys[offset + local_index]
        if config.route_mode == "route_diagonal":
            identity = jnp.broadcast_to(
                jnp.eye(width, dtype=jnp.float64),
                (size + 1, count, width, width),
            )
            noise = jax.random.normal(noise_key, identity.shape, dtype=jnp.float64)
            mixers.append(identity + (0.1 * scale / np.sqrt(width)) * noise)
        elif config.route_mode == "off_diagonal":
            identity = jnp.zeros(
                (size + 1, count, count, width, width), dtype=jnp.float64
            )
            diagonal = jnp.arange(count)
            identity = identity.at[:, diagonal, diagonal].set(
                jnp.eye(width, dtype=jnp.float64)
            )
            noise = jax.random.normal(noise_key, identity.shape, dtype=jnp.float64)
            mixers.append(
                identity
                + (0.1 * scale / np.sqrt(max(count * width, 1))) * noise
            )
        elif config.route_mode == "equivariant_route_diagonal":
            identity = jnp.broadcast_to(
                jnp.eye(width, dtype=jnp.float64),
                (size + 1, width, width),
            )
            noise = jax.random.normal(noise_key, identity.shape, dtype=jnp.float64)
            mixers.append(identity + (0.1 * scale / np.sqrt(width)) * noise)
        else:
            # Axis one stores the self (A) and distinct-route (B) maps.  The
            # latter is omitted for a one-route support, avoiding a trainable
            # parameter whose gradient would be identically zero.
            components = 1 if count == 1 else 2
            mixer = jnp.zeros(
                (size + 1, components, width, width), dtype=jnp.float64
            )
            mixer = mixer.at[:, 0].set(jnp.eye(width, dtype=jnp.float64))
            noise = jax.random.normal(noise_key, mixer.shape, dtype=jnp.float64)
            mixers.append(mixer + (0.1 * scale / np.sqrt(width)) * noise)
    readout = scale * jax.random.normal(
        keys[-1], (width,), dtype=jnp.float64
    )
    params: dict[str, object] = {
        "merges": tuple(merges),
        "readout": readout,
    }
    if config.route_mode != "single_route":
        params["route_mixers"] = tuple(mixers)
    return params


def _validate_parameter_shapes(
    params: Mapping[str, object], config: SpinReducedFusionDAGConfig
) -> None:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    expected_keys = {"merges", "readout"}
    if config.route_mode != "single_route":
        expected_keys.add("route_mixers")
    if set(params) != expected_keys:
        raise ValueError(
            f"expected parameter keys {sorted(expected_keys)}, got "
            f"{sorted(params)}"
        )
    merges = params["merges"]
    if not isinstance(merges, (tuple, list)):
        raise TypeError("params['merges'] must be a tuple or list")
    unique = _unique_merge_layouts(config)
    if len(merges) != len(unique):
        raise ValueError(f"expected {len(unique)} merge tensors")
    width = config.feature_dim
    for index, (value, layout) in enumerate(zip(merges, unique, strict=True)):
        left_width = 1 if layout.left_size == 1 else width
        right_width = 1 if layout.right_size == 1 else width
        expected = (len(layout.keys), width, left_width, right_width)
        if np.shape(value) != expected:
            raise ValueError(
                f"merge tensor {index} has shape {np.shape(value)}, "
                f"expected {expected}"
            )
        if not jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact):
            raise TypeError("merge tensors must have real or complex dtype")
    if np.shape(params["readout"]) != (width,):
        raise ValueError(f"readout must have shape ({width},)")
    if not jnp.issubdtype(jnp.asarray(params["readout"]).dtype, jnp.inexact):
        raise TypeError("readout must have real or complex dtype")

    if config.route_mode == "single_route":
        return
    mixers = params["route_mixers"]
    if not isinstance(mixers, (tuple, list)):
        raise TypeError("params['route_mixers'] must be a tuple or list")
    mixed_supports = _unique_mixer_layouts(config)
    if len(mixers) != len(mixed_supports):
        raise ValueError(f"expected {len(mixed_supports)} route mixers")
    for index, (value, layout) in enumerate(zip(mixers, mixed_supports, strict=True)):
        size = len(layout.support)
        count = len(layout.incoming_route_indices)
        if config.route_mode == "route_diagonal":
            expected = (size + 1, count, width, width)
        elif config.route_mode == "off_diagonal":
            expected = (size + 1, count, count, width, width)
        elif config.route_mode == "equivariant_route_diagonal":
            expected = (size + 1, width, width)
        else:
            components = 1 if count == 1 else 2
            expected = (size + 1, components, width, width)
        if np.shape(value) != expected:
            raise ValueError(
                f"route mixer {index} has shape {np.shape(value)}, "
                f"expected {expected}"
            )
        if not jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact):
            raise TypeError("route mixers must have real or complex dtype")


def _pool_routes(features: jax.Array) -> jax.Array:
    """Apply the fixed normalized linear readout of one support's routes."""

    return jnp.sum(features, axis=0) / jnp.sqrt(
        jnp.asarray(features.shape[0], dtype=jnp.float64)
    )


def spin_reduced_fusion_dag_amplitude(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinReducedFusionDAGConfig,
) -> jax.Array:
    """Evaluate one physical-spin amplitude in fixed ``(S,M)``.

    Route mixing is linear and block diagonal in the doubled total spin
    ``q``.  Magnetic dependence enters only through fixed Condon--Shortley CG
    coefficients.
    """

    _validate_parameter_shapes(params, config)
    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (config.n_sites,):
        raise ValueError(
            f"expected spin shape ({config.n_sites},), got {spins.shape}"
        )
    valid_spins = jnp.all((spins == 0) | (spins == 1))
    safe_spins = jnp.clip(spins, 0, 1)
    width = config.feature_dim
    merge_layouts = spin_reduced_fusion_dag_merge_layouts(config)
    support_values: dict[FusionSupport, tuple[jax.Array, jax.Array]] = {}

    for layout in spin_reduced_fusion_dag_support_layouts(config):
        support = layout.support
        if len(support) == 1:
            m2 = 2 * safe_spins[support.sites[0]] - 1
            value = jnp.zeros((1, 2, 1), dtype=jnp.float64)
            value = value.at[0, 1, 0].set(1.0)
            support_values[support] = (m2, value)
            continue

        route_values: list[jax.Array] = []
        parent_m2: jax.Array | None = None
        for route_index in layout.incoming_route_indices:
            merge = merge_layouts[route_index]
            left_m2, left_routes = support_values[merge.route.left]
            right_m2, right_routes = support_values[merge.route.right]
            left_by_q = _pool_routes(left_routes)
            right_by_q = _pool_routes(right_routes)
            weights = params["merges"][merge.parameter_index]  # type: ignore[index]
            channels = jnp.asarray(merge.keys, dtype=jnp.int32)
            q_left = channels[:, 0]
            q_right = channels[:, 1]
            q_total = channels[:, 2]
            reduced = jnp.einsum(
                "ca,cb,ckab->ck",
                left_by_q[q_left],
                right_by_q[q_right],
                weights,
            )
            cg = jnp.asarray(_cg_tensor(merge.left_size, merge.right_size))
            coefficient = cg[
                q_left,
                left_m2 + merge.left_size,
                q_right,
                right_m2 + merge.right_size,
                q_total,
            ]
            result_dtype = jnp.result_type(reduced, coefficient)
            result = jnp.zeros(
                (merge.size + 1, width), dtype=result_dtype
            )
            result = result.at[q_total].add(coefficient[:, None] * reduced)
            route_values.append(result)
            current_m2 = left_m2 + right_m2
            parent_m2 = current_m2 if parent_m2 is None else parent_m2

        raw = jnp.stack(route_values, axis=0)
        if config.route_mode == "single_route":
            mixed = raw
        elif config.route_mode == "route_diagonal":
            mixer = params["route_mixers"][layout.mixer_index]  # type: ignore[index]
            mixed = jnp.einsum("qrai,rqi->rqa", mixer, raw)
        elif config.route_mode == "off_diagonal":
            mixer = params["route_mixers"][layout.mixer_index]  # type: ignore[index]
            mixed = jnp.einsum("qrsai,sqi->rqa", mixer, raw)
        elif config.route_mode == "equivariant_route_diagonal":
            mixer = params["route_mixers"][layout.mixer_index]  # type: ignore[index]
            mixed = jnp.einsum("qai,rqi->rqa", mixer, raw)
        else:
            mixer = params["route_mixers"][layout.mixer_index]  # type: ignore[index]
            self_part = jnp.einsum("qai,rqi->rqa", mixer[:, 0], raw)
            count = raw.shape[0]
            if count == 1:
                distinct_part = jnp.zeros_like(self_part)
            else:
                distinct_sum = jnp.sum(raw, axis=0, keepdims=True) - raw
                distinct_part = jnp.einsum(
                    "qai,rqi->rqa", mixer[:, 1], distinct_sum
                ) / jnp.sqrt(jnp.asarray(count - 1, dtype=jnp.float64))
            mixed = self_part + distinct_part
        if parent_m2 is None:  # pragma: no cover - catalog validation forbids it
            raise RuntimeError("composite support has no incoming route")
        support_values[support] = (parent_m2, mixed)

    root = FusionSupport(tuple(range(config.n_sites)))
    total_m2, root_routes = support_values[root]
    root_by_q = _pool_routes(root_routes)
    value = jnp.dot(root_by_q[config.target_q], params["readout"])
    return jnp.where(
        valid_spins & (total_m2 == config.target_m2),
        value,
        jnp.asarray(0.0, dtype=value.dtype),
    )


def spin_reduced_fusion_dag_amplitudes(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinReducedFusionDAGConfig,
) -> jax.Array:
    """Vectorise :func:`spin_reduced_fusion_dag_amplitude` over states."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: spin_reduced_fusion_dag_amplitude(params, row, config)
    )(array)


def spin_reduced_fusion_dag_parameter_count(
    params: Mapping[str, object],
) -> int:
    """Count scalar trainable parameters in a reduced-fusion-DAG pytree."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


def collapse_off_diagonal_to_route_diagonal(
    params: Mapping[str, object],
    config: SpinReducedFusionDAGConfig,
) -> tuple[dict[str, object], SpinReducedFusionDAGConfig]:
    """Exactly collapse a Phase-1a dense mixer to a diagonal one.

    This constructive map is the Phase-1a NO-GO certificate.  Because every
    mixed support is consumed only through :func:`_pool_routes`, an
    unrestricted dense mixer is visible through its sum over output routes.
    The returned diagonal model therefore has identical amplitudes on every
    physical configuration, not merely an approximate fit.

    The same statement holds for the permutation-commutant A1 mixer: its
    pooled action is ``A + sqrt(R-1) B`` for ``R > 1``.
    """

    if config.route_mode not in {
        "off_diagonal",
        "equivariant_off_diagonal",
    }:
        raise ValueError("config must use an off-diagonal route mode")
    _validate_parameter_shapes(params, config)
    if config.route_mode == "off_diagonal":
        target_mode: SpinDAGRouteMode = "route_diagonal"
        collapsed_mixers = tuple(
            jnp.sum(jnp.asarray(mixer), axis=1)
            for mixer in params["route_mixers"]  # type: ignore[index]
        )
    else:
        target_mode = "equivariant_route_diagonal"
        collapsed = []
        for mixer, layout in zip(
            params["route_mixers"],  # type: ignore[index]
            _unique_mixer_layouts(config),
            strict=True,
        ):
            array = jnp.asarray(mixer)
            if layout.route_count == 1:
                collapsed.append(array[:, 0])
            else:
                collapsed.append(
                    array[:, 0]
                    + jnp.sqrt(
                        jnp.asarray(layout.route_count - 1, dtype=jnp.float64)
                    )
                    * array[:, 1]
                )
        collapsed_mixers = tuple(collapsed)
    target_config = replace(config, route_mode=target_mode)
    collapsed_params: dict[str, object] = {
        "merges": tuple(params["merges"]),  # type: ignore[arg-type]
        "route_mixers": collapsed_mixers,
        "readout": params["readout"],
    }
    _validate_parameter_shapes(collapsed_params, target_config)
    return collapsed_params, target_config


__all__ = [
    "SpinDAGMergeLayout",
    "SpinDAGRouteMode",
    "SpinDAGSupportLayout",
    "SpinReducedFusionDAGConfig",
    "collapse_off_diagonal_to_route_diagonal",
    "init_spin_reduced_fusion_dag",
    "spin_reduced_fusion_dag_amplitude",
    "spin_reduced_fusion_dag_amplitudes",
    "spin_reduced_fusion_dag_merge_layouts",
    "spin_reduced_fusion_dag_parameter_count",
    "spin_reduced_fusion_dag_support_layouts",
]
