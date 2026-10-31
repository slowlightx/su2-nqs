"""Persistent child-route-pair interactions for exact-SU(2) spin DAGs.

Phase-1a support mixers are followed by a linear route sum and therefore have
an exact diagonal collapse.  This independent Phase-1b engine introduces the
first non-redundant candidate primitive.  At a selected proper support, two
disjoint *multi-route* children are fused with a channel-resolved matrix

``K[channel, left_route, right_route]``.

The resulting feature is subsequently fused again on its path to the root.
A fixed all-ones ``K`` is a rank-one pooled control; a learned full matrix can
have larger route-pair rank.  Fixed CG coefficients still carry every
magnetic index, so both arms have exact total SU(2).

``full_rank`` is the unrestricted, preliminary kernel and does not impose
lattice symmetry.  ``equivariant_factorized_rank_one`` learns
``K=u(left route)v(right route)`` with each marginal decoration tied by its
actual group orbit.  ``equivariant_full_rank`` instead shares every joint
triple ``(parent route, left child route, right child route)`` over its exact
catalog-group orbit.  Together with the size-shared CG merge maps, the two
equivariant modes give internally A1-equivariant amplitudes without assuming
synchronized route permutations on the two children.
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
    route_image,
    support_image,
)
from ..su2.quantum_numbers import validate_m, validate_q
from .reduced_fusion_dag import (
    SpinDAGMergeLayout,
    SpinReducedFusionDAGConfig,
    spin_reduced_fusion_dag_merge_layouts,
    spin_reduced_fusion_dag_support_layouts,
)
from .spin_clebsch_tree import _cg_tensor


SpinRoutePairMode = Literal[
    "rank_one",
    "full_rank",
    "equivariant_factorized_rank_one",
    "equivariant_full_rank",
]


@dataclass(frozen=True)
class SpinPersistentRoutePairConfig:
    """Static contract for a persistent route-pair spin DAG.

    Every support in ``pair_supports`` must be a strict sub-support of the
    root, must be consumed by a later route, and every one of its incoming
    merges must join two children that each have at least two routes.
    """

    catalog: ReducedFusionCatalog
    pair_supports: tuple[FusionSupport, ...]
    target_q: int = 0
    target_m2: int = 0
    feature_dim: int = 4
    maximum_intermediate_q: int | None = None
    initialization_scale: float = 0.2
    pair_mode: SpinRoutePairMode = "full_rank"

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, ReducedFusionCatalog):
            raise TypeError("catalog must be a ReducedFusionCatalog")
        target_q = validate_q(self.target_q, name="target_q")
        target_m2 = validate_m(target_q, self.target_m2, name="target_m2")
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
        if self.pair_mode not in {
            "rank_one",
            "full_rank",
            "equivariant_factorized_rank_one",
            "equivariant_full_rank",
        }:
            raise ValueError(
                "pair_mode must be rank_one, full_rank, "
                "equivariant_factorized_rank_one, or equivariant_full_rank"
            )
        raw_supports = tuple(self.pair_supports)
        if not raw_supports:
            raise ValueError("pair_supports must contain at least one support")
        if any(not isinstance(support, FusionSupport) for support in raw_supports):
            raise TypeError("pair_supports must contain FusionSupport values")
        if len(set(raw_supports)) != len(raw_supports):
            raise ValueError("pair_supports must be duplicate free")
        support_set = set(self.catalog.supports)
        if any(support not in support_set for support in raw_supports):
            raise ValueError("every pair support must be present in the catalog")
        root = FusionSupport(tuple(range(self.catalog.n_sites)))
        if root in raw_supports:
            raise ValueError("route-pair interaction must occur below the root")
        selected = set(raw_supports)
        if self.pair_mode in {
            "equivariant_factorized_rank_one",
            "equivariant_full_rank",
        }:
            for support in raw_supports:
                for image in self.catalog.group:
                    if support_image(support, image) not in selected:
                        raise ValueError(
                            "equivariant pair_supports must be group closed"
                        )

        incoming = _incoming_route_indices(self.catalog)
        later_children = {
            route.left for route in self.catalog.routes
        } | {route.right for route in self.catalog.routes}
        for support in raw_supports:
            if support not in later_children:
                raise ValueError("each pair support must enter a later merge")
            for route_index in incoming[support]:
                route = self.catalog.routes[route_index]
                left_count = max(len(incoming[route.left]), 1)
                right_count = max(len(incoming[route.right]), 1)
                if left_count < 2 or right_count < 2:
                    raise ValueError(
                        "pair supports must merge two multi-route children"
                    )

        # Delegate global spin reachability and packed-channel validation to
        # the already exact Phase-1a compiler.  No route mixers are reused.
        base = _base_config_values(
            self.catalog,
            target_q,
            target_m2,
            int(self.feature_dim),
            self.maximum_intermediate_q,
            float(self.initialization_scale),
        )
        spin_reduced_fusion_dag_merge_layouts(base)
        object.__setattr__(self, "pair_supports", tuple(sorted(raw_supports)))
        object.__setattr__(self, "target_q", target_q)
        object.__setattr__(self, "target_m2", target_m2)
        object.__setattr__(self, "feature_dim", int(self.feature_dim))

    @property
    def n_sites(self) -> int:
        return self.catalog.n_sites


@dataclass(frozen=True)
class SpinRoutePairKernelLayout:
    """One learned route-pair kernel aligned with a catalog merge route."""

    merge_route_index: int
    kernel_index: int
    left_route_count: int
    right_route_count: int


@dataclass(frozen=True)
class SpinDecoratedRoutePair:
    """A parent merge decorated by one route of each child support."""

    parent_route: DisjointMergeRoute
    left_route: DisjointMergeRoute | None
    right_route: DisjointMergeRoute | None

    def __post_init__(self) -> None:
        if self.left_route is None:
            if len(self.parent_route.left) != 1:
                raise ValueError("only a leaf child has an implicit route")
        elif self.left_route.parent != self.parent_route.left:
            raise ValueError("left decorated route belongs to the wrong support")
        if self.right_route is None:
            if len(self.parent_route.right) != 1:
                raise ValueError("only a leaf child has an implicit route")
        elif self.right_route.parent != self.parent_route.right:
            raise ValueError("right decorated route belongs to the wrong support")


@dataclass(frozen=True)
class SpinRoutePairOrbitLayout:
    """One trainable decorated-route orbit and its packed channel count."""

    orbit_index: int
    representative: SpinDecoratedRoutePair
    channel_count: int


@dataclass(frozen=True)
class SpinDecoratedChildRoute:
    """One left or right child-route decoration of a parent merge."""

    parent_route: DisjointMergeRoute
    side: Literal["left", "right"]
    child_route: DisjointMergeRoute | None

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        support = (
            self.parent_route.left
            if self.side == "left"
            else self.parent_route.right
        )
        if self.child_route is None:
            if len(support) != 1:
                raise ValueError("only a leaf child has an implicit route")
        elif self.child_route.parent != support:
            raise ValueError("decorated child route belongs to the wrong support")


@dataclass(frozen=True)
class SpinRouteMarginalOrbitLayout:
    """One trainable marginal-decoration orbit for factorized rank one."""

    orbit_index: int
    representative: SpinDecoratedChildRoute
    channel_count: int


@lru_cache(maxsize=None)
def _base_config_values(
    catalog: ReducedFusionCatalog,
    target_q: int,
    target_m2: int,
    feature_dim: int,
    maximum_intermediate_q: int | None,
    initialization_scale: float,
) -> SpinReducedFusionDAGConfig:
    return SpinReducedFusionDAGConfig(
        catalog=catalog,
        target_q=target_q,
        target_m2=target_m2,
        feature_dim=feature_dim,
        maximum_intermediate_q=maximum_intermediate_q,
        initialization_scale=initialization_scale,
        route_mode="route_diagonal",
    )


def _base_config(
    config: SpinPersistentRoutePairConfig,
) -> SpinReducedFusionDAGConfig:
    return _base_config_values(
        config.catalog,
        config.target_q,
        config.target_m2,
        config.feature_dim,
        config.maximum_intermediate_q,
        float(config.initialization_scale),
    )


@lru_cache(maxsize=None)
def _incoming_route_indices(
    catalog: ReducedFusionCatalog,
) -> dict[FusionSupport, tuple[int, ...]]:
    grouped: dict[FusionSupport, list[int]] = {
        support: [] for support in catalog.supports
    }
    for index, route in enumerate(catalog.routes):
        grouped[route.parent].append(index)
    return {support: tuple(indices) for support, indices in grouped.items()}


def select_group_closed_persistent_pair_supports(
    catalog: ReducedFusionCatalog,
    *,
    minimum_child_routes: int = 2,
) -> tuple[FusionSupport, ...]:
    """Select every valid proper route-pair support in a geometry catalog.

    Eligibility is defined entirely by group-invariant catalog incidence:
    every incoming parent route must join two children with at least
    ``minimum_child_routes`` alternatives, and the resulting support must be
    consumed later below the root.  Hence the returned deterministic set is
    automatically closed under the catalog group; an internal assertion
    guards that statement against future catalog changes.
    """

    if not isinstance(catalog, ReducedFusionCatalog):
        raise TypeError("catalog must be a ReducedFusionCatalog")
    if (
        isinstance(minimum_child_routes, bool)
        or not isinstance(minimum_child_routes, Integral)
        or minimum_child_routes < 2
    ):
        raise ValueError("minimum_child_routes must be an integer at least two")
    threshold = int(minimum_child_routes)
    incoming = _incoming_route_indices(catalog)
    root = FusionSupport(tuple(range(catalog.n_sites)))
    later_children = {
        route.left for route in catalog.routes
    } | {route.right for route in catalog.routes}
    selected = tuple(
        support
        for support in catalog.supports
        if support != root
        and support in later_children
        and incoming[support]
        and all(
            max(len(incoming[catalog.routes[index].left]), 1) >= threshold
            and max(len(incoming[catalog.routes[index].right]), 1) >= threshold
            for index in incoming[support]
        )
    )
    selected_set = set(selected)
    if any(
        support_image(support, image) not in selected_set
        for support in selected
        for image in catalog.group
    ):
        raise RuntimeError("eligible persistent pair supports are not group closed")
    return selected


@lru_cache(maxsize=None)
def spin_route_pair_kernel_layouts(
    config: SpinPersistentRoutePairConfig,
) -> tuple[SpinRoutePairKernelLayout, ...]:
    """Return deterministic layouts for the selected full-rank kernels."""

    incoming = _incoming_route_indices(config.catalog)
    selected = set(config.pair_supports)
    layouts = []
    for merge_index, route in enumerate(config.catalog.routes):
        if route.parent not in selected:
            continue
        layouts.append(
            SpinRoutePairKernelLayout(
                merge_route_index=merge_index,
                kernel_index=len(layouts),
                left_route_count=max(len(incoming[route.left]), 1),
                right_route_count=max(len(incoming[route.right]), 1),
            )
        )
    return tuple(layouts)


def decorated_route_pair_image(
    decorated: SpinDecoratedRoutePair, image: tuple[int, ...]
) -> SpinDecoratedRoutePair:
    """Apply one active catalog site permutation to a decorated route."""

    return SpinDecoratedRoutePair(
        route_image(decorated.parent_route, image),
        None
        if decorated.left_route is None
        else route_image(decorated.left_route, image),
        None
        if decorated.right_route is None
        else route_image(decorated.right_route, image),
    )


def decorated_child_route_image(
    decorated: SpinDecoratedChildRoute, image: tuple[int, ...]
) -> SpinDecoratedChildRoute:
    """Apply one active site permutation to a marginal decoration."""

    return SpinDecoratedChildRoute(
        route_image(decorated.parent_route, image),
        decorated.side,
        None
        if decorated.child_route is None
        else route_image(decorated.child_route, image),
    )


@lru_cache(maxsize=None)
def _selected_decorated_route_pairs(
    config: SpinPersistentRoutePairConfig,
) -> tuple[SpinDecoratedRoutePair, ...]:
    incoming = _incoming_route_indices(config.catalog)
    selected = set(config.pair_supports)
    decorated = []
    for parent_route in config.catalog.routes:
        if parent_route.parent not in selected:
            continue
        left_routes: tuple[DisjointMergeRoute | None, ...]
        right_routes: tuple[DisjointMergeRoute | None, ...]
        left_indices = incoming[parent_route.left]
        right_indices = incoming[parent_route.right]
        left_routes = (
            tuple(config.catalog.routes[index] for index in left_indices)
            if left_indices
            else (None,)
        )
        right_routes = (
            tuple(config.catalog.routes[index] for index in right_indices)
            if right_indices
            else (None,)
        )
        decorated.extend(
            SpinDecoratedRoutePair(parent_route, left_route, right_route)
            for left_route in left_routes
            for right_route in right_routes
        )
    return tuple(decorated)


@lru_cache(maxsize=None)
def _decorated_route_pair_orbit_indices(
    config: SpinPersistentRoutePairConfig,
) -> tuple[int, ...]:
    decorated = _selected_decorated_route_pairs(config)
    positions = {value: index for index, value in enumerate(decorated)}
    labels = [-1] * len(decorated)
    next_label = 0
    for index, value in enumerate(decorated):
        if labels[index] >= 0:
            continue
        orbit = {
            decorated_route_pair_image(value, image)
            for image in config.catalog.group
        }
        try:
            member_positions = tuple(positions[member] for member in orbit)
        except KeyError as error:  # pragma: no cover - guarded at config boundary
            raise ValueError(
                "selected decorated route pairs are not group closed"
            ) from error
        for member_index in member_positions:
            labels[member_index] = next_label
        next_label += 1
    return tuple(labels)


@lru_cache(maxsize=None)
def spin_route_pair_orbit_layouts(
    config: SpinPersistentRoutePairConfig,
) -> tuple[SpinRoutePairOrbitLayout, ...]:
    """Compile deterministic full-group orbits of decorated route triples."""

    decorated = _selected_decorated_route_pairs(config)
    labels = _decorated_route_pair_orbit_indices(config)
    first_by_orbit: dict[int, SpinDecoratedRoutePair] = {}
    for value, label in zip(decorated, labels, strict=True):
        first_by_orbit.setdefault(label, value)
    route_positions = {
        route: index for index, route in enumerate(config.catalog.routes)
    }
    merge_layouts = spin_reduced_fusion_dag_merge_layouts(_base_config(config))
    return tuple(
        SpinRoutePairOrbitLayout(
            orbit_index=index,
            representative=first_by_orbit[index],
            channel_count=len(
                merge_layouts[
                    route_positions[first_by_orbit[index].parent_route]
                ].keys
            ),
        )
        for index in range(len(first_by_orbit))
    )


@lru_cache(maxsize=None)
def _selected_marginal_decorations(
    config: SpinPersistentRoutePairConfig,
    side: Literal["left", "right"],
) -> tuple[SpinDecoratedChildRoute, ...]:
    incoming = _incoming_route_indices(config.catalog)
    selected = set(config.pair_supports)
    values = []
    for parent_route in config.catalog.routes:
        if parent_route.parent not in selected:
            continue
        child_support = (
            parent_route.left if side == "left" else parent_route.right
        )
        indices = incoming[child_support]
        child_routes: tuple[DisjointMergeRoute | None, ...] = (
            tuple(config.catalog.routes[index] for index in indices)
            if indices
            else (None,)
        )
        values.extend(
            SpinDecoratedChildRoute(parent_route, side, child_route)
            for child_route in child_routes
        )
    return tuple(values)


@lru_cache(maxsize=None)
def _marginal_orbit_indices(
    config: SpinPersistentRoutePairConfig,
    side: Literal["left", "right"],
) -> tuple[int, ...]:
    values = _selected_marginal_decorations(config, side)
    positions = {value: index for index, value in enumerate(values)}
    labels = [-1] * len(values)
    next_label = 0
    for index, value in enumerate(values):
        if labels[index] >= 0:
            continue
        orbit = {
            decorated_child_route_image(value, image)
            for image in config.catalog.group
        }
        try:
            member_positions = tuple(positions[member] for member in orbit)
        except KeyError as error:  # pragma: no cover - config group closure
            raise ValueError(
                "selected marginal route decorations are not group closed"
            ) from error
        for member_index in member_positions:
            labels[member_index] = next_label
        next_label += 1
    return tuple(labels)


@lru_cache(maxsize=None)
def spin_route_marginal_orbit_layouts(
    config: SpinPersistentRoutePairConfig,
    side: Literal["left", "right"],
) -> tuple[SpinRouteMarginalOrbitLayout, ...]:
    """Compile actual group orbits for one factorized-kernel side."""

    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    values = _selected_marginal_decorations(config, side)
    labels = _marginal_orbit_indices(config, side)
    first_by_orbit: dict[int, SpinDecoratedChildRoute] = {}
    for value, label in zip(values, labels, strict=True):
        first_by_orbit.setdefault(label, value)
    route_positions = {
        route: index for index, route in enumerate(config.catalog.routes)
    }
    merge_layouts = spin_reduced_fusion_dag_merge_layouts(_base_config(config))
    return tuple(
        SpinRouteMarginalOrbitLayout(
            orbit_index=index,
            representative=first_by_orbit[index],
            channel_count=len(
                merge_layouts[
                    route_positions[first_by_orbit[index].parent_route]
                ].keys
            ),
        )
        for index in range(len(first_by_orbit))
    )


def _unique_merge_layouts(
    config: SpinPersistentRoutePairConfig,
) -> tuple[SpinDAGMergeLayout, ...]:
    unique: dict[int, SpinDAGMergeLayout] = {}
    for layout in spin_reduced_fusion_dag_merge_layouts(_base_config(config)):
        unique.setdefault(layout.parameter_index, layout)
    return tuple(unique[index] for index in range(len(unique)))


def init_spin_persistent_route_pair_dag(
    key: jax.Array,
    config: SpinPersistentRoutePairConfig,
) -> dict[str, object]:
    """Initialise the rank-one control or learned full-rank candidate."""

    unique = _unique_merge_layouts(config)
    pair_layouts = spin_route_pair_kernel_layouts(config)
    if config.pair_mode == "full_rank":
        n_pair_keys = len(pair_layouts)
    elif config.pair_mode == "equivariant_factorized_rank_one":
        n_pair_keys = len(spin_route_marginal_orbit_layouts(config, "left"))
        n_pair_keys += len(spin_route_marginal_orbit_layouts(config, "right"))
    elif config.pair_mode == "equivariant_full_rank":
        n_pair_keys = len(spin_route_pair_orbit_layouts(config))
    else:
        n_pair_keys = 0
    keys = jax.random.split(key, len(unique) + n_pair_keys + 1)
    width = config.feature_dim
    scale = float(config.initialization_scale)
    merges = []
    for index, layout in enumerate(unique):
        left_width = 1 if layout.left_size == 1 else width
        right_width = 1 if layout.right_size == 1 else width
        merges.append(
            (scale / np.sqrt(max(left_width * right_width, 1)))
            * jax.random.normal(
                keys[index],
                (len(layout.keys), width, left_width, right_width),
                dtype=jnp.float64,
            )
        )
    params: dict[str, object] = {"merges": tuple(merges)}
    if config.pair_mode == "full_rank":
        merge_layouts = spin_reduced_fusion_dag_merge_layouts(
            _base_config(config)
        )
        kernels = []
        for offset, pair_layout in enumerate(pair_layouts):
            merge = merge_layouts[pair_layout.merge_route_index]
            left_count = pair_layout.left_route_count
            right_count = pair_layout.right_route_count
            baseline = jnp.full(
                (len(merge.keys), left_count, right_count),
                1.0 / np.sqrt(left_count * right_count),
                dtype=jnp.float64,
            )
            noise = jax.random.normal(
                keys[len(unique) + offset], baseline.shape, dtype=jnp.float64
            )
            kernels.append(baseline + (0.1 * scale) * noise)
        params["pair_kernels"] = tuple(kernels)
    elif config.pair_mode == "equivariant_factorized_rank_one":
        incoming = _incoming_route_indices(config.catalog)
        left_layouts = spin_route_marginal_orbit_layouts(config, "left")
        right_layouts = spin_route_marginal_orbit_layouts(config, "right")
        left_weights = []
        for offset, layout in enumerate(left_layouts):
            parent = layout.representative.parent_route
            count = max(len(incoming[parent.left]), 1)
            baseline = jnp.full(
                (layout.channel_count,),
                1.0 / np.sqrt(count),
                dtype=jnp.float64,
            )
            noise = jax.random.normal(
                keys[len(unique) + offset], baseline.shape, dtype=jnp.float64
            )
            left_weights.append(baseline + (0.1 * scale) * noise)
        right_weights = []
        right_offset = len(unique) + len(left_layouts)
        for offset, layout in enumerate(right_layouts):
            parent = layout.representative.parent_route
            count = max(len(incoming[parent.right]), 1)
            baseline = jnp.full(
                (layout.channel_count,),
                1.0 / np.sqrt(count),
                dtype=jnp.float64,
            )
            noise = jax.random.normal(
                keys[right_offset + offset], baseline.shape, dtype=jnp.float64
            )
            right_weights.append(baseline + (0.1 * scale) * noise)
        params["left_route_orbit_weights"] = tuple(left_weights)
        params["right_route_orbit_weights"] = tuple(right_weights)
    elif config.pair_mode == "equivariant_full_rank":
        incoming = _incoming_route_indices(config.catalog)
        orbit_weights = []
        for offset, layout in enumerate(spin_route_pair_orbit_layouts(config)):
            representative = layout.representative
            left_count = max(len(incoming[representative.parent_route.left]), 1)
            right_count = max(len(incoming[representative.parent_route.right]), 1)
            baseline = jnp.full(
                (layout.channel_count,),
                1.0 / np.sqrt(left_count * right_count),
                dtype=jnp.float64,
            )
            noise = jax.random.normal(
                keys[len(unique) + offset], baseline.shape, dtype=jnp.float64
            )
            orbit_weights.append(baseline + (0.1 * scale) * noise)
        params["pair_orbit_weights"] = tuple(orbit_weights)
    params["readout"] = scale * jax.random.normal(
        keys[-1], (width,), dtype=jnp.float64
    )
    return params


def init_paired_rank_one_and_equivariant_route_pair_dag(
    key: jax.Array,
    candidate_config: SpinPersistentRoutePairConfig,
) -> tuple[
    SpinPersistentRoutePairConfig,
    dict[str, object],
    dict[str, object],
]:
    """Return an exactly common rank-one/candidate initialization.

    The first result is the learned equivariant-factorized control config,
    followed by control and candidate parameter pytrees.  Merge tensors and
    readout arrays are shared exactly.  Marginal and joint orbit weights start
    at the same uniform rank-one value, so both amplitudes are elementwise
    identical; optimization may subsequently move only the candidate into
    higher route-pair rank.
    """

    if candidate_config.pair_mode != "equivariant_full_rank":
        raise ValueError(
            "candidate_config must use equivariant_full_rank pair mode"
        )
    control_config = replace(
        candidate_config, pair_mode="equivariant_factorized_rank_one"
    )
    # Use a fixed rank-one initialization as the common backbone, then attach
    # exact uniform marginal factors to the learnable control.
    fixed_config = replace(candidate_config, pair_mode="rank_one")
    fixed = init_spin_persistent_route_pair_dag(key, fixed_config)
    incoming = _incoming_route_indices(candidate_config.catalog)
    left_weights = tuple(
        jnp.full(
            (layout.channel_count,),
            1.0
            / np.sqrt(
                max(
                    len(
                        incoming[
                            layout.representative.parent_route.left
                        ]
                    ),
                    1,
                )
            ),
            dtype=jnp.asarray(fixed["readout"]).dtype,
        )
        for layout in spin_route_marginal_orbit_layouts(
            control_config, "left"
        )
    )
    right_weights = tuple(
        jnp.full(
            (layout.channel_count,),
            1.0
            / np.sqrt(
                max(
                    len(
                        incoming[
                            layout.representative.parent_route.right
                        ]
                    ),
                    1,
                )
            ),
            dtype=jnp.asarray(fixed["readout"]).dtype,
        )
        for layout in spin_route_marginal_orbit_layouts(
            control_config, "right"
        )
    )
    control = {
        "merges": fixed["merges"],
        "left_route_orbit_weights": left_weights,
        "right_route_orbit_weights": right_weights,
        "readout": fixed["readout"],
    }
    orbit_weights = []
    for layout in spin_route_pair_orbit_layouts(candidate_config):
        parent = layout.representative.parent_route
        left_count = max(len(incoming[parent.left]), 1)
        right_count = max(len(incoming[parent.right]), 1)
        orbit_weights.append(
            jnp.full(
                (layout.channel_count,),
                1.0 / np.sqrt(left_count * right_count),
                dtype=jnp.asarray(fixed["readout"]).dtype,
            )
        )
    candidate = {
        "merges": fixed["merges"],
        "pair_orbit_weights": tuple(orbit_weights),
        "readout": fixed["readout"],
    }
    _validate_parameter_shapes(control, control_config)
    _validate_parameter_shapes(candidate, candidate_config)
    return control_config, control, candidate


def _validate_parameter_shapes(
    params: Mapping[str, object], config: SpinPersistentRoutePairConfig
) -> None:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    expected_keys = {"merges", "readout"}
    if config.pair_mode == "full_rank":
        expected_keys.add("pair_kernels")
    elif config.pair_mode == "equivariant_factorized_rank_one":
        expected_keys.update(
            {"left_route_orbit_weights", "right_route_orbit_weights"}
        )
    elif config.pair_mode == "equivariant_full_rank":
        expected_keys.add("pair_orbit_weights")
    if set(params) != expected_keys:
        raise ValueError(f"expected parameter keys {sorted(expected_keys)}")
    unique = _unique_merge_layouts(config)
    merges = params["merges"]
    if not isinstance(merges, (tuple, list)) or len(merges) != len(unique):
        raise ValueError(f"expected {len(unique)} merge tensors")
    width = config.feature_dim
    for index, (value, layout) in enumerate(zip(merges, unique, strict=True)):
        left_width = 1 if layout.left_size == 1 else width
        right_width = 1 if layout.right_size == 1 else width
        expected = (len(layout.keys), width, left_width, right_width)
        if np.shape(value) != expected:
            raise ValueError(
                f"merge tensor {index} has shape {np.shape(value)}, expected {expected}"
            )
    if np.shape(params["readout"]) != (width,):
        raise ValueError(f"readout must have shape ({width},)")
    if config.pair_mode == "rank_one":
        return
    if config.pair_mode == "equivariant_factorized_rank_one":
        for side in ("left", "right"):
            key = f"{side}_route_orbit_weights"
            values = params[key]
            layouts = spin_route_marginal_orbit_layouts(config, side)
            if not isinstance(values, (tuple, list)) or len(values) != len(layouts):
                raise ValueError(
                    f"expected {len(layouts)} {side} route-orbit weights"
                )
            for index, (value, layout) in enumerate(
                zip(values, layouts, strict=True)
            ):
                expected = (layout.channel_count,)
                if np.shape(value) != expected:
                    raise ValueError(
                        f"{side} route-orbit weight {index} has shape "
                        f"{np.shape(value)}, expected {expected}"
                    )
        return
    if config.pair_mode == "equivariant_full_rank":
        weights = params["pair_orbit_weights"]
        layouts = spin_route_pair_orbit_layouts(config)
        if not isinstance(weights, (tuple, list)) or len(weights) != len(layouts):
            raise ValueError(f"expected {len(layouts)} route-pair orbit weights")
        for index, (value, layout) in enumerate(zip(weights, layouts, strict=True)):
            expected = (layout.channel_count,)
            if np.shape(value) != expected:
                raise ValueError(
                    f"route-pair orbit weight {index} has shape "
                    f"{np.shape(value)}, expected {expected}"
                )
        return
    kernels = params["pair_kernels"]
    pair_layouts = spin_route_pair_kernel_layouts(config)
    if not isinstance(kernels, (tuple, list)) or len(kernels) != len(pair_layouts):
        raise ValueError(f"expected {len(pair_layouts)} route-pair kernels")
    merge_layouts = spin_reduced_fusion_dag_merge_layouts(_base_config(config))
    for index, (value, pair) in enumerate(zip(kernels, pair_layouts, strict=True)):
        channels = len(merge_layouts[pair.merge_route_index].keys)
        expected = (channels, pair.left_route_count, pair.right_route_count)
        if np.shape(value) != expected:
            raise ValueError(
                f"route-pair kernel {index} has shape {np.shape(value)}, "
                f"expected {expected}"
            )


def route_pair_bilinear(
    kernel: jax.Array, left_routes: jax.Array, right_routes: jax.Array
) -> jax.Array:
    """Contract one scalar route-pair response, useful for rank witnesses."""

    kernel = jnp.asarray(kernel)
    left_routes = jnp.asarray(left_routes)
    right_routes = jnp.asarray(right_routes)
    if kernel.ndim != 2:
        raise ValueError("kernel must be a rank-two route-pair matrix")
    if left_routes.shape != (kernel.shape[0],):
        raise ValueError("left route vector has the wrong shape")
    if right_routes.shape != (kernel.shape[1],):
        raise ValueError("right route vector has the wrong shape")
    return jnp.einsum("r,rs,s->", left_routes, kernel, right_routes)


def _fixed_rank_one_kernel(
    channels: int, left_count: int, right_count: int
) -> jax.Array:
    return jnp.full(
        (channels, left_count, right_count),
        1.0 / np.sqrt(left_count * right_count),
        dtype=jnp.float64,
    )


@lru_cache(maxsize=None)
def _equivariant_orbit_matrix(
    config: SpinPersistentRoutePairConfig, merge_index: int
) -> np.ndarray:
    """Return decorated-orbit labels for one selected parent route."""

    parent_route = config.catalog.routes[merge_index]
    incoming = _incoming_route_indices(config.catalog)
    left_values: tuple[DisjointMergeRoute | None, ...] = (
        tuple(config.catalog.routes[index] for index in incoming[parent_route.left])
        if incoming[parent_route.left]
        else (None,)
    )
    right_values: tuple[DisjointMergeRoute | None, ...] = (
        tuple(config.catalog.routes[index] for index in incoming[parent_route.right])
        if incoming[parent_route.right]
        else (None,)
    )
    decorations = _selected_decorated_route_pairs(config)
    labels = _decorated_route_pair_orbit_indices(config)
    label_by_value = dict(zip(decorations, labels, strict=True))
    matrix = np.asarray(
        [
            [
                label_by_value[
                    SpinDecoratedRoutePair(parent_route, left_route, right_route)
                ]
                for right_route in right_values
            ]
            for left_route in left_values
        ],
        dtype=np.int32,
    )
    matrix.setflags(write=False)
    return matrix


def _equivariant_kernel(
    params: Mapping[str, object],
    config: SpinPersistentRoutePairConfig,
    merge_index: int,
) -> jax.Array:
    orbit_matrix = _equivariant_orbit_matrix(config, merge_index)
    flat = tuple(int(value) for value in orbit_matrix.reshape(-1))
    columns = jnp.stack(
        [params["pair_orbit_weights"][index] for index in flat],  # type: ignore[index]
        axis=1,
    )
    return jnp.reshape(
        columns,
        (columns.shape[0], orbit_matrix.shape[0], orbit_matrix.shape[1]),
    )


@lru_cache(maxsize=None)
def _marginal_orbit_vector(
    config: SpinPersistentRoutePairConfig,
    merge_index: int,
    side: Literal["left", "right"],
) -> np.ndarray:
    parent_route = config.catalog.routes[merge_index]
    support = parent_route.left if side == "left" else parent_route.right
    incoming = _incoming_route_indices(config.catalog)[support]
    child_routes: tuple[DisjointMergeRoute | None, ...] = (
        tuple(config.catalog.routes[index] for index in incoming)
        if incoming
        else (None,)
    )
    decorations = _selected_marginal_decorations(config, side)
    labels = _marginal_orbit_indices(config, side)
    label_by_value = dict(zip(decorations, labels, strict=True))
    vector = np.asarray(
        [
            label_by_value[
                SpinDecoratedChildRoute(parent_route, side, child_route)
            ]
            for child_route in child_routes
        ],
        dtype=np.int32,
    )
    vector.setflags(write=False)
    return vector


def _factorized_equivariant_kernel(
    params: Mapping[str, object],
    config: SpinPersistentRoutePairConfig,
    merge_index: int,
) -> jax.Array:
    left_indices = _marginal_orbit_vector(config, merge_index, "left")
    right_indices = _marginal_orbit_vector(config, merge_index, "right")
    left = jnp.stack(
        [
            params["left_route_orbit_weights"][int(index)]  # type: ignore[index]
            for index in left_indices
        ],
        axis=1,
    )
    right = jnp.stack(
        [
            params["right_route_orbit_weights"][int(index)]  # type: ignore[index]
            for index in right_indices
        ],
        axis=1,
    )
    return left[:, :, None] * right[:, None, :]


def spin_persistent_route_pair_support_carriers(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinPersistentRoutePairConfig,
) -> dict[FusionSupport, tuple[jax.Array, jax.Array]]:
    """Return every ``(m2, route-by-q-by-width)`` support carrier.

    This diagnostic exposes intermediate covariance without changing the
    scalar amplitude path.  It is meaningful for valid binary spins; the
    public amplitude separately masks invalid inputs to zero.
    """

    _validate_parameter_shapes(params, config)
    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (config.n_sites,):
        raise ValueError(
            f"expected spin shape ({config.n_sites},), got {spins.shape}"
        )
    safe_spins = jnp.clip(spins, 0, 1)
    width = config.feature_dim
    base = _base_config(config)
    merge_layouts = spin_reduced_fusion_dag_merge_layouts(base)
    pair_by_merge = {
        layout.merge_route_index: layout
        for layout in spin_route_pair_kernel_layouts(config)
    }
    support_values: dict[FusionSupport, tuple[jax.Array, jax.Array]] = {}

    for support_layout in spin_reduced_fusion_dag_support_layouts(base):
        support = support_layout.support
        if len(support) == 1:
            m2 = 2 * safe_spins[support.sites[0]] - 1
            value = jnp.zeros((1, 2, 1), dtype=jnp.float64)
            support_values[support] = (m2, value.at[0, 1, 0].set(1.0))
            continue
        route_values = []
        parent_m2: jax.Array | None = None
        for merge_index in support_layout.incoming_route_indices:
            merge = merge_layouts[merge_index]
            left_m2, left_routes = support_values[merge.route.left]
            right_m2, right_routes = support_values[merge.route.right]
            channels = jnp.asarray(merge.keys, dtype=jnp.int32)
            q_left = channels[:, 0]
            q_right = channels[:, 1]
            q_total = channels[:, 2]
            left_selected = left_routes[:, q_left, :]
            right_selected = right_routes[:, q_right, :]
            pair_layout = pair_by_merge.get(merge_index)
            if config.pair_mode == "full_rank" and pair_layout is not None:
                kernel = params["pair_kernels"][pair_layout.kernel_index]  # type: ignore[index]
            elif (
                config.pair_mode == "equivariant_factorized_rank_one"
                and pair_layout is not None
            ):
                kernel = _factorized_equivariant_kernel(
                    params, config, merge_index
                )
            elif (
                config.pair_mode == "equivariant_full_rank"
                and pair_layout is not None
            ):
                kernel = _equivariant_kernel(
                    params, config, merge_index
                )
            else:
                kernel = _fixed_rank_one_kernel(
                    len(merge.keys), left_routes.shape[0], right_routes.shape[0]
                )
            weights = params["merges"][merge.parameter_index]  # type: ignore[index]
            reduced = jnp.einsum(
                "rca,scb,crs,coab->co",
                left_selected,
                right_selected,
                kernel,
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
            result = jnp.zeros(
                (merge.size + 1, width), dtype=jnp.result_type(reduced, coefficient)
            )
            route_values.append(
                result.at[q_total].add(coefficient[:, None] * reduced)
            )
            parent_m2 = left_m2 + right_m2
        if parent_m2 is None:  # pragma: no cover - catalog rejects this
            raise RuntimeError("composite support has no incoming route")
        support_values[support] = (parent_m2, jnp.stack(route_values, axis=0))

    return support_values


def spin_persistent_route_pair_dag_amplitude(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinPersistentRoutePairConfig,
) -> jax.Array:
    """Evaluate one exact-``(S,M)`` physical-spin amplitude."""

    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (config.n_sites,):
        raise ValueError(
            f"expected spin shape ({config.n_sites},), got {spins.shape}"
        )
    valid_spins = jnp.all((spins == 0) | (spins == 1))
    support_values = spin_persistent_route_pair_support_carriers(
        params, spins, config
    )
    root = FusionSupport(tuple(range(config.n_sites)))
    total_m2, root_routes = support_values[root]
    root_by_q = jnp.sum(root_routes, axis=0) / jnp.sqrt(
        jnp.asarray(root_routes.shape[0], dtype=jnp.float64)
    )
    value = jnp.dot(root_by_q[config.target_q], params["readout"])
    return jnp.where(
        valid_spins & (total_m2 == config.target_m2),
        value,
        jnp.asarray(0.0, dtype=value.dtype),
    )


def spin_persistent_route_pair_dag_amplitudes(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinPersistentRoutePairConfig,
) -> jax.Array:
    """Vectorise the persistent route-pair amplitude over configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: spin_persistent_route_pair_dag_amplitude(
            params, row, config
        )
    )(array)


def spin_persistent_route_pair_parameter_count(
    params: Mapping[str, object],
) -> int:
    """Count trainable scalars in a persistent route-pair parameter tree."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


def spin_route_pair_effective_kernels(
    params: Mapping[str, object], config: SpinPersistentRoutePairConfig
) -> tuple[jax.Array, ...]:
    """Return learned/fixed route-pair matrices aligned with public layouts.

    Each returned array has shape ``(allowed_channels, left_routes,
    right_routes)`` and is aligned with :func:`spin_route_pair_kernel_layouts`.
    This read-only diagnostic exposes the actual matrices used by the
    contraction, allowing channel-resolved singular-value/rank audits without
    duplicating equivariant-orbit bookkeeping in benchmark scripts.
    """

    _validate_parameter_shapes(params, config)
    pair_layouts = spin_route_pair_kernel_layouts(config)
    merge_layouts = spin_reduced_fusion_dag_merge_layouts(_base_config(config))
    kernels = []
    for pair in pair_layouts:
        channels = len(merge_layouts[pair.merge_route_index].keys)
        if config.pair_mode == "rank_one":
            kernel = _fixed_rank_one_kernel(
                channels, pair.left_route_count, pair.right_route_count
            )
        elif config.pair_mode == "full_rank":
            kernel = params["pair_kernels"][pair.kernel_index]  # type: ignore[index]
        elif config.pair_mode == "equivariant_factorized_rank_one":
            kernel = _factorized_equivariant_kernel(
                params, config, pair.merge_route_index
            )
        else:
            kernel = _equivariant_kernel(params, config, pair.merge_route_index)
        kernels.append(jnp.asarray(kernel))
    return tuple(kernels)


__all__ = [
    "SpinPersistentRoutePairConfig",
    "SpinDecoratedRoutePair",
    "SpinRoutePairKernelLayout",
    "SpinRoutePairOrbitLayout",
    "SpinRoutePairMode",
    "init_spin_persistent_route_pair_dag",
    "init_paired_rank_one_and_equivariant_route_pair_dag",
    "decorated_route_pair_image",
    "route_pair_bilinear",
    "select_group_closed_persistent_pair_supports",
    "spin_persistent_route_pair_dag_amplitude",
    "spin_persistent_route_pair_dag_amplitudes",
    "spin_persistent_route_pair_parameter_count",
    "spin_persistent_route_pair_support_carriers",
    "spin_route_pair_effective_kernels",
    "spin_route_pair_kernel_layouts",
    "spin_route_pair_orbit_layouts",
    "spin_route_marginal_orbit_layouts",
    "SpinDecoratedChildRoute",
    "SpinRouteMarginalOrbitLayout",
    "decorated_child_route_image",
]
