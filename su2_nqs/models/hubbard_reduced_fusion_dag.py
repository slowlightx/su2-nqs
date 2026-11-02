"""Exact Hubbard leaf/merge engine for prospective reduced-fusion DAGs.

This module is the Phase-1, single-route subset of the reduced-fusion-DAG
proposal.  It evaluates an arbitrary binary tree of disjoint physical support
sets while keeping charge and every doubled-spin channel explicit.  Magnetic
indices are contracted only through the repository's fixed Clebsch--Gordan
tables; learned tensors act on reduced multiplicities.

The tree restriction is intentional.  It establishes the exact primitive
needed by a future multi-route DAG without silently treating a terminal sum of
complete trees as feature-level route mixing.  In particular, this module
contains no optimizer and makes no energy-accuracy claim.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.fusion_catalog import DisjointMergeRoute, FusionSupport
from ..su2.graded_recoupling import (
    canonical_hubbard_child_merge_sign_jax,
    validated_child_merge_sites,
)
from ..su2.hubbard_cg_irrep import token_quantum_number_tables
from ..su2.quantum_numbers import coupling_range
from .hubbard_cg import (
    HubbardCGConfig,
    _cg_tensor,
    _charge_spin_channels,
    _coupled_q_indices,
    hubbard_cg_active_layout,
    hubbard_cg_tree_schedule,
)


@dataclass(frozen=True)
class HubbardReducedFusionTree:
    """A topologically ordered single-route fusion tree.

    ``routes`` must contain exactly one route for every composite support.
    Every composite child must have been produced earlier in the sequence,
    and the last parent must be the all-site root.  Child supports retain
    their orientation because both CG coefficients and fermionic signs depend
    on it.
    """

    n_sites: int
    routes: tuple[DisjointMergeRoute, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_sites, bool)
            or not isinstance(self.n_sites, Integral)
            or self.n_sites < 2
        ):
            raise ValueError("n_sites must be an integer of at least two")
        n_sites = int(self.n_sites)
        routes = tuple(self.routes)
        if len(routes) != n_sites - 1:
            raise ValueError("a binary fusion tree must have n_sites - 1 routes")

        available = {FusionSupport((site,)) for site in range(n_sites)}
        child_counts: Counter[FusionSupport] = Counter()
        for route in routes:
            if not isinstance(route, DisjointMergeRoute):
                raise TypeError("routes must contain DisjointMergeRoute values")
            for support in (route.parent, route.left, route.right):
                if support.sites[-1] >= n_sites:
                    raise ValueError("fusion-tree support lies outside n_sites")
            if route.parent in available:
                raise ValueError("each composite support must be produced exactly once")
            if route.left not in available or route.right not in available:
                raise ValueError("routes must be in dependency/topological order")
            available.add(route.parent)
            child_counts.update((route.left, route.right))

        root = FusionSupport(tuple(range(n_sites)))
        if routes[-1].parent != root:
            raise ValueError("the final route must produce the all-site root")
        expected_children = available - {root}
        if set(child_counts) != expected_children or any(
            count != 1 for count in child_counts.values()
        ):
            raise ValueError(
                "every non-root support must feed the root exactly once"
            )
        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "routes", routes)


@dataclass(frozen=True)
class HubbardReducedFusionMergeLayout:
    """Packed target-reachable charge/spin channels for one tree route."""

    route: DisjointMergeRoute
    keys: tuple[tuple[int, int, int, int, int], ...]
    lookup: np.ndarray

    @property
    def left_size(self) -> int:
        return len(self.route.left)

    @property
    def right_size(self) -> int:
        return len(self.route.right)

    @property
    def size(self) -> int:
        return len(self.route.parent)


def balanced_hubbard_reduced_fusion_tree(
    n_sites: int,
) -> HubbardReducedFusionTree:
    """Return the contiguous balanced tree used by :mod:`hubbard_cg`.

    This is the exact bridge for the single-route regression: all child
    supports are already in canonical order, so every graded merge sign is
    ``+1`` and mapped active-HubbardCG parameters replay element by element.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError("n_sites must be an integer of at least two")
    routes = []
    for start, split, stop, _ in hubbard_cg_tree_schedule(int(n_sites)):
        routes.append(
            DisjointMergeRoute(
                FusionSupport(tuple(range(start, stop))),
                FusionSupport(tuple(range(start, split))),
                FusionSupport(tuple(range(split, stop))),
            )
        )
    return HubbardReducedFusionTree(int(n_sites), tuple(routes))


@lru_cache(maxsize=None)
def hubbard_reduced_fusion_layouts(
    tree: HubbardReducedFusionTree,
    config: HubbardCGConfig,
) -> tuple[HubbardReducedFusionMergeLayout, ...]:
    """Compile packed charge/spin channels for a static single-route tree."""

    if tree.n_sites != config.n_sites:
        raise ValueError("tree and HubbardCGConfig disagree on n_sites")
    root = FusionSupport(tuple(range(config.n_sites)))
    layouts = []
    for route in tree.routes:
        left_size = len(route.left)
        right_size = len(route.right)
        size = len(route.parent)
        outside = config.n_sites - size
        minimum_charge = max(0, config.n_electrons - 2 * outside)
        maximum_charge = min(2 * size, config.n_electrons)
        lookup = np.full(
            (
                2 * left_size + 1,
                left_size + 1,
                2 * right_size + 1,
                right_size + 1,
                size + 1,
            ),
            -1,
            dtype=np.int32,
        )
        keys = []
        for left_charge, q_left in _charge_spin_channels(left_size):
            for right_charge, q_right in _charge_spin_channels(right_size):
                total_charge = left_charge + right_charge
                if not minimum_charge <= total_charge <= maximum_charge:
                    continue
                for q_total in coupling_range(q_left, q_right):
                    if route.parent == root and (
                        total_charge != config.n_electrons
                        or q_total != config.target_q
                    ):
                        continue
                    lookup[
                        left_charge,
                        q_left,
                        right_charge,
                        q_right,
                        q_total,
                    ] = len(keys)
                    keys.append(
                        (
                            left_charge,
                            q_left,
                            right_charge,
                            q_right,
                            q_total,
                        )
                    )
        lookup.setflags(write=False)
        layouts.append(
            HubbardReducedFusionMergeLayout(route, tuple(keys), lookup)
        )
    return tuple(layouts)


def init_hubbard_reduced_fusion_tree(
    key: jax.Array,
    tree: HubbardReducedFusionTree,
    config: HubbardCGConfig,
) -> dict[str, object]:
    """Initialize real reduced tensors for a validated single-route tree."""

    layouts = hubbard_reduced_fusion_layouts(tree, config)
    keys = jax.random.split(key, len(layouts) + 2)
    scale = float(config.initialization_scale)
    width = int(config.feature_dim)
    leaves = scale * jax.random.normal(
        keys[0], (config.n_sites, 3, width), dtype=jnp.float64
    )
    merges = tuple(
        (scale / width)
        * jax.random.normal(
            keys[index + 1],
            (len(layout.keys), width, width, width),
            dtype=jnp.float64,
        )
        for index, layout in enumerate(layouts)
    )
    readout = scale * jax.random.normal(keys[-1], (width,), dtype=jnp.float64)
    return {"leaves": leaves, "merges": merges, "readout": readout}


def _validate_parameter_shapes(
    params: Mapping[str, object],
    layouts: Sequence[HubbardReducedFusionMergeLayout],
    config: HubbardCGConfig,
) -> None:
    required = {"leaves", "merges", "readout"}
    if set(params) != required:
        raise ValueError(f"params must have exactly the keys {sorted(required)}")
    width = int(config.feature_dim)
    if np.shape(params["leaves"]) != (config.n_sites, 3, width):
        raise ValueError("leaf-parameter shape disagrees with config")
    if np.shape(params["readout"]) != (width,):
        raise ValueError("readout-parameter shape disagrees with config")
    if not jnp.issubdtype(jnp.asarray(params["leaves"]).dtype, jnp.inexact):
        raise TypeError("leaf parameters must have real or complex dtype")
    if not jnp.issubdtype(jnp.asarray(params["readout"]).dtype, jnp.inexact):
        raise TypeError("readout parameters must have real or complex dtype")
    merges = tuple(params["merges"])
    if len(merges) != len(layouts):
        raise ValueError("merge-parameter count disagrees with the tree")
    for weights, layout in zip(merges, layouts, strict=True):
        if np.shape(weights) != (len(layout.keys), width, width, width):
            raise ValueError("merge-parameter shape disagrees with its layout")
        if not jnp.issubdtype(jnp.asarray(weights).dtype, jnp.inexact):
            raise TypeError("merge parameters must have real or complex dtype")


def map_active_hubbard_cg_to_reduced_tree(
    active_params: Mapping[str, object],
    config: HubbardCGConfig,
) -> tuple[dict[str, object], HubbardReducedFusionTree]:
    """Map packed balanced-HubbardCG parameters without approximation.

    The input is the output of ``pack_hubbard_cg_parameters`` or
    ``init_active_hubbard_cg``.  Returned arrays are not numerically changed;
    the explicit layout comparison guards against a silent convention drift.
    """

    tree = balanced_hubbard_reduced_fusion_tree(config.n_sites)
    target_layouts = hubbard_reduced_fusion_layouts(tree, config)
    source_layouts = hubbard_cg_active_layout(config)
    if len(source_layouts) != len(target_layouts):
        raise RuntimeError("balanced HubbardCG and reduced-tree layouts disagree")
    for source, target in zip(source_layouts, target_layouts, strict=True):
        if source.keys != target.keys:
            raise RuntimeError("balanced HubbardCG channel ordering has drifted")
    mapped = {
        "leaves": jnp.asarray(active_params["leaves"]),
        "merges": tuple(jnp.asarray(value) for value in active_params["merges"]),
        "readout": jnp.asarray(active_params["readout"]),
    }
    _validate_parameter_shapes(mapped, target_layouts, config)
    return mapped, tree


def hubbard_reduced_fusion_merge(
    left_charge: jax.Array,
    left_m2: jax.Array,
    left_by_q: jax.Array,
    right_charge: jax.Array,
    right_m2: jax.Array,
    right_by_q: jax.Array,
    weights: jax.Array,
    tokens: jax.Array,
    layout: HubbardReducedFusionMergeLayout,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply one exact charge/SU(2)/fermion-graded disjoint merge.

    ``layout`` and its supports are static metadata.  The supports were
    host-validated when the tree was built; the pure-JAX sign kernel is used
    here so this primitive remains compatible with ``jit``, ``vmap`` and
    reverse-mode differentiation.
    """

    left_sites, right_sites = validated_child_merge_sites(
        layout.route.left.sites,
        layout.route.right.sites,
        tokens.shape[-1],
    )
    lookup = jnp.asarray(layout.lookup)
    cg = jnp.asarray(_cg_tensor(layout.left_size, layout.right_size))
    q_left, q_right, q_total = (
        jnp.asarray(values)
        for values in _coupled_q_indices(layout.left_size, layout.right_size)
    )
    packed_index = lookup[
        left_charge,
        q_left,
        right_charge,
        q_right,
        q_total,
    ]
    valid_channel = packed_index >= 0
    safe_index = jnp.maximum(packed_index, 0)
    reduced_product = jnp.einsum(
        "ca,cb,cabk->ck",
        left_by_q[q_left],
        right_by_q[q_right],
        weights[safe_index],
    )
    coefficient = cg[
        q_left,
        left_m2 + layout.left_size,
        q_right,
        right_m2 + layout.right_size,
        q_total,
    ]
    merge_sign = canonical_hubbard_child_merge_sign_jax(
        left_sites, right_sites, tokens
    )
    contribution = jnp.where(
        valid_channel[:, None],
        merge_sign * coefficient[:, None] * reduced_product,
        jnp.zeros_like(reduced_product),
    )
    result = jnp.zeros(
        (layout.size + 1, left_by_q.shape[-1]),
        dtype=jnp.result_type(reduced_product, coefficient),
    )
    result = result.at[q_total].add(contribution)
    return left_charge + right_charge, left_m2 + right_m2, result


def hubbard_reduced_fusion_tree_amplitude(
    params: Mapping[str, object],
    tokens: jax.Array,
    tree: HubbardReducedFusionTree,
    config: HubbardCGConfig,
) -> jax.Array:
    """Evaluate one physical configuration in fixed ``(Ne,S,M)``.

    Invalid tokens or configurations outside the declared charge/magnetic
    sector return zero.  Total spin is exact by construction for every tree;
    no projection or state-dependent nonlinear activation is applied.
    """

    if tree.n_sites != config.n_sites:
        raise ValueError("tree and HubbardCGConfig disagree on n_sites")
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.shape != (config.n_sites,):
        raise ValueError(f"expected token shape ({config.n_sites},), got {tokens.shape}")
    layouts = hubbard_reduced_fusion_layouts(tree, config)
    _validate_parameter_shapes(params, layouts, config)
    valid_tokens = jnp.all((tokens >= 0) & (tokens <= 3))
    safe_tokens = jnp.clip(tokens, 0, 3)
    charge_table, q_table, m2_table, reduced_table = (
        jnp.asarray(values, dtype=jnp.int32)
        for values in token_quantum_number_tables()
    )

    nodes: dict[FusionSupport, tuple[jax.Array, jax.Array, jax.Array]] = {}
    width = int(config.feature_dim)
    leaves = params["leaves"]
    for site in range(config.n_sites):
        token = safe_tokens[site]
        vector = leaves[site, reduced_table[token]]
        by_q = jnp.zeros((2, width), dtype=vector.dtype).at[q_table[token]].set(vector)
        nodes[FusionSupport((site,))] = (
            charge_table[token],
            m2_table[token],
            by_q,
        )

    for weights, layout in zip(params["merges"], layouts, strict=True):
        left_charge, left_m2, left_by_q = nodes[layout.route.left]
        right_charge, right_m2, right_by_q = nodes[layout.route.right]
        nodes[layout.route.parent] = hubbard_reduced_fusion_merge(
            left_charge,
            left_m2,
            left_by_q,
            right_charge,
            right_m2,
            right_by_q,
            weights,
            safe_tokens,
            layout,
        )

    root = FusionSupport(tuple(range(config.n_sites)))
    total_charge, total_m2, root_by_q = nodes[root]
    value = jnp.dot(root_by_q[config.target_q], params["readout"])
    valid_sector = (
        valid_tokens
        & (total_charge == config.n_electrons)
        & (total_m2 == config.target_m2)
    )
    return jnp.where(valid_sector, value, jnp.asarray(0.0, dtype=value.dtype))


__all__ = [
    "HubbardReducedFusionMergeLayout",
    "HubbardReducedFusionTree",
    "balanced_hubbard_reduced_fusion_tree",
    "hubbard_reduced_fusion_layouts",
    "hubbard_reduced_fusion_merge",
    "hubbard_reduced_fusion_tree_amplitude",
    "init_hubbard_reduced_fusion_tree",
    "map_active_hubbard_cg_to_reduced_tree",
]
