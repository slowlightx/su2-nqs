"""Clebsch tree with an exact local Schur-multiplicity front end.

Physical spin configurations are first transformed, cell by cell, into a
complete orthonormal Schur basis ``(q,m,a)``.  Fixed Clebsch--Gordan tensors
then merge cells, while trainable real maps act only on the reduced
multiplicity/feature axes.  The local multiplicity ``a`` is therefore not
compressed until the first *inter-cell* merge.

This is an unnormalised amplitude model.  A physical-space translation (or
other one-dimensional-character) projector can be attached without changing
the reduced parameter count.  All angular momenta and projections are doubled
integers and the computational-basis convention is ``0=down, 1=up`` with site
zero most significant.

The ``full_joint`` kernel is ``W[k,a,b]`` for every allowed angular-momentum
triple.  ``compressed_shared_tree`` first applies one cell-shared
``A_q[u,a]`` and then an ordinary ``B[k,u,v]`` merge; it is the primary
ordinary-tree boundary control.  The auxiliary ``factorized_rank_one`` arm
uses ``W[k,a,b] = L[k,a] R[k,b]``.  The latter has continuous scale gauges,
so scalar parameter counts must be supplemented by normalized-state Jacobian
ranks in any efficiency claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import FusionTree, Leaf, TreeNode
from ..su2.local_schur import (
    CellKind,
    CellSchurSpec,
    cell_schur_spec,
    cell_schur_tensor,
    cell_schur_unitarity_error,
)
from ..su2.quantum_numbers import coupling_range, validate_m, validate_q
from .spatial_projector import PhysicalSpatialProjector, spatial_projected_amplitude
from .spin_clebsch_tree import spin_cg_tensor


MergeMode = Literal[
    "full_joint", "compressed_shared_tree", "factorized_rank_one"
]


@dataclass(frozen=True)
class SpinCellMultiplicityConfig:
    """Static cell partition, exact-spin target, and reduced-map contract."""

    cell_kind: CellKind
    cells: tuple[tuple[int, ...], ...]
    target_q: int = 0
    target_m2: int = 0
    feature_dim: int = 2
    merge_mode: MergeMode = "full_joint"
    maximum_intermediate_q: int | None = None
    initialization_scale: float = 0.2
    projector: PhysicalSpatialProjector | None = None

    def __post_init__(self) -> None:
        spec = cell_schur_spec(self.cell_kind)
        raw_cells = tuple(tuple(cell) for cell in self.cells)
        if len(raw_cells) < 2:
            raise ValueError("cell-multiplicity trees require at least two cells")
        for cell in raw_cells:
            if len(cell) != spec.cell_size:
                raise ValueError(f"every {self.cell_kind} cell needs {spec.cell_size} sites")
            if any(isinstance(site, bool) or not isinstance(site, Integral) for site in cell):
                raise TypeError("cell site labels must be integers")
        checked_cells = tuple(tuple(int(site) for site in cell) for cell in raw_cells)
        flattened = tuple(site for cell in checked_cells for site in cell)
        if tuple(sorted(flattened)) != tuple(range(len(flattened))):
            raise ValueError("cells must partition physical sites 0..N-1 exactly once")
        target_q = validate_q(self.target_q, name="target_q")
        target_m2 = validate_m(target_q, self.target_m2, name="target_m2")
        n_sites = len(flattened)
        if target_q > n_sites or (n_sites - target_q) % 2:
            raise ValueError("target_q is unreachable for the physical site count")
        if (
            isinstance(self.feature_dim, bool)
            or not isinstance(self.feature_dim, Integral)
            or self.feature_dim < 1
        ):
            raise ValueError("feature_dim must be a positive integer")
        if self.merge_mode not in (
            "full_joint",
            "compressed_shared_tree",
            "factorized_rank_one",
        ):
            raise ValueError(f"unknown merge_mode: {self.merge_mode!r}")
        if self.maximum_intermediate_q is not None:
            maximum_q = validate_q(
                self.maximum_intermediate_q, name="maximum_intermediate_q"
            )
            if maximum_q < target_q:
                raise ValueError("maximum_intermediate_q cannot be below target_q")
        if (
            not np.isfinite(self.initialization_scale)
            or self.initialization_scale <= 0.0
        ):
            raise ValueError("initialization_scale must be positive and finite")
        if self.projector is not None:
            if not isinstance(self.projector, PhysicalSpatialProjector):
                raise TypeError("projector must be a PhysicalSpatialProjector")
            if self.projector.fermionic:
                raise ValueError("spin cell carriers require a bosonic projector")
            if self.projector.n_sites != n_sites:
                raise ValueError("projector and cell partition disagree on site count")
        object.__setattr__(self, "cells", checked_cells)
        object.__setattr__(self, "target_q", target_q)
        object.__setattr__(self, "target_m2", target_m2)
        object.__setattr__(self, "feature_dim", int(self.feature_dim))

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_sites(self) -> int:
        return sum(len(cell) for cell in self.cells)

    @property
    def cell_tree(self) -> FusionTree:
        return FusionTree.balanced(self.n_cells)


def active_cell_q_values(config: SpinCellMultiplicityConfig) -> tuple[int, ...]:
    """Return local q blocks that can enter the configured merge tree."""

    values = cell_schur_spec(config.cell_kind).q_values
    if config.maximum_intermediate_q is None:
        return values
    return tuple(q for q in values if q <= config.maximum_intermediate_q)


@dataclass(frozen=True)
class CellCGMergeLayout:
    """One postorder inter-cell CG merge and its packed reduced widths."""

    left_size: int
    right_size: int
    parameter_index: int
    keys: tuple[tuple[int, int, int], ...]
    left_widths: tuple[int, ...]
    right_widths: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.left_size + self.right_size


@lru_cache(maxsize=None)
def _cell_count(node: TreeNode) -> int:
    if isinstance(node, Leaf):
        return 1
    return _cell_count(node.left) + _cell_count(node.right)


def _reachable_q(size: int, maximum_q: int | None) -> tuple[int, ...]:
    upper = size if maximum_q is None else min(size, maximum_q)
    return tuple(range(size % 2, upper + 1, 2))


@lru_cache(maxsize=None)
def cell_multiplicity_layouts(
    config: SpinCellMultiplicityConfig,
) -> tuple[CellCGMergeLayout, ...]:
    """Compile inter-cell channels with size-shared reduced parameters."""

    spec = cell_schur_spec(config.cell_kind)
    tree = config.cell_tree
    groups: dict[tuple[int, int, bool], int] = {}
    layouts = []
    for node in tree.internal_nodes:
        left_cells = _cell_count(node.left)
        right_cells = _cell_count(node.right)
        left_size = left_cells * spec.cell_size
        right_size = right_cells * spec.cell_size
        is_root = node == tree.root
        group_key = (left_size, right_size, is_root)
        groups.setdefault(group_key, len(groups))
        keys = []
        left_widths = []
        right_widths = []
        for q_left in _reachable_q(left_size, config.maximum_intermediate_q):
            for q_right in _reachable_q(right_size, config.maximum_intermediate_q):
                for q_total in coupling_range(q_left, q_right):
                    if is_root and q_total != config.target_q:
                        continue
                    if (
                        not is_root
                        and config.maximum_intermediate_q is not None
                        and q_total > config.maximum_intermediate_q
                    ):
                        continue
                    keys.append((q_left, q_right, q_total))
                    left_widths.append(
                        config.feature_dim
                        if config.merge_mode == "compressed_shared_tree"
                        or not isinstance(node.left, Leaf)
                        else spec.multiplicity(q_left)
                    )
                    right_widths.append(
                        config.feature_dim
                        if config.merge_mode == "compressed_shared_tree"
                        or not isinstance(node.right, Leaf)
                        else spec.multiplicity(q_right)
                    )
        if not keys or any(width == 0 for width in left_widths + right_widths):
            raise ValueError("angular-momentum cutoff removes a required cell merge")
        layouts.append(
            CellCGMergeLayout(
                left_size,
                right_size,
                groups[group_key],
                tuple(keys),
                tuple(left_widths),
                tuple(right_widths),
            )
        )
    return tuple(layouts)


def _unique_layouts(
    config: SpinCellMultiplicityConfig,
) -> tuple[CellCGMergeLayout, ...]:
    unique: dict[int, CellCGMergeLayout] = {}
    for layout in cell_multiplicity_layouts(config):
        unique.setdefault(layout.parameter_index, layout)
    return tuple(unique[index] for index in range(len(unique)))


def init_spin_cell_multiplicity_tree(
    key: jax.Array, config: SpinCellMultiplicityConfig
) -> dict[str, object]:
    """Initialize reduced maps; local Schur and magnetic CG tensors are fixed."""

    layouts = _unique_layouts(config)
    is_factorized = config.merge_mode == "factorized_rank_one"
    compression_count = (
        len(active_cell_q_values(config))
        if config.merge_mode == "compressed_shared_tree"
        else 0
    )
    array_count = (
        compression_count
        + sum(
            len(layout.keys) * (2 if is_factorized else 1)
            for layout in layouts
        )
        + 1
    )
    keys = iter(jax.random.split(key, array_count))
    scale = float(config.initialization_scale)
    width = config.feature_dim
    local_compressions = None
    if config.merge_mode == "compressed_shared_tree":
        spec = cell_schur_spec(config.cell_kind)
        local_compressions = tuple(
            jax.random.normal(
                next(keys), (width, spec.multiplicity(q)), dtype=jnp.float64
            )
            / np.sqrt(spec.multiplicity(q))
            for q in active_cell_q_values(config)
        )
    merges = []
    for layout in layouts:
        if not is_factorized:
            joint = []
            for left_width, right_width in zip(
                layout.left_widths, layout.right_widths, strict=True
            ):
                joint.append(
                    scale
                    * jax.random.normal(
                        next(keys),
                        (width, left_width, right_width),
                        dtype=jnp.float64,
                    )
                    / np.sqrt(left_width * right_width)
                )
            merges.append({"joint": tuple(joint)})
        else:
            left_factors = []
            right_factors = []
            for left_width, right_width in zip(
                layout.left_widths, layout.right_widths, strict=True
            ):
                left_factors.append(
                    np.sqrt(scale)
                    * jax.random.normal(
                        next(keys), (width, left_width), dtype=jnp.float64
                    )
                    / np.sqrt(left_width)
                )
                right_factors.append(
                    np.sqrt(scale)
                    * jax.random.normal(
                        next(keys), (width, right_width), dtype=jnp.float64
                    )
                    / np.sqrt(right_width)
                )
            merges.append(
                {"left": tuple(left_factors), "right": tuple(right_factors)}
            )
    readout = scale * jax.random.normal(
        next(keys), (width,), dtype=jnp.float64
    )
    params = {"merges": tuple(merges), "readout": readout}
    if local_compressions is not None:
        params["local_compressions"] = local_compressions
    return params


def init_paired_full_and_factorized_cell_multiplicity(
    key: jax.Array, config: SpinCellMultiplicityConfig
) -> tuple[
    SpinCellMultiplicityConfig,
    dict[str, object],
    SpinCellMultiplicityConfig,
    dict[str, object],
]:
    """Initialize same-width full/factorized arms at exactly equal amplitudes."""

    full_config = replace(config, merge_mode="full_joint")
    factor_config = replace(config, merge_mode="factorized_rank_one")
    factor = init_spin_cell_multiplicity_tree(key, factor_config)
    full_merges = []
    for group in factor["merges"]:
        joint = tuple(
            left[:, :, None] * right[:, None, :]
            for left, right in zip(group["left"], group["right"], strict=True)
        )
        full_merges.append({"joint": joint})
    full = {"merges": tuple(full_merges), "readout": factor["readout"]}
    return full_config, full, factor_config, factor


def init_paired_full_and_compressed_cell_multiplicity(
    key: jax.Array, config: SpinCellMultiplicityConfig
) -> tuple[
    SpinCellMultiplicityConfig,
    dict[str, object],
    SpinCellMultiplicityConfig,
    dict[str, object],
]:
    """Embed the shared-cell compressed control exactly into ``full_joint``."""

    full_config = replace(config, merge_mode="full_joint")
    compressed_config = replace(config, merge_mode="compressed_shared_tree")
    compressed = init_spin_cell_multiplicity_tree(key, compressed_config)
    spec = cell_schur_spec(config.cell_kind)
    compressions = {
        q: compressed["local_compressions"][index]
        for index, q in enumerate(active_cell_q_values(compressed_config))
    }
    full_merges = []
    full_layouts = _unique_layouts(full_config)
    compressed_layouts = _unique_layouts(compressed_config)
    for full_layout, compressed_layout, group in zip(
        full_layouts,
        compressed_layouts,
        compressed["merges"],
        strict=True,
    ):
        if full_layout.keys != compressed_layout.keys:
            raise RuntimeError("full and compressed angular channels disagree")
        left_is_cell = full_layout.left_size == spec.cell_size
        right_is_cell = full_layout.right_size == spec.cell_size
        joint = []
        for channel, (q_left, q_right, _) in enumerate(full_layout.keys):
            kernel = group["joint"][channel]
            if left_is_cell and right_is_cell:
                effective = jnp.einsum(
                    "kuv,ua,vb->kab",
                    kernel,
                    compressions[q_left],
                    compressions[q_right],
                )
            elif left_is_cell:
                effective = jnp.einsum(
                    "kuv,ua->kav", kernel, compressions[q_left]
                )
            elif right_is_cell:
                effective = jnp.einsum(
                    "kuv,vb->kub", kernel, compressions[q_right]
                )
            else:
                effective = kernel
            expected_shape = (
                config.feature_dim,
                full_layout.left_widths[channel],
                full_layout.right_widths[channel],
            )
            if effective.shape != expected_shape:
                raise RuntimeError("compressed-to-full kernel has the wrong shape")
            joint.append(effective)
        full_merges.append({"joint": tuple(joint)})
    full = {
        "merges": tuple(full_merges),
        "readout": compressed["readout"],
    }
    return full_config, full, compressed_config, compressed


def init_common_d1_full_compressed_and_factorized_cell_multiplicity(
    key: jax.Array, config: SpinCellMultiplicityConfig
) -> tuple[
    SpinCellMultiplicityConfig,
    dict[str, object],
    SpinCellMultiplicityConfig,
    dict[str, object],
    SpinCellMultiplicityConfig,
    dict[str, object],
]:
    """Return three exactly equal D=1 arms from one compressed-tree state.

    At width one every compressed merge ``B`` is a scalar, so its effective
    kernel is rank one whenever a raw cell multiplicity axis enters.  This
    gives a rare three-way common initialization without padding parameters or
    changing any model definition.
    """

    if config.feature_dim != 1:
        raise ValueError("three-way exact common initialization requires feature_dim=1")
    full_config, full, compressed_config, compressed = (
        init_paired_full_and_compressed_cell_multiplicity(key, config)
    )
    factor_config = replace(config, merge_mode="factorized_rank_one")
    spec = cell_schur_spec(config.cell_kind)
    compressions = {
        q: compressed["local_compressions"][index]
        for index, q in enumerate(active_cell_q_values(compressed_config))
    }
    factor_merges = []
    factor_layouts = _unique_layouts(factor_config)
    compressed_layouts = _unique_layouts(compressed_config)
    for factor_layout, compressed_layout, group in zip(
        factor_layouts,
        compressed_layouts,
        compressed["merges"],
        strict=True,
    ):
        if factor_layout.keys != compressed_layout.keys:
            raise RuntimeError("factorized and compressed angular channels disagree")
        left_is_cell = factor_layout.left_size == spec.cell_size
        right_is_cell = factor_layout.right_size == spec.cell_size
        left_factors = []
        right_factors = []
        for channel, (q_left, q_right, _) in enumerate(factor_layout.keys):
            scalar = group["joint"][channel][0, 0, 0]
            left = (
                compressions[q_left]
                if left_is_cell
                else jnp.ones((1, 1), dtype=jnp.float64)
            )
            right = (
                compressions[q_right]
                if right_is_cell
                else jnp.ones((1, 1), dtype=jnp.float64)
            )
            left_factors.append(left)
            right_factors.append(scalar * right)
        factor_merges.append(
            {"left": tuple(left_factors), "right": tuple(right_factors)}
        )
    factor = {
        "merges": tuple(factor_merges),
        "readout": compressed["readout"],
    }
    return (
        full_config,
        full,
        compressed_config,
        compressed,
        factor_config,
        factor,
    )


def _reduced_merge(
    group: dict[str, object],
    channel: int,
    left: jax.Array,
    right: jax.Array,
    mode: MergeMode,
) -> jax.Array:
    if mode != "factorized_rank_one":
        return jnp.einsum("a,b,kab->k", left, right, group["joint"][channel])
    return (
        jnp.einsum("a,ka->k", left, group["left"][channel])
        * jnp.einsum("b,kb->k", right, group["right"][channel])
    )


def spin_cell_multiplicity_tree_raw_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinCellMultiplicityConfig,
) -> jax.Array:
    """Evaluate one unprojected physical amplitude in fixed ``(S,M)``."""

    states = jnp.asarray(spins, dtype=jnp.int32)
    if states.shape != (config.n_sites,):
        raise ValueError(f"expected spin shape ({config.n_sites},), got {states.shape}")
    valid = jnp.all((states == 0) | (states == 1))
    safe = jnp.clip(states, 0, 1)
    spec = cell_schur_spec(config.cell_kind)
    schur = jnp.asarray(cell_schur_tensor(config.cell_kind), dtype=jnp.float64)
    powers = jnp.asarray(
        1 << np.arange(spec.cell_size - 1, -1, -1), dtype=jnp.int32
    )
    tree = config.cell_tree
    node_values: dict[TreeNode, tuple[jax.Array, jax.Array]] = {}
    for leaf in tree.leaves:
        cell_spins = safe[jnp.asarray(config.cells[leaf.index], dtype=jnp.int32)]
        index = jnp.dot(cell_spins, powers)
        m2 = jnp.sum(2 * cell_spins - 1)
        local = schur[index, :, m2 + spec.cell_size, :]
        if config.merge_mode == "compressed_shared_tree":
            compressed = jnp.zeros(
                (spec.cell_size + 1, config.feature_dim), dtype=jnp.float64
            )
            for q_index, q in enumerate(active_cell_q_values(config)):
                multiplicity = spec.multiplicity(q)
                compressed = compressed.at[q].set(
                    params["local_compressions"][q_index]
                    @ local[q, :multiplicity]
                )
            local = compressed
        node_values[leaf] = (m2, local)

    for node, layout in zip(
        tree.internal_nodes, cell_multiplicity_layouts(config), strict=True
    ):
        left_m2, left_by_q = node_values[node.left]
        right_m2, right_by_q = node_values[node.right]
        group = params["merges"][layout.parameter_index]
        cg = jnp.asarray(spin_cg_tensor(layout.left_size, layout.right_size))
        result = jnp.zeros(
            (layout.size + 1, config.feature_dim), dtype=jnp.float64
        )
        for channel, ((q_left, q_right, q_total), left_width, right_width) in enumerate(
            zip(layout.keys, layout.left_widths, layout.right_widths, strict=True)
        ):
            reduced = _reduced_merge(
                group,
                channel,
                left_by_q[q_left, :left_width],
                right_by_q[q_right, :right_width],
                config.merge_mode,
            )
            coefficient = cg[
                q_left,
                left_m2 + layout.left_size,
                q_right,
                right_m2 + layout.right_size,
                q_total,
            ]
            result = result.at[q_total].add(coefficient * reduced)
        node_values[node] = (left_m2 + right_m2, result)

    total_m2, root_by_q = node_values[tree.root]
    value = jnp.dot(root_by_q[config.target_q], params["readout"])
    return jnp.where(
        valid & (total_m2 == config.target_m2),
        value,
        jnp.asarray(0.0, dtype=value.dtype),
    )


def spin_cell_multiplicity_tree_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinCellMultiplicityConfig,
) -> jax.Array:
    """Evaluate one amplitude, including the optional outer physical projector."""

    if config.projector is None:
        return spin_cell_multiplicity_tree_raw_amplitude(params, spins, config)
    return spatial_projected_amplitude(
        params,
        spins,
        config,
        config.projector,
        amplitude_fn=spin_cell_multiplicity_tree_raw_amplitude,
    )


def spin_cell_multiplicity_tree_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinCellMultiplicityConfig,
) -> jax.Array:
    """Vectorize the cell-multiplicity amplitude over physical states."""

    states = jnp.asarray(spins, dtype=jnp.int32)
    if states.ndim != 2 or states.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    return jax.vmap(
        lambda row: spin_cell_multiplicity_tree_amplitude(params, row, config)
    )(states)


def spin_cell_multiplicity_parameter_count(params: dict[str, object]) -> int:
    """Count stored trainable scalars (not gauge-reduced tangent dimension)."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


def spin_cell_multiplicity_parameter_count_from_config(
    config: SpinCellMultiplicityConfig,
) -> int:
    """Count stored scalars directly from static packed channel shapes."""

    count = config.feature_dim
    if config.merge_mode == "compressed_shared_tree":
        spec = cell_schur_spec(config.cell_kind)
        count += config.feature_dim * sum(
            spec.multiplicity(q) for q in active_cell_q_values(config)
        )
    for layout in _unique_layouts(config):
        for left_width, right_width in zip(
            layout.left_widths, layout.right_widths, strict=True
        ):
            if config.merge_mode != "factorized_rank_one":
                count += config.feature_dim * left_width * right_width
            else:
                count += config.feature_dim * (left_width + right_width)
    return int(count)


def capacity_matched_cell_multiplicity_configs(
    template: SpinCellMultiplicityConfig,
    *,
    maximum_feature_dim: int = 16,
) -> tuple[SpinCellMultiplicityConfig, SpinCellMultiplicityConfig, dict[str, int | float]]:
    """Find the closest scalar-count full/factorized width pair.

    Widths may differ, so this helper is for a parameter-budget control and
    not for the exact-common-initialization nesting test.
    """

    if (
        isinstance(maximum_feature_dim, bool)
        or not isinstance(maximum_feature_dim, Integral)
        or maximum_feature_dim < 1
    ):
        raise ValueError("maximum_feature_dim must be a positive integer")
    candidates = []
    for full_width in range(1, int(maximum_feature_dim) + 1):
        full = replace(template, feature_dim=full_width, merge_mode="full_joint")
        full_count = spin_cell_multiplicity_parameter_count_from_config(full)
        for factor_width in range(1, int(maximum_feature_dim) + 1):
            factor = replace(
                template,
                feature_dim=factor_width,
                merge_mode="factorized_rank_one",
            )
            factor_count = spin_cell_multiplicity_parameter_count_from_config(factor)
            difference = abs(full_count - factor_count)
            candidates.append(
                (
                    difference,
                    max(full_width, factor_width),
                    abs(full_width - factor_width),
                    full_count + factor_count,
                    full,
                    factor,
                    full_count,
                    factor_count,
                )
            )
    _, _, _, _, full, factor, full_count, factor_count = min(
        candidates, key=lambda value: value[:4]
    )
    relative = abs(full_count - factor_count) / max(
        0.5 * (full_count + factor_count), 1.0
    )
    return full, factor, {
        "full_parameter_count": full_count,
        "factorized_parameter_count": factor_count,
        "absolute_difference": abs(full_count - factor_count),
        "relative_difference": float(relative),
    }


def cell_multiplicity_relation_to_ordinary_tree(
    config: SpinCellMultiplicityConfig,
) -> dict[str, object]:
    """State the exact function-class relation to a grouped SpinClebschTree."""

    maximum_local_multiplicity = cell_schur_spec(
        config.cell_kind
    ).maximum_multiplicity
    return {
        "local_schur_transform_is_complete_unitary": True,
        "new_hilbert_sector_created": False,
        "comparison_tree": (
            "ordinary SpinClebschTree with sites reordered cell-contiguously, "
            "the same within-cell fusion topology, inter-cell tree, q cutoff, "
            "feature width, and size-sharing rule"
        ),
        "full_joint_contains_comparison_tree_at_same_width": (
            config.merge_mode == "full_joint"
        ),
        "full_joint_contains_compressed_shared_tree_exactly": (
            config.merge_mode == "full_joint"
        ),
        "compressed_shared_tree_is_primary_ordinary_boundary_control": (
            config.merge_mode == "compressed_shared_tree"
        ),
        "full_joint_is_reparameterization_when_width_at_least_local_rank": (
            config.merge_mode == "full_joint"
            and config.feature_dim >= maximum_local_multiplicity
        ),
        "maximum_local_multiplicity": maximum_local_multiplicity,
        "delayed_compression_is_only_structural_advantage": True,
        "possible_strict_advantage_requires_width_below_local_rank": (
            config.merge_mode == "full_joint"
            and config.feature_dim < maximum_local_multiplicity
        ),
        "factorized_control_is_not_a_superset_of_full_joint": (
            config.merge_mode == "factorized_rank_one"
        ),
        "factorized_control_is_only_an_auxiliary_ablation": (
            config.merge_mode == "factorized_rank_one"
        ),
        "claim_boundary": (
            "At sufficient width this is a symmetry-adapted gauge/parameterization, "
            "not a new variational family. Only a measured accuracy-or-cost gain "
            "at the delayed-compression bottleneck can establish an algorithmic win."
        ),
    }


__all__ = [
    "CellCGMergeLayout",
    "CellKind",
    "CellSchurSpec",
    "MergeMode",
    "SpinCellMultiplicityConfig",
    "active_cell_q_values",
    "capacity_matched_cell_multiplicity_configs",
    "cell_multiplicity_layouts",
    "cell_multiplicity_relation_to_ordinary_tree",
    "cell_schur_spec",
    "cell_schur_tensor",
    "cell_schur_unitarity_error",
    "init_paired_full_and_factorized_cell_multiplicity",
    "init_paired_full_and_compressed_cell_multiplicity",
    "init_common_d1_full_compressed_and_factorized_cell_multiplicity",
    "init_spin_cell_multiplicity_tree",
    "spin_cell_multiplicity_parameter_count",
    "spin_cell_multiplicity_parameter_count_from_config",
    "spin_cell_multiplicity_tree_amplitude",
    "spin_cell_multiplicity_tree_amplitudes",
    "spin_cell_multiplicity_tree_raw_amplitude",
]
