"""Same-geometry physical-tree controls for spin-1/2 wavefunctions.

The exact-SU(2) :mod:`su2_nqs.models.spin_clebsch_tree` model has two useful
controls which were previously missing from the physical-tree API:

``SpinU1TreeConfig``
    Keeps only doubled magnetization labels and returns zero outside one
    requested ``M2 = 2*Sz`` sector.

``SpinFullTreeConfig``
    Carries no spin quantum number.  It is an ordinary real tree tensor
    network with a fixed two-dimensional physical leg at every leaf.

Both controls consume the same supplied :class:`~su2_nqs.su2.fusion_tree.FusionTree`,
the same explicit physical ``site_order``, and use the same layer-sharing
rule as ``SpinClebschTree``: merge maps are shared by ordered child sizes and
the root has its own group.  ``feature_dim`` therefore defines a genuinely
matched hidden-width arm.  It does *not* match raw parameter count, because
the number of allowed symmetry channels differs.  Use
:func:`spin_control_parameter_budget_bracket` to preregister the two integer
widths bracketing a target parameter count; do not silently call equal width
an equal-parameter comparison.

The amplitudes are unnormalised and intentionally contain no Hamiltonian,
spatial-projector, optimizer, or sampling logic.  Bits follow the repository
convention ``0=down`` and ``1=up``.  Every magnetization is a doubled integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode


SpinControlKind = Literal["u1", "full"]


def _validated_site_order(
    tree: FusionTree, site_order: tuple[int, ...] | None
) -> tuple[int, ...]:
    n_sites = tree.n_leaves
    order = tuple(range(n_sites)) if site_order is None else tuple(site_order)
    if len(order) != n_sites or tuple(sorted(order)) != tuple(range(n_sites)):
        raise ValueError("site_order must contain every physical site once")
    if any(isinstance(site, bool) or not isinstance(site, Integral) for site in order):
        raise TypeError("site_order entries must be integers")
    return tuple(int(site) for site in order)


def _validate_width_and_scale(feature_dim: int, initialization_scale: float) -> None:
    if (
        isinstance(feature_dim, bool)
        or not isinstance(feature_dim, Integral)
        or feature_dim < 1
    ):
        raise ValueError("feature_dim must be a positive integer")
    if not np.isfinite(initialization_scale) or initialization_scale <= 0.0:
        raise ValueError("initialization_scale must be positive and finite")


@dataclass(frozen=True)
class SpinU1TreeConfig:
    """Fixed-magnetization physical tree with no total-spin constraint.

    ``target_m2`` is twice the total ``Sz``.  ``site_order[leaf]`` names the
    physical site contracted into that leaf of ``tree``.  The optional
    ``maximum_intermediate_abs_m2`` is the Abelian analogue of an SU(2)
    intermediate-``q`` cap: it is applied to non-leaf, non-root nodes and can
    be set to the same doubled integer in a cutoff-matched benchmark arm.
    """

    tree: FusionTree
    site_order: tuple[int, ...] | None = None
    target_m2: int = 0
    feature_dim: int = 4
    maximum_intermediate_abs_m2: int | None = None
    initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        if self.tree.n_leaves < 2:
            raise ValueError("a physical control tree requires at least two leaves")
        if isinstance(self.target_m2, bool) or not isinstance(self.target_m2, Integral):
            raise TypeError("target_m2 must be a doubled integer")
        n_sites = self.tree.n_leaves
        if abs(int(self.target_m2)) > n_sites or (n_sites - int(self.target_m2)) % 2:
            raise ValueError("target_m2 is unreachable for the supplied spin count")
        _validate_width_and_scale(self.feature_dim, self.initialization_scale)
        if self.maximum_intermediate_abs_m2 is not None and (
            isinstance(self.maximum_intermediate_abs_m2, bool)
            or not isinstance(self.maximum_intermediate_abs_m2, Integral)
            or self.maximum_intermediate_abs_m2 < 0
        ):
            raise ValueError(
                "maximum_intermediate_abs_m2 must be a non-negative doubled integer"
            )
        object.__setattr__(
            self, "site_order", _validated_site_order(self.tree, self.site_order)
        )

    @property
    def n_sites(self) -> int:
        return self.tree.n_leaves


@dataclass(frozen=True)
class SpinFullTreeConfig:
    """Physical tree with no spin-sector constraint."""

    tree: FusionTree
    site_order: tuple[int, ...] | None = None
    feature_dim: int = 4
    initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        if self.tree.n_leaves < 2:
            raise ValueError("a physical control tree requires at least two leaves")
        _validate_width_and_scale(self.feature_dim, self.initialization_scale)
        object.__setattr__(
            self, "site_order", _validated_site_order(self.tree, self.site_order)
        )

    @property
    def n_sites(self) -> int:
        return self.tree.n_leaves


@dataclass(frozen=True)
class SpinU1MergeLayout:
    """Packed reachable magnetization pairs for one tree merge."""

    left_size: int
    right_size: int
    parameter_index: int
    keys: tuple[tuple[int, int], ...]
    lookup: np.ndarray

    @property
    def size(self) -> int:
        return self.left_size + self.right_size


@dataclass(frozen=True)
class SpinFullMergeLayout:
    """Child sizes and shared parameter index for one unrestricted merge."""

    left_size: int
    right_size: int
    parameter_index: int

    @property
    def size(self) -> int:
        return self.left_size + self.right_size


@dataclass(frozen=True)
class SpinControlBudgetPoint:
    """One integer-width point in a control's raw-parameter ladder."""

    feature_dim: int
    parameter_count: int


@dataclass(frozen=True)
class SpinControlBudgetBracket:
    """Nearest parameter points below and above a requested raw budget."""

    control: SpinControlKind
    target_parameter_count: int
    lower: SpinControlBudgetPoint | None
    upper: SpinControlBudgetPoint | None


@dataclass(frozen=True)
class SpinControlParameterMatch:
    """Deterministic nearest-width audit against a relative raw-P tolerance."""

    control: SpinControlKind
    target_parameter_count: int
    relative_tolerance: float
    closest: SpinControlBudgetPoint
    relative_mismatch: float
    within_tolerance: bool


@lru_cache(maxsize=None)
def _node_size(node: TreeNode) -> int:
    if isinstance(node, Leaf):
        return 1
    return _node_size(node.left) + _node_size(node.right)


def _magnetizations(size: int) -> range:
    return range(-size, size + 1, 2)


def _node_magnetizations(
    node: TreeNode, maximum_intermediate_abs_m2: int | None
) -> tuple[int, ...]:
    size = _node_size(node)
    values = tuple(_magnetizations(size))
    if isinstance(node, Leaf) or maximum_intermediate_abs_m2 is None:
        return values
    cutoff = int(maximum_intermediate_abs_m2)
    return tuple(m2 for m2 in values if abs(m2) <= cutoff)


def _can_reach_target(m2: int, outside: int, target_m2: int) -> bool:
    remainder = target_m2 - m2
    return abs(remainder) <= outside and (outside - remainder) % 2 == 0


@lru_cache(maxsize=None)
def spin_u1_tree_layouts(
    config: SpinU1TreeConfig,
) -> tuple[SpinU1MergeLayout, ...]:
    """Compile target-reachable U(1) channels in tree postorder."""

    groups: dict[tuple[int, int, bool], int] = {}
    layouts: list[SpinU1MergeLayout] = []
    for node in config.tree.internal_nodes:
        left_size = _node_size(node.left)
        right_size = _node_size(node.right)
        size = left_size + right_size
        is_root = node == config.tree.root
        group = (left_size, right_size, is_root)
        if group not in groups:
            groups[group] = len(groups)
        lookup = np.full(
            (2 * left_size + 1, 2 * right_size + 1), -1, dtype=np.int32
        )
        keys: list[tuple[int, int]] = []
        outside = config.n_sites - size
        for left_m2 in _node_magnetizations(
            node.left, config.maximum_intermediate_abs_m2
        ):
            for right_m2 in _node_magnetizations(
                node.right, config.maximum_intermediate_abs_m2
            ):
                total_m2 = left_m2 + right_m2
                if is_root:
                    keep = total_m2 == config.target_m2
                else:
                    keep = _can_reach_target(
                        total_m2, outside, int(config.target_m2)
                    ) and (
                        config.maximum_intermediate_abs_m2 is None
                        or abs(total_m2)
                        <= int(config.maximum_intermediate_abs_m2)
                    )
                if not keep:
                    continue
                lookup[left_m2 + left_size, right_m2 + right_size] = len(keys)
                keys.append((left_m2, right_m2))
        if not keys:
            raise ValueError("target_m2 removes every channel from a required merge")
        lookup.setflags(write=False)
        layouts.append(
            SpinU1MergeLayout(
                left_size,
                right_size,
                groups[group],
                tuple(keys),
                lookup,
            )
        )
    return tuple(layouts)


@lru_cache(maxsize=None)
def spin_full_tree_layouts(
    config: SpinFullTreeConfig,
) -> tuple[SpinFullMergeLayout, ...]:
    """Compile the unrestricted tree schedule in postorder."""

    groups: dict[tuple[int, int, bool], int] = {}
    layouts: list[SpinFullMergeLayout] = []
    for node in config.tree.internal_nodes:
        left_size = _node_size(node.left)
        right_size = _node_size(node.right)
        group = (left_size, right_size, node == config.tree.root)
        if group not in groups:
            groups[group] = len(groups)
        layouts.append(
            SpinFullMergeLayout(left_size, right_size, groups[group])
        )
    return tuple(layouts)


def _unique_u1_layouts(
    config: SpinU1TreeConfig,
) -> tuple[SpinU1MergeLayout, ...]:
    unique: dict[int, SpinU1MergeLayout] = {}
    for layout in spin_u1_tree_layouts(config):
        unique.setdefault(layout.parameter_index, layout)
    return tuple(unique[index] for index in range(len(unique)))


def _unique_full_layouts(
    config: SpinFullTreeConfig,
) -> tuple[SpinFullMergeLayout, ...]:
    unique: dict[int, SpinFullMergeLayout] = {}
    for layout in spin_full_tree_layouts(config):
        unique.setdefault(layout.parameter_index, layout)
    return tuple(unique[index] for index in range(len(unique)))


def init_spin_u1_tree(
    key: jax.Array, config: SpinU1TreeConfig
) -> dict[str, object]:
    """Initialise the fixed-``M2`` tree; physical leaf scalars are fixed."""

    layouts = _unique_u1_layouts(config)
    keys = jax.random.split(key, len(layouts) + 1)
    width = int(config.feature_dim)
    scale = float(config.initialization_scale)
    merges = []
    for index, layout in enumerate(layouts):
        left_width = 1 if layout.left_size == 1 else width
        right_width = 1 if layout.right_size == 1 else width
        fan_in = max(left_width * right_width, 1)
        merges.append(
            (scale / np.sqrt(fan_in))
            * jax.random.normal(
                keys[index],
                (len(layout.keys), width, left_width, right_width),
                dtype=jnp.float64,
            )
        )
    readout = scale * jax.random.normal(keys[-1], (width,), dtype=jnp.float64)
    return {"merges": tuple(merges), "readout": readout}


def init_spin_full_tree(
    key: jax.Array, config: SpinFullTreeConfig
) -> dict[str, object]:
    """Initialise an unrestricted physical-leg tree tensor network."""

    layouts = _unique_full_layouts(config)
    keys = jax.random.split(key, len(layouts) + 1)
    width = int(config.feature_dim)
    scale = float(config.initialization_scale)
    merges = []
    for index, layout in enumerate(layouts):
        left_width = 2 if layout.left_size == 1 else width
        right_width = 2 if layout.right_size == 1 else width
        fan_in = max(left_width * right_width, 1)
        merges.append(
            (scale / np.sqrt(fan_in))
            * jax.random.normal(
                keys[index],
                (width, left_width, right_width),
                dtype=jnp.float64,
            )
        )
    readout = scale * jax.random.normal(keys[-1], (width,), dtype=jnp.float64)
    return {"merges": tuple(merges), "readout": readout}


def spin_u1_tree_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinU1TreeConfig,
) -> jax.Array:
    """Evaluate one physical configuration in exactly one ``M2`` sector."""

    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (config.n_sites,):
        raise ValueError(f"expected spin shape ({config.n_sites},), got {spins.shape}")
    valid_spins = jnp.all((spins == 0) | (spins == 1))
    safe_spins = jnp.clip(spins, 0, 1)
    ordered = safe_spins[jnp.asarray(config.site_order, dtype=jnp.int32)]
    nodes: dict[TreeNode, tuple[jax.Array, jax.Array]] = {}
    for leaf in config.tree.leaves:
        nodes[leaf] = (
            2 * ordered[leaf.index] - 1,
            jnp.ones((1,), dtype=jnp.float64),
        )

    for node, layout in zip(
        config.tree.internal_nodes, spin_u1_tree_layouts(config), strict=True
    ):
        left_m2, left = nodes[node.left]
        right_m2, right = nodes[node.right]
        packed = jnp.asarray(layout.lookup)[
            left_m2 + layout.left_size, right_m2 + layout.right_size
        ]
        safe_index = jnp.maximum(packed, 0)
        weights = params["merges"][layout.parameter_index]
        vector = jnp.einsum("a,b,kab->k", left, right, weights[safe_index])
        vector = jnp.where(packed >= 0, vector, jnp.zeros_like(vector))
        nodes[node] = (left_m2 + right_m2, vector)

    total_m2, root = nodes[config.tree.root]
    value = jnp.dot(root, params["readout"])
    return jnp.where(
        valid_spins & (total_m2 == config.target_m2),
        value,
        jnp.asarray(0.0, dtype=value.dtype),
    )


def spin_full_tree_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinFullTreeConfig,
) -> jax.Array:
    """Evaluate one configuration without imposing a spin quantum number."""

    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (config.n_sites,):
        raise ValueError(f"expected spin shape ({config.n_sites},), got {spins.shape}")
    valid_spins = jnp.all((spins == 0) | (spins == 1))
    safe_spins = jnp.clip(spins, 0, 1)
    ordered = safe_spins[jnp.asarray(config.site_order, dtype=jnp.int32)]
    nodes: dict[TreeNode, jax.Array] = {
        leaf: jax.nn.one_hot(
            ordered[leaf.index], 2, dtype=jnp.float64
        )
        for leaf in config.tree.leaves
    }
    for node, layout in zip(
        config.tree.internal_nodes, spin_full_tree_layouts(config), strict=True
    ):
        weights = params["merges"][layout.parameter_index]
        nodes[node] = jnp.einsum(
            "a,b,kab->k", nodes[node.left], nodes[node.right], weights
        )
    value = jnp.dot(nodes[config.tree.root], params["readout"])
    return jnp.where(valid_spins, value, jnp.asarray(0.0, dtype=value.dtype))


def spin_u1_tree_amplitudes(
    params: dict[str, object], spins: jax.Array, config: SpinU1TreeConfig
) -> jax.Array:
    """Vectorise :func:`spin_u1_tree_amplitude` over configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    return jax.vmap(lambda row: spin_u1_tree_amplitude(params, row, config))(array)


def spin_full_tree_amplitudes(
    params: dict[str, object], spins: jax.Array, config: SpinFullTreeConfig
) -> jax.Array:
    """Vectorise :func:`spin_full_tree_amplitude` over configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    return jax.vmap(lambda row: spin_full_tree_amplitude(params, row, config))(array)


def spin_control_tree_parameter_count(params: dict[str, object]) -> int:
    """Count real trainable scalars in either physical-tree control."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


def spin_u1_tree_parameter_count_from_config(config: SpinU1TreeConfig) -> int:
    """Return the U(1) raw count without random initialisation."""

    width = int(config.feature_dim)
    count = width
    for layout in _unique_u1_layouts(config):
        left_width = 1 if layout.left_size == 1 else width
        right_width = 1 if layout.right_size == 1 else width
        count += len(layout.keys) * width * left_width * right_width
    return int(count)


def spin_full_tree_parameter_count_from_config(config: SpinFullTreeConfig) -> int:
    """Return the unrestricted raw count without random initialisation."""

    width = int(config.feature_dim)
    count = width
    for layout in _unique_full_layouts(config):
        left_width = 2 if layout.left_size == 1 else width
        right_width = 2 if layout.right_size == 1 else width
        count += width * left_width * right_width
    return int(count)


def spin_control_parameter_budget_bracket(
    *,
    control: SpinControlKind,
    tree: FusionTree,
    site_order: tuple[int, ...],
    target_parameter_count: int,
    target_m2: int = 0,
    maximum_intermediate_abs_m2: int | None = None,
    maximum_feature_dim: int = 256,
) -> SpinControlBudgetBracket:
    """Bracket a raw parameter target with integer hidden widths.

    Fair benchmark runners should either execute both returned points or use
    a preregistered one-sided rule.  The function deliberately does not pick
    a single "closest" arm after observing energies.
    """

    if control not in ("u1", "full"):
        raise ValueError("control must be 'u1' or 'full'")
    if (
        isinstance(target_parameter_count, bool)
        or not isinstance(target_parameter_count, Integral)
        or target_parameter_count < 1
    ):
        raise ValueError("target_parameter_count must be a positive integer")
    if (
        isinstance(maximum_feature_dim, bool)
        or not isinstance(maximum_feature_dim, Integral)
        or maximum_feature_dim < 1
    ):
        raise ValueError("maximum_feature_dim must be a positive integer")

    points: list[SpinControlBudgetPoint] = []
    for width in range(1, int(maximum_feature_dim) + 1):
        if control == "u1":
            config = SpinU1TreeConfig(
                tree,
                site_order,
                target_m2=target_m2,
                feature_dim=width,
                maximum_intermediate_abs_m2=maximum_intermediate_abs_m2,
            )
            count = spin_u1_tree_parameter_count_from_config(config)
        else:
            config = SpinFullTreeConfig(tree, site_order, feature_dim=width)
            count = spin_full_tree_parameter_count_from_config(config)
        points.append(SpinControlBudgetPoint(width, count))

    below = [point for point in points if point.parameter_count <= target_parameter_count]
    above = [point for point in points if point.parameter_count >= target_parameter_count]
    lower = max(below, key=lambda point: point.parameter_count, default=None)
    upper = min(above, key=lambda point: point.parameter_count, default=None)
    return SpinControlBudgetBracket(
        control,
        int(target_parameter_count),
        lower,
        upper,
    )


def spin_control_parameter_match(
    *,
    control: SpinControlKind,
    tree: FusionTree,
    site_order: tuple[int, ...],
    target_parameter_count: int,
    target_m2: int = 0,
    maximum_intermediate_abs_m2: int | None = None,
    maximum_feature_dim: int = 256,
    relative_tolerance: float = 0.05,
) -> SpinControlParameterMatch:
    """Audit whether an integer-width control matches raw ``P`` closely enough.

    The closest point is selected only by raw parameter count, with a tie
    resolved toward the no-more-parameters point and then smaller width.
    This deterministic rule is safe to freeze before any energy is observed.
    ``within_tolerance=False`` must be treated as a failed parameter match,
    not rounded into an equal-budget claim.
    """

    if not np.isfinite(relative_tolerance) or not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be finite and lie in [0, 1)")
    bracket = spin_control_parameter_budget_bracket(
        control=control,
        tree=tree,
        site_order=site_order,
        target_parameter_count=target_parameter_count,
        target_m2=target_m2,
        maximum_intermediate_abs_m2=maximum_intermediate_abs_m2,
        maximum_feature_dim=maximum_feature_dim,
    )
    candidates = [point for point in (bracket.lower, bracket.upper) if point is not None]
    if not candidates:
        raise RuntimeError("parameter-budget scan unexpectedly produced no points")
    target = int(target_parameter_count)
    closest = min(
        candidates,
        key=lambda point: (
            abs(point.parameter_count - target),
            point.parameter_count > target,
            point.feature_dim,
        ),
    )
    mismatch = abs(closest.parameter_count - target) / target
    return SpinControlParameterMatch(
        control,
        target,
        float(relative_tolerance),
        closest,
        float(mismatch),
        bool(mismatch <= relative_tolerance),
    )


__all__ = [
    "SpinControlBudgetBracket",
    "SpinControlBudgetPoint",
    "SpinControlKind",
    "SpinControlParameterMatch",
    "SpinFullMergeLayout",
    "SpinFullTreeConfig",
    "SpinU1MergeLayout",
    "SpinU1TreeConfig",
    "init_spin_full_tree",
    "init_spin_u1_tree",
    "spin_control_parameter_budget_bracket",
    "spin_control_parameter_match",
    "spin_control_tree_parameter_count",
    "spin_full_tree_amplitude",
    "spin_full_tree_amplitudes",
    "spin_full_tree_layouts",
    "spin_full_tree_parameter_count_from_config",
    "spin_u1_tree_amplitude",
    "spin_u1_tree_amplitudes",
    "spin_u1_tree_layouts",
    "spin_u1_tree_parameter_count_from_config",
]
