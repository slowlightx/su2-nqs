"""Physical-spin ClebschTree amplitudes with exact total SU(2).

The input is a computational-basis spin configuration.  Fixed numerical
Clebsch--Gordan coefficients carry every magnetic index, while trainable
reduced maps act only on multiplicity features.  Maps are shared by tree
layer (more precisely, by the ordered child sizes), matching the essential
parameter-sharing structure of a balanced ClebschTree.

This is an unnormalised amplitude model.  Exact-small benchmarks may
normalise it exhaustively; scalable calculations must combine it with a
physical-basis sampler.  Bits obey the repository convention ``0=down`` and
``1=up`` and all angular momenta are doubled integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode
from ..su2.quantum_numbers import (
    coupling_range,
    magnetic_numbers,
    validate_m,
    validate_q,
)


@dataclass(frozen=True)
class SpinClebschTreeConfig:
    """Static exact-spin and reduced-width contract.

    ``maximum_intermediate_q`` is a doubled-spin cutoff.  It never truncates
    the requested root irrep; ``None`` retains every reachable intermediate
    angular momentum.
    """

    tree: FusionTree
    target_q: int = 0
    target_m2: int = 0
    feature_dim: int = 4
    maximum_intermediate_q: int | None = None
    initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        if self.tree.n_leaves < 2:
            raise ValueError("a ClebschTree requires at least two leaves")
        validate_q(self.target_q, name="target_q")
        validate_m(self.target_q, self.target_m2, name="target_m2")
        if not self.tree.target_is_reachable(self.target_q):
            raise ValueError("target_q is unreachable on the supplied tree")
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

    @property
    def n_sites(self) -> int:
        return self.tree.n_leaves


@dataclass(frozen=True)
class SpinCGMergeLayout:
    """Packed reachable reduced channels for one tree merge."""

    left_size: int
    right_size: int
    parameter_index: int
    keys: tuple[tuple[int, int, int], ...]
    lookup: np.ndarray

    @property
    def size(self) -> int:
        return self.left_size + self.right_size


@lru_cache(maxsize=None)
def _node_size(node: TreeNode) -> int:
    if isinstance(node, Leaf):
        return 1
    return _node_size(node.left) + _node_size(node.right)


def _reachable_q(size: int, maximum_q: int | None) -> tuple[int, ...]:
    upper = size if maximum_q is None else min(size, int(maximum_q))
    return tuple(range(size % 2, upper + 1, 2))


@lru_cache(maxsize=None)
def spin_clebsch_tree_layouts(
    config: SpinClebschTreeConfig,
) -> tuple[SpinCGMergeLayout, ...]:
    """Compile postorder channels and layer-shared parameter indices."""

    root = config.tree.root
    group_indices: dict[tuple[int, int, bool], int] = {}
    layouts: list[SpinCGMergeLayout] = []
    for node in config.tree.internal_nodes:
        left_size = _node_size(node.left)
        right_size = _node_size(node.right)
        is_root = node == root
        group = (left_size, right_size, is_root)
        if group not in group_indices:
            group_indices[group] = len(group_indices)
        parameter_index = group_indices[group]
        lookup = np.full(
            (left_size + 1, right_size + 1, left_size + right_size + 1),
            -1,
            dtype=np.int32,
        )
        keys: list[tuple[int, int, int]] = []
        for q_left in _reachable_q(left_size, config.maximum_intermediate_q):
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
                    lookup[q_left, q_right, q_total] = len(keys)
                    keys.append((q_left, q_right, q_total))
        if not keys:
            raise ValueError("angular-momentum cutoff removes a required merge")
        lookup.setflags(write=False)
        layouts.append(
            SpinCGMergeLayout(
                left_size,
                right_size,
                parameter_index,
                tuple(keys),
                lookup,
            )
        )
    return tuple(layouts)


@lru_cache(maxsize=None)
def _cg_tensor(left_size: int, right_size: int) -> np.ndarray:
    table = np.zeros(
        (
            left_size + 1,
            2 * left_size + 1,
            right_size + 1,
            2 * right_size + 1,
            left_size + right_size + 1,
        ),
        dtype=np.float64,
    )
    for q_left in range(left_size + 1):
        for m_left in magnetic_numbers(q_left):
            for q_right in range(right_size + 1):
                for m_right in magnetic_numbers(q_right):
                    for q_total in coupling_range(q_left, q_right):
                        table[
                            q_left,
                            m_left + left_size,
                            q_right,
                            m_right + right_size,
                            q_total,
                        ] = clebsch_gordan(
                            q_left,
                            m_left,
                            q_right,
                            m_right,
                            q_total,
                            m_left + m_right,
                        )
    table.setflags(write=False)
    return table


def spin_cg_tensor(left_size: int, right_size: int) -> np.ndarray:
    """Return the shared read-only CG lookup used by physical tree models.

    The two sizes count physical spin-half leaves in the left and right
    subtrees.  Exposing this immutable numerical table lets alternative exact
    Schur front ends reuse the established Condon--Shortley merge engine
    without duplicating the group-theory convention.
    """

    for name, value in (("left_size", left_size), ("right_size", right_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return _cg_tensor(int(left_size), int(right_size))


def init_spin_clebsch_tree(
    key: jax.Array, config: SpinClebschTreeConfig
) -> dict[str, object]:
    """Initialise real reduced maps; magnetic CG tensors remain fixed."""

    layouts = spin_clebsch_tree_layouts(config)
    unique: dict[int, SpinCGMergeLayout] = {}
    for layout in layouts:
        unique.setdefault(layout.parameter_index, layout)
    keys = jax.random.split(key, len(unique) + 1)
    width = int(config.feature_dim)
    scale = float(config.initialization_scale)
    merges = []
    for parameter_index in range(len(unique)):
        layout = unique[parameter_index]
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
    readout = scale * jax.random.normal(
        keys[-1], (width,), dtype=jnp.float64
    )
    return {"merges": tuple(merges), "readout": readout}


def spin_clebsch_tree_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinClebschTreeConfig,
) -> jax.Array:
    """Evaluate one physical-spin amplitude in fixed ``(S,M)``."""

    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (config.n_sites,):
        raise ValueError(
            f"expected spin shape ({config.n_sites},), got {spins.shape}"
        )
    valid_spins = jnp.all((spins == 0) | (spins == 1))
    safe_spins = jnp.clip(spins, 0, 1)
    width = int(config.feature_dim)
    node_values: dict[TreeNode, tuple[jax.Array, jax.Array]] = {}
    for leaf in config.tree.leaves:
        m2 = 2 * safe_spins[leaf.index] - 1
        by_q = jnp.zeros((2, 1), dtype=jnp.float64).at[1, 0].set(1.0)
        node_values[leaf] = (m2, by_q)

    for node, layout in zip(
        config.tree.internal_nodes,
        spin_clebsch_tree_layouts(config),
        strict=True,
    ):
        left_m2, left_by_q = node_values[node.left]
        right_m2, right_by_q = node_values[node.right]
        weights = params["merges"][layout.parameter_index]
        cg = jnp.asarray(_cg_tensor(layout.left_size, layout.right_size))
        # ``layout.keys`` is the packed, target-reachable channel list and has
        # the same order as the first axis of the shared reduced tensor.  A
        # single batched contraction keeps the XLA graph proportional to the
        # number of tree merges rather than statically unrolling one graph
        # fragment for every angular-momentum triple.
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

    total_m2, root_by_q = node_values[config.tree.root]
    value = jnp.dot(root_by_q[config.target_q], params["readout"])
    return jnp.where(
        valid_spins & (total_m2 == config.target_m2),
        value,
        jnp.asarray(0.0, dtype=value.dtype),
    )


def spin_clebsch_tree_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinClebschTreeConfig,
) -> jax.Array:
    """Vectorise :func:`spin_clebsch_tree_amplitude` over configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: spin_clebsch_tree_amplitude(params, row, config)
    )(array)


def spin_clebsch_tree_parameter_count(params: dict[str, object]) -> int:
    """Count scalar trainable parameters in a ClebschTree pytree."""

    return int(
        sum(
            np.prod(leaf.shape)
            for leaf in jax.tree_util.tree_leaves(params)
        )
    )


__all__ = [
    "SpinCGMergeLayout",
    "SpinClebschTreeConfig",
    "init_spin_clebsch_tree",
    "spin_clebsch_tree_amplitude",
    "spin_clebsch_tree_amplitudes",
    "spin_clebsch_tree_layouts",
    "spin_clebsch_tree_parameter_count",
    "spin_cg_tensor",
]
