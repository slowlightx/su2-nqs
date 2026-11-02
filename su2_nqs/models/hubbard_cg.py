"""Dense, exact-SU(2) Hubbard Clebsch--Gordan tree amplitudes.

This is the correctness-first evaluator for the HubbardCG project.  Physical
Fock configurations are leaves of a deterministic balanced binary tree.
Numerical Clebsch--Gordan tensors carry every magnetic index while trainable
maps act only on charge-resolved reduced multiplicity features.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.hubbard_cg_irrep import token_quantum_number_tables
from ..su2.quantum_numbers import coupling_range, magnetic_numbers, validate_m, validate_q


@dataclass(frozen=True)
class HubbardCGConfig:
    """Static sector and reduced-width contract for a HubbardCG tree."""

    n_sites: int
    n_electrons: int
    target_q: int = 0
    target_m2: int = 0
    feature_dim: int = 2
    initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        if isinstance(self.n_sites, bool) or not isinstance(self.n_sites, Integral) or self.n_sites < 1:
            raise ValueError("n_sites must be a positive integer")
        if (
            isinstance(self.n_electrons, bool)
            or not isinstance(self.n_electrons, Integral)
            or not 0 <= self.n_electrons <= 2 * self.n_sites
        ):
            raise ValueError("n_electrons must lie between 0 and 2*n_sites")
        validate_q(self.target_q, name="target_q")
        validate_m(self.target_q, self.target_m2, name="target_m2")
        if self.target_q > min(self.n_electrons, 2 * self.n_sites - self.n_electrons):
            raise ValueError("target_q is absent from the requested charge sector")
        if (self.n_electrons - self.target_q) % 2:
            raise ValueError("target_q has the wrong parity for n_electrons")
        if isinstance(self.feature_dim, bool) or not isinstance(self.feature_dim, Integral) or self.feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0.0:
            raise ValueError("initialization_scale must be positive and finite")


@dataclass(frozen=True)
class _TreeNode:
    start: int
    stop: int
    parameter_index: int | None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def is_leaf(self) -> bool:
        return self.stop - self.start == 1


@lru_cache(maxsize=None)
def _balanced_tree(n_sites: int) -> tuple[_TreeNode, tuple[_TreeNode, ...]]:
    merges: list[_TreeNode] = []

    def build(start: int, stop: int) -> _TreeNode:
        if stop - start == 1:
            return _TreeNode(start, stop, None)
        middle = start + (stop - start) // 2
        left = build(start, middle)
        right = build(middle, stop)
        node = _TreeNode(start, stop, len(merges), left, right)
        merges.append(node)
        return node

    root = build(0, n_sites)
    return root, tuple(merges)


def hubbard_cg_tree_schedule(n_sites: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return postorder ``(start, split, stop, parameter_index)`` merges."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    _, merges = _balanced_tree(int(n_sites))
    return tuple(
        (node.start, node.left.stop, node.stop, int(node.parameter_index))
        for node in merges
        if node.left is not None
    )


@dataclass(frozen=True)
class ActiveCGMergeLayout:
    """Packed reachable reduced channels for one deterministic tree merge."""

    start: int
    split: int
    stop: int
    keys: tuple[tuple[int, int, int, int, int], ...]
    lookup: np.ndarray

    @property
    def left_size(self) -> int:
        return self.split - self.start

    @property
    def right_size(self) -> int:
        return self.stop - self.split

    @property
    def size(self) -> int:
        return self.stop - self.start


@lru_cache(maxsize=None)
def _charge_spin_channels(size: int) -> tuple[tuple[int, int], ...]:
    channels = []
    for charge in range(2 * size + 1):
        maximum_singles = min(charge, 2 * size - charge)
        channels.extend(
            (charge, q) for q in range(charge % 2, maximum_singles + 1, 2)
        )
    return tuple(channels)


@lru_cache(maxsize=None)
def hubbard_cg_active_layout(
    config: HubbardCGConfig,
) -> tuple[ActiveCGMergeLayout, ...]:
    """Compile packed charge/spin channels that can reach the target charge."""

    layouts = []
    for start, split, stop, _ in hubbard_cg_tree_schedule(config.n_sites):
        left_size = split - start
        right_size = stop - split
        size = stop - start
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
        is_root = start == 0 and stop == config.n_sites
        for left_charge, q_left in _charge_spin_channels(left_size):
            for right_charge, q_right in _charge_spin_channels(right_size):
                total_charge = left_charge + right_charge
                if not minimum_charge <= total_charge <= maximum_charge:
                    continue
                for q_total in coupling_range(q_left, q_right):
                    if is_root and (
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
            ActiveCGMergeLayout(start, split, stop, tuple(keys), lookup)
        )
    return tuple(layouts)


@lru_cache(maxsize=None)
def _cg_tensor(left_size: int, right_size: int) -> np.ndarray:
    """Precompute all CG values needed at one pair of subtree sizes."""

    total_size = left_size + right_size
    table = np.zeros(
        (
            left_size + 1,
            2 * left_size + 1,
            right_size + 1,
            2 * right_size + 1,
            total_size + 1,
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


@lru_cache(maxsize=None)
def _coupled_q_indices(
    left_size: int, right_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return packed doubled-spin triples for one pair of subtree sizes.

    Charge reachability remains encoded by :class:`ActiveCGMergeLayout`.
    These arrays only replace Python-static angular-momentum loops by one
    vectorized gather/contract/scatter operation at each tree merge.
    """

    triples = tuple(
        (q_left, q_right, q_total)
        for q_left in range(left_size + 1)
        for q_right in range(right_size + 1)
        for q_total in coupling_range(q_left, q_right)
    )
    arrays = tuple(
        np.asarray([triple[index] for triple in triples], dtype=np.int32)
        for index in range(3)
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def init_hubbard_cg(key: jax.Array, config: HubbardCGConfig) -> dict[str, object]:
    """Initialize real reduced tensors; all magnetic CG tensors stay fixed."""

    _, merges = _balanced_tree(config.n_sites)
    keys = jax.random.split(key, config.n_sites + len(merges) + 1)
    scale = float(config.initialization_scale)
    width = int(config.feature_dim)
    leaves = scale * jax.random.normal(
        keys[0], (config.n_sites, 3, width), dtype=jnp.float64
    )
    merge_parameters = []
    for key_index, node in enumerate(merges, start=1):
        assert node.left is not None and node.right is not None
        left_size = node.left.size
        right_size = node.right.size
        shape = (
            2 * left_size + 1,
            left_size + 1,
            2 * right_size + 1,
            right_size + 1,
            node.size + 1,
            width,
            width,
            width,
        )
        merge_parameters.append(
            (scale / width)
            * jax.random.normal(keys[key_index], shape, dtype=jnp.float64)
        )
    readout = scale * jax.random.normal(
        keys[-1], (2 * config.n_sites + 1, config.n_sites + 1, width), dtype=jnp.float64
    )
    return {"leaves": leaves, "merges": tuple(merge_parameters), "readout": readout}


def init_active_hubbard_cg(
    key: jax.Array, config: HubbardCGConfig
) -> dict[str, object]:
    """Initialize only target-charge-reachable reduced channel tensors."""

    layouts = hubbard_cg_active_layout(config)
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


def pack_hubbard_cg_parameters(
    dense_params: dict[str, object], config: HubbardCGConfig
) -> dict[str, object]:
    """Pack reachable values from the dense correctness evaluator exactly."""

    layouts = hubbard_cg_active_layout(config)
    packed_merges = []
    for weights, layout in zip(dense_params["merges"], layouts, strict=True):
        packed_merges.append(
            jnp.stack(
                [weights[left_charge, q_left, right_charge, q_right, q_total]
                 for left_charge, q_left, right_charge, q_right, q_total in layout.keys]
            )
        )
    return {
        "leaves": jnp.asarray(dense_params["leaves"]),
        "merges": tuple(packed_merges),
        "readout": jnp.asarray(dense_params["readout"])[
            config.n_electrons, config.target_q
        ],
    }


def _evaluate_node(
    params: dict[str, object],
    tokens: jax.Array,
    node: _TreeNode,
    charge_table: jax.Array,
    q_table: jax.Array,
    m2_table: jax.Array,
    reduced_table: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    width = params["readout"].shape[-1]
    if node.is_leaf:
        token = tokens[node.start]
        charge = charge_table[token]
        q = q_table[token]
        m2 = m2_table[token]
        vector = params["leaves"][node.start, reduced_table[token]]
        by_q = jnp.zeros((2, width), dtype=vector.dtype).at[q].set(vector)
        return charge, m2, by_q

    assert node.left is not None and node.right is not None and node.parameter_index is not None
    left_charge, left_m2, left_by_q = _evaluate_node(
        params, tokens, node.left, charge_table, q_table, m2_table, reduced_table
    )
    right_charge, right_m2, right_by_q = _evaluate_node(
        params, tokens, node.right, charge_table, q_table, m2_table, reduced_table
    )
    weights = params["merges"][node.parameter_index]
    cg = jnp.asarray(_cg_tensor(node.left.size, node.right.size))
    result = jnp.zeros(
        (node.size + 1, width),
        dtype=jnp.result_type(left_by_q, right_by_q, weights),
    )
    for q_left in range(node.left.size + 1):
        for q_right in range(node.right.size + 1):
            reduced_product = jnp.einsum(
                "a,b,qabk->qk",
                left_by_q[q_left],
                right_by_q[q_right],
                weights[left_charge, q_left, right_charge, q_right],
            )
            for q_total in coupling_range(q_left, q_right):
                coefficient = cg[
                    q_left,
                    left_m2 + node.left.size,
                    q_right,
                    right_m2 + node.right.size,
                    q_total,
                ]
                result = result.at[q_total].add(coefficient * reduced_product[q_total])
    return left_charge + right_charge, left_m2 + right_m2, result


def hubbard_cg_amplitude(
    params: dict[str, object],
    tokens: jax.Array,
    config: HubbardCGConfig,
) -> jax.Array:
    """Evaluate one physical Fock amplitude in the fixed ``(Ne,S,M)`` sector."""

    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.shape != (config.n_sites,):
        raise ValueError(f"expected token shape ({config.n_sites},), got {tokens.shape}")
    valid_tokens = jnp.all((tokens >= 0) & (tokens <= 3))
    safe_tokens = jnp.clip(tokens, 0, 3)
    charge, q, m2, reduced = (
        jnp.asarray(values, dtype=jnp.int32) for values in token_quantum_number_tables()
    )
    root, _ = _balanced_tree(config.n_sites)
    total_charge, total_m2, by_q = _evaluate_node(
        params, safe_tokens, root, charge, q, m2, reduced
    )
    value = jnp.dot(
        by_q[config.target_q],
        params["readout"][total_charge, config.target_q],
    )
    valid_sector = (
        valid_tokens
        & (total_charge == config.n_electrons)
        & (total_m2 == config.target_m2)
    )
    return jnp.where(valid_sector, value, jnp.asarray(0.0, dtype=value.dtype))


def active_hubbard_cg_amplitude(
    params: dict[str, object],
    tokens: jax.Array,
    config: HubbardCGConfig,
) -> jax.Array:
    """Evaluate the packed evaluator exactly equivalent to the dense model."""

    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.shape != (config.n_sites,):
        raise ValueError(f"expected token shape ({config.n_sites},), got {tokens.shape}")
    valid_tokens = jnp.all((tokens >= 0) & (tokens <= 3))
    safe_tokens = jnp.clip(tokens, 0, 3)
    charge_table, q_table, m2_table, reduced_table = (
        jnp.asarray(values, dtype=jnp.int32)
        for values in token_quantum_number_tables()
    )
    nodes: dict[tuple[int, int], tuple[jax.Array, jax.Array, jax.Array]] = {}
    width = params["readout"].shape[-1]
    for site in range(config.n_sites):
        token = safe_tokens[site]
        vector = params["leaves"][site, reduced_table[token]]
        by_q = jnp.zeros((2, width), dtype=vector.dtype).at[q_table[token]].set(vector)
        nodes[(site, site + 1)] = (charge_table[token], m2_table[token], by_q)
    for weights, layout in zip(
        params["merges"], hubbard_cg_active_layout(config), strict=True
    ):
        left_charge, left_m2, left_by_q = nodes[(layout.start, layout.split)]
        right_charge, right_m2, right_by_q = nodes[(layout.split, layout.stop)]
        lookup = jnp.asarray(layout.lookup)
        cg = jnp.asarray(_cg_tensor(layout.left_size, layout.right_size))
        q_left, q_right, q_total = (
            jnp.asarray(values)
            for values in _coupled_q_indices(
                layout.left_size, layout.right_size
            )
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
        contribution = jnp.where(
            valid_channel[:, None],
            coefficient[:, None] * reduced_product,
            jnp.zeros_like(reduced_product),
        )
        result = jnp.zeros(
            (layout.size + 1, width),
            dtype=jnp.result_type(left_by_q, right_by_q, weights),
        )
        result = result.at[q_total].add(contribution)
        nodes[(layout.start, layout.stop)] = (
            left_charge + right_charge,
            left_m2 + right_m2,
            result,
        )
    total_charge, total_m2, root_by_q = nodes[(0, config.n_sites)]
    value = jnp.dot(root_by_q[config.target_q], params["readout"])
    valid_sector = (
        valid_tokens
        & (total_charge == config.n_electrons)
        & (total_m2 == config.target_m2)
    )
    return jnp.where(valid_sector, value, jnp.asarray(0.0, dtype=value.dtype))


def hubbard_cg_parameter_count(params: dict[str, object]) -> int:
    """Count scalar trainable parameters in a HubbardCG pytree."""

    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params)))


__all__ = [
    "ActiveCGMergeLayout",
    "HubbardCGConfig",
    "active_hubbard_cg_amplitude",
    "hubbard_cg_active_layout",
    "hubbard_cg_amplitude",
    "hubbard_cg_parameter_count",
    "hubbard_cg_tree_schedule",
    "init_active_hubbard_cg",
    "init_hubbard_cg",
    "pack_hubbard_cg_parameters",
]
