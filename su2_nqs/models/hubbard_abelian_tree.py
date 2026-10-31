"""Fixed-``(Ne,Sz)`` physical Hubbard tree used as the matched CG control."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .hubbard_cg import hubbard_cg_parameter_count, hubbard_cg_tree_schedule
from ..su2.hubbard_cg_irrep import token_quantum_number_tables


@dataclass(frozen=True)
class HubbardAbelianTreeConfig:
    """Sector and multiplicity width of the same-tree Abelian control."""

    n_sites: int
    n_electrons: int
    target_m2: int = 0
    feature_dim: int = 2
    initialization_scale: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.n_sites, bool) or not isinstance(self.n_sites, Integral) or self.n_sites < 1:
            raise ValueError("n_sites must be a positive integer")
        if (
            isinstance(self.n_electrons, bool)
            or not isinstance(self.n_electrons, Integral)
            or not 0 <= self.n_electrons <= 2 * self.n_sites
        ):
            raise ValueError("n_electrons must lie between 0 and 2*n_sites")
        if isinstance(self.target_m2, bool) or not isinstance(self.target_m2, Integral):
            raise TypeError("target_m2 must be a doubled integer")
        if (self.n_electrons + self.target_m2) % 2:
            raise ValueError("target_m2 has the wrong parity for n_electrons")
        n_up = (self.n_electrons + self.target_m2) // 2
        n_down = self.n_electrons - n_up
        if not (0 <= n_up <= self.n_sites and 0 <= n_down <= self.n_sites):
            raise ValueError("the requested fixed-(Ne,Sz) sector is empty")
        if isinstance(self.feature_dim, bool) or not isinstance(self.feature_dim, Integral) or self.feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0.0:
            raise ValueError("initialization_scale must be positive and finite")


@dataclass(frozen=True)
class ActiveAbelianMergeLayout:
    """Packed physical charge/magnetization channels for one tree merge."""

    start: int
    split: int
    stop: int
    keys: tuple[tuple[int, int, int, int], ...]
    lookup: np.ndarray

    @property
    def left_size(self) -> int:
        return self.split - self.start

    @property
    def right_size(self) -> int:
        return self.stop - self.split


@lru_cache(maxsize=None)
def _charge_magnetic_channels(size: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (n_up + n_down, n_up - n_down)
                for n_up in range(size + 1)
                for n_down in range(size + 1)
            }
        )
    )


def _can_reach_target(
    charge: int,
    m2: int,
    outside: int,
    config: HubbardAbelianTreeConfig,
) -> bool:
    target_up = (config.n_electrons + config.target_m2) // 2
    target_down = config.n_electrons - target_up
    current_up = (charge + m2) // 2
    current_down = charge - current_up
    return (
        0 <= target_up - current_up <= outside
        and 0 <= target_down - current_down <= outside
    )


@lru_cache(maxsize=None)
def hubbard_abelian_active_layout(
    config: HubbardAbelianTreeConfig,
) -> tuple[ActiveAbelianMergeLayout, ...]:
    """Compile target-reachable packed channels for the matched tree."""

    layouts = []
    for start, split, stop, _ in hubbard_cg_tree_schedule(config.n_sites):
        left_size = split - start
        right_size = stop - split
        outside = config.n_sites - (stop - start)
        lookup = np.full(
            (
                2 * left_size + 1,
                2 * left_size + 1,
                2 * right_size + 1,
                2 * right_size + 1,
            ),
            -1,
            dtype=np.int32,
        )
        keys = []
        for left_charge, left_m2 in _charge_magnetic_channels(left_size):
            for right_charge, right_m2 in _charge_magnetic_channels(right_size):
                if not _can_reach_target(
                    left_charge + right_charge,
                    left_m2 + right_m2,
                    outside,
                    config,
                ):
                    continue
                lookup[
                    left_charge,
                    left_m2 + left_size,
                    right_charge,
                    right_m2 + right_size,
                ] = len(keys)
                keys.append((left_charge, left_m2, right_charge, right_m2))
        lookup.setflags(write=False)
        layouts.append(
            ActiveAbelianMergeLayout(start, split, stop, tuple(keys), lookup)
        )
    return tuple(layouts)


def init_hubbard_abelian_tree(
    key: jax.Array, config: HubbardAbelianTreeConfig
) -> dict[str, object]:
    """Initialize the packed same-tree fixed-``(Ne,Sz)`` control."""

    layouts = hubbard_abelian_active_layout(config)
    keys = jax.random.split(key, len(layouts) + 2)
    scale = float(config.initialization_scale)
    width = int(config.feature_dim)
    leaves = scale * jax.random.normal(
        keys[0], (config.n_sites, 4, width), dtype=jnp.float64
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


def hubbard_abelian_tree_amplitude(
    params: dict[str, object],
    tokens: jax.Array,
    config: HubbardAbelianTreeConfig,
) -> jax.Array:
    """Evaluate one fixed-charge/fixed-magnetization physical amplitude."""

    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.shape != (config.n_sites,):
        raise ValueError(f"expected token shape ({config.n_sites},), got {tokens.shape}")
    valid_tokens = jnp.all((tokens >= 0) & (tokens <= 3))
    safe_tokens = jnp.clip(tokens, 0, 3)
    charge_table, _, m2_table, _ = (
        jnp.asarray(values, dtype=jnp.int32)
        for values in token_quantum_number_tables()
    )
    nodes: dict[tuple[int, int], tuple[jax.Array, jax.Array, jax.Array]] = {}
    for site in range(config.n_sites):
        token = safe_tokens[site]
        nodes[(site, site + 1)] = (
            charge_table[token],
            m2_table[token],
            params["leaves"][site, token],
        )
    for weights, layout in zip(
        params["merges"], hubbard_abelian_active_layout(config), strict=True
    ):
        left_charge, left_m2, left = nodes[(layout.start, layout.split)]
        right_charge, right_m2, right = nodes[(layout.split, layout.stop)]
        packed_index = jnp.asarray(layout.lookup)[
            left_charge,
            left_m2 + layout.left_size,
            right_charge,
            right_m2 + layout.right_size,
        ]
        safe_index = jnp.maximum(packed_index, 0)
        vector = jnp.einsum("a,b,abk->k", left, right, weights[safe_index])
        vector = jnp.where(packed_index >= 0, vector, jnp.zeros_like(vector))
        nodes[(layout.start, layout.stop)] = (
            left_charge + right_charge,
            left_m2 + right_m2,
            vector,
        )
    total_charge, total_m2, root = nodes[(0, config.n_sites)]
    value = jnp.dot(root, params["readout"])
    valid_sector = (
        valid_tokens
        & (total_charge == config.n_electrons)
        & (total_m2 == config.target_m2)
    )
    return jnp.where(valid_sector, value, jnp.asarray(0.0, dtype=value.dtype))


def hubbard_abelian_parameter_count(params: dict[str, object]) -> int:
    """Count scalar parameters in the Abelian control."""

    return hubbard_cg_parameter_count(params)


__all__ = [
    "ActiveAbelianMergeLayout",
    "HubbardAbelianTreeConfig",
    "hubbard_abelian_active_layout",
    "hubbard_abelian_parameter_count",
    "hubbard_abelian_tree_amplitude",
    "init_hubbard_abelian_tree",
]
