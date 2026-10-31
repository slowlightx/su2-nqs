"""Exact adjacent-singlet initialization for a pair-balanced ClebschTree."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    spin_clebsch_tree_layouts,
    spin_clebsch_tree_parameter_count,
)


def init_spin_clebsch_tree_dimer_product(
    config: SpinClebschTreeConfig,
) -> dict[str, object]:
    """Initialize the exact product of singlets on adjacent tree leaves.

    Every two-leaf node selects ``(q_left,q_right,q)=(1,1,0)``.  Every higher
    node selects ``(0,0,0)`` and propagates reduced feature zero.  The
    function fails if the supplied topology does not expose this channel at
    every merge; it therefore cannot silently reinterpret an arbitrary tree.
    """

    layouts = spin_clebsch_tree_layouts(config)
    unique = {}
    for layout in layouts:
        unique.setdefault(layout.parameter_index, layout)
    width = int(config.feature_dim)
    merges = []
    for parameter_index in range(len(unique)):
        layout = unique[parameter_index]
        target = (1, 1, 0) if (layout.left_size, layout.right_size) == (1, 1) else (0, 0, 0)
        if target not in layout.keys:
            raise ValueError(
                "tree topology does not retain the adjacent-singlet product channel"
            )
        left_width = 1 if layout.left_size == 1 else width
        right_width = 1 if layout.right_size == 1 else width
        block = jnp.zeros(
            (len(layout.keys), width, left_width, right_width),
            dtype=jnp.float64,
        )
        channel = layout.keys.index(target)
        block = block.at[channel, 0, 0, 0].set(1.0)
        merges.append(block)
    readout = jnp.zeros((width,), dtype=jnp.float64).at[0].set(1.0)
    params = {"merges": tuple(merges), "readout": readout}
    if spin_clebsch_tree_parameter_count(params) <= 0:
        raise RuntimeError("dimer-product initializer produced no parameters")
    return params


def adjacent_singlet_product_amplitudes(
    spins: np.ndarray,
) -> np.ndarray:
    """Independent Condon--Shortley product reference, ``0=down, 1=up``."""

    values = np.asarray(spins, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("spins must have shape (batch, even_n_sites)")
    result = np.ones(values.shape[0], dtype=np.float64)
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    for site in range(0, values.shape[1], 2):
        left, right = values[:, site], values[:, site + 1]
        local = np.where(
            (left == 0) & (right == 1),
            -inverse_sqrt_two,
            np.where(
                (left == 1) & (right == 0), inverse_sqrt_two, 0.0
            ),
        )
        result *= local
    return result


__all__ = [
    "adjacent_singlet_product_amplitudes",
    "init_spin_clebsch_tree_dimer_product",
]
