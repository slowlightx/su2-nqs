"""Route-tagged amplitudes for a physical-spin ClebschTree.

This module refines a fixed fusion hierarchy into two *coordinate-dependent*
amplitude components.  ``never_high`` contains fusion histories whose internal
doubled spins never exceed ``retained_q``; ``seen_high`` contains histories
that visit at least one larger internal doubled spin.  Their sum is the
ordinary ClebschTree amplitude.

The split is a diagnostic of one registered tree and its Condon--Shortley
fusion coordinates.  It is not a basis-independent physical sector.  In
particular, a subsequent lattice-symmetry projector need not leave the two
components orthogonal.
"""

from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import TreeNode
from ..su2.quantum_numbers import validate_q
from .spin_clebsch_tree import (
    SpinClebschTreeConfig,
    spin_cg_tensor,
    spin_clebsch_tree_layouts,
)


def _validate_retained_q(retained_q: int) -> int:
    if isinstance(retained_q, bool) or not isinstance(retained_q, Integral):
        raise ValueError("retained_q must be a doubled-integer spin label")
    value = int(retained_q)
    validate_q(value, name="retained_q")
    return value


def _contract_channels(
    left: jax.Array, right: jax.Array, weights: jax.Array
) -> jax.Array:
    """Contract multiplicity features for all packed channels."""

    return jnp.einsum("ca,cb,ckab->ck", left, right, weights)


def spin_clebsch_tree_route_parameter_masks(
    params: dict[str, object],
    config: SpinClebschTreeConfig,
    *,
    retained_q: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return disjoint ``(retained, added-route)`` scalar parameter masks.

    ``added-route`` marks entries belonging to packed merge keys with any
    doubled label above ``retained_q``.  The readout is retained.  These masks
    identify which scalar coordinates were introduced by the larger channel
    catalog; they do *not* imply that the high-route amplitude is independent
    of retained coordinates.
    """

    cutoff = _validate_retained_q(retained_q)
    layouts = spin_clebsch_tree_layouts(config)
    unique = {layout.parameter_index: layout for layout in layouts}
    merges = params.get("merges")
    readout = params.get("readout")
    if not isinstance(merges, (tuple, list)) or len(merges) != len(unique):
        raise ValueError("params merge registry does not match the tree layout")
    if readout is None:
        raise ValueError("params must contain a readout")

    retained_merges = []
    added_merges = []
    for parameter_index, weights in enumerate(merges):
        array = jnp.asarray(weights)
        layout = unique[parameter_index]
        if array.ndim != 4 or array.shape[0] != len(layout.keys):
            raise ValueError("params merge shape does not match packed channels")
        row_is_added = np.asarray(
            [max(key) > cutoff for key in layout.keys], dtype=bool
        )
        added = jnp.broadcast_to(
            jnp.asarray(row_is_added)[:, None, None, None], array.shape
        )
        added_merges.append(added)
        retained_merges.append(~added)

    readout_array = jnp.asarray(readout)
    retained = {
        "merges": tuple(retained_merges),
        "readout": jnp.ones(readout_array.shape, dtype=bool),
    }
    added = {
        "merges": tuple(added_merges),
        "readout": jnp.zeros(readout_array.shape, dtype=bool),
    }
    return retained, added


def spin_clebsch_tree_route_tagged_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinClebschTreeConfig,
    *,
    retained_q: int,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(never_high, seen_high)`` for one spin configuration.

    All angular momenta are doubled integers.  A route is tagged ``seen_high``
    when any *internal merge channel* has ``q_left``, ``q_right``, or
    ``q_total`` larger than ``retained_q``.  Physical leaf labels are not
    themselves treated as truncation decisions.

    The two outputs use the same trainable tensors and exact CG coefficients
    as :func:`spin_clebsch_tree_amplitude`; no subtraction is used to obtain
    the high-route component.
    """

    cutoff = _validate_retained_q(retained_q)
    spins = jnp.asarray(spins, dtype=jnp.int32)
    if spins.shape != (config.n_sites,):
        raise ValueError(
            f"expected spin shape ({config.n_sites},), got {spins.shape}"
        )
    valid_spins = jnp.all((spins == 0) | (spins == 1))
    safe_spins = jnp.clip(spins, 0, 1)
    width = int(config.feature_dim)

    node_values: dict[TreeNode, tuple[jax.Array, jax.Array, jax.Array]] = {}
    for leaf in config.tree.leaves:
        m2 = 2 * safe_spins[leaf.index] - 1
        never = jnp.zeros((2, 1), dtype=jnp.float64).at[1, 0].set(1.0)
        seen = jnp.zeros_like(never)
        node_values[leaf] = (m2, never, seen)

    for node, layout in zip(
        config.tree.internal_nodes,
        spin_clebsch_tree_layouts(config),
        strict=True,
    ):
        left_m2, left_never, left_seen = node_values[node.left]
        right_m2, right_never, right_seen = node_values[node.right]
        weights = params["merges"][layout.parameter_index]
        channels = jnp.asarray(layout.keys, dtype=jnp.int32)
        q_left = channels[:, 0]
        q_right = channels[:, 1]
        q_total = channels[:, 2]

        reduced_nn = _contract_channels(
            left_never[q_left], right_never[q_right], weights
        )
        reduced_ns = _contract_channels(
            left_never[q_left], right_seen[q_right], weights
        )
        reduced_sn = _contract_channels(
            left_seen[q_left], right_never[q_right], weights
        )
        reduced_ss = _contract_channels(
            left_seen[q_left], right_seen[q_right], weights
        )
        route_is_high = (
            (q_left > cutoff) | (q_right > cutoff) | (q_total > cutoff)
        )
        reduced_never = jnp.where(route_is_high[:, None], 0.0, reduced_nn)
        inherited_seen = reduced_ns + reduced_sn + reduced_ss
        reduced_seen = inherited_seen + jnp.where(
            route_is_high[:, None], reduced_nn, 0.0
        )

        cg = jnp.asarray(spin_cg_tensor(layout.left_size, layout.right_size))
        coefficient = cg[
            q_left,
            left_m2 + layout.left_size,
            q_right,
            right_m2 + layout.right_size,
            q_total,
        ]
        never = jnp.zeros((layout.size + 1, width), dtype=jnp.float64)
        seen = jnp.zeros_like(never)
        never = never.at[q_total].add(coefficient[:, None] * reduced_never)
        seen = seen.at[q_total].add(coefficient[:, None] * reduced_seen)
        node_values[node] = (left_m2 + right_m2, never, seen)

    total_m2, root_never, root_seen = node_values[config.tree.root]
    readout = params["readout"]
    never_value = jnp.dot(root_never[config.target_q], readout)
    seen_value = jnp.dot(root_seen[config.target_q], readout)
    valid = valid_spins & (total_m2 == config.target_m2)
    zero = jnp.asarray(0.0, dtype=never_value.dtype)
    return (
        jnp.where(valid, never_value, zero),
        jnp.where(valid, seen_value, zero),
    )


def spin_clebsch_tree_route_tagged_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SpinClebschTreeConfig,
    *,
    retained_q: int,
) -> tuple[jax.Array, jax.Array]:
    """Vectorize the explicit route split over spin configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: spin_clebsch_tree_route_tagged_amplitude(
            params, row, config, retained_q=retained_q
        )
    )(array)


__all__ = [
    "spin_clebsch_tree_route_parameter_masks",
    "spin_clebsch_tree_route_tagged_amplitude",
    "spin_clebsch_tree_route_tagged_amplitudes",
]
