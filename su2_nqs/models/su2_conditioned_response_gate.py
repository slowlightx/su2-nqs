"""Configuration-conditioned scalar response gates in SU(2) A1 space.

The gate reads only contextual reduced-label tokens from the four-cell
Spatial-ViT.  A diagonal function in the complete local-Schur coefficient
basis is projected into exact ``Gamma-A1`` coordinates and then coupled to a
prevalidated response-return operator through a Hermitian anticommutator.
Magnetic indices never enter a learned quantity.
"""

from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .su2_four_cell_sector_vit import (
    SU2FourCellSectorViTConfig,
    init_su2_four_cell_sector_vit,
    su2_four_cell_sector_vit_tokens,
)


jax.config.update("jax_enable_x64", True)


def init_su2_conditioned_response_gate(
    key: jax.Array,
    config: SU2FourCellSectorViTConfig,
    response_count: int,
    *,
    gate_scale: float = 0.08,
) -> dict[str, object]:
    """Initialize one shared backbone and response-specific scalar heads."""

    if (
        isinstance(response_count, bool)
        or not isinstance(response_count, Integral)
        or int(response_count) < 1
    ):
        raise ValueError("response_count must be a positive integer")
    scale = float(gate_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("gate_scale must be positive and finite")
    backbone_key, gate_key = jax.random.split(key)
    return {
        "backbone": init_su2_four_cell_sector_vit(backbone_key, config),
        "response_gate": {
            "head": scale
            * jax.random.normal(
                gate_key,
                (config.embedding_dim, int(response_count)),
                dtype=jnp.float64,
            ),
            # Start close to the fixed response loop while retaining a
            # nonzero configuration-dependent tangent.
            "bias": jnp.full(
                (int(response_count),), np.arctanh(0.5), dtype=jnp.float64
            ),
        },
    }


def su2_conditioned_response_gate_values(
    params: dict[str, object],
    actions: jax.Array,
    config: SU2FourCellSectorViTConfig,
) -> jax.Array:
    """Return bounded scalar gates for every reduced configuration/channel."""

    tokens = su2_four_cell_sector_vit_tokens(
        params["backbone"], actions, config
    )
    # All six causal decisions contribute.  Position information is already
    # present in every token and later decisions carry the earlier labels.
    pooled = jnp.mean(tokens, axis=1)
    gate = params["response_gate"]
    return jnp.tanh(pooled @ gate["head"] + gate["bias"])


def su2_conditioned_response_operators(
    params: dict[str, object],
    actions: jax.Array,
    config: SU2FourCellSectorViTConfig,
    coefficient_a1_basis: jax.Array,
    response_operators: jax.Array,
    *,
    scramble_rotation: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return normalized conditioned A1 operators and raw gate values.

    ``coefficient_a1_basis`` maps exact A1 coordinates into the complete
    local-Schur coefficient basis.  If supplied, ``scramble_rotation``
    orthogonally conjugates every final operator, preserving its spectrum at
    each parameter value.
    """

    coefficient = jnp.asarray(coefficient_a1_basis)
    response = jnp.asarray(response_operators)
    if coefficient.ndim != 2:
        raise ValueError("coefficient_a1_basis must be rank two")
    if response.ndim != 3 or response.shape[1:] != (
        coefficient.shape[1],
        coefficient.shape[1],
    ):
        raise ValueError("response operators and A1 basis have incompatible shapes")
    gates = su2_conditioned_response_gate_values(params, actions, config)
    if gates.shape != (coefficient.shape[0], response.shape[0]):
        raise ValueError("gate values do not match coefficient/response dimensions")
    gate_matrices = jnp.einsum(
        "ni,nk,nj->kij", coefficient.conj(), gates, coefficient
    )
    left = jnp.einsum("kij,kjl->kil", gate_matrices, response)
    right = jnp.einsum("kij,kjl->kil", response, gate_matrices)
    operators = 0.5 * (left + right)
    dimension = coefficient.shape[1]
    identity = jnp.eye(dimension, dtype=operators.dtype)
    traces = jnp.trace(operators, axis1=1, axis2=2) / dimension
    operators = operators - traces[:, None, None] * identity[None, :, :]
    norms = jnp.sqrt(
        jnp.sum(jnp.real(operators.conj() * operators), axis=(1, 2))
        + 1.0e-24
    )
    operators = operators / norms[:, None, None]
    if scramble_rotation is not None:
        rotation = jnp.asarray(scramble_rotation)
        if rotation.shape != (dimension, dimension):
            raise ValueError("scramble_rotation has the wrong shape")
        operators = jnp.einsum(
            "ai,kab,bj->kij", rotation, operators, rotation
        )
    return operators, gates


def su2_conditioned_response_parameter_count(params: object) -> int:
    """Return the number of real trainable scalars."""

    return int(
        sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params))
    )


__all__ = [
    "init_su2_conditioned_response_gate",
    "su2_conditioned_response_gate_values",
    "su2_conditioned_response_operators",
    "su2_conditioned_response_parameter_count",
]
