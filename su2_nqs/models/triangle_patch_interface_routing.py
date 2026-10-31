"""Opt-in support-local multiplicity routing for the triangle-patch tree.

The strict triangle-patch Clebsch tree already separates magnetic indices
from reduced multiplicity indices.  This module adds one deliberately small
piece of spatial information without changing that tree: when a six-site
support is consumed by its parent, its retained ``q=2`` multiplicity vector
is expressed in a different orthonormal frame on the left and right input
legs.  The same frame is shared by every translated occurrence.

The routing matrices are fixed numerical data, not trainable parameters.
They act identically for every magnetic component and only on the reduced
output index.  Exact SU(2), physical-leg multi-affinity, and the trainable
parameter count of the underlying tree are therefore unchanged.  An
external translation/space-group projector remains valid because this
module defines an ordinary unprojected scalar amplitude and introduces no
configuration-dependent branch.

This is an opt-in forward path.  The established triangle-patch module does
not import it, so sealed runs retain their source and parameter contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.fusion_catalog import FusionSupport
from .triangle_patch_clebsch_tree import (
    TrianglePatchClebschTreePlan,
    _channels_for_route,
    _compact_cg_tensor,
    _merge_carriers,
    _triangle_carrier,
    _validate_params,
    triangle_patch_pair_layout,
    triangle_root_layout,
    triangle_shared_reduced_layout,
)


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class TrianglePatchInterfaceRouting:
    """Fixed left/right frames on one reduced spin sector.

    ``left_rotation`` and ``right_rotation`` use the convention
    ``routed[m,c] = sum_d rotation[c,d] * carrier[m,d]``.  They must be real
    orthogonal matrices of size ``feature_dim``.  Only children whose support
    contains exactly ``support_size`` physical spins are routed, and a support
    is routed once, immediately before the unique parent consumes it.

    All angular momenta are doubled integers.  The initial physical route
    uses ``routed_q=2`` (a cluster triplet) and ``support_size=6``.
    """

    routed_q: int
    support_size: int
    left_rotation: np.ndarray
    right_rotation: np.ndarray
    orthogonality_tolerance: float = 2.0e-12

    def __post_init__(self) -> None:
        for name, value in (
            ("routed_q", self.routed_q),
            ("support_size", self.support_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        routed_q = int(self.routed_q)
        support_size = int(self.support_size)
        if routed_q < 0:
            raise ValueError("routed_q must be a nonnegative doubled spin")
        if support_size < 1:
            raise ValueError("support_size must be positive")
        tolerance = float(self.orthogonality_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("orthogonality_tolerance must be positive and finite")

        checked = []
        for name, raw in (
            ("left_rotation", self.left_rotation),
            ("right_rotation", self.right_rotation),
        ):
            matrix = np.asarray(raw, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"{name} must be a square matrix")
            if matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be nonempty and finite")
            error = float(np.linalg.norm(matrix @ matrix.T - np.eye(matrix.shape[0])))
            if error > tolerance:
                raise ValueError(f"{name} must be orthogonal")
            matrix = np.array(matrix, copy=True)
            matrix.setflags(write=False)
            checked.append(matrix)
        if checked[0].shape != checked[1].shape:
            raise ValueError("left and right rotations must have the same shape")
        object.__setattr__(self, "routed_q", routed_q)
        object.__setattr__(self, "support_size", support_size)
        object.__setattr__(self, "left_rotation", checked[0])
        object.__setattr__(self, "right_rotation", checked[1])
        object.__setattr__(self, "orthogonality_tolerance", tolerance)

    @property
    def feature_dim(self) -> int:
        """Reduced feature width expected by this routing contract."""

        return int(self.left_rotation.shape[0])

    @property
    def added_trainable_parameter_count(self) -> int:
        """The physical frames are fixed and add no variational scalars."""

        return 0


def _routed_child(
    carrier: dict[int, jax.Array],
    support: FusionSupport,
    rotation: np.ndarray,
    routing: TrianglePatchInterfaceRouting,
) -> dict[int, jax.Array]:
    if len(support) != routing.support_size or routing.routed_q not in carrier:
        return carrier
    values = carrier[routing.routed_q]
    if values.shape[1] != routing.feature_dim:
        raise ValueError("interface routing feature width does not match the plan")
    matrix = jnp.asarray(rotation, dtype=values.dtype)
    updated = dict(carrier)
    updated[routing.routed_q] = jnp.einsum("md,cd->mc", values, matrix)
    return updated


def _validate_routing_plan(
    plan: TrianglePatchClebschTreePlan,
    routing: TrianglePatchInterfaceRouting,
) -> None:
    if not isinstance(plan, TrianglePatchClebschTreePlan):
        raise TypeError("plan must be a TrianglePatchClebschTreePlan")
    if not isinstance(routing, TrianglePatchInterfaceRouting):
        raise TypeError("routing must be a TrianglePatchInterfaceRouting")
    if plan.feature_dim != routing.feature_dim:
        raise ValueError("routing feature width does not match the triangle-patch plan")
    if routing.routed_q > plan.maximum_intermediate_q:
        raise ValueError("routing spin is removed by the intermediate-spin cutoff")


def triangle_patch_interface_amplitude_from_spinors(
    params: Mapping[str, object],
    spinors: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    routing: TrianglePatchInterfaceRouting,
) -> jax.Array:
    """Evaluate a strict singlet with support-local left/right frames."""

    _validate_routing_plan(plan, routing)
    checked = _validate_params(params, plan)
    raw = jnp.asarray(spinors)
    values = jnp.asarray(raw, dtype=jnp.result_type(raw.dtype, jnp.float64))
    if values.shape != (plan.n_sites, 2):
        raise ValueError(f"expected spinor shape ({plan.n_sites}, 2)")

    carriers: dict[FusionSupport, dict[int, jax.Array]] = {
        patch.support: _triangle_carrier(values, patch) for patch in plan.patches
    }
    patch_set = set(plan.patch_supports)
    patch_layout = triangle_patch_pair_layout(plan)
    reduced_layout = triangle_shared_reduced_layout(plan)
    patch_blocks = {
        layout.key: block
        for layout, block in zip(patch_layout, checked["patch_pair"], strict=True)
    }
    reduced_blocks = {
        layout.key: block
        for layout, block in zip(reduced_layout, checked["reduced"], strict=True)
    }
    patch_route_index = 0
    reduced_route_index = 0
    for route in plan.non_root_routes:
        is_patch_pair = route.left in patch_set and route.right in patch_set
        layouts = _channels_for_route(plan, route)
        if is_patch_pair:
            gate = checked["patch_gates"][patch_route_index]
            source = patch_blocks
            patch_route_index += 1
            left = carriers[route.left]
            right = carriers[route.right]
        else:
            gate = checked["reduced_gates"][reduced_route_index]
            source = reduced_blocks
            reduced_route_index += 1
            left = _routed_child(
                carriers[route.left], route.left, routing.left_rotation, routing
            )
            right = _routed_child(
                carriers[route.right], route.right, routing.right_rotation, routing
            )
        blocks = tuple(
            source[layout.key] * gate[index]
            for index, layout in enumerate(layouts)
        )
        carriers[route.parent] = _merge_carriers(
            left, right, route, blocks, layouts, plan
        )

    root = plan.root_route
    left = _routed_child(
        carriers[root.left], root.left, routing.left_rotation, routing
    )
    right = _routed_child(
        carriers[root.right], root.right, routing.right_rotation, routing
    )
    result_dtype = jnp.result_type(
        *(value.dtype for value in left.values()),
        *(value.dtype for value in right.values()),
        *(block.dtype for block in checked["root"]),
    )
    result = jnp.asarray(0.0, dtype=result_dtype)
    for layout, block in zip(
        triangle_root_layout(plan), checked["root"], strict=True
    ):
        cg = jnp.asarray(
            _compact_cg_tensor(layout.q, layout.q, 0)[0], dtype=result_dtype
        )
        result = result + jnp.einsum(
            "la,rb,ab,lr->",
            left[layout.q],
            right[layout.q],
            block,
            cg,
        )
    return result


def triangle_patch_interface_amplitude(
    params: Mapping[str, object],
    spins: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    routing: TrianglePatchInterfaceRouting,
) -> jax.Array:
    """Evaluate one routed computational-basis coefficient."""

    bits = jnp.asarray(spins)
    if bits.shape != (plan.n_sites,):
        raise ValueError(f"expected spin shape ({plan.n_sites},)")
    if bits.dtype == jnp.bool_ or not jnp.issubdtype(bits.dtype, jnp.integer):
        raise TypeError("computational-basis spins must have an integer dtype")
    bits = bits.astype(jnp.int32)
    valid = jnp.all((bits == 0) | (bits == 1))
    safe = jnp.clip(bits, 0, 1)
    spinors = jnp.eye(2, dtype=jnp.float64)[safe]
    value = triangle_patch_interface_amplitude_from_spinors(
        params, spinors, plan, routing
    )
    in_sector = jnp.sum(safe) == plan.n_sites // 2
    return jnp.where(valid & in_sector, value, jnp.asarray(0.0, dtype=value.dtype))


def triangle_patch_interface_amplitudes(
    params: Mapping[str, object],
    spins: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    routing: TrianglePatchInterfaceRouting,
) -> jax.Array:
    """Vectorize the support-routed coefficient over configurations."""

    array = jnp.asarray(spins)
    if array.ndim != 2 or array.shape[1] != plan.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {plan.n_sites})")
    if array.dtype == jnp.bool_ or not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("computational-basis spins must have an integer dtype")
    array = array.astype(jnp.int32)
    return jax.vmap(
        lambda row: triangle_patch_interface_amplitude(params, row, plan, routing)
    )(array)


__all__ = [
    "TrianglePatchInterfaceRouting",
    "triangle_patch_interface_amplitude",
    "triangle_patch_interface_amplitude_from_spinors",
    "triangle_patch_interface_amplitudes",
]
