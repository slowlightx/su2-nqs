"""Exact real-parameter subspaces for staged variational optimization.

An :class:`ActiveParameterSubspace` records a real floating parameter pytree,
plus a boolean pytree selecting the entries that an optimizer may change.
The active entries are exposed as one deterministic one-dimensional vector.
Reconstruction replaces only those entries; all inactive entries are copied
bit-for-bit from the frozen reference tree.  The corresponding tangent map
puts exact zeros on every inactive entry.

This small abstraction is intentionally independent of any model or
Hamiltonian.  A physics workflow owns the mask.  For example, a Clebsch-tree
workflow can freeze its local carrier tensors and expose only selected
upper-tree reduced blocks without teaching generic SR code about angular
momentum labels.

The class is designed to be closed over by ``jax.jit``, ``jax.jvp`` and
``jax.grad`` functions.  Construction and structural validation happen on
the host; ``inject`` and ``tangent`` themselves contain only JAX array
operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


@dataclass(frozen=True)
class ActiveParameterSubspace:
    """A deterministic coordinate chart on selected real pytree entries.

    Use :meth:`from_mask` to construct the chart.  ``True`` entries in the
    mask are active.  The coordinate order is JAX's deterministic pytree leaf
    order followed by row-major order within each leaf.

    The reference tree is deliberately retained only in flattened form.  It
    defines the immutable values used by :meth:`inject`; it is not a moving
    optimizer state.  All parameter leaves must have one common real floating
    dtype so one active vector represents them without silent precision loss.
    """

    _reference_flat: jax.Array = field(repr=False)
    _active_indices: tuple[int, ...] = field(repr=False)
    _unravel: Callable[[jax.Array], Any] = field(repr=False, compare=False)
    _structure: jax.tree_util.PyTreeDef = field(repr=False, compare=False)
    _leaf_shapes: tuple[tuple[int, ...], ...] = field(repr=False)
    _dtype: np.dtype = field(repr=False)

    @classmethod
    def from_mask(
        cls,
        reference: object,
        active_mask: object,
    ) -> "ActiveParameterSubspace":
        """Build a coordinate chart from a reference tree and boolean mask.

        ``active_mask`` must have exactly the same pytree structure and leaf
        shapes as ``reference``.  Masks must be explicitly boolean; numeric
        zero/one arrays are rejected to avoid accidental parameter release.
        At least one scalar must be active.
        """

        reference_leaves, structure = jax.tree_util.tree_flatten(reference)
        mask_leaves, mask_structure = jax.tree_util.tree_flatten(active_mask)
        if not reference_leaves:
            raise ValueError("reference must contain at least one array leaf")
        if structure != mask_structure:
            raise ValueError("active_mask must have the reference pytree structure")

        checked_reference = tuple(jnp.asarray(leaf) for leaf in reference_leaves)
        dtypes = tuple(np.dtype(leaf.dtype) for leaf in checked_reference)
        if any(not np.issubdtype(dtype, np.floating) for dtype in dtypes):
            raise TypeError("reference leaves must have real floating dtype")
        if any(dtype != dtypes[0] for dtype in dtypes[1:]):
            raise TypeError("reference leaves must share one floating dtype")
        if any(not np.all(np.isfinite(np.asarray(leaf))) for leaf in checked_reference):
            raise FloatingPointError("reference contains non-finite values")

        shapes = tuple(tuple(int(size) for size in leaf.shape) for leaf in checked_reference)
        flattened_masks: list[np.ndarray] = []
        for shape, raw_mask in zip(shapes, mask_leaves, strict=True):
            concrete = np.asarray(raw_mask)
            if concrete.dtype != np.dtype(np.bool_):
                raise TypeError("active_mask leaves must have boolean dtype")
            if concrete.shape != shape:
                raise ValueError("an active_mask leaf has the wrong shape")
            flattened_masks.append(concrete.reshape(-1))
        flat_mask = np.concatenate(flattened_masks)
        active_indices = tuple(int(index) for index in np.flatnonzero(flat_mask))
        if not active_indices:
            raise ValueError("active_mask must select at least one scalar")

        reference_flat, unravel = ravel_pytree(reference)
        if reference_flat.ndim != 1 or reference_flat.size != flat_mask.size:
            raise RuntimeError("pytree flattening and mask flattening disagree")
        return cls(
            _reference_flat=reference_flat,
            _active_indices=active_indices,
            _unravel=unravel,
            _structure=structure,
            _leaf_shapes=shapes,
            _dtype=dtypes[0],
        )

    @property
    def dimension(self) -> int:
        """Number of active real scalar coordinates."""

        return len(self._active_indices)

    @property
    def total_parameter_count(self) -> int:
        """Total number of active plus frozen real scalar entries."""

        return int(self._reference_flat.size)

    @property
    def active_flat_indices(self) -> tuple[int, ...]:
        """Return active coordinates in deterministic flattened-tree order."""

        return self._active_indices

    def _checked_flatten(self, tree: object, *, name: str) -> jax.Array:
        leaves, structure = jax.tree_util.tree_flatten(tree)
        if structure != self._structure:
            raise ValueError(f"{name} has the wrong pytree structure")
        shapes = tuple(
            tuple(int(size) for size in jnp.shape(leaf)) for leaf in leaves
        )
        if shapes != self._leaf_shapes:
            raise ValueError(f"{name} has a leaf with the wrong shape")
        if any(np.dtype(jnp.asarray(leaf).dtype) != self._dtype for leaf in leaves):
            raise TypeError(f"{name} leaves must retain dtype {self._dtype}")
        flat, _ = ravel_pytree(tree)
        return flat

    def pack(self, tree: object) -> jax.Array:
        """Extract active coordinates from a compatible parameter tree."""

        flat = self._checked_flatten(tree, name="tree")
        return flat[jnp.asarray(self._active_indices, dtype=jnp.int32)]

    def inject(self, active: jax.Array) -> object:
        """Insert an active vector into the otherwise frozen reference tree."""

        vector = jnp.asarray(active)
        if vector.shape != (self.dimension,):
            raise ValueError(f"active must have shape ({self.dimension},)")
        if np.dtype(vector.dtype) != self._dtype:
            raise TypeError(f"active must have dtype {self._dtype}")
        indices = jnp.asarray(self._active_indices, dtype=jnp.int32)
        return self._unravel(self._reference_flat.at[indices].set(vector))

    def tangent(self, active_tangent: jax.Array) -> object:
        """Lift an active tangent, putting exact zeros on frozen entries."""

        vector = jnp.asarray(active_tangent)
        if vector.shape != (self.dimension,):
            raise ValueError(f"active_tangent must have shape ({self.dimension},)")
        if np.dtype(vector.dtype) != self._dtype:
            raise TypeError(f"active_tangent must have dtype {self._dtype}")
        indices = jnp.asarray(self._active_indices, dtype=jnp.int32)
        flat = jnp.zeros_like(self._reference_flat).at[indices].set(vector)
        return self._unravel(flat)


def _triangle_patch_structural_support(
    reference: Mapping[str, object],
    plan: object,
) -> tuple[dict[object, dict[int, np.ndarray]], dict[object, dict[int, np.ndarray]], list[tuple]]:
    """Propagate exact-zero support forward and root relevance backward."""

    # Imports stay local: the generic coordinate chart above has no model
    # dependency, while this optional selector follows the public strict-tree
    # layout contract.
    from ..models.triangle_patch_clebsch_tree import (
        TrianglePatchClebschTreePlan,
        triangle_patch_pair_layout,
        triangle_root_layout,
        triangle_route_layout,
        triangle_shared_reduced_layout,
    )
    from ..su2.local_schur import cell_schur_spec

    if not isinstance(plan, TrianglePatchClebschTreePlan):
        raise TypeError("plan must be a TrianglePatchClebschTreePlan")
    if not isinstance(reference, Mapping) or set(reference) != {
        "patch_pair",
        "reduced",
        "root",
        "patch_gates",
        "reduced_gates",
    }:
        raise ValueError("reference does not match the triangle-patch parameter tree")

    patch_layout = triangle_patch_pair_layout(plan)
    reduced_layout = triangle_shared_reduced_layout(plan)
    root_layout = triangle_root_layout(plan)
    expected = {
        "patch_pair": tuple(layout.shape for layout in patch_layout),
        "reduced": tuple(layout.shape for layout in reduced_layout),
        "root": tuple(layout.shape for layout in root_layout),
        "patch_gates": plan.patch_gate_shapes,
        "reduced_gates": plan.reduced_gate_shapes,
    }
    arrays: dict[str, tuple[np.ndarray, ...]] = {}
    for name, shapes in expected.items():
        raw = reference[name]
        if not isinstance(raw, (tuple, list)) or len(raw) != len(shapes):
            raise ValueError(f"reference {name} blocks do not match the plan")
        blocks = tuple(np.asarray(block) for block in raw)
        if tuple(block.shape for block in blocks) != shapes:
            raise ValueError(f"a reference {name} block has the wrong shape")
        arrays[name] = blocks

    patch_blocks = {
        layout.key: block
        for layout, block in zip(patch_layout, arrays["patch_pair"], strict=True)
    }
    reduced_blocks = {
        layout.key: block
        for layout, block in zip(reduced_layout, arrays["reduced"], strict=True)
    }
    local_spec = cell_schur_spec("triangular_triangle")
    forward: dict[object, dict[int, np.ndarray]] = {
        patch.support: {
            q: np.ones(local_spec.multiplicity(q), dtype=np.bool_)
            for q in local_spec.q_values
        }
        for patch in plan.patches
    }
    route_records: list[tuple] = []
    patch_set = set(plan.patch_supports)
    patch_route_index = 0
    reduced_route_index = 0
    for route in plan.non_root_routes:
        layouts = triangle_route_layout(plan, route)
        if route.left in patch_set and route.right in patch_set:
            gate = arrays["patch_gates"][patch_route_index]
            blocks = patch_blocks
            bank_name = "patch_pair"
            patch_route_index += 1
        else:
            gate = arrays["reduced_gates"][reduced_route_index]
            blocks = reduced_blocks
            bank_name = "reduced"
            reduced_route_index += 1
        output = {
            q: np.zeros(plan.feature_dim, dtype=np.bool_)
            for q in sorted({layout.q_parent for layout in layouts})
        }
        for position, layout in enumerate(layouts):
            effective_nonzero = (blocks[layout.key] * gate[position]) != 0.0
            reachable = (
                effective_nonzero
                & forward[route.left][layout.q_left][None, :, None]
                & forward[route.right][layout.q_right][None, None, :]
            )
            output[layout.q_parent] |= np.any(reachable, axis=(1, 2))
        forward[route.parent] = output
        route_records.append((route, layouts, gate, blocks, bank_name))

    backward = {
        support: {
            q: np.zeros_like(channels, dtype=np.bool_)
            for q, channels in sectors.items()
        }
        for support, sectors in forward.items()
    }
    root = plan.root_route
    for layout, block in zip(root_layout, arrays["root"], strict=True):
        nonzero = block != 0.0
        left_forward = forward[root.left][layout.q]
        right_forward = forward[root.right][layout.q]
        backward[root.left][layout.q] |= np.any(
            nonzero & right_forward[None, :], axis=1
        )
        backward[root.right][layout.q] |= np.any(
            nonzero & left_forward[:, None], axis=0
        )

    for route, layouts, gate, blocks, _ in reversed(route_records):
        for position, layout in enumerate(layouts):
            effective_nonzero = (blocks[layout.key] * gate[position]) != 0.0
            parent_relevant = backward[route.parent][layout.q_parent]
            left_forward = forward[route.left][layout.q_left]
            right_forward = forward[route.right][layout.q_right]
            backward[route.left][layout.q_left] |= np.any(
                effective_nonzero
                & parent_relevant[:, None, None]
                & right_forward[None, None, :],
                axis=(0, 2),
            )
            backward[route.right][layout.q_right] |= np.any(
                effective_nonzero
                & parent_relevant[:, None, None]
                & left_forward[None, :, None],
                axis=(0, 1),
            )
    return forward, backward, route_records


def triangle_patch_low_spin_upper_tree_mask(
    reference: Mapping[str, object],
    plan: object,
    *,
    stage: Literal["first_order", "complete"] = "first_order",
    maximum_active_q: Literal[0, 2] = 2,
) -> dict[str, tuple[np.ndarray, ...]]:
    """Select a canonical ``q=0,2`` upper-tree optimization stage.

    Both stages freeze the complete six-site patch bank, every route gate,
    the pure-core ``(0,0,0)[0,0,0]`` reduced coefficient, and the root
    ``q=0,[0,0]`` coefficient.  Freezing route gates removes the exact
    multiplicative bank/gate redundancy; the two scalar anchors preserve the
    normalized cluster-product branch and global scale convention.

    ``stage="complete"`` returns the 46-coordinate nonlinear family made of
    all other entries in the five reduced blocks whose doubled spins are at
    most two, plus the ``q=0,2`` root blocks.  ``stage="first_order"`` further
    intersects this family with an exact structural derivative test at the
    supplied reference: child channels must be reachable through nonzero base
    tensors and the output channel must be able to reach a nonzero root term.
    No numerical threshold or sampled configuration is used.  The registered
    physical initializer with nonzero background has 18 such coordinates.

    ``maximum_active_q=0`` is the nested singlet-resonance control: only the
    ``(0,0,0)`` reduced block and the ``q=0`` root can change.  It has ten
    first-order coordinates at the registered physical initializer.
    ``maximum_active_q=2`` additionally admits virtual cluster triplets and
    is the eighteen-coordinate treatment arm.  Keeping these as nested
    charts makes the incremental role of local ``S=1`` carriers measurable.
    """

    if stage not in {"first_order", "complete"}:
        raise ValueError("stage must be 'first_order' or 'complete'")
    if maximum_active_q not in (0, 2):
        raise ValueError("maximum_active_q must be doubled spin 0 or 2")
    from ..models.triangle_patch_clebsch_tree import (
        triangle_root_layout,
        triangle_shared_reduced_layout,
    )

    forward, backward, route_records = _triangle_patch_structural_support(
        reference, plan
    )
    mask = jax.tree_util.tree_map(
        lambda leaf: np.zeros(np.shape(leaf), dtype=np.bool_), reference
    )
    reduced_masks: list[np.ndarray] = []
    for layout, block in zip(
        triangle_shared_reduced_layout(plan), reference["reduced"], strict=True
    ):
        selected = np.full(
            np.shape(block), max(layout.key) <= maximum_active_q, dtype=np.bool_
        )
        if stage == "first_order" and np.any(selected):
            visible = np.zeros_like(selected)
            for route, layouts, gate, _, bank_name in route_records:
                if bank_name != "reduced":
                    continue
                position = next(
                    (
                        index
                        for index, candidate in enumerate(layouts)
                        if candidate.key == layout.key
                    ),
                    None,
                )
                if position is None or gate[position] == 0.0:
                    continue
                visible |= (
                    backward[route.parent][layout.q_parent][:, None, None]
                    & forward[route.left][layout.q_left][None, :, None]
                    & forward[route.right][layout.q_right][None, None, :]
                )
            selected &= visible
        if layout.key == (0, 0, 0):
            selected[0, 0, 0] = False
        reduced_masks.append(selected)

    root_masks: list[np.ndarray] = []
    root = plan.root_route
    for layout, block in zip(
        triangle_root_layout(plan), reference["root"], strict=True
    ):
        selected = np.full(
            np.shape(block), layout.q <= maximum_active_q, dtype=np.bool_
        )
        if stage == "first_order" and np.any(selected):
            selected &= (
                forward[root.left][layout.q][:, None]
                & forward[root.right][layout.q][None, :]
            )
        if layout.q == 0:
            selected[0, 0] = False
        root_masks.append(selected)
    mask["reduced"] = tuple(reduced_masks)
    mask["root"] = tuple(root_masks)
    return mask


def triangle_patch_low_spin_upper_tree_subspace(
    reference: Mapping[str, object],
    plan: object,
    *,
    stage: Literal["first_order", "complete"] = "first_order",
    maximum_active_q: Literal[0, 2] = 2,
) -> ActiveParameterSubspace:
    """Build the canonical strict-tree low-spin coordinate chart."""

    return ActiveParameterSubspace.from_mask(
        reference,
        triangle_patch_low_spin_upper_tree_mask(
            reference,
            plan,
            stage=stage,
            maximum_active_q=maximum_active_q,
        ),
    )


__all__ = [
    "ActiveParameterSubspace",
    "triangle_patch_low_spin_upper_tree_mask",
    "triangle_patch_low_spin_upper_tree_subspace",
]
