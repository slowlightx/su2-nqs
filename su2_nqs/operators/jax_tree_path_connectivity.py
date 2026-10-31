"""Accelerator-native local recoupling for arbitrary spin fusion trees.

The reference connectivity implementation in :mod:`tree_path_connectivity`
uses Python candidate lists and dictionaries.  That is ideal for validation,
but it puts every sampled VMC step on the host critical path.  This module
compiles the state-independent tree geometry and the finite SU(2) coefficient
tables once, then generates raw (not deduplicated) bond columns with JAX.

Per-bond output labels are returned raw.  For a complete graph action,
:func:`jax_coalesce_tree_connectivity` combines repeated labels on device
before a neural amplitude is evaluated.  Its mixed-radix sort keys are
collision free (unlike a hash). Candidate buffers are fixed capacity and
report overflow explicitly; no Hamiltonian branch is ever silently truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .tree_path_connectivity import (
    _left_vector_advance_coefficient,
    _right_vector_advance_coefficient,
    _scalar_vector_contraction_coefficient,
    compile_tree_heisenberg_graph_connectivity_plan,
)
from ..su2.fusion_tree import FusionTree
from ..su2.quantum_numbers import coupling_range, is_admissible_triangle


@dataclass(frozen=True)
class JaxTreeConnectivityPlan:
    """Static arrays and sparse coefficient tables for device recoupling."""

    n_sites: int
    target_q: int
    coefficient_tolerance: float
    bond_couplings: np.ndarray
    lca_indices: np.ndarray
    left_references: np.ndarray
    right_references: np.ndarray
    left_node_indices: np.ndarray
    left_active_references: np.ndarray
    left_spectator_references: np.ndarray
    left_step_is_left: np.ndarray
    left_step_mask: np.ndarray
    right_node_indices: np.ndarray
    right_active_references: np.ndarray
    right_spectator_references: np.ndarray
    right_step_is_left: np.ndarray
    right_step_mask: np.ndarray
    left_candidate_capacity_bounds: np.ndarray
    right_candidate_capacity_bounds: np.ndarray
    vector_left_keys: np.ndarray
    vector_left_values: np.ndarray
    vector_right_keys: np.ndarray
    vector_right_values: np.ndarray
    scalar_keys: np.ndarray
    scalar_values: np.ndarray

    def __post_init__(self) -> None:
        bond_count = len(np.asarray(self.bond_couplings))
        if self.n_sites < 2 or self.target_q < 0 or bond_count < 1:
            raise ValueError("invalid device connectivity plan identity")
        for name in ("lca_indices", "left_references", "right_references"):
            if np.asarray(getattr(self, name)).shape != (bond_count,):
                raise ValueError(f"{name} must have one entry per bond")
        for prefix in ("left", "right"):
            nodes = np.asarray(getattr(self, f"{prefix}_node_indices"))
            if nodes.ndim != 2 or nodes.shape[0] != bond_count:
                raise ValueError(f"{prefix} step arrays have the wrong shape")
            for suffix in (
                "active_references",
                "spectator_references",
                "step_is_left",
                "step_mask",
            ):
                if np.asarray(getattr(self, f"{prefix}_{suffix}")).shape != nodes.shape:
                    raise ValueError(f"{prefix}_{suffix} has the wrong shape")
            bounds = np.asarray(
                getattr(self, f"{prefix}_candidate_capacity_bounds")
            )
            if bounds.shape != (bond_count,) or np.any(bounds < 1):
                raise ValueError(f"{prefix} candidate bounds have the wrong shape")
        for prefix in ("vector_left", "vector_right", "scalar"):
            keys = np.asarray(getattr(self, f"{prefix}_keys"))
            values = np.asarray(getattr(self, f"{prefix}_values"))
            if keys.ndim != 1 or values.shape != keys.shape or len(keys) < 1:
                raise ValueError(f"{prefix} lookup must be a nonempty vector")
            if np.any(keys[1:] <= keys[:-1]):
                raise ValueError(f"{prefix} lookup keys must be strictly sorted")
            if np.any(~np.isfinite(values)):
                raise FloatingPointError(f"{prefix} lookup contains non-finite values")

    @property
    def bond_count(self) -> int:
        return int(len(self.bond_couplings))

    @property
    def maximum_depth(self) -> int:
        return int(
            max(self.left_node_indices.shape[1], self.right_node_indices.shape[1])
        )


def _reference_leaf_counts(
    left_references: tuple[int, ...], right_references: tuple[int, ...]
) -> tuple[int, ...]:
    counts: list[int] = []
    for left, right in zip(left_references, right_references, strict=True):
        left_count = 1 if left < 0 else counts[left]
        right_count = 1 if right < 0 else counts[right]
        counts.append(left_count + right_count)
    return tuple(counts)


def _reference_leaf_count(reference: int, node_counts: tuple[int, ...]) -> int:
    return 1 if reference < 0 else int(node_counts[reference])


def _pack_host(values: tuple[int, int, int, int, int], base: int) -> int:
    result = 0
    for value in values:
        result = result * base + int(value)
    return result


def _sorted_lookup(rows: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    retained = sorted((key, value) for key, value in rows.items() if value != 0.0)
    if not retained:
        raise RuntimeError("an SU(2) coefficient lookup unexpectedly has no entries")
    return (
        np.asarray([key for key, _ in retained], dtype=np.int64),
        np.asarray([value for _, value in retained], dtype=np.float64),
    )


def _vector_candidate_capacity_bound(
    steps,
    node_counts: tuple[int, ...],
    *,
    coefficient_tolerance: float,
) -> int:
    """Return an exhaustive safe candidate bound for one leaf-to-LCA path."""

    concrete_steps = tuple(steps)
    if not concrete_steps:
        return 1

    @lru_cache(maxsize=None)
    def visit(
        step_index: int,
        child_old: int,
        candidate_counts: tuple[tuple[int, int], ...],
    ) -> int:
        if step_index == len(concrete_steps):
            return sum(count for _, count in candidate_counts)
        step = concrete_steps[step_index]
        spectator_count = _reference_leaf_count(
            step.spectator_reference, node_counts
        )
        maximum = 0
        for spectator_q in range(
            spectator_count % 2, spectator_count + 1, 2
        ):
            for parent_old in coupling_range(child_old, spectator_q):
                following: dict[int, int] = {}
                for child_new, multiplicity in candidate_counts:
                    parent_range = (
                        coupling_range(child_new, spectator_q)
                        if step.active_is_left
                        else coupling_range(spectator_q, child_new)
                    )
                    for parent_new in parent_range:
                        if step.active_is_left:
                            coefficient = _left_vector_advance_coefficient(
                                child_new,
                                child_old,
                                spectator_q,
                                parent_new,
                                parent_old,
                            )
                        else:
                            coefficient = _right_vector_advance_coefficient(
                                spectator_q,
                                child_new,
                                child_old,
                                parent_new,
                                parent_old,
                            )
                        if abs(coefficient) > coefficient_tolerance:
                            following[int(parent_new)] = (
                                following.get(int(parent_new), 0) + multiplicity
                            )
                if not following:
                    continue
                concrete = tuple(sorted(following.items()))
                count = sum(value for _, value in concrete)
                maximum = max(
                    maximum,
                    count,
                    visit(step_index + 1, int(parent_old), concrete),
                )
        return maximum

    return max(1, visit(0, 1, ((1, 1),)))


def compile_jax_tree_connectivity_plan(
    tree: FusionTree,
    bonds,
    *,
    target_q: int = 0,
    coefficient_tolerance: float = 1.0e-14,
) -> JaxTreeConnectivityPlan:
    """Compile state-independent sparse SU(2) tables for JAX bond actions.

    This is an offline/initialization operation.  Symbolic or cached numerical
    coefficients never run inside JIT.  The resulting arrays can be serialized
    and reused by CPU, CUDA, or ROCm backends.
    """

    reference = compile_tree_heisenberg_graph_connectivity_plan(
        tree,
        bonds,
        target_q=target_q,
        coefficient_tolerance=coefficient_tolerance,
    )
    node_counts = _reference_leaf_counts(
        reference._left_references, reference._right_references
    )
    base = tree.n_leaves + 1

    left_lookup: dict[int, float] = {}
    right_lookup: dict[int, float] = {}
    step_types: set[tuple[int, int, bool]] = set()
    for bond in reference._compiled_bonds:
        for step in bond.left_steps + bond.right_steps:
            step_types.add(
                (
                    _reference_leaf_count(step.active_reference, node_counts),
                    _reference_leaf_count(step.spectator_reference, node_counts),
                    bool(step.active_is_left),
                )
            )
    for active_count, spectator_count, active_is_left in sorted(step_types):
        destination = left_lookup if active_is_left else right_lookup
        for child_old in range(active_count % 2, active_count + 1, 2):
            for spectator_q in range(
                spectator_count % 2, spectator_count + 1, 2
            ):
                for parent_old in coupling_range(child_old, spectator_q):
                    for child_new in (child_old - 2, child_old, child_old + 2):
                        if child_new < 0 or not is_admissible_triangle(
                            child_old, 2, child_new
                        ):
                            continue
                        for parent_new in (
                            parent_old - 2,
                            parent_old,
                            parent_old + 2,
                        ):
                            if (
                                parent_new < 0
                                or not is_admissible_triangle(
                                    parent_old, 2, parent_new
                                )
                                or not is_admissible_triangle(
                                    child_new, spectator_q, parent_new
                                )
                            ):
                                continue
                            if active_is_left:
                                coefficient = _left_vector_advance_coefficient(
                                    child_new,
                                    child_old,
                                    spectator_q,
                                    parent_new,
                                    parent_old,
                                )
                            else:
                                coefficient = _right_vector_advance_coefficient(
                                    spectator_q,
                                    child_new,
                                    child_old,
                                    parent_new,
                                    parent_old,
                                )
                            if abs(coefficient) > coefficient_tolerance:
                                destination[
                                    _pack_host(
                                        (
                                            child_new,
                                            child_old,
                                            spectator_q,
                                            parent_new,
                                            parent_old,
                                        ),
                                        base,
                                    )
                                ] = float(coefficient)

    scalar_lookup: dict[int, float] = {}
    subtree_pairs = {
        (
            _reference_leaf_count(bond.left_reference, node_counts),
            _reference_leaf_count(bond.right_reference, node_counts),
        )
        for bond in reference._compiled_bonds
    }
    for left_count, right_count in sorted(subtree_pairs):
        for left_old in range(left_count % 2, left_count + 1, 2):
            for right_old in range(right_count % 2, right_count + 1, 2):
                for total_q in coupling_range(left_old, right_old):
                    for left_new in (left_old - 2, left_old, left_old + 2):
                        if left_new < 0 or not is_admissible_triangle(
                            left_old, 2, left_new
                        ):
                            continue
                        for right_new in (
                            right_old - 2,
                            right_old,
                            right_old + 2,
                        ):
                            if (
                                right_new < 0
                                or not is_admissible_triangle(
                                    right_old, 2, right_new
                                )
                                or not is_admissible_triangle(
                                    left_new, right_new, total_q
                                )
                            ):
                                continue
                            coefficient = _scalar_vector_contraction_coefficient(
                                left_new,
                                right_new,
                                left_old,
                                right_old,
                                total_q,
                            )
                            if abs(coefficient) > coefficient_tolerance:
                                scalar_lookup[
                                    _pack_host(
                                        (
                                            left_new,
                                            right_new,
                                            left_old,
                                            right_old,
                                            total_q,
                                        ),
                                        base,
                                    )
                                ] = float(coefficient)

    maximum_left = max(
        (len(bond.left_steps) for bond in reference._compiled_bonds), default=0
    )
    maximum_right = max(
        (len(bond.right_steps) for bond in reference._compiled_bonds), default=0
    )

    def step_arrays(side: str, maximum: int):
        shape = (reference.bond_count, maximum)
        nodes = np.zeros(shape, dtype=np.int32)
        active = np.full(shape, -1, dtype=np.int32)
        spectator = np.full(shape, -1, dtype=np.int32)
        is_left = np.zeros(shape, dtype=np.bool_)
        mask = np.zeros(shape, dtype=np.bool_)
        for bond_index, bond in enumerate(reference._compiled_bonds):
            steps = bond.left_steps if side == "left" else bond.right_steps
            for step_index, step in enumerate(steps):
                nodes[bond_index, step_index] = step.node_index
                active[bond_index, step_index] = step.active_reference
                spectator[bond_index, step_index] = step.spectator_reference
                is_left[bond_index, step_index] = step.active_is_left
                mask[bond_index, step_index] = True
        return nodes, active, spectator, is_left, mask

    left_arrays = step_arrays("left", maximum_left)
    right_arrays = step_arrays("right", maximum_right)
    left_bounds = np.asarray(
        [
            _vector_candidate_capacity_bound(
                bond.left_steps,
                node_counts,
                coefficient_tolerance=coefficient_tolerance,
            )
            for bond in reference._compiled_bonds
        ],
        dtype=np.int32,
    )
    right_bounds = np.asarray(
        [
            _vector_candidate_capacity_bound(
                bond.right_steps,
                node_counts,
                coefficient_tolerance=coefficient_tolerance,
            )
            for bond in reference._compiled_bonds
        ],
        dtype=np.int32,
    )
    vector_left_keys, vector_left_values = _sorted_lookup(left_lookup)
    vector_right_keys, vector_right_values = _sorted_lookup(right_lookup)
    scalar_keys, scalar_values = _sorted_lookup(scalar_lookup)
    return JaxTreeConnectivityPlan(
        n_sites=tree.n_leaves,
        target_q=int(target_q),
        coefficient_tolerance=float(coefficient_tolerance),
        bond_couplings=np.asarray(
            [bond.coupling for bond in reference._compiled_bonds], dtype=np.float64
        ),
        lca_indices=np.asarray(
            [bond.lca_index for bond in reference._compiled_bonds], dtype=np.int32
        ),
        left_references=np.asarray(
            [bond.left_reference for bond in reference._compiled_bonds], dtype=np.int32
        ),
        right_references=np.asarray(
            [bond.right_reference for bond in reference._compiled_bonds], dtype=np.int32
        ),
        left_node_indices=left_arrays[0],
        left_active_references=left_arrays[1],
        left_spectator_references=left_arrays[2],
        left_step_is_left=left_arrays[3],
        left_step_mask=left_arrays[4],
        right_node_indices=right_arrays[0],
        right_active_references=right_arrays[1],
        right_spectator_references=right_arrays[2],
        right_step_is_left=right_arrays[3],
        right_step_mask=right_arrays[4],
        left_candidate_capacity_bounds=left_bounds,
        right_candidate_capacity_bounds=right_bounds,
        vector_left_keys=vector_left_keys,
        vector_left_values=vector_left_values,
        vector_right_keys=vector_right_keys,
        vector_right_values=vector_right_values,
        scalar_keys=scalar_keys,
        scalar_values=scalar_values,
    )


def _pack_device(values: tuple[jax.Array, ...], base: int) -> jax.Array:
    result = jnp.zeros_like(values[0], dtype=jnp.int64)
    for value in values:
        result = result * np.int64(base) + value.astype(jnp.int64)
    return result


def _lookup(keys, values, query):
    position = jnp.searchsorted(keys, query, side="left")
    clipped = jnp.minimum(position, len(keys) - 1)
    return jnp.where(
        (position < len(keys)) & (keys[clipped] == query),
        values[clipped],
        0.0,
    )


def _packed_label_chunks(labels: jax.Array, base: int) -> jax.Array:
    """Pack label rows into collision-free signed-int64 lexicographic keys."""

    maximum = np.iinfo(np.int64).max
    chunk_size = 1
    power = int(base)
    while power <= maximum // int(base):
        power *= int(base)
        chunk_size += 1
    chunks = []
    for start in range(0, labels.shape[1], chunk_size):
        value = jnp.zeros((labels.shape[0],), dtype=jnp.int64)
        for column in range(start, min(start + chunk_size, labels.shape[1])):
            value = value * np.int64(base) + labels[:, column].astype(jnp.int64)
        chunks.append(value)
    return jnp.stack(chunks, axis=0)


def jax_coalesce_tree_connectivity(
    connected_labels: jax.Array,
    coefficients: jax.Array,
    valid: jax.Array,
    *,
    n_sites: int,
    output_capacity: int,
    coefficient_tolerance: float = 1.0e-14,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sort and sum repeated tree labels entirely on the accelerator.

    Returns ``(labels, coefficients, valid, unique_count, overflow)``.  The
    fixed output shape is needed by XLA. If the exact number of unique valid
    labels exceeds ``output_capacity``, ``overflow`` is true; production
    callers must reject that batch rather than use the truncated buffer.
    """

    if isinstance(output_capacity, bool) or not isinstance(output_capacity, Integral):
        raise TypeError("output_capacity must be an integer")
    if output_capacity < 1:
        raise ValueError("output_capacity must be positive")
    if (
        not math.isfinite(float(coefficient_tolerance))
        or float(coefficient_tolerance) < 0.0
    ):
        raise ValueError("coefficient_tolerance must be finite and non-negative")
    labels = jnp.asarray(connected_labels, dtype=jnp.int32)
    values = jnp.asarray(coefficients, dtype=jnp.float64)
    retain = jnp.asarray(valid, dtype=jnp.bool_)
    if labels.ndim != 2 or labels.shape[1] != int(n_sites) - 1:
        raise ValueError("connected_labels have the wrong tree-label shape")
    if values.shape != (labels.shape[0],) or retain.shape != values.shape:
        raise ValueError("one coefficient and validity flag are required per row")
    if labels.shape[0] < 1:
        raise ValueError("at least one raw connectivity row is required")

    chunks = _packed_label_chunks(labels, int(n_sites) + 1)
    # jnp.lexsort uses the final key as primary. Invalid rows sort last; the
    # exact mixed-radix chunks preserve ordinary row lexicography.
    keys = jnp.concatenate(
        (jnp.flip(chunks, axis=0), (~retain)[None, :].astype(jnp.int64)),
        axis=0,
    )
    order = jnp.lexsort(keys, axis=-1)
    sorted_labels = labels[order]
    sorted_values = values[order]
    sorted_valid = retain[order]
    first = jnp.arange(labels.shape[0], dtype=jnp.int32) == 0
    changed = jnp.any(
        sorted_labels != jnp.roll(sorted_labels, 1, axis=0), axis=1
    )
    starts = sorted_valid & (first | changed)
    unique_count = jnp.sum(starts, dtype=jnp.int32)
    segment = jnp.cumsum(starts, dtype=jnp.int32) - 1
    safe_segment = jnp.maximum(segment, 0)
    summed = jax.ops.segment_sum(
        jnp.where(sorted_valid, sorted_values, 0.0),
        safe_segment,
        num_segments=int(output_capacity),
        indices_are_sorted=True,
        mode="drop",
    )
    start_indices = jnp.nonzero(
        starts, size=int(output_capacity), fill_value=0
    )[0]
    unique_labels = sorted_labels[start_indices]
    retained_unique = (
        (jnp.arange(int(output_capacity)) < unique_count)
        & (jnp.abs(summed) > float(coefficient_tolerance))
    )
    retained_count = jnp.sum(retained_unique, dtype=jnp.int32)
    retained_indices = jnp.nonzero(
        retained_unique, size=int(output_capacity), fill_value=0
    )[0]
    compact_labels = unique_labels[retained_indices]
    compact_valid = jnp.arange(int(output_capacity)) < retained_count
    compact_values = jnp.where(compact_valid, summed[retained_indices], 0.0)
    return (
        compact_labels,
        compact_values,
        compact_valid,
        retained_count,
        unique_count > int(output_capacity),
    )


def _reference_q(labels: jax.Array, reference: jax.Array) -> jax.Array:
    return jnp.where(reference < 0, 1, labels[jnp.maximum(reference, 0)])


def _vector_candidates(
    center: jax.Array,
    node_indices: jax.Array,
    active_references: jax.Array,
    spectator_references: jax.Array,
    step_is_left: jax.Array,
    step_mask: jax.Array,
    left_keys: jax.Array,
    left_values: jax.Array,
    right_keys: jax.Array,
    right_values: jax.Array,
    *,
    candidate_capacity: int,
    base: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    labels = jnp.repeat(center[None, :], candidate_capacity, axis=0)
    roots = jnp.ones((candidate_capacity,), dtype=jnp.int32)
    coefficients = jnp.zeros((candidate_capacity,), dtype=jnp.float64).at[0].set(
        np.sqrt(1.5)
    )
    valid = jnp.zeros((candidate_capacity,), dtype=jnp.bool_).at[0].set(True)
    overflow = jnp.asarray(False)
    deltas = jnp.asarray((-2, 0, 2), dtype=jnp.int32)

    def advance(carry, inputs):
        candidate_labels, candidate_roots, candidate_coefficients, candidate_valid, any_overflow = carry
        node_index, active_reference, spectator_reference, active_left, active_step = inputs

        def perform(current):
            labels_now, roots_now, coefficients_now, valid_now, overflow_now = current
            expanded_labels = jnp.repeat(labels_now, 3, axis=0)
            expanded_roots = jnp.repeat(roots_now, 3)
            expanded_coefficients = jnp.repeat(coefficients_now, 3)
            expanded_valid = jnp.repeat(valid_now, 3)
            parent_old = center[node_index]
            child_old = _reference_q(center, active_reference)
            spectator_q = _reference_q(center, spectator_reference)
            parent_new = parent_old + jnp.tile(deltas, candidate_capacity)
            keys = _pack_device(
                (
                    expanded_roots,
                    jnp.full_like(expanded_roots, child_old),
                    jnp.full_like(expanded_roots, spectator_q),
                    parent_new,
                    jnp.full_like(expanded_roots, parent_old),
                ),
                base,
            )
            factors = jax.lax.cond(
                active_left,
                lambda _: _lookup(left_keys, left_values, keys),
                lambda _: _lookup(right_keys, right_values, keys),
                operand=None,
            )
            expanded_valid = expanded_valid & (factors != 0.0)
            expanded_coefficients = expanded_coefficients * factors
            expanded_labels = expanded_labels.at[
                jnp.arange(candidate_capacity * 3), node_index
            ].set(parent_new)
            count = jnp.sum(expanded_valid, dtype=jnp.int32)
            indices = jnp.nonzero(
                expanded_valid, size=candidate_capacity, fill_value=0
            )[0]
            retained_valid = jnp.arange(candidate_capacity) < count
            return (
                expanded_labels[indices],
                parent_new[indices],
                expanded_coefficients[indices],
                retained_valid,
                overflow_now | (count > candidate_capacity),
            )

        following = jax.lax.cond(active_step, perform, lambda value: value, carry)
        return following, None

    (labels, roots, coefficients, valid, overflow), _ = jax.lax.scan(
        advance,
        (labels, roots, coefficients, valid, overflow),
        (
            node_indices,
            active_references,
            spectator_references,
            step_is_left,
            step_mask,
        ),
    )
    return labels, roots, coefficients, valid, overflow


def jax_tree_bond_connectivity(
    center_labels: jax.Array,
    plan: JaxTreeConnectivityPlan,
    bond_index: int | jax.Array,
    *,
    candidate_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return raw device bond connectivity and an explicit overflow flag.

    The returned arrays have shapes ``(candidate_capacity**2, n_sites-1)``,
    ``(candidate_capacity**2,)``, and ``(candidate_capacity**2,)``. Repeated
    states are valid and must be summed, not treated as independent samples.
    """

    if isinstance(candidate_capacity, bool) or not isinstance(
        candidate_capacity, Integral
    ):
        raise TypeError("candidate_capacity must be an integer")
    if candidate_capacity < 1:
        raise ValueError("candidate_capacity must be positive")
    center = jnp.asarray(center_labels, dtype=jnp.int32)
    if center.ndim != 1 or center.shape[0] != plan.n_sites - 1:
        raise ValueError(f"center_labels must have shape ({plan.n_sites - 1},)")
    index = jnp.asarray(bond_index, dtype=jnp.int32)
    base = plan.n_sites + 1
    left = _vector_candidates(
        center,
        jnp.asarray(plan.left_node_indices)[index],
        jnp.asarray(plan.left_active_references)[index],
        jnp.asarray(plan.left_spectator_references)[index],
        jnp.asarray(plan.left_step_is_left)[index],
        jnp.asarray(plan.left_step_mask)[index],
        jnp.asarray(plan.vector_left_keys),
        jnp.asarray(plan.vector_left_values),
        jnp.asarray(plan.vector_right_keys),
        jnp.asarray(plan.vector_right_values),
        candidate_capacity=candidate_capacity,
        base=base,
    )
    right = _vector_candidates(
        center,
        jnp.asarray(plan.right_node_indices)[index],
        jnp.asarray(plan.right_active_references)[index],
        jnp.asarray(plan.right_spectator_references)[index],
        jnp.asarray(plan.right_step_is_left)[index],
        jnp.asarray(plan.right_step_mask)[index],
        jnp.asarray(plan.vector_left_keys),
        jnp.asarray(plan.vector_left_values),
        jnp.asarray(plan.vector_right_keys),
        jnp.asarray(plan.vector_right_values),
        candidate_capacity=candidate_capacity,
        base=base,
    )
    left_labels, left_roots, left_coefficients, left_valid, left_overflow = left
    right_labels, right_roots, right_coefficients, right_valid, right_overflow = right
    left_old = _reference_q(center, jnp.asarray(plan.left_references)[index])
    right_old = _reference_q(center, jnp.asarray(plan.right_references)[index])
    total_q = center[jnp.asarray(plan.lca_indices)[index]]
    left_new = jnp.repeat(left_roots, candidate_capacity)
    right_new = jnp.tile(right_roots, candidate_capacity)
    scalar_keys = _pack_device(
        (
            left_new,
            right_new,
            jnp.full_like(left_new, left_old),
            jnp.full_like(left_new, right_old),
            jnp.full_like(left_new, total_q),
        ),
        base,
    )
    scalar = _lookup(
        jnp.asarray(plan.scalar_keys),
        jnp.asarray(plan.scalar_values),
        scalar_keys,
    )
    coefficients = (
        jnp.asarray(plan.bond_couplings)[index]
        * jnp.repeat(left_coefficients, candidate_capacity)
        * jnp.tile(right_coefficients, candidate_capacity)
        * scalar
    )
    valid = (
        jnp.repeat(left_valid, candidate_capacity)
        & jnp.tile(right_valid, candidate_capacity)
        & (jnp.abs(coefficients) > plan.coefficient_tolerance)
    )
    connected = (
        center[None, :]
        + jnp.repeat(left_labels - center[None, :], candidate_capacity, axis=0)
        + jnp.tile(right_labels - center[None, :], (candidate_capacity, 1))
    )
    return connected, coefficients, valid, left_overflow | right_overflow


def jax_tree_bond_connectivity_rectangular(
    center_labels: jax.Array,
    plan: JaxTreeConnectivityPlan,
    bond_index: int | jax.Array,
    *,
    left_candidate_capacity: int,
    right_candidate_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Rectangular-capacity variant used by path-structure bond buckets."""

    for name, capacity in (
        ("left_candidate_capacity", left_candidate_capacity),
        ("right_candidate_capacity", right_candidate_capacity),
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, Integral):
            raise TypeError(f"{name} must be an integer")
        if capacity < 1:
            raise ValueError(f"{name} must be positive")
    center = jnp.asarray(center_labels, dtype=jnp.int32)
    if center.ndim != 1 or center.shape[0] != plan.n_sites - 1:
        raise ValueError(f"center_labels must have shape ({plan.n_sites - 1},)")
    index = jnp.asarray(bond_index, dtype=jnp.int32)
    base = plan.n_sites + 1
    left = _vector_candidates(
        center,
        jnp.asarray(plan.left_node_indices)[index],
        jnp.asarray(plan.left_active_references)[index],
        jnp.asarray(plan.left_spectator_references)[index],
        jnp.asarray(plan.left_step_is_left)[index],
        jnp.asarray(plan.left_step_mask)[index],
        jnp.asarray(plan.vector_left_keys),
        jnp.asarray(plan.vector_left_values),
        jnp.asarray(plan.vector_right_keys),
        jnp.asarray(plan.vector_right_values),
        candidate_capacity=left_candidate_capacity,
        base=base,
    )
    right = _vector_candidates(
        center,
        jnp.asarray(plan.right_node_indices)[index],
        jnp.asarray(plan.right_active_references)[index],
        jnp.asarray(plan.right_spectator_references)[index],
        jnp.asarray(plan.right_step_is_left)[index],
        jnp.asarray(plan.right_step_mask)[index],
        jnp.asarray(plan.vector_left_keys),
        jnp.asarray(plan.vector_left_values),
        jnp.asarray(plan.vector_right_keys),
        jnp.asarray(plan.vector_right_values),
        candidate_capacity=right_candidate_capacity,
        base=base,
    )
    left_labels, left_roots, left_coefficients, left_valid, left_overflow = left
    right_labels, right_roots, right_coefficients, right_valid, right_overflow = right
    left_old = _reference_q(center, jnp.asarray(plan.left_references)[index])
    right_old = _reference_q(center, jnp.asarray(plan.right_references)[index])
    total_q = center[jnp.asarray(plan.lca_indices)[index]]
    left_new = jnp.repeat(left_roots, right_candidate_capacity)
    right_new = jnp.tile(right_roots, left_candidate_capacity)
    scalar_keys = _pack_device(
        (
            left_new,
            right_new,
            jnp.full_like(left_new, left_old),
            jnp.full_like(left_new, right_old),
            jnp.full_like(left_new, total_q),
        ),
        base,
    )
    scalar = _lookup(
        jnp.asarray(plan.scalar_keys),
        jnp.asarray(plan.scalar_values),
        scalar_keys,
    )
    coefficients = (
        jnp.asarray(plan.bond_couplings)[index]
        * jnp.repeat(left_coefficients, right_candidate_capacity)
        * jnp.tile(right_coefficients, left_candidate_capacity)
        * scalar
    )
    valid = (
        jnp.repeat(left_valid, right_candidate_capacity)
        & jnp.tile(right_valid, left_candidate_capacity)
        & (jnp.abs(coefficients) > plan.coefficient_tolerance)
    )
    connected = (
        center[None, :]
        + jnp.repeat(
            left_labels - center[None, :], right_candidate_capacity, axis=0
        )
        + jnp.tile(
            right_labels - center[None, :], (left_candidate_capacity, 1)
        )
    )
    return connected, coefficients, valid, left_overflow | right_overflow


__all__ = [
    "JaxTreeConnectivityPlan",
    "compile_jax_tree_connectivity_plan",
    "jax_coalesce_tree_connectivity",
    "jax_tree_bond_connectivity",
    "jax_tree_bond_connectivity_rectangular",
]
