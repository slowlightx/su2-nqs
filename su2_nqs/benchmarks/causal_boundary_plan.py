"""Offline compiler for production-facing causal SU(2) boundary messages."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..models.causal_boundary_tree import (
    BOUNDARY_DELTA_Q,
    CausalBoundaryMessagePlan,
)
from ..models.compact_tree_gru import split_tables
from ..operators.permutation import heisenberg_graph_fusion_matrix
from ..su2.fusion_path import enumerate_paths
from ..su2.fusion_tree import FusionTree
from ..su2.quantum_numbers import is_admissible_triangle
from ..su2.reduced_vector import compile_weighted_spin_vector
from .boundary_operator_channels import (
    BoundaryBond,
    compile_boundary_recoupling_plan,
    normalize_boundary_bonds,
    partition_boundary_bonds,
)


def _all_block_paths(n_sites: int) -> tuple[tuple[int, ...], ...]:
    parity = n_sites % 2
    return tuple(
        path
        for q in range(parity, n_sites + 1, 2)
        for path in enumerate_paths(n_sites, q)
    )


def _canonical_svd_right_weights(plan) -> np.ndarray:
    """Right endpoint vectors with channel scale and deterministic signs."""

    result = np.zeros(
        (plan.channel_rank, len(plan.prefix_sites)), dtype=np.float64
    )
    for channel in range(plan.channel_rank):
        scale = math.sqrt(float(plan.singular_values[channel]))
        left = scale * np.asarray(plan.left_vectors[:, channel])
        right = scale * np.asarray(plan.right_vectors[:, channel])
        if len(left):
            pivot = float(left[int(np.argmax(np.abs(left)))])
        elif len(right):
            pivot = float(right[int(np.argmax(np.abs(right)))])
        else:
            pivot = 0.0
        result[channel] = -right if pivot < 0.0 else right
    return result


def _action_prefix_tables(
    n_sites: int,
    decision: int,
    paths: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Map reverse-coupled revealed paths to root-first compact actions."""

    tree = FusionTree.left_associated(n_sites)
    table_left, table_right, table_valid = split_tables(tree)
    actions = np.empty((len(paths), decision + 1), dtype=np.int32)
    parents = np.empty_like(actions)
    final_decision = decision == n_sites - 2
    for row, path in enumerate(paths):
        for step in range(decision + 1):
            if step == 0:
                parent_q = 0
            else:
                parent_q = int(path[step - 1])
            if final_decision and step == decision:
                desired_left = desired_right = 1
            else:
                desired_left = int(path[step])
                desired_right = 1
            matches = np.flatnonzero(
                table_valid[step, parent_q]
                & (table_left[step, parent_q] == desired_left)
                & (table_right[step, parent_q] == desired_right)
            )
            if len(matches) != 1:
                raise RuntimeError(
                    "a revealed reverse path does not map to one compact action"
                )
            actions[row, step] = int(matches[0])
            parents[row, step] = parent_q
    if len({tuple(row) for row in actions}) != len(actions):
        raise RuntimeError("revealed paths do not have unique action prefixes")
    return actions, parents


def _closed_scalar_matrix(
    paths: tuple[tuple[int, ...], ...],
    revealed_order: np.ndarray,
    bonds: tuple[BoundaryBond, ...],
) -> np.ndarray:
    """Compile revealed-revealed Heisenberg bonds in the reverse path basis."""

    dimension = len(paths)
    answer = np.zeros((dimension, dimension), dtype=np.complex128)
    if not bonds:
        return answer
    positions = {path: index for index, path in enumerate(paths)}
    site_positions = {
        int(site): position for position, site in enumerate(revealed_order)
    }
    local_bonds = tuple(
        (
            site_positions[bond.first],
            site_positions[bond.second],
            bond.coupling,
        )
        for bond in bonds
    )
    for q in sorted({path[-1] for path in paths}):
        block_paths = enumerate_paths(len(revealed_order), q)
        matrix = np.asarray(
            heisenberg_graph_fusion_matrix(
                len(revealed_order), local_bonds, target_q=q
            ).toarray(),
            dtype=np.complex128,
        )
        indices = np.asarray([positions[path] for path in block_paths])
        answer[np.ix_(indices, indices)] = matrix
    return answer


def compile_causal_boundary_message_plan(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond
        | tuple[int, int]
        | tuple[int, int, float]
        | tuple[int, int, float, int]
    ],
    *,
    target_q: int = 0,
    rank_tolerance: float = 1.0e-12,
) -> CausalBoundaryMessagePlan:
    """Compile exact revealed-prefix operators for a left-associated singlet.

    Crossing couplings use the minimal fixed-cut SVD gauge.  Each physical
    channel is a single rank-1 SU(2) vector and all allowed doubled boundary
    changes ``(-2,0,+2)`` are retained.  Compilation uses no wavefunction and
    performs no contraction over an unrevealed suffix.
    """

    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    if target_q != 0:
        raise ValueError("the current reversed-prefix compiler requires target_q=0")
    if n_sites % 2:
        raise ValueError("a spin-half singlet requires an even site count")
    if rank_tolerance < 0.0:
        raise ValueError("rank_tolerance must be non-negative")
    normalized = normalize_boundary_bonds(bonds)
    for bond in normalized:
        if not 0 <= bond.first < n_sites or not 0 <= bond.second < n_sites:
            raise ValueError("a bond endpoint lies outside the system")
    cut_plans = tuple(
        compile_boundary_recoupling_plan(
            n_sites, normalized, decision, rank_tolerance=rank_tolerance
        )
        for decision in range(n_sites - 1)
    )
    maximum_channels = max(
        (plan.channel_rank for plan in cut_plans), default=0
    )

    revealed_paths = []
    action_sequences = []
    parent_q_sequences = []
    vector_operators = []
    vector_transition_valid = []
    channel_active = []
    closed_scalar_operators = []
    revealed_site_order = []
    vector_endpoint_weights = []

    for decision, cut in enumerate(cut_plans):
        final_decision = decision == n_sites - 2
        revealed_count = n_sites if final_decision else decision + 1
        sites = np.asarray(cut.prefix_sites[::-1], dtype=np.int32)
        if len(sites) != revealed_count:
            raise RuntimeError("boundary cut and AR revealed-site count disagree")
        paths = (
            tuple(enumerate_paths(n_sites, target_q))
            if final_decision
            else tuple(
                path
                for path in _all_block_paths(revealed_count)
                if path[-1] <= n_sites - revealed_count
                and ((n_sites - revealed_count - path[-1]) % 2 == 0)
            )
        )
        path_positions = {path: index for index, path in enumerate(paths)}
        actions, parents = _action_prefix_tables(n_sites, decision, paths)
        operators = np.zeros(
            (
                maximum_channels,
                len(BOUNDARY_DELTA_Q),
                len(paths),
                len(paths),
            ),
            dtype=np.complex128,
        )
        weights = np.zeros(
            (maximum_channels, revealed_count), dtype=np.float64
        )
        active = np.zeros((maximum_channels,), dtype=bool)
        right_weights = _canonical_svd_right_weights(cut)
        for channel in range(cut.channel_rank):
            # cut.prefix_sites is in ascending physical order, whereas the
            # causal revealed block is coupled from the right boundary inward.
            endpoint_weights = np.asarray(
                right_weights[channel, ::-1], dtype=np.float64
            )
            weights[channel] = endpoint_weights
            active[channel] = True
            table = compile_weighted_spin_vector(endpoint_weights)
            for new, old, q_new, q_old, value in zip(
                table.new_path_indices,
                table.old_path_indices,
                table.q_new,
                table.q_old,
                table.reduced_values,
                strict=True,
            ):
                new_path = table.paths[int(new)]
                old_path = table.paths[int(old)]
                new_position = path_positions.get(new_path)
                old_position = path_positions.get(old_path)
                if new_position is None or old_position is None:
                    continue
                delta = int(q_new - q_old)
                if delta in BOUNDARY_DELTA_Q:
                    slot = BOUNDARY_DELTA_Q.index(delta)
                    operators[
                        channel, slot, new_position, old_position
                    ] = value

        transition_mask = np.zeros(
            (len(paths), len(BOUNDARY_DELTA_Q)), dtype=bool
        )
        reachable_q = {path[-1] for path in paths}
        for old_index, old_path in enumerate(paths):
            for slot, delta in enumerate(BOUNDARY_DELTA_Q):
                q_new = old_path[-1] + delta
                transition_mask[old_index, slot] = (
                    q_new in reachable_q
                    and is_admissible_triangle(old_path[-1], 2, q_new)
                )

        closed_bonds = partition_boundary_bonds(
            n_sites, normalized, decision
        )["prefix"]
        scalar = _closed_scalar_matrix(paths, sites, closed_bonds)
        if tuple(path_positions) != paths:
            raise RuntimeError("revealed paths lost deterministic ordering")

        revealed_paths.append(np.asarray(paths, dtype=np.int32))
        action_sequences.append(actions)
        parent_q_sequences.append(parents)
        vector_operators.append(operators)
        vector_transition_valid.append(transition_mask)
        channel_active.append(active)
        closed_scalar_operators.append(scalar)
        revealed_site_order.append(sites)
        vector_endpoint_weights.append(weights)

    return CausalBoundaryMessagePlan(
        n_sites=n_sites,
        target_q=target_q,
        maximum_channels=maximum_channels,
        revealed_paths=tuple(revealed_paths),
        action_sequences=tuple(action_sequences),
        parent_q_sequences=tuple(parent_q_sequences),
        vector_operators=tuple(vector_operators),
        vector_transition_valid=tuple(vector_transition_valid),
        channel_active=tuple(channel_active),
        closed_scalar_operators=tuple(closed_scalar_operators),
        revealed_site_order=tuple(revealed_site_order),
        vector_endpoint_weights=tuple(vector_endpoint_weights),
    )


def estimate_dense_causal_boundary_storage(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond
        | tuple[int, int]
        | tuple[int, int, float]
        | tuple[int, int, float, int]
    ],
    *,
    target_q: int = 0,
    rank_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Estimate the exact dense reference-plan storage without building it.

    This deliberately counts the current padded implementation, whose vector
    tensors use the maximum retained spatial rank at every cut.  It is a guard
    against accidentally treating the exhaustive reference scaffold as an
    N=16 production kernel.
    """

    if n_sites < 2 or n_sites % 2 or target_q != 0:
        raise ValueError("the estimator currently supports even singlet systems")
    normalized = normalize_boundary_bonds(bonds)
    cuts = tuple(
        compile_boundary_recoupling_plan(
            n_sites, normalized, decision, rank_tolerance=rank_tolerance
        )
        for decision in range(n_sites - 1)
    )
    maximum_channels = max((cut.channel_rank for cut in cuts), default=0)
    rows = []
    total = 0
    for decision, cut in enumerate(cuts):
        final = decision == n_sites - 2
        revealed_count = n_sites if final else decision + 1
        if final:
            path_count = len(enumerate_paths(n_sites, target_q))
        else:
            hidden_count = n_sites - revealed_count
            path_count = sum(
                1
                for path in _all_block_paths(revealed_count)
                if path[-1] <= hidden_count
                and ((hidden_count - path[-1]) % 2 == 0)
            )
        vector_bytes = (
            maximum_channels
            * len(BOUNDARY_DELTA_Q)
            * path_count
            * path_count
            * np.dtype(np.complex128).itemsize
        )
        scalar_bytes = (
            path_count * path_count * np.dtype(np.complex128).itemsize
        )
        auxiliary_bytes = (
            path_count * revealed_count * np.dtype(np.int32).itemsize
            + 2
            * path_count
            * (decision + 1)
            * np.dtype(np.int32).itemsize
            + path_count * len(BOUNDARY_DELTA_Q) * np.dtype(bool).itemsize
            + maximum_channels
            * revealed_count
            * np.dtype(np.float64).itemsize
        )
        cut_total = int(vector_bytes + scalar_bytes + auxiliary_bytes)
        total += cut_total
        rows.append(
            {
                "decision": decision,
                "revealed_sites": revealed_count,
                "reachable_prefixes": path_count,
                "retained_spatial_rank": cut.channel_rank,
                "padded_maximum_spatial_rank": maximum_channels,
                "dense_vector_bytes": int(vector_bytes),
                "dense_scalar_bytes": int(scalar_bytes),
                "auxiliary_bytes": int(auxiliary_bytes),
                "total_bytes": cut_total,
            }
        )
    return {
        "n_sites": n_sites,
        "target_q": target_q,
        "maximum_retained_spatial_rank": maximum_channels,
        "total_bytes": int(total),
        "total_gibibytes": float(total / 2**30),
        "cuts": rows,
    }


__all__ = [
    "compile_causal_boundary_message_plan",
    "estimate_dense_causal_boundary_storage",
]
