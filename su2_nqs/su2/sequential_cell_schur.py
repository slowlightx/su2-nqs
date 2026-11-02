"""Exact cell-sequential coordinates for complete SU(2) Schur spaces.

Each four-spin cell contributes one reduced decision ``(q_i, a_i, Q_i)``.
Here ``q_i`` is the cell irrep, ``a_i`` resolves its complete local
multiplicity, and ``Q_i`` is the cumulative irrep after cells ``0..i``.
Candidate actions are masked *before* normalization by an exact dynamic
program which asks whether the remaining cells can still reach ``target_q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np

from .many_cell_schur import (
    ManyCellSchurLayout,
    four_spin_multiplicity_channels,
    validate_layout_state,
)
from .quantum_numbers import coupling_range, validate_q
from .tree_rotations import leaf_signature


@dataclass(frozen=True)
class SequentialCellActionTables:
    """Padded exact candidates indexed by ``[cell, previous_Q, action]``."""

    cell_q: np.ndarray
    cell_a: np.ndarray
    cumulative_q: np.ndarray
    valid: np.ndarray

    @property
    def action_count(self) -> int:
        return int(self.valid.shape[-1])


def _validate_problem(n_cells: int, target_q: int) -> tuple[int, int]:
    if isinstance(n_cells, bool) or not isinstance(n_cells, Integral):
        raise TypeError("n_cells must be an integer")
    if int(n_cells) < 1:
        raise ValueError("n_cells must be positive")
    validate_q(target_q, name="target_q")
    n_cells, target_q = int(n_cells), int(target_q)
    if target_q > 4 * n_cells or (target_q - 4 * n_cells) % 2:
        raise ValueError("target_q is unreachable for the requested cells")
    return n_cells, target_q


@lru_cache(maxsize=None)
def _can_finish(current_q: int, remaining_cells: int, target_q: int) -> bool:
    if remaining_cells == 0:
        return current_q == target_q
    for local_q in four_spin_multiplicity_channels():
        for new_q in coupling_range(current_q, local_q):
            if _can_finish(new_q, remaining_cells - 1, target_q):
                return True
    return False


@lru_cache(maxsize=None)
def sequential_cell_action_tables(
    n_cells: int,
    target_q: int = 0,
) -> SequentialCellActionTables:
    """Return all locally valid actions with exact future reachability masks."""

    n_cells, target_q = _validate_problem(n_cells, target_q)
    channels = four_spin_multiplicity_channels()
    maximum_q = 4 * n_cells
    rows: list[list[list[tuple[int, int, int]]]] = []
    maximum_actions = 1
    for cell in range(n_cells):
        cell_rows: list[list[tuple[int, int, int]]] = []
        remaining = n_cells - cell - 1
        for previous_q in range(maximum_q + 1):
            candidates: list[tuple[int, int, int]] = []
            for local_q, multiplicities in channels.items():
                possible_q = (local_q,) if cell == 0 else coupling_range(previous_q, local_q)
                for multiplicity in range(len(multiplicities)):
                    for cumulative_q in possible_q:
                        if _can_finish(cumulative_q, remaining, target_q):
                            candidates.append((local_q, multiplicity, cumulative_q))
            # Only the sentinel previous_q=0 is meaningful for the first cell.
            if cell == 0 and previous_q != 0:
                candidates = []
            cell_rows.append(candidates)
            maximum_actions = max(maximum_actions, len(candidates))
        rows.append(cell_rows)

    shape = (n_cells, maximum_q + 1, maximum_actions)
    local_q = np.zeros(shape, dtype=np.int32)
    multiplicity = np.zeros(shape, dtype=np.int32)
    cumulative_q = np.zeros(shape, dtype=np.int32)
    valid = np.zeros(shape, dtype=np.bool_)
    for cell, cell_rows in enumerate(rows):
        for previous_q, candidates in enumerate(cell_rows):
            for action, (q_value, a_value, total_value) in enumerate(candidates):
                local_q[cell, previous_q, action] = q_value
                multiplicity[cell, previous_q, action] = a_value
                cumulative_q[cell, previous_q, action] = total_value
                valid[cell, previous_q, action] = True
    for array in (local_q, multiplicity, cumulative_q, valid):
        array.setflags(write=False)
    return SequentialCellActionTables(local_q, multiplicity, cumulative_q, valid)


@lru_cache(maxsize=None)
def sequential_cumulative_indices(layout: ManyCellSchurLayout) -> tuple[int, ...]:
    """Postorder indices of cumulative ``Q_i`` for a sequential cell tree."""

    signatures = tuple(leaf_signature(node) for node in layout.tree.internal_nodes)
    indices = []
    for cell in range(layout.n_cells):
        signature = tuple(range(4 * (cell + 1)))
        try:
            index = signatures.index(signature)
        except ValueError as error:
            raise ValueError("layout is not a left-sequential four-spin cell tree") from error
        indices.append(index)
    if indices[0] != layout.cell_root_indices[0]:
        raise ValueError("the first cumulative irrep must be the first cell irrep")
    return tuple(indices)


@lru_cache(maxsize=None)
def sequential_cell_coordinates(
    layout: ManyCellSchurLayout,
) -> tuple[tuple[float, float], ...]:
    """Centered, scale-free coordinates of the physical four-spin cells."""

    coordinates = np.asarray(layout.site_coordinates, dtype=np.float64)
    centers = np.asarray(
        [np.mean(coordinates[4 * i : 4 * i + 4], axis=0) for i in range(layout.n_cells)]
    )
    span = np.maximum(np.ptp(centers, axis=0), 1.0)
    centers = (centers - np.mean(centers, axis=0, keepdims=True)) / span
    return tuple(tuple(float(value) for value in row) for row in centers)


def labels_to_sequential_actions(
    layout: ManyCellSchurLayout,
    labels: Sequence[Sequence[int]] | np.ndarray,
    *,
    target_q: int = 0,
) -> np.ndarray:
    """Decode complete tree labels as ``(..., cell, [q,a,Q])`` actions."""

    values = np.asarray(labels, dtype=np.int32)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != layout.n_sites - 1:
        raise ValueError(f"labels must have shape (...,{layout.n_sites - 1})")
    decoded = layout.decode(values)
    cumulative = values[:, np.asarray(sequential_cumulative_indices(layout))]
    actions = np.stack((decoded.cell_q, decoded.cell_a, cumulative), axis=-1)
    tables = sequential_cell_action_tables(layout.n_cells, target_q)
    previous = np.zeros(len(values), dtype=np.int32)
    state_valid = np.ones(len(values), dtype=np.bool_)
    for cell in range(layout.n_cells):
        safe_previous = np.clip(previous, 0, tables.valid.shape[1] - 1)
        matches = (
            tables.valid[cell, safe_previous]
            & (tables.cell_q[cell, safe_previous] == actions[:, cell, 0, None])
            & (tables.cell_a[cell, safe_previous] == actions[:, cell, 1, None])
            & (tables.cumulative_q[cell, safe_previous] == actions[:, cell, 2, None])
        )
        state_valid &= np.any(matches, axis=1)
        previous = actions[:, cell, 2]
    if not np.all(state_valid):
        raise ValueError("labels do not define a valid reachable cell sequence")
    return actions[0] if squeeze else actions


def sequential_actions_to_labels(
    layout: ManyCellSchurLayout,
    actions: Sequence[Sequence[Sequence[int]]] | np.ndarray,
    *,
    target_q: int = 0,
) -> np.ndarray:
    """Encode complete ``(q,a,Q)`` sequences as tree-postorder labels."""

    values = np.asarray(actions, dtype=np.int32)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (layout.n_cells, 3):
        raise ValueError(f"actions must have shape (...,{layout.n_cells},3)")
    tables = sequential_cell_action_tables(layout.n_cells, target_q)
    channels = four_spin_multiplicity_channels()
    cumulative_indices = sequential_cumulative_indices(layout)
    labels = np.zeros((len(values), layout.n_sites - 1), dtype=np.int32)
    for row, sequence in enumerate(values):
        previous_q = 0
        for cell, (local_q, multiplicity, cumulative_q) in enumerate(sequence):
            mask = tables.valid[cell, previous_q]
            matches = (
                mask
                & (tables.cell_q[cell, previous_q] == local_q)
                & (tables.cell_a[cell, previous_q] == multiplicity)
                & (tables.cumulative_q[cell, previous_q] == cumulative_q)
            )
            if not np.any(matches):
                raise ValueError(f"invalid or unreachable action at cell {cell}")
            try:
                left_q, right_q = channels[int(local_q)][int(multiplicity)]
            except (KeyError, IndexError) as error:
                raise ValueError("invalid local four-spin multiplicity") from error
            left_index, right_index = layout.cell_pair_indices[cell]
            labels[row, left_index] = left_q
            labels[row, right_index] = right_q
            labels[row, layout.cell_root_indices[cell]] = local_q
            labels[row, cumulative_indices[cell]] = cumulative_q
            previous_q = int(cumulative_q)
        validate_layout_state(layout, labels[row], target_q=target_q)
    return labels[0] if squeeze else labels


def enumerate_sequential_cell_actions(
    n_cells: int,
    target_q: int = 0,
) -> np.ndarray:
    """Enumerate the exact reduced support (intended for small validation)."""

    tables = sequential_cell_action_tables(n_cells, target_q)
    rows: list[tuple[tuple[int, int, int], ...]] = []

    def visit(cell: int, previous_q: int, prefix: tuple[tuple[int, int, int], ...]) -> None:
        if cell == n_cells:
            rows.append(prefix)
            return
        for action in np.flatnonzero(tables.valid[cell, previous_q]):
            candidate = (
                int(tables.cell_q[cell, previous_q, action]),
                int(tables.cell_a[cell, previous_q, action]),
                int(tables.cumulative_q[cell, previous_q, action]),
            )
            visit(cell + 1, candidate[2], prefix + (candidate,))

    visit(0, 0, ())
    return np.asarray(rows, dtype=np.int32)


__all__ = [
    "SequentialCellActionTables",
    "enumerate_sequential_cell_actions",
    "labels_to_sequential_actions",
    "sequential_actions_to_labels",
    "sequential_cell_action_tables",
    "sequential_cell_coordinates",
    "sequential_cumulative_indices",
]
