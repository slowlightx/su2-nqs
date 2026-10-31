"""Complete four-cell singlet configurations and autoregressive masks.

The physical cells are first transformed by a complete local Schur transform.
Four resulting irreps are coupled as ``((0,1),(2,3)) -> q=0``.  Neural
models may see only the doubled irrep labels and local multiplicities defined
here; magnetic indices remain in fixed Clebsch--Gordan tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import numpy as np

from .local_schur import CellKind, cell_schur_spec
from .quantum_numbers import coupling_range


@dataclass(frozen=True, order=True)
class FourCellSchurState:
    """One complete reduced configuration for four identical Schur cells."""

    cell_q: tuple[int, int, int, int]
    cell_a: tuple[int, int, int, int]
    pair_q: int

    def __post_init__(self) -> None:
        if len(self.cell_q) != 4 or len(self.cell_a) != 4:
            raise ValueError("four local q and multiplicity labels are required")
        if any(value < 0 for value in (*self.cell_q, *self.cell_a, self.pair_q)):
            raise ValueError("Schur labels must be nonnegative doubled integers")
        if self.pair_q not in coupling_range(self.cell_q[0], self.cell_q[1]):
            raise ValueError("left cell pair cannot reach pair_q")
        if self.pair_q not in coupling_range(self.cell_q[2], self.cell_q[3]):
            raise ValueError("right cell pair cannot reach pair_q")


@dataclass(frozen=True, order=True)
class FourCellSchurSectorState:
    """One four-cell multiplicity state in an arbitrary total-spin sector."""

    cell_q: tuple[int, int, int, int]
    cell_a: tuple[int, int, int, int]
    left_pair_q: int
    right_pair_q: int
    target_q: int

    def __post_init__(self) -> None:
        if len(self.cell_q) != 4 or len(self.cell_a) != 4:
            raise ValueError("four local q and multiplicity labels are required")
        labels = (
            *self.cell_q,
            *self.cell_a,
            self.left_pair_q,
            self.right_pair_q,
            self.target_q,
        )
        if any(value < 0 for value in labels):
            raise ValueError("Schur labels must be nonnegative doubled integers")
        if self.left_pair_q not in coupling_range(self.cell_q[0], self.cell_q[1]):
            raise ValueError("left cell pair cannot reach left_pair_q")
        if self.right_pair_q not in coupling_range(self.cell_q[2], self.cell_q[3]):
            raise ValueError("right cell pair cannot reach right_pair_q")
        if self.target_q not in coupling_range(self.left_pair_q, self.right_pair_q):
            raise ValueError("cell pairs cannot reach target_q")


@lru_cache(maxsize=None)
def local_schur_channels(kind: CellKind) -> tuple[tuple[int, int], ...]:
    """Return deterministic ``(q,a)`` local-channel actions."""

    spec = cell_schur_spec(kind)
    return tuple(
        (q, a)
        for q in spec.q_values
        for a in range(spec.multiplicity(q))
    )


@lru_cache(maxsize=None)
def four_cell_schur_states(kind: CellKind) -> tuple[FourCellSchurState, ...]:
    """Enumerate the complete four-cell total-singlet multiplicity basis."""

    channels = local_schur_channels(kind)
    states = []
    for q0, a0 in channels:
        for q1, a1 in channels:
            for pair_q in coupling_range(q0, q1):
                for q2, a2 in channels:
                    for q3, a3 in channels:
                        if pair_q not in coupling_range(q2, q3):
                            continue
                        states.append(
                            FourCellSchurState(
                                (q0, q1, q2, q3),
                                (a0, a1, a2, a3),
                                pair_q,
                            )
                        )
    return tuple(states)


@lru_cache(maxsize=None)
def four_cell_schur_sector_states(
    kind: CellKind,
    target_q: int,
) -> tuple[FourCellSchurSectorState, ...]:
    """Enumerate the complete four-cell multiplicity basis at ``target_q``."""

    if isinstance(target_q, bool) or not isinstance(target_q, Integral):
        raise TypeError("target_q must be a doubled integer")
    target_q = int(target_q)
    if target_q < 0:
        raise ValueError("target_q must be nonnegative")
    channels = local_schur_channels(kind)
    states = []
    for q0, a0 in channels:
        for q1, a1 in channels:
            for left_pair_q in coupling_range(q0, q1):
                for q2, a2 in channels:
                    for q3, a3 in channels:
                        for right_pair_q in coupling_range(q2, q3):
                            if target_q not in coupling_range(
                                left_pair_q, right_pair_q
                            ):
                                continue
                            states.append(
                                FourCellSchurSectorState(
                                    (q0, q1, q2, q3),
                                    (a0, a1, a2, a3),
                                    left_pair_q,
                                    right_pair_q,
                                    target_q,
                                )
                            )
    return tuple(states)


@lru_cache(maxsize=None)
def four_cell_action_table(kind: CellKind) -> np.ndarray:
    """Return immutable actions ``c0,c1,pair_q,c2,c3`` for every state."""

    channels = local_schur_channels(kind)
    channel_index = {channel: index for index, channel in enumerate(channels)}
    rows = [
        (
            channel_index[(state.cell_q[0], state.cell_a[0])],
            channel_index[(state.cell_q[1], state.cell_a[1])],
            state.pair_q,
            channel_index[(state.cell_q[2], state.cell_a[2])],
            channel_index[(state.cell_q[3], state.cell_a[3])],
        )
        for state in four_cell_schur_states(kind)
    ]
    result = np.asarray(rows, dtype=np.int32)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def four_cell_sector_action_table(kind: CellKind, target_q: int) -> np.ndarray:
    """Return actions ``c0,c1,q01,c2,c3,q23`` for one total-spin sector."""

    channels = local_schur_channels(kind)
    channel_index = {channel: index for index, channel in enumerate(channels)}
    rows = [
        (
            channel_index[(state.cell_q[0], state.cell_a[0])],
            channel_index[(state.cell_q[1], state.cell_a[1])],
            state.left_pair_q,
            channel_index[(state.cell_q[2], state.cell_a[2])],
            channel_index[(state.cell_q[3], state.cell_a[3])],
            state.right_pair_q,
        )
        for state in four_cell_schur_sector_states(kind, target_q)
    ]
    result = np.asarray(rows, dtype=np.int32)
    if result.size == 0:
        result = np.empty((0, 6), dtype=np.int32)
    result.setflags(write=False)
    return result


def four_cell_maximum_action_count(kind: CellKind) -> int:
    """Return the shared categorical width for channel and pair-q decisions."""

    spec = cell_schur_spec(kind)
    return max(len(local_schur_channels(kind)), 2 * spec.cell_size + 1)


@lru_cache(maxsize=None)
def four_cell_autoregressive_masks(kind: CellKind) -> np.ndarray:
    """Return exact pre-softmax masks for every complete action sequence.

    Shape is ``(configuration, decision, action)``.  Rows sharing the same
    prefix are required to carry identical masks.  The fourth decision masks
    local channels that cannot be completed by the final cell, so no invalid
    sample ever needs post-generation repair.
    """

    channels = local_schur_channels(kind)
    actions = four_cell_action_table(kind)
    width = four_cell_maximum_action_count(kind)
    reachable_pair_q = {
        q
        for q0, _ in channels
        for q1, _ in channels
        for q in coupling_range(q0, q1)
    }
    masks = np.zeros((len(actions), 5, width), dtype=np.bool_)
    masks[:, 0, : len(channels)] = True
    masks[:, 1, : len(channels)] = True
    for row_index, row in enumerate(actions):
        q0 = channels[int(row[0])][0]
        q1 = channels[int(row[1])][0]
        for q in coupling_range(q0, q1):
            if q in reachable_pair_q:
                masks[row_index, 2, q] = True
        pair_q = int(row[2])
        for channel2, (q2, _a2) in enumerate(channels):
            if any(pair_q in coupling_range(q2, q3) for q3, _a3 in channels):
                masks[row_index, 3, channel2] = True
        q2 = channels[int(row[3])][0]
        for channel3, (q3, _a3) in enumerate(channels):
            if pair_q in coupling_range(q2, q3):
                masks[row_index, 4, channel3] = True
    selected = np.take_along_axis(masks, actions[..., None], axis=-1)[..., 0]
    if not np.all(selected):
        raise RuntimeError("a registered four-cell state violates its exact mask")
    for decision in range(5):
        prefixes: dict[tuple[int, ...], np.ndarray] = {}
        for row, mask in zip(actions, masks[:, decision], strict=True):
            prefix = tuple(int(value) for value in row[:decision])
            previous = prefixes.setdefault(prefix, mask)
            if not np.array_equal(previous, mask):
                raise RuntimeError("autoregressive mask depends on a future action")
    masks.setflags(write=False)
    return masks


@lru_cache(maxsize=None)
def four_cell_sector_autoregressive_masks(
    kind: CellKind,
    target_q: int,
) -> np.ndarray:
    """Return exact six-decision masks for one four-cell spin sector."""

    channels = local_schur_channels(kind)
    actions = four_cell_sector_action_table(kind, target_q)
    if len(actions) == 0:
        raise ValueError("target_q is unreachable for four local cells")
    width = four_cell_maximum_action_count(kind)
    masks = np.zeros((len(actions), 6, width), dtype=np.bool_)
    masks[:, 0, : len(channels)] = True
    masks[:, 1, : len(channels)] = True
    possible_right_q = {
        q
        for q2, _ in channels
        for q3, _ in channels
        for q in coupling_range(q2, q3)
    }
    for row_index, row in enumerate(actions):
        q0 = channels[int(row[0])][0]
        q1 = channels[int(row[1])][0]
        for left_q in coupling_range(q0, q1):
            if any(
                target_q in coupling_range(left_q, right_q)
                for right_q in possible_right_q
            ):
                masks[row_index, 2, left_q] = True
        left_q = int(row[2])
        for channel2, (q2, _a2) in enumerate(channels):
            if any(
                target_q in coupling_range(left_q, right_q)
                for q3, _a3 in channels
                for right_q in coupling_range(q2, q3)
            ):
                masks[row_index, 3, channel2] = True
        q2 = channels[int(row[3])][0]
        for channel3, (q3, _a3) in enumerate(channels):
            if any(
                target_q in coupling_range(left_q, right_q)
                for right_q in coupling_range(q2, q3)
            ):
                masks[row_index, 4, channel3] = True
        q3 = channels[int(row[4])][0]
        for right_q in coupling_range(q2, q3):
            if target_q in coupling_range(left_q, right_q):
                masks[row_index, 5, right_q] = True
    selected = np.take_along_axis(masks, actions[..., None], axis=-1)[..., 0]
    if not np.all(selected):
        raise RuntimeError("a registered sector state violates its exact mask")
    for decision in range(6):
        prefixes: dict[tuple[int, ...], np.ndarray] = {}
        for row, mask in zip(actions, masks[:, decision], strict=True):
            prefix = tuple(int(value) for value in row[:decision])
            previous = prefixes.setdefault(prefix, mask)
            if not np.array_equal(previous, mask):
                raise RuntimeError("sector mask depends on a future action")
    masks.setflags(write=False)
    return masks


__all__ = [
    "FourCellSchurState",
    "FourCellSchurSectorState",
    "four_cell_action_table",
    "four_cell_autoregressive_masks",
    "four_cell_maximum_action_count",
    "four_cell_schur_states",
    "four_cell_schur_sector_states",
    "four_cell_sector_action_table",
    "four_cell_sector_autoregressive_masks",
    "local_schur_channels",
]
