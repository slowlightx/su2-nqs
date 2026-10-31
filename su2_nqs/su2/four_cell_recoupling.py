"""Exact recoupling between four-cell singlet pairing frames.

Four complete local Schur cells carry reduced labels ``(q,a)``.  A pairing
frame orders the physical cells as ``(i,j,k,l)`` and couples ``(i,j)`` and
``(k,l)`` to the same intermediate doubled spin before the final singlet.
This module changes only that association order.  Local multiplicity labels
are spectators and magnetic indices are summed with fixed Condon--Shortley
Clebsch--Gordan coefficients.

The returned matrix follows the repository connectivity convention:
``R[canonical, frame] = <canonical|frame>``.  It therefore maps frame
coefficients into the canonical ``((0,1),(2,3))`` basis.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from numbers import Integral

import numpy as np

from .clebsch_gordan import clebsch_gordan
from .local_schur import CellKind
from .local_schur_fusion import FourCellSchurState, four_cell_schur_states
from .quantum_numbers import magnetic_numbers


PairingOrder = tuple[int, int, int, int]


def validate_pairing_order(order: tuple[int, ...]) -> PairingOrder:
    """Return a checked permutation of the four physical cell indices."""

    raw = tuple(order)
    if len(raw) != 4:
        raise ValueError("a four-cell pairing order requires four entries")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError("pairing-order entries must be integers")
    checked = tuple(int(value) for value in raw)
    if tuple(sorted(checked)) != (0, 1, 2, 3):
        raise ValueError("pairing order must permute cell indices 0..3")
    return checked  # type: ignore[return-value]


def _physical_labels(
    state: FourCellSchurState,
    order: PairingOrder,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    q = [0] * 4
    a = [0] * 4
    for frame_cell, physical_cell in enumerate(order):
        q[physical_cell] = state.cell_q[frame_cell]
        a[physical_cell] = state.cell_a[frame_cell]
    return tuple(q), tuple(a)


def _frame_amplitude(
    physical_q: tuple[int, ...],
    physical_m: tuple[int, ...],
    pair_q: int,
    order: PairingOrder,
) -> float:
    """Return one fixed magnetic-index coefficient for a pairing frame."""

    first, second, third, fourth = order
    left_m = physical_m[first] + physical_m[second]
    right_m = physical_m[third] + physical_m[fourth]
    if left_m + right_m != 0:
        return 0.0
    left = clebsch_gordan(
        physical_q[first],
        physical_m[first],
        physical_q[second],
        physical_m[second],
        pair_q,
        left_m,
    )
    if left == 0.0:
        return 0.0
    right = clebsch_gordan(
        physical_q[third],
        physical_m[third],
        physical_q[fourth],
        physical_m[fourth],
        pair_q,
        right_m,
    )
    if right == 0.0:
        return 0.0
    root = clebsch_gordan(pair_q, left_m, pair_q, right_m, 0, 0)
    return float(left * right * root)


@lru_cache(maxsize=None)
def _magnetic_overlap(
    physical_q: tuple[int, int, int, int],
    canonical_pair_q: int,
    frame_pair_q: int,
    order: PairingOrder,
) -> float:
    total = 0.0
    magnetic = tuple(magnetic_numbers(q) for q in physical_q)
    canonical = (0, 1, 2, 3)
    for values in product(*magnetic):
        physical_m = tuple(int(value) for value in values)
        if sum(physical_m) != 0:
            continue
        left = _frame_amplitude(
            physical_q, physical_m, canonical_pair_q, canonical
        )
        if left == 0.0:
            continue
        right = _frame_amplitude(
            physical_q, physical_m, frame_pair_q, order
        )
        total += left * right
    return float(total)


@lru_cache(maxsize=None)
def four_cell_pairing_recoupling(
    kind: CellKind,
    order: PairingOrder,
) -> np.ndarray:
    """Return ``R[canonical,frame]=<canonical|frame>`` for four cells.

    Every angular momentum is a doubled integer.  The result is a real
    orthogonal matrix in the complete four-cell singlet multiplicity space.
    """

    checked = validate_pairing_order(order)
    states = four_cell_schur_states(kind)
    canonical_by_labels: dict[
        tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, FourCellSchurState]]
    ] = {}
    for index, state in enumerate(states):
        canonical_by_labels.setdefault((state.cell_q, state.cell_a), []).append(
            (index, state)
        )

    result = np.zeros((len(states), len(states)), dtype=np.float64)
    for frame_index, frame_state in enumerate(states):
        physical_q, physical_a = _physical_labels(frame_state, checked)
        candidates = canonical_by_labels.get((physical_q, physical_a), ())
        for canonical_index, canonical_state in candidates:
            result[canonical_index, frame_index] = _magnetic_overlap(
                physical_q,  # type: ignore[arg-type]
                canonical_state.pair_q,
                frame_state.pair_q,
                checked,
            )

    gram_error = float(
        max(
            np.max(np.abs(result.T @ result - np.eye(len(states)))),
            np.max(np.abs(result @ result.T - np.eye(len(states)))),
        )
    )
    if gram_error > 2.0e-12:
        raise RuntimeError(
            f"four-cell pairing recoupling is not orthogonal: {gram_error:.3e}"
        )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def pair_singlet_state_indices(kind: CellKind) -> np.ndarray:
    """Return states whose two cell pairs are independently singlets."""

    indices = np.asarray(
        [
            index
            for index, state in enumerate(four_cell_schur_states(kind))
            if state.pair_q == 0
        ],
        dtype=np.int32,
    )
    indices.setflags(write=False)
    return indices


__all__ = [
    "PairingOrder",
    "four_cell_pairing_recoupling",
    "pair_singlet_state_indices",
    "validate_pairing_order",
]
