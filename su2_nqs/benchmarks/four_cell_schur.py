"""Independent exact references for four-cell local-Schur singlet states."""

from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.local_schur import CellKind, cell_schur_spec, cell_schur_tensor
from ..su2.local_schur_fusion import (
    four_cell_schur_sector_states,
    four_cell_schur_states,
    local_schur_channels,
)
from ..su2.quantum_numbers import magnetic_numbers, validate_m
from .clebsch_tree_exact import fixed_m_spin_basis


def _default_supports(kind: CellKind) -> tuple[tuple[int, ...], ...]:
    size = cell_schur_spec(kind).cell_size
    return tuple(
        tuple(range(cell * size, (cell + 1) * size)) for cell in range(4)
    )


def _validate_supports(
    kind: CellKind,
    supports: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    spec = cell_schur_spec(kind)
    raw = _default_supports(kind) if supports is None else tuple(
        tuple(int(site) for site in support) for support in supports
    )
    if len(raw) != 4 or any(len(support) != spec.cell_size for support in raw):
        raise ValueError("supports must contain four complete equal Schur cells")
    flattened = tuple(site for support in raw for site in support)
    if tuple(sorted(flattened)) != tuple(range(4 * spec.cell_size)):
        raise ValueError("four supports must partition physical sites 0..N-1")
    return raw


@lru_cache(maxsize=None)
def _pair_carriers(
    kind: CellKind,
) -> tuple[tuple[int, np.ndarray, tuple[tuple[int, int, int, int], ...]], ...]:
    """Return orthonormal two-cell carriers grouped by total doubled spin."""

    spec = cell_schur_spec(kind)
    channels = local_schur_channels(kind)
    table = np.asarray(cell_schur_tensor(kind), dtype=np.float64)
    local_dimension = 1 << spec.cell_size
    maximum_pair_q = 2 * spec.cell_size
    grouped: dict[int, list[tuple[int, int, int, int]]] = {}
    for q0, a0 in channels:
        for q1, a1 in channels:
            for pair_q in range(abs(q0 - q1), q0 + q1 + 1, 2):
                grouped.setdefault(pair_q, []).append((q0, a0, q1, a1))
    output = []
    for pair_q in sorted(grouped):
        labels = tuple(grouped[pair_q])
        values = np.zeros(
            (
                local_dimension**2,
                2 * maximum_pair_q + 1,
                len(labels),
            ),
            dtype=np.float64,
        )
        for multiplicity, (q0, a0, q1, a1) in enumerate(labels):
            for m0 in magnetic_numbers(q0):
                left = table[:, q0, m0 + spec.cell_size, a0]
                for m1 in magnetic_numbers(q1):
                    total_m = m0 + m1
                    if abs(total_m) > pair_q:
                        continue
                    coefficient = clebsch_gordan(
                        q0, m0, q1, m1, pair_q, total_m
                    )
                    if coefficient:
                        values[:, total_m + maximum_pair_q, multiplicity] += (
                            coefficient * np.kron(left, table[:, q1, m1 + spec.cell_size, a1])
                        )
        values.setflags(write=False)
        output.append((pair_q, values, labels))
    return tuple(output)


def _packed_row_indices(
    kind: CellKind,
    supports: tuple[tuple[int, ...], ...],
    target_m2: int = 0,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    spec = cell_schur_spec(kind)
    n_sites = 4 * spec.cell_size
    validate_m(n_sites, target_m2, name="target_m2")
    configurations = fixed_m_spin_basis(n_sites, target_m2)
    local_dimension = 1 << spec.cell_size
    packed = []
    for configuration in configurations:
        local_indices = []
        for support in supports:
            value = 0
            for site in support:
                value = (value << 1) | int(configuration[site])
            local_indices.append(value)
        index = 0
        for value in local_indices:
            index = index * local_dimension + value
        packed.append(index)
    return np.asarray(packed, dtype=np.int64), configurations


def four_cell_schur_state_vector(
    coefficients: Sequence[complex],
    kind: CellKind,
    *,
    supports: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Expand reduced singlet coefficients in the complete fixed-M=0 basis."""

    states = four_cell_schur_states(kind)
    vector = np.asarray(coefficients)
    if vector.shape != (len(states),):
        raise ValueError(f"expected {len(states)} four-cell coefficients")
    checked_supports = _validate_supports(kind, supports)
    spec = cell_schur_spec(kind)
    local_dimension = 1 << spec.cell_size
    maximum_pair_q = 2 * spec.cell_size
    packed_state = np.zeros(local_dimension**4, dtype=np.result_type(vector, np.complex128))
    pair_data = {q: (carrier, labels) for q, carrier, labels in _pair_carriers(kind)}
    for pair_q in sorted(pair_data):
        carrier, labels = pair_data[pair_q]
        lookup = {label: index for index, label in enumerate(labels)}
        block = np.zeros((len(labels), len(labels)), dtype=packed_state.dtype)
        for state_index, state in enumerate(states):
            if state.pair_q != pair_q:
                continue
            left = (
                state.cell_q[0], state.cell_a[0],
                state.cell_q[1], state.cell_a[1],
            )
            right = (
                state.cell_q[2], state.cell_a[2],
                state.cell_q[3], state.cell_a[3],
            )
            block[lookup[left], lookup[right]] = vector[state_index]
        pair_matrix = np.zeros(
            (local_dimension**2, local_dimension**2), dtype=packed_state.dtype
        )
        for m2 in magnetic_numbers(pair_q):
            coefficient = clebsch_gordan(pair_q, m2, pair_q, -m2, 0, 0)
            left = carrier[:, m2 + maximum_pair_q, :]
            right = carrier[:, -m2 + maximum_pair_q, :]
            pair_matrix += coefficient * (left @ block @ right.T)
        packed_state += pair_matrix.reshape(-1)
    row_indices, configurations = _packed_row_indices(kind, checked_supports)
    result = np.asarray(packed_state[row_indices])
    return result, configurations


def four_cell_schur_sector_state_vector(
    coefficients: Sequence[complex],
    kind: CellKind,
    target_q: int,
    *,
    target_m2: int | None = None,
    supports: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Expand four-cell coefficients in an arbitrary total-spin sector.

    ``target_q`` and ``target_m2`` are doubled labels. The default
    ``target_m2=target_q`` gives the highest-weight fixed-M reference.
    """

    target_q = int(target_q)
    chosen_m2 = target_q if target_m2 is None else int(target_m2)
    validate_m(target_q, chosen_m2, name="target_m2")
    states = four_cell_schur_sector_states(kind, target_q)
    vector = np.asarray(coefficients)
    if vector.shape != (len(states),):
        raise ValueError(f"expected {len(states)} four-cell sector coefficients")
    checked_supports = _validate_supports(kind, supports)
    spec = cell_schur_spec(kind)
    local_dimension = 1 << spec.cell_size
    maximum_pair_q = 2 * spec.cell_size
    packed_state = np.zeros(
        local_dimension**4,
        dtype=np.result_type(vector, np.complex128),
    )
    pair_data = {q: (carrier, labels) for q, carrier, labels in _pair_carriers(kind)}
    for left_q, (left_carrier, left_labels) in pair_data.items():
        left_lookup = {label: index for index, label in enumerate(left_labels)}
        for right_q, (right_carrier, right_labels) in pair_data.items():
            if target_q not in range(
                abs(left_q - right_q), left_q + right_q + 1, 2
            ):
                continue
            right_lookup = {label: index for index, label in enumerate(right_labels)}
            block = np.zeros(
                (len(left_labels), len(right_labels)), dtype=packed_state.dtype
            )
            for state_index, state in enumerate(states):
                if state.left_pair_q != left_q or state.right_pair_q != right_q:
                    continue
                left = (
                    state.cell_q[0],
                    state.cell_a[0],
                    state.cell_q[1],
                    state.cell_a[1],
                )
                right = (
                    state.cell_q[2],
                    state.cell_a[2],
                    state.cell_q[3],
                    state.cell_a[3],
                )
                block[left_lookup[left], right_lookup[right]] = vector[state_index]
            if not np.any(block):
                continue
            pair_matrix = np.zeros(
                (local_dimension**2, local_dimension**2),
                dtype=packed_state.dtype,
            )
            right_magnetic = set(magnetic_numbers(right_q))
            for left_m2 in magnetic_numbers(left_q):
                right_m2 = chosen_m2 - left_m2
                if right_m2 not in right_magnetic:
                    continue
                coefficient = clebsch_gordan(
                    left_q,
                    left_m2,
                    right_q,
                    right_m2,
                    target_q,
                    chosen_m2,
                )
                if coefficient:
                    left = left_carrier[:, left_m2 + maximum_pair_q, :]
                    right = right_carrier[:, right_m2 + maximum_pair_q, :]
                    pair_matrix += coefficient * (left @ block @ right.T)
            packed_state += pair_matrix.reshape(-1)
    row_indices, configurations = _packed_row_indices(
        kind, checked_supports, chosen_m2
    )
    return np.asarray(packed_state[row_indices]), configurations


def four_cell_schur_sector_basis_matrix(
    kind: CellKind,
    target_q: int,
    *,
    target_m2: int | None = None,
    supports: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return exact fixed-M columns for one four-cell total-spin sector."""

    states = four_cell_schur_sector_states(kind, int(target_q))
    columns = []
    configurations = None
    for selected in range(len(states)):
        coefficients = np.zeros(len(states), dtype=np.float64)
        coefficients[selected] = 1.0
        column, basis = four_cell_schur_sector_state_vector(
            coefficients,
            kind,
            int(target_q),
            target_m2=target_m2,
            supports=supports,
        )
        columns.append(column)
        if configurations is None:
            configurations = basis
    matrix = np.column_stack(columns)
    if np.max(np.abs(matrix.imag), initial=0.0) < 2.0e-14:
        matrix = np.asarray(matrix.real, dtype=np.float64)
    matrix.setflags(write=False)
    assert configurations is not None
    return matrix, configurations


def four_cell_schur_basis_matrix(
    kind: CellKind,
    *,
    supports: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return exact fixed-M columns for every four-cell Schur configuration."""

    dimension = len(four_cell_schur_states(kind))
    columns = []
    configurations = None
    for selected in range(dimension):
        coefficients = np.zeros(dimension, dtype=np.float64)
        coefficients[selected] = 1.0
        column, basis = four_cell_schur_state_vector(
            coefficients, kind, supports=supports
        )
        columns.append(column)
        if configurations is None:
            configurations = basis
    matrix = np.column_stack(columns)
    if np.max(np.abs(matrix.imag), initial=0.0) < 2.0e-14:
        matrix = np.asarray(matrix.real, dtype=np.float64)
    if matrix.shape[0] != comb(4 * cell_schur_spec(kind).cell_size, 2 * cell_schur_spec(kind).cell_size):
        raise RuntimeError("fixed-M basis dimension changed")
    matrix.setflags(write=False)
    assert configurations is not None
    return matrix, configurations


__all__ = [
    "four_cell_schur_basis_matrix",
    "four_cell_schur_sector_basis_matrix",
    "four_cell_schur_sector_state_vector",
    "four_cell_schur_state_vector",
]
