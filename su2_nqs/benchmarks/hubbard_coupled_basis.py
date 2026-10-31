"""Computational-basis expansion of charge-resolved Hubbard fusion states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.hubbard_fusion import (
    HubbardFusionState,
    is_valid_hubbard_fusion_state,
    local_spin_q,
)
from ..su2.quantum_numbers import validate_m
from ..operators.hubbard_fock import tokens_to_fock_mask
from .hubbard_fock import enumerate_fock_basis


def hubbard_coupled_state(
    state: HubbardFusionState,
    *,
    total_m: int | None = None,
) -> dict[int, complex]:
    """Expand one fusion state into interleaved-orbital Fock bit masks."""

    if not is_valid_hubbard_fusion_state(state):
        raise ValueError(f"invalid Hubbard fusion state: {state!r}")
    total_q = state.target_q
    if total_m is None:
        total_m = total_q
    validate_m(total_q, total_m, name="total_m")

    # (running doubled m, Fock mask) -> amplitude.
    amplitudes: dict[tuple[int, int], complex] = {(0, 0): 1.0}
    running_q = 0
    for site, (occupation, next_q) in enumerate(zip(state.occupations, state.path)):
        local_q = local_spin_q(occupation)
        if occupation == 0:
            local_states = ((0, 0),)
        elif occupation == 1:
            # Local Condon--Shortley basis: up is m=+1, down is m=-1.
            local_states = ((1, 1 << (2 * site)), (-1, 1 << (2 * site + 1)))
        else:
            local_states = ((0, (1 << (2 * site)) | (1 << (2 * site + 1))),)
        updated: dict[tuple[int, int], complex] = {}
        for (running_m, mask), amplitude in amplitudes.items():
            for local_m, local_mask in local_states:
                coefficient = clebsch_gordan(
                    running_q,
                    running_m,
                    local_q,
                    local_m,
                    next_q,
                    running_m + local_m,
                )
                if coefficient:
                    key = (running_m + local_m, mask | local_mask)
                    updated[key] = updated.get(key, 0.0) + amplitude * coefficient
        amplitudes = updated
        running_q = next_q
    return {
        mask: amplitude
        for (magnetization, mask), amplitude in amplitudes.items()
        if magnetization == total_m and abs(amplitude) > 1.0e-15
    }


def hubbard_coupled_basis_matrix(
    states: Sequence[HubbardFusionState],
    *,
    total_m: int | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Stack fixed-``(N_e,S,M)`` coupled states as Fock-basis columns."""

    states = tuple(states)
    if not states:
        return np.empty((0, 0), dtype=np.complex128), ()
    first = states[0]
    if not is_valid_hubbard_fusion_state(first):
        raise ValueError("states contain an invalid Hubbard fusion state")
    n_sites = first.n_sites
    n_electrons = first.n_electrons
    target_q = first.target_q
    if any(
        not is_valid_hubbard_fusion_state(
            state, n_electrons=n_electrons, target_q=target_q
        )
        or state.n_sites != n_sites
        for state in states
    ):
        raise ValueError("states must belong to one common charge/spin sector")
    if total_m is None:
        total_m = target_q
    validate_m(target_q, total_m, name="total_m")
    basis = enumerate_fock_basis(n_sites, n_electrons)
    positions = {mask: row for row, mask in enumerate(basis)}
    matrix = np.zeros((len(basis), len(states)), dtype=np.complex128)
    for column, state in enumerate(states):
        for mask, amplitude in hubbard_coupled_state(state, total_m=total_m).items():
            matrix[positions[mask], column] = amplitude
    return matrix, basis


def hubbard_coupled_basis_coo(
    states: Sequence[HubbardFusionState],
    physical_basis: Sequence[Sequence[int]],
    *,
    total_m: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile the fixed-``M`` fusion-to-physical transform as COO arrays.

    The returned entries obey ``physical[row] += value * fusion[column]``.
    This avoids the full fixed-charge Fock matrix when only one fixed-``M``
    slice is needed for exact projected training.
    """

    states = tuple(states)
    physical_basis = tuple(tuple(int(token) for token in row) for row in physical_basis)
    if not states or not physical_basis:
        raise ValueError("states and physical_basis must be nonempty")
    first = states[0]
    if not is_valid_hubbard_fusion_state(first):
        raise ValueError("states contain an invalid Hubbard fusion state")
    n_sites = first.n_sites
    n_electrons = first.n_electrons
    target_q = first.target_q
    if any(
        state.n_sites != n_sites
        or not is_valid_hubbard_fusion_state(
            state, n_electrons=n_electrons, target_q=target_q
        )
        for state in states
    ):
        raise ValueError("states must belong to one common charge/spin sector")
    if any(len(row) != n_sites for row in physical_basis):
        raise ValueError("physical basis states have the wrong site count")
    if total_m is None:
        total_m = target_q
    validate_m(target_q, total_m, name="total_m")
    positions = {
        tokens_to_fock_mask(row): index for index, row in enumerate(physical_basis)
    }
    if len(positions) != len(physical_basis):
        raise ValueError("physical_basis must not contain duplicates")
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for column, state in enumerate(states):
        for mask, amplitude in hubbard_coupled_state(
            state, total_m=total_m
        ).items():
            try:
                row = positions[mask]
            except KeyError as error:
                raise ValueError(
                    "physical_basis does not span the requested fixed-M sector"
                ) from error
            rows.append(row)
            columns.append(column)
            values.append(complex(amplitude))
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.complex128),
    )


__all__ = [
    "hubbard_coupled_basis_coo",
    "hubbard_coupled_basis_matrix",
    "hubbard_coupled_state",
]
