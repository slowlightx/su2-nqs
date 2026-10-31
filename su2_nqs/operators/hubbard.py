"""Small-system Hubbard Hamiltonians in charge/spin fusion sectors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.hubbard_fusion import (
    HubbardFusionState,
    decode_hubbard_fusion_state,
    encode_hubbard_fusion_state,
    enumerate_hubbard_fusion_states,
    local_spin_q,
    occupations_from_fock_mask,
)
from ..su2.hubbard_spinor import directed_hopping_reduced_element


FockBond: TypeAlias = tuple[int, int] | tuple[int, int, float]


def _checked_bonds(
    n_sites: int, bonds: Sequence[FockBond]
) -> tuple[tuple[int, int, float], ...]:
    checked = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            hopping = 1.0
        elif len(bond) == 3:
            first, second, hopping = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        first = int(first)
        second = int(second)
        if first == second or not (0 <= first < n_sites and 0 <= second < n_sites):
            raise ValueError(f"invalid Hubbard bond: {bond!r}")
        checked.append((first, second, float(hopping)))
    return tuple(checked)


def _direct_coupled_coefficients(state: HubbardFusionState) -> dict[int, complex]:
    """Independent highest-weight CG expansion used by the direct operator."""

    amplitudes: dict[tuple[int, int], complex] = {(0, 0): 1.0}
    running_q = 0
    for site, (occupation, next_q) in enumerate(zip(state.occupations, state.path)):
        if occupation == 0:
            local_states = ((0, 0),)
        elif occupation == 1:
            local_states = ((1, 1 << (2 * site)), (-1, 1 << (2 * site + 1)))
        else:
            local_states = ((0, (1 << (2 * site)) | (1 << (2 * site + 1))),)
        updated: dict[tuple[int, int], complex] = {}
        for (running_m, mask), amplitude in amplitudes.items():
            for local_m, local_mask in local_states:
                coefficient = clebsch_gordan(
                    running_q,
                    running_m,
                    local_spin_q(occupation),
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
        if magnetization == state.target_q and abs(amplitude) > 1.0e-15
    }


def _apply_direct_hop(mask: int, destination: int, source: int) -> tuple[int, int] | None:
    """Independent canonical-Fock sign evaluation for one directed hop."""

    if not (mask & (1 << source)) or mask & (1 << destination):
        return None
    lower_source = (1 << source) - 1
    sign = -1 if (mask & lower_source).bit_count() % 2 else 1
    removed = mask & ~(1 << source)
    lower_destination = (1 << destination) - 1
    if (removed & lower_destination).bit_count() % 2:
        sign = -sign
    return removed | (1 << destination), sign


def hubbard_fusion_matrix(
    n_sites: int,
    n_electrons: int,
    target_q: int,
    bonds: Sequence[FockBond],
    *,
    interaction: float,
) -> tuple[np.ndarray, tuple[HubbardFusionState, ...]]:
    """Construct a fusion-sector Hamiltonian by direct finite CG contraction.

    This routine never constructs the full Fock Hamiltonian or multiplies
    ``C.conj().T @ H_fock @ C``.  For each local hop it contracts only the old
    and new coupled-state coefficients compatible with the changed occupation
    pattern.  It is an exact small-system implementation and the reference
    gate for a later reduced ``q_op=1`` spinor recurrence.
    """

    states = enumerate_hubbard_fusion_states(n_sites, n_electrons, target_q)
    positions_by_occupations: dict[tuple[int, ...], list[int]] = defaultdict(list)
    expansions: list[dict[int, complex]] = []
    for index, state in enumerate(states):
        positions_by_occupations[state.occupations].append(index)
        expansions.append(_direct_coupled_coefficients(state))
    matrix = np.zeros((len(states), len(states)), dtype=np.complex128)
    checked_bonds = _checked_bonds(n_sites, bonds)

    for column, (state, expansion) in enumerate(zip(states, expansions)):
        matrix[column, column] += float(interaction) * state.occupations.count(2)
        for old_mask, old_amplitude in expansion.items():
            for first, second, hopping in checked_bonds:
                for spin in (0, 1):
                    first_orbital = 2 * first + spin
                    second_orbital = 2 * second + spin
                    for destination, source in (
                        (first_orbital, second_orbital),
                        (second_orbital, first_orbital),
                    ):
                        result = _apply_direct_hop(old_mask, destination, source)
                        if result is None:
                            continue
                        new_mask, sign = result
                        new_occupations = occupations_from_fock_mask(new_mask, n_sites)
                        for row in positions_by_occupations.get(new_occupations, ()):
                            new_amplitude = expansions[row].get(new_mask, 0.0)
                            if new_amplitude:
                                matrix[row, column] += (
                                    -hopping
                                    * sign
                                    * np.conj(new_amplitude)
                                    * old_amplitude
                                )
    matrix = np.real_if_close(matrix, tol=1000)
    return np.asarray(matrix), states


def hubbard_reduced_fusion_matrix(
    n_sites: int,
    n_electrons: int,
    target_q: int,
    bonds: Sequence[FockBond],
    *,
    interaction: float,
) -> tuple[np.ndarray, tuple[HubbardFusionState, ...]]:
    """Compile the Hubbard Hamiltonian with reduced spinor transfers.

    Each directed hopping term is the scalar closure of two local
    ``q_op=1`` fermion spinors.  Canonical interleaved-orbital parity is
    carried explicitly by :func:`directed_hopping_reduced_element`.  Unlike
    :func:`hubbard_fusion_matrix`, this compiler never expands a fusion state
    into magnetic Fock configurations.
    """

    sparse, states = hubbard_reduced_fusion_sparse(
        n_sites,
        n_electrons,
        target_q,
        bonds,
        interaction=interaction,
    )
    matrix = np.real_if_close(sparse.toarray(), tol=1000)
    return np.asarray(matrix), states


def hubbard_reduced_fusion_sparse(
    n_sites: int,
    n_electrons: int,
    target_q: int,
    bonds: Sequence[FockBond],
    *,
    interaction: float,
):
    """Compile a CSR Hubbard matrix from reduced spinor connectivity."""

    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("sparse reduced Hubbard compilation requires scipy") from error
    states = enumerate_hubbard_fusion_states(n_sites, n_electrons, target_q)
    positions_by_occupations: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        positions_by_occupations[state.occupations].append(index)
    checked_bonds = _checked_bonds(n_sites, bonds)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[complex] = []
    for column, old_state in enumerate(states):
        diagonal = float(interaction) * old_state.occupations.count(2)
        if diagonal:
            row_indices.append(column)
            column_indices.append(column)
            values.append(diagonal)
        for first, second, hopping in checked_bonds:
            for destination, source in ((first, second), (second, first)):
                new_occupations = list(old_state.occupations)
                new_occupations[destination] += 1
                new_occupations[source] -= 1
                if not (
                    0 <= new_occupations[destination] <= 2
                    and 0 <= new_occupations[source] <= 2
                ):
                    continue
                for row in positions_by_occupations.get(
                    tuple(new_occupations), ()
                ):
                    element = directed_hopping_reduced_element(
                        states[row], old_state, destination, source
                    )
                    if element:
                        row_indices.append(row)
                        column_indices.append(column)
                        values.append(-hopping * element)
    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.complex128),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(states), len(states)),
        dtype=np.complex128,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix, states


def hubbard_fusion_connectivity(
    matrix: np.ndarray,
    states: Sequence[HubbardFusionState],
    state: HubbardFusionState,
    *,
    tolerance: float = 1.0e-14,
) -> tuple[tuple[HubbardFusionState, ...], np.ndarray]:
    """Return ``H[new,old]`` connectivity from a validated dense reference."""

    states = tuple(states)
    if matrix.shape != (len(states), len(states)):
        raise ValueError("matrix and state basis dimensions disagree")
    try:
        column = states.index(state)
    except ValueError as error:
        raise ValueError("state does not belong to this fusion sector") from error
    rows = np.flatnonzero(np.abs(np.asarray(matrix)[:, column]) > tolerance)
    return tuple(states[int(row)] for row in rows), np.asarray(matrix)[rows, column]


@dataclass(frozen=True)
class HubbardHamiltonian:
    """Small exact Hubbard operator compatible with generic VMC connectivity."""

    n_sites: int
    n_electrons: int
    target_q: int
    bonds: tuple[FockBond, ...]
    interaction: float
    compiler: str = "direct_cg"

    def __post_init__(self) -> None:
        if self.compiler not in ("direct_cg", "reduced_spinor"):
            raise ValueError("compiler must be direct_cg or reduced_spinor")

    @cached_property
    def compiled(self) -> tuple[np.ndarray, tuple[HubbardFusionState, ...]]:
        builder = (
            hubbard_fusion_matrix
            if self.compiler == "direct_cg"
            else hubbard_reduced_fusion_matrix
        )
        return builder(
            self.n_sites,
            self.n_electrons,
            self.target_q,
            self.bonds,
            interaction=self.interaction,
        )

    @property
    def basis(self) -> tuple[tuple[int, ...], ...]:
        return tuple(encode_hubbard_fusion_state(state) for state in self.compiled[1])

    def to_dense(self) -> np.ndarray:
        return self.compiled[0]

    def get_conn(
        self, labels: Sequence[int]
    ) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
        state = decode_hubbard_fusion_state(labels)
        connected, elements = hubbard_fusion_connectivity(
            self.compiled[0], self.compiled[1], state
        )
        return tuple(encode_hubbard_fusion_state(value) for value in connected), elements


__all__ = [
    "HubbardHamiltonian",
    "hubbard_fusion_connectivity",
    "hubbard_fusion_matrix",
    "hubbard_reduced_fusion_matrix",
    "hubbard_reduced_fusion_sparse",
]
