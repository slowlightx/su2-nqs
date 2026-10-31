"""Independent exact-small oracle for an XDiag total-:math:`S^+` bridge.

The source and target sectors use doubled projections ``m=0`` and ``m=2``.
This module contains no XDiag binding: it reconstructs normalized trivial-irrep
orbit bases in the full computational basis, applies the spin-raising operator
there, and projects back.  It is therefore an independent component-level
oracle for the external reduced-basis implementation.

Repository integers store site zero in the most-significant bit, with bit zero
meaning :math:`m_i=-1/2` and bit one meaning :math:`m_i=+1/2`.
"""

from __future__ import annotations

from math import comb, sqrt
from typing import Sequence

import numpy as np
from scipy import sparse

from su2_nqs.benchmarks.exact_diagonalization import computational_basis
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import (
    PhysicalSeedCase,
    configuration_from_repository_integer,
    configuration_from_xdiag_uint,
    ordered_gamma_a1_group,
    physical_seed_amplitudes,
    repository_integer_from_configuration,
    xdiag_uint_from_configuration,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds


def _active_transform(
    configuration: tuple[int, ...], image: Sequence[int]
) -> tuple[int, ...]:
    transformed = [0] * len(configuration)
    for old_site, new_site in enumerate(image):
        transformed[int(new_site)] = configuration[old_site]
    return tuple(transformed)


def explicit_a1_reduced_basis_sector(
    case: PhysicalSeedCase,
    representatives_xdiag: Sequence[int],
    n_up: int,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Construct normalized A1 orbit columns for one fixed-``n_up`` sector."""

    if isinstance(n_up, bool) or not isinstance(n_up, (int, np.integer)):
        raise ValueError("n_up must be an integer")
    n_up = int(n_up)
    if not 0 <= n_up <= case.n_sites:
        raise ValueError("n_up lies outside the spin-1/2 Hilbert space")
    basis = computational_basis(case.n_sites, n_up)
    positions = {state: index for index, state in enumerate(basis)}
    group = ordered_gamma_a1_group(case)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    orbit_sizes: list[int] = []
    stabilizers: list[int] = []
    observed_orbits: set[frozenset[int]] = set()
    for column, raw_xdiag in enumerate(representatives_xdiag):
        configuration = configuration_from_xdiag_uint(
            int(raw_xdiag), case.n_sites
        )
        if sum(configuration) != n_up:
            raise ValueError("emitted representative is in the wrong n_up sector")
        orbit_configurations = {
            _active_transform(configuration, image) for image in group
        }
        orbit = frozenset(
            repository_integer_from_configuration(state)
            for state in orbit_configurations
        )
        if orbit in observed_orbits:
            raise ValueError("emitted representatives repeat an A1 orbit")
        observed_orbits.add(orbit)
        packed_orbit = {
            xdiag_uint_from_configuration(state) for state in orbit_configurations
        }
        if int(raw_xdiag) != min(packed_orbit):
            raise ValueError("emitted representative is not minimum in XDiag order")
        orbit_size = len(orbit)
        if len(group) % orbit_size:
            raise RuntimeError("orbit size does not divide the group order")
        stabilizer = len(group) // orbit_size
        orbit_sizes.append(orbit_size)
        stabilizers.append(stabilizer)
        coefficient = 1.0 / sqrt(orbit_size)
        for state in orbit:
            row_indices.append(positions[state])
            column_indices.append(column)
            values.append(coefficient)
    if sum(orbit_sizes) != comb(case.n_sites, n_up):
        raise ValueError("emitted representatives do not cover the fixed-n_up basis")
    matrix = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(basis), len(representatives_xdiag)),
    ).tocsc()
    return matrix, np.asarray(orbit_sizes), np.asarray(stabilizers)


def physical_seed_full_columns(case: PhysicalSeedCase) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return all registered physical seeds in the full doubled-``m=0`` basis."""

    n_up = case.n_sites // 2
    basis = computational_basis(case.n_sites, n_up)
    configurations = np.asarray(
        [
            configuration_from_repository_integer(state, case.n_sites)
            for state in basis
        ],
        dtype=np.int8,
    )
    return physical_seed_amplitudes(configurations, case), basis


def computational_total_splus(
    source_columns: np.ndarray,
    source_basis: Sequence[int],
    n_sites: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply ``sum_i S_i^+`` from doubled ``m`` to doubled ``m+2``."""

    vectors = np.asarray(source_columns, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    if vectors.ndim != 2 or vectors.shape[0] != len(source_basis):
        raise ValueError("source columns and basis have incompatible shapes")
    if not source_basis:
        raise ValueError("source basis must not be empty")
    n_up_values = {int(state).bit_count() for state in source_basis}
    if len(n_up_values) != 1:
        raise ValueError("source basis does not have fixed n_up")
    n_up = n_up_values.pop()
    target_basis = computational_basis(n_sites, n_up + 1)
    positions = {state: index for index, state in enumerate(target_basis)}
    raised = np.zeros((len(target_basis), vectors.shape[1]), dtype=np.float64)
    for source_index, state in enumerate(source_basis):
        for site in range(n_sites):
            mask = 1 << (n_sites - 1 - site)
            if not int(state) & mask:
                raised[positions[int(state) | mask]] += vectors[source_index]
    return raised, target_basis


def fixed_sz_sparse_hamiltonian_sector(
    case: PhysicalSeedCase,
    coupling: float,
    n_up: int,
) -> tuple[sparse.csr_matrix, tuple[int, ...]]:
    """Build the independent J1-J2 Hamiltonian in one fixed-``n_up`` sector."""

    if not np.isfinite(coupling):
        raise ValueError("coupling must be finite")
    if isinstance(n_up, bool) or not isinstance(n_up, (int, np.integer)):
        raise ValueError("n_up must be an integer")
    n_up = int(n_up)
    if not 0 <= n_up <= case.n_sites:
        raise ValueError("n_up lies outside the spin-1/2 Hilbert space")
    basis = computational_basis(case.n_sites, n_up)
    positions = {state: index for index, state in enumerate(basis)}
    bonds = triangular_hnf_j1j2_bonds(
        case.supercell, j1=1.0, j2=float(coupling)
    )
    inverse = {old: new for new, old in enumerate(case.site_order)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(len(basis), dtype=np.float64)
    for column, state in enumerate(basis):
        for old_first, old_second, weight in bonds:
            first = inverse[int(old_first)]
            second = inverse[int(old_second)]
            first_mask = 1 << (case.n_sites - 1 - first)
            second_mask = 1 << (case.n_sites - 1 - second)
            first_bit = bool(state & first_mask)
            second_bit = bool(state & second_mask)
            diagonal[column] += weight * (
                0.25 if first_bit == second_bit else -0.25
            )
            if first_bit != second_bit:
                flipped = state ^ first_mask ^ second_mask
                rows.append(positions[flipped])
                columns.append(column)
                values.append(0.5 * weight)
    rows.extend(range(len(basis)))
    columns.extend(range(len(basis)))
    values.extend(float(value) for value in diagonal)
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    return matrix, basis


def deterministic_probe_columns(dimension: int, count: int = 3) -> np.ndarray:
    """Return the frozen real probe bank used by the independent C++ app."""

    if dimension <= 0 or count <= 0:
        raise ValueError("dimension and count must be positive")
    rows = np.arange(1, dimension + 1, dtype=np.float64)[:, None]
    columns = np.arange(count, dtype=np.float64)[None, :]
    result = (
        np.sin(rows * (columns + 2.0) * 0.17320508075688773)
        + np.cos((rows + 2.0) * (columns + 1.0) * 0.11785113019775793)
        + 0.01 * (np.mod(rows - 1.0 + columns, 7.0) - 3.0)
    )
    norms = np.linalg.norm(result, axis=0)
    if np.any(norms <= 1.0e-14):
        raise RuntimeError("frozen probe unexpectedly has a zero column")
    return result / norms[None, :]


def dicke_m_zero_full_column(n_sites: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return normalized ``|S=N/2,M=0>`` in the full computational basis."""

    if n_sites <= 0 or n_sites % 2:
        raise ValueError("Dicke M=0 requires a positive even number of sites")
    basis = computational_basis(n_sites, n_sites // 2)
    column = np.full((len(basis), 1), 1.0 / sqrt(len(basis)), dtype=np.float64)
    return column, basis

