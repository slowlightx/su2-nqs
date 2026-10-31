"""Independent geometry and exact-Hamiltonian checks for triangular J1-J2."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import (
    heisenberg_sz_matrix,
    spin_squared_matrix,
)
from su2_nqs.benchmarks.triangular_locality import (
    dimer_state_in_fusion_basis,
    nearest_neighbor_dimer_cover,
    physical_dimer_product,
)
from su2_nqs.lattices.triangular import triangular_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_path import enumerate_paths


J1_DISPLACEMENTS = ((1, 0), (0, 1), (1, 1))
J2_DISPLACEMENTS = ((1, -1), (1, 2), (2, 1))


def _triangular_squared_distance(displacement: tuple[int, int]) -> int:
    row, column = displacement
    return row * row + column * column - row * column


def _edge_weights(bonds):
    return {(first, second): coupling for first, second, coupling in bonds}


def _translate_site(site: int, rows: int, columns: int, dr: int, dc: int) -> int:
    row, column = divmod(site, columns)
    return ((row + dr) % rows) * columns + (column + dc) % columns


def test_displacements_are_first_and_second_physical_shells():
    assert {_triangular_squared_distance(delta) for delta in J1_DISPLACEMENTS} == {1}
    assert {_triangular_squared_distance(delta) for delta in J2_DISPLACEMENTS} == {3}


@pytest.mark.parametrize("rows,columns", [(2, 3), (3, 4), (4, 5)])
def test_open_shell_counts_have_closed_form(rows, columns):
    j1 = triangular_j1j2_bonds(rows, columns, j1=1.0, j2=0.0)
    j2 = triangular_j1j2_bonds(rows, columns, j1=0.0, j2=1.0)
    expected_j1 = (
        (rows - 1) * columns
        + rows * (columns - 1)
        + (rows - 1) * (columns - 1)
    )
    expected_j2 = (
        (rows - 1) * (columns - 1)
        + (rows - 1) * max(columns - 2, 0)
        + max(rows - 2, 0) * (columns - 1)
    )
    assert len(j1) == expected_j1
    assert len(j2) == expected_j2
    assert set(_edge_weights(j1)).isdisjoint(_edge_weights(j2))


@pytest.mark.parametrize("linear_size", [4, 5, 6])
def test_periodic_shells_have_coordination_six_and_translation_symmetry(linear_size):
    n = linear_size**2
    for use_j2 in (False, True):
        bonds = triangular_j1j2_bonds(
            linear_size,
            linear_size,
            j1=0.0 if use_j2 else 1.0,
            j2=1.0 if use_j2 else 0.0,
            periodic=True,
        )
        weights = _edge_weights(bonds)
        coordination = defaultdict(float)
        for (first, second), coupling in weights.items():
            coordination[first] += coupling
            coordination[second] += coupling
        assert len(bonds) == 3 * n
        assert set(coordination) == set(range(n))
        np.testing.assert_allclose(tuple(coordination.values()), 6.0, atol=0.0)

        for shift in ((1, 0), (0, 1), (1, 1)):
            translated = {}
            for (first, second), coupling in weights.items():
                mapped = tuple(
                    sorted(
                        (
                            _translate_site(first, linear_size, linear_size, *shift),
                            _translate_site(second, linear_size, linear_size, *shift),
                        )
                    )
                )
                translated[mapped] = coupling
            assert translated == weights


@pytest.mark.parametrize("periodic", [False, True])
def test_j1_shell_matches_existing_triangular_lattice_generator(periodic):
    weighted = triangular_j1j2_bonds(4, 4, j1=1.0, j2=0.0, periodic=periodic)
    assert tuple((first, second) for first, second, _ in weighted) == triangular_bonds(
        4, 4, periodic=periodic
    )


@pytest.mark.parametrize("rows,columns", [(2, 2), (2, 3), (2, 4)])
def test_triangular_fusion_hamiltonian_matches_computational_projection(rows, columns):
    n = rows * columns
    bonds = triangular_j1j2_bonds(rows, columns, j1=0.73, j2=0.19)
    paths = enumerate_paths(n, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    computational, _ = heisenberg_sz_matrix(n, bonds)
    fusion = heisenberg_graph_fusion_matrix(n, bonds, target_q=0).toarray()
    expected = transform.conj().T @ computational @ transform
    np.testing.assert_allclose(fusion, expected, atol=8e-14)
    np.testing.assert_allclose(fusion, fusion.T, atol=2e-14)


def test_triangular_hamiltonian_is_exactly_su2_scalar():
    n = 6
    bonds = triangular_j1j2_bonds(2, 3, j1=1.0, j2=0.27)
    hamiltonian, basis = heisenberg_sz_matrix(n, bonds)
    spin_squared, spin_basis = spin_squared_matrix(n)
    assert basis == spin_basis
    commutator = hamiltonian @ spin_squared - spin_squared @ hamiltonian
    np.testing.assert_allclose(commutator, 0.0, atol=3e-14)


def test_physical_dimer_pilot_state_is_an_exact_total_singlet():
    n = 6
    j1_bonds = triangular_j1j2_bonds(2, 3, j1=1.0, j2=0.0)
    matching = nearest_neighbor_dimer_cover(n, j1_bonds)
    assert len(matching) == n // 2
    assert all(
        (first, second, 1.0) in j1_bonds for first, second in matching
    )
    physical = physical_dimer_product(n, matching)
    spin_squared, _ = spin_squared_matrix(n)
    np.testing.assert_allclose(np.linalg.norm(physical), 1.0, atol=2e-15)
    np.testing.assert_allclose(spin_squared @ physical, 0.0, atol=3e-14)
    fusion, residual = dimer_state_in_fusion_basis(n, matching)
    np.testing.assert_allclose(np.linalg.norm(fusion), 1.0, atol=2e-15)
    assert residual < 2e-15
