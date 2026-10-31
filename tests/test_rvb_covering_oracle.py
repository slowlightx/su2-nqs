from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.rvb_covering_oracle import (
    dimer_matching_orbits,
    enumerate_perfect_matchings,
    fixed_m_dimer_amplitudes,
    lowest_ritz_state,
    rvb_orbit_columns,
    transform_matching,
    transform_oriented_matching,
    two_lanczos_state,
)
from su2_nqs.benchmarks.triangular_locality import physical_dimer_product
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds


TRIANGULAR_N12 = ((6, 4), (0, 2))


def test_fixed_m_dimer_amplitudes_match_full_condon_shortley_vector() -> None:
    matching = ((0, 1), (2, 5), (3, 4))
    configurations = fixed_m_spin_basis(6, 0)
    restricted = fixed_m_dimer_amplitudes(configurations, matching)
    full = physical_dimer_product(6, matching)
    indices = [sum(bit << (5 - site) for site, bit in enumerate(row)) for row in configurations]
    np.testing.assert_allclose(restricted, full[indices], atol=0.0)
    np.testing.assert_allclose(np.linalg.norm(restricted), 1.0, atol=2.0e-15)


def test_registered_nn_covering_and_orbit_counts_are_stable() -> None:
    triangular_bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N12, j1=1.0, j2=0.0
    )
    triangular = enumerate_perfect_matchings(12, triangular_bonds)
    triangular_group = triangular_hnf_gamma_a1_group(TRIANGULAR_N12)
    triangular_orbits = dimer_matching_orbits(triangular, triangular_group)
    assert len(triangular) == 348
    assert len(triangular_group) == 144
    assert len(triangular_orbits) == 7
    assert sorted(orbit.orbit_size for orbit in triangular_orbits) == [
        12,
        36,
        36,
        48,
        72,
        72,
        72,
    ]

    square_bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.0, periodic=True)
    square = enumerate_perfect_matchings(16, square_bonds)
    square_group = square_gamma_a1_group(4)
    square_orbits = dimer_matching_orbits(square, square_group)
    assert len(square) == 272
    assert len(square_group) == 128
    assert len(square_orbits) == 13
    assert sum(orbit.orbit_size for orbit in square_orbits) == 272


def test_matching_transform_respects_active_site_permutation() -> None:
    matching = ((0, 1), (2, 3))
    image = (1, 2, 3, 0)
    assert transform_matching(matching, image) == ((0, 3), (1, 2))
    assert transform_oriented_matching(matching, image) == (
        ((0, 3), (1, 2)),
        -1,
    )


def test_triangular_n12_orbit_columns_are_exact_singlets_and_a1() -> None:
    configurations = fixed_m_spin_basis(12, 0)
    group = triangular_hnf_gamma_a1_group(TRIANGULAR_N12)
    matchings = enumerate_perfect_matchings(
        12,
        triangular_hnf_j1j2_bonds(TRIANGULAR_N12, j1=1.0, j2=0.0),
    )
    orbits = dimer_matching_orbits(matchings, group)
    columns = rvb_orbit_columns(configurations, orbits).columns
    indices = physical_permutation_index_table(configurations, group)
    for column in columns.T:
        assert spin_raising_relative_norm(column, configurations) < 2.0e-14
        projected = np.mean(column[indices], axis=0)
        np.testing.assert_allclose(projected, column, atol=3.0e-15)


def test_ritz_solver_removes_dependent_columns_and_two_lanczos_is_nested() -> None:
    hamiltonian = np.diag(np.asarray((-2.0, -1.0, 0.5, 1.0)))
    seed = np.asarray((1.0, 2.0, 3.0, 4.0))
    two = two_lanczos_state(seed, hamiltonian)
    augmented = lowest_ritz_state(
        np.column_stack((seed, hamiltonian @ seed, hamiltonian @ (hamiltonian @ seed), seed)),
        hamiltonian,
    )
    assert two.rank == 3
    assert augmented.rank == 3
    np.testing.assert_allclose(augmented.energy, two.energy, atol=2.0e-14)
    np.testing.assert_allclose(np.linalg.norm(two.state), 1.0, atol=2.0e-15)
