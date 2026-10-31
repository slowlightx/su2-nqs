import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix
from su2_nqs.benchmarks.singlet_diagonal_estimator import (
    heisenberg_z_diagonal,
    singlet_diagonal_estimator_benchmark,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


CASES = (
    (
        4,
        square_j1j2_bonds(2, 2, j1=1.0, j2=0.5, periodic=False),
        2,
    ),
    (
        6,
        triangular_j1j2_bonds(2, 3, j1=1.0, j2=0.125, periodic=False),
        5,
    ),
    (
        8,
        square_j1j2_bonds(2, 4, j1=1.0, j2=0.5, periodic=False),
        14,
    ),
)


@pytest.mark.parametrize(("n_sites", "bonds", "singlet_dimension"), CASES)
def test_projected_diagonal_operator_matches_full_heisenberg_and_force(
    n_sites: int,
    bonds: tuple[tuple[int, int, float], ...],
    singlet_dimension: int,
) -> None:
    report = singlet_diagonal_estimator_benchmark(n_sites, bonds)
    assert report.singlet_dimension == singlet_dimension
    assert report.singlet_isometry_error < 2.0e-14
    assert report.projected_operator_maximum_error < 2.0e-14
    assert report.random_singlet_energy_difference < 2.0e-14
    assert report.random_singlet_force_difference < 3.0e-14


@pytest.mark.parametrize(("n_sites", "bonds", "_"), CASES)
def test_three_hz_loses_full_hamiltonian_zero_variance(
    n_sites: int,
    bonds: tuple[tuple[int, int, float], ...],
    _: int,
) -> None:
    report = singlet_diagonal_estimator_benchmark(n_sites, bonds)
    assert report.full_ground_local_energy_variance < 2.0e-27
    assert report.diagonal_ground_local_energy_variance > 1.0e-3
    assert report.diagonal_variance_outside_singlet_error < 2.0e-13


@pytest.mark.parametrize(("n_sites", "bonds", "_"), CASES)
def test_physical_connectivity_hz_diagonal_matches_dense_heisenberg(
    n_sites: int,
    bonds: tuple[tuple[int, int, float], ...],
    _: int,
) -> None:
    hamiltonian, basis = heisenberg_sz_matrix(
        n_sites, bonds, n_up=n_sites // 2
    )
    actual = heisenberg_z_diagonal(n_sites, bonds, basis=basis)
    np.testing.assert_allclose(actual, np.diag(hamiltonian), atol=0.0, rtol=0.0)


def test_exact_small_oracle_rejects_unregistered_size() -> None:
    with pytest.raises(ValueError, match="N=4, N=6, or N=8"):
        singlet_diagonal_estimator_benchmark(10, ((0, 1, 1.0),))
