import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.fusion_path import enumerate_paths


def test_spin_half_singlet_condon_shortley_signs():
    root_two = np.sqrt(2.0)
    assert clebsch_gordan(1, 1, 1, -1, 0, 0) == pytest.approx(1.0 / root_two)
    assert clebsch_gordan(1, -1, 1, 1, 0, 0) == pytest.approx(-1.0 / root_two)


@pytest.mark.parametrize(("n", "target_q"), [(2, 0), (4, 0), (6, 0), (6, 2)])
def test_coupled_basis_is_orthonormal_and_has_exact_total_spin(n, target_q):
    paths = enumerate_paths(n, target_q)
    basis = coupled_basis_matrix(paths, target_q, target_q)
    np.testing.assert_allclose(
        basis.conj().T @ basis,
        np.eye(len(paths)),
        atol=2e-14,
    )
    spin_squared, _ = spin_squared_matrix(n)
    spin = target_q / 2
    np.testing.assert_allclose(
        spin_squared @ basis,
        spin * (spin + 1) * basis,
        atol=2e-14,
    )

