import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix
from su2_nqs.operators.heisenberg_chain import HeisenbergChain
from su2_nqs.operators.sz_heisenberg import SzHeisenbergChain
from su2_nqs.su2.fusion_path import enumerate_paths


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
def test_fusion_hamiltonian_equals_projected_computational_hamiltonian(n):
    paths = enumerate_paths(n, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    computational, _ = heisenberg_sz_matrix(n)
    expected = transform.conj().T @ computational @ transform
    actual = HeisenbergChain(n, 0).to_dense()
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    np.testing.assert_allclose(actual, actual.T, atol=1e-14)


def test_sz_connectivity_has_same_sector_spectrum_as_dense_reference():
    n = 6
    connected = SzHeisenbergChain(n, n // 2).to_dense()
    reference, _ = heisenberg_sz_matrix(n, n_up=n // 2)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(connected),
        np.linalg.eigvalsh(reference),
        atol=2e-14,
    )

