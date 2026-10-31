from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.spectral_physics import (
    compile_sector_spin_vector,
    low_energy_spectral_weights,
    zero_temperature_response,
)
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.reduced_vector import compile_weighted_spin_vector


@pytest.mark.parametrize("n", (4, 6, 8))
def test_sector_vector_matches_all_irrep_reference(n: int) -> None:
    weights = [(-1.0) ** index * (index + 1) / n for index in range(n)]
    sector = compile_sector_spin_vector(weights, q_new=2, q_old=0)
    reference = compile_weighted_spin_vector(weights)
    new_paths, old_paths, matrix = reference.dense_block(2, 0)
    assert sector.new_paths == new_paths
    assert sector.old_paths == old_paths
    np.testing.assert_allclose(sector.matrix.toarray(), matrix, atol=2.0e-13)


def test_singlet_to_triplet_wigner_eckart_normalization() -> None:
    n = 6
    weights = np.asarray([1.0, -1.0, 0.3, -0.3, 0.7, -0.7])
    sector = compile_sector_spin_vector(weights, q_new=2, q_old=0)
    singlets = enumerate_paths(n, 0)
    triplets = enumerate_paths(n, 2)
    singlet_basis = coupled_basis_matrix(singlets, 0, 0)
    triplet_basis = coupled_basis_matrix(triplets, 2, 0)
    configurations = np.asarray(
        [
            [1 if (bits >> (n - site - 1)) & 1 else -1 for site in range(n)]
            for bits in range(1 << n)
        ],
        dtype=np.float64,
    )
    magnetic_z = 0.5 * configurations @ weights
    explicit = triplet_basis.conj().T @ (magnetic_z[:, None] * singlet_basis)
    np.testing.assert_allclose(
        sector.matrix.toarray() / math.sqrt(3.0), explicit, atol=2.0e-13
    )


def test_projected_response_matches_dense_spectrum() -> None:
    hamiltonian = sparse.csr_matrix(
        np.asarray(
            [[-2.0, 0.0, 0.0], [0.0, -0.5, 0.2], [0.0, 0.2, 1.0]]
        )
    )
    values, vectors = np.linalg.eigh(hamiltonian.toarray())
    ground = vectors[:, 0]
    raw_source = np.asarray([0.4, -0.7, 0.2])
    source = raw_source - ground * np.vdot(ground, raw_source)
    coefficients = vectors[:, 1:].conj().T @ source
    gaps = values[1:] - values[0]
    response = zero_temperature_response(
        hamiltonian, values[0], raw_source, null_vector=ground
    )
    expected_static = 2.0 * np.sum(np.abs(coefficients) ** 2 / gaps)
    expected_fidelity = np.sum(np.abs(coefficients) ** 2 / gaps**2)
    np.testing.assert_allclose(response.static_response, expected_static, atol=2.0e-12)
    np.testing.assert_allclose(
        response.fidelity_response, expected_fidelity, atol=2.0e-12
    )
    assert response.relative_linear_residual < 1.0e-11

    rows = low_energy_spectral_weights(
        values, vectors, source, ground_energy=values[0]
    )
    np.testing.assert_allclose(
        sum(row["static_response_weight"] for row in rows),
        expected_static,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint_weights": [], "q_new": 2, "q_old": 0}, "at least one"),
        ({"endpoint_weights": [1.0, np.nan], "q_new": 2, "q_old": 0}, "finite"),
        ({"endpoint_weights": [1.0, -1.0], "q_new": 0, "q_old": 0}, "q_op=2"),
    ],
)
def test_sector_vector_rejects_invalid_inputs(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compile_sector_spin_vector(**kwargs)
