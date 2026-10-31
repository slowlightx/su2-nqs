"""Independent fixed-Sz references for spin-sector purity."""

import numpy as np

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.spin_sector import (
    project_sz_to_singlet_multiplicity,
    singlet_projector_action,
    spin_purity_metrics,
    spin_squared_sz_sparse,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def test_sparse_spin_squared_matches_independent_dense_reference() -> None:
    sparse_matrix, sparse_basis = spin_squared_sz_sparse(6, n_up=3)
    dense_matrix, dense_basis = spin_squared_matrix(6, n_up=3)
    dense_positions = {state: index for index, state in enumerate(dense_basis)}
    sparse_as_integers = [
        sum(bit << (5 - site) for site, bit in enumerate(bits))
        for bits in sparse_basis
    ]
    permutation = [dense_positions[state] for state in sparse_as_integers]
    dense_in_sparse_order = dense_matrix[np.ix_(permutation, permutation)]
    np.testing.assert_allclose(
        sparse_matrix.toarray(), dense_in_sparse_order, atol=1.0e-14
    )


def test_singlet_polynomial_matches_coupled_basis_projector() -> None:
    n = 6
    spin_squared, basis = spin_squared_sz_sparse(n, n_up=n // 2)
    paths = enumerate_paths(n, 0)
    coupled = coupled_basis_matrix(paths, 0, 0)
    full_indices = [
        sum(bit << (n - site - 1) for site, bit in enumerate(bits))
        for bits in basis
    ]
    singlet_columns = coupled[np.asarray(full_indices), :]
    rng = np.random.default_rng(123)
    state = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    state /= np.linalg.norm(state)
    expected = singlet_columns @ (singlet_columns.conj().T @ state)
    actual = singlet_projector_action(
        state, spin_squared, maximum_spin=n // 2
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)
    metrics = spin_purity_metrics(
        state, spin_squared, maximum_spin=n // 2
    )
    np.testing.assert_allclose(
        metrics.singlet_weight,
        float(np.vdot(expected, expected).real),
        atol=2.0e-13,
    )
    assert metrics.projected_singlet_residual < 2.0e-13

    projection = project_sz_to_singlet_multiplicity(state, singlet_columns)
    np.testing.assert_allclose(
        projection.singlet_weight, metrics.singlet_weight, atol=2.0e-13
    )
    reconstructed = singlet_columns @ projection.state
    np.testing.assert_allclose(
        reconstructed,
        expected / np.sqrt(metrics.singlet_weight),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(np.vdot(projection.state, projection.state), 1.0)
    assert projection.isometry_error < 2.0e-13
