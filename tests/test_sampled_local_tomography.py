from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.local_indistinguishability import (
    pure_state_partial_trace,
    pure_state_transition_partial_trace,
)
from su2_nqs.benchmarks.sampled_local_tomography import (
    TOMOGRAPHY_VECTOR_NAMES,
    local_diagonal_rdm_from_moments,
    two_root_three_spin_tomography,
    two_root_tomography_vector,
    two_root_two_spin_tomography,
)
from su2_nqs.operators.permutation import transposition_matrix
from su2_nqs.su2.fusion_path import enumerate_paths


def _four_spin_moment_matrices() -> tuple[np.ndarray, np.ndarray]:
    n_sites = 4
    roots = np.eye(2, dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    swaps = {
        pair: transposition_matrix(n_sites, 0, *pair).toarray()
        for pair in ((0, 1), (1, 2), (0, 2))
    }
    pairs = tuple(0.5 * swaps[pair] - 0.25 * identity for pair in swaps)
    chirality = 0.25j * (
        swaps[(0, 1)] @ swaps[(1, 2)] - swaps[(1, 2)] @ swaps[(0, 1)]
    )
    moment_matrices = np.stack((identity, *pairs, chirality), axis=0)
    computational = coupled_basis_matrix(enumerate_paths(n_sites, 0), 0, 0) @ roots
    return moment_matrices, computational


def test_sampled_three_spin_tomography_matches_direct_local_blocks() -> None:
    moments, computational = _four_spin_moment_matrices()
    result = two_root_three_spin_tomography(moments)
    direct00 = pure_state_partial_trace(computational[:, 0], 4, (0, 1, 2))
    direct11 = pure_state_partial_trace(computational[:, 1], 4, (0, 1, 2))
    direct01 = pure_state_transition_partial_trace(
        computational[:, 0], computational[:, 1], 4, (0, 1, 2)
    )
    np.testing.assert_allclose(result.rho00, direct00, atol=4.0e-14)
    np.testing.assert_allclose(result.rho11, direct11, atol=4.0e-14)
    np.testing.assert_allclose(result.rho01, direct01, atol=4.0e-14)
    np.testing.assert_allclose(
        local_diagonal_rdm_from_moments(moments, root=0, region_size=3),
        direct00,
        atol=4.0e-14,
    )
    assert result.overlap_null_absolute < 3.0e-14
    assert result.maximum_operator_antihermiticity < 3.0e-14
    assert result.choi_minimum_eigenvalue > -4.0e-14

    vector = two_root_tomography_vector(moments, region_size=3)
    assert vector.shape == (len(TOMOGRAPHY_VECTOR_NAMES),)
    np.testing.assert_allclose(
        vector[3:6], result.qddm_singular_values, atol=2.0e-15
    )


def test_sampled_two_spin_tomography_matches_direct_local_blocks() -> None:
    moments, computational = _four_spin_moment_matrices()
    result = two_root_two_spin_tomography(moments[:2])
    direct00 = pure_state_partial_trace(computational[:, 0], 4, (0, 1))
    direct11 = pure_state_partial_trace(computational[:, 1], 4, (0, 1))
    direct01 = pure_state_transition_partial_trace(
        computational[:, 0], computational[:, 1], 4, (0, 1)
    )
    np.testing.assert_allclose(result.rho00, direct00, atol=4.0e-14)
    np.testing.assert_allclose(result.rho11, direct11, atol=4.0e-14)
    np.testing.assert_allclose(result.rho01, direct01, atol=4.0e-14)
    np.testing.assert_allclose(
        local_diagonal_rdm_from_moments(
            moments[:2], root=1, region_size=2, pair_indices=(1,)
        ),
        direct11,
        atol=4.0e-14,
    )

