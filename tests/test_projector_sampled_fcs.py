import numpy as np
import pytest

from su2_nqs.benchmarks.block_spin_fcs import (
    block_spin_fcs,
    grouped_fusion_tree,
    subsystem_spin_eigenbasis,
    tree_block_spin_fcs_with_tangent,
)
from su2_nqs.benchmarks.projector_sampled_fcs import (
    allowed_block_spins,
    block_spin_projector_action,
    fcs_outcomes,
    joint_projector_connections,
    project_distribution_to_simplex,
    projector_local_values,
    sample_tree_native_fcs,
    weighted_projector_fcs_statistics,
)


def _projector_matrix_from_connectivity(n_spins, target_q):
    matrix = np.zeros((2**n_spins, 2**n_spins), dtype=np.float64)
    for column in range(2**n_spins):
        for row, coefficient in block_spin_projector_action(
            n_spins, target_q, column
        ):
            matrix[row, column] = coefficient
    return matrix


def test_sparse_swap_polynomial_matches_dense_spin_projectors_through_six_sites():
    for n_spins in range(1, 7):
        labels, eigenvectors = subsystem_spin_eigenbasis(n_spins)
        projectors = []
        for target_q in allowed_block_spins(n_spins):
            selected = eigenvectors[:, labels == target_q]
            expected = selected @ selected.conj().T
            actual = _projector_matrix_from_connectivity(n_spins, target_q)
            np.testing.assert_allclose(actual, expected, atol=2.0e-12)
            np.testing.assert_allclose(actual @ actual, actual, atol=2.0e-12)
            projectors.append(actual)
        np.testing.assert_allclose(sum(projectors), np.eye(2**n_spins), atol=2.0e-12)


def test_joint_projector_connections_are_complete_on_noncontiguous_groups():
    bits = (1, 0, 1, 0, 1, 0)
    groups = ((0, 2, 4), (1, 3, 5))
    summed = {}
    for outcome in fcs_outcomes(groups):
        for target, coefficient in joint_projector_connections(bits, groups, outcome):
            summed[target] = summed.get(target, 0.0) + coefficient
    assert abs(summed.pop(bits) - 1.0) < 2.0e-12
    assert max((abs(value) for value in summed.values()), default=0.0) < 2.0e-12


def test_projector_local_estimator_closes_exact_complex_fcs_and_covariance():
    rng = np.random.default_rng(918)
    n_sites = 6
    wavefunction = rng.normal(size=2**n_sites) + 1j * rng.normal(size=2**n_sites)
    wavefunction /= np.linalg.norm(wavefunction)
    configurations = np.asarray(
        [
            [(integer >> (n_sites - site - 1)) & 1 for site in range(n_sites)]
            for integer in range(2**n_sites)
        ],
        dtype=np.int8,
    )

    def log_amplitudes(batch):
        integers = np.asarray(
            [
                sum(int(bit) << (n_sites - site - 1) for site, bit in enumerate(row))
                for row in np.asarray(batch)
            ]
        )
        values = wavefunction[integers]
        return np.log(np.abs(values)) + 1j * np.angle(values)

    groups = ((0, 2, 4), (1, 3, 5))
    result = weighted_projector_fcs_statistics(
        configurations,
        np.abs(wavefunction) ** 2,
        log_amplitudes,
        groups,
    )
    expected = block_spin_fcs(wavefunction, groups)
    keys = result["outcomes"]
    np.testing.assert_allclose(
        [result["probabilities"][key] for key in keys],
        [expected[key] for key in keys],
        atol=2.0e-12,
    )
    probabilities = np.asarray([expected[key] for key in keys])
    categorical_covariance = np.diag(probabilities) - np.outer(
        probabilities, probabilities
    )
    np.testing.assert_allclose(
        result["complex_covariance"], categorical_covariance, atol=3.0e-12
    )
    assert result["maximum_imaginary_expectation"] < 2.0e-12
    assert result["maximum_local_sum_rule_error"] < 2.0e-12


def test_local_values_use_amplitude_ratios_and_sum_to_identity():
    wavefunction = np.asarray([0.2, -0.3j, 0.4, 0.5j], dtype=np.complex128)
    wavefunction /= np.linalg.norm(wavefunction)

    def log_amplitudes(batch):
        integers = np.asarray([2 * int(row[0]) + int(row[1]) for row in batch])
        values = wavefunction[integers]
        return np.log(np.abs(values)) + 1j * np.angle(values)

    values, audit = projector_local_values((0, 1), log_amplitudes, ((0, 1),))
    assert set(values) == {(0,), (2,)}
    np.testing.assert_allclose(sum(values.values()), 1.0, atol=2.0e-12)
    assert audit["unique_amplitude_evaluations"] <= 2

    completed, completed_audit = projector_local_values(
        (0, 1),
        log_amplitudes,
        ((0, 1),),
        complete_last_outcome=True,
    )
    np.testing.assert_allclose(sum(completed.values()), 1.0, atol=1.0e-15)
    assert completed_audit["completeness_reconstructed_outcome"] == 1
    with pytest.raises(ValueError, match="canonical exhaustive support"):
        projector_local_values(
            (0, 1),
            log_amplitudes,
            ((0, 1),),
            outcomes=((0,),),
            complete_last_outcome=True,
        )


def test_tree_native_samples_read_group_root_labels_without_fcs_oracle():
    tree, indices = grouped_fusion_tree((2, 2))
    states = tree.enumerate_states(0)
    amplitudes = np.asarray([np.sqrt(0.35), np.sqrt(0.65)], dtype=np.complex128)
    exact, _ = tree_block_spin_fcs_with_tangent(
        amplitudes, np.zeros_like(amplitudes), states, indices
    )
    sampled = sample_tree_native_fcs(
        amplitudes,
        states,
        indices,
        50_000,
        72,
        support=tuple(exact),
    )
    np.testing.assert_allclose(
        [sampled[key] for key in exact], [exact[key] for key in exact], atol=0.01
    )
    with pytest.raises(ValueError, match="outside the fixed support"):
        sample_tree_native_fcs(
            amplitudes,
            states,
            indices,
            100,
            72,
            support=((0, 0),),
        )


def test_simplex_projection_is_normalized_nonnegative_and_identity_on_distribution():
    exact = {(0,): 0.2, (2,): 0.3, (4,): 0.5}
    assert project_distribution_to_simplex(exact) == exact
    projected = project_distribution_to_simplex({(0,): -0.1, (2,): 0.4, (4,): 0.8})
    assert min(projected.values()) >= 0.0
    np.testing.assert_allclose(sum(projected.values()), 1.0, atol=1.0e-15)
