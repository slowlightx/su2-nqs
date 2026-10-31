from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.local_indistinguishability import (
    local_code_decoupling_defect,
    local_code_qddm_singular_values,
    local_invariant_moments,
    pure_state_partial_trace,
    pure_state_transition_partial_trace,
    three_spin_su2_rdm,
    three_spin_su2_transition,
    three_spin_trace_distance_from_moments,
    trace_distance,
    transition_trace_norm,
    two_spin_su2_rdm,
    two_spin_su2_transition,
    two_spin_trace_distance_from_correlators,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def _random_four_spin_singlet(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amplitudes = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    amplitudes /= np.linalg.norm(amplitudes)
    coupled = coupled_basis_matrix(enumerate_paths(4, 0), 0, 0)
    return np.asarray(coupled @ amplitudes, dtype=np.complex128)


def test_reduced_two_and_three_spin_tomography_matches_partial_trace() -> None:
    state = _random_four_spin_singlet(714)
    rho2 = pure_state_partial_trace(state, 4, (0, 2))
    pair = local_invariant_moments(rho2)["pair_correlations"][0]
    reconstructed2 = two_spin_su2_rdm(pair)
    np.testing.assert_allclose(reconstructed2, rho2, atol=3.0e-14)

    rho3 = pure_state_partial_trace(state, 4, (0, 1, 3))
    moments = local_invariant_moments(rho3)
    reconstructed3 = three_spin_su2_rdm(
        moments["pair_correlations"], moments["scalar_chirality"]
    )
    np.testing.assert_allclose(reconstructed3, rho3, atol=3.0e-14)
    assert abs(np.trace(reconstructed3) - 1.0) < 2.0e-14
    assert np.min(np.linalg.eigvalsh(reconstructed3)) > -3.0e-14


def test_reduced_trace_distance_equals_direct_computational_distance() -> None:
    first = _random_four_spin_singlet(100)
    second = _random_four_spin_singlet(101)
    direct_first = pure_state_partial_trace(first, 4, (0, 1, 2))
    direct_second = pure_state_partial_trace(second, 4, (0, 1, 2))
    first_moments = local_invariant_moments(direct_first)
    second_moments = local_invariant_moments(direct_second)
    reduced_first = three_spin_su2_rdm(
        first_moments["pair_correlations"], first_moments["scalar_chirality"]
    )
    reduced_second = three_spin_su2_rdm(
        second_moments["pair_correlations"], second_moments["scalar_chirality"]
    )
    assert trace_distance(reduced_first, reduced_second) == pytest.approx(
        trace_distance(direct_first, direct_second), abs=3.0e-14
    )


def test_two_site_distance_is_absolute_correlator_difference() -> None:
    first, second = -0.55, 0.10
    assert trace_distance(two_spin_su2_rdm(first), two_spin_su2_rdm(second)) == pytest.approx(
        abs(first - second), abs=2.0e-14
    )
    assert two_spin_trace_distance_from_correlators(first, second) == pytest.approx(
        abs(first - second), abs=2.0e-14
    )


def test_time_reversal_real_singlet_has_zero_local_chirality() -> None:
    state = _random_four_spin_singlet(203).real
    state /= np.linalg.norm(state)
    rho3 = pure_state_partial_trace(state, 4, (0, 1, 2))
    moments = local_invariant_moments(rho3)
    assert abs(moments["scalar_chirality"]) < 2.0e-14


def test_closed_three_site_distance_matches_direct_trace_norm() -> None:
    first = _random_four_spin_singlet(301)
    second = _random_four_spin_singlet(302)
    first_rdm = pure_state_partial_trace(first, 4, (0, 2, 3))
    second_rdm = pure_state_partial_trace(second, 4, (0, 2, 3))
    first_moments = local_invariant_moments(first_rdm)
    second_moments = local_invariant_moments(second_rdm)
    closed = three_spin_trace_distance_from_moments(
        first_moments["pair_correlations"],
        first_moments["scalar_chirality"],
        second_moments["pair_correlations"],
        second_moments["scalar_chirality"],
    )
    assert closed == pytest.approx(trace_distance(first_rdm, second_rdm), abs=3.0e-14)


def test_complex_transition_tomography_matches_direct_partial_trace() -> None:
    left = _random_four_spin_singlet(401)
    right = _random_four_spin_singlet(402)
    overlap = np.vdot(right, left)

    direct2 = pure_state_transition_partial_trace(left, right, 4, (0, 3))
    identity_part = np.trace(direct2)
    pair_operator = (two_spin_su2_rdm(1.0) - np.eye(4) / 4.0) * 0.75
    pair2 = np.trace(direct2 @ pair_operator)
    reconstructed2 = two_spin_su2_transition(identity_part, pair2)
    np.testing.assert_allclose(reconstructed2, direct2, atol=4.0e-14)

    direct3 = pure_state_transition_partial_trace(left, right, 4, (0, 1, 3))
    # Obtain the five complex transition moments using dual reconstruction
    # identities from explicitly perturbed invariant density matrices.
    pair_operators = []
    for index in range(3):
        moments = [0.0, 0.0, 0.0]
        moments[index] = 1.0
        pair_operators.append((three_spin_su2_rdm(moments, 0.0) - np.eye(8) / 8.0) * 1.5)
    chirality_operator = (three_spin_su2_rdm((0.0, 0.0, 0.0), 1.0) - np.eye(8) / 8.0) * 0.75
    pair_moments = tuple(np.trace(direct3 @ operator) for operator in pair_operators)
    chirality_moment = np.trace(direct3 @ chirality_operator)
    reconstructed3 = three_spin_su2_transition(
        overlap, pair_moments, chirality_moment
    )
    np.testing.assert_allclose(reconstructed3, direct3, atol=4.0e-14)
    assert transition_trace_norm(reconstructed3) == pytest.approx(
        transition_trace_norm(direct3), abs=4.0e-14
    )


def test_local_tomography_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="three finite"):
        three_spin_su2_rdm((0.1, 0.2), 0.0)
    with pytest.raises(ValueError, match=r"2\*\*n_sites"):
        pure_state_partial_trace(np.ones(4), 3, (0, 1))
    with pytest.raises(ValueError, match="equal shape"):
        trace_distance(np.eye(2), np.eye(3))


def _two_root_local_blocks(
    roots: np.ndarray, retained_sites: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        pure_state_partial_trace(roots[:, 0], 4, retained_sites),
        pure_state_partial_trace(roots[:, 1], 4, retained_sites),
        pure_state_transition_partial_trace(
            roots[:, 0], roots[:, 1], 4, retained_sites
        ),
    )


@pytest.mark.parametrize("retained_sites", [(0, 2), (0, 1, 3)])
def test_local_code_metrics_are_invariant_under_root_rotation(
    retained_sites: tuple[int, ...],
) -> None:
    roots = np.asarray(
        coupled_basis_matrix(enumerate_paths(4, 0), 0, 0), dtype=np.complex128
    )
    rng = np.random.default_rng(917)
    matrix = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    rotation, _ = np.linalg.qr(matrix)
    rotated = roots @ rotation
    blocks = _two_root_local_blocks(roots, retained_sites)
    rotated_blocks = _two_root_local_blocks(rotated, retained_sites)
    assert local_code_decoupling_defect(*blocks) == pytest.approx(
        local_code_decoupling_defect(*rotated_blocks), abs=4.0e-14
    )
    np.testing.assert_allclose(
        local_code_qddm_singular_values(*blocks),
        local_code_qddm_singular_values(*rotated_blocks),
        atol=4.0e-14,
    )


def test_local_code_metrics_vanish_at_exact_local_indistinguishability() -> None:
    local_state = np.eye(4, dtype=np.complex128) / 4.0
    transition = np.zeros_like(local_state)
    assert local_code_decoupling_defect(
        local_state, local_state, transition
    ) == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        local_code_qddm_singular_values(local_state, local_state, transition),
        np.zeros(3),
        atol=1.0e-15,
    )


def test_local_code_metric_normalization_limits() -> None:
    zero = np.asarray([[1.0], [0.0]], dtype=np.complex128)
    one = np.asarray([[0.0], [1.0]], dtype=np.complex128)
    rho00, rho11 = zero @ zero.conj().T, one @ one.conj().T

    dephased_transition = np.zeros((2, 2), dtype=np.complex128)
    assert local_code_decoupling_defect(
        rho00, rho11, dephased_transition
    ) == pytest.approx(0.5, abs=2.0e-15)
    np.testing.assert_allclose(
        local_code_qddm_singular_values(rho00, rho11, dephased_transition),
        (1.0, 0.0, 0.0),
        atol=2.0e-15,
    )

    identity_transition = zero @ one.conj().T
    assert local_code_decoupling_defect(
        rho00, rho11, identity_transition
    ) == pytest.approx(0.75, abs=2.0e-15)
    np.testing.assert_allclose(
        local_code_qddm_singular_values(rho00, rho11, identity_transition),
        (1.0, 1.0, 1.0),
        atol=2.0e-15,
    )


def test_decoupling_choi_orientation_matches_reference_partial_trace() -> None:
    roots = np.asarray(
        coupled_basis_matrix(enumerate_paths(4, 0), 0, 0), dtype=np.complex128
    )
    retained = (0, 2)
    rho00, rho11, rho01 = _two_root_local_blocks(roots, retained)
    choi = np.block([[rho00, rho01], [rho01.conj().T, rho11]])
    assert np.trace(choi) == pytest.approx(2.0, abs=2.0e-14)
    assert np.min(np.linalg.eigvalsh(choi)) > -2.0e-14

    omega = np.concatenate((roots[:, 0], roots[:, 1])) / np.sqrt(2.0)
    direct_reference_state = pure_state_partial_trace(
        omega, 5, (0, *(site + 1 for site in retained))
    )
    np.testing.assert_allclose(choi / 2.0, direct_reference_state, atol=3.0e-14)

    average = (rho00 + rho11) / 2.0
    uncorrelated = np.kron(np.eye(2) / 2.0, average)
    assert local_code_decoupling_defect(rho00, rho11, rho01) == pytest.approx(
        trace_distance(direct_reference_state, uncorrelated), abs=3.0e-14
    )


def test_scalar_chirality_sign_matches_swap_commutator() -> None:
    swap = np.asarray(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )
    swap01 = np.kron(swap, np.eye(2))
    swap12 = np.kron(np.eye(2), swap)
    commutator_form = 0.25j * (swap01 @ swap12 - swap12 @ swap01)
    reconstructed_chirality = 0.75 * (
        three_spin_su2_rdm((0.0, 0.0, 0.0), 1.0) - np.eye(8) / 8.0
    )
    np.testing.assert_allclose(
        reconstructed_chirality, commutator_form, atol=2.0e-14
    )
