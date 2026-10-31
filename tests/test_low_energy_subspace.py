from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.low_energy_subspace import (
    low_pair_isolation_diagnostics,
    matrix_free_chirality_action,
    matrix_free_structure_factor_actions,
    matrix_free_structure_factor_subspace,
    matrix_free_weighted_bond_actions,
    operator_panel_in_subspace,
    orthonormal_subspace,
    projected_operator,
    traceless_frobenius_norm_squared,
    two_state_block_spin_projectors,
    two_state_projector_family_analysis,
)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.lattice_observables import spin_structure_factor_operator
from su2_nqs.benchmarks.lattice_observables import scalar_chirality_fusion_matrix
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_path import enumerate_paths


def test_projected_operator_and_panel_are_exact() -> None:
    states = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    operator = np.diag([2.0, -1.0, 7.0])
    np.testing.assert_allclose(projected_operator(operator, states), np.diag([2.0, -1.0]))
    panel = operator_panel_in_subspace({"probe": operator}, states)
    assert panel["probe"]["trace"] == pytest.approx(1.0)
    assert panel["probe"]["traceless_frobenius_norm_squared"] == pytest.approx(4.5)


def test_projector_family_contrast_and_axis_are_known() -> None:
    projectors = {
        "up": np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        "down": np.asarray([[0.0, 0.0], [0.0, 1.0]]),
    }
    comparator = np.diag([3.0, -3.0])
    result = two_state_projector_family_analysis(projectors, comparator)
    assert result["contrast"] == pytest.approx(1.0)
    assert result["leading_axis_comparator_absolute_cosine"] == pytest.approx(1.0)
    assert result["resolution_of_identity_error"] < 1.0e-14


def test_two_state_analysis_is_invariant_under_subspace_rotation() -> None:
    theta = 0.37
    unitary = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.complex128,
    )
    projectors = {
        "zero": np.asarray([[0.7, 0.2], [0.2, 0.3]]),
        "one": np.asarray([[0.3, -0.2], [-0.2, 0.7]]),
    }
    comparator = np.asarray([[1.0, 0.4j], [-0.4j, -1.0]])
    first = two_state_projector_family_analysis(projectors, comparator)
    rotated = {
        key: unitary.conj().T @ value @ unitary for key, value in projectors.items()
    }
    second = two_state_projector_family_analysis(
        rotated, unitary.conj().T @ comparator @ unitary
    )
    assert first["contrast"] == pytest.approx(second["contrast"])
    assert first["leading_axis_comparator_absolute_cosine"] == pytest.approx(
        second["leading_axis_comparator_absolute_cosine"]
    )


def test_invalid_subspace_and_nonhermitian_projector_fail_closed() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        orthonormal_subspace(np.ones((3, 2)))
    with pytest.raises(ValueError, match="Hermitian"):
        two_state_projector_family_analysis(
            {"bad": np.asarray([[1.0, 1.0], [0.0, 0.0]])}, np.eye(2)
        )


def test_traceless_norm_is_scalar_shift_invariant() -> None:
    matrix = np.asarray([[1.0, 2.0j], [-2.0j, -1.0]])
    assert traceless_frobenius_norm_squared(matrix) == pytest.approx(
        traceless_frobenius_norm_squared(matrix + 7.0 * np.eye(2))
    )


def test_low_pair_isolation_reports_raw_gaps_and_json_safe_ratio() -> None:
    ordinary = low_pair_isolation_diagnostics((-3.0, -2.9, -2.0))
    assert ordinary["ground_pair_splitting"] == pytest.approx(0.1)
    assert ordinary["separation_from_third"] == pytest.approx(0.9)
    assert ordinary["isolation_ratio"] == pytest.approx(9.0)

    degenerate = low_pair_isolation_diagnostics((-3.0, -3.0, -2.0))
    assert degenerate["ground_pair_splitting"] == 0.0
    assert np.isfinite(degenerate["isolation_ratio"])

    with pytest.raises(ValueError, match="ordered"):
        low_pair_isolation_diagnostics((-3.0, -2.0, -2.5))


def test_block_spin_projector_compression_resolves_low_subspace_identity() -> None:
    paths = enumerate_paths(4, 0)
    states = coupled_basis_matrix(paths, 0, 0)
    projectors = two_state_block_spin_projectors(states, ((0, 2), (1, 3)))
    total = sum(projectors.values(), start=np.zeros((2, 2), dtype=np.complex128))
    np.testing.assert_allclose(total, np.eye(2), atol=3.0e-13)
    for matrix in projectors.values():
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=3.0e-13)


def test_matrix_free_structure_factor_matches_explicit_operator() -> None:
    n_sites = 6
    paths = enumerate_paths(n_sites, 0)
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(len(paths), 3)) + 1.0j * rng.normal(
        size=(len(paths), 3)
    )
    states, _ = np.linalg.qr(raw)
    coordinates = tuple((float(site % 3), float(site // 3)) for site in range(n_sites))
    wavevectors = {"a": (0.37, -0.21), "b": (np.pi, 0.0)}
    mapping = {site: (5 * site) % n_sites for site in range(n_sites)}
    actual = matrix_free_structure_factor_subspace(
        states,
        coordinates,
        wavevectors,
        mapping,
        target_q=0,
    )
    for name, wavevector in wavevectors.items():
        explicit = spin_structure_factor_operator(
            coordinates, wavevector, mapping, target_q=0
        )
        expected = states.conj().T @ (explicit @ states)
        np.testing.assert_allclose(actual[name], expected, atol=5.0e-14)


def test_matrix_free_structure_factor_is_subspace_covariant() -> None:
    paths = enumerate_paths(4, 0)
    states = np.eye(len(paths), dtype=np.complex128)
    theta = 0.29
    unitary = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.complex128,
    )
    coordinates = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    first = matrix_free_structure_factor_subspace(
        states, coordinates, {"probe": (0.4, 0.7)}
    )["probe"]
    second = matrix_free_structure_factor_subspace(
        states @ unitary, coordinates, {"probe": (0.4, 0.7)}
    )["probe"]
    np.testing.assert_allclose(second, unitary.conj().T @ first @ unitary, atol=3.0e-14)


def test_matrix_free_structure_factor_zero_momentum_and_normalizations() -> None:
    n_sites = 4
    target_q = 2
    dimension = len(enumerate_paths(n_sites, target_q))
    states = np.eye(dimension, dtype=np.complex128)
    coordinates = tuple((float(site), 0.0) for site in range(n_sites))
    by_n_squared = matrix_free_structure_factor_subspace(
        states, coordinates, {"zero": (0.0, 0.0)}, target_q=target_q
    )["zero"]
    by_n = matrix_free_structure_factor_subspace(
        states,
        coordinates,
        {"zero": (0.0, 0.0)},
        target_q=target_q,
        normalize_by_n_squared=False,
    )["zero"]
    expected = target_q * (target_q + 2) / (4.0 * n_sites**2)
    np.testing.assert_allclose(by_n_squared, expected * np.eye(dimension), atol=5.0e-14)
    np.testing.assert_allclose(by_n, n_sites * by_n_squared, atol=5.0e-14)


def test_matrix_free_structure_factor_panel_matches_single_calls_and_minus_q() -> None:
    paths = enumerate_paths(6, 0)
    rng = np.random.default_rng(91)
    raw = rng.normal(size=(len(paths), 2))
    states, _ = np.linalg.qr(raw)
    coordinates = tuple((float(site % 2), float(site // 2)) for site in range(6))
    wavevectors = {"plus": (0.31, 0.77), "minus": (-0.31, -0.77)}
    together = matrix_free_structure_factor_subspace(states, coordinates, wavevectors)
    for name, wavevector in wavevectors.items():
        separate = matrix_free_structure_factor_subspace(
            states, coordinates, {name: wavevector}
        )[name]
        np.testing.assert_allclose(together[name], separate, atol=3.0e-14)
        assert np.min(np.linalg.eigvalsh(0.5 * (separate + separate.conj().T))) > -1.0e-12
    np.testing.assert_allclose(together["plus"], together["minus"], atol=3.0e-14)


def test_matrix_free_weighted_bonds_and_chirality_match_explicit_operators() -> None:
    n_sites = 6
    paths = enumerate_paths(n_sites, 0)
    rng = np.random.default_rng(7)
    states = rng.normal(size=(len(paths), 2)) + 1.0j * rng.normal(size=(len(paths), 2))
    bonds = ((0, 4, 0.7), (1, 3, -0.2), (0, 4, 0.1))
    actions = matrix_free_weighted_bond_actions(
        states,
        n_sites,
        {"probe": bonds},
        constants={"probe": 0.37},
    )["probe"]
    explicit = heisenberg_graph_fusion_matrix(
        n_sites, bonds, target_q=0
    ) + 0.37 * np.eye(len(paths))
    np.testing.assert_allclose(actions, explicit @ states, atol=5.0e-14)

    triangle = ((0, 2, 5),)
    actual_chirality = matrix_free_chirality_action(
        states, n_sites, triangle, target_q=0
    )
    expected_chirality = scalar_chirality_fusion_matrix(
        n_sites, *triangle[0], target_q=0
    ) @ states
    np.testing.assert_allclose(actual_chirality, expected_chirality, atol=8.0e-14)


def test_matrix_free_source_validation_fails_closed() -> None:
    state = np.ones(len(enumerate_paths(4, 0)), dtype=np.complex128)
    with pytest.raises(ValueError, match="coordinates"):
        matrix_free_structure_factor_actions(
            state,
            ((0.0, 0.0), (1.0, 0.0), (np.nan, 0.0), (2.0, 0.0)),
            {"q": (0.0, 0.0)},
        )
    with pytest.raises(ValueError, match="wavevectors"):
        matrix_free_structure_factor_actions(
            state,
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
            {"q": (np.inf, 0.0)},
        )
    with pytest.raises(ValueError, match="bond endpoints"):
        matrix_free_weighted_bond_actions(
            state, 4, {"bad": ((0, 0, 1.0),)}
        )
