from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.operator_string_locality import (
    ExchangeBond,
    connected_pair_catalog,
    fixed_rank_residual_basis,
    jordan_action_matrix,
    matched_farthest_disjoint_pairs,
    normalized_column_span,
    pair_shell_histogram,
    permutation_commutes_with_group,
    permute_exchange_bonds,
    residual_reference_capture,
    ritz_ground_state,
    weighted_graph_is_automorphism,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds


def _square_3x4_catalog() -> tuple[ExchangeBond, ...]:
    rows, columns = 3, 4
    entries = []
    for shell, couplings in (("J1", (1.0, 0.0)), ("J2", (0.0, 1.0))):
        for first, second, _ in square_j1j2_bonds(
            rows, columns, j1=couplings[0], j2=couplings[1], periodic=False
        ):
            entries.append(
                ExchangeBond(
                    len(entries),
                    (int(first), int(second)),
                    shell,
                    1.0 if shell == "J1" else 0.5,
                )
            )
    return tuple(entries)


def test_farthest_disjoint_catalog_exactly_matches_connected_shell_counts() -> None:
    bonds = _square_3x4_catalog()
    connected = connected_pair_catalog(bonds, n_sites=12)
    j1_edges = tuple(bond.sites for bond in bonds if bond.shell == "J1")
    disjoint = matched_farthest_disjoint_pairs(
        bonds, connected, n_sites=12, distance_edges=j1_edges
    )

    assert len(connected) == len(disjoint)
    assert pair_shell_histogram(connected) == pair_shell_histogram(disjoint)
    assert all(pair.connected for pair in connected)
    assert all(not pair.connected and pair.separation is not None for pair in disjoint)


def test_permutation_control_retains_labels_and_checks_centralizer() -> None:
    bonds = (
        ExchangeBond(0, (0, 1), "J1", 1.0),
        ExchangeBond(1, (1, 2), "J1", 1.0),
        ExchangeBond(2, (2, 3), "J2", 0.5),
    )
    reversal = (3, 2, 1, 0)
    image = (1, 0, 3, 2)
    transformed = permute_exchange_bonds(bonds, image, n_sites=4)

    assert [bond.shell for bond in transformed] == ["J1", "J1", "J2"]
    assert [bond.coupling for bond in transformed] == [1.0, 1.0, 0.5]
    assert permutation_commutes_with_group(image, ((0, 1, 2, 3), reversal))
    assert not weighted_graph_is_automorphism(bonds, image, n_sites=4)


def test_jordan_action_matrix_matches_explicit_two_orders_and_projection() -> None:
    rng = np.random.default_rng(1409)
    raw = rng.normal(size=(2, 4, 4))
    operators = tuple(matrix + matrix.T for matrix in raw)
    bonds = (
        ExchangeBond(0, (0, 1), "J1", 1.0),
        ExchangeBond(1, (1, 2), "J2", 0.25),
    )
    pair = connected_pair_catalog(bonds, n_sites=3)
    state = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    projector = np.diag([1.0, 1.0, 0.0, 0.0])

    observed = jordan_action_matrix(
        state, operators, bonds, pair, projector=projector
    )[:, 0]
    expected = 0.25 * projector @ (
        operators[1] @ (operators[0] @ state)
        + operators[0] @ (operators[1] @ state)
    )
    np.testing.assert_allclose(observed, expected, atol=1.0e-13)


def test_spin_half_connected_jordan_is_half_outer_exchange() -> None:
    identity = np.eye(2, dtype=np.complex128)
    pauli = (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    )

    def exchange(first: int, second: int) -> np.ndarray:
        result = np.zeros((8, 8), dtype=np.complex128)
        for sigma in pauli:
            factors = [identity, identity, identity]
            factors[first] = sigma
            factors[second] = sigma
            term = np.kron(np.kron(factors[0], factors[1]), factors[2])
            result += 0.25 * term
        return result

    h01 = exchange(0, 1)
    h12 = exchange(1, 2)
    h02 = exchange(0, 2)
    np.testing.assert_allclose(h01 @ h12 + h12 @ h01, 0.5 * h02, atol=1.0e-14)


def test_fixed_rank_basis_removes_complete_common_span_before_svd() -> None:
    common, common_audit = normalized_column_span(np.eye(8)[:, :2])
    candidates = np.column_stack(
        (
            100.0 * np.eye(8)[:, 0] + np.eye(8)[:, 2],
            np.eye(8)[:, 3],
            np.eye(8)[:, 4],
            np.eye(8)[:, 5],
        )
    )
    augmentation, audit = fixed_rank_residual_basis(
        common, candidates, 2, relative_tolerance=1.0e-12
    )

    assert common_audit["rank"] == 2
    assert audit["rank_satisfied"] is True
    assert audit["selected_rank"] == 2
    assert len(audit["residual_norm_quantiles"]) == 5
    np.testing.assert_allclose(common.conj().T @ augmentation, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        augmentation.conj().T @ augmentation, np.eye(2), atol=1.0e-13
    )


def test_capture_and_ritz_are_scored_after_basis_construction() -> None:
    hamiltonian = np.diag(np.arange(6.0))
    common, _ = normalized_column_span(np.column_stack((np.eye(6)[:, 1],)))
    augmentation, audit = fixed_rank_residual_basis(
        common,
        np.column_stack(
            (np.eye(6)[:, 0], np.eye(6)[:, 0], np.eye(6)[:, 2])
        ),
        1,
        relative_tolerance=1.0e-12,
    )
    basis = np.column_stack((common, augmentation))
    state, ritz = ritz_ground_state(hamiltonian, basis)
    capture = residual_reference_capture(np.eye(6)[:, 0], common, augmentation)

    assert audit["rank_satisfied"] is True
    assert audit["relative_boundary_gap"] > 0.0
    assert ritz["energy"] == 0.0
    assert ritz["projected_ritz_residual"] < 1.0e-14
    assert abs(state[0]) == 1.0
    assert capture["capture_fraction"] == pytest.approx(1.0)


def test_column_normalization_is_invariant_to_arbitrary_nonzero_scale() -> None:
    columns = np.column_stack(
        (1.0e-30 * np.eye(4)[:, 0], 1.0e30 * np.eye(4)[:, 1])
    )
    basis, audit = normalized_column_span(columns)

    assert audit["rank"] == 2
    np.testing.assert_allclose(
        basis @ basis.conj().T,
        np.diag([1.0, 1.0, 0.0, 0.0]),
        atol=1.0e-13,
    )


def test_capture_rejects_nonorthogonal_augmentation() -> None:
    common = np.eye(4)[:, :1]
    bad = np.eye(4)[:, :1]
    with np.testing.assert_raises(ValueError):
        residual_reference_capture(np.eye(4)[:, 1], common, bad)
