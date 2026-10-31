import math

import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_state_to_sz
from su2_nqs.benchmarks.reduced_vector_reference import (
    compile_weighted_spin_vector_reference,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.quantum_numbers import magnetic_numbers
from su2_nqs.su2.reduced_vector import (
    Q_OP_VECTOR,
    VECTOR_DELTA_Q,
    advance_weighted_spin_vector,
    compile_weighted_spin_vector,
    contract_reduced_vector_scalar,
    coupled_multiplet_basis,
    spherical_component_coupled_matrix,
)


def _local_operator(n_sites: int, site: int, value: np.ndarray) -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    answer = np.asarray([[1.0 + 0.0j]])
    for position in range(n_sites):
        answer = np.kron(answer, value if position == site else identity)
    return answer


def _spin_components() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Local computational order is bit 0 = down, bit 1 = up.
    sx = 0.5 * np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.asarray([[0.0, 1.0j], [-1.0j, 0.0]], dtype=np.complex128)
    sz = 0.5 * np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    return sx, sy, sz


def _direct_spherical(
    endpoint_weights: np.ndarray, mu_q: int
) -> np.ndarray:
    sx, sy, sz = _spin_components()
    if mu_q == 0:
        local = sz
    elif mu_q == 2:
        local = -(sx + 1.0j * sy) / math.sqrt(2.0)
    elif mu_q == -2:
        local = (sx - 1.0j * sy) / math.sqrt(2.0)
    else:
        raise ValueError("invalid spherical component")
    n_sites = len(endpoint_weights)
    return sum(
        (
            weight * _local_operator(n_sites, site, local)
            for site, weight in enumerate(endpoint_weights)
        ),
        start=np.zeros((1 << n_sites, 1 << n_sites), dtype=np.complex128),
    )


def _complete_coupled_transform(table) -> np.ndarray:
    return np.column_stack(
        [
            coupled_state_to_sz(path, path[-1], m)
            for path, m in coupled_multiplet_basis(table)
        ]
    )


def _two_block_transform(
    basis, target_q: int, total_m: int
) -> np.ndarray:
    columns = []
    for left_path, right_path in basis:
        q_left, q_right = left_path[-1], right_path[-1]
        vector = np.zeros(
            1 << (len(left_path) + len(right_path)), dtype=np.complex128
        )
        for m_left in magnetic_numbers(q_left):
            m_right = total_m - m_left
            if abs(m_right) > q_right or (q_right - m_right) % 2:
                continue
            coefficient = clebsch_gordan(
                q_left,
                m_left,
                q_right,
                m_right,
                target_q,
                total_m,
            )
            vector += coefficient * np.kron(
                coupled_state_to_sz(left_path, q_left, m_left),
                coupled_state_to_sz(right_path, q_right, m_right),
            )
        columns.append(vector)
    return np.column_stack(columns)


def _direct_cross_scalar(coupling: np.ndarray) -> np.ndarray:
    left_n, right_n = coupling.shape
    n_sites = left_n + right_n
    result = np.zeros((1 << n_sites, 1 << n_sites), dtype=np.complex128)
    for left_site in range(left_n):
        for right_site in range(right_n):
            for component in _spin_components():
                result += coupling[left_site, right_site] * (
                    _local_operator(n_sites, left_site, component)
                    @ _local_operator(n_sites, left_n + right_site, component)
                )
    return result


def _maximum_sparse_difference(first, second) -> float:
    left = first.entry_dict()
    right = second.entry_dict()
    return max(
        (abs(left.get(key, 0.0j) - right.get(key, 0.0j)) for key in set(left) | set(right)),
        default=0.0,
    )


def test_single_spin_condon_shortley_reduced_convention_and_orientation() -> None:
    table = compile_weighted_spin_vector([1.0])
    assert table.q_op == Q_OP_VECTOR == 2
    assert table.delta_q == VECTOR_DELTA_Q == (-2, 0, 2)
    assert table.reduced_element((1,), (1,)) == pytest.approx(math.sqrt(1.5))

    raising, basis = spherical_component_coupled_matrix(table, 2)
    positions = {state: index for index, state in enumerate(basis)}
    # T_+1=-S_+/sqrt(2), with rows new and columns old.
    assert raising[positions[((1,), 1)], positions[((1,), -1)]] == pytest.approx(
        -1.0 / math.sqrt(2.0)
    )
    assert raising[positions[((1,), -1)], positions[((1,), 1)]] == 0.0


def test_recursive_tables_match_independent_computational_reference_through_n8() -> None:
    rng = np.random.default_rng(2901)
    for n_sites in range(1, 9):
        weights = rng.normal(size=n_sites) + 0.23j * rng.normal(size=n_sites)
        recursive = compile_weighted_spin_vector(weights, zero_tolerance=1.0e-13)
        reference = compile_weighted_spin_vector_reference(
            weights, zero_tolerance=1.0e-13
        )
        assert recursive.paths == reference.paths
        assert _maximum_sparse_difference(recursive, reference) < 8.0e-14
        assert set(map(int, recursive.q_new - recursive.q_old)) <= set(
            VECTOR_DELTA_Q
        )
        payload = recursive.numeric_payload()
        assert all(isinstance(value, np.ndarray) for value in payload.values())
        assert payload["paths"].dtype == np.int32
        assert payload["reduced_values"].dtype == np.complex128


def test_advance_api_matches_direct_compile_and_all_spherical_components() -> None:
    weights = np.asarray([0.7 + 0.2j, -0.4, 1.1j, 0.3, -0.8j])
    table = compile_weighted_spin_vector(weights[:4])
    advanced = advance_weighted_spin_vector(table, weights[4])
    direct = compile_weighted_spin_vector(weights)
    assert _maximum_sparse_difference(advanced, direct) < 2.0e-14

    transform = _complete_coupled_transform(advanced)
    np.testing.assert_allclose(
        transform.conj().T @ transform,
        np.eye(transform.shape[1]),
        atol=3.0e-14,
    )
    for mu_q in (-2, 0, 2):
        reduced, _ = spherical_component_coupled_matrix(advanced, mu_q)
        exact = transform.conj().T @ _direct_spherical(weights, mu_q) @ transform
        np.testing.assert_allclose(reduced, exact, atol=8.0e-14)


@pytest.mark.parametrize("target_q", (0, 2))
def test_two_block_reduced_scalar_reconstructs_arbitrary_cross_couplings(
    target_q: int,
) -> None:
    rng = np.random.default_rng(740 + target_q)
    coupling = rng.normal(size=(3, 3))
    left_singular, singular, right_singular_h = np.linalg.svd(
        coupling, full_matrices=False
    )

    reduced_sum = None
    basis = None
    for channel, value in enumerate(singular):
        scale = math.sqrt(value)
        left = compile_weighted_spin_vector(scale * left_singular[:, channel])
        right = compile_weighted_spin_vector(
            scale * right_singular_h[channel, :]
        )
        contribution, channel_basis = contract_reduced_vector_scalar(
            left, right, target_q, total_m=0
        )
        if reduced_sum is None:
            reduced_sum = np.zeros_like(contribution)
            basis = channel_basis
        assert channel_basis == basis
        reduced_sum += contribution

    transform = _two_block_transform(basis, target_q, 0)
    exact = transform.conj().T @ _direct_cross_scalar(coupling) @ transform
    np.testing.assert_allclose(reduced_sum, exact, atol=1.2e-12)
    np.testing.assert_allclose(reduced_sum, reduced_sum.conj().T, atol=1.0e-13)


def test_scalar_contraction_is_independent_of_total_m() -> None:
    left = compile_weighted_spin_vector([0.2, -0.7, 1.1])
    right = compile_weighted_spin_vector([0.4, 0.9, -0.3])
    reference, basis = contract_reduced_vector_scalar(
        left, right, target_q=2, total_m=0
    )
    for total_m in (-2, 2):
        value, other_basis = contract_reduced_vector_scalar(
            left, right, target_q=2, total_m=total_m
        )
        assert other_basis == basis
        np.testing.assert_allclose(value, reference, atol=3.0e-14)
