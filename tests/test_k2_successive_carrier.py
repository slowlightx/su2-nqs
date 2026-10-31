import numpy as np

from su2_nqs.benchmarks.k2_successive_carrier import (
    born_centered_vector,
    decompose_k2_log_cumulant,
    decompose_k2_successive_action,
    k1_complement_decomposition,
    log_cumulant_diagnostics,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_path import count_paths


def _problem():
    n_sites = 8
    endpoints = ((0, 1), (1, 2), (3, 4), (1, 4), (2, 5))
    operators = tuple(
        heisenberg_graph_fusion_matrix(
            n_sites, ((first, second, 1.0),), target_q=0
        ).tocsr()
        for first, second in endpoints
    )
    couplings = (1.0, 0.7, -0.2, 0.4, 0.9)
    generator = np.random.default_rng(8101)
    vector = generator.normal(size=count_paths(n_sites, 0)) + 1j * generator.normal(
        size=count_paths(n_sites, 0)
    )
    vector /= np.linalg.norm(vector)
    return vector, operators, endpoints, couplings


def test_k2_components_reconstruct_direct_h_squared_in_fusion_space():
    vector, operators, endpoints, couplings = _problem()
    action = decompose_k2_successive_action(
        vector, operators, endpoints, couplings
    )
    assert action.bond_count == len(endpoints)
    assert action.repeated_history_count == len(endpoints)
    assert (
        action.repeated_history_count
        + action.connected_history_count
        + action.disconnected_history_count
        == len(endpoints) ** 2
    )
    assert action.connected_history_count > 0
    assert action.disconnected_history_count > 0
    np.testing.assert_allclose(action.total, action.direct, rtol=2e-13, atol=2e-13)
    assert action.relative_closure_error < 2.0e-13


def test_repeated_bond_sector_is_exactly_reducible_to_k1():
    vector, operators, endpoints, couplings = _problem()
    action = decompose_k2_successive_action(
        vector, operators, endpoints, couplings
    )
    direct_repeated = sum(
        (
            coupling**2 * np.asarray(operator @ (operator @ vector))
            for coupling, operator in zip(couplings, operators, strict=True)
        ),
        np.zeros_like(vector),
    )
    np.testing.assert_allclose(action.repeated, direct_repeated, atol=2e-13)
    h_vector = sum(
        (
            coupling * np.asarray(operator @ vector)
            for coupling, operator in zip(couplings, operators, strict=True)
        ),
        np.zeros_like(vector),
    )
    first_actions = tuple(np.asarray(operator @ vector) for operator in operators)
    diagnostics = k1_complement_decomposition(
        vector,
        h_vector,
        action,
        resolved_first_actions=first_actions,
    )
    assert diagnostics["k1_space"] == "resolved"
    assert diagnostics["k1_rank"] >= 2
    assert diagnostics["repeated_complement_relative_norm"] < 2.0e-13
    assert diagnostics["complement_reconstruction_relative_error"] < 2.0e-13


def test_k2_oracle_rejects_malformed_operator_metadata():
    vector, operators, endpoints, couplings = _problem()
    with np.testing.assert_raises_regex(ValueError, "equal length"):
        decompose_k2_successive_action(
            vector, operators, endpoints[:-1], couplings
        )
    with np.testing.assert_raises_regex(ValueError, "distinct"):
        decompose_k2_successive_action(
            vector,
            operators,
            (*endpoints[:-1], (2, 2)),
            couplings,
        )


def test_k2_log_cumulant_reconstructs_direct_second_log_response():
    vector, operators, endpoints, couplings = _problem()
    cumulant = decompose_k2_log_cumulant(
        vector, operators, endpoints, couplings
    )
    h_vector = sum(
        (
            coupling * np.asarray(operator @ vector)
            for coupling, operator in zip(couplings, operators, strict=True)
        ),
        np.zeros_like(vector),
    )
    expected = cumulant.action.direct - h_vector * h_vector / vector
    np.testing.assert_allclose(cumulant.direct, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(cumulant.total, expected, rtol=2e-13, atol=2e-13)
    assert cumulant.relative_closure_error < 2.0e-13
    diagnostics = log_cumulant_diagnostics(vector, cumulant)
    assert diagnostics["centered_reconstruction_relative_error"] < 2.0e-13
    assert np.isfinite(diagnostics["centered_cumulant_norm"])


def _exchange_operator(n_sites: int, first: int, second: int) -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
    result = np.zeros((2**n_sites, 2**n_sites), dtype=np.complex128)
    for pauli in (pauli_x, pauli_y, pauli_z):
        factors = [identity] * n_sites
        factors[first] = pauli
        factors[second] = pauli
        term = factors[0]
        for factor in factors[1:]:
            term = np.kron(term, factor)
        result += 0.25 * term
    return result


def test_disconnected_log_cumulant_vanishes_for_factorized_regions():
    spinors = (
        np.array([1.0, 0.4j]),
        np.array([0.7, -0.3]),
        np.array([1.1, 0.2j]),
        np.array([0.6, 0.5]),
    )
    vector = spinors[0]
    for spinor in spinors[1:]:
        vector = np.kron(vector, spinor)
    vector = vector.astype(np.complex128)
    vector /= np.linalg.norm(vector)
    endpoints = ((0, 1), (2, 3))
    operators = tuple(
        _exchange_operator(4, first, second) for first, second in endpoints
    )
    cumulant = decompose_k2_log_cumulant(
        vector, operators, endpoints, (0.8, -0.6)
    )
    assert cumulant.action.disconnected_history_count == 2
    np.testing.assert_allclose(cumulant.disconnected, 0.0, atol=2e-14)


def test_log_cumulant_rejects_nodes_and_centering_removes_gauge():
    vector, operators, endpoints, couplings = _problem()
    nodal = vector.copy()
    nodal[0] = 0.0
    with np.testing.assert_raises_regex(ValueError, "every initial amplitude"):
        decompose_k2_log_cumulant(nodal, operators, endpoints, couplings)
    centered = born_centered_vector(vector, 2.3j * vector)
    np.testing.assert_allclose(centered, 0.0, atol=2e-15)
