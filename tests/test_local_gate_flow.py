import numpy as np
from scipy.linalg import expm

from su2_nqs.benchmarks.local_gate_flow import (
    apply_exchange_gate,
    exchange_commutator_action,
    exchange_commutator_motif_diagnostics,
    exchange_second_order_action,
    lie_exchange_flow,
    orthogonal_component,
    strang_exchange_flow,
    twirled_strang_exchange_flow,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_path import count_paths


def _problem():
    n_sites = 6
    endpoints = ((0, 1), (1, 2), (3, 4), (2, 5))
    operators = tuple(
        heisenberg_graph_fusion_matrix(
            n_sites, ((first, second, 1.0),), target_q=0
        ).toarray()
        for first, second in endpoints
    )
    weights = (1.0, 0.4, -0.3, 0.7)
    generator = np.random.default_rng(9204)
    vector = generator.normal(size=count_paths(n_sites, 0))
    vector = vector.astype(np.complex128)
    vector /= np.linalg.norm(vector)
    hamiltonian = sum(
        (weight * operator for weight, operator in zip(weights, operators, strict=True)),
        np.zeros_like(operators[0]),
    )
    return vector, operators, endpoints, weights, hamiltonian


def test_spin_half_exchange_gate_matches_dense_matrix_exponential():
    vector, operators, _, weights, _ = _problem()
    step = 0.17
    actual = apply_exchange_gate(vector, operators[1], step, weights[1])
    expected = expm(-step * weights[1] * operators[1]) @ vector
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_lie_leading_error_is_a_connected_commutator_action():
    vector, operators, endpoints, weights, hamiltonian = _problem()
    action = exchange_commutator_action(vector, operators, endpoints, weights)
    assert action.connected_pair_count > 0
    assert action.disconnected_pair_count > 0
    np.testing.assert_allclose(action.disconnected, 0.0, atol=2e-15)
    assert action.disconnected_incoherent_norm < 2e-15
    step = 2.0e-4
    exact = expm(-step * hamiltonian) @ vector
    finite_difference = (
        lie_exchange_flow(vector, operators, weights, step) - exact
    ) / step**2
    np.testing.assert_allclose(
        finite_difference, action.connected, rtol=8e-4, atol=8e-5
    )


def test_palindromic_flow_cancels_the_connected_second_order_error():
    vector, operators, _, weights, hamiltonian = _problem()

    def error(method, step):
        exact = expm(-step * hamiltonian) @ vector
        return np.linalg.norm(method(vector, operators, weights, step) - exact)

    lie_ratio = error(lie_exchange_flow, 0.02) / error(
        lie_exchange_flow, 0.01
    )
    strang_ratio = error(strang_exchange_flow, 0.02) / error(
        strang_exchange_flow, 0.01
    )
    assert 3.8 < lie_ratio < 4.2
    assert 7.5 < strang_ratio < 8.5


def test_second_order_local_products_reconstruct_h_squared_exactly():
    vector, operators, endpoints, weights, hamiltonian = _problem()
    action = exchange_second_order_action(
        vector, operators, endpoints, weights
    )
    expected = hamiltonian @ (hamiltonian @ vector)
    np.testing.assert_allclose(action.total, expected, atol=3e-15)
    assert action.connected_pair_count > 0
    assert action.disconnected_pair_count > 0
    assert action.same_bond_columns.shape == (len(vector), len(operators))
    commutator = exchange_commutator_action(
        vector, operators, endpoints, weights
    )
    np.testing.assert_allclose(
        action.connected_commutator, commutator.connected, atol=3e-15
    )


def test_same_bond_second_order_action_reduces_to_k0_plus_k1():
    vector, operators, endpoints, weights, _ = _problem()
    action = exchange_second_order_action(
        vector, operators, endpoints, weights
    )
    expected = np.zeros_like(vector)
    for operator, weight in zip(operators, weights, strict=True):
        expected += weight**2 * (3.0 / 16.0 * vector - 0.5 * operator @ vector)
    np.testing.assert_allclose(action.same_bond, expected, atol=3e-15)


def test_group_twirled_flow_is_identity_at_zero_and_covariant():
    vector, operators, _, weights, _ = _problem()
    identity = np.eye(len(vector), dtype=np.complex128)
    reflection = np.fliplr(identity)
    group = (identity, reflection)
    zero = twirled_strang_exchange_flow(
        vector, operators, weights, 0.0, group
    )
    np.testing.assert_allclose(zero, vector, atol=2e-15)
    step = 0.07
    transformed_then_flowed = twirled_strang_exchange_flow(
        reflection @ vector, operators, weights, step, group
    )
    flowed_then_transformed = reflection @ twirled_strang_exchange_flow(
        vector, operators, weights, step, group
    )
    np.testing.assert_allclose(
        transformed_then_flowed, flowed_then_transformed, atol=2e-14
    )


def test_orthogonal_component_removes_only_scalar_flow():
    vector, _, _, _, _ = _problem()
    candidate = 2.0 * vector + np.arange(len(vector))
    residual = orthogonal_component(vector, candidate)
    assert abs(np.vdot(vector, residual)) < 2e-14


def test_commutator_is_an_oriented_three_spin_chirality_operator():
    identity = np.eye(2, dtype=np.complex128)
    paulis = (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.diag([1.0, -1.0]).astype(np.complex128),
    )

    def site_operator(site, matrix):
        factors = [identity, identity, identity]
        factors[site] = 0.5 * matrix
        return np.kron(np.kron(factors[0], factors[1]), factors[2])

    spins = tuple(
        tuple(site_operator(site, pauli) for pauli in paulis)
        for site in range(3)
    )
    h01 = sum(spins[0][axis] @ spins[1][axis] for axis in range(3))
    h12 = sum(spins[1][axis] @ spins[2][axis] for axis in range(3))
    chirality = (
        spins[0][0] @ (spins[1][1] @ spins[2][2] - spins[1][2] @ spins[2][1])
        + spins[0][1] @ (spins[1][2] @ spins[2][0] - spins[1][0] @ spins[2][2])
        + spins[0][2] @ (spins[1][0] @ spins[2][1] - spins[1][1] @ spins[2][0])
    )
    np.testing.assert_allclose(h01 @ h12 - h12 @ h01, -1.0j * chirality, atol=2e-15)


def test_commutator_motifs_separate_closed_loops_from_open_wedges():
    n_sites = 6
    endpoints = ((0, 1), (1, 2), (0, 2), (2, 3))
    shells = ("J1", "J1", "J2", "J1")
    operators = tuple(
        heisenberg_graph_fusion_matrix(
            n_sites, ((first, second, 1.0),), target_q=0
        ).tocsr()
        for first, second in endpoints
    )
    generator = np.random.default_rng(119)
    vector = generator.normal(size=count_paths(n_sites, 0)).astype(np.complex128)
    vector /= np.linalg.norm(vector)
    diagnostics = exchange_commutator_motif_diagnostics(
        vector, operators, endpoints, (1.0, 1.0, 0.4, 1.0), shells
    )
    action = exchange_commutator_action(
        vector, operators, endpoints, (1.0, 1.0, 0.4, 1.0)
    )
    assert diagnostics["connected_pair_count"] == action.connected_pair_count
    assert 0.0 < diagnostics["closed_loop_incoherent_norm_squared_fraction"] < 1.0
    closures = {row["outer_closure_shell"] for row in diagnostics["sectors"]}
    assert closures == {"J1", "J2", "open"}
    np.testing.assert_allclose(
        sum(
            row["bare_incoherent_norm_squared_fraction"]
            for row in diagnostics["sectors"]
        ),
        1.0,
        atol=2e-15,
    )
