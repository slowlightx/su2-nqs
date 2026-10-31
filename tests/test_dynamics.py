import numpy as np

from su2_nqs.benchmarks.dynamics import (
    bipartite_entropy_trajectory,
    cluster_local_action_basis,
    compare_trajectories,
    connected_local_action_basis,
    crank_nicolson_evolution,
    exact_time_evolution,
    expectation_trajectory,
    global_krylov_basis,
    ground_state,
    projected_time_evolution,
    principal_local_action_basis,
    state_fidelities,
)
from su2_nqs.lattices.chain import dimerized_chain_couplings
from su2_nqs.operators.heisenberg_chain import HeisenbergChain


def _bond_operators(n: int) -> tuple[np.ndarray, ...]:
    operators = []
    for bond in range(n - 1):
        couplings = [0.0] * (n - 1)
        couplings[bond] = 1.0
        operators.append(HeisenbergChain(n, 0, tuple(couplings)).to_dense())
    return tuple(operators)


def test_dimerized_couplings_fix_strong_bond_orientation_and_limits():
    np.testing.assert_allclose(
        dimerized_chain_couplings(6, coupling=2.0, dimerization=0.25),
        (2.5, 1.5, 2.5, 1.5, 2.5),
    )


def test_bipartite_entropy_distinguishes_product_and_bell_states():
    states = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    entropies = bipartite_entropy_trajectory(states, 2, base=2.0)
    np.testing.assert_allclose(entropies, (0.0, 1.0), atol=2e-14)
    np.testing.assert_allclose(
        dimerized_chain_couplings(4, dimerization=1.0), (2.0, 0.0, 2.0)
    )


def test_exact_quench_evolution_preserves_norm_and_postquench_energy():
    n = 6
    initial = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=0.8)
    ).to_dense()
    final = HeisenbergChain(n, 0).to_dense()
    _, state = ground_state(initial)
    times = np.linspace(0.0, 3.0, 61)
    trajectory = exact_time_evolution(final, state, times)
    np.testing.assert_allclose(np.sum(np.abs(trajectory) ** 2, axis=1), 1.0, atol=2e-14)
    energies = expectation_trajectory(trajectory, final)
    np.testing.assert_allclose(energies, energies[0], atol=2e-14)


def test_crank_nicolson_has_unit_norm_and_converges_at_second_order():
    n = 6
    initial = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=0.7)
    ).to_dense()
    final = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=-0.2)
    ).to_dense()
    _, state = ground_state(initial)
    total_time = 2.0
    coarse = crank_nicolson_evolution(final, state, time_step=0.1, steps=20)
    fine = crank_nicolson_evolution(final, state, time_step=0.05, steps=40)
    exact_coarse = exact_time_evolution(final, state, np.linspace(0.0, total_time, 21))
    exact_fine = exact_time_evolution(final, state, np.linspace(0.0, total_time, 41))
    coarse_error = compare_trajectories(exact_coarse, coarse, final)
    fine_error = compare_trajectories(exact_fine, fine, final)
    assert coarse_error.maximum_norm_error < 2e-14
    assert fine_error.maximum_norm_error < 3e-14
    assert fine_error.maximum_infidelity < coarse_error.maximum_infidelity / 8.0
    assert fine_error.maximum_energy_error < 3e-14


def test_resolved_local_action_hierarchy_reaches_exact_small_quench_space():
    n = 6
    initial_hamiltonian = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=1.0)
    ).to_dense()
    final_hamiltonian = HeisenbergChain(n, 0).to_dense()
    _, state = ground_state(initial_hamiltonian)
    local = _bond_operators(n)
    bonds = tuple((bond, bond + 1) for bond in range(n - 1))
    resolved, sequences = connected_local_action_basis(state, local, bonds, 4)
    collapsed = global_krylov_basis(final_hamiltonian, state, 3)
    assert () in sequences
    assert resolved.shape[1] >= collapsed.shape[1]
    assert resolved.shape[1] == final_hamiltonian.shape[0]
    times = np.linspace(0.0, 2.0, 21)
    exact = exact_time_evolution(final_hamiltonian, state, times)
    projected = projected_time_evolution(final_hamiltonian, state, resolved, times)
    np.testing.assert_allclose(state_fidelities(exact, projected), 1.0, atol=2e-13)


def test_local_action_hierarchy_can_be_anchored_to_one_perturbing_bond():
    n = 6
    hamiltonian = HeisenbergChain(n, 0).to_dense()
    _, state = ground_state(hamiltonian)
    bonds = tuple((bond, bond + 1) for bond in range(n - 1))
    _, sequences = connected_local_action_basis(
        state,
        _bond_operators(n),
        bonds,
        3,
        initial_operator_indices=(2,),
    )
    assert all(not sequence or sequence[0] == 2 for sequence in sequences)
    for sequence in sequences:
        support = set()
        for index in sequence:
            bond = set(bonds[index])
            assert not support or not support.isdisjoint(bond)
            support.update(bond)


def test_cluster_one_reproduces_connected_hierarchy_and_unrestricted_contains_krylov():
    n = 8
    initial = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=1.0)
    ).to_dense()
    final = HeisenbergChain(n, 0).to_dense()
    _, state = ground_state(initial)
    bonds = tuple((bond, bond + 1) for bond in range(n - 1))
    local = _bond_operators(n)
    connected, connected_sequences = connected_local_action_basis(
        state, local, bonds, 3
    )
    one_cluster, one_cluster_sequences = cluster_local_action_basis(
        state, local, bonds, 3, maximum_clusters=1
    )
    unrestricted, _ = cluster_local_action_basis(
        state, local, bonds, 3, maximum_clusters=None
    )
    krylov = global_krylov_basis(final, state, 3)
    assert one_cluster_sequences == connected_sequences
    np.testing.assert_allclose(
        connected @ connected.conj().T,
        one_cluster @ one_cluster.conj().T,
        atol=3e-13,
    )
    residual = krylov - unrestricted @ (unrestricted.conj().T @ krylov)
    assert np.linalg.norm(residual) < 2e-13


def test_disconnected_clusters_close_global_product_dimer_quench():
    n = 8
    initial = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=1.0)
    ).to_dense()
    final = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=-1.0)
    ).to_dense()
    _, state = ground_state(initial)
    bonds = tuple((bond, bond + 1) for bond in range(n - 1))
    local = _bond_operators(n)
    connected, _ = cluster_local_action_basis(
        state, local, bonds, 3, maximum_clusters=1
    )
    three_cluster, _ = cluster_local_action_basis(
        state, local, bonds, 3, maximum_clusters=3
    )
    times = np.linspace(0.0, 2.0, 41)
    exact = exact_time_evolution(final, state, times)
    connected_states = projected_time_evolution(final, state, connected, times)
    closed_states = projected_time_evolution(final, state, three_cluster, times)
    connected_error = np.max(1.0 - state_fidelities(exact, connected_states))
    closed_error = np.max(1.0 - state_fidelities(exact, closed_states))
    assert connected_error > 1e-2
    assert closed_error < 2e-13


def test_principal_action_control_preserves_initial_state_at_krylov_rank():
    n = 8
    initial = HeisenbergChain(
        n, 0, dimerized_chain_couplings(n, dimerization=0.7)
    ).to_dense()
    final = HeisenbergChain(n, 0).to_dense()
    _, state = ground_state(initial)
    bonds = tuple((bond, bond + 1) for bond in range(n - 1))
    global_basis = global_krylov_basis(final, state, 3)
    principal, _ = principal_local_action_basis(
        state,
        _bond_operators(n),
        bonds,
        3,
        global_basis.shape[1],
        maximum_clusters=1,
        initial_operator_indices=(3,),
    )
    assert principal.shape == global_basis.shape
    np.testing.assert_allclose(principal.conj().T @ principal, np.eye(4), atol=3e-13)
    np.testing.assert_allclose(
        principal @ (principal.conj().T @ state), state, atol=3e-13
    )
