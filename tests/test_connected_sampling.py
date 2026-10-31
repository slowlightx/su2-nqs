import numpy as np
from scipy import sparse

from su2_nqs.benchmarks.connected_sampling import (
    connected_mixture_distribution,
    hamiltonian_connected_distribution,
    probability_importance_diagnostics,
)
from su2_nqs.benchmarks.triangular_locality import (
    dimer_state_in_fusion_basis,
    nearest_neighbor_dimer_cover,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix


def test_connected_distribution_uses_h_new_old_orientation_and_squared_edges():
    # From old state zero, the two outgoing weights are |1|^2 and |2|^2.
    hamiltonian = sparse.csr_matrix(
        np.asarray(
            [
                [3.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 7.0],
            ]
        )
    )
    connected = hamiltonian_connected_distribution(
        hamiltonian, np.asarray([1.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(connected, np.asarray([0.0, 0.2, 0.8]))


def test_isolated_columns_use_self_loops_and_preserve_normalization():
    hamiltonian = np.diag([1.0, 2.0])
    probabilities = np.asarray([0.3, 0.7])
    np.testing.assert_allclose(
        hamiltonian_connected_distribution(hamiltonian, probabilities),
        probabilities,
    )


def test_connected_mixture_repairs_one_hop_target_support():
    hamiltonian = sparse.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    base = np.asarray([1.0, 0.0])
    target = np.asarray([0.8, 0.2])
    invalid = probability_importance_diagnostics(base, target)
    assert not invalid["importance_sampling_valid"]
    np.testing.assert_allclose(
        invalid["target_weight_outside_proposal_support"], 0.2
    )

    proposal = connected_mixture_distribution(
        hamiltonian, base, connected_fraction=0.25
    )
    np.testing.assert_allclose(proposal, np.asarray([0.75, 0.25]))
    repaired = probability_importance_diagnostics(proposal, target)
    assert repaired["importance_sampling_valid"]
    assert repaired["effective_sample_size_fraction"] > 0.98


def test_importance_diagnostics_respect_explicit_probability_cutoff():
    proposal = np.asarray([1.0 - 1e-30, 1e-30])
    target = np.asarray([0.9, 0.1])
    full = probability_importance_diagnostics(proposal, target)
    cut = probability_importance_diagnostics(
        proposal, target, zero_tolerance=1e-20
    )
    assert full["importance_sampling_valid"]
    assert full["effective_sample_size_fraction"] < 1e-20
    assert not cut["importance_sampling_valid"]
    np.testing.assert_allclose(cut["target_weight_outside_proposal_support"], 0.1)


def test_triangular_dimer_residual_is_exactly_covered_by_connected_kernel():
    rows, columns = 2, 4
    n = rows * columns
    j1 = triangular_j1j2_bonds(rows, columns, j1=1.0, j2=0.0)
    j2 = triangular_j1j2_bonds(rows, columns, j1=0.0, j2=1.0)
    matching = nearest_neighbor_dimer_cover(n, j1)
    state, _ = dimer_state_in_fusion_basis(n, matching)
    state[np.abs(state) < 1e-12] = 0.0
    state /= np.linalg.norm(state)
    hamiltonian = heisenberg_graph_fusion_matrix(
        n, j1 + tuple((i, j, 0.2 * coupling) for i, j, coupling in j2), target_q=0
    )
    applied = np.asarray(hamiltonian @ state)
    energy = np.vdot(state, applied)
    residual = applied - energy * state
    residual_probabilities = np.abs(residual) ** 2
    residual_probabilities /= np.sum(residual_probabilities)
    base = np.abs(state) ** 2
    connected = hamiltonian_connected_distribution(hamiltonian, base)

    base_diagnostics = probability_importance_diagnostics(
        base, residual_probabilities
    )
    assert not base_diagnostics["importance_sampling_valid"]
    np.testing.assert_allclose(
        base_diagnostics["target_weight_outside_proposal_support"], 1.0
    )
    np.testing.assert_allclose(connected, residual_probabilities, atol=2e-14)
    connected_diagnostics = probability_importance_diagnostics(
        connected, residual_probabilities
    )
    assert connected_diagnostics["importance_sampling_valid"]
    np.testing.assert_allclose(
        connected_diagnostics["effective_sample_size_fraction"], 1.0, atol=2e-14
    )
