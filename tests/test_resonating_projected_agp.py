from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from su2_nqs.benchmarks.projected_singlet_agp import permutation_group_closure
from su2_nqs.models.projected_singlet_agp import (
    ProjectedSingletAGPConfig,
    init_projected_singlet_agp_parameters,
)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.multistate import generalized_ritz_analysis
from su2_nqs.vmc.projected_agp import ProjectedAGPHeisenbergConfig
from su2_nqs.vmc.resonating_projected_agp import (
    ResonatingProjectedAGPAction,
    require_group_graph_compatibility,
    resonating_projected_agp_action,
    solve_three_root_resonating_subspace,
)
from su2_nqs.vmc.sampled_multistate import NumericalSubspaceRankError


def _dihedral_ring_group(n_sites: int) -> tuple[tuple[int, ...], ...]:
    translation = tuple((site + 1) % n_sites for site in range(n_sites))
    reflection = tuple((-site) % n_sites for site in range(n_sites))
    return permutation_group_closure((translation, reflection))


def _ring_bonds(n_sites: int, j2: float) -> tuple[tuple[int, int, float], ...]:
    nearest = tuple(
        (site, (site + 1) % n_sites, 1.0) for site in range(n_sites)
    )
    next_nearest = tuple(
        (site, (site + 2) % n_sites, j2) for site in range(n_sites)
    )
    return nearest + next_nearest


def _n8_problem(seed: int = 2026091370):
    n_sites = 8
    carrier = ProjectedSingletAGPConfig(
        n_sites, _dihedral_ring_group(n_sites), parameter_mode="real"
    )
    graph = ProjectedAGPHeisenbergConfig(n_sites, _ring_bonds(n_sites, 0.3))
    keys = jax.random.split(jax.random.key(seed), 3)
    bank = {
        "pairing_real": np.stack(
            [
                np.asarray(
                    init_projected_singlet_agp_parameters(key, carrier)[
                        "pairing_real"
                    ]
                )
                for key in keys
            ]
        )
    }
    configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    action = resonating_projected_agp_action(
        bank,
        configurations,
        carrier,
        graph,
        amplitude_microbatch_size=23,
    )
    return carrier, graph, bank, configurations, action


def _aligned_columns(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    aligned = np.array(candidate, dtype=np.complex128, copy=True)
    for column in range(reference.shape[1]):
        overlap = np.vdot(reference[:, column], aligned[:, column])
        if abs(overlap) > 0.0:
            aligned[:, column] *= np.conj(overlap / abs(overlap))
    return aligned


def test_stacked_action_and_three_root_moments_match_dense_reference() -> None:
    _, graph, _, configurations, action = _n8_problem()
    amplitudes = np.asarray(action.amplitudes)
    hamiltonian_actions = np.asarray(action.hamiltonian_actions)
    dense = SzHeisenbergGraph(
        graph.n_sites, graph.bonds, n_up=graph.n_sites // 2
    ).to_dense()
    np.testing.assert_allclose(
        hamiltonian_actions, dense @ amplitudes, atol=7.0e-13
    )

    proposal = np.full(len(configurations), 1.0 / len(configurations))
    result = solve_three_root_resonating_subspace(
        action, proposal, observation_weights=proposal
    )
    reference = generalized_ritz_analysis(amplitudes, dense)
    assert reference["rank"] == 3
    np.testing.assert_allclose(
        result.energies, reference["ritz_values"], atol=2.0e-13
    )
    np.testing.assert_allclose(
        result.residual_norms, reference["ritz_residuals"], atol=7.0e-13
    )

    scaled_amplitudes = amplitudes / result.column_rescales[None, :]
    scaled_actions = hamiltonian_actions / result.column_rescales[None, :]
    np.testing.assert_allclose(
        result.overlap,
        scaled_amplitudes.conj().T @ scaled_amplitudes,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.hamiltonian_raw,
        scaled_amplitudes.conj().T @ scaled_actions,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.action_gram,
        scaled_actions.conj().T @ scaled_actions,
        atol=2.0e-14,
    )
    assert result.overlap_hermiticity.relative_frobenius_error < 3.0e-15
    assert result.hamiltonian_hermiticity.relative_frobenius_error < 3.0e-14
    assert result.action_gram_hermiticity.relative_frobenius_error < 3.0e-15
    assert result.orthonormality_error < 3.0e-14
    assert np.max(np.abs(result.projected_residual_matrix)) < 3.0e-13


def test_three_root_solution_is_covariant_under_bank_basis_and_pairing_gauges() -> None:
    carrier, graph, bank, configurations, action = _n8_problem(2026091371)
    proposal = np.full(len(configurations), 1.0 / len(configurations))
    reference = solve_three_root_resonating_subspace(
        action, proposal, observation_weights=proposal
    )

    mixing = np.asarray(
        [[1.2, -0.3, 0.2], [0.1, 0.8, -0.25], [-0.2, 0.15, 1.1]],
        dtype=np.float64,
    )
    mixed_amplitudes = np.asarray(action.amplitudes) @ mixing
    mixed_actions = np.asarray(action.hamiltonian_actions) @ mixing
    mixed = ResonatingProjectedAGPAction(
        amplitudes=mixed_amplitudes,
        hamiltonian_actions=mixed_actions,
        center_exact_nodes=mixed_amplitudes == 0.0,
        minimum_connected_cancellation_ratio=np.ones_like(
            mixed_amplitudes, dtype=np.float64
        ),
    )
    transformed = solve_three_root_resonating_subspace(
        mixed, proposal, observation_weights=proposal
    )
    np.testing.assert_allclose(transformed.energies, reference.energies, atol=3.0e-13)
    np.testing.assert_allclose(
        transformed.residual_norms, reference.residual_norms, atol=7.0e-13
    )
    np.testing.assert_allclose(
        _aligned_columns(transformed.root_amplitudes, reference.root_amplitudes),
        reference.root_amplitudes,
        atol=2.0e-12,
    )
    transformed_in_original_coordinates = mixing @ transformed.coefficients
    np.testing.assert_allclose(
        _aligned_columns(
            transformed_in_original_coordinates, reference.coefficients
        ),
        reference.coefficients,
        atol=3.0e-12,
    )
    physical_from_original = np.asarray(action.amplitudes) @ reference.coefficients
    np.testing.assert_allclose(
        physical_from_original, reference.root_amplitudes, atol=3.0e-14
    )

    # Each projected AGP is homogeneous of degree N/2 in its pairing.  These
    # independent scales are pure basis gauges and must not move the Ritz data.
    scale = np.asarray([0.35, 1.7, -1.2])
    gauged_bank = {"pairing_real": bank["pairing_real"] * scale[:, None]}
    gauged_action = resonating_projected_agp_action(
        gauged_bank,
        configurations,
        carrier,
        graph,
        amplitude_microbatch_size=31,
    )
    gauged = solve_three_root_resonating_subspace(
        gauged_action, proposal, observation_weights=proposal
    )
    np.testing.assert_allclose(gauged.energies, reference.energies, atol=4.0e-13)
    np.testing.assert_allclose(
        gauged.residual_norms, reference.residual_norms, atol=8.0e-13
    )
    np.testing.assert_allclose(
        _aligned_columns(gauged.root_amplitudes, reference.root_amplitudes),
        reference.root_amplitudes,
        atol=3.0e-12,
    )


def test_nonuniform_exact_quadrature_and_true_nodes_are_admissible() -> None:
    _, _, _, configurations, action = _n8_problem(2026091372)
    rng = np.random.default_rng(2026091373)
    proposal = rng.random(len(configurations)) + 0.05
    proposal /= np.sum(proposal)
    result = solve_three_root_resonating_subspace(
        action, proposal, observation_weights=proposal
    )
    uniform = np.full(len(configurations), 1.0 / len(configurations))
    reference = solve_three_root_resonating_subspace(
        action, uniform, observation_weights=uniform
    )
    np.testing.assert_allclose(result.energies, reference.energies, atol=2.0e-13)
    np.testing.assert_allclose(
        result.residual_norms, reference.residual_norms, atol=5.0e-13
    )

    # Direct rows can contain nodes; no row-wise division is present.
    modified_amplitudes = np.array(action.amplitudes)
    modified_actions = np.array(action.hamiltonian_actions)
    modified_amplitudes[0, 0] = 0.0
    modified_actions[0, 0] = 0.0
    node_action = ResonatingProjectedAGPAction(
        amplitudes=modified_amplitudes,
        hamiltonian_actions=modified_actions,
        center_exact_nodes=modified_amplitudes == 0.0,
        minimum_connected_cancellation_ratio=np.ones_like(
            modified_amplitudes, dtype=np.float64
        ),
    )
    observed = solve_three_root_resonating_subspace(
        node_action, uniform, observation_weights=uniform
    )
    assert np.all(np.isfinite(observed.energies))
    assert np.all(np.isfinite(observed.residual_norms))


def test_three_root_rank_and_proposal_contracts_fail_closed() -> None:
    _, _, _, configurations, action = _n8_problem(2026091374)
    proposal = np.full(len(configurations), 1.0 / len(configurations))
    two_column = ResonatingProjectedAGPAction(
        *(np.asarray(field)[:, :2] for field in action)
    )
    with pytest.raises(NumericalSubspaceRankError, match="at least three"):
        solve_three_root_resonating_subspace(two_column, proposal)

    collapsed = ResonatingProjectedAGPAction(
        amplitudes=np.column_stack(
            (action.amplitudes[:, 0], action.amplitudes[:, 0], action.amplitudes[:, 2])
        ),
        hamiltonian_actions=np.column_stack(
            (
                action.hamiltonian_actions[:, 0],
                action.hamiltonian_actions[:, 0],
                action.hamiltonian_actions[:, 2],
            )
        ),
        center_exact_nodes=np.zeros((len(configurations), 3), dtype=bool),
        minimum_connected_cancellation_ratio=np.ones(
            (len(configurations), 3), dtype=np.float64
        ),
    )
    with pytest.raises(NumericalSubspaceRankError, match="not full rank"):
        solve_three_root_resonating_subspace(collapsed, proposal)
    invalid_proposal = proposal.copy()
    invalid_proposal[0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        solve_three_root_resonating_subspace(action, invalid_proposal)


def test_projected_group_must_be_a_symmetry_of_the_weighted_graph() -> None:
    carrier, graph, _, _, _ = _n8_problem(2026091375)
    require_group_graph_compatibility(carrier, graph)
    broken = ProjectedAGPHeisenbergConfig(
        graph.n_sites,
        graph.bonds[:-1] + ((0, 2, 0.37),),
    )
    with pytest.raises(ValueError, match="not an automorphism"):
        require_group_graph_compatibility(carrier, broken)
