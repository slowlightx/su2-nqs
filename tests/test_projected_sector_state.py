from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from su2_nqs.benchmarks.common_basis_gamma_a1 import (
    triangular_hnf_gamma_a1_circuit,
)
from su2_nqs.benchmarks.projected_sector_state import (
    ExactSmallProjectedSectorState,
    validated_full_support_proposal,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.operators.permutation import (
    heisenberg_graph_fusion_matrix,
    site_permutation_fusion_matrix,
)
from su2_nqs.vmc.sampled_multistate import (
    fail_closed_generalized_ritz,
    sampled_physical_residuals,
    sampled_subspace_moments,
)


@dataclass(frozen=True)
class _ProjectedProblem:
    state: ExactSmallProjectedSectorState
    hamiltonian: np.ndarray
    proposal: np.ndarray


@pytest.fixture(scope="module")
def projected_problem() -> _ProjectedProblem:
    """Build three independent q=0, Gamma.A1 vectors on the N=16 torus."""

    supercell = ((4, 0), (0, 4))
    circuit = triangular_hnf_gamma_a1_circuit(supercell, target_q=0)
    hamiltonian = heisenberg_graph_fusion_matrix(
        circuit.n_sites,
        triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.125),
        target_q=0,
    ).toarray()
    rng = np.random.default_rng(2026091304)
    base = rng.normal(size=(circuit.dimension, 3)) + 1.0j * rng.normal(
        size=(circuit.dimension, 3)
    )
    state = ExactSmallProjectedSectorState.from_base_amplitudes(
        circuit,
        hamiltonian,
        base,
        maximum_dimension=2000,
    )

    # This deliberately external proposal is smooth, strictly positive, and
    # unrelated to any of the three projected Born laws.
    index = np.arange(circuit.dimension, dtype=np.float64)
    weights = 0.35 + np.exp(
        0.72 * np.sin(0.173 * index) + 0.31 * np.cos(0.071 * index)
    )
    proposal = weights / np.sum(weights)
    return _ProjectedProblem(state=state, hamiltonian=hamiltonian, proposal=proposal)


def test_projected_state_and_hamiltonian_action_are_strict_gamma_a1(
    projected_problem: _ProjectedProblem,
) -> None:
    state = projected_problem.state
    assert state.dimension == 1430
    assert state.state_count == 3
    assert state.projector_residual < 1.0e-12
    assert state.action_projector_residual < 1.0e-12
    assert state.hamiltonian_hermiticity_error < 1.0e-14
    np.testing.assert_allclose(
        np.diag(state.amplitudes.conj().T @ state.amplitudes),
        np.ones(3),
        atol=3.0e-13,
    )

    # Check a nonidentity generator from every factor directly in the same
    # left-associated fusion basis.  These are physical symmetry actions, not
    # support-one changes between different tree frames.
    for factor in state.circuit.factors:
        if factor.order == 1:
            continue
        generator = site_permutation_fusion_matrix(
            factor.images[1], target_q=state.circuit.target_q
        )
        transformed_states = generator @ state.amplitudes
        transformed_actions = generator @ state.hamiltonian_actions
        assert (
            np.linalg.norm(transformed_states - state.amplitudes)
            / np.linalg.norm(state.amplitudes)
            < 2.0e-12
        )
        assert (
            np.linalg.norm(transformed_actions - state.hamiltonian_actions)
            / np.linalg.norm(state.hamiltonian_actions)
            < 2.0e-12
        )


def test_non_born_full_support_proposal_recovers_dense_three_root_ritz(
    projected_problem: _ProjectedProblem,
) -> None:
    state = projected_problem.state
    hamiltonian = projected_problem.hamiltonian
    proposal = projected_problem.proposal
    born_mixture = state.born_mixture_probabilities
    assert np.all(proposal > 0.0)
    assert 0.5 * np.sum(np.abs(proposal - born_mixture)) > 0.1

    rows = state.exhaustive_proposal_rows(proposal)
    moments = sampled_subspace_moments(
        rows.amplitudes,
        rows.hamiltonian_actions,
        rows.proposal_probabilities,
        observation_weights=rows.observation_weights,
    )
    ritz = fail_closed_generalized_ritz(
        moments,
        relative_rank_tolerance=1.0e-12,
        maximum_condition_number=1.0e8,
    )
    residuals = sampled_physical_residuals(
        rows.amplitudes,
        rows.hamiltonian_actions,
        rows.proposal_probabilities,
        ritz,
        observation_weights=rows.observation_weights,
    )

    exact_overlap = state.amplitudes.conj().T @ state.amplitudes
    exact_hamiltonian = state.amplitudes.conj().T @ state.hamiltonian_actions
    exact_action_gram = (
        state.hamiltonian_actions.conj().T @ state.hamiltonian_actions
    )
    np.testing.assert_allclose(moments.overlap, exact_overlap, atol=2.0e-12)
    np.testing.assert_allclose(
        moments.hamiltonian_raw, exact_hamiltonian, atol=3.0e-12
    )
    np.testing.assert_allclose(
        moments.action_gram, exact_action_gram, atol=8.0e-12
    )

    # An independent dense reference works in an orthonormal basis for the
    # same three-dimensional projected span; it does not reuse generalized
    # Ritz whitening or importance weights.
    orthonormal_span, _ = np.linalg.qr(state.amplitudes, mode="reduced")
    dense_effective = orthonormal_span.conj().T @ hamiltonian @ orthonormal_span
    dense_energies, dense_coefficients = np.linalg.eigh(
        0.5 * (dense_effective + dense_effective.conj().T)
    )
    dense_roots = orthonormal_span @ dense_coefficients
    dense_residual_rows = (
        hamiltonian @ dense_roots - dense_roots * dense_energies[None, :]
    )
    dense_residual_norms = np.linalg.norm(dense_residual_rows, axis=0)
    np.testing.assert_allclose(ritz.energies, dense_energies, atol=3.0e-12)
    np.testing.assert_allclose(
        residuals.residual_norms, dense_residual_norms, atol=8.0e-12
    )
    np.testing.assert_allclose(
        residuals.spectral_intervals,
        np.stack(
            (
                dense_energies - dense_residual_norms,
                dense_energies + dense_residual_norms,
            ),
            axis=1,
        ),
        atol=1.0e-11,
    )
    assert ritz.rank == 3
    assert ritz.orthonormality_error < 2.0e-12
    assert residuals.maximum_projected_residual < 2.0e-11


def test_proposal_sampling_is_external_and_full_support_is_fail_closed(
    projected_problem: _ProjectedProblem,
) -> None:
    state = projected_problem.state
    proposal = projected_problem.proposal
    rng = np.random.default_rng(2026091305)
    rows = state.sample_proposal_rows(proposal, rng, 4096)
    assert rows.sample_count == 4096
    assert rows.amplitudes.shape == (4096, 3)
    assert rows.hamiltonian_actions.shape == (4096, 3)
    assert rows.observation_weights is None
    assert not rows.is_exhaustive
    np.testing.assert_array_equal(
        rows.proposal_probabilities, proposal[rows.basis_indices]
    )

    missing_support = proposal.copy()
    missing_support[0] = 0.0
    missing_support /= np.sum(missing_support)
    with pytest.raises(ValueError, match="strictly positive"):
        validated_full_support_proposal(missing_support, state.dimension)

    unnormalized = proposal * 0.9
    with pytest.raises(ValueError, match="sum to one"):
        state.exhaustive_proposal_rows(unnormalized)

    with pytest.raises(ValueError, match="exact-small projected-state cap"):
        ExactSmallProjectedSectorState.from_base_amplitudes(
            state.circuit,
            np.empty((0, 0)),
            np.empty((0, 0)),
            maximum_dimension=100,
        )
