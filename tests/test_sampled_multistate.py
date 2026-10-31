from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.vmc.sampled_multistate import (
    EXACT_OVERLAP_DIAGONAL,
    NumericalSubspaceRankError,
    SAMPLED_OVERLAP_DIAGONAL,
    fail_closed_generalized_ritz,
    sampled_physical_residuals,
    sampled_subspace_moments,
)


def _complex_hermitian_problem(
    *, dimension: int = 11, state_count: int = 3, seed: int = 91301
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw_hamiltonian = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    hamiltonian = 0.5 * (raw_hamiltonian + raw_hamiltonian.conj().T)
    states = rng.normal(size=(dimension, state_count)) + 1j * rng.normal(
        size=(dimension, state_count)
    )
    states /= np.linalg.norm(states, axis=0, keepdims=True)
    return hamiltonian, states


def _shared_mixture(states: np.ndarray) -> np.ndarray:
    proposal = np.mean(np.abs(states) ** 2, axis=1)
    np.testing.assert_allclose(np.sum(proposal), 1.0, atol=3.0e-15)
    assert np.all(proposal > 0.0)
    return proposal


def test_exhaustive_shared_mixture_reconstructs_s_h_and_k() -> None:
    hamiltonian, states = _complex_hermitian_problem()
    actions = hamiltonian @ states
    proposal = _shared_mixture(states)
    moments = sampled_subspace_moments(
        states,
        actions,
        proposal,
        observation_weights=proposal,
    )

    np.testing.assert_allclose(
        moments.overlap, states.conj().T @ states, rtol=2.0e-14, atol=2.0e-14
    )
    np.testing.assert_allclose(
        moments.hamiltonian,
        states.conj().T @ hamiltonian @ states,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        moments.action_gram,
        actions.conj().T @ actions,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    assert moments.sample_count == len(states)
    assert moments.state_count == states.shape[1]
    assert moments.hamiltonian_hermiticity.maximum_absolute_error < 2.0e-15


def test_generalized_ritz_and_direct_residual_match_dense_vectors() -> None:
    hamiltonian, states = _complex_hermitian_problem(seed=91302)
    actions = hamiltonian @ states
    proposal = _shared_mixture(states)
    moments = sampled_subspace_moments(
        states, actions, proposal, observation_weights=proposal
    )
    ritz = fail_closed_generalized_ritz(moments)
    residual = sampled_physical_residuals(
        states,
        actions,
        proposal,
        ritz,
        observation_weights=proposal,
    )

    represented = states @ ritz.coefficients
    represented_actions = hamiltonian @ represented
    dense_residual = represented_actions - represented * ritz.energies[None, :]
    expected_gram = dense_residual.conj().T @ dense_residual
    np.testing.assert_allclose(
        ritz.coefficients.conj().T @ moments.overlap @ ritz.coefficients,
        np.eye(ritz.rank),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        residual.residual_gram, expected_gram, rtol=4.0e-14, atol=4.0e-14
    )
    np.testing.assert_allclose(
        residual.residual_norms,
        np.linalg.norm(dense_residual, axis=0),
        rtol=4.0e-14,
        atol=4.0e-14,
    )
    np.testing.assert_allclose(
        residual.spectral_intervals[:, 0],
        ritz.energies - residual.residual_norms,
    )
    np.testing.assert_allclose(
        residual.spectral_intervals[:, 1],
        ritz.energies + residual.residual_norms,
    )
    assert residual.maximum_projected_residual < 3.0e-14
    assert residual.residual_gram_minimum_eigenvalue > -3.0e-14


def test_rank_one_reduces_to_energy_action_norm_and_physical_variance() -> None:
    hamiltonian, states = _complex_hermitian_problem(
        dimension=9, state_count=1, seed=91303
    )
    state = states[:, 0]
    action = hamiltonian @ state
    proposal = np.abs(state) ** 2
    moments = sampled_subspace_moments(
        states,
        action[:, None],
        proposal,
        observation_weights=proposal,
    )
    ritz = fail_closed_generalized_ritz(moments)
    residual = sampled_physical_residuals(
        states,
        action[:, None],
        proposal,
        ritz,
        observation_weights=proposal,
    )

    energy = float(np.vdot(state, action).real)
    expected_residual_squared = float(
        np.vdot(action - energy * state, action - energy * state).real
    )
    np.testing.assert_allclose(moments.overlap, [[1.0]], atol=2.0e-14)
    np.testing.assert_allclose(moments.hamiltonian, [[energy]], atol=2.0e-14)
    np.testing.assert_allclose(
        moments.action_gram, [[np.vdot(action, action)]], atol=3.0e-14
    )
    np.testing.assert_allclose(ritz.energies, [energy], atol=2.0e-14)
    np.testing.assert_allclose(
        residual.residual_norms_squared,
        [expected_residual_squared],
        rtol=3.0e-14,
        atol=3.0e-14,
    )


def test_duplicate_roots_are_rejected_instead_of_truncated() -> None:
    hamiltonian, states = _complex_hermitian_problem(
        dimension=8, state_count=2, seed=91304
    )
    states[:, 1] = states[:, 0]
    actions = hamiltonian @ states
    proposal = _shared_mixture(states)
    moments = sampled_subspace_moments(
        states, actions, proposal, observation_weights=proposal
    )

    with pytest.raises(NumericalSubspaceRankError, match=r"rank=1/2"):
        fail_closed_generalized_ritz(moments)


def test_finite_sample_hamiltonian_keeps_raw_antihermitian_diagnostic() -> None:
    hamiltonian, states = _complex_hermitian_problem(
        dimension=13, state_count=3, seed=91305
    )
    actions = hamiltonian @ states
    proposal = _shared_mixture(states)
    # A deliberately incomplete empirical sample.  S and K remain Gram
    # matrices, while <Phi|H Phi> need not be Hermitian before averaging over
    # the full proposal.
    selected = np.asarray([0, 2, 3, 7, 11])
    moments = sampled_subspace_moments(
        states[selected], actions[selected], proposal[selected]
    )

    assert moments.hamiltonian_hermiticity.maximum_absolute_error > 1.0e-3
    assert moments.hamiltonian_hermiticity.relative_frobenius_error > 1.0e-3
    np.testing.assert_allclose(
        moments.hamiltonian,
        moments.hamiltonian.conj().T,
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(moments.hamiltonian_raw, moments.hamiltonian)
    assert moments.overlap_hermiticity.maximum_absolute_error < 2.0e-15
    assert moments.action_gram_hermiticity.maximum_absolute_error < 2.0e-14


def test_exact_overlap_diagonal_changes_only_sampled_s_diagonal() -> None:
    hamiltonian, states = _complex_hermitian_problem(
        dimension=13, state_count=3, seed=91307
    )
    actions = hamiltonian @ states
    proposal = _shared_mixture(states)
    selected = np.asarray([0, 1, 2, 4, 5, 7, 8, 10, 12])
    legacy = sampled_subspace_moments(
        states[selected], actions[selected], proposal[selected]
    )
    corrected = sampled_subspace_moments(
        states[selected],
        actions[selected],
        proposal[selected],
        exact_overlap_diagonal=np.ones(3),
    )

    assert legacy.overlap_diagonal_policy == SAMPLED_OVERLAP_DIAGONAL
    assert legacy.exact_overlap_diagonal is None
    assert corrected.overlap_diagonal_policy == EXACT_OVERLAP_DIAGONAL
    np.testing.assert_array_equal(corrected.exact_overlap_diagonal, np.ones(3))
    np.testing.assert_allclose(
        corrected.sampled_overlap_diagonal,
        np.real(np.diag(legacy.overlap)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(np.diag(corrected.overlap), np.ones(3))
    off_diagonal = ~np.eye(3, dtype=bool)
    np.testing.assert_array_equal(
        corrected.overlap[off_diagonal], legacy.overlap[off_diagonal]
    )
    np.testing.assert_array_equal(corrected.hamiltonian_raw, legacy.hamiltonian_raw)
    np.testing.assert_array_equal(corrected.hamiltonian, legacy.hamiltonian)
    np.testing.assert_array_equal(corrected.action_gram, legacy.action_gram)
    assert corrected.hamiltonian_hermiticity == legacy.hamiltonian_hermiticity
    assert corrected.overlap_hermiticity == legacy.overlap_hermiticity
    assert corrected.action_gram_hermiticity == legacy.action_gram_hermiticity
    assert corrected.hamiltonian_hermiticity.maximum_absolute_error > 1.0e-3

    with pytest.raises(ValueError, match="exact_overlap_diagonal"):
        sampled_subspace_moments(
            states[selected],
            actions[selected],
            proposal[selected],
            exact_overlap_diagonal=np.asarray([1.0, 0.0, 1.0]),
        )


def test_registered_condition_gate_also_fails_closed() -> None:
    hamiltonian, states = _complex_hermitian_problem(
        dimension=9, state_count=2, seed=91306
    )
    states[:, 1] = states[:, 0] + 1.0e-4 * states[:, 1]
    states[:, 1] /= np.linalg.norm(states[:, 1])
    actions = hamiltonian @ states
    proposal = _shared_mixture(states)
    moments = sampled_subspace_moments(
        states, actions, proposal, observation_weights=proposal
    )

    assert np.linalg.cond(moments.overlap) > 1.0e6
    with pytest.raises(
        NumericalSubspaceRankError, match="condition-number gate"
    ):
        fail_closed_generalized_ritz(
            moments,
            relative_rank_tolerance=1.0e-12,
            maximum_condition_number=1.0e6,
        )
