from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from su2_nqs.models.fusion_gru import (
    FusionGRUConfig,
    init_fusion_gru,
    log_psi,
    sample,
)
from su2_nqs.operators.heisenberg_chain import HeisenbergChain
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.vmc.gradients import energy_gradient
from su2_nqs.vmc.multistate import exact_ky_fan_loss
from su2_nqs.vmc.sampled_multistate import (
    NumericalSubspaceRankError,
    fail_closed_generalized_ritz,
    sampled_subspace_moments,
)
from su2_nqs.vmc.sampled_multistate_force import (
    prepare_sampled_ky_fan_force_reference,
    sampled_ky_fan_horizontal_force,
)


jax.config.update("jax_enable_x64", True)


def _fusion_multistate_problem(rank: int = 2):
    config = FusionGRUConfig(6, 0, embedding_dim=3, hidden_dim=5)
    params = tuple(
        init_fusion_gru(key, config)
        for key in jax.random.split(jax.random.PRNGKey(91320), rank)
    )
    paths = jnp.asarray(enumerate_paths(config.n, config.target_q))
    hamiltonian = jnp.asarray(HeisenbergChain(config.n, 0).to_dense())

    def log_matrix(candidate):
        return jnp.stack(
            [
                jax.vmap(lambda path: log_psi(head, path, config))(paths)
                for head in candidate
            ],
            axis=1,
        )

    def mean_ky_fan(candidate):
        return exact_ky_fan_loss(
            candidate,
            paths,
            hamiltonian,
            lambda values, path: jnp.stack(
                [log_psi(head, path, config) for head in values]
            ),
        ) / rank

    return params, hamiltonian, log_matrix, mean_ky_fan


def test_exhaustive_horizontal_force_matches_ky_fan_autodiff() -> None:
    params, hamiltonian, log_matrix, mean_ky_fan = _fusion_multistate_problem()
    logs = np.array(log_matrix(params), copy=True)
    states = np.exp(logs)
    actions = np.asarray(hamiltonian) @ states
    proposal = np.mean(np.abs(states) ** 2, axis=1)
    reference = prepare_sampled_ky_fan_force_reference(
        logs,
        actions,
        proposal,
        observation_weights=proposal,
    )
    force = sampled_ky_fan_horizontal_force(params, reference, log_matrix)
    exact_value, exact_gradient = jax.value_and_grad(mean_ky_fan)(params)
    observed_flat, _ = ravel_pytree(force.gradient)
    expected_flat, _ = ravel_pytree(exact_gradient)

    np.testing.assert_allclose(force.energy, exact_value, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        observed_flat, expected_flat, rtol=3.0e-10, atol=3.0e-10
    )
    assert reference.maximum_horizontality_error < 2.0e-14
    assert abs(force.surrogate_value_at_reference) < 2.0e-14


def test_exact_normalized_diagonal_is_rejected_by_force_but_remains_ritz_available() -> None:
    params, hamiltonian, log_matrix, mean_ky_fan = _fusion_multistate_problem(
        rank=3
    )
    logs = np.asarray(log_matrix(params))
    states = np.exp(logs)
    actions = np.asarray(hamiltonian) @ states
    index = np.arange(len(states), dtype=np.float64)
    weights = 0.4 + np.exp(0.37 * np.sin(index + 0.2))
    proposal = weights / np.sum(weights)
    moments = sampled_subspace_moments(
        states,
        actions,
        proposal,
        observation_weights=proposal,
        exact_overlap_diagonal=np.ones(3),
    )
    ritz = fail_closed_generalized_ritz(moments)
    np.testing.assert_array_equal(np.diag(moments.overlap), np.ones(3))
    assert ritz.rank == 3

    with pytest.raises(ValueError, match="Ritz only|moments/Ritz only"):
        prepare_sampled_ky_fan_force_reference(
            logs,
            actions,
            proposal,
            observation_weights=proposal,
            exact_normalized_overlap_diagonal=True,
        )


def test_finite_sample_force_rejects_exact_diagonal_before_surrogate_construction() -> None:
    params, hamiltonian, log_matrix, _ = _fusion_multistate_problem(rank=3)
    full_logs = np.asarray(log_matrix(params))
    full_states = np.exp(full_logs)
    full_actions = np.asarray(hamiltonian) @ full_states
    full_proposal = np.mean(np.abs(full_states) ** 2, axis=1)
    selected = np.asarray([0, 1, 1, 2, 3, 4, 4, 0, 2], dtype=np.int64)
    logs = full_logs[selected]
    actions = full_actions[selected]
    proposal = full_proposal[selected]
    legacy = prepare_sampled_ky_fan_force_reference(
        logs,
        actions,
        proposal,
        exact_normalized_overlap_diagonal=False,
        maximum_condition_number=1.0e10,
    )
    assert legacy.moments.hamiltonian_hermiticity.relative_frobenius_error > 1.0e-4
    with pytest.raises(ValueError, match="phase-gauge validation"):
        prepare_sampled_ky_fan_force_reference(
            logs,
            actions,
            proposal,
            exact_normalized_overlap_diagonal=True,
            maximum_condition_number=1.0e10,
        )


def test_finite_sample_force_is_rootwise_phase_invariant_and_matches_exact_zero_difference() -> None:
    params, hamiltonian, log_matrix, _ = _fusion_multistate_problem(rank=3)
    full_logs = np.asarray(log_matrix(params))
    full_states = np.exp(full_logs)
    full_actions = np.asarray(hamiltonian) @ full_states
    full_proposal = np.mean(np.abs(full_states) ** 2, axis=1)
    selected = np.asarray([0, 1, 1, 2, 3, 4, 4, 0, 2], dtype=np.int64)
    logs = full_logs[selected]
    actions = full_actions[selected]
    proposal = full_proposal[selected]
    reference = prepare_sampled_ky_fan_force_reference(
        logs,
        actions,
        proposal,
        maximum_condition_number=1.0e10,
    )

    def phase_log_matrix(phases):
        return jnp.asarray(logs) + 1.0j * phases[None, :]

    phases = jnp.zeros(3, dtype=jnp.float64)
    force = sampled_ky_fan_horizontal_force(
        phases, reference, phase_log_matrix
    )
    np.testing.assert_allclose(force.gradient, 0.0, atol=2.0e-12)
    assert reference.maximum_horizontality_error < 2.0e-12

    def corrected_ritz_energy(candidate_phases: np.ndarray) -> float:
        rotations = np.exp(1.0j * candidate_phases)[None, :]
        moments = sampled_subspace_moments(
            np.exp(logs) * rotations,
            actions * rotations,
            proposal,
            exact_overlap_diagonal=np.ones(3),
        )
        return float(np.mean(fail_closed_generalized_ritz(
            moments, maximum_condition_number=1.0e10
        ).energies))

    direction = np.asarray([0.31, -0.47, 0.83], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    # The exact phase orbit is flat.  A milliradian step keeps division of the
    # generalized eigensolver's O(1e-13) roundoff below the audit tolerance.
    epsilon = 1.0e-3
    finite_difference = (
        corrected_ritz_energy(epsilon * direction)
        - corrected_ritz_energy(-epsilon * direction)
    ) / (2.0 * epsilon)
    assert abs(finite_difference) < 2.0e-10


def test_exhaustive_horizontal_force_matches_directional_finite_difference() -> None:
    params, hamiltonian, log_matrix, mean_ky_fan = _fusion_multistate_problem()
    logs = np.asarray(log_matrix(params))
    states = np.exp(logs)
    actions = np.asarray(hamiltonian) @ states
    proposal = np.mean(np.abs(states) ** 2, axis=1)
    reference = prepare_sampled_ky_fan_force_reference(
        logs,
        actions,
        proposal,
        observation_weights=proposal,
    )
    force = sampled_ky_fan_horizontal_force(params, reference, log_matrix)
    flat_params, unravel = ravel_pytree(params)
    flat_gradient, _ = ravel_pytree(force.gradient)
    direction = jax.random.normal(
        jax.random.PRNGKey(91321), flat_params.shape, dtype=jnp.float64
    )
    direction /= jnp.linalg.norm(direction)
    epsilon = 2.0e-5
    finite_difference = (
        mean_ky_fan(unravel(flat_params + epsilon * direction))
        - mean_ky_fan(unravel(flat_params - epsilon * direction))
    ) / (2.0 * epsilon)
    tangent = jnp.vdot(flat_gradient, direction).real

    np.testing.assert_allclose(
        finite_difference, tangent, rtol=3.0e-6, atol=3.0e-8
    )


def test_rank_one_force_equals_standard_centered_local_energy_vmc_gradient() -> None:
    config = FusionGRUConfig(6, 0, embedding_dim=3, hidden_dim=6)
    params = init_fusion_gru(jax.random.PRNGKey(91322), config)
    operator = HeisenbergChain(config.n, config.target_q)
    log_fn = lambda candidate, path: log_psi(candidate, path, config)
    samples = sample(params, jax.random.PRNGKey(91323), 96, config)

    basis = tuple(enumerate_paths(config.n, config.target_q))
    basis_array = jnp.asarray(basis)
    full_logs = np.asarray(jax.vmap(lambda path: log_fn(params, path))(basis_array))
    full_state = np.exp(full_logs)
    full_action = operator.to_dense() @ full_state
    lookup = {tuple(path): index for index, path in enumerate(basis)}
    indices = np.asarray(
        [lookup[tuple(map(int, path))] for path in np.asarray(samples)],
        dtype=np.int64,
    )
    reference_logs = full_logs[indices, None]
    reference_actions = full_action[indices, None]
    proposal = np.abs(full_state[indices]) ** 2

    reference = prepare_sampled_ky_fan_force_reference(
        reference_logs, reference_actions, proposal
    )

    def sampled_log_matrix(candidate):
        return jax.vmap(lambda path: log_fn(candidate, path))(samples)[:, None]

    force = sampled_ky_fan_horizontal_force(
        params, reference, sampled_log_matrix
    )
    standard_energy, standard_gradient = energy_gradient(
        params, samples, log_fn, operator.get_conn
    )
    observed_flat, _ = ravel_pytree(force.gradient)
    expected_flat, _ = ravel_pytree(standard_gradient)

    np.testing.assert_allclose(
        force.energy, standard_energy, rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(
        observed_flat, expected_flat, rtol=3.0e-11, atol=3.0e-11
    )
    assert reference.ritz.rank == 1
    assert reference.maximum_horizontality_error < 2.0e-14


def test_force_reference_rejects_duplicate_roots_before_vjp() -> None:
    params, hamiltonian, log_matrix, _ = _fusion_multistate_problem()
    logs = np.array(log_matrix(params), copy=True)
    logs[:, 1] = logs[:, 0]
    states = np.exp(logs)
    actions = np.asarray(hamiltonian) @ states
    proposal = np.mean(np.abs(states) ** 2, axis=1)

    with pytest.raises(NumericalSubspaceRankError, match=r"rank=1/2"):
        prepare_sampled_ky_fan_force_reference(
            logs,
            actions,
            proposal,
            observation_weights=proposal,
        )
