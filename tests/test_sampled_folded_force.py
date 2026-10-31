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
)
from su2_nqs.operators.heisenberg_chain import HeisenbergChain
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.vmc.sampled_folded_force import (
    prepare_sampled_folded_force_reference,
    sampled_folded_spectrum_force,
)


jax.config.update("jax_enable_x64", True)


def _problem():
    config = FusionGRUConfig(6, 0, embedding_dim=3, hidden_dim=5)
    params = init_fusion_gru(jax.random.PRNGKey(91601), config)
    paths = jnp.asarray(enumerate_paths(config.n, config.target_q))
    hamiltonian = jnp.asarray(HeisenbergChain(config.n, 0).to_dense())
    shift = -1.37

    def state(candidate):
        logs = jax.vmap(lambda path: log_psi(candidate, path, config))(paths)
        return jnp.exp(logs)

    def residual(candidate):
        wavefunction = state(candidate)
        return hamiltonian @ wavefunction - shift * wavefunction

    def exact_folded(candidate):
        action = residual(candidate)
        return jnp.vdot(action, action).real

    return params, state, residual, exact_folded, shift


def test_exhaustive_folded_force_matches_direct_residual_autodiff() -> None:
    params, state, residual, exact_folded, shift = _problem()
    wavefunction = np.asarray(state(params))
    reference_residual = np.asarray(residual(params))
    proposal = np.abs(wavefunction) ** 2
    assert abs(float(np.sum(proposal)) - 1.0) < 2.0e-14
    reference = prepare_sampled_folded_force_reference(
        reference_residual,
        proposal,
        shift=shift,
        observation_weights=proposal,
    )
    force = sampled_folded_spectrum_force(params, reference, residual)
    exact_value, exact_gradient = jax.value_and_grad(exact_folded)(params)
    observed_flat, _ = ravel_pytree(force.gradient)
    expected_flat, _ = ravel_pytree(exact_gradient)

    np.testing.assert_allclose(
        force.folded_expectation, exact_value, rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(
        force.surrogate_value_at_reference,
        exact_value,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        observed_flat, expected_flat, rtol=3.0e-10, atol=3.0e-10
    )
    assert force.maximum_reference_action_error == 0.0


def test_exhaustive_folded_force_matches_directional_finite_difference() -> None:
    params, state, residual, exact_folded, shift = _problem()
    wavefunction = np.asarray(state(params))
    proposal = np.abs(wavefunction) ** 2
    reference = prepare_sampled_folded_force_reference(
        np.asarray(residual(params)),
        proposal,
        shift=shift,
        observation_weights=proposal,
    )
    force = sampled_folded_spectrum_force(params, reference, residual)
    flat_params, unravel = ravel_pytree(params)
    flat_gradient, _ = ravel_pytree(force.gradient)
    direction = jax.random.normal(
        jax.random.PRNGKey(91602), flat_params.shape, dtype=jnp.float64
    )
    direction /= jnp.linalg.norm(direction)
    epsilon = 2.0e-5
    finite_difference = (
        exact_folded(unravel(flat_params + epsilon * direction))
        - exact_folded(unravel(flat_params - epsilon * direction))
    ) / (2.0 * epsilon)
    tangent = jnp.vdot(flat_gradient, direction).real

    np.testing.assert_allclose(
        finite_difference, tangent, rtol=3.0e-6, atol=3.0e-8
    )


def test_folded_force_fails_closed_on_reference_drift_and_invalid_proposal() -> None:
    params, state, residual, _, shift = _problem()
    wavefunction = np.asarray(state(params))
    proposal = np.abs(wavefunction) ** 2
    reference_residual = np.asarray(residual(params))
    invalid_proposal = np.array(proposal, copy=True)
    invalid_proposal[0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        prepare_sampled_folded_force_reference(
            reference_residual,
            invalid_proposal,
            shift=shift,
        )
    reference = prepare_sampled_folded_force_reference(
        reference_residual,
        proposal,
        shift=shift,
        observation_weights=proposal,
    )

    def drifted(candidate):
        return residual(candidate) + jnp.asarray(1.0e-5 + 0.0j)

    with pytest.raises(ValueError, match="does not reproduce"):
        sampled_folded_spectrum_force(params, reference, drifted)
