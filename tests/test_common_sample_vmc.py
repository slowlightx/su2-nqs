from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.vmc.common_sample_vmc import (
    common_sample_vmc_value_and_grad,
    paired_common_sample_energy_difference,
)


jax.config.update("jax_enable_x64", True)


def test_common_sample_gradient_matches_complete_basis_rayleigh_derivative() -> None:
    hamiltonian = jnp.asarray(
        ((0.3, -0.7, 0.2), (-0.7, -0.1, 0.4), (0.2, 0.4, 0.8)),
        dtype=jnp.float64,
    )
    features = jnp.asarray(
        ((1.0, 0.2), (-0.3, 0.9), (0.7, -0.4)), dtype=jnp.float64
    )
    samples = jnp.arange(3)
    params = {"theta": jnp.asarray((0.6, -0.2), dtype=jnp.float64)}
    # One copy of every basis state is a sample from the uniform reference.
    # A nonuniform reference would require repeated/configuration-weighted
    # rows to represent its empirical sampling distribution.
    reference = jnp.ones((3,), dtype=jnp.complex128)

    def centers(candidate, indices):
        return (features @ candidate["theta"])[indices].astype(jnp.complex128)

    def hpsi(candidate, indices):
        state = features @ candidate["theta"]
        return state[indices].astype(jnp.complex128), (
            hamiltonian @ state
        )[indices].astype(jnp.complex128)

    result = common_sample_vmc_value_and_grad(
        params,
        samples,
        reference,
        hpsi_fn=hpsi,
        center_fn=centers,
    )

    def importance_energy(candidate):
        state = features @ candidate["theta"]
        weights = jnp.abs(state / reference) ** 2
        local = (hamiltonian @ state) / state
        return jnp.sum(weights * local) / jnp.sum(weights)

    expected = jax.grad(importance_energy)(params)
    np.testing.assert_allclose(
        result.gradient["theta"], expected["theta"], atol=2.0e-12, rtol=2.0e-12
    )
    assert 0.0 < result.ess_fraction <= 1.0


def test_common_sample_gradient_is_zero_at_an_eigenstate() -> None:
    params = {"theta": jnp.asarray((1.0, 0.0), dtype=jnp.float64)}
    samples = jnp.arange(2)
    reference = jnp.ones((2,), dtype=jnp.complex128)

    def centers(candidate, indices):
        state = candidate["theta"]
        return state[indices].astype(jnp.complex128)

    def hpsi(candidate, indices):
        state = candidate["theta"]
        return state[indices].astype(jnp.complex128), state[indices].astype(
            jnp.complex128
        )

    # A zero component is forbidden by the estimator, so use the nonzero
    # eigenvector (1,1) of the identity instead.
    params = {"theta": jnp.asarray((1.0, 1.0), dtype=jnp.float64)}
    result = common_sample_vmc_value_and_grad(
        params,
        samples,
        reference,
        hpsi_fn=hpsi,
        center_fn=centers,
    )
    np.testing.assert_allclose(result.gradient["theta"], 0.0, atol=1.0e-14)


def test_gradient_microbatching_matches_the_full_common_sample_objective() -> None:
    hamiltonian = jnp.asarray(
        (
            (0.1, -0.3, 0.0, 0.2),
            (-0.3, 0.4, -0.2, 0.0),
            (0.0, -0.2, -0.5, 0.6),
            (0.2, 0.0, 0.6, 0.7),
        ),
        dtype=jnp.float64,
    )
    features = jnp.asarray(
        ((1.0, 0.2), (-0.3, 0.9), (0.7, -0.4), (0.5, 0.8)),
        dtype=jnp.float64,
    )
    samples = jnp.arange(4)
    params = {"theta": jnp.asarray((0.6, -0.2), dtype=jnp.float64)}
    reference = jnp.asarray((1.0, 0.7, 1.2, 0.9), dtype=jnp.complex128)

    def centers(candidate, indices):
        return (features @ candidate["theta"])[indices].astype(jnp.complex128)

    def hpsi(candidate, indices):
        state = features @ candidate["theta"]
        return state[indices].astype(jnp.complex128), (
            hamiltonian @ state
        )[indices].astype(jnp.complex128)

    full = common_sample_vmc_value_and_grad(
        params,
        samples,
        reference,
        hpsi_fn=hpsi,
        center_fn=centers,
    )
    chunked = common_sample_vmc_value_and_grad(
        params,
        samples,
        reference,
        hpsi_fn=hpsi,
        center_fn=centers,
        gradient_microbatch_size=2,
    )
    assert abs(chunked.energy - full.energy) < 1.0e-14
    assert abs(chunked.variance - full.variance) < 1.0e-14
    assert abs(chunked.ess_fraction - full.ess_fraction) < 1.0e-14
    np.testing.assert_allclose(
        chunked.gradient["theta"],
        full.gradient["theta"],
        atol=2.0e-12,
        rtol=2.0e-12,
    )


def test_paired_energy_difference_keeps_common_sample_covariance() -> None:
    first_center = np.ones(4, dtype=np.complex128)
    second_center = np.ones(4, dtype=np.complex128)
    reference = np.ones(4, dtype=np.complex128)
    first_local = np.asarray((1.0, 2.0, 4.0, 5.0), dtype=np.complex128)
    second_local = np.asarray((0.5, 2.5, 3.0, 6.0), dtype=np.complex128)
    result = paired_common_sample_energy_difference(
        first_center,
        first_local,
        second_center,
        second_local,
        reference,
    )
    differences = (first_local.real - first_local.real.mean()) - (
        second_local.real - second_local.real.mean()
    )
    expected_se = np.std(differences, ddof=1) / np.sqrt(4)
    assert result.difference == first_local.real.mean() - second_local.real.mean()
    assert abs(result.standard_error - expected_se) < 1.0e-14


def test_identical_paired_estimators_have_exactly_zero_difference() -> None:
    centers = np.asarray((1.0, 0.7, 1.4), dtype=np.complex128)
    actions = np.asarray((0.3, -0.2, 0.8), dtype=np.complex128)
    reference = np.ones(3, dtype=np.complex128)
    result = paired_common_sample_energy_difference(
        centers, actions, centers, actions, reference
    )
    assert result.difference == 0.0
    assert result.standard_error == 0.0
    assert result.z_score == 0.0
