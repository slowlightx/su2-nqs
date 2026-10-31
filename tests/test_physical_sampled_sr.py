import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from su2_nqs.vmc.physical_sampled_sr import sampled_physical_sr_direction


jax.config.update("jax_enable_x64", True)


def test_matrix_free_direction_matches_explicit_complex_qgt() -> None:
    samples = jnp.arange(5, dtype=jnp.int32)[:, None]
    real_scores = jnp.asarray(
        [
            [1.0, -0.2, 0.3],
            [-0.4, 0.8, 0.1],
            [0.2, -0.3, 1.1],
            [0.7, 0.5, -0.6],
            [-0.9, 0.4, 0.2],
        ],
        dtype=jnp.float64,
    )
    phase_scores = jnp.asarray(
        [
            [0.1, 0.4, -0.2],
            [0.3, -0.7, 0.5],
            [-0.6, 0.2, 0.8],
            [0.9, -0.1, 0.3],
            [-0.2, 0.6, -0.4],
        ],
        dtype=jnp.float64,
    )
    signs = jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0], dtype=jnp.float64)
    params = {
        "head": jnp.asarray([0.25, -0.4], dtype=jnp.float64),
        "tail": jnp.asarray([0.15], dtype=jnp.float64),
    }
    gradient = {
        "head": jnp.asarray([0.7, -0.3], dtype=jnp.float64),
        "tail": jnp.asarray([0.2], dtype=jnp.float64),
    }

    def center_fn(candidate, batch):
        flat, _ = ravel_pytree(candidate)
        indices = batch[:, 0]
        exponent = real_scores[indices] @ flat + 1j * (phase_scores[indices] @ flat)
        return signs[indices] * jnp.exp(exponent)

    damping = 3.0e-2
    result = sampled_physical_sr_direction(
        params,
        samples,
        energy_gradient=gradient,
        center_fn=center_fn,
        damping=damping,
        cg_tolerance=1.0e-12,
        cg_max_iterations=30,
    )
    scores = np.asarray(real_scores + 1j * phase_scores)
    centered = scores - np.mean(scores, axis=0, keepdims=True)
    explicit_fisher = 2.0 * np.real(centered.conj().T @ centered) / len(samples)
    flat_gradient = np.asarray(ravel_pytree(gradient)[0])
    expected = np.linalg.solve(
        explicit_fisher + damping * np.eye(3), flat_gradient
    )
    actual = np.asarray(ravel_pytree(result.raw_direction)[0])
    np.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    residual = np.asarray(ravel_pytree(result.linear_residual)[0])
    np.testing.assert_allclose(residual, np.zeros(3), atol=2.0e-10)
    expected_fisher_norm = np.sqrt(expected @ explicit_fisher @ expected)
    assert result.fisher_norm == pytest.approx(expected_fisher_norm, rel=2.0e-10)
    assert result.diagnostics.linear_solve_converged
    assert result.diagnostics.qgt_matvec_count == result.diagnostics.cg_iterations + 2
    assert result.diagnostics.amplitude_source == "center_amplitude_complex_log"


def test_negative_real_centers_are_complex_logged_and_step_lowers_energy() -> None:
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)
    params = jnp.asarray([0.0], dtype=jnp.float64)

    def center_fn(candidate, batch):
        theta = candidate[0]
        return jnp.where(
            batch[:, 0] == 0,
            jnp.exp(theta),
            -jnp.exp(-theta),
        )

    def exact_energy(candidate):
        amplitudes = center_fn(candidate, samples)
        probabilities = jnp.abs(amplitudes) ** 2
        diagonal = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
        return jnp.sum(probabilities * diagonal) / jnp.sum(probabilities)

    gradient = jax.grad(exact_energy)(params)
    result = sampled_physical_sr_direction(
        params,
        samples,
        energy_gradient=gradient,
        center_fn=center_fn,
        damping=1.0e-3,
        cg_tolerance=1.0e-12,
        cg_max_iterations=10,
        step_size=0.1,
    )
    changed = params - result.direction
    assert np.isfinite(np.asarray(result.direction)).all()
    assert result.diagnostics.minimum_center_magnitude == pytest.approx(1.0)
    assert float(exact_energy(changed)) < float(exact_energy(params))
    assert result.diagnostics.directional_derivative > 0.0


def test_relative_and_fisher_trust_bounds_scale_one_common_direction() -> None:
    samples = jnp.asarray([[0], [1], [2], [3]], dtype=jnp.int32)
    params = jnp.asarray([0.4, -0.3], dtype=jnp.float64)
    gradient = jnp.asarray([1.0, -0.6], dtype=jnp.float64)
    scores = jnp.asarray(
        [[1.0, 0.2], [-0.4, 0.8], [0.3, -1.1], [-0.9, 0.1]],
        dtype=jnp.float64,
    )

    def log_amplitude_fn(candidate, batch):
        return (scores[batch[:, 0]] @ candidate).astype(jnp.complex128)

    maximum_relative = 2.0e-2
    fisher_radius = 1.0e-2
    result = sampled_physical_sr_direction(
        params,
        samples,
        energy_gradient=gradient,
        log_amplitude_fn=log_amplitude_fn,
        damping=1.0e-3,
        cg_tolerance=1.0e-12,
        cg_max_iterations=20,
        step_size=3.0,
        maximum_relative_parameter_step=maximum_relative,
        fisher_trust_radius=fisher_radius,
    )
    diagnostics = result.diagnostics
    assert diagnostics.trust_scale < 1.0
    assert diagnostics.relative_parameter_step <= maximum_relative * (1.0 + 1.0e-12)
    assert result.fisher_norm <= fisher_radius * (1.0 + 1.0e-12)
    raw = np.asarray(result.raw_direction)
    applied = np.asarray(result.direction)
    ratios = applied[np.abs(raw) > 1.0e-14] / raw[np.abs(raw) > 1.0e-14]
    np.testing.assert_allclose(ratios, ratios[0], atol=1.0e-12)
    assert diagnostics.amplitude_source == "direct_log_amplitude"


def test_zero_center_and_invalid_callback_choice_fail_closed() -> None:
    params = jnp.asarray([0.2], dtype=jnp.float64)
    gradient = jnp.asarray([0.1], dtype=jnp.float64)
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)

    with pytest.raises(FloatingPointError, match="zero center"):
        sampled_physical_sr_direction(
            params,
            samples,
            energy_gradient=gradient,
            center_fn=lambda _params, _samples: jnp.asarray([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="exactly one"):
        sampled_physical_sr_direction(
            params,
            samples,
            energy_gradient=gradient,
        )
    with pytest.raises(ValueError, match="exactly one"):
        sampled_physical_sr_direction(
            params,
            samples,
            energy_gradient=gradient,
            center_fn=lambda _params, _samples: jnp.ones(2),
            log_amplitude_fn=lambda _params, _samples: jnp.zeros(2),
        )


def test_relative_trust_bound_rejects_zero_parameter_norm() -> None:
    params = jnp.zeros(1, dtype=jnp.float64)
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="nonzero parameter norm"):
        sampled_physical_sr_direction(
            params,
            samples,
            energy_gradient=jnp.ones(1, dtype=jnp.float64),
            log_amplitude_fn=lambda candidate, batch: (
                (2 * batch[:, 0] - 1) * candidate[0]
            ).astype(jnp.complex128),
            maximum_relative_parameter_step=0.1,
        )


def test_true_residual_can_be_inspected_when_iteration_budget_is_too_small() -> None:
    samples = jnp.asarray([[0], [1], [2], [3]], dtype=jnp.int32)
    scores = jnp.asarray(
        [[1.0, 0.0], [0.0, 4.0], [-1.0, 0.0], [0.0, -4.0]],
        dtype=jnp.float64,
    )
    params = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    gradient = jnp.asarray([0.8, -0.7], dtype=jnp.float64)

    def log_amplitude_fn(candidate, batch):
        return (scores[batch[:, 0]] @ candidate).astype(jnp.complex128)

    result = sampled_physical_sr_direction(
        params,
        samples,
        energy_gradient=gradient,
        log_amplitude_fn=log_amplitude_fn,
        damping=1.0e-3,
        cg_tolerance=1.0e-14,
        cg_max_iterations=1,
        raise_on_nonconvergence=False,
    )
    centered = np.asarray(scores - jnp.mean(scores, axis=0, keepdims=True))
    fisher = 2.0 * centered.T @ centered / len(samples)
    raw = np.asarray(result.raw_direction)
    expected_residual = (fisher + 1.0e-3 * np.eye(2)) @ raw - np.asarray(
        gradient
    )
    np.testing.assert_allclose(
        np.asarray(result.linear_residual), expected_residual, atol=1.0e-12
    )
    assert not result.diagnostics.linear_solve_converged
    assert result.diagnostics.cg_iterations == 1
    assert result.diagnostics.relative_linear_residual > 1.0e-6
