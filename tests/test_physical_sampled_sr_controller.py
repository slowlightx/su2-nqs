import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.physical_sr_controller import (
    PersistentActiveSampledPhysicalSRSolver,
    correlated_physical_energy_difference,
    held_out_correlated_line_search,
)


jax.config.update("jax_enable_x64", True)


def test_persistent_active_solver_reuses_compiled_executable_and_direction() -> None:
    samples = jnp.arange(6, dtype=jnp.int32)
    real_scores = jnp.asarray(
        [
            [1.0, -0.2],
            [-0.4, 0.8],
            [0.2, -0.3],
            [0.7, 0.5],
            [-0.9, 0.4],
            [0.1, -1.2],
        ],
        dtype=jnp.float64,
    )
    phase_scores = jnp.asarray(
        [
            [0.1, 0.4],
            [0.3, -0.7],
            [-0.6, 0.2],
            [0.9, -0.1],
            [-0.2, 0.6],
            [0.5, 0.3],
        ],
        dtype=jnp.float64,
    )
    active = jnp.asarray([0.25, -0.4], dtype=jnp.float64)
    gradient = jnp.asarray([0.7, -0.3], dtype=jnp.float64)

    def log_amplitude_fn(candidate, batch):
        scores = real_scores[batch] + 1j * phase_scores[batch]
        return scores @ candidate

    damping = 3.0e-2
    solver = PersistentActiveSampledPhysicalSRSolver(
        log_amplitude_fn=log_amplitude_fn,
        damping=damping,
        cg_tolerance=1.0e-12,
        cg_max_iterations=20,
        step_size=0.2,
    )
    first = solver.solve(active, samples, energy_gradient=gradient)
    second = solver.solve(active, samples, energy_gradient=gradient)

    complex_scores = np.asarray(real_scores + 1j * phase_scores)
    centered = complex_scores - np.mean(complex_scores, axis=0, keepdims=True)
    fisher = 2.0 * np.real(centered.conj().T @ centered) / len(samples)
    expected_raw = np.linalg.solve(
        fisher + damping * np.eye(2), np.asarray(gradient)
    )
    np.testing.assert_allclose(first.raw_direction, expected_raw, atol=2.0e-11)
    np.testing.assert_array_equal(first.direction, second.direction)
    np.testing.assert_array_equal(first.raw_direction, second.raw_direction)
    assert not first.compilation.compile_cache_hit
    assert first.compilation.compiled_signature_count == 1
    assert first.compilation.compilation_seconds >= 0.0
    assert first.compilation.execution_seconds >= 0.0
    assert second.compilation.compile_cache_hit
    assert second.compilation.compiled_signature_count == 1
    assert second.compilation.compilation_seconds == 0.0
    assert solver.compiled_signature_count == 1
    assert first.diagnostics.parameter_count == 2
    assert first.diagnostics.sample_count == 6

    # Values can change without changing the compiled shape/dtype signature.
    third = solver.solve(active, samples[::-1], energy_gradient=gradient)
    assert third.compilation.compile_cache_hit
    assert solver.compiled_signature_count == 1


def test_active_solver_callback_can_close_over_a_frozen_backbone() -> None:
    frozen = jnp.asarray([0.2, -0.1, 0.7], dtype=jnp.float64)
    injection = jnp.asarray(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=jnp.float64
    )
    scores = jnp.asarray(
        [[1.0, 0.0, -0.2], [-0.5, 0.3, 0.8], [0.2, -0.7, 0.1]],
        dtype=jnp.float64,
    )

    def logs(active, samples):
        full = frozen + injection @ active
        return (scores[samples] @ full).astype(jnp.complex128)

    solver = PersistentActiveSampledPhysicalSRSolver(
        log_amplitude_fn=logs,
        damping=0.1,
        cg_tolerance=1.0e-10,
    )
    result = solver.solve(
        jnp.asarray([0.1, -0.2], dtype=jnp.float64),
        jnp.arange(3, dtype=jnp.int32),
        energy_gradient=jnp.asarray([0.4, -0.3], dtype=jnp.float64),
    )
    assert result.direction.shape == (2,)
    assert result.diagnostics.parameter_count == 2
    assert np.all(np.isfinite(np.asarray(result.direction)))


def test_persistent_center_path_complex_logs_negative_amplitudes() -> None:
    samples = jnp.asarray([0, 1], dtype=jnp.int32)

    def centers(active, batch):
        return jnp.where(
            batch == 0,
            jnp.exp(active[0]),
            -jnp.exp(-active[0]),
        )

    solver = PersistentActiveSampledPhysicalSRSolver(
        center_fn=centers,
        damping=1.0e-2,
        cg_tolerance=1.0e-12,
    )
    result = solver.solve(
        jnp.asarray([0.1], dtype=jnp.float64),
        samples,
        energy_gradient=jnp.asarray([0.2], dtype=jnp.float64),
    )
    assert np.isfinite(np.asarray(result.direction)).all()
    assert result.diagnostics.amplitude_source == "center_amplitude_complex_log"
    assert result.diagnostics.minimum_center_magnitude == pytest.approx(np.exp(-0.1))


def test_correlated_energy_uses_candidate_over_reference_born_weights() -> None:
    samples = jnp.asarray([0, 1] * 8, dtype=jnp.int32)
    reference = jnp.asarray([0.0], dtype=jnp.float64)
    candidate = jnp.asarray([0.4], dtype=jnp.float64)

    def logs(params, batch):
        spin = 2.0 * batch - 1.0
        return (params[0] * spin).astype(jnp.complex128)

    def local_energy(_params, batch):
        return batch.astype(jnp.complex128)

    result = correlated_physical_energy_difference(
        reference,
        candidate,
        samples,
        log_amplitude_fn=logs,
        local_energy_fn=local_energy,
        batch_count=4,
        standard_error_method="max",
    )
    expected_candidate = np.exp(0.8) / (np.exp(-0.8) + np.exp(0.8))
    assert result.reference.energy == pytest.approx(0.5)
    assert result.candidate.energy == pytest.approx(expected_candidate)
    assert result.energy_difference == pytest.approx(expected_candidate - 0.5)
    assert 0.5 < result.candidate.effective_sample_size_fraction < 1.0
    assert result.batch_standard_error is not None


def test_held_out_line_search_backtracks_then_accepts_improving_scale() -> None:
    samples = jnp.asarray([-1.0, 1.0] * 8, dtype=jnp.float64)
    reference = jnp.asarray([0.0], dtype=jnp.float64)
    direction = jnp.asarray([1.0], dtype=jnp.float64)

    def logs(_params, batch):
        return jnp.zeros(batch.shape, dtype=jnp.complex128)

    local_energy_calls = []

    def local_energy(params, batch):
        local_energy_calls.append(float(params[0]))
        shared_fluctuation = 0.1 * batch
        return ((params[0] + 0.5) ** 2 + shared_fluctuation).astype(
            jnp.complex128
        )

    decision = held_out_correlated_line_search(
        reference,
        direction,
        samples,
        log_amplitude_fn=logs,
        local_energy_fn=local_energy,
        step_scales=(1.5, 0.5, 0.25),
        maximum_variance_ratio=1.01,
        batch_count=4,
    )
    assert decision.accepted
    assert not decision.rollback_performed
    assert decision.accepted_step_scale == pytest.approx(0.5)
    np.testing.assert_allclose(decision.active_params, [-0.5])
    assert len(decision.attempts) == 2
    assert decision.attempts[0].comparison.confidence_level == pytest.approx(
        1.0 - 0.05 / 3.0
    )
    assert not decision.attempts[0].accepted
    assert "correlated_energy_bound" in decision.attempts[0].rejection_reasons
    assert decision.attempts[1].accepted
    # One immutable reference evaluation plus one per attempted candidate.
    assert len(local_energy_calls) == 3


def test_held_out_line_search_returns_exact_reference_on_rollback() -> None:
    samples = jnp.asarray([-1.0, 1.0] * 4, dtype=jnp.float64)
    reference = jnp.asarray([0.0], dtype=jnp.float64)
    direction = jnp.asarray([-1.0], dtype=jnp.float64)

    def logs(_params, batch):
        return jnp.zeros(batch.shape, dtype=jnp.complex128)

    def local_energy(params, batch):
        return ((params[0] + 0.5) ** 2 + 0.1 * batch).astype(jnp.complex128)

    decision = held_out_correlated_line_search(
        reference,
        direction,
        samples,
        log_amplitude_fn=logs,
        local_energy_fn=local_energy,
        step_scales=(1.0, 0.5),
        batch_count=2,
    )
    assert not decision.accepted
    assert decision.rollback_performed
    assert decision.accepted_step_scale is None
    np.testing.assert_array_equal(decision.active_params, reference)
    assert len(decision.attempts) == 2
    assert all(not attempt.accepted for attempt in decision.attempts)
