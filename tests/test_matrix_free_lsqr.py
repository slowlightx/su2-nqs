import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.matrix_free_lsqr import (
    damped_least_squares_residuals,
    damped_lsqr,
)


def _dense_damped_solution(matrix: np.ndarray, rhs: np.ndarray, damping: float) -> np.ndarray:
    augmented_matrix = np.vstack(
        (matrix, np.sqrt(damping) * np.eye(matrix.shape[1], dtype=matrix.dtype))
    )
    augmented_rhs = np.concatenate(
        (rhs, np.zeros(matrix.shape[1], dtype=rhs.dtype))
    )
    return np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)[0]


def _realify_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [matrix.real, -matrix.imag],
            [matrix.imag, matrix.real],
        ]
    )


def _realify_vector(vector: np.ndarray) -> np.ndarray:
    return np.concatenate((vector.real, vector.imag))


def test_jvp_vjp_lsqr_matches_dense_augmented_lstsq_and_exact_residuals():
    point = jnp.asarray([0.2, -0.35, 0.6, -0.1], dtype=jnp.float64)

    def features(parameters):
        p0, p1, p2, p3 = parameters
        return jnp.asarray(
            [
                jnp.sin(p0) + p1 * p2,
                jnp.exp(0.2 * p1) + p0 * p3,
                p0 * p0 - 0.3 * p2 + jnp.cos(p3),
                p1 - p2 * p3,
                jnp.tanh(p0 + p2) + 0.4 * p3,
                p0 - 2.0 * p1 + 0.5 * p2 + p3,
            ],
            dtype=jnp.float64,
        )

    _, jvp = jax.linearize(features, point)
    _, vjp = jax.vjp(features, point)
    matvec = lambda vector: jvp(vector)
    rmatvec = lambda vector: vjp(vector)[0]
    dense_jacobian = np.asarray(jax.jacfwd(features)(point))

    rhs = jnp.asarray([0.7, -0.2, 0.4, 1.1, -0.8, 0.3], dtype=jnp.float64)
    damping = 3.7e-2
    initial = jnp.asarray([-0.4, 0.1, 0.2, 0.3], dtype=jnp.float64)
    expected = _dense_damped_solution(dense_jacobian, np.asarray(rhs), damping)

    result = damped_lsqr(
        matvec,
        rmatvec,
        rhs,
        damping=damping,
        initial_solution=initial,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        max_iterations=40,
    )

    assert bool(result.converged)
    assert not bool(result.breakdown)
    np.testing.assert_allclose(np.asarray(result.solution), expected, rtol=2e-11, atol=2e-11)

    direct = damped_least_squares_residuals(
        matvec, rmatvec, rhs, result.solution, damping=damping
    )
    np.testing.assert_allclose(result.primal_residual_norm, direct.primal_norm, rtol=1e-13)
    np.testing.assert_allclose(
        result.augmented_residual_norm, direct.augmented_norm, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.normal_residual_norm, direct.normal_norm, rtol=5e-3, atol=5e-17
    )

    iterations = int(result.iterations)
    assert result.primal_residual_history.shape == (41,)
    assert result.normal_residual_history.shape == (41,)
    np.testing.assert_allclose(
        result.normal_residual_history[iterations], result.normal_residual_norm
    )
    np.testing.assert_allclose(
        result.augmented_residual_history[iterations:], result.augmented_residual_norm
    )
    np.testing.assert_allclose(
        result.lsqr_residual_estimate_history[iterations],
        result.augmented_residual_norm,
        rtol=5e-11,
        atol=5e-13,
    )
    assert float(result.relative_normal_residual) < 1e-12


def test_rank_deficient_complex_realified_problem_matches_dense_solution():
    complex_matrix = np.asarray(
        [
            [1.0 + 0.2j, -0.3 + 0.7j, 0.0 + 0.1j, 0.4 - 0.2j],
            [0.2 - 0.8j, 0.5 + 0.1j, 1.1 - 0.4j, -0.7 + 0.3j],
            [0.6 + 0.3j, -0.2 - 0.5j, 0.8 + 0.2j, 0.1 + 0.9j],
            [-0.4 + 0.1j, 0.9 - 0.6j, 0.3 + 0.7j, 0.2 - 0.2j],
            [0.7 - 0.4j, 0.1 + 0.8j, -0.5 + 0.6j, 0.3 + 0.1j],
        ],
        dtype=np.complex128,
    )
    # Make the fourth complex column exactly dependent on the first two.
    complex_matrix[:, 3] = complex_matrix[:, 0] - 2.0j * complex_matrix[:, 1]
    complex_rhs = np.asarray(
        [0.5 - 0.2j, -0.7 + 0.4j, 0.1 + 0.9j, 0.6 - 0.3j, -0.2 + 0.8j],
        dtype=np.complex128,
    )
    matrix = _realify_matrix(complex_matrix)
    rhs = _realify_vector(complex_rhs)
    damping = 2.0e-5
    expected = _dense_damped_solution(matrix, rhs, damping)
    matrix_jax = jnp.asarray(matrix, dtype=jnp.float64)
    rhs_jax = jnp.asarray(rhs, dtype=jnp.float64)

    result = damped_lsqr(
        lambda vector: matrix_jax @ vector,
        lambda vector: matrix_jax.T @ vector,
        rhs_jax,
        damping=damping,
        relative_tolerance=2e-12,
        absolute_tolerance=1e-14,
        max_iterations=100,
    )

    assert bool(result.converged)
    assert not bool(result.breakdown)
    np.testing.assert_allclose(np.asarray(result.solution), expected, rtol=2e-9, atol=2e-9)
    assert float(result.normal_residual_norm) < 2e-11
    # The undamped primal residual need not vanish for an inconsistent system;
    # the augmented and normal residuals answer different questions.
    assert float(result.primal_residual_norm) > float(result.normal_residual_norm)
    assert float(result.augmented_residual_norm) >= float(result.primal_residual_norm)


def test_ill_conditioned_complex_realified_problem_has_dense_objective_and_stationarity():
    generator = np.random.default_rng(1729)
    raw_left = generator.normal(size=(6, 4)) + 1j * generator.normal(size=(6, 4))
    raw_right = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
    left, _ = np.linalg.qr(raw_left)
    right, _ = np.linalg.qr(raw_right)
    singular_values = np.asarray([1.0, 2.0e-3, 5.0e-7, 0.0])
    complex_matrix = left @ np.diag(singular_values) @ right.conj().T
    complex_rhs = generator.normal(size=6) + 1j * generator.normal(size=6)
    matrix = _realify_matrix(complex_matrix)
    rhs = _realify_vector(complex_rhs)
    damping = 3.0e-10
    expected = _dense_damped_solution(matrix, rhs, damping)
    matrix_jax = jnp.asarray(matrix, dtype=jnp.float64)
    rhs_jax = jnp.asarray(rhs, dtype=jnp.float64)

    result = damped_lsqr(
        lambda vector: matrix_jax @ vector,
        lambda vector: matrix_jax.T @ vector,
        rhs_jax,
        damping=damping,
        relative_tolerance=5e-11,
        absolute_tolerance=1e-14,
        max_iterations=160,
    )

    assert bool(result.converged)
    assert not bool(result.breakdown)
    # With a condition number near 1/sqrt(damping), stationarity and objective
    # are more meaningful than demanding every weakly determined coefficient
    # agree to machine precision.
    expected_residual = matrix @ expected - rhs
    expected_objective = np.vdot(expected_residual, expected_residual).real + damping * np.vdot(
        expected, expected
    ).real
    actual_objective = float(result.augmented_residual_norm**2)
    np.testing.assert_allclose(actual_objective, expected_objective, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(np.asarray(result.solution), expected, rtol=3e-5, atol=2e-5)
    assert float(result.relative_normal_residual) < 5e-11


def test_zero_rhs_converges_without_iteration_and_is_jittable():
    matrix = jnp.asarray(
        [[1.0, -0.5, 0.2], [0.3, 0.7, -1.1], [0.4, 0.2, 0.9]],
        dtype=jnp.float64,
    )

    def solve(rhs):
        return damped_lsqr(
            lambda vector: matrix @ vector,
            lambda vector: matrix.T @ vector,
            rhs,
            damping=1e-3,
            relative_tolerance=1e-12,
            max_iterations=12,
        )

    result = jax.jit(solve)(jnp.zeros(3, dtype=jnp.float64))
    assert int(result.iterations) == 0
    assert bool(result.converged)
    assert not bool(result.breakdown)
    np.testing.assert_array_equal(np.asarray(result.solution), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(result.primal_residual_history), np.zeros(13))
    np.testing.assert_array_equal(np.asarray(result.normal_residual_history), np.zeros(13))


def test_zero_rhs_nonzero_warm_start_uses_initial_true_normal_residual_scale():
    matrix = jnp.asarray(
        [[2.0, -0.3], [0.1, 0.7], [-0.4, 1.2]], dtype=jnp.float64
    )
    rhs = jnp.zeros(3, dtype=jnp.float64)
    initial = jnp.asarray([1.5, -0.8], dtype=jnp.float64)
    damping = 2.0e-2
    relative_tolerance = 3.0e-11
    result = damped_lsqr(
        lambda vector: matrix @ vector,
        lambda vector: matrix.T @ vector,
        rhs,
        damping=damping,
        initial_solution=initial,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=1.0e-14,
        max_iterations=20,
    )

    initial_normal = matrix.T @ (matrix @ initial) + damping * initial
    expected_scale = float(jnp.linalg.norm(initial_normal))
    np.testing.assert_allclose(result.normal_residual_scale, expected_scale, rtol=2e-14)
    np.testing.assert_allclose(
        result.normal_residual_threshold,
        max(1.0e-14, relative_tolerance * expected_scale),
        rtol=2e-14,
    )
    np.testing.assert_allclose(result.relative_normal_residual_history[0], 1.0)
    assert int(result.iterations) > 0
    assert bool(result.converged)
    np.testing.assert_allclose(result.solution, jnp.zeros(2), atol=2e-11)
    assert float(result.normal_residual_norm) <= float(
        result.normal_residual_threshold
    )


def test_invalid_solver_arguments_are_rejected():
    identity = lambda vector: vector
    rhs = jnp.ones(2, dtype=jnp.float64)
    with pytest.raises(ValueError, match="damping"):
        damped_lsqr(identity, identity, rhs, damping=-1.0)
    with pytest.raises(ValueError, match="max_iterations"):
        damped_lsqr(identity, identity, rhs, damping=1e-3, max_iterations=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        damped_lsqr(identity, identity, rhs[:, None], damping=1e-3)
