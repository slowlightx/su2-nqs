import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from su2_nqs.vmc.krylov_deflation import (
    build_arnoldi_deflated_preconditioner,
)
from su2_nqs.vmc.preconditioned_sr_projection import (
    iteratively_refined_preconditioned_conjugate_gradient,
    preconditioned_gmres,
    preconditioned_conjugate_gradient,
)


def _dense_application(apply, dimension: int) -> np.ndarray:
    identity = np.eye(dimension)
    return np.column_stack(
        [np.asarray(apply(jnp.asarray(identity[:, column]))) for column in range(dimension)]
    )


def test_two_level_inverse_is_symmetric_positive_and_exact_on_coarse_space():
    generator = np.random.default_rng(7401)
    dimension = 24
    rotation, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
    eigenvalues = np.geomspace(4.0, 2.0e-5, dimension)
    matrix = rotation @ np.diag(eigenvalues) @ rotation.T
    diagonal_inverse = 1.0 / np.diag(matrix)
    rhs = generator.normal(size=dimension)
    matrix_jax = jnp.asarray(matrix)

    coarse = build_arnoldi_deflated_preconditioner(
        lambda vector: matrix_jax @ vector,
        jnp.asarray(rhs),
        lambda vector: jnp.asarray(diagonal_inverse) * vector,
        rank=7,
    )
    apply = lambda vector: coarse.apply(
        vector, lambda value: jnp.asarray(diagonal_inverse) * value
    )
    dense_inverse = _dense_application(apply, dimension)
    np.testing.assert_allclose(dense_inverse, dense_inverse.T, rtol=2e-11, atol=2e-11)
    assert np.linalg.eigvalsh(dense_inverse)[0] > 0.0
    for row, normal_row in zip(
        np.asarray(coarse.basis), np.asarray(coarse.normal_images), strict=True
    ):
        np.testing.assert_allclose(
            apply(jnp.asarray(normal_row)), row, rtol=2e-8, atol=2e-8
        )
    assert int(coarse.active_rank) == 7
    assert float(coarse.minimum_coarse_eigenvalue) > 0.0
    assert coarse.stored_matrix_elements == 2 * dimension * 7 + 7 * 7


def test_global_arnoldi_correction_reduces_true_pcg_residual():
    generator = np.random.default_rng(7402)
    dimension = 64
    local = np.geomspace(0.2, 3.0, dimension)
    carriers, _ = np.linalg.qr(generator.normal(size=(dimension, 6)))
    global_strength = np.geomspace(300.0, 3.0, 6)
    matrix = np.diag(local) + carriers @ np.diag(global_strength) @ carriers.T
    rhs = generator.normal(size=dimension)
    matrix_jax = jnp.asarray(matrix)
    base = lambda vector: vector / jnp.asarray(np.diag(matrix))
    coarse = build_arnoldi_deflated_preconditioner(
        lambda vector: matrix_jax @ vector,
        jnp.asarray(rhs),
        base,
        rank=8,
    )
    ordinary = preconditioned_conjugate_gradient(
        lambda vector: matrix_jax @ vector,
        jnp.asarray(rhs),
        preconditioner=base,
        tolerance=1.0e-12,
        max_iterations=12,
    )
    deflated = preconditioned_conjugate_gradient(
        lambda vector: matrix_jax @ vector,
        jnp.asarray(rhs),
        preconditioner=lambda vector: coarse.apply(vector, base),
        tolerance=1.0e-12,
        max_iterations=12,
    )
    assert not bool(ordinary.breakdown)
    assert not bool(deflated.breakdown)
    # At identical PCG depth, the global space removes more than an order of
    # magnitude of residual that the purely local diagonal metric leaves.
    assert float(deflated.true_relative_residual) < (
        0.1 * float(ordinary.true_relative_residual)
    )


def test_true_residual_replacement_multiplies_fixed_cycle_reductions():
    generator = np.random.default_rng(7402)
    dimension = 64
    local = np.geomspace(0.2, 3.0, dimension)
    carriers, _ = np.linalg.qr(generator.normal(size=(dimension, 6)))
    matrix = np.diag(local) + carriers @ np.diag(
        np.geomspace(300.0, 3.0, 6)
    ) @ carriers.T
    rhs = generator.normal(size=dimension)
    matrix_jax = jnp.asarray(matrix)
    base = lambda vector: vector / jnp.asarray(np.diag(matrix))
    coarse = build_arnoldi_deflated_preconditioner(
        lambda vector: matrix_jax @ vector,
        jnp.asarray(rhs),
        base,
        rank=8,
    )
    refined = iteratively_refined_preconditioned_conjugate_gradient(
        lambda vector: matrix_jax @ vector,
        jnp.asarray(rhs),
        preconditioner=lambda vector: coarse.apply(vector, base),
        tolerance=1.0e-7,
        iterations_per_cycle=12,
        maximum_cycles=6,
    )
    history = np.asarray(refined.true_relative_residual_history)[
        : int(refined.cycles)
    ]
    assert bool(refined.result.converged)
    assert not bool(refined.result.breakdown)
    assert int(refined.cycles) == 5
    assert int(refined.result.iterations) == 60
    assert np.all(np.diff(history) < 0.0)
    assert history[-1] < 1.0e-7
    np.testing.assert_allclose(
        matrix @ np.asarray(refined.result.solution), rhs, rtol=2e-7, atol=2e-7
    )


def test_gmres_reports_original_true_residual_after_full_orthogonalization():
    generator = np.random.default_rng(7403)
    dimension = 30
    rotation, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
    matrix = rotation @ np.diag(np.geomspace(5.0, 1.0e-4, dimension)) @ rotation.T
    rhs = generator.normal(size=dimension)
    matrix_jax = jnp.asarray(matrix)
    observed = preconditioned_gmres(
        lambda vector: matrix_jax @ vector,
        jnp.asarray(rhs),
        preconditioner=lambda vector: vector / jnp.asarray(np.diag(matrix)),
        tolerance=1.0e-8,
        restart=15,
        maximum_cycles=4,
    )
    direct = np.linalg.norm(matrix @ np.asarray(observed.result.solution) - rhs)
    direct /= np.linalg.norm(rhs)
    np.testing.assert_allclose(
        observed.result.true_relative_residual, direct, rtol=2e-8, atol=1e-12
    )
    assert int(observed.result.iterations) == 60
    assert not bool(observed.result.breakdown)


def test_arnoldi_deflation_validates_rank_and_real_vector():
    identity = lambda vector: vector
    with np.testing.assert_raises_regex(ValueError, "rank must be positive"):
        build_arnoldi_deflated_preconditioner(
            identity, jnp.ones(3), identity, rank=0
        )
    with np.testing.assert_raises_regex(ValueError, "exceed parameter count"):
        build_arnoldi_deflated_preconditioner(
            identity, jnp.ones(3), identity, rank=4
        )
    with np.testing.assert_raises_regex(TypeError, "real floating"):
        build_arnoldi_deflated_preconditioner(
            identity, jnp.ones(3, dtype=jnp.complex128), identity, rank=2
        )
