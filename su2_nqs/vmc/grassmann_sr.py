"""Exact-small Grassmann stochastic-reconfiguration linear algebra.

The metric follows the tangent-space projector of a represented subspace,

``QGT(mu,nu) = Re Tr[G^-1 dPhi_mu^dagger (I-P) dPhi_nu] / R``.

The generic implementation is matrix free in parameter space: JVP/VJP
products apply the tangent map and its adjoint, and conjugate gradients solve
the damped normal equations.  A mean-Ky--Fan specialization forms the small
exact feature-space kernel and uses a deterministic dual Cholesky solve.  Both
are exhaustive references before a sampled implementation, not scalable
production engines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class GrassmannSRDirection(NamedTuple):
    direction: Array
    loss: Array
    gradient_norm: Array
    direction_norm: Array
    relative_linear_residual: Array
    minimum_gram_eigenvalue: Array
    gram_condition_number: Array
    linear_solve_converged: Array


class GrassmannEnergySRDirection(NamedTuple):
    """Diagnostics for an exact-small mean-Ky--Fan SR direction."""

    direction: Array
    gradient: Array
    loss: Array
    gradient_norm: Array
    direction_norm: Array
    canonical_direction_norm: Array
    directional_derivative: Array
    relative_linear_residual: Array
    minimum_gram_eigenvalue: Array
    gram_condition_number: Array
    horizontality_error: Array
    damped_tangent_capture_fraction: Array
    linear_solve_converged: Array
    full_rank: Array
    orthonormality_error: Array
    hamiltonian_hermiticity_error: Array
    grassmann_residual_norm: Array
    grassmann_variance: Array


def _realified_tangent_feature(
    candidate: Array,
    *,
    reference: Array,
    orthonormal: Array,
    inverse_sqrt_gram: Array,
    state_fn: Callable[[Array], Array],
) -> Array:
    variation = state_fn(candidate) - reference
    perpendicular = variation - orthonormal @ (
        orthonormal.conj().T @ variation
    )
    weighted = perpendicular @ inverse_sqrt_gram
    weighted = weighted / jnp.sqrt(jnp.asarray(reference.shape[1], dtype=jnp.float64))
    return jnp.concatenate(
        (jnp.real(weighted).reshape(-1), jnp.imag(weighted).reshape(-1))
    )


def exact_grassmann_sr_direction(
    flat_parameters: Array,
    *,
    state_fn: Callable[[Array], Array],
    loss_fn: Callable[[Array], Array],
    damping: float = 1.0e-3,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 50,
    relative_rank_tolerance: float = 1.0e-10,
) -> GrassmannSRDirection:
    """Return one damped exact Grassmann natural-gradient direction.

    ``state_fn`` returns a full-Hilbert-space-by-rank complex matrix.  The
    parameters must be a flat real vector.  Rank-deficient reference states
    fail closed through non-finite diagnostics and direction values.
    """

    if not jax.config.x64_enabled:
        raise RuntimeError("exact Grassmann SR requires JAX x64 mode")
    flat = jnp.asarray(flat_parameters)
    if flat.ndim != 1 or not jnp.issubdtype(flat.dtype, jnp.floating):
        raise TypeError("flat_parameters must be a one-dimensional real array")
    if flat.dtype != jnp.float64:
        raise TypeError("exact Grassmann SR requires float64 parameters")
    if damping <= 0.0 or cg_tolerance <= 0.0 or cg_max_iterations < 1:
        raise ValueError("damping, tolerance, and iteration count must be positive")
    if not 0.0 < relative_rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie strictly between zero and one")

    loss, gradient = jax.value_and_grad(loss_fn)(flat)
    reference = state_fn(flat)
    if reference.ndim != 2 or reference.shape[1] < 1:
        raise ValueError("state_fn must return a nonempty matrix of columns")
    gram = reference.conj().T @ reference
    gram = 0.5 * (gram + gram.conj().T)
    gram_values, gram_vectors = jnp.linalg.eigh(gram)
    largest = gram_values[-1]
    full_rank = gram_values[0] > relative_rank_tolerance * largest
    safe_scale = jnp.maximum(largest, jnp.finfo(gram_values.dtype).tiny)
    safe_values = jnp.maximum(
        gram_values, relative_rank_tolerance * safe_scale
    )
    inverse_sqrt = (gram_vectors / jnp.sqrt(safe_values)[None, :]) @ gram_vectors.conj().T
    orthonormal = reference @ inverse_sqrt
    reference = jax.lax.stop_gradient(reference)
    inverse_sqrt = jax.lax.stop_gradient(inverse_sqrt)
    orthonormal = jax.lax.stop_gradient(orthonormal)

    feature_fn = lambda candidate: _realified_tangent_feature(
        candidate,
        reference=reference,
        orthonormal=orthonormal,
        inverse_sqrt_gram=inverse_sqrt,
        state_fn=state_fn,
    )
    _, pullback = jax.vjp(feature_fn, flat)

    def metric_product(vector: Array) -> Array:
        _, tangent = jax.jvp(feature_fn, (flat,), (vector,))
        return pullback(tangent)[0] + damping * vector

    solution, _ = jax.scipy.sparse.linalg.cg(
        metric_product,
        gradient,
        tol=cg_tolerance,
        maxiter=cg_max_iterations,
    )
    direction = -solution
    residual = metric_product(solution) - gradient
    relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
        jnp.linalg.norm(gradient), jnp.finfo(flat.dtype).tiny
    )
    solve_converged = relative_residual <= jnp.maximum(
        10.0 * cg_tolerance, jnp.sqrt(jnp.finfo(flat.dtype).eps)
    )
    invalid = (
        ~full_rank
        | ~solve_converged
        | ~jnp.isfinite(loss)
        | ~jnp.all(jnp.isfinite(direction))
    )
    direction = jnp.where(invalid, jnp.full_like(direction, jnp.nan), direction)
    return GrassmannSRDirection(
        direction=direction,
        loss=loss,
        gradient_norm=jnp.linalg.norm(gradient),
        direction_norm=jnp.linalg.norm(direction),
        relative_linear_residual=relative_residual,
        minimum_gram_eigenvalue=gram_values[0],
        gram_condition_number=largest / jnp.maximum(
            gram_values[0], jnp.finfo(gram_values.dtype).tiny
        ),
        linear_solve_converged=solve_converged,
    )


def exact_grassmann_energy_sr_direction(
    flat_parameters: Array,
    *,
    state_fn: Callable[[Array], Array],
    hamiltonian: Array,
    damping: float = 1.0e-3,
    relative_rank_tolerance: float = 1.0e-10,
) -> GrassmannEnergySRDirection:
    """Return the exact-small damped SR direction for a mean Ky--Fan loss.

    The represented columns need not be mutually orthogonal.  If ``Phi`` is
    their matrix and ``R`` their number, this routine minimizes

    ``Tr[Q^dagger H Q] / R``, where ``Q = Phi (Phi^dagger Phi)^(-1/2)``.

    The pullback metric is formed from the horizontal, realified tangent
    feature

    ``(I - Q Q^dagger) dPhi (Phi^dagger Phi)^(-1/2) / sqrt(R)``.

    This exhaustive reference explicitly constructs the feature Jacobian and
    solves the damped equations through the smaller dual kernel.  It is meant
    for exact-small validation; sampled production calculations should use a
    matrix-free estimator.  Rank-deficient input fails closed with a NaN
    direction rather than accepting a numerical QR completion.
    """

    if not jax.config.x64_enabled:
        raise RuntimeError("exact Grassmann SR requires JAX x64 mode")
    flat = jnp.asarray(flat_parameters)
    operator = jnp.asarray(hamiltonian)
    if flat.ndim != 1 or not jnp.issubdtype(flat.dtype, jnp.floating):
        raise TypeError("flat_parameters must be a one-dimensional real array")
    if flat.dtype != jnp.float64:
        raise TypeError("exact Grassmann SR requires float64 parameters")
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if not 0.0 < relative_rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie strictly between zero and one")

    reference = state_fn(flat)
    if reference.ndim != 2 or reference.shape[1] < 1:
        raise ValueError("state_fn must return a nonempty matrix of columns")
    if operator.shape != (reference.shape[0], reference.shape[0]):
        raise ValueError("hamiltonian shape does not match state_fn output")
    if reference.dtype not in (jnp.float64, jnp.complex128):
        raise TypeError("exact Grassmann SR requires float64 or complex128 states")
    if operator.dtype not in (jnp.float64, jnp.complex128):
        raise TypeError("exact Grassmann SR requires a float64 or complex128 Hamiltonian")

    rank = reference.shape[1]
    gram = reference.conj().T @ reference
    gram = 0.5 * (gram + gram.conj().T)
    gram_values, gram_vectors = jnp.linalg.eigh(gram)
    largest = gram_values[-1]
    full_rank = gram_values[0] > relative_rank_tolerance * largest
    safe_scale = jnp.maximum(largest, jnp.finfo(gram_values.dtype).tiny)
    safe_values = jnp.maximum(
        gram_values, relative_rank_tolerance * safe_scale
    )
    inverse_sqrt = (
        gram_vectors / jnp.sqrt(safe_values)[None, :]
    ) @ gram_vectors.conj().T
    orthonormal = reference @ inverse_sqrt

    # Geometry at the current point is fixed while differentiating the
    # horizontal tangent map.  Derivatives of the whitening/gauge are vertical
    # and are removed by the projector.
    fixed_reference = jax.lax.stop_gradient(reference)
    fixed_inverse_sqrt = jax.lax.stop_gradient(inverse_sqrt)
    fixed_orthonormal = jax.lax.stop_gradient(orthonormal)
    feature_fn = lambda candidate: _realified_tangent_feature(
        candidate,
        reference=fixed_reference,
        orthonormal=fixed_orthonormal,
        inverse_sqrt_gram=fixed_inverse_sqrt,
        state_fn=state_fn,
    )
    jacobian = jax.jacrev(feature_fn)(flat)

    projected_hamiltonian = orthonormal.conj().T @ operator @ orthonormal
    loss = jnp.real(jnp.trace(projected_hamiltonian)) / rank
    residual = operator @ orthonormal - orthonormal @ projected_hamiltonian
    residual_norm = jnp.linalg.norm(residual)
    grassmann_variance = residual_norm**2 / rank
    force = (2.0 / jnp.sqrt(jnp.asarray(rank, dtype=flat.dtype))) * jnp.concatenate(
        (jnp.real(residual).reshape(-1), jnp.imag(residual).reshape(-1))
    )
    gradient = jacobian.T @ force

    # (J^T J + lambda I)^-1 J^T = J^T (J J^T + lambda I)^-1.
    # The feature-space system is only 2 * dim(H) * rank for exact-small
    # problems and gives a deterministic solve residual, unlike a truncated
    # parameter-space CG iteration.
    kernel = jacobian @ jacobian.T
    kernel = 0.5 * (kernel + kernel.T)
    regularized_kernel = kernel + damping * jnp.eye(
        kernel.shape[0], dtype=kernel.dtype
    )
    cholesky = jnp.linalg.cholesky(regularized_kernel)
    dual_solution = jax.scipy.linalg.cho_solve((cholesky, True), force)
    positive_solution = jacobian.T @ dual_solution
    direction = -positive_solution
    predicted_tangent = jacobian @ positive_solution
    linear_residual = (
        jacobian.T @ predicted_tangent
        + damping * positive_solution
        - gradient
    )
    relative_residual = jnp.linalg.norm(linear_residual) / jnp.maximum(
        jnp.linalg.norm(gradient), jnp.finfo(flat.dtype).tiny
    )
    canonical_norm = jnp.linalg.norm(predicted_tangent)
    directional_derivative = jnp.vdot(gradient, direction).real

    entries = reference.shape[0] * rank
    signed_feature = -predicted_tangent
    horizontal_tangent = signed_feature[:entries].reshape(reference.shape) + 1j * (
        signed_feature[entries:].reshape(reference.shape)
    )
    horizontality_error = jnp.linalg.norm(
        orthonormal.conj().T @ horizontal_tangent
    )
    force_norm_squared = jnp.vdot(force, force).real
    capture = jnp.vdot(force, predicted_tangent).real / jnp.maximum(
        force_norm_squared, jnp.finfo(flat.dtype).tiny
    )

    solve_converged = relative_residual <= 1.0e-8
    orthonormality_error = jnp.max(
        jnp.abs(
            orthonormal.conj().T @ orthonormal
            - jnp.eye(rank, dtype=orthonormal.dtype)
        )
    )
    hamiltonian_hermiticity_error = jnp.max(
        jnp.abs(operator - operator.conj().T)
    )
    finite = (
        jnp.isfinite(loss)
        & jnp.all(jnp.isfinite(direction))
        & jnp.all(jnp.isfinite(gradient))
        & jnp.isfinite(relative_residual)
        & jnp.isfinite(canonical_norm)
        & jnp.isfinite(directional_derivative)
        & jnp.isfinite(horizontality_error)
        & jnp.isfinite(capture)
        & jnp.isfinite(orthonormality_error)
        & jnp.isfinite(hamiltonian_hermiticity_error)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(grassmann_variance)
    )
    invalid = (
        ~full_rank
        | ~solve_converged
        | ~finite
        | (hamiltonian_hermiticity_error > 1.0e-10)
    )
    direction = jnp.where(invalid, jnp.full_like(direction, jnp.nan), direction)
    gradient = jnp.where(invalid, jnp.full_like(gradient, jnp.nan), gradient)
    return GrassmannEnergySRDirection(
        direction=direction,
        gradient=gradient,
        loss=loss,
        gradient_norm=jnp.linalg.norm(gradient),
        direction_norm=jnp.linalg.norm(direction),
        canonical_direction_norm=canonical_norm,
        directional_derivative=directional_derivative,
        relative_linear_residual=relative_residual,
        minimum_gram_eigenvalue=gram_values[0],
        gram_condition_number=largest
        / jnp.maximum(gram_values[0], jnp.finfo(gram_values.dtype).tiny),
        horizontality_error=horizontality_error,
        damped_tangent_capture_fraction=capture,
        linear_solve_converged=solve_converged,
        full_rank=full_rank,
        orthonormality_error=orthonormality_error,
        hamiltonian_hermiticity_error=hamiltonian_hermiticity_error,
        grassmann_residual_norm=residual_norm,
        grassmann_variance=grassmann_variance,
    )


__all__ = [
    "GrassmannEnergySRDirection",
    "GrassmannSRDirection",
    "exact_grassmann_energy_sr_direction",
    "exact_grassmann_sr_direction",
]
