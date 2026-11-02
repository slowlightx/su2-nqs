"""Matrix-free damped least squares using Golub--Kahan LSQR.

The public solver minimizes

``||A x - b||_2**2 + damping * ||x||_2**2``

without materializing either ``A`` or the normal matrix ``A.T @ A``.  The
operator is supplied only through a forward product and its Euclidean
adjoint, which makes the routine directly usable with JAX JVP/VJP closures.

Unlike a normal-equation CG diagnostic, the reported ``primal`` and
``augmented`` residuals are recomputed with the original operator.  The
normal residual is likewise the true stationarity residual
``A.T @ (A @ x - b) + damping * x``.  The native LSQR residual estimate is
reported separately.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array
LinearProduct = Callable[[Array], Array]


class DampedLeastSquaresResiduals(NamedTuple):
    """Exact residual norms for a damped least-squares iterate."""

    primal_norm: Array
    augmented_norm: Array
    normal_norm: Array


class DampedLSQRResult(NamedTuple):
    """Result and exact convergence history from :func:`damped_lsqr`.

    Histories include iteration zero and therefore have length
    ``max_iterations + 1``.  Entries after convergence repeat the converged
    value so that the result has static shapes under :func:`jax.jit`; only
    entries through ``iterations`` are logically active.
    """

    solution: Array
    iterations: Array
    converged: Array
    breakdown: Array
    primal_residual_norm: Array
    augmented_residual_norm: Array
    normal_residual_norm: Array
    normal_residual_scale: Array
    normal_residual_threshold: Array
    relative_normal_residual: Array
    lsqr_residual_estimate: Array
    primal_residual_history: Array
    augmented_residual_history: Array
    normal_residual_history: Array
    relative_normal_residual_history: Array
    lsqr_residual_estimate_history: Array


class _LSQRState(NamedTuple):
    solution: Array
    left_data: Array
    left_regularizer: Array
    right: Array
    search: Array
    alpha: Array
    rho_bar: Array
    phi_bar: Array
    iterations: Array
    active: Array
    converged: Array
    breakdown: Array
    primal_residual: Array
    augmented_residual: Array
    normal_residual: Array
    relative_normal_residual: Array


def _norm(vector: Array) -> Array:
    """Return a real Euclidean norm for real or complex arrays."""

    return jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(vector, vector)), 0.0))


def _joint_norm(first: Array, second: Array) -> Array:
    """Euclidean norm of two logically concatenated vectors."""

    squared = jnp.real(jnp.vdot(first, first)) + jnp.real(jnp.vdot(second, second))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def _safe_denominator(value: Array) -> Array:
    """Replace zero or non-finite normalization scalars by one."""

    return jnp.where(jnp.isfinite(value) & (value > 0.0), value, 1.0)


def _concrete_nonnegative(name: str, value: float) -> None:
    """Validate scalar options without forcing a traced JAX value concrete."""

    if isinstance(value, Real) and value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def damped_least_squares_residuals(
    matvec: LinearProduct,
    rmatvec: LinearProduct,
    rhs: Array,
    solution: Array,
    *,
    damping: float,
) -> DampedLeastSquaresResiduals:
    """Recompute true primal, augmented, and normal residual norms.

    ``rmatvec`` must be the Euclidean adjoint of ``matvec``.  For a complex
    operator this means the Hermitian adjoint.  A real-parameter complex
    problem should ordinarily be explicitly realified before calling this
    function, exactly as JAX does for a real-valued JVP/VJP pair.
    """

    rhs_array = jnp.asarray(rhs)
    solution_array = jnp.asarray(solution)
    image = jnp.asarray(matvec(solution_array))
    if image.shape != rhs_array.shape:
        raise ValueError("matvec(solution) must have the same shape as rhs")
    residual = image - rhs_array
    normal = jnp.asarray(rmatvec(residual)) + damping * solution_array
    if normal.shape != solution_array.shape:
        raise ValueError("rmatvec(residual) must have the same shape as solution")
    primal_norm = _norm(residual)
    solution_norm = _norm(solution_array)
    augmented_norm = jnp.sqrt(
        jnp.maximum(primal_norm * primal_norm + damping * solution_norm * solution_norm, 0.0)
    )
    return DampedLeastSquaresResiduals(
        primal_norm=primal_norm,
        augmented_norm=augmented_norm,
        normal_norm=_norm(normal),
    )


def damped_lsqr(
    matvec: LinearProduct,
    rmatvec: LinearProduct,
    rhs: Array,
    *,
    damping: float,
    initial_solution: Array | None = None,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    max_iterations: int = 200,
) -> DampedLSQRResult:
    """Solve a damped matrix-free least-squares problem with LSQR.

    The implementation applies ordinary Golub--Kahan LSQR to the implicit
    augmented system

    ``[A; sqrt(damping) I] x = [rhs; 0]``.

    No normal-equation product is used by the Krylov iteration.  Exact normal
    residuals are evaluated only as diagnostics and for the stopping test.
    Convergence means

    ``||A.T (A x - rhs) + damping*x|| <= max(atol, rtol*scale)``,

    where ``scale`` is the maximum of ``||A.T rhs||``, the true normal
    residual at the initial solution, and the smallest positive number of the
    working dtype.  Including the initial residual is essential for a warm
    start when ``rhs == 0``.

    The iteration and its result are compatible with :func:`jax.jit` when
    ``max_iterations`` and the operator output shapes are static.

    Args:
        matvec: Function implementing ``x -> A @ x``.
        rmatvec: Function implementing the Euclidean adjoint ``y -> A.T @ y``.
        rhs: One-dimensional right-hand side in the data space.
        damping: Non-negative coefficient multiplying ``||x||**2``.
        initial_solution: Optional starting point.  The augmented residual of
            this point is used as the LSQR right-hand side for the correction.
        relative_tolerance: Relative tolerance for the true normal residual.
        absolute_tolerance: Absolute tolerance for the true normal residual.
        max_iterations: Maximum number of Golub--Kahan steps.

    Returns:
        A :class:`DampedLSQRResult` with a static-shape residual history.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    _concrete_nonnegative("damping", damping)
    _concrete_nonnegative("relative_tolerance", relative_tolerance)
    _concrete_nonnegative("absolute_tolerance", absolute_tolerance)

    rhs_array = jnp.asarray(rhs)
    if rhs_array.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    adjoint_rhs = jnp.asarray(rmatvec(rhs_array))
    if adjoint_rhs.ndim != 1:
        raise ValueError("rmatvec(rhs) must be one-dimensional")
    if initial_solution is None:
        solution = jnp.zeros_like(adjoint_rhs)
    else:
        solution = jnp.asarray(initial_solution)
        if solution.shape != adjoint_rhs.shape:
            raise ValueError("initial_solution has the wrong shape")

    image = jnp.asarray(matvec(solution))
    if image.shape != rhs_array.shape:
        raise ValueError("matvec(initial_solution) must have the same shape as rhs")

    real_dtype = jnp.real(solution).dtype
    if not jnp.issubdtype(real_dtype, jnp.floating):
        raise TypeError("the parameter-space dtype must be floating or complex")
    damping_array = jnp.asarray(damping, dtype=real_dtype)
    relative_tolerance_array = jnp.asarray(relative_tolerance, dtype=real_dtype)
    absolute_tolerance_array = jnp.asarray(absolute_tolerance, dtype=real_dtype)
    square_root_damping = jnp.sqrt(damping_array)

    # LSQR is run on the correction to an optional initial solution.  Keeping
    # the two blocks separate avoids ever constructing the augmented matrix or
    # even a full augmented vector.
    left_data = rhs_array - image
    left_regularizer = -square_root_damping * solution
    beta = _joint_norm(left_data, left_regularizer)
    beta_denominator = _safe_denominator(beta)
    left_data = left_data / beta_denominator
    left_regularizer = left_regularizer / beta_denominator

    right_raw = jnp.asarray(rmatvec(left_data)) + square_root_damping * left_regularizer
    if right_raw.shape != solution.shape:
        raise ValueError("rmatvec output has the wrong parameter-space shape")
    alpha = _norm(right_raw)
    right = right_raw / _safe_denominator(alpha)

    initial_residuals = damped_least_squares_residuals(
        matvec,
        rmatvec,
        rhs_array,
        solution,
        damping=damping_array,
    )
    rhs_normal_norm = _norm(adjoint_rhs)
    normal_scale = jnp.maximum(
        jnp.maximum(rhs_normal_norm, initial_residuals.normal_norm),
        jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
    )
    threshold = jnp.maximum(
        absolute_tolerance_array,
        relative_tolerance_array * normal_scale,
    )
    finite_initial = (
        jnp.all(jnp.isfinite(solution))
        & jnp.isfinite(beta)
        & jnp.isfinite(alpha)
        & jnp.isfinite(initial_residuals.primal_norm)
        & jnp.isfinite(initial_residuals.augmented_norm)
        & jnp.isfinite(initial_residuals.normal_norm)
    )
    initially_converged = finite_initial & (initial_residuals.normal_norm <= threshold)
    initial_state = _LSQRState(
        solution=solution,
        left_data=left_data,
        left_regularizer=left_regularizer,
        right=right,
        search=right,
        alpha=alpha,
        rho_bar=alpha,
        phi_bar=beta,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        active=finite_initial & ~initially_converged,
        converged=initially_converged,
        breakdown=~finite_initial,
        primal_residual=initial_residuals.primal_norm,
        augmented_residual=initial_residuals.augmented_norm,
        normal_residual=initial_residuals.normal_norm,
        relative_normal_residual=initial_residuals.normal_norm / normal_scale,
    )

    def active_iteration(state: _LSQRState) -> _LSQRState:
        # One Golub--Kahan bidiagonalization step on the augmented operator.
        next_left_data_raw = matvec(state.right) - state.alpha * state.left_data
        next_left_regularizer_raw = (
            square_root_damping * state.right
            - state.alpha * state.left_regularizer
        )
        next_beta = _joint_norm(next_left_data_raw, next_left_regularizer_raw)
        beta_normalizer = _safe_denominator(next_beta)
        next_left_data = next_left_data_raw / beta_normalizer
        next_left_regularizer = next_left_regularizer_raw / beta_normalizer

        next_right_raw = (
            rmatvec(next_left_data)
            + square_root_damping * next_left_regularizer
            - next_beta * state.right
        )
        next_alpha = _norm(next_right_raw)
        next_right = next_right_raw / _safe_denominator(next_alpha)

        rho = jnp.hypot(state.rho_bar, next_beta)
        rho_denominator = _safe_denominator(rho)
        cosine = state.rho_bar / rho_denominator
        sine = next_beta / rho_denominator
        theta = sine * next_alpha
        next_rho_bar = -cosine * next_alpha
        phi = cosine * state.phi_bar
        next_phi_bar = sine * state.phi_bar
        next_solution = state.solution + (phi / rho_denominator) * state.search
        next_search = next_right - (theta / rho_denominator) * state.search

        residuals = damped_least_squares_residuals(
            matvec,
            rmatvec,
            rhs_array,
            next_solution,
            damping=damping_array,
        )
        finite_step = (
            jnp.isfinite(next_beta)
            & jnp.isfinite(next_alpha)
            & jnp.isfinite(rho)
            & (rho > 0.0)
            & jnp.all(jnp.isfinite(next_solution))
            & jnp.all(jnp.isfinite(next_right))
            & jnp.all(jnp.isfinite(next_search))
            & jnp.isfinite(residuals.primal_norm)
            & jnp.isfinite(residuals.augmented_norm)
            & jnp.isfinite(residuals.normal_norm)
        )
        converged = finite_step & (residuals.normal_norm <= threshold)
        attempted_iterations = state.iterations + jnp.asarray(1, dtype=jnp.int32)
        return _LSQRState(
            solution=jnp.where(finite_step, next_solution, state.solution),
            left_data=jnp.where(finite_step, next_left_data, state.left_data),
            left_regularizer=jnp.where(
                finite_step, next_left_regularizer, state.left_regularizer
            ),
            right=jnp.where(finite_step, next_right, state.right),
            search=jnp.where(finite_step, next_search, state.search),
            alpha=jnp.where(finite_step, next_alpha, state.alpha),
            rho_bar=jnp.where(finite_step, next_rho_bar, state.rho_bar),
            phi_bar=jnp.where(finite_step, next_phi_bar, state.phi_bar),
            iterations=attempted_iterations,
            active=finite_step & ~converged,
            converged=converged,
            breakdown=~finite_step,
            primal_residual=jnp.where(
                finite_step, residuals.primal_norm, state.primal_residual
            ),
            augmented_residual=jnp.where(
                finite_step, residuals.augmented_norm, state.augmented_residual
            ),
            normal_residual=jnp.where(
                finite_step, residuals.normal_norm, state.normal_residual
            ),
            relative_normal_residual=jnp.where(
                finite_step,
                residuals.normal_norm / normal_scale,
                state.relative_normal_residual,
            ),
        )

    def scan_step(
        state: _LSQRState, _: Array
    ) -> tuple[_LSQRState, tuple[Array, Array, Array, Array, Array]]:
        next_state = jax.lax.cond(state.active, active_iteration, lambda current: current, state)
        diagnostics = (
            next_state.primal_residual,
            next_state.augmented_residual,
            next_state.normal_residual,
            next_state.relative_normal_residual,
            jnp.abs(next_state.phi_bar),
        )
        return next_state, diagnostics

    final_state, scanned_history = jax.lax.scan(
        scan_step,
        initial_state,
        xs=jnp.arange(max_iterations, dtype=jnp.int32),
    )
    (
        primal_tail,
        augmented_tail,
        normal_tail,
        relative_normal_tail,
        estimate_tail,
    ) = scanned_history

    def prepend(initial: Array, tail: Array) -> Array:
        return jnp.concatenate((initial[None], tail), axis=0)

    return DampedLSQRResult(
        solution=final_state.solution,
        iterations=final_state.iterations,
        converged=final_state.converged,
        breakdown=final_state.breakdown,
        primal_residual_norm=final_state.primal_residual,
        augmented_residual_norm=final_state.augmented_residual,
        normal_residual_norm=final_state.normal_residual,
        normal_residual_scale=normal_scale,
        normal_residual_threshold=threshold,
        relative_normal_residual=final_state.relative_normal_residual,
        lsqr_residual_estimate=jnp.abs(final_state.phi_bar),
        primal_residual_history=prepend(
            initial_state.primal_residual, primal_tail
        ),
        augmented_residual_history=prepend(
            initial_state.augmented_residual, augmented_tail
        ),
        normal_residual_history=prepend(
            initial_state.normal_residual, normal_tail
        ),
        relative_normal_residual_history=prepend(
            initial_state.relative_normal_residual, relative_normal_tail
        ),
        lsqr_residual_estimate_history=prepend(
            jnp.abs(initial_state.phi_bar), estimate_tail
        ),
    )


__all__ = [
    "DampedLSQRResult",
    "DampedLeastSquaresResiduals",
    "damped_least_squares_residuals",
    "damped_lsqr",
]
