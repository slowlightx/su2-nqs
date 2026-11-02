"""Provenance-isolated physical-block PCG projection for sampled SR.

This module intentionally does not alter :mod:`su2_nqs.vmc.sampled_sr`.  It
uses the same centered complex teacher features, damping, candidate step
schedule, and optional energy backtracking, while exposing an explicit solver
control:

``identity``
    Ordinary unpreconditioned conjugate gradients.

``physical_block``
    PCG with the diagonal/small-block Fisher inverse from
    :mod:`su2_nqs.vmc.fisher_preconditioner`.

``physical_block_step``
    Apply that inverse once to the normal-equation right-hand side.  This is
    an explicitly approximate block-natural-gradient direction, not a claim
    to have solved the global damped normal equation.

``physical_block_deflated``
    PCG with the same local inverse plus a fixed-rank, direction-adapted
    Arnoldi coarse space.  The coarse Ritz solve restores global cross-block
    interference without constructing a parameter-square Fisher matrix.

``physical_block_deflated_refined``
    The same two-level inverse with periodic true-residual replacement.  Each
    refinement cycle restarts PCG from the explicitly recomputed residual,
    preventing a long recursive CG trajectory from silently losing useful
    conjugacy on an ill-conditioned sampled Fisher system.

``physical_block_deflated_gmres``
    A fixed restarted-GMRES audit of the same normal equation and two-level
    inverse.  Full Arnoldi orthogonalization makes this the solver-independent
    K1 check after long and iteratively refined PCG have both failed.

The normal operator remains a JVP/VJP product and is never materialized.  A
per-sample centered score matrix is formed only to estimate the Jacobi
preconditioner.  Every reported residual is recomputed with the original,
unpreconditioned normal operator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .fisher_preconditioner import (
    ChunkedFisherDiagnostics,
    FisherBlockSpec,
    FisherPreconditioner,
    build_chunked_fisher_preconditioner,
    physical_carrier_head_block_specs,
)
from .krylov_deflation import build_arnoldi_deflated_preconditioner


SolverControl = Literal[
    "identity",
    "physical_block",
    "physical_block_step",
    "physical_block_deflated",
    "physical_block_deflated_refined",
    "physical_block_deflated_gmres",
]


class PCGResult(NamedTuple):
    """Result of a matrix-free real symmetric positive-definite PCG solve."""

    solution: jax.Array
    iterations: jax.Array
    converged: jax.Array
    breakdown: jax.Array
    true_residual_norm: jax.Array
    true_relative_residual: jax.Array


class RefinedPCGResult(NamedTuple):
    """PCG result plus true residuals after iterative-refinement cycles."""

    result: PCGResult
    cycles: jax.Array
    true_relative_residual_history: jax.Array


class GMRESResult(NamedTuple):
    """Restarted-GMRES result with an independently recomputed residual."""

    result: PCGResult
    info: jax.Array


def centered_complex_log_distance(
    log_ratios: jax.Array,
    sample_weights: jax.Array,
) -> jax.Array:
    """Estimate the gauge-invariant complex-log distance on sampled support.

    ``log_ratios`` is ``log(psi_target / psi_base)`` on samples drawn from an
    arbitrary proposal, and ``sample_weights`` is proportional to the target
    Born density divided by that proposal density.  Removing the weighted
    complex mean eliminates normalization and global phase.  The factor two
    matches the real/imaginary feature convention used by sampled SR.
    """

    ratios = jnp.asarray(log_ratios)
    weights = jnp.asarray(sample_weights)
    if ratios.ndim != 1 or weights.ndim != 1 or ratios.shape != weights.shape:
        raise ValueError("log ratios and sample weights must be matching vectors")
    if len(ratios) < 1:
        raise ValueError("sampled complex-log distance requires nonempty support")
    if not jnp.issubdtype(ratios.dtype, jnp.complexfloating):
        ratios = ratios.astype(jnp.result_type(ratios.dtype, jnp.complex64))
    real_dtype = jnp.real(ratios).dtype
    weights = weights.astype(real_dtype)
    total = jnp.sum(weights)
    normalized = weights / jnp.maximum(total, jnp.finfo(real_dtype).tiny)
    centered = ratios - jnp.sum(normalized * ratios)
    return jnp.sqrt(2.0 * jnp.sum(normalized * jnp.abs(centered) ** 2))


def preconditioned_conjugate_gradient(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    preconditioner: Callable[[jax.Array], jax.Array] | None = None,
    tolerance: float = 1.0e-5,
    absolute_tolerance: float = 0.0,
    max_iterations: int = 30,
) -> PCGResult:
    """Solve ``A x = rhs`` by PCG and recompute the true normal residual.

    ``preconditioner`` applies an approximate inverse, not its factor.  The
    stopping criterion is
    ``||rhs - A x|| <= max(absolute_tolerance, tolerance * ||rhs||)``.
    Damping in the caller's normal operator is expected to make it positive
    definite.  Non-positive curvature or a non-SPD preconditioner is reported
    as ``breakdown`` rather than silently producing a non-finite direction.
    """

    if tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("PCG tolerances must be non-negative")
    if max_iterations < 1:
        raise ValueError("PCG max_iterations must be positive")
    vector = jnp.asarray(rhs)
    if not jnp.issubdtype(vector.dtype, jnp.floating):
        raise TypeError("PCG currently requires a real floating right-hand side")
    apply_preconditioner = (
        (lambda value: value) if preconditioner is None else preconditioner
    )
    solution = jnp.zeros_like(vector)
    residual = vector - matvec(solution)
    preconditioned = apply_preconditioner(residual)
    direction = preconditioned
    residual_product = jnp.vdot(residual, preconditioned).real
    rhs_norm = jnp.linalg.norm(vector)
    residual_norm = jnp.linalg.norm(residual)
    threshold = jnp.maximum(
        jnp.asarray(absolute_tolerance, dtype=vector.dtype),
        jnp.asarray(tolerance, dtype=vector.dtype) * rhs_norm,
    )
    initial_breakdown = (
        ~jnp.isfinite(residual_product)
        | ~jnp.isfinite(residual_norm)
        | (residual_product < 0.0)
    ) & (residual_norm > threshold)

    # solution, residual, preconditioned residual, direction, r.T M r,
    # residual norm, iterations, breakdown
    initial_state = (
        solution,
        residual,
        preconditioned,
        direction,
        residual_product,
        residual_norm,
        jnp.asarray(0, dtype=jnp.int32),
        initial_breakdown,
    )

    def continue_iteration(state):
        return (
            (state[6] < max_iterations)
            & (state[5] > threshold)
            & ~state[7]
        )

    def iteration(state):
        (
            current_solution,
            current_residual,
            _,
            current_direction,
            current_residual_product,
            _,
            iterations,
            previous_breakdown,
        ) = state
        product = matvec(current_direction)
        curvature = jnp.vdot(current_direction, product).real
        valid_curvature = (
            jnp.isfinite(curvature)
            & (curvature > 0.0)
            & jnp.isfinite(current_residual_product)
            & (current_residual_product >= 0.0)
        )
        safe_curvature = jnp.where(valid_curvature, curvature, 1.0)
        step = jnp.where(
            valid_curvature, current_residual_product / safe_curvature, 0.0
        )
        next_solution = current_solution + step * current_direction
        next_residual = current_residual - step * product
        next_residual_norm = jnp.linalg.norm(next_residual)
        next_preconditioned = apply_preconditioner(next_residual)
        next_residual_product = jnp.vdot(
            next_residual, next_preconditioned
        ).real
        has_converged = next_residual_norm <= threshold
        valid_next = (
            jnp.isfinite(next_residual_norm)
            & jnp.isfinite(next_residual_product)
            & ((next_residual_product >= 0.0) | has_converged)
        )
        safe_residual_product = jnp.where(
            current_residual_product > 0.0, current_residual_product, 1.0
        )
        beta = jnp.where(
            valid_curvature & valid_next & (current_residual_product > 0.0),
            next_residual_product / safe_residual_product,
            0.0,
        )
        next_direction = next_preconditioned + beta * current_direction
        breakdown = previous_breakdown | ~valid_curvature | ~valid_next
        return (
            next_solution,
            next_residual,
            next_preconditioned,
            next_direction,
            next_residual_product,
            next_residual_norm,
            iterations + 1,
            breakdown,
        )

    final_state = jax.lax.while_loop(
        continue_iteration, iteration, initial_state
    )
    final_solution = final_state[0]
    true_residual = vector - matvec(final_solution)
    true_residual_norm = jnp.linalg.norm(true_residual)
    tiny = jnp.finfo(vector.dtype).tiny
    true_relative_residual = true_residual_norm / jnp.maximum(rhs_norm, tiny)
    converged = true_residual_norm <= threshold
    return PCGResult(
        solution=final_solution,
        iterations=final_state[6],
        converged=converged,
        breakdown=final_state[7],
        true_residual_norm=true_residual_norm,
        true_relative_residual=true_relative_residual,
    )


def iteratively_refined_preconditioned_conjugate_gradient(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    preconditioner: Callable[[jax.Array], jax.Array] | None = None,
    tolerance: float = 1.0e-5,
    absolute_tolerance: float = 0.0,
    iterations_per_cycle: int = 100,
    maximum_cycles: int = 6,
) -> RefinedPCGResult:
    """Solve by PCG cycles separated by explicit true-residual replacement.

    Each inner solve has fixed depth; the original relative/absolute stopping
    threshold is applied to the true residual at cycle boundaries.  The
    accumulated solution and its residual are recomputed with ``matvec`` at
    every boundary.  This is iterative refinement of one fixed linear system,
    not a damping, model, or preconditioner scan.
    """

    if tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("refined PCG tolerances must be non-negative")
    if iterations_per_cycle < 1 or maximum_cycles < 1:
        raise ValueError("refined PCG cycle sizes must be positive")
    vector = jnp.asarray(rhs)
    if vector.ndim != 1 or not jnp.issubdtype(vector.dtype, jnp.floating):
        raise TypeError("refined PCG requires a flat real floating right-hand side")
    rhs_norm = jnp.linalg.norm(vector)
    threshold = jnp.maximum(
        jnp.asarray(absolute_tolerance, dtype=vector.dtype),
        jnp.asarray(tolerance, dtype=vector.dtype) * rhs_norm,
    )
    history = jnp.zeros((maximum_cycles,), dtype=vector.dtype)
    initial_residual_norm = rhs_norm
    initial_breakdown = ~jnp.isfinite(rhs_norm)
    # solution, residual norm, total iterations, breakdown, cycles, history
    initial_state = (
        jnp.zeros_like(vector),
        initial_residual_norm,
        jnp.asarray(0, dtype=jnp.int32),
        initial_breakdown,
        jnp.asarray(0, dtype=jnp.int32),
        history,
    )

    def refinement_cycle(index, state):
        solution, residual_norm, total_iterations, breakdown, cycles, values = state
        active = (residual_norm > threshold) & ~breakdown

        def perform(_):
            residual = vector - matvec(solution)
            correction = preconditioned_conjugate_gradient(
                matvec,
                residual,
                preconditioner=preconditioner,
                tolerance=0.0,
                absolute_tolerance=0.0,
                max_iterations=iterations_per_cycle,
            )
            candidate = solution + correction.solution
            true_residual = vector - matvec(candidate)
            true_norm = jnp.linalg.norm(true_residual)
            candidate_breakdown = (
                breakdown
                | correction.breakdown
                | ~jnp.isfinite(true_norm)
            )
            candidate_values = values.at[index].set(
                true_norm / jnp.maximum(rhs_norm, jnp.finfo(vector.dtype).tiny)
            )
            return (
                candidate,
                true_norm,
                total_iterations + correction.iterations,
                candidate_breakdown,
                cycles + 1,
                candidate_values,
            )

        return jax.lax.cond(active, perform, lambda _: state, operand=None)

    final_state = jax.lax.fori_loop(
        0, maximum_cycles, refinement_cycle, initial_state
    )
    final_solution = final_state[0]
    true_residual = vector - matvec(final_solution)
    true_residual_norm = jnp.linalg.norm(true_residual)
    tiny = jnp.finfo(vector.dtype).tiny
    result = PCGResult(
        solution=final_solution,
        iterations=final_state[2],
        converged=true_residual_norm <= threshold,
        breakdown=final_state[3],
        true_residual_norm=true_residual_norm,
        true_relative_residual=true_residual_norm / jnp.maximum(rhs_norm, tiny),
    )
    return RefinedPCGResult(
        result=result,
        cycles=final_state[4],
        true_relative_residual_history=final_state[5],
    )


def preconditioned_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    preconditioner: Callable[[jax.Array], jax.Array] | None = None,
    tolerance: float = 1.0e-5,
    absolute_tolerance: float = 0.0,
    restart: int = 64,
    maximum_cycles: int = 8,
) -> GMRESResult:
    """Apply fixed-depth restarted GMRES and recompute the true residual.

    The internal tolerance is zero so that a preconditioned-residual estimate
    cannot terminate the audit before the declared Krylov budget.  Acceptance
    uses only ``||b - A x||`` in the original unpreconditioned coordinates.
    """

    if tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("GMRES tolerances must be non-negative")
    if restart < 1 or maximum_cycles < 1:
        raise ValueError("GMRES restart and maximum_cycles must be positive")
    vector = jnp.asarray(rhs)
    if vector.ndim != 1 or not jnp.issubdtype(vector.dtype, jnp.floating):
        raise TypeError("GMRES requires a flat real floating right-hand side")
    solution, info = jax.scipy.sparse.linalg.gmres(
        matvec,
        vector,
        tol=0.0,
        atol=0.0,
        restart=restart,
        maxiter=maximum_cycles,
        M=preconditioner,
        solve_method="incremental",
    )
    true_residual = vector - matvec(solution)
    true_residual_norm = jnp.linalg.norm(true_residual)
    rhs_norm = jnp.linalg.norm(vector)
    threshold = jnp.maximum(
        jnp.asarray(absolute_tolerance, dtype=vector.dtype),
        jnp.asarray(tolerance, dtype=vector.dtype) * rhs_norm,
    )
    breakdown = (
        ~jnp.all(jnp.isfinite(solution))
        | ~jnp.isfinite(true_residual_norm)
        | (jnp.asarray(info) < 0)
    )
    result = PCGResult(
        solution=solution,
        iterations=jnp.asarray(restart * maximum_cycles, dtype=jnp.int32),
        converged=(true_residual_norm <= threshold) & ~breakdown,
        breakdown=breakdown,
        true_residual_norm=true_residual_norm,
        true_relative_residual=true_residual_norm
        / jnp.maximum(rhs_norm, jnp.finfo(vector.dtype).tiny),
    )
    return GMRESResult(result=result, info=jnp.asarray(info))


def log_derivative_scores(
    params: object,
    samples: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> object:
    """Return uncentered complex log-derivative scores for one sample chunk."""

    flat, unravel = ravel_pytree(params)
    sample_values = jnp.asarray(samples)

    def real_imag_log(values, sample):
        log_value = log_psi_fn(unravel(values), sample)
        return jnp.stack((jnp.real(log_value), jnp.imag(log_value)))

    real_imag_jacobian = jax.vmap(
        jax.jacrev(real_imag_log, argnums=0), in_axes=(None, 0)
    )(flat, sample_values)
    score_matrix = (
        real_imag_jacobian[:, 0, :]
        + 1j * real_imag_jacobian[:, 1, :]
    )
    parameter_leaves, structure = jax.tree_util.tree_flatten(params)
    score_leaves = []
    offset = 0
    for parameter in parameter_leaves:
        size = int(np.prod(parameter.shape, dtype=np.int64))
        score_leaves.append(
            jnp.reshape(
                score_matrix[:, offset : offset + size],
                (sample_values.shape[0],) + parameter.shape,
            )
        )
        offset += size
    if offset != score_matrix.shape[1]:
        raise AssertionError("score unflattening did not consume every parameter")
    return jax.tree_util.tree_unflatten(structure, score_leaves)


def centered_log_derivative_scores(
    params: object,
    samples: jax.Array,
    sample_weights: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> object:
    """Return the weighted-centered complex score pytree for one batch."""

    weights = jnp.asarray(sample_weights)
    if samples.shape[0] != weights.shape[0] or weights.ndim != 1:
        raise ValueError("samples and sample weights disagree")
    weights = weights.astype(ravel_pytree(params)[0].dtype)
    normalized_weights = weights / jnp.maximum(
        jnp.sum(weights), jnp.finfo(weights.dtype).tiny
    )
    scores = log_derivative_scores(params, samples, log_psi_fn)
    return jax.tree_util.tree_map(
        lambda score: score
        - jnp.sum(
            normalized_weights.reshape(
                (len(weights),) + (1,) * (score.ndim - 1)
            )
            * score,
            axis=0,
            keepdims=True,
        ),
        scores,
    )


def train_preconditioned_sr_projection(
    params: object,
    key: jax.Array,
    *,
    sample_pool: jax.Array,
    pool_importance_weights: jax.Array,
    pool_base_logs: jax.Array,
    pool_target_log_ratios: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    batch_size: int,
    solver_control: SolverControl,
    block_specs: tuple[FisherBlockSpec, ...] | None = None,
    score_chunk_size: int = 32,
    damping: float = 1.0e-3,
    step_size: float = 1.0,
    cg_tolerance: float = 1.0e-5,
    cg_absolute_tolerance: float = 0.0,
    cg_max_iterations: int = 30,
    deflation_rank: int = 0,
    refinement_cycles: int = 0,
    gmres_restart: int = 0,
    gmres_cycles: int = 0,
    backtracking_steps: int = 6,
    maximum_feature_distance: float | None = None,
    acceptance_score_fn: Callable[[object], float] | None = None,
    random_batches: bool = True,
) -> tuple[object, list[dict[str, object]]]:
    """Project a complex teacher by identity-CG or physical-block PCG.

    Apart from the linear solver preconditioner, feature construction and
    backtracking semantics match the existing sampled SR projection.  If
    ``maximum_feature_distance`` is supplied, the full proposed direction is
    first scaled by its linearized centered-complex-log displacement and every
    nonlinear candidate is then required to remain inside the same output
    ball.  The physical control rebuilds its Fisher block inverse at the
    current point and batch before each solve.  ``identity`` is the required
    unpreconditioned control and never constructs score blocks.
    """

    if steps < 1 or batch_size < 1 or len(sample_pool) < 1:
        raise ValueError("steps, batch size, and sample pool must be positive")
    if damping <= 0.0 or step_size <= 0.0:
        raise ValueError("damping and step size must be positive")
    if backtracking_steps < 1:
        raise ValueError("backtracking_steps must be positive")
    if maximum_feature_distance is not None and (
        not np.isfinite(maximum_feature_distance)
        or maximum_feature_distance <= 0.0
    ):
        raise ValueError("maximum_feature_distance must be positive and finite")
    if score_chunk_size < 1:
        raise ValueError("score_chunk_size must be positive")
    if solver_control not in (
        "identity",
        "physical_block",
        "physical_block_step",
        "physical_block_deflated",
        "physical_block_deflated_refined",
        "physical_block_deflated_gmres",
    ):
        raise ValueError(
            "solver_control must be 'identity', 'physical_block', or "
            "'physical_block_step', 'physical_block_deflated', or "
            "'physical_block_deflated_refined', or "
            "'physical_block_deflated_gmres'"
        )
    if solver_control == "identity" and block_specs:
        raise ValueError("identity CG may not receive physical block specs")
    flat, unravel = ravel_pytree(params)
    if not jnp.issubdtype(flat.dtype, jnp.floating):
        raise TypeError("SR projection requires real model parameters")
    if (
        solver_control in (
            "physical_block",
            "physical_block_step",
            "physical_block_deflated",
            "physical_block_deflated_refined",
            "physical_block_deflated_gmres",
        )
        and block_specs is None
    ):
        block_specs = physical_carrier_head_block_specs(params)
    if solver_control in (
        "physical_block_deflated",
        "physical_block_deflated_refined",
        "physical_block_deflated_gmres",
    ):
        if deflation_rank < 1:
            raise ValueError("physical_block_deflated requires positive deflation_rank")
        if deflation_rank > flat.size:
            raise ValueError("deflation_rank may not exceed parameter count")
    elif deflation_rank != 0:
        raise ValueError("deflation_rank is only valid for deflated controls")
    if solver_control == "physical_block_deflated_refined":
        if refinement_cycles < 1:
            raise ValueError("refined deflation requires positive refinement_cycles")
    elif refinement_cycles != 0:
        raise ValueError("refinement_cycles is only valid for the refined control")
    if solver_control == "physical_block_deflated_gmres":
        if gmres_restart < 1 or gmres_cycles < 1:
            raise ValueError("deflated GMRES requires positive restart and cycles")
    elif gmres_restart != 0 or gmres_cycles != 0:
        raise ValueError("GMRES controls are only valid for the GMRES control")

    def centered_features(values, samples, importance):
        weights = importance / jnp.maximum(
            jnp.sum(importance), jnp.finfo(importance.dtype).tiny
        )
        square_root_weights = jnp.sqrt(2.0 * weights)
        logs = jax.vmap(
            lambda sample: log_psi_fn(unravel(values), sample)
        )(samples)
        centered = logs - jnp.sum(weights * logs)
        return jnp.concatenate(
            (
                square_root_weights * jnp.real(centered),
                square_root_weights * jnp.imag(centered),
            )
        )

    def target_features(importance, base_logs, target_log_ratios):
        weights = importance / jnp.maximum(
            jnp.sum(importance), jnp.finfo(importance.dtype).tiny
        )
        square_root_weights = jnp.sqrt(2.0 * weights)
        target_logs = base_logs + target_log_ratios
        centered = target_logs - jnp.sum(weights * target_logs)
        return jnp.concatenate(
            (
                square_root_weights * jnp.real(centered),
                square_root_weights * jnp.imag(centered),
            )
        )

    def solve_and_candidates(
        values,
        samples,
        importance,
        base_logs,
        target_log_ratios,
        physical_preconditioner: FisherPreconditioner | None,
    ):
        desired_features = target_features(
            importance, base_logs, target_log_ratios
        )

        def features(candidate):
            return centered_features(candidate, samples, importance)

        current_features, pullback = jax.vjp(features, values)
        residual = desired_features - current_features
        rhs = pullback(residual)[0]

        def normal_product(vector):
            _, tangent = jax.jvp(features, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        if physical_preconditioner is None:
            apply_inverse = None
        else:
            def apply_inverse(vector):
                tree_vector = unravel(vector)
                applied = physical_preconditioner.apply(tree_vector)
                return ravel_pytree(applied)[0]

        deflation_active_rank = jnp.asarray(0, dtype=jnp.int32)
        deflation_minimum_eigenvalue = jnp.asarray(0.0, dtype=values.dtype)
        deflation_maximum_eigenvalue = jnp.asarray(0.0, dtype=values.dtype)
        deflation_stored_matrix_elements = jnp.asarray(0, dtype=jnp.int64)
        if solver_control in (
            "physical_block_deflated",
            "physical_block_deflated_refined",
            "physical_block_deflated_gmres",
        ):
            assert apply_inverse is not None
            deflated = build_arnoldi_deflated_preconditioner(
                normal_product,
                rhs,
                apply_inverse,
                rank=deflation_rank,
            )
            base_inverse = apply_inverse

            def apply_inverse(vector):
                return deflated.apply(vector, base_inverse)

            deflation_active_rank = deflated.active_rank
            deflation_minimum_eigenvalue = (
                deflated.minimum_coarse_eigenvalue
            )
            deflation_maximum_eigenvalue = (
                deflated.maximum_coarse_eigenvalue
            )
            deflation_stored_matrix_elements = jnp.asarray(
                deflated.stored_matrix_elements, dtype=jnp.int64
            )

        refinement_cycles_used = jnp.asarray(0, dtype=jnp.int32)
        refinement_residual_history = jnp.zeros(
            (refinement_cycles,), dtype=values.dtype
        )
        gmres_info = jnp.asarray(0, dtype=jnp.int32)
        if solver_control == "physical_block_step":
            assert apply_inverse is not None
            solution = apply_inverse(rhs)
            true_residual = rhs - normal_product(solution)
            true_residual_norm = jnp.linalg.norm(true_residual)
            rhs_norm_for_gate = jnp.linalg.norm(rhs)
            threshold = jnp.maximum(
                jnp.asarray(cg_absolute_tolerance, dtype=values.dtype),
                jnp.asarray(cg_tolerance, dtype=values.dtype) * rhs_norm_for_gate,
            )
            cg_result = PCGResult(
                solution=solution,
                iterations=jnp.asarray(1, dtype=jnp.int32),
                converged=true_residual_norm <= threshold,
                breakdown=jnp.asarray(False),
                true_residual_norm=true_residual_norm,
                true_relative_residual=true_residual_norm
                / jnp.maximum(rhs_norm_for_gate, jnp.finfo(values.dtype).tiny),
            )
        elif solver_control == "physical_block_deflated_refined":
            refined_result = iteratively_refined_preconditioned_conjugate_gradient(
                normal_product,
                rhs,
                preconditioner=apply_inverse,
                tolerance=cg_tolerance,
                absolute_tolerance=cg_absolute_tolerance,
                iterations_per_cycle=cg_max_iterations,
                maximum_cycles=refinement_cycles,
            )
            cg_result = refined_result.result
            refinement_cycles_used = refined_result.cycles
            refinement_residual_history = (
                refined_result.true_relative_residual_history
            )
        elif solver_control == "physical_block_deflated_gmres":
            gmres_result = preconditioned_gmres(
                normal_product,
                rhs,
                preconditioner=apply_inverse,
                tolerance=cg_tolerance,
                absolute_tolerance=cg_absolute_tolerance,
                restart=gmres_restart,
                maximum_cycles=gmres_cycles,
            )
            cg_result = gmres_result.result
            gmres_info = gmres_result.info
        else:
            cg_result = preconditioned_conjugate_gradient(
                normal_product,
                rhs,
                preconditioner=apply_inverse,
                tolerance=cg_tolerance,
                absolute_tolerance=cg_absolute_tolerance,
                max_iterations=cg_max_iterations,
            )
        _, raw_feature_tangent = jax.jvp(
            features, (values,), (cg_result.solution,)
        )
        raw_full_step_feature_norm = step_size * jnp.linalg.norm(
            raw_feature_tangent
        )
        if maximum_feature_distance is None:
            trust_scale = jnp.asarray(1.0, dtype=values.dtype)
        else:
            trust_scale = jnp.minimum(
                1.0,
                jnp.asarray(maximum_feature_distance, dtype=values.dtype)
                / jnp.maximum(
                    raw_full_step_feature_norm, jnp.finfo(values.dtype).tiny
                ),
            )
        trusted_direction = trust_scale * cg_result.solution
        candidate_steps = step_size * 0.5 ** jnp.arange(backtracking_steps)
        candidate_steps = jnp.concatenate((jnp.zeros((1,)), candidate_steps))
        candidate_values = (
            values[None, :]
            + candidate_steps[:, None] * trusted_direction[None, :]
        )
        candidate_features = jax.vmap(features)(candidate_values)
        candidate_losses = jnp.mean(
            (desired_features[None, :] - candidate_features) ** 2, axis=1
        )
        candidate_feature_distances = jnp.linalg.norm(
            candidate_features - current_features[None, :], axis=1
        )
        return (
            candidate_values,
            candidate_losses,
            candidate_feature_distances,
            jnp.mean(residual**2),
            jnp.linalg.norm(rhs),
            jnp.linalg.norm(cg_result.solution),
            trust_scale,
            raw_full_step_feature_norm,
            cg_result.iterations,
            cg_result.converged,
            cg_result.breakdown,
            cg_result.true_residual_norm,
            cg_result.true_relative_residual,
            deflation_active_rank,
            deflation_minimum_eigenvalue,
            deflation_maximum_eigenvalue,
            deflation_stored_matrix_elements,
            refinement_cycles_used,
            refinement_residual_history,
            gmres_info,
        )

    if solver_control == "identity":
        compiled_solve = jax.jit(
            lambda values, samples, importance, base_logs, target_ratios: (
                solve_and_candidates(
                    values,
                    samples,
                    importance,
                    base_logs,
                    target_ratios,
                    None,
                )
            )
        )
    else:
        compiled_solve = jax.jit(solve_and_candidates)
        compiled_scores = jax.jit(
            lambda values, samples: log_derivative_scores(
                unravel(values), samples, log_psi_fn
            )
        )

    history: list[dict[str, object]] = []
    pool_size = len(sample_pool)
    accepted_score = (
        float(acceptance_score_fn(unravel(flat)))
        if acceptance_score_fn is not None
        else None
    )
    for step in range(1, steps + 1):
        if random_batches:
            key, subkey = jax.random.split(key)
            indices = jax.random.randint(subkey, (batch_size,), 0, pool_size)
        else:
            if batch_size != pool_size:
                raise ValueError(
                    "deterministic batches require batch_size == pool size"
                )
            indices = jnp.arange(pool_size)
        samples = sample_pool[indices]
        importance = pool_importance_weights[indices]
        base_logs = pool_base_logs[indices]
        target_ratios = pool_target_log_ratios[indices]
        score_seconds = 0.0
        preconditioner_seconds = 0.0
        preconditioner: FisherPreconditioner | None = None
        accumulator_diagnostics: ChunkedFisherDiagnostics | None = None
        if solver_control in (
            "physical_block",
            "physical_block_step",
            "physical_block_deflated",
            "physical_block_deflated_refined",
            "physical_block_deflated_gmres",
        ):
            preconditioner_started = time.perf_counter()
            score_seconds_holder = [0.0]

            def score_chunks():
                for start in range(0, batch_size, score_chunk_size):
                    stop = min(start + score_chunk_size, batch_size)
                    score_started = time.perf_counter()
                    chunk_scores = compiled_scores(flat, samples[start:stop])
                    jax.block_until_ready(chunk_scores)
                    score_seconds_holder[0] += time.perf_counter() - score_started
                    yield chunk_scores, importance[start:stop]

            preconditioner, accumulator_diagnostics = (
                build_chunked_fisher_preconditioner(
                    unravel(flat),
                    score_chunks(),
                    damping=damping,
                    block_specs=block_specs or (),
                )
            )
            jax.block_until_ready(preconditioner)
            total_preconditioner_seconds = (
                time.perf_counter() - preconditioner_started
            )
            score_seconds = score_seconds_holder[0]
            preconditioner_seconds = max(
                total_preconditioner_seconds - score_seconds, 0.0
            )

        previous_flat = flat
        solve_started = time.perf_counter()
        if solver_control == "identity":
            solve_outputs = compiled_solve(
                flat, samples, importance, base_logs, target_ratios
            )
        else:
            assert preconditioner is not None
            solve_outputs = compiled_solve(
                flat,
                samples,
                importance,
                base_logs,
                target_ratios,
                preconditioner,
            )
        (
            candidate_flats,
            candidate_losses,
            candidate_feature_distances,
            loss_before,
            rhs_norm,
            raw_direction_norm,
            trust_scale,
            raw_full_step_feature_norm,
            cg_iterations,
            cg_converged,
            cg_breakdown,
            true_normal_residual_norm,
            true_relative_normal_residual,
            deflation_active_rank,
            deflation_minimum_eigenvalue,
            deflation_maximum_eigenvalue,
            deflation_stored_matrix_elements,
            refinement_cycles_used,
            refinement_residual_history,
            gmres_info,
        ) = solve_outputs
        jax.block_until_ready(candidate_losses)
        solve_seconds = time.perf_counter() - solve_started
        if not all(
            np.isfinite(float(value))
            for value in (
                loss_before,
                rhs_norm,
                raw_direction_norm,
                trust_scale,
                raw_full_step_feature_norm,
                true_normal_residual_norm,
                true_relative_normal_residual,
            )
        ):
            raise FloatingPointError("non-finite PCG projection diagnostic")

        accepted = True
        candidate_score = None
        if acceptance_score_fn is not None:
            accepted = False
            selected = 0
            evaluations = 0
            for candidate_index in range(1, len(candidate_flats)):
                if float(candidate_losses[candidate_index]) > float(loss_before):
                    continue
                if (
                    maximum_feature_distance is not None
                    and float(candidate_feature_distances[candidate_index])
                    > maximum_feature_distance * (1.0 + 1.0e-10)
                ):
                    continue
                evaluations += 1
                score = float(
                    acceptance_score_fn(unravel(candidate_flats[candidate_index]))
                )
                candidate_score = (
                    score if candidate_score is None else min(candidate_score, score)
                )
                if score <= accepted_score:
                    accepted = True
                    selected = candidate_index
                    accepted_score = score
                    candidate_score = score
                    break
            flat = candidate_flats[selected] if accepted else previous_flat
        else:
            eligible_losses = candidate_losses
            if maximum_feature_distance is not None:
                eligible_losses = jnp.where(
                    candidate_feature_distances
                    <= maximum_feature_distance * (1.0 + 1.0e-10),
                    candidate_losses,
                    jnp.inf,
                )
            selected = int(jnp.argmin(eligible_losses))
            flat = candidate_flats[selected]
            evaluations = 0
        effective_step = (
            step_size * 0.5 ** (selected - 1) if selected > 0 else 0.0
        )
        loss_after = candidate_losses[selected] if accepted else loss_before
        history.append(
            {
                "step": float(step),
                "solver_control": solver_control,
                "feature_loss_before": float(loss_before),
                "feature_loss_after": float(loss_after),
                "rhs_norm": float(rhs_norm),
                "direction_norm": float(raw_direction_norm * trust_scale),
                "raw_direction_norm": float(raw_direction_norm),
                "output_trust_scale": float(trust_scale),
                "raw_full_step_feature_norm": float(
                    raw_full_step_feature_norm
                ),
                "maximum_feature_distance": (
                    float(maximum_feature_distance)
                    if maximum_feature_distance is not None
                    else None
                ),
                "selected_feature_distance": float(
                    candidate_feature_distances[selected]
                ),
                "cg_iterations": int(cg_iterations),
                "cg_converged": bool(cg_converged),
                "cg_breakdown": bool(cg_breakdown),
                "true_normal_residual_norm": float(true_normal_residual_norm),
                "true_relative_normal_residual": float(
                    true_relative_normal_residual
                ),
                "cg_relative_residual": float(true_relative_normal_residual),
                "deflation_rank": int(deflation_rank),
                "deflation_active_rank": int(deflation_active_rank),
                "deflation_minimum_coarse_eigenvalue": float(
                    deflation_minimum_eigenvalue
                ),
                "deflation_maximum_coarse_eigenvalue": float(
                    deflation_maximum_eigenvalue
                ),
                "deflation_stored_matrix_elements": int(
                    deflation_stored_matrix_elements
                ),
                "refinement_cycles": int(refinement_cycles_used),
                "refinement_true_relative_residuals": [
                    float(value)
                    for value in np.asarray(refinement_residual_history)[
                        : int(refinement_cycles_used)
                    ]
                ],
                "gmres_restart": int(gmres_restart),
                "gmres_cycles": int(gmres_cycles),
                "gmres_info": int(gmres_info),
                "effective_step_size": float(effective_step),
                "accepted": bool(accepted),
                "energy_backtracking_evaluations": int(evaluations),
                "score_construction_seconds": float(score_seconds),
                "preconditioner_construction_seconds": float(
                    preconditioner_seconds
                ),
                "solve_and_candidate_seconds": float(solve_seconds),
                "preconditioner_stored_matrix_elements": (
                    int(preconditioner.stored_matrix_elements)
                    if preconditioner is not None
                    else 0
                ),
                "preconditioner_block_sizes": (
                    {
                        name: [int(count), int(size)]
                        for name, (count, size) in preconditioner.block_sizes.items()
                    }
                    if preconditioner is not None
                    else {}
                ),
                "preconditioner_score_chunk_shape": (
                    list(accumulator_diagnostics.maximum_score_chunk_shape)
                    if accumulator_diagnostics is not None
                    else [0, 0]
                ),
                "preconditioner_score_chunk_bytes": (
                    int(accumulator_diagnostics.maximum_score_chunk_bytes)
                    if accumulator_diagnostics is not None
                    else 0
                ),
                "preconditioner_real_imag_jacobian_shape": (
                    list(
                        accumulator_diagnostics.maximum_real_imag_jacobian_shape
                    )
                    if accumulator_diagnostics is not None
                    else [0, 0, 0]
                ),
                "preconditioner_estimated_peak_score_workspace_bytes": (
                    int(
                        accumulator_diagnostics.estimated_peak_score_workspace_bytes
                    )
                    if accumulator_diagnostics is not None
                    else 0
                ),
                "preconditioner_largest_block_dimension": (
                    int(accumulator_diagnostics.largest_block_dimension)
                    if accumulator_diagnostics is not None
                    else 0
                ),
                **(
                    {"acceptance_score": float(candidate_score)}
                    if candidate_score is not None
                    else {}
                ),
            }
        )
    return unravel(flat), history


__all__ = [
    "PCGResult",
    "RefinedPCGResult",
    "GMRESResult",
    "SolverControl",
    "centered_log_derivative_scores",
    "log_derivative_scores",
    "iteratively_refined_preconditioned_conjugate_gradient",
    "preconditioned_gmres",
    "preconditioned_conjugate_gradient",
    "train_preconditioned_sr_projection",
]
