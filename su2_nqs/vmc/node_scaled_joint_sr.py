"""Full-native cached SR with node-scaled ridge and the complete Fisher metric.

Node labels identify primitive reduced-tensor parameter collections, not spin
quantum numbers. Magnetic structure and doubled ``q`` labels are unchanged.
Only the regularizer is block diagonal; the sampled physical Fisher is not.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .cached_amplitude_operators import (
    CachedAmplitudeJacobian,
    _physical_data,
    _positive_int,
    _require_x64,
    cached_derivative_memory_plan,
)

_RELATIVE_SOLVE_TOLERANCE = 1e-7


def solve_cached_node_scaled_joint_sr(
    cache: CachedAmplitudeJacobian,
    psi: Any,
    hpsi: Any,
    density: Any,
    *,
    blocks: Sequence[tuple[str, int, int]],
    relative_damping: float = .5,
    ridge_floor_relative: float = 1e-12,
    parameter_chunk_size: int = 16384,
    maximum_device_bytes: int,
    maximum_host_bytes: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Solve ``(A.T A + Lambda)d = A.T r`` in ALL native coordinates.

    For ``n`` real records, ``Z=sum(psi**2/Q)``, ``u=psi/sqrt(Q*Z)``,
    ``A=(I-u u.T) J/sqrt(Q*Z)`` and ``r=2(Hpsi-E psi)/sqrt(Q*Z)``.
    This is the existing metric-F/population-force-2 convention. The caller
    applies ``theta - scale*d`` and independently validates the finite state.

    ``Lambda`` is scalar within each explicit contiguous node block:
    ``lambda_b=max(relative_damping*trace(A_b A_b.T)/(n-1), floor)``,
    where ``floor=ridge_floor_relative*max_b(trace(A_b A_b.T)/(n-1))``.
    Zero physical blocks have zero direction with this disclosed ridge floor;
    an entirely zero physical Jacobian is rejected rather than given a scale.

    The exact dual is ``c=(I+sum_b A_b A_b.T/lambda_b)^-1 r`` and
    ``d_b=A_b.T c/lambda_b``. Three streamed J passes build per-node sample
    Grams, recover the native vectors and verify the TRUE coupled normal
    residual. No parameter-square matrix, AD, H action or new sample occurs.
    Stored node Grams require ``len(blocks)*n*n*8`` additional host bytes.

    Positive Q at target nodes is mandatory. A numerically exact sample solve
    does not certify independent direction, tangent support or finite descent.
    A cache may only be reused at its original frozen parameters/records/Q.
    Both dual and true native relative normal residuals must be below
    ``1e-7``; finite but inaccurate solves raise ``ArithmeticError``.
    """
    _require_x64()
    if (cache.jacobian.ndim != 2 or cache.jacobian.dtype != np.dtype('float64')
            or min(cache.jacobian.shape) < 1 or cache.jacobian.shape[0] < 2):
        raise ValueError('real float64 cached J[n>=2,P>=1] required')
    if (isinstance(relative_damping, bool) or isinstance(ridge_floor_relative, bool)
            or not isfinite(relative_damping) or not isfinite(ridge_floor_relative)
            or relative_damping <= 0 or ridge_floor_relative <= 0):
        raise ValueError('positive finite relative damping and ridge floor required')
    _positive_int(parameter_chunk_size, 'parameter_chunk_size')
    _positive_int(maximum_device_bytes, 'maximum_device_bytes')
    _positive_int(maximum_host_bytes, 'maximum_host_bytes')
    count, parameters = cache.jacobian.shape
    cursor = 0
    labels: set[str] = set()
    if not blocks:
        raise ValueError('named node blocks must partition all native columns')
    for label, first, final in blocks:
        if (not isinstance(label, str) or not label or label in labels
                or isinstance(first, bool) or isinstance(final, bool)
                or not isinstance(first, int) or not isinstance(final, int)
                or first != cursor or not first < final <= parameters):
            raise ValueError('named contiguous node blocks must partition all native columns once')
        labels.add(label)
        cursor = final
    if cursor != parameters:
        raise ValueError('node blocks must retain all native columns')
    width = min(parameters, parameter_chunk_size)
    plan = cached_derivative_memory_plan(count, parameters, parameter_chunk_size=width)
    host_extra = (len(blocks)+1)*count*count*8
    if plan['host_solve_array_bound_bytes']+host_extra > maximum_host_bytes:
        raise MemoryError('joint node sample Grams exceed registered host array budget')
    if plan['streamed_solve_device_array_bound_bytes'] > maximum_device_bytes:
        raise MemoryError('joint streamed algebra exceeds registered device array budget')
    scale, unit, force, energy, norm = _physical_data(cache, psi, density, hpsi)
    grams: list[np.ndarray] = []
    notes: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    direction, gradient = np.zeros(parameters), np.zeros(parameters)

    @jax.jit
    def horizontal(raw):
        weighted = raw*scale[:, None]
        return weighted-unit[:, None]*(unit@weighted)[None, :]

    @jax.jit
    def add_gram(gram, raw):
        a = horizontal(raw)
        return gram+a@a.T

    @jax.jit
    def vectors(raw, coefficients, ridge):
        a = horizontal(raw)
        d, g = (a.T@coefficients)/ridge, a.T@force
        return d, g, a@d

    @jax.jit
    def residual_squared(raw, product, d, g, ridge):
        residual = horizontal(raw).T@product+ridge*d-g
        return jnp.vdot(residual, residual).real

    def chunks(first: int, final: int):
        for start in range(first, final, width):
            stop = min(start+width, final)
            raw = np.ascontiguousarray(cache.jacobian[:, start:stop])
            if not np.isfinite(raw).all():
                raise FloatingPointError('nonfinite cached derivative; no clipped columns')
            if stop-start != width:
                raw = np.pad(raw, ((0, 0), (0, width-(stop-start))))
            yield start, stop, jnp.asarray(raw)

    def report(stage: str, tick: float, **extra: Any) -> None:
        note = dict(stage=stage, seconds=perf_counter()-tick, **extra)
        timings.append(note)
        if progress is not None:
            progress(note)

    for label, first, final in blocks:
        tick = perf_counter()
        gram = jnp.zeros((count, count), jnp.float64)
        for _, _, raw in chunks(first, final):
            gram = add_gram(gram, raw)
            gram.block_until_ready()
        matrix = np.asarray(gram)
        trace = float(np.trace(matrix))
        if not np.isfinite(matrix).all() or not np.isfinite(trace) or trace < 0:
            raise ArithmeticError('nonfinite/negative node physical Fisher Gram')
        grams.append(matrix)
        notes.append(dict(label=label, first=first, final=final, parameters=final-first,
                          Fisher_trace=trace, sample_eigenvalue_scale=trace/(count-1)))
        report('joint_node_Gram_cached', tick, label=label, completed_nodes=len(grams))
    reference = max(note['sample_eigenvalue_scale'] for note in notes)
    if reference <= 0:
        raise ArithmeticError('all native blocks have zero physical metric')
    floor = ridge_floor_relative*reference
    dual_gram = jnp.zeros((count, count), jnp.float64)
    tick = perf_counter()
    for matrix, note in zip(grams, notes):
        ridge = max(relative_damping*note['sample_eigenvalue_scale'], floor)
        note['ridge'] = ridge
        note['numerical_floor_active'] = ridge == floor
        dual_gram = dual_gram+jnp.asarray(matrix)/ridge
        dual_gram.block_until_ready()
    coefficients = jax.scipy.linalg.solve(dual_gram+jnp.eye(count), force, assume_a='pos')
    coefficients.block_until_ready()
    dual_residual = (dual_gram+jnp.eye(count))@coefficients-force
    dual_relative = float(jnp.linalg.norm(dual_residual)/jnp.maximum(jnp.linalg.norm(force), 1e-300))
    report('joint_dual_solve', tick, nodes=len(blocks), records=count)
    del grams
    product = jnp.zeros(count, jnp.float64)
    for note in notes:
        tick = perf_counter()
        for first, final, raw in chunks(note['first'], note['final']):
            d, g, image = vectors(raw, coefficients, note['ridge'])
            direction[first:final] = np.asarray(d)[:final-first]
            gradient[first:final] = np.asarray(g)[:final-first]
            product = product+image
            product.block_until_ready()
        report('joint_native_direction', tick, label=note['label'])
    squared = 0.
    for note in notes:
        tick = perf_counter()
        local_squared = jnp.asarray(0., jnp.float64)
        for first, final, raw in chunks(note['first'], note['final']):
            d = np.pad(direction[first:final], (0, width-(final-first)))
            g = np.pad(gradient[first:final], (0, width-(final-first)))
            local_squared += residual_squared(raw, product, jnp.asarray(d), jnp.asarray(g), note['ridge'])
            local_squared.block_until_ready()
        squared += float(local_squared)
        rhs = float(np.linalg.norm(gradient[note['first']:note['final']]))
        note['relative_coupled_block_normal_residual'] = float(np.sqrt(float(local_squared)))/max(rhs, 1e-300)
        report('joint_true_native_normal_residual', tick, label=note['label'])
    gradient_norm = float(np.linalg.norm(gradient))
    relative = np.sqrt(squared)/gradient_norm if gradient_norm else (0. if squared == 0 else float('inf'))
    if (not np.isfinite(direction).all() or not np.isfinite(gradient).all()
            or not np.isfinite(relative) or not np.isfinite(dual_relative)):
        raise FloatingPointError('nonfinite full-native joint node-scaled SR result')
    if relative >= _RELATIVE_SOLVE_TOLERANCE or dual_relative >= _RELATIVE_SOLVE_TOLERANCE:
        raise ArithmeticError('joint solve failed true native or dual relative normal residual tolerance 1e-7')
    tangent = np.asarray(product)
    tangent_mass = tangent**2
    absolute_force_mass = abs(tangent*np.asarray(force))

    def concentration(mass: np.ndarray) -> tuple[float, float]:
        total = float(mass.sum())
        if total == 0:
            return 0., 0.
        return total**2/float(mass@mass), float(mass.max()/total)

    tangent_ess, tangent_max = concentration(tangent_mass)
    force_ess, force_max = concentration(absolute_force_mass)
    return jnp.asarray(direction), jnp.asarray(gradient), dict(
        numerically_valid=True, full_global_Fisher_solved=True,
        update_geometry='complete_native_Fisher_with_node_scaled_ridge',
        relative_linear_residual=float(relative), global_cross_node_normal_residual=float(relative),
        normal_residual_norm=float(np.sqrt(squared)), dual_relative_linear_residual=dual_relative,
        relative_solve_tolerance=_RELATIVE_SOLVE_TOLERANCE,
        gradient_norm=gradient_norm, raw_fisher_norm=float(np.linalg.norm(tangent)),
        chosen_direction_tangent=tangent, sampled_parent_norm=norm/count, sampled_energy_total=energy,
        exact_node_records=int(np.count_nonzero(np.asarray(psi) == 0)),
        blocks=notes, relative_damping=float(relative_damping), ridge_floor_relative=float(ridge_floor_relative),
        numerical_ridge_floor=float(floor), ridge_scale='trace(F_node)/(records-1), not trace/P',
        all_native_columns_retained=True, all_records_retained=True, metric_is_F_not_two_F=True,
        force_is_population_Hermitian_estimator=True, no_AD_recomputed_in_solve=True,
        exact_Jacobian_passes=3, retained_raw_jacobian_entries=int(cache.jacobian.size),
        raw_J_storage_bytes=int(cache.jacobian.nbytes), additional_node_Gram_bytes=host_extra,
        chosen_direction_tangent_square_ESS=tangent_ess,
        chosen_direction_tangent_square_maximum_fraction=tangent_max,
        chosen_direction_absolute_force_projection_ESS=force_ess,
        chosen_direction_absolute_force_projection_maximum_fraction=force_max,
        concentration_is_chosen_direction_only_not_full_tangent_coverage=True,
        finite_quadrature_is_not_independent_direction_or_finite_step_confirmation=True,
        memory_plan=plan, timings=timings)


__all__ = ['solve_cached_node_scaled_joint_sr']
