"""Gauge-aware diagnostics for distilling fixed carrier-resonance teachers.

These are approximation losses, NOT energy estimators or a new variational
claim. A small state loss on a saved finite-Q corpus does not prove equality
on the full Hilbert space. Conversely, equal complete generator tables up
to their common identity shift do produce the same physical ray for the
shared two-gate construction, independent of the corpus.

Only candidate tables/psi/Hpsi are differentiated. Teacher, carrier, fixed Q
and the chosen real H-response shift are data. No amplitude ratios occur at
configuration nodes; the only division is by GLOBAL nonzero carrier-chart
overlaps. Undefined charts/zero target signals have explicit flags and NaN
relative losses, never an epsilon-regularized claim of successful replay.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class TableDistillationMetrics(NamedTuple):
    relative_mse: jax.Array
    hs_relative_mse: jax.Array
    absolute_mse: jax.Array
    removed_identity_shift: jax.Array
    defined: jax.Array


class PhysicalDistillationMetrics(NamedTuple):
    ray_infidelity: jax.Array
    innovation_relative_mse: jax.Array
    h_response_relative_mse: jax.Array
    innovation_absolute_mse: jax.Array
    h_response_absolute_mse: jax.Array
    candidate_carrier_fidelity: jax.Array
    target_carrier_fidelity: jax.Array
    chart_valid: jax.Array
    innovation_defined: jax.Array
    h_response_defined: jax.Array
    energy_shift: jax.Array


def _array(value: jax.Array, name: str, *, real: bool = False, fixed: bool = False) -> jax.Array:
    array = jnp.asarray(value)
    if real and jnp.iscomplexobj(array):
        raise ValueError(f'{name} must be real')
    array = jnp.asarray(array, dtype=jnp.float64 if real else jnp.complex128)
    if not isinstance(array, jax.core.Tracer) and not np.isfinite(np.asarray(array)).all():
        raise ValueError(f'{name} must be finite')
    return jax.lax.stop_gradient(array) if fixed else array


def _threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def table_distillation_metrics(
    candidate_table: jax.Array, target_table: jax.Array, *, minimum_target_signal: float = 1e-24,
) -> TableDistillationMetrics:
    """Complete[6,6,10] generator losses modulo ONE common identity shift.

    ``G -> G+kappa I`` in every context/q block scales the shared two-gate
    state by exp(2*kappa); removing context-dependent traces is NOT allowed.
    The main metric is the uniform reduced-coordinate Frobenius MSE ratio.
    The additional HS metric weights every complete magnetic multiplet by
    (q_left+1)*(q_right+1)*(q_active+1), using doubled integer q labels.
    ``absolute_mse`` is the gauge-removed uniform mean squared table error.
    """
    tolerance = _threshold(minimum_target_signal, 'minimum_target_signal')
    candidate = _array(candidate_table, 'candidate_table', real=True)
    target = _array(target_table, 'target_table', real=True, fixed=True)
    if candidate.shape != (6, 6, 10) or target.shape != candidate.shape:
        raise ValueError('candidate and target need complete real tables[6,6,10]')
    diagonal = jnp.asarray([1., 1., 0., 1., 1., 1., 0., 0., 0., 1.])
    identity = jnp.broadcast_to(diagonal, candidate.shape)
    environment_dimensions = jnp.asarray([1., 1., 3., 3., 3., 5.])
    active_dimensions = jnp.asarray([1., 1., 1., 3., 3., 3., 3., 3., 3., 5.])
    hs_weights = (environment_dimensions[:, None, None] * environment_dimensions[None, :, None]
                  * active_dimensions[None, None, :])
    def project(value, weights):
        coefficient = jnp.sum(weights * identity * value) / jnp.sum(weights * identity**2)
        return value - coefficient * identity, coefficient
    delta, shift = project(candidate - target, jnp.ones_like(candidate))
    centered_target, _ = project(target, jnp.ones_like(candidate))
    denominator = jnp.sum(centered_target**2)
    delta_hs, _ = project(candidate - target, hs_weights)
    target_hs, _ = project(target, hs_weights)
    denominator_hs = jnp.sum(hs_weights * target_hs**2)
    finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(target))
    defined = finite & (denominator > tolerance)
    defined_hs = finite & (denominator_hs > tolerance)
    error = jnp.sum(delta**2)
    return TableDistillationMetrics(
        jnp.where(defined, error / jnp.where(defined, denominator, 1.), jnp.nan),
        jnp.where(defined_hs, jnp.sum(hs_weights * delta_hs**2) / jnp.where(defined_hs, denominator_hs, 1.), jnp.nan),
        jnp.where(finite, error / candidate.size, jnp.nan),
        jnp.where(finite, shift, jnp.nan), defined)


def gauge_invariant_table_loss(candidate_table: jax.Array, target_table: jax.Array) -> jax.Array:
    """Uniform complete-table relative MSE, with the teacher held fixed."""
    return table_distillation_metrics(candidate_table, target_table).relative_mse


def physical_distillation_metrics(
    psi: jax.Array, hpsi: jax.Array, target: jax.Array, htarget: jax.Array,
    carrier: jax.Array, hcarrier: jax.Array, q: jax.Array, *,
    energy_shift: float | jax.Array | None = None,
    minimum_carrier_fidelity: float = 1e-12,
    minimum_target_signal: float = 1e-24,
) -> PhysicalDistillationMetrics:
    """Ray, carrier-innovation and centered-H errors on a FIXED quadrature.

    With a_v=<C,v>_Q/<C,C>_Q, d_v=v/a_v-C and Hd_v=Hv/a_v-HC,
    compare d_psi to d_target and (H-e)d_psi to (H-e)d_target. The fixed real
    shift e defaults to the target's empirical real energy in Hpsi units.
    Candidate/target phases and scales and carrier phase/scale cancel.
    ``innovation_absolute_mse`` is ||d_psi-d_t||²_Q/||C||²_Q; its H analogue
    has energy-squared units. Relative metrics retain small innovations that
    total-state fidelity can hide. A finite-corpus fit is not a global proof.
    """
    minimum_fidelity = _threshold(minimum_carrier_fidelity, 'minimum_carrier_fidelity')
    signal = _threshold(minimum_target_signal, 'minimum_target_signal')
    if minimum_fidelity >= 1:
        raise ValueError('minimum_carrier_fidelity must be smaller than one')
    candidate = _array(psi, 'psi')
    action = _array(hpsi, 'hpsi')
    teacher = _array(target, 'target', fixed=True)
    hteacher = _array(htarget, 'htarget', fixed=True)
    anchor = _array(carrier, 'carrier', fixed=True)
    hanchor = _array(hcarrier, 'hcarrier', fixed=True)
    proposal = _array(q, 'q', real=True, fixed=True)
    values = (candidate, action, teacher, hteacher, anchor, hanchor, proposal)
    if candidate.ndim != 1 or candidate.size < 1 or any(x.shape != candidate.shape for x in values):
        raise ValueError('all inputs must be matching nonempty sample vectors')
    if not isinstance(proposal, jax.core.Tracer) and np.any(np.asarray(proposal) <= 0):
        raise ValueError('q must be strictly positive at every sampled configuration')
    def inner(left, right):
        return jnp.sum(left.conj() * right / proposal)
    def norm(value):
        return inner(value, value).real
    zc, zp, zt = norm(anchor), norm(candidate), norm(teacher)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in values])) & jnp.all(proposal > 0)
    norm_valid = finite & (zc > 0) & (zp > 0) & (zt > 0) & jnp.all(jnp.isfinite(jnp.asarray([zc, zp, zt])))
    safe_zc = jnp.where(norm_valid, zc, 1.)
    safe_zp = jnp.where(norm_valid, zp, 1.)
    safe_zt = jnp.where(norm_valid, zt, 1.)
    ap, at = inner(anchor, candidate) / safe_zc, inner(anchor, teacher) / safe_zc
    fp = abs(ap)**2 * safe_zc / safe_zp
    ft = abs(at)**2 * safe_zc / safe_zt
    chart_valid = norm_valid & (fp > minimum_fidelity) & (ft > minimum_fidelity)
    # Substituting one only on explicitly invalid paths prevents invalid
    # divisions inside AD; the returned chart losses are still NaN/undefined.
    safe_ap, safe_at = jnp.where(chart_valid, ap, 1.+0j), jnp.where(chart_valid, at, 1.+0j)
    dp, dt = candidate/safe_ap-anchor, teacher/safe_at-anchor
    hdp, hdt = action/safe_ap-hanchor, hteacher/safe_at-hanchor
    if energy_shift is None:
        shift = jax.lax.stop_gradient(inner(teacher, hteacher).real / safe_zt)
    else:
        shift = _array(energy_shift, 'energy_shift', real=True, fixed=True)
        if shift.shape != ():
            raise ValueError('energy_shift must be one fixed real scalar')
    rp, rt = hdp-shift*dp, hdt-shift*dt
    dnorm, rnorm = norm(dt), norm(rt)
    # Relative degeneracy checks are scale/phase invariant for C and for H.
    response_scale = norm(hteacher/safe_at) + norm(hanchor) + abs(shift)**2*safe_zc
    innovation_defined = chart_valid & (dnorm > 0) & (dnorm > signal*safe_zc)
    response_defined = chart_valid & (rnorm > 0) & (rnorm > signal*response_scale)
    innovation_error, response_error = norm(dp-dt), norm(rp-rt)
    teacher_projection = inner(teacher, candidate) / safe_zt
    perpendicular = candidate-teacher_projection*teacher
    infidelity = jnp.clip(norm(perpendicular)/safe_zp, 0., 1.)
    return PhysicalDistillationMetrics(
        jnp.where(norm_valid, infidelity, jnp.nan),
        jnp.where(innovation_defined, innovation_error/jnp.where(innovation_defined, dnorm, 1.), jnp.nan),
        jnp.where(response_defined, response_error/jnp.where(response_defined, rnorm, 1.), jnp.nan),
        jnp.where(chart_valid, innovation_error/safe_zc, jnp.nan),
        jnp.where(chart_valid, response_error/safe_zc, jnp.nan),
        jnp.where(norm_valid, fp, jnp.nan), jnp.where(norm_valid, ft, jnp.nan),
        chart_valid, innovation_defined, response_defined, jnp.where(norm_valid, shift, jnp.nan))


__all__ = ['TableDistillationMetrics', 'PhysicalDistillationMetrics',
           'table_distillation_metrics', 'gauge_invariant_table_loss', 'physical_distillation_metrics']
