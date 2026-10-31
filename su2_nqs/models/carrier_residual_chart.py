"""Global, differentiable coordinates for a carrier plus additive residual.

This is an amplitude-coordinate adapter, not a new ansatz or a projector.
One frozen development quadrature defines the carrier/reference directions.
Residual coefficients are fit ONCE on that explicit calibration corpus, with
derivatives retained, then reused on arbitrary evaluation chunks. No physical
bits, per-configuration gates, Hamiltonian, sampling or optimization enter.

If C, D0 and Dtheta are exact SU(2) singlets in lattice A1, every returned
state stays in that sector: all transformations are global complex scalars.
The adapter cannot establish these symmetries for arbitrary supplied arrays.
Exact old-state ray replay is guaranteed where the chart is valid. If mix
and gain are subsequently optimized independently of the replay map, equality
of the old and new trainable state families needs a separate expressivity
argument; it does not follow from this coordinate identity.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ResidualChartAnchor(NamedTuple):
    """Frozen calibration data; Q may have an unknown common normalization."""

    calibration_carrier: jax.Array
    calibration_reference: jax.Array
    calibration_qtilde: jax.Array
    calibration_s0: jax.Array
    carrier_norm: jax.Array
    reference_overlap: jax.Array
    reference_perp_scale: jax.Array
    valid: jax.Array


class ResidualChartCoefficients(NamedTuple):
    """Differentiable global projections a=<C,f>/ZC, b=<s0,f>/ZC."""

    a: jax.Array
    b: jax.Array
    valid: jax.Array


class ResidualChartEvaluation(NamedTuple):
    """Values on requested points; no coefficient fit occurs here."""

    amplitudes: jax.Array
    innovation: jax.Array
    reference_direction: jax.Array


class AdditiveRayParameters(NamedTuple):
    """old_additive = carrier_coefficient * chart_state on every point."""

    mix: jax.Array
    gain: jax.Array
    carrier_coefficient: jax.Array
    valid: jax.Array


def _vector(value: jax.Array, name: str, *, real: bool = False,
            frozen: bool = False) -> jax.Array:
    result = jnp.asarray(value)
    if result.ndim != 1 or result.shape[0] < 1:
        raise ValueError(f'{name} must be a nonempty one-dimensional array')
    if real and jnp.iscomplexobj(result):
        raise TypeError(f'{name} must be real')
    result = jnp.asarray(result, dtype=jnp.float64 if real else jnp.complex128)
    if not isinstance(result, jax.core.Tracer) and not np.isfinite(np.asarray(result)).all():
        raise ValueError(f'{name} must be finite')
    return jax.lax.stop_gradient(result) if frozen else result


def _scalar(value: jax.Array, name: str, *, real: bool = False) -> jax.Array:
    result = jnp.asarray(value)
    if result.shape != ():
        raise ValueError(f'{name} must be one global scalar, never a physical-bit gate')
    if real and jnp.iscomplexobj(result):
        raise TypeError(f'{name} must be real')
    result = jnp.asarray(result, dtype=jnp.float64 if real else jnp.complex128)
    if not isinstance(result, jax.core.Tracer) and not np.isfinite(np.asarray(result)).all():
        raise ValueError(f'{name} must be finite')
    return result


def _checked(valid: jax.Array, message: str) -> jax.Array:
    """Eager invalid data raise; traced invalid data are masked to NaN."""
    if not isinstance(valid, jax.core.Tracer) and not bool(np.asarray(valid)):
        raise ValueError(message)
    return valid


def _anchor(anchor: ResidualChartAnchor) -> ResidualChartAnchor:
    if not isinstance(anchor, ResidualChartAnchor):
        raise TypeError('anchor must come from make_residual_chart_anchor')
    return jax.tree_util.tree_map(jax.lax.stop_gradient, anchor)


def make_residual_chart_anchor(
    calibration_carrier: jax.Array, calibration_reference: jax.Array,
    calibration_qtilde: jax.Array, *, relative_tolerance: float = 1e-12,
) -> ResidualChartAnchor:
    """Freeze C,D0,Q and form s0=P_C^perp D0 / sqrt(||P_C^perp D0||²/ZC).

    The inner product is sum(conj(u)*v/Q). Thus C and s0 are orthogonal and
    have the same calibration norm. ``relative_tolerance`` is a minimum
    reference/carrier separation sin(angle), not an absolute amplitude floor.
    A zero or nearly parallel reference fails closed; it is never silently
    replaced by a new direction. Q is a positive fixed development metric,
    not necessarily a normalized sampling density.
    """
    c = _vector(calibration_carrier, 'calibration_carrier', frozen=True)
    d0 = _vector(calibration_reference, 'calibration_reference', frozen=True)
    q = _vector(calibration_qtilde, 'calibration_qtilde', real=True, frozen=True)
    if c.shape != d0.shape or c.shape != q.shape:
        raise ValueError('calibration arrays must have matching shapes')
    tolerance = jax.lax.stop_gradient(_scalar(relative_tolerance, 'relative_tolerance', real=True))
    norm = jnp.sum(abs(c)**2/q)
    reference_norm = jnp.sum(abs(d0)**2/q)
    overlap = jnp.sum(c.conj()*d0/q)/jnp.where(norm > 0, norm, 1.)
    perpendicular = d0-overlap*c
    perpendicular_norm = jnp.sum(abs(perpendicular)**2/q)
    scale = jnp.sqrt(perpendicular_norm/jnp.where(norm > 0, norm, 1.))
    valid = (jnp.all(jnp.isfinite(c)) & jnp.all(jnp.isfinite(d0))
             & jnp.all(jnp.isfinite(q)) & jnp.all(q > 0)
             & jnp.isfinite(norm) & (norm > 0)
             & jnp.isfinite(reference_norm) & (reference_norm > 0)
             & jnp.isfinite(scale) & (scale > 0)
             & jnp.isfinite(tolerance) & (tolerance > 0) & (tolerance < 1)
             & (perpendicular_norm > tolerance**2*reference_norm))
    _checked(valid, 'invalid metric, zero carrier, or reference nearly parallel to carrier')
    s0 = perpendicular/jnp.where(scale > 0, scale, 1.)
    return jax.tree_util.tree_map(jax.lax.stop_gradient, ResidualChartAnchor(
        jnp.where(valid, c, jnp.nan+0j), jnp.where(valid, d0, jnp.nan+0j),
        jnp.where(valid, q, jnp.nan), jnp.where(valid, s0, jnp.nan+0j),
        jnp.where(valid, norm, jnp.nan), jnp.where(valid, overlap, jnp.nan+0j),
        jnp.where(valid, scale, jnp.nan), valid))


def fit_residual_chart_coefficients(
    anchor: ResidualChartAnchor, calibration_difference: jax.Array,
) -> ResidualChartCoefficients:
    """Differentiate THROUGH a,b computed from full-corpus f=Dtheta-D0.

    Do not call this on each evaluation microbatch. A and b are global scalar
    functionals of the explicitly supplied calibration residual; detaching
    them changes the tangent and is not this coordinate transformation.
    """
    frozen = _anchor(anchor)
    f = _vector(calibration_difference, 'calibration_difference')
    if f.shape != frozen.calibration_carrier.shape:
        raise ValueError('calibration_difference must match the frozen calibration corpus')
    a = jnp.sum(frozen.calibration_carrier.conj()*f/frozen.calibration_qtilde)/frozen.carrier_norm
    b = jnp.sum(frozen.calibration_s0.conj()*f/frozen.calibration_qtilde)/frozen.carrier_norm
    valid = frozen.valid & jnp.all(jnp.isfinite(f)) & jnp.isfinite(a) & jnp.isfinite(b)
    _checked(valid, 'invalid calibration residual or chart anchor')
    return ResidualChartCoefficients(jnp.where(valid, a, jnp.nan+0j),
                                     jnp.where(valid, b, jnp.nan+0j), valid)


def residual_chart_amplitudes(
    anchor: ResidualChartAnchor, coefficients: ResidualChartCoefficients,
    evaluation_carrier: jax.Array, evaluation_reference: jax.Array,
    evaluation_difference: jax.Array, *, mix: jax.Array = 0j,
    gain: jax.Array = 1.+0j,
) -> ResidualChartEvaluation:
    """Evaluate C+mix*s0+gain*I with the globally calibrated a,b reused.

    I=f-a*C-b*s0; s0=(D0-anchor.reference_overlap*C)/anchor.reference_perp_scale.
    Evaluation C,D0 are frozen, f and a,b remain differentiable. Different
    evaluation sizes/orderings produce consistent pointwise values. The
    caller is responsible for supplying the same physical C,D0 and model f
    as at calibration. There is no Q argument or local/batch coefficient fit.

    At f=0,mix=0 the state is exactly C. Gain defaults to one, so the
    innovation tangent is not algebraically killed by a zero residual gate.
    This does not guarantee that a particular model has a nonzero innovation.
    """
    frozen = _anchor(anchor)
    if not isinstance(coefficients, ResidualChartCoefficients):
        raise TypeError('coefficients must come from fit_residual_chart_coefficients')
    c = _vector(evaluation_carrier, 'evaluation_carrier', frozen=True)
    d0 = _vector(evaluation_reference, 'evaluation_reference', frozen=True)
    f = _vector(evaluation_difference, 'evaluation_difference')
    if c.shape != d0.shape or c.shape != f.shape:
        raise ValueError('evaluation arrays must have matching shapes')
    a = _scalar(coefficients.a, 'coefficient a'); b = _scalar(coefficients.b, 'coefficient b')
    mixture = _scalar(mix, 'mix'); strength = _scalar(gain, 'gain')
    s0 = (d0-frozen.reference_overlap*c)/frozen.reference_perp_scale
    innovation = f-a*c-b*s0
    values = c+mixture*s0+strength*innovation
    valid = (frozen.valid & coefficients.valid & jnp.all(jnp.isfinite(c))
             & jnp.all(jnp.isfinite(d0)) & jnp.all(jnp.isfinite(f))
             & jnp.isfinite(a) & jnp.isfinite(b) & jnp.isfinite(mixture)
             & jnp.isfinite(strength) & jnp.all(jnp.isfinite(values)))
    _checked(valid, 'invalid evaluation arrays, scalar chart coordinates or frozen anchor')
    return ResidualChartEvaluation(jnp.where(valid, values, jnp.nan+0j),
        jnp.where(valid, innovation, jnp.nan+0j), jnp.where(valid, s0, jnp.nan+0j))


def additive_ray_parameters(
    coefficients: ResidualChartCoefficients, scale: jax.Array, *,
    overlap_tolerance: float = 1e-12,
) -> AdditiveRayParameters:
    """Replay old=C+f/scale up to one nonzero global complex coefficient.

    old=(1+a/scale)*[C + b/(scale+a)*s0 + I/(scale+a)]. Scale must be finite,
    real and positive. Cancellation scale+a near zero invalidates this chart;
    no clipping or replacement is performed. The threshold is relative to
    abs(scale)+abs(a), detecting cancellation independently of amplitude units.
    This is ray equivalence, not numerical equality before global rescaling
    or an increase of the physical state manifold.
    """
    if not isinstance(coefficients, ResidualChartCoefficients):
        raise TypeError('coefficients must come from fit_residual_chart_coefficients')
    a = _scalar(coefficients.a, 'coefficient a'); b = _scalar(coefficients.b, 'coefficient b')
    divisor = _scalar(scale, 'scale', real=True)
    tolerance = jax.lax.stop_gradient(_scalar(overlap_tolerance, 'overlap_tolerance', real=True))
    denominator = divisor+a
    valid = (coefficients.valid & jnp.isfinite(a) & jnp.isfinite(b)
             & jnp.isfinite(divisor) & (divisor > 0)
             & jnp.isfinite(tolerance) & (tolerance > 0) & (tolerance < 1)
             & (abs(denominator) > tolerance*(abs(divisor)+abs(a))))
    _checked(valid, 'invalid scale or near-zero carrier overlap in additive ray chart')
    safe = jnp.where(valid, denominator, 1.+0j)
    return AdditiveRayParameters(jnp.where(valid, b/safe, jnp.nan+0j),
        jnp.where(valid, 1/safe, jnp.nan+0j),
        jnp.where(valid, denominator/jnp.where(divisor > 0, divisor, 1.), jnp.nan+0j), valid)


__all__ = [
    'ResidualChartAnchor', 'ResidualChartCoefficients', 'ResidualChartEvaluation',
    'AdditiveRayParameters', 'make_residual_chart_anchor',
    'fit_residual_chart_coefficients', 'residual_chart_amplitudes',
    'additive_ray_parameters',
]
