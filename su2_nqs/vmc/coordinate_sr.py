"""Fixed positive coordinate changes for real stochastic reconfiguration."""
from __future__ import annotations

from math import isfinite
import jax
import jax.numpy as jnp


def rescaled_real_sr_direction(
    scores: jax.Array, energies: jax.Array, coordinate_scale: jax.Array,
    *, damping: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return an old-coordinate descent direction from ``theta'=D theta``.

    ``scores`` are real physical log derivatives, with shape ``(records,P)``.
    ``coordinate_scale`` is a strictly positive diagonal ``D`` chosen before
    evaluating held-out data.  The same physical ray then has scores ``O/D``
    in primed coordinates.  A uniform ridge is applied there, and the update
    is mapped back using ``delta theta = delta theta'/D``.  The ridge is thus
    a declared changed optimization condition, not coordinate-invariant SR.
    """
    values=jnp.asarray(scores); energy=jnp.asarray(energies); scale=jnp.asarray(coordinate_scale)
    if values.ndim != 2 or energy.shape != (values.shape[0],) or scale.shape != (values.shape[1],):
        raise ValueError('scores, energies and coordinate scales have incompatible shapes')
    if values.shape[0] < 2 or not isfinite(damping) or damping <= 0:
        raise ValueError('at least two records and positive damping are required')
    transformed=values/scale
    a=(transformed-transformed.mean(axis=0))/jnp.sqrt(len(values))
    gradient=2*a.T@(energy-energy.mean())/jnp.sqrt(len(values))
    dual=jax.scipy.linalg.solve(damping*jnp.eye(len(values))+a@a.T,a@gradient,assume_a='pos')
    natural=(gradient-a.T@dual)/damping
    residual=a.T@(a@natural)+damping*natural-gradient
    valid=jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(scale)) & jnp.all(scale > 0)
    direction=jnp.where(valid,-natural/scale,jnp.nan)
    return direction,dict(relative_SR_residual=jnp.linalg.norm(residual)/jnp.maximum(jnp.linalg.norm(gradient),1e-25),
        transformed_gradient_norm=jnp.linalg.norm(gradient),valid=valid)


def diagonal_regularized_real_sr_direction(
    scores: jax.Array, energies: jax.Array, *, relative_shift: float,
    absolute_shift: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Solve ``[F + relative_shift*diag(F) + absolute_shift*I] d = -g``.

    ``F=Re Cov(O,O)`` and ``g=2 Re Cov(O,E)`` use real scores. All diagonal
    statistics must come from fitting chains only; this routine receives no
    held-chain data. Use a dual solve with D equal to the square root of
    the declared diagonal penalty. Center before rescaling so an exactly
    constant normalization-gauge column remains exactly zero. This changes the
    regularized direction, not the wavefunction, width or physical metric.
    Finite-step FS calibration and independent-chain validation remain the
    caller's responsibility. No energy benefit is implied by a small residual.
    """
    if (not isfinite(relative_shift) or relative_shift < 0
            or not isfinite(absolute_shift) or absolute_shift <= 0):
        raise ValueError('nonnegative relative and positive absolute shifts required')
    values = jnp.asarray(scores)
    if values.ndim != 2 or jnp.iscomplexobj(values):
        raise ValueError('real matrix of physical log derivatives required')
    centered = values-values.mean(axis=0)
    diagonal = jnp.mean(centered**2, axis=0)
    penalty = relative_shift*diagonal+absolute_shift
    energy = jnp.asarray(energies)
    if energy.shape != (len(values),) or len(values) < 2:
        raise ValueError('at least two matched score/energy records required')
    normalized = centered/jnp.sqrt(len(values))
    scale = jnp.sqrt(penalty)
    a = normalized/scale
    rhs = -2*(energy-energy.mean())/jnp.sqrt(len(values))
    factor = jax.scipy.linalg.cho_factor(jnp.eye(len(values))+a@a.T, lower=True)
    dual = jax.scipy.linalg.cho_solve(factor, rhs)
    # A.T (I+A A.T)^-1 rhs avoids subtracting two almost equal primal
    # gradients in the Woodbury formula at small absolute shifts.
    direction = (a.T@dual)/scale
    gradient = -normalized.T@rhs
    # Fixed iterative refinement in ORIGINAL coordinates. Reuse the same
    # factorization; no P-by-P Fisher matrix and no residual-based stopping
    # or tolerance relaxation. This matters for widely differing feature
    # scales and an almost unregularized dual null space.
    for _ in range(2):
        residual = normalized.T@(normalized@direction)+penalty*direction+gradient
        whitened = residual/scale
        correction = (whitened-a.T@jax.scipy.linalg.cho_solve(factor, a@whitened))/scale
        direction = direction-correction
    residual = normalized.T@(normalized@direction)+penalty*direction+gradient
    relative = jnp.linalg.norm(residual)/jnp.maximum(jnp.linalg.norm(gradient), 1e-25)
    valid = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(energy)) & (relative <= 1e-7)
    return jnp.where(valid, direction, jnp.nan), dict(valid=valid,
        relative_SR_residual=relative,
        penalty_min=jnp.min(penalty), penalty_max=jnp.max(penalty))
