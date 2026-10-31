"""Amplitude-only calibration of a proposed physical-state update.

The caller owns sampling and optimization. This utility matches an empirical
Fubini--Study sine distance along one frozen parameter direction, without
evaluating the Hamiltonian or changing that direction. Independent Born
samples should be supplied by the caller; a finite-sample match is not a
global Hilbert-space or equilibration certificate.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .dual_sampled_sr import sampled_overlap_diagnostics


def make_fs_calibrator(
    prepare: Callable, log_prepared: Callable, template: object, *,
    target_fs_sine: float, relative_tolerance: float = .02, max_factor: float = 2.,
    bracket_steps: int = 8, bisection_steps: int = 16,
    minimum_ess_fraction: float = .8,
    batch_size: int | None = None,
) -> Callable:
    """Match a proposed direction's empirical FS sine on independent samples.

    Search only a bounded, locally monotone bracket beginning at the unchanged
    state. An unreachable/nonfinite/nonmonotone/poor-ESS direction is rejected,
    rather than substituted with a shorter step or selected by candidate energy.
    This empirical distance is not an exact global Hilbert-space certificate.
    """
    if not (0 < target_fs_sine < 1 and 0 < relative_tolerance < 1
            and math.isfinite(max_factor) and max_factor > 0
            and 0 < minimum_ess_fraction <= 1):
        raise ValueError('invalid physical calibration scales')
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1
           for k in (bracket_steps, bisection_steps)):
        raise ValueError('positive integer calibration search counts required')
    if batch_size is not None and (isinstance(batch_size, bool)
                                  or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('batch_size must be None or a positive integer')
    _, unravel = ravel_pytree(template)
    if batch_size is None:
        logs = jax.vmap(log_prepared, in_axes=(None, 0))
    else:
        # Complete coherent projections can be large. Preserve the same rows
        # and values, bounding only simultaneous amplitude intermediates.
        logs = lambda prepared, rows: jax.lax.map(
            lambda row: log_prepared(prepared, row), rows, batch_size=batch_size)

    @jax.jit
    def calibrate(parent, proposed, rows):
        old, _ = ravel_pytree(parent)
        candidate, _ = ravel_pytree(proposed)
        direction = candidate-old
        old_logs = logs(prepare(parent), rows)

        def diagnose(amount):
            ratios = logs(prepare(unravel(old+amount*direction)), rows)-old_logs
            info = sampled_overlap_diagnostics(ratios)
            finite = jnp.all(jnp.isfinite(ratios)) & jnp.all(jnp.isfinite(old_logs))
            return info['sampled_fs_sine'], info['reweighting_ess_fraction'], finite

        # Geometric expansion finds the first target crossing. Once found, do
        # not inspect later nonlocal crossings of this nonlinear state curve.
        initial = (jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.),
                   jnp.asarray(False), jnp.asarray(True), jnp.asarray(0))

        def expand(i, carry):
            lo, hi, previous_fs, found, valid, calls = carry

            def check(_):
                factor = max_factor*2.**(i-(bracket_steps-1))
                fs, ess, finite = diagnose(factor)
                okay = finite & (fs+1e-12 >= previous_fs)
                crossed = fs >= target_fs_sine
                return (jnp.where(crossed, lo, factor), factor, fs,
                        crossed, valid & okay, calls+1)
            return jax.lax.cond(found, lambda _: carry, check, operand=None)

        lo, hi, maximum_fs, found, valid, calls = jax.lax.fori_loop(
            0, bracket_steps, expand, initial)

        def refine(_, bounds):
            left, right = bounds
            fs, _, _ = diagnose((left+right)/2)
            return (jnp.where(fs < target_fs_sine, (left+right)/2, left),
                    jnp.where(fs >= target_fs_sine, (left+right)/2, right))
        lo, hi = jax.lax.cond(found & valid,
            lambda bounds: jax.lax.fori_loop(0, bisection_steps, refine, bounds),
            lambda bounds: bounds, (lo, hi))
        amount = (lo+hi)/2
        fs, ess, finite = diagnose(amount)
        matched = jnp.abs(fs-target_fs_sine) <= relative_tolerance*target_fs_sine
        accepted = (found & valid & finite & matched &
                    (ess >= minimum_ess_fraction) & jnp.all(jnp.isfinite(direction)))
        chosen = jnp.where(accepted, old+amount*direction, old)
        return unravel(chosen), dict(accepted=accepted,
            proposed_factor=amount, applied_factor=jnp.where(accepted, amount, 0.),
            sampled_fs_sine=jnp.where(accepted, fs, 0.), attempted_fs_sine=fs,
            reweighting_ess_fraction=ess, target_fs_sine=jnp.asarray(target_fs_sine),
            target_bracketed=found, locally_monotone_finite=valid,
            maximum_bracket_fs_sine=maximum_fs,
            amplitude_batch_evaluations=1+calls+jnp.where(found & valid, bisection_steps, 0)+1)
    return calibrate
