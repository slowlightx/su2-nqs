"""Exact finite-quadrature full-H objectives, differentiating both amplitudes.

Samples have a fixed, positive, possibly unnormalized importance density Q.
The empirical normalized ratio is NOT a variational upper bound. In
particular H need not be Hermitian in the finite-Q empirical inner product:
both ``psi`` AND ``hpsi`` must remain differentiable. The familiar population
force2 Re<dpsi|(H-E)psi> is not its general finite-quadrature derivative.

No division by a wavefunction amplitude occurs; Hpsi at psi=0 contributes
to second moments and gradients. Q is stop-gradient, never differentiated
through a sampler. Sampling, chain uncertainties and tail coverage remain
the caller's responsibilities.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class FixedProposalHMoments(NamedTuple):
    """Empirical moments in the energy units of the supplied Hpsi.

    norm_sum is only sum_i|psi_i|²/Q_i, not an absolute Hilbert norm.
    variance is nonnegative weighted residual variance and retains the
    complex empirical energy when centering. With Hpsi/N, it is Var(H)/N²,
    not Var(H)/N. No quantity here estimates its own sampling uncertainty.
    """

    energy_complex: jax.Array
    norm_sum: jax.Array
    second_moment: jax.Array
    variance: jax.Array


def _inputs(
    psi: jax.Array, hpsi: jax.Array, q: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    amplitude, action = jnp.asarray(psi), jnp.asarray(hpsi)
    density = jnp.asarray(q)
    if (amplitude.ndim != 1 or amplitude.size < 1 or action.shape != amplitude.shape
            or density.shape != amplitude.shape or jnp.iscomplexobj(density)):
        raise ValueError('psi, hpsi and real q must be matching nonempty vectors')
    amplitude, action = (jnp.asarray(x, dtype=jnp.complex128) for x in (amplitude, action))
    density = jax.lax.stop_gradient(jnp.asarray(density, dtype=jnp.float64))
    for name, value in (('psi', amplitude), ('hpsi', action), ('q', density)):
        if not isinstance(value, jax.core.Tracer) and not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'{name} must be finite')
    if not isinstance(density, jax.core.Tracer) and np.any(np.asarray(density) <= 0):
        raise ValueError('q must be strictly positive, including wavefunction nodes')
    norm = jnp.sum(jnp.real(amplitude.conj() * amplitude) / density)
    if not isinstance(norm, jax.core.Tracer) and (not np.isfinite(float(norm)) or float(norm) <= 0):
        raise ValueError('psi must have finite positive empirical norm')
    valid = (jnp.all(jnp.isfinite(amplitude)) & jnp.all(jnp.isfinite(action))
             & jnp.all(jnp.isfinite(density)) & jnp.all(density > 0)
             & jnp.isfinite(norm) & (norm > 0))
    return amplitude, action, density, norm, valid


def weighted_full_h_moments(
    psi: jax.Array, hpsi: jax.Array, q: jax.Array,
) -> FixedProposalHMoments:
    """Return node-safe empirical energy, norm, second moment and variance.

    E=sum(conj(psi)*hpsi/Q)/Z and V=sum(|hpsi-E*psi|²/Q)/Z.
    This positive residual form equals <Hpsi|Hpsi>_Q/Z-|E|² even when E is
    complex. Do not force Im(E)=0 or replace the full action by a diagonal
    surrogate when reporting the physical variance.
    """
    amplitude, action, density, norm, valid = _inputs(psi, hpsi, q)
    energy = jnp.sum(amplitude.conj() * action / density) / norm
    second = jnp.sum(jnp.real(action.conj() * action) / density) / norm
    residual = action - energy * amplitude
    variance = jnp.sum(jnp.real(residual.conj() * residual) / density) / norm
    valid = valid & jnp.isfinite(energy) & jnp.isfinite(second) & jnp.isfinite(variance)
    return FixedProposalHMoments(jnp.where(valid, energy, jnp.nan + 0j),
                                jnp.where(valid, norm, jnp.nan),
                                jnp.where(valid, second, jnp.nan),
                                jnp.where(valid, variance, jnp.nan))


def weighted_full_h_loss(psi: jax.Array, hpsi: jax.Array, q: jax.Array) -> jax.Array:
    """Real normalized full-H empirical objective with dual-sided AD.

    Hpsi must be recomputed differentiably from the same variational
    parameters as psi. This function does not stop its gradient. Compute one
    global numerator/denominator across all microbatches; averaging separately
    normalized minibatch losses is a different objective.
    """
    amplitude, action, density, norm, valid = _inputs(psi, hpsi, q)
    energy = jnp.sum(amplitude.conj() * action / density).real / norm
    return jnp.where(valid & jnp.isfinite(energy), energy, jnp.nan)


__all__ = ['FixedProposalHMoments', 'weighted_full_h_loss', 'weighted_full_h_moments']
