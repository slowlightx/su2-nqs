"""Hermitian, ray-invariant carrier control variate for strict SU(2) singlets.

For a fixed carrier C with known HC and D=3 H_z, use the positive empirical
inner product <u|v>_Q=sum(conj(u)*v/qtilde).  The rank-two Hermitian completion
of D agrees with the carrier energy/force and with the full Hamiltonian in
the population limit restricted to singlets.  It is NOT a finite-sample
variational bound or a replacement for an untouched full-H variance audit.

All inputs are physical amplitudes at the same configurations. No division
by psi or C occurs, including at carrier nodes. Q may be unnormalized but is
fixed. C, HC, Q and D are stop-gradient data; only the candidate state is
variational. The Hamiltonian must be isotropic, and states AND tangents must
be exact singlets for the population energy identity.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class CarrierControlEvaluation(NamedTuple):
    """JAX-pytree evaluation in the same energy units as HC and diagonal.

    action is H_eff psi; residual is (H_eff-energy)psi. norm_sum is only an
    empirical unnormalized-Q quadrature, not an absolute Hilbert norm.
    carrier_energy_complex retains the parent full-H imaginary-energy check.
    """

    energy: jax.Array
    norm_sum: jax.Array
    action: jax.Array
    residual: jax.Array
    carrier_energy_complex: jax.Array


def _vector(value: jax.Array, name: str, *, real: bool = False,
            fixed: bool = False) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ValueError(f'{name} must be a nonempty vector')
    if real and jnp.iscomplexobj(array):
        raise TypeError(f'{name} must be real')
    array = jnp.asarray(array, dtype=jnp.float64 if real else jnp.complex128)
    if not isinstance(array, jax.core.Tracer) and not np.isfinite(np.asarray(array)).all():
        raise ValueError(f'{name} must be finite')
    return jax.lax.stop_gradient(array) if fixed else array


def _carrier_data(carrier: jax.Array, hcarrier: jax.Array,
                  qtilde: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    c = _vector(carrier, 'carrier', fixed=True)
    hc = _vector(hcarrier, 'hcarrier', fixed=True)
    q = _vector(qtilde, 'qtilde', real=True, fixed=True)
    if hc.shape != c.shape or q.shape != c.shape:
        raise ValueError('carrier, hcarrier and qtilde must have matching shapes')
    if not isinstance(q, jax.core.Tracer) and np.any(np.asarray(q) <= 0):
        raise ValueError('qtilde must be strictly positive')
    norm = jnp.sum(jnp.real(c.conj()*c)/q)
    if not isinstance(norm, jax.core.Tracer) and float(norm) <= 0:
        raise ValueError('carrier needs positive quadrature norm')
    valid = (jnp.all(jnp.isfinite(c)) & jnp.all(jnp.isfinite(hc))
             & jnp.all(jnp.isfinite(q)) & jnp.all(q > 0) & jnp.isfinite(norm) & (norm > 0))
    return c, hc, q, norm, valid


def _action_data(
    psi: jax.Array, carrier: jax.Array, hcarrier: jax.Array,
    qtilde: jax.Array, diagonal: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    c, hc, q, zc, valid = _carrier_data(carrier, hcarrier, qtilde)
    candidate = _vector(psi, 'psi')
    d = _vector(diagonal, 'diagonal', real=True, fixed=True)
    if candidate.shape != c.shape or d.shape != c.shape:
        raise ValueError('psi and diagonal must match carrier shape')
    r = hc-d*c
    a = jnp.sum(c.conj()*r/q).real
    b = jnp.sum(c.conj()*candidate/q)
    r_psi = jnp.sum(r.conj()*candidate/q)
    action = d*candidate+r*b/zc+c*r_psi/zc-a*c*b/zc**2
    carrier_energy = jnp.sum(c.conj()*hc/q)/zc
    valid = valid & jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(d))
    return candidate, q, action, carrier_energy, valid


def carrier_control_action(
    psi: jax.Array, carrier: jax.Array, hcarrier: jax.Array,
    qtilde: jax.Array, diagonal: jax.Array,
) -> jax.Array:
    """Return H_eff psi for the weighted-Hermitian rank-two completion.

    H_eff = D + |r><C|/Z + |C><r|/Z - a|C><C|/Z²,
    where r=HC-DC, Z=<C|C>_Q and a=Re<C|r>_Q. A zero candidate is allowed;
    its action is zero. Concrete invalid data raise; invalid traced data give
    NaNs so a compiled caller can fail closed.
    """
    _, _, action, _, valid = _action_data(psi, carrier, hcarrier, qtilde, diagonal)
    return jnp.where(valid, action, jnp.nan+0j)


def carrier_control_evaluation(
    psi: jax.Array, carrier: jax.Array, hcarrier: jax.Array,
    qtilde: jax.Array, diagonal: jax.Array,
) -> CarrierControlEvaluation:
    """Energy, action and centered force residual without amplitude ratios.

    D must be total 3Hz if HC is total H C; for per-site units divide BOTH
    HC and D by N. Do not use the surrogate action to report physical Var(H).
    Finite-Q gauge invariance holds independently of the singlet assumption.
    """
    candidate, q, action, parent_energy, valid = _action_data(
        psi, carrier, hcarrier, qtilde, diagonal)
    norm = jnp.sum(jnp.real(candidate.conj()*candidate)/q)
    if not isinstance(norm, jax.core.Tracer) and float(norm) <= 0:
        raise ValueError('psi needs positive quadrature norm for an energy')
    energy = jnp.sum(candidate.conj()*action/q).real/norm
    valid = valid & jnp.isfinite(norm) & (norm > 0) & jnp.isfinite(energy)
    energy = jnp.where(valid, energy, jnp.nan)
    return CarrierControlEvaluation(energy, jnp.where(valid, norm, jnp.nan),
                                    jnp.where(valid, action, jnp.nan+0j),
                                    jnp.where(valid, action-energy*candidate, jnp.nan+0j),
                                    jnp.where(valid, parent_energy, jnp.nan+0j))


def carrier_control_loss(
    psi: jax.Array, carrier: jax.Array, hcarrier: jax.Array,
    qtilde: jax.Array, diagonal: jax.Array,
) -> jax.Array:
    """Differentiable ray-invariant energy surrogate; only psi is variable."""
    return carrier_control_evaluation(psi, carrier, hcarrier, qtilde, diagonal).energy


def _direction_matrix(directions: jax.Array, count: int) -> tuple[jax.Array, bool]:
    array = jnp.asarray(directions, dtype=jnp.complex128)
    scalar = array.ndim == 1
    if scalar:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 1:
        raise ValueError('directions must have shape (sample,) or (sample, direction)')
    if not isinstance(array, jax.core.Tracer) and not np.isfinite(np.asarray(array)).all():
        raise ValueError('directions must be finite')
    return array, scalar


def parent_directional_force(
    carrier: jax.Array, hcarrier: jax.Array, directions: jax.Array,
    qtilde: jax.Array,
) -> jax.Array:
    """Real-parameter dE/dt at psi=C+t*direction, including carrier nodes.

    Returns 2 Re<direction | HC-E_complex C>_Q/Z. Complex centering removes
    finite-quadrature phase force as well as norm force. For an exact carrier
    eigenstate this is zero on ANY positive quadrature, before averaging.
    Pass hcarrier=D*C to obtain the ordinary diagonal parent force instead.
    The result is scalar for a vector direction and (k,) for k directions.
    """
    c, hc, q, norm, valid = _carrier_data(carrier, hcarrier, qtilde)
    tangent, scalar = _direction_matrix(directions, c.shape[0])
    energy = jnp.sum(c.conj()*hc/q)/norm
    residual = hc-energy*c
    force = 2*jnp.sum(tangent.conj()*residual[:, None]/q[:, None], axis=0).real/norm
    force = jnp.where(valid & jnp.all(jnp.isfinite(tangent)), force, jnp.nan)
    return force[0] if scalar else force


def carrier_control_directional_force(
    psi: jax.Array, directions: jax.Array, carrier: jax.Array,
    hcarrier: jax.Array, qtilde: jax.Array, diagonal: jax.Array,
) -> jax.Array:
    """Directional derivative of the completed surrogate at arbitrary psi.

    Uses the full-corpus centered action. For chunked neural gradients freeze
    that action and norm, sum 2 Re<dpsi|residual>_Q/norm across microbatches,
    and then pull back through the amplitude/backbone map.
    """
    evaluated = carrier_control_evaluation(psi, carrier, hcarrier, qtilde, diagonal)
    tangent, scalar = _direction_matrix(directions, evaluated.action.shape[0])
    q = jax.lax.stop_gradient(jnp.asarray(qtilde, dtype=jnp.float64))
    result = 2*jnp.sum(tangent.conj()*evaluated.residual[:, None]/q[:, None], axis=0).real/evaluated.norm_sum
    result = jnp.where(jnp.all(jnp.isfinite(tangent)), result, jnp.nan)
    return result[0] if scalar else result


__all__ = ['CarrierControlEvaluation', 'carrier_control_action',
           'carrier_control_evaluation', 'carrier_control_loss',
           'parent_directional_force', 'carrier_control_directional_force']
