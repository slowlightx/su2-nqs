"""Node-safe amplitude-space SR and imaginary-time alignment on positive Q.

Real parameter tangents are centered in physical Hilbert-space geometry,
not divided by a carrier amplitude. Only supplied amplitudes and TOTAL Hpsi
are read; there is no Hamiltonian AD or model-specific logic. Finite-Q fits
and cross-chain predictions are optimizer diagnostics, not global descent
certificates. Incomplete quadrature need not preserve H self-adjointness.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


def make_amplitude_jacobian(amplitude: Callable, template_params: object,
                            *, chunk_size: int = 4) -> Callable:
    """Return JIT (params, configurations) -> complex amplitude Jacobian.

    Columns follow ravel_pytree(template_params); every leaf is real float64.
    The callback takes a SINGLE configuration, including physical nodes.
    Explicit O(records*parameters) storage is used; not matrix-free AD.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('amplitude tangents require x64')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('positive integer chunk size required')
    leaves = jax.tree.leaves(template_params)
    if not leaves or any(np.asarray(v).dtype != np.dtype('float64') or
                         not np.isfinite(v).all() for v in leaves):
        raise ValueError('finite real float64 parameter leaves required')
    _, unravel = ravel_pytree(template_params)

    @jax.jit
    def jacobian(params, configurations):
        flat, _ = ravel_pytree(params)
        def row(x):
            def components(values):
                z = amplitude(unravel(values), x)
                return jnp.stack((z.real, z.imag))
            values = jax.jacrev(components)(flat)
            return values[0]+1j*values[1]
        return jax.lax.map(row, configurations, batch_size=chunk_size)
    return jacobian


def tangent_system(psi: np.ndarray, hpsi: np.ndarray, jacobian: np.ndarray,
                   q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return real SR A,b and complex centered J,residual in normalized Q units.

    S=A.T A=2 Re(J_perp^dagger J_perp), g=A.T b. Remove BOTH
    normalization and phase gauge by a complex projection onto the parent
    ray. Hpsi is total, not Hpsi/N. Unknown positive Q normalization cancels.
    """
    p, hp, j, density = (np.asarray(psi, complex), np.asarray(hpsi, complex),
                          np.asarray(jacobian, complex), np.asarray(q, float))
    if p.ndim != 1 or len(p) < 2 or hp.shape != p.shape or density.shape != p.shape:
        raise ValueError('matching amplitudes/products/Q with at least two records required')
    if j.ndim != 2 or j.shape[0] != len(p) or not j.shape[1]:
        raise ValueError('Jacobian must be records by real parameters')
    if not all(np.isfinite(x).all() for x in (p, hp, j, density)) or np.any(density <= 0):
        raise ValueError('finite inputs and strictly positive Q required')
    norm = np.sum(abs(p)**2/density)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('nonzero finite parent quadrature norm required')
    scale = np.sqrt(density*norm)
    u, hu, tangent = p/scale, hp/scale, j/scale[:, None]
    tangent -= u[:, None]*(u.conj()@tangent)[None, :]
    residual = hu-u*np.vdot(u, hu)
    a = np.sqrt(2.)*np.concatenate((tangent.real, tangent.imag))
    b = np.sqrt(2.)*np.concatenate((residual.real, residual.imag))
    return a, b, tangent, residual


def direction_diagnostics(psi: np.ndarray, hpsi: np.ndarray, jacobian: np.ndarray,
                          q: np.ndarray, descent_direction: np.ndarray) -> dict:
    """Predict -J*d alignment with -(H-E)psi; no candidate H evaluation.

    Positive cosine and negative one-sided energy slope are desirable. On a
    finite incomplete Q cache they need not imply actual full-H descent: the
    missing reverse-action coherence is tested only on actual candidates.
    """
    _, _, tangent, residual = tangent_system(psi, hpsi, jacobian, q)
    d = np.asarray(descent_direction, float)
    if d.shape != (tangent.shape[1],) or not np.isfinite(d).all():
        raise ValueError('finite real parameter direction of correct shape required')
    v = tangent@d
    size, target = float(np.linalg.norm(v)), float(np.linalg.norm(residual))
    inner = float(np.vdot(v, residual).real)
    denominator = size*target
    return dict(tangent_fs_speed=size, energy_residual_norm=target,
        descent_alignment_cosine=inner/denominator if denominator > 0 else 0.,
        predicted_one_sided_energy_slope_total=-2*inner,
        predicted_one_sided_unit_fs_slope_total=-2*inner/size if size > 0 else 0.,
        residual_fit_fraction=1-float(np.linalg.norm(residual-v)**2)/target**2 if target > 0 else 0.,
        finite_quadrature_is_not_descent_certificate=True)


def importance_sr_direction(psi: np.ndarray, hpsi: np.ndarray, jacobian: np.ndarray,
                            q: np.ndarray, *, damping: float = .1) -> tuple[np.ndarray, dict]:
    """Solve (2 Re J†J + damping I)d=2 Re J†r; the update is -d.

    Explicit sample-space ridge solve; same 2*FS convention as Born SR. No
    truncation/clipping of rare amplitudes, H products or derivatives. Physical
    trust matching and carrier protection must be applied by the caller.
    """
    if isinstance(damping, bool) or not math.isfinite(damping) or damping <= 0:
        raise ValueError('positive finite damping required')
    a, b, _, _ = tangent_system(psi, hpsi, jacobian, q)
    coefficients = np.linalg.solve(a@a.T+damping*np.eye(len(b)), b)
    direction = a.T@coefficients
    gradient = a.T@b
    error = a.T@(a@direction)+damping*direction-gradient
    relative = float(np.linalg.norm(error)/max(np.linalg.norm(gradient), 1e-30))
    if not np.isfinite(direction).all() or relative > 1e-7:
        raise FloatingPointError('amplitude-space SR solve failed its independent residual')
    result = direction_diagnostics(psi, hpsi, jacobian, q, direction)
    result.update(relative_linear_residual=relative, gradient_norm=float(np.linalg.norm(gradient)),
                  raw_direction_norm=float(np.linalg.norm(direction)),
                  sample_matrix_dimension=len(b), damping=float(damping),
                  exact_node_records=int(np.sum(np.asarray(psi) == 0)))
    return direction, result


__all__ = ['make_amplitude_jacobian', 'tangent_system',
           'direction_diagnostics', 'importance_sr_direction']
