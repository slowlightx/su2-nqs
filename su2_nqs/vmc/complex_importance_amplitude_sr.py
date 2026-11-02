"""Node-safe complex-amplitude SR in real native parameter coordinates.

The metric is ``F = Re(J_perp.conj().T @ J_perp)`` (not ``2F``), and the
population Hermitian force is ``g = 2 Re(J_perp^dagger (H-E)psi)``.  Amplitudes
may be complex; all trainable parameter leaves remain real float64.  A complex
ray projection removes both normalization and global-phase directions.

Only one raw complex ``[records, parameters]`` Jacobian is retained.  Bounded
column blocks are projected and realified to form a ``[2N, 2N]`` dual Gram,
never a parameter-space Fisher matrix or another full normalized Jacobian.
The realification is essential: ``Re(A @ A.conj().T)`` alone is NOT the dual
system for a real parameter metric.  Positive proposal support for the whole
physical tangent, including exact target nodes, is the caller's responsibility.

Supplied ``Hpsi`` and positive ``Q`` are fixed data.  The one-sided Hermitian
force is a population estimator, not the derivative of an arbitrary finite
importance quadrature.  A numerically valid proposed update still requires
fresh finite-state-distance and independent full-H acceptance checks.
"""
from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')


def solve_complex_projected_block_sr(
    jacobian: jax.Array,
    psi: jax.Array,
    hpsi: jax.Array,
    density: jax.Array,
    *,
    damping: float,
    parameter_chunk_size: int = 8192,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Return real ``direction, gradient, info`` for complex physical states.

    For N records, ``Zsum=sum(|psi|^2/Q)``, ``u=psi/sqrt(Q Zsum)``,
    ``D=J/sqrt(Q Zsum)``, ``A=(I-u u^dagger)D`` and
    ``b=2(Hpsi-E psi)/sqrt(Q Zsum)``.  The solver uses
    ``F=Re(A^dagger A)``, ``g=Re(A^dagger b)`` and ``(F+lambda I)d=g``.
    No division by psi, phase repair, amplitude floor or density clipping occurs.

    J is an amplitude derivative with respect to REAL parameters.  J, psi and
    Hpsi must have dtype complex128 or float64 (the latter are promoted), and
    Q must be float64 and strictly positive even at target nodes.  Common
    nonzero complex rescaling of J/psi/Hpsi, positive rescaling of Q, and a real
    Hamiltonian energy shift leave direction and gradient invariant.

    Invalid concrete psi/Hpsi/Q values raise ValueError.  The large Jacobian
    is checked blockwise, without copying it to the host.  Under JIT, invalid
    dynamic inputs yield ``numerically_valid=False`` and NaN vectors, never a
    repaired proposal.  Complex sampled Rayleigh imaginary parts are retained
    as an estimator diagnostic; importance sampling need not make them zero.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('complex importance amplitude SR requires x64')
    jacobian, psi, hpsi, density = map(jnp.asarray, (jacobian, psi, hpsi, density))
    if (jacobian.ndim != 2 or min(jacobian.shape) < 1 or jacobian.shape[0] < 2
            or any(x.shape != (jacobian.shape[0],) for x in (psi, hpsi, density))
            or any(x.dtype not in (jnp.float64, jnp.complex128) for x in (jacobian, psi, hpsi))
            or density.dtype != jnp.float64):
        raise ValueError('float64/complex128 J[N>=2,P>=1], psi/Hpsi and real float64 Q required')
    if isinstance(damping, bool) or not isfinite(damping) or damping <= 0:
        raise ValueError('positive finite damping required')
    _positive_integer(parameter_chunk_size, 'parameter_chunk_size')
    for name, value in (('psi', psi), ('Hpsi', hpsi), ('Q', density)):
        if not isinstance(value, jax.core.Tracer):
            host = np.asarray(value)
            if not np.isfinite(host).all() or (name == 'Q' and np.any(host <= 0)):
                raise ValueError(name + ' must be finite; Q must be strictly positive')
    jacobian, psi, hpsi = (x.astype(jnp.complex128) for x in (jacobian, psi, hpsi))
    density, hpsi = jax.lax.stop_gradient(density), jax.lax.stop_gradient(hpsi)
    count, parameters = jacobian.shape
    width = min(parameter_chunk_size, parameters)
    blocks = (parameters + width - 1) // width
    real_dtype = jnp.float64
    mass = jnp.abs(psi)**2 / density
    norm = mass.sum()
    if not isinstance(norm, jax.core.Tracer) and (not np.isfinite(float(norm)) or float(norm) <= 0):
        raise ValueError('positive finite sampled parent norm required')
    scale = jax.lax.rsqrt(density * norm)
    u = psi * scale
    rayleigh = jnp.sum(jnp.conj(psi) * hpsi / density) / norm
    energy = jnp.real(rayleigh)
    force = 2 * (hpsi - energy * psi) * scale
    real_force = jnp.concatenate((jnp.real(force), jnp.imag(force)))
    valid = (jnp.all(jnp.isfinite(psi)) & jnp.all(jnp.isfinite(hpsi))
             & jnp.all(jnp.isfinite(density)) & jnp.all(density > 0)
             & jnp.isfinite(norm) & (norm > 0) & jnp.isfinite(rayleigh)
             & jnp.all(jnp.isfinite(force)))

    def block(i: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        first = jnp.minimum(i * width, parameters - width)
        values = jax.lax.dynamic_slice(jacobian, (0, first), (count, width))
        scaled = values * scale[:, None]
        projected = scaled - u[:, None] * (jnp.conj(u) @ scaled)[None, :]
        columns = first + jnp.arange(width) >= i * width
        projected = jnp.where(columns[None, :], projected, 0.)
        a = jnp.concatenate((jnp.real(projected), jnp.imag(projected)), axis=0)
        return first, a, jnp.all(jnp.isfinite(values))

    def gram_body(i: jax.Array, carry: tuple[jax.Array, jax.Array]):
        gram, finite = carry
        _, a, ok = block(i)
        return gram + a @ a.T, finite & ok

    gram, valid = jax.lax.fori_loop(0, blocks, gram_body,
        (jnp.zeros((2 * count, 2 * count), real_dtype), valid))
    coefficients = jax.scipy.linalg.solve(
        gram + damping * jnp.eye(2 * count, dtype=real_dtype), real_force, assume_a='pos')

    def vectors_body(i: jax.Array, carry: tuple[jax.Array, jax.Array, jax.Array]):
        direction, gradient, product = carry
        first, a, _ = block(i)
        d, g = a.T @ coefficients, a.T @ real_force
        old_d = jax.lax.dynamic_slice(direction, (first,), (width,))
        old_g = jax.lax.dynamic_slice(gradient, (first,), (width,))
        # The masked final partial block can overlap a completed prefix.
        direction = jax.lax.dynamic_update_slice(direction, old_d + d, (first,))
        gradient = jax.lax.dynamic_update_slice(gradient, old_g + g, (first,))
        return direction, gradient, product + a @ d

    direction, gradient, real_product = jax.lax.fori_loop(0, blocks, vectors_body,
        (jnp.zeros(parameters, real_dtype), jnp.zeros(parameters, real_dtype),
         jnp.zeros(2 * count, real_dtype)))

    def residual_body(i: jax.Array, squared: jax.Array) -> jax.Array:
        first, a, _ = block(i)
        d = jax.lax.dynamic_slice(direction, (first,), (width,))
        g = jax.lax.dynamic_slice(gradient, (first,), (width,))
        columns = first + jnp.arange(width) >= i * width
        residual = jnp.where(columns, a.T @ real_product + damping * d - g, 0.)
        return squared + jnp.vdot(residual, residual).real

    squared = jax.lax.fori_loop(0, blocks, residual_body, jnp.asarray(0., real_dtype))
    gradient_norm = jnp.linalg.norm(gradient)
    relative = jnp.where(gradient_norm > 0, jnp.sqrt(squared) / gradient_norm,
        jnp.where(squared == 0, 0., jnp.inf))
    valid = (valid & jnp.all(jnp.isfinite(direction)) & jnp.all(jnp.isfinite(gradient))
             & jnp.isfinite(relative))
    product = real_product[:count] + 1j * real_product[count:]
    tangent_mass = jnp.abs(product)**2
    signed_force_mass = jnp.real(jnp.conj(product) * force)
    absolute_force_mass = jnp.abs(signed_force_mass)
    tangent_total, force_total = tangent_mass.sum(), absolute_force_mass.sum()
    tangent_defined = jnp.isfinite(tangent_total) & (tangent_total > 0)
    force_defined = jnp.isfinite(force_total) & (force_total > 0)
    return jnp.where(valid, direction, jnp.nan), jnp.where(valid, gradient, jnp.nan), dict(
        numerically_valid=valid, relative_linear_residual=relative, gradient_norm=gradient_norm,
        raw_fisher_norm=jnp.linalg.norm(real_product), sampled_parent_norm=norm / count,
        sampled_energy_total=energy, sampled_energy_imaginary=jnp.imag(rayleigh),
        exact_node_records=jnp.sum(psi == 0),
        norm_effective_contributions=norm**2 / jnp.sum(mass**2),
        maximum_norm_contribution=jnp.max(mass) / norm,
        parent_ray_norm_squared=jnp.vdot(u, u).real, parameter_blocks=jnp.asarray(blocks),
        dual_real_dimension=jnp.asarray(2 * count),
        chosen_direction_tangent_square_mass=tangent_total,
        chosen_direction_tangent_concentration_defined=tangent_defined,
        chosen_direction_tangent_square_ESS=jnp.where(tangent_defined,
            tangent_total**2 / jnp.sum(tangent_mass**2), 0.),
        chosen_direction_tangent_square_maximum_fraction=jnp.where(tangent_defined,
            jnp.max(tangent_mass) / tangent_total, 0.),
        chosen_direction_absolute_force_projection_mass=force_total,
        chosen_direction_signed_force_projection_sum=signed_force_mass.sum(),
        chosen_direction_force_concentration_defined=force_defined,
        chosen_direction_absolute_force_projection_ESS=jnp.where(force_defined,
            force_total**2 / jnp.sum(absolute_force_mass**2), 0.),
        chosen_direction_absolute_force_projection_maximum_fraction=jnp.where(force_defined,
            jnp.max(absolute_force_mass) / force_total, 0.),
        concentration_is_chosen_direction_only_not_full_tangent_coverage=jnp.asarray(True),
        force_projection_contributions_are_real=jnp.asarray(True),
        maximum_normalized_jacobian_block_entries=jnp.asarray(count * width),
        retained_raw_jacobian_entries=jnp.asarray(jacobian.size),
        metric_is_F_not_two_F=jnp.asarray(True), real_parameter_coordinates=jnp.asarray(True),
        complex_ray_projection_removes_scaling_and_phase=jnp.asarray(True),
        force_is_population_Hermitian_estimator=jnp.asarray(True))


def make_complex_importance_amplitude_sr(
    amplitude: Callable,
    template: object,
    *,
    damping: float = .5,
    score_chunk: int = 2,
    parameter_chunk_size: int = 8192,
    sr_tangent_fs_radius: float = .003,
    prepare: Callable | None = None,
    prepared_amplitude: Callable | None = None,
) -> Callable:
    """Build ``update(params,moments,rows,Hpsi,Q,lr)`` with real parameters.

    Callbacks evaluate one configuration and one complex scalar amplitude.
    Real and imaginary amplitude derivatives are pulled back separately to
    every original real parameter.  Optional preparation is shared, with its
    VJP differentiated through the full native coordinates, never substituted
    by sampling-gauge/canonical tensors as if these were native parameters.

    Hpsi and the callback must use the SAME fixed complex amplitude scale.
    The provisional step is ``theta-scale*d``, where scale is the smaller of
    lr and the requested tangent FS radius divided by ``||A d||``.  Inherited
    moment trees are unchanged and their counter increments only on numerical
    validity.  ``accepted`` marks a PROPOSAL, not finite-FS/energy acceptance;
    the caller must guard fresh physical-state updates before committing.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('complex importance amplitude SR requires x64 before loading parameters')
    leaves = jax.tree.leaves(template)
    if not leaves or any(np.asarray(v).dtype != np.dtype('float64')
            or not np.isfinite(np.asarray(v)).all() for v in leaves):
        raise ValueError('finite real float64 native parameter leaves required')
    if not callable(amplitude):
        raise ValueError('one-row amplitude callback required')
    if isinstance(damping, bool) or not isfinite(damping) or damping <= 0:
        raise ValueError('positive finite damping required')
    _positive_integer(score_chunk, 'score_chunk')
    _positive_integer(parameter_chunk_size, 'parameter_chunk_size')
    if (isinstance(sr_tangent_fs_radius, bool) or not isfinite(sr_tangent_fs_radius)
            or sr_tangent_fs_radius <= 0):
        raise ValueError('positive finite tangent FS radius required')
    if (prepare is None) != (prepared_amplitude is None) or (prepare is not None
            and (not callable(prepare) or not callable(prepared_amplitude))):
        raise ValueError('both callable preparation callbacks, or neither, required')
    _, unravel = ravel_pytree(template)

    @jax.jit
    def update(params, moments, rows, hpsi, density, learning_rate):
        theta, _ = ravel_pytree(params)
        if theta.dtype != jnp.float64:
            raise ValueError('updated native parameters must remain real float64')
        one = jnp.asarray(1., jnp.complex128)
        minus_i = jnp.asarray(-1j, jnp.complex128)
        if prepare is None:
            def row_value(row):
                value, pullback = jax.vjp(
                    lambda values: jnp.asarray(amplitude(unravel(values), row), jnp.complex128), theta)
                return value, pullback(one)[0] + 1j * pullback(minus_i)[0]
        else:
            bank, native_pullback = jax.vjp(lambda values: prepare(unravel(values)), theta)
            def row_value(row):
                value, pullback = jax.vjp(
                    lambda data: jnp.asarray(prepared_amplitude(data, row), jnp.complex128), bank)
                real_derivative = native_pullback(pullback(one)[0])[0]
                imag_derivative = native_pullback(pullback(minus_i)[0])[0]
                return value, real_derivative + 1j * imag_derivative
        psi, jacobian = jax.lax.map(row_value, rows, batch_size=score_chunk)
        direction, gradient, info = solve_complex_projected_block_sr(jacobian, psi, hpsi, density,
            damping=damping, parameter_chunk_size=parameter_chunk_size)
        natural_norm = info['raw_fisher_norm']
        scale = jnp.where(natural_norm > 0,
            jnp.minimum(learning_rate, sr_tangent_fs_radius / natural_norm), learning_rate)
        proposed = theta - scale * direction
        valid = (info['numerically_valid'] & (info['relative_linear_residual'] < 1e-7)
            & jnp.isfinite(learning_rate) & (learning_rate > 0) & jnp.all(jnp.isfinite(proposed)))
        next_moments = (jnp.where(valid, moments[0] + 1, moments[0]), moments[1], moments[2])
        return unravel(jnp.where(valid, proposed, theta)), next_moments, dict(info,
            accepted=valid, proposal_only_requires_fresh_finite_guard=jnp.asarray(True),
            update_scale=jnp.where(valid, scale, 0.),
            sampled_tangent_FS=jnp.where(valid, scale * natural_norm, 0.),
            relative_SR_residual=info['relative_linear_residual'],
            energy_gradient_norm=info['gradient_norm'],
            raw_energy_directional_derivative=jnp.vdot(gradient, proposed - theta).real,
            applied_energy_directional_derivative=jnp.where(valid,
                jnp.vdot(gradient, proposed - theta).real, 0.))
    return update


# Same names/signatures can be imported from this OPTIONAL complex module;
# the existing real implementation and every production default are unchanged.
solve_projected_block_sr = solve_complex_projected_block_sr
make_importance_amplitude_sr = make_complex_importance_amplitude_sr

__all__ = ['solve_complex_projected_block_sr', 'make_complex_importance_amplitude_sr',
           'solve_projected_block_sr', 'make_importance_amplitude_sr']
