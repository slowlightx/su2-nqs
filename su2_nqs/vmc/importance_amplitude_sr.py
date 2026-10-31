"""Real, node-safe importance-amplitude SR in original parameter coordinates.

The metric is F=Re<J_perp|J_perp>, NOT 2F. The Hermitian population
force is g=2 Re<J_perp|(H-E)psi>, preserving the existing physical-SR
ridge convention. Supplied Hpsi and strictly positive Q are fixed data.
No Hamiltonian, sampler, amplitude floor, or log derivative is owned here.

One real [records,parameters] amplitude Jacobian is retained. Only bounded
parameter blocks are normalized/projected; no second full Jacobian or
parameter-space Fisher matrix is constructed. Proposal support for ALL
tangent-square contributions remains the caller's responsibility. Finite
importance ratios and the one-sided Hermitian force are not certificates
of population geometry or of the finite-quadrature full-H derivative.
"""
from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(name+' must be a positive integer')


def solve_projected_block_sr(
    jacobian: jax.Array,
    psi: jax.Array,
    hpsi: jax.Array,
    density: jax.Array,
    *,
    damping: float,
    parameter_chunk_size: int=8192,
) -> tuple[jax.Array,jax.Array,dict[str,jax.Array]]:
    """Solve (A.T A+lambda I)d=A.T b with node-safe physical projection.

    For n records, Z=mean(psi²/Q), u=psi/sqrt(n Q Z),
    D=J/sqrt(n Q Z), A=(I-u u.T)D, and b=2(Hpsi-E psi)/sqrt(n Q Z).
    Real float64 arrays are required; J contains amplitude, not log-amplitude,
    derivatives. Q is positive even at target nodes. A common fixed scaling
    of (psi,Hpsi,J), or of Q, does not change the solution.

    Under JIT invalid dynamic values produce ``numerically_valid=False`` and
    NaN direction/force, never repaired Q or clipped derivatives. Static
    contract errors and concrete invalid psi/Hpsi/Q raise ValueError. The
    large Jacobian is checked blockwise, without a host copy.
    """
    if not jax.config.x64_enabled:raise RuntimeError('importance amplitude SR requires x64')
    jacobian,psi,hpsi,density=map(jnp.asarray,(jacobian,psi,hpsi,density))
    if (jacobian.ndim!=2 or min(jacobian.shape)<1 or jacobian.shape[0]<2
            or any(x.shape!=(jacobian.shape[0],) for x in (psi,hpsi,density))
            or any(x.dtype!=jnp.float64 for x in (jacobian,psi,hpsi,density))):
        raise ValueError('real float64 J[N>=2,P>=1] and matching psi/Hpsi/Q required')
    if isinstance(damping,bool) or not isfinite(damping) or damping<=0:
        raise ValueError('positive finite damping required')
    _positive_integer(parameter_chunk_size,'parameter_chunk_size')
    for name,value in (('psi',psi),('Hpsi',hpsi),('Q',density)):
        if not isinstance(value,jax.core.Tracer):
            host=np.asarray(value)
            if not np.isfinite(host).all() or (name=='Q' and np.any(host<=0)):
                raise ValueError(name+' must be finite; Q must be strictly positive')
    density=jax.lax.stop_gradient(density);hpsi=jax.lax.stop_gradient(hpsi)
    count,parameters=jacobian.shape;width=min(parameter_chunk_size,parameters)
    blocks=(parameters+width-1)//width;dtype=jacobian.dtype
    mass=psi**2/density;norm=mass.sum()
    if not isinstance(norm,jax.core.Tracer) and (not np.isfinite(float(norm)) or float(norm)<=0):
        raise ValueError('positive finite sampled parent norm required')
    scale=jax.lax.rsqrt(density*norm)
    u=psi*scale;energy=jnp.sum(psi*hpsi/density)/norm
    force=2*(hpsi-energy*psi)*scale
    valid=(jnp.all(jnp.isfinite(psi)) & jnp.all(jnp.isfinite(hpsi))
        & jnp.all(jnp.isfinite(density)) & jnp.all(density>0)
        & jnp.isfinite(norm) & (norm>0) & jnp.all(jnp.isfinite(force)))

    def block(i):
        first=jnp.minimum(i*width,parameters-width)
        values=jax.lax.dynamic_slice(jacobian,(0,first),(count,width))
        scaled=values*scale[:,None]
        a=scaled-u[:,None]*(u@scaled)[None,:]
        columns=first+jnp.arange(width)>=i*width
        return first,jnp.where(columns[None,:],a,0.),jnp.all(jnp.isfinite(values))

    def gram_body(i,carry):
        gram,finite=carry;_,a,ok=block(i)
        return gram+a@a.T,finite & ok
    gram,valid=jax.lax.fori_loop(0,blocks,gram_body,(jnp.zeros((count,count),dtype),valid))
    coefficients=jax.scipy.linalg.solve(gram+damping*jnp.eye(count,dtype=dtype),force,assume_a='pos')

    def vectors_body(i,carry):
        direction,gradient,product=carry;first,a,_=block(i)
        d,g=a.T@coefficients,a.T@force
        old_d=jax.lax.dynamic_slice(direction,(first,),(width,))
        old_g=jax.lax.dynamic_slice(gradient,(first,),(width,))
        # The masked final partial block may overlap a completed prefix.
        direction=jax.lax.dynamic_update_slice(direction,old_d+d,(first,))
        gradient=jax.lax.dynamic_update_slice(gradient,old_g+g,(first,))
        return direction,gradient,product+a@d
    direction,gradient,product=jax.lax.fori_loop(0,blocks,vectors_body,
        (jnp.zeros(parameters,dtype),jnp.zeros(parameters,dtype),jnp.zeros(count,dtype)))

    def residual_body(i,squared):
        first,a,_=block(i)
        d=jax.lax.dynamic_slice(direction,(first,),(width,))
        g=jax.lax.dynamic_slice(gradient,(first,),(width,))
        columns=first+jnp.arange(width)>=i*width
        residual=jnp.where(columns,a.T@product+damping*d-g,0.)
        return squared+jnp.vdot(residual,residual).real
    squared=jax.lax.fori_loop(0,blocks,residual_body,jnp.asarray(0.,dtype))
    gradient_norm=jnp.linalg.norm(gradient)
    # The denominator convention is diagnostic only, not a wavefunction or Q floor.
    relative=jnp.where(gradient_norm>0,jnp.sqrt(squared)/gradient_norm,
        jnp.where(squared==0,0.,jnp.inf))
    valid=valid & jnp.all(jnp.isfinite(direction)) & jnp.all(jnp.isfinite(gradient)) & jnp.isfinite(relative)
    # These N-entry diagnostics concern this ONE fitted direction, not all
    # native tangents or unseen tails. A zero mass is explicitly undefined;
    # its displayed ESS/max are zero, without inserting a positive floor.
    tangent_mass=product**2;absolute_force_mass=jnp.abs(product*force)
    tangent_total=tangent_mass.sum();force_total=absolute_force_mass.sum()
    tangent_defined=jnp.isfinite(tangent_total) & (tangent_total>0)
    force_defined=jnp.isfinite(force_total) & (force_total>0)
    return jnp.where(valid,direction,jnp.nan),jnp.where(valid,gradient,jnp.nan),dict(
        numerically_valid=valid,relative_linear_residual=relative,gradient_norm=gradient_norm,
        raw_fisher_norm=jnp.linalg.norm(product),sampled_parent_norm=norm/count,
        sampled_energy_total=energy,exact_node_records=jnp.sum(psi==0),
        norm_effective_contributions=norm**2/jnp.sum(mass**2),maximum_norm_contribution=jnp.max(mass)/norm,
        parent_ray_norm_squared=jnp.vdot(u,u).real,parameter_blocks=jnp.asarray(blocks),
        chosen_direction_tangent_square_mass=tangent_total,
        chosen_direction_tangent_concentration_defined=tangent_defined,
        chosen_direction_tangent_square_ESS=jnp.where(tangent_defined,tangent_total**2/jnp.sum(tangent_mass**2),0.),
        chosen_direction_tangent_square_maximum_fraction=jnp.where(tangent_defined,jnp.max(tangent_mass)/tangent_total,0.),
        chosen_direction_absolute_force_projection_mass=force_total,
        chosen_direction_signed_force_projection_sum=jnp.vdot(product,force).real,
        chosen_direction_force_concentration_defined=force_defined,
        chosen_direction_absolute_force_projection_ESS=jnp.where(force_defined,force_total**2/jnp.sum(absolute_force_mass**2),0.),
        chosen_direction_absolute_force_projection_maximum_fraction=jnp.where(force_defined,jnp.max(absolute_force_mass)/force_total,0.),
        concentration_is_chosen_direction_only_not_full_tangent_coverage=jnp.asarray(True),
        maximum_normalized_jacobian_block_entries=jnp.asarray(count*width),
        retained_raw_jacobian_entries=jnp.asarray(jacobian.size),
        metric_is_F_not_two_F=jnp.asarray(True),force_is_population_Hermitian_estimator=jnp.asarray(True))


def make_importance_amplitude_sr(
    amplitude: Callable,
    template: object,
    *,
    damping: float=.5,
    score_chunk: int=2,
    parameter_chunk_size: int=8192,
    sr_tangent_fs_radius: float=.003,
    prepare: Callable|None=None,
    prepared_amplitude: Callable|None=None,
) -> Callable:
    """Build update(params,moments,rows,Hpsi,Q,lr) -> provisional native state.

    Callbacks evaluate ONE configuration and a real scalar amplitude. Optional
    shared preparation is differentiated by one VJP; each prepared-amplitude
    derivative is pulled back to ALL original native parameter columns.
    Canonical sampling/gauge tables must not replace these native coordinates.

    Hpsi and Q use the SAME fixed amplitude scale as the callback. No local
    energy or division by psi occurs. The proposal is theta−scale*d, with
    scale=min(lr, radius/||A d||). On numerical validity only, the proposed
    optimizer counter increments once and both inherited moment trees remain
    unchanged. ``accepted`` means a numerically valid PROPOSAL, not an energy
    or finite-FS acceptance: the caller must apply its fresh finite-state guard
    before committing either parameters or optimizer. No hidden retries occur.
    """
    if not jax.config.x64_enabled:raise RuntimeError('importance amplitude SR requires x64 before loading parameters')
    leaves=jax.tree.leaves(template)
    if not leaves or any(np.asarray(v).dtype!=np.dtype('float64') or not np.isfinite(np.asarray(v)).all() for v in leaves):
        raise ValueError('finite real float64 native parameter leaves required')
    if not callable(amplitude):raise ValueError('one-row amplitude callback required')
    if isinstance(damping,bool) or not isfinite(damping) or damping<=0:
        raise ValueError('positive finite damping required')
    _positive_integer(score_chunk,'score_chunk');_positive_integer(parameter_chunk_size,'parameter_chunk_size')
    if isinstance(sr_tangent_fs_radius,bool) or not isfinite(sr_tangent_fs_radius) or sr_tangent_fs_radius<=0:
        raise ValueError('positive finite tangent FS radius required')
    if (prepare is None)!=(prepared_amplitude is None) or (prepare is not None and
            (not callable(prepare) or not callable(prepared_amplitude))):
        raise ValueError('both callable preparation callbacks, or neither, required')
    _,unravel=ravel_pytree(template)

    @jax.jit
    def update(params,moments,rows,hpsi,density,learning_rate):
        theta,_=ravel_pytree(params)
        if prepare is None:
            def value(values,row):return amplitude(unravel(values),row)
            evaluate=jax.value_and_grad(value)
            psi,jacobian=jax.lax.map(lambda row:evaluate(theta,row),rows,batch_size=score_chunk)
        else:
            bank,pullback=jax.vjp(lambda values:prepare(unravel(values)),theta)
            evaluate=jax.value_and_grad(prepared_amplitude)
            def row_value(row):
                value,derivative=evaluate(bank,row)
                return value,pullback(derivative)[0]
            psi,jacobian=jax.lax.map(row_value,rows,batch_size=score_chunk)
        direction,gradient,info=solve_projected_block_sr(jacobian,psi,hpsi,density,
            damping=damping,parameter_chunk_size=parameter_chunk_size)
        natural_norm=info['raw_fisher_norm']
        scale=jnp.where(natural_norm>0,jnp.minimum(learning_rate,sr_tangent_fs_radius/natural_norm),learning_rate)
        proposed=theta-scale*direction
        valid=(info['numerically_valid'] & (info['relative_linear_residual']<1e-7)
            & jnp.isfinite(learning_rate) & (learning_rate>0) & jnp.all(jnp.isfinite(proposed)))
        next_moments=(jnp.where(valid,moments[0]+1,moments[0]),moments[1],moments[2])
        return unravel(jnp.where(valid,proposed,theta)),next_moments,dict(info,
            accepted=valid,proposal_only_requires_fresh_finite_guard=jnp.asarray(True),
            update_scale=jnp.where(valid,scale,0.),sampled_tangent_FS=jnp.where(valid,scale*natural_norm,0.),
            relative_SR_residual=info['relative_linear_residual'],energy_gradient_norm=info['gradient_norm'],
            raw_energy_directional_derivative=jnp.vdot(gradient,proposed-theta).real,
            applied_energy_directional_derivative=jnp.where(valid,jnp.vdot(gradient,proposed-theta).real,0.))
    return update


__all__=['solve_projected_block_sr','make_importance_amplitude_sr']
