"""Optional two-force natural updates of complex states in real coordinates.

By default, energy and the real-centred full-H residual are differentiated on
the same fixed importance panel. Optional modes use a population-Hermitian
energy proposal or filter empirical Fisher support; they change the proposal
rule, not the independent acceptance test. No target-amplitude division or
H-squared query is needed. Quantum labels and physical projectors belong to
the caller.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def product_cotangents(products: jax.Array, density: jax.Array) -> tuple:
    """Return exact fixed-panel E/variance cotangents for (psi,Hpsi).

    The centre E is real. Cotangents follow JAX's complex VJP convention.
    Both coherent energy derivatives and normalization derivatives are kept.
    """
    p, q = jnp.asarray(products), jnp.asarray(density)
    if p.ndim != 2 or p.shape[1] != 2 or q.shape != (len(p),):
        raise ValueError('products[N,2] and positive density[N] required')
    if not jax.config.x64_enabled:
        raise RuntimeError('float64/complex128 required')
    z = jnp.sum(abs(p[:, 0])**2 / q)
    energy = jnp.sum(jnp.real(jnp.conj(p[:, 0])*p[:, 1])/q)/z
    variance = jnp.sum(abs(p[:, 1]-energy*p[:, 0])**2/q)/z
    def energy_loss(v):
        return jnp.sum((jnp.real(jnp.conj(v[:, 0])*v[:, 1])-energy*abs(v[:, 0])**2)/q)/z
    def variance_loss(v):
        return jnp.sum((abs(v[:, 1]-energy*v[:, 0])**2-variance*abs(v[:, 0])**2)/q)/z
    return (jax.grad(energy_loss)(p), jax.grad(variance_loss)(p),
            dict(energy=energy, variance=variance, norm=z))


def solve_balanced_forces(jacobian: jax.Array, psi: jax.Array, density: jax.Array,
                          forces: jax.Array, *, damping: float,
                          parameter_chunk_size: int = 8192,
                          support_filter: bool = False,
                          damping_candidates: tuple[float, ...] | None = None,
                          minimum_tangent_ess_fraction: float = 0.) -> tuple:
    """Solve two regularized full Fisher systems, retaining cross-node terms.

    forces has shape [2,P]: actual energy and residual-variance gradients.
    The direction is d_E/||g_E||_(F+lambda)^-1 +
    d_V/||g_V||_(F+lambda)^-1. Its negative is a joint descent direction
    unless the gradients are opposite in this metric. The Woodbury inverse
    includes the force component outside the sampled Jacobian row space.
    Optional support_filter uses F(F+lambda I)^-2 on each force, suppressing
    empirical null directions. This filtered rule is not the original SR
    inverse, and does not certify population support. An optional increasing
    damping grid reuses the same Gram matrix and full derivatives, selecting
    the first joint direction with the requested empirical tangent ESS. This
    is FIT-only proposal design, never independent confirmation.
    """
    if (type(support_filter) is not bool or not isfinite(damping) or damping <= 0 or type(parameter_chunk_size) is not int
            or parameter_chunk_size < 1):
        raise ValueError('positive damping and integer parameter chunk required')
    grid = (damping,) if damping_candidates is None else damping_candidates
    if (not isinstance(grid, tuple) or not grid or any(not isfinite(x) or x <= 0 for x in grid)
            or grid[0] != damping or any(b <= a for a, b in zip(grid, grid[1:]))
            or not isfinite(minimum_tangent_ess_fraction) or not 0 <= minimum_tangent_ess_fraction <= 1):
        raise ValueError('increasing positive damping tuple starting at damping and ESS fraction in[0,1] required')
    j, p, q, g = map(jnp.asarray, (jacobian, psi, density, forces))
    if (j.ndim != 2 or len(j) < 2 or p.shape != (len(j),) or q.shape != p.shape
            or g.shape != (2,j.shape[1]) or g.dtype != jnp.float64):
        raise ValueError('complex J[N,P], psi/Q[N], real gradients[2,P] required')
    n, size = j.shape
    width = min(parameter_chunk_size,size); blocks=(size+width-1)//width
    z=jnp.sum(abs(p)**2/q); scale=jax.lax.rsqrt(q*z); u=p*scale
    def block(i):
        start=jnp.minimum(i*width,size-width)
        dj=jax.lax.dynamic_slice(j,(0,start),(n,width))*scale[:,None]
        dj=dj-u[:,None]*(jnp.conj(u)@dj)[None,:]
        keep=start+jnp.arange(width)>=i*width
        return start,jnp.where(keep[None,:],jnp.concatenate([dj.real,dj.imag],axis=0),0.),keep
    def first(i,carry):
        gram,ag=carry;start,a,_=block(i)
        gb=jax.lax.dynamic_slice(g,(0,start),(2,width))
        return gram+a@a.T,ag+a@gb.T
    gram,ag=jax.lax.fori_loop(0,blocks,first,(jnp.zeros((2*n,2*n)),jnp.zeros((2*n,2))))
    def at_damping(ridge):
        coefficients=jax.scipy.linalg.solve(gram+ridge*jnp.eye(2*n),ag,assume_a='pos')
        if support_filter:
            supported_coefficients=jax.scipy.linalg.solve(
                gram+ridge*jnp.eye(2*n),coefficients,assume_a='pos')
        def second(i,out):
            start,a,keep=block(i);gb=jax.lax.dynamic_slice(g,(0,start),(2,width))
            raw=(a.T@supported_coefficients).T if support_filter else (gb-(a.T@coefficients).T)/ridge
            value=jnp.where(keep[None,:],raw,0.)
            previous=jax.lax.dynamic_slice(out,(0,start),(2,width))
            return jax.lax.dynamic_update_slice(out,previous+value,(0,start))
        directions=jax.lax.fori_loop(0,blocks,second,jnp.zeros_like(g))
        norm2=jnp.sum(g*directions,axis=1)
        metric_norm=jnp.sqrt(jnp.maximum(norm2,0.))
        cosine=jnp.vdot(g[0],directions[1]).real/jnp.prod(metric_norm)
        direction=jnp.sum(directions/metric_norm[:,None],axis=0)
        slopes=g@direction
        def action(i,carry):
            products=carry;start,a,_=block(i)
            db=jax.lax.dynamic_slice(directions,(0,start),(2,width))
            return products+a@db.T
        products=jax.lax.fori_loop(0,blocks,action,jnp.zeros((2*n,2)))
        def residual(i,carry):
            sq,physical,rhs_squared=carry;start,a,keep=block(i)
            db=jax.lax.dynamic_slice(directions,(0,start),(2,width))
            gb=jax.lax.dynamic_slice(g,(0,start),(2,width))
            rhs=(a.T@coefficients).T if support_filter else gb
            rhs=jnp.where(keep[None,:],rhs,0.)
            res=jnp.where(keep[None,:],(a.T@products).T+ridge*db-rhs,0.)
            dbal=jax.lax.dynamic_slice(direction,(start,),(width,))
            return sq+jnp.sum(res*res,axis=1),physical+a@dbal,rhs_squared+jnp.sum(rhs*rhs,axis=1)
        sq,physical,rhs_squared=jax.lax.fori_loop(0,blocks,residual,
            (jnp.zeros(2),jnp.zeros(2*n),jnp.zeros(2)))
        relative=jnp.sqrt(sq/rhs_squared)
        valid=(jnp.all(jnp.isfinite(direction)) & jnp.all(jnp.isfinite(g)) &
               jnp.all(norm2>0) & jnp.all(slopes>0) & jnp.all(relative<1e-7) &
               jnp.all(q>0) & (z>0) & jnp.isfinite(cosine) & (cosine>-.999999))
        mass=physical[:n]**2+physical[n:]**2
        tangent_ess=jnp.sum(mass)**2/jnp.maximum(jnp.sum(mass**2),1e-300)
        return direction,dict(numerically_valid=valid,relative_linear_residual=jnp.max(relative),
            force_relative_residuals=relative,metric_force_cosine=cosine,force_metric_norms=metric_norm,
            predicted_energy_descent=slopes[0],predicted_variance_descent=slopes[1],
            raw_fisher_norm=jnp.linalg.norm(physical),gradient_norm=jnp.linalg.norm(g[0]),
            parameter_blocks=blocks,dual_real_dimension=2*n,retained_raw_jacobian_entries=j.size,
            arbitrary_force_outside_sampled_rowspace_retained=not support_filter,
            empirical_support_filter=support_filter,
            supported_force_norms=jnp.sqrt(rhs_squared),selected_damping=ridge,
            metric_tangent_square_ESS=tangent_ess,metric_tangent_square_maximum_fraction=jnp.max(mass)/jnp.maximum(jnp.sum(mass),1e-300))
    # Only small dual systems and streamed Jacobian products are repeated.
    # The full Jacobian, H force, Gram matrix and force projections are built ONCE.
    direction, info = at_damping(jnp.asarray(grid[0]))
    eligible = lambda x: x['numerically_valid'] & (x['metric_tangent_square_ESS'] >= minimum_tangent_ess_fraction*n)
    found = eligible(info)
    trace = jnp.zeros((len(grid), 4)).at[0].set(jnp.array([grid[0], info['metric_tangent_square_ESS'], info['predicted_energy_descent'], info['predicted_variance_descent']]))
    def choose(i, carry):
        best, note, accepted, history = carry
        trial, meta = at_damping(jnp.asarray(grid)[i])
        passes = eligible(meta)
        replace = ~accepted & (passes | (meta['metric_tangent_square_ESS'] > note['metric_tangent_square_ESS']))
        best = jnp.where(replace, trial, best)
        note = jax.tree.map(lambda a, b: jnp.where(replace, a, b), meta, note)
        history = history.at[i].set(jnp.array([jnp.asarray(grid)[i], meta['metric_tangent_square_ESS'], meta['predicted_energy_descent'], meta['predicted_variance_descent']]))
        return best, note, accepted | passes, history
    if len(grid) > 1:
        direction, info, found, trace = jax.lax.fori_loop(1, len(grid), choose, (direction, info, found, trace))
    info = dict(info, numerically_valid=info['numerically_valid'] & found,
                adaptive_support_passed=found, damping_design_trace=trace,
                damping_selection_uses_fit_only=True, Gram_constructions=1,
                minimum_tangent_ess_fraction=minimum_tangent_ess_fraction)
    return direction, info


def residual_augmented_metric_rows(jacobian: jax.Array, h_jacobian: jax.Array,
                                   psi: jax.Array, hpsi: jax.Array,
                                   density: jax.Array, *, weight: float) -> tuple:
    """Add a positive, ray-invariant frozen-energy residual metric.

    The extra rows represent (H-E)delta_psi-r*<psi|delta_psi>/Z, scaled by
    the current residual width. This is a Gauss--Newton surrogate, not the
    exact energy Hessian. Every supplied derivative must include the same
    physical source, projector and full H action as the amplitudes. It changes
    only the update geometry; physical FS calibration and acceptance remain
    separate. Quantum labels are entirely owned by the caller.
    """
    if not isfinite(weight) or weight < 0:
        raise ValueError('finite nonnegative residual metric weight required')
    j, hj, p, h, q = map(jnp.asarray, (jacobian,h_jacobian,psi,hpsi,density))
    if (j.ndim!=2 or hj.shape!=j.shape or p.shape!=(len(j),)
            or h.shape!=p.shape or q.shape!=p.shape):
        raise ValueError('matching full amplitude/H derivative rows and products required')
    z=jnp.sum(abs(p)**2/q)
    energy=jnp.sum(jnp.real(jnp.conj(p)*h)/q)/z
    r=h-energy*p
    width2=jnp.sum(abs(r)**2/q)/z
    alpha=(jnp.conj(p)/q)@j/z
    b=hj-energy*j-r[:,None]*alpha[None,:]
    b=b*jnp.sqrt(weight/jnp.maximum(width2,1e-24))
    return (jnp.concatenate([j,b]),jnp.concatenate([p,jnp.zeros_like(p)]),
            jnp.concatenate([q,q]),
            dict(residual_metric_weight=weight,residual_metric_width_squared=width2,
                 frozen_energy_total=energy,not_exact_energy_Hessian=True))


def make_variance_balanced_update(template: Any, *, prepare: Callable,
                                  prepared_amplitude: Callable,
                                  prepared_products: Callable,
                                  damping: float=.2, tangent_fs: float=.02,
                                  action_chunk: int=8,
                                  support_filter: bool=False,
                                  energy_force: str='bilateral',
                                  residual_metric_weight: float=0.,
                                  damping_candidates: tuple[float, ...] | None=None,
                                  minimum_tangent_ess_fraction: float=0.) -> Callable:
    """Full-native proposal with streamed full-H reverse products and SR.

    prepared_products(bank,rows,source,reference) returns [N,2]. All fixed
    sources/references and Q stay frozen; the preparation VJP is retained.
    The caller supplies already measured parent products for replay/cotangents.
    """
    if tangent_fs<=0 or type(action_chunk) is not int or action_chunk<1:
        raise ValueError('positive FS radius and integer streaming chunk required')
    if energy_force not in ('bilateral','hermitian_population'):
        raise ValueError('bilateral or Hermitian-population energy-force estimator required')
    if not isfinite(residual_metric_weight) or residual_metric_weight<0:
        raise ValueError('finite nonnegative residual metric weight required')
    if residual_metric_weight and (damping_candidates is not None or minimum_tangent_ess_fraction):
        raise ValueError('adaptive tangent coverage currently requires the physical Fisher metric')
    _,unravel=ravel_pytree(template)
    @jax.jit
    def update(params,moments,rows,parent_products,density,source,reference,learning_rate):
        theta,_=ravel_pytree(params);n=len(rows)
        if n%action_chunk:raise ValueError('records must be divisible by action_chunk')
        bank,pull_native=jax.vjp(lambda x:prepare(unravel(x)),theta)
        e_cot,v_cot,point=product_cotangents(parent_products,density)
        zeros=jax.tree.map(jnp.zeros_like,bank)
        xs=(rows.reshape((-1,action_chunk,rows.shape[-1])),
            source.reshape((-1,action_chunk,2)),reference.reshape((-1,action_chunk,2)),
            e_cot.reshape((-1,action_chunk,2)),v_cot.reshape((-1,action_chunk,2)),
            parent_products.reshape((-1,action_chunk,2)))
        def scan(carry,x):
            ge,gv,error=carry;r,s,ref,ce,cv,expected=x
            value,pull=jax.vjp(lambda b:prepared_products(b,r,s,ref),bank)
            if energy_force=='bilateral':ge=jax.tree.map(jnp.add,ge,pull(ce)[0])
            gv=jax.tree.map(jnp.add,gv,pull(cv)[0])
            error=jnp.maximum(error,jnp.max(abs(value-expected))/jnp.maximum(jnp.max(abs(expected)),1e-300))
            return (ge,gv,error),None
        (ge,gv,error),_=jax.lax.scan(scan,(zeros,zeros,jnp.asarray(0.)),xs)
        forces=jnp.stack([pull_native(ge)[0],pull_native(gv)[0]])
        def row_value(row):
            value,pull=jax.vjp(lambda b:prepared_amplitude(b,row),bank)
            return value,pull_native(pull(jnp.asarray(1.,jnp.complex128))[0])[0]+1j*pull_native(pull(jnp.asarray(-1j,jnp.complex128))[0])[0]
        if residual_metric_weight:
            def product_row(x):
                row,s,ref=x
                value,pull=jax.vjp(lambda b:prepared_products(b,row[None],s[None],ref[None])[0],bank)
                def derivative(cot):return pull_native(pull(cot)[0])[0]
                jp=derivative(jnp.array([1.,0.],jnp.complex128))+1j*derivative(jnp.array([-1j,0.],jnp.complex128))
                jh=derivative(jnp.array([0.,1.],jnp.complex128))+1j*derivative(jnp.array([0.,-1j],jnp.complex128))
                return value[0],jp,jh
            psi,jac,hjac=jax.lax.map(product_row,(rows,source,reference),batch_size=1)
        else:
            psi,jac=jax.lax.map(row_value,rows,batch_size=1)
        if energy_force=='hermitian_population':
            # This unbiased population form uses H Hermiticity, rather than
            # differentiating a noisy finite-sample non-Hermitian quotient.
            # Fresh confirmation still checks BOTH coherent derivatives.
            mass=jnp.sum(abs(psi)**2/density)
            e_complex=jnp.sum(jnp.conj(psi)*parent_products[:,1]/density)/mass
            cot=2*(parent_products[:,1]-e_complex*psi)/density/mass
            size=jac.shape[1];width=min(8192,size);count=(size+width-1)//width
            def force_block(i,out):
                start=jnp.minimum(i*width,size-width)
                a=jax.lax.dynamic_slice(jac,(0,start),(n,width))
                g=jnp.real(jnp.conj(a).T@cot)
                g=jnp.where(start+jnp.arange(width)>=i*width,g,0.)
                previous=jax.lax.dynamic_slice(out,(start,),(width,))
                return jax.lax.dynamic_update_slice(out,previous+g,(start,))
            population=jax.lax.fori_loop(0,count,force_block,jnp.zeros(size))
            forces=forces.at[0].set(population)
        metric_j,metric_p,metric_q=jac,psi,density
        if residual_metric_weight:
            metric_j,metric_p,metric_q,metric_note=residual_augmented_metric_rows(
                jac,hjac,psi,parent_products[:,1],density,weight=residual_metric_weight)
        direction,info=solve_balanced_forces(metric_j,metric_p,metric_q,forces,damping=damping,
                                           support_filter=support_filter,damping_candidates=damping_candidates,
                                           minimum_tangent_ess_fraction=minimum_tangent_ess_fraction)
        _,d_bank=jax.jvp(prepare,(params,),(unravel(direction),))
        def tangent_part(x):
            r,s,ref=x
            return jax.jvp(lambda b:prepared_products(b,r,s,ref),(bank,),(d_bank,))[1]
        tangent=jax.lax.map(tangent_part,xs[:3]).reshape((n,2))
        z=point['norm'];ray=jnp.sum(jnp.conj(psi)*tangent[:,0]/density)/z
        t=tangent[:,0]-ray*psi;ht=tangent[:,1]-ray*parent_products[:,1]
        tmass=abs(t)**2/density/z
        contributions=jnp.abs(jnp.real(jnp.conj(t)*parent_products[:,1]+jnp.conj(psi)*ht)/density/z)
        tm=tmass.sum();fm=contributions.sum()
        # The trust radius is ALWAYS the physical amplitude FS norm, even when
        # the solve also penalizes residual motion. The augmented norm is not FS.
        norm=jnp.sqrt(tmass.sum()) if residual_metric_weight else info['raw_fisher_norm']
        scale=jnp.minimum(learning_rate,tangent_fs/jnp.maximum(norm,1e-300))
        proposed=theta-scale*direction
        amp_error=jnp.linalg.norm(psi-parent_products[:,0])/jnp.maximum(jnp.linalg.norm(psi),1e-300)
        valid=info['numerically_valid'] & (error<2e-10) & (amp_error<2e-10) & jnp.all(jnp.isfinite(proposed))
        next_m=(jnp.where(valid,moments[0]+1,moments[0]),moments[1],moments[2])
        return unravel(jnp.where(valid,proposed,theta)),next_m,dict(info,accepted=valid,
            relative_SR_residual=info['relative_linear_residual'],sampled_energy_total=point['energy'],
            sampled_variance_total=point['variance'],sampled_tangent_FS=scale*norm,
            applied_energy_directional_derivative=-scale*info['predicted_energy_descent'],
            applied_variance_directional_derivative=-scale*info['predicted_variance_descent'],
            update_scale=scale,full_H_preparation_derivative_retained=True,
            energy_force_uses_Hermiticity=energy_force=='hermitian_population',
            residual_metric_weight=residual_metric_weight,
            full_H_derivative_metric=residual_metric_weight>0,
            physical_FS_norm_before_scale=norm,
            sampled_population_Cauchy_bound=2*jnp.sqrt(point['variance'])*scale*norm,
            full_H_reverse_rows=n,full_H_replay_relative_error=error,amplitude_replay_error=amp_error,
            full_H_directional_rows=n,
            chosen_direction_tangent_square_ESS=tm*tm/jnp.sum(tmass*tmass),
            chosen_direction_tangent_square_maximum_fraction=jnp.max(tmass)/tm,
            chosen_direction_absolute_force_projection_ESS=fm*fm/jnp.sum(contributions**2),
            chosen_direction_absolute_force_projection_maximum_fraction=jnp.max(contributions)/fm,
            force_coverage_retains_both_coherent_derivatives=True,
            proposal_only_requires_fresh_finite_guard=True)
    return update
