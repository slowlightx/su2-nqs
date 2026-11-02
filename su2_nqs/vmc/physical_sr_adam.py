"""Real physical-amplitude SR-preconditioned Adam with an overlap safeguard.

Hamiltonian products are supplied by the caller. The same score-based energy
gradient is used in SR and ordinary Adam phases. No diagonal-only H shortcut,
energy clipping, nonzero node floor or magnetic-index parameters are added.
"""
from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .adam import _manual_adam_update
from .dual_sampled_sr import sampled_overlap_diagnostics
from .parameter_block_sr import solve_parameter_block_sr


def make_physical_sr_adam(amplitude: Callable,template: object,*,damping: float=.01,
                         score_chunk: int=1,maximum_fs_sine: float=.05,
                         minimum_ess_fraction: float=.95,halvings: int=8,
                         rematerialize_adam: bool=True,sr_use_adam: bool=True,
                         sr_tangent_fs_radius: float | None=None,
                         sr_solver: str='woodbury_gradient',
                         parameter_chunk_size: int | None=None,
                         prepare: Callable | None=None,
                         prepared_amplitude: Callable | None=None) -> Callable:
    """Build (params,moments,samples,E,lr,warm)->update; warm is a static bool.

    Real parameters and real amplitudes are required. Exact nodes are excluded
    from the actual Born sample, not repaired. Gauge directions are removed
    by score centering. Adam moments commit only for an accepted update.

    Optional ``prepare(params)`` and ``prepared_amplitude(bank,bits)`` must be
    supplied together and represent exactly ``amplitude(params,bits)``. The
    caller must independently validate that identity. This execution-only
    path prepares a shared differentiable bank once for the entire score
    batch and pulls every bank score back to ALL original parameter columns.
    It retains the real metric ``F=Cov(O,O)`` and force ``g=2 Cov(O,E)``;
    neither the variational family nor the ridge convention changes. Omitting
    both callbacks keeps the literal legacy computation unchanged.

    ``sr_solver='dual_force'`` uses the algebraically identical dual-force
    ridge solution without subtracting two nearly equal parameter vectors.
    The default retains the original arithmetic for existing registrations.
    """
    if not jax.config.x64_enabled: raise RuntimeError('physical SR needs float64')
    if not jax.tree.leaves(template) or any(v.dtype != jnp.float64 for v in jax.tree.leaves(template)):
        raise ValueError('physical SR-Adam requires real float64 parameter leaves')
    if (not isfinite(damping) or damping <= 0 or score_chunk < 1 or halvings < 1
            or not 0 < maximum_fs_sine < 1 or not 0 < minimum_ess_fraction <= 1):
        raise ValueError('invalid ridge/chunk/physical trust settings')
    if not isinstance(sr_use_adam,bool):raise ValueError('SR Adam selection must be boolean')
    if sr_solver not in ('woodbury_gradient', 'dual_force'):
        raise ValueError('unknown SR ridge solver')
    if parameter_chunk_size is not None and (isinstance(parameter_chunk_size,bool)
            or not isinstance(parameter_chunk_size,int) or parameter_chunk_size < 1
            or sr_solver != 'dual_force'):
        raise ValueError('parameter chunk requires positive integer size and dual_force solver')
    if sr_tangent_fs_radius is not None and (not isfinite(sr_tangent_fs_radius) or sr_tangent_fs_radius<=0):
        raise ValueError('SR tangent FS radius must be positive and finite')
    if (prepare is None)!=(prepared_amplitude is None):
        raise ValueError('prepare and prepared_amplitude must be supplied together')
    if prepare is not None and (not callable(prepare) or not callable(prepared_amplitude)):
        raise ValueError('prepared execution callbacks must be callable')
    _,unravel=ravel_pytree(template)
    def log_abs(theta,bits): return jnp.log(jnp.abs(amplitude(unravel(theta),bits)))
    score=jax.grad(log_abs)
    def log_prepared(bank,bits):return jnp.log(jnp.abs(prepared_amplitude(bank,bits)))
    def update(params,moments,samples,energies,learning_rate,*,warm: bool):
        theta,_=ravel_pytree(params)
        e=jax.lax.stop_gradient(energies-energies.mean())
        if warm:
            if prepare is None:
                scores=jax.lax.map(lambda row:score(theta,row),samples,batch_size=score_chunk)
            else:
                prepared,pullback=jax.vjp(lambda values:prepare(unravel(values)),theta)
                prepared_score=jax.grad(log_prepared)
                scores=jax.lax.map(lambda row:pullback(prepared_score(prepared,row))[0],
                    samples,batch_size=score_chunk)
            if parameter_chunk_size is not None:
                direction,gradient,blocked=solve_parameter_block_sr(scores[:,None,:],
                    2*e/jnp.sqrt(len(samples)), damping=damping,
                    score_scale=1/len(samples)**.5,parameter_chunk_size=parameter_chunk_size)
                relative=blocked['relative_linear_residual']
                natural_norm=blocked['raw_fisher_norm']
            else:
                a=(scores-scores.mean(axis=0))/jnp.sqrt(len(samples))
                gradient=2*a.T@e/jnp.sqrt(len(samples))
                metric=damping*jnp.eye(len(samples))+a@a.T
            if parameter_chunk_size is not None:
                pass
            elif sr_solver == 'dual_force':
                # g=A.T b.  (A.T A+lambda I)^-1 A.T b equals
                # A.T (A A.T+lambda I)^-1 b, with no g-A.T@dual
                # cancellation when the ridge is small relative to A A.T.
                force=2*e/jnp.sqrt(len(samples))
                dual=jax.scipy.linalg.solve(metric,force,assume_a='pos')
                direction=a.T@dual
            else:
                dual=jax.scipy.linalg.solve(metric,a@gradient,assume_a='pos')
                direction=(gradient-a.T@dual)/damping
            if parameter_chunk_size is None:
                residual=a.T@(a@direction)+damping*direction-gradient
                relative=jnp.linalg.norm(residual)/jnp.maximum(jnp.linalg.norm(gradient),1e-25)
                natural_norm=jnp.linalg.norm(a@direction)
        else:
            def surrogate(values):
                # Reverse AD of the outer scan otherwise stacks all CT/orbit
                # contraction residuals over4096 samples (hundreds of GiB).
                # Retain only row inputs; recompute the IDENTICAL scalar log
                # within each reverse step. This changes memory, not the loss,
                # sample set, physical state, clipping or Adam moments.
                if prepare is None:
                    evaluate=jax.checkpoint(log_abs) if rematerialize_adam else log_abs
                    logs=jax.lax.map(lambda row:evaluate(values,row),samples,batch_size=score_chunk)
                else:
                    bank=prepare(unravel(values))
                    evaluate=jax.checkpoint(log_prepared) if rematerialize_adam else log_prepared
                    logs=jax.lax.map(lambda row:evaluate(bank,row),samples,batch_size=score_chunk)
                return 2*jnp.mean(e*logs)
            gradient=jax.grad(surrogate)(theta)
            direction=gradient*jnp.minimum(1.,.001/jnp.maximum(jnp.linalg.norm(gradient),1e-25))
            relative=jnp.asarray(0.)
        if warm and not sr_use_adam:
            # Plain natural-gradient direction; a second coordinate-wise Adam
            # normalization need not preserve descent. Retain the original
            # moment arrays for exact checkpoint continuity, but increment the
            # update counter. No Adam moments enter this direction.
            scale=learning_rate
            if sr_tangent_fs_radius is not None:
                scale*=jnp.minimum(1.,sr_tangent_fs_radius/jnp.maximum(
                    learning_rate*natural_norm,1e-25))
            proposed=theta-scale*direction
            next_moments=(moments[0]+1,moments[1],moments[2])
        else:
            raw,next_moments=_manual_adam_update(params,unravel(direction),moments,learning_rate)
            proposed,_=ravel_pytree(raw)
        if prepare is None:
            old=jax.lax.map(lambda row:amplitude(params,row),samples,batch_size=score_chunk)
        else:
            if not warm:prepared=prepare(params)
            old=jax.lax.map(lambda row:prepared_amplitude(prepared,row),samples,batch_size=score_chunk)
        def trial(k,carry):
            selected,accepted,fs,ess,scale=carry
            def evaluate(_):
                alpha=.5**k; candidate=theta+alpha*(proposed-theta)
                if prepare is None:
                    new=jax.lax.map(lambda row:amplitude(unravel(candidate),row),samples,batch_size=score_chunk)
                else:
                    bank=prepare(unravel(candidate))
                    new=jax.lax.map(lambda row:prepared_amplitude(bank,row),samples,batch_size=score_chunk)
                # Complex logarithms retain sign reversals in the overlap.
                diagnostics=sampled_overlap_diagnostics(jnp.log((new/old).astype(jnp.complex128)))
                ok=(jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(gradient))
                    & (relative < 1e-7) & (diagnostics['sampled_fs_sine'] <= maximum_fs_sine)
                    & (diagnostics['reweighting_ess_fraction'] >= minimum_ess_fraction)
                    & jnp.all(new != 0))
                return (jnp.where(ok,candidate,selected),ok,
                        jnp.where(ok,diagnostics['sampled_fs_sine'],0.),
                        jnp.where(ok,diagnostics['reweighting_ess_fraction'],1.),jnp.where(ok,alpha,0.))
            return jax.lax.cond(accepted,lambda _:carry,evaluate,operand=None)
        selected,accepted,fs,ess,scale=jax.lax.fori_loop(0,halvings,trial,
            (theta,jnp.asarray(False),jnp.asarray(0.),jnp.asarray(1.),jnp.asarray(0.)))
        committed=jax.tree.map(lambda new,old:jnp.where(accepted,new,old),next_moments,moments)
        return unravel(selected),committed,dict(accepted=accepted,FS_sine=fs,ESS_fraction=ess,
            update_scale=scale,relative_SR_residual=relative,energy_gradient_norm=jnp.linalg.norm(gradient),
            raw_energy_directional_derivative=jnp.vdot(gradient,proposed-theta).real,
            applied_energy_directional_derivative=jnp.vdot(gradient,selected-theta).real)
    return jax.jit(update,static_argnames=('warm',))
