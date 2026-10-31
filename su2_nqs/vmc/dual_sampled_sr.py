"""Bounded sample-space ridge SR for real parameters and complex wavefunctions.

Uses the same convention as ``physical_sampled_sr``: S = 2 Re Cov(O,O),
g = 2 Re Cov(O,E_local).  The identity
(A.T A + lambda I)^-1 A.T b = A.T (A A.T + lambda I)^-1 b
avoids a parameter-square matrix. Scores are explicit, chunked, and cost
O(samples * parameters) memory: this is NOT a score-matrix-free method.
The unscaled solve residual is checked independently of trust scaling.
No Hamiltonian, sampler, or model architecture is owned by this module.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


def sampled_overlap_diagnostics(log_ratios: jax.Array) -> dict[str, jax.Array]:
    """Born-sample estimates of sine FS distance and reweighting ESS fraction.

    Ratios are log(psi_new / psi_old) on samples from the old state. These
    are finite-sample diagnostics, not a globally certified trust bound.
    """
    shifted = log_ratios-jnp.max(log_ratios.real)
    ratios = jnp.exp(shifted)
    weights = abs(ratios)**2
    overlap2 = abs(jnp.mean(ratios))**2/jnp.mean(weights)
    return dict(sampled_fs_sine=jnp.sqrt(jnp.maximum(0.,1.-jnp.minimum(overlap2,1.))),
        reweighting_ess_fraction=jnp.mean(weights)**2/jnp.mean(weights**2))


def importance_overlap_diagnostics(
    first: np.ndarray, second: np.ndarray, proposal: np.ndarray, *,
    observation_weights: np.ndarray | None = None,
) -> dict:
    """Finite-proposal overlap, including zeros of either physical amplitude.

    Both states are measured on the SAME rows from a positive proposal Q.
    A common unknown normalization of Q cancels. In particular these rows
    need NOT be Born samples of ``first``; using the ordinary unweighted
    Born-ratio formula in that case would be incorrect. For an exhaustive
    reference sum, observation weights proportional to Q cancel importance
    sampling. Default equal weights describe an empirical sample pool.

    No division by either state occurs. Independent amplitude rescalings
    preserve fidelity/ESS but not the explicitly reported norm ratio. This
    is NOT an equilibrium, complete-support, or global-fidelity certificate;
    chain uncertainties and proposal node coverage belong to the caller.
    """
    a, b = np.asarray(first, complex), np.asarray(second, complex)
    q = np.asarray(proposal)
    if (a.ndim != 1 or not a.size or b.shape != a.shape or q.shape != a.shape
            or np.iscomplexobj(q) or any(not np.isfinite(v).all() for v in (a, b, q))
            or np.any(q <= 0)):
        raise ValueError('matching finite amplitudes and positive real proposal required')
    weights = np.ones(len(a)) if observation_weights is None else np.asarray(observation_weights)
    if (weights.shape != a.shape or np.iscomplexobj(weights) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError('finite nonnegative observation weights required')
    weights = np.asarray(weights, float)/np.max(weights)
    qa = np.sqrt(np.asarray(q, float))
    scaled, norm_logs, diagnostics = [], [], []
    for state in (a, b):
        scale = float(np.max(abs(state)))
        if scale == 0:
            raise ValueError('both states need nonzero measured norm')
        vector = (state/scale)*np.sqrt(weights)/qa
        rms_scale = float(np.max(abs(vector)))
        if not np.isfinite(vector).all() or rms_scale == 0:
            raise FloatingPointError('importance amplitude scaling is invalid')
        vector = vector/rms_scale
        mass = abs(vector)**2
        norm = float(mass.sum())
        norm_logs.append(2*np.log(scale)+2*np.log(rms_scale)+np.log(norm))
        scaled.append(vector/np.sqrt(norm))
        diagnostics.append(dict(effective_contributions=float(norm**2/np.sum(mass**2)),
                                maximum_norm_record_fraction=float(mass.max()/norm)))
    overlap = np.vdot(scaled[0], scaled[1])
    fidelity = float(abs(overlap)**2)
    if not np.isfinite(fidelity) or fidelity > 1+2e-12:
        raise FloatingPointError('normalized empirical overlap violates Cauchy-Schwarz')
    log_ratio = float(norm_logs[1]-norm_logs[0])
    return dict(fidelity_squared=fidelity,
        sampled_fs_sine=float(np.sqrt(max(0., 1-min(fidelity, 1.)))),
        second_over_first_norm_ratio=float(np.exp(log_ratio)) if abs(log_ratio) < 700 else None,
        log_second_over_first_norm_ratio=log_ratio, first=diagnostics[0], second=diagnostics[1],
        samples=len(a), empirical_pool_only=True, equilibrium_certified=False,
        complete_support_certified=False, global_fidelity_certified=False)


def make_dual_sampled_sr_update(
    log_amplitude: Callable,
    template_params: object,
    *,
    damping: float = 0.1,
    step_size: float = 0.03,
    fisher_trust_radius: float = 0.05,
    maximum_overlap_distance: float = 0.05,
    minimum_ess_fraction: float = 0.8,
    backtracking_steps: int = 6,
    score_chunk_size: int = 8,
    linear_solver: str = "dual",
) -> Callable:
    """Build a JIT update ``(params,samples,local_E)->(params,diagnostics)``.

    Learnable parameters must have real float64 leaves. The exact complex
    log-amplitude callback operates on a single physical configuration. The
    local-energy vector is supplied, detached, by the caller. No clipping of
    energies or scores is used. The update is a descent natural gradient;
    normalization and phase directions are removed by score centering.

    A fixed list of step halvings checks a sampled nonlinear overlap/ESS
    bound without any candidate-energy calls. If none passes, parameters
    stay unchanged; the attempted update still consumes the caller's budget.

    ``linear_solver='svd'`` solves the SAME ridge problem using the thin
    singular-value decomposition of centered scores. It avoids cancellation
    from null-space coefficients in an ill-conditioned dual Gram solve.
    Neither damping nor the independent 1e-7 residual guard is changed.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("dual sampled SR requires x64")
    if linear_solver not in {"dual", "svd"}:
        raise ValueError("linear_solver must be dual or svd")
    for name,value in (("damping",damping),("step_size",step_size),
                       ("fisher_trust_radius",fisher_trust_radius),
                       ("maximum_overlap_distance",maximum_overlap_distance)):
        if not math.isfinite(value) or value<=0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 < minimum_ess_fraction <= 1:
        raise ValueError("minimum_ess_fraction must be in (0,1]")
    for value in (backtracking_steps,score_chunk_size):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError("chunk and backtracking counts must be positive integers")
    leaves=jax.tree_util.tree_leaves(template_params)
    if not leaves or any(np.asarray(v).dtype!=np.dtype('float64') or
                         not np.isfinite(v).all() for v in leaves):
        raise ValueError("template parameters must have finite real float64 leaves")
    _,unravel=ravel_pytree(template_params)
    log_batch=jax.vmap(log_amplitude,in_axes=(None,0))

    @jax.jit
    def update(params,samples,energies):
        n=len(samples)
        if n<2 or energies.shape!=(n,):
            raise ValueError("need at least two samples and one energy per sample")
        flat,_=ravel_pytree(params)
        def score(x):
            def components(values):
                z=log_amplitude(unravel(values),x)
                return jnp.stack((jnp.real(z),jnp.imag(z)))
            return jax.jacrev(components)(flat)
        scores=jax.lax.map(score,samples,batch_size=score_chunk_size)
        centered=scores-scores.mean(axis=0,keepdims=True)
        a=jnp.sqrt(2./n)*jnp.concatenate((centered[:,0,:],centered[:,1,:]),axis=0)
        e=jax.lax.stop_gradient(energies-energies.mean())
        b=jnp.sqrt(2./n)*jnp.concatenate((e.real,e.imag))
        if linear_solver == 'svd':
            u, singular, vh = jnp.linalg.svd(a, full_matrices=False)
            direction = vh.T @ ((singular/(singular**2+damping)) * (u.T@b))
        else:
            gram=a@a.T
            coefficients=jax.scipy.linalg.solve(gram+damping*jnp.eye(2*n),b,assume_a='pos')
            direction=a.T@coefficients
        gradient=a.T@b
        residual=a.T@(a@direction)+damping*direction-gradient
        norm=jnp.linalg.norm(gradient)
        relative_residual=jnp.linalg.norm(residual)/jnp.maximum(norm,1e-30)
        raw_fisher_norm=jnp.linalg.norm(a@direction)
        scale=step_size*jnp.minimum(1.,fisher_trust_radius/jnp.maximum(step_size*raw_fisher_norm,1e-30))
        old_logs=log_batch(params,samples)
        # First passing halving, not minimum candidate energy or audit selection.
        initial=(flat,jnp.asarray(False),jnp.asarray(0.),jnp.asarray(1.),jnp.asarray(0.),jnp.asarray(0))
        def trial(k,carry):
            selected,accepted,fs,ess,used_scale,attempts=carry
            def evaluate(_):
                candidate_scale=scale*0.5**k
                candidate=flat-candidate_scale*direction
                ratios=log_batch(unravel(candidate),samples)-old_logs
                diag=sampled_overlap_diagnostics(ratios)
                passed=(jnp.all(jnp.isfinite(ratios)) & jnp.all(jnp.isfinite(candidate)) &
                    (relative_residual<=1e-7) & (diag['sampled_fs_sine']<=maximum_overlap_distance) &
                    (diag['reweighting_ess_fraction']>=minimum_ess_fraction))
                return (jnp.where(passed,candidate,selected),passed,
                    jnp.where(passed,diag['sampled_fs_sine'],0.),
                    jnp.where(passed,diag['reweighting_ess_fraction'],1.),
                    jnp.where(passed,candidate_scale,0.),attempts+1)
            return jax.lax.cond(accepted,lambda _:carry,evaluate,operand=None)
        chosen,accepted,fs,ess,used_scale,attempts=jax.lax.fori_loop(0,backtracking_steps,trial,initial)
        return unravel(chosen),dict(gradient_norm=norm,relative_linear_residual=relative_residual,
            raw_direction_norm=jnp.linalg.norm(direction),raw_fisher_norm=raw_fisher_norm,
            fisher_norm=used_scale*raw_fisher_norm,applied_scale=used_scale,
            parameter_step_norm=jnp.linalg.norm(chosen-flat),sampled_fs_sine=fs,
            reweighting_ess_fraction=ess,accepted=accepted,backtracking_evaluations=attempts,
            directional_derivative=used_scale*jnp.vdot(gradient,direction).real,
            score_real_entries=scores.size,sample_matrix_dimension=2*n)
    return update
