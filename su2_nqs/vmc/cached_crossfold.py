"""One-step, whole-chain directional diagnostics on already cached H actions.

The empirical full-H ratio differentiates BOTH amplitudes. A fresh Adam
direction is computed using fit rows only. Its parameter is then expressed
in units of fit-set projective tangent speed before a fixed-work FS search.
Held-out rows never choose the direction, radius, or scalar step. They are
diagnostic only when the frozen parent or protocol has already seen them.
No Hamiltonian, sampler, or physical-basis construction occurs here.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .matched_state_step import matched_state_step, _stable_distance


@dataclass(frozen=True)
class CrossfoldConfig:
    """Fixed one-step protocol; ``normalized_max_alpha`` has tangent-FS units."""

    n_sites: int = 36
    target_fs: float = .001
    normalized_max_alpha: float = .002
    bisection_steps: int = 16
    relative_tolerance: float = .01
    adam_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if isinstance(self.n_sites, bool) or not isinstance(self.n_sites, int) or self.n_sites < 1:
            raise ValueError('n_sites must be positive integer')
        if isinstance(self.bisection_steps, bool) or not isinstance(self.bisection_steps, int) or self.bisection_steps < 0:
            raise ValueError('bisection_steps must be nonnegative integer')
        for key in ('target_fs', 'normalized_max_alpha', 'relative_tolerance', 'adam_epsilon'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key+' must be finite and positive')
        if self.target_fs >= np.pi/2 or self.relative_tolerance >= 1:
            raise ValueError('invalid FS radius/tolerance')


def whole_chain_folds(groups: np.ndarray, count: int = 4) -> np.ndarray:
    """Assign sorted independent chain labels round-robin, never split a chain."""
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.size == 0 or count < 2:
        raise ValueError('nonempty groups and at least two folds are required')
    unique, inverse = np.unique(labels, return_inverse=True)
    if len(unique) < count or not np.isfinite(labels).all():
        raise ValueError('need finite labels and at least one chain per fold')
    return inverse % count


def directional_moments(
    psi: np.ndarray, hpsi: np.ndarray, dpsi: np.ndarray, dhpsi: np.ndarray,
    q: np.ndarray, n_sites: int,
) -> dict[str, float]:
    """Exact finite-Q energy/variance directional derivatives, node-safe.

    Hpsi and dHpsi use TOTAL-H units. The energy may be complex; its full
    value is retained in centering the nonnegative residual variance.
    """
    p, h, dp, dh = [np.asarray(x, dtype=np.complex128) for x in (psi, hpsi, dpsi, dhpsi)]
    if np.iscomplexobj(q):
        raise ValueError('proposal must be real')
    proposal = np.asarray(q, dtype=np.float64)
    if (p.ndim != 1 or not p.size or any(x.shape != p.shape for x in (h, dp, dh, proposal))
            or not all(np.isfinite(x).all() for x in (p, h, dp, dh, proposal)) or np.any(proposal <= 0)
            or n_sites < 1):
        raise ValueError('invalid finite amplitude/action/tangent/proposal vectors')
    w = 1/proposal
    z = float(np.sum(w*abs(p)**2))
    if not np.isfinite(z) or z <= 0:
        raise ValueError('zero or nonfinite physical norm')
    dz = 2*float(np.real(np.sum(w*p.conj()*dp)))
    e = np.sum(w*p.conj()*h)/z
    de = (np.sum(w*(dp.conj()*h+p.conj()*dh))-e*dz)/z
    residual = h-e*p
    dr = dh-de*p-e*dp
    variance = float(np.sum(w*abs(residual)**2)/z)
    dvariance = (2*float(np.real(np.sum(w*residual.conj()*dr)))-variance*dz)/z
    overlap = np.sum(w*p.conj()*dp)/z
    tangent_perp = dp-overlap*p
    speed = float(np.sqrt(np.sum(w*abs(tangent_perp)**2)/z))
    return dict(energy_per_site=float(e.real/n_sites), imaginary_energy_per_site=float(e.imag/n_sites),
        variance_per_site=variance/n_sites, energy_slope_per_site=float(de.real/n_sites),
        imaginary_energy_slope_per_site=float(de.imag/n_sites), variance_slope_per_site=dvariance/n_sites,
        projective_speed=speed, norm_sum=z)


class CachedCrossfoldOracle:
    """Compile once per parameterization; fold masks remain dynamic arguments."""

    def __init__(self, state_pair: Callable, q: np.ndarray, config: CrossfoldConfig):
        if np.iscomplexobj(q):
            raise ValueError('proposal must be real')
        self.q = np.asarray(q, dtype=np.float64)
        if self.q.ndim != 1 or not self.q.size or not np.isfinite(self.q).all() or np.any(self.q <= 0):
            raise ValueError('q must be a positive finite vector')
        self.config = config
        self.state = jax.jit(state_pair)
        def objective(x, mask):
            p, hp = state_pair(x)
            weights = jax.lax.stop_gradient(mask/jnp.asarray(self.q))
            return jnp.sum(weights*jnp.real(p.conj()*hp))/jnp.sum(weights*abs(p)**2)/config.n_sites
        self.gradient = jax.jit(jax.value_and_grad(objective))
        self.tangent = jax.jit(lambda x, v: jax.jvp(state_pair, (x,), (v,)))

    def trial(self, initial: np.ndarray, fit_mask: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """One independent fresh-Adam step; never update the input parent.

        Fixed 1+bisection_steps candidate calls unless the tangent is null,
        which is reported explicitly and not counted as matched work.
        """
        if np.iscomplexobj(initial):
            raise ValueError('parameters must be real')
        x = np.asarray(initial, dtype=np.float64)
        fit = np.asarray(fit_mask, dtype=bool)
        if x.ndim != 1 or not np.isfinite(x).all() or fit.shape != self.q.shape or not fit.any() or fit.all():
            raise ValueError('need finite parameters and nonempty disjoint fit/held-out sets')
        value, gradient = self.gradient(jnp.asarray(x), jnp.asarray(fit, dtype=jnp.float64))
        gradient = np.asarray(gradient)
        if not np.isfinite(float(value)) or not np.isfinite(gradient).all():
            raise FloatingPointError('nonfinite finite-Q full gradient')
        direction = -gradient/(abs(gradient)+self.config.adam_epsilon)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction /= direction_norm
        (old, hold), (tangent, htangent) = self.tangent(jnp.asarray(x), jnp.asarray(direction))
        old, hold, tangent, htangent = map(np.asarray, (old, hold, tangent, htangent))
        raw_fit = directional_moments(old[fit], hold[fit], tangent[fit], htangent[fit], self.q[fit], self.config.n_sites)
        speed = raw_fit['projective_speed']
        if speed <= 1e-14 or not np.isfinite(speed):
            raise FloatingPointError('zero/gauge fit tangent: no physical-distance comparison possible')
        direction = direction/speed
        tangent, htangent = tangent/speed, htangent/speed
        counts = dict(candidate_state_calls=0)
        def candidate(alpha):
            counts['candidate_state_calls'] += 1
            return np.asarray(self.state(jnp.asarray(x+alpha*direction))[0])[fit]
        matched = matched_state_step(candidate, old[fit], self.q[fit], self.config.target_fs,
            self.config.normalized_max_alpha, self.config.bisection_steps,
            relative_tolerance=self.config.relative_tolerance)
        endpoint = x+matched.alpha*direction
        after, hafter = map(np.asarray, self.state(jnp.asarray(endpoint)))
        reports = {}
        for name, mask in (('fit', fit), ('held_out', ~fit)):
            before = directional_moments(old[mask], hold[mask], tangent[mask], htangent[mask], self.q[mask], self.config.n_sites)
            final = directional_moments(after[mask], hafter[mask], tangent[mask]*0, htangent[mask]*0, self.q[mask], self.config.n_sites)
            reports[name] = dict(initial=before, endpoint=final,
                delta_energy_per_site=final['energy_per_site']-before['energy_per_site'],
                delta_variance_per_site=final['variance_per_site']-before['variance_per_site'],
                distance=_stable_distance(old[mask], after[mask], self.q[mask]))
        record = dict(status='complete', matched=matched.matched, physical_step=matched.as_dict(),
            raw_parameter_direction_norm=float(direction_norm), gradient_norm=float(np.linalg.norm(gradient)),
            fit_tangent_speed_before_normalization=speed, fit_gradient_dot_direction=float(gradient@direction),
            direction='fresh Adam first step; zero moments; then unit fit-projective tangent speed',
            **reports, ledger=dict(full_gradient_calls=1, state_jvp_calls=1,
                candidate_state_calls=counts['candidate_state_calls'], final_state_calls=1,
                new_samples=0, new_H_queries=0, new_ED_calls=0))
        arrays = dict(initial=x, endpoint=endpoint, gradient=gradient, direction=direction,
            psi_initial=old, hpsi_initial=hold, psi_endpoint=after, hpsi_endpoint=hafter,
            tangent=tangent, htangent=htangent, fit_mask=fit)
        return record, arrays


__all__ = ['CrossfoldConfig', 'whole_chain_folds', 'directional_moments', 'CachedCrossfoldOracle']
