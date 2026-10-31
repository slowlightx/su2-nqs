"""Small reusable helpers for bounded, resumable physical-space VMC pilots."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from .adam import _manual_adam_init, _manual_adam_update
from .physical_trust_region import bound_physical_state_update
from .projected_spin_work import make_frozen_vmc_force


def make_dynamic_force(amplitudes: Callable, n_sites: int) -> Callable:
    """JIT aggregate VMC force with dynamic, stopped local energies; no H replay."""
    def force(params, states, energies):
        return make_frozen_vmc_force(amplitudes, energies, n_sites)(params, states)
    return jax.jit(force)


def initialize_pilot_adam(theta: jax.Array) -> tuple:
    """Reuse the existing real-vector Adam moments without a new optimizer."""
    return _manual_adam_init(theta)


_adam_update = jax.jit(_manual_adam_update)


def bounded_adam_step(
    theta: jax.Array, gradient: jax.Array, optimizer: tuple, states: jax.Array,
    *, amplitudes: Callable, learning_rate: float, clip_norm: float,
    maximum_fs: float, bisections: int,
) -> tuple[jax.Array, tuple, dict]:
    """Adam direction, then empirical physical-state FS cap on training samples.

    Moments advance on every attempt, including shortened/zero steps. This is
    not an independent-sample trust guarantee and uses no candidate H calls.
    """
    if any(not np.isfinite(v) or v <= 0 for v in (learning_rate, clip_norm, maximum_fs)):
        raise ValueError('learning rate, clip norm and FS cap must be positive')
    g = np.asarray(gradient)
    if not np.isfinite(g).all(): raise FloatingPointError('nonfinite VMC force')
    norm = float(np.linalg.norm(g)); clipping = min(1., clip_norm/max(norm,1e-300))
    candidate, updated = _adam_update(theta, clipping*gradient, optimizer, learning_rate)
    bound = bound_physical_state_update(theta, theta-candidate, states,
        center_fn=amplitudes, maximum_fubini_study_distance=maximum_fs,
        maximum_bisection_iterations=bisections)
    result = theta-bound.direction
    if not np.isfinite(np.asarray(result)).all(): raise FloatingPointError('nonfinite update')
    return result, updated, dict(gradient_norm=norm, gradient_clip_scale=clipping,
        step_scale=bound.scale, distance=asdict(bound.distance), bisections=bound.bisection_iterations)


def energy_summary(local_energies: np.ndarray, n_sites: int) -> dict:
    """Describe complex local energies (time, chains); SE uses whole-chain means.

    Chain means are not guaranteed equilibrated. The variance estimate is
    mean(|E_L-mean(E_L)|^2)/N, not Var(Re E_L)/N. No iid-snapshot error bar.
    """
    values = np.asarray(local_energies, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape)<2 or not np.isfinite(values).all() or n_sites<1:
        raise ValueError('need finite (time>=2, chains>=2) energies and positive N')
    mean = np.mean(values); chains = values.mean(axis=0)
    return dict(energy_per_site=float(mean.real/n_sites),
        imaginary_energy_per_site=float(mean.imag/n_sites),
        variance_per_site=float(np.mean(abs(values-mean)**2)/n_sites),
        energy_chain_se=float(chains.real.std(ddof=1)/np.sqrt(values.shape[1])/n_sites),
        imaginary_chain_se=float(chains.imag.std(ddof=1)/np.sqrt(values.shape[1])/n_sites),
        snapshots=values.shape[0], chains=values.shape[1], samples=values.size,
        equilibrium_certified=False)


def independent_audit_comparison(
    parent: np.ndarray, terminal: np.ndarray, *, n_sites: int, seed: int, replicates: int,
) -> dict:
    """Independent-stream change; chain bootstrap stratified by two start cohorts.

    Training and audit RNG streams must be disjoint. Parent/terminal chains are
    NOT paired; their chain-mean variances add. Bootstrap uncertainty remains
    conditional on equilibration and the finite chain population.
    """
    old = np.asarray(parent); new = np.asarray(terminal)
    if old.shape!=new.shape or old.ndim!=2 or old.shape[1]%2 or replicates<2:
        raise ValueError('require equal even-chain audit shapes and replicates>=2')
    a=energy_summary(old,n_sites);b=energy_summary(new,n_sites)
    rng=np.random.default_rng(seed);half=old.shape[1]//2
    def draw(values):
        indices=np.concatenate((rng.integers(0,half,(replicates,half)),
                                half+rng.integers(0,half,(replicates,half))),axis=1)
        x=values[:,indices];mean=x.mean(axis=(0,2))
        return mean.real/n_sites,np.mean(abs(x-mean[None,:,None])**2,axis=(0,2))/n_sites
    eo,vo=draw(old);en,vn=draw(new)
    return dict(delta_energy_per_site=b['energy_per_site']-a['energy_per_site'],
        delta_energy_chain_se=float(np.hypot(a['energy_chain_se'],b['energy_chain_se'])),
        delta_variance_per_site=b['variance_per_site']-a['variance_per_site'],
        delta_energy_bootstrap_95=list(map(float,np.quantile(en-eo,[.025,.975]))),
        delta_variance_bootstrap_95=list(map(float,np.quantile(vn-vo,[.025,.975]))),
        paired=False, bootstrap_unit='whole_chain_stratified_by_initial_cohort',
        uncertainty_conditional_on_equilibration=True)
