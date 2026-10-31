"""Full-support frame-mixture proposals for a *separately specified* Born target.

Frames describe product proposal densities, never the learned wavefunction.
Magnetic bits use site zero as MSB and every four-site factor has doubled
projection m=0. The uniform fixed-m component covers configurations on which
all product frames vanish. Only the full mixture density enters Hastings.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from .physical_exchange import (
    canonical_exchange_edges, initialize_spin_exchange_chains,
    spin_bond_exchange_proposals, spin_exchange_proposals,
)


def frame_proposal(
    frames: Sequence, weights: Sequence[float], local_probability: Sequence[float],
    *, uniform_fraction: float,
) -> tuple[Callable, Callable]:
    """Return ``log_density(bits)`` and ``draw(key, count)`` for normalized Q.

    ``Q = (1-epsilon) sum_f w_f prod_c p_c + epsilon / binomial(N,N/2)``.
    ``count`` is static when jitted. Ordered sites within each cell are kept.
    """
    f = np.asarray(frames)
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(local_probability, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != 4 or min(f.shape) < 1 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError('frames must be integer (frames,cells,4) arrays')
    n = f.shape[1] * 4
    if any(not np.array_equal(np.sort(row.ravel()), np.arange(n)) for row in f):
        raise ValueError('every frame must partition the sites')
    if w.shape != (len(f),) or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-13, rtol=0):
        raise ValueError('frame weights must be normalized and nonnegative')
    bits = (np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1
    if (p.shape != (16,) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1, atol=1e-13, rtol=0)
            or np.any(p[bits.sum(axis=1) != 2] != 0)):
        raise ValueError('local probabilities must be normalized with exact m=0 support')
    if not 0 < uniform_fraction <= 1:
        raise ValueError('a positive full-support uniform fraction is required')
    frames_j, weights_j, probabilities = jnp.asarray(f), jnp.asarray(w), jnp.asarray(p)
    positions = jnp.asarray([8, 4, 2, 1])
    floor = uniform_fraction / comb(n, n//2)

    def log_density(states: jax.Array) -> jax.Array:
        x = jnp.asarray(states, dtype=jnp.int32)
        indices = jnp.sum(x[:, frames_j] * positions, axis=-1)
        density = (1-uniform_fraction) * (jnp.prod(probabilities[indices], axis=-1) @ weights_j) + floor
        valid = jnp.all((x == 0) | (x == 1), axis=1) & (x.sum(axis=1) == n//2)
        return jnp.where(valid, jnp.log(density), -jnp.inf)

    def draw(key: jax.Array, count: int) -> jax.Array:
        kf, kl, ku, km = jax.random.split(key, 4)
        component = jax.random.categorical(kf, jnp.log(weights_j), shape=(count,))
        choices = jax.random.categorical(kl, jnp.log(probabilities), shape=(count, f.shape[1]))
        local_bits = ((choices[..., None] >> jnp.arange(3, -1, -1)) & 1).reshape(count, n).astype(jnp.int32)
        indices = frames_j[component].reshape(count, n)
        product = jnp.zeros((count, n), dtype=jnp.int32).at[jnp.arange(count)[:, None], indices].set(local_bits)
        uniform = initialize_spin_exchange_chains(ku, n_chains=count, n_sites=n, n_up=n//2)
        use_uniform = jax.random.bernoulli(km, uniform_fraction, (count,))
        return jnp.where(use_uniform[:, None], uniform, product)

    return log_density, draw


def hastings_log_acceptance(
    old_log_weight: jax.Array, new_log_weight: jax.Array,
    old_log_q: jax.Array, new_log_q: jax.Array,
) -> jax.Array:
    """Log acceptance for an independence proposal; reject zero-to-zero moves."""
    ratio = new_log_weight - old_log_weight + old_log_q - new_log_q
    return jnp.where(jnp.isnan(ratio), -jnp.inf, jnp.minimum(0., ratio))


def make_frame_metropolis(
    log_amplitude: Callable, *, mode: str, proposals: int,
    edges: Sequence[Sequence[int]], n_sites: int,
    log_proposal: Callable, draw_proposal: Callable, frame_probability: float = .5,
) -> Callable:
    """Make a batched fixed-target sampler with NN/global/frame-mixture modes.

    ``frame_mix`` chooses independently per chain between global two-site
    exchange and independence MH. Mixture selection is state independent;
    each component satisfies detailed balance. Returns states, RNG, and
    (kernel, attempted/accepted/accepted-nonnull) counts; kernels NN/global/frame.
    No Hamiltonian or gradient is evaluated here.
    """
    if mode not in ('nearest', 'global', 'frame_mix') or proposals < 1:
        raise ValueError('invalid sampler mode or proposal count')
    if not 0 < frame_probability <= 1:
        raise ValueError('frame_probability must lie in (0,1]')
    checked = canonical_exchange_edges(edges, n_sites=n_sites)
    batched_log = jax.vmap(log_amplitude, in_axes=(None, 0))

    @jax.jit
    def sample(params, key, states):
        current = 2*batched_log(params, states).real

        def step(_, carry):
            x, log_weight, rng, counts = carry
            rng, kp, kf, km, ka = jax.random.split(rng, 5)
            if mode == 'nearest':
                candidate = spin_bond_exchange_proposals(kp, x, checked)
                kernel = jnp.zeros(x.shape[0], dtype=jnp.int32)
                correction = jnp.zeros(x.shape[0])
            else:
                candidate = spin_exchange_proposals(kp, x)
                kernel = jnp.ones(x.shape[0], dtype=jnp.int32)
                correction = jnp.zeros(x.shape[0])
                if mode == 'frame_mix':
                    use_frame = jax.random.bernoulli(km, frame_probability, (x.shape[0],))
                    candidate = jnp.where(use_frame[:, None], draw_proposal(kf, x.shape[0]), candidate)
                    correction = jnp.where(use_frame, log_proposal(x)-log_proposal(candidate), 0.)
                    kernel = jnp.where(use_frame, 2, 1)
            proposed = 2*batched_log(params, candidate).real
            log_accept = hastings_log_acceptance(log_weight, proposed, correction, jnp.zeros_like(correction))
            accepted = jnp.log(jax.random.uniform(ka, log_weight.shape)) < log_accept
            moved = accepted & jnp.any(candidate != x, axis=1)
            counts = counts.at[kernel, 0].add(1).at[kernel, 1].add(accepted).at[kernel, 2].add(moved)
            return jnp.where(accepted[:, None], candidate, x), jnp.where(accepted, proposed, log_weight), rng, counts

        x, _, rng, counts = jax.lax.fori_loop(0, proposals, step,
            (states, current, key, jnp.zeros((3, 3), dtype=jnp.int32)))
        return x, rng, counts

    return sample
