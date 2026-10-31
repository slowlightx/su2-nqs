"""On-demand projected factored-ViT amplitudes and sampled full-H calibration.

No Hilbert enumeration or ED is used. Quotienting exact patch translations
reduces, but does not remove, the Reynolds sum. This is fixed-Sz, not SU(2).
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, fixed_sz_spatial_vit_log_amplitudes
from ..vmc.physical_exchange import spin_exchange_proposals


def patch_translation_cosets(
    group: Sequence[Sequence[int]] | np.ndarray,
    config: FixedSzSpatialViTConfig,
) -> np.ndarray:
    """Represent right cosets by model-invariant patch translations.

    Physical images are indexed as ``bits[permutation]``. Thus postcomposing
    a permutation with a patch translation leaves its raw amplitude unchanged.
    """
    group = np.asarray(group, dtype=np.int32)
    grid = np.arange(config.n_sites).reshape(config.rows, config.columns)
    translations = [np.roll(grid, (dy, dx), (0, 1)).ravel()
                    for dy in range(0, config.rows, config.patch_rows)
                    for dx in range(0, config.columns, config.patch_columns)]
    keys = [min(tuple(g[k]) for k in translations) for g in group]
    unique, counts = np.unique(np.asarray(keys), axis=0, return_counts=True)
    if not np.all(counts == len(translations)):
        raise ValueError("group must contain the complete patch-translation subgroup")
    return unique


def projected_log_function(
    config: FixedSzSpatialViTConfig,
    group: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[Callable, int]:
    """Return a stable exact-A1 log amplitude using a quotient Reynolds sum."""
    images = jnp.asarray(patch_translation_cosets(group, config))

    def log_amplitude(params, bits):
        logs = fixed_sz_spatial_vit_log_amplitudes(params, bits[images], config)
        shift = jax.lax.stop_gradient(jnp.max(jnp.real(logs)))
        return shift + jnp.log(jnp.mean(jnp.exp(logs-shift)))

    return log_amplitude, len(images)


def make_log_exchange_sampler(log_amplitude: Callable, proposals: int) -> Callable:
    """Fixed-Sz global exchanges, with log-domain detailed-balance acceptance.

    A 'proposal' is one two-site proposal per chain, not an N-site sweep.
    All current amplitudes are refreshed at each call after parameter updates.
    """
    if proposals < 1:
        raise ValueError("proposals must be positive")
    batched = jax.vmap(log_amplitude, in_axes=(None, 0))

    @jax.jit
    def sample(params, key, states):
        logs = batched(params, states)

        def step(_, carry):
            rng, states, logs, accepted, moved = carry
            rng, propose_key, accept_key = jax.random.split(rng, 3)
            candidate = spin_exchange_proposals(propose_key, states)
            candidate_logs = batched(params, candidate)
            log_ratio = 2*(jnp.real(candidate_logs)-jnp.real(logs))
            accept = jnp.log(jax.random.uniform(accept_key, (len(states),))) < jnp.minimum(0., log_ratio)
            changed = jnp.any(candidate != states, axis=1)
            return (rng, jnp.where(accept[:, None], candidate, states),
                    jnp.where(accept, candidate_logs, logs), accepted+jnp.sum(accept),
                    moved+jnp.sum(accept & changed))

        rng, final, logs, accepted, moved = jax.lax.fori_loop(
            0, proposals, step, (key, states, logs, 0, 0))
        return final, rng, accepted/(proposals*len(states)), moved/(proposals*len(states))

    return sample


def make_full_heisenberg_ratios(
    log_amplitude: Callable, bonds: Sequence[tuple[int, int, float]],
) -> Callable:
    """Batched JAX full-H ratios; bounded connector memory, no host columns."""
    edges = jnp.asarray([(a, b) for a, b, _ in bonds], dtype=jnp.int32)
    couplings = jnp.asarray([v for _, _, v in bonds])

    def one(params, bits):
        center = log_amplitude(params, bits)

        def contribution(row):
            edge, coupling = row
            a, b = edge
            opposite = bits[a] != bits[b]
            swapped = bits.at[a].set(bits[b]).at[b].set(bits[a])
            ratio = jnp.exp(log_amplitude(params, swapped)-center)
            return coupling*jnp.where(opposite, -.25+.5*ratio, .25)

        return jnp.sum(jax.lax.map(contribution, (edges, couplings), batch_size=4))

    return jax.jit(jax.vmap(one, in_axes=(None, 0)))
