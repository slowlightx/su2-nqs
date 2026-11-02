"""Node-safe, division-free full H products for centered additive carriers.

Reuse the exact device-only connection plan. A residual need not be nonzero at
a carrier sample: Hphi must be contracted directly, NEVER reconstructed as
(Hphi/phi)*phi. Only divide AFTER combining the carrier and residual products.
This forward-only work path is not differentiated; VMC score AD is separate.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from ..models.spatial_projector import PhysicalSpatialProjector
from .projected_spin_work import make_deduplicated_a1_heisenberg


def make_projected_heisenberg_products(
    raw_amplitude: Callable, projector: PhysicalSpatialProjector,
    bonds: Sequence[tuple[int, int, float]], *, chunk_size: int = 64,
) -> tuple[Callable, Callable, Callable]:
    """Return plan(states), evaluate(bank,plan), action(bank,states).

    Evaluate returns (Hpsi,psi,unique_count,padded_evaluation_count), including
    at EXACT psi=0. Complete A1 image multiplicities and every Hamiltonian
    bond are retained, with repeated bonds combined by the existing planner.
    As with that planner, exact uint64 encoding currently requires N<=63 and
    JAX x64. Runtime performs no host or symbolic work. Raw amplitudes may be
    scalar-valued invariant tensor contractions or other physical ansatze.
    """
    plan, _, _ = make_deduplicated_a1_heisenberg(
        raw_amplitude, projector, bonds, chunk_size=chunk_size)
    shifts = jnp.arange(projector.n_sites-1, -1, -1, dtype=jnp.uint64)
    evaluate_raw = jax.vmap(raw_amplitude, in_axes=(None, 0))

    def evaluate(bank, prepared):
        keys, inverse, count, diagonal, offdiagonal = prepared
        chunks = (count+chunk_size-1)//chunk_size
        values = jnp.zeros(keys.shape, dtype=jnp.complex128)
        def step(index, output):
            locations = index*chunk_size+jnp.arange(chunk_size)
            selected = jax.lax.dynamic_slice(keys, (index*chunk_size,), (chunk_size,))
            selected = jnp.where(locations < count, selected, keys[0])
            bits = ((selected[:, None] >> shifts) & 1).astype(jnp.int32)
            value = evaluate_raw(bank, bits)
            return jax.lax.dynamic_update_slice(output,
                jnp.where(locations < count, value, 0), (index*chunk_size,))
        values = jax.lax.fori_loop(0, chunks, step, values)
        projected = jnp.mean(values[inverse], axis=-1)
        psi = projected[:, 0]
        hpsi = diagonal*psi+jnp.sum(offdiagonal*projected[:, 1:], axis=1)
        return hpsi, psi, count, chunks*chunk_size

    def action(bank, states):
        return evaluate(bank, plan(states))
    return plan, evaluate, action


__all__ = ['make_projected_heisenberg_products']
