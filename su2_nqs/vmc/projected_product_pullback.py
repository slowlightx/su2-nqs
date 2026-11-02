"""Explicit pullback of the existing full-H/A1 connection plan.

The plan's linear map is transposed exactly, including repeated images and
bonds. Raw-amplitude AD is performed one bounded chunk at a time. No reverse
AD is attempted through the dynamic forward loop, no H/psi ratios are used,
and no commutation or raw-model invariance is assumed.
"""
from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import jax
import jax.numpy as jnp


def make_projected_product_pullback(raw_amplitude: Callable, n_sites: int,
                                    *, chunk_size: int = 64) -> Callable:
    """Return JIT (bank,plan,wpsi,wH) -> real objective, bank gradient, counts.

    The objective is Re sum(wpsi*phi + wH*Hphi). Weights are COMPLEX
    coefficients, not conjugated cotangents. The returned gradient follows
    JAX's real-objective complex convention: dL=Re sum(gradient*delta_bank).
    Feed it directly to a generator VJP; do not conjugate it again.

    Use plans from make_projected_heisenberg_products with the SAME chunk
    size and bosonic A1 projector. All plan data is parameter independent.
    Runtime memory holds one raw contraction/VJP chunk plus a bank gradient,
    not an H-amplitude Jacobian with one column per model parameter.
    """
    if not jax.config.x64_enabled:
        raise ValueError('exact keys and pullbacks require JAX x64')
    if (isinstance(n_sites, bool) or not isinstance(n_sites, Integral)
            or not 1 <= n_sites <= 63):
        raise ValueError('exact spin keys require integer 1 <= N <= 63')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError('positive integer chunk size required')
    shifts = jnp.arange(n_sites-1, -1, -1, dtype=jnp.uint64)
    raw_batch = jax.vmap(raw_amplitude, in_axes=(None, 0))

    @jax.jit
    def pullback(bank, prepared, wpsi, wh):
        keys, inverse, count, diagonal, offdiagonal = prepared
        if wpsi.shape != diagonal.shape or wh.shape != diagonal.shape:
            raise ValueError('one complex weight per physical center required')
        group_order = inverse.shape[-1]
        connection_weights = jnp.concatenate(
            ((wpsi+diagonal*wh)[:, None], wh[:, None]*offdiagonal), axis=1)
        image_weights = jnp.broadcast_to(connection_weights[:, :, None]/group_order, inverse.shape)
        weights = jnp.zeros(keys.shape, dtype=jnp.complex128).at[inverse.reshape(-1)].add(image_weights.reshape(-1))
        gradient = jax.tree.map(jnp.zeros_like, bank)
        chunks = (count+chunk_size-1)//chunk_size

        def step(index, carry):
            value, total = carry
            offset = index*chunk_size
            locations = offset+jnp.arange(chunk_size)
            selected = jax.lax.dynamic_slice(keys, (offset,), (chunk_size,))
            selected = jnp.where(locations < count, selected, keys[0])
            bits = ((selected[:, None] >> shifts) & 1).astype(jnp.int32)
            coefficients = jax.lax.dynamic_slice(weights, (offset,), (chunk_size,))
            coefficients = jnp.where(locations < count, coefficients, 0)
            def objective(b):
                return jnp.real(jnp.sum(coefficients*raw_batch(b, bits)))
            part, derivative = jax.value_and_grad(objective)(bank)
            return value+part, jax.tree.map(lambda a, b: a+b, total, derivative)

        value, gradient = jax.lax.fori_loop(0, chunks, step, (jnp.array(0., jnp.float64), gradient))
        return value, gradient, count, chunks*chunk_size
    return pullback


__all__ = ['make_projected_product_pullback']
