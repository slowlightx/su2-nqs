"""Exact sample-dual ridge SR without full centered-score copies.

The original real score array is retained. Parameter columns are centered,
scaled and contracted in fixed-size blocks, so neither a full normalized
score matrix nor a parameter-space Fisher matrix is materialized. This is
not a matrix-free score generator and does not change the sampled metric.
"""
from __future__ import annotations

from math import isfinite

import jax
import jax.numpy as jnp


def solve_parameter_block_sr(
    scores: jax.Array,
    force: jax.Array,
    *,
    damping: float,
    score_scale: float,
    parameter_chunk_size: int = 8192,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Solve ``(A.T A + damping I)d=A.T force`` with bounded column work.

    ``scores`` has shape ``[samples, components, parameters]`` and real
    entries. Each component is independently sample-centered. Rows of A
    are component-major, with scale ``score_scale``; ``force`` uses this
    same row order. One component describes real amplitudes; two describe
    real and imaginary log derivatives of a complex amplitude. The caller
    explicitly chooses the metric/force convention, including factors of 2.

    The final partial block is masked without padding the full score array.
    A true primal residual is accumulated from A*d, not inferred solely
    from the small dual solve. No damping, samples or columns are dropped.
    """
    if (scores.ndim != 3 or min(scores.shape) < 1 or scores.shape[0] < 2
            or force.shape != (scores.shape[0]*scores.shape[1],)):
        raise ValueError('scores [N>=2,C>=1,P>=1] and matching component-major force required')
    if not jnp.issubdtype(scores.dtype, jnp.floating) or not jnp.issubdtype(force.dtype, jnp.floating):
        raise ValueError('real floating scores and force required')
    if (not isfinite(damping) or damping <= 0 or not isfinite(score_scale) or score_scale <= 0
            or isinstance(parameter_chunk_size, bool) or not isinstance(parameter_chunk_size, int)
            or parameter_chunk_size < 1):
        raise ValueError('positive finite ridge/scale and integer parameter chunk required')
    count, components, parameters = scores.shape
    rows = count*components
    width = min(parameter_chunk_size, parameters)
    blocks = (parameters+width-1)//width
    dtype = scores.dtype

    def block(i):
        first = jnp.minimum(i*width, parameters-width)
        values = jax.lax.dynamic_slice(scores, (0, 0, first), (count, components, width))
        values = values-values.mean(axis=0, keepdims=True)
        a = values.transpose(1, 0, 2).reshape(rows, width)*score_scale
        valid = first+jnp.arange(width) >= i*width
        return first, jnp.where(valid[None, :], a, 0.)

    def gram_body(i, gram):
        _, a = block(i)
        return gram+a@a.T

    gram = jax.lax.fori_loop(0, blocks, gram_body, jnp.zeros((rows, rows), dtype))
    coefficients = jax.scipy.linalg.solve(gram+damping*jnp.eye(rows, dtype=dtype), force, assume_a='pos')

    def vectors_body(i, carry):
        direction, gradient, product = carry
        first, a = block(i)
        d, g = a.T@coefficients, a.T@force
        old_d = jax.lax.dynamic_slice(direction, (first,), (width,))
        old_g = jax.lax.dynamic_slice(gradient, (first,), (width,))
        # The masked last block may overlap the previous one. Add, do not
        # overwrite its already-filled prefix with masked zeros.
        direction = jax.lax.dynamic_update_slice(direction, old_d+d, (first,))
        gradient = jax.lax.dynamic_update_slice(gradient, old_g+g, (first,))
        return direction, gradient, product+a@d

    direction, gradient, product = jax.lax.fori_loop(0, blocks, vectors_body,
        (jnp.zeros(parameters, dtype), jnp.zeros(parameters, dtype), jnp.zeros(rows, dtype)))

    def residual_body(i, squared):
        first, a = block(i)
        d = jax.lax.dynamic_slice(direction, (first,), (width,))
        g = jax.lax.dynamic_slice(gradient, (first,), (width,))
        valid = first+jnp.arange(width) >= i*width
        residual = jnp.where(valid, a.T@product+damping*d-g, 0.)
        return squared+jnp.vdot(residual, residual).real

    squared_residual = jax.lax.fori_loop(0, blocks, residual_body, jnp.asarray(0., dtype))
    gradient_norm = jnp.linalg.norm(gradient)
    return direction, gradient, dict(
        gradient_norm=gradient_norm,
        relative_linear_residual=jnp.sqrt(squared_residual)/jnp.maximum(gradient_norm, 1e-30),
        raw_fisher_norm=jnp.linalg.norm(product),
        parameter_blocks=jnp.asarray(blocks),
        maximum_normalized_score_block_entries=jnp.asarray(rows*width),
        retained_raw_score_entries=jnp.asarray(scores.size),
    )
