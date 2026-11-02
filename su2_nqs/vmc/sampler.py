"""Shared sampler protocol and reproducible sampling helper."""

from __future__ import annotations

from collections.abc import Callable

import jax


def draw_samples(
    sample_fn: Callable[[object, jax.Array, int], jax.Array],
    params: object,
    key: jax.Array,
    n_samples: int,
) -> tuple[jax.Array, jax.Array]:
    """Split an RNG key and draw a fresh independent autoregressive batch."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    key, sample_key = jax.random.split(key)
    return sample_fn(params, sample_key, n_samples), key

