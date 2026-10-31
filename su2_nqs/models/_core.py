"""Small dependency-free JAX GRU building blocks."""

from __future__ import annotations

from collections.abc import Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _matrix(key: jax.Array, shape: tuple[int, int]) -> jax.Array:
    scale = (2.0 / (shape[0] + shape[1])) ** 0.5
    return scale * jax.random.normal(key, shape, dtype=jnp.float64)


def init_gru_core(
    key: jax.Array,
    *,
    embedding_sizes: Sequence[int],
    embedding_dim: int,
    hidden_dim: int,
    output_dim: int = 2,
) -> dict[str, object]:
    """Initialize embeddings, one GRU cell, and probability/phase heads."""

    keys = iter(jax.random.split(key, 16))
    embeddings = tuple(
        0.05 * jax.random.normal(next(keys), (size, embedding_dim), dtype=jnp.float64)
        for size in embedding_sizes
    )
    input_dim = embedding_dim * len(embedding_sizes)
    gru = {
        "wz": _matrix(next(keys), (input_dim, hidden_dim)),
        "uz": _matrix(next(keys), (hidden_dim, hidden_dim)),
        "bz": jnp.zeros((hidden_dim,), dtype=jnp.float64),
        "wr": _matrix(next(keys), (input_dim, hidden_dim)),
        "ur": _matrix(next(keys), (hidden_dim, hidden_dim)),
        "br": jnp.zeros((hidden_dim,), dtype=jnp.float64),
        "wh": _matrix(next(keys), (input_dim, hidden_dim)),
        "uh": _matrix(next(keys), (hidden_dim, hidden_dim)),
        "bh": jnp.zeros((hidden_dim,), dtype=jnp.float64),
    }
    return {
        "embeddings": embeddings,
        "gru": gru,
        "probability": {
            "w": _matrix(next(keys), (hidden_dim, output_dim)),
            "b": jnp.zeros((output_dim,), dtype=jnp.float64),
        },
        "phase": {
            "w": _matrix(next(keys), (hidden_dim, output_dim)),
            "b": jnp.zeros((output_dim,), dtype=jnp.float64),
        },
        "h0": jnp.zeros((hidden_dim,), dtype=jnp.float64),
    }


def gru_step(
    params: dict[str, jax.Array],
    hidden: jax.Array,
    inputs: jax.Array,
) -> jax.Array:
    """Apply one GRU update to an unbatched or batched leading shape."""

    sigmoid = jax.nn.sigmoid
    update = sigmoid(inputs @ params["wz"] + hidden @ params["uz"] + params["bz"])
    reset = sigmoid(inputs @ params["wr"] + hidden @ params["ur"] + params["br"])
    candidate = jnp.tanh(
        inputs @ params["wh"] + (reset * hidden) @ params["uh"] + params["bh"]
    )
    return (1.0 - update) * hidden + update * candidate


def heads(params: dict[str, object], hidden: jax.Array) -> tuple[jax.Array, jax.Array]:
    probability = params["probability"]
    phase = params["phase"]
    logits = hidden @ probability["w"] + probability["b"]
    phase_values = jnp.pi * jnp.tanh(hidden @ phase["w"] + phase["b"])
    return logits, phase_values


def embed(params: dict[str, object], indices: Sequence[jax.Array | int]) -> jax.Array:
    arrays = [table[index] for table, index in zip(params["embeddings"], indices)]
    return jnp.concatenate(arrays, axis=-1)

