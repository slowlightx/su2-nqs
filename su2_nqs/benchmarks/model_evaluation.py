"""Exact vectors for exhaustively evaluating small autoregressive models."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def model_vector(
    params: object,
    basis: Sequence[Sequence[int]],
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> np.ndarray:
    """Return model amplitudes in a supplied small-system basis order."""

    configurations = jnp.asarray(np.asarray(basis), dtype=jnp.int32)
    logs = jax.vmap(lambda configuration: log_psi_fn(params, configuration))(configurations)
    return np.asarray(jnp.exp(logs), dtype=np.complex128)


def parameter_count(params: object) -> int:
    """Count real scalar parameters in a JAX pytree."""

    return sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params))

