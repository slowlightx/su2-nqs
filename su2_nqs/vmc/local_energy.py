"""Fusion-space or computational-space local-energy estimators."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

LogPsi = Callable[[object, jax.Array], jax.Array]
Connectivity = Callable[[Sequence[int]], tuple[Sequence[Sequence[int]], np.ndarray]]


def local_energy(
    params: object,
    sample: Sequence[int] | jax.Array,
    log_psi_fn: LogPsi,
    get_conn_fn: Connectivity,
) -> jax.Array:
    """Compute ``sum_x' H_xx' psi(x') / psi(x)`` for one sample."""

    concrete_sample = tuple(int(value) for value in np.asarray(sample))
    connected, elements = get_conn_fn(concrete_sample)
    if len(elements) == 0:
        return jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)
    connected_array = jnp.asarray(np.asarray(connected), dtype=jnp.int32)
    log_value = log_psi_fn(params, jnp.asarray(concrete_sample, dtype=jnp.int32))
    connected_logs = jax.vmap(lambda value: log_psi_fn(params, value))(connected_array)
    ratios = jnp.exp(connected_logs - log_value)
    return jnp.sum(jnp.asarray(elements, dtype=jnp.complex128) * ratios)


def local_energies(
    params: object,
    samples: Sequence[Sequence[int]] | jax.Array,
    log_psi_fn: LogPsi,
    get_conn_fn: Connectivity,
) -> jax.Array:
    """Evaluate local energies for a concrete Monte Carlo batch.

    Duplicate autoregressive samples share one local-energy evaluation.  This
    is a substantial speedup in the small fusion spaces used for validation
    and does not change the Monte Carlo weighting.
    """

    concrete = np.asarray(samples, dtype=np.int32)
    if concrete.ndim != 2 or concrete.shape[0] == 0:
        raise ValueError("samples must be a non-empty rank-two array")
    unique, inverse = np.unique(concrete, axis=0, return_inverse=True)
    connected_batches: list[np.ndarray] = []
    element_batches: list[np.ndarray] = []
    owner_batches: list[np.ndarray] = []
    for owner, sample in enumerate(unique):
        connected, elements = get_conn_fn(tuple(int(value) for value in sample))
        if len(elements) == 0:
            continue
        connected_batches.append(np.asarray(connected, dtype=np.int32))
        element_batches.append(np.asarray(elements, dtype=np.complex128))
        owner_batches.append(np.full(len(elements), owner, dtype=np.int32))

    if not connected_batches:
        return jnp.zeros((concrete.shape[0],), dtype=jnp.complex128)
    connected_array = jnp.asarray(np.concatenate(connected_batches), dtype=jnp.int32)
    elements = jnp.asarray(np.concatenate(element_batches), dtype=jnp.complex128)
    owners = jnp.asarray(np.concatenate(owner_batches), dtype=jnp.int32)
    unique_array = jnp.asarray(unique, dtype=jnp.int32)
    base_logs = jax.vmap(lambda value: log_psi_fn(params, value))(unique_array)
    connected_logs = jax.vmap(lambda value: log_psi_fn(params, value))(connected_array)
    contributions = elements * jnp.exp(connected_logs - base_logs[owners])
    unique_energies = jnp.zeros((len(unique),), dtype=jnp.complex128).at[owners].add(
        contributions
    )
    return unique_energies[jnp.asarray(inverse, dtype=jnp.int32)]


def energy_and_variance(
    params: object,
    samples: Sequence[Sequence[int]] | jax.Array,
    log_psi_fn: LogPsi,
    get_conn_fn: Connectivity,
) -> tuple[jax.Array, jax.Array]:
    """Monte Carlo energy and variance from exact autoregressive samples."""

    energies = local_energies(params, samples, log_psi_fn, get_conn_fn)
    mean = jnp.mean(energies)
    variance = jnp.mean(jnp.abs(energies - mean) ** 2)
    return jnp.real(mean), jnp.real(variance)
