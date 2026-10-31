"""Exact reuse of physical spin configurations, without changing an ansatz.

The A1 sum retains every group element. Only *identical bit strings* are
shared, including their full multiplicities. Labels use site-zero-most-
significant uint64 keys (at most 63 sites). This is a bounded implementation,
not an arbitrary-size encoding or an assumed model symmetry quotient.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..models.spatial_projector import PhysicalSpatialProjector


def _check(projector: PhysicalSpatialProjector, chunk_size: int) -> None:
    if projector.fermionic or any(z != 1 for z in projector.characters):
        raise ValueError('this evaluator supports bosonic A1 only')
    if projector.n_sites > 63:
        raise ValueError('exact uint64 keys require at most 63 sites')
    if not jax.config.x64_enabled:
        raise ValueError('enable JAX x64 before constructing exact spin keys')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')


def _batch(states: jax.Array, n_sites: int) -> jax.Array:
    states = jnp.asarray(states, dtype=jnp.int32)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] != n_sites:
        raise ValueError('expected nonempty (batch, n_sites) binary states')
    return states


def make_chunked_a1_amplitudes(
    raw_amplitude: Callable, projector: PhysicalSpatialProjector, *, chunk_size: int = 64,
) -> Callable:
    """Full coherent A1 amplitudes; scan/rematerialization bounds gradient memory.

    Accepts dynamic ``(params, binary_states)`` and returns a complex batch.
    Every image is retained; padding repeats a valid input and has zero weight.
    Binary input validity is the caller's responsibility inside JIT.
    """
    _check(projector, chunk_size)
    images = jnp.asarray(projector.site_permutations, dtype=jnp.int32)
    group = projector.group_order
    evaluate = jax.checkpoint(jax.vmap(raw_amplitude, in_axes=(None, 0)))

    def amplitudes(params, states):
        states = _batch(states, projector.n_sites)
        count = states.shape[0] * group
        chunks = (count + chunk_size - 1) // chunk_size
        def step(_, index):
            locations = index * chunk_size + jnp.arange(chunk_size)
            valid = locations < count
            safe = jnp.minimum(locations, count - 1)
            bits = states[safe // group]
            pulled = jnp.take_along_axis(bits, images[safe % group], axis=1)
            values = evaluate(params, pulled)
            return None, jnp.where(valid, values, 0)
        _, values = jax.lax.scan(step, None, jnp.arange(chunks))
        return values.reshape(-1)[:count].reshape(-1, group).mean(axis=1)
    return amplitudes


class ProjectedConnectionPlan(NamedTuple):
    """Device-only exact unique keys plus inverse map; no parameter-dependent data."""
    keys: jax.Array
    inverse: jax.Array
    count: jax.Array
    diagonal: jax.Array
    offdiagonal: jax.Array


def make_deduplicated_a1_heisenberg(
    raw_amplitude: Callable, projector: PhysicalSpatialProjector,
    bonds: Sequence[tuple[int, int, float]], *, chunk_size: int = 64,
) -> tuple[Callable, Callable, Callable]:
    """Return ``plan(states), evaluate(params, plan), action(params, states)``.

    Evaluate returns ``(Hpsi/psi, psi, unique_count, padded_evaluation_count)``.
    Repeated bonds are combined before tracing. Like-spin offdiagonal terms
    are absent, and identical raw configurations across centers/connectors/
    images are evaluated once. No commutation or raw-model invariance is used.
    Projection multiplicities remain in the inverse map, not averaged away.

    The dynamic-length evaluation loop is forward-only: use the separately
    differentiable chunked A1 amplitude for VMC score gradients and STOP the
    local-energy gradient, as in the usual VMC covariance estimator. Do not
    differentiate through this H action. Node ratios remain undefined rather
    than being clipped. Plans must be rebuilt when configurations change.
    """
    _check(projector, chunk_size)
    n = projector.n_sites
    merged: dict[tuple[int, int], float] = {}
    for a, b, value in bonds:
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (a, b)):
            raise TypeError('bond sites must be integers')
        if not (0 <= a < n and 0 <= b < n) or a == b or not np.isfinite(value):
            raise ValueError('invalid Heisenberg bond')
        edge = tuple(sorted((int(a), int(b))))
        merged[edge] = merged.get(edge, 0.) + float(value)
    edges = [edge for edge in sorted(merged) if merged[edge] != 0]
    sites = jnp.asarray(edges, dtype=jnp.int32).reshape(-1, 2)
    weights = jnp.asarray([merged[e] for e in edges], dtype=jnp.float64)
    images_np = np.asarray(projector.site_permutations, dtype=np.int32)
    images = jnp.asarray(images_np)
    shifts = jnp.arange(n - 1, -1, -1, dtype=jnp.uint64)
    masks = np.zeros((len(edges), projector.group_order), dtype=np.uint64)
    inverse_images = np.argsort(images_np, axis=1)
    for i, (a, b) in enumerate(edges):
        masks[i] = (np.uint64(1) << (n - 1 - inverse_images[:, a]).astype(np.uint64)) | (
            np.uint64(1) << (n - 1 - inverse_images[:, b]).astype(np.uint64))
    swap_masks = jnp.asarray(masks)
    sentinel = jnp.asarray(np.iinfo(np.uint64).max, dtype=jnp.uint64)

    def plan(states):
        states = _batch(states, n)
        center_keys = jnp.sum(states[:, images].astype(jnp.uint64) << shifts, axis=-1)
        unlike = states[:, sites[:, 0]] != states[:, sites[:, 1]]
        swapped = center_keys[:, None, :] ^ swap_masks[None, :, :]
        swapped = jnp.where(unlike[:, :, None], swapped, sentinel)
        queries = jnp.concatenate((center_keys[:, None, :], swapped), axis=1)
        keys, inverse = jnp.unique(queries, return_inverse=True, size=queries.size, fill_value=sentinel)
        count = jnp.sum(keys != sentinel, dtype=jnp.int32)
        # One extra chunk makes every dynamic slice valid, including the tail.
        keys = jnp.pad(keys, (0, chunk_size), constant_values=sentinel)
        return ProjectedConnectionPlan(keys, inverse.reshape(queries.shape), count,
            jnp.sum(jnp.where(unlike, -.25, .25) * weights, axis=1),
            .5 * unlike * weights)

    evaluate_raw = jax.vmap(raw_amplitude, in_axes=(None, 0))
    def evaluate(params, prepared):
        keys, inverse, count, diagonal, offdiagonal = prepared
        chunks = (count + chunk_size - 1) // chunk_size
        values = jnp.zeros(keys.shape, dtype=jnp.complex128)
        def step(index, output):
            locations = index * chunk_size + jnp.arange(chunk_size)
            selected = jax.lax.dynamic_slice(keys, (index * chunk_size,), (chunk_size,))
            selected = jnp.where(locations < count, selected, keys[0])
            bits = ((selected[:, None] >> shifts) & 1).astype(jnp.int32)
            raw = evaluate_raw(params, bits)
            return jax.lax.dynamic_update_slice(output, jnp.where(locations < count, raw, 0),
                                                (index * chunk_size,))
        values = jax.lax.fori_loop(0, chunks, step, values)
        projected = jnp.mean(values[inverse], axis=-1)
        psi = projected[:, 0]
        ratios = diagonal + jnp.sum(offdiagonal * projected[:, 1:], axis=1) / psi
        return ratios, psi, count, chunks * chunk_size

    def action(params, states):
        return evaluate(params, plan(states))
    return plan, evaluate, action


def make_frozen_vmc_force(amplitudes: Callable, local_energies: jax.Array, n_sites: int) -> Callable:
    """Real-parameter covariance force per site, with fixed local energies.

    Equal-weight samples give ``2 Re mean(conj(O) (E_L-mean(E_L)))/N``.
    On frozen non-Born probes this is a cost/replay diagnostic, not an energy
    derivative estimate. Returns a pytree matching the real input parameters.
    """
    energies = jax.lax.stop_gradient(jnp.asarray(local_energies, dtype=jnp.complex128))
    if energies.ndim != 1 or n_sites < 1:
        raise ValueError('require vector energies and positive site count')
    centered = energies - jnp.mean(energies)
    def objective(params, states):
        if states.shape[0] != energies.shape[0]:
            raise ValueError('one frozen local energy per state is required')
        logs = jnp.log(amplitudes(params, states))
        return 2 * jnp.real(jnp.mean(jnp.conj(logs) * centered)) / n_sites
    return jax.grad(objective)
