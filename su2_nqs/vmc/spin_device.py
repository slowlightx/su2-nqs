"""Accelerator-native full-Heisenberg action on physical spin bits."""

from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp

from ..models.physical_tree import (
    PhysicalSpinTreeConfig,
    PhysicalTreeAmplitude,
    physical_spin_tree_amplitude,
)
from ..operators.spin_config import SpinBond


def spin_device_hpsi(
    params: object,
    samples: jax.Array,
    bonds: Iterable[SpinBond],
    config: PhysicalSpinTreeConfig,
    *,
    amplitude_fn: PhysicalTreeAmplitude = physical_spin_tree_amplitude,
) -> tuple[jax.Array, jax.Array]:
    """Apply the full graph with batched JAX exchanges and no host columns."""

    concrete_bonds = tuple(bonds)
    if not concrete_bonds:
        raise ValueError("bonds must be nonempty")
    normalized = []
    for bond in concrete_bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        first, second = int(first), int(second)
        if (
            first == second
            or not 0 <= first < config.n_sites
            or not 0 <= second < config.n_sites
        ):
            raise ValueError("bond endpoints must be distinct physical sites")
        normalized.append((first, second, float(coupling)))

    states = jnp.asarray(samples, dtype=jnp.int32)
    if states.ndim != 2 or states.shape[1] != config.n_sites:
        raise ValueError("samples have the wrong physical-spin shape")
    first = jnp.asarray([row[0] for row in normalized], dtype=jnp.int32)
    second = jnp.asarray([row[1] for row in normalized], dtype=jnp.int32)
    coupling = jnp.asarray([row[2] for row in normalized], dtype=jnp.float64)
    centers = jax.vmap(lambda row: amplitude_fn(params, row, config))(states)
    first_bits = states[:, first]
    second_bits = states[:, second]
    antiparallel = first_bits != second_bits
    connected = jnp.repeat(states[:, None, :], len(normalized), axis=1)
    batch = jnp.arange(len(states), dtype=jnp.int32)[:, None]
    bond_axis = jnp.arange(len(normalized), dtype=jnp.int32)[None, :]
    connected = connected.at[batch, bond_axis, first[None, :]].set(second_bits)
    connected = connected.at[batch, bond_axis, second[None, :]].set(first_bits)
    connected_values = jax.vmap(lambda row: amplitude_fn(params, row, config))(
        connected.reshape((-1, config.n_sites))
    ).reshape(antiparallel.shape)
    diagonal = jnp.sum(
        coupling[None, :] * jnp.where(antiparallel, -0.25, 0.25), axis=1
    )
    off_diagonal = jnp.sum(
        jnp.where(
            antiparallel,
            0.5 * coupling[None, :] * connected_values,
            0.0 + 0.0j,
        ),
        axis=1,
    )
    return centers, diagonal * centers + off_diagonal


__all__ = ["spin_device_hpsi"]
