"""Shared, node-safe physical Heisenberg queries for expensive amplitudes.

This only coalesces exact configurations; it never truncates Hamiltonian
terms or changes a sampling distribution. Bits use zero=down, site zero MSB.
The underlying real Heisenberg connectivity is H[new,old]; its symmetry makes
these same entries the row action needed for Hpsi(old).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..operators.spin_config import SpinBond, heisenberg_spin_connectivity


@dataclass(frozen=True)
class PhysicalHQueries:
    """One shared amplitude support and padded exact row map."""

    configurations: np.ndarray
    centers: np.ndarray
    center_indices: np.ndarray
    connected_indices: np.ndarray
    elements: np.ndarray
    valid: np.ndarray
    diagnostics: dict[str, int]


def compile_physical_h_queries(samples: np.ndarray, bonds: Sequence[SpinBond]) -> PhysicalHQueries:
    """Deduplicate all centers and full-H connections without amplitude tests."""
    bits = np.asarray(samples)
    if bits.ndim != 2 or not len(bits) or bits.shape[1] < 2 or bits.dtype.kind not in 'biu':
        raise ValueError('samples must be a nonempty integer bit matrix')
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError('physical configurations must contain only zero or one')
    checked = []
    for bond in bonds:
        if len(bond) not in (2, 3):
            raise ValueError('each bond must contain two sites and optionally J')
        a, b = bond[:2]
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in (a, b)):
            raise ValueError('bond sites must be integers')
        j = float(bond[2]) if len(bond) == 3 else 1.
        if a == b or min(a, b) < 0 or max(a, b) >= bits.shape[1] or not np.isfinite(j):
            raise ValueError('invalid bond or nonfinite coupling')
        checked.append((int(a), int(b), j))
    bits = bits.astype(np.int8, copy=True)
    columns = [heisenberg_spin_connectivity(row, checked) for row in bits]
    width = max(len(states) for states, _ in columns)
    all_states = np.concatenate((bits, *(np.asarray(states, dtype=np.int8) for states, _ in columns)))
    unique, inverse = np.unique(all_states, axis=0, return_inverse=True)
    center_indices = inverse[:len(bits)].astype(np.int32)
    indices = np.zeros((len(bits), width), dtype=np.int32)
    values = np.zeros_like(indices, dtype=np.float64)
    mask = np.zeros_like(indices, dtype=bool)
    offset = len(bits)
    for row, (states, elements) in enumerate(columns):
        size = len(states)
        indices[row, :size] = inverse[offset:offset+size]
        values[row, :size] = elements
        mask[row, :size] = True
        offset += size
    for array in (unique, bits, center_indices, indices, values, mask):
        array.setflags(write=False)
    return PhysicalHQueries(unique, bits, center_indices, indices, values, mask,
        dict(center_count=len(bits), bond_count=len(checked), full_H_centers=len(bits),
             raw_amplitude_requests=len(all_states), unique_amplitude_queries=len(unique),
             maximum_connections=width, duplicate_requests_removed=len(all_states)-len(unique)))


def gather_physical_h_action(plan: PhysicalHQueries, amplitudes: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return psi and Hpsi including nodes; deliberately performs no division."""
    values = jnp.asarray(amplitudes)
    if values.shape != (len(plan.configurations),):
        raise ValueError('one amplitude for every unique physical query required')
    connected = values[jnp.asarray(plan.connected_indices)]
    action = jnp.sum(jnp.where(jnp.asarray(plan.valid), connected*jnp.asarray(plan.elements), 0), axis=1)
    return values[jnp.asarray(plan.center_indices)], action
