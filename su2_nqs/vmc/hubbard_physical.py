"""Padded on-demand physical Hubbard actions for HubbardCG VMC."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
import jax
import jax.numpy as jnp
import numpy as np

from ..models.physical_tree import (
    PhysicalTreeAmplitude,
    PhysicalTreeConfig,
    physical_hubbard_tree_amplitude,
)
from ..operators.hubbard_fock import HubbardBond, hubbard_fock_connectivity


HubbardAmplitude = PhysicalTreeAmplitude


def padded_hubbard_fock_connectivity(
    samples: Sequence[Sequence[int]] | np.ndarray,
    bonds: Iterable[HubbardBond],
    *,
    interaction: float,
    padded_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compile only sampled physical columns into fixed-width arrays.

    Returns ``connected, elements, valid, widths`` with connectivity following
    ``H[new, old]``.  No Hilbert-space enumeration or global sparse matrix is
    performed.
    """

    concrete = np.asarray(samples, dtype=np.int32)
    if concrete.ndim != 2 or concrete.shape[0] < 1:
        raise ValueError("samples must be a nonempty rank-two array")
    checked_bonds = tuple(bonds)
    columns = [
        hubbard_fock_connectivity(
            tuple(int(token) for token in row),
            checked_bonds,
            interaction=interaction,
        )
        for row in concrete
    ]
    widths = np.asarray([len(states) for states, _ in columns], dtype=np.int32)
    required_width = int(np.max(widths))
    if padded_width is not None and (
        isinstance(padded_width, bool) or not isinstance(padded_width, Integral)
    ):
        raise TypeError("padded_width must be an integer")
    maximum_width = required_width if padded_width is None else int(padded_width)
    if maximum_width < required_width:
        raise ValueError(
            f"padded_width={maximum_width} is smaller than required {required_width}"
        )
    if maximum_width < 1:
        raise ValueError("padded_width must be positive")
    connected = np.zeros(
        (len(concrete), maximum_width, concrete.shape[1]), dtype=np.int32
    )
    elements = np.zeros((len(concrete), maximum_width), dtype=np.complex128)
    valid = np.zeros((len(concrete), maximum_width), dtype=bool)
    for row, (states, values) in enumerate(columns):
        width = len(states)
        connected[row, :width] = np.asarray(states, dtype=np.int32)
        elements[row, :width] = np.asarray(values, dtype=np.complex128)
        valid[row, :width] = True
    return connected, elements, valid, widths


def hubbard_physical_hpsi(
    params: object,
    samples: jax.Array,
    connected: jax.Array,
    elements: jax.Array,
    valid: jax.Array,
    config: PhysicalTreeConfig,
    *,
    amplitude_fn: HubbardAmplitude = physical_hubbard_tree_amplitude,
) -> tuple[jax.Array, jax.Array]:
    """Return center amplitudes and ``H|psi>`` on sampled Fock states."""

    samples = jnp.asarray(samples, dtype=jnp.int32)
    connected = jnp.asarray(connected, dtype=jnp.int32)
    elements = jnp.asarray(elements, dtype=jnp.complex128)
    valid = jnp.asarray(valid)
    if samples.ndim != 2 or samples.shape[1] != config.n_sites:
        raise ValueError("samples have the wrong HubbardCG shape")
    if connected.ndim != 3 or connected.shape[:1] != samples.shape[:1]:
        raise ValueError("connected states have the wrong batch shape")
    if connected.shape[2] != config.n_sites:
        raise ValueError("connected states have the wrong site count")
    if elements.shape != connected.shape[:2] or valid.shape != elements.shape:
        raise ValueError("one element and validity flag are required per slot")
    centers = jax.vmap(lambda row: amplitude_fn(params, row, config))(samples)
    flattened = connected.reshape((-1, config.n_sites))
    connected_values = jax.vmap(
        lambda row: amplitude_fn(params, row, config)
    )(flattened).reshape(elements.shape)
    hpsi = jnp.sum(
        jnp.where(valid, elements * connected_values, 0.0 + 0.0j), axis=1
    )
    return centers, hpsi


def hubbard_physical_local_energies(
    params: object,
    samples: jax.Array,
    connected: jax.Array,
    elements: jax.Array,
    valid: jax.Array,
    config: PhysicalTreeConfig,
    *,
    amplitude_fn: HubbardAmplitude = physical_hubbard_tree_amplitude,
) -> jax.Array:
    """Return sampled local energies, retaining zero centers as nonfinite."""

    centers, hpsi = hubbard_physical_hpsi(
        params,
        samples,
        connected,
        elements,
        valid,
        config,
        amplitude_fn=amplitude_fn,
    )
    return hpsi / centers


__all__ = [
    "hubbard_physical_hpsi",
    "hubbard_physical_local_energies",
    "padded_hubbard_fock_connectivity",
]
