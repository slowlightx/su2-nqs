"""Exact-small evaluation of physical-basis Hubbard tree controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..operators.hubbard_fock import (
    HubbardBond,
    enumerate_hubbard_tokens,
    hubbard_fock_sector_matrix,
)
from .hubbard_cg_exact import hubbard_spin_squared_matrix


PhysicalAmplitude = Callable[[dict[str, object], jax.Array, object], jax.Array]


def reconstruct_physical_tree_state(
    params: dict[str, object],
    config: object,
    *,
    n_sites: int,
    n_electrons: int,
    total_m2: int,
    amplitude_fn: PhysicalAmplitude,
) -> tuple[jax.Array, tuple[tuple[int, ...], ...]]:
    """Evaluate any physical-tree amplitude over a fixed sector."""

    basis = enumerate_hubbard_tokens(
        n_sites, n_electrons, total_m2=total_m2
    )
    batch = jnp.asarray(basis, dtype=jnp.int32)
    amplitudes = jax.vmap(lambda tokens: amplitude_fn(params, tokens, config))(batch)
    return amplitudes, basis


def exact_physical_tree_energy(
    params: dict[str, object],
    config: object,
    hamiltonian: jax.Array,
    *,
    n_sites: int,
    n_electrons: int,
    total_m2: int,
    amplitude_fn: PhysicalAmplitude,
) -> jax.Array:
    """Differentiable exact-support Rayleigh quotient for a physical tree."""

    amplitudes, _ = reconstruct_physical_tree_state(
        params,
        config,
        n_sites=n_sites,
        n_electrons=n_electrons,
        total_m2=total_m2,
        amplitude_fn=amplitude_fn,
    )
    norm = jnp.vdot(amplitudes, amplitudes).real
    return jnp.vdot(amplitudes, hamiltonian @ amplitudes).real / norm


def evaluate_physical_tree_exact(
    params: dict[str, object],
    config: object,
    bonds: Sequence[HubbardBond],
    *,
    n_sites: int,
    n_electrons: int,
    total_m2: int,
    interaction: float,
    amplitude_fn: PhysicalAmplitude,
) -> dict[str, float]:
    """Return energy, variance, fidelity, spin contamination and doublons."""

    amplitudes_jax, basis = reconstruct_physical_tree_state(
        params,
        config,
        n_sites=n_sites,
        n_electrons=n_electrons,
        total_m2=total_m2,
        amplitude_fn=amplitude_fn,
    )
    amplitudes = np.asarray(amplitudes_jax, dtype=np.complex128)
    hamiltonian, h_basis = hubbard_fock_sector_matrix(
        n_sites,
        n_electrons,
        total_m2,
        bonds,
        interaction=interaction,
    )
    spin_squared, spin_basis = hubbard_spin_squared_matrix(
        n_sites, n_electrons, total_m2=total_m2
    )
    if basis != h_basis or basis != spin_basis:
        raise RuntimeError("physical basis ordering mismatch")
    norm = float(np.vdot(amplitudes, amplitudes).real)
    if not np.isfinite(norm) or norm <= 1.0e-28:
        raise ValueError("physical tree state has zero or non-finite norm")
    state = amplitudes / np.sqrt(norm)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    exact_energy = float(eigenvalues[0])
    exact_state = eigenvectors[:, 0]
    h_state = hamiltonian @ state
    energy = float(np.vdot(state, h_state).real)
    spin_expectation = float(np.vdot(state, spin_squared @ state).real)
    doublons = np.asarray(
        [sum(token == 3 for token in configuration) for configuration in basis],
        dtype=np.float64,
    )
    return {
        "basis_size": float(len(basis)),
        "norm_before_normalization": norm,
        "energy": energy,
        "energy_error": energy - exact_energy,
        "energy_variance": float(np.vdot(h_state - energy * state, h_state - energy * state).real),
        "exact_energy": exact_energy,
        "ground_state_fidelity": float(abs(np.vdot(exact_state, state)) ** 2),
        "spin_squared_expectation": spin_expectation,
        "spin_contamination_from_singlet": spin_expectation,
        "double_occupancy_total": float(np.abs(state) ** 2 @ doublons),
    }


__all__ = [
    "PhysicalAmplitude",
    "evaluate_physical_tree_exact",
    "exact_physical_tree_energy",
    "reconstruct_physical_tree_state",
]
