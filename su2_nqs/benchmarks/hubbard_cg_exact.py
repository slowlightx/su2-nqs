"""Independent exact-small diagnostics for physical-basis HubbardCG states."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.hubbard_cg import HubbardCGConfig, hubbard_cg_amplitude
from ..operators.hubbard_fock import (
    HubbardBond,
    apply_canonical_hop,
    enumerate_hubbard_tokens,
    fock_mask_to_tokens,
    hubbard_fock_sector_matrix,
    tokens_to_fock_mask,
)
from .hubbard_fock import apply_fermion_hop, enumerate_fock_basis


def hubbard_spin_squared_matrix(
    n_sites: int,
    n_electrons: int,
    *,
    total_m2: int,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Build physical total ``S^2`` in one fixed ``(Ne,2*Sz)`` sector."""

    full_basis = enumerate_fock_basis(n_sites, n_electrons)
    positions = {mask: index for index, mask in enumerate(full_basis)}
    s_plus = np.zeros((len(full_basis), len(full_basis)), dtype=np.float64)
    sz = np.zeros((len(full_basis), len(full_basis)), dtype=np.float64)
    for column, mask in enumerate(full_basis):
        n_up = sum((mask >> (2 * site)) & 1 for site in range(n_sites))
        n_down = sum((mask >> (2 * site + 1)) & 1 for site in range(n_sites))
        sz[column, column] = 0.5 * (n_up - n_down)
        for site in range(n_sites):
            result = apply_fermion_hop(mask, 2 * site, 2 * site + 1)
            if result is not None:
                new_mask, sign = result
                s_plus[positions[new_mask], column] += sign
    s_minus = s_plus.T
    full_s2 = sz @ sz + 0.5 * (s_plus @ s_minus + s_minus @ s_plus)
    selected_masks = tuple(
        mask
        for mask in full_basis
        if sum((mask >> (2 * site)) & 1 for site in range(n_sites))
        - sum((mask >> (2 * site + 1)) & 1 for site in range(n_sites))
        == total_m2
    )
    indices = [positions[mask] for mask in selected_masks]
    sector = full_s2[np.ix_(indices, indices)]
    tokens = tuple(fock_mask_to_tokens(mask, n_sites) for mask in selected_masks)
    return sector, tokens


def hubbard_spin_squared_sector_sparse(
    n_sites: int,
    n_electrons: int,
    *,
    total_m2: int,
):
    """Build sparse total ``S^2`` without a full-Fock dense allocation.

    The construction applies the physical spin ladders between the requested
    fixed-``(Ne,2*Sz)`` sector and its two adjacent magnetic sectors, then
    forms

    ``S^2 = Sz^2 + (S+ S- + S- S+)/2``.

    It is an observable/reference operator only: no Hamiltonian
    diagonalization is performed.
    """

    from scipy import sparse

    basis = enumerate_hubbard_tokens(
        n_sites, n_electrons, total_m2=total_m2
    )
    source_masks = tuple(tokens_to_fock_mask(tokens) for tokens in basis)

    def ladder(*, raise_spin: bool):
        target_m2 = total_m2 + (2 if raise_spin else -2)
        target = enumerate_hubbard_tokens(
            n_sites, n_electrons, total_m2=target_m2
        )
        target_positions = {
            tokens_to_fock_mask(tokens): index
            for index, tokens in enumerate(target)
        }
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for column, mask in enumerate(source_masks):
            for site in range(n_sites):
                up = 2 * site
                down = up + 1
                destination, source = (up, down) if raise_spin else (down, up)
                result = apply_canonical_hop(mask, destination, source)
                if result is None:
                    continue
                new_mask, sign = result
                rows.append(target_positions[new_mask])
                columns.append(column)
                values.append(float(sign))
        return sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(target), len(basis)),
            dtype=np.float64,
        ).tocsr()

    s_plus = ladder(raise_spin=True)
    s_minus = ladder(raise_spin=False)
    sz = 0.5 * float(total_m2)
    identity = sparse.identity(len(basis), dtype=np.float64, format="csr")
    spin_squared = (
        (sz * sz) * identity
        + 0.5 * (s_plus.T @ s_plus + s_minus.T @ s_minus)
    ).tocsr()
    spin_squared.sum_duplicates()
    spin_squared.eliminate_zeros()
    spin_squared.sort_indices()
    return spin_squared, basis


def reconstruct_hubbard_cg_state(
    params: dict[str, object],
    config: HubbardCGConfig,
    *,
    amplitude_fn: Callable[[dict[str, object], jax.Array, HubbardCGConfig], jax.Array] = hubbard_cg_amplitude,
) -> tuple[jax.Array, tuple[tuple[int, ...], ...]]:
    """Exhaustively evaluate one HubbardCG state in its physical sector."""

    basis = enumerate_hubbard_tokens(
        config.n_sites, config.n_electrons, total_m2=config.target_m2
    )
    if not basis:
        return jnp.empty((0,), dtype=jnp.float64), basis
    token_batch = jnp.asarray(basis, dtype=jnp.int32)
    amplitudes = jax.vmap(lambda tokens: amplitude_fn(params, tokens, config))(
        token_batch
    )
    return amplitudes, basis


def target_spin_subspace(
    spin_squared: np.ndarray,
    target_q: int,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return orthonormal columns spanning doubled spin ``target_q``."""

    eigenvalues, eigenvectors = np.linalg.eigh(spin_squared)
    spin = 0.5 * target_q
    selected = np.abs(eigenvalues - spin * (spin + 1.0)) <= tolerance
    if not np.any(selected):
        raise ValueError("target spin is absent from this physical sector")
    return eigenvectors[:, selected]


def exact_target_spin_ground_state(
    hamiltonian: np.ndarray,
    spin_squared: np.ndarray,
    target_q: int,
) -> tuple[float, np.ndarray]:
    """Diagonalize ``H`` after an independently constructed spin projection."""

    subspace = target_spin_subspace(spin_squared, target_q)
    projected = subspace.conj().T @ hamiltonian @ subspace
    values, vectors = np.linalg.eigh(projected)
    state = subspace @ vectors[:, 0]
    return float(values[0]), state


def exact_hubbard_cg_energy(
    params: dict[str, object],
    config: HubbardCGConfig,
    hamiltonian: jax.Array,
    *,
    amplitude_fn: Callable[[dict[str, object], jax.Array, HubbardCGConfig], jax.Array] = hubbard_cg_amplitude,
) -> jax.Array:
    """Differentiable exhaustive Rayleigh quotient for tiny calibration runs."""

    amplitudes, _ = reconstruct_hubbard_cg_state(
        params, config, amplitude_fn=amplitude_fn
    )
    norm = jnp.vdot(amplitudes, amplitudes).real
    numerator = jnp.vdot(amplitudes, hamiltonian @ amplitudes).real
    return numerator / norm


def evaluate_hubbard_cg_exact(
    params: dict[str, object],
    config: HubbardCGConfig,
    bonds: Sequence[HubbardBond],
    *,
    interaction: float,
    amplitude_fn: Callable[[dict[str, object], jax.Array, HubbardCGConfig], jax.Array] = hubbard_cg_amplitude,
) -> dict[str, float]:
    """Measure H0 spin purity and exact-small H1/H2 energy diagnostics."""

    amplitudes_jax, basis = reconstruct_hubbard_cg_state(
        params, config, amplitude_fn=amplitude_fn
    )
    amplitudes = np.asarray(amplitudes_jax, dtype=np.complex128)
    hamiltonian, h_basis = hubbard_fock_sector_matrix(
        config.n_sites,
        config.n_electrons,
        config.target_m2,
        bonds,
        interaction=interaction,
    )
    spin_squared, spin_basis = hubbard_spin_squared_matrix(
        config.n_sites, config.n_electrons, total_m2=config.target_m2
    )
    if basis != h_basis or basis != spin_basis:
        raise RuntimeError("physical basis ordering mismatch")
    norm = float(np.vdot(amplitudes, amplitudes).real)
    if not np.isfinite(norm) or norm <= 1.0e-28:
        raise ValueError("HubbardCG state has zero or non-finite norm")
    normalized = amplitudes / np.sqrt(norm)
    target_s = 0.5 * config.target_q
    spin_residual = spin_squared @ normalized - target_s * (target_s + 1.0) * normalized
    energy = float(np.vdot(normalized, hamiltonian @ normalized).real)
    variance = float(
        np.vdot(
            hamiltonian @ normalized - energy * normalized,
            hamiltonian @ normalized - energy * normalized,
        ).real
    )
    exact_energy, exact_state = exact_target_spin_ground_state(
        hamiltonian, spin_squared, config.target_q
    )
    fidelity = float(abs(np.vdot(exact_state, normalized)) ** 2)
    doublons = np.asarray(
        [sum(token == 3 for token in state) for state in basis], dtype=np.float64
    )
    probabilities = np.abs(normalized) ** 2
    return {
        "basis_size": float(len(basis)),
        "norm_before_normalization": norm,
        "spin_squared_expectation": float(np.vdot(normalized, spin_squared @ normalized).real),
        "spin_residual_norm": float(np.linalg.norm(spin_residual)),
        "energy": energy,
        "energy_error": energy - exact_energy,
        "energy_variance": variance,
        "exact_target_spin_energy": exact_energy,
        "ground_state_fidelity": fidelity,
        "double_occupancy_total": float(probabilities @ doublons),
    }


def basis_masks(basis: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Expose the physical ordering for comparisons with independent oracles."""

    return tuple(tokens_to_fock_mask(tokens) for tokens in basis)


__all__ = [
    "basis_masks",
    "evaluate_hubbard_cg_exact",
    "exact_hubbard_cg_energy",
    "exact_target_spin_ground_state",
    "hubbard_spin_squared_matrix",
    "hubbard_spin_squared_sector_sparse",
    "reconstruct_hubbard_cg_state",
    "target_spin_subspace",
]
