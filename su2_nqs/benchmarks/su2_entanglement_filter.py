"""Exact-small SU(2)-scalar entanglement-Hamiltonian filters.

The trainable filter acts on one subsystem as

``exp(theta_1 H_A^J1 + theta_2 H_A^J2 + theta_q S_A^2)``.

Every generator commutes with subsystem SU(2).  The filter can therefore
rotate and reweight multiplicity states inside fixed boundary-spin sectors
without introducing magnetic-index parameters.  This module is an
exact-small mechanism test; exponentiating a half-system operator is not a
scalable NQS evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from .su2_entanglement_spectrum import bipartite_amplitude_matrix


def spin_pair_operator(n_sites: int, first: int, second: int) -> np.ndarray:
    """Return dense ``S_first . S_second`` in the MSB-first bit convention."""

    if n_sites < 2:
        raise ValueError("at least two sites are required")
    first, second = int(first), int(second)
    if first == second or min(first, second) < 0 or max(first, second) >= n_sites:
        raise ValueError("pair sites must be distinct valid subsystem indices")
    dimension = 2**n_sites
    powers = 2 ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    bits = ((np.arange(dimension)[:, None] // powers[None, :]) % 2).astype(np.int8)
    operator = np.zeros((dimension, dimension), dtype=np.float64)
    diagonal = np.where(bits[:, first] == bits[:, second], 0.25, -0.25)
    operator[np.arange(dimension), np.arange(dimension)] = diagonal
    different = np.flatnonzero(bits[:, first] != bits[:, second])
    flipped = bits[different].copy()
    flipped[:, [first, second]] = flipped[:, [second, first]]
    operator[flipped @ powers, different] += 0.5
    return operator


def subsystem_spin_squared(n_sites: int) -> np.ndarray:
    """Return ``S_A^2`` using the same pair-operator implementation."""

    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    result = np.eye(2**n_sites, dtype=np.float64) * (0.75 * n_sites)
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            result += 2.0 * spin_pair_operator(n_sites, first, second)
    return result


def triangular_half_system_generators(
    supercell: tuple[tuple[int, int], tuple[int, int]],
    subsystem: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return internal J1, internal J2, and ``S_A^2`` generators."""

    sites = tuple(int(site) for site in subsystem)
    if not sites or len(set(sites)) != len(sites):
        raise ValueError("subsystem must contain distinct sites")
    site_to_local = {site: local for local, site in enumerate(sites)}
    n_total = abs(
        int(supercell[0][0]) * int(supercell[1][1])
        - int(supercell[0][1]) * int(supercell[1][0])
    )
    if any(site < 0 or site >= n_total for site in sites):
        raise ValueError("subsystem site lies outside the supercell")
    shell_operators = []
    for j1, j2 in ((1.0, 0.0), (0.0, 1.0)):
        operator = np.zeros((2 ** len(sites), 2 ** len(sites)), dtype=np.float64)
        for first, second, _coupling in triangular_hnf_j1j2_bonds(
            supercell, j1=j1, j2=j2
        ):
            if first in site_to_local and second in site_to_local:
                operator += spin_pair_operator(
                    len(sites), site_to_local[first], site_to_local[second]
                )
        shell_operators.append(operator)
    return shell_operators[0], shell_operators[1], subsystem_spin_squared(len(sites))


def spin_permutation_operator(permutation: Sequence[int]) -> np.ndarray:
    """Return the computational-basis unitary for a subsystem-site permutation."""

    order = tuple(int(site) for site in permutation)
    n_sites = len(order)
    if tuple(sorted(order)) != tuple(range(n_sites)):
        raise ValueError("permutation must contain each subsystem site exactly once")
    dimension = 2**n_sites
    powers = 2 ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    bits = ((np.arange(dimension)[:, None] // powers[None, :]) % 2).astype(np.int8)
    images = bits[:, order] @ powers
    operator = np.zeros((dimension, dimension), dtype=np.float64)
    operator[images, np.arange(dimension)] = 1.0
    return operator


def scramble_generators(
    generators: Sequence[np.ndarray], permutation: Sequence[int]
) -> tuple[np.ndarray, ...]:
    """Apply one common SU(2)-preserving isospectral site scramble."""

    unitary = spin_permutation_operator(permutation)
    dimension = unitary.shape[0]
    output = []
    for generator in generators:
        value = np.asarray(generator, dtype=np.float64)
        if value.shape != (dimension, dimension):
            raise ValueError("generator and permutation dimensions disagree")
        output.append(unitary @ value @ unitary.T)
    return tuple(output)


def exponential_filter(
    parameters: Sequence[float], generators: Sequence[np.ndarray]
) -> np.ndarray:
    """Return a stable positive exponential of a real-symmetric generator."""

    theta = np.asarray(parameters, dtype=np.float64)
    if theta.shape != (len(generators),) or np.any(~np.isfinite(theta)):
        raise ValueError("one finite parameter is required per generator")
    combined = np.zeros_like(np.asarray(generators[0]), dtype=np.float64)
    for coefficient, generator in zip(theta, generators, strict=True):
        value = np.asarray(generator, dtype=np.float64)
        if value.shape != combined.shape:
            raise ValueError("all generators must have equal square shape")
        combined += coefficient * value
    combined = 0.5 * (combined + combined.T)
    eigenvalues, eigenvectors = np.linalg.eigh(combined)
    weights = np.exp(np.clip(eigenvalues, -40.0, 40.0))
    return (eigenvectors * weights) @ eigenvectors.T


def apply_subsystem_filter(
    state: np.ndarray,
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    operator: np.ndarray,
) -> np.ndarray:
    """Apply a subsystem operator and return coefficients in the input basis."""

    configurations = np.asarray(basis, dtype=np.int8)
    sites_a = tuple(int(site) for site in subsystem)
    set_a = set(sites_a)
    sites_b = tuple(site for site in range(configurations.shape[1]) if site not in set_a)
    expected = 2 ** len(sites_a)
    value = np.asarray(operator, dtype=np.complex128)
    if value.shape != (expected, expected):
        raise ValueError("operator shape does not match subsystem size")
    amplitudes = bipartite_amplitude_matrix(state, basis, sites_a)
    filtered = value @ amplitudes
    powers_a = 2 ** np.arange(len(sites_a) - 1, -1, -1, dtype=np.int64)
    powers_b = 2 ** np.arange(len(sites_b) - 1, -1, -1, dtype=np.int64)
    rows = configurations[:, sites_a] @ powers_a
    columns = configurations[:, sites_b] @ powers_b
    return np.asarray(filtered[rows, columns])


def bures_schmidt_transport(
    candidate_state: np.ndarray,
    target_state: np.ndarray,
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    *,
    relative_cutoff: float = 1.0e-12,
) -> np.ndarray:
    """Return the positive SU(2)-scalar transport from candidate to target RDM.

    The target state enters only in this oracle ceiling.  Production ansatzes
    must learn or approximate the resulting multiplicity rotation without
    access to the target reduced density matrix.
    """

    if not 0.0 < relative_cutoff < 1.0:
        raise ValueError("relative_cutoff must lie between zero and one")
    candidate = bipartite_amplitude_matrix(candidate_state, basis, subsystem)
    target = bipartite_amplitude_matrix(target_state, basis, subsystem)
    rho_candidate = candidate @ candidate.conj().T
    rho_target = target @ target.conj().T

    def power(matrix: np.ndarray, exponent: float, cutoff: float) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
        threshold = cutoff * max(float(np.max(eigenvalues)), np.finfo(float).tiny)
        weights = np.zeros_like(eigenvalues)
        support = eigenvalues > threshold
        weights[support] = eigenvalues[support] ** exponent
        return (eigenvectors * weights) @ eigenvectors.conj().T

    target_root = power(rho_target, 0.5, np.finfo(float).eps)
    middle = target_root @ rho_candidate @ target_root
    transport = target_root @ power(middle, -0.5, relative_cutoff) @ target_root
    return 0.5 * (transport + transport.conj().T)


__all__ = [
    "apply_subsystem_filter",
    "bures_schmidt_transport",
    "exponential_filter",
    "scramble_generators",
    "spin_pair_operator",
    "spin_permutation_operator",
    "subsystem_spin_squared",
    "triangular_half_system_generators",
]
