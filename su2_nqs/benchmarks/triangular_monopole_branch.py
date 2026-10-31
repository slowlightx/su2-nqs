"""Exact-small source diagnostics for triangular monopole branches.

The routines here are deliberately dense and are intended only for local
many-body calibration.  They resolve one reciprocal character and one
one-dimensional little-group character in a fixed total-spin multiplicity
space, then measure how much of a physical source is carried by a selected
low-energy eigenspace.  Angular momenta remain doubled integers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from ..lattices.triangular_hnf import (
    IntMatrix,
    IntVector,
    hnf_point_group_permutations,
    hnf_sites,
    hnf_volume,
    reduce_hnf,
)
from ..operators.permutation import (
    site_permutation_fusion_matrix,
    transposition_matrix,
)
from ..su2.fusion_path import count_paths
from .triangular_hnf_space_group import (
    FractionMomentum,
    hnf_fraction_character_numerator,
    momentum_projector_matrix,
    reciprocal_little_group,
)


@dataclass(frozen=True)
class SourceSpectrum:
    """Exact zeroth/first moments and lowest-pole saturation of one source."""

    m0: float
    m1: float
    mean_excitation: float
    lowest_gap: float
    lowest_pole_fraction: float
    m0_closure_error: float
    m1_closure_error: float
    weights: np.ndarray


def little_group_character_projector(
    supercell: IntMatrix,
    momentum: FractionMomentum,
    stabilizer_characters: Mapping[str, int],
    *,
    target_q: int,
) -> np.ndarray:
    """Return a dense exact-small momentum/little-group projector.

    ``stabilizer_characters`` currently supports real one-dimensional
    characters only.  Its keys must equal the exact reciprocal little group.
    """

    little_group = reciprocal_little_group(supercell, momentum)
    characters = {str(name): int(value) for name, value in stabilizer_characters.items()}
    if set(characters) != set(little_group):
        raise ValueError("stabilizer characters must cover the exact little group")
    if any(value not in (-1, 1) for value in characters.values()):
        raise ValueError("only real one-dimensional stabilizer characters are supported")
    elements = {
        name: image
        for name, _, image in hnf_point_group_permutations(supercell)
    }
    dimension = count_paths(hnf_volume(supercell), int(target_q))
    little_projector = np.zeros((dimension, dimension), dtype=np.complex128)
    for name in little_group:
        little_projector += characters[name] * site_permutation_fusion_matrix(
            elements[name], target_q=int(target_q)
        ).toarray()
    little_projector /= len(little_group)
    numerator = hnf_fraction_character_numerator(supercell, momentum)
    momentum_projector = momentum_projector_matrix(
        supercell, numerator, target_q=int(target_q)
    )
    result = momentum_projector @ little_projector
    return np.asarray(0.5 * (result + result.conj().T), dtype=np.complex128)


def projector_basis(
    projector: np.ndarray, *, threshold: float = 0.5
) -> np.ndarray:
    """Return deterministic column-orthonormal coordinates for a projector."""

    value = np.asarray(projector, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("projector must be square")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (value + value.conj().T))
    return np.asarray(eigenvectors[:, eigenvalues > threshold], dtype=np.complex128)


def bond_center_orientation_operators(
    supercell: IntMatrix,
    momentum: FractionMomentum,
    displacements: Sequence[IntVector],
    *,
    target_q: int = 0,
) -> tuple[np.ndarray, ...]:
    """Return Fourier bond-energy operators, one per oriented displacement.

    The phase is attached to the geometric bond center,
    ``exp(2*pi*i*k.(r+delta/2))``.  Distinct periodic images are retained,
    matching the Hamiltonian convention on small aliased tori.
    """

    steps = tuple((int(step[0]), int(step[1])) for step in displacements)
    if not steps:
        raise ValueError("displacements must be nonempty")
    sites = hnf_sites(supercell)
    positions = {site: index for index, site in enumerate(sites)}
    n_sites = len(sites)
    dimension = count_paths(n_sites, int(target_q))
    identity = np.eye(dimension, dtype=np.complex128)
    answer: list[np.ndarray] = []
    for dx, dy in steps:
        if dx == 0 and dy == 0:
            raise ValueError("a bond displacement must be nonzero")
        operator = np.zeros((dimension, dimension), dtype=np.complex128)
        for site in sites:
            target = reduce_hnf((site[0] + dx, site[1] + dy), supercell)
            first, second = positions[site], positions[target]
            if first == second:
                raise ValueError("a displacement aliases a site to itself")
            phase_argument = (
                momentum[0] * (2 * site[0] + dx) / 2
                + momentum[1] * (2 * site[1] + dy) / 2
            )
            phase = np.exp(2.0j * np.pi * float(phase_argument))
            pair = 0.5 * transposition_matrix(
                n_sites, int(target_q), first, second
            ).toarray() - 0.25 * identity
            operator += phase * pair
        answer.append(operator)
    return tuple(answer)


def maximum_low_subspace_source_fraction(
    source_columns: np.ndarray,
    low_vectors: np.ndarray,
    coefficient_map: np.ndarray | None = None,
    *,
    rank_tolerance: float = 1.0e-12,
) -> tuple[float, np.ndarray]:
    """Optimize source coefficients for overlap with a fixed low subspace.

    The generalized Rayleigh quotient is solved only on the numerically
    supported source Gram space.  ``coefficient_map`` can impose a frozen
    symmetry form, for example ``(c0,c0,c1)`` for mirror-even bond sources.
    """

    sources = np.asarray(source_columns, dtype=np.complex128)
    low = np.asarray(low_vectors, dtype=np.complex128)
    if sources.ndim != 2 or low.ndim != 2 or sources.shape[0] != low.shape[0]:
        raise ValueError("source_columns and low_vectors need a common leading dimension")
    mapping = (
        np.eye(sources.shape[1], dtype=np.complex128)
        if coefficient_map is None
        else np.asarray(coefficient_map, dtype=np.complex128)
    )
    if mapping.ndim != 2 or mapping.shape[0] != sources.shape[1]:
        raise ValueError("coefficient_map has incompatible shape")
    reduced_sources = sources @ mapping
    gram = reduced_sources.conj().T @ reduced_sources
    numerator = reduced_sources.conj().T @ low @ low.conj().T @ reduced_sources
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    if eigenvalues.size == 0 or eigenvalues[-1] <= 0.0:
        raise ValueError("source columns have zero norm")
    keep = eigenvalues > rank_tolerance * eigenvalues[-1]
    whitening = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    reduced = whitening.conj().T @ numerator @ whitening
    fractions, vectors = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
    internal = whitening @ vectors[:, -1]
    coefficients = mapping @ internal
    coefficients /= np.linalg.norm(coefficients)
    pivot = int(np.argmax(np.abs(coefficients)))
    coefficients *= np.exp(-1.0j * np.angle(coefficients[pivot]))
    fraction = float(np.clip(fractions[-1].real, 0.0, 1.0))
    return fraction, coefficients


def source_spectrum(
    hamiltonian: np.ndarray,
    ground_energy: float,
    source: np.ndarray,
    target_eigenvalues: Sequence[float],
    target_eigenvectors: np.ndarray,
    *,
    pole_tolerance: float = 1.0e-10,
) -> SourceSpectrum:
    """Evaluate exact spectral moment closure and lowest-pole fraction."""

    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    vector = np.asarray(source, dtype=np.complex128)
    values = np.asarray(tuple(target_eigenvalues), dtype=np.float64)
    states = np.asarray(target_eigenvectors, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if vector.shape != (matrix.shape[0],) or states.shape != (matrix.shape[0], len(values)):
        raise ValueError("source/eigensystem shapes do not match the Hamiltonian")
    if len(values) == 0:
        raise ValueError("target eigensystem must be nonempty")
    amplitudes = states.conj().T @ vector
    weights = np.abs(amplitudes) ** 2
    gaps = values - float(ground_energy)
    m0 = float(np.vdot(vector, vector).real)
    if m0 <= 0.0:
        raise ValueError("source must have positive norm")
    shifted = matrix @ vector - float(ground_energy) * vector
    m1 = float(np.vdot(vector, shifted).real)
    spectral_m0 = float(np.sum(weights))
    spectral_m1 = float(np.dot(weights, gaps))
    lowest = float(np.min(gaps))
    low_mask = gaps <= lowest + float(pole_tolerance)
    return SourceSpectrum(
        m0=m0,
        m1=m1,
        mean_excitation=m1 / m0,
        lowest_gap=lowest,
        lowest_pole_fraction=float(np.sum(weights[low_mask]) / m0),
        m0_closure_error=abs(spectral_m0 - m0),
        m1_closure_error=abs(spectral_m1 - m1),
        weights=np.asarray(weights, dtype=np.float64),
    )


__all__ = [
    "SourceSpectrum",
    "bond_center_orientation_operators",
    "little_group_character_projector",
    "maximum_low_subspace_source_fraction",
    "projector_basis",
    "source_spectrum",
]
