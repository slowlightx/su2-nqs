"""Exact-small overlapping Schur views and local recoupling operators.

A single Schur tiling is a complete orthonormal SU(2) multiplicity basis, but
it privileges one disjoint partition of a two-dimensional torus.  This module
keeps each individual tiling complete and orthonormal while combining several
translated, hence mutually overlapping, tilings.  If ``U_0`` is the primary
basis and ``U_v`` another view, their exact recoupling is
``R_v = U_v.T @ U_0``.  We never need to materialize the dense ``R_v``:
``U_v.T @ Phi_A1`` applies it directly to the small Gamma-A1 subspace.

The resulting operator bank is built only from reduced local-channel and
inter-cell doubled-spin labels.  Magnetic indices never enter a learned
quantity.  A deterministic orthogonal conjugation *inside the same A1
subspace* supplies the matched scrambled-locality falsifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from ..su2.local_schur import CellKind
from ..su2.local_schur_fusion import (
    four_cell_sector_action_table,
    local_schur_channels,
)
from .four_cell_schur import four_cell_schur_sector_basis_matrix
from .four_cell_schur_frontier import (
    SQUARE_N16_SUPPORTS,
    TRIANGULAR_N12_SUPERCELL,
    TRIANGULAR_N12_SUPPORTS,
)


Supports = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class OverlappingSchurA1Representation:
    """Exact Gamma-A1 coordinates and matched local/scrambled operators."""

    kind: CellKind
    tilings: tuple[Supports, ...]
    coefficient_a1_basis: np.ndarray
    physical_a1_basis: np.ndarray
    physical_operators: np.ndarray
    scrambled_operators: np.ndarray
    operator_labels: tuple[str, ...]
    diagnostics: dict[str, float | int]


def _canonical_tiling(supports: Sequence[Sequence[int]]) -> Supports:
    return tuple(sorted(tuple(sorted(int(site) for site in support)) for support in supports))


def translated_tiling_orbit(case_name: str) -> tuple[Supports, ...]:
    """Return distinct translated partitions while preserving view ordering."""

    if case_name == "triangular_n12_j2_0125":
        primary = TRIANGULAR_N12_SUPPORTS
        group = triangular_hnf_translation_group(TRIANGULAR_N12_SUPERCELL)
    elif case_name == "square_n16_j2_05":
        primary = SQUARE_N16_SUPPORTS
        group = rectangular_translation_group(4, 4)
    else:
        raise ValueError(f"unknown overlapping-Schur case: {case_name}")
    result: list[Supports] = [tuple(tuple(support) for support in primary)]
    seen = {_canonical_tiling(primary)}
    for image in group:
        mapped = tuple(
            tuple(int(image[site]) for site in support) for support in primary
        )
        signature = _canonical_tiling(mapped)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(mapped)
    n_sites = len(group[0])
    for tiling in result:
        flattened = tuple(site for support in tiling for site in support)
        if tuple(sorted(flattened)) != tuple(range(n_sites)):
            raise RuntimeError("a translated Schur view is not a site partition")
    return tuple(result)


def _modified_gram_schmidt(
    vectors: Sequence[np.ndarray],
    *,
    tolerance: float,
    maximum_count: int | None = None,
) -> tuple[np.ndarray, ...]:
    basis: list[np.ndarray] = []
    for raw in vectors:
        candidate = np.asarray(raw, dtype=np.float64).copy()
        for _ in range(2):
            for vector in basis:
                candidate -= np.vdot(vector, candidate).real * vector
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate.flat[pivot] < 0.0:
            candidate *= -1.0
        candidate.setflags(write=False)
        basis.append(candidate)
        if maximum_count is not None and len(basis) == maximum_count:
            break
    return tuple(basis)


def _tagged_modified_gram_schmidt(
    vectors: Sequence[np.ndarray],
    *,
    tolerance: float,
    maximum_count: int,
) -> tuple[tuple[int, np.ndarray], ...]:
    basis: list[tuple[int, np.ndarray]] = []
    for index, raw in enumerate(vectors):
        candidate = np.asarray(raw, dtype=np.float64).copy()
        for _ in range(2):
            for _old_index, vector in basis:
                candidate -= np.vdot(vector, candidate).real * vector
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate.flat[pivot] < 0.0:
            candidate *= -1.0
        candidate.setflags(write=False)
        basis.append((index, candidate))
        if len(basis) == maximum_count:
            break
    return tuple(basis)


def gamma_a1_coefficient_basis(
    singlet_basis: np.ndarray,
    permutation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return deterministic A1 coordinates inside one complete singlet basis."""

    unitary = np.asarray(singlet_basis, dtype=np.float64)
    indices = np.asarray(permutation_indices, dtype=np.int64)
    if unitary.ndim != 2 or indices.ndim != 2 or indices.shape[1] != unitary.shape[0]:
        raise ValueError("singlet basis and group-index shapes are inconsistent")
    orbit_label = np.min(indices, axis=0)
    orbit_rows = []
    for label in np.unique(orbit_label):
        selected = orbit_label == label
        orbit_rows.append(
            np.sum(unitary[selected], axis=0) / np.sqrt(float(np.count_nonzero(selected)))
        )
    coefficient_vectors = _modified_gram_schmidt(
        orbit_rows, tolerance=2.0e-11
    )
    if not coefficient_vectors:
        raise RuntimeError("the singlet space has no Gamma-A1 component")
    coefficient_basis = np.column_stack(coefficient_vectors)
    physical_basis = unitary @ coefficient_basis
    projected = np.mean(physical_basis[indices], axis=0)
    spatial_residual = float(np.linalg.norm(projected - physical_basis))
    if spatial_residual > 5.0e-11:
        raise RuntimeError("constructed coefficient basis is not Gamma-A1")
    gram_error = float(
        max(
            np.linalg.norm(coefficient_basis.T @ coefficient_basis - np.eye(coefficient_basis.shape[1])),
            np.linalg.norm(physical_basis.T @ physical_basis - np.eye(physical_basis.shape[1])),
        )
    )
    if gram_error > 5.0e-11:
        raise RuntimeError("constructed Gamma-A1 basis is not orthonormal")
    coefficient_basis.setflags(write=False)
    physical_basis.setflags(write=False)
    return coefficient_basis, physical_basis, spatial_residual


def _reduced_feature_table(kind: CellKind) -> tuple[np.ndarray, tuple[str, ...]]:
    actions = four_cell_sector_action_table(kind, 0)
    channels = local_schur_channels(kind)
    local = actions[:, (0, 1, 3, 4)]
    counts = np.column_stack(
        [np.sum(local == channel, axis=1) for channel in range(len(channels))]
    ).astype(np.float64)
    pair_q = actions[:, 2]
    columns: list[np.ndarray] = []
    labels: list[str] = []

    def add(label: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        standard = float(np.std(values))
        if standard <= 1.0e-12:
            return
        columns.append((values - np.mean(values)) / standard)
        labels.append(label)

    for index, (q, a) in enumerate(channels):
        add(f"count_q{q}_a{a}", counts[:, index])
    pair_values = tuple(sorted({int(value) for value in pair_q}))
    pair_indicators = {
        value: np.asarray(pair_q == value, dtype=np.float64) for value in pair_values
    }
    for value in pair_values:
        add(f"pair_q{value}", pair_indicators[value])
    for left in range(len(channels)):
        for right in range(left, len(channels)):
            add(
                f"count_product_{left}_{right}",
                counts[:, left] * counts[:, right],
            )
    for channel in range(len(channels)):
        for value in pair_values:
            add(
                f"count_{channel}_times_pair_q{value}",
                counts[:, channel] * pair_indicators[value],
            )
    return np.column_stack(columns), tuple(labels)


def _haar_orthogonal(dimension: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    orthogonal, triangular = np.linalg.qr(raw)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    return orthogonal * signs[None, :]


def build_overlapping_schur_a1_representation(
    case_name: str,
    singlet_basis: np.ndarray,
    configurations: Sequence[Sequence[int]],
    permutation_indices: np.ndarray,
    *,
    maximum_operator_count: int = 8,
) -> OverlappingSchurA1Representation:
    """Build the exact-small overlapping-view operator representation."""

    if isinstance(maximum_operator_count, bool) or maximum_operator_count < 1:
        raise ValueError("maximum_operator_count must be a positive integer")
    if case_name == "triangular_n12_j2_0125":
        kind: CellKind = "triangular_triangle"
        scramble_seed = 2026092401
    elif case_name == "square_n16_j2_05":
        kind = "square_plaquette"
        scramble_seed = 2026092402
    else:
        raise ValueError(f"unknown overlapping-Schur case: {case_name}")
    primary = np.asarray(singlet_basis, dtype=np.float64)
    coefficient_a1, physical_a1, spatial_residual = gamma_a1_coefficient_basis(
        primary, permutation_indices
    )
    features, feature_labels = _reduced_feature_table(kind)
    tilings = translated_tiling_orbit(case_name)
    accumulated = np.zeros(
        (features.shape[1], coefficient_a1.shape[1], coefficient_a1.shape[1]),
        dtype=np.float64,
    )
    maximum_view_gram_error = 0.0
    checked_configurations = tuple(tuple(int(bit) for bit in row) for row in configurations)
    for tiling in tilings:
        if _canonical_tiling(tiling) == _canonical_tiling(tilings[0]):
            view = primary
            view_configurations = checked_configurations
        else:
            view, view_configurations = four_cell_schur_sector_basis_matrix(
                kind, 0, target_m2=0, supports=tiling
            )
        if view_configurations != checked_configurations:
            raise RuntimeError("overlapping Schur views use different physical bases")
        reduced_a1 = np.asarray(view, dtype=np.float64).T @ physical_a1
        maximum_view_gram_error = max(
            maximum_view_gram_error,
            float(
                np.linalg.norm(
                    reduced_a1.T @ reduced_a1
                    - np.eye(reduced_a1.shape[1])
                )
            ),
        )
        accumulated += np.einsum(
            "ni,nk,nj->kij", reduced_a1, features, reduced_a1
        )
    accumulated /= float(len(tilings))
    identity = np.eye(coefficient_a1.shape[1])
    candidates = []
    for operator in accumulated:
        hermitian = 0.5 * (operator + operator.T)
        hermitian -= np.trace(hermitian) / hermitian.shape[0] * identity
        candidates.append(hermitian.reshape(-1))
    orthogonalized = _tagged_modified_gram_schmidt(
        candidates,
        tolerance=2.0e-10,
        maximum_count=int(maximum_operator_count),
    )
    if not orthogonalized:
        raise RuntimeError("overlapping Schur features generated no A1 operators")
    physical_operators = np.stack(
        [vector.reshape(identity.shape) for _index, vector in orthogonalized]
    )
    operator_labels = tuple(
        f"orthogonalized_from_{feature_labels[index]}"
        for index, _vector in orthogonalized
    )
    rotation = _haar_orthogonal(coefficient_a1.shape[1], scramble_seed)
    scrambled_operators = np.einsum(
        "ai,kab,bj->kij", rotation, physical_operators, rotation
    )
    eigenvalue_error = max(
        float(
            np.max(
                np.abs(
                    np.linalg.eigvalsh(physical) - np.linalg.eigvalsh(scrambled)
                )
            )
        )
        for physical, scrambled in zip(
            physical_operators, scrambled_operators, strict=True
        )
    )
    hermiticity_error = float(
        max(
            np.max(np.abs(physical_operators - physical_operators.transpose(0, 2, 1))),
            np.max(np.abs(scrambled_operators - scrambled_operators.transpose(0, 2, 1))),
        )
    )
    if max(maximum_view_gram_error, eigenvalue_error, hermiticity_error) > 5.0e-10:
        raise RuntimeError("overlapping Schur representation failed an exact invariant")
    for array in (
        physical_operators,
        scrambled_operators,
    ):
        array.setflags(write=False)
    return OverlappingSchurA1Representation(
        kind=kind,
        tilings=tilings,
        coefficient_a1_basis=coefficient_a1,
        physical_a1_basis=physical_a1,
        physical_operators=physical_operators,
        scrambled_operators=scrambled_operators,
        operator_labels=operator_labels,
        diagnostics={
            "tiling_count": len(tilings),
            "singlet_dimension": int(primary.shape[1]),
            "gamma_a1_dimension": int(coefficient_a1.shape[1]),
            "operator_count": int(len(physical_operators)),
            "spatial_residual": spatial_residual,
            "maximum_view_gram_error": maximum_view_gram_error,
            "maximum_operator_hermiticity_error": hermiticity_error,
            "maximum_scrambled_spectrum_error": eigenvalue_error,
        },
    )


__all__ = [
    "OverlappingSchurA1Representation",
    "build_overlapping_schur_a1_representation",
    "gamma_a1_coefficient_basis",
    "translated_tiling_orbit",
]
