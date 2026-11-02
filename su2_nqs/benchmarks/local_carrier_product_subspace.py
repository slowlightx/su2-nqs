"""Exact product-carrier subspaces for optimization-independent Ritz oracles.

The routines in this module construct physical fixed-``M=0`` columns from
small, exactly validated SU(2) multiplicity carriers.  They never diagonalize
the full Hamiltonian.  A caller may translation-project the columns and solve
the resulting small generalized Ritz problem to distinguish representational
capacity from optimizer behavior.

Quantum numbers follow the repository doubled-integer convention.  Physical
bits use ``0=down``, ``1=up``, with site zero most significant.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np

from ..su2.fusion_tree import FusionTree
from ..su2.recoupling import tree_basis_matrix
from .local_multiplicity_carriers import (
    square_plaquette_singlet_carrier,
    triangular_triangle_spin_half_carrier,
)


CellPartition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GeneralizedRitzSubspace:
    """Rank-revealed solution of ``C^dagger H C`` versus ``C^dagger C``."""

    raw_column_count: int
    projected_rank: int
    overlap_eigenvalues: np.ndarray
    retained_overlap_condition_number: float
    orthonormal_basis: np.ndarray
    ritz_eigenvalues: np.ndarray
    ground_state: np.ndarray
    ground_energy: float
    variance: float
    orthonormality_error: float
    residual_norm: float


def validate_cell_partition(
    cells: Sequence[Sequence[int]],
    *,
    n_sites: int,
    cell_size: int,
) -> CellPartition:
    """Return a deterministic disjoint exact cover of physical sites."""

    checked = tuple(tuple(int(site) for site in cell) for cell in cells)
    if not checked or any(len(cell) != cell_size for cell in checked):
        raise ValueError(f"every cell must contain exactly {cell_size} sites")
    flattened = tuple(site for cell in checked for site in cell)
    if len(flattened) != n_sites or tuple(sorted(flattened)) != tuple(
        range(n_sites)
    ):
        raise ValueError("cells must form a disjoint exact cover of all sites")
    return checked


def _basis_array(
    basis: Sequence[Sequence[int]], *, n_sites: int
) -> np.ndarray:
    array = np.asarray(basis, dtype=np.int8)
    if array.ndim != 2 or array.shape[1] != n_sites:
        raise ValueError(f"basis must have shape (dimension, {n_sites})")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("basis must be binary")
    return array


def _cell_basis_indices(basis: np.ndarray, cells: CellPartition) -> np.ndarray:
    indices = []
    for cell in cells:
        weights = 1 << np.arange(len(cell) - 1, -1, -1, dtype=np.int64)
        indices.append(basis[:, np.asarray(cell, dtype=np.int64)] @ weights)
    return np.column_stack(indices).astype(np.int64, copy=False)


def square_a1_b1_product_columns(
    basis: Sequence[Sequence[int]], cells: Sequence[Sequence[int]]
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return all ``2^cells`` products of local square ``(A1,B1)`` singlets."""

    n_sites = len(basis[0])
    checked = validate_cell_partition(cells, n_sites=n_sites, cell_size=4)
    states = _basis_array(basis, n_sites=n_sites)
    local_indices = _cell_basis_indices(states, checked)
    carrier = square_plaquette_singlet_carrier()
    local_vectors = np.real_if_close(carrier.vectors, tol=1000)
    if np.iscomplexobj(local_vectors):
        raise RuntimeError("square carrier unexpectedly became complex")
    local_vectors = np.asarray(local_vectors, dtype=np.float64)
    labels = tuple(product((0, 1), repeat=len(checked)))
    label_array = np.asarray(labels, dtype=np.int64)
    columns = np.ones((len(states), len(labels)), dtype=np.float64)
    for cell_index in range(len(checked)):
        columns *= local_vectors[
            local_indices[:, cell_index, None],
            label_array[None, :, cell_index],
        ]
    gram_error = np.linalg.norm(columns.conj().T @ columns - np.eye(len(labels)))
    if gram_error > 3.0e-13:
        raise RuntimeError("square product carriers are not orthonormal")
    return columns, labels


def coherent_square_plaquette_product(
    columns: np.ndarray,
    labels: Sequence[Sequence[int]],
    angle: float,
) -> np.ndarray:
    """Return ``tensor_cells(cos(angle) A1 + sin(angle) B1)``.

    ``columns`` and ``labels`` must be the output of
    :func:`square_a1_b1_product_columns`.  The one-angle restriction is the
    size-transferable physical carrier; unlike an unrestricted Ritz solve in
    the full product-column span, its parameter count stays constant with the
    number of cells.
    """

    values = np.asarray(columns)
    checked = tuple(tuple(int(value) for value in row) for row in labels)
    if values.ndim != 2 or values.shape[1] != len(checked) or not checked:
        raise ValueError("columns and labels must be aligned and nonempty")
    n_cells = len(checked[0])
    if n_cells < 1 or any(
        len(row) != n_cells or any(value not in (0, 1) for value in row)
        for row in checked
    ):
        raise ValueError("labels must be equal-length binary local-channel rows")
    theta = float(angle)
    if not np.isfinite(theta):
        raise ValueError("angle must be finite")
    local = (np.cos(theta), np.sin(theta))
    coefficients = np.asarray(
        [np.prod([local[value] for value in row]) for row in checked],
        dtype=np.float64,
    )
    state = values @ coefficients
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise RuntimeError("coherent plaquette product vanished")
    return np.asarray(state / norm)


def _triangle_magnetic_carriers() -> np.ndarray:
    """Return triangle carrier amplitudes as ``[bits,m_bit,E_label]``."""

    tree = FusionTree.left_associated(3)
    minus, minus_states = tree_basis_matrix(tree, target_q=1, target_m=-1)
    plus, plus_states = tree_basis_matrix(tree, target_q=1, target_m=1)
    if tuple(state.labels for state in minus_states) != tuple(
        state.labels for state in plus_states
    ):
        raise RuntimeError("triangle magnetic partners have mismatched labels")
    registered = triangular_triangle_spin_half_carrier()
    if np.linalg.norm(plus - registered.vectors) > 3.0e-14:
        raise RuntimeError("triangle +1 magnetic slice changed convention")
    magnetic = np.stack((minus, plus), axis=1)
    gram = np.einsum("ime,imf->ef", magnetic.conj(), magnetic)
    if np.linalg.norm(gram - 2.0 * np.eye(2)) > 4.0e-13:
        raise RuntimeError("triangle E magnetic carrier normalization failed")
    magnetic = np.real_if_close(magnetic, tol=1000)
    if np.iscomplexobj(magnetic):
        raise RuntimeError("triangle magnetic carriers unexpectedly became complex")
    return np.asarray(magnetic, dtype=np.float64)


def triangular_e_singlet_product_columns(
    basis: Sequence[Sequence[int]], cells: Sequence[Sequence[int]]
) -> tuple[
    np.ndarray,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    """Fuse four local ``q=1,E`` triangles into total ``q=0`` columns.

    Every exact fusion multiplicity of the effective spin halves is retained
    for every local ``E``-label product.  Four triangles therefore give
    ``2 * 2^4 = 32`` columns, while six give ``5 * 2^6 = 320`` columns.
    """

    n_sites = len(basis[0])
    checked = validate_cell_partition(cells, n_sites=n_sites, cell_size=3)
    if len(checked) % 2 or len(checked) < 2:
        raise ValueError("a singlet requires a positive even number of triangles")
    states = _basis_array(basis, n_sites=n_sites)
    local_indices = _cell_basis_indices(states, checked)
    magnetic = _triangle_magnetic_carriers()
    n_cells = len(checked)
    local_up_counts = tuple(
        np.sum(states[:, np.asarray(cell, dtype=np.int64)], axis=1)
        for cell in checked
    )
    effective_bits = tuple(
        np.clip(counts - 1, 0, 1).astype(np.int64)
        for counts in local_up_counts
    )
    local = tuple(
        magnetic[
            local_indices[:, index],
            effective_bits[index],
            :,
        ]
        for index in range(n_cells)
    )
    effective, effective_states = tree_basis_matrix(
        FusionTree.left_associated(n_cells), target_q=0, target_m=0
    )
    multiplicity = effective.shape[1]
    effective = np.real_if_close(effective, tol=1000)
    if np.iscomplexobj(effective):
        raise RuntimeError("effective-spin CG basis unexpectedly became complex")
    effective = np.asarray(effective, dtype=np.float64)
    e_labels = tuple(product((0, 1), repeat=n_cells))
    column_labels = tuple(
        (labels, fusion_index)
        for labels in e_labels
        for fusion_index in range(multiplicity)
    )
    # Local total M fixes the effective magnetic bit uniquely: a q=1 triangle
    # has one up spin for m2=-1 and two for m2=+1.  Exploiting this selection
    # rule avoids a nominal 2**n_cells magnetic contraction for every E label.
    e_products = np.ones((len(states), 1), dtype=np.float64)
    for amplitudes in local:
        e_products = (
            e_products[:, :, None] * amplitudes[:, None, :]
        ).reshape(len(states), -1)
    weights = 1 << np.arange(n_cells - 1, -1, -1, dtype=np.int64)
    effective_indices = np.column_stack(effective_bits) @ weights
    fusion_amplitudes = effective[effective_indices]
    result = (
        e_products[:, :, None] * fusion_amplitudes[:, None, :]
    ).reshape(len(states), len(column_labels))
    # The construction is an exact tensor product of orthonormal local
    # carriers and an orthonormal effective-spin fusion basis.  Rechecking the
    # dense Gram is useful at N=12 but would be a needless O(dim*320^2) N=18
    # profile cost.
    if result.shape[1] <= 64:
        gram_error = np.linalg.norm(
            result.conj().T @ result - np.eye(result.shape[1])
        )
        if gram_error > 8.0e-13:
            raise RuntimeError("triangle product carriers are not orthonormal")
    fusion_labels = tuple(
        tuple(int(value) for value in state.labels)
        for state in effective_states
    )
    return result, e_labels, fusion_labels


def project_columns_by_index_table(
    columns: np.ndarray,
    permutation_indices: np.ndarray,
    *,
    characters: np.ndarray | Sequence[complex] | None = None,
) -> np.ndarray:
    """Apply a terminal character projector columnwise.

    The convention is exactly the physical-amplitude convention
    ``P_chi = |G|^-1 sum_g conj(chi[g]) T_g``.  ``characters=None`` retains
    the historical equal-character/Gamma behavior.  Character order must
    match the rows of ``permutation_indices``.
    """

    values = np.asarray(columns)
    if values.dtype.kind not in "fc":
        values = values.astype(np.float64)
    indices = np.asarray(permutation_indices, dtype=np.int64)
    if values.ndim != 2 or indices.ndim != 2:
        raise ValueError("columns and permutation_indices must be rank two")
    if indices.shape[1] != values.shape[0]:
        raise ValueError("projector index width and basis dimension disagree")
    if characters is None:
        weights = np.ones((indices.shape[0],), dtype=np.float64)
    else:
        weights = np.asarray(characters, dtype=np.complex128)
        if weights.shape != (indices.shape[0],):
            raise ValueError("one character is required per projector row")
        if not np.all(np.isfinite(weights)):
            raise ValueError("characters must be finite")
        if np.max(np.abs(np.abs(weights) - 1.0), initial=0.0) > 1.0e-10:
            raise ValueError("characters must have unit modulus")
    dtype = np.result_type(values.dtype, weights.dtype)
    projected = np.zeros(values.shape, dtype=dtype)
    for character, row in zip(weights, indices, strict=True):
        projected += np.conj(character) * values[row]
    return projected / indices.shape[0]


def generalized_ritz_subspace(
    columns: np.ndarray,
    hamiltonian: Any,
    *,
    relative_rank_tolerance: float = 1.0e-11,
    absolute_rank_tolerance: float = 1.0e-13,
    maximum_rank: int | None = None,
) -> GeneralizedRitzSubspace:
    """Rank reveal projected columns and solve only their small Ritz matrix.

    When ``maximum_rank`` is supplied, the retained directions are the largest
    overlap-eigenvalue modes.  This truncation is deterministic and
    Hamiltonian blind.
    """

    values = np.asarray(columns)
    if values.dtype.kind not in "fc":
        values = values.astype(np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("columns must be a nonempty rank-two matrix")
    if hamiltonian.shape != (values.shape[0], values.shape[0]):
        raise ValueError("Hamiltonian and column dimensions disagree")
    overlap = values.conj().T @ values
    overlap = (overlap + overlap.conj().T) / 2.0
    overlap_eigenvalues, overlap_vectors = np.linalg.eigh(overlap)
    largest = float(max(overlap_eigenvalues[-1], 0.0))
    threshold = max(absolute_rank_tolerance, relative_rank_tolerance * largest)
    retained = overlap_eigenvalues > threshold
    numerical_rank = int(np.count_nonzero(retained))
    if maximum_rank is not None:
        if maximum_rank < 1:
            raise ValueError("maximum_rank must be positive when supplied")
        if maximum_rank < numerical_rank:
            retained_indices = np.flatnonzero(retained)[-maximum_rank:]
            retained = np.zeros_like(retained, dtype=bool)
            retained[retained_indices] = True
    rank = int(np.count_nonzero(retained))
    if rank < 1:
        raise ValueError("projected columns have numerically zero rank")
    orthonormal = values @ (
        overlap_vectors[:, retained]
        / np.sqrt(overlap_eigenvalues[retained])[None, :]
    )
    orthonormality_error = float(
        np.linalg.norm(orthonormal.conj().T @ orthonormal - np.eye(rank))
    )
    action = hamiltonian @ orthonormal
    ritz_matrix = orthonormal.conj().T @ action
    ritz_matrix = (ritz_matrix + ritz_matrix.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(ritz_matrix)
    state = orthonormal @ eigenvectors[:, 0]
    state /= np.linalg.norm(state)
    h_state = hamiltonian @ state
    energy = float(np.real(np.vdot(state, h_state)))
    residual = h_state - energy * state
    return GeneralizedRitzSubspace(
        raw_column_count=values.shape[1],
        projected_rank=rank,
        overlap_eigenvalues=np.asarray(overlap_eigenvalues),
        retained_overlap_condition_number=float(
            overlap_eigenvalues[retained][-1]
            / overlap_eigenvalues[retained][0]
        ),
        orthonormal_basis=np.asarray(orthonormal),
        ritz_eigenvalues=np.asarray(eigenvalues),
        ground_state=np.asarray(state),
        ground_energy=energy,
        variance=float(np.real(np.vdot(residual, residual))),
        orthonormality_error=orthonormality_error,
        residual_norm=float(np.linalg.norm(residual)),
    )


def local_binary_channel_diagnostics(
    raw_columns: np.ndarray,
    state: np.ndarray,
    *,
    n_cells: int,
    spectator_dimension: int = 1,
) -> dict[str, object]:
    """Resolve local binary-channel occupations/coherences in one cell frame.

    The translation-projected state need not lie wholly in the unprojected
    partition bank.  ``raw_carrier_subspace_weight`` reports the retained
    component; local density matrices are conditional on that component.
    """

    columns = np.asarray(raw_columns)
    vector = np.asarray(state)
    expected = (2**n_cells) * spectator_dimension
    if columns.shape != (vector.size, expected):
        raise ValueError("raw bank shape disagrees with channel dimensions")
    coordinates = columns.conj().T @ vector
    weight = float(np.real(np.vdot(coordinates, coordinates)))
    if weight <= 1.0e-14:
        raise ValueError("state has no resolvable weight in raw carrier bank")
    tensor = (coordinates / np.sqrt(weight)).reshape(
        (2,) * n_cells + (spectator_dimension,)
    )
    cells = []
    for cell in range(n_cells):
        matrix = np.moveaxis(tensor, cell, 0).reshape(2, -1)
        density = matrix @ matrix.conj().T
        cells.append(
            {
                "channel_0_occupation": float(np.real(density[0, 0])),
                "channel_1_occupation": float(np.real(density[1, 1])),
                "pseudospin_x_coherence": float(2.0 * np.real(density[0, 1])),
                "pseudospin_y_coherence": float(-2.0 * np.imag(density[0, 1])),
                "pseudospin_z_polarization": float(
                    np.real(density[0, 0] - density[1, 1])
                ),
                "density_matrix": [
                    [
                        [float(value.real), float(value.imag)]
                        for value in row
                    ]
                    for row in density
                ],
            }
        )
    spectator = np.moveaxis(tensor, -1, 0).reshape(spectator_dimension, -1)
    spectator_density = spectator @ spectator.conj().T
    return {
        "raw_carrier_subspace_weight": weight,
        "cells": cells,
        "mean_channel_1_occupation": float(
            np.mean([row["channel_1_occupation"] for row in cells])
        ),
        "mean_absolute_pseudospin_x_coherence": float(
            np.mean([abs(row["pseudospin_x_coherence"]) for row in cells])
        ),
        "spectator_occupations": [
            float(value) for value in np.real(np.diag(spectator_density))
        ],
    }


def subspace_principal_singular_values(
    first: GeneralizedRitzSubspace, second: GeneralizedRitzSubspace
) -> np.ndarray:
    """Return cosines of principal angles between two projected subspaces."""

    return np.linalg.svd(
        first.orthonormal_basis.conj().T @ second.orthonormal_basis,
        compute_uv=False,
    )


__all__ = [
    "CellPartition",
    "GeneralizedRitzSubspace",
    "coherent_square_plaquette_product",
    "generalized_ritz_subspace",
    "local_binary_channel_diagnostics",
    "project_columns_by_index_table",
    "square_a1_b1_product_columns",
    "subspace_principal_singular_values",
    "triangular_e_singlet_product_columns",
    "validate_cell_partition",
]
