"""Exact-N16 gate for a Krylov-anchored full SU(2) Spatial-ViT.

The comparison deliberately changes the state manifold rather than appending
another low-rank response head.  The exact-SU(2) arm generates all 1430
balanced-tree singlet coefficients with :class:`ManyCellSchurViT`; its
fixed-``Sz`` control uses the parameter-identical autoregressive Spatial-ViT
on physical bits.  Both neural arms are coherently anchored to the same
two-Lanczos state and lattice ``Gamma-A1`` is imposed before every energy
evaluation.

This module contains only exact-small numerical helpers.  It consumes the
sealed N=16 representation produced by the existing benchmark and does not
diagonalize a new many-body Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.fixed_sz_autoregressive_spatial_vit import (
    FixedSzAutoregressiveSpatialViTConfig,
    fixed_sz_autoregressive_spatial_log_amplitudes,
    init_fixed_sz_autoregressive_spatial_vit,
)
from ..models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from ..su2.many_cell_schur import (
    ManyCellSchurLayout,
    rectangular_four_spin_layout,
)


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class A1OrbitReduction:
    """Sparse orbit-sum coordinates for a trivial spatial irrep."""

    orbit_id: np.ndarray
    orbit_sizes: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.orbit_id)
        sizes = np.asarray(self.orbit_sizes)
        if ids.ndim != 1 or sizes.ndim != 1 or len(sizes) == 0:
            raise ValueError("orbit data must be nonempty one-dimensional arrays")
        if ids.dtype.kind not in "iu" or sizes.dtype.kind not in "iu":
            raise TypeError("orbit labels and sizes must be integers")
        if np.min(ids) < 0 or np.max(ids) >= len(sizes):
            raise ValueError("orbit labels are outside the orbit table")
        observed = np.bincount(ids, minlength=len(sizes))
        if not np.array_equal(observed, sizes):
            raise ValueError("orbit sizes do not match the membership labels")

    @property
    def physical_dimension(self) -> int:
        return int(len(self.orbit_id))

    @property
    def reduced_dimension(self) -> int:
        return int(len(self.orbit_sizes))


def a1_orbit_reduction(permutation_indices: np.ndarray) -> A1OrbitReduction:
    """Build exact orbit-sum coordinates from physical permutation indices."""

    images = np.asarray(permutation_indices, dtype=np.int64)
    if images.ndim != 2 or images.shape[0] == 0 or images.shape[1] == 0:
        raise ValueError("permutation_indices must have shape (group,dimension)")
    dimension = images.shape[1]
    expected = np.arange(dimension)
    if any(not np.array_equal(np.sort(row), expected) for row in images):
        raise ValueError("each spatial action must be a complete permutation")
    parent = np.arange(dimension, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            if left < right:
                parent[right] = left
            else:
                parent[left] = right

    for row in images:
        for old, new in enumerate(row):
            union(old, int(new))
    roots = np.asarray([find(index) for index in range(dimension)])
    unique = np.unique(roots)
    lookup = {int(root): index for index, root in enumerate(unique)}
    orbit_id = np.asarray([lookup[int(root)] for root in roots], dtype=np.int32)
    sizes = np.bincount(orbit_id, minlength=len(unique)).astype(np.int32)
    orbit_id.setflags(write=False)
    sizes.setflags(write=False)
    return A1OrbitReduction(orbit_id=orbit_id, orbit_sizes=sizes)


def a1_orbit_coordinates(
    amplitudes: jax.Array,
    reduction: A1OrbitReduction,
) -> jax.Array:
    """Return coefficients in the normalized orbit-sum basis."""

    values = jnp.asarray(amplitudes)
    if values.shape != (reduction.physical_dimension,):
        raise ValueError("amplitude dimension and orbit reduction disagree")
    ids = jnp.asarray(reduction.orbit_id, dtype=jnp.int32)
    sizes = jnp.asarray(reduction.orbit_sizes, dtype=jnp.float64)
    sums = jnp.zeros((reduction.reduced_dimension,), dtype=values.dtype)
    sums = sums.at[ids].add(values)
    return sums / jnp.sqrt(sizes)


def a1_orbit_hamiltonian(
    rows: Sequence[int],
    columns: Sequence[int],
    values: Sequence[complex],
    reduction: A1OrbitReduction,
) -> np.ndarray:
    """Reduce a physical-basis COO Hamiltonian to normalized orbit sums."""

    row = np.asarray(rows, dtype=np.int64)
    column = np.asarray(columns, dtype=np.int64)
    elements = np.asarray(values)
    if row.shape != column.shape or row.shape != elements.shape:
        raise ValueError("COO rows, columns, and values must have equal shapes")
    if len(row) and (
        np.min(row) < 0
        or np.min(column) < 0
        or np.max(row) >= reduction.physical_dimension
        or np.max(column) >= reduction.physical_dimension
    ):
        raise ValueError("COO index lies outside the physical basis")
    orbit_row = reduction.orbit_id[row]
    orbit_column = reduction.orbit_id[column]
    normalization = np.sqrt(
        reduction.orbit_sizes[orbit_row] * reduction.orbit_sizes[orbit_column]
    )
    result = np.zeros(
        (reduction.reduced_dimension, reduction.reduced_dimension),
        dtype=np.result_type(elements, np.complex128),
    )
    np.add.at(result, (orbit_row, orbit_column), elements / normalization)
    result = 0.5 * (result + result.conj().T)
    if np.max(np.abs(result.imag), initial=0.0) < 2.0e-14:
        result = np.asarray(result.real, dtype=np.float64)
    return result


def anchored_unit_residual_state(
    carrier: jax.Array,
    residual: jax.Array,
    residual_scale: jax.Array | float,
) -> jax.Array:
    """Add a normalized carrier-orthogonal residual to a strong anchor."""

    anchor = jnp.asarray(carrier)
    candidate = jnp.asarray(residual)
    scale = jnp.asarray(residual_scale)
    if anchor.ndim != 1 or candidate.shape != anchor.shape:
        raise ValueError("carrier and residual must be equal vectors")
    if scale.ndim != 0:
        raise ValueError("residual_scale must be scalar")
    anchor = anchor / jnp.linalg.norm(anchor)
    orthogonal = candidate - anchor * jnp.vdot(anchor, candidate)
    norm = jnp.linalg.norm(orthogonal)
    unit = orthogonal / jnp.maximum(norm, jnp.asarray(1.0e-14))
    return anchor + scale * unit


def lowest_krylov_state(
    carrier: np.ndarray,
    hamiltonian: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, int]:
    """Return the lowest Ritz state in ``span{psi,...,H^degree psi}``."""

    if degree < 0:
        raise ValueError("degree must be nonnegative")
    matrix = np.asarray(hamiltonian)
    state = np.asarray(carrier, dtype=np.result_type(matrix, np.complex128))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if state.shape != (matrix.shape[0],):
        raise ValueError("carrier and Hamiltonian dimensions disagree")
    state = state / np.linalg.norm(state)
    columns = [state]
    for _ in range(degree):
        columns.append(matrix @ columns[-1])
    stacked = np.column_stack(columns)
    left, singular, _ = np.linalg.svd(stacked, full_matrices=False)
    threshold = singular[0] * 1.0e-12 if len(singular) else 0.0
    rank = int(np.count_nonzero(singular > threshold))
    basis = left[:, :rank]
    reduced = basis.conj().T @ matrix @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
    result = basis @ eigenvectors[:, int(np.argmin(eigenvalues))]
    result /= np.linalg.norm(result)
    return result, rank


def scrambled_coordinate_layout(
    layout: ManyCellSchurLayout,
    permutation: Sequence[int],
) -> ManyCellSchurLayout:
    """Relabel spatial coordinates without changing the SU(2) fusion tree."""

    order = tuple(int(value) for value in permutation)
    if tuple(sorted(order)) != tuple(range(layout.n_sites)):
        raise ValueError("permutation must contain every model site exactly once")
    return ManyCellSchurLayout(
        tree=layout.tree,
        site_order=layout.site_order,
        physical_cells=layout.physical_cells,
        site_coordinates=tuple(layout.site_coordinates[index] for index in order),
        cell_root_indices=layout.cell_root_indices,
        cell_pair_indices=layout.cell_pair_indices,
        intercell_indices=layout.intercell_indices,
    )


def matched_backbone_initializations(
    seed: int,
    *,
    embedding_dim: int,
    head_count: int,
    depth: int,
    mlp_ratio: int,
    phase_scale: float,
    initialization_scale: float,
    use_distance_envelope: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    ManyCellSchurViTConfig,
    FixedSzAutoregressiveSpatialViTConfig,
]:
    """Initialize exact-SU2 and fixed-Sz arms with bitwise-identical arrays."""

    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    shared = dict(
        embedding_dim=int(embedding_dim),
        head_count=int(head_count),
        depth=int(depth),
        mlp_ratio=int(mlp_ratio),
        phase_scale=float(phase_scale),
        initialization_scale=float(initialization_scale),
        use_distance_envelope=bool(use_distance_envelope),
    )
    su2_config = ManyCellSchurViTConfig(
        layout,
        target_q=0,
        geometry="triangular",
        **shared,
    )
    fixed_config = FixedSzAutoregressiveSpatialViTConfig(
        "triangular",
        4,
        4,
        target_m2=0,
        **shared,
    )
    key = jax.random.PRNGKey(int(seed))
    su2 = init_many_cell_schur_vit(key, su2_config)
    fixed = init_fixed_sz_autoregressive_spatial_vit(key, fixed_config)
    left = jax.tree_util.tree_leaves(su2)
    right = jax.tree_util.tree_leaves(fixed)
    if len(left) != len(right) or any(
        first.shape != second.shape or not np.array_equal(first, second)
        for first, second in zip(left, right, strict=True)
    ):
        raise RuntimeError("matched SU2/fixed-Sz initialization contract failed")
    return su2, fixed, su2_config, fixed_config


def fixed_sz_projected_residual(
    params: object,
    configurations: jax.Array,
    config: FixedSzAutoregressiveSpatialViTConfig,
    reduction: A1OrbitReduction,
) -> jax.Array:
    """Evaluate a fixed-Sz ViT and project it exactly to lattice A1."""

    logs = fixed_sz_autoregressive_spatial_log_amplitudes(
        params, configurations, config
    )
    return a1_orbit_coordinates(jnp.exp(logs), reduction)


def su2_projected_residual(
    params: object,
    labels: jax.Array,
    config: ManyCellSchurViTConfig,
    balanced_a1_basis: jax.Array,
) -> jax.Array:
    """Evaluate the complete singlet ViT and project coefficients to A1."""

    logs = many_cell_schur_log_amplitudes(params, labels, config)
    coefficients = jnp.exp(logs)
    basis = jnp.asarray(balanced_a1_basis)
    if basis.ndim != 2 or basis.shape[0] != coefficients.shape[0]:
        raise ValueError("balanced A1 basis and complete singlet labels disagree")
    return basis.conj().T @ coefficients


__all__ = [
    "A1OrbitReduction",
    "a1_orbit_coordinates",
    "a1_orbit_hamiltonian",
    "a1_orbit_reduction",
    "anchored_unit_residual_state",
    "fixed_sz_projected_residual",
    "lowest_krylov_state",
    "matched_backbone_initializations",
    "scrambled_coordinate_layout",
    "su2_projected_residual",
]
