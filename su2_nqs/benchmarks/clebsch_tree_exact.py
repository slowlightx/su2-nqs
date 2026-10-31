"""Exact-support utilities for physical-input ClebschTree benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..operators.sz_heisenberg import SzHeisenbergGraph
from ..lattices.spatial_symmetry import validate_site_permutation
from ..su2.sz_basis import enumerate_sz_configs


def fixed_m_spin_basis(
    n_sites: int, total_m2: int
) -> tuple[tuple[int, ...], ...]:
    """Return the deterministic physical-spin basis for doubled ``M``."""

    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    if abs(total_m2) > n_sites or (n_sites + total_m2) % 2:
        raise ValueError("total_m2 is unreachable for n_sites spin halves")
    return enumerate_sz_configs(n_sites, (n_sites + total_m2) // 2)


def physical_permutation_index_table(
    basis: Sequence[Sequence[int]],
    group_images: Iterable[Sequence[int]],
) -> np.ndarray:
    """Compile pullbacks into ``[group, target_basis_index]`` indices."""

    checked_basis = tuple(tuple(int(bit) for bit in row) for row in basis)
    if not checked_basis:
        raise ValueError("basis must be nonempty")
    n_sites = len(checked_basis[0])
    if any(
        len(row) != n_sites or any(bit not in (0, 1) for bit in row)
        for row in checked_basis
    ):
        raise ValueError("basis must contain equal-length binary rows")
    positions = {row: index for index, row in enumerate(checked_basis)}
    if len(positions) != len(checked_basis):
        raise ValueError("basis must not contain duplicates")
    rows = []
    for image in group_images:
        checked_image = validate_site_permutation(image, n_sites=n_sites)
        source = [
            positions[
                tuple(
                    configuration[checked_image[old]]
                    for old in range(n_sites)
                )
            ]
            for configuration in checked_basis
        ]
        rows.append(source)
    if not rows:
        raise ValueError("group_images must be nonempty")
    result = np.asarray(rows, dtype=np.int32)
    result.setflags(write=False)
    return result


def reynolds_project_state(
    amplitudes: jax.Array,
    permutation_indices: jax.Array | np.ndarray,
    *,
    characters: jax.Array | np.ndarray | None = None,
) -> jax.Array:
    """Project one physical-basis vector using precompiled group gathers."""

    vector = jnp.asarray(amplitudes)
    indices = jnp.asarray(permutation_indices, dtype=jnp.int32)
    if vector.ndim != 1 or indices.ndim != 2 or indices.shape[1] != vector.size:
        raise ValueError("amplitude and permutation-index shapes are inconsistent")
    if characters is None:
        weights = jnp.ones((indices.shape[0],), dtype=vector.dtype)
    else:
        weights = jnp.asarray(characters)
        if weights.shape != (indices.shape[0],):
            raise ValueError("one character is required per group image")
    return jnp.einsum("g,gn->n", jnp.conj(weights), vector[indices]) / indices.shape[0]


def graph_hamiltonian_coo(
    n_sites: int,
    total_m2: int,
    bonds: Sequence[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return exact fixed-``M`` Heisenberg COO data in basis order."""

    basis = fixed_m_spin_basis(n_sites, total_m2)
    positions = {configuration: index for index, configuration in enumerate(basis)}
    graph = SzHeisenbergGraph(
        n_sites,
        tuple((int(i), int(j), float(value)) for i, j, value in bonds),
        n_up=(n_sites + total_m2) // 2,
    )
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for column, configuration in enumerate(basis):
        connected, elements = graph.get_conn(configuration)
        for candidate, element in zip(connected, elements, strict=True):
            rows.append(positions[candidate])
            columns.append(column)
            values.append(float(element))
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
        basis,
    )


def coo_matvec(
    rows: jax.Array | np.ndarray,
    columns: jax.Array | np.ndarray,
    values: jax.Array | np.ndarray,
    vector: jax.Array,
) -> jax.Array:
    """Differentiable COO matrix-vector product with repeated-row addition."""

    row_array = jnp.asarray(rows, dtype=jnp.int32)
    column_array = jnp.asarray(columns, dtype=jnp.int32)
    value_array = jnp.asarray(values)
    state = jnp.asarray(vector)
    if row_array.shape != column_array.shape or row_array.shape != value_array.shape:
        raise ValueError("COO arrays must have identical shapes")
    if state.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    return jnp.zeros_like(state).at[row_array].add(
        value_array * state[column_array]
    )


__all__ = [
    "coo_matvec",
    "fixed_m_spin_basis",
    "graph_hamiltonian_coo",
    "physical_permutation_index_table",
    "reynolds_project_state",
]
