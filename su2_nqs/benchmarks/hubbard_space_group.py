"""Exact-small signed spatial projectors for Hubbard Fock amplitudes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import validated_permutation_group
from ..operators.hubbard_fock import apply_site_permutation


def inverse_site_permutation(image: Sequence[int]) -> tuple[int, ...]:
    """Invert an active ``old_site -> new_site`` permutation."""

    permutation = tuple(int(site) for site in image)
    if tuple(sorted(permutation)) != tuple(range(len(permutation))):
        raise ValueError("image must contain every site exactly once")
    inverse = [0] * len(permutation)
    for old, new in enumerate(permutation):
        inverse[new] = old
    return tuple(inverse)


def rectangular_translation_group(
    rows: int, columns: int
) -> tuple[tuple[int, ...], ...]:
    """Return all distinct translations of a rectangular periodic cell."""

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    images = []
    for dy in range(rows):
        for dx in range(columns):
            images.append(
                tuple(
                    ((row + dy) % rows) * columns
                    + ((column + dx) % columns)
                    for row in range(rows)
                    for column in range(columns)
                )
            )
    return validated_permutation_group(tuple(sorted(set(images))))


def fermionic_permutation_index_sign_tables(
    basis: Sequence[Sequence[int]],
    group_images: Iterable[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compile the physical coefficient action of site permutations.

    If ``U_g|n> = s_g(n)|g n>``, a coefficient transforms as
    ``(U_g psi)(n)=s_g(g^-1 n) psi(g^-1 n)``.  The returned arrays have shape
    ``[group, target_basis_index]`` and implement precisely this preimage
    convention.
    """

    checked_basis = tuple(tuple(int(token) for token in row) for row in basis)
    if not checked_basis:
        raise ValueError("basis must be nonempty")
    n_sites = len(checked_basis[0])
    positions = {state: index for index, state in enumerate(checked_basis)}
    if len(positions) != len(checked_basis):
        raise ValueError("basis must not contain duplicates")
    images = validated_permutation_group(group_images)
    if any(len(image) != n_sites for image in images):
        raise ValueError("group image size disagrees with basis")
    index_rows = []
    sign_rows = []
    for image in images:
        inverse = inverse_site_permutation(image)
        indices = []
        signs = []
        for target in checked_basis:
            source, sign = apply_site_permutation(target, inverse)
            try:
                indices.append(positions[source])
            except KeyError as error:
                raise ValueError("site permutation does not preserve basis") from error
            signs.append(sign)
        index_rows.append(indices)
        sign_rows.append(signs)
    index_table = np.asarray(index_rows, dtype=np.int32)
    sign_table = np.asarray(sign_rows, dtype=np.float64)
    index_table.setflags(write=False)
    sign_table.setflags(write=False)
    return index_table, sign_table


def fermionic_reynolds_project_state(
    amplitudes: jax.Array,
    permutation_indices: jax.Array | np.ndarray,
    permutation_signs: jax.Array | np.ndarray,
    *,
    characters: jax.Array | np.ndarray | None = None,
) -> jax.Array:
    """Apply a one-dimensional-character signed Reynolds projector."""

    vector = jnp.asarray(amplitudes)
    indices = jnp.asarray(permutation_indices, dtype=jnp.int32)
    signs = jnp.asarray(permutation_signs)
    if vector.ndim != 1 or indices.ndim != 2 or indices.shape[1] != vector.size:
        raise ValueError("amplitude and permutation-index shapes are inconsistent")
    if signs.shape != indices.shape:
        raise ValueError("one fermionic sign is required per gathered amplitude")
    if characters is None:
        weights = jnp.ones((indices.shape[0],), dtype=vector.dtype)
    else:
        weights = jnp.asarray(characters)
        if weights.shape != (indices.shape[0],):
            raise ValueError("one character is required per group image")
    return jnp.einsum(
        "g,gn,gn->n", jnp.conj(weights), signs, vector[indices]
    ) / indices.shape[0]


__all__ = [
    "fermionic_permutation_index_sign_tables",
    "fermionic_reynolds_project_state",
    "inverse_site_permutation",
    "rectangular_translation_group",
]
