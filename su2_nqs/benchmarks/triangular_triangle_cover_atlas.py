"""Elementary-triangle exact covers and translation-orbit diagnostics.

This module contains no Hamiltonian or optimizer logic.  It enumerates exact
covers of a rectangular periodic triangular lattice by elementary triangles,
then quotients those covers by physical translations.  The resulting finite
atlas is useful for testing whether a local-carrier result depends on one
arbitrarily selected cell tiling.

Coordinates follow :mod:`su2_nqs.lattices.triangular_j1j2`: the primitive
nearest-neighbour displacements are ``(1,0)``, ``(0,1)``, and ``(1,1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..lattices.spatial_symmetry import rectangular_translation_group


TrianglePartition = tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ElementaryTriangle:
    """One elementary triangle with a canonical orientation and anchor."""

    sites: tuple[int, int, int]
    orientation: str
    anchor_row: int
    anchor_column: int


@dataclass(frozen=True)
class TriangleCoverOrbit:
    """One translation-canonical exact-cover orbit."""

    representative: TrianglePartition
    orbit_size: int
    stabilizer_translations: tuple[tuple[int, int], ...]
    up_count: int
    down_count: int
    row_orientation_words: tuple[str, ...]


@dataclass(frozen=True)
class TranslationCharacterOrbitBasis:
    """Sparse orbit basis for one rectangular translation character."""

    momentum_indices: tuple[int, int]
    translation_eigenvalues: tuple[complex, complex]
    columns: object
    orbit_sizes: tuple[int, ...]
    orthonormality_error: float
    character_residual: float


def _site(row: int, column: int, *, rows: int, columns: int) -> int:
    return (row % rows) * columns + (column % columns)


def triangular_elementary_triangles(
    rows: int, columns: int
) -> tuple[ElementaryTriangle, ...]:
    """Return the two elementary-triangle orientations at every anchor.

    ``U(r,c)={(r,c),(r,c+1),(r+1,c+1)}`` and
    ``D(r,c)={(r,c),(r+1,c),(r+1,c+1)}``.  Very small periodic tori can alias
    triangles; aliased site sets are returned only once.
    """

    if rows < 2 or columns < 2:
        raise ValueError("an elementary-triangle torus needs both periods >= 2")
    output: list[ElementaryTriangle] = []
    seen: set[tuple[int, int, int]] = set()
    for row in range(rows):
        for column in range(columns):
            definitions = (
                (
                    "U",
                    (
                        _site(row, column, rows=rows, columns=columns),
                        _site(row, column + 1, rows=rows, columns=columns),
                        _site(row + 1, column + 1, rows=rows, columns=columns),
                    ),
                ),
                (
                    "D",
                    (
                        _site(row, column, rows=rows, columns=columns),
                        _site(row + 1, column, rows=rows, columns=columns),
                        _site(row + 1, column + 1, rows=rows, columns=columns),
                    ),
                ),
            )
            for orientation, sites in definitions:
                canonical = tuple(sorted(sites))
                if len(set(canonical)) != 3 or canonical in seen:
                    continue
                seen.add(canonical)
                output.append(
                    ElementaryTriangle(
                        sites=canonical,
                        orientation=orientation,
                        anchor_row=row,
                        anchor_column=column,
                    )
                )
    return tuple(output)


def enumerate_triangle_exact_covers(
    rows: int, columns: int
) -> tuple[TrianglePartition, ...]:
    """Enumerate every disjoint elementary-triangle exact cover.

    The exact-cover recursion always branches on the least-covered remaining
    site, making the enumeration deterministic without depending on an
    external solver.
    """

    n_sites = rows * columns
    if n_sites % 3:
        return ()
    triangles = triangular_elementary_triangles(rows, columns)
    site_sets = tuple(frozenset(item.sites) for item in triangles)
    by_site: list[list[int]] = [[] for _ in range(n_sites)]
    for index, cell in enumerate(site_sets):
        for site in cell:
            by_site[site].append(index)
    covers: set[TrianglePartition] = set()

    def visit(uncovered: frozenset[int], chosen: tuple[int, ...]) -> None:
        if not uncovered:
            covers.add(
                tuple(sorted(triangles[index].sites for index in chosen))
            )
            return
        site = min(
            uncovered,
            key=lambda value: sum(
                site_sets[index] <= uncovered for index in by_site[value]
            ),
        )
        for index in by_site[site]:
            cell = site_sets[index]
            if cell <= uncovered:
                visit(uncovered - cell, chosen + (index,))

    visit(frozenset(range(n_sites)), ())
    return tuple(sorted(covers))


def translate_triangle_partition(
    partition: Sequence[Sequence[int]], image: Sequence[int]
) -> TrianglePartition:
    """Apply one physical site permutation and canonicalize the cell order."""

    return tuple(
        sorted(tuple(sorted(int(image[site]) for site in cell)) for cell in partition)
    )


def canonical_triangle_partition(
    partition: Sequence[Sequence[int]],
    translations: Sequence[Sequence[int]],
) -> TrianglePartition:
    """Return the lexicographically least translation image of a cover."""

    return min(translate_triangle_partition(partition, image) for image in translations)


def triangular_triangle_cover_orbits(
    rows: int, columns: int
) -> tuple[TriangleCoverOrbit, ...]:
    """Return all exact covers modulo translations with orientation metadata."""

    triangles = triangular_elementary_triangles(rows, columns)
    metadata = {item.sites: item for item in triangles}
    translations = rectangular_translation_group(rows, columns)
    covers = enumerate_triangle_exact_covers(rows, columns)
    buckets: dict[TrianglePartition, set[TrianglePartition]] = {}
    for cover in covers:
        representative = canonical_triangle_partition(cover, translations)
        buckets.setdefault(representative, set()).add(cover)
    output = []
    for representative in sorted(buckets):
        images = {
            translate_triangle_partition(representative, image)
            for image in translations
        }
        stabilizer_images = tuple(
            image
            for image in translations
            if translate_triangle_partition(representative, image) == representative
        )
        # Recover the displacement from the active image of the origin.  The
        # validated group is lexicographically sorted, so its tuple index is
        # deliberately not interpreted as a displacement.
        stabilizers = tuple(divmod(image[0], columns) for image in stabilizer_images)
        rows_words = [["." for _ in range(columns)] for _ in range(rows)]
        up_count = 0
        for cell in representative:
            item = metadata[cell]
            rows_words[item.anchor_row][item.anchor_column] = item.orientation
            up_count += item.orientation == "U"
        output.append(
            TriangleCoverOrbit(
                representative=representative,
                orbit_size=len(images),
                stabilizer_translations=stabilizers,
                up_count=int(up_count),
                down_count=len(representative) - int(up_count),
                row_orientation_words=tuple("".join(word) for word in rows_words),
            )
        )
    if sum(item.orbit_size for item in output) != len(covers):
        raise RuntimeError("translation cover orbits do not partition the atlas")
    return tuple(output)


def rectangular_translation_displacements(
    translations: Sequence[Sequence[int]], *, rows: int, columns: int
) -> tuple[tuple[int, int], ...]:
    """Recover ``(dr,dc)`` labels from active translation images."""

    n_sites = rows * columns
    displacements = []
    for image in translations:
        checked = tuple(int(site) for site in image)
        if len(checked) != n_sites or tuple(sorted(checked)) != tuple(range(n_sites)):
            raise ValueError("translations must be site permutations of the torus")
        dr, dc = divmod(checked[0], columns)
        expected = tuple(
            _site(row + dr, column + dc, rows=rows, columns=columns)
            for row in range(rows)
            for column in range(columns)
        )
        if checked != expected:
            raise ValueError("group image is not a rectangular translation")
        displacements.append((dr, dc))
    if len(set(displacements)) != rows * columns:
        raise ValueError("translations must contain every torus displacement once")
    return tuple(displacements)


def translation_character_values(
    translations: Sequence[Sequence[int]],
    *,
    rows: int,
    columns: int,
    momentum_indices: tuple[int, int],
) -> np.ndarray:
    """Return eigencharacters for ``T_(dr,dc)|k>=chi_k(dr,dc)|k>``."""

    kr = int(momentum_indices[0]) % rows
    kc = int(momentum_indices[1]) % columns
    displacements = rectangular_translation_displacements(
        translations, rows=rows, columns=columns
    )
    return np.asarray(
        [
            np.exp(-2.0j * np.pi * (kr * dr / rows + kc * dc / columns))
            for dr, dc in displacements
        ],
        dtype=np.complex128,
    )


def project_columns_by_translation_character(
    columns_to_project: np.ndarray,
    permutation_indices: np.ndarray,
    translations: Sequence[Sequence[int]],
    *,
    rows: int,
    columns: int,
    momentum_indices: tuple[int, int],
) -> np.ndarray:
    """Project dense columns into one rectangular momentum sector.

    ``permutation_indices[g,target]`` is the physical pullback compiled for
    ``translations[g]``.  The conjugated character therefore matches the
    repository Reynolds-projector convention.
    """

    values = np.asarray(columns_to_project)
    indices = np.asarray(permutation_indices, dtype=np.int64)
    if values.ndim != 2 or indices.shape != (len(translations), values.shape[0]):
        raise ValueError("column/projector dimensions disagree")
    characters = translation_character_values(
        translations,
        rows=rows,
        columns=columns,
        momentum_indices=momentum_indices,
    )
    projected = np.zeros(values.shape, dtype=np.complex128)
    for character, row in zip(characters, indices, strict=True):
        projected += np.conj(character) * values[row]
    return projected / len(translations)


def translation_character_orbit_basis(
    permutation_indices: np.ndarray,
    translations: Sequence[Sequence[int]],
    *,
    rows: int,
    columns: int,
    momentum_indices: tuple[int, int],
    zero_tolerance: float = 1.0e-13,
) -> TranslationCharacterOrbitBasis:
    """Build a sparse exact orbit basis in one translation character.

    Projected columns from distinct computational-basis translation orbits
    have disjoint support.  One normalized nonzero column per compatible orbit
    is therefore already an orthonormal basis; no dense projector is formed.
    """

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover - core tests install scipy
        raise RuntimeError("translation orbit bases require scipy") from error
    indices = np.asarray(permutation_indices, dtype=np.int64)
    dimension = indices.shape[1] if indices.ndim == 2 else -1
    if indices.shape != (len(translations), dimension) or dimension < 1:
        raise ValueError("permutation_indices must have shape (group,dimension)")
    if zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be positive")
    characters = translation_character_values(
        translations,
        rows=rows,
        columns=columns,
        momentum_indices=momentum_indices,
    )
    inverse = np.empty_like(indices)
    target = np.arange(dimension, dtype=np.int64)
    for group_index, row in enumerate(indices):
        if tuple(sorted(int(value) for value in row)) != tuple(range(dimension)):
            raise ValueError("each pullback index row must be a permutation")
        inverse[group_index, row] = target
    visited = np.zeros(dimension, dtype=bool)
    output_rows: list[int] = []
    output_columns: list[int] = []
    output_values: list[complex] = []
    orbit_sizes: list[int] = []
    for seed in range(dimension):
        if visited[seed]:
            continue
        orbit = np.unique(inverse[:, seed])
        visited[orbit] = True
        accumulated: dict[int, complex] = {}
        for character, destination in zip(characters, inverse[:, seed], strict=True):
            key = int(destination)
            accumulated[key] = accumulated.get(key, 0.0j) + np.conj(character)
        norm = float(np.sqrt(sum(abs(value) ** 2 for value in accumulated.values())))
        if norm <= zero_tolerance:
            continue
        column = len(orbit_sizes)
        orbit_sizes.append(len(orbit))
        for row, value in accumulated.items():
            normalized = value / norm
            if abs(normalized) > zero_tolerance:
                output_rows.append(row)
                output_columns.append(column)
                output_values.append(normalized)
    basis = sparse.coo_matrix(
        (output_values, (output_rows, output_columns)),
        shape=(dimension, len(orbit_sizes)),
        dtype=np.complex128,
    ).tocsr()
    gram = (basis.conj().T @ basis).toarray()
    orthonormality_error = float(
        np.max(np.abs(gram - np.eye(len(orbit_sizes))), initial=0.0)
    )
    displacements = rectangular_translation_displacements(
        translations, rows=rows, columns=columns
    )
    residual = 0.0
    for generator, expected in (
        ((1 % rows, 0), np.exp(-2.0j * np.pi * momentum_indices[0] / rows)),
        ((0, 1 % columns), np.exp(-2.0j * np.pi * momentum_indices[1] / columns)),
    ):
        group_index = displacements.index(generator)
        difference = basis[indices[group_index]] - expected * basis
        residual = max(
            residual, float(np.max(np.abs(difference.data), initial=0.0))
        )
    return TranslationCharacterOrbitBasis(
        momentum_indices=(
            int(momentum_indices[0]) % rows,
            int(momentum_indices[1]) % columns,
        ),
        translation_eigenvalues=(
            complex(np.exp(-2.0j * np.pi * momentum_indices[0] / rows)),
            complex(np.exp(-2.0j * np.pi * momentum_indices[1] / columns)),
        ),
        columns=basis,
        orbit_sizes=tuple(orbit_sizes),
        orthonormality_error=orthonormality_error,
        character_residual=residual,
    )


__all__ = [
    "ElementaryTriangle",
    "TriangleCoverOrbit",
    "TrianglePartition",
    "TranslationCharacterOrbitBasis",
    "canonical_triangle_partition",
    "enumerate_triangle_exact_covers",
    "project_columns_by_translation_character",
    "rectangular_translation_displacements",
    "translate_triangle_partition",
    "translation_character_orbit_basis",
    "translation_character_values",
    "triangular_elementary_triangles",
    "triangular_triangle_cover_orbits",
]
