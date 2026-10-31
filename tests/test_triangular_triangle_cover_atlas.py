from su2_nqs.benchmarks.triangular_triangle_cover_atlas import (
    canonical_triangle_partition,
    enumerate_triangle_exact_covers,
    project_columns_by_translation_character,
    translation_character_orbit_basis,
    triangular_elementary_triangles,
    triangular_triangle_cover_orbits,
)
from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
)
from su2_nqs.lattices.spatial_symmetry import rectangular_translation_group

import numpy as np


def test_triangular_3x6_elementary_exact_cover_atlas_is_complete() -> None:
    triangles = triangular_elementary_triangles(3, 6)
    covers = enumerate_triangle_exact_covers(3, 6)
    orbits = triangular_triangle_cover_orbits(3, 6)
    assert len(triangles) == 36
    assert {item.orientation for item in triangles} == {"U", "D"}
    assert len(covers) == 60
    assert len(orbits) == 7
    assert sorted(item.orbit_size for item in orbits) == [3, 3, 6, 6, 6, 18, 18]
    assert sorted(item.up_count for item in orbits) == [0, 3, 3, 3, 3, 3, 6]
    assert sum(item.orbit_size for item in orbits) == len(covers)


def test_cover_representatives_are_translation_canonical_exact_partitions() -> None:
    translations = rectangular_translation_group(3, 6)
    for orbit in triangular_triangle_cover_orbits(3, 6):
        flattened = tuple(site for cell in orbit.representative for site in cell)
        assert sorted(flattened) == list(range(18))
        assert canonical_triangle_partition(
            orbit.representative, translations
        ) == orbit.representative
        assert orbit.orbit_size * len(orbit.stabilizer_translations) == 18
        assert len(orbit.row_orientation_words) == 3
        assert all(len(word) == 6 for word in orbit.row_orientation_words)


def test_translation_character_orbit_bases_resolve_identity() -> None:
    rows, columns = 2, 3
    basis = fixed_m_spin_basis(rows * columns, 0)
    translations = rectangular_translation_group(rows, columns)
    indices = physical_permutation_index_table(basis, translations)
    sectors = [
        translation_character_orbit_basis(
            indices,
            translations,
            rows=rows,
            columns=columns,
            momentum_indices=(kr, kc),
        )
        for kr in range(rows)
        for kc in range(columns)
    ]
    assert sum(item.columns.shape[1] for item in sectors) == len(basis)
    assert max(item.orthonormality_error for item in sectors) < 3.0e-14
    assert max(item.character_residual for item in sectors) < 3.0e-14
    cross = sectors[0].columns.conj().T @ sectors[1].columns
    assert np.max(np.abs(cross.toarray()), initial=0.0) < 3.0e-14


def test_dense_character_projection_matches_sparse_orbit_range() -> None:
    rows, columns = 2, 3
    basis = fixed_m_spin_basis(rows * columns, 0)
    translations = rectangular_translation_group(rows, columns)
    indices = physical_permutation_index_table(basis, translations)
    seed = np.arange(2 * len(basis), dtype=np.float64).reshape(len(basis), 2)
    momentum = (1, 2)
    projected = project_columns_by_translation_character(
        seed,
        indices,
        translations,
        rows=rows,
        columns=columns,
        momentum_indices=momentum,
    )
    sector = translation_character_orbit_basis(
        indices,
        translations,
        rows=rows,
        columns=columns,
        momentum_indices=momentum,
    )
    residual = projected - sector.columns @ (sector.columns.conj().T @ projected)
    assert np.linalg.norm(residual) < 2.0e-13
