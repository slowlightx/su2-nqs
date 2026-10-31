from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.coherent_plaquette_frames import (
    coherent_plaquette_frame_columns,
    distinct_partition_orbit,
    nongeometric_site_permutation,
    scrambled_coherent_frame_columns,
)
from su2_nqs.benchmarks.four_cell_schur_frontier import SQUARE_N16_SUPPORTS
from su2_nqs.benchmarks.local_carrier_product_subspace import (
    coherent_square_plaquette_product,
    project_columns_by_index_table,
    square_a1_b1_product_columns,
)
from su2_nqs.lattices.spatial_symmetry import (
    square_gamma_a1_group,
    rectangular_translation_group,
)


def test_square_frame_orbit_and_explicit_projector_agree() -> None:
    basis = fixed_m_spin_basis(16, 0)
    group = square_gamma_a1_group(4)
    coherent = coherent_plaquette_frame_columns(
        basis, SQUARE_N16_SUPPORTS, group
    )
    assert coherent.distinct_partition_count == 4
    assert len(
        distinct_partition_orbit(
            SQUARE_N16_SUPPORTS, rectangular_translation_group(4, 4)
        )
    ) == 4
    primary, labels = square_a1_b1_product_columns(basis, SQUARE_N16_SUPPORTS)
    angle = 0.137
    raw = coherent_square_plaquette_product(primary, labels, angle)
    from_columns = coherent_square_plaquette_product(
        coherent.columns, coherent.labels, angle
    )
    indices = physical_permutation_index_table(basis, group)
    projected = project_columns_by_index_table(raw[:, None], indices)[:, 0]
    projected /= np.linalg.norm(projected)
    if np.vdot(projected, from_columns).real < 0.0:
        projected *= -1.0
    np.testing.assert_allclose(from_columns, projected, atol=2.0e-14)


def test_scramble_preserves_column_gram_before_terminal_projection() -> None:
    basis = fixed_m_spin_basis(16, 0)
    group = square_gamma_a1_group(4)
    coherent = coherent_plaquette_frame_columns(
        basis, SQUARE_N16_SUPPORTS, group
    )
    permutation = nongeometric_site_permutation(
        n_sites=16, group=group, seed=2026092501
    )
    assert permutation not in group
    row = physical_permutation_index_table(basis, (permutation,))[0]
    moved = coherent.columns[row]
    np.testing.assert_allclose(
        moved.T @ moved, coherent.columns.T @ coherent.columns, atol=2.0e-13
    )
    scrambled = scrambled_coherent_frame_columns(
        coherent, basis, group, permutation
    )
    assert scrambled.shape == coherent.columns.shape
    assert np.all(np.isfinite(scrambled))
