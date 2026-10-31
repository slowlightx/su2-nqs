from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.local_carrier_product_subspace import (
    coherent_square_plaquette_product,
    generalized_ritz_subspace,
    local_binary_channel_diagnostics,
    project_columns_by_index_table,
    square_a1_b1_product_columns,
    triangular_e_singlet_product_columns,
    validate_cell_partition,
)
from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_characters,
    rectangular_translation_group,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitudes,
)


jax.config.update("jax_enable_x64", True)


SQUARE_CELLS = (
    (0, 1, 4, 5),
    (2, 3, 6, 7),
    (8, 9, 12, 13),
    (10, 11, 14, 15),
)
TRIANGLE_CELLS = (
    (0, 1, 5),
    (2, 3, 7),
    (4, 8, 9),
    (6, 10, 11),
)


def test_square_A1_B1_products_are_exact_orthonormal_singlets() -> None:
    basis = fixed_m_spin_basis(16, 0)
    columns, labels = square_a1_b1_product_columns(basis, SQUARE_CELLS)
    assert columns.shape == (12870, 16)
    assert columns.dtype == np.float64
    assert labels[0] == (0, 0, 0, 0)
    assert labels[-1] == (1, 1, 1, 1)
    np.testing.assert_allclose(columns.T @ columns, np.eye(16), atol=3.0e-14)
    spin_squared, spin_basis = spin_squared_sz_sparse(16, n_up=8)
    assert tuple(spin_basis) == basis
    assert np.linalg.norm(spin_squared @ columns) < 3.0e-13


def test_coherent_plaquette_product_has_size_independent_one_angle_form() -> None:
    basis = fixed_m_spin_basis(16, 0)
    columns, labels = square_a1_b1_product_columns(basis, SQUARE_CELLS)
    np.testing.assert_allclose(
        coherent_square_plaquette_product(columns, labels, 0.0),
        columns[:, 0],
        atol=2.0e-15,
    )
    angle = 0.37
    state = coherent_square_plaquette_product(columns, labels, angle)
    coefficients = columns.T @ state
    expected = np.asarray(
        [
            np.prod(
                [np.cos(angle) if value == 0 else np.sin(angle) for value in row]
            )
            for row in labels
        ]
    )
    np.testing.assert_allclose(coefficients, expected, atol=2.0e-15)
    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_triangle_E_products_keep_all_effective_singlet_multiplicities() -> None:
    basis = fixed_m_spin_basis(12, 0)
    columns, e_labels, fusion_labels = triangular_e_singlet_product_columns(
        basis, TRIANGLE_CELLS
    )
    assert columns.shape == (924, 32)
    assert columns.dtype == np.float64
    assert len(e_labels) == 16
    assert len(fusion_labels) == 2
    np.testing.assert_allclose(columns.T @ columns, np.eye(32), atol=3.0e-14)
    spin_squared, spin_basis = spin_squared_sz_sparse(12, n_up=6)
    assert tuple(spin_basis) == basis
    assert np.linalg.norm(spin_squared @ columns) < 3.0e-13


def test_terminal_projection_and_H_blind_rank_truncation() -> None:
    basis = fixed_m_spin_basis(12, 0)
    columns, _, _ = triangular_e_singlet_product_columns(
        basis, TRIANGLE_CELLS
    )
    indices = physical_permutation_index_table(
        basis, rectangular_translation_group(3, 4)
    )
    projected = project_columns_by_index_table(columns, indices)
    identity_h = sparse.identity(len(basis), format="csr")
    full = generalized_ritz_subspace(projected, identity_h)
    truncated = generalized_ritz_subspace(
        projected, identity_h, maximum_rank=5
    )
    assert full.projected_rank == 14
    assert truncated.projected_rank == 5
    assert full.ground_energy == pytest.approx(1.0, abs=2.0e-14)
    assert truncated.ground_energy == pytest.approx(1.0, abs=2.0e-14)
    assert full.orthonormality_error < 2.0e-13
    with pytest.raises(ValueError, match="positive"):
        generalized_ritz_subspace(projected, identity_h, maximum_rank=0)


def test_character_weighted_columns_match_physical_projector_convention() -> None:
    basis = fixed_m_spin_basis(3, 1)
    group = rectangular_translation_group(1, 3)
    indices = physical_permutation_index_table(basis, group)
    characters = np.asarray(rectangular_translation_characters(1, 3, 0, 1))
    columns = np.asarray(
        (
            (1.0 + 0.2j, -0.3 + 0.7j),
            (0.4 - 0.5j, 1.1 + 0.1j),
            (-0.8 + 0.3j, 0.2 - 0.9j),
        )
    )
    projected = project_columns_by_index_table(
        columns, indices, characters=characters
    )
    expected = sum(
        np.conj(character) * columns[row]
        for character, row in zip(characters, indices, strict=True)
    ) / len(group)
    np.testing.assert_allclose(projected, expected, atol=2.0e-15)
    np.testing.assert_allclose(
        project_columns_by_index_table(
            projected, indices, characters=characters
        ),
        projected,
        atol=3.0e-15,
    )
    for character, row in zip(characters, indices, strict=True):
        np.testing.assert_allclose(
            projected[row], character * projected, atol=3.0e-15
        )

    projector = PhysicalSpatialProjector(group, tuple(characters))
    powers = jnp.asarray((4, 2, 1), dtype=jnp.int32)

    def lookup_amplitude(params, state, _config):
        return params[jnp.dot(state, powers)]

    for column in range(columns.shape[1]):
        full = np.zeros((8,), dtype=np.complex128)
        for row, state in enumerate(basis):
            index = sum(bit << (2 - site) for site, bit in enumerate(state))
            full[index] = columns[row, column]
        model_projected = spatial_projected_amplitudes(
            jnp.asarray(full),
            jnp.asarray(basis),
            None,
            projector,
            amplitude_fn=lookup_amplitude,
        )
        np.testing.assert_allclose(
            np.asarray(model_projected), projected[:, column], atol=3.0e-15
        )

    with pytest.raises(ValueError, match="one character"):
        project_columns_by_index_table(columns, indices, characters=(1.0,))
    with pytest.raises(ValueError, match="unit modulus"):
        project_columns_by_index_table(
            columns, indices, characters=(1.0, 0.5, 1.0)
        )


def test_triangular_3x6_k_character_is_aligned_and_projector_valid() -> None:
    group = rectangular_translation_group(3, 6)
    # Triangular primitive coordinates: K=(1,1)/3, represented by integer
    # momentum indices (1,2) on a 3x6 torus.  This checks sector
    # addressability only and makes no statement that K is the ground sector.
    characters = rectangular_translation_characters(3, 6, 1, 2)
    assert len(group) == len(characters) == 18
    by_origin_image = {
        divmod(image[0], 6): character
        for image, character in zip(group, characters, strict=True)
    }
    np.testing.assert_allclose(by_origin_image[(0, 0)], 1.0, atol=0.0)
    np.testing.assert_allclose(
        by_origin_image[(1, 0)], np.exp(-2.0j * np.pi / 3.0), atol=2.0e-15
    )
    np.testing.assert_allclose(
        by_origin_image[(0, 1)], np.exp(-2.0j * np.pi / 3.0), atol=2.0e-15
    )
    assert abs(sum(characters)) < 3.0e-14
    projector = PhysicalSpatialProjector(group, characters)
    assert projector.characters == characters
    with pytest.raises(TypeError, match="momentum indices"):
        rectangular_translation_characters(3, 6, True, 2)


def test_triangular_3x6_hnf_numerator_9_0_m_character_is_representable() -> None:
    group = rectangular_translation_group(3, 6)
    # For the equivalent HNF diag(6,3), coordinates are ordered (x=column,
    # y=row). Numerator (9,0) over volume 18 is k=(1/2,0), hence rectangular
    # indices (momentum_row,momentum_column)=(0,3). This is M, not K.
    characters = rectangular_translation_characters(3, 6, 0, 3)
    by_origin_image = {
        divmod(image[0], 6): character
        for image, character in zip(group, characters, strict=True)
    }
    np.testing.assert_allclose(by_origin_image[(0, 1)], -1.0, atol=2.0e-15)
    np.testing.assert_allclose(by_origin_image[(1, 0)], 1.0, atol=2.0e-15)
    projector = PhysicalSpatialProjector(group, characters)
    assert projector.characters == characters


def test_local_channel_readout_tracks_known_product() -> None:
    basis = fixed_m_spin_basis(16, 0)
    columns, _ = square_a1_b1_product_columns(basis, SQUARE_CELLS)
    result = local_binary_channel_diagnostics(
        columns, columns[:, 0], n_cells=4
    )
    assert result["raw_carrier_subspace_weight"] == pytest.approx(1.0)
    assert result["mean_channel_1_occupation"] == pytest.approx(0.0)
    assert result["mean_absolute_pseudospin_x_coherence"] == pytest.approx(0.0)


def test_cell_partition_requires_disjoint_exact_cover() -> None:
    with pytest.raises(ValueError, match="disjoint exact cover"):
        validate_cell_partition(((0, 1, 2), (2, 3, 4)), n_sites=6, cell_size=3)
