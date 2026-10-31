from __future__ import annotations

import math

import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import (
    heisenberg_sz_matrix,
    spin_squared_matrix,
)
from su2_nqs.benchmarks.lattice_observables import (
    compile_sector_rank2_composite,
    computational_rank2_component,
    oriented_triangular_plaquettes,
    scalar_chirality_computational_matrix,
    scalar_chirality_fusion_matrix,
    square_obc_channel_operators,
    square_obc_channel_operators_sz,
    triangular_nearest_neighbor_orientation_bonds,
    triangular_obc_channel_operators,
    triangular_obc_channel_operators_sz,
    uniform_triangular_chirality_fusion_matrix,
    weighted_spherical_spin_component,
)
from su2_nqs.benchmarks.spectral_physics import compile_sector_spin_vector
from su2_nqs.su2.fusion_path import enumerate_paths


@pytest.mark.parametrize("n_sites", (4, 6))
def test_vector_probe_matches_independent_computational_projection(n_sites: int) -> None:
    weights = np.asarray(
        [np.exp(2.0j * np.pi * site / n_sites) for site in range(n_sites)]
    )
    reduced = compile_sector_spin_vector(weights, q_new=2, q_old=0)
    singlet = coupled_basis_matrix(enumerate_paths(n_sites, 0), 0, 0)
    triplet = coupled_basis_matrix(enumerate_paths(n_sites, 2), 2, 0)
    explicit = triplet.conj().T @ (
        weighted_spherical_spin_component(weights, 0) @ singlet
    )
    np.testing.assert_allclose(
        reduced.matrix.toarray() / math.sqrt(3.0), explicit, atol=3.0e-13
    )


@pytest.mark.parametrize("n_sites", (4, 6))
def test_rank2_composite_matches_independent_computational_projection(
    n_sites: int,
) -> None:
    generator = np.random.default_rng(26060909 + n_sites)
    left = generator.normal(size=n_sites) + 0.2j * generator.normal(size=n_sites)
    right = generator.normal(size=n_sites) - 0.3j * generator.normal(size=n_sites)
    reduced = compile_sector_rank2_composite(left, right, q_new=4, q_old=0)
    singlet = coupled_basis_matrix(reduced.old_paths, 0, 0)
    quintuplet = coupled_basis_matrix(reduced.new_paths, 4, 0)
    explicit = quintuplet.conj().T @ (
        computational_rank2_component(left, right, 0) @ singlet
    )
    np.testing.assert_allclose(
        reduced.matrix / math.sqrt(5.0), explicit, atol=8.0e-13
    )
    assert np.linalg.norm(reduced.matrix) > 1.0e-8


@pytest.mark.parametrize("n_sites", (4, 6))
def test_vector_probe_has_exact_q0_to_q4_wigner_eckart_zero(n_sites: int) -> None:
    weights = np.asarray(
        [np.exp(2.0j * np.pi * site / n_sites) for site in range(n_sites)]
    )
    singlet = coupled_basis_matrix(enumerate_paths(n_sites, 0), 0, 0)
    maximum = 0.0
    for component_q in (-2, 0, 2):
        quintuplet = coupled_basis_matrix(
            enumerate_paths(n_sites, 4), 4, component_q
        )
        forbidden = quintuplet.conj().T @ (
            weighted_spherical_spin_component(weights, component_q) @ singlet
        )
        maximum = max(maximum, float(np.max(np.abs(forbidden), initial=0.0)))
    assert maximum < 2.0e-15


def test_scalar_chirality_commutator_orientation_and_total_spin() -> None:
    chirality = scalar_chirality_computational_matrix(3, 0, 1, 2)
    h01, _ = heisenberg_sz_matrix(3, ((0, 1, 1.0),))
    h12, _ = heisenberg_sz_matrix(3, ((1, 2, 1.0),))
    commutator = 1.0j * (h01 @ h12 - h12 @ h01)
    np.testing.assert_allclose(chirality, commutator, atol=2.0e-15)
    np.testing.assert_allclose(chirality, chirality.conj().T, atol=2.0e-15)
    np.testing.assert_allclose(
        scalar_chirality_computational_matrix(3, 2, 1, 0),
        -chirality,
        atol=2.0e-15,
    )
    spin_squared, _ = spin_squared_matrix(3)
    np.testing.assert_allclose(
        chirality @ spin_squared - spin_squared @ chirality,
        0.0,
        atol=3.0e-15,
    )


@pytest.mark.parametrize("n_sites", (4, 6))
def test_scalar_chirality_fusion_projection_matches_computational_reference(
    n_sites: int,
) -> None:
    paths = enumerate_paths(n_sites, 0)
    coupled = coupled_basis_matrix(paths, 0, 0)
    computational = scalar_chirality_computational_matrix(n_sites, 0, 1, 2)
    expected = coupled.conj().T @ computational @ coupled
    fusion = scalar_chirality_fusion_matrix(
        n_sites, 0, 1, 2, target_q=0
    ).toarray()
    np.testing.assert_allclose(fusion, expected, atol=3.0e-14)
    np.testing.assert_allclose(fusion, fusion.conj().T, atol=3.0e-14)


def test_triangular_plaquettes_are_complete_and_counterclockwise() -> None:
    rows, columns = 3, 4
    triangles = oriented_triangular_plaquettes(rows, columns)
    assert len(triangles) == 2 * (rows - 1) * (columns - 1)
    assert len(set(tuple(sorted(triangle)) for triangle in triangles)) == len(triangles)

    def point(site: int) -> np.ndarray:
        row, column = divmod(site, columns)
        return np.asarray([row - 0.5 * column, math.sqrt(3.0) * column / 2.0])

    for first, middle, last in triangles:
        a, b, c = point(first), point(middle), point(last)
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        assert cross > 0.0

    uniform = uniform_triangular_chirality_fusion_matrix(2, 3)
    assert uniform.shape == (5, 5)
    np.testing.assert_allclose(uniform.toarray(), uniform.toarray().conj().T, atol=3e-14)


def test_shared_obc_channel_panels_are_complete_and_hermitian() -> None:
    square = square_obc_channel_operators(2, 2)
    triangular = triangular_obc_channel_operators(2, 3)
    assert set(square) == {
        "neel", "stripe_x", "stripe_y", "b1g", "columnar_x", "columnar_y"
    }
    assert set(triangular) == {
        "k", "m1", "m2", "m3", "nematic_1", "nematic_2", "chirality"
    }
    assert tuple(
        len(family)
        for family in triangular_nearest_neighbor_orientation_bonds(3, 4)
    ) == (8, 9, 6)
    for operator in (*square.values(), *triangular.values()):
        np.testing.assert_allclose(
            operator.toarray(), operator.toarray().conj().T, atol=3.0e-14
        )


@pytest.mark.parametrize(
    ("lattice", "fusion_builder", "sz_builder"),
    (
        ("square", square_obc_channel_operators, square_obc_channel_operators_sz),
        (
            "triangular",
            triangular_obc_channel_operators,
            triangular_obc_channel_operators_sz,
        ),
    ),
)
def test_fixed_sz_channel_panels_project_to_fusion_exactly(
    lattice, fusion_builder, sz_builder
) -> None:
    del lattice
    rows, columns = 2, 3
    n_sites = rows * columns
    fusion = fusion_builder(rows, columns)
    sz, configurations = sz_builder(rows, columns, n_up=n_sites // 2)
    paths = enumerate_paths(n_sites, 0)
    coupled = coupled_basis_matrix(paths, 0, 0)
    indices = [
        sum(bit << (n_sites - site - 1) for site, bit in enumerate(config))
        for config in configurations
    ]
    transform = coupled[np.asarray(indices)]
    assert set(fusion) == set(sz)
    for name in fusion:
        projected = transform.conj().T @ (sz[name] @ transform)
        np.testing.assert_allclose(
            projected, fusion[name].toarray(), atol=8.0e-14
        )
        np.testing.assert_allclose(
            sz[name].toarray(), sz[name].toarray().conj().T, atol=3.0e-14
        )
