from __future__ import annotations

import math

import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.sign_frame_obstruction import (
    aggregate_antiferromagnetic_bonds,
    align_state_real,
    all_born_sign_statistics,
    all_configuration_gauge_diagonals,
    born_sign_statistics,
    configuration_gauge_diagonal,
    edge_obstruction_costs,
    enumerate_site_z2_gauges,
    gauge_bitstring,
    gauge_minimum,
    matrix_positive_offdiagonal_fractions,
    named_order_frames,
    phase_frame_cost,
    z2_gauge_from_angles,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def _basis_indices(basis: tuple[tuple[int, ...], ...]) -> np.ndarray:
    n_sites = len(basis[0])
    return np.asarray(
        [
            sum(bit << (n_sites - site - 1) for site, bit in enumerate(state))
            for state in basis
        ],
        dtype=np.int64,
    )


def test_z2_gauges_are_unique_lexicographic_and_fix_global_redundancy() -> None:
    gauges = enumerate_site_z2_gauges(4)
    bitstrings = tuple(gauge_bitstring(row) for row in gauges)
    assert gauges.shape == (8, 4)
    assert np.all(gauges[:, 0] == 1)
    assert bitstrings == tuple(sorted(bitstrings))
    assert bitstrings == (
        "1000",
        "1001",
        "1010",
        "1011",
        "1100",
        "1101",
        "1110",
        "1111",
    )


def test_configuration_gauge_diagonal_matches_vectorized_definition() -> None:
    basis = enumerate_sz_configs(4, 2)
    gauges = enumerate_site_z2_gauges(4)
    actual = all_configuration_gauge_diagonals(basis, gauges)
    expected = np.stack(
        [configuration_gauge_diagonal(basis, gauge) for gauge in gauges]
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.square(actual), np.ones_like(actual))

    # z -> -z differs only by (-1)^N_down in a fixed magnetization sector,
    # so it cannot change C or V.  Use odd N_down to make the global factor
    # explicit rather than accidentally testing equality with +1.
    basis_six = enumerate_sz_configs(6, 3)
    gauge = np.asarray((1, -1, 1, 1, -1, -1), dtype=np.int8)
    diagonal = configuration_gauge_diagonal(basis_six, gauge)
    opposite = configuration_gauge_diagonal(basis_six, -gauge)
    np.testing.assert_array_equal(opposite, -diagonal)
    state = np.linspace(-1.0, 1.0, len(basis_six))
    first_stats = born_sign_statistics(state, diagonal, zero_tolerance=0.0)
    second_stats = born_sign_statistics(state, opposite, zero_tolerance=0.0)
    assert first_stats.coherence == pytest.approx(second_stats.coherence)
    assert first_stats.violation == pytest.approx(second_stats.violation)
    costs = edge_obstruction_costs(
        np.stack((gauge, -gauge)),
        ((0, 1, 1.0), (1, 2, 0.4), (3, 5, 0.2)),
        weight_power=1,
    )
    assert costs[0] == pytest.approx(costs[1])


def test_phi1_and_phi2_close_independently_against_fixed_sz_matrix() -> None:
    # Unequal weights ensure that the L1 and Frobenius normalizations are
    # genuinely independent rather than two copies of the same unit graph.
    n_sites = 4
    bonds = aggregate_antiferromagnetic_bonds(
        n_sites,
        ((0, 1, 1.0), (1, 2, 0.5), (0, 2, 0.25)),
    )
    gauges = enumerate_site_z2_gauges(n_sites)
    basis = enumerate_sz_configs(n_sites, n_sites // 2)
    diagonals = all_configuration_gauge_diagonals(basis, gauges)
    hamiltonian, matrix_basis = heisenberg_sz_graph_sparse(
        n_sites, bonds, n_up=n_sites // 2
    )
    assert matrix_basis == basis
    matrix_phi1, matrix_phi2, maximum_imaginary = (
        matrix_positive_offdiagonal_fractions(
            hamiltonian, diagonals, chunk_size=3
        )
    )
    graph_phi1 = edge_obstruction_costs(gauges, bonds, weight_power=1)
    graph_phi2 = edge_obstruction_costs(gauges, bonds, weight_power=2)
    np.testing.assert_allclose(matrix_phi1, graph_phi1, atol=2e-16)
    np.testing.assert_allclose(matrix_phi2, graph_phi2, atol=2e-16)
    assert maximum_imaginary == 0.0
    assert not np.allclose(graph_phi1, graph_phi2)


def test_unfrustrated_square_null_has_exact_marshall_frame() -> None:
    rows, columns = 2, 2
    bonds = aggregate_antiferromagnetic_bonds(
        rows * columns,
        square_j1j2_bonds(rows, columns, j1=1.0, j2=0.0, periodic=False),
    )
    gauges = enumerate_site_z2_gauges(rows * columns)
    phi1 = edge_obstruction_costs(gauges, bonds, weight_power=1)
    phi2 = edge_obstruction_costs(gauges, bonds, weight_power=2)
    minimum1 = gauge_minimum(phi1, gauges, tie_tolerance=1e-14)
    minimum2 = gauge_minimum(phi2, gauges, tie_tolerance=1e-14)
    neel = z2_gauge_from_angles(
        named_order_frames("square", rows, columns)["neel"]
    )
    neel_index = tuple(gauge_bitstring(row) for row in gauges).index(
        gauge_bitstring(neel)
    )
    assert minimum1.value == 0.0
    assert minimum2.value == 0.0
    assert neel_index in minimum1.indices
    assert neel_index in minimum2.indices

    basis = enumerate_sz_configs(rows * columns, 2)
    paths = enumerate_paths(rows * columns, 0)
    full_transform = coupled_basis_matrix(paths, 0, 0)
    transform = full_transform[_basis_indices(basis)]
    fusion = heisenberg_graph_fusion_matrix(rows * columns, bonds, target_q=0)
    _, vectors = np.linalg.eigh(fusion.toarray())
    ground, imaginary_norm, norm = align_state_real(transform @ vectors[:, 0])
    diagonal = configuration_gauge_diagonal(basis, neel)
    signs = born_sign_statistics(ground, diagonal, zero_tolerance=1e-14)
    assert imaginary_norm < 1e-15
    assert abs(norm - 1.0) < 1e-14
    assert signs.violation < 1e-28
    assert signs.zero_weight < 1e-28


def test_triangle_has_irreducible_z2_cost_but_equal_fixed_120_chiralities() -> None:
    bonds = aggregate_antiferromagnetic_bonds(
        3, ((0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0))
    )
    gauges = enumerate_site_z2_gauges(3)
    costs = edge_obstruction_costs(gauges, bonds, weight_power=1)
    minimum = gauge_minimum(costs, gauges, tie_tolerance=1e-14)
    assert minimum.value == pytest.approx(1.0 / 3.0)
    assert minimum.bitstrings == ("100", "101", "110")

    plus = np.asarray((0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0))
    minus = -plus
    assert phase_frame_cost(plus, bonds) == pytest.approx(0.25, abs=2e-16)
    assert phase_frame_cost(minus, bonds) == pytest.approx(0.25, abs=2e-16)


def test_registered_triangular_120_frames_have_equal_cost_on_both_shapes() -> None:
    for rows, columns in ((3, 4), (2, 6)):
        frames = named_order_frames("triangular", rows, columns)
        for ratio in (0.0, 0.125, 0.2):
            bonds = aggregate_antiferromagnetic_bonds(
                rows * columns,
                triangular_j1j2_bonds(
                    rows, columns, j1=1.0, j2=ratio, periodic=False
                ),
            )
            plus = phase_frame_cost(frames["120_plus"], bonds)
            minus = phase_frame_cost(frames["120_minus"], bonds)
            assert plus == pytest.approx(minus, abs=2e-15)


@pytest.mark.parametrize(
    ("lattice", "bonds"),
    (
        (
            "square",
            square_j1j2_bonds(2, 2, j1=1.0, j2=0.4, periodic=False),
        ),
        (
            "triangular",
            triangular_j1j2_bonds(2, 2, j1=1.0, j2=0.2, periodic=False),
        ),
    ),
)
def test_small_graph_fusion_hamiltonian_closes_against_sz_projection(
    lattice: str, bonds: tuple[tuple[int, int, float], ...]
) -> None:
    del lattice
    n_sites = 4
    paths = enumerate_paths(n_sites, 0)
    basis = enumerate_sz_configs(n_sites, n_sites // 2)
    full_transform = coupled_basis_matrix(paths, 0, 0)
    transform = full_transform[_basis_indices(basis)]
    h_sz, matrix_basis = heisenberg_sz_graph_sparse(n_sites, bonds, n_up=2)
    h_fusion = heisenberg_graph_fusion_matrix(n_sites, bonds, target_q=0)
    assert matrix_basis == basis
    np.testing.assert_allclose(
        transform.conj().T @ (h_sz @ transform),
        h_fusion.toarray(),
        atol=3e-14,
    )
    np.testing.assert_allclose(
        transform.conj().T @ transform, np.eye(len(paths)), atol=3e-14
    )


def test_born_sign_statistics_vectorization_and_zero_handling() -> None:
    state = np.asarray((math.sqrt(0.6), -math.sqrt(0.3), math.sqrt(0.1)))
    diagonals = np.asarray(((1, 1, 1), (-1, 1, -1)), dtype=np.int8)
    vectorized = all_born_sign_statistics(
        state, diagonals, zero_tolerance=0.4
    )
    for index, diagonal in enumerate(diagonals):
        direct = born_sign_statistics(state, diagonal, zero_tolerance=0.4)
        assert vectorized["positive_weight"][index] == pytest.approx(
            direct.positive_weight
        )
        assert vectorized["negative_weight"][index] == pytest.approx(
            direct.negative_weight
        )
        assert vectorized["zero_weight"][index] == pytest.approx(
            direct.zero_weight
        )
        assert vectorized["coherence"][index] == pytest.approx(direct.coherence)
        assert vectorized["violation"][index] == pytest.approx(direct.violation)
        assert 0.0 <= direct.violation <= 0.5
    assert vectorized["zero_weight"][0] == pytest.approx(0.1)
    assert vectorized["positive_weight"][0] == pytest.approx(0.6)
    assert vectorized["negative_weight"][0] == pytest.approx(0.3)

    # A real eigensolver may choose either global sign.  W+ and W- swap, but
    # the registered coherence and violation are exactly invariant.
    positive_phase = born_sign_statistics(
        state, diagonals[0], zero_tolerance=0.4
    )
    negative_phase = born_sign_statistics(
        -state, diagonals[0], zero_tolerance=0.4
    )
    assert positive_phase.positive_weight == pytest.approx(
        negative_phase.negative_weight
    )
    assert positive_phase.negative_weight == pytest.approx(
        negative_phase.positive_weight
    )
    assert positive_phase.coherence == pytest.approx(negative_phase.coherence)
    assert positive_phase.violation == pytest.approx(negative_phase.violation)


def test_sign_frame_helpers_reject_out_of_scope_inputs() -> None:
    gauges = enumerate_site_z2_gauges(3)
    with pytest.raises(ValueError, match="J >= 0"):
        aggregate_antiferromagnetic_bonds(3, ((0, 1, -1.0),))
    with pytest.raises(ValueError, match="weight_power"):
        edge_obstruction_costs(gauges, ((0, 1, 1.0),), weight_power=3)
    with pytest.raises(ValueError, match="gauge entries"):
        edge_obstruction_costs(
            np.asarray(((1, 0, -1),)), ((0, 1, 1.0),), weight_power=1
        )
    with pytest.raises(ValueError, match="dimension"):
        matrix_positive_offdiagonal_fractions(
            np.asarray(((0.0, 1.0), (1.0, 0.0))),
            np.ones((1, 3), dtype=np.int8),
        )
    with pytest.raises(ValueError, match="not Z2"):
        z2_gauge_from_angles((0.0, math.pi / 2.0))
