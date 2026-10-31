from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix
from su2_nqs.benchmarks.two_cluster_virtual_processes import (
    translated_covering_resonance_report,
    two_cluster_open_partition,
    two_cluster_virtual_process_report,
)
from su2_nqs.frontier.triangle_patch_physical_init import (
    triangle_patch_cluster_reference,
)


def _embedded_cluster_product(geometry: str, ratio: float, basis) -> np.ndarray:
    local = triangle_patch_cluster_reference(geometry, ratio).computational_state
    clusters = ((0, 1, 2, 6, 7, 8), (3, 4, 5, 9, 10, 11))
    answer = []
    for configuration in basis:
        indices = []
        for sites in clusters:
            index = 0
            for site in sites:
                index = (index << 1) | ((configuration >> (11 - site)) & 1)
            indices.append(index)
        answer.append(local[indices[0]] * local[indices[1]])
    return np.asarray(answer)


@pytest.mark.parametrize(
    ("geometry", "ratio", "expected_bonds"),
    (
        (
            "square",
            0.5,
            (
                (2, 3, 2, 0, 1.0, "J1"),
                (2, 9, 2, 3, 0.5, "J2"),
                (8, 3, 5, 0, 0.5, "J2"),
                (8, 9, 5, 3, 1.0, "J1"),
            ),
        ),
        (
            "triangular",
            0.125,
            (
                (1, 9, 1, 3, 0.125, "J2"),
                (2, 3, 2, 0, 1.0, "J1"),
                (2, 9, 2, 3, 1.0, "J1"),
                (2, 10, 2, 4, 0.125, "J2"),
                (8, 3, 5, 0, 0.125, "J2"),
                (8, 9, 5, 3, 1.0, "J1"),
            ),
        ),
    ),
)
def test_open_two_cluster_partition_reconstructs_the_graph_and_core(
    geometry: str,
    ratio: float,
    expected_bonds,
) -> None:
    partition = two_cluster_open_partition(geometry, ratio)
    observed = tuple(
        (
            bond.first,
            bond.second,
            bond.first_local,
            bond.second_local,
            bond.coupling,
            bond.shell,
        )
        for bond in partition.intercluster_bonds
    )
    assert observed == expected_bonds
    assert partition.reconstruction_error == 0.0
    assert partition.cluster_bonds == triangle_patch_cluster_reference(
        geometry, ratio
    ).bonds


@pytest.mark.parametrize(
    (
        "geometry",
        "ratio",
        "norm_squared",
        "second_order",
        "lowest_e2_fraction",
        "schmidt_rank",
        "singular_values",
        "cumulative_e2",
        "raw_carrier_e2",
        "dressed_carrier_e2",
    ),
    (
        (
            "square",
            0.5,
            0.16907690268924586,
            -0.07992938861199121,
            0.34749254741427715,
            2,
            (0.3839938906, 0.1470564320),
            (
                0.34749255,
                0.81336837,
                0.84430400,
                0.86770395,
                0.92788166,
                0.98553836,
                0.98613859,
                0.99433705,
                1.0,
            ),
            (0.8557865299838664, 0.9398308229670403),
            (0.915240692894982, 0.999418541793712),
        ),
        (
            "triangular",
            0.125,
            0.18833242144942128,
            -0.08906449675587971,
            0.16523109478608175,
            3,
            (0.393257138, 0.183523057, 0.000729410),
            (
                0.16523109,
                0.20147469,
                0.54572654,
                0.81927493,
                0.92778002,
                0.96982435,
                0.97800379,
                0.99190634,
                1.0,
            ),
            (0.8175081293617719, 0.9363549990252542),
            (0.8750714158657737, 0.9996527123992027),
        ),
    ),
)
def test_virtual_boundary_action_is_exactly_a_triplet_pair_and_low_rank(
    geometry: str,
    ratio: float,
    norm_squared: float,
    second_order: float,
    lowest_e2_fraction: float,
    schmidt_rank: int,
    singular_values,
    cumulative_e2,
    raw_carrier_e2,
    dressed_carrier_e2,
) -> None:
    report = two_cluster_virtual_process_report(geometry, ratio)
    assert report.response_amplitude_matrix.shape == (9, 9)
    assert report.response_schmidt_rank == schmidt_rank
    np.testing.assert_allclose(report.action_norm_squared, norm_squared, atol=2.0e-14)
    np.testing.assert_allclose(report.q2_q2_norm_squared, norm_squared, atol=2.0e-14)
    np.testing.assert_allclose(report.second_order_energy, second_order, atol=2.0e-14)
    np.testing.assert_allclose(
        report.lowest_triplet_pair_e2_fraction,
        lowest_e2_fraction,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        report.response_singular_values[:schmidt_rank], singular_values, atol=7.0e-10
    )
    np.testing.assert_allclose(
        report.cumulative_low_energy_triplet_e2_fractions,
        cumulative_e2,
        atol=6.0e-9,
    )
    np.testing.assert_allclose(
        report.raw_response_carrier_e2_fractions[:2],
        raw_carrier_e2,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        report.correction_carrier_e2_fractions[:2],
        dressed_carrier_e2,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        report.correction_truncated_svd_e2_fractions[:2],
        dressed_carrier_e2,
        atol=2.0e-14,
    )
    assert report.response_leading_left_right_overlap < 0.08
    assert report.correction_leading_left_right_overlap < 0.21
    assert report.rank_one_selection_residual < 4.0e-15
    assert report.outside_q2_q2_norm_bound < 1.0e-27
    assert abs(report.first_order_expectation) < 2.0e-31
    assert np.all(np.diff(report.cumulative_low_energy_triplet_e2_fractions) >= 0.0)
    np.testing.assert_allclose(
        report.response_cumulative_norm_fractions[-1], 1.0, atol=3.0e-15
    )


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_response_norm_matches_an_independent_n12_boundary_hamiltonian(
    geometry: str, ratio: float
) -> None:
    partition = two_cluster_open_partition(geometry, ratio)
    boundary_bonds = tuple(
        (min(bond.first, bond.second), max(bond.first, bond.second), bond.coupling)
        for bond in partition.intercluster_bonds
    )
    boundary_h, basis = heisenberg_sz_matrix(12, boundary_bonds, n_up=6)
    product = _embedded_cluster_product(geometry, ratio, basis)
    action = boundary_h @ product
    report = two_cluster_virtual_process_report(geometry, ratio)
    np.testing.assert_allclose(np.linalg.norm(product), 1.0, atol=2.0e-13)
    np.testing.assert_allclose(product @ action, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        np.vdot(action, action).real,
        report.action_norm_squared,
        atol=2.0e-14,
    )


@pytest.mark.parametrize(
    (
        "geometry",
        "ratio",
        "exact_energy",
        "product_fidelity",
        "sector_weights",
        "excited_q0_weight",
        "second_order_gap_fraction",
        "carrier_gap_fractions",
    ),
    (
        (
            "square",
            0.5,
            -5.543196174870516,
            0.7981338665796421,
            (0.8017884046869482, 0.1981686040324492, 4.299111249023144e-5, 1.681132424133259e-10),
            0.003654538107304095,
            0.42646368713671046,
            (
                0.20823693225369833,
                0.7947055147869999,
                0.831025701097883,
                0.842536827830477,
                0.8801344957577519,
                0.9748104873868502,
            ),
        ),
        (
            "triangular",
            0.125,
            -5.506113554412756,
            0.7896967321560497,
            (0.7916048544574605, 0.20832679161778306, 6.835387772982053e-5, 4.7028432126079545e-11),
            0.001908122301409354,
            0.4631873830475741,
            (
                0.1237005371224334,
                0.6938015002778276,
                0.7760752296932711,
                0.7536675737036723,
                0.842012392399442,
                0.9829432808546574,
            ),
        ),
    ),
)
def test_full_coupling_open_ground_confirms_core_plus_virtual_triplet_picture(
    geometry: str,
    ratio: float,
    exact_energy: float,
    product_fidelity: float,
    sector_weights,
    excited_q0_weight: float,
    second_order_gap_fraction: float,
    carrier_gap_fractions,
) -> None:
    report = two_cluster_virtual_process_report(geometry, ratio)
    np.testing.assert_allclose(report.open_exact_ground_energy, exact_energy, atol=3.0e-14)
    np.testing.assert_allclose(
        report.open_product_energy, 2.0 * report.cluster_energy, atol=3.0e-14
    )
    np.testing.assert_allclose(
        report.open_product_ground_fidelity, product_fidelity, atol=3.0e-14
    )
    np.testing.assert_array_equal(report.exact_ground_sector_q_values, (0, 2, 4, 6))
    np.testing.assert_allclose(
        report.exact_ground_equal_q_sector_weights, sector_weights, atol=4.0e-14
    )
    np.testing.assert_allclose(
        report.exact_ground_excited_q0_pair_weight,
        excited_q0_weight,
        atol=4.0e-14,
    )
    np.testing.assert_allclose(
        report.second_order_open_gap_fraction,
        second_order_gap_fraction,
        atol=4.0e-14,
    )
    assert report.exact_ground_offdiagonal_q_weight < 2.0e-28
    assert report.response_matrix_physical_residual < 1.0e-14
    np.testing.assert_allclose(
        [row.recovered_exact_gap_fraction for row in report.full_coupling_carriers],
        carrier_gap_fractions,
        atol=5.0e-14,
    )
    assert [row.carrier for row in report.full_coupling_carriers] == [
        "lowest_triplet_pair",
        "raw_response_schmidt_k1",
        "raw_response_schmidt_k2",
        "resolvent_dressed_schmidt_k1",
        "resolvent_dressed_schmidt_k2",
        "complete_q2_pair_space",
    ]


@pytest.mark.parametrize(
    (
        "geometry",
        "ratio",
        "distinct_count",
        "single",
        "column_energy",
        "all_translation_energy",
        "exact",
        "all_translation_fraction",
    ),
    (
        (
            "square",
            0.5,
            3,
            -7.641550870832068,
            -7.97022804918142,
            -7.97022804918142,
            -9.0,
            0.24195030295368225,
        ),
        (
            "triangular",
            0.125,
            6,
            -7.715314761541281,
            -8.072525649576278,
            -8.100300187679318,
            -9.018707191613895,
            0.2953718444694273,
        ),
    ),
)
def test_periodic_translated_coverings_quantify_a_separate_singlet_resonance(
    geometry: str,
    ratio: float,
    distinct_count: int,
    single: float,
    column_energy: float,
    all_translation_energy: float,
    exact: float,
    all_translation_fraction: float,
) -> None:
    report = translated_covering_resonance_report(geometry, ratio)
    assert report.raw_translation_count == 12
    assert report.distinct_covering_count == distinct_count
    assert report.subspace_rank == distinct_count
    assert report.column_only_covering_count == 3
    assert report.column_only_subspace_rank == 3
    assert "thin torus" in report.boundary_convention
    np.testing.assert_allclose(report.single_covering_energy, single, atol=3.0e-13)
    np.testing.assert_allclose(
        report.column_only_ground_energy, column_energy, atol=3.0e-13
    )
    np.testing.assert_allclose(
        report.generalized_ground_energy, all_translation_energy, atol=3.0e-13
    )
    np.testing.assert_allclose(report.uniform_coherent_energy, all_translation_energy, atol=3.0e-13)
    np.testing.assert_allclose(report.exact_ground_energy, exact, atol=4.0e-13)
    np.testing.assert_allclose(
        report.recovered_exact_gap_fraction,
        all_translation_fraction,
        atol=3.0e-13,
    )
    assert np.min(report.overlap_eigenvalues) > 1.0e-3
    assert report.generalized_ground_energy < report.single_covering_energy


def test_two_cluster_diagnostic_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="geometry"):
        two_cluster_open_partition("honeycomb", 0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        two_cluster_virtual_process_report("square", -0.1)
