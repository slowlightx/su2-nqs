from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.two_cluster_support_orientation import (
    support_orientation_carrier_report,
)
from su2_nqs.benchmarks.two_cluster_virtual_processes import (
    cluster_q2_response_basis,
    two_cluster_open_partition,
    two_cluster_virtual_process_report,
)


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_public_cluster_response_basis_reconstructs_the_canonical_open_cut(
    geometry: str, ratio: float
) -> None:
    local = cluster_q2_response_basis(geometry, ratio)
    partition = two_cluster_open_partition(geometry, ratio)
    coupling = np.zeros((6, 6), dtype=np.float64)
    for bond in partition.intercluster_bonds:
        coupling[bond.first_local, bond.second_local] += bond.coupling
    reconstructed = (
        np.sqrt(3.0)
        * local.sz_transition_matrix
        @ coupling
        @ local.sz_transition_matrix.T
    )
    report = two_cluster_virtual_process_report(geometry, ratio)
    assert local.triplet_energies.shape == (9,)
    assert local.sz_transition_matrix.shape == (9, 6)
    assert np.all(local.triplet_gaps > 0.0)
    # Total S^z annihilates the singlet, hence each transition row sums to zero.
    np.testing.assert_allclose(
        np.sum(local.sz_transition_matrix, axis=1), 0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(
        reconstructed, report.response_amplitude_matrix, atol=2.0e-15
    )


@pytest.mark.parametrize(
    (
        "geometry",
        "ratio",
        "direction_count",
        "direction_interface_count",
        "direction_occurrences",
        "pair_occurrences",
        "direction_route_k1",
        "direction_shared_e2",
        "pair_route_k1",
        "pair_shared_e2",
        "pair_mean_e2",
        "direction_minima",
        "pair_minima",
        "pair_route_ranks",
    ),
    (
        (
            "square",
            0.5,
            4,
            8,
            288,
            90,
            0.9225913554158073,
            (0.5164042114420191, 0.7987970636475531, 0.9127881605172853,
             0.9696434556699364, 0.9996737811043953),
            0.5970276063740122,
            (0.5558908172026749, 0.8001386922519965, 0.868288995251855,
             0.9431478252358402, 0.9998052380965441),
            -1.456376468356561,
            (3, 5, 5),
            (4, 5, 5),
            (4, 7, 9),
        ),
        (
            "triangular",
            0.125,
            6,
            13,
            468,
            90,
            0.9250222834070703,
            (0.4923990986017413, 0.7615785049811, 0.8964580487838454,
             0.9572741893367261, 0.9995422977637276),
            0.6324327397404883,
            (0.5027631534668692, 0.7244381510416965, 0.8177780269697064,
             0.922281912159803, 0.999680223065249),
            -1.6701680807507262,
            (4, 5, 5),
            (4, 5, 5),
            (5, 8, 9),
        ),
    ),
)
def test_translated_orientation_union_quantifies_shared_carrier_cost(
    geometry: str,
    ratio: float,
    direction_count: int,
    direction_interface_count: int,
    direction_occurrences: int,
    pair_occurrences: int,
    direction_route_k1: float,
    direction_shared_e2,
    pair_route_k1: float,
    pair_shared_e2,
    pair_mean_e2: float,
    direction_minima,
    pair_minima,
    pair_route_ranks,
) -> None:
    report = support_orientation_carrier_report(geometry, ratio)
    assert report.lattice_shape == (6, 6)
    assert report.cluster_shape == (2, 3)
    assert report.translation_group_order == 36
    assert report.distinct_tiling_count == 6
    assert report.tiling_stabilizer_order == 6
    assert report.representative_translation_shifts == (
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    )
    assert len(report.direction_classes) == direction_count
    assert all(row.periodic_bond_count == 36 for row in report.direction_classes)
    assert report.periodic_bond_reconstruction_error == 0.0
    assert len(report.direction_resolved_interfaces) == direction_interface_count
    assert len(report.physical_pair_total_interfaces) == 3

    direction = report.direction_resolved_union
    pair = report.physical_pair_total_union
    assert direction.weighted_occurrence_count == direction_occurrences
    assert pair.weighted_occurrence_count == pair_occurrences
    assert direction.union_rank == pair.union_rank == 9
    np.testing.assert_array_equal(direction.route_k_union_ranks, np.full(9, 9))
    np.testing.assert_array_equal(pair.route_k_union_ranks[:3], pair_route_ranks)
    np.testing.assert_allclose(
        direction.route_conditioned_e2_fractions[0],
        direction_route_k1,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        pair.route_conditioned_e2_fractions[0], pair_route_k1, atol=3.0e-14
    )
    np.testing.assert_allclose(
        direction.shared_bank_e2_fractions[:5], direction_shared_e2, atol=3.0e-14
    )
    np.testing.assert_allclose(
        pair.shared_bank_e2_fractions[:5], pair_shared_e2, atol=3.0e-14
    )
    np.testing.assert_allclose(
        pair.mean_second_order_energy_per_distinct_tiling,
        pair_mean_e2,
        atol=3.0e-14,
    )
    assert (
        direction.minimum_shared_carriers_for_90pct_e2,
        direction.minimum_shared_carriers_for_99pct_e2,
        direction.minimum_shared_carriers_for_99p9pct_e2,
    ) == direction_minima
    assert (
        pair.minimum_shared_carriers_for_90pct_e2,
        pair.minimum_shared_carriers_for_99pct_e2,
        pair.minimum_shared_carriers_for_99p9pct_e2,
    ) == pair_minima
    for union in (direction, pair):
        assert np.all(np.diff(union.union_cumulative_weight_fractions) >= -2.0e-15)
        assert np.all(np.diff(union.shared_bank_e2_fractions) >= -2.0e-13)
        assert np.all(np.diff(union.route_conditioned_e2_fractions) >= -2.0e-13)
        np.testing.assert_allclose(union.shared_bank_e2_fractions[-1], 1.0, atol=2.0e-15)
        np.testing.assert_allclose(
            union.route_conditioned_e2_fractions[-1], 1.0, atol=2.0e-15
        )
    assert "does not include 3x2 rotated/reflected" in report.claim_boundary


def test_square_and_triangular_direction_ledgers_use_the_registered_axes() -> None:
    square = support_orientation_carrier_report("square", 0.5)
    triangular = support_orientation_carrier_report("triangular", 0.125)
    assert [(row.shell, row.displacement) for row in square.direction_classes] == [
        ("J1", (0, 1)),
        ("J1", (1, 0)),
        ("J2", (1, 1)),
        ("J2", (1, -1)),
    ]
    assert [(row.shell, row.displacement) for row in triangular.direction_classes] == [
        ("J1", (1, 0)),
        ("J1", (0, 1)),
        ("J1", (1, 1)),
        ("J2", (1, -1)),
        ("J2", (1, 2)),
        ("J2", (2, 1)),
    ]
