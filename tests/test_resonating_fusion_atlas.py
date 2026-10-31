from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.resonating_fusion_atlas import (
    atlas_subset_complexity,
    fusion_chart_components,
    globally_scrambled_atlas_orders,
    rectangular_fusion_atlas_orders,
    resonating_atlas_analysis,
)
from su2_nqs.su2.fusion_path import count_paths


@pytest.mark.parametrize("lattice", ("square", "triangular"))
def test_rectangular_atlas_has_eight_distinct_site_permutations(lattice: str) -> None:
    orders = rectangular_fusion_atlas_orders(lattice, 3, 4)
    assert len(orders) == 8
    assert len(set(orders.values())) == 8
    assert next(iter(orders.values())) == tuple(range(12))
    for order in orders.values():
        assert tuple(sorted(order)) == tuple(range(12))


def test_global_scramble_preserves_base_and_relabels_other_charts() -> None:
    orders = rectangular_fusion_atlas_orders("square", 3, 4)
    permutation = tuple(reversed(range(12)))
    scrambled = globally_scrambled_atlas_orders(orders, permutation)
    names = tuple(orders)
    assert scrambled[names[0]] == orders[names[0]]
    for name in names[1:]:
        assert scrambled[name] == tuple(permutation[site] for site in orders[name])


def test_fusion_chart_transport_is_normalized_and_has_exact_base_column() -> None:
    orders = rectangular_fusion_atlas_orders("square", 3, 4)
    dimension = count_paths(12, 0)
    rng = np.random.default_rng(17)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    state /= np.linalg.norm(state)
    components, audits = fusion_chart_components(state, orders, target_q=0)
    np.testing.assert_allclose(components[:, 0], state, atol=2.0e-13)
    np.testing.assert_allclose(
        np.sum(np.abs(components) ** 2, axis=0), np.ones(8), atol=3.0e-13
    )
    assert max(float(row["inverse_recovery_error"]) for row in audits) < 3.0e-13


def test_subset_complexity_counts_coherent_two_chart_gain() -> None:
    components = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    components /= np.linalg.norm(components, axis=0)
    hamiltonian = np.diag([0.0, 2.0, 5.0])
    result = atlas_subset_complexity(
        components, hamiltonian, recovery_fraction=0.9
    )
    assert result["best_single_energy"] == pytest.approx(1.0)
    assert result["full_ritz_energy"] == pytest.approx(0.0, abs=1.0e-13)
    assert result["minimum_chart_count_for_recovery"] == 2


def test_resonating_analysis_ritz_does_not_need_target_to_lower_energy() -> None:
    components = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    hamiltonian = np.diag([-2.0, 1.0])
    target = np.asarray([1.0, 0.0], dtype=np.complex128)
    result = resonating_atlas_analysis(
        components,
        hamiltonian,
        target,
        component_names=("left", "right"),
        exact_energy=-2.0,
    )
    assert result["hamiltonian_ritz"]["energy"] == pytest.approx(-2.0)
    assert result["hamiltonian_ritz"]["fidelity"] == pytest.approx(1.0)
    assert result["fractional_energy_error_reduction_from_base"] == pytest.approx(1.0)


def test_invalid_atlas_requests_fail_closed() -> None:
    with pytest.raises(ValueError, match="lattice"):
        rectangular_fusion_atlas_orders("honeycomb", 3, 4)
    with pytest.raises(ValueError, match="permutation"):
        globally_scrambled_atlas_orders(
            rectangular_fusion_atlas_orders("square", 3, 4),
            tuple(range(11)),
        )

