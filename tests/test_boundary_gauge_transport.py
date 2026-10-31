import numpy as np
import pytest

from su2_nqs.benchmarks.boundary_gauge_transport import (
    compile_boundary_gauge_transport,
    endpoint_gauge_matrix,
    persistent_subspace_transport,
    svd_truncation_audit,
)
from su2_nqs.benchmarks.boundary_operator_channels import (
    BoundaryBond,
    compile_boundary_recoupling_plan,
    partition_boundary_bonds,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix


def _graph(n: int) -> tuple[BoundaryBond, ...]:
    return tuple(
        BoundaryBond(site + 1, site, 1.0, 0) for site in range(n - 1)
    ) + tuple(
        BoundaryBond(site + 2, site, 0.39, 1) for site in range(n - 2)
    )


def _endpoint_bonds(plan):
    matrix = endpoint_gauge_matrix(plan)
    return tuple(
        (int(first), int(second), float(matrix[i, j]))
        for i, first in enumerate(plan.hidden_sites)
        for j, second in enumerate(plan.prefix_sites)
        if abs(matrix[i, j]) > 1.0e-14
    )


@pytest.mark.parametrize("n", (4, 6, 8, 10))
def test_successive_gauges_reconstruct_crossing_and_track_rank_changes(n: int) -> None:
    bonds = _graph(n)
    plans = [
        compile_boundary_recoupling_plan(n, bonds, decision)
        for decision in range(n - 1)
    ]
    for plan in plans:
        np.testing.assert_allclose(
            endpoint_gauge_matrix(plan), plan.coupling_matrix, atol=1e-14
        )
        crossing = partition_boundary_bonds(n, bonds, plan.decision)["crossing"]
        expected = heisenberg_graph_fusion_matrix(
            n,
            tuple((bond.first, bond.second, bond.coupling) for bond in crossing),
            target_q=0,
        )
        actual = heisenberg_graph_fusion_matrix(
            n, _endpoint_bonds(plan), target_q=0
        )
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=8e-12)

        truncation = svd_truncation_audit(plan, 3)
        assert truncation.retained_rank == min(3, plan.channel_rank)
        np.testing.assert_allclose(
            truncation.relative_frobenius_error**2,
            1.0 - truncation.cumulative_squared_weight,
            atol=3e-14,
        )

    for old, new in zip(plans[:-1], plans[1:], strict=True):
        transport = compile_boundary_gauge_transport(old, new)
        assert transport.old_svd_rank == old.channel_rank
        assert transport.new_svd_rank == new.channel_rank
        assert transport.svd_rank_births == max(
            0, new.channel_rank - old.channel_rank
        )
        assert transport.svd_rank_deaths == max(
            0, old.channel_rank - new.channel_rank
        )
        assert transport.common_coupling_error < 1e-14
        removed = tuple(
            sorted(set(map(int, old.hidden_sites)) - set(map(int, new.hidden_sites)))
        )
        assert transport.migrating_sites == removed
        assert set(transport.endpoint_births) == set(
            map(int, new.endpoint_hidden_sites)
        ) - set(map(int, old.endpoint_hidden_sites))
        assert set(transport.endpoint_deaths) == set(
            map(int, old.endpoint_hidden_sites)
        ) - set(map(int, new.endpoint_hidden_sites))
        for update in transport.endpoint_updates:
            assert update.common_prefix_error < 1e-14
            assert -1e-14 <= update.cosine_overlap <= 1.0 + 1e-14


def test_endpoint_orientation_is_independent_of_input_bond_direction() -> None:
    n = 8
    forward = tuple(
        BoundaryBond(site, site + 1, 1.0) for site in range(n - 1)
    ) + tuple(BoundaryBond(site, site + 2, 0.4) for site in range(n - 2))
    reversed_bonds = tuple(
        BoundaryBond(bond.second, bond.first, bond.coupling) for bond in forward
    )
    for decision in range(n - 1):
        first = compile_boundary_recoupling_plan(n, forward, decision)
        second = compile_boundary_recoupling_plan(n, reversed_bonds, decision)
        np.testing.assert_array_equal(first.hidden_sites, second.hidden_sites)
        np.testing.assert_array_equal(first.prefix_sites, second.prefix_sites)
        np.testing.assert_allclose(
            endpoint_gauge_matrix(first), endpoint_gauge_matrix(second), atol=1e-14
        )


def test_principal_angles_ignore_internal_gauge_rotation() -> None:
    rng = np.random.default_rng(771)
    raw = rng.normal(size=(7, 3))
    basis, _ = np.linalg.qr(raw)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    sites = np.arange(7, dtype=np.int32)
    result = persistent_subspace_transport(
        basis, sites, basis @ rotation, sites
    )
    np.testing.assert_allclose(result.principal_cosines, 1.0, atol=3e-15)
    np.testing.assert_allclose(result.principal_angles_degrees, 0.0, atol=5e-6)
    assert result.shared_procrustes_overlap == pytest.approx(1.0, abs=2e-15)
    assert result.dimension_penalized_procrustes_overlap == pytest.approx(
        1.0, abs=2e-15
    )
    assert result.normalized_procrustes_residual < 3e-8
