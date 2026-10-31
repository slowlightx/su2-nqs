from __future__ import annotations

import numpy as np

from su2_nqs.lattices.pair_balanced_tree import pair_balanced_fusion_tree
from su2_nqs.lattices.physical_route_banks import (
    triangular_dimer_complete_route_orders,
)
from su2_nqs.lattices.route_carrier_templates import (
    triangular_route_carrier_templates,
)


def test_n36_physical_and_generic_route_carriers_are_full_rank_and_disjoint() -> None:
    routes = triangular_dimer_complete_route_orders(6, 6)
    result = triangular_route_carrier_templates(
        pair_balanced_fusion_tree(36),
        routes,
        ((6, 0), (0, 6)),
        generic_seed=23,
    )
    assert result.route_names == tuple(routes)
    assert len(result.a1) == len(result.e2) == 3
    assert len(result.generic) == 6
    physical = np.asarray((*result.a1, *result.e2), dtype=np.float64).reshape(6, -1)
    generic = np.asarray(result.generic, dtype=np.float64).reshape(6, -1)
    np.testing.assert_allclose(physical @ physical.T, np.eye(6), atol=2.0e-12)
    np.testing.assert_allclose(generic @ generic.T, np.eye(6), atol=2.0e-12)
    np.testing.assert_allclose(physical @ generic.T, 0.0, atol=2.0e-12)
    assert np.linalg.matrix_rank(np.vstack((physical, generic)), tol=1.0e-11) == 12
    assert all(label.startswith(("A1:", "E2xE2->A1:")) for label in result.physical_labels)


def test_n36_carrier_templates_are_seeded_before_any_energy_evaluation() -> None:
    routes = triangular_dimer_complete_route_orders(6, 6)
    kwargs = (pair_balanced_fusion_tree(36), routes, ((6, 0), (0, 6)))
    first = triangular_route_carrier_templates(*kwargs, generic_seed=17)
    second = triangular_route_carrier_templates(*kwargs, generic_seed=17)
    third = triangular_route_carrier_templates(*kwargs, generic_seed=18)
    assert first.a1 == second.a1
    assert first.e2 == second.e2
    assert first.generic == second.generic
    assert first.generic != third.generic

