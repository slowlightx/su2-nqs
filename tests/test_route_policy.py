from __future__ import annotations

import pytest

from su2_nqs.frontier.route_policy import (
    DIRECT_FUSION_STATUS,
    FROZEN_DIRECT_FUSION_MODULE_PREFIXES,
    FROZEN_EXPLORATORY_MODULE_PREFIXES,
    FROZEN_PRODUCTION_MODULE_PREFIXES,
    FrozenResearchRouteError,
    assert_no_frozen_runtime_modules,
    frozen_production_modules,
    require_active_production_method,
)


def test_direct_fusion_route_is_frozen_fail_closed() -> None:
    assert DIRECT_FUSION_STATUS == "frozen_exact_reference_only"
    for name in (
        "direct_fusion",
        "fusion-path GRU",
        "FusionGRU",
        "hubbard_fusion_gru_translation",
    ):
        with pytest.raises(FrozenResearchRouteError):
            require_active_production_method(name)


def test_physical_routes_remain_available() -> None:
    assert require_active_production_method(" Physical_SU2_CG ") == "physical_su2_cg"
    assert require_active_production_method("space_group_physical_tree") == (
        "space_group_physical_tree"
    )
    with pytest.raises(ValueError):
        require_active_production_method("  ")


def test_central_runtime_registry_rejects_direct_and_exploratory_modules() -> None:
    names = (
        "su2_nqs.models.physical_tree",
        "su2_nqs.models.tree_gru",
        "su2_nqs.models.shared_cut",
        "su2_nqs.models.reduced_fusion_dag.internal",
        "su2_nqs.operators.permutation.extra",
    )
    assert frozen_production_modules(names) == (
        "su2_nqs.models.reduced_fusion_dag.internal",
        "su2_nqs.models.shared_cut",
        "su2_nqs.models.tree_gru",
        "su2_nqs.operators.permutation.extra",
    )
    with pytest.raises(FrozenResearchRouteError):
        assert_no_frozen_runtime_modules(names)
    assert assert_no_frozen_runtime_modules(("su2_nqs.su2.fusion_tree",)) == ()
    assert "su2_nqs.su2.fusion_tree" not in FROZEN_PRODUCTION_MODULE_PREFIXES


def test_frozen_route_registries_are_explicit_and_disjoint() -> None:
    assert FROZEN_EXPLORATORY_MODULE_PREFIXES == (
        "su2_nqs.models.shared_cut",
        "su2_nqs.models.spin_clebsch_tree_routes",
        "su2_nqs.models.cell_multiplicity_clebsch_tree",
        "su2_nqs.models.hubbard_reduced_fusion_dag",
        "su2_nqs.models.reduced_fusion_dag",
        "su2_nqs.models.persistent_route_pair_dag",
    )
    assert set(FROZEN_DIRECT_FUSION_MODULE_PREFIXES).isdisjoint(
        FROZEN_EXPLORATORY_MODULE_PREFIXES
    )
    assert FROZEN_PRODUCTION_MODULE_PREFIXES == (
        FROZEN_DIRECT_FUSION_MODULE_PREFIXES + FROZEN_EXPLORATORY_MODULE_PREFIXES
    )


def test_active_clebsch_tree_and_exact_algebra_modules_remain_allowed() -> None:
    active_modules = (
        "su2_nqs.su2.fusion_tree",
        "su2_nqs.su2.clebsch_gordan",
        "su2_nqs.models.physical_tree",
        "su2_nqs.models.spin_clebsch_tree",
        "su2_nqs.models.hubbard_cg",
    )
    assert frozen_production_modules(active_modules) == ()
    assert assert_no_frozen_runtime_modules(active_modules) == ()
