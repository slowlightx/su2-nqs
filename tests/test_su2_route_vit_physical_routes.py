from __future__ import annotations

import jax

from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_physical_routes import (
    _physical_comparisons,
    _spatial_banks,
)
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_parameter_count,
)


def _model() -> dict[str, object]:
    return {
        "base_feature_dim": 4,
        "maximum_intermediate_q": 8,
        "initialization_scale": 1.0,
        "embedding_dim": 12,
        "head_count": 3,
        "depth": 1,
        "mlp_ratio": 2,
        "modulation_rank": 2,
        "modulation_scale": 0.15,
    }


def test_complete_dimer_bank_adds_route_without_trainable_parameters() -> None:
    base, complete, ledger = _spatial_banks(_model())
    assert base.n_routes == 8
    assert complete.n_routes == 9
    assert ledger["added_route"] == "diagonal_dimers"
    assert ledger["added_trainable_parameters"] == 0
    common = dict(
        embedding_dim=12,
        head_count=3,
        depth=1,
        mlp_ratio=2,
        modulation_rank=2,
        modulation_scale=0.15,
        initialization_scale=1.0,
    )
    base_params = init_su2_route_vit(
        jax.random.PRNGKey(11), SU2RouteViTConfig(spatial=base, **common)
    )
    complete_params = init_su2_route_vit(
        jax.random.PRNGKey(11), SU2RouteViTConfig(spatial=complete, **common)
    )
    assert su2_route_vit_parameter_count(base_params) == su2_route_vit_parameter_count(
        complete_params
    )


def test_physical_comparison_records_zero_parameter_delta() -> None:
    def arm(name: str, energy: float, count: int = 12) -> dict[str, object]:
        return {
            "arm_id": name,
            "parameter_count": count,
            "metrics": {"energy": energy},
        }

    cases = [
        {
            "case_id": "triangular",
            "trials": [
                {
                    "seed": 3,
                    "arms": [
                        arm("route_vit_base_8_routes_d4", -1.0),
                        arm("route_vit_dimer_complete_9_routes_d4", -1.1),
                    ],
                }
            ],
        }
    ]
    row = _physical_comparisons(cases)["triangular"]
    assert row["complete_bank_win_count"] == 1
    assert row["per_seed"][0]["added_trainable_parameters"] == 0
