from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_causal_scan import (
    _causal_comparisons,
    _scrambled_spatial_config,
)
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_amplitudes,
)


jax.config.update("jax_enable_x64", True)


def test_scrambled_tokens_preserve_nested_initial_state_and_token_multiset() -> None:
    spatial, _ledger = spatial_cg_route_spec(
        "triangular",
        2,
        4,
        feature_dim=3,
        maximum_intermediate_q=4,
        initialization_scale=0.4,
    )
    scrambled, ledger = _scrambled_spatial_config(spatial)
    assert ledger["token_multiset_preserved"] is True
    assert ledger["fixed_route_count"] == 0
    np.testing.assert_allclose(
        np.sort(np.asarray(scrambled.route_features), axis=0),
        np.sort(np.asarray(spatial.route_features), axis=0),
        atol=0.0,
        rtol=0.0,
    )
    config = SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=8,
        head_count=2,
        modulation_rank=2,
    )
    scrambled_config = SU2RouteViTConfig(
        spatial=scrambled,
        embedding_dim=8,
        head_count=2,
        modulation_rank=2,
    )
    params = init_su2_route_vit(jax.random.PRNGKey(7), config)
    spins = jnp.asarray(
        [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]],
        dtype=jnp.int32,
    )
    np.testing.assert_allclose(
        su2_route_vit_amplitudes(params, spins, config),
        su2_route_vit_amplitudes(params, spins, scrambled_config),
        atol=0.0,
        rtol=0.0,
    )


def test_causal_summary_uses_paired_seed_differences() -> None:
    def arm(name: str, energy: float) -> dict[str, object]:
        return {"arm_id": name, "metrics": {"energy": energy}}

    cases = [
        {
            "case_id": "case",
            "trials": [
                {
                    "seed": 1,
                    "arms": [
                        arm("route_vit_conditioned_d4", -2.0),
                        arm("route_vit_scrambled_tokens_d4", -1.9),
                        arm("route_local_conditioned_d4", -2.1),
                    ],
                },
                {
                    "seed": 2,
                    "arms": [
                        arm("route_vit_conditioned_d4", -3.0),
                        arm("route_vit_scrambled_tokens_d4", -2.8),
                        arm("route_local_conditioned_d4", -2.7),
                    ],
                },
            ],
        }
    ]
    result = _causal_comparisons(cases)["case"]
    assert result["route_vit_scrambled_tokens_d4"]["global_win_count"] == 2
    assert result["route_local_conditioned_d4"]["global_win_count"] == 1
