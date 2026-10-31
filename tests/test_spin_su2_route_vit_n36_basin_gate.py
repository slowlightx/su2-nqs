from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_basin_gate import (
    STRATA,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/"
    "spin_su2_route_vit_n36_basin_gate_global.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/"
    "spin_su2_route_vit_n36_basin_gate_global_v1/results.json"
)


def test_global_basin_gate_config_is_registered() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["sampler"]["chains_per_stratum"] == 8
    assert sum(spec["sampler"]["kernel_weights"]) == 1.0
    assert spec["sampler"]["uniform_floor"] > 0.0


def test_sealed_global_basin_gate_passes_every_admission_check() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["all_gates_passed"]
    assert all(payload["gates"].values())
    assert payload["sampler"]["strata"] == list(STRATA)
    assert payload["sampler"]["space_group_complete_dimer_covering_count"] == 12
    assert payload["checkpoint"]["replay_state_equal"]
    assert payload["checkpoint"]["replay_index_equal"]
    assert payload["checkpoint"]["replay_max_abs_amplitude_error"] == 0.0
    threshold = payload["config"]["gates"]["split_rhat_maximum"]
    assert max(
        row["split_rhat"] for row in payload["correlations"].values()
    ) < threshold
    assert all(
        row["within_two_combined_standard_errors"]
        for row in payload["full_h"]["pairwise_stratum_checks"]
    )
