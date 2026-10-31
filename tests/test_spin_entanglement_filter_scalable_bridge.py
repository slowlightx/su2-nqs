from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_entanglement_filter_scalable_bridge import (
    _validate,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_entanglement_filter_scalable_bridge.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/spin_entanglement_filter_scalable_bridge_v1/results.json"
)


def test_scalable_bridge_command_and_preregistered_config() -> None:
    parsed = _build_parser().parse_args(
        ["spin-entanglement-filter-scalable-bridge", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-entanglement-filter-scalable-bridge"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert len(spec["seeds"]) == 3
    assert spec["architecture"]["q_conditioned_attention"] is True
    assert spec["n36_policy"] == "fail_closed_until_exact_a1_and_carrier_exist"


def test_scalable_bridge_result_separates_capacity_from_spatial_obstruction() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["result_sha256"] == canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    state_bundle = ROOT / payload["state_bundle"]["path"]
    assert file_sha256(state_bundle) == payload["state_bundle"]["sha256"]
    assert payload["repeated_ed_performed"] is False
    assert payload["n36_preflight"]["parameter_shapes_match_n16"]
    assert payload["n36_preflight"]["sampler_gate_passed"]
    assert payload["n36_preflight"]["matched_external_projected_carrier_available"]
    assert not payload["n36_preflight"][
        "carrier_residual_common_basis_interface_available"
    ]
    assert not payload["gates"]["n16_compression_passed_both_couplings"]
    assert not payload["gates"]["n36_gpu_production_authorized"]
    assert all(
        row["summary"]["postprojection_repairs_median_variance_vs_carrier"]
        for row in payload["couplings"]
    )
    assert all(
        row["summary"]["minimum_a1_projection_norm"] > 0.999
        for row in payload["couplings"]
    )
