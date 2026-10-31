from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_carrier_cell_scalar_circuit import (
    _validate,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_carrier_cell_scalar_circuit.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_carrier_cell_scalar_circuit_v1/results.json"


def test_carrier_cell_command_and_falsification_config() -> None:
    parsed = _build_parser().parse_args(
        ["spin-carrier-cell-scalar-circuit", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-carrier-cell-scalar-circuit"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["positive_claim_audit"] is False
    assert spec["run_full_coupling_scan_only_if_calibration_passes"] is True


def test_sealed_carrier_cell_result_records_fail_closed_decision() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["result_sha256"] == canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    bundle = ROOT / payload["state_bundle"]["path"]
    assert file_sha256(bundle) == payload["state_bundle"]["sha256"]
    assert payload["repeated_ed_performed"] is False
    assert payload["frontier"]["representation"]["gamma_a1_dimension"] == 14
    assert payload["frontier"]["representation"]["parameter_count"] == 16
    assert not payload["frontier"]["gates"]["license_n36_matrix_free_pilot"]
    assert not payload["full_coupling_scan_started"]
    assert not payload["n36_gpu_production_authorized"]
    assert payload["decision"] == "stop_cell_scalar_route_small_a1_space_not_discriminating"
