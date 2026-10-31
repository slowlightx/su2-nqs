from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_sequential_cell_gate import _validate
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_sequential_cell_gate.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_sequential_cell_gate_v1/results.json"


def test_sequential_cell_command_and_preregistered_config() -> None:
    parsed = _build_parser().parse_args(
        ["spin-sequential-cell-gate", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-sequential-cell-gate"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert len(spec["matched_arm"]["seeds"]) == 3
    assert spec["n36_preflight"]["run_energy_training_only_if_all_gates_pass"] is True


def test_sealed_sequential_cell_result_records_win_and_fail_closed_boundary() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["result_sha256"] == canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    state_bundle = ROOT / payload["state_bundle"]["path"]
    assert file_sha256(state_bundle) == payload["state_bundle"]["sha256"]
    assert payload["repeated_ed_performed"] is False
    assert payload["exact_recoupling"]["singlet_dimension"] == 1430
    assert payload["target_replay"]["energy_abs_error"] < 2.0e-11
    assert payload["matched_summary"]["fidelity_gain_over_root_first"] > 0.05
    assert payload["matched_summary"]["energy_gain_over_root_first"] > 0.05
    assert payload["capacity_trial"]["fidelity"] > 0.8
    gates = payload["gate_results"]
    assert gates["matched_cell_factorization_win"]
    assert gates["n36_autoregressive_sampler_smoke"]
    assert not gates["sealed_state_reproduction"]
    assert not gates["physical_carrier_reproduction"]
    assert not gates["scalable_full_gamma_a1_projector"]
    assert not payload["n36_gpu_production_authorized"]
    assert payload["n36_preflight"]["energy_training_started"] is False
