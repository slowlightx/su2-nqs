import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.spin_n36_many_cell_simplex_gate import (
    TASK,
    _validate,
)
from su2_nqs.frontier.artifacts import canonical_sha256


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n36_many_cell_simplex_gate.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/spin_n36_many_cell_simplex_gate_v4/results.json"
)


def test_simplex_gate_contract_is_local_fail_closed_and_matched() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["task"] == TASK
    assert spec["model"]["virtual_contraction"] == "exact_local_F_q23_q12"
    assert spec["model"]["magnetic_indices_visible"] is False
    assert spec["model"]["backbone_frozen_for_local_gate"] is True
    assert spec["controls"]["total_steps"] == 2
    assert len(spec["controls"]["audit_seeds"]) == 5
    assert spec["decision"]["remote_production_authorized_by_this_config"] is False


def test_sealed_simplex_gate_records_the_narrow_pass_and_variational_stop() -> None:
    payload = json.loads(RESULT.read_text())
    stored = payload.pop("result_sha256")
    assert stored == canonical_sha256(payload)
    assert stored == "11cb772101140819de5c3b0dbb05ee568f9c2dc8854daefc1490276faa90f028"
    assert payload["gates"]["local_prototype_complete"] is True
    assert payload["gates"]["physical_beats_scramble"] is True
    assert payload["gates"]["remote_production_authorized"] is False
    paired = payload["paired_audit"]
    assert paired["physical_energy_win_count"] == 5
    assert paired["physical_joint_win_count"] == 4
    assert paired["physical_minus_scramble_energy_per_site_mean"] < 0.0
    assert payload["aggregate_multi_channel_fraction"]["physical_simplex"] > 0.05
