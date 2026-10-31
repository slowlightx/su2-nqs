from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_triangular_n16_state_manifold import (
    _validate,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_triangular_n16_state_manifold.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_triangular_n16_state_manifold_v1/results.json"


def test_state_manifold_command_and_preregistered_config() -> None:
    parsed = _build_parser().parse_args(
        ["spin-triangular-n16-state-manifold", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-triangular-n16-state-manifold"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["preregistered_before_training"] is True
    assert spec["frozen_state_source"]["exact_ground_role"] == "post_training_audit_only"
    assert spec["architecture"]["singlet_dimension"] == 1430
    assert spec["optimization"]["seeds"] == [2111, 2112, 2113]


def test_sealed_state_manifold_result_records_variational_win_without_new_ed() -> None:
    payload = json.loads(RESULT.read_text())
    seal = payload["result_sha256"]
    assert seal == canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    assert payload["repeated_ed_performed"] is False
    state_bundle = ROOT / payload["state_bundle"]["path"]
    assert file_sha256(state_bundle) == payload["state_bundle"]["sha256"]
    suite = payload["suite"]
    assert suite["complete_singlet_dimension"] == 1430
    assert suite["space_group_order"] == 192
    assert suite["promotion_gates_passed"]
    anchor = suite["frozen_operator_vit_comparator"]
    anchored = suite["summary"]["operator_anchored_complete_multiplicity_vit"]
    assert anchored["median_energy_total"] < anchor["energy_total"]
    assert anchored["median_variance_total"] < anchor["variance_total"]
    assert max(anchored["all_seed_matched_step_energy_errors_total"]) < anchor[
        "energy_error_total"
    ]
