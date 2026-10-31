from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.spin_scalar_operator_frontier import (
    TASK,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_scalar_operator_frontier.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_scalar_operator_frontier_v1/results.json"


def test_scalar_operator_scan_config_covers_qsl_and_stripe_side() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["task"] == TASK
    assert 0.125 in spec["couplings"]
    assert 0.175 in spec["couplings"]


def test_scalar_operator_result_records_matched_krylov_decision_when_present() -> None:
    if not RESULT.exists():
        return
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["maximum_exact_basis_dimension"] == 12870
    assert payload["decision"]["qsl_target_0p125_licenses_shell_attention"] is False
    assert payload["decision"]["qsl_target_0p125_licenses_e2_energy_head"] is False
    assert payload["decision"]["retain_e2_as_response_probe"] is True
