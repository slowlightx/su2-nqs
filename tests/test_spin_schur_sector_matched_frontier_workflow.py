from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_four_cell_additive_carrier import (
    _validate as validate_carrier,
)
from projects.clebsch_tree_nqs.workflows.spin_four_cell_additive_carrier_confirmation import (
    _validate as validate_carrier_confirmation,
)
from projects.clebsch_tree_nqs.workflows.spin_schur_sector_matched_frontier import (
    _validate as validate_matched,
)


ROOT = Path(__file__).resolve().parents[1]
MATCHED_CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_schur_sector_matched_frontier.json"
MATCHED_RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_schur_sector_matched_frontier_v1/results.json"
CARRIER_CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_four_cell_additive_carrier.json"
CARRIER_RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_four_cell_additive_carrier_v1/results.json"
CONFIRMATION_CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_four_cell_additive_carrier_confirmation.json"
CONFIRMATION_RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_four_cell_additive_carrier_confirmation_v1/results.json"


def test_new_commands_and_configs_are_registered() -> None:
    for command, config in (
        ("spin-schur-sector-matched-frontier", MATCHED_CONFIG),
        ("spin-four-cell-additive-carrier", CARRIER_CONFIG),
        ("spin-four-cell-additive-carrier-confirmation", CONFIRMATION_CONFIG),
    ):
        parsed = _build_parser().parse_args([command, "--config", str(config)])
        assert parsed.command == command
    validate_matched(json.loads(MATCHED_CONFIG.read_text()))
    validate_carrier(json.loads(CARRIER_CONFIG.read_text()))
    validate_carrier_confirmation(json.loads(CONFIRMATION_CONFIG.read_text()))


def test_strict_sector_heads_pass_and_n36_remains_held() -> None:
    payload = json.loads(MATCHED_RESULT.read_text())
    assert payload["status"] == "complete"
    sector = payload["sector_head_gate"]
    assert sector["exact_gates_passed"]
    assert sector["trainability_gates_passed"]
    assert set(sector["heads"]) == {"A1", "E2", "A2", "K", "M"}
    assert all(
        row["tangent_projector_closure"] < 5.0e-12
        for row in sector["heads"].values()
    )
    assert payload["matched_frontier"]["promotion_gates_passed"] is False
    assert payload["promotion_decision"] == "hold_n36_improve_local_controls_or_backbone"


def test_physical_additive_carrier_repairs_square_slow_basin() -> None:
    payload = json.loads(CARRIER_RESULT.read_text())
    assert payload["status"] == "complete"
    suite = payload["suite"]
    square = suite["cases"]["square_n16_j2_05"]
    assert square["gates_passed"]
    physical = square["summary"]["physical_additive_carrier"]
    raw = square["summary"]["no_carrier"]
    scrambled = square["summary"]["scrambled_additive_carrier"]
    assert physical["maximum_energy_error_total"] < 1.0e-5
    assert physical["median_energy_error_total"] < raw["median_energy_error_total"]
    assert physical["median_energy_error_total"] < scrambled["median_energy_error_total"]
    assert suite["all_cases_passed"] is False
    assert payload["promotion_decision"] == "carrier_mechanism_not_robust"


def test_untouched_seeds_reject_n36_carrier_promotion() -> None:
    payload = json.loads(CONFIRMATION_RESULT.read_text())
    assert payload["status"] == "complete"
    suite = payload["suite"]
    assert suite["fresh_seeds"] == [101, 102, 103]
    assert suite["confirmation_passed"] is False
    assert payload["promotion_decision"] == "hold_n36_carrier_confirmation_failed"

    square = suite["suite"]["cases"]["square_n16_j2_05"]["summary"]
    physical = square["physical_additive_carrier"]
    raw = square["no_carrier"]
    scrambled = square["scrambled_additive_carrier"]
    assert physical["median_energy_error_total"] < raw["median_energy_error_total"]
    assert physical["maximum_energy_error_total"] < raw["maximum_energy_error_total"]
    assert physical["maximum_energy_error_total"] > 1.0e-5
    assert scrambled["median_energy_error_total"] < physical["median_energy_error_total"]
