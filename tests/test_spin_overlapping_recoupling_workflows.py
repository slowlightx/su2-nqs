from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_four_cell_overlapping_recoupling import (
    _validate as validate_overlapping,
)
from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_ritz import (
    _validate as validate_ritz,
)
from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_variable_projection import (
    _validate as validate_variable_projection,
)
from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_variable_projection_audit import (
    _validate as validate_audit,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects/clebsch_tree_nqs"
CASES = (
    (
        "spin-four-cell-overlapping-recoupling",
        PROJECT
        / "configs/validation/spin_four_cell_overlapping_recoupling_development.json",
        PROJECT
        / "results/spin_four_cell_overlapping_recoupling_development_v1/results.json",
        PROJECT / "workflows/spin_four_cell_overlapping_recoupling.py",
        validate_overlapping,
        "promotion_decision",
        "overlapping_recoupling_development_failed",
    ),
    (
        "spin-four-cell-recoupling-ritz",
        PROJECT
        / "configs/validation/spin_four_cell_recoupling_ritz_calibration.json",
        PROJECT
        / "results/spin_four_cell_recoupling_ritz_calibration_v1/results.json",
        PROJECT / "workflows/spin_four_cell_recoupling_ritz.py",
        validate_ritz,
        "decision",
        "physical_subspace_failed_stop_recoupling_route",
    ),
    (
        "spin-four-cell-recoupling-variable-projection",
        PROJECT
        / "configs/validation/spin_four_cell_recoupling_variable_projection_development.json",
        PROJECT
        / "results/spin_four_cell_recoupling_variable_projection_development_v1/results.json",
        PROJECT
        / "workflows/spin_four_cell_recoupling_variable_projection.py",
        validate_variable_projection,
        "promotion_decision",
        "freeze_variable_projection_and_preregister_untouched_audit",
    ),
    (
        "spin-four-cell-recoupling-variable-projection-audit",
        PROJECT
        / "configs/validation/spin_four_cell_recoupling_variable_projection_audit.json",
        PROJECT
        / "results/spin_four_cell_recoupling_variable_projection_audit_v1/results.json",
        PROJECT
        / "workflows/spin_four_cell_recoupling_variable_projection_audit.py",
        validate_audit,
        "promotion_decision",
        "hold_n36_variable_projection_audit_failed",
    ),
)


def test_recoupling_commands_configs_seals_and_current_sources() -> None:
    for command, config, result, workflow, validate, key, decision in CASES:
        parsed = _build_parser().parse_args([command, "--config", str(config)])
        assert parsed.command == command
        validate(json.loads(config.read_text()))
        payload = json.loads(result.read_text())
        assert payload["status"] == "complete"
        assert payload[key] == decision
        assert payload["result_sha256"] == canonical_sha256(
            {name: value for name, value in payload.items() if name != "result_sha256"}
        )
        assert payload["source_manifest_sha256"] == canonical_sha256(
            payload["source_manifest"]
        )
        assert payload["source_manifest"] == transitive_local_source_manifest(
            (workflow.resolve(),),
            root=ROOT,
            package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
        )


def test_variable_projection_development_passes_but_untouched_audit_blocks_n36() -> None:
    development = json.loads(CASES[2][2].read_text())
    audit = json.loads(CASES[3][2].read_text())
    assert development["suite"]["all_cases_passed"] is True
    assert audit["frozen_development_result_sha256"] == development["result_sha256"]
    assert audit["suite"]["all_cases_passed"] is False
    square = audit["suite"]["cases"]["square_n16_j2_05"]
    physical = square["summary"]["physical_overlapping_variable_projection"]
    scrambled = square["summary"]["a1_scrambled_variable_projection"]
    assert physical["median_energy_error_total"] < scrambled[
        "median_energy_error_total"
    ]
    assert physical["median_variance_total"] < scrambled["median_variance_total"]
    assert square["paired_physical_win_flags"] == [False, True, False]
    assert square["gates"]["physical_all_seeds_below_1e-5"] is False
    assert square["gates"]["physical_beats_scrambled_every_paired_seed"] is False


def test_frozen_ritz_calibration_separates_physical_subspace_from_training() -> None:
    calibration = json.loads(CASES[1][2].read_text())
    square = calibration["suite"]["cases"]["square_n16_j2_05"]
    assert square["physical_hamiltonian_span_fraction"] > 1.0 - 1.0e-12
    assert square["scrambled_hamiltonian_span_fraction"] < 0.03
    assert square["paired_physical_win_flags"] == [True, True, True]
    assert square["physical_median_gain_beyond_one_lanczos"] > 0.0
