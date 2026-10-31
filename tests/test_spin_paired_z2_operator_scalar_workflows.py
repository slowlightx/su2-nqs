from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_four_cell_operator_scalar_frontier import (
    _validate as validate_operator,
)
from projects.clebsch_tree_nqs.workflows.spin_four_cell_paired_z2_frontier import (
    _validate as validate_paired,
)
from projects.clebsch_tree_nqs.workflows.spin_four_cell_scalar_correlator_frontier import (
    _validate as validate_scalar,
)
from projects.clebsch_tree_nqs.workflows.spin_four_cell_staged_operator_scalar import (
    _validate as validate_staged,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects/clebsch_tree_nqs"
CASES = (
    (
        "spin-four-cell-paired-z2-frontier",
        PROJECT / "configs/validation/spin_four_cell_paired_z2_development.json",
        PROJECT / "results/spin_four_cell_paired_z2_development_v1/results.json",
        validate_paired,
        "paired_z2_development_failed",
        PROJECT / "workflows/spin_four_cell_paired_z2_frontier.py",
    ),
    (
        "spin-four-cell-scalar-correlator-frontier",
        PROJECT / "configs/validation/spin_four_cell_scalar_correlator_development.json",
        PROJECT / "results/spin_four_cell_scalar_correlator_development_v1/results.json",
        validate_scalar,
        "scalar_correlator_development_failed",
        PROJECT / "workflows/spin_four_cell_scalar_correlator_frontier.py",
    ),
    (
        "spin-four-cell-operator-scalar-frontier",
        PROJECT / "configs/validation/spin_four_cell_operator_scalar_development.json",
        PROJECT / "results/spin_four_cell_operator_scalar_development_v1/results.json",
        validate_operator,
        "operator_scalar_development_failed",
        PROJECT / "workflows/spin_four_cell_operator_scalar_frontier.py",
    ),
    (
        "spin-four-cell-staged-operator-scalar",
        PROJECT
        / "configs/validation/spin_four_cell_staged_operator_scalar_development.json",
        PROJECT
        / "results/spin_four_cell_staged_operator_scalar_development_v1/results.json",
        validate_staged,
        "staged_operator_scalar_development_failed",
        PROJECT / "workflows/spin_four_cell_staged_operator_scalar.py",
    ),
)


def test_new_commands_configs_and_result_seals() -> None:
    for command, config, result, validate, decision, workflow in CASES:
        parsed = _build_parser().parse_args([command, "--config", str(config)])
        assert parsed.command == command
        validate(json.loads(config.read_text()))
        payload = json.loads(result.read_text())
        assert payload["status"] == "complete"
        assert payload["phase"] == "development"
        assert payload["promotion_decision"] == decision
        seal = payload["result_sha256"]
        assert seal == canonical_sha256(
            {key: value for key, value in payload.items() if key != "result_sha256"}
        )
        assert payload["source_manifest_sha256"] == canonical_sha256(
            payload["source_manifest"]
        )
        assert payload["source_manifest"] == transitive_local_source_manifest(
            (workflow.resolve(),),
            root=ROOT,
            package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
        )


def test_paired_z2_is_physical_but_fails_scrambled_gate() -> None:
    payload = json.loads(CASES[0][2].read_text())
    assert payload["suite"]["all_cases_passed"] is False
    square = payload["suite"]["cases"]["square_n16_j2_05"]
    assert square["summary"]["paired_z2_additive_carrier"][
        "median_energy_error_total"
    ] < square["summary"]["no_carrier"]["median_energy_error_total"]
    assert square["gates"]["physical_beats_scrambled_median"] is False
    assert all(
        row["continuous_igg_nullity"] == 0
        for row in square["carrier_diagnostics"]
    )


def test_staged_operator_result_blocks_untouched_audit() -> None:
    payload = json.loads(CASES[3][2].read_text())
    square = payload["suite"]["cases"]["square_n16_j2_05"]
    assert square["paired_physical_win_flags"] == [False, False, False]
    physical = square["summary"]["physical_su2_operator_correlator"]
    scrambled = square["summary"]["scrambled_operator_correlator"]
    assert scrambled["median_energy_error_total"] < physical[
        "median_energy_error_total"
    ]
    assert payload["suite"]["all_cases_passed"] is False
