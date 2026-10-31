from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_four_cell_schur_frontier import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_four_cell_schur_frontier.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_four_cell_schur_frontier_v1/results.json"


def test_four_cell_command_and_config_are_registered() -> None:
    parsed = _build_parser().parse_args(
        ["spin-four-cell-schur-frontier", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-four-cell-schur-frontier"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["architecture"]["magnetic_indices_visible_to_vit"] is False
    assert spec["claim_boundary"]["variational_frontier_win"] is False


def test_sealed_four_cell_gate_records_exact_invariants_and_claim_boundary() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    suite = payload["suite"]
    assert suite["exact_invariants_passed"]
    assert suite["trainability_diagnostics_passed"]
    assert suite["triangular_n12_j2_0125"]["singlet_dimension"] == 132
    assert suite["square_n16_j2_05"]["singlet_dimension"] == 1430
    assert suite["square_n16_j2_05"]["full_basis_gram_materialized"] is False
    assert "no scalable variational-win claim" in suite["claim_boundary"]
