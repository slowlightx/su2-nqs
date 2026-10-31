from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n36_multibody_frame_resonance import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n36_multibody_frame_resonance.json"
)
SEALED = (
    ROOT
    / "projects/clebsch_tree_nqs/results/spin_n36_multibody_frame_resonance_v1/results.json"
)


def test_multibody_frame_command_and_preregistered_contract() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n36-multibody-frame-resonance", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n36-multibody-frame-resonance"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert len(spec["controls"]["untouched_audit_seeds"]) == 4
    assert len(spec["controls"]["scramble_seeds"]) == 4
    assert spec["physical_trust"]["maximum_fubini_study_distance"] == 0.03
    assert spec["remote_training_on_pass"] is False


def test_sealed_multibody_frame_result_preserves_mixed_verdict() -> None:
    result = json.loads(SEALED.read_text())
    assert result["decision"] == "freeze_coherent_local_singlet_resonance_route"
    assert result["remote_pilot_reapplication_allowed"] is False
    cases = {row["case_id"]: row for row in result["cases"]}
    square = cases["square_6x6_j2_0.5"]
    triangular = cases["triangular_6x6_j2_0.125"]
    assert square["passed"] is True
    assert triangular["passed"] is False
    assert (
        square["physical_multibody_bank"]["audit_summary"]
        ["paired_energy_change_per_site_mean"]
        < 0.0
    )
    assert (
        triangular["physical_multibody_bank"]["audit_summary"]
        ["maximum_fubini_study_distance"]
        > 0.3
    )
