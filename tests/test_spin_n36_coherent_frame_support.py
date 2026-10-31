from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n36_coherent_frame_support import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n36_coherent_frame_support.json"


def test_n36_coherent_frame_command_and_fail_closed_contract() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n36-coherent-frame-support", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n36-coherent-frame-support"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["remote_training_on_pass"] is False
    assert len(spec["controls"]["sample_seeds"]) == 4
    assert spec["gates"]["minimum_paired_casimir_z_score"] == 2.0
