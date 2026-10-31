from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n16_imaginary_time_tangent import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n16_imaginary_time_tangent.json"
)


def test_imaginary_time_tangent_command_and_preregistered_gate() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n16-imaginary-time-tangent", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n16-imaginary-time-tangent"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["preregistered_before_run"] is True
    assert spec["controls"]["lsqr_max_iterations"] == 48
    assert spec["gates"]["minimum_feature_loss_recovery_fraction"] == 0.5
