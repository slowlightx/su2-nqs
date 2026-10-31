from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_coherent_plaquette_frame_gate import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_coherent_plaquette_frame_gate.json"


def test_coherent_frame_command_and_preregistered_local_gate() -> None:
    parsed = _build_parser().parse_args(
        ["spin-coherent-plaquette-frame-gate", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-coherent-plaquette-frame-gate"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert len(spec["development_scramble_seeds"]) == 16
    assert spec["remote_training_on_pass"] is False
    assert spec["gates"]["minimum_joint_energy_variance_win_fraction"] == 0.75
