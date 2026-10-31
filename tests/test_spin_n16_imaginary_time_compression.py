from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n16_imaginary_time_compression import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n16_imaginary_time_compression.json"
)


def test_imaginary_time_compression_command_and_preregistered_gate() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n16-imaginary-time-compression", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n16-imaginary-time-compression"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["preregistered_before_run"] is True
    assert spec["controls"]["maximum_fubini_study_distance"] == 0.03
    assert spec["controls"]["loss"] == "hamiltonian_graph_fidelity"
    assert spec["gates"]["minimum_student_energy_recovery_fraction"] == 0.5
