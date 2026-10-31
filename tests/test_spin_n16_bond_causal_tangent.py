from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n16_bond_causal_tangent import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n16_bond_causal_tangent.json"
)


def test_bond_causal_tangent_command_and_preregistered_controls() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n16-bond-causal-tangent", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n16-bond-causal-tangent"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["preregistered_before_run"] is True
    assert len(spec["controls"]["scramble_seeds"]) == 3
    assert spec["gates"]["minimum_physical_over_scramble_recovery"] == 0.02
