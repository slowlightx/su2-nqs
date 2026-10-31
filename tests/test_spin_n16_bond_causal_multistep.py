from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n16_bond_causal_multistep import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n16_bond_causal_multistep.json"


def test_bond_causal_multistep_command_and_preregistered_seed_split() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n16-bond-causal-multistep", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n16-bond-causal-multistep"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["preregistered_before_run"] is True
    assert len(spec["controls"]["head_seeds"]) == 3
    assert len(spec["controls"]["scramble_seeds"]) == 3
    assert set(spec["controls"]["head_seeds"]).isdisjoint(
        spec["controls"]["untouched_audit_seeds"]
    )
    assert spec["gates"]["minimum_positive_paired_replicates"] == 2
