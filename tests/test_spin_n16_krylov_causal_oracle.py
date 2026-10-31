from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n16_krylov_causal_oracle import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n16_krylov_causal_oracle.json"


def test_krylov_causal_oracle_command_and_preregistered_gate() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n16-krylov-causal-oracle", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n16-krylov-causal-oracle"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["preregistered_before_run"] is True
    assert len(spec["controls"]["scramble_seeds"]) == 3
    assert spec["gates"]["maximum_physical_energy_above_three_lanczos_per_site"] == 1e-8
