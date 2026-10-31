from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n16_causal_krylov_hybrid import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n16_causal_krylov_hybrid.json"


def test_causal_krylov_hybrid_command_and_rank_contract() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n16-causal-krylov-hybrid", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n16-causal-krylov-hybrid"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["comparison"]["control_extra_column"] == "h4_psi"
    assert spec["comparison"]["require_equal_numerical_rank"] is True
