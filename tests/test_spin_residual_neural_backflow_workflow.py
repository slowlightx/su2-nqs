from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_residual_neural_backflow import _validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_residual_neural_backflow.json"


def test_registered_residual_backflow_config_and_cli() -> None:
    parsed = _build_parser().parse_args(
        ["spin-residual-neural-backflow", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-residual-neural-backflow"
    _validate(json.loads(CONFIG.read_text()))
