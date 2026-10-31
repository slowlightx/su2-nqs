from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_resonating_neural_pfaffian import _validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_resonating_neural_pfaffian.json"


def test_resonating_neural_pfaffian_config_and_cli() -> None:
    parsed = _build_parser().parse_args(["spin-resonating-neural-pfaffian", "--config", str(CONFIG)])
    assert parsed.command == "spin-resonating-neural-pfaffian"
    _validate(json.loads(CONFIG.read_text()))
