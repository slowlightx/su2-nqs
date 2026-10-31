from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n16_hamiltonian_filtration import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n16_hamiltonian_filtration.json"
)


def test_hamiltonian_filtration_command_and_contract() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n16-hamiltonian-filtration", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n16-hamiltonian-filtration"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["controls"]["audit_batch_size"] == 512
