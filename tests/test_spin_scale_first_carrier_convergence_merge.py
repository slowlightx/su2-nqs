from __future__ import annotations

from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_convergence_merge import (
    _load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/remote"
    / "spin_scale_first_carrier_convergence_merge.json"
)


def test_carrier_convergence_merge_contract_and_cli() -> None:
    spec, training, base = _load_contract(CONFIG)
    assert spec["convergence_gate"]["block_size"] == 8
    assert spec["convergence_gate"]["maximum_gpu_peak_memory_mib"] == 70000.0
    assert training["controls"]["total_steps"] == 64
    assert len(base["systems"]) == 2
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-carrier-convergence-merge",
            "--config",
            str(CONFIG),
            "--input-root",
            "input",
            "--output",
            "output",
        ]
    )
    assert parsed.command == "spin-scale-first-carrier-convergence-merge"
