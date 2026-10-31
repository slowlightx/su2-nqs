from __future__ import annotations

import numpy as np
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_scale_first_initialization_pilot import (
    _load_pilot_contract,
)
from projects.clebsch_tree_nqs.workflows.spin_scale_first_convergence_merge import (
    _trend_summary,
)


def test_block_trend_summary_detects_sustained_descent() -> None:
    values = []
    for block in range(8):
        values.extend(np.linspace(10.0 - block, 10.2 - block, 8))
    result = _trend_summary(values, block_size=8, edge_block_count=2)
    assert len(result["block_means"]) == 8
    assert result["linear_block_slope"] < 0.0
    assert result["final_minus_initial"] < 0.0
    assert result["negative_adjacent_block_fraction"] == 1.0


def test_convergence_merge_cli_is_registered() -> None:
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-convergence-merge",
            "--config",
            "config.json",
            "--input-root",
            "input",
            "--output",
            "output",
        ]
    )
    assert parsed.command == "spin-scale-first-convergence-merge"


def test_cg256_replay_changes_only_the_solver_iteration_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    original, _, _ = _load_pilot_contract(
        root
        / "projects/clebsch_tree_nqs/configs/remote"
        / "spin_scale_first_single_seed_convergence.json"
    )
    replay, _, _ = _load_pilot_contract(
        root
        / "projects/clebsch_tree_nqs/configs/remote"
        / "spin_scale_first_single_seed_convergence_cg256.json"
    )
    original_controls = dict(original["controls"])
    replay_controls = dict(replay["controls"])
    assert original_controls.pop("cg_max_iterations") == 64
    assert replay_controls.pop("cg_max_iterations") == 256
    assert replay_controls == original_controls
    assert replay["convergence_gate"] == original["convergence_gate"]
