from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_long_range_loop_rvb import _validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_long_range_loop_rvb.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_long_range_loop_rvb_v1/results.json"


def test_registered_long_range_loop_rvb_config_and_cli() -> None:
    parsed = _build_parser().parse_args(
        ["spin-long-range-loop-rvb", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-long-range-loop-rvb"
    _validate(json.loads(CONFIG.read_text()))
    mutated = json.loads(CONFIG.read_text())
    mutated["architecture"]["contingency_matched_krylov_dimension"] = 6
    with pytest.raises(ValueError, match="architecture"):
        _validate(mutated)


@pytest.mark.skipif(not RESULT.is_file(), reason="official loop-RVB gate not run")
def test_official_loop_rvb_gate_is_fail_closed() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["gpu_production_licensed"] is False
    assert payload["suite"]["exact_s0_gamma_a1_dimension"] == 14
    assert payload["suite"]["decision"]["triangular_n36_reopened"] is False
