from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_rvb_covering_frontier import (
    _one_sided_sign_probability,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects"
    / "clebsch_tree_nqs"
    / "configs"
    / "validation"
    / "spin_rvb_covering_frontier.json"
)
RESULT = (
    ROOT
    / "projects"
    / "clebsch_tree_nqs"
    / "results"
    / "spin_rvb_covering_frontier_v1"
    / "results.json"
)


def test_registered_rvb_covering_frontier_config_is_fail_closed() -> None:
    parsed = _build_parser().parse_args(
        ["spin-rvb-covering-frontier", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-rvb-covering-frontier"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    mutated = json.loads(CONFIG.read_text())
    mutated["architecture"]["depth"] = 1
    with pytest.raises(ValueError, match="depth"):
        _validate(mutated)


def test_exact_one_sided_sign_probability() -> None:
    assert _one_sided_sign_probability(8, 8) == pytest.approx(1.0 / 256.0)
    assert _one_sided_sign_probability(7, 7) == pytest.approx(1.0 / 128.0)


def test_sealed_rvb_covering_frontier_records_the_local_win_and_boundary() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["oracle"]["exact_sector_gate_passed"]
    assert payload["oracle"]["rvb_direction_beyond_two_lanczos_present_both_cases"]
    assert payload["attention"]["exact_spin_gate_passed"]
    assert payload["attention"]["full_coefficient_tangent_rank_both_cases"]
    decision = payload["decision"]
    assert decision["local_energy_efficiency_gate_passed"]
    assert decision["square_paired_energy"]["wins"] == 8
    assert decision["square_paired_energy"]["losses"] == 0
    assert decision["large_size_variational_win_established"] is False
