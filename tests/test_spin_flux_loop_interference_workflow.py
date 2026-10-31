from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_flux_loop_interference_confirmation import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects"
    / "clebsch_tree_nqs"
    / "configs"
    / "validation"
    / "spin_flux_loop_interference_confirmation.json"
)
RESULT = (
    ROOT
    / "projects"
    / "clebsch_tree_nqs"
    / "results"
    / "spin_flux_loop_interference_confirmation_v1"
    / "results.json"
)


def test_registered_flux_loop_confirmation_is_fail_closed() -> None:
    parsed = _build_parser().parse_args(
        ["spin-flux-loop-interference-confirmation", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-flux-loop-interference-confirmation"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    mutated = json.loads(CONFIG.read_text())
    mutated["scramble_seeds"] = mutated["scramble_seeds"][:-1]
    with pytest.raises(ValueError, match="16 confirmation"):
        _validate(mutated)


@pytest.mark.skipif(not RESULT.is_file(), reason="official confirmation not run")
def test_sealed_flux_loop_decision_separates_mechanism_from_frontier() -> None:
    payload = json.loads(RESULT.read_text())
    decision = payload["suite"]["decision"]
    assert decision["cross_geometry_mechanism_gate_passed"]
    assert not decision["variational_frontier_gate_passed"]
    assert decision["n36_sampled_preflight_licensed"]
    assert not decision["n36_gpu_production_licensed"]
