from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_neural_backflow_pfaffian import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_neural_backflow_pfaffian.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/spin_neural_backflow_pfaffian_v1/results.json"
)


def test_registered_neural_backflow_gate_is_fail_closed() -> None:
    parsed = _build_parser().parse_args(
        ["spin-neural-backflow-pfaffian", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-neural-backflow-pfaffian"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    mutated = json.loads(CONFIG.read_text())
    mutated["architecture"]["compact_matched_krylov_dimension"] = 4
    with pytest.raises(ValueError, match="architecture"):
        _validate(mutated)


@pytest.mark.skipif(not RESULT.is_file(), reason="official backflow gate not run")
def test_sealed_backflow_gate_never_directly_licenses_gpu_production() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["gpu_production_licensed"] is False
    assert payload["suite"]["exact_s0_gamma_a1_dimension"] == 14
