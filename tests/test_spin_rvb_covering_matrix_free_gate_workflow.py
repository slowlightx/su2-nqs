from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_rvb_covering_matrix_free_gate import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects"
    / "clebsch_tree_nqs"
    / "configs"
    / "validation"
    / "spin_rvb_covering_matrix_free_gate.json"
)
RESULT = (
    ROOT
    / "projects"
    / "clebsch_tree_nqs"
    / "results"
    / "spin_rvb_covering_matrix_free_gate_v1"
    / "results.json"
)


def test_registered_matrix_free_gate_is_fail_closed() -> None:
    parsed = _build_parser().parse_args(
        ["spin-rvb-covering-matrix-free-gate", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-rvb-covering-matrix-free-gate"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    mutated = json.loads(CONFIG.read_text())
    mutated["overlap_loop_gas"]["sample_steps"] = 64
    with pytest.raises(ValueError, match="overlap_loop_gas"):
        _validate(mutated)


@pytest.mark.skipif(not RESULT.is_file(), reason="official matrix-free gate not run")
def test_sealed_gate_records_control_and_negative_promotion_decision() -> None:
    payload = json.loads(RESULT.read_text())
    decision = payload["decision"]
    assert payload["status"] == "complete"
    assert decision["exact_estimator_gate_passed"]
    assert decision["local_cpu_throughput_gate_passed"]
    assert decision["square_bipartite_control_gate_passed"]
    assert decision["triangular_equal_amplitude_estimator_gate_passed"] is False
    assert decision["phase_carrier_transfer_gate_passed"] is False
    assert decision["gpu_training_authorized"] is False
    assert payload["square_n36_overlap_loop_gas_control"]["average_sign"] == 1.0
