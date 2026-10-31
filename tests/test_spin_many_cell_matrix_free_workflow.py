from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_many_cell_matrix_free import _validate
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_many_cell_matrix_free.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_many_cell_matrix_free_v1/results.json"


def test_many_cell_command_and_fail_closed_config() -> None:
    parsed = _build_parser().parse_args(
        ["spin-many-cell-matrix-free", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-many-cell-matrix-free"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["many_cell_architecture"]["additive_carrier"] is True
    assert spec["many_cell_architecture"]["magnetic_indices_visible_to_vit"] is False
    assert spec["n36_preflight"]["run_training_only_if_all_gates_pass"] is True


def test_sealed_many_cell_result_separates_exact_and_neural_gates() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["result_sha256"] == canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    assert payload["repeated_ed_performed"] is False
    state_bundle = ROOT / payload["state_bundle"]["path"]
    assert file_sha256(state_bundle) == payload["state_bundle"]["sha256"]
    replay = payload["n16_exact_replay"]
    assert replay["physical_amplitude_max_abs_error"] < 2.0e-12
    assert replay["matrix_free_vs_sealed_energy_abs_error"] < 2.0e-11
    assert replay["matrix_free_variance_total"] < 5.0e-18
    gates = payload["gate_results"]
    assert gates["n16_physical_amplitude_replay"]
    assert gates["n16_matrix_free_energy"]
    assert gates["n16_n36_parameter_shape_transfer"]
    assert gates["n36_autoregressive_sampler_smoke"]
    assert not gates["transferable_vit_distillation"]
    assert not gates["scalable_full_gamma_a1_projector"]
    assert not payload["n36_gpu_production_authorized"]
    assert payload["n36_preflight"]["energy_training_started"] is False

