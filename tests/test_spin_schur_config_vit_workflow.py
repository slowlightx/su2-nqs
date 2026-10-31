from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_schur_config_vit.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_schur_config_vit_v1/results.json"


def test_schur_config_vit_command_and_config_are_registered() -> None:
    parsed = _build_parser().parse_args(
        ["spin-schur-config-vit", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-schur-config-vit"
    spec = json.loads(CONFIG.read_text())
    assert spec["task"] == "spin_schur_config_vit"
    assert spec["architecture"]["magnetic_indices_visible_to_vit"] is False
    assert spec["claim_boundary"]["variational_win"] is False


def test_sealed_local_gate_records_full_rank_and_exact_energy() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["suite"]["passed"]
    for report in payload["suite"]["reports"].values():
        assert report["coefficient_tangent_rank"] == report[
            "complete_singlet_dimension"
        ]
        assert abs(report["best"]["energy_error_total"]) < 2.0e-12
        assert report["best"]["variance_total"] < 2.0e-20
        assert report["best"]["spin_squared_residual_norm"] < 3.0e-14
        assert "not evidence" in report["interpretation"]

