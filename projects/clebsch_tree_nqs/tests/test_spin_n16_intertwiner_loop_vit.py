from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows.spin_n16_intertwiner_loop_vit import (
    _validate,
)
from su2_nqs.frontier.artifacts import canonical_sha256


def _sealed(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    seal = payload["result_sha256"]
    assert canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    ) == seal
    return payload


def test_intertwiner_loop_configs_are_valid_and_artifacts_are_sealed() -> None:
    config_root = ROOT / "projects/clebsch_tree_nqs/configs/validation"
    result_root = ROOT / "projects/clebsch_tree_nqs/results"
    for config_name in (
        "spin_n16_intertwiner_loop_vit.json",
        "spin_n16_intertwiner_loop_vit_audit.json",
    ):
        _validate(json.loads((config_root / config_name).read_text()))
    development = _sealed(
        result_root / "spin_n16_intertwiner_loop_vit_v1/results.json"
    )
    audit = _sealed(
        result_root / "spin_n16_intertwiner_loop_vit_audit_v1/results.json"
    )
    assert audit["audit_parent_result_sha256"] == development["result_sha256"]


def test_intertwiner_loop_mechanism_and_untouched_gates_pass() -> None:
    result_root = ROOT / "projects/clebsch_tree_nqs/results"
    for directory in (
        "spin_n16_intertwiner_loop_vit_v1",
        "spin_n16_intertwiner_loop_vit_audit_v1",
    ):
        payload = _sealed(result_root / directory / "results.json")
        audit = payload["representation_audit"]
        assert audit["canonical_tangent_rank"] < payload["system"][
            "complete_singlet_dimension"
        ]
        assert audit["gamma_a1_tangent_rank"] == payload["system"][
            "gamma_a1_dimension"
        ]
        assert audit["single_frame_intercell_q_positive_weight"] < 1.0e-20
        assert audit["coherent_frame_intercell_q_positive_weight"] > 0.3
        assert payload["promotion_gate_passed"]
        assert all(payload["gates"].values())
