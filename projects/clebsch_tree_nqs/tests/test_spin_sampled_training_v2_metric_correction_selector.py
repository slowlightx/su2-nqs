from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2 as scan_v2
from projects.clebsch_tree_nqs.workflows import (
    spin_sampled_training_v2_metric_correction_selector as correction,
)
from projects.clebsch_tree_nqs.workflows import (
    spin_sampled_training_v2_selector as literal_selector,
)
from su2_nqs.frontier.provenance import file_sha256


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled"
    / "spin_6x6_center_only_lr_metric_correction_selector.json"
)


def _leaf_differences(
    left: object, right: object, prefix: str = ""
) -> list[tuple[str, object, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[tuple[str, object, object]] = []
        for key in sorted(set(left) | set(right)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                differences.append((name, None, right[key]))
            elif key not in right:
                differences.append((name, left[key], None))
            else:
                differences.extend(_leaf_differences(left[key], right[key], name))
        return differences
    return [] if left == right else [(prefix, left, right)]


def _synthetic_candidate(*, nontrivial: float = 0.5) -> dict[str, object]:
    conditional = 0.2
    rows = [
        {
            "sampler": {
                "accepted_move_fraction": conditional * nontrivial,
                "conditional_move_acceptance": conditional,
                "nontrivial_proposal_fraction": nontrivial,
                "zero_amplitude_chain_fraction": 0.0,
                "chain_snapshot_move_fraction": 0.9,
            },
            "adam": {"relative_update_norm": 0.001},
        }
        for _ in range(50)
    ]
    arms: dict[str, object] = {}
    for backbone in correction.EXPECTED_ARMS:
        report = {
            "status_complete_without_hard_failure": True,
            "all_readiness_gates_passed": True,
            "stability_gate_passed": False,
            "last10_accepted_move_fraction_median": conditional * nontrivial,
            "relative_update_quantile_value": 0.001,
            "energy_lowering_per_site": 0.04,
            "energy_lowering_paired_chain_se": 0.005,
            "energy_lowering_paired_z": 8.0,
            "energy_endpoint_gate_passed": True,
            "variance_worsening_per_site": -0.02,
            "variance_worsening_paired_chain_se": 0.005,
            "variance_worsening_paired_z": -4.0,
            "variance_endpoint_gate_passed": True,
            "primary_arm_eligible": False,
            "fallback_arm_eligible": False,
            "chain_count": 32,
        }
        arms[backbone] = {
            "trial": {"history": deepcopy(rows)},
            "metrics": {"report": report},
        }
    cases = {
        case_id: {"arms": deepcopy(arms)} for case_id in correction.EXPECTED_CASES
    }
    return {"cases": cases}


def test_registered_policy_changes_only_the_acceptance_field_and_cli_exists() -> None:
    spec = json.loads(CONFIG.read_text())
    correction._validate_spec(spec)
    old = scan_v2._expected_selection_policy(50)
    new = spec["selection_policy"]
    differences = _leaf_differences(old, new)
    assert differences == [
        (
            "stability_gates.last10_accepted_move_fraction_median_minimum",
            0.15,
            None,
        ),
        (
            "stability_gates.last10_conditional_move_acceptance_median_minimum",
            None,
            0.15,
        ),
    ]
    source = (ROOT / "projects/clebsch_tree_nqs/run.py").read_text()
    assert '"sampled-optimizer-select-v2-metric-correction"' in source
    assert "run_spin_sampled_training_v2_metric_correction_selector" in source


def test_corrected_metric_uses_conditional_and_secondary_audit_is_not_a_gate() -> None:
    spec = json.loads(CONFIG.read_text())
    candidate = _synthetic_candidate(nontrivial=0.3)
    correction._apply_corrected_metric(candidate, spec)
    for case_id in correction.EXPECTED_CASES:
        for backbone in correction.EXPECTED_ARMS:
            report = candidate["cases"][case_id]["arms"][backbone]["metrics"][
                "report"
            ]
            assert report["last10_accepted_move_fraction_median"] == pytest.approx(
                0.06
            )
            assert report["last10_conditional_move_acceptance_median"] == pytest.approx(
                0.2
            )
            assert report["literal_v1_stability_gate_passed"] is False
            assert report["corrected_stability_gate_passed"] is True
            assert report["diagnostic_nontrivial_fraction_passed"] is False
            assert report["secondary_sampler_audit_passed"] is False
            assert report["secondary_sampler_audit_changes_candidate_eligibility"] is False
            assert report["primary_arm_eligible"] is True


def test_official_metric_correction_is_sealed_and_preserves_all_inputs(
    tmp_path: Path,
) -> None:
    spec = json.loads(CONFIG.read_text())
    tracked = [
        correction._resolve_path(spec["literal_primary_selector"]["config_path"]),
        correction._resolve_path(spec["literal_primary_selector"]["result_path"]),
    ]
    for item in spec["input_candidates"]:
        tracked.extend(
            [
                correction._resolve_path(item["config_path"]),
                correction._resolve_path(item["result_path"]),
            ]
        )
    before = {path: file_sha256(path) for path in tracked}
    result = correction.run_spin_sampled_training_v2_metric_correction_selector(
        CONFIG, tmp_path / "correction", overwrite=True
    )
    after = {path: file_sha256(path) for path in tracked}
    assert before == after
    assert result["result_payload_sha256"] == correction._result_payload_sha256(
        result
    )
    assert result["literal_primary_binding"]["actions_by_case"] == {
        case_id: "no_adam_promotion" for case_id in correction.EXPECTED_CASES
    }
    assert result["decision_hierarchy"] == {
        "primary": "sealed_literal_v1_no_adam_promotion",
        "secondary": "conditional_acceptance_metric_sensitivity",
        "secondary_overrides_primary": False,
    }
    by_geometry = {row["geometry"]: row for row in result["geometry_results"]}
    expected = {"square": "lr_2e-3", "triangular": "lr_1e-3"}
    required = set(spec["parent_binding_contract"]["required_fields"])
    for geometry, candidate_id in expected.items():
        row = by_geometry[geometry]
        decision = row["decision"]
        assert decision["selected_candidate_id"] == candidate_id
        assert decision["training_authorized_by_secondary_selector"] is False
        assert decision["overrides_literal_primary_no_adam_promotion"] is False
        assert set(decision["selected_step50_parent_bindings_for_new_identity_only"]) == {
            "u1",
            "su2",
        }
        for binding in decision[
            "selected_step50_parent_bindings_for_new_identity_only"
        ].values():
            assert required <= set(binding)
        for candidate in row["candidate_diagnostics"]:
            for arm in candidate["arms"].values():
                assert "last10_conditional_move_acceptance_median" in arm
                assert "last10_accepted_move_fraction_median" in arm
                assert "last10_nontrivial_proposal_fraction_median" in arm
                assert "last10_chain_snapshot_move_fraction_median" in arm
                assert arm[
                    "secondary_sampler_audit_changes_candidate_eligibility"
                ] is False

    again = correction.run_spin_sampled_training_v2_metric_correction_selector(
        CONFIG, tmp_path / "correction", overwrite=False
    )
    assert again["result_payload_sha256"] == result["result_payload_sha256"]


def test_literal_selector_decision_tampering_fails_closed(tmp_path: Path) -> None:
    spec = json.loads(CONFIG.read_text())
    literal_result = correction._resolve_path(
        spec["literal_primary_selector"]["result_path"]
    )
    payload = json.loads(literal_result.read_text())
    payload["geometry_results"][0]["decision"]["action"] = "tampered_action"
    payload["result_payload_sha256"] = literal_selector._result_payload_sha256(payload)
    tampered = tmp_path / "literal_tampered.json"
    tampered.write_text(json.dumps(payload))
    modified = deepcopy(spec)
    modified["literal_primary_selector"]["result_path"] = str(tampered)
    config = tmp_path / "metric_correction.json"
    config.write_text(json.dumps(modified))
    with pytest.raises(RuntimeError, match="fail recomputation"):
        correction.run_spin_sampled_training_v2_metric_correction_selector(
            config, tmp_path / "output", overwrite=True
        )
