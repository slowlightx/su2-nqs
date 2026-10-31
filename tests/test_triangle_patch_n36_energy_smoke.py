from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from su2_nqs.frontier.artifacts import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_triangle_patch_n36_energy_smoke as smoke


CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_triangle_patch_n36_energy_smoke.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_triangle_patch_n36_energy_smoke_v1/results.json"


def _spec():
    return json.loads(CONFIG.read_text())


def test_energy_smoke_config_freezes_projection_schedule_and_claim_boundary() -> None:
    spec = _spec(); smoke._validate(spec)
    assert spec["model"]["maximum_intermediate_q"] == 10
    assert spec["model"]["parameter_count"] == 1252
    assert spec["projection"]["group_order"] == 36
    assert spec["projection"]["applied_from_update"] == 0
    assert spec["sampler"]["chains"] == 16
    assert spec["sampler"]["initial_burn_proposals_per_chain"] == 288
    assert spec["sampler"]["proposals_between_updates"] == 72
    assert spec["optimizer"]["updates"] == 10
    assert spec["audit"]["updates"] == [0, 5, 10]
    assert all(case["ed"] is False for case in spec["cases"])


@pytest.mark.parametrize(
    "mutation",
    (
        lambda x: x["model"].__setitem__("maximum_intermediate_q", 4),
        lambda x: x["projection"].__setitem__("applied_from_update", 1),
        lambda x: x["sampler"].__setitem__("chains", 8),
        lambda x: x["optimizer"].__setitem__("updates", 25),
        lambda x: x["audit"].__setitem__("updates", [0, 10]),
        lambda x: x["cases"][0].__setitem__("ed", True),
    ),
)
def test_energy_smoke_config_fails_closed(mutation) -> None:
    spec = copy.deepcopy(_spec()); mutation(spec)
    with pytest.raises((TypeError, ValueError)):
        smoke._validate(spec)


def test_energy_smoke_artifact_is_sealed_and_honest() -> None:
    result = json.loads(RESULT.read_text())
    unsealed = copy.deepcopy(result); seal = unsealed.pop("result_sha256")
    assert seal == canonical_sha256(unsealed)
    assert result["config"] == _spec()
    assert result["hamiltonian_diagonalization_performed"] is False
    assert result["translation_projection_applied_from_update_zero"] is True
    assert result["training_force"] == "exact_singlet_3hz_center_only"
    assert "no converged energy" in result["claim_boundary"]
    assert {case["geometry"] for case in result["cases"]} == {"square", "triangular"}
    for case in result["cases"]:
        assert case["parameter_count"] == 1252
        assert case["translation_group_order"] == 36
        assert case["completed_updates"] <= 10
        assert [audit["update"] for audit in case["full_h_paired_audits"]] == (
            [0, 5, 10] if case["status"] == "complete" else [0]
        )[: len(case["full_h_paired_audits"])]
        for audit in case["full_h_paired_audits"]:
            assert audit["same_samples_used_for_full_h_and_three_hz"]
            assert audit["three_hz_variance_is_not_physical_h_variance"]
            assert audit["sample_count"] == 32
            assert audit["chain_count"] == 16
            assert audit["energy_per_site_chain_mcse"] >= 0.0
            assert audit["variance_h_per_site"] >= 0.0
        for row in case["history"]:
            assert row["three_hz_variance_is_not_physical_h_variance"]
            assert row["adam"]["relative_update_norm"] <= 0.05
        latest = ROOT / case["checkpoint_directory"] / "LATEST.json"
        manifest = json.loads(latest.read_text())
        assert [row["completed_steps"] for row in manifest["generations"]] == [
            0,
            5,
            10,
        ]
