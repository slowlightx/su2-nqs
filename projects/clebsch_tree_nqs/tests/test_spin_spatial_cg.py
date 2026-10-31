from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest

from projects.clebsch_tree_nqs.workflows import spin_spatial_cg
from projects.clebsch_tree_nqs.workflows import spin_spatial_cg_capacity


PROJECT = ROOT / "projects/clebsch_tree_nqs"


def _load(name: str) -> dict[str, object]:
    return json.loads((PROJECT / "results" / name / "results.json").read_text())


def _assert_result_seal(payload: dict[str, object]) -> str:
    body = deepcopy(payload)
    seal = body.pop("result_sha256")
    assert isinstance(seal, str)
    assert seal == canonical_sha256(body)
    return seal


def test_spatial_cg_artifacts_are_complete_and_current_source_sealed() -> None:
    for payload, module in (
        (_load("spin_spatial_cg_local_pilot_v1"), spin_spatial_cg),
        (_load("spin_spatial_cg_capacity_v1"), spin_spatial_cg_capacity),
    ):
        assert payload["status"] == "complete"
        _assert_result_seal(payload)
        manifest = transitive_local_source_manifest(
            (Path(module.__file__).resolve(),),
            root=ROOT,
            package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
        )
        assert payload["source_manifest"] == manifest
        assert payload["source_manifest_sha256"] == canonical_sha256(manifest)

    pilot = _load("spin_spatial_cg_local_pilot_v1")
    capacity = _load("spin_spatial_cg_capacity_v1")
    assert capacity["pilot_result_sha256"] == pilot["result_sha256"]


def test_spatial_cg_result_seal_detects_numerical_payload_tampering() -> None:
    payload = _load("spin_spatial_cg_local_pilot_v1")
    body = deepcopy(payload)
    seal = body.pop("result_sha256")
    body["summary"] = {}
    assert seal != canonical_sha256(body)


def test_n8_spatial_cg_oracles_pass_state_tangent_and_rank_gates() -> None:
    payload = _load("spin_spatial_cg_local_pilot_v1")
    for report in payload["exact_small_oracles"].values():
        assert report["passed"]
        assert report["nested_single_tree_max_abs_error"] < 1.0e-13
        assert report["spin_squared_residual_ratio"] < 5.0e-12
        assert report["maximum_tangent_spin_squared_residual_ratio"] < 5.0e-12
        assert (
            report["translation_projected_component_span"][
                "rank_at_1e_minus_10"
            ]
            > 1
        )


def test_paired_initialization_and_negative_matched_work_decision() -> None:
    payload = _load("spin_spatial_cg_local_pilot_v1")
    assert len(payload["cases"]) == 2
    for case in payload["cases"]:
        assert len(case["trials"]) == 12
        backbone_hashes: dict[int, set[str]] = {}
        for trial in case["trials"]:
            assert trial["initial_spatial_to_morton_max_abs_error"] < 1.0e-13
            backbone_hashes.setdefault(trial["seed"], set()).add(
                trial["initial_backbone_sha256"]
            )
        assert all(len(values) == 1 for values in backbone_hashes.values())
        summary = payload["summary"][case["case_id"]]
        work = summary["paired_comparisons"][
            "spatial_vs_work_control_single_tree"
        ]
        assert work["candidate_energy_win_count"] == 0
        assert work["median_paired_energy_error_ratio"] > 100.0

    square = payload["summary"]["square_4x4_j2_0p5_spatial_cg"]
    triangular = payload["summary"]["triangular_4x4_j2_0p125_spatial_cg"]
    assert square["paired_comparisons"][
        "spatial_vs_same_step_single_tree"
    ]["candidate_energy_win_count"] == 1
    assert triangular["paired_comparisons"][
        "spatial_vs_same_step_single_tree"
    ]["candidate_energy_win_count"] == 3
    assert square["paired_comparisons"][
        "physical_vs_scrambled_spatial_cg"
    ]["candidate_energy_win_count"] == 1
    assert triangular["paired_comparisons"][
        "physical_vs_scrambled_spatial_cg"
    ]["candidate_energy_win_count"] == 2


def test_capacity_replay_is_exact_and_global_route_headroom_is_small() -> None:
    payload = _load("spin_spatial_cg_capacity_v1")
    for case in payload["cases"]:
        for trial in case["trials"]:
            assert trial["single_tree_replay_energy_error"] < 2.0e-11
            physical = trial["physical_routes"]
            assert physical["span_rank_at_1e_minus_10"] > 1
            assert physical["unconstrained_ritz_gain_below_reference"] >= 0.0
            assert physical["fraction_of_ritz_gain_recovered"] < 0.5
        summary = payload["summary"][case["case_id"]]
        # Total-energy headroom below 8e-3 means less than 5e-4 per site.
        assert summary["median_physical_ritz_gain_below_reference"] < 8.0e-3
        assert summary["seed_count"] == 3
        assert summary["physical_two_feature_win_count_over_scrambled"] <= 1
