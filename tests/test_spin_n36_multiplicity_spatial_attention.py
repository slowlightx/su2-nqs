from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.spin_n36_multiplicity_spatial_attention import (
    ARM_IDS,
    EXACT_ENERGY_PER_SITE,
    GCNN_ENERGY_PER_SITE,
    TASK,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n36_multiplicity_spatial_attention.json"
JOINT_CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n36_joint_multiplicity_spatial_attention.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_n36_multiplicity_spatial_attention_v1/results.json"


def test_registered_multiplicity_spatial_config() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["task"] == TASK
    assert spec["common_sample"]["replicates"] >= 2


def test_joint_config_trains_reduced_backbone_and_keeps_scrambled_control() -> None:
    spec = json.loads(JOINT_CONFIG.read_text())
    _validate(spec)
    assert spec["training"]["train_reduced_backbone"] is True
    assert spec["training"]["include_fixed_sz_control"] is False
    assert spec["common_sample"]["replicates"] == 3


def test_literature_targets_match_registered_same_torus_values() -> None:
    assert EXACT_ENERGY_PER_SITE == -0.515564
    assert GCNN_ENERGY_PER_SITE == -0.515386


def test_repeated_paired_result_is_complete_when_present() -> None:
    if not RESULT.exists():
        return
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["reference"]["no_ed_repeated"]
    assert payload["gates"]["all_admission_gates_passed"]
    assert payload["attention_contract"][
        "acts_only_on_static_support_descriptors_and_reduced_multiplicity_maps"
    ]
    assert payload["attention_contract"]["magnetic_indices_carried_only_by_fixed_cg_tensors"]
    assert len(payload["replicates"]) == 3
    for replicate in payload["replicates"]:
        assert tuple(row["arm_id"] for row in replicate["arms"]) == ARM_IDS
        work = [row["work_ledger"]["full_h_configuration_calls_total"] for row in replicate["arms"]]
        assert len(set(work)) == 1
    causal = payload["paired_aggregate"]["spatial_minus_scrambled"]
    assert causal["negative_replicate_count"] == 3
    assert causal["mean_difference_per_site"] < 0.0
