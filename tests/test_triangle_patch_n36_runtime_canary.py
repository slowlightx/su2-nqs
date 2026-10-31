from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_triangle_patch_n36_runtime


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_triangle_patch_n36_runtime_canary.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results"
    / "spin_triangle_patch_n36_runtime_canary_v1/results.json"
)


def _spec() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_n36_runtime_config_is_strictly_registered_without_ed_or_projection() -> None:
    spec = _spec()
    spin_triangle_patch_n36_runtime._validate(spec)
    assert spec["model"]["feature_dim"] == 2
    assert spec["model"]["maximum_intermediate_q"] == 10
    assert spec["model"]["expected_real_parameter_count"] == 1252
    assert spec["model"]["lattice_projection"] == "none"
    assert {case["geometry"] for case in spec["cases"]} == {
        "square",
        "triangular",
    }
    assert all(case["ed"] is False for case in spec["cases"])
    assert spec["runtime"]["cutoff_stress_q_values"] == [4, 6, 8, 10]
    assert spec["runtime"]["translation_projected_hpsi_probe"] is True
    assert spec["runtime"]["perform_parameter_update"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda spec: spec["model"].__setitem__("maximum_intermediate_q", 4),
        lambda spec: spec["model"].__setitem__("lattice_projection", "translation"),
        lambda spec: spec["cases"][0].__setitem__("ed", True),
        lambda spec: spec["runtime"].__setitem__(
            "cutoff_stress_q_values", [4, 6, 8]
        ),
        lambda spec: spec["runtime"].__setitem__("perform_parameter_update", True),
        lambda spec: spec["runtime"].__setitem__(
            "translation_projected_hpsi_probe", False
        ),
        lambda spec: spec.pop("claim_boundary"),
    ),
)
def test_n36_runtime_config_fails_closed(mutation) -> None:
    spec = copy.deepcopy(_spec())
    mutation(spec)
    with pytest.raises((TypeError, ValueError)):
        spin_triangle_patch_n36_runtime._validate(spec)


def test_n36_probe_states_and_registered_bond_widths() -> None:
    states = spin_triangle_patch_n36_runtime._fixed_m0_candidates(7, 11)
    assert states.shape == (11, 36)
    assert states.dtype == np.int32
    assert np.all(np.sum(states, axis=1) == 18)
    square = spin_triangle_patch_n36_runtime._bonds("square", 0.5)
    triangular = spin_triangle_patch_n36_runtime._bonds("triangular", 0.125)
    assert (len(square), len(square) + 1) == (144, 145)
    assert (len(triangular), len(triangular) + 1) == (216, 217)


def test_n36_runtime_artifact_records_only_kernel_readiness() -> None:
    result = json.loads(RESULT.read_text())
    unsealed = copy.deepcopy(result)
    seal = unsealed.pop("result_sha256")
    assert seal == canonical_sha256(unsealed)
    assert result["config"] == _spec()
    assert result["config_sha256"] == canonical_sha256(_spec())
    manifest = transitive_local_source_manifest(
        (Path(spin_triangle_patch_n36_runtime.__file__).resolve(),),
        root=spin_triangle_patch_n36_runtime.ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    assert result["source_manifest"] == manifest
    assert result["source_manifest_sha256"] == canonical_sha256(manifest)
    assert result["status"] == "complete"
    assert result["task"] == spin_triangle_patch_n36_runtime.TASK
    assert result["hamiltonian_diagonalization_performed"] is False
    assert result["hilbert_space_enumerated"] is False
    assert result["training_performed"] is False
    assert result["primary_gradient_lattice_projection_applied"] is False
    assert result["translation_projected_hpsi_probe_performed"] is True
    assert result["promotion_decision"] == (
        "pass_n36_unprojected_gradient_and_translation_hpsi_runtime_gate_only"
    )
    assert "not Born samples" in result["claim_boundary"]

    stress = result["cutoff_stress_scan"]["rows"]
    assert [row["maximum_intermediate_q"] for row in stress] == [4, 6, 8, 10]
    zero_counts = [row["exact_zero_count"] for row in stress]
    assert zero_counts[0] > 0
    assert all(left >= right for left, right in zip(zero_counts, zero_counts[1:]))
    assert zero_counts[-1] == 0
    triangular = result["cutoff_stress_scan"]["triangular_qmax10_confirmation"]
    assert triangular["maximum_intermediate_q"] == 10
    assert triangular["exact_zero_count"] == 0
    assert triangular["all_values_finite"]

    assert {case["geometry"] for case in result["cases"]} == {
        "square",
        "triangular",
    }
    for case in result["cases"]:
        assert case["parameter_count_real_scalars"] == 1252
        expected_bonds = 144 if case["geometry"] == "square" else 216
        assert case["bond_count"] == expected_bonds
        assert case["padded_width"] == expected_bonds + 1
        assert case["all_profiled_values_finite"]
        assert case["nonfinite_local_count"] == 0
        assert case["tangent_output_l2_norm"] > 0.0
        assert case["gradient_l2_norm"] > 0.0
        assert case["forward_hpsi_center_replay_max_abs_error"] < 1.0e-13
        assert case["hamiltonian_diagonalization_performed"] is False
        assert case["hilbert_space_enumerated"] is False
        assert case["lattice_projection_applied"] is False
        assert case["translation_projected_hpsi_probe_performed"] is True
        projected = case["translation_projected_hpsi_probe"]
        assert projected["group_order"] == 36
        assert projected["all_values_finite"]
        assert projected["gradient_profiled"] is False
        assert case["parameter_update_performed"] is False
