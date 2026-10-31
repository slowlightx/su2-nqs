from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from su2_nqs.frontier.artifacts import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import (
    spin_triangle_patch_n36_physical_init_readout as readout,
)


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/"
    "spin_triangle_patch_n36_physical_init_readout.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/"
    "spin_triangle_patch_n36_physical_init_readout_v1/results.json"
)


def _spec() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_physical_init_readout_config_freezes_projection_and_mcse_contract() -> None:
    spec = _spec()
    readout._validate(spec)
    assert spec["initialization"]["pre_projection_background_coefficient"] == 0.05
    assert spec["initialization"]["projected_branch_norm_ratio_not_assumed"]
    assert spec["projection"]["group_order"] == 36
    assert spec["sampler"]["chains"] == 32
    assert spec["sampler"]["burn_proposals_per_chain"] == 500
    assert spec["sampler"]["snapshots"] == 20
    assert spec["sampler"]["proposals_between_snapshots"] == 100
    assert spec["estimator"]["kind"] == "exact_singlet_3hz_diagonal"
    assert spec["estimator"]["sample_variance_is_physical_h_variance"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["initialization"].__setitem__(
            "pre_projection_background_coefficient", 0.1
        ),
        lambda value: value["projection"].__setitem__("group_order", 1),
        lambda value: value["sampler"].__setitem__("chains", 16),
        lambda value: value["estimator"].__setitem__(
            "sample_variance_is_physical_h_variance", True
        ),
    ),
)
def test_physical_init_readout_config_fails_closed(mutation) -> None:
    spec = copy.deepcopy(_spec())
    mutation(spec)
    with pytest.raises((TypeError, ValueError)):
        readout._validate(spec)


def test_promoted_scratch_artifact_is_sealed_and_explicit_about_limitations() -> None:
    result = json.loads(RESULT.read_text())
    unsealed = copy.deepcopy(result)
    seal = unsealed.pop("result_sha256")
    assert seal == canonical_sha256(unsealed)
    assert result["config_sha256"] == canonical_sha256(_spec())
    assert result["artifact_origin"] == "promoted_from_same_session_scratch_stdout"
    assert result["workflow_reexecuted_for_artifact"] is False
    assert result["raw_chain_states_persisted"] is False
    assert "triangle-patch-n36-physical-init-readout" in result["reproduction_command"]
    assert result["optimization_performed"] is False
    assert result["physical_hamiltonian_variance_reported"] is False
    assert "no optimized or converged energy" in result["claim_boundary"]
    assert {case["geometry"] for case in result["cases"]} == {
        "square",
        "triangular",
    }
    for case in result["cases"]:
        assert case["three_hz_sample_count"] == 640
        assert case["three_hz_energy_per_site"] < -0.4
        assert case["three_hz_energy_per_site_chain_mcse"] > 0.0
        assert case["post_burn_zero_amplitude_chain_fraction_maximum"] == 0.0
        pure, mixed = case["support_probe_arms"]
        assert pure["exact_zero_count"] == 1976
        assert mixed["exact_zero_count"] == 0
        assert mixed["below_relative_rms_threshold_count"] == 0


def test_support_summary_distinguishes_exact_and_relative_zeros() -> None:
    values = np.asarray((0.0, 1.0e-20, 1.0, -2.0))
    row = readout._support_summary(values, 1.0e-14)
    assert row["exact_zero_count"] == 1
    assert row["below_relative_rms_threshold_count"] == 2
    assert row["maximum_magnitude"] == 2.0
