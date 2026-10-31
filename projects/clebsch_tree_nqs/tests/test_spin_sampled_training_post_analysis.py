from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs import run
from projects.clebsch_tree_nqs.workflows import (
    spin_sampled_training_post_analysis as workflow,
)
from su2_nqs.frontier.artifacts import (
    array_sha256,
    canonical_sha256,
    embedded_trial_manifest,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/"
    "spin_6x6_sampled_training_post_analysis.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/"
    "spin_6x6_sampled_training_post_analysis_v1/results.json"
)


def _pairs(values: object) -> np.ndarray:
    concrete = np.asarray(values, dtype=np.float64)
    # Assign components rather than adding ``1j*imag`` so signed-zero bits in
    # the sealed complex array survive a JSON round trip.
    result = np.empty(concrete.shape[:-1], dtype=np.complex128)
    result.real = concrete[..., 0]
    result.imag = concrete[..., 1]
    return result


def test_registered_post_analysis_config_and_cli_contract() -> None:
    spec = json.loads(CONFIG.read_text())
    workflow._validate_spec(spec)
    args = run._build_parser().parse_args(
        ["sampled-training-post-analysis", "--config", str(CONFIG)]
    )
    assert args.command == "sampled-training-post-analysis"
    assert args.config == CONFIG
    assert spec["energy_normalization"] == "E/N"
    assert spec["variance_normalization"] == "Var(H)/N"
    assert spec["claim_boundary"] == workflow.POST_ANALYSIS_CLAIM_BOUNDARY


def test_variance_chain_mcse_and_per_site_normalization() -> None:
    values = np.asarray(
        [[1.0 + 1.0j, 3.0 - 1.0j], [2.0 + 0.5j, 6.0 - 0.5j]],
        dtype=np.complex128,
    )
    result = workflow._variance_summary(values, n_sites=4, block_length=1)
    contributions = np.abs(values - np.mean(values)) ** 2
    chain_means = np.mean(contributions, axis=0)
    expected_mcse = np.std(chain_means, ddof=1) / np.sqrt(2)
    assert result["variance_H"] == pytest.approx(np.mean(contributions))
    assert result["variance_H_chain_mcse"] == pytest.approx(expected_mcse)
    assert result["variance_H_per_site"] == pytest.approx(
        np.mean(contributions) / 4
    )
    assert result["variance_H_per_site_chain_mcse"] == pytest.approx(
        expected_mcse / 4
    )
    assert result["variance_contributions_sha256"] == array_sha256(
        contributions.astype(np.float64)
    )


@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_star_mean_is_formed_before_chain_mcse_and_max_is_secondary(
    geometry: str,
) -> None:
    # Perfectly anticorrelated star components make their per-sample mean
    # constant.  Combining marginal error bars would incorrectly miss this.
    first = np.asarray([[1.0, 3.0], [3.0, 1.0]], dtype=np.complex128)
    second = 4.0 - first
    if geometry == "square":
        names = ("q0", "neel", "stripe_x", "stripe_y")
        mode_values = np.stack((first * 0, first, first, second), axis=-1)
        key = "stripe_star_mean"
    else:
        names = ("q0", "k", "m1", "m2", "m3")
        third = np.full_like(first, 2.0)
        mode_values = np.stack(
            (first * 0, first, first, second, third), axis=-1
        )
        key = "M_star_mean"
    arrays, contract = workflow._derived_mode_arrays(
        geometry, mode_values, names
    )
    summary = workflow._chain_summary(arrays[key])
    assert summary["mean_real"] == pytest.approx(2.0)
    assert summary["chain_mean_mcse"] == pytest.approx(0.0)
    assert contract["star_mean_mcse_uses_per_sample_component_average"] is True
    assert contract["component_covariance_retained_in_chain_means"] is True
    assert contract["max_star_has_selection_bias_and_is_not_primary"] is True


def test_paired_chain_endpoint_uses_lineage_covariance() -> None:
    updates = [5, 10, 15, 20]
    offsets = np.asarray([0.0, 2.0, -1.0, 4.0])
    rows = [offsets + shift for shift in (0.0, -0.5, -1.0, -1.5)]
    result = workflow._paired_chain_trend(updates, rows)
    assert result["paired_endpoint_delta_last_minus_first"] == pytest.approx(-1.5)
    assert result["paired_endpoint_chain_mcse"] == pytest.approx(0.0)
    assert result["chain_lineage_index_preserved"] is True
    assert result["primary_endpoint_resolution_uses_paired_chain_difference"] is True


def test_resolution_labels_and_claim_boundary() -> None:
    assert workflow._resolution_label(0.99) == "unresolved_within_MCSE"
    assert workflow._resolution_label(1.0) == "suggestive_exploratory"
    assert workflow._resolution_label(2.0).startswith("MC_resolved_exploratory")
    assert "do not establish convergence" in workflow.POST_ANALYSIS_CLAIM_BOUNDARY
    assert "QSL" in workflow.POST_ANALYSIS_CLAIM_BOUNDARY


def test_upstream_training_and_runtime_seals_are_read_only_valid() -> None:
    spec = json.loads(CONFIG.read_text())
    training, training_path = workflow._load_training_result(spec)
    baselines = workflow._load_runtime_baselines(spec)
    assert training["status"] == "complete"
    assert training_path.is_file()
    assert set(baselines) == {"square", "triangular"}
    assert training["source_sha256"] == workflow.training_workflow._source_manifest()
    assert all(path.is_file() for _, _, path in baselines.values())


@pytest.mark.skipif(not RESULT.is_file(), reason="official post-analysis not run")
def test_official_post_analysis_artifact_is_fully_sealed_and_reproducible() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["config_sha256"] == canonical_sha256(payload["config"])
    assert payload["source_sha256"] == workflow._source_manifest()
    assert payload["source_manifest_sha256"] == canonical_sha256(
        payload["source_sha256"]
    )
    assert payload["runtime_identity"] == numerical_runtime_identity()
    assert payload["runtime_identity_sha256"] == canonical_sha256(
        payload["runtime_identity"]
    )
    assert payload["input_results_manifest_sha256"] == canonical_sha256(
        payload["input_results_manifest"]
    )
    spec = payload["config"]
    assert payload["input_results_manifest"]["sampled_training_pilot"] == file_sha256(
        ROOT / spec["input_result"]
    )
    for entry in spec["runtime_baselines"]:
        assert payload["input_results_manifest"][f"runtime_{entry['geometry']}"] == file_sha256(
            ROOT / entry["result"]
        )
    assert payload["trial_manifest_sha256"] == canonical_sha256(
        embedded_trial_manifest(payload)
    )
    assert payload["all_post_analysis_gates_passed"] is True
    for case in payload["cases"]:
        assert case["same_schedule_SU2_vs_U1_step20"]["matched_contract"][
            "parameter_count_matched"
        ] is False
        for trial in case["trials"]:
            assert trial["trial_payload_sha256"] == trial_payload_sha256(trial)
            assert trial["all_post_analysis_gates_passed"] is True
            assert trial["post_analysis_timing_claim_eligible"] is False
            assert len(trial["replay_updates"]) == 4
            for row in trial["replay_updates"]:
                raw = row["raw_replay"]
                energy = _pairs(raw["total_H_local_values_complex_pairs"])
                modes = _pairs(raw["full_Sq_local_values_complex_pairs"])
                assert list(energy.shape) == raw["shape_energy_sweep_chain"]
                assert list(modes.shape) == raw["shape_modes_sweep_chain_mode"]
                assert array_sha256(energy) == raw["total_H_local_values_sha256"]
                assert array_sha256(modes) == raw["full_Sq_local_values_sha256"]
                assert row["registered_summary_replay_within_tolerance"] is True
                assert row["variance_H_per_site"][
                    "variance_H_per_site_chain_mcse"
                ] >= 0.0
