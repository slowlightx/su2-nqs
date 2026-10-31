from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime
from projects.clebsch_tree_nqs.workflows import spin_sampled_training as training_v1
from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2 as training_v2
from su2_nqs.frontier.artifacts import array_sha256, canonical_sha256
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint


CONFIG_DIRECTORY = ROOT / "projects/clebsch_tree_nqs/configs/sampled"
CONFIGS = tuple(
    CONFIG_DIRECTORY / name
    for name in (
        "spin_6x6_center_only_lr_1e3.json",
        "spin_6x6_center_only_lr_2e3.json",
        "spin_6x6_center_only_lr_5e3.json",
    )
)


def _tiny_spec() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    case = {
        "case_id": "square_2x2_center_only_test",
        "geometry": "square",
        "rows": 2,
        "columns": 2,
        "periodic": True,
        "j2_over_j1": 0.0,
        "support": "fixed_m0",
        "ed": False,
        "coordinate_convention": "row_major_cartesian_column_row",
    }
    arm = {
        "arm_id": "u1_D1_translation_test",
        "backbone": "u1",
        "projection": "translation",
        "seed": 17,
        "model": {
            "feature_dim": 1,
            "maximum_intermediate_q": 4,
            "initialization_scale": 0.2,
        },
    }
    spec = {
        "schema_version": 2,
        "domain": "spin",
        "task": training_v2.TASK,
        "optimizer_backend": training_v2.OPTIMIZER_BACKEND,
        "scan_family_id": "tiny_center_only_scan",
        "candidate_id": "lr_1e-4",
        "claim_boundary": training_v2.CLAIM_BOUNDARY,
        "cases": [case],
        "projection": {
            "kind": "translation",
            "orbit_chunk_size": 1,
            "rematerialize_orbit": True,
        },
        "arms": [arm],
        "sampler": {
            "chains": 8,
            "initial_burn_sweeps": 1,
            "decorrelation_sweeps_per_update": 1,
            "proposals_per_sweep": 4,
            "bond_proposal_probability": 0.5,
            "seed": 101,
        },
        "optimizer": {
            "name": "local_adam",
            "updates": 2,
            "learning_rate": 1.0e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "microbatch_size": 4,
            "maximum_relative_update_norm": 0.05,
            "checkpoint_every_updates": 1,
        },
        "measurement": {
            "updates": [0, 1, 2],
            "burn_sweeps": 1,
            "measurement_sweeps": 2,
            "seed": 202,
            "energy_variance_only": True,
        },
        "diagnostics": {
            "acceptance_window": 10,
            "low_acceptance_median_threshold": 0.15,
            "relative_update_quantile": 0.99,
            "relative_update_threshold": 0.02,
        },
        "selection_policy": training_v2._expected_selection_policy(2),
    }
    return spec, case, arm


def _tiny_runtime(arm: dict[str, object]) -> runtime._ArmRuntime:
    projector = PhysicalSpatialProjector(
        (tuple(range(4)),), orbit_chunk_size=1, rematerialize_orbit=True
    )
    return runtime._initialize_arm(arm, rows=2, columns=2, projector=projector)


def _run_tiny(
    directory: Path,
    *,
    overwrite: bool,
    maximum_updates_this_invocation: int | None = None,
    learning_rate: float = 1.0e-4,
) -> dict[str, object]:
    spec, case, arm = _tiny_spec()
    spec["optimizer"]["learning_rate"] = learning_rate
    spec["candidate_id"] = f"lr_{learning_rate:.8g}"
    bonds = (
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
    )
    return training_v2._run_scan_arm(
        runtime=_tiny_runtime(arm),
        arm=arm,
        case=case,
        bonds=bonds,
        group=(tuple(range(4)),),
        spec=spec,
        checkpoint_directory=directory,
        config_hash=canonical_sha256(spec),
        source_hash="tiny-v2-source",
        runtime_hash="tiny-runtime",
        common_profile_hash=training_v2.common_scan_profile_sha256(spec),
        overwrite=overwrite,
        maximum_updates_this_invocation=maximum_updates_this_invocation,
    )


def test_registered_configs_form_one_matched_three_rate_scan() -> None:
    specs = [json.loads(path.read_text()) for path in CONFIGS]
    for spec in specs:
        training_v2._validate_spec(spec)
        assert spec["optimizer_backend"] == training_v2.OPTIMIZER_BACKEND
        assert spec["optimizer"]["updates"] == 50
        assert spec["optimizer"]["checkpoint_every_updates"] == 25
        assert spec["optimizer"]["microbatch_size"] == 8
        assert spec["optimizer"]["maximum_relative_update_norm"] == 0.05
        assert spec["sampler"]["chains"] == 32
        assert spec["sampler"]["initial_burn_sweeps"] == 16
        assert spec["sampler"]["decorrelation_sweeps_per_update"] == 2
        assert spec["measurement"] == {
            "updates": [0, 25, 50],
            "burn_sweeps": 8,
            "measurement_sweeps": 16,
            "seed": 20260919,
            "energy_variance_only": True,
        }
        assert spec["selection_policy"] == training_v2._expected_selection_policy(50)
        assert {case["geometry"] for case in spec["cases"]} == {
            "square",
            "triangular",
        }
        assert all(case["rows"] == case["columns"] == 6 for case in spec["cases"])
        assert {arm["backbone"] for arm in spec["arms"]} == {"u1", "su2"}
        assert all(arm["seed"] == 11 for arm in spec["arms"])
        assert all(arm["model"]["feature_dim"] == 4 for arm in spec["arms"])
    assert {spec["optimizer"]["learning_rate"] for spec in specs} == {
        1.0e-3,
        2.0e-3,
        5.0e-3,
    }
    assert len({training_v2.common_scan_profile_sha256(spec) for spec in specs}) == 1
    for first, second in zip(specs[:-1], specs[1:], strict=True):
        assert training_v2._common_scan_profile(first) == training_v2._common_scan_profile(
            second
        )
    broken = deepcopy(specs[0])
    broken["selection_policy"]["geometry_selection"][
        "cross_arm_uncertainty_combination"
    ] = "quadrature"
    with pytest.raises(ValueError, match="exactly match"):
        training_v2._validate_spec(broken)


def test_v2_source_is_versioned_without_reverse_contaminating_v1_seal() -> None:
    v2_manifest = training_v2._source_manifest()
    assert "projects/clebsch_tree_nqs/workflows/spin_sampled_training_v2.py" in v2_manifest
    assert "su2_nqs/vmc/physical_sampled_adam_center_only.py" in v2_manifest
    assert "projects/clebsch_tree_nqs/workflows/spin_sampled_training.py" in v2_manifest

    v1_manifest = training_v1._source_manifest()
    assert "projects/clebsch_tree_nqs/workflows/spin_sampled_training_v2.py" not in v1_manifest
    assert "su2_nqs/vmc/physical_sampled_adam_center_only.py" not in v1_manifest


def test_center_only_resume_matches_uninterrupted_and_work_ledger_is_correct(
    tmp_path: Path,
) -> None:
    resumed_directory = tmp_path / "resumed"
    interrupted = _run_tiny(
        resumed_directory,
        overwrite=True,
        maximum_updates_this_invocation=1,
    )
    assert interrupted["status"] == "running"
    assert interrupted["completed_updates"] == 1
    resumed = _run_tiny(resumed_directory, overwrite=False)
    direct = _run_tiny(tmp_path / "direct", overwrite=True)
    assert resumed["status"] == direct["status"] == "complete"
    assert resumed["final_params_sha256"] == direct["final_params_sha256"]
    assert resumed["final_chain_states_sha256"] == direct["final_chain_states_sha256"]
    assert resumed["checkpoint"]["optimizer_state_sha256"] == direct["checkpoint"][
        "optimizer_state_sha256"
    ]
    assert resumed["checkpoint"]["sampling_key_sha256"] == direct["checkpoint"][
        "sampling_key_sha256"
    ]
    assert resumed["sampler"]["resume_count"] == 1
    assert direct["sampler"]["resume_count"] == 0
    assert [item["update"] for item in resumed["fixed_parameter_measurements"]] == [
        0,
        1,
        2,
    ]
    for resumed_measurement, direct_measurement in zip(
        resumed["fixed_parameter_measurements"],
        direct["fixed_parameter_measurements"],
        strict=True,
    ):
        for name in (
            "fixed_params_sha256",
            "training_chains_sha256",
            "energy_per_site",
            "energy_per_site_chain_mcse",
            "energy_per_site_chain_means",
            "energy_per_site_chain_means_sha256",
            "variance_h_per_site",
            "variance_h_per_site_chain_mcse",
            "variance_h_per_site_chain_contributions",
            "variance_h_per_site_chain_contributions_sha256",
        ):
            assert resumed_measurement[name] == direct_measurement[name]
        assert resumed_measurement["full_spin_structure_factor_computed"] is False
    for resumed_row, direct_row in zip(
        resumed["history"], direct["history"], strict=True
    ):
        for name in (
            "params_before_sha256",
            "params_after_sha256",
            "chain_states_sha256",
            "sampling_key_sha256",
            "pre_update_sampled_energy_per_site",
            "pre_update_sampled_variance_h_per_site",
        ):
            assert resumed_row[name] == direct_row[name]
        assert resumed_row["gradient"]["connected_hpsi_recomputed_in_backward"] is False
        ledger = resumed_row["work_ledger"]
        assert ledger["hpsi_forward_projected_amplitude_evaluations"] == 48
        assert ledger["center_only_primal_projected_amplitude_evaluations"] == 8
        assert ledger["center_only_forward_projected_amplitude_evaluations"] == 56
        assert ledger["legacy_two_hpsi_projected_amplitude_evaluations_proxy"] == 96
        assert (
            ledger[
                "avoided_second_hpsi_connected_projected_amplitude_evaluations"
            ]
            == 40
        )
        assert ledger["hpsi_actual_connected_configurations"] <= 40
        assert ledger["hpsi_padded_connected_configurations"] == 40
    assert all(resumed["readiness_gates"].values())
    assert resumed["selection_diagnostics"][
        "diagnostics_are_not_hard_stops_or_wall_time_filters"
    ] is True
    policy = resumed["endpoint_policy_diagnostics"]
    first, final = resumed["fixed_parameter_measurements"][0::2]
    expected_energy_lowering = (
        first["energy_per_site"] - final["energy_per_site"]
    )
    energy_differences = np.asarray(
        first["energy_per_site_chain_means"], dtype=np.float64
    ) - np.asarray(final["energy_per_site_chain_means"], dtype=np.float64)
    expected_energy_mcse = np.std(energy_differences, ddof=1) / np.sqrt(
        len(energy_differences)
    )
    expected_independent_mcse = np.hypot(
        first["energy_per_site_chain_mcse"],
        final["energy_per_site_chain_mcse"],
    )
    np.testing.assert_allclose(
        policy["signed_energy_lowering_per_site"],
        expected_energy_lowering,
        atol=2.0e-14,
        rtol=2.0e-12,
    )
    assert policy["energy_combined_chain_mcse_per_site"] == expected_energy_mcse
    assert policy["signed_energy_lowering_combined_mcse"] == (
        np.mean(energy_differences) / expected_energy_mcse
    )
    assert policy["energy_independent_hypot_mcse_per_site"] == (
        expected_independent_mcse
    )
    assert policy["primary_endpoint_uncertainty"] == (
        "same_chain_lineage_paired_difference_se"
    )
    assert policy[
        "geometry_common_lr_selection_deferred_to_cross_artifact_analysis"
    ] is True
    assert policy["arm_specific_optimum_diagnostic_only"] is True
    assert policy["diagnostic_does_not_stop_training_early"] is True
    assert policy["eligible_for_cross_artifact_geometry_common_lr_analysis"] == (
        policy["energy_endpoint_gate_passed"]
        and policy["variance_endpoint_gate_passed"]
        and policy["stability_gate_passed"]
    )
    assert resumed["work_ledger"]["wall_time_used_for_candidate_selection"] is False

    checkpoint = load_latest_resumable_checkpoint(resumed_directory)
    assert checkpoint["run_identity"]["schema"] == (
        "physical_sampled_spin_adam_center_only_v2"
    )
    assert checkpoint["run_identity"]["optimizer_backend"] == (
        training_v2.OPTIMIZER_BACKEND
    )
    assert checkpoint["metadata"]["amplitude_cache_present"] is False
    assert "amplitudes" not in checkpoint["metadata"]


def test_common_profile_ignores_only_candidate_and_learning_rate() -> None:
    spec, _case, _arm = _tiny_spec()
    alternative = deepcopy(spec)
    alternative["candidate_id"] = "lr_2e-4"
    alternative["optimizer"]["learning_rate"] = 2.0e-4
    assert training_v2.common_scan_profile_sha256(spec) == (
        training_v2.common_scan_profile_sha256(alternative)
    )
    alternative["sampler"]["seed"] += 1
    assert training_v2.common_scan_profile_sha256(spec) != (
        training_v2.common_scan_profile_sha256(alternative)
    )


def test_three_learning_rates_share_dynamic_initial_state_and_update_zero_measurement(
    tmp_path: Path,
) -> None:
    states = [
        _run_tiny(
            tmp_path / f"lr_{index}",
            overwrite=True,
            maximum_updates_this_invocation=0,
            learning_rate=rate,
        )
        for index, rate in enumerate((1.0e-3, 2.0e-3, 5.0e-3))
    ]
    assert all(item["status"] == "running" for item in states)
    for field in (
        "common_scan_profile_sha256",
        "initial_params_sha256",
        "initial_chain_states_sha256",
        "initial_post_burn_chain_states_sha256",
        "initial_training_key_sha256",
        "initial_post_burn_training_key_sha256",
    ):
        assert len({item[field] for item in states}) == 1
    measurements = [item["fixed_parameter_measurements"][0] for item in states]
    for field in (
        "fixed_params_sha256",
        "training_chains_sha256",
        "energy_per_site",
        "energy_per_site_chain_mcse",
        "variance_h_per_site",
        "variance_h_per_site_chain_mcse",
        "measurement_random_stream_fold",
        "measurement_random_stream_identity_sha256",
    ):
        assert len({item[field] for item in measurements}) == 1
    for field, hash_field in (
        ("energy_per_site_chain_means", "energy_per_site_chain_means_sha256"),
        (
            "variance_h_per_site_chain_contributions",
            "variance_h_per_site_chain_contributions_sha256",
        ),
    ):
        arrays = [np.asarray(item[field], dtype=np.float64) for item in measurements]
        assert all(array.shape == (8,) for array in arrays)
        for array, item in zip(arrays, measurements, strict=True):
            assert item[hash_field] == array_sha256(array)
            assert item["raw_measurement_sweeps_stored"] is False
        for array in arrays[1:]:
            np.testing.assert_array_equal(array, arrays[0])
        roundtrip = json.loads(json.dumps(measurements[0]))
        assert roundtrip[hash_field] == array_sha256(
            np.asarray(roundtrip[field], dtype=np.float64)
        )
    assert all(
        item["latest_valid_checkpoint"]["completed_updates"] == 0
        for item in states
    )


def test_endpoint_policy_fails_closed_on_zero_combined_mcse() -> None:
    spec, _case, _arm = _tiny_spec()
    rows = []
    for update in (0, 2):
        energy = np.full(4, -0.1 - 0.01 * update, dtype=np.float64)
        variance = np.full(4, 0.4, dtype=np.float64)
        rows.append(
            {
                "update": update,
                "chain_count": 4,
                "energy_per_site": float(np.mean(energy)),
                "energy_per_site_chain_mcse": 0.0,
                "energy_per_site_chain_means": energy.tolist(),
                "energy_per_site_chain_means_sha256": array_sha256(energy),
                "variance_h_per_site": float(np.mean(variance)),
                "variance_h_per_site_chain_mcse": 0.0,
                "variance_h_per_site_chain_contributions": variance.tolist(),
                "variance_h_per_site_chain_contributions_sha256": (
                    array_sha256(variance)
                ),
                "chain_index_lineage": (
                    "persistent_training_chain_index_then_independent_measurement_copy"
                ),
            }
        )
    result = training_v2._endpoint_policy_diagnostics(rows, spec)
    assert result["policy_diagnostic_valid"] is False
    assert result["energy_endpoint_gate_passed"] is False
    assert result["variance_endpoint_gate_passed"] is False
    assert result["eligible_for_cross_artifact_geometry_common_lr_analysis"] is False
    assert result["signed_energy_lowering_combined_mcse"] is None
    assert result["signed_variance_worsening_combined_mcse"] is None


def test_top_level_hard_failures_are_terminal_and_manifest_sealed(
    tmp_path: Path, monkeypatch
) -> None:
    spec, _case, arm = _tiny_spec()
    su2_arm = deepcopy(arm)
    su2_arm["arm_id"] = "su2_D1_translation_test"
    su2_arm["backbone"] = "su2"
    spec["arms"] = [arm, su2_arm]
    training_v2._validate_spec(spec)
    config = tmp_path / "hard_fail_config.json"
    config.write_text(json.dumps(spec))

    class FakeRuntime:
        def __init__(self, selected_arm):
            self.arm_id = selected_arm["arm_id"]
            self.backbone = selected_arm["backbone"]

    monkeypatch.setattr(
        runtime,
        "_initialize_arm",
        lambda selected_arm, **_kwargs: FakeRuntime(selected_arm),
    )

    def fake_hard_fail(*, runtime, arm, spec, **_kwargs):
        return {
            "status": "hard_failed",
            "candidate_id": spec["candidate_id"],
            "learning_rate": spec["optimizer"]["learning_rate"],
            "optimizer_backend": training_v2.OPTIMIZER_BACKEND,
            "arm_id": runtime.arm_id,
            "backbone": runtime.backbone,
            "seed": arm["seed"],
            "completed_updates": 0,
            "requested_updates": spec["optimizer"]["updates"],
            "history": [],
            "fixed_parameter_measurements": [],
            "selection_diagnostics": {},
            "latest_valid_checkpoint": None,
            "hard_failure": {
                "category": "relative_update_exceeded",
                "attempted_update": 1,
                "message": "registered test hard failure",
            },
            "claim_boundary": training_v2.CLAIM_BOUNDARY,
        }

    monkeypatch.setattr(training_v2, "_run_scan_arm", fake_hard_fail)
    payload = training_v2.run_spin_sampled_training_v2(
        config, tmp_path / "hard_fail_result", overwrite=True
    )
    assert payload["status"] == "complete"
    assert payload["hard_failed_trial_count"] == 2
    assert payload["terminal_trial_count"] == 2
    assert payload["all_non_hard_failed_candidates_completed"] is True
    assert len(payload["trial_manifest"]) == 2
    assert all(
        trial["status"] == "hard_failed" and "seed" in trial
        for case in payload["cases"]
        for trial in case["trials"]
    )


def test_cli_registers_versioned_optimizer_scan() -> None:
    source = (ROOT / "projects/clebsch_tree_nqs/run.py").read_text()
    assert '"sampled-optimizer-scan-v2"' in source
    assert "run_spin_sampled_training_v2" in source
    assert canonical_sha256({"registered": True})
