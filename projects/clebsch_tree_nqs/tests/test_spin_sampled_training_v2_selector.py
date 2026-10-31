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

from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2 as scan_v2
from projects.clebsch_tree_nqs.workflows import (
    spin_sampled_training_v2_selector as selector,
)
from su2_nqs.frontier.artifacts import (
    array_sha256,
    canonical_sha256,
    embedded_trial_manifest,
    pytree_sha256,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.resumable_checkpoint import (
    canonical_json_sha256,
    save_resumable_checkpoint,
)


CONFIG_DIRECTORY = ROOT / "projects/clebsch_tree_nqs/configs/sampled"
SELECTOR_CONFIG = CONFIG_DIRECTORY / "spin_6x6_center_only_lr_selector.json"
SCAN_CONFIGS = (
    CONFIG_DIRECTORY / "spin_6x6_center_only_lr_1e3.json",
    CONFIG_DIRECTORY / "spin_6x6_center_only_lr_2e3.json",
    CONFIG_DIRECTORY / "spin_6x6_center_only_lr_5e3.json",
)
PATTERN = np.asarray([-1.5, -0.5, 0.5, 1.5] * 2, dtype=np.float64)
SYNTHETIC_RUNTIME = {"kind": "synthetic-selector-runtime", "precision": "float64"}


def _measurement(
    *,
    spec: dict[str, object],
    case_id: str,
    arm_id: str,
    update: int,
    energy: np.ndarray,
    variance: np.ndarray,
) -> dict[str, object]:
    energy = np.asarray(energy, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    fold = scan_v2._measurement_fold(
        spec=spec, case_id=case_id, arm_id=arm_id, update=update
    )
    seed = int(spec["measurement"]["seed"])
    return {
        "update": update,
        "fixed_params_sha256": f"params-{case_id}-{arm_id}-update{update}",
        "training_chains_sha256": f"chains-{case_id}-{arm_id}-update{update}",
        "params_frozen_throughout_block": True,
        "training_chains_copied_and_not_mutated": True,
        "independent_measurement_random_stream": True,
        "measurement_random_stream_fold": fold,
        "measurement_random_stream_identity_sha256": canonical_sha256(
            {
                "measurement_seed": seed,
                "fold": fold,
                "case_id": case_id,
                "arm_id": arm_id,
                "update": update,
            }
        ),
        "energy_per_site": float(np.mean(energy)),
        "energy_per_site_chain_mcse": float(
            np.std(energy, ddof=1) / np.sqrt(len(energy))
        ),
        "energy_per_site_chain_means": energy.tolist(),
        "energy_per_site_chain_means_sha256": array_sha256(energy),
        "energy_imaginary_abs": 0.0,
        "variance_h_per_site": float(np.mean(variance)),
        "variance_h_per_site_chain_mcse": float(
            np.std(variance, ddof=1) / np.sqrt(len(variance))
        ),
        "variance_h_per_site_chain_contributions": variance.tolist(),
        "variance_h_per_site_chain_contributions_sha256": array_sha256(variance),
        "chain_index_lineage": selector.CHAIN_LINEAGE,
        "raw_measurement_sweeps_stored": False,
        "sample_count": 128,
        "chain_count": len(energy),
        "measurement_sweeps": 16,
        "samples_are_autocorrelated": True,
        "chain_mcse_does_not_pool_optimizer_updates": True,
        "sampler_diagnostics": {"accepted_move_fraction_mean": 0.4},
        "full_spin_structure_factor_computed": False,
        "work_ledger": {"fixed_parameter_energy_samples": 128},
    }


def _lowering(case_id: str, rate: float, backbone: str) -> tuple[float, float]:
    arm_shift = 0.001 if backbone == "su2" else 0.0
    if case_id.startswith("square"):
        means = {0.001: 0.040, 0.002: 0.045, 0.005: 0.047}
        # lr=2e-3 and 5e-3 have a noisy paired gap and form a tie; lr=1e-3
        # has an essentially deterministic, resolved gap from the raw best.
        amplitudes = {0.001: 0.006, 0.002: -0.006, 0.005: 0.006}
    else:
        means = {0.001: 0.005, 0.002: 0.010, 0.005: 0.007}
        amplitudes = {0.001: 0.003, 0.002: 0.003, 0.005: 0.003}
    return means[rate] + arm_shift, amplitudes[rate]


def _write_checkpoint(
    root: Path,
    *,
    spec: dict[str, object],
    case_id: str,
    arm_id: str,
    backbone: str,
    rate: float,
    measurements: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, str]]:
    series = root / f"checkpoint_{case_id}_{arm_id}"
    series.mkdir(parents=True, exist_ok=True)
    generation = series / "step_00000050_test.pkl"
    params = {"weight": np.asarray([rate, len(case_id)], dtype=np.float64)}
    optimizer_state = {
        "step": np.asarray(50, dtype=np.int64),
        "moment": np.asarray([rate], dtype=np.float64),
    }
    states = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    key = np.asarray([123, 456], dtype=np.uint32)
    metadata = {
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(optimizer_state),
        "chain_states_sha256": array_sha256(states),
        "sampling_key_sha256": array_sha256(key),
        "chain_states": states,
        "sampling_key": key,
        "fixed_parameter_measurements": measurements,
    }
    identity = {
        "schema": "synthetic_selector_parent",
        "task": scan_v2.TASK,
        "optimizer_backend": scan_v2.OPTIMIZER_BACKEND,
        "config_sha256": canonical_sha256(spec),
        "source_manifest_sha256": canonical_sha256(scan_v2._source_manifest()),
        "runtime_identity_sha256": canonical_sha256(SYNTHETIC_RUNTIME),
        "common_scan_profile_sha256": scan_v2.common_scan_profile_sha256(spec),
        "scan_family_id": spec["scan_family_id"],
        "candidate_id": spec["candidate_id"],
        "case_id": case_id,
        "arm_id": arm_id,
        "backbone": backbone,
    }
    save_resumable_checkpoint(
        generation,
        params,
        optimizer_state=optimizer_state,
        completed_steps=50,
        next_sample_index=1600,
        run_identity=identity,
        history=[{"update": index + 1} for index in range(50)],
        metadata=metadata,
    )
    checkpoint = {
        "series_directory": series.name,
        "latest_generation": generation.name,
        "latest_generation_sha256": file_sha256(generation),
        "run_identity_sha256": canonical_json_sha256(identity),
        "completed_updates": 50,
        "params_sha256": metadata["params_sha256"],
        "optimizer_state_sha256": metadata["optimizer_state_sha256"],
        "chain_states_sha256": metadata["chain_states_sha256"],
        "sampling_key_sha256": metadata["sampling_key_sha256"],
        "amplitude_cache_present": False,
    }
    return checkpoint, metadata


def _trial(
    root: Path,
    *,
    spec: dict[str, object],
    case_id: str,
    backbone: str,
    rate: float,
) -> dict[str, object]:
    arm = next(item for item in spec["arms"] if item["backbone"] == backbone)
    arm_id = str(arm["arm_id"])
    base_energy = -0.42 if case_id.startswith("square") else -0.31
    base_energy += -0.015 if backbone == "su2" else 0.0
    energy0 = base_energy + 0.002 * PATTERN
    lowering_mean, lowering_amplitude = _lowering(case_id, rate, backbone)
    lowering = lowering_mean + lowering_amplitude * PATTERN
    energy50 = energy0 - lowering
    energy25 = 0.5 * (energy0 + energy50)
    variance0 = 0.40 + 0.01 * PATTERN
    variance50 = variance0 - 0.03 + 0.004 * np.roll(PATTERN, 1)
    variance25 = 0.5 * (variance0 + variance50)
    measurements = [
        _measurement(
            spec=spec,
            case_id=case_id,
            arm_id=arm_id,
            update=update,
            energy=energy,
            variance=variance,
        )
        for update, energy, variance in (
            (0, energy0, variance0),
            (25, energy25, variance25),
            (50, energy50, variance50),
        )
    ]
    endpoint = scan_v2._endpoint_policy_diagnostics(measurements, spec)
    checkpoint, metadata = _write_checkpoint(
        root,
        spec=spec,
        case_id=case_id,
        arm_id=arm_id,
        backbone=backbone,
        rate=rate,
        measurements=measurements,
    )
    initial_prefix = f"initial-{case_id}-{backbone}"
    trial = {
        "status": "complete",
        "hard_failure": None,
        "candidate_id": spec["candidate_id"],
        "learning_rate": rate,
        "optimizer_backend": scan_v2.OPTIMIZER_BACKEND,
        "common_scan_profile_sha256": scan_v2.common_scan_profile_sha256(spec),
        "arm_id": arm_id,
        "backbone": backbone,
        "seed": int(arm["seed"]),
        "initial_params_sha256": f"{initial_prefix}-params",
        "final_params_sha256": metadata["params_sha256"],
        "initial_chain_states_sha256": f"{initial_prefix}-raw-chains",
        "initial_post_burn_chain_states_sha256": f"{initial_prefix}-burn-chains",
        "initial_training_key_sha256": f"{initial_prefix}-raw-key",
        "initial_post_burn_training_key_sha256": f"{initial_prefix}-burn-key",
        "final_chain_states_sha256": metadata["chain_states_sha256"],
        "history": [{"update": index + 1} for index in range(50)],
        "fixed_parameter_measurements": measurements,
        "selection_diagnostics": {
            "last_updates_accepted_move_fraction_median": 0.35,
            "relative_update_quantile_value": 0.01,
        },
        "endpoint_policy_diagnostics": endpoint,
        "readiness_gates": {"synthetic_finite": True},
        "all_readiness_gates_passed": True,
        "checkpoint": checkpoint,
        "claim_boundary": scan_v2.CLAIM_BOUNDARY,
    }
    trial["trial_payload_sha256"] = trial_payload_sha256(trial)
    return trial


def _write_scan_artifacts(tmp_path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    source = scan_v2._source_manifest()
    runtime = SYNTHETIC_RUNTIME
    for config_path in SCAN_CONFIGS:
        spec = json.loads(config_path.read_text())
        rate = float(spec["optimizer"]["learning_rate"])
        output = tmp_path / str(spec["candidate_id"])
        output.mkdir(parents=True)
        cases = []
        for case in spec["cases"]:
            case_id = str(case["case_id"])
            trials = [
                _trial(
                    output,
                    spec=spec,
                    case_id=case_id,
                    backbone=backbone,
                    rate=rate,
                )
                for backbone in ("u1", "su2")
            ]
            cases.append(
                {
                    "case_id": case_id,
                    "geometry": case["geometry"],
                    "trials": trials,
                }
            )
        payload = {
            "schema_version": 2,
            "domain": "spin",
            "task": scan_v2.TASK,
            "optimizer_backend": scan_v2.OPTIMIZER_BACKEND,
            "scan_family_id": spec["scan_family_id"],
            "candidate_id": spec["candidate_id"],
            "learning_rate": rate,
            "status": "complete",
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": canonical_sha256(spec),
            "config": spec,
            "common_scan_profile": scan_v2._common_scan_profile(spec),
            "common_scan_profile_sha256": scan_v2.common_scan_profile_sha256(spec),
            "source_sha256": source,
            "source_manifest_sha256": canonical_sha256(source),
            "runtime_identity": runtime,
            "runtime_identity_sha256": canonical_sha256(runtime),
            "cases": cases,
            "terminal_trial_count": 4,
            "hard_failed_trial_count": 0,
            "all_non_hard_failed_candidates_completed": True,
            "claim_boundary": scan_v2.CLAIM_BOUNDARY,
        }
        manifest = embedded_trial_manifest(payload)
        payload["trial_manifest"] = manifest
        payload["trial_manifest_sha256"] = canonical_sha256(manifest)
        result_path = output / "results.json"
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        entries.append(
            {
                "candidate_id": spec["candidate_id"],
                "learning_rate": rate,
                "config_path": str(config_path),
                "result_path": str(result_path),
            }
        )
    return entries


def _selector_spec(entries: list[dict[str, object]]) -> dict[str, object]:
    spec = json.loads(SELECTOR_CONFIG.read_text())
    spec["input_candidates"] = entries
    return spec


def test_registered_selector_config_is_exact_and_cli_is_registered() -> None:
    spec = json.loads(SELECTOR_CONFIG.read_text())
    selector._validate_spec(spec)
    assert spec["selection_policy"] == scan_v2._expected_selection_policy(50)
    assert spec["fallback_extension_contract"] == selector._expected_extension_contract()
    source = (ROOT / "projects/clebsch_tree_nqs/run.py").read_text()
    assert '"sampled-optimizer-select-v2"' in source
    assert "run_spin_sampled_training_v2_selector" in source


def test_selector_uses_paired_tie_and_emits_primary_and_unique_fallback(
    tmp_path: Path,
) -> None:
    entries = _write_scan_artifacts(tmp_path / "inputs")
    spec = _selector_spec(entries)
    config = tmp_path / "selector.json"
    config.write_text(json.dumps(spec))
    result = selector.run_spin_sampled_training_v2_selector(
        config, tmp_path / "selected", overwrite=True
    )
    assert result["status"] == "complete"
    assert result["result_payload_sha256"] == selector._result_payload_sha256(result)
    geometries = {item["geometry"]: item for item in result["geometry_results"]}

    square = geometries["square"]["decision"]
    assert square["action"] == (
        "freeze_lr_pending_reference_targets_then_heldout_64x200"
    )
    assert square["tie_audit"]["raw_best_candidate_id"] == "lr_5e-3"
    assert square["selected_candidate_id"] == "lr_2e-3"
    comparison = next(
        item
        for item in square["tie_audit"]["tie_comparisons"]
        if item["candidate_id"] == "lr_2e-3"
    )
    arm_ses = [
        item["paired_E50_other_minus_raw_best_se"]
        for item in comparison["same_arm_cross_lr_pairing"].values()
    ]
    assert comparison["geometry_score_gap_se_upper_bound"] == pytest.approx(
        0.5 * sum(arm_ses)
    )
    assert comparison["cross_arm_covariance_estimated"] is False
    assert comparison["in_raw_best_tie_set"] is True

    triangular = geometries["triangular"]["decision"]
    assert triangular["action"] == "run_single_step50_to100_extension"
    assert triangular["selected_candidate_id"] == "lr_2e-3"
    assert set(triangular["fallback_parent_bindings"]) == {"u1", "su2"}
    for binding in triangular["fallback_parent_bindings"].values():
        assert set(spec["fallback_extension_contract"]["parent_binding_required"]) <= set(
            binding
        )
    assert result["cross_candidate_audit"]["wall_time_read_or_used"] is False
    assert all(item["wall_time_read_or_used"] is False for item in result["geometry_results"])

    # A second invocation revalidates all input file hashes and returns the seal.
    again = selector.run_spin_sampled_training_v2_selector(
        config, tmp_path / "selected", overwrite=False
    )
    assert again["result_payload_sha256"] == result["result_payload_sha256"]


def test_selector_fails_closed_on_update_zero_or_trial_seal_tampering(
    tmp_path: Path,
) -> None:
    entries = _write_scan_artifacts(tmp_path / "inputs")
    tampered_path = Path(str(entries[1]["result_path"]))
    payload = json.loads(tampered_path.read_text())
    measurement = payload["cases"][0]["trials"][0]["fixed_parameter_measurements"][0]
    measurement["energy_per_site_chain_means"][0] += 0.1
    # Even if an attacker refreshes only the outer trial/result seals, the
    # independently stored per-chain array seal must fail.
    trial = payload["cases"][0]["trials"][0]
    trial["trial_payload_sha256"] = trial_payload_sha256(trial)
    manifest = embedded_trial_manifest(payload)
    payload["trial_manifest"] = manifest
    payload["trial_manifest_sha256"] = canonical_sha256(manifest)
    tampered_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    spec = _selector_spec(entries)
    config = tmp_path / "selector.json"
    config.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="vector seal failed"):
        selector.run_spin_sampled_training_v2_selector(
            config, tmp_path / "selected", overwrite=True
        )


def test_selector_outputs_no_adam_promotion_when_no_fallback_pool(
    tmp_path: Path,
) -> None:
    entries = _write_scan_artifacts(tmp_path / "inputs")
    for entry in entries:
        result_path = Path(str(entry["result_path"]))
        payload = json.loads(result_path.read_text())
        for case in payload["cases"]:
            if case["geometry"] != "triangular":
                continue
            for trial in case["trials"]:
                trial["selection_diagnostics"][
                    "last_updates_accepted_move_fraction_median"
                ] = 0.01
                trial["trial_payload_sha256"] = trial_payload_sha256(trial)
        manifest = embedded_trial_manifest(payload)
        payload["trial_manifest"] = manifest
        payload["trial_manifest_sha256"] = canonical_sha256(manifest)
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    spec = _selector_spec(entries)
    config = tmp_path / "selector.json"
    config.write_text(json.dumps(spec))
    result = selector.run_spin_sampled_training_v2_selector(
        config, tmp_path / "selected", overwrite=True
    )
    triangular = next(
        item for item in result["geometry_results"] if item["geometry"] == "triangular"
    )
    assert triangular["decision"]["action"] == "no_adam_promotion"
    assert triangular["decision"]["selected_candidate_id"] is None
    assert triangular["decision"]["recommended_next_route"] == (
        "matrix_free_physical_sr"
    )
