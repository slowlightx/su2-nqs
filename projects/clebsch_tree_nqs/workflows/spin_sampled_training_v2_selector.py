"""Sealed cross-artifact selector for the center-only 6x6 LR scan.

The three learning-rate candidates are produced by independent v2 workflow
invocations.  This module is deliberately separate from that runner: it never
trains a model, never reads wall time as a decision variable, and never edits a
parent checkpoint.  It verifies the three artifacts against the current v2
source/configuration, reconstructs paired-chain endpoint statistics, and makes
one geometry-common U1/SU2 decision per lattice.

The selector is still a single-seed optimizer-route diagnostic.  Its output is
not a converged-energy, efficiency, phase, or QSL claim.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2 as scan_v2
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.resumable_checkpoint import (
    canonical_json_sha256,
    load_resumable_checkpoint,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "physical_sampled_optimizer_scan_v2_selector"
RESULT_SCHEMA_VERSION = 1
CLAIM_BOUNDARY = (
    "single-seed sealed optimizer-route selection only; no converged-energy, "
    "wall-time efficiency, magnetic-order, phase, QSL, or physics-win claim"
)
EXPECTED_LEARNING_RATES = (1.0e-3, 2.0e-3, 5.0e-3)
EXPECTED_CASES = {
    "square_6x6_j2_0p5": "square",
    "triangular_6x6_j2_0p125": "triangular",
}
EXPECTED_ARMS = {
    "u1": "u1_D4_translation",
    "su2": "su2_D4_translation",
}
CHAIN_LINEAGE = "persistent_training_chain_index_then_independent_measurement_copy"


def _expected_extension_contract() -> dict[str, object]:
    """Return the immutable contract for the sole permitted 50->100 extension."""

    return {
        "schema": "physical_sampled_optimizer_extension_v1",
        "trigger": "selector_action_run_single_step50_to100_extension_only",
        "maximum_uses_per_geometry": 1,
        "parent_binding_required": [
            "result_file_sha256",
            "config_sha256",
            "source_manifest_sha256",
            "runtime_identity_sha256",
            "trial_payload_sha256",
            "checkpoint_generation_sha256",
            "checkpoint_run_identity_sha256",
            "params_sha256",
            "optimizer_state_sha256",
            "chain_states_sha256",
            "sampling_key_sha256",
        ],
        "parent_step": 50,
        "start_update": 51,
        "final_update": 100,
        "new_measurement_updates": [75, 100],
        "inherited_parent_measurement_updates": [0, 25, 50],
        "resume_exact_parent_params_adam_chains_and_key": True,
        "new_checkpoint_schema_and_series_required": True,
        "parent_checkpoint_identity_must_not_be_rewritten": True,
        "parent_checkpoint_files_must_not_be_modified": True,
        "only_selected_candidate_and_both_arms_continue": True,
        "learning_rate_locked": True,
        "all_other_sampler_model_optimizer_choices_locked": True,
        "unselected_candidates_forbidden": True,
    }


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        # Selector outputs must be able to revalidate external/synthetic inputs
        # on a second invocation; retaining the absolute path is therefore
        # preferable to the non-resolvable display-only ``external/name`` form.
        return str(resolved)


def _resolve_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact paths must be nonempty strings")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _result_payload_sha256(payload: Mapping[str, object]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_payload_sha256"}
    )


def _validate_spec(spec: dict[str, object]) -> None:
    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
    ):
        raise ValueError("selector requires schema_version=1, domain=spin, and its task")
    if spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("selector claim_boundary must match the enforced boundary")
    if not isinstance(spec.get("scan_family_id"), str) or not spec["scan_family_id"]:
        raise ValueError("scan_family_id must be a nonempty string")
    if spec.get("expected_cases") != EXPECTED_CASES:
        raise ValueError("selector expected_cases must match the registered 6x6 cases")
    if spec.get("expected_arms") != EXPECTED_ARMS:
        raise ValueError("selector expected_arms must match the registered U1/SU2 arms")
    if spec.get("selection_policy") != scan_v2._expected_selection_policy(50):
        raise ValueError("selector must seal the exact registered v2 selection policy")
    if spec.get("fallback_extension_contract") != _expected_extension_contract():
        raise ValueError("selector fallback extension contract is not registered")

    inputs = spec.get("input_candidates")
    if not isinstance(inputs, list) or len(inputs) != len(EXPECTED_LEARNING_RATES):
        raise ValueError("selector requires exactly three candidate artifacts")
    identifiers: set[str] = set()
    rates: list[float] = []
    for item in inputs:
        if not isinstance(item, dict):
            raise TypeError("every input candidate must be an object")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in identifiers:
            raise ValueError("candidate_id values must be nonempty and unique")
        identifiers.add(candidate_id)
        rate = float(item.get("learning_rate", np.nan))
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("candidate learning rates must be positive and finite")
        rates.append(rate)
        for name in ("config_path", "result_path"):
            if not isinstance(item.get(name), str) or not item[name]:
                raise ValueError(f"input candidate requires {name}")
    if tuple(sorted(rates)) != EXPECTED_LEARNING_RATES:
        raise ValueError("selector learning rates must be 1e-3, 2e-3, and 5e-3")


def _close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=2.0e-12, abs_tol=2.0e-14):
        raise RuntimeError(f"numeric audit failed for {name}: {actual} != {expected}")


def _measurement_map(trial: Mapping[str, object]) -> dict[int, dict[str, object]]:
    raw = trial.get("fixed_parameter_measurements")
    if not isinstance(raw, list):
        raise RuntimeError("trial has no fixed-parameter measurement list")
    result: dict[int, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("fixed-parameter measurement is not an object")
        update = int(item.get("update", -1))
        if update in result:
            raise RuntimeError("duplicate fixed-parameter measurement update")
        result[update] = item
    return result


def _checked_chain_vector(
    measurement: Mapping[str, object], value_name: str, hash_name: str
) -> np.ndarray:
    values = np.asarray(measurement.get(value_name), dtype=np.float64)
    chain_count = int(measurement.get("chain_count", -1))
    if chain_count < 2 or values.shape != (chain_count,) or np.any(~np.isfinite(values)):
        raise RuntimeError(f"invalid per-chain measurement vector: {value_name}")
    if measurement.get(hash_name) != array_sha256(values):
        raise RuntimeError(f"per-chain measurement vector seal failed: {value_name}")
    if measurement.get("chain_index_lineage") != CHAIN_LINEAGE:
        raise RuntimeError("measurement does not preserve registered chain-index lineage")
    return values


def _audit_measurement(
    measurement: dict[str, object],
    *,
    measurement_seed: int,
    case_id: str,
    arm_id: str,
) -> dict[str, np.ndarray]:
    if measurement.get("raw_measurement_sweeps_stored") is not False:
        raise RuntimeError("selector expects chain summaries, not unsealed raw sweeps")
    if measurement.get("independent_measurement_random_stream") is not True:
        raise RuntimeError("fixed measurement is not isolated from the training stream")
    update = int(measurement["update"])
    fold = int(measurement["measurement_random_stream_fold"])
    expected_stream_hash = canonical_sha256(
        {
            "measurement_seed": measurement_seed,
            "fold": fold,
            "case_id": case_id,
            "arm_id": arm_id,
            "update": update,
        }
    )
    if measurement.get("measurement_random_stream_identity_sha256") != expected_stream_hash:
        raise RuntimeError("measurement random-stream identity seal failed")

    energy = _checked_chain_vector(
        measurement,
        "energy_per_site_chain_means",
        "energy_per_site_chain_means_sha256",
    )
    variance = _checked_chain_vector(
        measurement,
        "variance_h_per_site_chain_contributions",
        "variance_h_per_site_chain_contributions_sha256",
    )
    if energy.shape != variance.shape:
        raise RuntimeError("energy and variance chain vectors have different sizes")
    chain_count = len(energy)
    _close(np.mean(energy), float(measurement["energy_per_site"]), name="energy mean")
    _close(
        np.std(energy, ddof=1) / np.sqrt(chain_count),
        float(measurement["energy_per_site_chain_mcse"]),
        name="energy chain MCSE",
    )
    _close(
        np.mean(variance),
        float(measurement["variance_h_per_site"]),
        name="variance mean",
    )
    _close(
        np.std(variance, ddof=1) / np.sqrt(chain_count),
        float(measurement["variance_h_per_site_chain_mcse"]),
        name="variance chain MCSE",
    )
    return {"energy": energy, "variance": variance}


def _audit_complete_trial_metrics(
    trial: dict[str, object],
    *,
    policy: dict[str, object],
    measurement_seed: int,
    case_id: str,
    arm_id: str,
) -> dict[str, object]:
    measurements = _measurement_map(trial)
    if sorted(measurements) != [0, 25, 50]:
        raise RuntimeError("complete scan trial must contain update 0/25/50 measurements")
    audited = {
        update: _audit_measurement(
            measurement,
            measurement_seed=measurement_seed,
            case_id=case_id,
            arm_id=arm_id,
        )
        for update, measurement in measurements.items()
    }
    energy_lowering_chains = audited[0]["energy"] - audited[50]["energy"]
    variance_worsening_chains = audited[50]["variance"] - audited[0]["variance"]
    chains = len(energy_lowering_chains)
    energy_lowering = float(np.mean(energy_lowering_chains))
    energy_se = float(np.std(energy_lowering_chains, ddof=1) / np.sqrt(chains))
    variance_worsening = float(np.mean(variance_worsening_chains))
    variance_se = float(np.std(variance_worsening_chains, ddof=1) / np.sqrt(chains))
    valid = all(
        math.isfinite(value)
        for value in (energy_lowering, energy_se, variance_worsening, variance_se)
    ) and energy_se > 0.0 and variance_se > 0.0
    energy_z = energy_lowering / energy_se if valid else None
    variance_z = variance_worsening / variance_se if valid else None
    energy_gate = bool(
        valid
        and energy_lowering >= float(policy["required_energy_lowering_per_site"])
        and float(energy_z) >= float(policy["minimum_energy_lowering_combined_mcse"])
    )
    variance_gate = bool(
        valid
        and float(variance_z)
        <= float(policy["maximum_variance_worsening_combined_mcse"])
    )

    endpoint = trial.get("endpoint_policy_diagnostics")
    if not isinstance(endpoint, dict):
        raise RuntimeError("complete scan trial lacks endpoint policy diagnostics")
    for name, value in (
        ("signed_energy_lowering_per_site", energy_lowering),
        ("energy_paired_chain_mcse_per_site", energy_se),
        ("signed_variance_worsening_per_site", variance_worsening),
        ("variance_paired_chain_mcse_per_site", variance_se),
    ):
        _close(float(endpoint[name]), value, name=f"endpoint.{name}")
    if endpoint.get("primary_endpoint_uncertainty") != (
        "same_chain_lineage_paired_difference_se"
    ):
        raise RuntimeError("endpoint diagnostics do not use paired chain uncertainty")
    if bool(endpoint.get("energy_endpoint_gate_passed")) != energy_gate:
        raise RuntimeError("stored energy endpoint gate disagrees with recomputation")
    if bool(endpoint.get("variance_endpoint_gate_passed")) != variance_gate:
        raise RuntimeError("stored variance endpoint gate disagrees with recomputation")

    selection = trial.get("selection_diagnostics")
    if not isinstance(selection, dict):
        raise RuntimeError("complete scan trial lacks stability diagnostics")
    stability_policy = dict(policy["stability_gates"])
    acceptance = selection.get("last_updates_accepted_move_fraction_median")
    relative = selection.get("relative_update_quantile_value")
    stable = bool(
        acceptance is not None
        and relative is not None
        and math.isfinite(float(acceptance))
        and math.isfinite(float(relative))
        and float(acceptance)
        >= float(stability_policy["last10_accepted_move_fraction_median_minimum"])
        and float(relative)
        <= float(stability_policy["relative_update_quantile_maximum"])
    )
    readiness = bool(trial.get("all_readiness_gates_passed"))
    complete = trial.get("status") == "complete" and trial.get("hard_failure") is None
    primary_eligible = complete and readiness and stable and energy_gate and variance_gate
    fallback_arm_eligible = bool(
        complete
        and readiness
        and stable
        and variance_gate
        and energy_z is not None
        and float(energy_z)
        >= float(
            policy["continuation_if_no_eligible_candidate"][
                "pool_minimum_each_arm_energy_lowering_paired_mcse"
            ]
        )
    )
    report = {
        "status_complete_without_hard_failure": complete,
        "all_readiness_gates_passed": readiness,
        "stability_gate_passed": stable,
        "last10_accepted_move_fraction_median": float(acceptance),
        "relative_update_quantile_value": float(relative),
        "energy_lowering_per_site": energy_lowering,
        "energy_lowering_paired_chain_se": energy_se,
        "energy_lowering_paired_z": energy_z,
        "energy_endpoint_gate_passed": energy_gate,
        "variance_worsening_per_site": variance_worsening,
        "variance_worsening_paired_chain_se": variance_se,
        "variance_worsening_paired_z": variance_z,
        "variance_endpoint_gate_passed": variance_gate,
        "primary_arm_eligible": primary_eligible,
        "fallback_arm_eligible": fallback_arm_eligible,
        "chain_count": chains,
        "endpoint_chain_vectors_sha256": {
            "energy_update0": array_sha256(audited[0]["energy"]),
            "energy_update50": array_sha256(audited[50]["energy"]),
            "variance_update0": array_sha256(audited[0]["variance"]),
            "variance_update50": array_sha256(audited[50]["variance"]),
        },
    }
    return {
        "report": report,
        "energy_update0": audited[0]["energy"],
        "energy_update50": audited[50]["energy"],
        "variance_update0": audited[0]["variance"],
        "variance_update50": audited[50]["variance"],
    }


def _audit_checkpoint(
    trial: dict[str, object],
    result_path: Path,
    *,
    expected_update: int,
    expected_identity_fields: Mapping[str, object],
) -> dict[str, object]:
    checkpoint = trial.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("complete trial has no checkpoint identity")
    generation = (
        result_path.parent
        / str(checkpoint["series_directory"])
        / str(checkpoint["latest_generation"])
    )
    if not generation.is_file():
        raise RuntimeError(f"missing selected parent checkpoint generation: {generation}")
    digest = file_sha256(generation)
    if digest != checkpoint.get("latest_generation_sha256"):
        raise RuntimeError("checkpoint generation file seal failed")
    payload = load_resumable_checkpoint(generation)
    if int(payload["completed_steps"]) != expected_update:
        raise RuntimeError("checkpoint is not at the registered final update")
    if canonical_json_sha256(payload["run_identity"]) != checkpoint.get(
        "run_identity_sha256"
    ):
        raise RuntimeError("checkpoint run-identity seal disagrees with result")
    for name, expected in expected_identity_fields.items():
        if payload["run_identity"].get(name) != expected:
            raise RuntimeError(f"checkpoint run identity mismatch: {name}")
    metadata = payload["metadata"]
    for name in (
        "params_sha256",
        "optimizer_state_sha256",
        "chain_states_sha256",
        "sampling_key_sha256",
    ):
        if metadata.get(name) != checkpoint.get(name):
            raise RuntimeError(f"checkpoint metadata/result mismatch: {name}")
    if pytree_sha256(payload["params"]) != checkpoint["params_sha256"]:
        raise RuntimeError("checkpoint parameter payload seal failed")
    if pytree_sha256(payload["optimizer_state"]) != checkpoint["optimizer_state_sha256"]:
        raise RuntimeError("checkpoint optimizer payload seal failed")
    if array_sha256(np.asarray(metadata["chain_states"], dtype=np.int32)) != checkpoint[
        "chain_states_sha256"
    ]:
        raise RuntimeError("checkpoint chain-state payload seal failed")
    if array_sha256(np.asarray(metadata["sampling_key"], dtype=np.uint32)) != checkpoint[
        "sampling_key_sha256"
    ]:
        raise RuntimeError("checkpoint sampling-key payload seal failed")
    if canonical_sha256(metadata.get("fixed_parameter_measurements")) != canonical_sha256(
        trial.get("fixed_parameter_measurements")
    ):
        raise RuntimeError("checkpoint/result fixed measurements disagree")
    if trial.get("final_params_sha256") != checkpoint["params_sha256"]:
        raise RuntimeError("trial final parameter hash disagrees with checkpoint")
    if trial.get("final_chain_states_sha256") != checkpoint["chain_states_sha256"]:
        raise RuntimeError("trial final chain hash disagrees with checkpoint")
    return {
        "checkpoint_generation_path": str(generation),
        "checkpoint_generation_sha256": digest,
        "checkpoint_run_identity_sha256": checkpoint["run_identity_sha256"],
        "params_sha256": checkpoint["params_sha256"],
        "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
        "chain_states_sha256": checkpoint["chain_states_sha256"],
        "sampling_key_sha256": checkpoint["sampling_key_sha256"],
    }


def _audit_candidate(
    entry: dict[str, object],
    *,
    selector_spec: dict[str, object],
    current_scan_source: dict[str, str],
) -> dict[str, object]:
    config_path = _resolve_path(entry["config_path"])
    result_path = _resolve_path(entry["result_path"])
    if not config_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"missing selector input: {config_path} or {result_path}")
    config = json.loads(config_path.read_text())
    scan_v2._validate_spec(config)
    if config.get("scan_family_id") != selector_spec["scan_family_id"]:
        raise RuntimeError("candidate scan_family_id mismatch")
    if config.get("candidate_id") != entry["candidate_id"]:
        raise RuntimeError("candidate_id does not match its current config")
    _close(
        float(config["optimizer"]["learning_rate"]),
        float(entry["learning_rate"]),
        name="candidate learning rate",
    )
    if config.get("selection_policy") != selector_spec["selection_policy"]:
        raise RuntimeError("candidate config selection policy differs from selector")

    payload = json.loads(result_path.read_text())
    if payload.get("schema_version") != 2 or payload.get("task") != scan_v2.TASK:
        raise RuntimeError("input is not a v2 center-only scan artifact")
    if payload.get("status") != "complete":
        raise RuntimeError("selector refuses a nonterminal scan artifact")
    if payload.get("config") != config:
        raise RuntimeError("embedded scan config differs from current config file")
    config_hash = canonical_sha256(config)
    if payload.get("config_sha256") != config_hash:
        raise RuntimeError("scan artifact config seal failed")
    if payload.get("candidate_id") != entry["candidate_id"]:
        raise RuntimeError("scan artifact candidate_id mismatch")
    _close(
        float(payload["learning_rate"]),
        float(entry["learning_rate"]),
        name="artifact learning rate",
    )
    if payload.get("source_sha256") != current_scan_source:
        raise RuntimeError("scan artifact was not produced by the current v2 source")
    scan_source_hash = canonical_sha256(current_scan_source)
    if payload.get("source_manifest_sha256") != scan_source_hash:
        raise RuntimeError("scan artifact source-manifest seal failed")
    runtime_identity = payload.get("runtime_identity")
    if not isinstance(runtime_identity, dict) or payload.get(
        "runtime_identity_sha256"
    ) != canonical_sha256(runtime_identity):
        raise RuntimeError("scan artifact runtime-identity seal failed")
    expected_profile = scan_v2._common_scan_profile(config)
    expected_profile_hash = canonical_sha256(expected_profile)
    if (
        payload.get("common_scan_profile") != expected_profile
        or payload.get("common_scan_profile_sha256") != expected_profile_hash
    ):
        raise RuntimeError("scan artifact common-profile seal failed")
    manifest = embedded_trial_manifest(payload)
    if payload.get("trial_manifest") != manifest or payload.get(
        "trial_manifest_sha256"
    ) != canonical_sha256(manifest):
        raise RuntimeError("scan artifact trial manifest seal failed")
    expected_trial_count = len(EXPECTED_CASES) * len(EXPECTED_ARMS)
    if (
        len(manifest) != expected_trial_count
        or payload.get("terminal_trial_count") != expected_trial_count
        or payload.get("all_non_hard_failed_candidates_completed") is not True
    ):
        raise RuntimeError("scan artifact terminal trial accounting is incomplete")

    cases = payload.get("cases")
    if not isinstance(cases, list) or {item.get("case_id") for item in cases} != set(
        EXPECTED_CASES
    ):
        raise RuntimeError("scan artifact case set is incomplete")
    audited_cases: dict[str, object] = {}
    measurement_seed = int(config["measurement"]["seed"])
    for case in cases:
        case_id = str(case["case_id"])
        if case.get("geometry") != EXPECTED_CASES[case_id]:
            raise RuntimeError("scan artifact geometry label mismatch")
        trials = case.get("trials")
        if not isinstance(trials, list) or len(trials) != len(EXPECTED_ARMS):
            raise RuntimeError("scan artifact does not contain exactly U1 and SU2 trials")
        by_backbone = {str(trial.get("backbone")): trial for trial in trials}
        if set(by_backbone) != set(EXPECTED_ARMS):
            raise RuntimeError("scan artifact backbone set is incomplete")
        audited_arms: dict[str, object] = {}
        for backbone, expected_arm_id in EXPECTED_ARMS.items():
            trial = by_backbone[backbone]
            if trial.get("arm_id") != expected_arm_id:
                raise RuntimeError("scan artifact arm_id mismatch")
            measurements = _measurement_map(trial)
            if 0 not in measurements:
                raise RuntimeError("every candidate must retain the common update-zero block")
            update_zero = _audit_measurement(
                measurements[0],
                measurement_seed=measurement_seed,
                case_id=case_id,
                arm_id=expected_arm_id,
            )
            if trial.get("status") == "complete":
                metrics = _audit_complete_trial_metrics(
                    trial,
                    policy=dict(selector_spec["selection_policy"]),
                    measurement_seed=measurement_seed,
                    case_id=case_id,
                    arm_id=expected_arm_id,
                )
                checkpoint = _audit_checkpoint(
                    trial,
                    result_path,
                    expected_update=50,
                    expected_identity_fields={
                        "task": scan_v2.TASK,
                        "optimizer_backend": scan_v2.OPTIMIZER_BACKEND,
                        "config_sha256": config_hash,
                        "source_manifest_sha256": scan_source_hash,
                        "runtime_identity_sha256": payload["runtime_identity_sha256"],
                        "common_scan_profile_sha256": expected_profile_hash,
                        "scan_family_id": selector_spec["scan_family_id"],
                        "candidate_id": entry["candidate_id"],
                        "case_id": case_id,
                        "arm_id": expected_arm_id,
                        "backbone": backbone,
                    },
                )
            elif trial.get("status") == "hard_failed":
                metrics = {
                    "report": {
                        "status_complete_without_hard_failure": False,
                        "all_readiness_gates_passed": False,
                        "stability_gate_passed": False,
                        "primary_arm_eligible": False,
                        "fallback_arm_eligible": False,
                        "hard_failure": deepcopy(trial.get("hard_failure")),
                    },
                    "energy_update0": update_zero["energy"],
                    "energy_update50": None,
                    "variance_update0": update_zero["variance"],
                    "variance_update50": None,
                }
                checkpoint = None
            else:
                raise RuntimeError("scan trial is neither complete nor hard_failed")
            audited_arms[backbone] = {
                "trial": trial,
                "metrics": metrics,
                "checkpoint": checkpoint,
                "update_zero_measurement_sha256": canonical_sha256(measurements[0]),
                "measurement_streams": {
                    str(update): measurement["measurement_random_stream_identity_sha256"]
                    for update, measurement in measurements.items()
                },
            }
        if (
            audited_arms["u1"]["measurement_streams"]["0"]
            == audited_arms["su2"]["measurement_streams"]["0"]
        ):
            raise RuntimeError("U1 and SU2 measurement streams must be distinct")
        audited_cases[case_id] = {"case": case, "arms": audited_arms}

    return {
        "candidate_id": str(entry["candidate_id"]),
        "learning_rate": float(entry["learning_rate"]),
        "config_path": config_path,
        "config_file_sha256": file_sha256(config_path),
        "config_sha256": config_hash,
        "result_path": result_path,
        "result_file_sha256": file_sha256(result_path),
        "source_manifest_sha256": scan_source_hash,
        "runtime_identity_sha256": str(payload["runtime_identity_sha256"]),
        "common_scan_profile_sha256": expected_profile_hash,
        "payload": payload,
        "cases": audited_cases,
    }


def _audit_cross_candidate_matching(candidates: list[dict[str, object]]) -> None:
    if len({item["runtime_identity_sha256"] for item in candidates}) != 1:
        raise RuntimeError("learning-rate candidates used different numerical runtimes")
    if len({item["common_scan_profile_sha256"] for item in candidates}) != 1:
        raise RuntimeError("learning-rate candidates do not share one common profile")
    initial_fields = (
        "initial_params_sha256",
        "initial_chain_states_sha256",
        "initial_post_burn_chain_states_sha256",
        "initial_training_key_sha256",
        "initial_post_burn_training_key_sha256",
    )
    for case_id in EXPECTED_CASES:
        for backbone in EXPECTED_ARMS:
            arms = [item["cases"][case_id]["arms"][backbone] for item in candidates]
            for field in initial_fields:
                if len({arm["trial"].get(field) for arm in arms}) != 1:
                    raise RuntimeError(
                        f"cross-LR initial identity mismatch: {case_id}/{backbone}/{field}"
                    )
            if len({arm["update_zero_measurement_sha256"] for arm in arms}) != 1:
                raise RuntimeError(
                    f"cross-LR update-zero mismatch: {case_id}/{backbone}"
                )
            updates = set.intersection(
                *(set(arm["measurement_streams"]) for arm in arms)
            )
            for update in updates:
                if len({arm["measurement_streams"][update] for arm in arms}) != 1:
                    raise RuntimeError(
                        f"cross-LR measurement stream mismatch: "
                        f"{case_id}/{backbone}/update{update}"
                    )


def _candidate_geometry_report(
    candidate: dict[str, object], case_id: str
) -> dict[str, object]:
    arms = candidate["cases"][case_id]["arms"]
    arm_reports = {
        backbone: deepcopy(arms[backbone]["metrics"]["report"])
        for backbone in EXPECTED_ARMS
    }
    if all(report.get("status_complete_without_hard_failure") for report in arm_reports.values()):
        score = 0.5 * sum(
            float(report["energy_lowering_per_site"]) for report in arm_reports.values()
        )
        score_se_upper = 0.5 * sum(
            float(report["energy_lowering_paired_chain_se"])
            for report in arm_reports.values()
        )
    else:
        score = None
        score_se_upper = None
    primary = bool(
        score is not None
        and all(report.get("primary_arm_eligible") for report in arm_reports.values())
    )
    fallback = bool(
        score is not None
        and float(score) > 0.0
        and all(report.get("fallback_arm_eligible") for report in arm_reports.values())
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "learning_rate": candidate["learning_rate"],
        "arms": arm_reports,
        "geometry_score_energy_lowering_per_site": score,
        "geometry_score_endpoint_se_worst_case_upper_bound": score_se_upper,
        "primary_candidate_eligible": primary,
        "fallback_candidate_eligible": fallback,
        "wall_time_read_or_used": False,
    }


def _tie_audit(
    *,
    raw_best: dict[str, object],
    other: dict[str, object],
    case_id: str,
    threshold: float,
) -> dict[str, object]:
    arm_rows: dict[str, object] = {}
    arm_ses: list[float] = []
    arm_gaps: list[float] = []
    for backbone in EXPECTED_ARMS:
        best_values = raw_best["cases"][case_id]["arms"][backbone]["metrics"][
            "energy_update50"
        ]
        other_values = other["cases"][case_id]["arms"][backbone]["metrics"][
            "energy_update50"
        ]
        if best_values is None or other_values is None:
            raise RuntimeError("tie comparison requires complete update-50 chain vectors")
        difference = np.asarray(other_values) - np.asarray(best_values)
        gap = float(np.mean(difference))
        se = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
        if not math.isfinite(gap) or not math.isfinite(se) or se < 0.0:
            raise RuntimeError("nonfinite paired cross-candidate tie statistic")
        arm_rows[backbone] = {
            "paired_E50_other_minus_raw_best_mean": gap,
            "paired_E50_other_minus_raw_best_se": se,
            "paired_difference_sha256": array_sha256(difference.astype(np.float64)),
        }
        arm_gaps.append(gap)
        arm_ses.append(se)
    score_gap = 0.5 * sum(arm_gaps)
    expected_gap = float(raw_best["geometry_report"][
        "geometry_score_energy_lowering_per_site"
    ]) - float(other["geometry_report"]["geometry_score_energy_lowering_per_site"])
    _close(score_gap, expected_gap, name="cross-candidate geometry score gap")
    worst_case_se = 0.5 * sum(arm_ses)
    tied = bool(score_gap <= threshold * worst_case_se)
    return {
        "candidate_id": other["candidate_id"],
        "learning_rate": other["learning_rate"],
        "raw_best_candidate_id": raw_best["candidate_id"],
        "geometry_score_gap": score_gap,
        "same_arm_cross_lr_pairing": arm_rows,
        "cross_arm_covariance_estimated": False,
        "cross_arm_uncertainty_rule": "cauchy_worst_case_sum_upper_bound",
        "geometry_score_gap_se_upper_bound": worst_case_se,
        "tie_threshold_mcse": threshold,
        "in_raw_best_tie_set": tied,
    }


def _choose_from_pool(
    pool: list[dict[str, object]], *, case_id: str, threshold: float
) -> tuple[dict[str, object], dict[str, object]]:
    if not pool:
        raise ValueError("cannot choose from an empty candidate pool")
    ordered = sorted(
        pool,
        key=lambda item: (
            -float(item["geometry_report"]["geometry_score_energy_lowering_per_site"]),
            float(item["learning_rate"]),
            str(item["candidate_id"]),
        ),
    )
    raw_best = ordered[0]
    comparisons = [
        _tie_audit(
            raw_best=raw_best,
            other=candidate,
            case_id=case_id,
            threshold=threshold,
        )
        for candidate in ordered
    ]
    tied_ids = {
        item["candidate_id"] for item in comparisons if item["in_raw_best_tie_set"]
    }
    selected = min(
        (candidate for candidate in ordered if candidate["candidate_id"] in tied_ids),
        key=lambda item: (float(item["learning_rate"]), str(item["candidate_id"])),
    )
    return selected, {
        "raw_best_candidate_id": raw_best["candidate_id"],
        "raw_best_learning_rate": raw_best["learning_rate"],
        "tie_comparisons": comparisons,
        "tie_set_candidate_ids": sorted(tied_ids),
        "selected_candidate_id": selected["candidate_id"],
        "selected_learning_rate": selected["learning_rate"],
        "tie_breaker": "smallest_learning_rate_within_raw_best_2se_upper_bound",
    }


def _parent_bindings(
    candidate: dict[str, object], case_id: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for backbone in EXPECTED_ARMS:
        arm = candidate["cases"][case_id]["arms"][backbone]
        checkpoint = arm["checkpoint"]
        if checkpoint is None:
            raise RuntimeError("fallback candidate has no valid step-50 checkpoint")
        trial = arm["trial"]
        result[backbone] = {
            "result_path": _portable_path(candidate["result_path"]),
            "result_file_sha256": candidate["result_file_sha256"],
            "config_sha256": candidate["config_sha256"],
            "source_manifest_sha256": candidate["source_manifest_sha256"],
            "runtime_identity_sha256": candidate["runtime_identity_sha256"],
            "trial_payload_sha256": trial["trial_payload_sha256"],
            "checkpoint_generation_path": _portable_path(
                Path(checkpoint["checkpoint_generation_path"])
            ),
            "checkpoint_generation_sha256": checkpoint[
                "checkpoint_generation_sha256"
            ],
            "checkpoint_run_identity_sha256": checkpoint[
                "checkpoint_run_identity_sha256"
            ],
            "params_sha256": checkpoint["params_sha256"],
            "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
            "chain_states_sha256": checkpoint["chain_states_sha256"],
            "sampling_key_sha256": checkpoint["sampling_key_sha256"],
        }
    return result


def _select_geometry(
    candidates: list[dict[str, object]],
    *,
    case_id: str,
    policy: dict[str, object],
    extension_contract: dict[str, object],
) -> dict[str, object]:
    decorated: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for candidate in candidates:
        report = _candidate_geometry_report(candidate, case_id)
        decorated_candidate = {**candidate, "geometry_report": report}
        decorated.append(decorated_candidate)
        reports.append(report)
    primary_pool = [
        candidate
        for candidate in decorated
        if candidate["geometry_report"]["primary_candidate_eligible"]
    ]
    threshold = float(policy["geometry_selection"]["tie_threshold_combined_mcse"])
    if primary_pool:
        selected, tie = _choose_from_pool(
            primary_pool, case_id=case_id, threshold=threshold
        )
        action = "freeze_lr_pending_reference_targets_then_heldout_64x200"
        decision = {
            "action": action,
            "selected_candidate_id": selected["candidate_id"],
            "selected_learning_rate": selected["learning_rate"],
            "selection_pool": "primary_eligible",
            "tie_audit": tie,
            "heldout_policy": deepcopy(policy["heldout_policy"]),
            "heldout_blocked_until_absolute_energy_targets_are_frozen": True,
            "fallback_parent_bindings": None,
        }
    else:
        fallback_pool = [
            candidate
            for candidate in decorated
            if candidate["geometry_report"]["fallback_candidate_eligible"]
        ]
        if fallback_pool:
            selected, tie = _choose_from_pool(
                fallback_pool, case_id=case_id, threshold=threshold
            )
            decision = {
                "action": "run_single_step50_to100_extension",
                "selected_candidate_id": selected["candidate_id"],
                "selected_learning_rate": selected["learning_rate"],
                "selection_pool": "single_preregistered_falsification",
                "tie_audit": tie,
                "fallback_extension_contract": deepcopy(extension_contract),
                "fallback_promotion_gate": deepcopy(
                    policy["continuation_if_no_eligible_candidate"][
                        "promotion_0_to_100_each_arm"
                    ]
                ),
                "fallback_failure_action": policy[
                    "continuation_if_no_eligible_candidate"
                ]["failure_action"],
                "fallback_parent_bindings": _parent_bindings(selected, case_id),
            }
        else:
            decision = {
                "action": "no_adam_promotion",
                "selected_candidate_id": None,
                "selected_learning_rate": None,
                "selection_pool": "empty",
                "tie_audit": None,
                "fallback_parent_bindings": None,
                "recommended_next_route": "matrix_free_physical_sr",
            }
    return {
        "case_id": case_id,
        "geometry": EXPECTED_CASES[case_id],
        "candidate_diagnostics": sorted(
            reports, key=lambda item: float(item["learning_rate"])
        ),
        "decision": decision,
        "wall_time_read_or_used": False,
        "physics_or_efficiency_claim": False,
    }


def run_spin_sampled_training_v2_selector(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    """Audit three completed v2 candidates and seal the preregistered decision."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    source_manifest = _source_manifest()
    source_hash = canonical_sha256(source_manifest)
    current_scan_source = scan_v2._source_manifest()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text())
        if (
            existing.get("schema_version") != RESULT_SCHEMA_VERSION
            or existing.get("config_sha256") != config_hash
            or existing.get("source_manifest_sha256") != source_hash
            or existing.get("result_payload_sha256") != _result_payload_sha256(existing)
        ):
            raise RuntimeError(f"refusing stale or unsealed selector result: {result_path}")
        for item in existing.get("input_manifest", []):
            current = _resolve_path(item["result_path"])
            if not current.is_file() or file_sha256(current) != item["result_file_sha256"]:
                raise RuntimeError("selector input changed after the decision was sealed")
        return existing

    candidates = [
        _audit_candidate(
            dict(entry),
            selector_spec=spec,
            current_scan_source=current_scan_source,
        )
        for entry in spec["input_candidates"]
    ]
    _audit_cross_candidate_matching(candidates)
    policy = dict(spec["selection_policy"])
    extension_contract = dict(spec["fallback_extension_contract"])
    geometry_results = [
        _select_geometry(
            candidates,
            case_id=case_id,
            policy=policy,
            extension_contract=extension_contract,
        )
        for case_id in EXPECTED_CASES
    ]
    payload: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": _portable_path(config_path),
        "config_sha256": config_hash,
        "config": spec,
        "source_sha256": source_manifest,
        "source_manifest_sha256": source_hash,
        "current_scan_source_manifest_sha256": canonical_sha256(current_scan_source),
        "scan_family_id": spec["scan_family_id"],
        "common_scan_profile_sha256": candidates[0]["common_scan_profile_sha256"],
        "runtime_identity_sha256": candidates[0]["runtime_identity_sha256"],
        "input_manifest": [
            {
                "candidate_id": candidate["candidate_id"],
                "learning_rate": candidate["learning_rate"],
                "config_path": _portable_path(candidate["config_path"]),
                "config_file_sha256": candidate["config_file_sha256"],
                "config_sha256": candidate["config_sha256"],
                "result_path": _portable_path(candidate["result_path"]),
                "result_file_sha256": candidate["result_file_sha256"],
                "source_manifest_sha256": candidate["source_manifest_sha256"],
                "runtime_identity_sha256": candidate["runtime_identity_sha256"],
                "common_scan_profile_sha256": candidate[
                    "common_scan_profile_sha256"
                ],
            }
            for candidate in sorted(candidates, key=lambda item: item["learning_rate"])
        ],
        "cross_candidate_audit": {
            "current_v2_source_required": True,
            "input_config_and_trial_seals_verified": True,
            "input_checkpoint_generation_and_state_hashes_verified": True,
            "common_profile_and_runtime_required": True,
            "initial_params_chains_and_training_keys_equal_across_lr": True,
            "update_zero_blocks_bitwise_equal_across_lr": True,
            "measurement_stream_identities_equal_across_lr_per_case_arm_update": True,
            "measurement_stream_identities_distinct_between_u1_su2": True,
            "per_chain_array_hashes_and_summary_statistics_verified": True,
            "same_arm_cross_lr_E50_differences_paired_by_chain": True,
            "cross_arm_covariance_not_assumed": True,
            "cross_arm_se_uses_cauchy_worst_case_sum_upper_bound": True,
            "wall_time_read_or_used": False,
        },
        "selection_policy": policy,
        "fallback_extension_contract": extension_contract,
        "geometry_results": geometry_results,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_payload_sha256"] = _result_payload_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "CLAIM_BOUNDARY",
    "TASK",
    "run_spin_sampled_training_v2_selector",
]
