"""Parent-bound 50->100 confirmatory center-only Adam extension.

This workflow starts a *new* checkpoint identity from the immutable update-50
states selected by the sealed metric-correction analysis.  The selector is an
input-integrity and learning-rate choice, not a training authorization: this
module implements the separately preregistered confirmatory campaign and
leaves every parent result and checkpoint byte untouched.

The extension is deliberately narrower than a production held-out campaign.
It continues only square ``lr=2e-3`` and triangular ``lr=1e-3`` U1/SU2 arms,
replays update zero with a long fixed-parameter measurement, and measures the
new trajectories at updates 75 and 100.  Raw per-sweep local energies and
diagonal ordering slow modes are retained so sampler mixing can be falsified
before an optimization claim is considered.  No full spin structure factor,
ED, phase, QSL, or efficiency claim is made here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime_workflow
from projects.clebsch_tree_nqs.workflows import spin_sampled_training as v1_helpers
from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2 as scan_v2
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    pytree_sha256,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.physical_sampled_adam_center_only import (
    SampledAdamState,
    sampled_adam_update,
    sampled_vmc_value_and_grad,
)
from su2_nqs.vmc.resumable_checkpoint import (
    canonical_json_sha256,
    load_latest_resumable_checkpoint,
    load_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "physical_sampled_optimizer_confirmatory_extension_v1"
RESULT_SCHEMA_VERSION = 1
OPTIMIZER_BACKEND = "center_only_sampled_adam_confirmatory_extension_v1"
SELECTOR_TASK = "physical_sampled_optimizer_scan_v2_metric_correction_selector"
EXPECTED_SELECTIONS = {
    "square_6x6_j2_0p5": {
        "geometry": "square",
        "candidate_id": "lr_2e-3",
        "learning_rate": 2.0e-3,
    },
    "triangular_6x6_j2_0p125": {
        "geometry": "triangular",
        "candidate_id": "lr_1e-3",
        "learning_rate": 1.0e-3,
    },
}
EXPECTED_ARMS = {"u1": "u1_D4_translation", "su2": "su2_D4_translation"}
CLAIM_BOUNDARY = (
    "single-seed parent-bound Adam confirmatory extension and sampler-mixing "
    "falsification only; no converged-energy, wall-time efficiency, magnetic-"
    "order, phase, QSL, heldout, or physics-win claim"
)


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _result_payload_sha256(payload: Mapping[str, object]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_payload_sha256"}
    )


def _expected_protocol() -> dict[str, object]:
    """Return the immutable training, measurement, gate, and routing contract."""

    return {
        "parent_step": 50,
        "start_update": 51,
        "final_update": 100,
        "training_chains": 32,
        "decorrelation_sweeps_per_update": 2,
        "microbatch_size": 8,
        "checkpoint_updates": [75, 100],
        "resume_exact_parent": [
            "params",
            "adam_state",
            "training_chains",
            "training_sampling_key",
        ],
        "fixed_measurement": {
            "updates": [0, 75, 100],
            "chains": 64,
            "parent_lineage_chains": 32,
            "independent_fresh_m0_chains": 32,
            "burn_sweeps": 64,
            "production_sweeps": 256,
            "raw_local_energy_shape": [256, 64],
            "raw_real_local_energy_stored": True,
            "raw_imaginary_local_energy_stored_for_variance_audit": True,
            "diagonal_slow_modes": {
                "square": [
                    "diagonal_Szz_Neel_Q",
                    "diagonal_Szz_stripe_star",
                ],
                "triangular": [
                    "diagonal_Szz_K_star",
                    "diagonal_Szz_M_star",
                ],
            },
            "full_spin_structure_factor_computed": False,
            "parent_short_update50_measurement_role": "trajectory_diagnostic_only",
            "baseline_update0_must_be_replayed_from_parent_generation": True,
            "missing_parent_update0_generation_action": "fail_closed",
        },
        "training_stability_gates": {
            "last10_conditional_move_acceptance_median_minimum": 0.15,
            "relative_update_quantile": 0.99,
            "relative_update_quantile_maximum": 0.02,
        },
        "fixed_measurement_mixing_gates": {
            "zero_amplitude_chain_fraction_maximum": 0.0,
            "frozen_chain_count_maximum": 0,
            "split_rhat_strict_maximum": 1.01,
            "runtime_initial_positive_window_ess_minimum": 1000.0,
            "rank_normalized_bulk_ess_computed": False,
            "production_sweeps_minimum_tau_multiple": 50.0,
            "checkpoint_vs_fresh_each_observable_maximum_combined_mcse": 2.0,
            "energy_first_vs_second_half_maximum_combined_mcse": 2.0,
        },
        "final_optimization_gates_each_arm": {
            "endpoint_updates": [0, 100],
            "minimum_energy_lowering_per_site": 0.05,
            "minimum_energy_lowering_paired_mcse": 5.0,
            "minimum_variance_fractional_decrease": 0.25,
            "minimum_variance_decrease_paired_mcse": 3.0,
            "evaluate_final_update_only": True,
        },
        "routing": {
            "any_mixing_or_training_stability_failure": "sampler_audit",
            "mixing_pass_but_any_optimization_failure": "matrix_free_SR",
            "all_four_arms_pass": "heldout",
        },
    }


def _validate_spec(spec: dict[str, object]) -> None:
    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
    ):
        raise ValueError("confirmatory extension requires its v1 spin schema/task")
    if spec.get("optimizer_backend") != OPTIMIZER_BACKEND:
        raise ValueError("confirmatory extension optimizer_backend mismatch")
    if spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("confirmatory extension claim boundary mismatch")
    if spec.get("protocol") != _expected_protocol():
        raise ValueError("confirmatory extension protocol must match preregistration")
    if spec.get("selector_is_training_authority") is not False:
        raise ValueError("metric-correction selector must not be called training authority")
    if spec.get("campaign_authorization") != (
        "separate_preregistered_confirmatory_campaign_requested_after_selector_audit"
    ):
        raise ValueError("extension requires its separate campaign authorization label")
    if not isinstance(spec.get("campaign_id"), str) or not spec["campaign_id"]:
        raise ValueError("campaign_id must be a nonempty string")
    if isinstance(spec.get("measurement_seed"), bool) or not isinstance(
        spec.get("measurement_seed"), int
    ):
        raise TypeError("measurement_seed must be an integer")

    selector = spec.get("selector_binding")
    required_selector = {
        "config_path",
        "config_file_sha256",
        "config_sha256",
        "result_path",
        "result_file_sha256",
        "result_payload_sha256",
        "source_manifest_sha256",
        "runtime_identity_sha256",
    }
    if not isinstance(selector, dict) or set(selector) != required_selector:
        raise ValueError("selector_binding has the wrong exact field set")
    if any(not isinstance(selector[name], str) or not selector[name] for name in selector):
        raise ValueError("selector binding values must be nonempty strings")

    selections = spec.get("selections")
    if not isinstance(selections, list) or len(selections) != 2:
        raise ValueError("exactly two geometry selections are required")
    seen: set[str] = set()
    for item in selections:
        if not isinstance(item, dict):
            raise TypeError("selection entries must be objects")
        case_id = str(item.get("case_id", ""))
        if case_id in seen or case_id not in EXPECTED_SELECTIONS:
            raise ValueError("selection case set is not the registered pair")
        seen.add(case_id)
        expected = EXPECTED_SELECTIONS[case_id]
        if (
            item.get("geometry") != expected["geometry"]
            or item.get("candidate_id") != expected["candidate_id"]
            or float(item.get("learning_rate", np.nan)) != expected["learning_rate"]
            or not isinstance(item.get("selector_decision_sha256"), str)
            or len(item["selector_decision_sha256"]) != 64
        ):
            raise ValueError("selection differs from the registered corrected decision")
        if item.get("selector_action") != (
            "freeze_lr_pending_reference_targets_then_heldout_64x200"
        ):
            raise ValueError("extension must retain the selector's literal action")
    if seen != set(EXPECTED_SELECTIONS):
        raise ValueError("selection case set is incomplete")


def _case_and_trial(
    payload: Mapping[str, object], case_id: str, backbone: str
) -> tuple[dict[str, object], dict[str, object]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("parent scan result has no case list")
    matched = [item for item in cases if item.get("case_id") == case_id]
    if len(matched) != 1:
        raise RuntimeError("parent scan result case identity is ambiguous")
    case = dict(matched[0])
    trials = case.get("trials")
    if not isinstance(trials, list):
        raise RuntimeError("parent scan case has no trial list")
    selected = [item for item in trials if item.get("backbone") == backbone]
    if len(selected) != 1:
        raise RuntimeError("parent scan backbone identity is ambiguous")
    return case, dict(selected[0])


def _verify_checkpoint_payload(
    path: Path,
    binding: Mapping[str, object],
    *,
    expected_step: int,
) -> dict[str, object]:
    if not path.is_file() or file_sha256(path) != binding["checkpoint_generation_sha256"]:
        raise RuntimeError("parent checkpoint generation file seal failed")
    payload = load_resumable_checkpoint(path)
    if int(payload["completed_steps"]) != expected_step:
        raise RuntimeError("parent checkpoint is at the wrong update")
    if (
        len(payload["history"]) != expected_step
        or int(payload["next_sample_index"]) != expected_step * 32
    ):
        raise RuntimeError("parent checkpoint counters are inconsistent")
    if canonical_json_sha256(payload["run_identity"]) != binding[
        "checkpoint_run_identity_sha256"
    ]:
        raise RuntimeError("parent checkpoint run identity seal failed")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("parent checkpoint metadata is invalid")
    expected_hashes = {
        "params_sha256": pytree_sha256(payload["params"]),
        "optimizer_state_sha256": pytree_sha256(payload["optimizer_state"]),
        "chain_states_sha256": array_sha256(
            np.asarray(metadata.get("chain_states"), dtype=np.int32)
        ),
        "sampling_key_sha256": array_sha256(
            np.asarray(metadata.get("sampling_key"), dtype=np.uint32)
        ),
    }
    for name, actual in expected_hashes.items():
        if metadata.get(name) != actual or binding.get(name) != actual:
            raise RuntimeError(f"parent checkpoint state seal failed: {name}")
    optimizer_state = payload["optimizer_state"]
    if not isinstance(optimizer_state, SampledAdamState) or optimizer_state.step != expected_step:
        raise RuntimeError("parent checkpoint does not contain the expected Adam state")
    return payload


def _find_and_verify_step_zero(
    step50_path: Path,
    step50_payload: Mapping[str, object],
    trial: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    """Locate the committed update-zero generation or fail closed."""

    latest_path = step50_path.parent / "LATEST.json"
    if not latest_path.is_file():
        raise RuntimeError("parent update-zero generation unavailable: no series manifest")
    series = json.loads(latest_path.read_text())
    if series.get("run_identity_sha256") != canonical_json_sha256(
        step50_payload["run_identity"]
    ):
        raise RuntimeError("parent checkpoint series run identity changed")
    series_entries = series.get("generations", [])
    if (
        not isinstance(series_entries, list)
        or sorted(int(item.get("completed_steps", -1)) for item in series_entries)
        != [0, 25, 50]
    ):
        raise RuntimeError("parent checkpoint series generation set changed")
    step50_entries = [
        item
        for item in series_entries
        if int(item.get("completed_steps", -1)) == 50
        and item.get("file") == step50_path.name
    ]
    if (
        len(step50_entries) != 1
        or step50_entries[0].get("sha256") != file_sha256(step50_path)
    ):
        raise RuntimeError("parent step-50 generation is not sealed by its series")
    entries = [
        item
        for item in series_entries
        if int(item.get("completed_steps", -1)) == 0
    ]
    if len(entries) != 1:
        raise RuntimeError("parent update-zero generation unavailable or ambiguous")
    entry = entries[0]
    path = step50_path.parent / str(entry["file"])
    if not path.is_file() or file_sha256(path) != entry.get("sha256"):
        raise RuntimeError("parent update-zero generation seal failed")
    payload = load_resumable_checkpoint(path)
    if int(payload["completed_steps"]) != 0:
        raise RuntimeError("parent baseline replay source is not update zero")
    if payload["history"] or int(payload["next_sample_index"]) != 0:
        raise RuntimeError("parent update-zero counters are inconsistent")
    if payload["run_identity"] != step50_payload["run_identity"]:
        raise RuntimeError("parent update-zero and update-50 run identities differ")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("parent update-zero metadata is invalid")
    if not isinstance(payload["optimizer_state"], SampledAdamState):
        raise RuntimeError("parent update-zero optimizer type mismatch")
    if payload["optimizer_state"].step != 0:
        raise RuntimeError("parent update-zero Adam counter is nonzero")
    checks = {
        "params_sha256": trial.get("initial_params_sha256"),
        "chain_states_sha256": trial.get("initial_post_burn_chain_states_sha256"),
        "sampling_key_sha256": trial.get("initial_post_burn_training_key_sha256"),
    }
    actual = {
        "params_sha256": pytree_sha256(payload["params"]),
        "chain_states_sha256": array_sha256(
            np.asarray(metadata.get("chain_states"), dtype=np.int32)
        ),
        "sampling_key_sha256": array_sha256(
            np.asarray(metadata.get("sampling_key"), dtype=np.uint32)
        ),
    }
    if actual != checks:
        raise RuntimeError("parent update-zero state does not match sealed trial lineage")
    return path, payload


def _audit_selector_and_parents(
    spec: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Audit selector plus all four selected parents without mutating them."""

    selector_binding = dict(spec["selector_binding"])
    selector_config_path = _resolve_path(selector_binding["config_path"])
    selector_result_path = _resolve_path(selector_binding["result_path"])
    if (
        not selector_config_path.is_file()
        or file_sha256(selector_config_path) != selector_binding["config_file_sha256"]
    ):
        raise RuntimeError("metric-correction selector config file seal failed")
    selector_config = json.loads(selector_config_path.read_text())
    if canonical_sha256(selector_config) != selector_binding["config_sha256"]:
        raise RuntimeError("metric-correction selector canonical config seal failed")
    if (
        not selector_result_path.is_file()
        or file_sha256(selector_result_path) != selector_binding["result_file_sha256"]
    ):
        raise RuntimeError("metric-correction selector result file seal failed")
    selector_result = json.loads(selector_result_path.read_text())
    if (
        selector_result.get("task") != SELECTOR_TASK
        or selector_result.get("status") != "complete"
        or selector_result.get("config_sha256") != selector_binding["config_sha256"]
        or selector_result.get("source_manifest_sha256")
        != selector_binding["source_manifest_sha256"]
        or selector_result.get("runtime_identity_sha256")
        != selector_binding["runtime_identity_sha256"]
    ):
        raise RuntimeError("metric-correction selector header binding failed")
    if selector_result.get("config") != selector_config:
        raise RuntimeError("selector embedded config differs from sealed config file")
    if (
        selector_result.get("result_payload_sha256")
        != selector_binding["result_payload_sha256"]
        or _result_payload_sha256(selector_result)
        != selector_binding["result_payload_sha256"]
    ):
        raise RuntimeError("metric-correction selector payload seal failed")
    parent_contract = selector_result.get("parent_binding_contract")
    if not isinstance(parent_contract, dict) or (
        parent_contract.get("training_authorized_by_this_selector") is not False
        or parent_contract.get("new_run_identity_and_checkpoint_series_required")
        is not True
    ):
        raise RuntimeError("selector parent-binding boundary changed")

    decisions = {
        str(item["case_id"]): item
        for item in selector_result.get("geometry_results", [])
    }
    audited: dict[str, dict[str, object]] = {}
    for requested in spec["selections"]:
        case_id = str(requested["case_id"])
        if case_id not in decisions:
            raise RuntimeError("selector decision is missing a requested geometry")
        geometry_result = decisions[case_id]
        decision = geometry_result.get("decision")
        if not isinstance(decision, dict):
            raise RuntimeError("selector geometry has no decision")
        if canonical_sha256(decision) != requested["selector_decision_sha256"]:
            raise RuntimeError("selector decision seal changed")
        expected = EXPECTED_SELECTIONS[case_id]
        if (
            decision.get("selected_candidate_id") != expected["candidate_id"]
            or float(decision.get("selected_learning_rate", np.nan))
            != expected["learning_rate"]
            or decision.get("action") != requested["selector_action"]
            or decision.get("training_authorized_by_secondary_selector") is not False
        ):
            raise RuntimeError("selector decision no longer matches extension selection")
        bindings = decision.get("selected_step50_parent_bindings_for_new_identity_only")
        if not isinstance(bindings, dict) or set(bindings) != set(EXPECTED_ARMS):
            raise RuntimeError("selector decision lacks both parent bindings")

        audited_arms: dict[str, object] = {}
        for backbone, arm_id in EXPECTED_ARMS.items():
            binding = dict(bindings[backbone])
            result_path = _resolve_path(binding["result_path"])
            if (
                not result_path.is_file()
                or file_sha256(result_path) != binding["result_file_sha256"]
            ):
                raise RuntimeError("selected parent result file seal failed")
            parent_result = json.loads(result_path.read_text())
            if (
                parent_result.get("status") != "complete"
                or parent_result.get("config_sha256") != binding["config_sha256"]
                or parent_result.get("source_manifest_sha256")
                != binding["source_manifest_sha256"]
                or parent_result.get("runtime_identity_sha256")
                != binding["runtime_identity_sha256"]
            ):
                raise RuntimeError("selected parent result header binding failed")
            parent_config = dict(parent_result["config"])
            scan_v2._validate_spec(parent_config)
            if canonical_sha256(parent_config) != binding["config_sha256"]:
                raise RuntimeError("selected parent config seal failed")
            parent_case, trial = _case_and_trial(parent_result, case_id, backbone)
            if (
                trial.get("status") != "complete"
                or trial.get("arm_id") != arm_id
                or trial.get("candidate_id") != expected["candidate_id"]
                or float(trial.get("learning_rate", np.nan))
                != expected["learning_rate"]
                or trial.get("trial_payload_sha256")
                != binding["trial_payload_sha256"]
                or trial_payload_sha256(trial) != binding["trial_payload_sha256"]
            ):
                raise RuntimeError("selected parent trial payload binding failed")
            if (
                int(parent_config["sampler"]["chains"]) != 32
                or int(parent_config["sampler"]["decorrelation_sweeps_per_update"])
                != 2
                or int(parent_config["optimizer"]["microbatch_size"]) != 8
                or float(parent_config["optimizer"]["learning_rate"])
                != expected["learning_rate"]
            ):
                raise RuntimeError("selected parent training choices are not locked")
            step50_path = _resolve_path(binding["checkpoint_generation_path"])
            step50 = _verify_checkpoint_payload(
                step50_path, binding, expected_step=50
            )
            parent_identity = step50["run_identity"]
            expected_identity_fields = {
                "task": scan_v2.TASK,
                "optimizer_backend": scan_v2.OPTIMIZER_BACKEND,
                "config_sha256": binding["config_sha256"],
                "source_manifest_sha256": binding["source_manifest_sha256"],
                "runtime_identity_sha256": binding["runtime_identity_sha256"],
                "candidate_id": expected["candidate_id"],
                "case_id": case_id,
                "arm_id": arm_id,
                "backbone": backbone,
            }
            if any(
                parent_identity.get(name) != value
                for name, value in expected_identity_fields.items()
            ):
                raise RuntimeError("selected parent checkpoint identity fields changed")
            if (
                step50["metadata"].get("amplitude_cache_present") is not False
                or "amplitudes" in step50["metadata"]
            ):
                raise RuntimeError("selected parent checkpoint retained amplitude cache")
            step0_path, step0 = _find_and_verify_step_zero(step50_path, step50, trial)
            arm_spec = next(
                item
                for item in parent_config["arms"]
                if item["backbone"] == backbone
            )
            case_spec = next(
                item
                for item in parent_config["cases"]
                if item["case_id"] == case_id
            )
            audited_arms[backbone] = {
                "binding": binding,
                "parent_result_path": result_path,
                "parent_result": parent_result,
                "parent_case_result": parent_case,
                "parent_trial": trial,
                "parent_config": parent_config,
                "case_spec": case_spec,
                "arm_spec": arm_spec,
                "step50_path": step50_path,
                "step50_payload": step50,
                "step0_path": step0_path,
                "step0_path_sha256": file_sha256(step0_path),
                "step0_payload": step0,
            }
        audited[case_id] = {
            "geometry": expected["geometry"],
            "candidate_id": expected["candidate_id"],
            "learning_rate": expected["learning_rate"],
            "selector_decision": decision,
            "selector_decision_sha256": canonical_sha256(decision),
            "arms": audited_arms,
        }
    return audited


def _compatibility_z(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or min(len(a), len(b)) < 2:
        raise ValueError("compatibility requires two one-dimensional sample vectors")
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    se_a = float(np.std(a, ddof=1) / np.sqrt(len(a)))
    se_b = float(np.std(b, ddof=1) / np.sqrt(len(b)))
    combined = math.hypot(se_a, se_b)
    finite = all(math.isfinite(x) for x in (mean_a, mean_b, se_a, se_b, combined))
    exactly_equal_zero_error = bool(finite and combined == 0.0 and mean_a == mean_b)
    valid = bool(finite and (combined > 0.0 or exactly_equal_zero_error))
    return {
        "first_mean": mean_a,
        "second_mean": mean_b,
        "first_mcse": se_a,
        "second_mcse": se_b,
        "combined_mcse": combined,
        "absolute_difference": abs(mean_a - mean_b),
        "absolute_difference_combined_mcse": (
            abs(mean_a - mean_b) / combined
            if finite and combined > 0.0
            else (0.0 if exactly_equal_zero_error else None)
        ),
        "valid_nonzero_combined_mcse": valid,
        "zero_mcse_exact_equal_special_case": exactly_equal_zero_error,
    }


def _series_mean_compatibility(
    first: np.ndarray, second: np.ndarray, *, n_sites: int
) -> dict[str, object]:
    """Compare two MCMC series with runtime autocorrelation-adjusted MCSEs."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    first_correlation = runtime_workflow._autocorrelation_summary(first_values)
    second_correlation = runtime_workflow._autocorrelation_summary(second_values)

    def summary(values: np.ndarray, correlation: Mapping[str, object]) -> tuple[float, float]:
        mean = float(np.mean(values) / n_sites)
        ess = float(correlation["ess"])
        mcse = (
            float(np.std(values, ddof=1) / np.sqrt(ess) / n_sites)
            if ess > 0.0
            else math.nan
        )
        return mean, mcse

    mean_a, se_a = summary(first_values, first_correlation)
    mean_b, se_b = summary(second_values, second_correlation)
    combined = math.hypot(se_a, se_b)
    finite = all(math.isfinite(x) for x in (mean_a, mean_b, se_a, se_b, combined))
    exactly_equal_zero_error = bool(finite and combined == 0.0 and mean_a == mean_b)
    valid = bool(finite and (combined > 0.0 or exactly_equal_zero_error))
    z = (
        abs(mean_a - mean_b) / combined
        if finite and combined > 0.0
        else (0.0 if exactly_equal_zero_error else None)
    )
    return {
        "first_mean": mean_a,
        "second_mean": mean_b,
        "first_mcse": se_a,
        "second_mcse": se_b,
        "combined_mcse": combined,
        "absolute_difference": abs(mean_a - mean_b),
        "absolute_difference_combined_mcse": z,
        "valid_nonzero_combined_mcse": valid,
        "zero_mcse_exact_equal_special_case": exactly_equal_zero_error,
        "first_autocorrelation": first_correlation,
        "second_autocorrelation": second_correlation,
        "mcse_contract": (
            "std_of_real_local_energy_samples_over_sqrt_runtime_initial_positive_"
            "window_ess_divided_by_N"
        ),
    }


def _weighted_sampler_summary(
    rows: list[dict[str, float]], attempts_per_chain: list[int], *, chains: int
) -> dict[str, object]:
    if not rows or len(rows) != len(attempts_per_chain):
        raise ValueError("sampler diagnostics and attempt weights must align")
    weights = np.asarray(attempts_per_chain, dtype=np.float64) * chains
    total = float(np.sum(weights))
    nontrivial = float(
        np.sum(weights * np.asarray([row["nontrivial_proposal_fraction"] for row in rows]))
    )
    accepted_move = float(
        np.sum(weights * np.asarray([row["accepted_move_fraction"] for row in rows]))
    )
    accepted_including_null = float(
        np.sum(
            weights
            * np.asarray([row["acceptance_fraction_including_null"] for row in rows])
        )
    )
    zero = np.asarray([row["zero_amplitude_chain_fraction"] for row in rows])
    return {
        "proposal_attempts": int(total),
        "nontrivial_proposal_count_proxy": nontrivial,
        "accepted_move_count_proxy": accepted_move,
        "accepted_including_null_count_proxy": accepted_including_null,
        "nontrivial_proposal_fraction_attempt_weighted": nontrivial / total,
        "accepted_move_fraction_attempt_weighted": accepted_move / total,
        "acceptance_fraction_including_null_attempt_weighted": (
            accepted_including_null / total
        ),
        "conditional_move_acceptance_attempt_weighted": (
            accepted_move / nontrivial if nontrivial > 0.0 else 0.0
        ),
        "zero_amplitude_chain_fraction_attempt_weighted_snapshot_mean": float(
            np.sum(weights * zero) / total
        ),
        "zero_amplitude_chain_fraction_maximum": float(np.max(zero)),
        "aggregation": "proposal_attempt_count_weighted_not_mean_of_ratios",
    }


def _correlation_gates(
    arrays: Mapping[str, np.ndarray], protocol: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, bool]]:
    summaries = {
        name: runtime_workflow._autocorrelation_summary(values)
        for name, values in arrays.items()
    }
    gates_spec = dict(protocol["fixed_measurement_mixing_gates"])
    gates: dict[str, bool] = {}
    for name, summary in summaries.items():
        finite = bool(
            summary["finite"] is True
            and summary["tau_int"] is not None
            and summary["split_rhat"] is not None
            and all(
                math.isfinite(float(summary[field]))
                for field in ("tau_int", "ess", "split_rhat")
            )
        )
        gates[f"{name}_finite"] = finite
        gates[f"{name}_split_rhat_below_1p01"] = bool(
            finite
            and float(summary["split_rhat"])
            < float(gates_spec["split_rhat_strict_maximum"])
        )
        gates[f"{name}_initial_positive_window_ess_at_least_1000"] = bool(
            finite
            and float(summary["ess"])
            >= float(gates_spec["runtime_initial_positive_window_ess_minimum"])
        )
        gates[f"{name}_production_at_least_50_tau"] = bool(
            finite
            and int(protocol["fixed_measurement"]["production_sweeps"])
            >= float(gates_spec["production_sweeps_minimum_tau_multiple"])
            * float(summary["tau_int"])
        )
        summary["estimator_contract"] = (
            "spin_sampled_runtime_initial_positive_window_ess_and_classical_split_rhat"
        )
        summary["rank_normalized_bulk_ess"] = None
    return summaries, gates


def _fixed_confirmatory_measurement(
    *,
    runtime: runtime_workflow._ArmRuntime,
    params: object,
    lineage_states: np.ndarray,
    case: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    protocol: dict[str, object],
    measurement_seed: int,
    campaign_id: str,
    update: int,
    advance_uncached: Callable,
    advance_cached: Callable,
    hpsi_fn: Callable,
    microbatch_size: int,
    proposals_per_sweep: int,
    bond_proposal_probability: float,
) -> tuple[dict[str, object], dict[str, int]]:
    """Run one long 32-lineage + 32-fresh fixed-parameter block."""

    measurement = dict(protocol["fixed_measurement"])
    lineage = np.asarray(lineage_states, dtype=np.int32)
    lineage_count = int(measurement["parent_lineage_chains"])
    fresh_count = int(measurement["independent_fresh_m0_chains"])
    chains = int(measurement["chains"])
    rows, columns = int(case["rows"]), int(case["columns"])
    n_sites = rows * columns
    if lineage.shape != (lineage_count, n_sites) or lineage_count + fresh_count != chains:
        raise ValueError("fixed measurement lineage chain shape is wrong")
    params_hash = pytree_sha256(params)
    lineage_hash = array_sha256(lineage)

    common_fresh_fold = int(
        canonical_sha256(
            {
                "purpose": "confirmatory_fresh_m0_common_across_updates",
                "campaign_id": campaign_id,
                "measurement_seed": measurement_seed,
                "case_id": case["case_id"],
                "arm_id": runtime.arm_id,
            }
        )[:8],
        16,
    )
    fresh_key = jax.random.fold_in(
        jax.random.PRNGKey(measurement_seed), common_fresh_fold
    )
    fresh = np.asarray(
        initialize_spin_exchange_chains(
            fresh_key,
            n_chains=fresh_count,
            n_sites=n_sites,
            n_up=n_sites // 2,
        ),
        dtype=np.int32,
    )
    states = np.concatenate((lineage, fresh), axis=0)
    initial_states = np.array(states, copy=True)
    group_labels = ["parent_lineage"] * lineage_count + ["fresh_random_m0"] * fresh_count
    measurement_fold = int(
        canonical_sha256(
            {
                "purpose": "confirmatory_long_fixed_parameter_measurement",
                "campaign_id": campaign_id,
                "measurement_seed": measurement_seed,
                "case_id": case["case_id"],
                "arm_id": runtime.arm_id,
                "update": int(update),
            }
        )[:8],
        16,
    )
    key = jax.random.fold_in(jax.random.PRNGKey(measurement_seed), measurement_fold)
    amplitudes = None
    burn_snapshot_changed_ever = np.zeros(chains, dtype=bool)
    production_snapshot_changed_ever = np.zeros(chains, dtype=bool)
    sampler_rows: list[dict[str, float]] = []
    attempt_weights: list[int] = []
    burn_sampler_rows: list[dict[str, float]] = []
    production_sampler_rows: list[dict[str, float]] = []
    initial_amplitudes = jax.vmap(
        lambda state: runtime.projected_amplitude(params, state)
    )(jnp.asarray(states))
    jax.block_until_ready(initial_amplitudes)
    initial_amplitude_host = np.asarray(initial_amplitudes, dtype=np.complex128)
    if np.any(~np.isfinite(initial_amplitude_host)):
        raise FloatingPointError("long measurement initial amplitudes are nonfinite")
    initial_zero_amplitude_fraction = float(
        np.mean(np.abs(initial_amplitude_host) == 0.0)
    )

    def one_sweep(*, phase: str) -> None:
        nonlocal states, amplitudes, key
        before = np.array(states, copy=True)
        if amplitudes is None:
            output = advance_uncached(
                params,
                key,
                jnp.asarray(states),
                proposals_per_sweep,
                bond_proposal_probability,
            )
        else:
            output = advance_cached(
                params,
                key,
                jnp.asarray(states),
                amplitudes,
                proposals_per_sweep,
                bond_proposal_probability,
            )
        states_device, amplitudes_device, key_device, row_device = output
        jax.block_until_ready(output)
        states = np.asarray(states_device, dtype=np.int32)
        amplitudes = jnp.asarray(amplitudes_device)
        key = jnp.asarray(key_device)
        changed = np.any(states != before, axis=1)
        if phase == "burn":
            burn_snapshot_changed_ever[:] |= changed
        elif phase == "production":
            production_snapshot_changed_ever[:] |= changed
        else:  # pragma: no cover - internal call sites are literal
            raise ValueError("unknown fixed-measurement phase")
        row = v1_helpers._float_diagnostics(row_device)
        sampler_rows.append(row)
        (burn_sampler_rows if phase == "burn" else production_sampler_rows).append(row)
        attempt_weights.append(proposals_per_sweep)

    for _ in range(int(measurement["burn_sweeps"])):
        one_sweep(phase="burn")

    production = int(measurement["production_sweeps"])
    energy_rows: list[np.ndarray] = []
    slow_rows: dict[str, list[np.ndarray]] = {}
    actual_connected = 0
    minimum_center_magnitude = math.inf
    maximum_local_energy_magnitude = 0.0
    padded_width = len(bonds) + 1
    for _ in range(production):
        one_sweep(phase="production")
        host_states = np.asarray(states, dtype=np.int32)
        batches, widths, _ = v1_helpers._hpsi_batches(
            host_states,
            bonds,
            padded_width=padded_width,
            microbatch_size=microbatch_size,
        )
        actual_connected += int(np.sum(widths))
        chunks: list[np.ndarray] = []
        for batch in batches:
            centers, actions = hpsi_fn(params, batch)
            jax.block_until_ready((centers, actions))
            center_array = np.asarray(centers, dtype=np.complex128)
            if np.any(np.abs(center_array) == 0.0):
                raise FloatingPointError("long measurement encountered zero center")
            minimum_center_magnitude = min(
                minimum_center_magnitude, float(np.min(np.abs(center_array)))
            )
            local_chunk = np.asarray(actions, dtype=np.complex128) / center_array
            maximum_local_energy_magnitude = max(
                maximum_local_energy_magnitude,
                float(np.max(np.abs(local_chunk))),
            )
            chunks.append(local_chunk)
        energy_rows.append(np.concatenate(chunks))
        slow = runtime_workflow._ordering_structure_factor_series(
            host_states,
            geometry=str(case["geometry"]),
            rows=rows,
            columns=columns,
        )
        for name, values in slow.items():
            slow_rows.setdefault(name, []).append(np.asarray(values, dtype=np.float64))

    if params_hash != pytree_sha256(params) or lineage_hash != array_sha256(lineage):
        raise RuntimeError("fixed confirmatory measurement mutated frozen inputs")
    local = np.stack(energy_rows, axis=0).astype(np.complex128)
    if (
        local.shape != (production, chains)
        or list(local.shape) != list(measurement["raw_local_energy_shape"])
        or np.any(~np.isfinite(local))
    ):
        raise FloatingPointError("long measurement local-energy array is invalid")
    slow_arrays = {
        name: np.stack(values, axis=0).astype(np.float64)
        for name, values in slow_rows.items()
    }
    expected_slow = set(measurement["diagonal_slow_modes"][str(case["geometry"])])
    if set(slow_arrays) != expected_slow or any(
        values.shape != (production, chains) for values in slow_arrays.values()
    ):
        raise RuntimeError("long measurement slow-mode contract mismatch")

    real = np.asarray(local.real, dtype=np.float64)
    imaginary = np.asarray(local.imag, dtype=np.float64)
    mean = complex(np.mean(local))
    variance = float(np.mean(np.abs(local - mean) ** 2).real)
    chain_energy = np.asarray(np.mean(real, axis=0) / n_sites, dtype=np.float64)
    chain_variance = np.asarray(
        np.mean(np.abs(local - mean) ** 2, axis=0) / n_sites,
        dtype=np.float64,
    )
    correlation, correlation_gates = _correlation_gates(
        {"real_local_energy": real, **slow_arrays}, protocol
    )
    group_compatibility = {
        "real_local_energy": _series_mean_compatibility(
            real[:, :lineage_count], real[:, lineage_count:], n_sites=n_sites
        ),
        **{
            name: _series_mean_compatibility(
                values[:, :lineage_count], values[:, lineage_count:], n_sites=1
            )
            for name, values in slow_arrays.items()
        },
    }
    half = production // 2
    half_compatibility = _series_mean_compatibility(
        real[:half], real[half:], n_sites=n_sites
    )
    sampler_summary = _weighted_sampler_summary(
        sampler_rows, attempt_weights, chains=chains
    )
    burn_sampler_summary = _weighted_sampler_summary(
        burn_sampler_rows,
        [proposals_per_sweep] * len(burn_sampler_rows),
        chains=chains,
    )
    production_sampler_summary = _weighted_sampler_summary(
        production_sampler_rows,
        [proposals_per_sweep] * len(production_sampler_rows),
        chains=chains,
    )
    sampler_summary["zero_amplitude_chain_fraction_initial"] = (
        initial_zero_amplitude_fraction
    )
    sampler_summary["zero_amplitude_chain_fraction_maximum"] = max(
        float(sampler_summary["zero_amplitude_chain_fraction_maximum"]),
        initial_zero_amplitude_fraction,
    )
    gates_spec = dict(protocol["fixed_measurement_mixing_gates"])
    frozen_count = int(np.count_nonzero(~production_snapshot_changed_ever))
    sampler_finite = all(
        math.isfinite(float(value))
        for row in sampler_rows
        for value in row.values()
    ) and all(
        math.isfinite(float(value))
        for name, value in sampler_summary.items()
        if name not in {"aggregation"}
    )
    mixing_gates = {
        **correlation_gates,
        "sampler_diagnostics_finite": sampler_finite,
        "no_zero_amplitude_chains": bool(
            sampler_summary["zero_amplitude_chain_fraction_maximum"]
            <= float(gates_spec["zero_amplitude_chain_fraction_maximum"])
        ),
        "no_frozen_chains": frozen_count
        <= int(gates_spec["frozen_chain_count_maximum"]),
        **{
            f"checkpoint_and_fresh_{name}_means_compatible": bool(
                comparison["valid_nonzero_combined_mcse"]
                and float(comparison["absolute_difference_combined_mcse"])
                <= float(
                    gates_spec[
                        "checkpoint_vs_fresh_each_observable_maximum_combined_mcse"
                    ]
                )
            )
            for name, comparison in group_compatibility.items()
        },
        "energy_first_and_second_half_compatible": bool(
            half_compatibility["valid_nonzero_combined_mcse"]
            and float(half_compatibility["absolute_difference_combined_mcse"])
            <= float(gates_spec["energy_first_vs_second_half_maximum_combined_mcse"])
        ),
    }
    result: dict[str, object] = {
        "update": int(update),
        "fixed_params_sha256": params_hash,
        "parent_lineage_start_states_sha256": lineage_hash,
        "fresh_start_states_sha256": array_sha256(fresh),
        "combined_start_states_sha256": array_sha256(initial_states),
        "final_measurement_states_sha256": array_sha256(states),
        "group_labels": group_labels,
        "group_label_counts": {
            "parent_lineage": lineage_count,
            "fresh_random_m0": fresh_count,
        },
        "group_index_contract": (
            f"indices_0_{lineage_count - 1}_parent_training_lineage_indices_0_"
            f"{lineage_count - 1};indices_{lineage_count}_{chains - 1}_common_"
            f"fresh_m0_indices_0_{fresh_count - 1}"
        ),
        "common_fresh_start_fold": common_fresh_fold,
        "measurement_random_stream_fold": measurement_fold,
        "measurement_random_stream_identity_sha256": canonical_sha256(
            {
                "measurement_seed": measurement_seed,
                "fold": measurement_fold,
                "case_id": case["case_id"],
                "arm_id": runtime.arm_id,
                "update": int(update),
            }
        ),
        "burn_sweeps": int(measurement["burn_sweeps"]),
        "production_sweeps": production,
        "chain_count": chains,
        "changed_ever": production_snapshot_changed_ever.tolist(),
        "changed_ever_sha256": array_sha256(production_snapshot_changed_ever),
        "changed_ever_contract": (
            "changed_between_at_least_one_adjacent production lattice-sweep snapshot; "
            "within-sweep move-and-return is not observable"
        ),
        "burn_snapshot_changed_ever": burn_snapshot_changed_ever.tolist(),
        "burn_snapshot_changed_ever_sha256": array_sha256(
            burn_snapshot_changed_ever
        ),
        "frozen_chain_count": frozen_count,
        "frozen_chain_indices": np.flatnonzero(
            ~production_snapshot_changed_ever
        ).astype(int).tolist(),
        "sampler_diagnostics_attempt_weighted": sampler_summary,
        "burn_sampler_diagnostics_attempt_weighted": burn_sampler_summary,
        "production_sampler_diagnostics_attempt_weighted": (
            production_sampler_summary
        ),
        "real_local_energy_raw": real.tolist(),
        "real_local_energy_raw_sha256": array_sha256(real),
        "imaginary_local_energy_raw": imaginary.tolist(),
        "imaginary_local_energy_raw_sha256": array_sha256(imaginary),
        "raw_local_energy_shape": list(real.shape),
        "energy_per_site": float(mean.real / n_sites),
        "energy_imaginary_abs_per_site": float(abs(mean.imag) / n_sites),
        "minimum_center_magnitude": minimum_center_magnitude,
        "maximum_local_energy_magnitude": maximum_local_energy_magnitude,
        "energy_per_site_chain_means": chain_energy.tolist(),
        "energy_per_site_chain_means_sha256": array_sha256(chain_energy),
        "energy_per_site_chain_mcse": float(
            np.std(chain_energy, ddof=1) / np.sqrt(chains)
        ),
        "variance_h_per_site": variance / n_sites,
        "variance_h_per_site_chain_contributions": chain_variance.tolist(),
        "variance_h_per_site_chain_contributions_sha256": array_sha256(
            chain_variance
        ),
        "variance_h_per_site_chain_mcse": float(
            np.std(chain_variance, ddof=1) / np.sqrt(chains)
        ),
        "diagonal_slow_mode_raw": {
            name: {
                "values": values.tolist(),
                "sha256": array_sha256(values),
                "shape": list(values.shape),
            }
            for name, values in slow_arrays.items()
        },
        "autocorrelation": correlation,
        "checkpoint_start_vs_fresh_start_compatibility": group_compatibility,
        "energy_first_vs_second_half_compatibility": half_compatibility,
        "mixing_gates": mixing_gates,
        "all_mixing_gates_passed": all(mixing_gates.values()),
        "params_frozen_throughout_block": True,
        "training_chains_copied_and_not_mutated": True,
        "full_spin_structure_factor_computed": False,
    }
    sampling_evaluations = chains * (
        1 + proposals_per_sweep * (int(measurement["burn_sweeps"]) + production)
    )
    hpsi_evaluations = production * chains * (1 + padded_width)
    ledger = {
        "fixed_parameter_energy_samples": production * chains,
        "measurement_sampler_proposal_attempts": chains
        * proposals_per_sweep
        * (int(measurement["burn_sweeps"]) + production),
        "measurement_sampler_projected_amplitude_evaluations": sampling_evaluations,
        "measurement_hpsi_forward_projected_amplitude_evaluations": hpsi_evaluations,
        "measurement_hpsi_actual_connected_configurations": actual_connected,
        "measurement_hpsi_padded_connected_configurations": production
        * chains
        * padded_width,
        "measurement_base_tree_orbit_image_work_proxy": (
            (sampling_evaluations + hpsi_evaluations)
            * runtime.projector.group_order
        ),
    }
    return result, ledger


def _training_stability(
    history: list[dict[str, object]], protocol: Mapping[str, object]
) -> dict[str, object]:
    policy = dict(protocol["training_stability_gates"])
    conditional = np.asarray(
        [float(row["sampler"]["conditional_move_acceptance"]) for row in history],
        dtype=np.float64,
    )
    relative = np.asarray(
        [float(row["adam"]["relative_update_norm"]) for row in history],
        dtype=np.float64,
    )
    last = conditional[-10:]
    median = float(np.median(last)) if len(last) else None
    quantile = (
        float(np.quantile(relative, float(policy["relative_update_quantile"])))
        if len(relative)
        else None
    )
    gates = {
        "last10_conditional_move_acceptance_median_at_least_0p15": bool(
            median is not None
            and median
            >= float(policy["last10_conditional_move_acceptance_median_minimum"])
        ),
        "relative_update_q99_at_most_0p02": bool(
            quantile is not None
            and quantile <= float(policy["relative_update_quantile_maximum"])
        ),
    }
    return {
        "last_updates_window": min(10, len(history)),
        "last10_conditional_move_acceptance_median": median,
        "relative_update_quantile": float(policy["relative_update_quantile"]),
        "relative_update_quantile_value": quantile,
        "gates": gates,
        "all_training_stability_gates_passed": all(gates.values()),
    }


def _paired_final_optimization_gates(
    baseline: Mapping[str, object], final: Mapping[str, object], protocol: Mapping[str, object]
) -> dict[str, object]:
    policy = dict(protocol["final_optimization_gates_each_arm"])
    energy0 = np.asarray(baseline["energy_per_site_chain_means"], dtype=np.float64)
    energy1 = np.asarray(final["energy_per_site_chain_means"], dtype=np.float64)
    variance0 = np.asarray(
        baseline["variance_h_per_site_chain_contributions"], dtype=np.float64
    )
    variance1 = np.asarray(
        final["variance_h_per_site_chain_contributions"], dtype=np.float64
    )
    if not (energy0.shape == energy1.shape == variance0.shape == variance1.shape == (64,)):
        raise RuntimeError("confirmatory endpoint paired vectors have wrong shape")
    energy_diff = energy0 - energy1
    variance_diff = variance0 - variance1
    energy_lowering = float(np.mean(energy_diff))
    energy_se = float(np.std(energy_diff, ddof=1) / np.sqrt(len(energy_diff)))
    variance_decrease = float(np.mean(variance_diff))
    variance_se = float(np.std(variance_diff, ddof=1) / np.sqrt(len(variance_diff)))
    variance_start = float(np.mean(variance0))
    valid = (
        all(
            math.isfinite(value)
            for value in (
                energy_lowering,
                energy_se,
                variance_decrease,
                variance_se,
                variance_start,
            )
        )
        and energy_se > 0.0
        and variance_se > 0.0
        and variance_start > 0.0
    )
    energy_z = energy_lowering / energy_se if valid else None
    variance_z = variance_decrease / variance_se if valid else None
    fractional = variance_decrease / variance_start if valid else None
    gates = {
        "energy_lowering_per_site_at_least_0p05": bool(
            valid
            and energy_lowering >= float(policy["minimum_energy_lowering_per_site"])
        ),
        "energy_lowering_paired_z_at_least_5": bool(
            valid
            and float(energy_z) >= float(policy["minimum_energy_lowering_paired_mcse"])
        ),
        "variance_fractional_decrease_at_least_0p25": bool(
            valid
            and float(fractional)
            >= float(policy["minimum_variance_fractional_decrease"])
        ),
        "variance_decrease_paired_z_at_least_3": bool(
            valid
            and float(variance_z)
            >= float(policy["minimum_variance_decrease_paired_mcse"])
        ),
    }
    return {
        "endpoint_updates": [0, 100],
        "pairing_contract": (
            "same_group_and_chain_index; parent lineage indices and common fresh-M0 "
            "indices retained separately"
        ),
        "energy_lowering_per_site": energy_lowering,
        "energy_lowering_paired_chain_se": energy_se,
        "energy_lowering_paired_z": energy_z,
        "energy_difference_sha256": array_sha256(energy_diff),
        "variance_start_per_site": variance_start,
        "variance_final_per_site": float(np.mean(variance1)),
        "variance_decrease_per_site": variance_decrease,
        "variance_fractional_decrease": fractional,
        "variance_decrease_paired_chain_se": variance_se,
        "variance_decrease_paired_z": variance_z,
        "variance_difference_sha256": array_sha256(variance_diff),
        "valid_nonzero_paired_mcse": valid,
        "gates": gates,
        "all_final_optimization_gates_passed": all(gates.values()),
        "evaluated_at_final_update_only": True,
    }


def _route_decision(trials: list[Mapping[str, object]]) -> dict[str, object]:
    """Apply the registered four-arm sampler -> SR -> heldout hierarchy."""

    mixing_pass = bool(
        len(trials) == 4
        and all(
            trial.get("status") == "complete"
            and trial.get("all_mixing_and_training_stability_gates_passed") is True
            for trial in trials
        )
    )
    optimization_pass = bool(
        mixing_pass
        and all(
            trial.get("all_final_optimization_gates_passed") is True
            for trial in trials
        )
    )
    if not mixing_pass:
        route = "sampler_audit"
    elif not optimization_pass:
        route = "matrix_free_SR"
    else:
        route = "heldout"
    return {
        "all_four_mixing_and_training_stability_pass": mixing_pass,
        "all_four_final_optimization_gates_pass": optimization_pass,
        "next_route": route,
        "full_spin_structure_factor_computed": False,
    }


def _finalize_extension_payload(payload: dict[str, object]) -> dict[str, object]:
    """Seal a terminal four-arm payload after all trial seals are present."""

    trials = [
        trial
        for case in payload.get("cases", [])
        for trial in case.get("trials", [])
    ]
    if len(trials) != 4 or any(
        trial.get("status") not in {"complete", "hard_failed"} for trial in trials
    ):
        raise RuntimeError("extension finalization requires four terminal trials")
    payload["route_decision"] = _route_decision(trials)
    payload["trial_manifest"] = embedded_trial_manifest(payload)
    if len(payload["trial_manifest"]) != 4:
        raise RuntimeError("extension trial manifest must contain four unique arms")
    payload["trial_manifest_sha256"] = canonical_sha256(payload["trial_manifest"])
    payload["status"] = "complete"
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["result_payload_sha256"] = _result_payload_sha256(payload)
    return payload


def _extension_identity(
    *,
    config_hash: str,
    source_hash: str,
    runtime_hash: str,
    campaign_id: str,
    case_id: str,
    backbone: str,
    arm_id: str,
    selector_decision_sha256: str,
    parent: Mapping[str, object],
) -> dict[str, object]:
    binding = dict(parent["binding"])
    return {
        "schema": "physical_sampled_optimizer_confirmatory_extension_v1",
        "task": TASK,
        "optimizer_backend": OPTIMIZER_BACKEND,
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "runtime_identity_sha256": runtime_hash,
        "campaign_id": campaign_id,
        "case_id": case_id,
        "arm_id": arm_id,
        "backbone": backbone,
        "selector_decision_sha256": selector_decision_sha256,
        "parent_result_file_sha256": binding["result_file_sha256"],
        "parent_config_sha256": binding["config_sha256"],
        "parent_source_manifest_sha256": binding["source_manifest_sha256"],
        "parent_runtime_identity_sha256": binding["runtime_identity_sha256"],
        "parent_trial_payload_sha256": binding["trial_payload_sha256"],
        "parent_step50_generation_sha256": binding["checkpoint_generation_sha256"],
        "parent_step50_run_identity_sha256": binding[
            "checkpoint_run_identity_sha256"
        ],
        "parent_step50_params_sha256": binding["params_sha256"],
        "parent_step50_optimizer_state_sha256": binding[
            "optimizer_state_sha256"
        ],
        "parent_step50_chain_states_sha256": binding["chain_states_sha256"],
        "parent_step50_sampling_key_sha256": binding["sampling_key_sha256"],
        "parent_step0_generation_sha256": parent["step0_path_sha256"],
    }


def _extension_metadata(
    *,
    params: object,
    optimizer_state: SampledAdamState,
    states: np.ndarray,
    sampling_key: np.ndarray,
    parent_binding: Mapping[str, object],
    step0_binding: Mapping[str, object],
    measurements: list[dict[str, object]],
    resume_count: int,
) -> dict[str, object]:
    return {
        "optimizer_backend": OPTIMIZER_BACKEND,
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(optimizer_state),
        "chain_states": np.asarray(states, dtype=np.int32),
        "chain_states_sha256": array_sha256(np.asarray(states, dtype=np.int32)),
        "sampling_key": np.asarray(sampling_key, dtype=np.uint32),
        "sampling_key_sha256": array_sha256(np.asarray(sampling_key, dtype=np.uint32)),
        "parent_step50_binding": dict(parent_binding),
        "parent_step0_binding": dict(step0_binding),
        "fixed_parameter_measurements": measurements,
        "resume_count": int(resume_count),
        "amplitude_cache_present": False,
        "amplitude_cache_policy": "never checkpointed; recompute after every update",
    }


def _load_extension_checkpoint(
    directory: Path,
    identity: dict[str, object],
    *,
    n_sites: int,
) -> dict[str, object] | None:
    if not directory.is_dir():
        return None
    payload = load_latest_resumable_checkpoint(directory, expected_run_identity=identity)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("optimizer_backend") != OPTIMIZER_BACKEND:
        raise RuntimeError("refusing checkpoint from another optimizer workflow")
    if metadata.get("amplitude_cache_present") is not False or "amplitudes" in metadata:
        raise RuntimeError("extension checkpoint must not contain an amplitude cache")
    states = np.asarray(metadata.get("chain_states"), dtype=np.int32)
    if states.shape != (32, n_sites) or np.any(np.sum(states, axis=1) != n_sites // 2):
        raise RuntimeError("extension checkpoint training-chain support is invalid")
    if metadata.get("chain_states_sha256") != array_sha256(states):
        raise RuntimeError("extension checkpoint chain seal failed")
    key = np.asarray(metadata.get("sampling_key"), dtype=np.uint32)
    if metadata.get("sampling_key_sha256") != array_sha256(key):
        raise RuntimeError("extension checkpoint key seal failed")
    if metadata.get("params_sha256") != pytree_sha256(payload["params"]):
        raise RuntimeError("extension checkpoint parameter seal failed")
    state = payload["optimizer_state"]
    if not isinstance(state, SampledAdamState) or state.step != int(payload["completed_steps"]):
        raise RuntimeError("extension checkpoint Adam counter mismatch")
    if metadata.get("optimizer_state_sha256") != pytree_sha256(state):
        raise RuntimeError("extension checkpoint Adam seal failed")
    if int(payload["next_sample_index"]) != int(payload["completed_steps"]) * 32:
        raise RuntimeError("extension checkpoint sample counter mismatch")
    expected_measurements = {0}
    if int(payload["completed_steps"]) >= 75:
        expected_measurements.add(75)
    if int(payload["completed_steps"]) >= 100:
        expected_measurements.add(100)
    actual_measurements = {
        int(item["update"]) for item in metadata.get("fixed_parameter_measurements", [])
    }
    if actual_measurements != expected_measurements:
        raise RuntimeError("extension checkpoint fixed measurements are incomplete")
    return payload


def _run_extension_arm(
    *,
    runtime: runtime_workflow._ArmRuntime,
    parent: dict[str, object],
    case: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    spec: dict[str, object],
    selector_decision_sha256: str,
    checkpoint_directory: Path,
    config_hash: str,
    source_hash: str,
    runtime_hash: str,
    overwrite: bool,
    stop_after_update: int | None = None,
) -> dict[str, object]:
    protocol = dict(spec["protocol"])
    parent_config = dict(parent["parent_config"])
    sampler = dict(parent_config["sampler"])
    optimizer = dict(parent_config["optimizer"])
    n_sites = int(case["rows"]) * int(case["columns"])
    chains = int(protocol["training_chains"])
    proposals = int(sampler["proposals_per_sweep"])
    probability = float(sampler["bond_proposal_probability"])
    microbatch = int(protocol["microbatch_size"])
    padded_width = len(bonds) + 1
    advance_uncached, advance_cached, hpsi_fn, center_fn = scan_v2._compiled_kernels(
        runtime, bonds
    )
    identity = _extension_identity(
        config_hash=config_hash,
        source_hash=source_hash,
        runtime_hash=runtime_hash,
        campaign_id=str(spec["campaign_id"]),
        case_id=str(case["case_id"]),
        backbone=runtime.backbone,
        arm_id=runtime.arm_id,
        selector_decision_sha256=selector_decision_sha256,
        parent=parent,
    )
    if overwrite and checkpoint_directory.is_dir():
        shutil.rmtree(checkpoint_directory)
    checkpoint = None if overwrite else _load_extension_checkpoint(
        checkpoint_directory, identity, n_sites=n_sites
    )
    step50 = dict(parent["step50_payload"])
    step0 = dict(parent["step0_payload"])
    parent_binding = dict(parent["binding"])
    step0_binding = {
        "checkpoint_generation_path": _portable_path(parent["step0_path"]),
        "checkpoint_generation_sha256": parent["step0_path_sha256"],
        "checkpoint_run_identity_sha256": canonical_json_sha256(
            step0["run_identity"]
        ),
        "params_sha256": pytree_sha256(step0["params"]),
        "optimizer_state_sha256": pytree_sha256(step0["optimizer_state"]),
        "chain_states_sha256": step0["metadata"]["chain_states_sha256"],
        "sampling_key_sha256": step0["metadata"]["sampling_key_sha256"],
    }

    if checkpoint is None:
        params = step50["params"]
        optimizer_state = step50["optimizer_state"]
        states = np.asarray(step50["metadata"]["chain_states"], dtype=np.int32)
        sampling_key = np.asarray(step50["metadata"]["sampling_key"], dtype=np.uint32)
        if (
            pytree_sha256(params) != parent_binding["params_sha256"]
            or pytree_sha256(optimizer_state)
            != parent_binding["optimizer_state_sha256"]
            or array_sha256(states) != parent_binding["chain_states_sha256"]
            or array_sha256(sampling_key) != parent_binding["sampling_key_sha256"]
        ):
            raise RuntimeError("parent step-50 state changed before extension import")
        baseline, baseline_work = _fixed_confirmatory_measurement(
            runtime=runtime,
            params=step0["params"],
            lineage_states=np.asarray(step0["metadata"]["chain_states"], dtype=np.int32),
            case=case,
            bonds=bonds,
            protocol=protocol,
            measurement_seed=int(spec["measurement_seed"]),
            campaign_id=str(spec["campaign_id"]),
            update=0,
            advance_uncached=advance_uncached,
            advance_cached=advance_cached,
            hpsi_fn=hpsi_fn,
            microbatch_size=microbatch,
            proposals_per_sweep=proposals,
            bond_proposal_probability=probability,
        )
        baseline["work_ledger"] = baseline_work
        measurements = [baseline]
        parent_history = list(step50["history"])
        if len(parent_history) != 50:
            raise RuntimeError("parent step-50 checkpoint history is incomplete")
        checkpoint_history = parent_history
        history: list[dict[str, object]] = []
        resume_count = 0
        completed = 50
        metadata = _extension_metadata(
            params=params,
            optimizer_state=optimizer_state,
            states=states,
            sampling_key=sampling_key,
            parent_binding=parent_binding,
            step0_binding=step0_binding,
            measurements=measurements,
            resume_count=resume_count,
        )
        save_resumable_checkpoint_generation(
            checkpoint_directory,
            params,
            optimizer_state=optimizer_state,
            completed_steps=50,
            next_sample_index=50 * chains,
            run_identity=identity,
            history=checkpoint_history,
            metadata=metadata,
        )
    else:
        params = checkpoint["params"]
        optimizer_state = checkpoint["optimizer_state"]
        completed = int(checkpoint["completed_steps"])
        checkpoint_history = list(checkpoint["history"])
        if len(checkpoint_history) != completed or completed < 50:
            raise RuntimeError("extension checkpoint history/counter mismatch")
        history = checkpoint_history[50:]
        metadata = dict(checkpoint["metadata"])
        states = np.asarray(metadata["chain_states"], dtype=np.int32)
        sampling_key = np.asarray(metadata["sampling_key"], dtype=np.uint32)
        measurements = list(metadata["fixed_parameter_measurements"])
        resume_count = int(metadata["resume_count"]) + 1
        if metadata.get("parent_step50_binding") != parent_binding:
            raise RuntimeError("extension checkpoint parent binding changed")
        if metadata.get("parent_step0_binding") != step0_binding:
            raise RuntimeError("extension checkpoint baseline binding changed")

    final_update = int(protocol["final_update"])
    invocation_end = final_update if stop_after_update is None else int(stop_after_update)
    if invocation_end not in {75, 100} or invocation_end < completed:
        raise ValueError("extension invocation may stop only at checkpoint 75 or 100")
    hard_failure: dict[str, object] | None = None
    for update in range(completed + 1, invocation_end + 1):
        params_before = params
        optimizer_before = optimizer_state
        states_before = np.array(states, copy=True)
        key_before = np.array(sampling_key, copy=True)
        attempts = int(protocol["decorrelation_sweeps_per_update"]) * proposals
        sampled = advance_uncached(
            params,
            jnp.asarray(sampling_key),
            jnp.asarray(states),
            attempts,
            probability,
        )
        jax.block_until_ready(sampled)
        states = np.asarray(sampled[0], dtype=np.int32)
        sampling_key = np.asarray(sampled[2], dtype=np.uint32)
        sampler_diagnostics = v1_helpers._float_diagnostics(sampled[3])
        moved = float(np.mean(np.any(states != states_before, axis=1)))
        del sampled
        if moved <= 0.0:
            hard_failure = {
                "category": "frozen_training_chain",
                "attempted_update": update,
            }
            states, sampling_key = states_before, key_before
            break
        batches, widths, _ = v1_helpers._hpsi_batches(
            states, bonds, padded_width=padded_width, microbatch_size=microbatch
        )
        try:
            gradient = sampled_vmc_value_and_grad(
                params, batches, hpsi_fn=hpsi_fn, center_fn=center_fn
            )
            params, optimizer_state, adam = sampled_adam_update(
                params,
                gradient.gradient,
                optimizer_state,
                learning_rate=float(optimizer["learning_rate"]),
                max_relative_update=float(optimizer["maximum_relative_update_norm"]),
                beta1=float(optimizer["beta1"]),
                beta2=float(optimizer["beta2"]),
                epsilon=float(optimizer["epsilon"]),
            )
            jax.block_until_ready(params)
        except FloatingPointError as error:
            hard_failure = {
                "category": "nonfinite_or_optimizer_limit",
                "attempted_update": update,
                "message": str(error),
            }
            params, optimizer_state = params_before, optimizer_before
            states, sampling_key = states_before, key_before
            break
        fixed = None
        fixed_work: dict[str, int] = {}
        if update in {75, 100}:
            try:
                fixed, fixed_work = _fixed_confirmatory_measurement(
                    runtime=runtime,
                    params=params,
                    lineage_states=states,
                    case=case,
                    bonds=bonds,
                    protocol=protocol,
                    measurement_seed=int(spec["measurement_seed"]),
                    campaign_id=str(spec["campaign_id"]),
                    update=update,
                    advance_uncached=advance_uncached,
                    advance_cached=advance_cached,
                    hpsi_fn=hpsi_fn,
                    microbatch_size=microbatch,
                    proposals_per_sweep=proposals,
                    bond_proposal_probability=probability,
                )
            except FloatingPointError as error:
                hard_failure = {
                    "category": "nonfinite_fixed_measurement",
                    "attempted_update": update,
                    "message": str(error),
                }
                params, optimizer_state = params_before, optimizer_before
                states, sampling_key = states_before, key_before
                break
            fixed["work_ledger"] = fixed_work
            measurements.append(fixed)
        sampler_projected = chains * (1 + attempts)
        hpsi_projected = chains * (1 + padded_width)
        center_projected = chains
        row = {
            "update": update,
            "params_before_sha256": pytree_sha256(params_before),
            "params_after_sha256": pytree_sha256(params),
            "chain_states_sha256": array_sha256(states),
            "sampling_key_sha256": array_sha256(sampling_key),
            "sampler": {
                **sampler_diagnostics,
                "chain_snapshot_move_fraction": moved,
            },
            "pre_update_sampled_energy_per_site": gradient.energy / n_sites,
            "pre_update_sampled_variance_h_per_site": gradient.variance / n_sites,
            "gradient": {
                "norm": gradient.gradient_norm,
                "sample_count": gradient.sample_count,
                "microbatch_count": gradient.microbatch_count,
                "connected_hpsi_recomputed_in_backward": False,
            },
            "adam": asdict(adam),
            "fixed_parameter_measurement": fixed,
            "work_ledger": {
                "training_samples": chains,
                "sampler_proposal_attempts": chains * attempts,
                "sampler_projected_amplitude_evaluations": sampler_projected,
                "hpsi_forward_projected_amplitude_evaluations": hpsi_projected,
                "center_only_primal_projected_amplitude_evaluations": center_projected,
                "center_only_adjoint_center_equivalent_work_proxy": center_projected,
                "center_only_forward_projected_amplitude_evaluations": (
                    hpsi_projected + center_projected
                ),
                "legacy_two_hpsi_projected_amplitude_evaluations_proxy": (
                    2 * hpsi_projected
                ),
                "avoided_second_hpsi_connected_projected_amplitude_evaluations": (
                    chains * padded_width
                ),
                "hpsi_actual_connected_configurations": int(np.sum(widths)),
                "hpsi_padded_connected_configurations": chains * padded_width,
                **fixed_work,
            },
        }
        history.append(row)
        checkpoint_history.append(row)
        completed = update
        if update in {75, 100}:
            metadata = _extension_metadata(
                params=params,
                optimizer_state=optimizer_state,
                states=states,
                sampling_key=sampling_key,
                parent_binding=parent_binding,
                step0_binding=step0_binding,
                measurements=measurements,
                resume_count=resume_count,
            )
            save_resumable_checkpoint_generation(
                checkpoint_directory,
                params,
                optimizer_state=optimizer_state,
                completed_steps=completed,
                next_sample_index=completed * chains,
                run_identity=identity,
                history=checkpoint_history,
                metadata=metadata,
            )

    stability = _training_stability(history, protocol)
    latest = load_latest_resumable_checkpoint(
        checkpoint_directory, expected_run_identity=identity
    )
    if hard_failure is not None:
        return {
            "status": "hard_failed",
            "hard_failure": hard_failure,
            "case_id": case["case_id"],
            "arm_id": runtime.arm_id,
            "backbone": runtime.backbone,
            "seed": int(parent["arm_spec"]["seed"]),
            "learning_rate": float(optimizer["learning_rate"]),
            "parent_step50_binding": parent_binding,
            "parent_step0_binding": step0_binding,
            "completed_updates": completed,
            "history": history,
            "fixed_parameter_measurements": measurements,
            "training_stability": stability,
            "latest_valid_extension_checkpoint": {
                "completed_steps": int(latest["completed_steps"]),
                "generation_path": _portable_path(
                    Path(str(latest["checkpoint_generation_path"]))
                ),
            },
            "claim_boundary": CLAIM_BOUNDARY,
        }
    if completed < final_update:
        return {
            "status": "running",
            "hard_failure": None,
            "case_id": case["case_id"],
            "arm_id": runtime.arm_id,
            "backbone": runtime.backbone,
            "seed": int(parent["arm_spec"]["seed"]),
            "learning_rate": float(optimizer["learning_rate"]),
            "completed_updates": completed,
            "history": history,
            "fixed_parameter_measurements": measurements,
            "training_stability": stability,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    by_update = {int(item["update"]): item for item in measurements}
    if set(by_update) != {0, 75, 100}:
        raise RuntimeError("completed extension lacks registered measurements")
    optimization = _paired_final_optimization_gates(
        by_update[0], by_update[100], protocol
    )
    fixed_mixing = all(
        bool(by_update[update]["all_mixing_gates_passed"])
        for update in (0, 75, 100)
    )
    all_mixing = bool(
        fixed_mixing and stability["all_training_stability_gates_passed"]
    )
    metadata = dict(latest["metadata"])
    generation = Path(str(latest["checkpoint_generation_path"]))
    return {
        "status": "complete",
        "hard_failure": None,
        "case_id": case["case_id"],
        "arm_id": runtime.arm_id,
        "backbone": runtime.backbone,
        "seed": int(parent["arm_spec"]["seed"]),
        "learning_rate": float(optimizer["learning_rate"]),
        "optimizer_backend": OPTIMIZER_BACKEND,
        "parent_step50_binding": parent_binding,
        "parent_step0_binding": step0_binding,
        "inherited_parent_state_exact": True,
        "completed_updates": completed,
        "history": history,
        "fixed_parameter_measurements": measurements,
        "parent_short_update50_measurement": {
            "role": "trajectory_diagnostic_only",
            "canonical_sha256": canonical_sha256(
                next(
                    item
                    for item in parent["parent_trial"]["fixed_parameter_measurements"]
                    if int(item["update"]) == 50
                )
            ),
        },
        "training_stability": stability,
        "all_mixing_and_training_stability_gates_passed": all_mixing,
        "final_optimization_diagnostics": optimization,
        "all_final_optimization_gates_passed": optimization[
            "all_final_optimization_gates_passed"
        ],
        "checkpoint": {
            "series_directory": checkpoint_directory.name,
            "latest_generation": generation.name,
            "latest_generation_sha256": file_sha256(generation),
            "run_identity_sha256": canonical_json_sha256(identity),
            "completed_updates": int(latest["completed_steps"]),
            "params_sha256": metadata["params_sha256"],
            "optimizer_state_sha256": metadata["optimizer_state_sha256"],
            "chain_states_sha256": metadata["chain_states_sha256"],
            "sampling_key_sha256": metadata["sampling_key_sha256"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def preflight_spin_sampled_training_v2_extension(
    config_path: Path,
) -> dict[str, object]:
    """Validate the sealed selector/parents and instantiate all four models."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    audited = _audit_selector_and_parents(spec)
    rows: list[dict[str, object]] = []
    for selection in spec["selections"]:
        case_id = str(selection["case_id"])
        selected = audited[case_id]
        for backbone in EXPECTED_ARMS:
            parent = selected["arms"][backbone]
            case = dict(parent["case_spec"])
            bonds, _group, projector = v1_helpers._geometry(
                case, dict(parent["parent_config"]["projection"])
            )
            runtime = runtime_workflow._initialize_arm(
                dict(parent["arm_spec"]),
                rows=int(case["rows"]),
                columns=int(case["columns"]),
                projector=projector,
            )
            step0_params = pytree_sha256(parent["step0_payload"]["params"])
            if pytree_sha256(runtime.params) != step0_params:
                raise RuntimeError("fresh runtime initialization disagrees with parent")
            rows.append(
                {
                    "case_id": case_id,
                    "geometry": case["geometry"],
                    "backbone": backbone,
                    "arm_id": runtime.arm_id,
                    "learning_rate": selected["learning_rate"],
                    "bond_count": len(bonds),
                    "translation_group_order": projector.group_order,
                    "parent_step0_generation_sha256": parent["step0_path_sha256"],
                    "parent_step50_generation_sha256": parent["binding"][
                        "checkpoint_generation_sha256"
                    ],
                    "parent_params_adam_chains_key_verified": True,
                    "runtime_initialization_matches_parent_step0": True,
                }
            )
    preflight = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "preflight_complete",
        "config_path": _portable_path(config_path),
        "config_sha256": canonical_sha256(spec),
        "selector_result_payload_sha256": spec["selector_binding"][
            "result_payload_sha256"
        ],
        "selector_is_training_authority": False,
        "parent_artifacts_mutated": False,
        "long_training_or_measurement_started": False,
        "arms": rows,
        "protocol": deepcopy(spec["protocol"]),
        "source_manifest_sha256": canonical_sha256(_source_manifest()),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    preflight["runtime_identity_sha256"] = canonical_sha256(
        preflight["runtime_identity"]
    )
    preflight["result_payload_sha256"] = _result_payload_sha256(preflight)
    return preflight


def run_spin_sampled_training_v2_extension(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or exactly resume all four parent-bound confirmatory arms."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    audited = _audit_selector_and_parents(spec)
    config_hash = canonical_sha256(spec)
    source_manifest = _source_manifest()
    source_hash = canonical_sha256(source_manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_hash = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if overwrite:
        if result_path.exists():
            result_path.unlink()
        for child in output.glob("checkpoint_extension_*"):
            if child.is_dir():
                shutil.rmtree(child)
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        if (
            payload.get("schema_version") != RESULT_SCHEMA_VERSION
            or payload.get("config_sha256") != config_hash
            or payload.get("source_manifest_sha256") != source_hash
            or payload.get("runtime_identity_sha256") != runtime_hash
        ):
            raise RuntimeError("refusing stale confirmatory extension result")
        if payload.get("status") == "complete":
            if payload.get("result_payload_sha256") != _result_payload_sha256(payload):
                raise RuntimeError("confirmatory extension result seal failed")
            return payload
    else:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "domain": "spin",
            "task": TASK,
            "optimizer_backend": OPTIMIZER_BACKEND,
            "status": "running",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": _portable_path(config_path),
            "config_sha256": config_hash,
            "config": spec,
            "source_sha256": source_manifest,
            "source_manifest_sha256": source_hash,
            "runtime_identity": runtime_identity,
            "runtime_identity_sha256": runtime_hash,
            "selector_binding": dict(spec["selector_binding"]),
            "selector_is_training_authority": False,
            "parent_artifacts_and_checkpoint_series_immutable": True,
            "new_identity_and_checkpoint_series": True,
            "cases": [],
            "claim_boundary": CLAIM_BOUNDARY,
        }
        atomic_json(result_path, payload)
    try:
        existing = {item["case_id"]: item for item in payload["cases"]}
        for selection in spec["selections"]:
            case_id = str(selection["case_id"])
            selected = audited[case_id]
            case_payload = existing.get(case_id)
            if case_payload is None:
                case_payload = {
                    "case_id": case_id,
                    "geometry": selected["geometry"],
                    "candidate_id": selected["candidate_id"],
                    "learning_rate": selected["learning_rate"],
                    "selector_decision_sha256": selected[
                        "selector_decision_sha256"
                    ],
                    "trials": [],
                }
                payload["cases"].append(case_payload)
                existing[case_id] = case_payload
                atomic_json(result_path, payload)
            complete = {item["backbone"] for item in case_payload["trials"]}
            for backbone in EXPECTED_ARMS:
                if backbone in complete:
                    continue
                parent = selected["arms"][backbone]
                case = dict(parent["case_spec"])
                bonds, _group, projector = v1_helpers._geometry(
                    case, dict(parent["parent_config"]["projection"])
                )
                runtime = runtime_workflow._initialize_arm(
                    dict(parent["arm_spec"]),
                    rows=int(case["rows"]),
                    columns=int(case["columns"]),
                    projector=projector,
                )
                trial = _run_extension_arm(
                    runtime=runtime,
                    parent=parent,
                    case=case,
                    bonds=bonds,
                    spec=spec,
                    selector_decision_sha256=selected[
                        "selector_decision_sha256"
                    ],
                    checkpoint_directory=output
                    / f"checkpoint_extension_{case_id}_{runtime.arm_id}",
                    config_hash=config_hash,
                    source_hash=source_hash,
                    runtime_hash=runtime_hash,
                    overwrite=overwrite,
                )
                if trial["status"] == "running":  # pragma: no cover
                    raise RuntimeError("production extension stopped before update 100")
                trial["trial_payload_sha256"] = trial_payload_sha256(trial)
                case_payload["trials"].append(trial)
                atomic_json(result_path, payload)
        _finalize_extension_payload(payload)
        atomic_json(result_path, payload)
        return payload
    except BaseException as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(result_path, payload)
        raise


__all__ = [
    "CLAIM_BOUNDARY",
    "OPTIMIZER_BACKEND",
    "TASK",
    "preflight_spin_sampled_training_v2_extension",
    "run_spin_sampled_training_v2_extension",
]
