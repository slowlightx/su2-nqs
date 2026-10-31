"""Immutable-preserving semantic correction for the sealed v2 LR selector.

The primary selector used ``accepted_move_fraction`` for its 0.15 stability
threshold.  The preregistered physical intent was instead the conditional
Metropolis acceptance among nontrivial proposals.  This workflow does not
rewrite that primary decision or any scan artifact.  It strictly revalidates
the three scans and the literal primary selector, changes exactly that one
field in the selection policy, and emits a sealed *secondary sensitivity*
decision with immutable step-50 parent bindings.

No training, timing comparison, or physics claim is made here.
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
from projects.clebsch_tree_nqs.workflows import (
    spin_sampled_training_v2_selector as literal_selector,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest


ROOT = Path(__file__).resolve().parents[3]
TASK = "physical_sampled_optimizer_scan_v2_metric_correction_selector"
RESULT_SCHEMA_VERSION = 1
CLAIM_BOUNDARY = (
    "secondary single-seed preregistration-field sensitivity only; the sealed "
    "literal v1 no-Adam-promotion decision remains immutable and primary; no "
    "training authorization, converged-energy, efficiency, magnetic-order, "
    "phase, QSL, or physics-win claim"
)
EXPECTED_LEARNING_RATES = literal_selector.EXPECTED_LEARNING_RATES
EXPECTED_CASES = literal_selector.EXPECTED_CASES
EXPECTED_ARMS = literal_selector.EXPECTED_ARMS


def _expected_semantic_correction() -> dict[str, object]:
    return {
        "classification": "preregistration_field_semantic_correction",
        "primary_literal_decision_immutable": True,
        "secondary_sensitivity_only": True,
        "old_field": "last10_accepted_move_fraction_median_minimum",
        "new_field": "last10_conditional_move_acceptance_median_minimum",
        "threshold_unchanged": 0.15,
        "all_other_selection_policy_fields_unchanged": True,
        "old_primary_expected_action_each_geometry": "no_adam_promotion",
    }


def _expected_corrected_policy() -> dict[str, object]:
    policy = deepcopy(scan_v2._expected_selection_policy(50))
    stability = dict(policy["stability_gates"])
    threshold = stability.pop("last10_accepted_move_fraction_median_minimum")
    stability["last10_conditional_move_acceptance_median_minimum"] = threshold
    policy["stability_gates"] = stability
    return policy


def _expected_secondary_sampler_audit() -> dict[str, object]:
    return {
        "candidate_eligibility_rule": False,
        "diagnostic_only": True,
        "last10_nontrivial_proposal_fraction_median_minimum": 0.4,
        "last10_chain_snapshot_move_fraction_median_minimum": 0.8,
        "require_all_history_zero_amplitude_chain_fraction_equal_zero": True,
        "accepted_move_fraction_reported_but_not_gated": True,
        "does_not_establish_trained_state_ess_or_autocorrelation": True,
    }


def _expected_parent_binding_contract() -> dict[str, object]:
    return {
        "schema": "metric_correction_step50_parent_binding_v1",
        "purpose": "new_identity_extension_input_only",
        "training_authorized_by_this_selector": False,
        "parent_step": 50,
        "parent_artifacts_and_checkpoint_series_immutable": True,
        "new_run_identity_and_checkpoint_series_required": True,
        "required_fields": list(
            literal_selector._expected_extension_contract()[
                "parent_binding_required"
            ]
        ),
    }


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
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
        raise ValueError("metric-correction selector has an invalid schema/domain/task")
    if spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("metric-correction claim boundary is not registered")
    if spec.get("scan_family_id") != "spin_6x6_center_only_lr_scan_seed11":
        raise ValueError("metric correction is registered only for the seed-11 scan")
    if spec.get("expected_cases") != EXPECTED_CASES:
        raise ValueError("metric-correction expected_cases mismatch")
    if spec.get("expected_arms") != EXPECTED_ARMS:
        raise ValueError("metric-correction expected_arms mismatch")
    if spec.get("semantic_correction") != _expected_semantic_correction():
        raise ValueError("semantic correction must exactly match the registered change")
    if spec.get("selection_policy") != _expected_corrected_policy():
        raise ValueError("corrected selection policy is not exactly registered")
    if spec.get("secondary_sampler_audit") != _expected_secondary_sampler_audit():
        raise ValueError("secondary sampler audit is not exactly registered")
    if spec.get("parent_binding_contract") != _expected_parent_binding_contract():
        raise ValueError("parent binding contract is not exactly registered")

    literal = spec.get("literal_primary_selector")
    if not isinstance(literal, dict) or set(literal) != {"config_path", "result_path"}:
        raise ValueError("literal_primary_selector requires config_path and result_path")
    for value in literal.values():
        if not isinstance(value, str) or not value:
            raise ValueError("literal selector paths must be nonempty strings")

    inputs = spec.get("input_candidates")
    if not isinstance(inputs, list) or len(inputs) != len(EXPECTED_LEARNING_RATES):
        raise ValueError("metric correction requires exactly three scan artifacts")
    identifiers: set[str] = set()
    rates: list[float] = []
    for item in inputs:
        if not isinstance(item, dict):
            raise TypeError("candidate entries must be objects")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in identifiers:
            raise ValueError("candidate identifiers must be nonempty and unique")
        identifiers.add(candidate_id)
        rate = float(item.get("learning_rate", np.nan))
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("candidate learning rates must be positive and finite")
        rates.append(rate)
        for name in ("config_path", "result_path"):
            if not isinstance(item.get(name), str) or not item[name]:
                raise ValueError(f"candidate entry requires {name}")
    if tuple(sorted(rates)) != EXPECTED_LEARNING_RATES:
        raise ValueError("metric correction requires learning rates 1e-3/2e-3/5e-3")


def _load_literal_primary(spec: dict[str, object]) -> dict[str, object]:
    entry = dict(spec["literal_primary_selector"])
    config_path = _resolve_path(entry["config_path"])
    result_path = _resolve_path(entry["result_path"])
    if not config_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("literal primary selector config/result is missing")
    literal_spec = json.loads(config_path.read_text())
    literal_selector._validate_spec(literal_spec)
    if literal_spec.get("input_candidates") != spec.get("input_candidates"):
        raise RuntimeError("corrected selector inputs differ from literal primary inputs")
    payload = json.loads(result_path.read_text())
    if (
        payload.get("schema_version") != literal_selector.RESULT_SCHEMA_VERSION
        or payload.get("task") != literal_selector.TASK
        or payload.get("status") != "complete"
    ):
        raise RuntimeError("literal primary selector artifact is not complete")
    if payload.get("config") != literal_spec:
        raise RuntimeError("literal primary embedded config differs from current file")
    config_hash = canonical_sha256(literal_spec)
    if payload.get("config_sha256") != config_hash:
        raise RuntimeError("literal primary config seal failed")
    current_source = literal_selector._source_manifest()
    source_hash = canonical_sha256(current_source)
    if (
        payload.get("source_sha256") != current_source
        or payload.get("source_manifest_sha256") != source_hash
    ):
        raise RuntimeError("literal primary current-source seal failed")
    if payload.get("result_payload_sha256") != literal_selector._result_payload_sha256(
        payload
    ):
        raise RuntimeError("literal primary result payload seal failed")
    if payload.get("current_scan_source_manifest_sha256") != canonical_sha256(
        scan_v2._source_manifest()
    ):
        raise RuntimeError("literal primary scan-source binding is stale")
    return {
        "config": literal_spec,
        "payload": payload,
        "config_path": config_path,
        "config_file_sha256": file_sha256(config_path),
        "config_sha256": config_hash,
        "result_path": result_path,
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": payload["result_payload_sha256"],
        "source_manifest_sha256": source_hash,
    }


def _history_sampler_audit(
    trial: Mapping[str, object], audit_policy: Mapping[str, object]
) -> dict[str, object]:
    history = trial.get("history")
    if not isinstance(history, list) or len(history) < 10:
        raise RuntimeError("conditional stability audit requires at least ten updates")
    rows: list[dict[str, float]] = []
    for item in history:
        if not isinstance(item, dict) or not isinstance(item.get("sampler"), dict):
            raise RuntimeError("training history lacks sampler diagnostics")
        sampler = item["sampler"]
        row = {
            name: float(sampler[name])
            for name in (
                "accepted_move_fraction",
                "conditional_move_acceptance",
                "nontrivial_proposal_fraction",
                "zero_amplitude_chain_fraction",
                "chain_snapshot_move_fraction",
            )
        }
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in row.values()):
            raise RuntimeError("invalid sampler fraction in training history")
        if not math.isclose(
            row["accepted_move_fraction"],
            row["conditional_move_acceptance"]
            * row["nontrivial_proposal_fraction"],
            rel_tol=2.0e-6,
            abs_tol=2.0e-7,
        ):
            raise RuntimeError("accepted/conditional/nontrivial sampler identity failed")
        rows.append(row)
    tail = rows[-10:]
    accepted = float(np.median([row["accepted_move_fraction"] for row in tail]))
    conditional = float(
        np.median([row["conditional_move_acceptance"] for row in tail])
    )
    nontrivial = float(
        np.median([row["nontrivial_proposal_fraction"] for row in tail])
    )
    chain_move = float(
        np.median([row["chain_snapshot_move_fraction"] for row in tail])
    )
    all_zero_amplitude = all(
        row["zero_amplitude_chain_fraction"] == 0.0 for row in rows
    )
    nontrivial_ok = nontrivial >= float(
        audit_policy["last10_nontrivial_proposal_fraction_median_minimum"]
    )
    chain_move_ok = chain_move >= float(
        audit_policy["last10_chain_snapshot_move_fraction_median_minimum"]
    )
    zero_ok = bool(all_zero_amplitude)
    return {
        "last10_accepted_move_fraction_median": accepted,
        "last10_conditional_move_acceptance_median": conditional,
        "last10_nontrivial_proposal_fraction_median": nontrivial,
        "last10_chain_snapshot_move_fraction_median": chain_move,
        "all_history_zero_amplitude_chain_fraction_equal_zero": zero_ok,
        "diagnostic_nontrivial_fraction_passed": nontrivial_ok,
        "diagnostic_chain_snapshot_move_fraction_passed": chain_move_ok,
        "diagnostic_zero_amplitude_passed": zero_ok,
        "secondary_sampler_audit_passed": bool(
            nontrivial_ok and chain_move_ok and zero_ok
        ),
        "secondary_sampler_audit_changes_candidate_eligibility": False,
    }


def _apply_corrected_metric(
    candidate: dict[str, object], spec: dict[str, object]
) -> None:
    policy = dict(spec["selection_policy"])
    stability = dict(policy["stability_gates"])
    sampler_audit_policy = dict(spec["secondary_sampler_audit"])
    conditional_minimum = float(
        stability["last10_conditional_move_acceptance_median_minimum"]
    )
    relative_maximum = float(stability["relative_update_quantile_maximum"])
    fallback_minimum_z = float(
        policy["continuation_if_no_eligible_candidate"][
            "pool_minimum_each_arm_energy_lowering_paired_mcse"
        ]
    )
    for case_id in EXPECTED_CASES:
        for backbone in EXPECTED_ARMS:
            arm = candidate["cases"][case_id]["arms"][backbone]
            trial = arm["trial"]
            report = deepcopy(arm["metrics"]["report"])
            if not report.get("status_complete_without_hard_failure"):
                report["corrected_stability_gate_passed"] = False
                report["primary_arm_eligible"] = False
                report["fallback_arm_eligible"] = False
                arm["metrics"]["report"] = report
                continue
            history_audit = _history_sampler_audit(trial, sampler_audit_policy)
            relative = float(report["relative_update_quantile_value"])
            corrected_stable = bool(
                history_audit["last10_conditional_move_acceptance_median"]
                >= conditional_minimum
                and relative <= relative_maximum
            )
            report["literal_v1_stability_gate_passed"] = bool(
                report["stability_gate_passed"]
            )
            report["literal_v1_primary_arm_eligible"] = bool(
                report["primary_arm_eligible"]
            )
            report.update(history_audit)
            report["stability_metric"] = (
                "last10_conditional_move_acceptance_median"
            )
            report["stability_gate_passed"] = corrected_stable
            report["corrected_stability_gate_passed"] = corrected_stable
            complete = bool(report["status_complete_without_hard_failure"])
            readiness = bool(report["all_readiness_gates_passed"])
            energy_gate = bool(report["energy_endpoint_gate_passed"])
            variance_gate = bool(report["variance_endpoint_gate_passed"])
            energy_z = report["energy_lowering_paired_z"]
            report["primary_arm_eligible"] = bool(
                complete
                and readiness
                and corrected_stable
                and energy_gate
                and variance_gate
            )
            report["fallback_arm_eligible"] = bool(
                complete
                and readiness
                and corrected_stable
                and variance_gate
                and energy_z is not None
                and float(energy_z) >= fallback_minimum_z
            )
            arm["metrics"]["report"] = report


def _audit_literal_decision(
    literal: dict[str, object], candidates: list[dict[str, object]]
) -> dict[str, object]:
    payload = literal["payload"]
    literal_spec = literal["config"]
    literal_selector._audit_cross_candidate_matching(candidates)
    recomputed = [
        literal_selector._select_geometry(
            candidates,
            case_id=case_id,
            policy=dict(literal_spec["selection_policy"]),
            extension_contract=dict(literal_spec["fallback_extension_contract"]),
        )
        for case_id in EXPECTED_CASES
    ]
    if payload.get("geometry_results") != recomputed:
        raise RuntimeError("literal primary geometry decisions fail recomputation")
    actions = {
        str(item["case_id"]): str(item["decision"]["action"])
        for item in recomputed
    }
    if set(actions) != set(EXPECTED_CASES) or any(
        action != "no_adam_promotion" for action in actions.values()
    ):
        raise RuntimeError("literal primary is not the registered no-promotion result")
    literal_manifest = {
        str(item["candidate_id"]): item for item in payload["input_manifest"]
    }
    for candidate in candidates:
        item = literal_manifest.get(str(candidate["candidate_id"]))
        if item is None:
            raise RuntimeError("literal primary input manifest is incomplete")
        for name in (
            "config_sha256",
            "result_file_sha256",
            "source_manifest_sha256",
            "runtime_identity_sha256",
            "common_scan_profile_sha256",
        ):
            if item.get(name) != candidate[name]:
                raise RuntimeError(f"literal primary input binding mismatch: {name}")
    return {
        "role": "immutable_primary",
        "status": "complete",
        "actions_by_case": actions,
        "config_path": _portable_path(literal["config_path"]),
        "config_file_sha256": literal["config_file_sha256"],
        "config_sha256": literal["config_sha256"],
        "result_path": _portable_path(literal["result_path"]),
        "result_file_sha256": literal["result_file_sha256"],
        "result_payload_sha256": literal["result_payload_sha256"],
        "source_manifest_sha256": literal["source_manifest_sha256"],
        "decision_is_not_overwritten": True,
    }


def _corrected_geometry_results(
    candidates: list[dict[str, object]], spec: dict[str, object]
) -> list[dict[str, object]]:
    policy = dict(spec["selection_policy"])
    extension_contract = literal_selector._expected_extension_contract()
    results: list[dict[str, object]] = []
    for case_id in EXPECTED_CASES:
        result = literal_selector._select_geometry(
            candidates,
            case_id=case_id,
            policy=policy,
            extension_contract=extension_contract,
        )
        decision = result["decision"]
        candidate_id = decision.get("selected_candidate_id")
        if candidate_id is None:
            parent_bindings = None
        else:
            selected = next(
                item for item in candidates if item["candidate_id"] == candidate_id
            )
            parent_bindings = literal_selector._parent_bindings(selected, case_id)
        decision["selected_step50_parent_bindings_for_new_identity_only"] = (
            parent_bindings
        )
        decision["parent_binding_contract"] = deepcopy(
            spec["parent_binding_contract"]
        )
        decision["training_authorized_by_secondary_selector"] = False
        decision["overrides_literal_primary_no_adam_promotion"] = False
        result["decision_role"] = "secondary_metric_semantics_sensitivity"
        result["literal_primary_action"] = "no_adam_promotion"
        result["secondary_sampler_audit_is_candidate_rule"] = False
        results.append(result)
    return results


def run_spin_sampled_training_v2_metric_correction_selector(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    """Seal the conditional-acceptance sensitivity without mutating v1 inputs."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    source_manifest = _source_manifest()
    source_hash = canonical_sha256(source_manifest)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text())
        if (
            existing.get("schema_version") != RESULT_SCHEMA_VERSION
            or existing.get("config_sha256") != config_hash
            or existing.get("source_manifest_sha256") != source_hash
            or existing.get("result_payload_sha256")
            != _result_payload_sha256(existing)
        ):
            raise RuntimeError(f"refusing stale metric-correction result: {result_path}")
        bound_files = [existing["literal_primary_binding"]] + list(
            existing.get("input_manifest", [])
        )
        for item in bound_files:
            for path_name, hash_name in (
                ("config_path", "config_file_sha256"),
                ("result_path", "result_file_sha256"),
            ):
                path = _resolve_path(item[path_name])
                if not path.is_file() or file_sha256(path) != item[hash_name]:
                    raise RuntimeError("metric-correction input changed after sealing")
        return existing

    literal = _load_literal_primary(spec)
    literal_spec = dict(literal["config"])
    current_scan_source = scan_v2._source_manifest()
    candidates = [
        literal_selector._audit_candidate(
            dict(entry),
            selector_spec=literal_spec,
            current_scan_source=current_scan_source,
        )
        for entry in spec["input_candidates"]
    ]
    literal_primary_binding = _audit_literal_decision(literal, candidates)

    corrected_candidates = deepcopy(candidates)
    for candidate in corrected_candidates:
        _apply_corrected_metric(candidate, spec)
    literal_selector._audit_cross_candidate_matching(corrected_candidates)
    geometry_results = _corrected_geometry_results(corrected_candidates, spec)

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
        "scan_family_id": spec["scan_family_id"],
        "common_scan_profile_sha256": candidates[0][
            "common_scan_profile_sha256"
        ],
        "runtime_identity_sha256": candidates[0]["runtime_identity_sha256"],
        "literal_primary_binding": literal_primary_binding,
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
            for candidate in sorted(
                candidates, key=lambda item: float(item["learning_rate"])
            )
        ],
        "semantic_correction": deepcopy(spec["semantic_correction"]),
        "policy_change_audit": {
            "only_changed_gate": (
                "last10_accepted_move_fraction_median_minimum -> "
                "last10_conditional_move_acceptance_median_minimum"
            ),
            "threshold_before_and_after": 0.15,
            "energy_variance_relative_update_score_and_tie_rules_unchanged": True,
            "literal_primary_decision_recomputed_and_verified": True,
            "secondary_sampler_audit_does_not_change_candidate_eligibility": True,
        },
        "selection_policy": deepcopy(spec["selection_policy"]),
        "secondary_sampler_audit": deepcopy(spec["secondary_sampler_audit"]),
        "parent_binding_contract": deepcopy(spec["parent_binding_contract"]),
        "geometry_results": geometry_results,
        "decision_hierarchy": {
            "primary": "sealed_literal_v1_no_adam_promotion",
            "secondary": "conditional_acceptance_metric_sensitivity",
            "secondary_overrides_primary": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_payload_sha256"] = _result_payload_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "CLAIM_BOUNDARY",
    "TASK",
    "run_spin_sampled_training_v2_metric_correction_selector",
]
