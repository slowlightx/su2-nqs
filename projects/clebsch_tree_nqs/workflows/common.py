"""Shared exact-support campaign orchestration for spin and Hubbard studies.

The domain modules own only physical system construction, model evaluation,
and observables.  This module owns artifact identity, resume semantics, paired
initialization checks, and the common experiment loop so that a new system or
scan does not require another top-level runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax

from su2_nqs.frontier.artifacts import (
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    paired_initialization_audit,
    repo_relative_path,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.frontier.route_policy import (
    FROZEN_PRODUCTION_MODULE_PREFIXES,
    assert_no_frozen_runtime_modules,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExactWorkflow:
    """Callbacks and metadata needed by one exact-support physical domain."""

    domain: str
    source_entry_paths: tuple[Path, ...]
    prepare_case: Callable[
        [dict[str, object], dict[str, object]], dict[str, object]
    ]
    case_record: Callable[
        [dict[str, object], dict[str, object], float, dict[str, object]],
        dict[str, object],
    ]
    run_trial: Callable[..., dict[str, object]]
    trial_identity: Callable[..., str]
    timing_scope: str
    route_metadata: Mapping[str, object] = field(default_factory=dict)


def source_manifest(workflow: ExactWorkflow) -> dict[str, str]:
    """Return the transitive source seal for one domain workflow."""

    return transitive_local_source_manifest(
        workflow.source_entry_paths,
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _load_reusable_result(
    result_path: Path,
    *,
    config_sha256: str,
    source_manifest_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    existing = json.loads(result_path.read_text())
    if existing.get("config_sha256") != config_sha256:
        raise RuntimeError(f"refusing differently configured result: {result_path}")
    if existing.get("source_manifest_sha256") != source_manifest_sha256:
        raise RuntimeError(f"refusing stale-source result: {result_path}")
    if existing.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise RuntimeError(f"refusing different-runtime result: {result_path}")
    if existing.get("status") == "complete":
        embedded = embedded_trial_manifest(existing)
        if existing.get("trial_manifest_sha256") != canonical_sha256(embedded):
            raise RuntimeError(f"refusing result with a broken trial seal: {result_path}")
        return existing
    if existing.get("status") != "running":
        raise RuntimeError(f"refusing result with invalid status: {result_path}")
    return existing


def validate_complete_payload(
    payload: Mapping[str, object], workflow: ExactWorkflow
) -> None:
    """Validate a paper-facing result against current source and trial seals."""

    if payload.get("status") != "complete":
        raise RuntimeError("paper-facing result is not complete")
    if payload.get("domain") != workflow.domain:
        raise RuntimeError("paper-facing result belongs to a different domain")
    current_manifest = source_manifest(workflow)
    if payload.get("source_sha256") != current_manifest:
        raise RuntimeError("paper-facing result embeds a different source manifest")
    if payload.get("source_manifest_sha256") != canonical_sha256(current_manifest):
        raise RuntimeError("paper-facing result is stale relative to current source")
    embedded = embedded_trial_manifest(payload)
    if payload.get("trial_manifest_sha256") != canonical_sha256(embedded):
        raise RuntimeError("paper-facing result has a broken embedded-trial seal")


def run_exact_campaign(
    config_path: Path,
    output: Path,
    *,
    workflow: ExactWorkflow,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or resume one immutable exact-support campaign."""

    runtime_offenders = assert_no_frozen_runtime_modules(tuple(sys.modules))
    spec = json.loads(config_path.read_text())
    if str(spec.get("domain", "")) != workflow.domain:
        raise ValueError(
            f"config domain {spec.get('domain')!r} does not match "
            f"selected workflow {workflow.domain!r}"
        )
    config_hash = canonical_sha256(spec)
    manifest = source_manifest(workflow)
    manifest_hash = canonical_sha256(manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_identity_hash = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        existing = _load_reusable_result(
            result_path,
            config_sha256=config_hash,
            source_manifest_sha256=manifest_hash,
            runtime_identity_sha256=runtime_identity_hash,
        )
        if existing["status"] == "complete":
            print(f"reused immutable complete result: {result_path}", flush=True)
            return existing
        print(f"resuming checkpointed campaign: {result_path}", flush=True)

    route_boundary: dict[str, object] = {
        "historical_project_imports": False,
        "forbidden_module_prefixes": list(FROZEN_PRODUCTION_MODULE_PREFIXES),
        "forbidden_registry_sha256": canonical_sha256(
            FROZEN_PRODUCTION_MODULE_PREFIXES
        ),
        "runtime_offenders": list(runtime_offenders),
        "hamiltonian_diagonalization_performed": False,
        **dict(workflow.route_metadata),
    }
    payload: dict[str, Any] = {
        "schema_version": 4,
        "timing_schema_version": 2,
        "domain": workflow.domain,
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": repo_relative_path(config_path, root=ROOT),
        "config_sha256": config_hash,
        "config": spec,
        "source_sha256": manifest,
        "source_manifest_sha256": manifest_hash,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_hash,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "route_boundary": route_boundary,
        "timing_scope": workflow.timing_scope,
        "cases": [],
    }

    for case_spec in spec["cases"]:
        preparation_started = time.perf_counter()
        data = workflow.prepare_case(case_spec, spec)
        preparation_seconds = time.perf_counter() - preparation_started
        case = workflow.case_record(case_spec, data, preparation_seconds, spec)
        case["trials"] = []

        arm_ids = set(case_spec["arm_ids"])
        arms = [arm for arm in spec["arms"] if arm["arm_id"] in arm_ids]
        if len(arms) != len(arm_ids):
            raise ValueError(f"case {case_spec['case_id']} references an unknown arm")
        for arm in arms:
            for seed_value in spec["optimization"]["seeds"]:
                seed = int(seed_value)
                trial_path = output / (
                    f"trial_{case_spec['case_id']}_{arm['arm_id']}_seed{seed}.json"
                )
                trial_identity = workflow.trial_identity(
                    config_sha256=config_hash,
                    source_manifest_sha256=manifest_hash,
                    runtime_identity_sha256=runtime_identity_hash,
                    case_id=str(case_spec["case_id"]),
                    arm=arm,
                    seed=seed,
                    data=data,
                )
                if trial_path.exists() and not overwrite:
                    trial = json.loads(trial_path.read_text())
                    valid = (
                        trial.get("config_sha256") == config_hash
                        and trial.get("source_manifest_sha256") == manifest_hash
                        and trial.get("runtime_identity_sha256")
                        == runtime_identity_hash
                        and trial.get("trial_identity_sha256") == trial_identity
                        and trial.get("trial_payload_sha256")
                        == trial_payload_sha256(trial)
                    )
                    if not valid:
                        raise RuntimeError(f"refusing stale trial: {trial_path}")
                    source = "reused"
                else:
                    trial = workflow.run_trial(arm, seed=seed, spec=spec, data=data)
                    trial["config_sha256"] = config_hash
                    trial["source_manifest_sha256"] = manifest_hash
                    trial["runtime_identity_sha256"] = runtime_identity_hash
                    trial["trial_identity_sha256"] = trial_identity
                    trial["trial_payload_sha256"] = trial_payload_sha256(trial)
                    atomic_json(trial_path, trial)
                    source = "computed"
                case["trials"].append(trial)
                metrics = trial["metrics"]
                print(
                    f"{source} {case_spec['case_id']} {arm['arm_id']} seed={seed}: "
                    f"dE/N={metrics['energy_error_per_site']:.3e}, "
                    f"var/N={metrics['variance_per_site']:.3e}, "
                    f"wall={trial['trial_wall_seconds_excluding_case_preparation']:.2f}s",
                    flush=True,
                )
                atomic_json(result_path, payload)

        case["paired_initialization_audit"] = paired_initialization_audit(
            case["trials"]
        )
        payload["cases"].append(case)
        atomic_json(result_path, payload)

    final_runtime_offenders = assert_no_frozen_runtime_modules(tuple(sys.modules))
    payload["route_boundary"]["final_runtime_offenders"] = list(
        final_runtime_offenders
    )
    trial_manifest = embedded_trial_manifest(payload)
    payload["trial_manifest"] = trial_manifest
    payload["trial_manifest_sha256"] = canonical_sha256(trial_manifest)
    payload["status"] = "complete"
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "ExactWorkflow",
    "run_exact_campaign",
    "source_manifest",
    "validate_complete_payload",
]
