"""Untouched audit of the frozen overlapping-recoupling variable projection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_recoupling_variable_projection import (
    recoupling_variable_projection_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_four_cell_recoupling_variable_projection_audit"
DEVELOPMENT_WORKFLOW = (
    ROOT
    / "projects/clebsch_tree_nqs/workflows"
    / "spin_four_cell_recoupling_variable_projection.py"
)


def _validated_frozen_development(spec: dict[str, object]) -> dict[str, object]:
    relative = spec.get("frozen_development_artifact")
    expected_hash = spec.get("frozen_development_result_sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise TypeError("frozen development path and hash are required")
    path = ROOT / relative
    payload = json.loads(path.read_text())
    stored_hash = payload.get("result_sha256")
    calculated_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    if stored_hash != calculated_hash or stored_hash != expected_hash:
        raise ValueError("frozen development result seal failed")
    if payload.get("promotion_decision") != (
        "freeze_variable_projection_and_preregister_untouched_audit"
    ):
        raise ValueError("development artifact did not license an audit")
    current_manifest = transitive_local_source_manifest(
        (DEVELOPMENT_WORKFLOW,),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    if payload.get("source_manifest") != current_manifest:
        raise ValueError("frozen development source has changed")
    return payload


def _validate(spec: dict[str, object]) -> dict[str, object]:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid variable-projection audit identity")
    if spec.get("phase") != "untouched_audit":
        raise ValueError("phase must be untouched_audit")
    frozen = _validated_frozen_development(spec)
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("at least three audit seeds are required")
    checked = tuple(int(value) for value in seeds)
    if len(set(checked)) != len(checked) or any(value < 0 for value in checked):
        raise ValueError("audit seeds must be distinct and nonnegative")
    excluded = spec.get("excluded_all_previous_seeds")
    if not isinstance(excluded, list):
        raise TypeError("excluded_all_previous_seeds must be a list")
    if set(checked) & {int(value) for value in excluded}:
        raise ValueError("audit seeds overlap a previous seed set")
    frozen_optimization = frozen["config"]["optimization"]
    if (
        int(optimization.get("steps", 0)) != int(frozen_optimization["steps"])
        or float(optimization.get("learning_rate", 0.0))
        != float(frozen_optimization["learning_rate"])
    ):
        raise ValueError("audit optimization differs from frozen development")
    if spec.get("architecture") != frozen["config"].get("architecture"):
        raise ValueError("audit architecture differs from frozen development")
    return frozen


def run_spin_four_cell_recoupling_variable_projection_audit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the untouched seeds without changing the frozen implementation."""

    spec = json.loads(config_path.read_text())
    frozen = _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = recoupling_variable_projection_frontier(
        seeds=tuple(int(value) for value in optimization["seeds"]),
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "untouched_audit",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "frozen_development_result_sha256": frozen["result_sha256"],
        "frozen_development_source_manifest_sha256": frozen[
            "source_manifest_sha256"
        ],
        "suite": suite,
        "promotion_decision": (
            "license_sampled_n36_variable_projection_canary"
            if suite["all_cases_passed"]
            else "hold_n36_variable_projection_audit_failed"
        ),
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "TASK",
    "run_spin_four_cell_recoupling_variable_projection_audit",
]
