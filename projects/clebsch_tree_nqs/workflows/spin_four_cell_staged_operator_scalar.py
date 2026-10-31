"""Seal staged operator-scalar development and untouched audits."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_staged_operator_scalar_frontier import (
    staged_operator_scalar_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_four_cell_staged_operator_scalar"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid staged operator-scalar identity")
    if spec.get("phase") not in {"development", "untouched_audit"}:
        raise ValueError("phase must be development or untouched_audit")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("at least three seeds are required")
    checked = tuple(int(value) for value in seeds)
    if len(set(checked)) != len(checked) or any(value < 0 for value in checked):
        raise ValueError("seeds must be distinct and nonnegative")
    previous = spec.get("excluded_previous_seeds")
    if not isinstance(previous, list):
        raise TypeError("excluded_previous_seeds must be a list")
    if set(checked) & {int(value) for value in previous}:
        raise ValueError("new seeds overlap a previous set")
    if min(
        int(optimization.get("warmup_steps", 0)),
        int(optimization.get("branch_steps", 0)),
    ) < 1:
        raise ValueError("warmup_steps and branch_steps must be positive")
    if spec.get("phase") == "untouched_audit" and not isinstance(
        spec.get("frozen_development_artifact"), str
    ):
        raise ValueError("untouched audit requires a frozen development artifact")


def run_spin_four_cell_staged_operator_scalar(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal one staged phase."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = staged_operator_scalar_frontier(
        seeds=tuple(int(value) for value in optimization["seeds"]),
        warmup_steps=int(optimization["warmup_steps"]),
        branch_steps=int(optimization["branch_steps"]),
        learning_rate=float(optimization["learning_rate"]),
    )
    phase = str(spec["phase"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": phase,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": (
            "freeze_staged_operator_scalar_and_preregister_audit"
            if phase == "development" and suite["all_cases_passed"]
            else "staged_operator_scalar_development_failed"
            if phase == "development"
            else "prepare_n36_staged_operator_canary"
            if suite["all_cases_passed"]
            else "hold_n36_staged_operator_audit_failed"
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


__all__ = ["TASK", "run_spin_four_cell_staged_operator_scalar"]
