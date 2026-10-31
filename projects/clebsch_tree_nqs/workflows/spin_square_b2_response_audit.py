"""Seal the untouched square-B2 response audit."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_b2_response_audit import (
    square_b2_response_audit,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_square_b2_response_audit"
FROZEN_HASH = "9f181b8f4893c1213780af7c2c2457f95937edac5c3251564b22c09fa57db24b"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid B2 audit identity")
    if spec.get("phase") != "untouched_audit":
        raise ValueError("the B2 workflow requires untouched_audit phase")
    if spec.get("frozen_development_result_sha256") != FROZEN_HASH:
        raise ValueError("the frozen development hash changed")
    artifact = ROOT / str(spec.get("frozen_development_artifact"))
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    development = json.loads(artifact.read_text())
    if development.get("result_sha256") != FROZEN_HASH:
        raise ValueError("the development artifact does not match its frozen hash")
    architecture = spec.get("architecture")
    if not isinstance(architecture, dict) or architecture != {
        "common_physical_overlapping_operators": 8,
        "added_response_operators": 1,
        "physical_channel": "B2_J2_nematic_return",
        "matched_control": "isospectral in-A1 orthogonal conjugate of the same B2 return operator",
    }:
        raise ValueError("the frozen B2 architecture changed")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    seeds = optimization.get("seeds")
    if seeds != [1211, 1212, 1213, 1214, 1215]:
        raise ValueError("the untouched B2 seeds changed")
    previous = spec.get("excluded_all_previous_seeds")
    if not isinstance(previous, list) or set(seeds) & set(previous):
        raise ValueError("the B2 audit reuses previous seeds")
    if optimization.get("steps") != 1000 or optimization.get("learning_rate") != 0.01:
        raise ValueError("the B2 audit hyperparameters changed")


def run_spin_square_b2_response_audit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the untouched B2 audit."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = square_b2_response_audit(
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
        "suite": suite,
        "promotion_decision": (
            "retain_B2_error_filter_and_design_scalable_triangular_analogue"
            if suite["gates_passed"]
            else "reject_B2_seed_robustness_hypothesis"
        ),
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(
        payload["source_manifest"]
    )
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_square_b2_response_audit"]
