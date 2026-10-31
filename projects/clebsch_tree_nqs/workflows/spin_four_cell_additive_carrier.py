"""Seal the four-cell physical-versus-scrambled additive-carrier test."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_additive_carrier import (
    additive_carrier_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_four_cell_additive_carrier"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid four-cell additive-carrier config identity")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("optimization.seeds must be nonempty")
    checked = tuple(int(value) for value in seeds)
    if any(value < 0 for value in checked) or len(set(checked)) != len(checked):
        raise ValueError("optimization.seeds must be distinct and nonnegative")
    if int(optimization.get("steps", 0)) < 1:
        raise ValueError("optimization.steps must be positive")
    if spec.get("arms") != [
        "no_carrier",
        "physical_additive_carrier",
        "scrambled_additive_carrier",
    ]:
        raise ValueError("all three additive-carrier arms must be registered")


def run_spin_four_cell_additive_carrier(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the carrier mechanism test."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = additive_carrier_frontier(
        seeds=tuple(int(value) for value in optimization["seeds"]),
        steps=int(optimization["steps"]),
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": (
            "carrier_mechanism_pass"
            if suite["all_cases_passed"]
            else "carrier_mechanism_not_robust"
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


__all__ = ["TASK", "run_spin_four_cell_additive_carrier"]
