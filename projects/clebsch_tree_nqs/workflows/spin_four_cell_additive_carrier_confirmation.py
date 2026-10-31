"""Seal the fresh-seed additive-carrier confirmation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_additive_carrier_confirmation import (
    additive_carrier_confirmation,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_four_cell_additive_carrier_confirmation"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid additive-carrier confirmation identity")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    seeds = optimization.get("fresh_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("at least three fresh seeds are required")
    checked = tuple(int(value) for value in seeds)
    if len(set(checked)) != len(checked) or any(value < 0 for value in checked):
        raise ValueError("fresh seeds must be distinct and nonnegative")
    if set(checked) & {0, 1, 2}:
        raise ValueError("development seeds 0,1,2 may not enter confirmation")
    if int(optimization.get("steps", 0)) < 1:
        raise ValueError("optimization.steps must be positive")
    floor = float(spec.get("triangular_numerical_floor", 0.0))
    if not 0.0 < floor <= 1.0e-8:
        raise ValueError("triangular_numerical_floor must lie in (0,1e-8]")


def run_spin_four_cell_additive_carrier_confirmation(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the untouched-seed confirmation."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = additive_carrier_confirmation(
        seeds=tuple(int(value) for value in optimization["fresh_seeds"]),
        steps=int(optimization["steps"]),
        triangular_numerical_floor=float(spec["triangular_numerical_floor"]),
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
            "prepare_n36_sampled_canary"
            if suite["confirmation_passed"]
            else "hold_n36_carrier_confirmation_failed"
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


__all__ = ["TASK", "run_spin_four_cell_additive_carrier_confirmation"]
