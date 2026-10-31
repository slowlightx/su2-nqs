"""Seal fresh-seed shared-backbone recoupling Ritz calibrations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_recoupling_ritz_frontier import (
    recoupling_ritz_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_four_cell_recoupling_ritz"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid recoupling-Ritz identity")
    if spec.get("phase") != "development_calibration":
        raise ValueError("this workflow is development_calibration only")
    comparison = spec.get("comparison")
    if not isinstance(comparison, dict):
        raise TypeError("comparison must be an object")
    required = {
        "shared_frozen_backbone": True,
        "physical_and_scrambled_operator_count": 8,
        "scramble_inside_gamma_a1": True,
        "report_one_lanczos": True,
    }
    if any(comparison.get(key) != value for key, value in required.items()):
        raise ValueError("the preregistered Ritz comparison changed")
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
    if int(optimization.get("warmup_steps", 0)) < 1:
        raise ValueError("warmup_steps must be positive")
    rate = float(optimization.get("learning_rate", 0.0))
    if not 0.0 < rate <= 0.1:
        raise ValueError("learning_rate must lie in (0,0.1]")


def run_spin_four_cell_recoupling_ritz(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal one variable-projection calibration."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = recoupling_ritz_frontier(
        seeds=tuple(int(value) for value in optimization["seeds"]),
        warmup_steps=int(optimization["warmup_steps"]),
        learning_rate=float(optimization["learning_rate"]),
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "development_calibration",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "decision": (
            "physical_subspace_passes_develop_differentiable_variable_projection"
            if suite["all_cases_passed"]
            else "physical_subspace_failed_stop_recoupling_route"
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


__all__ = ["TASK", "run_spin_four_cell_recoupling_ritz"]
