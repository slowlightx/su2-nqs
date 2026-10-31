"""Seal the preregistered symmetry-labelled response-operator development."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_response_operator_frontier import (
    symmetry_response_operator_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_symmetry_response_operator"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid symmetry-response workflow identity")
    if spec.get("phase") != "development":
        raise ValueError("the current response workflow is development only")
    mechanism = spec.get("mechanism")
    if not isinstance(mechanism, dict):
        raise TypeError("mechanism must be an object")
    required = {
        "ground_sector": "Gamma-A1 singlet",
        "triangular_certificate_channels": [
            "E2_J1_nematic_return",
            "A2_chirality_return",
        ],
        "square_training_channels": [
            "B1_J1_nematic_return",
            "B2_J2_nematic_return",
            "A2_chirality_return",
        ],
        "scalar_return": "sum_mu O_Gamma,mu^dagger O_Gamma,mu",
        "common_physical_overlapping_operators": 8,
    }
    if any(mechanism.get(key) != value for key, value in required.items()):
        raise ValueError("the preregistered response mechanism changed")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 5:
        raise ValueError("at least five preregistered seeds are required")
    checked = tuple(int(value) for value in seeds)
    if len(set(checked)) != len(checked) or any(value < 0 for value in checked):
        raise ValueError("seeds must be distinct and nonnegative")
    previous = spec.get("excluded_previous_seeds")
    if not isinstance(previous, list) or set(checked) & {
        int(value) for value in previous
    }:
        raise ValueError("response seeds overlap a previous seed set")
    if int(optimization.get("steps", 0)) != 1000:
        raise ValueError("the preregistered run requires exactly 1000 steps")
    if float(optimization.get("learning_rate", 0.0)) != 0.01:
        raise ValueError("the preregistered run requires learning_rate=0.01")


def run_spin_symmetry_response_operator(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run, validate, and provenance-seal the response development."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = symmetry_response_operator_frontier(
        seeds=tuple(int(value) for value in optimization["seeds"]),
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "development",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": (
            "freeze_response_mechanism_and_preregister_untouched_audit"
            if suite["all_gates_passed"]
            else "hold_response_mechanism_development_failed"
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


__all__ = ["TASK", "run_spin_symmetry_response_operator"]
