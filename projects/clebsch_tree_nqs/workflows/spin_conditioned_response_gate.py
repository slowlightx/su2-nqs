"""Seal the preregistered configuration-conditioned response development."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_conditioned_response_frontier import (
    conditioned_response_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_conditioned_response_gate"
SEEDS = [1311, 1312, 1313, 1314, 1315]


def _validate(spec: dict[str, object]) -> None:
    """Reject any drift from the frozen exact-small development contract."""

    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid conditioned-response workflow identity")
    if spec.get("phase") != "development":
        raise ValueError("the conditioned-response workflow is development only")
    mechanism = spec.get("mechanism")
    required = {
        "ground_sector": "Gamma-A1 singlet",
        "reduced_inputs": "six causal local-Schur q,a and two-dimensional position tokens",
        "triangular_certificate_channels": [
            "E2_J1_nematic_return",
            "A2_chirality_return",
        ],
        "square_training_channels": [
            "B1_J1_nematic_return",
            "B2_J2_nematic_return",
            "A2_chirality_return",
        ],
        "conditioned_operator": "trace-centered Frobenius-normalized anticommutator of an A1-projected diagonal gate and a prevalidated response-return operator",
        "common_physical_overlapping_operators": 8,
        "matched_control": "a fixed orthogonal conjugation of every conditioned operator at every parameter point, preserving its spectrum, gate values, parameter count, and operator count",
    }
    if not isinstance(mechanism, dict) or any(
        mechanism.get(key) != value for key, value in required.items()
    ):
        raise ValueError("the preregistered conditioned-response mechanism changed")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    if optimization.get("seeds") != SEEDS:
        raise ValueError("the conditioned-response development seeds changed")
    previous = spec.get("excluded_all_previous_seeds")
    if not isinstance(previous, list) or set(SEEDS) & {
        int(value) for value in previous
    }:
        raise ValueError("conditioned-response seeds overlap previous seeds")
    if optimization.get("steps") != 1000:
        raise ValueError("the conditioned-response development requires 1000 steps")
    if optimization.get("learning_rate") != 0.01:
        raise ValueError("the conditioned-response development requires lr=0.01")


def run_spin_conditioned_response_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run, validate, and provenance-seal the development campaign."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    suite = conditioned_response_frontier(
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
            "freeze_conditioned_response_and_preregister_untouched_audit"
            if suite["all_gates_passed"]
            else "hold_conditioned_response_development_failed"
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


__all__ = ["TASK", "run_spin_conditioned_response_gate"]
