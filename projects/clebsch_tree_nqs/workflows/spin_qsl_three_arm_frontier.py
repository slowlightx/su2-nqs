"""Reproducible three-arm pi-flux QSL variational benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.qsl_three_arm import (
    triangular_n36_operator_connectivity_preflight,
    triangular_operator_vit_phase_scan,
    triangular_qsl_three_arm_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_qsl_pi_flux_three_arm_frontier"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid QSL three-arm config identity")
    if abs(float(spec.get("j2_over_j1", -1.0)) - 0.125) > 1.0e-14:
        raise ValueError("the preregistered local QSL point is J2/J1=0.125")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    for key in ("jastrow_seeds", "vit_seeds", "operator_seeds"):
        values = optimization.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"optimization.{key} must be a nonempty list")
        checked = tuple(int(value) for value in values)
        if len(set(checked)) != len(checked) or any(value < 0 for value in checked):
            raise ValueError(f"optimization.{key} must contain distinct seeds")
    for key in ("vit_steps", "operator_steps"):
        if int(optimization.get(key, 0)) < 1:
            raise ValueError(f"optimization.{key} must be positive")
    phase_scan = spec.get("phase_scan")
    if not isinstance(phase_scan, dict):
        raise TypeError("phase_scan must be an object")
    anchors = phase_scan.get("exact_energy_anchors_total")
    if not isinstance(anchors, dict) or len(anchors) < 3:
        raise ValueError("phase_scan requires at least three stored anchors")
    if int(phase_scan.get("steps", 0)) < 1:
        raise ValueError("phase_scan.steps must be positive")
    preflight = spec.get("n36_connectivity_preflight")
    if not isinstance(preflight, dict) or not isinstance(
        preflight.get("seeds"), list
    ):
        raise ValueError("n36_connectivity_preflight.seeds must be a list")


def run_spin_qsl_three_arm_frontier(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    optimization = spec["optimization"]
    frontier = triangular_qsl_three_arm_frontier(
        float(spec["j2_over_j1"]),
        jastrow_seeds=tuple(int(value) for value in optimization["jastrow_seeds"]),
        vit_seeds=tuple(int(value) for value in optimization["vit_seeds"]),
        operator_seeds=tuple(
            int(value) for value in optimization["operator_seeds"]
        ),
        vit_steps=int(optimization["vit_steps"]),
        operator_steps=int(optimization["operator_steps"]),
    )
    phase_scan_spec = spec["phase_scan"]
    phase_scan = triangular_operator_vit_phase_scan(
        {
            float(coupling): float(energy)
            for coupling, energy in phase_scan_spec[
                "exact_energy_anchors_total"
            ].items()
        },
        seed=int(phase_scan_spec["seed"]),
        steps=int(phase_scan_spec["steps"]),
    )
    n36_preflight = triangular_n36_operator_connectivity_preflight(
        seeds=tuple(int(value) for value in spec["n36_connectivity_preflight"]["seeds"])
    )
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "frontier": frontier,
        "phase_scan": phase_scan,
        "n36_connectivity_preflight": n36_preflight,
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


__all__ = ["TASK", "run_spin_qsl_three_arm_frontier"]
