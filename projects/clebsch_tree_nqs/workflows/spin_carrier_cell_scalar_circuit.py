"""Seal the exact-small carrier-preserving cell-scalar falsification gate."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.carrier_cell_scalar_circuit import (
    carrier_cell_scalar_circuit_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_carrier_cell_scalar_circuit"


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid carrier-cell scalar-circuit identity")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("this falsification gate is fixed to triangular 4x4")
    if spec.get("purpose") != "route_falsification_calibration":
        raise ValueError("this artifact must not be labelled as a positive audit")
    couplings = tuple(float(value) for value in spec.get("couplings", ()))
    if couplings != (0.0, 0.125):
        raise ValueError("the calibration couplings must be (0, 0.125)")
    seeds = tuple(int(value) for value in spec.get("seeds", ()))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct paired seeds are required")
    if int(spec.get("depth", 0)) < 1 or int(spec.get("steps", 0)) < 1:
        raise ValueError("depth and steps must be positive")
    if float(spec.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if spec.get("run_full_coupling_scan_only_if_calibration_passes") is not True:
        raise ValueError("the full scan must fail closed")


def run_spin_carrier_cell_scalar_circuit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the paired physical-locality versus scramble calibration."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    frontier, arrays = carrier_cell_scalar_circuit_frontier(
        couplings=tuple(float(value) for value in spec["couplings"]),
        seeds=tuple(int(value) for value in spec["seeds"]),
        depth=int(spec["depth"]),
        steps=int(spec["steps"]),
        learning_rate=float(spec["learning_rate"]),
        maximum_operator_count=int(spec["maximum_operator_count"]),
    )
    _atomic_npz(state_path, arrays)
    passed = bool(frontier["gates"]["license_n36_matrix_free_pilot"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "frontier": frontier,
        "decision": (
            "promote_to_full_n16_coupling_scan" if passed
            else "stop_cell_scalar_route_small_a1_space_not_discriminating"
        ),
        "full_coupling_scan_started": False,
        "n36_gpu_production_authorized": False,
        "repeated_ed_performed": False,
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "runtime_identity": numerical_runtime_identity(),
        "interpretation": (
            "The 14-dimensional exact S=0/Gamma-A1 space makes a 16-parameter "
            "scrambled circuit nearly universal. Failure against that control "
            "invalidates this finite-size locality test and blocks both the full "
            "coupling scan and N36 promotion for this mechanism."
        ),
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


__all__ = ["TASK", "run_spin_carrier_cell_scalar_circuit"]
