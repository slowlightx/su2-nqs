"""Fail-closed merger for the six single-seed N=36 convergence slices."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.vmc.scale_first_arms import SCALE_FIRST_ARM_IDS

from .spin_scale_first_initialization_pilot import _load_pilot_contract


TASK = "spin_scale_first_single_seed_convergence_merge"


def _trend_summary(
    values: list[float],
    *,
    block_size: int,
    edge_block_count: int,
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) % block_size:
        raise ValueError("trajectory length must be divisible by block_size")
    blocks = np.mean(array.reshape(-1, block_size), axis=1)
    if len(blocks) < 2 * edge_block_count:
        raise ValueError("trajectory has too few blocks for the edge comparison")
    differences = np.diff(blocks)
    slope = float(np.polyfit(np.arange(len(blocks)), blocks, 1)[0])
    initial = float(np.mean(blocks[:edge_block_count]))
    final = float(np.mean(blocks[-edge_block_count:]))
    return {
        "block_means": [float(value) for value in blocks],
        "linear_block_slope": slope,
        "initial_edge_mean": initial,
        "final_edge_mean": final,
        "final_minus_initial": final - initial,
        "negative_adjacent_block_fraction": float(np.mean(differences < 0.0)),
    }


def _telemetry(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    memory = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    utilization = np.asarray([float(row[4]) for row in rows], dtype=np.float64)
    return {
        "sample_count": len(rows),
        "peak_memory_mib": float(np.max(memory)),
        "mean_gpu_utilization_percent": float(np.mean(utilization)),
        "maximum_gpu_utilization_percent": float(np.max(utilization)),
    }


def run_spin_scale_first_convergence_merge(
    config_path: Path,
    input_root: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Merge all six slices and decide whether production may be released."""

    pilot, audit, base = _load_pilot_contract(config_path)
    gate = pilot.get("convergence_gate")
    if not isinstance(gate, dict):
        raise TypeError("convergence_gate must be an object")
    block_size = int(gate["block_size"])
    edge_count = int(gate["initial_and_final_block_count"])
    if block_size < 1 or edge_count < 1:
        raise ValueError("block sizes must be positive")
    expected = {
        (str(case["case_id"]), arm_id)
        for case in base["systems"]
        for arm_id in SCALE_FIRST_ARM_IDS
    }
    files = sorted(input_root.glob("*/*/results.json"))
    rows = []
    observed = set()
    for path in files:
        payload = json.loads(path.read_text())
        if len(payload.get("runs", [])) != 1:
            raise ValueError(f"slice must contain exactly one run: {path}")
        run = payload["runs"][0]
        identity = (str(run["case_id"]), str(run["arm_id"]))
        if identity in observed:
            raise ValueError(f"duplicate convergence slice: {identity}")
        observed.add(identity)
        history = run["history"]
        energy = _trend_summary(
            [entry["energy_before_update"] for entry in history],
            block_size=block_size,
            edge_block_count=edge_count,
        )
        variance = _trend_summary(
            [entry["variance_before_update"] for entry in history],
            block_size=block_size,
            edge_block_count=edge_count,
        )
        telemetry_path = path.parent / "gpu_telemetry.csv"
        device = _telemetry(telemetry_path)
        maximum_residual = float(
            max(entry["sr_relative_linear_residual"] for entry in history)
        )
        negative_energy = float(gate["minimum_negative_energy_block_fraction"])
        negative_variance = float(
            gate["minimum_negative_variance_block_fraction"]
        )
        trend_passed = bool(
            (not gate["require_negative_energy_block_slope"]
             or energy["linear_block_slope"] < 0.0)
            and (not gate["require_negative_variance_block_slope"]
                 or variance["linear_block_slope"] < 0.0)
            and (not gate["require_final_energy_below_initial"]
                 or energy["final_minus_initial"] < 0.0)
            and (not gate["require_final_variance_below_initial"]
                 or variance["final_minus_initial"] < 0.0)
            and energy["negative_adjacent_block_fraction"] >= negative_energy
            and variance["negative_adjacent_block_fraction"] >= negative_variance
        )
        numerical_passed = bool(
            payload.get("passed") is True
            and payload.get("case_filter") == identity[0]
            and payload.get("arm_filter") == identity[1]
            and payload["runtime_identity"]["jax_backend"] == "gpu"
            and run["checkpoint_created"] is True
            and run["completed_steps"] == int(pilot["controls"]["total_steps"])
            and all(entry["full_h_evaluations"] == 1 for entry in history)
            and maximum_residual <= float(gate["maximum_sr_relative_residual"])
            and device["peak_memory_mib"]
            <= float(gate["maximum_gpu_peak_memory_mib"])
        )
        rows.append(
            {
                "case_id": identity[0],
                "arm_id": identity[1],
                "initial_params_sha256": run["initial_params_sha256"],
                "final_params_sha256": run["final_params_sha256"],
                "energy": energy,
                "variance": variance,
                "maximum_sr_relative_residual": maximum_residual,
                "cumulative_wall_seconds": float(run["cumulative_wall_seconds"]),
                "cumulative_samples": int(run["cumulative_samples"]),
                "cumulative_full_h_bond_terms": int(
                    run["cumulative_full_h_bond_terms"]
                ),
                "telemetry": device,
                "numerical_gate_passed": numerical_passed,
                "sustained_descent_gate_passed": trend_passed,
            }
        )
    coverage_passed = observed == expected
    shared_initialization = all(
        len(
            {
                row["initial_params_sha256"]
                for row in rows
                if row["case_id"] == str(case["case_id"])
            }
        )
        == 1
        for case in base["systems"]
    )
    production_release = bool(
        coverage_passed
        and shared_initialization
        and all(row["numerical_gate_passed"] for row in rows)
        and all(row["sustained_descent_gate_passed"] for row in rows)
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "summary.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"summary exists; pass --overwrite: {destination}")
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(pilot),
        "initialization_audit_config_sha256": canonical_sha256(audit),
        "source_config_sha256": canonical_sha256(base),
        "coverage_passed": coverage_passed,
        "shared_initialization_arrays_within_system": shared_initialization,
        "runs": sorted(rows, key=lambda row: (row["case_id"], row["arm_id"])),
        "total_gpu_wall_seconds": float(
            sum(row["cumulative_wall_seconds"] for row in rows)
        ),
        "all_numerical_gates_passed": bool(
            rows and all(row["numerical_gate_passed"] for row in rows)
        ),
        "all_sustained_descent_gates_passed": bool(
            rows and all(row["sustained_descent_gate_passed"] for row in rows)
        ),
        "production_release_allowed": production_release,
        "production_seed_count_if_released": 3,
        "claim_boundary": pilot["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_trend_summary", "run_spin_scale_first_convergence_merge"]
