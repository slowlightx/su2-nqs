"""Fail-closed summary of carrier-initialized N=36 development trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.vmc.scale_first_arms import SCALE_FIRST_ARM_IDS

from .spin_scale_first_carrier_single_seed import _load_contract as _load_training_contract
from .spin_scale_first_convergence_merge import _telemetry, _trend_summary


TASK = "spin_scale_first_carrier_convergence_merge"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_contract(
    config_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    spec = json.loads(config_path.read_text())
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid carrier convergence-merge identity")
    training_path = _resolve(config_path, spec["source_training"])
    training, base, initialization, _artifact_root = _load_training_contract(
        training_path
    )
    gate = spec.get("convergence_gate")
    if not isinstance(gate, dict):
        raise TypeError("convergence_gate must be an object")
    return spec, training, base


def run_spin_scale_first_carrier_convergence_merge(
    config_path: Path,
    input_root: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Merge all six trajectories; production remains locked pending audit."""

    spec, training, base = _load_contract(config_path)
    gate = spec["convergence_gate"]
    block_size = int(gate["block_size"])
    edge_count = int(gate["initial_and_final_block_count"])
    expected = {
        (str(case["case_id"]), arm_id)
        for case in base["systems"]
        for arm_id in SCALE_FIRST_ARM_IDS
    }
    observed = set()
    rows = []
    for path in sorted(input_root.glob("*/*/results.json")):
        payload = json.loads(path.read_text())
        identity = (str(payload.get("case_id")), str(payload.get("arm_id")))
        if identity in observed:
            raise ValueError(f"duplicate carrier convergence slice: {identity}")
        observed.add(identity)
        history = payload.get("history", [])
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
        device = _telemetry(path.parent / "gpu_telemetry.csv")
        maximum_residual = float(
            max(entry["sr_relative_linear_residual"] for entry in history)
        )
        trend_passed = bool(
            energy["linear_block_slope"] < 0.0
            and variance["linear_block_slope"] < 0.0
            and energy["final_minus_initial"] < 0.0
            and variance["final_minus_initial"] < 0.0
            and energy["negative_adjacent_block_fraction"]
            >= float(gate["minimum_negative_energy_block_fraction"])
            and variance["negative_adjacent_block_fraction"]
            >= float(gate["minimum_negative_variance_block_fraction"])
        )
        numerical_passed = bool(
            payload.get("passed") is True
            and payload.get("status") == "complete"
            and payload.get("completed_steps") == int(training["controls"]["total_steps"])
            and payload.get("runtime_identity", {}).get("jax_backend") == "gpu"
            and payload.get("latest_checkpoint") is not None
            and all(entry["full_h_evaluations"] == 1 for entry in history)
            and maximum_residual <= float(gate["maximum_sr_relative_residual"])
            and device["peak_memory_mib"] <= float(gate["maximum_gpu_peak_memory_mib"])
            and payload.get("production_budget_released") is False
        )
        rows.append(
            {
                "case_id": identity[0],
                "arm_id": identity[1],
                "initial_params_sha256": payload["initial_params_sha256"],
                "final_params_sha256": payload["final_params_sha256"],
                "energy": energy,
                "variance": variance,
                "maximum_sr_relative_residual": maximum_residual,
                "cumulative_wall_seconds": float(payload["cumulative_wall_seconds"]),
                "cumulative_samples": int(payload["cumulative_samples"]),
                "cumulative_full_h_bond_terms": int(
                    payload["cumulative_full_h_bond_terms"]
                ),
                "telemetry": device,
                "numerical_gate_passed": numerical_passed,
                "sustained_descent_gate_passed": trend_passed,
            }
        )
    coverage = observed == expected
    numerical = bool(
        coverage and rows and all(row["numerical_gate_passed"] for row in rows)
    )
    sustained = bool(
        coverage and rows and all(row["sustained_descent_gate_passed"] for row in rows)
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
        "config_sha256": canonical_sha256(spec),
        "source_training_config_sha256": canonical_sha256(training),
        "coverage_passed": coverage,
        "runs": sorted(rows, key=lambda row: (row["case_id"], row["arm_id"])),
        "total_gpu_wall_seconds": float(
            sum(row["cumulative_wall_seconds"] for row in rows)
        ),
        "all_numerical_gates_passed": numerical,
        "all_sustained_descent_gates_passed": sustained,
        "untouched_checkpoint_audit_allowed": numerical,
        "production_release_candidate": bool(numerical and sustained),
        "production_release_allowed": False,
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_load_contract", "run_spin_scale_first_carrier_convergence_merge"]
