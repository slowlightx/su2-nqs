"""Compile plus warm matched-work SR steps per N=36 system and spin arm."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.vmc.matched_work_sr import (
    MatchedWorkSRControls,
    run_resumable_matched_work_sr,
)
from su2_nqs.vmc.scale_first_arms import build_scale_first_arm_runtime

from .spin_scale_first_accelerator_slices import _selection, _shared_kwargs
from .spin_scale_first_three_arm_preflight import ARM_IDS, _bonds, _validate


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_training_calibration"


def run_spin_scale_first_training_calibration(
    config_path: Path,
    output: Path,
    *,
    case_id: str,
    arm_id: str,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run registered timing steps; forbid interpretation of their energies."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    calibration = spec.get("training_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("training_calibration is required")
    sample_count = int(calibration["sample_count"])
    case, case_index, _ = _selection(spec, case_id, arm_id, sample_count)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    arm_index = ARM_IDS.index(arm_id)
    initialization_seed = int(spec["seed"]) + 10 * case_index
    runtime = build_scale_first_arm_runtime(
        arm_id,
        geometry=str(case["geometry"]),
        rows=int(case["rows"]),
        columns=int(case["columns"]),
        bonds=_bonds(str(case["geometry"]), float(case["j2_over_j1"])),
        shared_model=_shared_kwargs(spec["shared_spatial_backbone"]),
        initialization_key=jax.random.PRNGKey(initialization_seed),
        su2_connected_capacity=int(
            calibration["su2_connected_capacity"]
        ),
        su2_amplitude_tile_size=int(
            calibration["su2_amplitude_tile_size"]
        ),
    )
    initial_hash = pytree_sha256(runtime.params)
    controls = MatchedWorkSRControls(
        total_steps=int(calibration["total_steps"]),
        sample_count=sample_count,
        seed=int(calibration["seed"]) + 10 * case_index + arm_index,
        damping=float(calibration["damping"]),
        cg_tolerance=float(calibration["cg_tolerance"]),
        cg_max_iterations=int(calibration["cg_max_iterations"]),
        step_size=float(calibration["step_size"]),
        maximum_relative_parameter_step=float(
            calibration["maximum_relative_parameter_step"]
        ),
        checkpoint_every_steps=1,
    )
    result = run_resumable_matched_work_sr(
        runtime.params,
        sample_fn=runtime.sample_fn,
        local_energy_fn=runtime.local_energy_fn,
        log_amplitude_fn=runtime.log_amplitude_fn,
        controls=controls,
        checkpoint_directory=output / "checkpoints",
        run_identity={
            "task": TASK,
            "config_sha256": canonical_sha256(spec),
            "case_id": case_id,
            "arm_id": arm_id,
            "initialization_seed": initialization_seed,
            "initial_params_sha256": initial_hash,
        },
    )
    history = result["history"]
    last = history[-1]
    runtime_identity = numerical_runtime_identity()
    passed = bool(
        runtime_identity["jax_backend"] == "gpu"
        and result["completed_steps"] == int(calibration["total_steps"])
        and result["cumulative_samples"]
        == sample_count * int(calibration["total_steps"])
        and all(row["full_h_evaluations"] == 1 for row in history)
        and all(
            row["full_h_bond_terms"] == sample_count * runtime.bond_count
            for row in history
        )
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "case_id": case_id,
        "arm_id": arm_id,
        "initialization_seed": initialization_seed,
        "initial_params_sha256": initial_hash,
        "final_params_sha256": pytree_sha256(result["params"]),
        "sample_count": sample_count,
        "total_steps": int(calibration["total_steps"]),
        "bond_count": runtime.bond_count,
        "su2_amplitude_tile_size": int(
            calibration["su2_amplitude_tile_size"]
        ),
        "su2_connected_capacity": int(
            calibration["su2_connected_capacity"]
        ),
        "history": history,
        "cumulative_wall_seconds": result["cumulative_wall_seconds"],
        "checkpoint_created": result["latest_checkpoint"] is not None,
        "passed": passed,
        "physics_interpretation_forbidden": True,
        "trained_energy_claim_allowed": False,
        "runtime_identity": runtime_identity,
        "claim_boundary": (
            "Compile-plus-warm force/SR timing calibration only; these "
            "short-trajectory energies are not variational results."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["TASK", "run_spin_scale_first_training_calibration"]
