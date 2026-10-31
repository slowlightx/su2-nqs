"""Checkpointed N=36 one-seed VMC from the admitted analytic carrier."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.vmc.matched_work_sr import MatchedWorkSRControls, run_resumable_matched_work_sr
from su2_nqs.vmc.scale_first_arms import SCALE_FIRST_ARM_IDS, build_scale_first_arm_runtime

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _bonds, _validate


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_carrier_single_seed"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_contract(
    config_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    spec = json.loads(config_path.read_text())
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid carrier single-seed identity")
    base = json.loads(_resolve(config_path, spec["source_preflight"]).read_text())
    _validate(base)
    initialization_path = _resolve(config_path, spec["source_initialization_result"])
    initialization = json.loads(initialization_path.read_text())
    if (
        initialization.get("task")
        != "spin_scale_first_n36_carrier_distillation"
        or initialization.get("remote_single_seed_training_allowed") is not True
    ):
        raise ValueError("the N36 carrier initialization has not passed")
    controls = spec.get("controls")
    if not isinstance(controls, dict):
        raise TypeError("controls must be an object")
    return spec, base, initialization, initialization_path.parent


def run_spin_scale_first_carrier_single_seed(
    config_path: Path,
    output: Path,
    *,
    case_id: str,
    arm_id: str,
    maximum_steps_this_call: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or resume one selected carrier-initialized system/arm trajectory."""

    spec, base, initialization, artifact_root = _load_contract(config_path)
    if arm_id not in SCALE_FIRST_ARM_IDS:
        raise ValueError(f"unknown arm_id: {arm_id!r}")
    selected = [
        (index, case)
        for index, case in enumerate(base["systems"])
        if str(case["case_id"]) == case_id
    ]
    if len(selected) != 1:
        raise ValueError(f"unknown case_id: {case_id!r}")
    case_index, case = selected[0]
    source = [
        row
        for row in initialization["runs"]
        if row["case_id"] == case_id and row["arm_id"] == arm_id
    ]
    if len(source) != 1 or source[0]["passed"] is not True:
        raise ValueError("selected carrier artifact is missing or failed")
    source = source[0]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    controls_spec = spec["controls"]
    shared_raw = dict(base["shared_spatial_backbone"])
    shared_raw["direct_phase_initialization_scale"] = 0.0
    runtime = build_scale_first_arm_runtime(
        arm_id,
        geometry=str(case["geometry"]),
        rows=6,
        columns=6,
        bonds=_bonds(str(case["geometry"]), float(case["j2_over_j1"])),
        shared_model=_shared_kwargs(shared_raw),
        initialization_key=jax.random.PRNGKey(
            int(controls_spec["initialization_template_seed"]) + case_index
        ),
        su2_connected_capacity=int(controls_spec["su2_connected_capacity"]),
        su2_amplitude_tile_size=int(controls_spec["su2_amplitude_tile_size"]),
        physical_phase_convention="marshall_checkerboard",
    )
    manifest_path = artifact_root / str(source["parameter_artifact"])
    params = load_parameter_artifact(manifest_path, runtime.params)
    if pytree_sha256(params) != source["trained_params_sha256"]:
        raise RuntimeError("carrier parameter artifact changed")
    runtime = replace(runtime, params=params)
    arm_index = SCALE_FIRST_ARM_IDS.index(arm_id)
    controls = MatchedWorkSRControls(
        total_steps=int(controls_spec["total_steps"]),
        sample_count=int(controls_spec["sample_count"]),
        seed=int(controls_spec["seed"]) + 100 * case_index + arm_index,
        damping=float(controls_spec["damping"]),
        cg_tolerance=float(controls_spec["cg_tolerance"]),
        cg_max_iterations=int(controls_spec["cg_max_iterations"]),
        step_size=float(controls_spec["step_size"]),
        maximum_relative_parameter_step=float(
            controls_spec["maximum_relative_parameter_step"]
        ),
        checkpoint_every_steps=int(controls_spec["checkpoint_every_steps"]),
    )
    initial_hash = pytree_sha256(params)
    result = run_resumable_matched_work_sr(
        params,
        sample_fn=runtime.sample_fn,
        local_energy_fn=runtime.local_energy_fn,
        log_amplitude_fn=runtime.log_amplitude_fn,
        controls=controls,
        checkpoint_directory=output / "checkpoints",
        run_identity={
            "task": TASK,
            "config_sha256": canonical_sha256(spec),
            "source_preflight_sha256": canonical_sha256(base),
            "source_initialization_result_sha256": initialization["result_sha256"],
            "case_id": case_id,
            "arm_id": arm_id,
            "initial_params_sha256": initial_hash,
            "physical_phase_convention": "marshall_checkerboard",
        },
        maximum_steps_this_call=maximum_steps_this_call,
    )
    history = result["history"]
    complete = int(result["completed_steps"]) == int(controls_spec["total_steps"])
    passed = bool(
        complete
        and len(history) == int(controls_spec["total_steps"])
        and all(
            np.isfinite(
                [
                    row["energy_before_update"],
                    row["variance_before_update"],
                    row["sr_relative_linear_residual"],
                ]
            ).all()
            for row in history
        )
        and all(row["full_h_evaluations"] == 1 for row in history)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete" if complete else "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_preflight_sha256": canonical_sha256(base),
        "source_initialization_result_sha256": initialization["result_sha256"],
        "case_id": case_id,
        "arm_id": arm_id,
        "initial_params_sha256": initial_hash,
        "final_params_sha256": pytree_sha256(result["params"]),
        "parameter_count": int(
            sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params))
        ),
        "source_parameter_artifact": str(manifest_path),
        "history": history,
        "completed_steps": int(result["completed_steps"]),
        "cumulative_samples": int(result["cumulative_samples"]),
        "cumulative_full_h_bond_terms": int(result["cumulative_full_h_bond_terms"]),
        "cumulative_wall_seconds": float(result["cumulative_wall_seconds"]),
        "latest_checkpoint": result["latest_checkpoint"],
        "resumed": result["resumed"],
        "passed": passed,
        "maximum_steps_this_call": maximum_steps_this_call,
        "production_budget_released": False,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["TASK", "_load_contract", "run_spin_scale_first_carrier_single_seed"]
