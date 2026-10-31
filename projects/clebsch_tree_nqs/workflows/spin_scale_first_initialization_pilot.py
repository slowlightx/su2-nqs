"""Checkpointed two-step pilot for the corrected shared initialization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.vmc.matched_work_sr import MatchedWorkSRControls, run_resumable_matched_work_sr
from su2_nqs.vmc.scale_first_arms import SCALE_FIRST_ARM_IDS, build_scale_first_arm_runtime

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_initialization_audit import _load_contract
from .spin_scale_first_three_arm_preflight import _bonds


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_initialization_pilot"


def _load_pilot_contract(
    config_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    pilot_spec = json.loads(config_path.read_text())
    if (
        pilot_spec.get("schema_version"),
        pilot_spec.get("domain"),
        pilot_spec.get("task"),
    ) != (1, "spin", TASK):
        raise ValueError("invalid initialization-pilot identity")
    source = Path(str(pilot_spec.get("source_audit")))
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()
    audit, base = _load_contract(source)
    controls = pilot_spec.get("controls")
    if not isinstance(controls, dict):
        raise TypeError("controls must be an object")
    for name in ("total_steps", "sample_count", "cg_max_iterations"):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")
    if int(controls["sample_count"]) < 2:
        raise ValueError("controls.sample_count must be at least two")
    for name in (
        "damping",
        "cg_tolerance",
        "step_size",
        "maximum_relative_parameter_step",
    ):
        value = float(controls.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"controls.{name} must be positive and finite")
    minimum_variance = float(controls.get("minimum_variance", np.nan))
    if not np.isfinite(minimum_variance) or minimum_variance < 0.0:
        raise ValueError("controls.minimum_variance must be nonnegative and finite")
    return pilot_spec, audit, base


def run_spin_scale_first_initialization_pilot(
    config_path: Path,
    output: Path,
    *,
    case_id: str | None = None,
    arm_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run two matched SR evaluations per system/arm with resumable checkpoints."""

    pilot_spec, audit, base = _load_pilot_contract(config_path)
    pilot = pilot_spec["controls"]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    selected_scale = float(
        audit["controls"]["selected_direct_phase_initialization_scale"]
    )
    shared = dict(base["shared_spatial_backbone"])
    shared["direct_phase_initialization_scale"] = selected_scale
    if arm_id is not None and arm_id not in SCALE_FIRST_ARM_IDS:
        raise ValueError(f"unknown scale-first arm: {arm_id!r}")
    selected_cases = [
        (index, case)
        for index, case in enumerate(base["systems"])
        if case_id is None or str(case["case_id"]) == case_id
    ]
    if not selected_cases:
        raise ValueError(f"unknown case_id: {case_id!r}")
    selected_arms = (
        SCALE_FIRST_ARM_IDS if arm_id is None else (arm_id,)
    )
    rows = []
    for case_index, case in selected_cases:
        initialization_seed = int(pilot["seed"]) + 100 * case_index
        for selected_arm in selected_arms:
            arm_index = SCALE_FIRST_ARM_IDS.index(selected_arm)
            runtime = build_scale_first_arm_runtime(
                selected_arm,
                geometry=str(case["geometry"]),
                rows=int(case["rows"]),
                columns=int(case["columns"]),
                bonds=_bonds(str(case["geometry"]), float(case["j2_over_j1"])),
                shared_model=_shared_kwargs(shared),
                initialization_key=jax.random.PRNGKey(initialization_seed),
                su2_connected_capacity=int(
                    audit["controls"]["su2_connected_capacity"]
                ),
                su2_amplitude_tile_size=int(
                    audit["controls"]["su2_amplitude_tile_size"]
                ),
            )
            initial_hash = pytree_sha256(runtime.params)
            controls = MatchedWorkSRControls(
                total_steps=int(pilot["total_steps"]),
                sample_count=int(pilot["sample_count"]),
                seed=int(pilot["seed"]) + 100 * case_index + arm_index,
                damping=float(pilot["damping"]),
                cg_tolerance=float(pilot["cg_tolerance"]),
                cg_max_iterations=int(pilot["cg_max_iterations"]),
                step_size=float(pilot["step_size"]),
                maximum_relative_parameter_step=float(
                    pilot["maximum_relative_parameter_step"]
                ),
                checkpoint_every_steps=1,
            )
            result = run_resumable_matched_work_sr(
                runtime.params,
                sample_fn=runtime.sample_fn,
                local_energy_fn=runtime.local_energy_fn,
                log_amplitude_fn=runtime.log_amplitude_fn,
                controls=controls,
                checkpoint_directory=(
                    output / "checkpoints" / str(case["case_id"]) / selected_arm
                ),
                run_identity={
                    "task": TASK,
                    "config_sha256": canonical_sha256(pilot_spec),
                    "initialization_audit_config_sha256": canonical_sha256(audit),
                    "source_config_sha256": canonical_sha256(base),
                    "case_id": str(case["case_id"]),
                    "arm_id": selected_arm,
                    "initialization_seed": initialization_seed,
                    "initial_params_sha256": initial_hash,
                    "direct_phase_initialization_scale": selected_scale,
                },
            )
            history = result["history"]
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "arm_id": selected_arm,
                    "initialization_seed": initialization_seed,
                    "initial_params_sha256": initial_hash,
                    "final_params_sha256": pytree_sha256(result["params"]),
                    "history": history,
                    "completed_steps": int(result["completed_steps"]),
                    "cumulative_samples": int(result["cumulative_samples"]),
                    "cumulative_full_h_bond_terms": int(
                        result["cumulative_full_h_bond_terms"]
                    ),
                    "cumulative_wall_seconds": float(
                        result["cumulative_wall_seconds"]
                    ),
                    "checkpoint_created": result["latest_checkpoint"] is not None,
                    "energy_change_between_sampled_steps": float(
                        history[-1]["energy_before_update"]
                        - history[0]["energy_before_update"]
                    ),
                }
            )
    all_history = [entry for row in rows for entry in row["history"]]
    passed = bool(
        len(rows) == len(selected_cases) * len(selected_arms)
        and all(row["completed_steps"] == int(pilot["total_steps"]) for row in rows)
        and all(row["checkpoint_created"] for row in rows)
        and all(
            np.isfinite(
                [
                    entry["energy_before_update"],
                    entry["variance_before_update"],
                    entry["gradient_norm"],
                    entry["sr_relative_linear_residual"],
                ]
            ).all()
            for entry in all_history
        )
        and all(
            entry["variance_before_update"] >= float(pilot["minimum_variance"])
            for entry in all_history
        )
        and all(entry["full_h_evaluations"] == 1 for entry in all_history)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(pilot_spec),
        "initialization_audit_config_sha256": canonical_sha256(audit),
        "source_config_sha256": canonical_sha256(base),
        "selected_direct_phase_initialization_scale": selected_scale,
        "case_filter": case_id,
        "arm_filter": arm_id,
        "runs": rows,
        "passed": passed,
        "production_seeds_untouched": True,
        "physics_interpretation_forbidden": True,
        "trained_energy_claim_allowed": False,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Two-step numerical trainability pilot only. Independent samples make "
            "the reported step-to-step energy differences diagnostic, not claims."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["TASK", "_load_pilot_contract", "run_spin_scale_first_initialization_pilot"]
