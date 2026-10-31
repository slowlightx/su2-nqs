"""Untouched-chain audit for carrier-initialized N=36 checkpoints.

This workflow is read-only: it reconstructs the admitted carrier initialization,
loads sealed VMC checkpoints, and evaluates every registered checkpoint on common
random streams that were not used by optimization.  The merge gate releases no
multi-seed production budget unless both energy and variance show a resolved
paired decrease for all six system/arm trajectories.
"""

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
from su2_nqs.vmc.matched_work_sr import MatchedWorkSRControls
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint
from su2_nqs.vmc.scale_first_arms import SCALE_FIRST_ARM_IDS, build_scale_first_arm_runtime

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_carrier_single_seed import _load_contract as _load_training_contract
from .spin_scale_first_checkpoint_audit import (
    _paired_summary,
    _seed_statistics,
)
from .spin_scale_first_three_arm_preflight import _bonds


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_carrier_checkpoint_audit"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_contract(
    config_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    Path,
]:
    spec = json.loads(config_path.read_text())
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid carrier checkpoint-audit identity")
    training_path = _resolve(config_path, spec["source_training"])
    training, base, initialization, artifact_root = _load_training_contract(training_path)
    controls = spec.get("controls")
    if not isinstance(controls, dict):
        raise TypeError("controls must be an object")
    steps = controls.get("checkpoint_steps")
    seeds = controls.get("audit_seeds")
    if (
        not isinstance(steps, list)
        or not steps
        or any(isinstance(value, bool) or not isinstance(value, int) for value in steps)
        or sorted(set(steps)) != steps
        or steps[0] != 0
        or steps[-1] > int(training["controls"]["total_steps"])
    ):
        raise ValueError("checkpoint_steps must be sorted unique integers beginning at zero")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("audit_seeds must contain at least two unique integers")
    training_seeds = {
        int(training["controls"]["seed"]) + 100 * case_index + arm_index
        for case_index, _case in enumerate(base["systems"])
        for arm_index, _arm in enumerate(SCALE_FIRST_ARM_IDS)
    }
    production_seeds = set(base.get("production_contract", {}).get("seeds", []))
    if training_seeds.intersection(seeds) or production_seeds.intersection(seeds):
        raise ValueError("audit seeds must be disjoint from training and production seeds")
    count = controls.get("sample_count")
    microbatch = controls.get("device_microbatch_size")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if (
        isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch < 1
        or count % microbatch
    ):
        raise ValueError("device_microbatch_size must divide sample_count")
    return spec, training, base, initialization, artifact_root


def _training_controls(
    training: dict[str, object], *, case_index: int, arm_id: str
) -> MatchedWorkSRControls:
    raw = training["controls"]
    arm_index = SCALE_FIRST_ARM_IDS.index(arm_id)
    return MatchedWorkSRControls(
        total_steps=int(raw["total_steps"]),
        sample_count=int(raw["sample_count"]),
        seed=int(raw["seed"]) + 100 * case_index + arm_index,
        damping=float(raw["damping"]),
        cg_tolerance=float(raw["cg_tolerance"]),
        cg_max_iterations=int(raw["cg_max_iterations"]),
        step_size=float(raw["step_size"]),
        maximum_relative_parameter_step=float(raw["maximum_relative_parameter_step"]),
        checkpoint_every_steps=int(raw["checkpoint_every_steps"]),
    )


def _checkpoint_path(root: Path, case_id: str, arm_id: str, step: int) -> Path:
    """Return one carrier-run checkpoint from its process-isolated slice."""

    directory = root / case_id / arm_id / "checkpoints"
    matches = sorted(directory.glob(f"step_{step:08d}_*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one carrier checkpoint for {case_id}/{arm_id}/"
            f"step {step}; found {len(matches)}"
        )
    return matches[0]


def _run_identity(
    *,
    training: dict[str, object],
    base: dict[str, object],
    initialization: dict[str, object],
    case_index: int,
    case_id: str,
    arm_id: str,
    initial_params_sha256: str,
) -> dict[str, object]:
    controls = _training_controls(training, case_index=case_index, arm_id=arm_id)
    return {
        "task": "spin_scale_first_carrier_single_seed",
        "config_sha256": canonical_sha256(training),
        "source_preflight_sha256": canonical_sha256(base),
        "source_initialization_result_sha256": initialization["result_sha256"],
        "case_id": case_id,
        "arm_id": arm_id,
        "initial_params_sha256": initial_params_sha256,
        "physical_phase_convention": "marshall_checkerboard",
        "driver": "resumable_matched_work_matrix_free_sr_v1",
        "controls": {
            name: getattr(controls, name) for name in controls.__dataclass_fields__
        },
    }


def _build_carrier_runtime(
    *,
    training: dict[str, object],
    base: dict[str, object],
    initialization: dict[str, object],
    artifact_root: Path,
    case: dict[str, object],
    case_index: int,
    arm_id: str,
) -> tuple[object, object, str, dict[str, object], Path]:
    """Reconstruct one admitted carrier runtime and its sealed identity."""

    case_id = str(case["case_id"])
    source_rows = [
        row
        for row in initialization["runs"]
        if row["case_id"] == case_id and row["arm_id"] == arm_id
    ]
    if len(source_rows) != 1 or source_rows[0]["passed"] is not True:
        raise ValueError("selected carrier initialization is missing or failed")
    shared = dict(base["shared_spatial_backbone"])
    shared["direct_phase_initialization_scale"] = 0.0
    training_raw = training["controls"]
    runtime = build_scale_first_arm_runtime(
        arm_id,
        geometry=str(case["geometry"]),
        rows=int(case["rows"]),
        columns=int(case["columns"]),
        bonds=_bonds(str(case["geometry"]), float(case["j2_over_j1"])),
        shared_model=_shared_kwargs(shared),
        initialization_key=jax.random.PRNGKey(
            int(training_raw["initialization_template_seed"]) + case_index
        ),
        su2_connected_capacity=int(training_raw["su2_connected_capacity"]),
        su2_amplitude_tile_size=int(training_raw["su2_amplitude_tile_size"]),
        physical_phase_convention="marshall_checkerboard",
    )
    source = source_rows[0]
    manifest_path = artifact_root / str(source["parameter_artifact"])
    params = load_parameter_artifact(manifest_path, runtime.params)
    if pytree_sha256(params) != source["trained_params_sha256"]:
        raise RuntimeError("carrier parameter artifact changed")
    runtime = replace(runtime, params=params)
    initial_hash = pytree_sha256(params)
    expected_identity = _run_identity(
        training=training,
        base=base,
        initialization=initialization,
        case_index=case_index,
        case_id=case_id,
        arm_id=arm_id,
        initial_params_sha256=initial_hash,
    )
    return runtime, params, initial_hash, expected_identity, manifest_path


def run_spin_scale_first_carrier_checkpoint_audit(
    config_path: Path,
    checkpoint_root: Path,
    output: Path,
    *,
    case_id: str,
    arm_id: str,
    overwrite: bool = False,
) -> dict[str, object]:
    """Audit one carrier-initialized trajectory without resuming training."""

    spec, training, base, initialization, artifact_root = _load_contract(config_path)
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
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")

    runtime, params, initial_hash, expected_identity, manifest_path = (
        _build_carrier_runtime(
        training=training,
        base=base,
        initialization=initialization,
        artifact_root=artifact_root,
        case=case,
        case_index=case_index,
        arm_id=arm_id,
        )
    )

    rows = []
    controls = spec["controls"]
    for raw_step in controls["checkpoint_steps"]:
        step = int(raw_step)
        if step == 0:
            checkpoint_params = params
            source_file = None
        else:
            source_path = _checkpoint_path(checkpoint_root, case_id, arm_id, step)
            checkpoint = load_resumable_checkpoint(
                source_path, expected_run_identity=expected_identity
            )
            if int(checkpoint["completed_steps"]) != step:
                raise ValueError("checkpoint step does not match its filename")
            checkpoint_params = checkpoint["params"]
            source_file = str(source_path.relative_to(checkpoint_root))
        params_hash = pytree_sha256(checkpoint_params)
        for raw_seed in controls["audit_seeds"]:
            seed = int(raw_seed)
            samples = runtime.sample_fn(
                checkpoint_params,
                jax.random.PRNGKey(seed),
                int(controls["sample_count"]),
            )
            jax.block_until_ready(samples)
            local_values = []
            overflow_count = 0
            full_h_seconds = 0.0
            bond_count = None
            microbatch = int(controls["device_microbatch_size"])
            for start in range(0, int(controls["sample_count"]), microbatch):
                local = runtime.local_energy_fn(
                    checkpoint_params, samples[start : start + microbatch]
                )
                local_values.append(np.asarray(local.local_energies))
                overflow_count += int(local.overflow_count)
                full_h_seconds += float(local.evaluation_seconds)
                if bond_count is None:
                    bond_count = int(local.bond_count)
                elif bond_count != int(local.bond_count):
                    raise RuntimeError("microbatches disagree on Hamiltonian bond count")
            statistics = _seed_statistics(np.concatenate(local_values))
            rows.append(
                {
                    "case_id": case_id,
                    "arm_id": arm_id,
                    "checkpoint_step": step,
                    "checkpoint_file": source_file,
                    "params_sha256": params_hash,
                    "audit_seed": seed,
                    "sample_count": int(controls["sample_count"]),
                    "device_microbatch_size": microbatch,
                    "full_h_kernel_calls": int(controls["sample_count"]) // microbatch,
                    "overflow_count": overflow_count,
                    "full_h_bond_terms": int(controls["sample_count"]) * int(bond_count),
                    "full_h_seconds": full_h_seconds,
                    **statistics,
                }
            )

    runtime_identity = numerical_runtime_identity()
    passed = bool(
        len(rows) == len(controls["checkpoint_steps"]) * len(controls["audit_seeds"])
        and runtime_identity["jax_backend"] == "gpu"
        and all(row["overflow_count"] == 0 for row in rows)
        and all(
            np.isfinite(
                [
                    row["energy"],
                    row["energy_imaginary_abs"],
                    row["variance"],
                    row["energy_standard_error"],
                    row["variance_standard_error"],
                ]
            ).all()
            for row in rows
        )
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_training_config_sha256": canonical_sha256(training),
        "case_id": case_id,
        "arm_id": arm_id,
        "initial_params_sha256": initial_hash,
        "rows": rows,
        "passed": passed,
        "runtime_identity": runtime_identity,
        "training_performed": False,
        "audit_seeds_untouched": True,
        "production_budget_released": False,
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


def _resolved_descent(summary: dict[str, object], quantity: str) -> bool:
    mean = float(summary[f"paired_{quantity}_change_from_step_0"])
    error = float(summary[f"paired_{quantity}_change_standard_error"])
    return mean + 2.0 * error < 0.0


def run_spin_scale_first_carrier_checkpoint_audit_merge(
    config_path: Path,
    input_root: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Merge all six audits and apply the pre-registered production gate."""

    spec, training, base, _initialization, _artifact_root = _load_contract(config_path)
    expected = {
        (str(case["case_id"]), arm_id)
        for case in base["systems"]
        for arm_id in SCALE_FIRST_ARM_IDS
    }
    payloads = []
    observed = set()
    rows = []
    for path in sorted(input_root.glob("*/*/results.json")):
        payload = json.loads(path.read_text())
        identity = (str(payload.get("case_id")), str(payload.get("arm_id")))
        if identity in observed:
            raise ValueError(f"duplicate carrier checkpoint-audit slice: {identity}")
        observed.add(identity)
        payloads.append(payload)
        rows.extend(payload.get("rows", []))
    summaries = _paired_summary(rows)
    final_step = max(int(step) for step in spec["controls"]["checkpoint_steps"])
    final = [row for row in summaries if int(row["checkpoint_step"]) == final_step]
    energy_gate = bool(final) and all(_resolved_descent(row, "energy") for row in final)
    variance_gate = bool(final) and all(_resolved_descent(row, "variance") for row in final)
    coverage = observed == expected
    audit_passed = bool(
        coverage
        and payloads
        and all(payload.get("passed") is True for payload in payloads)
        and len(final) == len(expected)
    )
    production_release = bool(audit_passed and energy_gate and variance_gate)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "summary.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"summary exists; pass --overwrite: {destination}")
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": f"{TASK}_merge",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_training_config_sha256": canonical_sha256(training),
        "coverage_passed": coverage,
        "all_slice_gates_passed": audit_passed,
        "final_energy_resolved_descent_all_arms": energy_gate,
        "final_variance_resolved_descent_all_arms": variance_gate,
        "summaries": summaries,
        "production_release_allowed": production_release,
        "audit_seeds_untouched": True,
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = [
    "TASK",
    "_build_carrier_runtime",
    "_load_contract",
    "_resolved_descent",
    "_run_identity",
    "run_spin_scale_first_carrier_checkpoint_audit",
    "run_spin_scale_first_carrier_checkpoint_audit_merge",
]
