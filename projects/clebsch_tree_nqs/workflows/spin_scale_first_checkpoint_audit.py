"""Held-out checkpoint audit for the scale-first N=36 trajectories.

The workflow never resumes optimization.  It reconstructs the registered
native sampler and full-H evaluator, loads sealed parameter generations, and
uses common fresh random streams to obtain paired energy/variance changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.vmc.matched_work_sr import MatchedWorkSRControls
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint
from su2_nqs.vmc.scale_first_arms import SCALE_FIRST_ARM_IDS, build_scale_first_arm_runtime

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_initialization_pilot import _load_pilot_contract
from .spin_scale_first_three_arm_preflight import _bonds


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_checkpoint_audit"


def _standard_error(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _load_contract(
    config_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    spec = json.loads(config_path.read_text())
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid checkpoint-audit identity")
    source = Path(str(spec.get("source_convergence")))
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()
    pilot, audit, base = _load_pilot_contract(source)
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
        or steps[-1] > int(pilot["controls"]["total_steps"])
    ):
        raise ValueError("checkpoint_steps must be sorted unique integers beginning at zero")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("audit_seeds must contain at least two unique integers")
    production_seeds = set(base.get("production_contract", {}).get("seeds", []))
    if production_seeds.intersection(seeds):
        raise ValueError("audit seeds must be disjoint from production seeds")
    count = controls.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    microbatch = controls.get("device_microbatch_size")
    if (
        isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch < 1
        or count % microbatch
    ):
        raise ValueError("device_microbatch_size must divide sample_count")
    return spec, pilot, audit, base


def _expected_checkpoint_identity(
    *,
    pilot: dict[str, object],
    audit: dict[str, object],
    base: dict[str, object],
    case_id: str,
    arm_id: str,
    case_index: int,
    initialization_seed: int,
    initial_params_sha256: str,
    selected_scale: float,
) -> dict[str, object]:
    controls_spec = pilot["controls"]
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
        checkpoint_every_steps=1,
    )
    return {
        "task": "spin_scale_first_initialization_pilot",
        "config_sha256": canonical_sha256(pilot),
        "initialization_audit_config_sha256": canonical_sha256(audit),
        "source_config_sha256": canonical_sha256(base),
        "case_id": case_id,
        "arm_id": arm_id,
        "initialization_seed": initialization_seed,
        "initial_params_sha256": initial_params_sha256,
        "direct_phase_initialization_scale": selected_scale,
        "driver": "resumable_matched_work_matrix_free_sr_v1",
        "controls": {
            name: getattr(controls, name) for name in controls.__dataclass_fields__
        },
    }


def _checkpoint_path(root: Path, case_id: str, arm_id: str, step: int) -> Path:
    directory = root / case_id / arm_id / "checkpoints" / case_id / arm_id
    matches = sorted(directory.glob(f"step_{step:08d}_*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one checkpoint for {case_id}/{arm_id}/step {step}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _seed_statistics(local_energies: np.ndarray) -> dict[str, float]:
    values = np.asarray(local_energies, dtype=np.complex128)
    mean = np.mean(values)
    residual = np.abs(values - mean) ** 2
    variance = float(np.mean(residual))
    return {
        "energy": float(mean.real),
        "energy_imaginary_abs": float(abs(mean.imag)),
        "variance": variance,
        "energy_standard_error": float(np.sqrt(variance / len(values))),
        "variance_standard_error": _standard_error([float(value) for value in residual]),
    }


def run_spin_scale_first_checkpoint_audit(
    config_path: Path,
    checkpoint_root: Path,
    output: Path,
    *,
    case_id: str,
    arm_id: str,
    overwrite: bool = False,
) -> dict[str, object]:
    """Audit one system/arm checkpoint sequence on untouched random streams."""

    spec, pilot, audit, base = _load_contract(config_path)
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

    selected_scale = float(
        audit["controls"]["selected_direct_phase_initialization_scale"]
    )
    shared = dict(base["shared_spatial_backbone"])
    shared["direct_phase_initialization_scale"] = selected_scale
    initialization_seed = int(pilot["controls"]["seed"]) + 100 * case_index
    runtime = build_scale_first_arm_runtime(
        arm_id,
        geometry=str(case["geometry"]),
        rows=int(case["rows"]),
        columns=int(case["columns"]),
        bonds=_bonds(str(case["geometry"]), float(case["j2_over_j1"])),
        shared_model=_shared_kwargs(shared),
        initialization_key=jax.random.PRNGKey(initialization_seed),
        su2_connected_capacity=int(audit["controls"]["su2_connected_capacity"]),
        su2_amplitude_tile_size=int(audit["controls"]["su2_amplitude_tile_size"]),
    )
    initial_hash = pytree_sha256(runtime.params)
    expected_identity = _expected_checkpoint_identity(
        pilot=pilot,
        audit=audit,
        base=base,
        case_id=case_id,
        arm_id=arm_id,
        case_index=case_index,
        initialization_seed=initialization_seed,
        initial_params_sha256=initial_hash,
        selected_scale=selected_scale,
    )
    rows = []
    controls = spec["controls"]
    for step in controls["checkpoint_steps"]:
        step = int(step)
        if step == 0:
            params = runtime.params
            source_file = None
        else:
            source = _checkpoint_path(checkpoint_root, case_id, arm_id, step)
            checkpoint = load_resumable_checkpoint(
                source, expected_run_identity=expected_identity
            )
            if int(checkpoint["completed_steps"]) != step:
                raise ValueError("checkpoint step does not match its filename")
            params = checkpoint["params"]
            source_file = str(source.relative_to(checkpoint_root))
        params_hash = pytree_sha256(params)
        for seed in controls["audit_seeds"]:
            seed = int(seed)
            samples = runtime.sample_fn(
                params, jax.random.PRNGKey(seed), int(controls["sample_count"])
            )
            jax.block_until_ready(samples)
            local_values = []
            overflow_count = 0
            full_h_seconds = 0.0
            bond_count = None
            microbatch = int(controls["device_microbatch_size"])
            for start in range(0, int(controls["sample_count"]), microbatch):
                local = runtime.local_energy_fn(
                    params, samples[start : start + microbatch]
                )
                local_values.append(np.asarray(local.local_energies))
                overflow_count += int(local.overflow_count)
                full_h_seconds += float(local.evaluation_seconds)
                if bond_count is None:
                    bond_count = int(local.bond_count)
                elif bond_count != int(local.bond_count):
                    raise RuntimeError("microbatches disagree on Hamiltonian bond count")
            combined = np.concatenate(local_values)
            statistics = _seed_statistics(combined)
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
                    "full_h_bond_terms": int(controls["sample_count"])
                    * int(bond_count),
                    "full_h_seconds": full_h_seconds,
                    **statistics,
                }
            )

    runtime_identity = numerical_runtime_identity()
    passed = bool(
        len(rows)
        == len(controls["checkpoint_steps"]) * len(controls["audit_seeds"])
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
        "source_convergence_config_sha256": canonical_sha256(pilot),
        "case_id": case_id,
        "arm_id": arm_id,
        "initial_params_sha256": initial_hash,
        "rows": rows,
        "passed": passed,
        "runtime_identity": runtime_identity,
        "production_seeds_untouched": True,
        "training_performed": False,
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


def _paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case_id"]), str(row["arm_id"])), []).append(row)
    summaries = []
    for (case_id, arm_id), arm_rows in sorted(grouped.items()):
        by_step: dict[int, dict[int, dict[str, object]]] = {}
        for row in arm_rows:
            by_step.setdefault(int(row["checkpoint_step"]), {})[
                int(row["audit_seed"])
            ] = row
        baseline = by_step[min(by_step)]
        for step, seed_rows in sorted(by_step.items()):
            seeds = sorted(seed_rows)
            energies = [float(seed_rows[seed]["energy"]) for seed in seeds]
            variances = [float(seed_rows[seed]["variance"]) for seed in seeds]
            energy_delta = [
                float(seed_rows[seed]["energy"]) - float(baseline[seed]["energy"])
                for seed in seeds
            ]
            variance_delta = [
                float(seed_rows[seed]["variance"])
                - float(baseline[seed]["variance"])
                for seed in seeds
            ]
            delta_mean = float(np.mean(variance_delta))
            delta_se = _standard_error(variance_delta)
            if delta_mean - 2.0 * delta_se > 0.0:
                variance_classification = "resolved_increase"
            elif delta_mean + 2.0 * delta_se < 0.0:
                variance_classification = "resolved_decrease"
            else:
                variance_classification = "unresolved"
            summaries.append(
                {
                    "case_id": case_id,
                    "arm_id": arm_id,
                    "checkpoint_step": step,
                    "audit_seed_count": len(seeds),
                    "energy_mean": float(np.mean(energies)),
                    "energy_between_seed_standard_error": _standard_error(energies),
                    "variance_mean": float(np.mean(variances)),
                    "variance_between_seed_standard_error": _standard_error(variances),
                    "paired_energy_change_from_step_0": float(np.mean(energy_delta)),
                    "paired_energy_change_standard_error": _standard_error(energy_delta),
                    "paired_variance_change_from_step_0": delta_mean,
                    "paired_variance_change_standard_error": delta_se,
                    "variance_change_classification": variance_classification,
                }
            )
    return summaries


def run_spin_scale_first_checkpoint_audit_merge(
    config_path: Path,
    input_root: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Merge all six read-only checkpoint audits and classify variance trends."""

    spec, pilot, _audit, base = _load_contract(config_path)
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
            raise ValueError(f"duplicate checkpoint-audit slice: {identity}")
        observed.add(identity)
        payloads.append(payload)
        rows.extend(payload.get("rows", []))
    summaries = _paired_summary(rows)
    coverage = observed == expected
    expected_summaries = len(expected) * len(spec["controls"]["checkpoint_steps"])
    passed = bool(
        coverage
        and payloads
        and all(payload.get("passed") is True for payload in payloads)
        and len(summaries) == expected_summaries
    )
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
        "source_convergence_config_sha256": canonical_sha256(pilot),
        "coverage_passed": coverage,
        "all_slice_gates_passed": passed,
        "summaries": summaries,
        "production_release_allowed": False,
        "production_seeds_untouched": True,
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = [
    "TASK",
    "_load_contract",
    "_paired_summary",
    "run_spin_scale_first_checkpoint_audit",
    "run_spin_scale_first_checkpoint_audit_merge",
]
