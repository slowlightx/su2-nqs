"""Pre-production audit of the shared N=36 three-arm initialization.

The development scan is deliberately separated from the untouched production
seeds.  It asks only whether all controls begin in a nonstationary,
finite-variance state; its energies are not variational results.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.vmc.accelerator_sampled_force import sampled_vmc_log_value_and_grad
from su2_nqs.vmc.scale_first_arms import (
    EXACT_SU2_ARM,
    FIXED_SZ_ARM,
    SCALE_FIRST_ARM_IDS,
    SYMMETRY_UNAWARE_ARM,
    build_scale_first_arm_runtime,
)

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _bonds, _validate


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_initialization_audit"
PHYSICAL_ARMS = (FIXED_SZ_ARM, SYMMETRY_UNAWARE_ARM)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _load_contract(config_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    audit = json.loads(config_path.read_text())
    if (audit.get("schema_version"), audit.get("domain"), audit.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid initialization-audit identity")
    source = Path(str(audit.get("source_config")))
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()
    base = json.loads(source.read_text())
    _validate(base)
    controls = audit.get("controls")
    if not isinstance(controls, dict):
        raise TypeError("controls must be an object")
    scales = controls.get("direct_phase_initialization_scales")
    if not isinstance(scales, list) or len(scales) < 2:
        raise ValueError("at least two direct-phase scales are required")
    scales = [float(value) for value in scales]
    if not all(np.isfinite(scales)) or any(value < 0.0 for value in scales):
        raise ValueError("direct-phase scales must be nonnegative and finite")
    selected = float(controls.get("selected_direct_phase_initialization_scale"))
    if selected not in scales:
        raise ValueError("selected direct-phase scale must be in the scan")
    seeds = controls.get("development_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct development seeds are required")
    if set(int(seed) for seed in seeds) & set(base["production_contract"]["seeds"]):
        raise ValueError("development and production seeds must be disjoint")
    for name in ("scan_sample_count", "selected_sample_count", "su2_connected_capacity"):
        _positive_integer(controls.get(name), name)
    if _positive_integer(controls.get("selected_sample_count"), "selected_sample_count") < 2:
        raise ValueError("selected_sample_count must be at least two")
    tile = _positive_integer(controls.get("su2_amplitude_tile_size"), "tile")
    capacity = int(controls["su2_connected_capacity"])
    if capacity % tile:
        raise ValueError("SU2 amplitude tile must divide connected capacity")
    gates = audit.get("gates")
    if not isinstance(gates, dict):
        raise TypeError("gates must be an object")
    for name in (
        "minimum_generic_variance",
        "minimum_gradient_norm",
        "maximum_generic_fixed_energy_gap",
        "minimum_generic_fixed_variance_ratio",
        "maximum_generic_fixed_variance_ratio",
    ):
        value = float(gates.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"gates.{name} must be positive and finite")
    return audit, base


def _evaluate(
    *,
    base: dict[str, object],
    case: dict[str, object],
    case_index: int,
    arm_id: str,
    seed: int,
    sample_count: int,
    direct_phase_scale: float,
    capacity: int,
    tile_size: int,
    gradient: bool,
) -> dict[str, object]:
    shared = dict(base["shared_spatial_backbone"])
    shared["direct_phase_initialization_scale"] = float(direct_phase_scale)
    runtime = build_scale_first_arm_runtime(
        arm_id,
        geometry=str(case["geometry"]),
        rows=int(case["rows"]),
        columns=int(case["columns"]),
        bonds=_bonds(str(case["geometry"]), float(case["j2_over_j1"])),
        shared_model=_shared_kwargs(shared),
        initialization_key=jax.random.PRNGKey(int(seed)),
        su2_connected_capacity=capacity,
        su2_amplitude_tile_size=tile_size,
    )
    arm_index = SCALE_FIRST_ARM_IDS.index(arm_id)
    sample_key = jax.random.PRNGKey(
        int(seed) + 10_000 + 100 * case_index + arm_index
    )
    samples = runtime.sample_fn(runtime.params, sample_key, sample_count)
    local = runtime.local_energy_fn(runtime.params, samples)
    values = np.asarray(local.local_energies)
    row: dict[str, object] = {
        "case_id": str(case["case_id"]),
        "arm_id": arm_id,
        "seed": int(seed),
        "sample_count": int(sample_count),
        "direct_phase_initialization_scale": float(direct_phase_scale),
        "energy": float(np.real(np.mean(values))),
        "variance": float(np.mean(np.abs(values - np.mean(values)) ** 2)),
        "energy_imaginary_abs": float(abs(np.imag(np.mean(values)))),
        "overflow_count": int(local.overflow_count),
        "parameter_sha256": pytree_sha256(runtime.params),
        "parameter_count": int(
            sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(runtime.params))
        ),
    }
    if arm_id != EXACT_SU2_ARM:
        bits = np.asarray(samples)
        row["mean_absolute_m2"] = float(
            np.mean(np.abs(2 * np.sum(bits, axis=1) - bits.shape[1]))
        )
    if gradient:
        force = sampled_vmc_log_value_and_grad(
            runtime.params,
            samples,
            values,
            log_amplitude_fn=runtime.log_amplitude_fn,
        )
        row["gradient_norm"] = float(force.gradient_norm)
    return row


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["case_id"]),
                str(row["arm_id"]),
                float(row["direct_phase_initialization_scale"]),
            )
            for row in rows
        }
    )
    result = []
    for case_id, arm_id, scale in keys:
        group = [
            row
            for row in rows
            if row["case_id"] == case_id
            and row["arm_id"] == arm_id
            and row["direct_phase_initialization_scale"] == scale
        ]
        result.append(
            {
                "case_id": case_id,
                "arm_id": arm_id,
                "direct_phase_initialization_scale": scale,
                "seed_count": len(group),
                "energy_mean": float(np.mean([row["energy"] for row in group])),
                "energy_minimum": float(np.min([row["energy"] for row in group])),
                "energy_maximum": float(np.max([row["energy"] for row in group])),
                "variance_mean": float(np.mean([row["variance"] for row in group])),
                "variance_minimum": float(np.min([row["variance"] for row in group])),
                "gradient_norm_mean": (
                    None
                    if "gradient_norm" not in group[0]
                    else float(np.mean([row["gradient_norm"] for row in group]))
                ),
                "gradient_norm_minimum": (
                    None
                    if "gradient_norm" not in group[0]
                    else float(np.min([row["gradient_norm"] for row in group]))
                ),
                "overflow_count": int(sum(row["overflow_count"] for row in group)),
                "parameter_counts": sorted(
                    {int(row["parameter_count"]) for row in group}
                ),
            }
        )
    return result


def run_spin_scale_first_initialization_audit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Scan physical controls, then audit the selected protocol on all arms."""

    audit, base = _load_contract(config_path)
    controls = audit["controls"]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    capacity = int(controls["su2_connected_capacity"])
    tile = int(controls["su2_amplitude_tile_size"])
    seeds = [int(seed) for seed in controls["development_seeds"]]
    scan_rows = []
    for case_index, case in enumerate(base["systems"]):
        for scale in controls["direct_phase_initialization_scales"]:
            for arm_id in PHYSICAL_ARMS:
                for seed in seeds:
                    scan_rows.append(
                        _evaluate(
                            base=base,
                            case=case,
                            case_index=case_index,
                            arm_id=arm_id,
                            seed=seed,
                            sample_count=int(controls["scan_sample_count"]),
                            direct_phase_scale=float(scale),
                            capacity=capacity,
                            tile_size=tile,
                            gradient=False,
                        )
                    )
    selected_scale = float(controls["selected_direct_phase_initialization_scale"])
    selected_rows = []
    for case_index, case in enumerate(base["systems"]):
        for arm_id in SCALE_FIRST_ARM_IDS:
            for seed in seeds:
                selected_rows.append(
                    _evaluate(
                        base=base,
                        case=case,
                        case_index=case_index,
                        arm_id=arm_id,
                        seed=seed,
                        sample_count=int(controls["selected_sample_count"]),
                        direct_phase_scale=selected_scale,
                        capacity=capacity,
                        tile_size=tile,
                        gradient=True,
                    )
                )
    selected_summary = _aggregate(selected_rows)
    gates = audit["gates"]
    checks = []
    for case in base["systems"]:
        case_id = str(case["case_id"])
        by_arm = {
            row["arm_id"]: row
            for row in selected_summary
            if row["case_id"] == case_id
        }
        generic = by_arm[SYMMETRY_UNAWARE_ARM]
        fixed = by_arm[FIXED_SZ_ARM]
        ratio = float(generic["variance_mean"] / fixed["variance_mean"])
        checks.append(
            {
                "case_id": case_id,
                "generic_variance_nontrivial": bool(
                    generic["variance_minimum"]
                    >= float(gates["minimum_generic_variance"])
                ),
                "generic_fixed_energy_gap": float(
                    abs(generic["energy_mean"] - fixed["energy_mean"])
                ),
                "generic_fixed_variance_ratio": ratio,
                "physical_controls_comparable": bool(
                    abs(generic["energy_mean"] - fixed["energy_mean"])
                    <= float(gates["maximum_generic_fixed_energy_gap"])
                    and ratio >= float(gates["minimum_generic_fixed_variance_ratio"])
                    and ratio <= float(gates["maximum_generic_fixed_variance_ratio"])
                ),
            }
        )
    all_counts = {row["parameter_count"] for row in selected_rows}
    passed = bool(
        len(all_counts) == 1
        and all(row["overflow_count"] == 0 for row in selected_rows)
        and all(np.isfinite(row["energy"]) for row in selected_rows)
        and all(np.isfinite(row["variance"]) for row in selected_rows)
        and all(
            row["gradient_norm"] >= float(gates["minimum_gradient_norm"])
            for row in selected_rows
        )
        and all(row["generic_variance_nontrivial"] for row in checks)
        and all(row["physical_controls_comparable"] for row in checks)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(audit),
        "source_config_sha256": canonical_sha256(base),
        "selected_direct_phase_initialization_scale": selected_scale,
        "development_scan": _aggregate(scan_rows),
        "selected_protocol_audit": selected_summary,
        "gate_checks": checks,
        "matched_parameter_count": next(iter(all_counts)) if len(all_counts) == 1 else None,
        "passed": passed,
        "production_seeds_untouched": True,
        "physics_interpretation_forbidden": True,
        "trained_energy_claim_allowed": False,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Initialization-development audit only. Energies select a finite-variance, "
            "nonstationary common protocol and cannot support a variational claim."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["TASK", "run_spin_scale_first_initialization_audit"]
