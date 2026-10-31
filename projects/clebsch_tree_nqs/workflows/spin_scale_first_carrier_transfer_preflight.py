"""Local N=36 transfer canary for the common low-energy carrier artifacts."""

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
from su2_nqs.vmc.scale_first_arms import (
    SCALE_FIRST_ARM_IDS,
    build_scale_first_arm_runtime,
)

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _bonds, _validate


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_carrier_transfer_preflight"


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
        raise ValueError("invalid carrier-transfer preflight identity")
    base = json.loads(_resolve(config_path, spec["source_preflight"]).read_text())
    _validate(base)
    distillation_path = _resolve(config_path, spec["source_distillation_result"])
    distillation = json.loads(distillation_path.read_text())
    if distillation.get("task") != "spin_scale_first_carrier_distillation":
        raise ValueError("source result is not a carrier-distillation artifact")
    if distillation.get("carrier_definition", {}).get("carrier_policy") != (
        "aligned_local_a1_no_reynolds"
    ):
        raise ValueError("only the aligned no-Reynolds carrier may transfer")
    controls = spec.get("controls")
    admission = spec.get("n16_physical_admission")
    gates = spec.get("n36_canary_gates")
    if not all(isinstance(value, dict) for value in (controls, admission, gates)):
        raise TypeError("controls and both gate sections must be objects")
    sample_count = controls.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("controls.sample_count must be an integer of at least two")
    return spec, base, distillation, distillation_path.parent


def _admit_n16(
    distillation: dict[str, object], admission: dict[str, object]
) -> tuple[bool, list[dict[str, object]]]:
    rows = []
    for case in distillation["cases"]:
        for arm in case["arms"]:
            phase_coherence = float(arm["training_history"][-1]["phase_coherence"])
            passed = bool(
                float(arm["energy_degradation_per_site"])
                <= float(admission["maximum_energy_degradation_per_site"])
                and float(arm["variance_degradation_per_site"])
                <= float(admission["maximum_variance_degradation_per_site"])
                and float(arm["teacher_sector_weight"])
                >= float(admission["minimum_teacher_sector_weight"])
                and phase_coherence >= float(admission["minimum_phase_coherence"])
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "arm_id": arm["arm_id"],
                    "energy_degradation_per_site": arm[
                        "energy_degradation_per_site"
                    ],
                    "variance_degradation_per_site": arm[
                        "variance_degradation_per_site"
                    ],
                    "teacher_sector_weight": arm["teacher_sector_weight"],
                    "phase_coherence": phase_coherence,
                    "fidelity_diagnostic_not_a_gate": arm["fidelity"],
                    "passed": passed,
                }
            )
    return bool(rows and all(row["passed"] for row in rows)), rows


def run_spin_scale_first_carrier_transfer_preflight(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Load N=16 artifacts into N=36 templates and run native-H canaries."""

    spec, base, distillation, artifact_root = _load_contract(config_path)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    n16_passed, n16_rows = _admit_n16(
        distillation, spec["n16_physical_admission"]
    )
    if not n16_passed:
        raise RuntimeError("N16 physical carrier admission failed; N36 remains locked")
    controls = spec["controls"]
    gates = spec["n36_canary_gates"]
    shared_raw = dict(base["shared_spatial_backbone"])
    shared_raw["direct_phase_initialization_scale"] = float(
        distillation["carrier_definition"].get(
            "direct_phase_initialization_scale", 0.0
        )
    )
    # The sealed distillation identity records the actual model dictionary.
    shared_raw["direct_phase_initialization_scale"] = 0.0
    shared = _shared_kwargs(shared_raw)
    phase_convention = str(
        distillation["carrier_definition"]["physical_phase_convention"]
    )
    source_cases = {str(case["case_id"]): case for case in distillation["cases"]}
    rows = []
    for case_index, case in enumerate(base["systems"]):
        case_id = str(case["case_id"])
        geometry = str(case["geometry"])
        source_case = source_cases[case_id]
        source_arms = {str(row["arm_id"]): row for row in source_case["arms"]}
        for arm_index, arm_id in enumerate(SCALE_FIRST_ARM_IDS):
            template_runtime = build_scale_first_arm_runtime(
                arm_id,
                geometry=geometry,
                rows=6,
                columns=6,
                bonds=_bonds(geometry, float(case["j2_over_j1"])),
                shared_model=shared,
                initialization_key=jax.random.PRNGKey(
                    int(controls["seed"]) + case_index
                ),
                su2_connected_capacity=int(controls["su2_connected_capacity"]),
                su2_amplitude_tile_size=int(controls["su2_amplitude_tile_size"]),
                physical_phase_convention=phase_convention,
            )
            manifest_path = artifact_root / str(
                source_arms[arm_id]["parameter_artifact"]
            )
            params = load_parameter_artifact(
                manifest_path, template_runtime.params
            )
            runtime = replace(template_runtime, params=params)
            samples = runtime.sample_fn(
                params,
                jax.random.PRNGKey(
                    int(controls["seed"]) + 100 * case_index + arm_index
                ),
                int(controls["sample_count"]),
            )
            jax.block_until_ready(samples)
            local = runtime.local_energy_fn(params, samples)
            energies = np.asarray(local.local_energies, dtype=np.complex128)
            mean = np.mean(energies)
            variance = float(np.mean(np.abs(energies - mean) ** 2))
            energy_per_site = float(mean.real / 36.0)
            variance_per_site = variance / 36.0
            passed = bool(
                np.all(np.isfinite(energies))
                and local.overflow_count == 0
                and abs(float(mean.imag))
                <= float(gates["maximum_energy_imaginary_abs"])
                and energy_per_site <= float(gates["maximum_energy_per_site"])
                and variance_per_site <= float(gates["maximum_variance_per_site"])
            )
            rows.append(
                {
                    "case_id": case_id,
                    "geometry": geometry,
                    "arm_id": arm_id,
                    "source_parameter_artifact": str(manifest_path),
                    "n16_params_sha256": source_arms[arm_id][
                        "trained_params_sha256"
                    ],
                    "n36_params_sha256": pytree_sha256(params),
                    "size_transfer_preserved_parameter_arrays": (
                        source_arms[arm_id]["trained_params_sha256"]
                        == pytree_sha256(params)
                    ),
                    "sample_count": int(controls["sample_count"]),
                    "energy_total": float(mean.real),
                    "energy_per_site": energy_per_site,
                    "energy_imaginary_abs": float(abs(mean.imag)),
                    "variance_total": variance,
                    "variance_per_site": variance_per_site,
                    "overflow_count": local.overflow_count,
                    "full_h_seconds": local.evaluation_seconds,
                    "passed": passed,
                }
            )
    passed = bool(len(rows) == 6 and all(row["passed"] for row in rows))
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_preflight_sha256": canonical_sha256(base),
        "source_distillation_result_sha256": distillation["result_sha256"],
        "n16_physical_admission_passed": n16_passed,
        "n16_physical_admission": n16_rows,
        "n36_canaries": rows,
        "all_n36_canaries_passed": passed,
        "n36_single_seed_remote_promotion_allowed": passed,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = [
    "TASK",
    "_load_contract",
    "run_spin_scale_first_carrier_transfer_preflight",
]
