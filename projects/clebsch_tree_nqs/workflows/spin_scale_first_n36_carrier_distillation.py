"""Direct N=36 initialization from an analytically sampled SU(2) carrier."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.analytic_plaquette_carrier import (
    sample_aligned_plaquette_a1_physical,
    sample_aligned_plaquette_a1_schur,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.models.fixed_sz_autoregressive_spatial_vit import (
    fixed_sz_autoregressive_spatial_log_amplitudes,
)
from su2_nqs.models.many_cell_schur_vit import many_cell_schur_log_amplitudes
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import (
    evaluate_log_amplitudes_in_batches,
    train_sampled_carrier_pool_distillation,
)
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

TASK = "spin_scale_first_n36_carrier_distillation"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_contract(config_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    spec = json.loads(config_path.read_text())
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 carrier-distillation identity")
    base = json.loads(_resolve(config_path, spec["source_preflight"]).read_text())
    _validate(base)
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    for name in (
        "pool_size",
        "held_out_size",
        "steps",
        "batch_size",
        "audit_batch_size",
        "record_every",
        "native_sample_count",
        "sector_audit_sample_count",
        "su2_connected_capacity",
        "su2_amplitude_tile_size",
    ):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")
    return spec, base


def _teacher_pool(
    arm_id: str,
    *,
    geometry: str,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    layout = rectangular_four_spin_layout(
        6, 6, triangular_coordinates=geometry == "triangular"
    )
    if arm_id == EXACT_SU2_ARM:
        return sample_aligned_plaquette_a1_schur(
            layout, sample_count=count, seed=seed
        )
    return sample_aligned_plaquette_a1_physical(
        layout, sample_count=count, seed=seed
    )


def run_spin_scale_first_n36_carrier_distillation(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Fit all three N=36 arms to the same analytic carrier and audit H."""

    spec, base = _load_contract(config_path)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    controls = spec["controls"]
    gates = spec["gates"]
    shared_raw = dict(base["shared_spatial_backbone"])
    shared_raw["direct_phase_initialization_scale"] = float(
        controls["direct_phase_initialization_scale"]
    )
    shared = _shared_kwargs(shared_raw)
    phase_convention = "marshall_checkerboard"
    rows = []
    for case_index, case in enumerate(base["systems"]):
        case_id = str(case["case_id"])
        geometry = str(case["geometry"])
        initial_hashes = []
        for arm_index, arm_id in enumerate(SCALE_FIRST_ARM_IDS):
            runtime = build_scale_first_arm_runtime(
                arm_id,
                geometry=geometry,
                rows=6,
                columns=6,
                bonds=_bonds(geometry, float(case["j2_over_j1"])),
                shared_model=shared,
                initialization_key=jax.random.PRNGKey(
                    int(controls["initialization_seed"]) + case_index
                ),
                su2_connected_capacity=int(controls["su2_connected_capacity"]),
                su2_amplitude_tile_size=int(controls["su2_amplitude_tile_size"]),
                physical_phase_convention=phase_convention,
            )
            initial_hashes.append(pytree_sha256(runtime.params))
            pool, pool_phases = _teacher_pool(
                arm_id,
                geometry=geometry,
                count=int(controls["pool_size"]),
                seed=int(controls["teacher_seed"]) + 100 * case_index,
            )
            if arm_id == EXACT_SU2_ARM:
                log_fn = lambda candidate, batch, model=runtime.model_config: (
                    many_cell_schur_log_amplitudes(candidate, batch, model)
                )
            else:
                log_fn = lambda candidate, batch, model=runtime.model_config: (
                    fixed_sz_autoregressive_spatial_log_amplitudes(
                        candidate, batch, model
                    )
                )
            trained, history = train_sampled_carrier_pool_distillation(
                runtime.params,
                sample_pool=pool,
                target_phases=pool_phases,
                log_amplitude_fn=log_fn,
                seed=int(controls["distillation_seed"]) + 100 * case_index,
                steps=int(controls["steps"]),
                batch_size=int(controls["batch_size"]),
                learning_rate=float(controls["learning_rate"]),
                phase_weight=float(controls["phase_weight"]),
                record_every=int(controls["record_every"]),
            )
            held_out, held_out_phases = _teacher_pool(
                arm_id,
                geometry=geometry,
                count=int(controls["held_out_size"]),
                seed=int(controls["held_out_seed"]) + 100 * case_index,
            )
            held_out_logs = evaluate_log_amplitudes_in_batches(
                trained,
                held_out,
                log_fn,
                batch_size=int(controls["audit_batch_size"]),
            )
            phase_coherence = float(
                abs(np.mean(np.exp(1.0j * (held_out_logs.imag - held_out_phases))))
            )
            held_out_cross_entropy = float(np.mean(-2.0 * held_out_logs.real))
            trained_runtime = replace(runtime, params=trained)
            native_samples = trained_runtime.sample_fn(
                trained,
                jax.random.PRNGKey(
                    int(controls["native_seed"]) + 100 * case_index + arm_index
                ),
                int(controls["native_sample_count"]),
            )
            jax.block_until_ready(native_samples)
            local = trained_runtime.local_energy_fn(trained, native_samples)
            energies = np.asarray(local.local_energies, dtype=np.complex128)
            mean = np.mean(energies)
            variance = float(np.mean(np.abs(energies - mean) ** 2))
            if arm_id == EXACT_SU2_ARM:
                sector_weight = 1.0
            else:
                sector_samples = trained_runtime.sample_fn(
                    trained,
                    jax.random.PRNGKey(
                        int(controls["sector_seed"])
                        + 100 * case_index
                        + arm_index
                    ),
                    int(controls["sector_audit_sample_count"]),
                )
                concrete = np.asarray(jax.device_get(sector_samples))
                sector_weight = float(np.mean(np.sum(concrete, axis=1) == 18))
            energy_per_site = float(mean.real / 36.0)
            variance_per_site = variance / 36.0
            sector_gate_applies = arm_id != SYMMETRY_UNAWARE_ARM
            passed = bool(
                phase_coherence >= float(gates["minimum_phase_coherence"])
                and (
                    not sector_gate_applies
                    or sector_weight >= float(gates["minimum_m0_sector_weight"])
                )
                and local.overflow_count == 0
                and energy_per_site <= float(gates["maximum_energy_per_site"])
                and variance_per_site <= float(gates["maximum_variance_per_site"])
                and abs(float(mean.imag))
                <= float(gates["maximum_energy_imaginary_abs"])
            )
            identity = {
                "task": TASK,
                "config_sha256": canonical_sha256(spec),
                "source_preflight_sha256": canonical_sha256(base),
                "case_id": case_id,
                "geometry": geometry,
                "arm_id": arm_id,
                "carrier": "aligned_2x2_local_A1_product",
                "physical_phase_convention": phase_convention,
                "model": shared_raw,
            }
            manifest_path = output / "artifacts" / f"{geometry}_{arm_id}.json"
            manifest = save_parameter_artifact(
                manifest_path, trained, identity=identity, overwrite=overwrite
            )
            rows.append(
                {
                    "case_id": case_id,
                    "geometry": geometry,
                    "arm_id": arm_id,
                    "initial_params_sha256": initial_hashes[-1],
                    "trained_params_sha256": pytree_sha256(trained),
                    "parameter_artifact": str(manifest_path.relative_to(output)),
                    "parameter_artifact_manifest_sha256": manifest[
                        "manifest_sha256"
                    ],
                    "training_history": history,
                    "held_out_carrier_cross_entropy": held_out_cross_entropy,
                    "held_out_phase_coherence": phase_coherence,
                    "native_m0_sector_weight": sector_weight,
                    "m0_sector_weight_is_a_gate": sector_gate_applies,
                    "native_sample_count": int(controls["native_sample_count"]),
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
        "carrier": {
            "state": "aligned tensor product of 2x2 local-A1 singlets",
            "su2_sampling": "cell q=0,a in {0,1}; every intercell q=0",
            "physical_sampling": "IID exact local A1 Born samples",
            "physical_phase_convention": phase_convention,
            "hamiltonian_calls_during_distillation": 0,
        },
        "shared_initial_parameter_arrays_within_geometry": all(
            len(
                {
                    row["initial_params_sha256"]
                    for row in rows
                    if row["case_id"] == case["case_id"]
                }
            )
            == 1
            for case in base["systems"]
        ),
        "runs": rows,
        "all_n36_carrier_gates_passed": passed,
        "remote_single_seed_training_allowed": passed,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["TASK", "_load_contract", "run_spin_scale_first_n36_carrier_distillation"]
