"""Encode one exact-small plaquette-RVB carrier in all three backbones.

The carrier is a physical wavefunction, not a shared random parameter tree.
It is built once in the fixed-``M=0`` basis, converted exactly to the balanced
Schur singlet basis, and distilled independently into the three normalized
autoregressive representations without Hamiltonian calls.  Exhaustive N=16
audits then compare physical fidelity, sector weight, energy, and variance.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.four_cell_schur import four_cell_schur_basis_matrix
from su2_nqs.benchmarks.four_cell_schur_frontier import SQUARE_N16_SUPPORTS
from su2_nqs.benchmarks.local_carrier_product_subspace import (
    coherent_square_plaquette_product,
    project_columns_by_index_table,
    square_a1_b1_product_columns,
)
from su2_nqs.benchmarks.many_cell_replay import (
    balanced_tree_to_rotated_coefficients,
    rotated_to_balanced_tree_coefficients,
)
from su2_nqs.benchmarks.triangular_state_manifold import (
    TRIANGULAR_N16_RHOMBUS_SUPPORTS,
)
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.fixed_sz_autoregressive_spatial_vit import (
    FixedSzAutoregressiveSpatialViTConfig,
    fixed_sz_autoregressive_spatial_log_amplitudes,
    init_fixed_sz_autoregressive_spatial_vit,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.operators.computational_graph_sparse import (
    heisenberg_computational_graph_sparse,
)
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import (
    evaluate_log_amplitudes_in_batches,
    normalized_carrier_audit,
    train_normalized_carrier_distillation,
)
from su2_nqs.vmc.scale_first_arms import (
    EXACT_SU2_ARM,
    FIXED_SZ_ARM,
    SCALE_FIRST_ARM_IDS,
    SYMMETRY_UNAWARE_ARM,
)

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _validate


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_carrier_distillation"


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
        raise ValueError("invalid carrier-distillation identity")
    base = json.loads(_resolve(config_path, spec.get("source_preflight")).read_text())
    _validate(base)
    carrier = json.loads(
        _resolve(config_path, spec.get("source_carrier_gate_result")).read_text()
    )
    if (
        carrier.get("task") != "spin_scale_first_plaquette_carrier_gate"
        or carrier.get("all_carrier_gates_passed") is not True
    ):
        raise ValueError("the source plaquette-carrier gate has not passed")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    for name in ("steps", "batch_size", "audit_batch_size", "record_every"):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")
    for name in ("learning_rate", "phase_weight"):
        value = float(controls.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"controls.{name} must be positive and finite")
    if not isinstance(controls.get("exact_teacher_distribution", False), bool):
        raise TypeError("controls.exact_teacher_distribution must be boolean")
    if controls.get("angle_policy") != "local_A1_theta_zero":
        raise ValueError("only the size-transferable local-A1 angle policy is admitted")
    if controls.get("carrier_policy", "projected_gamma_a1") not in {
        "projected_gamma_a1",
        "aligned_local_a1_no_reynolds",
    }:
        raise ValueError("unsupported controls.carrier_policy")
    if controls.get("physical_phase_convention", "none") not in {
        "none",
        "marshall_checkerboard",
    }:
        raise ValueError("unsupported controls.physical_phase_convention")
    return spec, base, carrier


def _state_metrics(state: np.ndarray, hamiltonian) -> dict[str, float]:
    vector = np.asarray(state, dtype=np.complex128)
    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have positive finite norm")
    vector = vector / np.sqrt(norm)
    action = np.asarray(hamiltonian @ vector)
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return {
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "variance_total": float(np.vdot(residual, residual).real),
        "variance_per_site": float(np.vdot(residual, residual).real / 16.0),
    }


def _n16_bonds(geometry: str, j2: float):
    if geometry == "triangular":
        return triangular_hnf_j1j2_bonds(
            ((4, 0), (0, 4)), j1=1.0, j2=float(j2)
        )
    if geometry == "square":
        return square_j1j2_bonds(4, 4, j1=1.0, j2=float(j2), periodic=True)
    raise ValueError(f"unsupported geometry: {geometry!r}")


def _target_data(
    geometry: str,
    *,
    triangular_cache: Path,
    project_gamma_a1: bool = True,
) -> dict[str, object]:
    fixed_basis = fixed_m_spin_basis(16, 0)
    supports = (
        SQUARE_N16_SUPPORTS
        if geometry == "square"
        else TRIANGULAR_N16_RHOMBUS_SUPPORTS
    )
    columns, labels = square_a1_b1_product_columns(fixed_basis, supports)
    unprojected = coherent_square_plaquette_product(columns, labels, 0.0)
    if not project_gamma_a1:
        physical = unprojected
    elif geometry == "square":
        from su2_nqs.lattices.spatial_symmetry import square_gamma_a1_group
        from su2_nqs.benchmarks.clebsch_tree_exact import (
            physical_permutation_index_table,
        )

        indices = physical_permutation_index_table(
            fixed_basis, square_gamma_a1_group(4)
        )
        projected_columns = project_columns_by_index_table(columns, indices)
        physical = coherent_square_plaquette_product(
            projected_columns, labels, 0.0
        )
    elif geometry == "triangular":
        with np.load(triangular_cache) as cache:
            phi = np.asarray(cache["physical_a1_basis"], dtype=np.float64)
        physical = phi @ (phi.T @ unprojected)
        physical = physical / np.linalg.norm(physical)
    else:
        raise ValueError(f"unsupported geometry: {geometry!r}")

    schur_basis, schur_configurations = four_cell_schur_basis_matrix(
        "square_plaquette", supports=supports
    )
    if schur_configurations != fixed_basis:
        raise RuntimeError("Schur and fixed-M physical bases disagree")
    rotated = np.asarray(schur_basis.T.conj() @ physical)
    layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=geometry == "triangular"
    )
    balanced, balanced_labels = rotated_to_balanced_tree_coefficients(
        rotated, layout
    )
    reconstructed = schur_basis @ balanced_tree_to_rotated_coefficients(
        balanced, layout
    )
    reconstruction_error = float(np.linalg.norm(reconstructed - physical))
    if reconstruction_error > 3.0e-12:
        raise RuntimeError("physical carrier failed exact balanced-Schur conversion")
    return {
        "fixed_basis": np.asarray(fixed_basis, dtype=np.int32),
        "physical": np.asarray(physical, dtype=np.complex128),
        "balanced_labels": np.asarray(balanced_labels, dtype=np.int32),
        "balanced": np.asarray(balanced, dtype=np.complex128),
        "schur_basis": schur_basis,
        "layout": layout,
        "reconstruction_error": reconstruction_error,
    }


def _student_physical_state(
    arm_id: str,
    params: object,
    *,
    geometry: str,
    target: dict[str, object],
    shared: dict[str, object],
    physical_phase_convention: str,
    audit_batch_size: int,
) -> tuple[np.ndarray, float, dict[str, float], np.ndarray | None]:
    if arm_id == EXACT_SU2_ARM:
        config = ManyCellSchurViTConfig(
            target["layout"], target_q=0, geometry=geometry, **shared
        )
        logs = evaluate_log_amplitudes_in_batches(
            params,
            target["balanced_labels"],
            lambda candidate, rows: many_cell_schur_log_amplitudes(
                candidate, rows, config
            ),
            batch_size=audit_batch_size,
        )
        coefficients = np.exp(logs)
        native_norm = float(np.vdot(coefficients, coefficients).real)
        audit = normalized_carrier_audit(
            coefficients, target["balanced"], native_norm=native_norm
        )
        rotated = balanced_tree_to_rotated_coefficients(
            coefficients, target["layout"]
        )
        physical = np.asarray(target["schur_basis"] @ rotated)
        return physical, native_norm, audit, coefficients

    target_m2 = 0 if arm_id == FIXED_SZ_ARM else None
    config = FixedSzAutoregressiveSpatialViTConfig(
        geometry,
        4,
        4,
        target_m2=target_m2,
        phase_convention=physical_phase_convention,
        **shared,
    )
    fixed_logs = evaluate_log_amplitudes_in_batches(
        params,
        target["fixed_basis"],
        lambda candidate, rows: fixed_sz_autoregressive_spatial_log_amplitudes(
            candidate, rows, config
        ),
        batch_size=audit_batch_size,
    )
    fixed_amplitudes = np.exp(fixed_logs)
    full_amplitudes = None
    if arm_id == FIXED_SZ_ARM:
        native_norm = float(np.vdot(fixed_amplitudes, fixed_amplitudes).real)
    else:
        integers = np.arange(1 << 16, dtype=np.uint32)
        full_basis = (
            (integers[:, None] >> np.arange(15, -1, -1, dtype=np.uint32)) & 1
        ).astype(np.int32)
        full_logs = evaluate_log_amplitudes_in_batches(
            params,
            full_basis,
            lambda candidate, rows: fixed_sz_autoregressive_spatial_log_amplitudes(
                candidate, rows, config
            ),
            batch_size=audit_batch_size,
        )
        full_amplitudes = np.exp(full_logs)
        native_norm = float(np.vdot(full_amplitudes, full_amplitudes).real)
    audit = normalized_carrier_audit(
        fixed_amplitudes, target["physical"], native_norm=native_norm
    )
    return fixed_amplitudes, native_norm, audit, full_amplitudes


def run_spin_scale_first_carrier_distillation(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Distill and exhaustively audit the common N=16 physical carrier."""

    spec, base, source_carrier = _load_contract(config_path)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    controls = spec["controls"]
    gates = spec["gates"]
    triangular_cache = _resolve(config_path, spec["triangular_cache"])
    shared_raw = dict(base["shared_spatial_backbone"])
    shared_raw["direct_phase_initialization_scale"] = float(
        controls["direct_phase_initialization_scale"]
    )
    shared = _shared_kwargs(shared_raw)
    physical_phase_convention = str(
        controls.get("physical_phase_convention", "none")
    )
    carrier_policy = str(
        controls.get("carrier_policy", "projected_gamma_a1")
    )
    rows = []
    artifacts = []
    for case_index, case in enumerate(base["systems"]):
        geometry = str(case["geometry"])
        case_id = str(case["case_id"])
        target = _target_data(
            geometry,
            triangular_cache=triangular_cache,
            project_gamma_a1=carrier_policy == "projected_gamma_a1",
        )
        bonds = _n16_bonds(geometry, float(case["j2_over_j1"]))
        fixed_h, fixed_h_basis = heisenberg_sz_graph_sparse(16, bonds, n_up=8)
        if tuple(fixed_h_basis) != tuple(map(tuple, target["fixed_basis"])):
            raise RuntimeError("fixed-M Hamiltonian basis ordering changed")
        target_metrics = _state_metrics(target["physical"], fixed_h)
        initial_hashes = []
        case_rows = []
        for arm_id in SCALE_FIRST_ARM_IDS:
            initialization_key = jax.random.PRNGKey(
                int(controls["initialization_seed"]) + case_index
            )
            if arm_id == EXACT_SU2_ARM:
                model = ManyCellSchurViTConfig(
                    target["layout"], target_q=0, geometry=geometry, **shared
                )
                params = init_many_cell_schur_vit(initialization_key, model)
                training_basis = target["balanced_labels"]
                training_target = target["balanced"]
                log_fn = lambda candidate, batch, model=model: (
                    many_cell_schur_log_amplitudes(candidate, batch, model)
                )
            else:
                model = FixedSzAutoregressiveSpatialViTConfig(
                    geometry,
                    4,
                    4,
                    target_m2=0 if arm_id == FIXED_SZ_ARM else None,
                    phase_convention=physical_phase_convention,
                    **shared,
                )
                params = init_fixed_sz_autoregressive_spatial_vit(
                    initialization_key, model
                )
                training_basis = target["fixed_basis"]
                training_target = target["physical"]
                log_fn = lambda candidate, batch, model=model: (
                    fixed_sz_autoregressive_spatial_log_amplitudes(
                        candidate, batch, model
                    )
                )
            initial_hashes.append(pytree_sha256(params))
            trained, history = train_normalized_carrier_distillation(
                params,
                basis=training_basis,
                target=training_target,
                log_amplitude_fn=log_fn,
                seed=int(controls["distillation_seed"]) + 100 * case_index,
                steps=int(controls["steps"]),
                batch_size=int(controls["batch_size"]),
                learning_rate=float(controls["learning_rate"]),
                phase_weight=float(controls["phase_weight"]),
                record_every=int(controls["record_every"]),
                exact_teacher_distribution=bool(
                    controls.get("exact_teacher_distribution", False)
                ),
            )
            physical, native_norm, audit, full = _student_physical_state(
                arm_id,
                trained,
                geometry=geometry,
                target=target,
                shared=shared,
                physical_phase_convention=physical_phase_convention,
                audit_batch_size=int(controls["audit_batch_size"]),
            )
            if arm_id == SYMMETRY_UNAWARE_ARM:
                full_h, _ = heisenberg_computational_graph_sparse(16, bonds)
                metrics = _state_metrics(full, full_h)
            else:
                metrics = _state_metrics(physical, fixed_h)
            energy_degradation = (
                metrics["energy_per_site"] - target_metrics["energy_per_site"]
            )
            variance_degradation = (
                metrics["variance_per_site"] - target_metrics["variance_per_site"]
            )
            passed = bool(
                audit["fidelity"] >= float(gates["minimum_fidelity"])
                and abs(native_norm - 1.0)
                <= float(gates["maximum_normalization_error"])
                and audit["teacher_sector_weight"]
                >= float(gates["minimum_teacher_sector_weight"])
                and energy_degradation
                <= float(gates["maximum_energy_degradation_per_site"])
                and variance_degradation
                <= float(gates["maximum_variance_degradation_per_site"])
            )
            identity = {
                "task": TASK,
                "config_sha256": canonical_sha256(spec),
                "source_preflight_sha256": canonical_sha256(base),
                "source_carrier_result_sha256": source_carrier["result_sha256"],
                "case_id": case_id,
                "geometry": geometry,
                "arm_id": arm_id,
                "n16_teacher_sha256": array_sha256(target["physical"]),
                "angle_policy": controls["angle_policy"],
                "carrier_policy": carrier_policy,
                "physical_phase_convention": physical_phase_convention,
                "model": shared_raw,
            }
            manifest_path = output / "artifacts" / f"{geometry}_{arm_id}.json"
            manifest = save_parameter_artifact(
                manifest_path, trained, identity=identity, overwrite=overwrite
            )
            artifacts.append(str(manifest_path.relative_to(output)))
            case_rows.append(
                {
                    "arm_id": arm_id,
                    "initial_params_sha256": initial_hashes[-1],
                    "trained_params_sha256": pytree_sha256(trained),
                    "parameter_artifact_manifest_sha256": manifest["manifest_sha256"],
                    "parameter_artifact": str(manifest_path.relative_to(output)),
                    "training_history": history,
                    **audit,
                    **metrics,
                    "energy_degradation_per_site": energy_degradation,
                    "variance_degradation_per_site": variance_degradation,
                    "passed": passed,
                }
            )
        rows.append(
            {
                "case_id": case_id,
                "geometry": geometry,
                "target_reconstruction_l2_error": target["reconstruction_error"],
                "target_physical_sha256": array_sha256(target["physical"]),
                "target": target_metrics,
                "shared_initial_parameter_arrays": len(set(initial_hashes)) == 1,
                "arms": case_rows,
                "passed": all(row["passed"] for row in case_rows),
            }
        )
    passed = bool(rows and all(row["passed"] for row in rows))
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_preflight_sha256": canonical_sha256(base),
        "source_carrier_result_sha256": source_carrier["result_sha256"],
        "carrier_definition": {
            "physical_state": (
                "Gamma-A1 projection of tensor 2x2 local-A1 singlets"
                if carrier_policy == "projected_gamma_a1"
                else "aligned tensor product of 2x2 local-A1 singlets"
            ),
            "angle": 0.0,
            "angle_policy": controls["angle_policy"],
            "carrier_policy": carrier_policy,
            "physical_phase_convention": physical_phase_convention,
            "reason": (
                "size-transferable common carrier at the preregistered no-Reynolds lattice-symmetry level"
                if carrier_policy == "aligned_local_a1_no_reynolds"
                else "size-transferable local rule with terminal exact-small Gamma-A1 projection"
            ),
        },
        "cases": rows,
        "artifacts": artifacts,
        "all_representation_gates_passed": passed,
        "n36_single_seed_promotion_allowed": passed,
        "hamiltonian_calls_during_distillation": 0,
        "repeated_diagonalization_performed": False,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["TASK", "_load_contract", "run_spin_scale_first_carrier_distillation"]
