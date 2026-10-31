"""Exact N=16 gate for compressing a finite imaginary-time SU(2) step.

This workflow starts from the sealed balanced-Schur Spatial-ViT carrier, acts
once with the complete Hamiltonian in the exhaustive singlet basis, and then
fits the resulting finite imaginary-time ray back into the *same* normalized
autoregressive model.  It therefore tests a new projection/training mechanism
without changing the carrier, representation, or parameter count.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import (
    load_parameter_artifact,
    save_parameter_artifact,
)
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    many_cell_schur_log_psi,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.distillation import train_exact_fidelity_distillation
from su2_nqs.vmc.imaginary_time_projection import (
    exact_state_distance,
    imaginary_time_euler_target,
    select_imaginary_time_euler_step,
)
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity
from su2_nqs.vmc.scale_first_arms import EXACT_SU2_ARM

from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _validate as _validate_preflight


jax.config.update("jax_enable_x64", True)

TASK = "spin_n16_imaginary_time_compression"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 imaginary-time compression identity")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    for name in ("steps", "record_every", "audit_batch_size"):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")
    for name in (
        "learning_rate",
        "maximum_tau",
        "maximum_fubini_study_distance",
        "maximum_symmetric_kl",
        "minimum_importance_effective_sample_fraction",
    ):
        value = float(controls.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"controls.{name} must be positive and finite")
    if float(controls["minimum_importance_effective_sample_fraction"]) > 1.0:
        raise ValueError("minimum importance ESS fraction cannot exceed one")
    if controls.get("loss") not in {"fidelity", "hamiltonian_graph_fidelity"}:
        raise ValueError("controls.loss must select a registered exact loss")
    for name in (
        "minimum_target_energy_gain_per_site",
        "minimum_student_energy_recovery_fraction",
        "maximum_student_variance_change_per_site",
        "maximum_student_target_infidelity",
        "maximum_student_fubini_study_distance",
        "maximum_student_symmetric_kl",
    ):
        value = float(gates.get(name, np.nan))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"gates.{name} must be nonnegative and finite")


def _bonds(geometry: str, j2: float):
    if geometry == "triangular":
        return triangular_hnf_j1j2_bonds(((4, 0), (0, 4)), j1=1.0, j2=j2)
    if geometry == "square":
        return square_j1j2_bonds(4, 4, j1=1.0, j2=j2, periodic=True)
    raise ValueError(f"unsupported geometry: {geometry!r}")


def _dense_matrix(workload) -> np.ndarray:
    labels = np.asarray(workload.center_labels, dtype=np.int32)
    positions = {tuple(map(int, row)): index for index, row in enumerate(labels)}
    if len(positions) != len(labels):
        raise RuntimeError("exhaustive labels are not unique")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for old in range(len(labels)):
        for slot in range(int(workload.widths[old])):
            key = tuple(map(int, workload.connected_labels[old, slot]))
            if key not in positions:
                raise RuntimeError("Hamiltonian connectivity left the singlet basis")
            matrix[positions[key], old] += float(workload.coefficients[old, slot])
    hermiticity_error = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    if hermiticity_error > 3.0e-12:
        raise RuntimeError(
            f"exact reduced Hamiltonian is not Hermitian: {hermiticity_error}"
        )
    return matrix


def _state_metrics(state: np.ndarray, matrix: np.ndarray) -> dict[str, float]:
    vector = np.asarray(state, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    action = matrix @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return {
        "normalization": float(np.vdot(vector, vector).real),
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "variance_total": float(np.vdot(residual, residual).real),
        "variance_per_site": float(np.vdot(residual, residual).real / 16.0),
    }


def _distance_payload(distance) -> dict[str, float]:
    return {name: float(value) for name, value in distance.__dict__.items()}


def _source_artifact(
    source: dict[str, object], geometry: str, source_root: Path
) -> Path:
    for case in source["cases"]:
        if str(case["geometry"]) != geometry:
            continue
        for arm in case["arms"]:
            if str(arm["arm_id"]) == EXACT_SU2_ARM:
                return source_root / str(arm["parameter_artifact"])
    raise KeyError(f"missing exact-SU2 source artifact for {geometry}")


def _run_case(
    case: dict[str, object],
    *,
    shared: dict[str, object],
    source: dict[str, object],
    source_root: Path,
    controls: dict[str, object],
    gates: dict[str, object],
    output: Path,
    case_index: int,
) -> dict[str, object]:
    geometry = str(case["geometry"])
    j2 = float(case["j2_over_j1"])
    layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=geometry == "triangular"
    )
    config = ManyCellSchurViTConfig(
        layout, target_q=0, geometry=geometry, **shared
    )
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)],
        dtype=np.int32,
    )
    template = init_many_cell_schur_vit(
        jax.random.PRNGKey(2026092500 + case_index), config
    )
    source_artifact = _source_artifact(source, geometry, source_root)
    initial_params = load_parameter_artifact(source_artifact, template)
    initial_logs = evaluate_log_amplitudes_in_batches(
        initial_params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, config
        ),
        batch_size=int(controls["audit_batch_size"]),
    )
    initial = np.exp(initial_logs)
    initial_native_norm = float(np.vdot(initial, initial).real)
    initial = initial / math.sqrt(initial_native_norm)

    workload = build_many_cell_tree_connectivity(labels, config, _bonds(geometry, j2))
    matrix = _dense_matrix(workload)
    initial_metrics = _state_metrics(initial, matrix)
    initial_action = matrix @ initial
    selected = select_imaginary_time_euler_step(
        initial,
        initial_action,
        maximum_tau=float(controls["maximum_tau"]),
        maximum_fubini_study_distance=float(
            controls["maximum_fubini_study_distance"]
        ),
        maximum_symmetric_kl=float(controls["maximum_symmetric_kl"]),
        minimum_importance_effective_sample_fraction=float(
            controls["minimum_importance_effective_sample_fraction"]
        ),
    )
    target = selected.target
    target_metrics = _state_metrics(target, matrix)
    variance_backtracks = 0
    while (
        target_metrics["variance_total"] > initial_metrics["variance_total"]
        and variance_backtracks < int(controls.get("maximum_variance_backtracks", 0))
    ):
        target, _ = imaginary_time_euler_target(
            initial,
            initial_action,
            0.5 * selected.tau,
            energy_shift=selected.energy_shift,
        )
        selected = select_imaginary_time_euler_step(
            initial,
            initial_action,
            maximum_tau=0.5 * selected.tau,
            maximum_fubini_study_distance=float(
                controls["maximum_fubini_study_distance"]
            ),
            maximum_symmetric_kl=float(controls["maximum_symmetric_kl"]),
            minimum_importance_effective_sample_fraction=float(
                controls["minimum_importance_effective_sample_fraction"]
            ),
        )
        target = selected.target
        target_metrics = _state_metrics(target, matrix)
        variance_backtracks += 1

    graph_matrix = None
    graph_scale = None
    if controls["loss"] == "hamiltonian_graph_fidelity":
        graph_matrix = jnp.asarray(matrix)
        graph_scale = max(math.sqrt(target_metrics["variance_total"]), 1.0e-8)
    trained, history = train_exact_fidelity_distillation(
        initial_params,
        basis=jnp.asarray(labels),
        target=jnp.asarray(target),
        log_psi_fn=lambda candidate, row: many_cell_schur_log_psi(
            candidate, row, config
        ),
        steps=int(controls["steps"]),
        learning_rate=float(controls["learning_rate"]),
        record_every=int(controls["record_every"]),
        hamiltonian=graph_matrix,
        graph_scale=graph_scale,
    )
    trained_logs = evaluate_log_amplitudes_in_batches(
        trained,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, config
        ),
        batch_size=int(controls["audit_batch_size"]),
    )
    student = np.exp(trained_logs)
    student_native_norm = float(np.vdot(student, student).real)
    student = student / math.sqrt(student_native_norm)
    student_metrics = _state_metrics(student, matrix)
    initial_to_target = exact_state_distance(initial, target)
    initial_to_student = exact_state_distance(initial, student)
    student_to_target = exact_state_distance(student, target)
    target_gain = initial_metrics["energy_per_site"] - target_metrics["energy_per_site"]
    student_gain = initial_metrics["energy_per_site"] - student_metrics["energy_per_site"]
    recovery = student_gain / max(target_gain, np.finfo(float).tiny)
    student_variance_change = (
        student_metrics["variance_per_site"] - initial_metrics["variance_per_site"]
    )
    passed = bool(
        target_gain >= float(gates["minimum_target_energy_gain_per_site"])
        and target_metrics["variance_per_site"] <= initial_metrics["variance_per_site"]
        and recovery >= float(gates["minimum_student_energy_recovery_fraction"])
        and student_variance_change
        <= float(gates["maximum_student_variance_change_per_site"])
        and 1.0 - student_to_target.normalized_overlap**2
        <= float(gates["maximum_student_target_infidelity"])
        and initial_to_student.fubini_study_distance
        <= float(gates["maximum_student_fubini_study_distance"])
        and initial_to_student.symmetric_kl
        <= float(gates["maximum_student_symmetric_kl"])
        and abs(student_native_norm - 1.0) <= 1.0e-9
    )
    artifact_path = output / "artifacts" / f"{geometry}_exact_su2_it_compressed.json"
    manifest = save_parameter_artifact(
        artifact_path,
        trained,
        identity={
            "task": TASK,
            "case_id": str(case["case_id"]),
            "geometry": geometry,
            "j2_over_j1": j2,
            "source_parameter_artifact": str(source_artifact),
            "source_params_sha256": pytree_sha256(initial_params),
            "trained_params_sha256": pytree_sha256(trained),
            "tau": float(selected.tau),
            "loss": str(controls["loss"]),
        },
    )
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "hamiltonian_connectivity_maximum_width": int(workload.maximum_width),
        "hamiltonian_hermiticity_max_abs_error": float(
            np.max(np.abs(matrix - matrix.T), initial=0.0)
        ),
        "source_parameter_artifact": str(source_artifact),
        "source_params_sha256": pytree_sha256(initial_params),
        "trained_params_sha256": pytree_sha256(trained),
        "trained_parameter_artifact": str(artifact_path.relative_to(output)),
        "trained_parameter_artifact_manifest_sha256": manifest["manifest_sha256"],
        "initial_native_normalization": initial_native_norm,
        "student_native_normalization": student_native_norm,
        "selected_tau": float(selected.tau),
        "variance_backtracks": variance_backtracks,
        "initial": initial_metrics,
        "uncompressed_target": target_metrics,
        "compressed_student": student_metrics,
        "initial_to_target_distance": _distance_payload(initial_to_target),
        "initial_to_student_distance": _distance_payload(initial_to_student),
        "student_to_target_distance": _distance_payload(student_to_target),
        "target_energy_gain_per_site": target_gain,
        "student_energy_gain_per_site": student_gain,
        "student_energy_recovery_fraction": recovery,
        "student_variance_change_per_site": student_variance_change,
        "graph_scale": graph_scale,
        "training_history": history,
        "full_h_actions": 1,
        "passed": passed,
    }


def run_spin_n16_imaginary_time_compression(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the exhaustive triangular and square compression gates."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    preflight_path = _resolve(config_path, spec["source_preflight"])
    preflight = json.loads(preflight_path.read_text())
    _validate_preflight(preflight)
    source_path = _resolve(config_path, spec["source_initialization_result"])
    source = json.loads(source_path.read_text())
    if (
        source.get("task") != "spin_scale_first_carrier_distillation"
        or "cases" not in source
    ):
        raise ValueError("invalid source initialization result")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    shared_raw = dict(preflight["shared_spatial_backbone"])
    shared_raw["direct_phase_initialization_scale"] = 0.0
    shared = _shared_kwargs(shared_raw)
    controls = dict(spec["controls"])
    gates = dict(spec["gates"])
    cases = [
        _run_case(
            case,
            shared=shared,
            source=source,
            source_root=source_path.parent,
            controls=controls,
            gates=gates,
            output=output,
            case_index=index,
        )
        for index, case in enumerate(preflight["systems"])
    ]
    passed = bool(all(case["passed"] for case in cases))
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_preflight_sha256": canonical_sha256(preflight),
        "source_initialization_result_sha256": source["result_sha256"],
        "mechanism": {
            "physical_step": "normalized [1-tau(H-<H>)] psi",
            "representation": "exact-S0 balanced four-cell Schur tree",
            "compression_model": "same normalized autoregressive Spatial-ViT",
            "changed_carrier_or_parameter_count": False,
        },
        "cases": cases,
        "all_n16_compression_gates_passed": passed,
        "n36_untouched_canary_allowed": passed,
        "remote_training_allowed": False,
        "decision": (
            "run_bounded_local_n36_untouched_imaginary_time_canary"
            if passed
            else "freeze_imaginary_time_compression_route"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Exact N=16 representation-and-compression gate only. Passing does "
            "not establish an N=36 energy, efficiency, or PRL-level win."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_imaginary_time_compression"]
