"""Matrix-free tangent audit of the N=16 imaginary-time target.

The preceding finite-fit gate can fail either because the target lies outside
the neural tangent or because a generic Adam fit does not resolve a small
physical displacement.  This workflow distinguishes those explanations with
one full-support, Born-weighted LSQR projection and exact energy acceptance.
It does not alter the ansatz and cannot authorize remote production.
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
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    many_cell_schur_log_psi,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.imaginary_time_projection import (
    exact_state_distance,
    select_imaginary_time_euler_step,
)
from su2_nqs.vmc.lsqr_projection import train_lsqr_projection
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity

from .spin_n16_imaginary_time_compression import (
    _bonds,
    _dense_matrix,
    _distance_payload,
    _source_artifact,
    _state_metrics,
)
from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _validate as _validate_preflight


jax.config.update("jax_enable_x64", True)

TASK = "spin_n16_imaginary_time_tangent"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 imaginary-time tangent identity")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    for name in ("lsqr_max_iterations", "backtracking_steps", "audit_batch_size"):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")
    for name in (
        "maximum_tau",
        "maximum_fubini_study_distance",
        "maximum_symmetric_kl",
        "minimum_importance_effective_sample_fraction",
        "damping",
        "step_size",
        "lsqr_tolerance",
    ):
        value = float(controls.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"controls.{name} must be positive and finite")
    for name in (
        "minimum_target_energy_gain_per_site",
        "minimum_student_energy_recovery_fraction",
        "minimum_feature_loss_recovery_fraction",
        "maximum_student_variance_change_per_site",
        "maximum_student_fubini_study_distance",
        "maximum_student_symmetric_kl",
    ):
        value = float(gates.get(name, np.nan))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"gates.{name} must be nonnegative and finite")


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
    model = ManyCellSchurViTConfig(
        layout, target_q=0, geometry=geometry, **shared
    )
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    template = init_many_cell_schur_vit(
        jax.random.PRNGKey(2026092510 + case_index), model
    )
    source_artifact = _source_artifact(source, geometry, source_root)
    initial_params = load_parameter_artifact(source_artifact, template)
    log_batch = jax.jit(
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, model
        )
    )
    initial_logs = evaluate_log_amplitudes_in_batches(
        initial_params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, model
        ),
        batch_size=int(controls["audit_batch_size"]),
    )
    initial = np.exp(initial_logs)
    initial_native_norm = float(np.vdot(initial, initial).real)
    initial /= math.sqrt(initial_native_norm)
    workload = build_many_cell_tree_connectivity(labels, model, _bonds(geometry, j2))
    matrix = _dense_matrix(workload)
    initial_metrics = _state_metrics(initial, matrix)
    selected = select_imaginary_time_euler_step(
        initial,
        matrix @ initial,
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
    target_log_ratios = np.log(target / initial)
    probabilities = np.abs(initial) ** 2

    def exact_energy(candidate: object) -> float:
        logs = np.asarray(
            jax.device_get(log_batch(candidate, jnp.asarray(labels))),
            dtype=np.complex128,
        )
        state = np.exp(logs)
        return float(_state_metrics(state, matrix)["energy_total"])

    trained, history = train_lsqr_projection(
        initial_params,
        jax.random.PRNGKey(int(controls["seed"]) + case_index),
        sample_pool=jnp.asarray(labels),
        pool_importance_weights=jnp.asarray(probabilities),
        pool_base_logs=jnp.asarray(initial_logs),
        pool_target_log_ratios=jnp.asarray(target_log_ratios),
        log_psi_fn=lambda candidate, row: many_cell_schur_log_psi(
            candidate, row, model
        ),
        steps=1,
        batch_size=len(labels),
        damping=float(controls["damping"]),
        step_size=float(controls["step_size"]),
        lsqr_tolerance=float(controls["lsqr_tolerance"]),
        lsqr_max_iterations=int(controls["lsqr_max_iterations"]),
        backtracking_steps=int(controls["backtracking_steps"]),
        acceptance_score_fn=exact_energy,
        random_batches=False,
    )
    trained_logs = evaluate_log_amplitudes_in_batches(
        trained,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, model
        ),
        batch_size=int(controls["audit_batch_size"]),
    )
    student = np.exp(trained_logs)
    student_native_norm = float(np.vdot(student, student).real)
    student /= math.sqrt(student_native_norm)
    student_metrics = _state_metrics(student, matrix)
    initial_to_target = exact_state_distance(initial, target)
    initial_to_student = exact_state_distance(initial, student)
    student_to_target = exact_state_distance(student, target)
    target_gain = initial_metrics["energy_per_site"] - target_metrics["energy_per_site"]
    student_gain = initial_metrics["energy_per_site"] - student_metrics["energy_per_site"]
    recovery = student_gain / max(target_gain, np.finfo(float).tiny)
    feature_before = float(history[-1]["feature_loss_before"])
    feature_after = float(history[-1]["feature_loss_after"])
    feature_recovery = 1.0 - feature_after / max(feature_before, np.finfo(float).tiny)
    variance_change = (
        student_metrics["variance_per_site"] - initial_metrics["variance_per_site"]
    )
    solver_converged = bool(history[-1]["lsqr_converged"])
    passed = bool(
        solver_converged
        and
        target_gain >= float(gates["minimum_target_energy_gain_per_site"])
        and target_metrics["variance_per_site"] <= initial_metrics["variance_per_site"]
        and recovery >= float(gates["minimum_student_energy_recovery_fraction"])
        and feature_recovery >= float(gates["minimum_feature_loss_recovery_fraction"])
        and variance_change <= float(gates["maximum_student_variance_change_per_site"])
        and initial_to_student.fubini_study_distance
        <= float(gates["maximum_student_fubini_study_distance"])
        and initial_to_student.symmetric_kl
        <= float(gates["maximum_student_symmetric_kl"])
        and abs(student_native_norm - 1.0) <= 1.0e-9
    )
    artifact_path = output / "artifacts" / f"{geometry}_exact_su2_lsqr_tangent.json"
    manifest = save_parameter_artifact(
        artifact_path,
        trained,
        identity={
            "task": TASK,
            "case_id": str(case["case_id"]),
            "geometry": geometry,
            "source_parameter_artifact": str(source_artifact),
            "source_params_sha256": pytree_sha256(initial_params),
            "trained_params_sha256": pytree_sha256(trained),
            "tau": float(selected.tau),
            "damping": float(controls["damping"]),
        },
    )
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "source_parameter_artifact": str(source_artifact),
        "trained_parameter_artifact": str(artifact_path.relative_to(output)),
        "trained_parameter_artifact_manifest_sha256": manifest["manifest_sha256"],
        "initial_native_normalization": initial_native_norm,
        "student_native_normalization": student_native_norm,
        "selected_tau": float(selected.tau),
        "initial": initial_metrics,
        "uncompressed_target": target_metrics,
        "tangent_student": student_metrics,
        "initial_to_target_distance": _distance_payload(initial_to_target),
        "initial_to_student_distance": _distance_payload(initial_to_student),
        "student_to_target_distance": _distance_payload(student_to_target),
        "target_energy_gain_per_site": target_gain,
        "student_energy_gain_per_site": student_gain,
        "student_energy_recovery_fraction": recovery,
        "student_variance_change_per_site": variance_change,
        "feature_loss_recovery_fraction": feature_recovery,
        "linear_solver_converged": solver_converged,
        "lsqr_history": history,
        "full_h_actions": 1,
        "passed": passed,
    }


def run_spin_n16_imaginary_time_tangent(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the exact full-support tangent audit on both registered systems."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    preflight_path = _resolve(config_path, spec["source_preflight"])
    preflight = json.loads(preflight_path.read_text())
    _validate_preflight(preflight)
    source_path = _resolve(config_path, spec["source_initialization_result"])
    source = json.loads(source_path.read_text())
    if source.get("task") != "spin_scale_first_carrier_distillation":
        raise ValueError("invalid source initialization result")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    shared_raw = dict(preflight["shared_spatial_backbone"])
    shared_raw["direct_phase_initialization_scale"] = 0.0
    shared = _shared_kwargs(shared_raw)
    cases = [
        _run_case(
            case,
            shared=shared,
            source=source,
            source_root=source_path.parent,
            controls=dict(spec["controls"]),
            gates=dict(spec["gates"]),
            output=output,
            case_index=index,
        )
        for index, case in enumerate(preflight["systems"])
    ]
    passed = bool(all(case["passed"] for case in cases))
    all_solvers_converged = bool(
        all(case["linear_solver_converged"] for case in cases)
    )
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
        "cases": cases,
        "all_n16_tangent_gates_passed": passed,
        "all_linear_solvers_converged": all_solvers_converged,
        "n36_untouched_canary_allowed": passed,
        "remote_training_allowed": False,
        "decision": (
            "run_bounded_local_n36_untouched_imaginary_time_canary"
            if passed
            else (
                "current_balanced_schur_vit_misses_imaginary_time_tangent"
                if all_solvers_converged
                else "inconclusive_lsqr_tangent_solve_not_converged"
            )
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Exact N=16 instantaneous tangent diagnostic. It separates optimizer "
            "failure from local neural-manifold capacity but is not a finite-size "
            "or variational-win claim."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_imaginary_time_tangent"]
