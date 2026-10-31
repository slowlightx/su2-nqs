"""Exact tangent gate for a bond-causal many-cell Schur Spatial-ViT head."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from su2_nqs.benchmarks.energy_tangent_projection import (
    project_born_complex_tangent,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.models.hamiltonian_conditioned_tree import HamiltonianBond
from su2_nqs.models.many_cell_hamiltonian_conditioned_vit import (
    ManyCellHamiltonianConditionedViTConfig,
    condition_many_cell_schur_vit,
    many_cell_hamiltonian_conditioned_log_amplitudes,
    many_cell_hamiltonian_conditioned_log_psi,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.imaginary_time_projection import (
    exact_state_distance,
    select_imaginary_time_euler_step,
)
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

TASK = "spin_n16_bond_causal_tangent"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 bond-causal tangent identity")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    seeds = controls.get("scramble_seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("at least two integer scramble seeds are required")
    for name in (
        "local_embedding_dim",
        "local_hidden_dim",
        "backtracking_steps",
        "audit_batch_size",
        "tangent_batch_size",
    ):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")


def _conditioned_bonds(layout, physical_bonds, j2: float, permutation=None):
    if permutation is None:
        transformed = tuple(physical_bonds)
    else:
        transformed = tuple(
            (int(permutation[first]), int(permutation[second]), float(coupling))
            for first, second, coupling in physical_bonds
        )
    remapped = layout.remap_bonds(transformed)
    return tuple(
        HamiltonianBond(
            int(first),
            int(second),
            float(coupling),
            int(abs(float(coupling) - j2) < 1.0e-12 and abs(j2 - 1.0) > 1.0e-12),
        )
        for first, second, coupling in remapped
    )


def _fit_head(
    *,
    base_params,
    base_model,
    bonds,
    labels,
    initial_logs,
    initial_state,
    target,
    matrix,
    controls,
    seed: int,
) -> dict[str, object]:
    conditioned_model = ManyCellHamiltonianConditionedViTConfig(
        base_model,
        bonds,
        n_channels=2,
        local_embedding_dim=int(controls["local_embedding_dim"]),
        local_hidden_dim=int(controls["local_hidden_dim"]),
        train_local_encoder=False,
    )
    combined = condition_many_cell_schur_vit(
        base_params, jax.random.PRNGKey(int(controls["head_seed"])), conditioned_model
    )
    initial_local = combined["local"]
    probabilities = np.abs(initial_state) ** 2
    target_log_ratios = np.log(target / initial_state)
    initial_heads = {
        "probability_head": initial_local["probability_head"],
        "phase_head": initial_local["phase_head"],
    }
    flat_heads, unravel_heads = ravel_pytree(initial_heads)

    def local_from_flat(values):
        answer = dict(initial_local)
        answer.update(unravel_heads(values))
        return answer

    tangent_rows = []
    tangent_batch_size = int(controls["tangent_batch_size"])
    for start in range(0, len(labels), tangent_batch_size):
        batch = jnp.asarray(labels[start : start + tangent_batch_size])

        def batch_logs(values):
            return many_cell_hamiltonian_conditioned_log_amplitudes(
                {"base": base_params, "local": local_from_flat(values)},
                batch,
                conditioned_model,
            )

        tangent_rows.append(
            np.asarray(jax.device_get(jax.jacfwd(batch_logs)(flat_heads)))
        )
    log_derivatives = np.concatenate(tangent_rows, axis=0)
    projection = project_born_complex_tangent(
        log_derivatives,
        probabilities,
        target_log_ratios,
        relative_rank_cutoff=float(controls["relative_rank_cutoff"]),
    )
    direction = jnp.asarray(projection.least_squares_coefficients)

    def evaluate(values):
        local = local_from_flat(values)
        logs = evaluate_log_amplitudes_in_batches(
            local,
            labels,
            lambda candidate, batch: many_cell_hamiltonian_conditioned_log_amplitudes(
                {"base": base_params, "local": candidate},
                batch,
                conditioned_model,
            ),
            batch_size=int(controls["audit_batch_size"]),
        )
        candidate = np.exp(logs)
        native = float(np.vdot(candidate, candidate).real)
        candidate /= math.sqrt(native)
        return local, candidate, native, _state_metrics(candidate, matrix)

    candidate_rows = []
    best = None
    for exponent in range(int(controls["backtracking_steps"])):
        scale = float(controls["step_size"]) * 0.5**exponent
        local, candidate, native, metrics = evaluate(flat_heads + scale * direction)
        distance = exact_state_distance(initial_state, candidate)
        row = {
            "scale": scale,
            "energy_total": metrics["energy_total"],
            "variance_total": metrics["variance_total"],
            "fubini_study_distance": distance.fubini_study_distance,
            "symmetric_kl": distance.symmetric_kl,
        }
        candidate_rows.append(row)
        if (
            distance.fubini_study_distance
            <= float(controls["candidate_maximum_fubini_study_distance"])
            and (best is None or metrics["energy_total"] < best[3]["energy_total"])
        ):
            best = (local, candidate, native, metrics, distance, scale)
    if best is None:
        best = (
            initial_local,
            initial_state,
            1.0,
            _state_metrics(initial_state, matrix),
            exact_state_distance(initial_state, initial_state),
            0.0,
        )
    trained_local, state, native_norm, metrics, initial_distance, scale = best
    return {
        "head_seed": int(controls["head_seed"]),
        "fit_seed": int(seed),
        "initial_local_params_sha256": pytree_sha256(initial_local),
        "trained_local_params_sha256": pytree_sha256(trained_local),
        "native_normalization": native_norm,
        "metrics": metrics,
        "selected_backtracking_scale": float(scale),
        "candidate_scan": candidate_rows,
        "initial_to_student_distance": _distance_payload(initial_distance),
        "student_to_target_distance": _distance_payload(
            exact_state_distance(state, target)
        ),
        "feature_loss_recovery_fraction": float(projection.projection_fraction),
        "tangent_projection": projection.scalar_summary(),
        "tangent_svd_valid": bool(
            projection.numerical_rank > 0
            and np.all(np.isfinite(projection.singular_values))
        ),
    }


def _run_case(
    case,
    *,
    shared,
    source,
    source_root,
    controls,
    gates,
    case_index,
):
    geometry = str(case["geometry"])
    j2 = float(case["j2_over_j1"])
    layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=geometry == "triangular"
    )
    base_model = ManyCellSchurViTConfig(
        layout, target_q=0, geometry=geometry, **shared
    )
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    template = init_many_cell_schur_vit(
        jax.random.PRNGKey(2026092540 + case_index), base_model
    )
    artifact = _source_artifact(source, geometry, source_root)
    base_params = load_parameter_artifact(artifact, template)
    initial_logs = evaluate_log_amplitudes_in_batches(
        base_params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, base_model
        ),
        batch_size=int(controls["audit_batch_size"]),
    )
    initial = np.exp(initial_logs)
    initial /= np.linalg.norm(initial)
    physical_bonds = _bonds(geometry, j2)
    matrix = _dense_matrix(
        build_many_cell_tree_connectivity(labels, base_model, physical_bonds)
    )
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
    physical = _fit_head(
        base_params=base_params,
        base_model=base_model,
        bonds=_conditioned_bonds(layout, physical_bonds, j2),
        labels=labels,
        initial_logs=initial_logs,
        initial_state=initial,
        target=target,
        matrix=matrix,
        controls=controls,
        seed=int(controls["fit_seed"]) + 100 * case_index,
    )
    controls_rows = []
    for control_index, scramble_seed in enumerate(controls["scramble_seeds"]):
        permutation = np.random.default_rng(int(scramble_seed)).permutation(16)
        controls_rows.append(
            _fit_head(
                base_params=base_params,
                base_model=base_model,
                bonds=_conditioned_bonds(
                    layout, physical_bonds, j2, permutation=permutation
                ),
                labels=labels,
                initial_logs=initial_logs,
                initial_state=initial,
                target=target,
                matrix=matrix,
                controls=controls,
                seed=int(controls["fit_seed"])
                + 100 * case_index
                + control_index
                + 1,
            )
        )
        controls_rows[-1]["scramble_seed"] = int(scramble_seed)
    target_gain = initial_metrics["energy_per_site"] - target_metrics["energy_per_site"]

    def gain(row):
        return initial_metrics["energy_per_site"] - row["metrics"]["energy_per_site"]

    physical_gain = gain(physical)
    control_gains = [gain(row) for row in controls_rows]
    physical_recovery = physical_gain / max(target_gain, np.finfo(float).tiny)
    control_recoveries = [value / max(target_gain, np.finfo(float).tiny) for value in control_gains]
    advantage = physical_recovery - float(np.median(control_recoveries))
    variance_change = (
        physical["metrics"]["variance_per_site"] - initial_metrics["variance_per_site"]
    )
    passed = bool(
        physical["tangent_svd_valid"]
        and all(row["tangent_svd_valid"] for row in controls_rows)
        and physical_recovery >= float(gates["minimum_physical_energy_recovery_fraction"])
        and advantage >= float(gates["minimum_physical_over_scramble_recovery"])
        and physical["feature_loss_recovery_fraction"]
        >= float(gates["minimum_physical_feature_recovery_fraction"])
        and variance_change <= float(gates["maximum_physical_variance_change_per_site"])
        and physical["initial_to_student_distance"]["fubini_study_distance"]
        <= float(gates["maximum_physical_fubini_study_distance"])
    )
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "source_parameter_artifact": str(artifact),
        "selected_tau": float(selected.tau),
        "initial": initial_metrics,
        "uncompressed_target": target_metrics,
        "target_energy_gain_per_site": target_gain,
        "physical_bond_head": physical,
        "scrambled_bond_heads": controls_rows,
        "physical_energy_gain_per_site": physical_gain,
        "scrambled_energy_gain_per_site": control_gains,
        "physical_energy_recovery_fraction": physical_recovery,
        "scrambled_energy_recovery_fraction": control_recoveries,
        "physical_over_median_scramble_recovery": advantage,
        "physical_variance_change_per_site": variance_change,
        "passed": passed,
    }


def run_spin_n16_bond_causal_tangent(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Compare the physical bond head with matched site-scrambled controls."""

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
            case_index=index,
        )
        for index, case in enumerate(preflight["systems"])
    ]
    passed = bool(all(case["passed"] for case in cases))
    result = {
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
        "all_bond_causal_tangent_gates_passed": passed,
        "full_trainable_head_implementation_allowed": passed,
        "n36_training_allowed": False,
        "decision": (
            "implement_sampleable_bond_causal_schur_vit"
            if passed
            else "freeze_current_bond_path_encoder_and_change_recoupling_features"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Exact N=16 one-step tangent development gate with matched bond "
            "scrambles. It is not a finite-training or large-size energy claim."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_bond_causal_tangent"]
