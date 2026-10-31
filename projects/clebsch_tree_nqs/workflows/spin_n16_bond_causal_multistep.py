"""Exact N=16 multistep gate for the bond-causal Schur Spatial-ViT head."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import (
    load_parameter_artifact,
    save_parameter_artifact,
)
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.models.many_cell_hamiltonian_conditioned_vit import (
    ManyCellHamiltonianConditionedViTConfig,
    condition_many_cell_schur_vit,
    many_cell_hamiltonian_conditioned_head_log_amplitudes,
    prepare_many_cell_hamiltonian_conditioned_head_design,
    sample_many_cell_hamiltonian_conditioned_vit,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.exact_log_derivative_sr import (
    exact_log_derivative_natural_gradient_from_data,
)
from su2_nqs.vmc.imaginary_time_projection import exact_state_distance
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity

from .spin_n16_bond_causal_tangent import _conditioned_bonds
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

TASK = "spin_n16_bond_causal_multistep"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 bond-causal multistep identity")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    head_seeds = controls.get("head_seeds")
    scramble_seeds = controls.get("scramble_seeds")
    audit_seeds = controls.get("untouched_audit_seeds")
    for name, values in (
        ("head_seeds", head_seeds),
        ("scramble_seeds", scramble_seeds),
        ("untouched_audit_seeds", audit_seeds),
    ):
        if not isinstance(values, list) or len(values) < 2 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"controls.{name} requires at least two integer seeds")
        if len(set(values)) != len(values):
            raise ValueError(f"controls.{name} seeds must be unique")
    if len(head_seeds) != len(scramble_seeds):
        raise ValueError("head_seeds and scramble_seeds must be paired")
    if (
        set(head_seeds) & set(scramble_seeds)
        or set(head_seeds) & set(audit_seeds)
        or set(scramble_seeds) & set(audit_seeds)
    ):
        raise ValueError("development and untouched seed sets must be disjoint")
    for name in (
        "local_embedding_dim",
        "local_hidden_dim",
        "steps",
        "backtracking_steps",
        "audit_batch_size",
        "audit_sample_count",
    ):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")
    for name in (
        "learning_rate",
        "damping",
        "relative_rank_cutoff",
        "maximum_step_fubini_study_distance",
        "maximum_total_fubini_study_distance",
    ):
        value = controls.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"controls.{name} must be positive and finite")


def _normalized_state(logs: np.ndarray) -> np.ndarray:
    state = np.exp(np.asarray(logs, dtype=np.complex128))
    state /= np.linalg.norm(state)
    return state


def _train_active_heads(
    *,
    base_params,
    model: ManyCellHamiltonianConditionedViTConfig,
    labels: np.ndarray,
    matrix: np.ndarray,
    head_seed: int,
    controls: dict[str, object],
) -> dict[str, object]:
    combined = condition_many_cell_schur_vit(
        base_params, jax.random.PRNGKey(head_seed), model
    )
    initial_local = combined["local"]
    heads = {
        "probability_head": initial_local["probability_head"],
        "phase_head": initial_local["phase_head"],
    }
    flat, unravel = ravel_pytree(heads)
    matrix_array = jnp.asarray(matrix)

    def local_from_flat(values):
        local = dict(initial_local)
        local.update(unravel(values))
        return local

    design = prepare_many_cell_hamiltonian_conditioned_head_design(
        base_params,
        initial_local,
        jnp.asarray(labels),
        model,
    )

    def logs_from_flat(values):
        return many_cell_hamiltonian_conditioned_head_log_amplitudes(
            unravel(values), design
        )

    compiled_logs = jax.jit(logs_from_flat)
    compiled_logs_and_derivatives = jax.jit(
        lambda values: (logs_from_flat(values), jax.jacfwd(logs_from_flat)(values))
    )

    def evaluate(values):
        logs = np.asarray(jax.device_get(compiled_logs(values)))
        state = _normalized_state(logs)
        return state, _state_metrics(state, matrix)

    initial_state, initial_metrics = evaluate(flat)
    current_state = initial_state
    current_metrics = initial_metrics
    history: list[dict[str, object]] = []
    for step in range(1, int(controls["steps"]) + 1):
        logs, derivatives = compiled_logs_and_derivatives(flat)
        jax.block_until_ready((logs, derivatives))
        natural = exact_log_derivative_natural_gradient_from_data(
            logs,
            derivatives,
            hamiltonian=matrix_array,
            damping=float(controls["damping"]),
            relative_rank_cutoff=float(controls["relative_rank_cutoff"]),
        )
        direction = jnp.asarray(natural.descent_direction)
        candidates = []
        selected = None
        for exponent in range(int(controls["backtracking_steps"])):
            scale = float(controls["learning_rate"]) * 0.5**exponent
            candidate_flat = flat + scale * direction
            candidate_state, candidate_metrics = evaluate(candidate_flat)
            step_distance = exact_state_distance(current_state, candidate_state)
            total_distance = exact_state_distance(initial_state, candidate_state)
            row = {
                "scale": scale,
                "energy_per_site": candidate_metrics["energy_per_site"],
                "variance_per_site": candidate_metrics["variance_per_site"],
                "step_fubini_study_distance": step_distance.fubini_study_distance,
                "total_fubini_study_distance": total_distance.fubini_study_distance,
            }
            candidates.append(row)
            if (
                candidate_metrics["energy_total"] < current_metrics["energy_total"]
                and step_distance.fubini_study_distance
                <= float(controls["maximum_step_fubini_study_distance"])
                and total_distance.fubini_study_distance
                <= float(controls["maximum_total_fubini_study_distance"])
                and (
                    selected is None
                    or candidate_metrics["energy_total"]
                    < selected[2]["energy_total"]
                )
            ):
                selected = (
                    candidate_flat,
                    candidate_state,
                    candidate_metrics,
                    step_distance,
                    total_distance,
                    scale,
                )
        row = {
            "step": step,
            "energy_before_per_site": current_metrics["energy_per_site"],
            "variance_before_per_site": current_metrics["variance_per_site"],
            "gradient_norm": float(np.linalg.norm(natural.gradient)),
            "natural_direction_norm": float(np.linalg.norm(natural.descent_direction)),
            "metric_numerical_rank": natural.numerical_rank,
            "metric_maximum_eigenvalue": natural.maximum_metric_eigenvalue,
            "linear_relative_residual": natural.relative_linear_residual,
            "candidate_scan": candidates,
            "accepted": selected is not None,
        }
        if selected is None:
            history.append(row)
            break
        (
            flat,
            current_state,
            current_metrics,
            step_distance,
            total_distance,
            scale,
        ) = selected
        row.update(
            {
                "accepted_scale": scale,
                "energy_after_per_site": current_metrics["energy_per_site"],
                "variance_after_per_site": current_metrics["variance_per_site"],
                "step_distance": _distance_payload(step_distance),
                "total_distance": _distance_payload(total_distance),
            }
        )
        history.append(row)
    final_local = local_from_flat(flat)
    return {
        "head_seed": head_seed,
        "initial_local_params_sha256": pytree_sha256(initial_local),
        "final_local_params_sha256": pytree_sha256(final_local),
        "initial": initial_metrics,
        "final": current_metrics,
        "energy_gain_per_site": (
            initial_metrics["energy_per_site"] - current_metrics["energy_per_site"]
        ),
        "variance_change_per_site": (
            current_metrics["variance_per_site"] - initial_metrics["variance_per_site"]
        ),
        "initial_to_final_distance": _distance_payload(
            exact_state_distance(initial_state, current_state)
        ),
        "accepted_steps": sum(bool(row["accepted"]) for row in history),
        "history": history,
        "final_local": final_local,
        "final_state": current_state,
    }


def _sampled_audit(
    *,
    params,
    model,
    labels: np.ndarray,
    state: np.ndarray,
    matrix: np.ndarray,
    controls: dict[str, object],
) -> dict[str, object]:
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(labels)}
    h_state = matrix @ state
    energy = float(np.vdot(state, h_state).real)
    local_energy = h_state / state
    rows = []
    for seed in controls["untouched_audit_seeds"]:
        sampled = np.asarray(
            jax.device_get(
                sample_many_cell_hamiltonian_conditioned_vit(
                    params,
                    jax.random.PRNGKey(int(seed)),
                    int(controls["audit_sample_count"]),
                    model,
                )
            )
        )
        indices = np.asarray([lookup[tuple(int(value) for value in row)] for row in sampled])
        values = local_energy[indices]
        real_values = np.real(values)
        squared = np.abs(values - energy) ** 2
        rows.append(
            {
                "seed": int(seed),
                "sample_count": int(len(indices)),
                "energy_total": float(np.mean(real_values)),
                "energy_standard_error": float(np.std(real_values, ddof=1) / math.sqrt(len(indices))),
                "energy_imaginary_abs": float(abs(np.mean(np.imag(values)))),
                "variance_total": float(np.mean(squared)),
                "variance_standard_error": float(np.std(squared, ddof=1) / math.sqrt(len(indices))),
                "energy_z_score": float(
                    abs(np.mean(real_values) - energy)
                    / max(np.std(real_values, ddof=1) / math.sqrt(len(indices)), np.finfo(float).tiny)
                ),
            }
        )
    return {
        "design": "independent_untouched_autoregressive_sample_streams",
        "exact_energy_total": energy,
        "replicates": rows,
        "maximum_energy_z_score": max(row["energy_z_score"] for row in rows),
    }


def _public_training_row(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"final_local", "final_state"}}


def _run_case(case, *, shared, source, source_root, controls, gates, output, case_index):
    geometry = str(case["geometry"])
    j2 = float(case["j2_over_j1"])
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=geometry == "triangular")
    base_model = ManyCellSchurViTConfig(layout, target_q=0, geometry=geometry, **shared)
    labels = np.asarray([state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32)
    template = init_many_cell_schur_vit(jax.random.PRNGKey(2026092570 + case_index), base_model)
    source_artifact = _source_artifact(source, geometry, source_root)
    base_params = load_parameter_artifact(source_artifact, template)
    base_logs = evaluate_log_amplitudes_in_batches(
        base_params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(candidate, batch, base_model),
        batch_size=int(controls["audit_batch_size"]),
    )
    base_state = _normalized_state(base_logs)
    physical_bonds = _bonds(geometry, j2)
    matrix = _dense_matrix(build_many_cell_tree_connectivity(labels, base_model, physical_bonds))
    base_metrics = _state_metrics(base_state, matrix)
    physical_model = ManyCellHamiltonianConditionedViTConfig(
        base_model,
        _conditioned_bonds(layout, physical_bonds, j2),
        n_channels=2,
        local_embedding_dim=int(controls["local_embedding_dim"]),
        local_hidden_dim=int(controls["local_hidden_dim"]),
        train_local_encoder=False,
    )
    physical_rows = []
    scramble_rows = []
    for replicate, (head_seed, scramble_seed) in enumerate(
        zip(controls["head_seeds"], controls["scramble_seeds"], strict=True)
    ):
        physical = _train_active_heads(
            base_params=base_params,
            model=physical_model,
            labels=labels,
            matrix=matrix,
            head_seed=int(head_seed),
            controls=controls,
        )
        final_params = {"base": base_params, "local": physical["final_local"]}
        physical["sampled_audit"] = _sampled_audit(
            params=final_params,
            model=physical_model,
            labels=labels,
            state=physical["final_state"],
            matrix=matrix,
            controls=controls,
        )
        identity = {
            "task": TASK,
            "case_id": str(case["case_id"]),
            "arm": "physical_bond_causal_head",
            "head_seed": int(head_seed),
            "source_base_params_sha256": pytree_sha256(base_params),
        }
        manifest_path = output / "parameters" / f"{case['case_id']}_physical_seed{head_seed}.json"
        manifest = save_parameter_artifact(
            manifest_path,
            physical["final_local"],
            identity=identity,
            overwrite=True,
        )
        physical["parameter_artifact"] = str(manifest_path.relative_to(output))
        physical["parameter_manifest_sha256"] = manifest["manifest_sha256"]
        physical_rows.append(_public_training_row(physical))

        permutation = np.random.default_rng(int(scramble_seed)).permutation(16)
        scrambled_model = ManyCellHamiltonianConditionedViTConfig(
            base_model,
            _conditioned_bonds(layout, physical_bonds, j2, permutation=permutation),
            n_channels=2,
            local_embedding_dim=int(controls["local_embedding_dim"]),
            local_hidden_dim=int(controls["local_hidden_dim"]),
            train_local_encoder=False,
        )
        scrambled = _train_active_heads(
            base_params=base_params,
            model=scrambled_model,
            labels=labels,
            matrix=matrix,
            head_seed=int(head_seed),
            controls=controls,
        )
        scrambled["scramble_seed"] = int(scramble_seed)
        scramble_rows.append(_public_training_row(scrambled))

    physical_gains = np.asarray([row["energy_gain_per_site"] for row in physical_rows])
    scramble_gains = np.asarray([row["energy_gain_per_site"] for row in scramble_rows])
    paired_advantages = physical_gains - scramble_gains
    audit_z = [
        row["sampled_audit"]["maximum_energy_z_score"] for row in physical_rows
    ]
    passed = bool(
        np.median(physical_gains) >= float(gates["minimum_median_physical_energy_gain_per_site"])
        and np.median(paired_advantages) >= float(gates["minimum_median_paired_advantage_per_site"])
        and sum(value > 0.0 for value in paired_advantages)
        >= int(gates["minimum_positive_paired_replicates"])
        and max(row["variance_change_per_site"] for row in physical_rows)
        <= float(gates["maximum_physical_variance_change_per_site"])
        and min(row["accepted_steps"] for row in physical_rows)
        >= int(gates["minimum_accepted_steps"])
        and max(audit_z) <= float(gates["maximum_untouched_energy_z_score"])
    )
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "source_parameter_artifact": str(source_artifact),
        "base": base_metrics,
        "physical_replicates": physical_rows,
        "scrambled_replicates": scramble_rows,
        "physical_energy_gains_per_site": physical_gains.tolist(),
        "scrambled_energy_gains_per_site": scramble_gains.tolist(),
        "paired_physical_advantages_per_site": paired_advantages.tolist(),
        "median_physical_energy_gain_per_site": float(np.median(physical_gains)),
        "median_scrambled_energy_gain_per_site": float(np.median(scramble_gains)),
        "median_paired_advantage_per_site": float(np.median(paired_advantages)),
        "passed": passed,
    }


def run_spin_n16_bond_causal_multistep(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run paired exact multistep fits and untouched sampler replays."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    preflight_path = _resolve(config_path, spec["source_preflight"])
    preflight = json.loads(preflight_path.read_text())
    _validate_preflight(preflight)
    source_path = _resolve(config_path, spec["source_initialization_result"])
    source = json.loads(source_path.read_text())
    if source.get("task") != "spin_scale_first_carrier_distillation":
        raise ValueError("invalid source initialization result")
    tangent_path = _resolve(config_path, spec["source_tangent_result"])
    tangent = json.loads(tangent_path.read_text())
    if tangent.get("task") != "spin_n16_bond_causal_tangent" or not tangent.get(
        "all_bond_causal_tangent_gates_passed"
    ):
        raise ValueError("a passing bond-causal tangent artifact is required")
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
    result = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_tangent_result_sha256": tangent["result_sha256"],
        "cases": cases,
        "all_multistep_gates_passed": passed,
        "n36_sampled_canary_allowed": passed,
        "n36_gpu_production_allowed": False,
        "decision": (
            "prepare_bounded_n36_sampled_bond_causal_canary"
            if passed
            else "hold_n36_and_revise_bond_causal_feature_manifold"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Exact N=16 multistep energy optimization of 128 active output-head "
            "parameters, paired with geometry scrambles and independent sampler "
            "replays. It is not an N=36 variational or scaling claim."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_bond_causal_multistep"]
