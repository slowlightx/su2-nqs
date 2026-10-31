"""Exact N=16 oracle for causal Hamiltonian-residual dressing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.causal_residual import (
    compact_tree_action_sequences,
    martingale_residual_increments,
)
from su2_nqs.benchmarks.rvb_covering_oracle import two_lanczos_state
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.models.tree_gru import tree_decisions
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.exact_log_derivative_sr import (
    exact_log_derivative_natural_gradient_from_data,
)
from su2_nqs.vmc.imaginary_time_projection import exact_state_distance
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity

from .spin_n16_hamiltonian_filtration import _decision_category
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

TASK = "spin_n16_causal_operator_oracle"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 causal-operator oracle identity")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    seeds = controls.get("scramble_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("at least three integer scramble seeds are required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("scramble seeds must be distinct")
    for name in ("steps", "backtracking_steps", "audit_batch_size"):
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


def _normalized(logs: np.ndarray) -> np.ndarray:
    values = np.asarray(logs, dtype=np.complex128)
    state = np.exp(values - np.max(np.real(values)))
    state /= np.linalg.norm(state)
    return state


def _causal_features(layout, actions, probabilities, local_energy):
    increments, mean = martingale_residual_increments(
        probabilities, local_energy, actions
    )
    within = np.zeros(len(probabilities), dtype=np.complex128)
    inter = np.zeros(len(probabilities), dtype=np.complex128)
    fractions = {"within_cell": 0.0, "intercell": 0.0}
    total_variance = float(
        np.sum(probabilities * np.abs(local_energy - mean) ** 2)
    )
    for decision, column in zip(
        tree_decisions(layout.tree), increments.T, strict=True
    ):
        category = _decision_category(layout, int(decision.node_index))
        key = "intercell" if category == "intercell_q" else "within_cell"
        if key == "intercell":
            inter += column
        else:
            within += column
        fractions[key] += float(np.sum(probabilities * np.abs(column) ** 2))
    features = np.column_stack((within, inter))
    closure = float(
        np.max(np.abs(np.sum(features, axis=1) - (local_energy - mean)))
    )
    return features, complex(mean), closure, {
        key: value / total_variance if total_variance > 0.0 else 0.0
        for key, value in fractions.items()
    }


def _train_exponential_oracle(
    *,
    base_logs,
    base_state,
    features,
    matrix,
    controls,
):
    theta = np.zeros(features.shape[1], dtype=np.float64)
    initial_metrics = _state_metrics(base_state, matrix)
    current_state = base_state
    current_metrics = initial_metrics
    history = []
    for step in range(1, int(controls["steps"]) + 1):
        logs = base_logs + features @ theta
        natural = exact_log_derivative_natural_gradient_from_data(
            logs,
            features,
            hamiltonian=matrix,
            damping=float(controls["damping"]),
            relative_rank_cutoff=float(controls["relative_rank_cutoff"]),
        )
        candidates = []
        selected = None
        for exponent in range(int(controls["backtracking_steps"])):
            scale = float(controls["learning_rate"]) * 0.5**exponent
            candidate_theta = theta + scale * natural.descent_direction
            candidate_state = _normalized(base_logs + features @ candidate_theta)
            metrics = _state_metrics(candidate_state, matrix)
            step_distance = exact_state_distance(current_state, candidate_state)
            total_distance = exact_state_distance(base_state, candidate_state)
            candidates.append(
                {
                    "scale": scale,
                    "energy_per_site": metrics["energy_per_site"],
                    "variance_per_site": metrics["variance_per_site"],
                    "step_fubini_study_distance": step_distance.fubini_study_distance,
                    "total_fubini_study_distance": total_distance.fubini_study_distance,
                }
            )
            if (
                metrics["energy_total"] < current_metrics["energy_total"]
                and step_distance.fubini_study_distance
                <= float(controls["maximum_step_fubini_study_distance"])
                and total_distance.fubini_study_distance
                <= float(controls["maximum_total_fubini_study_distance"])
                and (selected is None or metrics["energy_total"] < selected[2]["energy_total"])
            ):
                selected = (
                    candidate_theta,
                    candidate_state,
                    metrics,
                    step_distance,
                    total_distance,
                    scale,
                )
        row = {
            "step": step,
            "metric_numerical_rank": natural.numerical_rank,
            "gradient_norm": float(np.linalg.norm(natural.gradient)),
            "candidate_scan": candidates,
            "accepted": selected is not None,
        }
        if selected is None:
            history.append(row)
            break
        theta, current_state, current_metrics, step_distance, total_distance, scale = selected
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
    return {
        "parameters": theta.tolist(),
        "initial": initial_metrics,
        "final": current_metrics,
        "energy_gain_per_site": initial_metrics["energy_per_site"]
        - current_metrics["energy_per_site"],
        "variance_change_per_site": current_metrics["variance_per_site"]
        - initial_metrics["variance_per_site"],
        "initial_to_final_distance": _distance_payload(
            exact_state_distance(base_state, current_state)
        ),
        "accepted_steps": sum(bool(row["accepted"]) for row in history),
        "history": history,
    }


def _run_case(case, *, shared, source, source_root, controls, gates, case_index):
    geometry = str(case["geometry"])
    j2 = float(case["j2_over_j1"])
    layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=geometry == "triangular"
    )
    model = ManyCellSchurViTConfig(layout, target_q=0, geometry=geometry, **shared)
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    template = init_many_cell_schur_vit(
        jax.random.PRNGKey(2026092660 + case_index), model
    )
    source_artifact = _source_artifact(source, geometry, source_root)
    base_params = load_parameter_artifact(source_artifact, template)
    base_logs = evaluate_log_amplitudes_in_batches(
        base_params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, model
        ),
        batch_size=int(controls["audit_batch_size"]),
    )
    base_state = _normalized(base_logs)
    probabilities = np.abs(base_state) ** 2
    actions = compact_tree_action_sequences(layout.tree, 0, labels)
    physical_bonds = _bonds(geometry, j2)
    matrix = _dense_matrix(
        build_many_cell_tree_connectivity(labels, model, physical_bonds)
    )
    local_energy = (matrix @ base_state) / base_state
    features, mean, closure, fractions = _causal_features(
        layout, actions, probabilities, local_energy
    )
    physical = _train_exponential_oracle(
        base_logs=base_logs,
        base_state=base_state,
        features=features,
        matrix=matrix,
        controls=controls,
    )
    two_lanczos = two_lanczos_state(base_state, matrix)
    two_lanczos_metrics = _state_metrics(two_lanczos.state, matrix)
    scrambles = []
    for seed in controls["scramble_seeds"]:
        permutation = np.random.default_rng(int(seed)).permutation(16)
        scrambled_bonds = tuple(
            (int(permutation[first]), int(permutation[second]), float(coupling))
            for first, second, coupling in physical_bonds
        )
        scrambled_matrix = _dense_matrix(
            build_many_cell_tree_connectivity(labels, model, scrambled_bonds)
        )
        scrambled_local = (scrambled_matrix @ base_state) / base_state
        scrambled_features, _, scrambled_closure, scrambled_fractions = _causal_features(
            layout, actions, probabilities, scrambled_local
        )
        trained = _train_exponential_oracle(
            base_logs=base_logs,
            base_state=base_state,
            features=scrambled_features,
            matrix=matrix,
            controls=controls,
        )
        trained.update(
            {
                "scramble_seed": int(seed),
                "feature_closure_error": scrambled_closure,
                "filtration_variance_fraction": scrambled_fractions,
            }
        )
        scrambles.append(trained)
    scramble_energies = np.asarray(
        [row["final"]["energy_per_site"] for row in scrambles]
    )
    physical_energy = float(physical["final"]["energy_per_site"])
    physical_variance = float(physical["final"]["variance_per_site"])
    passed = bool(
        closure <= float(gates["maximum_feature_closure_error"])
        and physical["accepted_steps"] >= int(gates["minimum_accepted_steps"])
        and physical["energy_gain_per_site"]
        >= float(gates["minimum_physical_energy_gain_per_site"])
        and physical["variance_change_per_site"]
        <= float(gates["maximum_physical_variance_change_per_site"])
        and float(np.median(scramble_energies)) - physical_energy
        >= float(gates["minimum_physical_over_scramble_energy_per_site"])
        and physical_energy
        <= two_lanczos_metrics["energy_per_site"]
        + float(gates["maximum_physical_energy_above_two_lanczos_per_site"])
        and physical_variance
        <= two_lanczos_metrics["variance_per_site"]
        + float(gates["maximum_physical_variance_above_two_lanczos_per_site"])
    )
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "source_parameter_artifact": str(source_artifact),
        "feature_definition": [
            "within_cell_Doob_increment_sum",
            "intercell_q_Doob_increment_sum",
        ],
        "feature_closure_error": closure,
        "filtration_variance_fraction": fractions,
        "base_local_energy_mean_real": float(np.real(mean)),
        "base_local_energy_mean_imag": float(np.imag(mean)),
        "physical_oracle": physical,
        "scrambled_oracles": scrambles,
        "two_lanczos": {
            "rank": int(two_lanczos.rank),
            **two_lanczos_metrics,
            "base_to_two_lanczos_distance": _distance_payload(
                exact_state_distance(base_state, two_lanczos.state)
            ),
        },
        "physical_below_median_scramble_per_site": float(
            np.median(scramble_energies) - physical_energy
        ),
        "physical_minus_two_lanczos_energy_per_site": float(
            physical_energy - two_lanczos_metrics["energy_per_site"]
        ),
        "physical_minus_two_lanczos_variance_per_site": float(
            physical_variance - two_lanczos_metrics["variance_per_site"]
        ),
        "passed": passed,
    }


def run_spin_n16_causal_operator_oracle(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Test the best exact causal residual message before approximating it."""

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
        "cases": cases,
        "all_oracle_gates_passed": passed,
        "scalable_recoupling_kernel_allowed": passed,
        "n36_training_allowed": False,
        "decision": (
            "compress_exact_causal_operator_into_local_f_move_kernel"
            if passed
            else "do_not_approximate_causal_operator_change_state_manifold"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Exact-support N=16 causal-operator oracle. It deliberately uses "
            "full-support conditional expectations and is not a scalable ansatz, "
            "an A1 proof, or an N36 variational result."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_causal_operator_oracle"]
