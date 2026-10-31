"""Exact N=16 causal dressing on top of a two-Lanczos carrier."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.benchmarks.rvb_covering_oracle import (
    lowest_ritz_state,
    two_lanczos_state,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.imaginary_time_projection import exact_state_distance
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity

from .spin_n16_causal_operator_oracle import (
    _causal_features,
    _distance_payload,
    _train_exponential_oracle,
)
from .spin_n16_imaginary_time_compression import (
    _bonds,
    _dense_matrix,
    _source_artifact,
    _state_metrics,
)
from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _validate as _validate_preflight


jax.config.update("jax_enable_x64", True)

TASK = "spin_n16_krylov_causal_oracle"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 Krylov-causal oracle identity")
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


def _state_logs(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.complex128)
    if np.min(np.abs(values)) <= 1.0e-14:
        raise RuntimeError("the two-Lanczos carrier contains exact-support nodes")
    return np.log(np.abs(values)) + 1.0j * np.angle(values)


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
        jax.random.PRNGKey(2026092740 + case_index), model
    )
    source_artifact = _source_artifact(source, geometry, source_root)
    base_params = load_parameter_artifact(source_artifact, template)
    raw_logs = evaluate_log_amplitudes_in_batches(
        base_params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, model
        ),
        batch_size=int(controls["audit_batch_size"]),
    )
    raw_state = np.exp(raw_logs)
    raw_state /= np.linalg.norm(raw_state)
    physical_bonds = _bonds(geometry, j2)
    matrix = _dense_matrix(
        build_many_cell_tree_connectivity(labels, model, physical_bonds)
    )
    two_lanczos = two_lanczos_state(raw_state, matrix)
    anchor = two_lanczos.state
    anchor_logs = _state_logs(anchor)
    probabilities = np.abs(anchor) ** 2
    actions = compact_tree_action_sequences(layout.tree, 0, labels)
    local_energy = (matrix @ anchor) / anchor
    features, mean, closure, fractions = _causal_features(
        layout, actions, probabilities, local_energy
    )
    physical = _train_exponential_oracle(
        base_logs=anchor_logs,
        base_state=anchor,
        features=features,
        matrix=matrix,
        controls=controls,
    )

    first = matrix @ raw_state
    second = matrix @ first
    third = matrix @ second
    three_lanczos = lowest_ritz_state(
        np.column_stack((raw_state, first, second, third)), matrix
    )
    three_metrics = _state_metrics(three_lanczos.state, matrix)
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
        scrambled_local = (scrambled_matrix @ anchor) / anchor
        scrambled_features, _, scrambled_closure, scrambled_fractions = _causal_features(
            layout, actions, probabilities, scrambled_local
        )
        trained = _train_exponential_oracle(
            base_logs=anchor_logs,
            base_state=anchor,
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
        <= three_metrics["energy_per_site"]
        + float(gates["maximum_physical_energy_above_three_lanczos_per_site"])
        and physical_variance
        <= three_metrics["variance_per_site"]
        + float(gates["maximum_physical_variance_above_three_lanczos_per_site"])
    )
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "source_parameter_artifact": str(source_artifact),
        "carrier": "exact_two_lanczos_ritz_state",
        "minimum_carrier_amplitude_abs": float(np.min(np.abs(anchor))),
        "feature_closure_error": closure,
        "filtration_variance_fraction": fractions,
        "anchor_local_energy_mean_real": float(np.real(mean)),
        "anchor_local_energy_mean_imag": float(np.imag(mean)),
        "two_lanczos_anchor": {
            "rank": int(two_lanczos.rank),
            **_state_metrics(anchor, matrix),
        },
        "physical_oracle": physical,
        "scrambled_oracles": scrambles,
        "three_lanczos": {
            "rank": int(three_lanczos.rank),
            **three_metrics,
            "anchor_to_three_lanczos_distance": _distance_payload(
                exact_state_distance(anchor, three_lanczos.state)
            ),
        },
        "physical_below_median_scramble_per_site": float(
            np.median(scramble_energies) - physical_energy
        ),
        "physical_minus_three_lanczos_energy_per_site": float(
            physical_energy - three_metrics["energy_per_site"]
        ),
        "physical_minus_three_lanczos_variance_per_site": float(
            physical_variance - three_metrics["variance_per_site"]
        ),
        "passed": passed,
    }


def run_spin_n16_krylov_causal_oracle(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Test non-Krylov causal gain after a two-Lanczos carrier."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    preflight_path = _resolve(config_path, spec["source_preflight"])
    preflight = json.loads(preflight_path.read_text())
    _validate_preflight(preflight)
    source_path = _resolve(config_path, spec["source_initialization_result"])
    source = json.loads(source_path.read_text())
    if source.get("task") != "spin_scale_first_carrier_distillation":
        raise ValueError("invalid source initialization result")
    oracle_path = _resolve(config_path, spec["source_causal_oracle_result"])
    oracle = json.loads(oracle_path.read_text())
    if oracle.get("task") != "spin_n16_causal_operator_oracle":
        raise ValueError("invalid source causal-operator oracle")
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
        "source_causal_oracle_result_sha256": oracle["result_sha256"],
        "cases": cases,
        "all_krylov_causal_gates_passed": passed,
        "additive_krylov_causal_sampler_allowed": passed,
        "n36_training_allowed": False,
        "decision": (
            "implement_additive_two_lanczos_carrier_plus_causal_su2_vit"
            if passed
            else "freeze_multiplicative_causal_dressing_and_change_carrier_manifold"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Exact N=16 oracle on a two-Lanczos carrier. Full-support Ritz "
            "states and conditional expectations are not scalable samplers or "
            "large-size variational results."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_krylov_causal_oracle"]
