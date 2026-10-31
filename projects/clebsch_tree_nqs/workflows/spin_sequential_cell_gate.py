"""Seal the cell-sequential SU(2)-Spatial-ViT representation gate."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.qsl_three_arm import triangular_pi_flux_operator_components
from su2_nqs.benchmarks.sequential_cell_replay import (
    balanced_to_sequential_coefficients,
    computational_to_sequential_coefficients,
    distill_sequential_cell_vit,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.sequential_cell_schur_vit import (
    SequentialCellSchurViTConfig,
    init_sequential_cell_schur_vit,
    sample_sequential_cell_schur_vit,
)
from su2_nqs.su2.many_cell_schur import (
    rectangular_sequential_four_spin_layout,
    validate_layout_state,
)
from su2_nqs.su2.sequential_cell_schur import labels_to_sequential_actions
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_sequential_cell_schur_gate"


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid sequential-cell gate identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("production seeds must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("the local gate is fixed to triangular 4x4")
    for name in ("source", "matched_arm", "capacity_arm", "physical_carrier_arm", "gates", "n36_preflight"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    if len(spec["matched_arm"].get("seeds", ())) < 3:
        raise ValueError("the matched arm requires at least three untouched seeds")
    if spec["n36_preflight"].get("run_energy_training_only_if_all_gates_pass") is not True:
        raise ValueError("N36 training must fail closed")


def _parameter_shapes(params: object) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in leaf.shape)
        for leaf in jax.tree_util.tree_leaves(params)
    )


def _energy_variance(workload, state: np.ndarray) -> tuple[float, float]:
    positions = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(workload.center_labels)
    }
    action = np.zeros_like(state, dtype=np.complex128)
    for old, coefficient in enumerate(state):
        for slot in range(int(workload.widths[old])):
            new = tuple(int(value) for value in workload.connected_labels[old, slot])
            action[positions[new]] += workload.coefficients[old, slot] * coefficient
    norm = np.asarray(state, dtype=np.complex128)
    norm = norm / np.linalg.norm(norm)
    action = action / np.linalg.norm(state)
    energy = float(np.vdot(norm, action).real)
    residual = action - energy * norm
    return energy, float(np.vdot(residual, residual).real)


def _arm_kwargs(arm: dict[str, object]) -> dict[str, object]:
    return {
        "steps": int(arm["steps"]),
        "learning_rate": float(arm["learning_rate"]),
        "embedding_dim": int(arm["embedding_dim"]),
        "head_count": int(arm["head_count"]),
        "depth": int(arm["depth"]),
        "mlp_ratio": int(arm["mlp_ratio"]),
        "objective": str(arm["objective"]),
    }


def run_spin_sequential_cell_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run exact recoupling, matched capacity, carrier, and N36 smoke gates."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")

    source = spec["source"]
    source_result_path = ROOT / source["result"]
    source_bundle_path = ROOT / source["bundle"]
    source_result = json.loads(source_result_path.read_text())
    if source_result.get("result_sha256") != source["result_sha256"]:
        raise RuntimeError("source result seal changed")
    if canonical_sha256(
        {key: value for key, value in source_result.items() if key != "result_sha256"}
    ) != source["result_sha256"]:
        raise RuntimeError("source result self-seal is invalid")
    if file_sha256(source_bundle_path) != source["bundle_file_sha256"]:
        raise RuntimeError("source bundle seal changed")
    with np.load(source_bundle_path) as bundle:
        balanced = np.asarray(bundle[source["coefficient_key"]], dtype=np.complex128)

    target, labels, recoupling = balanced_to_sequential_coefficients(balanced)
    layout = rectangular_sequential_four_spin_layout(4, 4)
    matched_arm = spec["matched_arm"]
    matched_config = SequentialCellSchurViTConfig(
        layout,
        embedding_dim=int(matched_arm["embedding_dim"]),
        head_count=int(matched_arm["head_count"]),
        depth=int(matched_arm["depth"]),
        mlp_ratio=int(matched_arm["mlp_ratio"]),
    )
    bonds = triangular_hnf_j1j2_bonds(
        ((4, 0), (0, 4)), j1=1.0, j2=float(spec["j2_over_j1"])
    )
    workload = build_many_cell_tree_connectivity(labels, matched_config, bonds)
    target_energy, target_variance = _energy_variance(workload, target)
    expected_energy = float(source_result["n16_exact_replay"]["matrix_free_energy_total"])

    matched_trials = []
    matched_states = []
    for seed in matched_arm["seeds"]:
        report, state, _ = distill_sequential_cell_vit(
            labels,
            target,
            workload,
            seed=int(seed),
            **_arm_kwargs(matched_arm),
        )
        matched_trials.append(report)
        matched_states.append(state)

    capacity_arm = spec["capacity_arm"]
    capacity_report, capacity_state, _ = distill_sequential_cell_vit(
        labels,
        target,
        workload,
        seed=int(capacity_arm["seed"]),
        **_arm_kwargs(capacity_arm),
    )

    carrier_data = triangular_pi_flux_operator_components(float(spec["j2_over_j1"]))
    carrier_target, carrier_labels, carrier_projection = computational_to_sequential_coefficients(
        carrier_data.carrier, carrier_data.configurations
    )
    if not np.array_equal(carrier_labels, labels):
        raise RuntimeError("carrier and sealed targets use different sequential basis orders")
    carrier_arm = spec["physical_carrier_arm"]
    carrier_report, carrier_state, _ = distill_sequential_cell_vit(
        labels,
        carrier_target,
        workload,
        seed=int(carrier_arm["seed"]),
        **_arm_kwargs(carrier_arm),
    )

    large_layout = rectangular_sequential_four_spin_layout(6, 6)
    large_config = SequentialCellSchurViTConfig(
        large_layout,
        embedding_dim=int(matched_arm["embedding_dim"]),
        head_count=int(matched_arm["head_count"]),
        depth=int(matched_arm["depth"]),
        mlp_ratio=int(matched_arm["mlp_ratio"]),
    )
    small_params = init_sequential_cell_schur_vit(jax.random.PRNGKey(0), matched_config)
    large_params = init_sequential_cell_schur_vit(jax.random.PRNGKey(0), large_config)
    shape_transfer = _parameter_shapes(small_params) == _parameter_shapes(large_params)
    sample_count = int(spec["n36_preflight"]["sample_count"])
    samples = np.asarray(
        sample_sequential_cell_schur_vit(
            large_params, jax.random.PRNGKey(1), sample_count, large_config
        )
    )
    for row in samples:
        validate_layout_state(large_layout, row, target_q=0)
    sample_actions = labels_to_sequential_actions(large_layout, samples)

    root_reference = source_result["transferable_vit_distillation"]
    median_fidelity = float(np.median([row["fidelity"] for row in matched_trials]))
    median_energy = float(np.median([row["energy_total"] for row in matched_trials]))
    best_index = int(np.argmax([row["fidelity"] for row in matched_trials]))
    gates = spec["gates"]
    gate_results = {
        "exact_sequential_recoupling": bool(
            recoupling["rotation_orthogonality_max_abs_error"] <= float(gates["recoupling_max_abs_error"])
            and recoupling["coefficient_roundtrip_max_abs_error"] <= float(gates["recoupling_max_abs_error"])
        ),
        "matrix_free_energy_replay": bool(
            abs(target_energy - expected_energy) <= float(gates["matrix_free_energy_abs_error"])
        ),
        "matched_cell_factorization_win": bool(
            median_fidelity - float(root_reference["fidelity"])
            >= float(gates["matched_median_fidelity_gain_over_root_first"])
            and float(root_reference["energy_total"]) - median_energy
            >= float(gates["matched_median_energy_gain_over_root_first"])
        ),
        "sealed_state_reproduction": bool(
            capacity_report["fidelity"] >= float(gates["sealed_state_fidelity"])
        ),
        "physical_carrier_reproduction": bool(
            carrier_report["fidelity"] >= float(gates["physical_carrier_fidelity"])
        ),
        "n16_n36_parameter_shape_transfer": bool(shape_transfer),
        "n36_autoregressive_sampler_smoke": bool(
            samples.shape == (sample_count, 35)
            and sample_actions.shape == (sample_count, 9, 3)
            and np.all(samples[:, -1] == 0)
        ),
        "scalable_full_gamma_a1_projector": False,
    }
    production_authorized = all(gate_results.values())
    arrays = {
        "sequential_labels": labels,
        "sealed_target_coefficients": target,
        "matched_best_coefficients": matched_states[best_index],
        "capacity_coefficients": capacity_state,
        "physical_carrier_target_coefficients": carrier_target,
        "physical_carrier_distilled_coefficients": carrier_state,
        "n36_sampler_smoke_labels": samples,
    }
    _atomic_npz(state_path, arrays)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "exact_recoupling": recoupling,
        "target_replay": {
            "energy_total": target_energy,
            "variance_total": target_variance,
            "expected_energy_total": expected_energy,
            "energy_abs_error": abs(target_energy - expected_energy),
            "connectivity_maximum_width": workload.maximum_width,
            "connectivity_total_entries": workload.total_entries,
        },
        "root_first_reference": root_reference,
        "matched_cell_trials": matched_trials,
        "matched_summary": {
            "median_fidelity": median_fidelity,
            "median_energy_total": median_energy,
            "fidelity_gain_over_root_first": median_fidelity - float(root_reference["fidelity"]),
            "energy_gain_over_root_first": float(root_reference["energy_total"]) - median_energy,
        },
        "capacity_trial": capacity_report,
        "physical_carrier_projection": carrier_projection,
        "physical_carrier_trial": carrier_report,
        "n36_preflight": {
            "n_sites": 36,
            "n_cells": 9,
            "action_shape": list(sample_actions.shape),
            "parameter_shapes_match_n16": shape_transfer,
            "energy_training_started": False,
        },
        "gate_results": gate_results,
        "n36_gpu_production_authorized": production_authorized,
        "blocking_reasons": [name for name, passed in gate_results.items() if not passed],
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "repeated_ed_performed": False,
        "runtime_identity": numerical_runtime_identity(),
        "interpretation": (
            "A shorter physical cell factorization is tested separately from exact basis replay. "
            "It may be a matched-capacity improvement without yet reproducing either the near-ground "
            "state or the projected carrier; those failures keep N36 production closed."
        ),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_sequential_cell_gate"]
