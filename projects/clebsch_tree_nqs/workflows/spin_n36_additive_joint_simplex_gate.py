"""Local carrier-retaining joint-training gate for the N=36 simplex state."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.many_cell_schur_vit import (
    many_cell_schur_log_amplitudes,
    sample_many_cell_schur_vit,
)
from su2_nqs.models.many_cell_simplex_vit import (
    ManyCellSimplexViTConfig,
    init_many_cell_simplex_vit,
    many_cell_simplex_parameter_count,
    sample_many_cell_simplex_vit,
)
from su2_nqs.su2.many_cell_schur import (
    scramble_simplex_coordinates,
    triangular_six_by_six_simplex_layout,
    validate_layout_state,
)
from su2_nqs.vmc.additive_simplex_training import (
    AdditiveSimplexTrainingControls,
    additive_simplex_importance_metrics,
    run_resumable_additive_simplex_training,
)
from su2_nqs.vmc.many_cell_simplex import (
    audit_sampled_local_f_moves,
    evaluate_many_cell_simplex_additive_local_energy,
)
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_additive_joint_simplex_gate"
ARM_IDS = ("physical_simplex_residual", "scrambled_simplex_residual")


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid additive joint-simplex gate identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the local gate must be preregistered")
    for name in ("system", "model", "training", "audit", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    system = spec["system"]
    if system.get("geometry") != "triangular" or (
        int(system.get("rows", 0)), int(system.get("columns", 0))
    ) != (6, 6):
        raise ValueError("the gate is fixed to triangular 6x6")
    model = spec["model"]
    if model.get("magnetic_indices_visible") is not False:
        raise ValueError("magnetic indices may not enter the learned state")
    if model.get("joint_residual_backbone_training") is not True:
        raise ValueError("the complete residual backbone must be trainable")
    permutation = model.get("scramble_cell_permutation")
    if not isinstance(permutation, list) or sorted(permutation) != list(range(9)):
        raise ValueError("scramble_cell_permutation must permute nine cells")
    if permutation == list(range(9)):
        raise ValueError("the spatial scramble must be nontrivial")
    audit = spec["audit"]
    seeds = audit.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("audit.seeds must contain at least three distinct seeds")


def _model_config(spec: dict[str, object], *, scrambled: bool) -> ManyCellSimplexViTConfig:
    model = spec["model"]
    layout = triangular_six_by_six_simplex_layout()
    if scrambled:
        layout = scramble_simplex_coordinates(
            layout, model["scramble_cell_permutation"]
        )
    return ManyCellSimplexViTConfig(
        layout,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        depth=int(model["depth"]),
        mlp_ratio=int(model["mlp_ratio"]),
        phase_scale=float(model["phase_scale"]),
        initialization_scale=float(model["initialization_scale"]),
        direct_phase_initialization_scale=float(
            model["direct_phase_initialization_scale"]
        ),
        initial_distance_decay=float(model["initial_distance_decay"]),
        use_distance_envelope=bool(model["use_distance_envelope"]),
        simplex_head_initialization_scale=float(
            model["simplex_head_initialization_scale"]
        ),
        initial_pair_resonance_bias=float(model["initial_pair_resonance_bias"]),
        initial_root_resonance_bias=float(model["initial_root_resonance_bias"]),
        initial_inter_simplex_bias=float(model["initial_inter_simplex_bias"]),
    )


def _mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _audit_seed(
    *,
    seed: int,
    count: int,
    carrier: object,
    carrier_config,
    arm_params: dict[str, object],
    arm_config: ManyCellSimplexViTConfig,
    baseline_residual: object,
    bonds: tuple[tuple[int, int, float], ...],
) -> dict[str, object]:
    labels = np.asarray(
        sample_many_cell_schur_vit(
            carrier, jax.random.PRNGKey(seed), count, carrier_config
        )
    )
    for row in labels:
        validate_layout_state(carrier_config.layout, row, target_q=0)
    carrier_logs = np.asarray(
        many_cell_schur_log_amplitudes(carrier, labels, carrier_config)
    )
    workload = build_many_cell_tree_connectivity(labels, carrier_config, bonds)
    baseline = evaluate_many_cell_simplex_additive_local_energy(
        carrier,
        baseline_residual,
        0.0,
        workload,
        carrier_config,
        arm_config,
    )
    target = evaluate_many_cell_simplex_additive_local_energy(
        carrier,
        arm_params["residual"],
        arm_params["residual_scale"],
        workload,
        carrier_config,
        arm_config,
    )
    log_q = 2.0 * carrier_logs.real
    baseline_metrics = additive_simplex_importance_metrics(
        baseline.center_amplitudes, baseline.action_amplitudes, log_q
    )
    target_metrics = additive_simplex_importance_metrics(
        target.center_amplitudes, target.action_amplitudes, log_q
    )
    nested_error = float(
        np.max(np.abs(baseline.center_amplitudes - np.exp(carrier_logs)))
    )
    native = np.asarray(
        sample_many_cell_simplex_vit(
            arm_params["residual"],
            jax.random.PRNGKey(seed + 1_000_003),
            count,
            arm_config,
        )
    )
    for row in native:
        validate_layout_state(arm_config.simplex_layout.schur, row, target_q=0)
    f_audit = audit_sampled_local_f_moves(native, arm_config)
    return {
        "seed": int(seed),
        "sample_count": int(count),
        "carrier": baseline_metrics,
        "target": target_metrics,
        "target_minus_carrier_energy_per_site": float(
            (target_metrics["energy"] - baseline_metrics["energy"]) / 36.0
        ),
        "target_minus_carrier_variance_per_site": float(
            (target_metrics["variance"] - baseline_metrics["variance"]) / 36.0
        ),
        "zero_residual_carrier_max_abs_error": nested_error,
        "finite_target_h_ratios": bool(
            np.all(np.isfinite(target.ratios[workload.mask]))
        ),
        "maximum_connectivity_width": int(np.max(workload.widths)),
        "native_residual_local_f_audit": f_audit,
    }


def run_spin_n36_additive_joint_simplex_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run joint residual training and the untouched carrier-retention gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    if overwrite and (output / "checkpoints").exists():
        raise FileExistsError("refusing to overwrite checkpoint generations")

    physical_config = _model_config(spec, scrambled=False)
    scrambled_config = _model_config(spec, scrambled=True)
    carrier_config = physical_config.base_config
    artifact = _resolve(config_path, spec["carrier_parameter_artifact"])
    template = init_many_cell_simplex_vit(jax.random.PRNGKey(0), physical_config)
    carrier = load_parameter_artifact(artifact, template["backbone"])
    model = spec["model"]
    residuals = {}
    for arm_id, arm_config in zip(
        ARM_IDS, (physical_config, scrambled_config), strict=True
    ):
        candidate = init_many_cell_simplex_vit(
            jax.random.PRNGKey(int(model["initialization_seed"])), arm_config
        )
        residuals[arm_id] = {**candidate, "backbone": carrier}
    if pytree_sha256(residuals[ARM_IDS[0]]) != pytree_sha256(residuals[ARM_IDS[1]]):
        raise RuntimeError("physical and scrambled residual arrays must start identically")

    system = spec["system"]
    bonds = tuple(
        triangular_hnf_j1j2_bonds(
            ((6, 0), (0, 6)),
            j1=1.0,
            j2=float(system["j2_over_j1"]),
        )
    )
    training = spec["training"]
    controls = AdditiveSimplexTrainingControls(
        total_steps=int(training["total_steps"]),
        sample_count=int(training["sample_count"]),
        seed=int(training["seed"]),
        learning_rate=float(training["learning_rate"]),
        carrier_probability=float(training["carrier_probability"]),
        minimum_ess_fraction=float(training["minimum_ess_fraction"]),
        maximum_variance_increase=float(
            training.get("maximum_variance_increase", 0.0)
        ),
        line_search_factors=tuple(float(value) for value in training["line_search_factors"]),
        checkpoint_every_steps=int(training["checkpoint_every_steps"]),
    )
    initial_scale = float(training["initial_residual_scale"])
    trained = {}
    arm_reports = {}
    config_hash = canonical_sha256(spec)
    for arm_id, arm_config in zip(
        ARM_IDS, (physical_config, scrambled_config), strict=True
    ):
        identity = {
            "task": TASK,
            "config_sha256": config_hash,
            "arm_id": arm_id,
            "carrier_sha256": pytree_sha256(carrier),
            "initial_residual_sha256": pytree_sha256(residuals[arm_id]),
        }
        checkpoint = output / "checkpoints" / arm_id
        first = run_resumable_additive_simplex_training(
            residuals[arm_id],
            initial_scale,
            carrier_params=carrier,
            carrier_config=carrier_config,
            residual_config=arm_config,
            physical_bonds=bonds,
            controls=controls,
            checkpoint_directory=checkpoint,
            run_identity=identity,
            maximum_steps_this_call=1,
        )
        second = run_resumable_additive_simplex_training(
            residuals[arm_id],
            initial_scale,
            carrier_params=carrier,
            carrier_config=carrier_config,
            residual_config=arm_config,
            physical_bonds=bonds,
            controls=controls,
            checkpoint_directory=checkpoint,
            run_identity=identity,
        )
        restored = load_latest_resumable_checkpoint(
            checkpoint, expected_run_identity=second["run_identity"]
        )
        trained[arm_id] = second["params"]
        arm_reports[arm_id] = {
            "parameter_count": int(
                many_cell_simplex_parameter_count(second["params"]["residual"])
                + 1
            ),
            "initial_residual_sha256": pytree_sha256(residuals[arm_id]),
            "final_params_sha256": pytree_sha256(second["params"]),
            "final_backbone_sha256": pytree_sha256(
                second["params"]["residual"]["backbone"]
            ),
            "backbone_changed": bool(
                pytree_sha256(second["params"]["residual"]["backbone"])
                != pytree_sha256(carrier)
            ),
            "final_residual_scale": float(second["params"]["residual_scale"]),
            "first_invocation_completed_steps": int(first["completed_steps"]),
            "second_invocation_completed_steps": int(second["completed_steps"]),
            "second_invocation_resumed": bool(second["resumed"]),
            "checkpoint_restore_matches_final": bool(
                pytree_sha256(restored["params"])
                == pytree_sha256(second["params"])
            ),
            "history": second["history"],
            "cumulative_samples": int(second["cumulative_samples"]),
            "cumulative_full_h_calls": int(second["cumulative_full_h_calls"]),
            "cumulative_wall_seconds": float(second["cumulative_wall_seconds"]),
        }

    audit = spec["audit"]
    rows = {arm: [] for arm in ARM_IDS}
    for seed_value in audit["seeds"]:
        seed = int(seed_value)
        for arm_id, arm_config in zip(
            ARM_IDS, (physical_config, scrambled_config), strict=True
        ):
            rows[arm_id].append(
                _audit_seed(
                    seed=seed,
                    count=int(audit["sample_count"]),
                    carrier=carrier,
                    carrier_config=carrier_config,
                    arm_params=trained[arm_id],
                    arm_config=arm_config,
                    baseline_residual=residuals[arm_id],
                    bonds=bonds,
                )
            )

    physical_rows = rows[ARM_IDS[0]]
    scrambled_rows = rows[ARM_IDS[1]]
    carrier_delta, carrier_delta_se = _mean_and_standard_error(
        [float(row["target_minus_carrier_energy_per_site"]) for row in physical_rows]
    )
    carrier_variance_delta, carrier_variance_delta_se = _mean_and_standard_error(
        [float(row["target_minus_carrier_variance_per_site"]) for row in physical_rows]
    )
    geometry_delta, geometry_delta_se = _mean_and_standard_error(
        [
            float(left["target"]["energy"] - right["target"]["energy"]) / 36.0
            for left, right in zip(physical_rows, scrambled_rows, strict=True)
        ]
    )
    geometry_variance_delta, geometry_variance_delta_se = _mean_and_standard_error(
        [
            float(left["target"]["variance"] - right["target"]["variance"]) / 36.0
            for left, right in zip(physical_rows, scrambled_rows, strict=True)
        ]
    )
    gates = spec["gates"]
    maximum_nesting_error = max(
        float(row["zero_residual_carrier_max_abs_error"])
        for arm in ARM_IDS
        for row in rows[arm]
    )
    minimum_ess = min(
        float(row["target"]["ess_fraction"])
        for arm in ARM_IDS
        for row in rows[arm]
    )
    support = {
        arm: float(
            np.mean(
                [
                    row["native_residual_local_f_audit"]["multi_channel_fraction"]
                    for row in rows[arm]
                ]
            )
        )
        for arm in ARM_IDS
    }
    checkpoint_gate = all(
        arm_reports[arm]["first_invocation_completed_steps"] == 1
        and arm_reports[arm]["second_invocation_completed_steps"]
        == int(training["total_steps"])
        and arm_reports[arm]["second_invocation_resumed"]
        and arm_reports[arm]["checkpoint_restore_matches_final"]
        for arm in ARM_IDS
    )
    exact_gate = bool(
        maximum_nesting_error <= float(gates["maximum_zero_residual_nesting_error"])
        and all(
            row["finite_target_h_ratios"]
            and row["native_residual_local_f_audit"]["maximum_f_column_norm_error"]
            <= float(gates["maximum_f_column_norm_error"])
            for arm in ARM_IDS
            for row in rows[arm]
        )
    )
    carrier_gate = bool(
        carrier_delta <= float(gates["maximum_carrier_energy_penalty_per_site"])
        and carrier_variance_delta
        <= float(gates["maximum_carrier_variance_penalty_per_site"])
    )
    geometry_gate = bool(
        geometry_delta <= float(gates["maximum_physical_minus_scramble_energy_per_site"])
    )
    support_gate = all(
        value >= float(gates["minimum_multi_channel_fraction"])
        for value in support.values()
    )
    ess_gate = minimum_ess >= float(gates["minimum_audit_ess_fraction"])
    local_pass = bool(
        exact_gate and checkpoint_gate and carrier_gate and geometry_gate and support_gate and ess_gate
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": config_hash,
        "carrier_parameter_artifact": str(artifact),
        "carrier_sha256": pytree_sha256(carrier),
        "hamiltonian_bond_count": len(bonds),
        "arms": arm_reports,
        "untouched_common_carrier_audit": rows,
        "paired_summary": {
            "physical_minus_carrier_energy_per_site_mean": carrier_delta,
            "physical_minus_carrier_energy_per_site_standard_error": carrier_delta_se,
            "physical_minus_carrier_variance_per_site_mean": carrier_variance_delta,
            "physical_minus_carrier_variance_per_site_standard_error": carrier_variance_delta_se,
            "physical_minus_scramble_energy_per_site_mean": geometry_delta,
            "physical_minus_scramble_energy_per_site_standard_error": geometry_delta_se,
            "physical_minus_scramble_variance_per_site_mean": geometry_variance_delta,
            "physical_minus_scramble_variance_per_site_standard_error": geometry_variance_delta_se,
        },
        "aggregate_multi_channel_fraction": support,
        "minimum_audit_ess_fraction": minimum_ess,
        "maximum_zero_residual_nesting_error": maximum_nesting_error,
        "gates": {
            "exact_additive_nesting_and_local_f": exact_gate,
            "joint_backbone_changed": bool(
                all(arm_reports[arm]["backbone_changed"] for arm in ARM_IDS)
            ),
            "checkpoint_resume": checkpoint_gate,
            "carrier_energy_and_variance_retained": carrier_gate,
            "physical_residual_beats_scramble": geometry_gate,
            "multi_channel_support": support_gate,
            "importance_ess": ess_gate,
            "local_additive_joint_gate_passed": local_pass,
            "remote_production_authorized": False,
        },
        "decision": (
            "license_stronger_common_basis_carrier_integration"
            if local_pass
            else "hold_and_change_carrier_or_residual_initialization"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Laptop-scale common-proposal gate. It tests whether a jointly trainable "
            "simplex residual can retain its exact nested carrier and beat a spatial "
            "scramble. The carrier is the existing reduced-basis plaquette-distilled "
            "state, not the physical pi-flux/two-Lanczos frontier, so passing is not "
            "a variational or remote-production claim."
        ),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["ARM_IDS", "TASK", "_validate", "run_spin_n36_additive_joint_simplex_gate"]
