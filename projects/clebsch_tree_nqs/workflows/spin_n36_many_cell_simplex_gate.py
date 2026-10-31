"""Local N=36 gate for the F-move virtual-intertwiner/simplex ansatz."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.many_cell_schur_vit import init_many_cell_schur_vit
from su2_nqs.models.many_cell_simplex_vit import (
    ManyCellSimplexViTConfig,
    frozen_backbone_simplex_log_amplitudes,
    init_many_cell_simplex_head,
    many_cell_simplex_geometry_features,
    sample_frozen_backbone_simplex_vit,
)
from su2_nqs.su2.many_cell_schur import (
    scramble_simplex_coordinates,
    triangular_six_by_six_simplex_layout,
    validate_layout_state,
)
from su2_nqs.vmc.many_cell_simplex import (
    audit_sampled_local_f_moves,
    evaluate_many_cell_simplex_local_energy,
)
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity
from su2_nqs.vmc.matched_work_sr import (
    MatchedWorkSRControls,
    SampledLocalEnergyBatch,
    run_resumable_matched_work_sr,
)
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_many_cell_simplex_gate"
ARM_IDS = ("physical_simplex", "scrambled_geometry")


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid many-cell simplex gate identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the local physical-versus-scramble gate must be preregistered")
    system = spec.get("system")
    model = spec.get("model")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not all(isinstance(value, dict) for value in (system, model, controls, gates)):
        raise TypeError("system, model, controls, and gates must be objects")
    if system.get("geometry") != "triangular" or (
        int(system.get("rows", 0)), int(system.get("columns", 0))
    ) != (6, 6):
        raise ValueError("the first simplex gate is fixed to triangular 6x6")
    if model.get("magnetic_indices_visible") is not False:
        raise ValueError("magnetic indices may not enter the learned model")
    permutation = controls.get("scramble_cell_permutation")
    if not isinstance(permutation, list) or sorted(permutation) != list(range(9)):
        raise ValueError("scramble_cell_permutation must permute nine cells")
    if permutation == list(range(9)):
        raise ValueError("scramble_cell_permutation must be nontrivial")
    for name in (
        "total_steps",
        "sample_count",
        "cg_max_iterations",
        "checkpoint_every_steps",
        "audit_sample_count",
    ):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controls.{name} must be a positive integer")
    seeds = controls.get("audit_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("at least three integer audit_seeds are required")


def _model_config(
    spec: dict[str, object], *, scrambled: bool
) -> ManyCellSimplexViTConfig:
    model = spec["model"]
    controls = spec["controls"]
    layout = triangular_six_by_six_simplex_layout()
    if scrambled:
        layout = scramble_simplex_coordinates(
            layout, controls["scramble_cell_permutation"]
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


def _runtime(
    config: ManyCellSimplexViTConfig,
    backbone: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
) -> tuple[
    Callable[[object, jax.Array, int], jax.Array],
    Callable[[object, jax.Array], SampledLocalEnergyBatch],
    Callable[[object, jax.Array], jax.Array],
]:
    def sample_fn(head, key, count):
        return sample_frozen_backbone_simplex_vit(
            head, key, count, config, backbone
        )

    def log_fn(head, labels):
        return frozen_backbone_simplex_log_amplitudes(
            head, labels, config, backbone
        )

    def local_fn(head, labels):
        workload = build_many_cell_tree_connectivity(
            np.asarray(labels), config.base_config, bonds
        )
        evaluated = evaluate_many_cell_simplex_local_energy(
            head, backbone, workload, config
        )
        return SampledLocalEnergyBatch(
            local_energies=evaluated.local_energies,
            bond_count=len(bonds),
            evaluation_seconds=(
                workload.connectivity_seconds + evaluated.amplitude_seconds
            ),
        )

    return sample_fn, local_fn, log_fn


def _audit(
    head: dict[str, jax.Array],
    *,
    config: ManyCellSimplexViTConfig,
    backbone: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    seed: int,
    count: int,
) -> dict[str, object]:
    samples = np.asarray(
        sample_frozen_backbone_simplex_vit(
            head, jax.random.PRNGKey(seed), count, config, backbone
        )
    )
    for row in samples:
        validate_layout_state(config.simplex_layout.schur, row, target_q=0)
    logs = np.asarray(
        frozen_backbone_simplex_log_amplitudes(
            head, samples, config, backbone
        )
    )
    workload = build_many_cell_tree_connectivity(
        samples, config.base_config, bonds
    )
    evaluated = evaluate_many_cell_simplex_local_energy(
        head, backbone, workload, config
    )
    local = np.asarray(evaluated.local_energies, dtype=np.complex128)
    mean = np.mean(local)
    variance = float(np.mean(np.abs(local - mean) ** 2))
    valid_ratios = np.asarray(evaluated.ratios)[workload.mask]
    f_audit = audit_sampled_local_f_moves(samples, config)
    return {
        "seed": int(seed),
        "sample_count": int(count),
        "all_labels_valid": True,
        "all_log_amplitudes_finite": bool(np.all(np.isfinite(logs))),
        "maximum_log_probability": float(np.max(2.0 * logs.real)),
        "energy_total": float(mean.real),
        "energy_per_site": float(mean.real / 36.0),
        "energy_imaginary_abs": float(abs(mean.imag)),
        "variance_total": variance,
        "variance_per_site": float(variance / 36.0),
        "maximum_connectivity_width": int(np.max(workload.widths)),
        "mean_connectivity_width": float(np.mean(workload.widths)),
        "finite_h_ratios": bool(np.all(np.isfinite(valid_ratios))),
        "maximum_h_ratio_magnitude": float(
            np.max(np.abs(valid_ratios), initial=0.0)
        ),
        "full_h_bond_terms": int(count * len(bonds)),
        "connectivity_seconds": float(workload.connectivity_seconds),
        "amplitude_ratio_seconds": float(evaluated.amplitude_seconds),
        "local_f_move_audit": f_audit,
    }


def _paired_summary(
    physical: list[dict[str, object]],
    scrambled: list[dict[str, object]],
) -> dict[str, float | int]:
    energy = np.asarray(
        [
            float(left["energy_per_site"]) - float(right["energy_per_site"])
            for left, right in zip(physical, scrambled, strict=True)
        ]
    )
    variance = np.asarray(
        [
            float(left["variance_per_site"]) - float(right["variance_per_site"])
            for left, right in zip(physical, scrambled, strict=True)
        ]
    )

    def standard_error(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    return {
        "pair_count": int(len(energy)),
        "physical_minus_scramble_energy_per_site_mean": float(np.mean(energy)),
        "physical_minus_scramble_energy_per_site_standard_error": standard_error(energy),
        "physical_minus_scramble_variance_per_site_mean": float(np.mean(variance)),
        "physical_minus_scramble_variance_per_site_standard_error": standard_error(variance),
        "physical_energy_win_count": int(np.count_nonzero(energy < 0.0)),
        "physical_joint_win_count": int(
            np.count_nonzero((energy < 0.0) & (variance <= 0.0))
        ),
    }


def run_spin_n36_many_cell_simplex_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the local amplitude/H-ratio/checkpoint/geometry gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    if overwrite and (output / "checkpoints").exists():
        raise FileExistsError(
            "refusing to overwrite checkpoint generations; select a fresh output directory"
        )

    physical_config = _model_config(spec, scrambled=False)
    scrambled_config = _model_config(spec, scrambled=True)
    template = init_many_cell_schur_vit(
        jax.random.PRNGKey(0), physical_config.base_config
    )
    artifact = _resolve(config_path, spec["backbone_parameter_artifact"])
    backbone = load_parameter_artifact(artifact, template)
    if pytree_sha256(backbone) != pytree_sha256(
        load_parameter_artifact(
            artifact,
            init_many_cell_schur_vit(
                jax.random.PRNGKey(1), scrambled_config.base_config
            ),
        )
    ):
        raise RuntimeError("physical and scrambled controls did not share a backbone")

    system = spec["system"]
    bonds = tuple(
        triangular_hnf_j1j2_bonds(
            ((6, 0), (0, 6)),
            j1=1.0,
            j2=float(system["j2_over_j1"]),
        )
    )
    controls_spec = spec["controls"]
    controls = MatchedWorkSRControls(
        total_steps=int(controls_spec["total_steps"]),
        sample_count=int(controls_spec["sample_count"]),
        seed=int(controls_spec["training_seed"]),
        damping=float(controls_spec["damping"]),
        cg_tolerance=float(controls_spec["cg_tolerance"]),
        cg_max_iterations=int(controls_spec["cg_max_iterations"]),
        step_size=float(controls_spec["step_size"]),
        maximum_relative_parameter_step=float(
            controls_spec["maximum_relative_parameter_step"]
        ),
        checkpoint_every_steps=int(controls_spec["checkpoint_every_steps"]),
    )
    arm_results = {}
    final_heads = {}
    for arm_index, (arm_id, model_config) in enumerate(
        zip(ARM_IDS, (physical_config, scrambled_config), strict=True)
    ):
        initial_head = init_many_cell_simplex_head(
            jax.random.PRNGKey(int(controls_spec["head_seed"])), model_config
        )
        sample_fn, local_fn, log_fn = _runtime(model_config, backbone, bonds)
        identity = {
            "task": TASK,
            "config_sha256": canonical_sha256(spec),
            "arm_id": arm_id,
            "backbone_sha256": pytree_sha256(backbone),
            "head_initial_sha256": pytree_sha256(initial_head),
            "training_seed_offset": arm_index * 0,
        }
        checkpoint_directory = output / "checkpoints" / arm_id
        first = run_resumable_matched_work_sr(
            initial_head,
            sample_fn=sample_fn,
            local_energy_fn=local_fn,
            log_amplitude_fn=log_fn,
            controls=controls,
            checkpoint_directory=checkpoint_directory,
            run_identity=identity,
            maximum_steps_this_call=1,
        )
        second = run_resumable_matched_work_sr(
            initial_head,
            sample_fn=sample_fn,
            local_energy_fn=local_fn,
            log_amplitude_fn=log_fn,
            controls=controls,
            checkpoint_directory=checkpoint_directory,
            run_identity=identity,
        )
        restored = load_latest_resumable_checkpoint(
            checkpoint_directory, expected_run_identity=second["run_identity"]
        )
        final_heads[arm_id] = second["params"]
        arm_results[arm_id] = {
            "initial_head_sha256": pytree_sha256(initial_head),
            "final_head_sha256": pytree_sha256(second["params"]),
            "first_invocation_completed_steps": int(first["completed_steps"]),
            "second_invocation_completed_steps": int(second["completed_steps"]),
            "second_invocation_resumed": bool(second["resumed"]),
            "checkpoint_restore_matches_final": bool(
                pytree_sha256(restored["params"])
                == pytree_sha256(second["params"])
            ),
            "history": second["history"],
            "cumulative_samples": int(second["cumulative_samples"]),
            "cumulative_full_h_bond_terms": int(
                second["cumulative_full_h_bond_terms"]
            ),
            "cumulative_wall_seconds": float(second["cumulative_wall_seconds"]),
        }

    audit_rows = {arm: [] for arm in ARM_IDS}
    for seed in controls_spec["audit_seeds"]:
        audit_rows[ARM_IDS[0]].append(
            _audit(
                final_heads[ARM_IDS[0]],
                config=physical_config,
                backbone=backbone,
                bonds=bonds,
                seed=int(seed),
                count=int(controls_spec["audit_sample_count"]),
            )
        )
        audit_rows[ARM_IDS[1]].append(
            _audit(
                final_heads[ARM_IDS[1]],
                config=scrambled_config,
                backbone=backbone,
                bonds=bonds,
                seed=int(seed),
                count=int(controls_spec["audit_sample_count"]),
            )
        )
    paired = _paired_summary(audit_rows[ARM_IDS[0]], audit_rows[ARM_IDS[1]])
    gates_spec = spec["gates"]
    exact_algebra_gate = all(
        row["all_labels_valid"]
        and row["all_log_amplitudes_finite"]
        and row["finite_h_ratios"]
        and float(row["local_f_move_audit"]["maximum_f_column_norm_error"])
        <= float(gates_spec["maximum_f_column_norm_error"])
        for arm in ARM_IDS
        for row in audit_rows[arm]
    )
    support_by_arm = {
        arm: float(
            np.mean(
                [
                    row["local_f_move_audit"]["multi_channel_fraction"]
                    for row in audit_rows[arm]
                ]
            )
        )
        for arm in ARM_IDS
    }
    support_gate = all(
        value >= float(gates_spec["minimum_multi_channel_fraction"])
        for value in support_by_arm.values()
    )
    exact_gate = bool(exact_algebra_gate and support_gate)
    checkpoint_gate = all(
        arm_results[arm]["first_invocation_completed_steps"] == 1
        and arm_results[arm]["second_invocation_completed_steps"]
        == int(controls_spec["total_steps"])
        and arm_results[arm]["second_invocation_resumed"]
        and arm_results[arm]["checkpoint_restore_matches_final"]
        for arm in ARM_IDS
    )
    sigma = float(gates_spec["required_paired_standard_errors"])
    energy_gate = (
        float(paired["physical_minus_scramble_energy_per_site_mean"])
        + sigma
        * float(paired["physical_minus_scramble_energy_per_site_standard_error"])
        < 0.0
    )
    variance_gate = (
        float(paired["physical_minus_scramble_variance_per_site_mean"])
        - float(gates_spec["variance_nonworsening_standard_errors"])
        * float(paired["physical_minus_scramble_variance_per_site_standard_error"])
        <= float(gates_spec["maximum_variance_penalty_per_site"])
    )
    physical_gate = bool(energy_gate and variance_gate)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "backbone_parameter_artifact": str(artifact),
        "backbone_sha256": pytree_sha256(backbone),
        "hamiltonian_bond_count": len(bonds),
        "physical_geometry_features": many_cell_simplex_geometry_features(
            physical_config
        ).tolist(),
        "scrambled_geometry_features": many_cell_simplex_geometry_features(
            scrambled_config
        ).tolist(),
        "arms": arm_results,
        "untouched_audit": audit_rows,
        "paired_audit": paired,
        "aggregate_multi_channel_fraction": support_by_arm,
        "gates": {
            "sampled_amplitude_and_local_f_move": bool(exact_gate),
            "exact_local_f_algebra": bool(exact_algebra_gate),
            "aggregate_multi_channel_support": bool(support_gate),
            "finite_full_h_ratios": bool(
                all(row["finite_h_ratios"] for arm in ARM_IDS for row in audit_rows[arm])
            ),
            "checkpoint_resume": bool(checkpoint_gate),
            "physical_energy": bool(energy_gate),
            "physical_variance": bool(variance_gate),
            "physical_beats_scramble": physical_gate,
            "local_prototype_complete": bool(exact_gate and checkpoint_gate),
            "remote_production_authorized": False,
        },
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "This is a laptop-scale N=36 sampled mechanism gate. It validates "
            "normalized reduced amplitudes, exact local-F Hamiltonian ratios, "
            "and checkpoint replay. A physical-versus-scramble win, if present, "
            "does not replace the later fixed-Sz/common-GPU-hours comparison."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = ["ARM_IDS", "TASK", "_validate", "run_spin_n36_many_cell_simplex_gate"]
