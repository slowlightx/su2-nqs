"""Development audit of full-manifold tangent rank and exact-state SR."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n16_krylov_anchored_full_vit import (
    _balanced_a1_basis,
    _load_source,
    _metrics,
)
from su2_nqs.benchmarks.krylov_anchored_full_vit import (
    anchored_unit_residual_state,
    lowest_krylov_state,
    matched_backbone_initializations,
    su2_projected_residual,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.vmc.adam import train_exact_adam
from su2_nqs.vmc.exact_state_sr import (
    train_exact_state_sr,
    train_exact_state_sr_trust_region,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n16_full_vit_sr_reachability"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid full-ViT SR reachability identity")
    if spec.get("development_only") is not True:
        raise ValueError("this opened-seed audit must remain development-only")
    source = spec.get("source_gate")
    optimization = spec.get("optimization")
    if not isinstance(source, dict) or not isinstance(optimization, dict):
        raise TypeError("source_gate and optimization must be objects")
    if int(optimization.get("seed", -1)) < 0:
        raise ValueError("optimization.seed must be nonnegative")
    for block_name in ("adam", "sr"):
        block = optimization.get(block_name)
        if not isinstance(block, dict) or int(block.get("steps", 0)) < 1:
            raise ValueError(f"optimization.{block_name}.steps must be positive")
    if float(optimization["adam"].get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("optimization.adam.learning_rate must be positive")
    sr = optimization["sr"]
    optimizer = sr.get("optimizer", "fixed_step")
    if optimizer not in {"fixed_step", "fs_trust_region"}:
        raise ValueError("optimization.sr.optimizer is unsupported")
    learning_key = (
        "learning_rate" if optimizer == "fixed_step" else "initial_learning_rate"
    )
    damping_key = "damping" if optimizer == "fixed_step" else "initial_damping"
    if float(sr.get(learning_key, 0.0)) <= 0.0:
        raise ValueError(f"optimization.sr.{learning_key} must be positive")
    if float(sr.get(damping_key, 0.0)) <= 0.0:
        raise ValueError(f"optimization.sr.{damping_key} must be positive")
    if optimizer == "fs_trust_region" and not 0.0 < float(
        sr.get("maximum_fs_infidelity", 0.0)
    ) < 1.0:
        raise ValueError("trust-region FS infidelity must lie in (0,1)")


def _load_gate(spec: dict[str, object]) -> dict[str, object]:
    path = ROOT / str(spec["result"])
    payload = json.loads(path.read_text())
    seal = str(spec["result_sha256"])
    if payload.get("result_sha256") != seal or canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    ) != seal:
        raise RuntimeError("source full-ViT gate seal changed")
    if payload.get("task") != "spin_n16_krylov_anchored_full_vit":
        raise ValueError("source result is not the full-ViT gate")
    return payload


def _tangent_audit(params, state_fn, hamiltonian: np.ndarray) -> dict[str, object]:
    flat, unravel = ravel_pytree(params)

    def coordinates(values):
        state = state_fn(unravel(values))
        state = state / jnp.linalg.norm(state)
        return jnp.concatenate((jnp.real(state), jnp.imag(state)))

    started = time.perf_counter()
    jacobian = np.asarray(jax.jit(jax.jacrev(coordinates))(flat))
    wall = time.perf_counter() - started
    left, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    cutoff = max(float(singular[0]) * 1.0e-10, 100.0 * np.finfo(float).eps)
    rank = int(np.count_nonzero(singular > cutoff))
    state = np.array(state_fn(params), dtype=np.complex128, copy=True)
    state /= np.linalg.norm(state)
    action = hamiltonian @ state
    energy = float(np.vdot(state, action).real)
    residual = action - energy * state
    residual_coordinates = np.concatenate((residual.real, residual.imag))
    projected = left[:, :rank] @ (left[:, :rank].T @ residual_coordinates)
    denominator = max(
        float(np.vdot(residual_coordinates, residual_coordinates).real),
        np.finfo(float).tiny,
    )
    capture = float(np.vdot(projected, projected).real / denominator)
    return {
        "real_normalized_state_dimension": int(jacobian.shape[0]),
        "parameter_count": int(jacobian.shape[1]),
        "numerical_tangent_rank": rank,
        "maximum_projective_rank": int(2 * len(state) - 2),
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[rank - 1]),
        "next_singular_value": (
            None if rank == len(singular) else float(singular[rank])
        ),
        "hamiltonian_residual_capture_fraction": capture,
        "wall_seconds_including_compile": wall,
    }


def run_spin_n16_full_vit_sr_reachability(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Separate N16 full-manifold capacity from Adam optimization failure."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    parameter_path = output / "sr_parameters.json"
    if (result_path.exists() or parameter_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    source_gate = _load_gate(spec["source_gate"])
    source_spec = source_gate["config"]["source"]
    _source_result, bundle_path = _load_source(source_spec)
    with np.load(bundle_path, allow_pickle=False) as arrays:
        coefficient_a1 = np.asarray(arrays["coefficient_a1_basis"])
        projected_vmc = np.asarray(arrays["coupling_0.125_carrier_a1"])
        two_lanczos = np.asarray(arrays["coupling_0.125_two_lanczos_a1"])
        hamiltonian = np.asarray(arrays["coupling_0.125_h_a1"])
    projected_vmc = projected_vmc / np.linalg.norm(projected_vmc)
    two_lanczos = two_lanczos / np.linalg.norm(two_lanczos)
    balanced_a1, labels = _balanced_a1_basis(coefficient_a1)
    labels_jax = jnp.asarray(labels, dtype=jnp.int32)
    basis_jax = jnp.asarray(balanced_a1)
    architecture = source_gate["config"]["architecture"]
    shared = {
        key: architecture[key]
        for key in (
            "embedding_dim",
            "head_count",
            "depth",
            "mlp_ratio",
            "phase_scale",
            "initialization_scale",
            "use_distance_envelope",
        )
    }
    seed = int(spec["optimization"]["seed"])
    initial_model, _fixed, model_config, _fixed_config = (
        matched_backbone_initializations(seed, **shared)
    )
    params = {
        "model": initial_model,
        "residual_scale": jnp.asarray(
            float(source_gate["config"]["optimization"]["initial_residual_scale"]),
            dtype=jnp.float64,
        ),
    }
    matrix = jnp.asarray(hamiltonian)

    def state_fn(candidate):
        residual = su2_projected_residual(
            candidate["model"], labels_jax, model_config, basis_jax
        )
        return anchored_unit_residual_state(
            jnp.asarray(two_lanczos), residual, candidate["residual_scale"]
        )

    def loss(candidate):
        state = state_fn(candidate)
        return jnp.real(jnp.vdot(state, matrix @ state) / jnp.vdot(state, state))

    adam_spec = spec["optimization"]["adam"]
    adam_params, adam_report = train_exact_adam(
        params,
        loss_fn=loss,
        steps=int(adam_spec["steps"]),
        learning_rate=float(adam_spec["learning_rate"]),
        record_every=max(1, int(adam_spec["steps"]) // 8),
    )
    adam_state = np.asarray(state_fn(adam_params))
    adam_metrics = _metrics(adam_state, hamiltonian)
    tangent = _tangent_audit(adam_params, state_fn, hamiltonian)

    sr_spec = spec["optimization"]["sr"]
    if sr_spec.get("optimizer", "fixed_step") == "fs_trust_region":
        sr_params, sr_report = train_exact_state_sr_trust_region(
            adam_params,
            state_fn=state_fn,
            hamiltonian=matrix,
            steps=int(sr_spec["steps"]),
            initial_learning_rate=float(sr_spec["initial_learning_rate"]),
            initial_damping=float(sr_spec["initial_damping"]),
            maximum_fs_infidelity=float(sr_spec["maximum_fs_infidelity"]),
            cg_tolerance=float(sr_spec["cg_tolerance"]),
            cg_max_iterations=int(sr_spec["cg_max_iterations"]),
            maximum_backtracks=int(sr_spec["maximum_backtracks"]),
            record_every=max(1, int(sr_spec["steps"]) // 8),
        )
    else:
        sr_params, sr_report = train_exact_state_sr(
            adam_params,
            state_fn=state_fn,
            hamiltonian=matrix,
            steps=int(sr_spec["steps"]),
            learning_rate=float(sr_spec["learning_rate"]),
            damping=float(sr_spec["damping"]),
            cg_tolerance=float(sr_spec["cg_tolerance"]),
            cg_max_iterations=int(sr_spec["cg_max_iterations"]),
            record_every=max(1, int(sr_spec["steps"]) // 8),
        )
    sr_state = np.asarray(state_fn(sr_params))
    sr_metrics = _metrics(sr_state, hamiltonian)
    degree_four, rank = lowest_krylov_state(projected_vmc, hamiltonian, degree=4)
    degree_four_metrics = {**_metrics(degree_four, hamiltonian), "rank": rank}
    energy_win = bool(
        sr_metrics["energy_per_site"]
        < degree_four_metrics["energy_per_site"] - 1.0e-10
    )
    variance_win = bool(
        sr_metrics["variance_per_site"]
        < degree_four_metrics["variance_per_site"] - 1.0e-10
    )
    manifest = save_parameter_artifact(
        parameter_path,
        sr_params,
        identity={
            "task": TASK,
            "seed": seed,
            "development_only": True,
            "source_gate_sha256": source_gate["result_sha256"],
        },
        overwrite=overwrite,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_gate_sha256": source_gate["result_sha256"],
        "adam": {"metrics": adam_metrics, "optimizer": adam_report},
        "tangent_audit_after_adam": tangent,
        "exact_state_sr": {"metrics": sr_metrics, "optimizer": sr_report},
        "degree_four_krylov": degree_four_metrics,
        "development_gates": {
            "full_projective_tangent_rank": (
                tangent["numerical_tangent_rank"]
                >= tangent["maximum_projective_rank"]
            ),
            "hamiltonian_residual_capture_above_0p999": (
                tangent["hamiltonian_residual_capture_fraction"] > 0.999
            ),
            "sr_energy_beats_degree_four": energy_win,
            "sr_variance_beats_degree_four": variance_win,
            "development_passed": energy_win and variance_win,
        },
        "decision": (
            "preregister_fresh_seed_sr_confirmation"
            if energy_win and variance_win
            else "state_manifold_or_parameterization_still_insufficient"
        ),
        "parameter_artifact": manifest,
        "claim_boundary": (
            "Opened-seed optimizer/capacity diagnosis only. Even a pass cannot "
            "license N36 until an untouched-seed confirmation repeats both wins."
        ),
        "repeated_many_body_ed_performed": False,
        "runtime_identity": numerical_runtime_identity(),
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


__all__ = ["TASK", "run_spin_n16_full_vit_sr_reachability"]
