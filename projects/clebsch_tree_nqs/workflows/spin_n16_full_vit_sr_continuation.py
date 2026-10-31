"""One bounded continuation of the sealed N16 trust-region SR state."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import jax.numpy as jnp
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
from su2_nqs.frontier.parameter_artifact import (
    load_parameter_artifact,
    save_parameter_artifact,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.vmc.exact_state_sr import train_exact_state_sr_trust_region


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n16_full_vit_sr_continuation"


def _sealed_result(reference: dict[str, object], expected_task: str) -> dict[str, object]:
    path = ROOT / str(reference["result"])
    payload = json.loads(path.read_text())
    seal = str(reference["result_sha256"])
    if payload.get("task") != expected_task:
        raise ValueError(f"unexpected source task: {payload.get('task')}")
    if payload.get("result_sha256") != seal or canonical_sha256(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    ) != seal:
        raise RuntimeError("source result seal changed")
    return payload


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid full-ViT SR continuation identity")
    if spec.get("development_only") is not True or spec.get(
        "final_development_optimizer_attempt"
    ) is not True:
        raise ValueError("the bounded development/stop contract changed")
    optimizer = spec.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("name") != (
        "exact_pure_state_sr_fs_trust_region"
    ):
        raise ValueError("the continuation requires the sealed trust-region optimizer")
    if int(optimizer.get("steps", 0)) < 1:
        raise ValueError("optimizer.steps must be positive")
    for name in (
        "initial_learning_rate",
        "initial_damping",
        "maximum_fs_infidelity",
        "cg_tolerance",
    ):
        if float(optimizer.get(name, 0.0)) <= 0.0:
            raise ValueError(f"optimizer.{name} must be positive")


def run_spin_n16_full_vit_sr_continuation(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Continue the one successful monotone development trajectory once."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    parameter_path = output / "sr_parameters.json"
    if (result_path.exists() or parameter_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    source_sr = _sealed_result(
        spec["source_sr"], "spin_n16_full_vit_sr_reachability"
    )
    source_gate = _sealed_result(
        spec["source_gate"], "spin_n16_krylov_anchored_full_vit"
    )
    original_source = source_gate["config"]["source"]
    _source_result, bundle_path = _load_source(original_source)
    with np.load(bundle_path, allow_pickle=False) as arrays:
        coefficient_a1 = np.asarray(arrays["coefficient_a1_basis"])
        projected_vmc = np.array(
            arrays["coupling_0.125_carrier_a1"], copy=True
        )
        two_lanczos = np.array(
            arrays["coupling_0.125_two_lanczos_a1"], copy=True
        )
        hamiltonian = np.asarray(arrays["coupling_0.125_h_a1"])
    projected_vmc /= np.linalg.norm(projected_vmc)
    two_lanczos /= np.linalg.norm(two_lanczos)
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
    seed = int(source_sr["config"]["optimization"]["seed"])
    initial_model, _fixed, model_config, _fixed_config = (
        matched_backbone_initializations(seed, **shared)
    )
    template = {
        "model": initial_model,
        "residual_scale": jnp.asarray(
            float(source_gate["config"]["optimization"]["initial_residual_scale"]),
            dtype=jnp.float64,
        ),
    }
    expected_identity = {
        "task": "spin_n16_full_vit_sr_reachability",
        "seed": seed,
        "development_only": True,
        "source_gate_sha256": source_gate["result_sha256"],
    }
    source_manifest = ROOT / str(spec["source_parameters"])
    params = load_parameter_artifact(
        source_manifest,
        template,
        expected_identity=expected_identity,
    )
    matrix = jnp.asarray(hamiltonian)

    def state_fn(candidate):
        residual = su2_projected_residual(
            candidate["model"], labels_jax, model_config, basis_jax
        )
        return anchored_unit_residual_state(
            jnp.asarray(two_lanczos), residual, candidate["residual_scale"]
        )

    initial_metrics = _metrics(np.asarray(state_fn(params)), hamiltonian)
    optimizer = spec["optimizer"]
    trained, report = train_exact_state_sr_trust_region(
        params,
        state_fn=state_fn,
        hamiltonian=matrix,
        steps=int(optimizer["steps"]),
        initial_learning_rate=float(optimizer["initial_learning_rate"]),
        initial_damping=float(optimizer["initial_damping"]),
        maximum_fs_infidelity=float(optimizer["maximum_fs_infidelity"]),
        cg_tolerance=float(optimizer["cg_tolerance"]),
        cg_max_iterations=int(optimizer["cg_max_iterations"]),
        maximum_backtracks=int(optimizer["maximum_backtracks"]),
        record_every=max(1, int(optimizer["steps"]) // 12),
    )
    final_metrics = _metrics(np.asarray(state_fn(trained)), hamiltonian)
    degree_four, rank = lowest_krylov_state(projected_vmc, hamiltonian, degree=4)
    krylov_metrics = {**_metrics(degree_four, hamiltonian), "rank": rank}
    energy_win = bool(
        final_metrics["energy_per_site"]
        < krylov_metrics["energy_per_site"] - 1.0e-10
    )
    variance_win = bool(
        final_metrics["variance_per_site"]
        < krylov_metrics["variance_per_site"] - 1.0e-10
    )
    manifest = save_parameter_artifact(
        parameter_path,
        trained,
        identity={
            "task": TASK,
            "seed": seed,
            "development_only": True,
            "source_sr_sha256": source_sr["result_sha256"],
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
        "source_sr_sha256": source_sr["result_sha256"],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "degree_four_krylov": krylov_metrics,
        "optimizer": report,
        "development_gates": {
            "energy_beats_degree_four": energy_win,
            "variance_beats_degree_four": variance_win,
            "development_passed": energy_win and variance_win,
        },
        "decision": (
            "preregister_untouched_three_seed_confirmation"
            if energy_win and variance_win
            else "freeze_current_full_vit_parameterization"
        ),
        "parameter_artifact": manifest,
        "claim_boundary": (
            "Final opened-seed optimizer attempt. Passing only licenses a "
            "fresh three-seed N16 confirmation; failure freezes this parameterization."
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


__all__ = ["TASK", "run_spin_n16_full_vit_sr_continuation"]
