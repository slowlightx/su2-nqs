"""Matrix-free exact-state SR canary for the fixed-``Sz`` Spatial-ViT.

This is an optimizer diagnosis, not a second architecture frontier.  It starts
from a preregistered initialization that collapsed under exact-support Adam
and asks whether the pure-state tangent metric rescues the same ansatz without
forming a dense Hamiltonian or a dense quantum-geometric tensor.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import coo_matvec, reynolds_project_state
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.models.fixed_sz_spatial_vit import (
    fixed_sz_spatial_vit_amplitudes,
    fixed_sz_spatial_vit_parameter_count,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.vmc.exact_state_sr import train_exact_state_sr

from projects.clebsch_tree_nqs.workflows.spin import _metrics, _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit import (
    _distance_diagnostics,
    _model_config,
    _projected_normalized_state,
    _spin_projection_diagnostics,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "fixed_sz_spatial_vit_matrix_free_sr_canary"


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("Spatial-ViT SR canary requires schema_version=1 and spin")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    for key in ("steps", "cg_max_iterations", "record_every"):
        if int(optimization.get(key, 0)) < 1:
            raise ValueError(f"optimization.{key} must be positive")
    for key in ("learning_rate", "damping", "cg_tolerance"):
        value = float(optimization.get(key, 0.0))
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"optimization.{key} must be positive and finite")
    if int(spec.get("seed", -1)) < 0:
        raise ValueError("seed must be nonnegative")
    arm = spec.get("arm")
    if not isinstance(arm, dict) or not bool(arm.get("use_distance_envelope")):
        raise ValueError("the canary requires one distance-aware Spatial-ViT arm")
    if int(arm.get("embedding_dim", 0)) % int(arm.get("head_count", 1)):
        raise ValueError("embedding_dim must be divisible by head_count")
    model = spec.get("model")
    if not isinstance(model, dict) or (
        int(model.get("patch_rows", 0)), int(model.get("patch_columns", 0))
    ) != (2, 2):
        raise ValueError("the 4x4 canary requires 2x2 patches")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        if case.get("geometry") not in {"square", "triangular"}:
            raise ValueError("case geometry must be square or triangular")
        if (case.get("rows"), case.get("columns")) != (4, 4):
            raise ValueError("the local canary is restricted to 4x4")
        if case.get("projection") != "translation":
            raise ValueError("the canary requires the translation projector")
        if "exact_energy_anchor" not in case:
            raise ValueError("every case requires a stored exact energy anchor")


def _run_case(
    spec: dict[str, object],
    case_spec: dict[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    adapted = {**case_spec, "support": "fixed_m0"}
    data = _prepare_case(adapted, "fixed_m0")
    arm = spec["arm"]
    config = _model_config(spec, arm, case_spec)
    seed = int(spec["seed"])
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(seed), config)
    basis = jnp.asarray(data["basis"], dtype=jnp.int32)
    indices = jnp.asarray(data["tables"]["translation"], dtype=jnp.int32)
    rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    values = jnp.asarray(data["h_values"], dtype=jnp.float64)

    def raw_amplitudes(candidate):
        return fixed_sz_spatial_vit_amplitudes(candidate, basis, config)

    def projected_amplitudes(candidate):
        return reynolds_project_state(raw_amplitudes(candidate), indices)

    def hamiltonian_action(state):
        return coo_matvec(rows, columns, values, state)

    initial_raw = np.asarray(raw_amplitudes(params), dtype=np.complex128)
    initial_metrics = _metrics(initial_raw, data, projection="translation")
    initial_state = _projected_normalized_state(initial_raw, np.asarray(indices))
    initial_spin = _spin_projection_diagnostics(
        initial_state,
        data,
        raw_energy=float(initial_metrics["energy"]),
    )
    initial_params_sha256 = pytree_sha256(params)
    initial_raw_sha256 = array_sha256(initial_raw)
    optimization = spec["optimization"]
    trained, training = train_exact_state_sr(
        params,
        state_fn=projected_amplitudes,
        hamiltonian_action=hamiltonian_action,
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
        damping=float(optimization["damping"]),
        cg_tolerance=float(optimization["cg_tolerance"]),
        cg_max_iterations=int(optimization["cg_max_iterations"]),
        record_every=int(optimization["record_every"]),
    )
    raw = np.asarray(raw_amplitudes(trained), dtype=np.complex128)
    metrics = _metrics(raw, data, projection="translation")
    state = _projected_normalized_state(raw, np.asarray(indices))
    spin = _spin_projection_diagnostics(
        state,
        data,
        raw_energy=float(metrics["energy"]),
    )
    if abs(float(training["final_energy"]) - float(metrics["energy"])) > 2.0e-9:
        raise RuntimeError("SR and independent physical metric energies disagree")
    gates = spec["rescue_gates"]
    gate_results = {
        "energy_error_per_site_below": bool(
            float(metrics["energy_error_per_site"])
            < float(gates["energy_error_per_site_below"])
        ),
        "singlet_weight_above": bool(
            float(spin["singlet_weight"]) > float(gates["singlet_weight_above"])
        ),
        "variance_per_site_below": bool(
            float(metrics["variance_per_site"])
            < float(gates["variance_per_site_below"])
        ),
    }
    return {
        "case_id": str(case_spec["case_id"]),
        "geometry": str(case_spec["geometry"]),
        "rows": int(case_spec["rows"]),
        "columns": int(case_spec["columns"]),
        "j2_over_j1": float(case_spec["j2_over_j1"]),
        "projection": "translation",
        "support": "fixed_m0",
        "architecture": "factored_spatial_vit",
        "optimizer_diagnosis": "matrix_free_exact_pure_state_sr",
        "seed": seed,
        "embedding_dim": config.embedding_dim,
        "head_count": config.head_count,
        "depth": config.depth,
        "parameter_count": fixed_sz_spatial_vit_parameter_count(params),
        "basis_size": len(data["basis"]),
        "hamiltonian_nnz": int(data["hamiltonian"].nnz),
        "exact_energy_anchor": float(data["exact_energy"]),
        "anchor_provenance": str(data["anchor_provenance"]),
        "initial_params_sha256": initial_params_sha256,
        "initial_raw_amplitudes_sha256": initial_raw_sha256,
        "initial_metrics": initial_metrics,
        "initial_spin_sector_diagnostics": initial_spin,
        "training": training,
        "metrics": metrics,
        "spin_sector_diagnostics": spin,
        "spatial_kernel_diagnostics": _distance_diagnostics(trained),
        "rescue_gate_results": gate_results,
        "rescue_gate_passed": all(gate_results.values()),
        "case_wall_seconds": time.perf_counter() - started,
        "work_ledger": {
            "hamiltonian_representation": "sparse_coo_action",
            "dense_hamiltonian_formed": False,
            "dense_quantum_geometric_tensor_formed": False,
            "maximum_cg_iterations_per_step": int(
                optimization["cg_max_iterations"]
            ),
        },
    }


def run_spin_spatial_vit_sr(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the preregistered optimizer-rescue canary without diagonalization."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    runtime = numerical_runtime_identity()
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "hamiltonian_diagonalization_performed": False,
        "cases": [],
        "claim_boundary": (
            "Optimizer-rescue canary on fully enumerated 4x4 fixed-M0 support. "
            "It diagnoses Adam collapse but is not a scalable VMC or efficiency "
            "claim; no dense Hamiltonian or dense metric is formed."
        ),
    }
    atomic_json(result_path, payload)
    for case_spec in spec["cases"]:
        case = _run_case(spec, case_spec)
        payload["cases"].append(case)
        atomic_json(result_path, payload)
        print(
            f"{case['case_id']}: dE/N={case['metrics']['energy_error_per_site']:.3e}, "
            f"w0={case['spin_sector_diagnostics']['singlet_weight']:.4f}, "
            f"rescue={case['rescue_gate_passed']}, "
            f"wall={case['case_wall_seconds']:.2f}s",
            flush=True,
        )
    payload["status"] = "complete"
    payload["rescue_gate_pass_count"] = sum(
        bool(case["rescue_gate_passed"]) for case in payload["cases"]
    )
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_spatial_vit_sr"]
