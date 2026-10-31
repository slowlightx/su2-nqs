"""Diagnose whether the sealed full ViT supplies non-Krylov low-energy content."""

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
)
from su2_nqs.benchmarks.krylov_anchored_full_vit import (
    anchored_unit_residual_state,
    matched_backbone_initializations,
    su2_projected_residual,
)
from su2_nqs.benchmarks.qsl_symmetry_diagnostics import (
    triangular_qsl_symmetry_diagnostics,
)
from su2_nqs.benchmarks.qsl_three_arm import (
    triangular_pi_flux_operator_components,
)
from su2_nqs.benchmarks.spectral_error_tomography import (
    krylov_columns,
    lowest_subspace_state,
    normalized_state,
    orthogonal_weight,
    spectral_state_profile,
    state_metrics,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n16_spectral_error_tomography"


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
        raise ValueError("invalid spectral-error tomography identity")
    if spec.get("diagnostic_only") is not True:
        raise ValueError("spectral tomography must remain diagnostic only")
    for name in ("source_gate", "source_final", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    gates = spec["gates"]
    threshold = float(gates.get("minimum_non_krylov_weight", -1.0))
    if not 0.0 <= threshold < 1.0:
        raise ValueError("minimum_non_krylov_weight must lie in [0,1)")


def _with_per_site(metrics: dict[str, float], n_sites: int) -> dict[str, float]:
    return {
        **metrics,
        "energy_per_site": metrics["energy_total"] / n_sites,
        "variance_per_site": metrics["variance_total"] / n_sites,
    }


def run_spin_n16_spectral_error_tomography(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the sealed matched-rank neural-versus-Krylov complementarity gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")

    gate = _sealed_result(
        spec["source_gate"], "spin_n16_krylov_anchored_full_vit"
    )
    final = _sealed_result(
        spec["source_final"], "spin_n16_full_vit_sr_continuation"
    )
    source_result, bundle_path = _load_source(gate["config"]["source"])
    with np.load(bundle_path, allow_pickle=False) as arrays:
        coefficient_a1 = np.asarray(arrays["coefficient_a1_basis"])
        physical_a1 = np.asarray(arrays["physical_a1_basis"])
        carrier = normalized_state(arrays["coupling_0.125_carrier_a1"])
        two_lanczos = normalized_state(arrays["coupling_0.125_two_lanczos_a1"])
        hamiltonian = np.asarray(arrays["coupling_0.125_h_a1"])

    balanced_a1, labels = _balanced_a1_basis(coefficient_a1)
    architecture = gate["config"]["architecture"]
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
    identity = final["parameter_artifact"]["identity"]
    seed = int(identity["seed"])
    initial_model, _fixed, model_config, _fixed_config = (
        matched_backbone_initializations(seed, **shared)
    )
    template = {
        "model": initial_model,
        "residual_scale": jnp.asarray(
            float(gate["config"]["optimization"]["initial_residual_scale"]),
            dtype=jnp.float64,
        ),
    }
    parameter_path = ROOT / str(spec["source_final"]["parameters"])
    params = load_parameter_artifact(
        parameter_path,
        template,
        expected_identity=identity,
    )
    labels_jax = jnp.asarray(labels, dtype=jnp.int32)
    basis_jax = jnp.asarray(balanced_a1)

    residual = su2_projected_residual(
        params["model"], labels_jax, model_config, basis_jax
    )
    vit_state = normalized_state(
        np.asarray(
            anchored_unit_residual_state(
                jnp.asarray(two_lanczos), residual, params["residual_scale"]
            )
        )
    )
    n_sites = int(gate["system"]["n_sites"])

    k4_columns = krylov_columns(carrier, hamiltonian, degree=4)
    k5_columns = krylov_columns(carrier, hamiltonian, degree=5)
    k6_columns = krylov_columns(carrier, hamiltonian, degree=6)
    k4, rank4 = lowest_subspace_state(k4_columns, hamiltonian)
    k5, rank5 = lowest_subspace_state(k5_columns, hamiltonian)
    k6, rank6 = lowest_subspace_state(k6_columns, hamiltonian)
    hybrid6, hybrid_rank6 = lowest_subspace_state(
        np.column_stack((k4_columns, vit_state)), hamiltonian
    )
    hybrid7, hybrid_rank7 = lowest_subspace_state(
        np.column_stack((k4_columns, vit_state, hamiltonian @ vit_state)),
        hamiltonian,
    )

    named_states = {
        "projected_vmc": carrier,
        "two_lanczos": two_lanczos,
        "full_su2_spatial_vit": vit_state,
        "degree_four_krylov": k4,
        "degree_five_krylov": k5,
        "degree_six_krylov": k6,
        "k4_plus_vit": hybrid6,
        "k4_plus_vit_plus_hvit": hybrid7,
    }
    metrics = {
        name: _with_per_site(state_metrics(state, hamiltonian), n_sites)
        for name, state in named_states.items()
    }
    profiles = {
        name: spectral_state_profile(state, hamiltonian)
        for name, state in named_states.items()
        if name in {
            "full_su2_spatial_vit",
            "degree_four_krylov",
            "degree_five_krylov",
            "k4_plus_vit",
        }
    }
    exact_values, exact_vectors = np.linalg.eigh(
        0.5 * (hamiltonian + hamiltonian.conj().T)
    )
    exact_reduced = normalized_state(exact_vectors[:, int(np.argmin(exact_values))])
    observable_data = triangular_pi_flux_operator_components(0.125)
    if physical_a1.shape != (len(observable_data.configurations), len(carrier)):
        raise RuntimeError("sealed physical A1 basis and observable support disagree")
    reduced_replay = physical_a1.conj().T @ (
        observable_data.hamiltonian @ physical_a1
    )
    if np.max(np.abs(reduced_replay - hamiltonian)) > 2.0e-10:
        raise RuntimeError("observable Hamiltonian and sealed A1 matrix disagree")
    physical_states = {
        name: normalized_state(physical_a1 @ state)
        for name, state in named_states.items()
        if name in {
            "full_su2_spatial_vit",
            "degree_four_krylov",
            "degree_five_krylov",
            "k4_plus_vit",
        }
    }
    observable_diagnostics = triangular_qsl_symmetry_diagnostics(
        observable_data,
        physical_states,
        normalized_state(physical_a1 @ exact_reduced),
    )
    observable_errors = {
        name: float(row["median_symmetry_observable_relative_error"])
        for name, row in observable_diagnostics["variational"].items()
    }
    novelty = orthogonal_weight(vit_state, k4_columns)
    energy_tol = float(spec["gates"].get("energy_tolerance", 1.0e-12))
    variance_tol = float(spec["gates"].get("variance_tolerance", 1.0e-12))
    rank6_energy_win = bool(
        metrics["k4_plus_vit"]["energy_total"]
        < metrics["degree_five_krylov"]["energy_total"] - energy_tol
    )
    rank6_variance_win = bool(
        metrics["k4_plus_vit"]["variance_total"]
        < metrics["degree_five_krylov"]["variance_total"] - variance_tol
    )
    rank7_energy_win = bool(
        metrics["k4_plus_vit_plus_hvit"]["energy_total"]
        < metrics["degree_six_krylov"]["energy_total"] - energy_tol
    )
    rank7_variance_win = bool(
        metrics["k4_plus_vit_plus_hvit"]["variance_total"]
        < metrics["degree_six_krylov"]["variance_total"] - variance_tol
    )
    novelty_pass = bool(
        novelty >= float(spec["gates"]["minimum_non_krylov_weight"])
    )
    complementarity = bool(
        novelty_pass
        and rank6_energy_win
        and rank6_variance_win
        and rank7_energy_win
        and rank7_variance_win
    )
    vit_observable_win = bool(
        observable_errors["full_su2_spatial_vit"]
        < observable_errors["degree_four_krylov"]
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_gate_sha256": gate["result_sha256"],
        "source_final_sha256": final["result_sha256"],
        "source_small_representation_sha256": source_result["result_sha256"],
        "system": gate["system"],
        "subspace_ranks": {
            "degree_four_krylov": rank4,
            "degree_five_krylov": rank5,
            "degree_six_krylov": rank6,
            "k4_plus_vit": hybrid_rank6,
            "k4_plus_vit_plus_hvit": hybrid_rank7,
        },
        "metrics": metrics,
        "spectral_profiles": profiles,
        "qsl_observable_diagnostics": observable_diagnostics,
        "median_symmetry_observable_relative_errors": observable_errors,
        "neural_weight_outside_degree_four_krylov": novelty,
        "gates": {
            "non_krylov_weight_passed": novelty_pass,
            "rank6_hybrid_energy_beats_rank6_krylov": rank6_energy_win,
            "rank6_hybrid_variance_beats_rank6_krylov": rank6_variance_win,
            "rank7_hybrid_energy_beats_rank7_krylov": rank7_energy_win,
            "rank7_hybrid_variance_beats_rank7_krylov": rank7_variance_win,
            "neural_krylov_complementarity_passed": complementarity,
            "vit_observables_closer_than_degree_four_krylov": vit_observable_win,
        },
        "decision": (
            "preregister_scalable_alternating_neural_krylov_ansatz"
            if complementarity
            else "freeze_neural_krylov_hybrid_for_current_parameterization"
        ),
        "claim_boundary": (
            "Exact-small sealed-state diagnosis only. A pass identifies a "
            "matched-rank non-Krylov direction but does not license N36 energy training."
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


__all__ = ["TASK", "_validate", "run_spin_n16_spectral_error_tomography"]
