"""Seal the carrier-preserving neural-Pfaffian contingency."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.residual_neural_backflow import (
    residual_neural_backflow_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_residual_neural_backflow"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid residual neural-backflow identity")
    if spec.get("preregistered_after_direct_backflow_failure") is not True:
        raise ValueError("the carrier-preserving contingency must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic" or abs(
        float(spec.get("j2_over_j1", -1.0)) - 0.125
    ) > 1.0e-14:
        raise ValueError("only triangular 4x4 at J2/J1=0.125 is registered")
    architecture = spec.get("architecture")
    optimization = spec.get("optimization")
    gates = spec.get("gates")
    boundary = spec.get("claim_boundary")
    if not all(isinstance(value, dict) for value in (architecture, optimization, gates, boundary)):
        raise TypeError("registered residual protocol sections must be objects")
    if architecture != {
        "anchor": "exact_S0_A1_projection_of_optimized_projected_VMC_two_Lanczos",
        "residual": "orthogonal_configuration_dependent_neural_pfaffian",
        "compact_parameter_count": 5,
        "extended_parameter_count": 7,
        "compact_matched_krylov_dimension": 6,
        "extended_matched_krylov_dimension": 8,
        "scrambled_control": "same_anchor_and_parameter_count_site_scrambled_relations",
    }:
        raise ValueError("residual neural-backflow architecture changed")
    if optimization != {
        "seeds": [2026092521, 2026092522, 2026092523, 2026092524],
        "scramble_seed": 2026092530,
        "initial_scale": 0.12,
        "maximum_iterations": 250,
        "parameter_bound": 3.0,
        "optimizer": "L-BFGS-B_exact_energy_and_gradient",
    }:
        raise ValueError("residual neural-backflow optimization changed")
    if gates != {
        "minimum_resolved_energy_gain_per_site": 1e-10,
        "maximum_spin_raising_relative_norm": 1e-11,
        "majority_seed_rule": "at_least_3_of_4",
    }:
        raise ValueError("residual neural-backflow gates changed")
    if boundary != {
        "repeat_ed": False,
        "exact_ground_role": "post_training_audit_only",
        "n36_closed_unless_variational_win": True,
        "gpu_production_allowed": False,
    }:
        raise ValueError("residual claim boundary changed")


def _verify_result(path: Path, expected: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    body = dict(payload)
    recorded = body.pop("result_sha256", None)
    if recorded != expected or canonical_sha256(body) != expected:
        raise ValueError(f"sealed prerequisite is invalid: {path}")
    return payload


def _verify_bundle(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"sealed bundle is invalid: {path}")


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def run_spin_residual_neural_backflow(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    sources = spec["sealed_sources"]
    direct = _verify_result(
        ROOT / sources["direct_backflow_failure"]["result"],
        sources["direct_backflow_failure"]["result_sha256"],
    )
    if direct["suite"]["decision"]["triangular_n36_reopened"] is not False:
        raise ValueError("direct backflow did not license this failure contingency")
    qsl_result = _verify_result(
        ROOT / sources["qsl_states"]["result"],
        sources["qsl_states"]["result_sha256"],
    )
    qsl_path = ROOT / sources["qsl_states"]["bundle"]
    a1_path = ROOT / sources["a1_oracle"]["bundle"]
    _verify_bundle(qsl_path, sources["qsl_states"]["bundle_sha256"])
    _verify_result(
        ROOT / sources["a1_oracle"]["result"],
        sources["a1_oracle"]["result_sha256"],
    )
    _verify_bundle(a1_path, sources["a1_oracle"]["bundle_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    with np.load(qsl_path) as loaded:
        qsl = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    with np.load(a1_path) as loaded:
        a1 = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    diagnostics = qsl_result["diagnostics"]["variational"]
    baselines = {
        "projected_vmc_two_lanczos": {
            "energy_per_site": diagnostics["optimized_pi_flux_jastrow_two_lanczos"]["energy_per_site"],
            "variance_per_site": diagnostics["optimized_pi_flux_jastrow_two_lanczos"]["variance_per_site"],
        },
        "fixed_sz_spatial_vit": {
            "energy_per_site": diagnostics["pi_flux_fixed_sz_gamma_a1_spatial_vit"]["energy_per_site"],
            "variance_per_site": diagnostics["pi_flux_fixed_sz_gamma_a1_spatial_vit"]["variance_per_site"],
        },
        "exact_su2_operator_vit": {
            "energy_per_site": diagnostics["pi_flux_exact_su2_a1_operator_vit"]["energy_per_site"],
            "variance_per_site": diagnostics["pi_flux_exact_su2_a1_operator_vit"]["variance_per_site"],
        },
    }
    optimization = spec["optimization"]
    suite, arrays = residual_neural_backflow_frontier(
        configurations=qsl["basis"],
        a1_basis=a1["physical_a1_basis"],
        hamiltonian_a1=a1["coupling_0.125_h_a1"],
        exact_state_physical=qsl["exact_ground"],
        anchor_state_physical=qsl["optimized_pi_flux_jastrow_two_lanczos"],
        frozen_baselines=baselines,
        seeds=optimization["seeds"],
        scramble_seed=optimization["scramble_seed"],
        initial_scale=optimization["initial_scale"],
        maximum_iterations=optimization["maximum_iterations"],
        parameter_bound=optimization["parameter_bound"],
        gates=spec["gates"],
    )
    _atomic_npz(state_path, arrays)
    selected = suite["decision"]["selected_model"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": (
            "license_bounded_n36_matrix_free_design_only"
            if selected is not None
            else "freeze_all_current_neural_pfaffian_backflow_routes"
        ),
        "repeated_ed_performed": False,
        "gpu_production_licensed": False,
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
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


__all__ = ["TASK", "run_spin_residual_neural_backflow"]
