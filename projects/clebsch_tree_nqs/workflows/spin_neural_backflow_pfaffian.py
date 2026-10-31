"""Seal the triangular-N16 symmetry-projected neural-backflow gate."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.neural_backflow_pfaffian import (
    neural_backflow_pfaffian_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_neural_backflow_pfaffian"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid neural-backflow Pfaffian identity")
    if spec.get("preregistered_before_training") is not True:
        raise ValueError("the exact-support comparison must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic" or abs(
        float(spec.get("j2_over_j1", -1.0)) - 0.125
    ) > 1.0e-14:
        raise ValueError("only triangular 4x4 at J2/J1=0.125 is registered")
    sources = spec.get("sealed_sources")
    architecture = spec.get("architecture")
    optimization = spec.get("optimization")
    gates = spec.get("gates")
    boundary = spec.get("claim_boundary")
    if not all(
        isinstance(value, dict)
        for value in (sources, architecture, optimization, gates, boundary)
    ):
        raise TypeError("registered protocol sections must be objects")
    if set(sources) != {"qsl_states", "a1_oracle"}:
        raise ValueError("the two sealed source roles are fixed")
    if architecture != {
        "raw_carrier": "configuration_dependent_real_singlet_pfaffian",
        "projection": "sealed_complete_S0_Gamma_A1_basis",
        "compact_model": "4_parameter_J1_relational_backflow",
        "contingency_model": "6_parameter_J1_J2_relational_backflow",
        "compact_matched_krylov_dimension": 5,
        "contingency_matched_krylov_dimension": 7,
        "scrambled_control": "same_model_with_fixed_site_scrambled_relations",
    }:
        raise ValueError("neural-backflow architecture differs from registration")
    if optimization != {
        "seeds": [2026092501, 2026092502, 2026092503, 2026092504],
        "scramble_seed": 2026092510,
        "initial_scale": 0.12,
        "maximum_iterations": 200,
        "parameter_bound": 3.0,
        "optimizer": "L-BFGS-B_exact_energy_and_gradient",
    }:
        raise ValueError("neural-backflow optimization differs from registration")
    if gates != {
        "minimum_base_carrier_fidelity": 0.9999999999,
        "minimum_resolved_energy_gain_per_site": 1e-10,
        "maximum_spin_raising_relative_norm": 1e-11,
        "majority_seed_rule": "at_least_3_of_4",
        "n36_requires_energy_and_variance_win_over_all_registered_controls": True,
    }:
        raise ValueError("neural-backflow gates differ from registration")
    if boundary != {
        "repeat_ed": False,
        "exact_ground_role": "post_training_audit_only",
        "n36_closed_unless_variational_win": True,
        "gpu_production_allowed": False,
    }:
        raise ValueError("neural-backflow claim boundary differs from registration")


def _verified_source(cell: dict[str, object]) -> tuple[dict[str, object], Path]:
    result_path = ROOT / str(cell["result"])
    bundle_path = ROOT / str(cell["bundle"])
    payload = json.loads(result_path.read_text())
    recorded = payload.get("result_sha256")
    body = dict(payload)
    body.pop("result_sha256", None)
    if recorded != cell["result_sha256"] or canonical_sha256(body) != recorded:
        raise ValueError(f"sealed source result is invalid: {result_path}")
    if file_sha256(bundle_path) != cell["bundle_sha256"]:
        raise ValueError(f"sealed source bundle is invalid: {bundle_path}")
    return payload, bundle_path


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def run_spin_neural_backflow_pfaffian(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    qsl_result, qsl_bundle_path = _verified_source(spec["sealed_sources"]["qsl_states"])
    _a1_result, a1_bundle_path = _verified_source(spec["sealed_sources"]["a1_oracle"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    with np.load(qsl_bundle_path) as loaded:
        qsl = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    with np.load(a1_bundle_path) as loaded:
        a1 = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    diagnostics = qsl_result["diagnostics"]["variational"]
    baselines = {
        "projected_vmc_two_lanczos": {
            "energy_per_site": float(
                diagnostics["optimized_pi_flux_jastrow_two_lanczos"]["energy_per_site"]
            ),
            "variance_per_site": float(
                diagnostics["optimized_pi_flux_jastrow_two_lanczos"]["variance_per_site"]
            ),
        },
        "fixed_sz_spatial_vit": {
            "energy_per_site": float(
                diagnostics["pi_flux_fixed_sz_gamma_a1_spatial_vit"]["energy_per_site"]
            ),
            "variance_per_site": float(
                diagnostics["pi_flux_fixed_sz_gamma_a1_spatial_vit"]["variance_per_site"]
            ),
        },
        "exact_su2_operator_vit": {
            "energy_per_site": float(
                diagnostics["pi_flux_exact_su2_a1_operator_vit"]["energy_per_site"]
            ),
            "variance_per_site": float(
                diagnostics["pi_flux_exact_su2_a1_operator_vit"]["variance_per_site"]
            ),
        },
    }
    optimization = spec["optimization"]
    suite, arrays = neural_backflow_pfaffian_frontier(
        configurations=qsl["basis"],
        a1_basis=a1["physical_a1_basis"],
        hamiltonian_a1=a1["coupling_0.125_h_a1"],
        exact_state_physical=qsl["exact_ground"],
        base_carrier_a1=a1["carrier_a1"],
        projected_vmc_physical=qsl["optimized_pi_flux_jastrow_two_lanczos"],
        frozen_baselines=baselines,
        seeds=tuple(int(value) for value in optimization["seeds"]),
        scramble_seed=int(optimization["scramble_seed"]),
        initial_scale=float(optimization["initial_scale"]),
        maximum_iterations=int(optimization["maximum_iterations"]),
        parameter_bound=float(optimization["parameter_bound"]),
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
            else "freeze_projected_neural_backflow_pfaffian_route"
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


__all__ = ["TASK", "run_spin_neural_backflow_pfaffian"]
