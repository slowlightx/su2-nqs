"""Seal the exact-N16 phase-aware long-range loop-RVB gate."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.long_range_loop_rvb import (
    long_range_loop_rvb_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_long_range_loop_rvb"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid long-range loop-RVB identity")
    if spec.get("preregistered_after_prototype_screen") is not True:
        raise ValueError("the fresh-seed loop-RVB audit must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic" or abs(
        float(spec.get("j2_over_j1", -1.0)) - 0.125
    ) > 1.0e-14:
        raise ValueError("only triangular 4x4 at J2/J1=0.125 is registered")
    if spec.get("architecture") != {
        "support": "J1_coverings_plus_at_most_one_J2_bond",
        "phase_carrier": "A1_projected_pi_zero_flux_covering_coefficients",
        "loop_features": [
            "long_bond_count",
            "four_site_alternating_resonance_count",
            "six_site_alternating_resonance_count",
        ],
        "primary": "single_PSG_three_parameter_loop_Jastrow_plus_anchor",
        "contingency": "pi_zero_flux_plus_loop_parity_plus_paired_Z2_E2_bank",
        "primary_matched_krylov_dimension": 5,
        "contingency_matched_krylov_dimension": 7,
        "scrambled_control": "paired_independent_orbit_feature_permutation",
        "projection": "sealed_complete_S0_Gamma_A1_basis",
    }:
        raise ValueError("loop-RVB architecture differs from registration")
    if spec.get("optimization") != {
        "seeds": [2026092411, 2026092412, 2026092413, 2026092414],
        "scramble_seeds": [2026092421, 2026092422, 2026092423, 2026092424],
        "initial_scale": 0.12,
        "maximum_iterations": 300,
        "parameter_bound": 3.0,
        "optimizer": "L_BFGS_B_exact_energy_and_gradient",
    }:
        raise ValueError("loop-RVB optimization differs from registration")
    if spec.get("gates") != {
        "minimum_resolved_energy_gain_per_site": 1e-10,
        "maximum_spin_raising_relative_norm": 1e-11,
        "maximum_gamma_a1_relative_residual": 1e-11,
        "minimum_paired_joint_wins": 3,
        "n36_requires_energy_and_variance_win_over_operator_vit_and_matched_krylov": True,
    }:
        raise ValueError("loop-RVB gates differ from registration")
    if spec.get("claim_boundary") != {
        "repeat_ed": False,
        "exact_ground_role": "post_training_audit_only",
        "n36_closed_unless_variational_win": True,
        "gpu_production_allowed": False,
        "no_posthoc_width_seed_or_feature_tuning": True,
    }:
        raise ValueError("loop-RVB claim boundary differs from registration")


def _verified_source(cell: dict[str, object]) -> tuple[dict[str, object], Path]:
    result_path = ROOT / str(cell["result"])
    bundle_path = ROOT / str(cell["bundle"])
    payload = json.loads(result_path.read_text())
    body = dict(payload)
    recorded = body.pop("result_sha256", None)
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


def run_spin_long_range_loop_rvb(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    qsl_result, qsl_path = _verified_source(spec["sealed_sources"]["qsl_states"])
    _a1_result, a1_path = _verified_source(spec["sealed_sources"]["a1_oracle"])
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
    suite, arrays = long_range_loop_rvb_frontier(
        configurations=qsl["basis"],
        a1_basis=a1["physical_a1_basis"],
        hamiltonian_a1=a1["coupling_0.125_h_a1"],
        anchor_physical=qsl["optimized_pi_flux_jastrow_two_lanczos"],
        exact_physical=qsl["exact_ground"],
        frozen_baselines=baselines,
        seeds=tuple(int(value) for value in optimization["seeds"]),
        scramble_seeds=tuple(
            int(value) for value in optimization["scramble_seeds"]
        ),
        initial_scale=float(optimization["initial_scale"]),
        maximum_iterations=int(optimization["maximum_iterations"]),
        parameter_bound=float(optimization["parameter_bound"]),
        minimum_resolved_energy_gain_per_site=float(
            spec["gates"]["minimum_resolved_energy_gain_per_site"]
        ),
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
            "license_bounded_n36_loop_rvb_design_only"
            if selected is not None
            else "freeze_current_long_range_loop_rvb_route"
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
    payload["source_manifest_sha256"] = canonical_sha256(
        payload["source_manifest"]
    )
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_long_range_loop_rvb"]
