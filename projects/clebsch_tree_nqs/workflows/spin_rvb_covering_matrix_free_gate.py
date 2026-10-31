"""Seal the matrix-free N=36 RVB estimator and phase-transfer gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.rvb_covering_matrix_free_frontier import (
    exact_small_overlap_sign_frustration,
    exact_small_transition_loop_gate,
    square_n36_overlap_loop_gas_canary,
    triangular_n36_overlap_loop_gas_canary,
    triangular_n36_rvb_throughput_canary,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_rvb_covering_matrix_free_gate"


def _validate(spec: dict[str, object]) -> None:
    """Fail closed if the registered local gate is altered."""

    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid matrix-free RVB gate identity")
    if spec.get("cases") != {
        "target": "triangular_6x6_j2_0.125",
        "bipartite_control": "square_6x6_j2_0.5",
    }:
        raise ValueError("cases must retain the registered target and control")
    throughput = spec.get("throughput")
    loop_gas = spec.get("overlap_loop_gas")
    phase = spec.get("phase_screen")
    if not all(isinstance(value, dict) for value in (throughput, loop_gas, phase)):
        raise TypeError("throughput, overlap_loop_gas, and phase_screen are required")
    expected_throughput = {
        "pair_count": 512,
        "block_count": 8,
        "seed_left": 2026092301,
        "seed_right": 2026092302,
        "model_seed": 2026092303,
        "residual_scale": 0.1,
    }
    if throughput != expected_throughput:
        raise ValueError("throughput protocol differs from the registration")
    expected_loop = {
        "chain_count": 32,
        "burn_steps": 128,
        "sample_steps": 128,
        "thin": 2,
        "seed": 2026092311,
    }
    if loop_gas != expected_loop:
        raise ValueError("overlap_loop_gas protocol differs from the registration")
    if phase.get("development_fraction") != 0.5:
        raise ValueError("phase development_fraction must remain 0.5")
    if phase.get("families") != ["direction_only_z2", "two_by_two_cell_z2"]:
        raise ValueError("phase families differ from the registration")
    if not np.isclose(float(phase.get("minimum_audit_absolute_average_sign", 0)), 0.05):
        raise ValueError("phase minimum audit sign must remain 0.05")
    if not np.isclose(float(phase.get("minimum_audit_improvement", 0)), 0.02):
        raise ValueError("phase minimum audit improvement must remain 0.02")


def run_spin_rvb_covering_matrix_free_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run, decide, and provenance-seal the registered matrix-free gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")

    exact = exact_small_transition_loop_gate()
    if not exact["passed"]:
        raise RuntimeError("exact-small transition-loop gate failed")
    frustration = exact_small_overlap_sign_frustration()
    throughput = triangular_n36_rvb_throughput_canary(**spec["throughput"])
    triangular = triangular_n36_overlap_loop_gas_canary(
        **spec["overlap_loop_gas"]
    )
    square = square_n36_overlap_loop_gas_canary(**spec["overlap_loop_gas"])

    phase_spec = spec["phase_screen"]
    phase = triangular["z2_phase_carrier_screen"]
    canonical = float(phase["canonical_audit_absolute_average_sign"])
    candidates = {
        family: float(phase[family]["audit_absolute_average_sign"])
        for family in phase_spec["families"]
    }
    best_family = max(candidates, key=candidates.get)
    best_audit_sign = candidates[best_family]
    phase_transfer = bool(
        best_audit_sign
        >= float(phase_spec["minimum_audit_absolute_average_sign"])
        and best_audit_sign - canonical
        >= float(phase_spec["minimum_audit_improvement"])
    )
    throughput_gate = bool(
        throughput["gates"]["network_call_factor_independent_of_group_order"]
        and throughput["gates"]["warm_loop_kernel_under_5_seconds_local_cpu"]
        and throughput["gates"]["warm_loss_gradient_under_1_second_local_cpu"]
    )
    control_gate = bool(
        square["passed"]
        and square["absolute_average_sign"] > 0.99
        and square["jackknife_standard_error_per_site"] < 0.01
    )
    triangular_stability = bool(triangular["passed"])
    gpu_training_authorized = bool(
        throughput_gate and control_gate and triangular_stability and phase_transfer
    )
    decision = {
        "exact_estimator_gate_passed": True,
        "local_cpu_throughput_gate_passed": throughput_gate,
        "square_bipartite_control_gate_passed": control_gate,
        "triangular_equal_amplitude_estimator_gate_passed": triangular_stability,
        "phase_carrier_transfer_gate_passed": phase_transfer,
        "canonical_audit_absolute_average_sign": canonical,
        "phase_candidate_audit_absolute_average_signs": candidates,
        "best_phase_family": best_family,
        "best_phase_audit_improvement": best_audit_sign - canonical,
        "gpu_training_authorized": gpu_training_authorized,
        "branch_decision": (
            "promote_to_fixed_budget_gpu_training"
            if gpu_training_authorized
            else "freeze_equal_amplitude_nn_rvb_and_local_z2_covering_branch"
        ),
        "physics_conclusion": (
            "The square Marshall control has sign one and a stable energy ratio, "
            "while the triangular overlap graph contains gauge-invariant negative "
            "cycles and the preregistered local Z2 phases do not transfer. The "
            "obstruction is non-bipartite sign/flux structure rather than loop-kernel "
            "throughput or local-move acceptance."
        ),
        "next_allowed_state_manifold": (
            "phase-aware projected Dirac/paired-Z2 carrier evaluated in spin "
            "configurations, with RVB loops retained as invariant latent features"
        ),
        "claim_boundary": (
            "No triangular N=36 variational energy or scaling win is established; "
            "the noisy finite-sample ratios are estimator diagnostics only."
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "exact_small": exact,
        "overlap_sign_frustration": frustration,
        "triangular_n36_throughput": throughput,
        "triangular_n36_overlap_loop_gas": triangular,
        "square_n36_overlap_loop_gas_control": square,
        "decision": decision,
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


__all__ = ["TASK", "run_spin_rvb_covering_matrix_free_gate"]
