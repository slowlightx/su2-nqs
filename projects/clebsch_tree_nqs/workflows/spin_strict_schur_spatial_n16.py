"""Seal the square-4x4 strict Schur/CG forward, JVP, and Hpsi smoke."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from su2_nqs.benchmarks.strict_schur_spatial import four_plaquette_n16_smoke
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "strict_schur_spatial_n16_smoke"
CLAIM_BOUNDARY = (
    "Strict square-4x4 forward/JVP/Hpsi readiness only; no optimized energy, "
    "efficiency, scaling, phase, QSL, or PRL win."
)

GATE_FIELDS = {
    "spin_raising_relative_norm_below": (
        (
            "state_spin_raising_relative_norm",
            "tangent_spin_raising_relative_norm",
            "hpsi_spin_raising_relative_norm",
            "translation_projected_spin_raising_relative_norm",
            "translation_projected_hpsi_spin_raising_relative_norm",
        ),
        1.0e-11,
    ),
    "physical_leg_multi_affinity_defect_below": (
        ("physical_leg_multi_affinity_defect",),
        5.0e-12,
    ),
    "translation_projection_closure_below": (
        ("translation_projection_closure_error",),
        1.0e-11,
    ),
    "hpsi_jvp_linearity_relative_error_below": (
        ("hpsi_jvp_linearity_relative_error",),
        1.0e-11,
    ),
    "jvp_central_difference_relative_error_below": (
        (
            "state_jvp_central_difference_relative_error",
            "hpsi_jvp_central_difference_relative_error",
        ),
        2.0e-8,
    ),
    "translation_hpsi_commutator_relative_error_below": (
        (
            "translation_hpsi_commutator_relative_error",
            "translation_tangent_hpsi_commutator_relative_error",
        ),
        1.0e-11,
    ),
    "full_energy_vs_three_hz_abs_error_below": (
        ("full_energy_vs_three_hz_abs_error",),
        1.0e-11,
    ),
    "rayleigh_imaginary_abs_below": (
        ("random_state_energy_imaginary_abs",),
        1.0e-11,
    ),
}


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("N16 smoke requires schema_version=1 and domain='spin'")
    if spec.get("task") != TASK or spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("task or claim boundary does not match the N16 smoke")
    if isinstance(spec.get("seed"), bool) or not isinstance(spec.get("seed"), int):
        raise TypeError("seed must be an integer")
    case = spec.get("case")
    if not isinstance(case, dict) or case != {
        "geometry": "square",
        "rows": 4,
        "columns": 4,
        "j2_over_j1": 0.5,
        "periodic": True,
        "support": "fixed_m0",
        "basis_size": 12870,
        "ed": False,
    }:
        raise ValueError("case must be the registered square-4x4 J2/J1=0.5 smoke")
    model = spec.get("model")
    if not isinstance(model, dict) or model != {
        "local_cells": "four_complete_2x2_plaquettes",
        "level1_merges": "two_unshared_4_plus_4",
        "root_merge": "8_plus_8_to_q0",
        "reduced_width": 4,
        "raw_real_trainable_scalar_count": 619,
        "internal_gauge_nullity": 130,
        "unprojected_generic_tangent_rank": 489,
        "translation_invariant_singlet_codomain_dimension": 107,
        "bias": False,
        "extra_readout": False,
    }:
        raise ValueError("model must be the registered four-plaquette D=4 bridge")
    gates = spec.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(GATE_FIELDS):
        raise ValueError("gates must register the complete N16 smoke schema")
    for name, (_fields, ceiling) in GATE_FIELDS.items():
        try:
            threshold = float(gates[name])
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric") from error
        if not math.isfinite(threshold) or threshold <= 0.0 or threshold > ceiling:
            raise ValueError(f"{name} must be positive and no looser than {ceiling}")


def run_strict_schur_spatial_n16_smoke(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the N16 runtime smoke without diagonalization or training."""

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
    report = four_plaquette_n16_smoke(seed=int(spec["seed"]))
    if not report["passed"] or not report["finite_state_tangent_hpsi"]:
        raise RuntimeError("four-plaquette N16 runtime smoke failed")
    if (
        report["fixed_m0_basis_size"] != spec["case"]["basis_size"]
        or report["raw_trainable_scalar_count"]
        != spec["model"]["raw_real_trainable_scalar_count"]
        or report["internal_gauge_nullity"]
        != spec["model"]["internal_gauge_nullity"]
        or report["unprojected_generic_effective_tangent_dimension"]
        != spec["model"]["unprojected_generic_tangent_rank"]
        or report["translation_projected_random_tangent_rank_observed"]
        != spec["model"]["translation_invariant_singlet_codomain_dimension"]
        or report["hamiltonian_diagonalization_performed"] is not False
    ):
        raise RuntimeError("N16 report disagrees with the registered model/case")
    for gate_name, (fields, _ceiling) in GATE_FIELDS.items():
        threshold = float(spec["gates"][gate_name])
        if any(float(report[field]) >= threshold for field in fields):
            raise RuntimeError(f"N16 report failed registered gate {gate_name}")

    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": now,
        "completed_utc": now,
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "hamiltonian_diagonalization_performed": False,
        "training_performed": False,
        "report": report,
        "promotion_decision": "pass_n16_runtime_smoke_only",
        "next_gate": (
            "Build the preregistered same-backbone U1, SU2-nonspatial, "
            "SU2-physical-geometry, and SU2-scrambled-geometry energy arms."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "CLAIM_BOUNDARY",
    "GATE_FIELDS",
    "run_strict_schur_spatial_n16_smoke",
]
