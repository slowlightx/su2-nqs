"""Seal the exact-small gate for strict spatially generated Schur/CG states."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from su2_nqs.benchmarks.strict_schur_spatial import (
    strict_schur_spatial_oracle_suite,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "strict_schur_spatial_exact_gate"

PERMANENT_GATE_FIELDS = {
    "coefficient_basis_gram_error_below": (
        ("coefficient_basis_gram_error",),
        5.0e-12,
    ),
    "state_spin_squared_residual_below": (
        ("state_spin_squared_residual_ratio",),
        5.0e-12,
    ),
    "tangent_spin_squared_residual_below": (
        ("generator_tangent_spin_squared_residual_ratio",),
        2.0e-10,
    ),
    "global_rotation_residual_below": (
        (
            "global_rotation_residual_ratio",
            "generator_tangent_global_rotation_residual_ratio",
        ),
        2.0e-10,
    ),
    "physical_leg_multi_affinity_defect_below": (
        ("physical_leg_multi_affinity_defect",),
        5.0e-12,
    ),
    "translation_projection_closure_below": (
        ("translation_projection_closure_error",),
        5.0e-12,
    ),
    "generator_nesting_error_below": (
        ("generator_nested_direct_max_abs_error",),
        1.0e-14,
    ),
    "jvp_finite_difference_relative_error_below": (
        ("generator_jvp_central_difference_relative_error",),
        2.0e-8,
    ),
}


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("strict Schur-spatial gate requires schema_version=1 and spin")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    plans = spec.get("plans")
    if not isinstance(plans, list) or len(plans) != 3:
        raise ValueError("plans must register exactly N=4,6,8")
    expected = {
        "square_n4": (4, 2),
        "triangular_n6": (6, 5),
        "square_n8": (8, 14),
    }
    observed = {
        str(row.get("name")): (int(row.get("n_sites", 0)), int(row.get("rank", 0)))
        for row in plans
        if isinstance(row, dict)
    }
    if observed != expected:
        raise ValueError("registered plans must have exact ranks 2,5,14")
    permanent_gates = spec.get("permanent_gates")
    if not isinstance(permanent_gates, dict) or set(permanent_gates) != set(
        PERMANENT_GATE_FIELDS
    ):
        raise ValueError("permanent_gates must register the complete exact schema")
    for name, (_fields, ceiling) in PERMANENT_GATE_FIELDS.items():
        try:
            threshold = float(permanent_gates[name])
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric") from error
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        if threshold > ceiling:
            raise ValueError(f"{name} may not be looser than {ceiling}")
    next_gate = spec.get("next_gate")
    if not isinstance(next_gate, dict):
        raise TypeError("next_gate must be an object")
    if (
        next_gate.get("geometry") != "square"
        or (next_gate.get("rows"), next_gate.get("columns")) != (4, 4)
        or int(next_gate.get("reduced_width", 0)) != 4
    ):
        raise ValueError("next gate must be the registered square 4x4 D=4 bridge")


def run_strict_schur_spatial_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the coefficient-space gate; no Hamiltonian is diagonalized."""

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
    suite = strict_schur_spatial_oracle_suite()
    if not suite["passed"]:
        raise RuntimeError("strict Schur-spatial exact-small gate failed")
    for row in spec["plans"]:
        report = suite["reports"][row["name"]]
        if (
            int(report["n_sites"]) != int(row["n_sites"])
            or int(report["complete_singlet_rank_observed"]) != int(row["rank"])
        ):
            raise RuntimeError("oracle result disagrees with the registered rank")
        for gate_name, (report_fields, _ceiling) in PERMANENT_GATE_FIELDS.items():
            threshold = float(spec["permanent_gates"][gate_name])
            if any(float(report[field]) >= threshold for field in report_fields):
                raise RuntimeError(
                    f"{row['name']} failed registered gate {gate_name}"
                )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "hamiltonian_diagonalization_performed": False,
        "exact_small_suite": suite,
        "promotion_decision": "pass_exact_algebra_only",
        "claim_boundary": (
            "The strict support-ledger/local-Schur coefficient model and its "
            "geometry-only reduced generator pass N=4,6,8 exactness. This does "
            "not establish an N=16 energy, efficiency, scaling, or PRL win."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "PERMANENT_GATE_FIELDS",
    "run_strict_schur_spatial_gate",
]
