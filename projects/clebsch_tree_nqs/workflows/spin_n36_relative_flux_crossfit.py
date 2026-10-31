"""Seal the cross-fitted triangular-6x6 relative-flux response gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.relative_flux_crossfit import (
    triangular_n36_relative_flux_crossfit,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_relative_flux_crossfit"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 relative-flux crossfit identity")
    if spec.get("execution_tier") != "bounded_local_cpu_crossfit":
        raise ValueError("execution_tier must remain bounded_local_cpu_crossfit")
    if (
        spec.get("sealed_n36_support_result")
        != "projects/clebsch_tree_nqs/results/"
        "spin_n36_flux_loop_preflight_v1/results.json"
    ):
        raise ValueError("sealed N36 support prerequisite changed")
    seeds = spec.get("seeds")
    sampler = spec.get("sampler")
    action = spec.get("action")
    spectral = spec.get("spectral_protocol")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not all(
        isinstance(value, dict)
        for value in (seeds, sampler, action, spectral, controls, gates)
    ):
        raise TypeError("all registered protocol sections must be objects")
    if seeds != {
        "development": 2026092402,
        "audit": 2026092403,
        "scrambled_control": 2026092404,
    }:
        raise ValueError("crossfit seeds differ from registration")
    if sampler != {
        "chain_count": 16,
        "burn_sweeps": 64,
        "measurement_blocks": 2,
        "decorrelation_sweeps": 8,
    }:
        raise ValueError("crossfit sampler protocol differs from registration")
    if action != {
        "center_count": 16,
        "center_batch_size": 2,
        "amplitude_microbatch_size": 32,
        "base_space": ["carrier_modes", "H_carrier_modes", "H2_pi_flux"],
        "response": "dimer_on_relative_flux_modes",
    }:
        raise ValueError("crossfit action protocol differs from registration")
    if spectral != {
        "relative_carrier_eigenvalue_cutoff": 0.005,
        "relative_base_singular_value_cutoff": 1e-08,
        "fit_split": "development_only",
        "mode_zero_role": "excluded_collective_mode",
    }:
        raise ValueError("crossfit spectral protocol differs from registration")
    if controls != {
        "scrambled_configuration_count": 32,
        "independent_row_permutation_per_split": True,
    }:
        raise ValueError("crossfit controls differ from registration")
    if gates != {
        "minimum_retained_mode_count": 2,
        "maximum_development_condition_number": 200.0,
        "maximum_audit_condition_number": 500.0,
        "minimum_audit_nonkrylov_residual_fraction": 0.05,
        "maximum_physical_to_scrambled_residual_ratio": 0.9,
    }:
        raise ValueError("crossfit gates differ from registration")
    if spec.get("claim_boundary") != {
        "variational_energy_result": False,
        "gpu_production_allowed": False,
        "pilot_only_if_both_gates_pass": True,
        "freeze_carrier_bank_if_response_gate_fails": True,
    }:
        raise ValueError("crossfit claim boundary differs from registration")


def _verified_prerequisite(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    recorded = payload.get("result_sha256")
    body = dict(payload)
    body.pop("result_sha256", None)
    if not isinstance(recorded, str) or canonical_sha256(body) != recorded:
        raise ValueError("N36 support prerequisite seal is invalid")
    gates = payload.get("preflight", {}).get("gates", {})
    if (
        gates.get("paired_carrier_support_gate_passed") is not True
        or gates.get("matched_energy_pilot_licensed") is not False
        or gates.get("gpu_production_licensed") is not False
    ):
        raise ValueError("N36 prerequisite does not license a response revision")
    return payload


def run_spin_n36_relative_flux_crossfit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    prerequisite = _verified_prerequisite(
        ROOT / str(spec["sealed_n36_support_result"])
    )
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    seeds = spec["seeds"]
    sampler = spec["sampler"]
    action = spec["action"]
    spectral = spec["spectral_protocol"]
    controls = spec["controls"]
    result = triangular_n36_relative_flux_crossfit(
        development_seed=int(seeds["development"]),
        audit_seed=int(seeds["audit"]),
        control_seed=int(seeds["scrambled_control"]),
        chain_count=int(sampler["chain_count"]),
        burn_sweeps=int(sampler["burn_sweeps"]),
        measurement_blocks=int(sampler["measurement_blocks"]),
        decorrelation_sweeps=int(sampler["decorrelation_sweeps"]),
        action_center_count=int(action["center_count"]),
        amplitude_microbatch_size=int(action["amplitude_microbatch_size"]),
        center_batch_size=int(action["center_batch_size"]),
        relative_eigenvalue_cutoff=float(
            spectral["relative_carrier_eigenvalue_cutoff"]
        ),
        relative_singular_value_cutoff=float(
            spectral["relative_base_singular_value_cutoff"]
        ),
        control_count=int(controls["scrambled_configuration_count"]),
        gates=spec["gates"],
    )
    licensed = bool(result["gates"]["matched_energy_pilot_licensed"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "execution_tier": spec["execution_tier"],
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "prerequisite_result_sha256": prerequisite["result_sha256"],
        "crossfit": result,
        "promotion_decision": (
            "license_one_small_matched_energy_pilot"
            if licensed
            else "freeze_current_flux_carrier_bank"
        ),
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


__all__ = ["TASK", "run_spin_n36_relative_flux_crossfit"]
