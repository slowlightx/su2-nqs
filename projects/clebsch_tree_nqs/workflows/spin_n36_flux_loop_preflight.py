"""Seal the bounded triangular-6x6 flux/loop support preflight."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.flux_loop_n36_preflight import (
    triangular_n36_flux_loop_preflight,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_flux_loop_preflight"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 flux-loop preflight identity")
    if spec.get("execution_tier") != "bounded_local_cpu_preflight":
        raise ValueError("execution_tier must remain bounded_local_cpu_preflight")
    if (
        spec.get("sealed_exact_small_result")
        != "projects/clebsch_tree_nqs/results/"
        "spin_flux_loop_interference_confirmation_v1/results.json"
    ):
        raise ValueError("sealed exact-small prerequisite changed")
    sampler = spec.get("sampler")
    action = spec.get("action")
    gates = spec.get("gates")
    if not all(isinstance(value, dict) for value in (sampler, action, gates)):
        raise TypeError("sampler, action, and gates must be objects")
    if sampler != {
        "chain_count": 16,
        "burn_sweeps": 64,
        "measurement_blocks": 4,
        "decorrelation_sweeps": 8,
    }:
        raise ValueError("bounded sampler protocol differs from registration")
    if action != {
        "center_count": 8,
        "center_batch_size": 2,
        "amplitude_microbatch_size": 32,
        "token_labels": [
            "J1_shell",
            "J2_shell",
            "rhombus_dimer_resonance",
        ],
    }:
        raise ValueError("action protocol differs from registration")
    if gates != {
        "minimum_sample_rank": 4,
        "minimum_importance_ess_fraction": 0.02,
        "maximum_exact_node_fraction": 0.1,
        "minimum_median_cancellation_ratio": 0.00001,
        "maximum_gamma_a1_residual": 1e-10,
        "minimum_dimer_ratio_standard_deviation": 0.05,
        "minimum_dimer_orthogonal_fraction": 0.05,
    }:
        raise ValueError("preflight gates differ from registration")
    if spec.get("claim_boundary") != {
        "variational_energy_result": False,
        "gpu_production_allowed": False,
        "matched_energy_pilot_requires_both_preflight_gates": True,
    }:
        raise ValueError("claim boundary differs from registration")


def _verified_prerequisite(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    recorded = payload.get("result_sha256")
    body = dict(payload)
    body.pop("result_sha256", None)
    if not isinstance(recorded, str) or canonical_sha256(body) != recorded:
        raise ValueError("exact-small prerequisite seal is invalid")
    decision = payload.get("suite", {}).get("decision", {})
    if (
        decision.get("cross_geometry_mechanism_gate_passed") is not True
        or decision.get("n36_sampled_preflight_licensed") is not True
        or decision.get("n36_gpu_production_licensed") is not False
    ):
        raise ValueError("exact-small prerequisite does not license this preflight")
    return payload


def run_spin_n36_flux_loop_preflight(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    prerequisite = _verified_prerequisite(ROOT / str(spec["sealed_exact_small_result"]))
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    sampler = spec["sampler"]
    action = spec["action"]
    result = triangular_n36_flux_loop_preflight(
        seed=int(spec["seed"]),
        chain_count=int(sampler["chain_count"]),
        burn_sweeps=int(sampler["burn_sweeps"]),
        measurement_blocks=int(sampler["measurement_blocks"]),
        decorrelation_sweeps=int(sampler["decorrelation_sweeps"]),
        action_center_count=int(action["center_count"]),
        amplitude_microbatch_size=int(action["amplitude_microbatch_size"]),
        center_batch_size=int(action["center_batch_size"]),
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
        "preflight": result,
        "promotion_decision": (
            "license_small_matched_energy_pilot"
            if licensed
            else "hold_and_revise_phase_aware_carrier_or_action"
        ),
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


__all__ = ["TASK", "run_spin_n36_flux_loop_preflight"]
