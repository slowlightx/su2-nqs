"""Seal the preregistered triangular A2 loop-response coupling scan."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.symmetry_labelled_loop_coupling_scan import (
    COUPLINGS,
    FRESH_SEEDS,
    HELDOUT_ORDERED_COUPLINGS,
    HELDOUT_QSL_COUPLINGS,
    symmetry_labelled_loop_coupling_scan,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_symmetry_loop_coupling_scan"
SOURCE_HASH = "e228b98ac387e9bd1dc53b475fe21ae8eb35fcbf96c1e4a4ae99ff83ac17a9ba"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid symmetry-loop coupling-scan identity")
    if spec.get("phase") != "preregistered_local_validation":
        raise ValueError("the symmetry-loop coupling scan phase changed")
    boundary = spec.get("discovery_boundary")
    if not isinstance(boundary, dict) or boundary != {
        "coupling": 0.125,
        "source_artifact": "projects/clebsch_tree_nqs/results/spin_symmetry_labelled_loop_screen_v1/results.json",
        "source_result_sha256": SOURCE_HASH,
        "use": "replication_only_not_phase_contrast",
    }:
        raise ValueError("the discovery boundary changed")
    artifact = ROOT / str(boundary["source_artifact"])
    source = json.loads(artifact.read_text())
    if source.get("result_sha256") != SOURCE_HASH:
        raise ValueError("the discovery artifact hash changed")
    model = spec.get("model")
    if not isinstance(model, dict) or model != {
        "family": "exact_SU2_Gamma_A1_two_layer_Operator_ViT",
        "operator_tokens": [
            "J1_shell",
            "J2_shell",
            "rhombus_ring_exchange",
            "rhombus_dimer_resonance",
            "six_site_loop_rearrangement",
        ],
        "response_channels": [
            "E2_rhombus_ring_return",
            "E2_rhombus_dimer_return",
            "A2_directed_ring_return",
        ],
    }:
        raise ValueError("the loop-response model or channels changed")
    scan = spec.get("scan")
    if not isinstance(scan, dict) or scan != {
        "j2_over_j1": list(COUPLINGS),
        "fresh_seeds": list(FRESH_SEEDS),
        "steps": 160,
        "learning_rate": 0.015,
        "heldout_qsl_couplings": list(HELDOUT_QSL_COUPLINGS),
        "heldout_ordered_couplings": list(HELDOUT_ORDERED_COUPLINGS),
    }:
        raise ValueError("the preregistered coupling scan changed")


def run_spin_symmetry_loop_coupling_scan(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the coupling-resolved response test."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    scan = spec["scan"]
    suite = symmetry_labelled_loop_coupling_scan(
        couplings=tuple(float(value) for value in scan["j2_over_j1"]),
        seeds=tuple(int(value) for value in scan["fresh_seeds"]),
        steps=int(scan["steps"]),
        learning_rate=float(scan["learning_rate"]),
    )
    if suite["response_signal_passed"] and suite["variational_promotion_passed"]:
        decision = "preregister_deep_A2_multiplicity_correlator"
    elif suite["response_signal_passed"]:
        decision = "retain_A2_as_response_observable_not_energy_head"
    else:
        decision = "reject_A2_phase_selective_organizing_principle"
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "preregistered_local_validation",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": decision,
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


__all__ = ["TASK", "run_spin_symmetry_loop_coupling_scan"]
