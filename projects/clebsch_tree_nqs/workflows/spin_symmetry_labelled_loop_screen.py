"""Seal the preregistered E2/A2 loop-resonance residual screen."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.symmetry_labelled_loop_resonance import (
    triangular_symmetry_labelled_loop_screen,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_symmetry_labelled_loop_screen"
FROZEN_HASH = "8eef991116fc990c4827fd105ebbe79cb87261a2e0de0a96d670c2cb8e63f740"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid symmetry-labelled loop-screen identity")
    if spec.get("phase") != "development":
        raise ValueError("the loop-resonance screen is development only")
    if spec.get("frozen_state_result_sha256") != FROZEN_HASH:
        raise ValueError("the frozen state artifact hash changed")
    if spec.get("frozen_operator_vit_seeds") != [0, 1, 2]:
        raise ValueError("the frozen Operator-ViT seed panel changed")
    mechanism = spec.get("mechanism")
    if not isinstance(mechanism, dict) or mechanism != {
        "geometry": "triangular_4x4_periodic",
        "j2_over_j1": 0.125,
        "ground_sector": "S=0 Gamma-A1",
        "response_channels": [
            "E2_rhombus_ring_return",
            "E2_rhombus_dimer_return",
            "A2_directed_ring_return",
        ],
        "scalar_return": "sum_mu O_Gamma,mu^dagger O_Gamma,mu",
        "matched_dimension_controls": [
            "degree4_Krylov",
            "existing_A1_ring_dimer_six_site_bank",
            "degree7_Krylov_for_combined_bank",
        ],
    }:
        raise ValueError("the preregistered loop-resonance mechanism changed")


def run_spin_symmetry_labelled_loop_screen(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the frozen-state residual screen."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    artifact_path = ROOT / str(spec["frozen_state_artifact"])
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("result_sha256") != FROZEN_HASH:
        raise ValueError("the frozen state artifact does not match its hash")
    trials = artifact["frontier"]["pi_flux_exact_su2_a1_operator_vit_trials"]
    if [int(row["seed"]) for row in trials] != spec["frozen_operator_vit_seeds"]:
        raise ValueError("the frozen artifact seed order changed")
    coefficients = [row["operator_coefficients"] for row in trials]
    suite = triangular_symmetry_labelled_loop_screen(coefficients)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "development",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": (
            "preregister_fresh_seed_labelled_loop_operator_vit"
            if suite["all_gates_passed"]
            else "do_not_train_labelled_loop_operator_vit"
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


__all__ = ["TASK", "run_spin_symmetry_labelled_loop_screen"]
