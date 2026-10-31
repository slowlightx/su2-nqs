"""Seal the exact-N16 coherent multi-Pfaffian comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.resonating_neural_pfaffian import (
    resonating_neural_pfaffian_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_resonating_neural_pfaffian"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (1, "spin", TASK):
        raise ValueError("invalid resonating neural-Pfaffian identity")
    if spec.get("preregistered_after_single_basin_failures") is not True:
        raise ValueError("the resonating contingency must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic" or abs(float(spec.get("j2_over_j1", -1)) - 0.125) > 1e-14:
        raise ValueError("only triangular 4x4 at J2/J1=0.125 is registered")
    if spec.get("banks") != {
        "direct_only": ["anchor", "compact_direct", "extended_direct"],
        "residual_only": ["anchor", "compact_residual", "extended_residual"],
        "combined": ["anchor", "compact_direct", "extended_direct", "compact_residual", "extended_residual"],
    }:
        raise ValueError("resonating bank definitions changed")
    if spec.get("controls") != {
        "scrambled_bank_uses_identical_column_count": True,
        "krylov_dimension_equals_observed_physical_rank": True,
    }:
        raise ValueError("resonating controls changed")
    if spec.get("gates") != {
        "minimum_resolved_energy_gain_per_site": 1e-10,
        "n36_requires_combined_energy_and_variance_win": True,
    }:
        raise ValueError("resonating gates changed")
    if spec.get("claim_boundary") != {
        "repeat_ed": False,
        "exact_ground_role": "post_fit_audit_only",
        "n36_closed_unless_variational_win": True,
        "gpu_production_allowed": False,
    }:
        raise ValueError("resonating claim boundary changed")


def _load_source(cell):
    result_path = ROOT / cell["result"]
    bundle_path = ROOT / cell["bundle"]
    payload = json.loads(result_path.read_text())
    body = dict(payload)
    recorded = body.pop("result_sha256", None)
    if recorded != cell["result_sha256"] or canonical_sha256(body) != recorded:
        raise ValueError(f"invalid sealed result: {result_path}")
    if file_sha256(bundle_path) != cell["bundle_sha256"]:
        raise ValueError(f"invalid sealed bundle: {bundle_path}")
    with np.load(bundle_path) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    return payload, arrays


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def run_spin_resonating_neural_pfaffian(config_path: Path, output: Path, *, overwrite: bool = False):
    spec = json.loads(config_path.read_text())
    _validate(spec)
    direct_result, direct = _load_source(spec["sealed_sources"]["direct"])
    residual_result, residual = _load_source(spec["sealed_sources"]["residual"])
    _a1_result, a1 = _load_source(spec["sealed_sources"]["a1_oracle"])
    if direct_result["suite"]["decision"]["triangular_n36_reopened"] is not False or residual_result["suite"]["decision"]["triangular_n36_reopened"] is not False:
        raise ValueError("single-basin prerequisites do not license this contingency")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    suite, arrays = resonating_neural_pfaffian_frontier(
        hamiltonian_a1=a1["coupling_0.125_h_a1"],
        exact_a1=residual["exact_a1"],
        anchor_a1=residual["anchor_a1"],
        direct_states={
            "compact_physical": direct["compact_j1_best_physical_a1"],
            "compact_scrambled": direct["compact_j1_best_scrambled_a1"],
            "extended_physical": direct["extended_j1j2_best_physical_a1"],
            "extended_scrambled": direct["extended_j1j2_best_scrambled_a1"],
        },
        residual_states={
            "compact_physical": residual["compact_j1_best_physical_a1"],
            "compact_scrambled": residual["compact_j1_best_scrambled_a1"],
            "extended_physical": residual["extended_j1j2_best_physical_a1"],
            "extended_scrambled": residual["extended_j1j2_best_scrambled_a1"],
        },
        operator_vit_energy_per_site=-0.534574778552453,
        minimum_resolved_energy_gain_per_site=spec["gates"]["minimum_resolved_energy_gain_per_site"],
    )
    _atomic_npz(state_path, arrays)
    win = suite["decision"]["combined_variational_win"]
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": "license_bounded_n36_multi_pfaffian_design_only" if win else "freeze_resonating_neural_pfaffian_route",
        "repeated_ed_performed": False,
        "gpu_production_licensed": False,
        "state_bundle": {"path": str(state_path.resolve().relative_to(ROOT.resolve())), "sha256": file_sha256(state_path)},
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["source_manifest"] = transitive_local_source_manifest((Path(__file__).resolve(),), root=ROOT, package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_resonating_neural_pfaffian"]
