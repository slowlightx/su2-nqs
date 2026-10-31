"""Seal the preregistered triangular-4x4 complete state-manifold gate."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.triangular_state_manifold import (
    ARM_IDS,
    TRIANGULAR_N16_RHOMBUS_SUPPORTS,
    triangular_n16_complete_manifold_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_triangular_n16_state_manifold"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid triangular N16 state-manifold config identity")
    if spec.get("preregistered_before_training") is not True:
        raise ValueError("the fresh-seed experiment must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("only the triangular 4x4 periodic gate is registered")
    if abs(float(spec.get("j2_over_j1", -1.0)) - 0.125) > 1.0e-14:
        raise ValueError("the state-manifold gate is fixed at J2/J1=0.125")
    architecture = spec.get("architecture")
    optimization = spec.get("optimization")
    source = spec.get("frozen_state_source")
    if not all(isinstance(value, dict) for value in (architecture, optimization, source)):
        raise TypeError("architecture, optimization, and frozen source must be objects")
    if architecture.get("basis") != "complete_four_rhombus_local_Schur_multiplicity":
        raise ValueError("the complete four-rhombus Schur basis is required")
    if int(architecture.get("singlet_dimension", 0)) != 1430:
        raise ValueError("the N16 singlet dimension must be 1430")
    if architecture.get("cell_supports") != [
        list(support) for support in TRIANGULAR_N16_RHOMBUS_SUPPORTS
    ]:
        raise ValueError("the preregistered rhombus partition changed")
    if architecture.get("magnetic_indices_visible_to_vit") is not False:
        raise ValueError("the ViT must not see magnetic indices")
    if tuple(architecture.get("arms", ())) != ARM_IDS:
        raise ValueError("the standalone and anchored arms must both be registered")
    embedding = int(architecture.get("embedding_dim", 0))
    heads = int(architecture.get("head_count", 0))
    depth = int(architecture.get("depth", 0))
    if min(embedding, heads, depth) < 1 or embedding % heads:
        raise ValueError("invalid ViT dimensions")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("at least three fresh seeds are required")
    checked_seeds = tuple(int(seed) for seed in seeds)
    if min(checked_seeds) < 0 or len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("seeds must be distinct and nonnegative")
    steps = int(optimization.get("steps", 0))
    matched = int(optimization.get("matched_step", 0))
    if not 0 < matched < steps or matched % 4:
        raise ValueError("matched_step must be below steps and divisible by four")
    if min(
        float(optimization.get("learning_rate", 0.0)),
        float(optimization.get("initial_residual_scale", 0.0)),
    ) <= 0.0:
        raise ValueError("optimization scales must be positive")
    if source.get("exact_ground_role") != "post_training_audit_only":
        raise ValueError("the exact ground state must not enter training")


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def run_spin_triangular_n16_state_manifold(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run without repeating ED and seal the complete-manifold comparison."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    source = spec["frozen_state_source"]
    source_result_path = ROOT / source["result"]
    source_bundle_path = ROOT / source["bundle"]
    source_payload = json.loads(source_result_path.read_text())
    if source_payload.get("result_sha256") != source["result_sha256"]:
        raise RuntimeError("frozen source result seal changed")
    if canonical_sha256(
        {key: value for key, value in source_payload.items() if key != "result_sha256"}
    ) != source["result_sha256"]:
        raise RuntimeError("frozen source result self-seal is invalid")
    if file_sha256(source_bundle_path) != source["bundle_file_sha256"]:
        raise RuntimeError("frozen state bundle file seal changed")
    with np.load(source_bundle_path) as loaded:
        state_bundle = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    architecture = spec["architecture"]
    optimization = spec["optimization"]
    suite, states = triangular_n16_complete_manifold_frontier(
        state_bundle,
        seeds=tuple(int(seed) for seed in optimization["seeds"]),
        steps=int(optimization["steps"]),
        matched_step=int(optimization["matched_step"]),
        learning_rate=float(optimization["learning_rate"]),
        initial_residual_scale=float(optimization["initial_residual_scale"]),
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
        coupling=float(spec["j2_over_j1"]),
    )
    _atomic_npz(state_path, states)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "repeated_ed_performed": False,
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


__all__ = ["TASK", "run_spin_triangular_n16_state_manifold"]
