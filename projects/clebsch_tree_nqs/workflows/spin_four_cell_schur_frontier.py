"""Seal the complete N=12/N=16 local-Schur Spatial-ViT physics gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_schur_frontier import (
    four_cell_schur_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_four_cell_schur_frontier"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid four-cell Schur frontier config identity")
    architecture = spec.get("architecture")
    optimization = spec.get("optimization")
    cases = spec.get("cases")
    if not isinstance(architecture, dict) or not isinstance(optimization, dict):
        raise TypeError("architecture and optimization must be objects")
    if not isinstance(cases, list) or cases != [
        "triangular_n12_j2_0125",
        "square_n16_j2_05",
    ]:
        raise ValueError("cases must register the N=12 and N=16 physics gates")
    if architecture.get("magnetic_indices_visible_to_vit") is not False:
        raise ValueError("the ViT must not see magnetic indices")
    if architecture.get("basis") != "complete_four_cell_local_Schur_multiplicity":
        raise ValueError("the complete four-cell Schur basis must be registered")
    embedding_dim = int(architecture.get("embedding_dim", 0))
    head_count = int(architecture.get("head_count", 0))
    depth = int(architecture.get("depth", 0))
    if min(embedding_dim, head_count, depth) < 1:
        raise ValueError("architecture dimensions must be positive")
    if embedding_dim % head_count:
        raise ValueError("embedding_dim must be divisible by head_count")
    for key in ("triangular_seeds", "square_seeds"):
        seeds = optimization.get(key)
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"optimization.{key} must be nonempty")
        checked = tuple(int(value) for value in seeds)
        if any(value < 0 for value in checked) or len(set(checked)) != len(checked):
            raise ValueError(f"optimization.{key} must be distinct and nonnegative")
    for key in ("triangular_steps", "square_steps"):
        if int(optimization.get(key, 0)) < 1:
            raise ValueError(f"optimization.{key} must be positive")
    if float(optimization.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("optimization.learning_rate must be positive")


def run_spin_four_cell_schur_frontier(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the complete-space N=12/N=16 gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    architecture = spec["architecture"]
    optimization = spec["optimization"]
    suite = four_cell_schur_frontier(
        triangular_seeds=tuple(int(value) for value in optimization["triangular_seeds"]),
        square_seeds=tuple(int(value) for value in optimization["square_seeds"]),
        triangular_steps=int(optimization["triangular_steps"]),
        square_steps=int(optimization["square_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
    )
    if not suite["exact_invariants_passed"]:
        raise RuntimeError("four-cell exact sector/tangent invariants failed")
    if not suite["trainability_diagnostics_passed"]:
        raise RuntimeError("four-cell registered trainability diagnostics failed")
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
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


__all__ = ["TASK", "run_spin_four_cell_schur_frontier"]
