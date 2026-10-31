"""Seal strict N=12 sector heads and the N=12/N=16 matched-work frontier."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.four_cell_matched_frontier import (
    four_cell_matched_frontier,
)
from su2_nqs.benchmarks.four_cell_sector_heads import (
    shared_four_cell_sector_head_gate,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_schur_sector_matched_frontier"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid Schur sector matched-frontier identity")
    sector = spec.get("sector_head_training")
    matched = spec.get("matched_frontier")
    if not isinstance(sector, dict) or not isinstance(matched, dict):
        raise TypeError("sector_head_training and matched_frontier must be objects")
    for label, block in (("sector_head_training", sector), ("matched_frontier", matched)):
        seeds = block.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{label}.seeds must be nonempty")
        checked = tuple(int(value) for value in seeds)
        if any(value < 0 for value in checked) or len(set(checked)) != len(checked):
            raise ValueError(f"{label}.seeds must be distinct and nonnegative")
        if int(block.get("steps", 0)) < 1:
            raise ValueError(f"{label}.steps must be positive")
    if float(sector.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("sector_head_training.learning_rate must be positive")
    controls = matched.get("controls")
    if controls != [
        "fixed_sz_A1_spatial_vit",
        "symmetry_unaware_fixed_sz_vit",
        "optimized_physical_jastrow_two_lanczos",
    ]:
        raise ValueError("matched_frontier must register all three controls")


def run_spin_schur_sector_matched_frontier(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run, validate, and provenance-seal both local promotion gates."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    sector_spec = spec["sector_head_training"]
    matched_spec = spec["matched_frontier"]
    sector = shared_four_cell_sector_head_gate(
        seeds=tuple(int(value) for value in sector_spec["seeds"]),
        steps=int(sector_spec["steps"]),
        learning_rate=float(sector_spec["learning_rate"]),
    )
    if not sector["exact_gates_passed"]:
        raise RuntimeError("a strict sector/tangent invariant failed")
    matched = four_cell_matched_frontier(
        seeds=tuple(int(value) for value in matched_spec["seeds"]),
        steps=int(matched_spec["steps"]),
    )
    promotion = (
        "pass_local_matched_frontier_prepare_n36_canary"
        if matched["promotion_gates_passed"]
        else "hold_n36_improve_local_controls_or_backbone"
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "sector_head_gate": sector,
        "matched_frontier": matched,
        "promotion_decision": promotion,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "A passed local gate licenses implementation of a sampled N=36 "
            "canary, not a large-size variational or QSL claim. The physical "
            "Jastrow/Lanczos arm is an optimized repository control, not a "
            "claim to reproduce every published projected-VMC ansatz."
        ),
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


__all__ = ["TASK", "run_spin_schur_sector_matched_frontier"]
