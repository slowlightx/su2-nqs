"""Seal the exact-small invariant RVB-covering Transformer frontier."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.rvb_covering_attention_frontier import (
    rvb_covering_attention_frontier,
)
from su2_nqs.benchmarks.rvb_covering_frontier import rvb_covering_frontier
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_rvb_covering_frontier"
CASES = ("triangular_n12_j2_0.125", "square_n16_j2_0.5")


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid RVB-covering frontier config identity")
    if tuple(spec.get("cases", ())) != CASES:
        raise ValueError("cases must register triangular N=12 and square N=16")
    architecture = spec.get("architecture")
    optimization = spec.get("optimization")
    if not isinstance(architecture, dict) or not isinstance(optimization, dict):
        raise TypeError("architecture and optimization must be objects")
    required_architecture = {
        "basis": "signed_nearest_neighbour_rvb_covering_orbits",
        "exact_spin_sector": "S=0",
        "exact_lattice_sector": "Gamma-A1",
        "magnetic_indices_visible": False,
        "forward_reynolds_sum_required": False,
        "additive_projected_carrier": True,
        "embedding_dim": 4,
        "head_count": 1,
        "depth": 2,
        "mlp_ratio": 2,
    }
    for key, expected in required_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"architecture.{key} must equal {expected!r}")
    seeds = tuple(int(seed) for seed in optimization.get("paired_seeds", ()))
    if seeds != tuple(range(8)):
        raise ValueError("paired_seeds must be the preregistered range 0..7")
    if int(optimization.get("steps", 0)) != 500:
        raise ValueError("the registered comparison uses exactly 500 steps")
    if not np.isclose(float(optimization.get("learning_rate", 0.0)), 3.0e-3):
        raise ValueError("the registered learning rate is 3e-3")
    if optimization.get("control") != "paired_orbit_feature_reassignment":
        raise ValueError("the matched scrambled control must remain registered")


def _one_sided_sign_probability(wins: int, trials: int) -> float:
    if not 0 <= wins <= trials:
        raise ValueError("wins must lie between zero and trials")
    return float(sum(comb(trials, count) for count in range(wins, trials + 1)) / 2**trials)


def _paired_audit(case: dict[str, object], *, tolerance: float = 1.0e-6) -> dict[str, object]:
    differences = np.asarray(
        case["paired_physical_minus_scrambled_per_site"], dtype=np.float64
    )
    wins = int(np.count_nonzero(differences < -tolerance))
    losses = int(np.count_nonzero(differences > tolerance))
    ties = int(differences.size - wins - losses)
    non_ties = wins + losses
    return {
        "lower_is_better": True,
        "tolerance_per_site": tolerance,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "one_sided_exact_sign_probability": (
            _one_sided_sign_probability(wins, non_ties) if non_ties else 1.0
        ),
        "mean_difference_per_site": float(np.mean(differences)),
        "standard_error_per_site": float(
            np.std(differences, ddof=1) / np.sqrt(differences.size)
        ),
    }


def run_spin_rvb_covering_frontier(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the state-manifold and compression gates."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    architecture = spec["architecture"]
    optimization = spec["optimization"]
    oracle = rvb_covering_frontier()
    attention = rvb_covering_attention_frontier(
        seeds=tuple(int(seed) for seed in optimization["paired_seeds"]),
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
        mlp_ratio=int(architecture["mlp_ratio"]),
    )
    if not oracle["exact_sector_gate_passed"]:
        raise RuntimeError("RVB covering state-manifold sector audit failed")
    if not attention["exact_spin_gate_passed"]:
        raise RuntimeError("learned covering state leaked out of S=0")
    if not attention["full_coefficient_tangent_rank_both_cases"]:
        raise RuntimeError("registered covering Transformer is tangent-rank deficient")

    triangular_audit = _paired_audit(attention["triangular_n12"])
    square_audit = _paired_audit(attention["square_n16"])
    square_summary = attention["square_n16"]["summary"]
    local_energy_gate = bool(
        triangular_audit["mean_difference_per_site"] < 0.0
        and square_audit["mean_difference_per_site"] < 0.0
        and square_audit["losses"] == 0
        and square_summary["physical_relation_attention"][
            "seeds_below_two_lanczos"
        ]
        >= 7
        and square_summary["scrambled_relation_attention"][
            "seeds_below_two_lanczos"
        ]
        <= 2
    )
    decision = {
        "triangular_paired_energy": triangular_audit,
        "square_paired_energy": square_audit,
        "local_energy_efficiency_gate_passed": local_energy_gate,
        "large_size_variational_win_established": False,
        "promotion": (
            "implement nonorthogonal covering sampling and run an N=36 "
            "forward/backward throughput canary before any production energy job"
            if local_energy_gate
            else "freeze the covering-attention branch"
        ),
        "claim_boundary": (
            "The sealed result establishes an exact-small optimization-efficiency "
            "advantage from physical spatial labels. It does not yet establish "
            "an N=36 energy, GPU-hour, scaling, or QSL-physics advantage."
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "oracle": oracle,
        "attention": attention,
        "decision": decision,
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


__all__ = ["TASK", "run_spin_rvb_covering_frontier"]
