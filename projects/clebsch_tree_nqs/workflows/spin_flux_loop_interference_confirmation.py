"""Seal the cross-geometry flux-interference/dimer confirmation gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.flux_loop_interference_frontier import (
    flux_loop_interference_frontier,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_flux_loop_interference_confirmation"


def _validate(spec: dict[str, object]) -> None:
    """Fail closed on the confirmation protocol and claim boundary."""

    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid flux-loop confirmation identity")
    if spec.get("phase") != "confirmation":
        raise ValueError("phase must remain confirmation")
    if spec.get("cases") != [
        "triangular_n16_j2_0.125",
        "square_n16_j2_0.5",
    ]:
        raise ValueError("both registered N16 geometries are required")
    if (
        spec.get("physical_extension")
        != "full_A1_opposite_edge_singlet_projector_dimer_resonance"
    ):
        raise ValueError("the physical extension differs from the registration")
    seeds = spec.get("scramble_seeds")
    excluded = spec.get("excluded_development_seeds")
    if not isinstance(seeds, list) or len(seeds) != 16:
        raise ValueError("exactly 16 confirmation scramble seeds are required")
    if not isinstance(excluded, list):
        raise TypeError("excluded_development_seeds must be a list")
    checked = tuple(int(seed) for seed in seeds)
    if len(set(checked)) != 16 or any(seed < 0 for seed in checked):
        raise ValueError("confirmation seeds must be distinct and nonnegative")
    if set(checked) & {int(seed) for seed in excluded}:
        raise ValueError("confirmation and development seeds overlap")
    if spec.get("controls") != {
        "same_exact_sector": True,
        "same_column_count": True,
        "same_variational_rank": True,
        "dimension_matched_hamiltonian_krylov": True,
        "full_hamiltonian_energy_and_variance": True,
    }:
        raise ValueError("matched controls differ from the registration")
    if spec.get("claim_boundary") != {
        "n12_is_saturation_control_only": True,
        "mechanism_gate_does_not_imply_variational_frontier": True,
        "n36_gpu_production_requires_dimension_matched_krylov_win": True,
    }:
        raise ValueError("claim boundary differs from the registration")


def run_spin_flux_loop_interference_confirmation(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and provenance-seal the registered confirmation."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    suite = flux_loop_interference_frontier(
        scramble_seeds=tuple(int(seed) for seed in spec["scramble_seeds"])
    )
    decision = suite["decision"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "confirmation",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "suite": suite,
        "promotion_decision": decision["branch"],
        "physics_conclusion": (
            "Cross-flux interference and the common local dimer-resonance action "
            "form a reproducible non-Krylov descent mechanism only if both N16 "
            "geometries beat every exact-sector geometry scramble. A failure "
            "against dimension-matched Krylov forbids a variational-frontier or "
            "GPU-production claim."
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


__all__ = ["TASK", "run_spin_flux_loop_interference_confirmation"]
