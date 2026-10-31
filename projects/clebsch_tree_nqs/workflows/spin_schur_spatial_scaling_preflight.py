"""Run the local scale-first admission benchmark for the spin-only PRL route."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.schur_spatial_scaling import (
    profile_schur_spatial_scaling,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_schur_spatial_scaling_preflight"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid Schur spatial scaling config identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the scaling preflight must be preregistered")
    architecture = spec.get("architecture")
    if not isinstance(architecture, dict):
        raise TypeError("architecture must be an object")
    if architecture.get("tree") != "balanced_many_cell":
        raise ValueError("the production candidate must use a balanced many-cell tree")
    if architecture.get("exact_su2") is not True:
        raise ValueError("the production candidate must retain exact SU(2)")
    if architecture.get("spatial_distance_envelope") != "positive_learned_periodic":
        raise ValueError("the size-consistent periodic distance envelope is required")
    systems = spec.get("systems")
    if not isinstance(systems, list) or not systems:
        raise TypeError("systems must be a nonempty list")
    identities = []
    for system in systems:
        if not isinstance(system, dict):
            raise TypeError("every system must be an object")
        geometry = system.get("geometry")
        if geometry not in {"square", "triangular"}:
            raise ValueError("invalid system geometry")
        size = system.get("linear_size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 2 or size % 2:
            raise ValueError("linear_size must be a positive even integer")
        identities.append((geometry, size))
    if len(set(identities)) != len(identities):
        raise ValueError("system entries must be unique")


def run_spin_schur_spatial_scaling_preflight(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Profile local scaling; do not start variational optimization."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    architecture = spec["architecture"]
    profiles = []
    for index, system in enumerate(spec["systems"]):
        profiles.append(
            profile_schur_spatial_scaling(
                geometry=str(system["geometry"]),
                linear_size=int(system["linear_size"]),
                j2_over_j1=float(system["j2_over_j1"]),
                sample_count=int(system["sample_count"]),
                connectivity_sample_count=int(system["connectivity_sample_count"]),
                seed=int(spec["seed"]) + 10 * index,
                embedding_dim=int(architecture["embedding_dim"]),
                head_count=int(architecture["head_count"]),
                depth=int(architecture["depth"]),
                mlp_ratio=int(architecture["mlp_ratio"]),
                timing_repeats=int(spec["timing_repeats"]),
            )
        )

    parameter_counts = {int(row["balanced_parameter_count"]) for row in profiles}
    all_samples_valid = all(bool(row["all_samples_valid_singlets"]) for row in profiles)
    n64_rows = [row for row in profiles if int(row["n_sites"]) == 64]
    gates = {
        "constant_parameter_count_across_sizes_and_lattices": len(parameter_counts) == 1,
        "all_autoregressive_samples_are_exact_singlets": all_samples_valid,
        "lattice_unit_distance_does_not_shrink_with_size": all(
            abs(float(row["minimum_positive_support_distance"]) - 1.0) <= 1.0e-12
            for row in profiles
        ),
        "balanced_tree_reduces_every_n64_connectivity_width": bool(n64_rows)
        and all(
            float(row["balanced_to_sequential_width_ratio"])
            > float(spec["gates"]["minimum_n64_width_reduction"])
            for row in n64_rows
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
        "profiles": profiles,
        "gate_results": gates,
        "local_preflight_passed": all(gates.values()),
        "energy_training_started": False,
        "claim_boundary": (
            "This local run establishes representation and cost scaling only. "
            "It contains no trained energy, V-score, phase, or GPU-hour result."
        ),
        "interpretation": (
            "The balanced geometry-aware Schur tree is the production backbone. "
            "The left-associated cell sequence remains a normalization/expressivity "
            "control because its exact Hamiltonian branching grows too rapidly."
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
    atomic_json(destination, payload)
    return payload


__all__ = ["TASK", "run_spin_schur_spatial_scaling_preflight"]
