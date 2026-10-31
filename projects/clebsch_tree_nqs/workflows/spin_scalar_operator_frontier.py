"""Reproducible exact-small gate for scalar operators on the pi-flux carrier."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from su2_nqs.benchmarks.scalar_operator_frontier import (
    triangular_scalar_operator_scan,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_scalar_operator_frontier"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid scalar-operator frontier identity")
    couplings = spec.get("couplings")
    if not isinstance(couplings, list) or not couplings:
        raise ValueError("couplings must be a nonempty list")
    checked = tuple(float(value) for value in couplings)
    if len(set(checked)) != len(checked) or any(value < 0.0 for value in checked):
        raise ValueError("couplings must be distinct and nonnegative")


def run_spin_scalar_operator_frontier(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    rows = triangular_scalar_operator_scan(spec["couplings"])
    shell_wins = tuple(
        row["j2_over_j1"]
        for row in rows
        if row["same_dimension_gate"]["shell_beats_same_dimension_two_lanczos"]
    )
    e2_wins = tuple(
        row["j2_over_j1"]
        for row in rows
        if row["e2_scalar_gate"]["e2_bank_beats_matched_krylov"]
        or row["e2_scalar_gate"]["e2_adds_after_matched_krylov"]
    )
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "boundary": "4x4 periodic torus",
            "carrier": "projected pi/0-flux S=0 Gamma-A1",
            "purpose": "architecture gate only; not a large-size energy claim",
        },
        "scan": rows,
        "decision": {
            "shell_resolved_beats_matched_krylov_at": shell_wins,
            "first_registered_shell_win": shell_wins[0] if shell_wins else None,
            "e2_scalar_beats_or_adds_after_matched_krylov_at": e2_wins,
            "qsl_target_0p125_licenses_shell_attention": 0.125 in shell_wins,
            "qsl_target_0p125_licenses_e2_energy_head": 0.125 in e2_wins,
            "next_qsl_carriers": (
                "gauge_flux",
                "valence_bond_resonance",
                "long_range_singlet_rearrangement",
            ),
            "retain_e2_as_response_probe": True,
        },
        "hamiltonian_diagonalization_performed": True,
        "maximum_exact_basis_dimension": 12870,
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


__all__ = ["TASK", "run_spin_scalar_operator_frontier"]
