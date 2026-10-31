"""Diagnose the SU(2) boundary-spin channel missing from sealed N=16 states."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from su2_nqs.benchmarks.su2_entanglement_spectrum import (
    compare_entanglement_reports,
    json_ready_entanglement_report,
    su2_resolved_entanglement,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_entanglement_channel_audit"


def _validate(spec: Mapping[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid entanglement-channel audit identity")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("this audit is fixed to the sealed triangular 4x4 states")
    if spec.get("purpose") != "physics_mechanism_diagnosis":
        raise ValueError("the audit must be labelled as a mechanism diagnosis")
    couplings = tuple(float(value) for value in spec.get("couplings", ()))
    if couplings != (0.0, 0.125):
        raise ValueError("couplings must be the sealed pair (0, 0.125)")
    cuts = spec.get("cuts")
    if not isinstance(cuts, dict) or not cuts:
        raise ValueError("at least one named spatial cut is required")
    for name, sites in cuts.items():
        if not isinstance(name, str) or not isinstance(sites, list):
            raise ValueError("cuts must map names to site lists")
        checked = tuple(int(site) for site in sites)
        if not checked or len(set(checked)) != len(checked):
            raise ValueError("every cut must contain distinct sites")
    candidates = spec.get("candidates")
    if not isinstance(candidates, dict) or "two_lanczos" not in candidates:
        raise ValueError("candidate mapping must include two_lanczos")


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state has zero or nonfinite norm")
    return values / norm


def _state_metrics(
    state: np.ndarray,
    hamiltonian: np.ndarray,
    exact: np.ndarray,
    exact_energy: float,
) -> dict[str, float]:
    state = _normalized(state)
    action = hamiltonian @ state
    energy = float(np.vdot(state, action).real)
    variance = max(0.0, float(np.vdot(action, action).real - energy * energy))
    return {
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "energy_error_per_site": (energy - exact_energy) / 16.0,
        "variance_total": variance,
        "variance_per_site": variance / 16.0,
        "fidelity_to_exact": float(abs(np.vdot(exact, state)) ** 2),
    }


def _mean_metric(
    records: Mapping[str, Mapping[str, Mapping[str, float]]],
    candidate: str,
    metric: str,
) -> float:
    return float(
        np.mean(
            [comparison[candidate][metric] for comparison in records.values()]
        )
    )


def _mechanism_diagnosis(
    comparisons: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, object]:
    candidate = "two_lanczos"
    trace = _mean_metric(comparisons, candidate, "trace_distance")
    spin_tv = _mean_metric(
        comparisons, candidate, "sector_weight_total_variation"
    )
    spectrum_tv = _mean_metric(
        comparisons, candidate, "resolved_spectrum_total_variation"
    )
    scale = max(trace, 1.0e-15)
    spin_fraction = spin_tv / scale
    spectrum_fraction = spectrum_tv / scale
    if spin_fraction >= 0.5:
        primary = "boundary_spin_sector_weights"
        architecture = (
            "multi-scale SU(2) fusion tokens that can populate and transport "
            "additional boundary-spin q channels"
        )
    elif spectrum_fraction >= 0.5:
        primary = "multiplicity_schmidt_values_within_existing_q"
        architecture = (
            "configuration-dependent SU(2)-scalar attention acting inside each "
            "q sector, with explicit multi-scale multiplicity tokens"
        )
    else:
        primary = "multiplicity_schmidt_vectors_within_existing_q"
        architecture = (
            "a projected carrier/backflow or resonating-loop layer that rotates "
            "multiplicity states inside fixed q sectors before scalar attention"
        )
    return {
        "reference_candidate": candidate,
        "mean_trace_distance": trace,
        "mean_boundary_spin_weight_tv": spin_tv,
        "mean_resolved_spectrum_tv": spectrum_tv,
        "boundary_spin_fraction_of_trace_distance": spin_fraction,
        "resolved_spectrum_fraction_of_trace_distance": spectrum_fraction,
        "primary_missing_structure": primary,
        "next_architecture": architecture,
        "scope": (
            "This is an exact-small mechanism discriminator. It does not by "
            "itself establish thermodynamic entanglement scaling or a QSL."
        ),
    }


def run_spin_entanglement_channel_audit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Audit sealed states without rebuilding a many-body Hamiltonian."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")

    state_path = ROOT / str(spec["state_bundle"])
    source_result_path = ROOT / str(spec["source_results"])
    actual_state_hash = file_sha256(state_path)
    expected_state_hash = str(spec["state_bundle_sha256"])
    if actual_state_hash != expected_state_hash:
        raise RuntimeError("sealed state-bundle SHA256 mismatch")
    source_payload = json.loads(source_result_path.read_text())
    if source_payload["state_bundle"]["sha256"] != actual_state_hash:
        raise RuntimeError("source result and state bundle seals disagree")

    basis = enumerate_sz_configs(16, 8)
    cuts = {
        str(name): tuple(int(site) for site in sites)
        for name, sites in spec["cuts"].items()
    }
    candidate_suffixes = {
        str(name): str(suffix) for name, suffix in spec["candidates"].items()
    }
    couplings: list[dict[str, object]] = []
    all_comparisons: dict[
        str, dict[str, dict[str, float]]
    ] = {}
    maximum_a1_orthonormality_error = 0.0
    maximum_singlet_commutator = 0.0
    with np.load(state_path, allow_pickle=False) as arrays:
        physical_a1 = np.asarray(arrays["physical_a1_basis"], dtype=np.float64)
        maximum_a1_orthonormality_error = float(
            np.max(np.abs(physical_a1.T @ physical_a1 - np.eye(physical_a1.shape[1])))
        )
        if physical_a1.shape != (len(basis), 14):
            raise RuntimeError("sealed physical A1 basis has unexpected shape")
        if maximum_a1_orthonormality_error > 5.0e-11:
            raise RuntimeError("sealed physical A1 basis is not orthonormal")
        for coupling in (float(value) for value in spec["couplings"]):
            tag = f"{coupling:g}"
            hamiltonian = np.asarray(
                arrays[f"coupling_{tag}_h_a1"], dtype=np.float64
            )
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
            exact_energy = float(eigenvalues[0])
            exact_reduced = _normalized(eigenvectors[:, 0])
            expected_anchor = float(spec["exact_energy_per_site"][tag])
            anchor_error = abs(exact_energy / 16.0 - expected_anchor)
            if anchor_error > 2.0e-12:
                raise RuntimeError("sealed reduced Hamiltonian misses energy anchor")

            reduced_states: dict[str, np.ndarray] = {"exact": exact_reduced}
            for candidate, suffix in candidate_suffixes.items():
                reduced_states[candidate] = _normalized(
                    np.asarray(arrays[f"coupling_{tag}_{suffix}"])
                )
            metrics = {
                name: _state_metrics(state, hamiltonian, exact_reduced, exact_energy)
                for name, state in reduced_states.items()
            }
            physical_states = {
                name: _normalized(physical_a1 @ state)
                for name, state in reduced_states.items()
            }
            cut_payload: dict[str, object] = {}
            for cut_name, sites in cuts.items():
                reports = {
                    name: su2_resolved_entanglement(state, basis, sites)
                    for name, state in physical_states.items()
                }
                maximum_singlet_commutator = max(
                    maximum_singlet_commutator,
                    max(float(report["commutator_frobenius"]) for report in reports.values()),
                )
                comparisons = {
                    name: compare_entanglement_reports(report, reports["exact"])
                    for name, report in reports.items()
                    if name != "exact"
                }
                all_comparisons[f"j2_{tag}/{cut_name}"] = comparisons
                cut_payload[cut_name] = {
                    "states": {
                        name: json_ready_entanglement_report(report)
                        for name, report in reports.items()
                    },
                    "comparison_to_exact": comparisons,
                }
            couplings.append(
                {
                    "j2_over_j1": coupling,
                    "exact_energy_anchor_error_per_site": anchor_error,
                    "state_metrics": metrics,
                    "cuts": cut_payload,
                }
            )

    diagnosis = _mechanism_diagnosis(all_comparisons)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_artifact": {
            "state_bundle": str(state_path.resolve().relative_to(ROOT.resolve())),
            "state_bundle_sha256": actual_state_hash,
            "source_results": str(
                source_result_path.resolve().relative_to(ROOT.resolve())
            ),
            "source_results_sha256": file_sha256(source_result_path),
        },
        "couplings": couplings,
        "diagnosis": diagnosis,
        "exact_invariants": {
            "maximum_a1_orthonormality_error": maximum_a1_orthonormality_error,
            "maximum_rho_s2_commutator_frobenius": maximum_singlet_commutator,
        },
        "repeated_ed_performed": False,
        "new_many_body_hamiltonian_built": False,
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


__all__ = ["TASK", "run_spin_entanglement_channel_audit"]
