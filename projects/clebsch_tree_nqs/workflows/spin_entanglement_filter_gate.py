"""Run the fail-closed N=16 SU(2) entanglement-filter gates."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from su2_nqs.benchmarks.su2_entanglement_filter import (
    apply_subsystem_filter,
    bures_schmidt_transport,
    exponential_filter,
    scramble_generators,
    subsystem_spin_squared,
    triangular_half_system_generators,
)
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
TASK = "spin_entanglement_filter_gate"


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _validate(spec: Mapping[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid entanglement-filter gate identity")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("this gate is fixed to triangular 4x4")
    if spec.get("purpose") != "fail_closed_physics_mechanism_gate":
        raise ValueError("the workflow must be labelled as a fail-closed gate")
    if tuple(float(value) for value in spec.get("couplings", ())) != (0.0, 0.125):
        raise ValueError("couplings must be the registered pair (0, 0.125)")
    subsystem = tuple(int(site) for site in spec.get("subsystem", ()))
    if len(subsystem) != 8 or len(set(subsystem)) != 8:
        raise ValueError("the registered half-torus subsystem must have eight sites")
    seeds = tuple(int(seed) for seed in spec.get("paired_seeds", ()))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct paired seeds are required")
    if float(spec.get("boundary_q_tv_ceiling", 0.0)) <= 0.0:
        raise ValueError("boundary_q_tv_ceiling must be positive")
    if int(spec.get("maximum_iterations", 0)) < 1:
        raise ValueError("maximum_iterations must be positive")
    if not isinstance(spec.get("scalable_filter_implementation_available"), bool):
        raise ValueError("scalable_filter_implementation_available must be boolean")


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state has zero or nonfinite norm")
    return values / norm


def _metrics(
    reduced: np.ndarray,
    hamiltonian: np.ndarray,
    exact: np.ndarray,
) -> dict[str, float]:
    state = _normalize(reduced)
    action = hamiltonian @ state
    energy = float(np.vdot(state, action).real)
    variance = max(0.0, float(np.vdot(action, action).real - energy * energy))
    return {
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "energy_error_per_site": (energy - float(np.linalg.eigvalsh(hamiltonian)[0])) / 16.0,
        "variance_total": variance,
        "variance_per_site": variance / 16.0,
        "fidelity_to_exact": float(abs(np.vdot(exact, state)) ** 2),
    }


def _filtered_reduced_state(
    parameters: Sequence[float],
    generators: Sequence[np.ndarray],
    anchor_physical: np.ndarray,
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    physical_a1: np.ndarray,
) -> np.ndarray:
    operator = exponential_filter(parameters, generators)
    filtered = apply_subsystem_filter(
        anchor_physical, basis, subsystem, operator
    )
    return _normalize(physical_a1.conj().T @ filtered)


def _optimize(
    initial: np.ndarray,
    generators: Sequence[np.ndarray],
    anchor_physical: np.ndarray,
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    physical_a1: np.ndarray,
    hamiltonian: np.ndarray,
    maximum_iterations: int,
    bound: float,
) -> tuple[np.ndarray, dict[str, object]]:
    def objective(parameters: np.ndarray) -> float:
        state = _filtered_reduced_state(
            parameters,
            generators,
            anchor_physical,
            basis,
            subsystem,
            physical_a1,
        )
        return float(np.vdot(state, hamiltonian @ state).real)

    result = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-bound, bound)] * len(initial),
        options={"maxiter": int(maximum_iterations), "ftol": 1.0e-14, "gtol": 1.0e-10},
    )
    state = _filtered_reduced_state(
        result.x,
        generators,
        anchor_physical,
        basis,
        subsystem,
        physical_a1,
    )
    return state, {
        "parameters": np.asarray(result.x).tolist(),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "full_h_calls": int(result.nfev),
        "final_objective_total": float(result.fun),
    }


def _audit_state(
    reduced: np.ndarray,
    physical_a1: np.ndarray,
    hamiltonian: np.ndarray,
    exact_reduced: np.ndarray,
    exact_report: dict[str, object],
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
) -> dict[str, object]:
    report = su2_resolved_entanglement(
        physical_a1 @ _normalize(reduced), basis, subsystem
    )
    return {
        "metrics": _metrics(reduced, hamiltonian, exact_reduced),
        "entanglement": json_ready_entanglement_report(report),
        "entanglement_error": compare_entanglement_reports(report, exact_report),
    }


def run_spin_entanglement_filter_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the oracle, trainable, and matched-scramble gates in order."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    bundle_path = ROOT / str(spec["state_bundle"])
    if file_sha256(bundle_path) != str(spec["state_bundle_sha256"]):
        raise RuntimeError("sealed state bundle SHA256 mismatch")

    basis = enumerate_sz_configs(16, 8)
    subsystem = tuple(int(site) for site in spec["subsystem"])
    generators = triangular_half_system_generators(
        ((4, 0), (0, 4)), subsystem
    )
    spin_squared = subsystem_spin_squared(len(subsystem))
    generator_su2_commutator = max(
        float(np.linalg.norm(generator @ spin_squared - spin_squared @ generator))
        for generator in generators
    )
    paired_seeds = tuple(int(seed) for seed in spec["paired_seeds"])
    maximum_iterations = int(spec["maximum_iterations"])
    parameter_bound = float(spec["parameter_bound"])
    initialization_scale = float(spec["initialization_scale"])
    q_tv_ceiling = float(spec["boundary_q_tv_ceiling"])

    coupling_payloads = []
    saved_states: dict[str, np.ndarray] = {}
    gate1_rows = []
    gate2_rows = []
    maximum_scrambled_spectrum_error = 0.0
    with np.load(bundle_path, allow_pickle=False) as arrays:
        physical_a1 = np.asarray(arrays["physical_a1_basis"], dtype=np.float64)
        for coupling in (float(value) for value in spec["couplings"]):
            tag = f"{coupling:g}"
            hamiltonian = np.asarray(arrays[f"coupling_{tag}_h_a1"], dtype=np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
            exact_reduced = _normalize(eigenvectors[:, 0])
            anchor_reduced = _normalize(
                np.asarray(arrays[f"coupling_{tag}_two_lanczos_a1"])
            )
            exact_physical = _normalize(physical_a1 @ exact_reduced)
            anchor_physical = _normalize(physical_a1 @ anchor_reduced)
            exact_report = su2_resolved_entanglement(
                exact_physical, basis, subsystem
            )
            anchor_audit = _audit_state(
                anchor_reduced,
                physical_a1,
                hamiltonian,
                exact_reduced,
                exact_report,
                basis,
                subsystem,
            )

            transport = bures_schmidt_transport(
                anchor_physical,
                exact_physical,
                basis,
                subsystem,
                relative_cutoff=float(spec["oracle_relative_cutoff"]),
            )
            oracle_physical = apply_subsystem_filter(
                anchor_physical, basis, subsystem, transport
            )
            oracle_reduced = _normalize(physical_a1.conj().T @ oracle_physical)
            oracle_audit = _audit_state(
                oracle_reduced,
                physical_a1,
                hamiltonian,
                exact_reduced,
                exact_report,
                basis,
                subsystem,
            )
            oracle_commutator = float(
                np.linalg.norm(transport @ spin_squared - spin_squared @ transport)
            )

            paired_trials = []
            physical_states = []
            scrambled_states = []
            for seed in paired_seeds:
                rng = np.random.default_rng(seed)
                initial = rng.normal(
                    scale=initialization_scale, size=len(generators)
                )
                permutation = tuple(int(site) for site in rng.permutation(len(subsystem)))
                scrambled = scramble_generators(generators, permutation)
                maximum_scrambled_spectrum_error = max(
                    maximum_scrambled_spectrum_error,
                    max(
                        float(
                            np.max(
                                np.abs(
                                    np.linalg.eigvalsh(physical)
                                    - np.linalg.eigvalsh(control)
                                )
                            )
                        )
                        for physical, control in zip(
                            generators, scrambled, strict=True
                        )
                    ),
                )
                physical_state, physical_optimization = _optimize(
                    initial,
                    generators,
                    anchor_physical,
                    basis,
                    subsystem,
                    physical_a1,
                    hamiltonian,
                    maximum_iterations,
                    parameter_bound,
                )
                scrambled_state, scrambled_optimization = _optimize(
                    initial,
                    scrambled,
                    anchor_physical,
                    basis,
                    subsystem,
                    physical_a1,
                    hamiltonian,
                    maximum_iterations,
                    parameter_bound,
                )
                physical_audit = _audit_state(
                    physical_state,
                    physical_a1,
                    hamiltonian,
                    exact_reduced,
                    exact_report,
                    basis,
                    subsystem,
                )
                scrambled_audit = _audit_state(
                    scrambled_state,
                    physical_a1,
                    hamiltonian,
                    exact_reduced,
                    exact_report,
                    basis,
                    subsystem,
                )
                paired_trials.append(
                    {
                        "seed": seed,
                        "scrambled_site_permutation": list(permutation),
                        "common_initial_parameters": initial.tolist(),
                        "physical": {
                            **physical_audit,
                            "optimization": physical_optimization,
                        },
                        "scrambled": {
                            **scrambled_audit,
                            "optimization": scrambled_optimization,
                        },
                        "paired_physical_minus_scrambled_energy_per_site": (
                            float(physical_audit["metrics"]["energy_per_site"])
                            - float(scrambled_audit["metrics"]["energy_per_site"])
                        ),
                        "paired_physical_minus_scrambled_variance_per_site": (
                            float(physical_audit["metrics"]["variance_per_site"])
                            - float(scrambled_audit["metrics"]["variance_per_site"])
                        ),
                    }
                )
                physical_states.append(physical_state)
                scrambled_states.append(scrambled_state)

            physical_energy = np.asarray(
                [trial["physical"]["metrics"]["energy_per_site"] for trial in paired_trials]
            )
            physical_variance = np.asarray(
                [trial["physical"]["metrics"]["variance_per_site"] for trial in paired_trials]
            )
            physical_trace = np.asarray(
                [trial["physical"]["entanglement_error"]["trace_distance"] for trial in paired_trials]
            )
            physical_spectrum = np.asarray(
                [
                    trial["physical"]["entanglement_error"][
                        "resolved_spectrum_total_variation"
                    ]
                    for trial in paired_trials
                ]
            )
            physical_q_tv = np.asarray(
                [
                    trial["physical"]["entanglement_error"][
                        "sector_weight_total_variation"
                    ]
                    for trial in paired_trials
                ]
            )
            scrambled_energy = np.asarray(
                [trial["scrambled"]["metrics"]["energy_per_site"] for trial in paired_trials]
            )
            scrambled_variance = np.asarray(
                [trial["scrambled"]["metrics"]["variance_per_site"] for trial in paired_trials]
            )
            anchor_metrics = anchor_audit["metrics"]
            anchor_error = anchor_audit["entanglement_error"]
            oracle_error = oracle_audit["entanglement_error"]
            gate1 = {
                "oracle_lowers_energy": oracle_audit["metrics"]["energy_per_site"]
                < anchor_metrics["energy_per_site"],
                "oracle_lowers_variance": oracle_audit["metrics"]["variance_per_site"]
                < anchor_metrics["variance_per_site"],
                "oracle_lowers_trace_distance": oracle_error["trace_distance"]
                < anchor_error["trace_distance"],
                "oracle_lowers_resolved_spectrum_error": oracle_error[
                    "resolved_spectrum_total_variation"
                ]
                < anchor_error["resolved_spectrum_total_variation"],
                "trainable_median_lowers_energy": float(np.median(physical_energy))
                < anchor_metrics["energy_per_site"],
                "trainable_median_lowers_variance": float(np.median(physical_variance))
                < anchor_metrics["variance_per_site"],
                "trainable_median_lowers_trace_distance": float(np.median(physical_trace))
                < anchor_error["trace_distance"],
                "trainable_median_lowers_resolved_spectrum_error": float(
                    np.median(physical_spectrum)
                )
                < anchor_error["resolved_spectrum_total_variation"],
                "trainable_boundary_q_weights_remain_accurate": float(
                    np.median(physical_q_tv)
                )
                <= max(
                    q_tv_ceiling,
                    float(anchor_error["sector_weight_total_variation"]),
                ),
            }
            gate2 = {
                "physical_beats_two_lanczos_energy_every_seed": bool(
                    np.all(physical_energy < anchor_metrics["energy_per_site"])
                ),
                "physical_beats_two_lanczos_variance_every_seed": bool(
                    np.all(physical_variance < anchor_metrics["variance_per_site"])
                ),
                "physical_beats_scrambled_median_energy": float(
                    np.median(physical_energy)
                )
                < float(np.median(scrambled_energy)),
                "physical_beats_scrambled_median_variance": float(
                    np.median(physical_variance)
                )
                < float(np.median(scrambled_variance)),
                "physical_beats_paired_scramble_majority_energy_and_variance": sum(
                    trial["paired_physical_minus_scrambled_energy_per_site"] < 0.0
                    and trial["paired_physical_minus_scrambled_variance_per_site"] < 0.0
                    for trial in paired_trials
                )
                >= len(paired_trials) // 2 + 1,
            }
            gate1_rows.append(all(gate1.values()))
            gate2_rows.append(all(gate2.values()))
            best_physical = int(np.argmin(physical_energy))
            best_scrambled = int(np.argmin(scrambled_energy))
            saved_states[f"coupling_{tag}_anchor_a1"] = anchor_reduced
            saved_states[f"coupling_{tag}_oracle_a1"] = oracle_reduced
            saved_states[f"coupling_{tag}_physical_best_a1"] = physical_states[
                best_physical
            ]
            saved_states[f"coupling_{tag}_scrambled_best_a1"] = scrambled_states[
                best_scrambled
            ]
            coupling_payloads.append(
                {
                    "j2_over_j1": coupling,
                    "exact_energy_per_site": float(eigenvalues[0] / 16.0),
                    "two_lanczos": anchor_audit,
                    "oracle_schmidt_transport": {
                        **oracle_audit,
                        "rho_s2_commutator_frobenius": oracle_commutator,
                        "uses_exact_target_rdm": True,
                    },
                    "paired_trials": paired_trials,
                    "summary": {
                        "physical_median_energy_per_site": float(np.median(physical_energy)),
                        "scrambled_median_energy_per_site": float(np.median(scrambled_energy)),
                        "physical_median_variance_per_site": float(np.median(physical_variance)),
                        "scrambled_median_variance_per_site": float(np.median(scrambled_variance)),
                        "physical_median_trace_distance": float(np.median(physical_trace)),
                        "physical_median_resolved_spectrum_tv": float(
                            np.median(physical_spectrum)
                        ),
                        "physical_median_boundary_q_tv": float(np.median(physical_q_tv)),
                    },
                    "gate1_entanglement_mechanism": gate1,
                    "gate1_passed": bool(all(gate1.values())),
                    "gate2_matched_physical_control": gate2,
                    "gate2_passed": bool(all(gate2.values())),
                }
            )

    gate1_passed = bool(all(gate1_rows))
    gate2_passed = bool(gate1_passed and all(gate2_rows))
    mechanism_promoted = bool(gate1_passed and gate2_passed)
    scalable_implementation = bool(spec["scalable_filter_implementation_available"])
    n36_authorized = bool(mechanism_promoted and scalable_implementation)
    _atomic_npz(state_path, saved_states)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "mechanism": {
            "formula": "P_A1 exp(theta1 H_A_J1 + theta2 H_A_J2 + thetaq S_A^2) psi_2L",
            "trainable_parameter_count": 3,
            "exact_su2": True,
            "exact_gamma_a1_after_projection": True,
            "oracle_uses_exact_target_only_as_ceiling": True,
            "production_scalable": False,
        },
        "couplings": coupling_payloads,
        "gates": {
            "gate1_entanglement_mechanism_passed_both_couplings": gate1_passed,
            "gate2_matched_physical_control_passed_both_couplings": gate2_passed,
            "mechanism_promoted_to_scalable_implementation": mechanism_promoted,
            "scalable_filter_implementation_available": scalable_implementation,
            "gate3_triangular_n36_authorized": n36_authorized,
        },
        "gate3": {
            "status": "not_started" if not n36_authorized else "authorized_not_run",
            "reason": (
                "mechanism_passed_but_half_system_exponential_has_no_scalable_evaluator"
                if mechanism_promoted and not scalable_implementation
                else "earlier_mechanism_or_control_gate_failed"
                if not mechanism_promoted
                else "requires_a_separate_scalable_filter_implementation"
            ),
        },
        "exact_invariants": {
            "maximum_generator_su2_commutator_frobenius": generator_su2_commutator,
            "maximum_scrambled_generator_spectrum_error": maximum_scrambled_spectrum_error,
        },
        "repeated_ed_performed": False,
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "decision": (
            "implement_and_validate_scalable_q_resolved_filter_before_n36"
            if mechanism_promoted and not scalable_implementation
            else "stop_entanglement_filter_mechanism_gate_failed"
            if not mechanism_promoted
            else "implement_scalable_filter_then_run_n36"
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
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_entanglement_filter_gate"]
