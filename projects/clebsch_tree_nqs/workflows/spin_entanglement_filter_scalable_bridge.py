"""Compress the exact-small entanglement filter into a transferable SU(2) NQS."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Mapping

import jax
import numpy as np

from su2_nqs.benchmarks.four_cell_schur import four_cell_schur_basis_matrix
from su2_nqs.benchmarks.many_cell_replay import TRIANGULAR_N16_SUPPORTS
from su2_nqs.benchmarks.sequential_cell_replay import (
    a1_reduced_to_sequential_coefficients,
    distill_carrier_anchored_sequential_cell_vit,
    sequential_to_rotated_coefficients,
)
from su2_nqs.benchmarks.su2_entanglement_spectrum import (
    compare_entanglement_reports,
    json_ready_entanglement_report,
    su2_resolved_entanglement,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    induced_representation_audit,
    projected_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.sequential_cell_schur_vit import (
    SequentialCellSchurViTConfig,
    init_sequential_cell_schur_vit,
    sample_sequential_cell_schur_vit,
)
from su2_nqs.su2.many_cell_schur import (
    rectangular_sequential_four_spin_layout,
    validate_layout_state,
)
from su2_nqs.su2.sequential_cell_schur import labels_to_sequential_actions
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_entanglement_filter_scalable_bridge"


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
        raise ValueError("invalid scalable entanglement-filter bridge identity")
    if spec.get("geometry") != "triangular_4x4_to_6x6_periodic":
        raise ValueError("this bridge is fixed to triangular 4x4 -> 6x6")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("fresh compression seeds must be preregistered")
    if tuple(float(value) for value in spec.get("couplings", ())) != (0.0, 0.125):
        raise ValueError("couplings must be the registered pair (0, 0.125)")
    seeds = tuple(int(seed) for seed in spec.get("seeds", ()))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct compression seeds are required")
    for name in ("filter_source", "a1_basis_source", "architecture", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    architecture = spec["architecture"]
    if architecture.get("q_conditioned_attention") is not True:
        raise ValueError("the registered bridge must use q-conditioned attention")
    if int(architecture.get("steps", 0)) < 1:
        raise ValueError("steps must be positive")
    if float(architecture.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("learning_rate must be positive")
    for name in ("n36_sample_count", "n36_carrier_probe_count"):
        if int(spec.get(name, 0)) < 1:
            raise ValueError(f"{name} must be positive")
    if spec.get("n36_policy") != "fail_closed_until_exact_a1_and_carrier_exist":
        raise ValueError("N36 policy must remain fail closed")


def _load_sealed(source: Mapping[str, object]) -> tuple[dict[str, object], Path]:
    result_path = ROOT / str(source["result"])
    bundle_path = ROOT / str(source["bundle"])
    result = json.loads(result_path.read_text())
    expected_result = str(source["result_sha256"])
    if result.get("result_sha256") != expected_result:
        raise RuntimeError(f"source result seal changed: {result_path}")
    if canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ) != expected_result:
        raise RuntimeError(f"source result self-seal is invalid: {result_path}")
    if file_sha256(bundle_path) != str(source["bundle_sha256"]):
        raise RuntimeError(f"source bundle seal changed: {bundle_path}")
    return result, bundle_path


def _matrix_free_action(workload, coefficients: np.ndarray) -> np.ndarray:
    positions = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(workload.center_labels)
    }
    action = np.zeros_like(coefficients, dtype=np.complex128)
    for old, coefficient in enumerate(coefficients):
        for slot in range(int(workload.widths[old])):
            new = tuple(int(value) for value in workload.connected_labels[old, slot])
            action[positions[new]] += workload.coefficients[old, slot] * coefficient
    return action


def _energy_variance(workload, state: np.ndarray) -> dict[str, float]:
    normalized = np.asarray(state, dtype=np.complex128)
    normalized /= np.linalg.norm(normalized)
    action = _matrix_free_action(workload, normalized)
    energy = float(np.vdot(normalized, action).real)
    residual = action - energy * normalized
    return {
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "variance_total": float(np.vdot(residual, residual).real),
        "variance_per_site": float(np.vdot(residual, residual).real / 16.0),
    }


def _parameter_shapes(params: object) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in leaf.shape)
        for leaf in jax.tree_util.tree_leaves(params)
    )


def _physical_audit(
    sequential: np.ndarray,
    *,
    schur_basis: np.ndarray,
    configurations,
    coefficient_a1_basis: np.ndarray,
    exact_report: dict[str, object],
    subsystem: tuple[int, ...],
) -> dict[str, object]:
    rotated, inverse = sequential_to_rotated_coefficients(sequential)
    physical = np.asarray(schur_basis @ rotated, dtype=np.complex128)
    physical /= np.linalg.norm(physical)
    a1_coordinates = coefficient_a1_basis.conj().T @ rotated
    a1_closure = float(np.vdot(a1_coordinates, a1_coordinates).real)
    report = su2_resolved_entanglement(physical, configurations, subsystem)
    return {
        "a1_projection_norm": a1_closure,
        "a1_leakage_norm": max(0.0, 1.0 - a1_closure),
        "inverse_recoupling": inverse,
        "entanglement": json_ready_entanglement_report(report),
        "entanglement_error": compare_entanglement_reports(report, exact_report),
    }


def _postprojected_a1_audit(
    sequential: np.ndarray,
    *,
    coefficient_a1_basis: np.ndarray,
    physical_a1_basis: np.ndarray,
    hamiltonian_a1: np.ndarray,
    configurations,
    exact_report: dict[str, object],
    subsystem: tuple[int, ...],
) -> dict[str, object]:
    """Diagnose whether tiny spatial leakage is responsible for bad variance."""

    rotated, _ = sequential_to_rotated_coefficients(sequential)
    coordinates = coefficient_a1_basis.conj().T @ rotated
    projection_norm = float(np.vdot(coordinates, coordinates).real)
    coordinates /= np.linalg.norm(coordinates)
    action = hamiltonian_a1 @ coordinates
    energy = float(np.vdot(coordinates, action).real)
    residual = action - energy * coordinates
    report = su2_resolved_entanglement(
        physical_a1_basis @ coordinates, configurations, subsystem
    )
    return {
        "preprojection_a1_norm": projection_norm,
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "variance_total": float(np.vdot(residual, residual).real),
        "variance_per_site": float(np.vdot(residual, residual).real / 16.0),
        "entanglement_error": compare_entanglement_reports(report, exact_report),
    }


def _architecture_kwargs(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "steps": int(spec["steps"]),
        "learning_rate": float(spec["learning_rate"]),
        "embedding_dim": int(spec["embedding_dim"]),
        "head_count": int(spec["head_count"]),
        "depth": int(spec["depth"]),
        "mlp_ratio": int(spec["mlp_ratio"]),
        "initial_residual_scale": float(spec["initial_residual_scale"]),
        "q_conditioned_attention": bool(spec["q_conditioned_attention"]),
    }


def run_spin_entanglement_filter_scalable_bridge(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run fresh-seed N16 compression and a fail-closed N36 preflight."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")

    filter_result, filter_bundle_path = _load_sealed(spec["filter_source"])
    a1_result, a1_bundle_path = _load_sealed(spec["a1_basis_source"])
    if not filter_result["gates"][
        "gate2_matched_physical_control_passed_both_couplings"
    ]:
        raise RuntimeError("upstream entanglement-filter mechanism gate did not pass")

    with np.load(a1_bundle_path, allow_pickle=False) as arrays:
        coefficient_a1_basis = np.asarray(
            arrays["coefficient_a1_basis"], dtype=np.complex128
        )
        physical_a1_basis = np.asarray(arrays["physical_a1_basis"], dtype=np.complex128)
        h_a1 = {
            coupling: np.asarray(arrays[f"coupling_{coupling:g}_h_a1"], dtype=np.float64)
            for coupling in (float(value) for value in spec["couplings"])
        }
    with np.load(filter_bundle_path, allow_pickle=False) as arrays:
        anchors = {
            coupling: np.asarray(arrays[f"coupling_{coupling:g}_anchor_a1"])
            for coupling in (float(value) for value in spec["couplings"])
        }
        targets = {
            coupling: np.asarray(arrays[f"coupling_{coupling:g}_physical_best_a1"])
            for coupling in (float(value) for value in spec["couplings"])
        }

    schur_basis, configurations = four_cell_schur_basis_matrix(
        "square_plaquette", supports=TRIANGULAR_N16_SUPPORTS
    )
    subsystem = tuple(int(site) for site in spec["subsystem"])
    architecture = spec["architecture"]
    saved: dict[str, np.ndarray] = {}
    coupling_rows = []
    all_gate_rows = []
    best_models: dict[float, dict[str, object]] = {}
    labels_reference = None

    for coupling in (float(value) for value in spec["couplings"]):
        target, labels, target_recoupling = a1_reduced_to_sequential_coefficients(
            targets[coupling], coefficient_a1_basis
        )
        carrier, carrier_labels, carrier_recoupling = a1_reduced_to_sequential_coefficients(
            anchors[coupling], coefficient_a1_basis
        )
        if not np.array_equal(labels, carrier_labels):
            raise RuntimeError("target and carrier sequential bases disagree")
        labels_reference = labels
        layout = rectangular_sequential_four_spin_layout(4, 4)
        model_config = SequentialCellSchurViTConfig(
            layout,
            embedding_dim=int(architecture["embedding_dim"]),
            head_count=int(architecture["head_count"]),
            depth=int(architecture["depth"]),
            mlp_ratio=int(architecture["mlp_ratio"]),
            q_conditioned_attention=True,
        )
        workload = build_many_cell_tree_connectivity(
            labels,
            model_config,
            triangular_hnf_j1j2_bonds(
                ((4, 0), (0, 4)), j1=1.0, j2=coupling
            ),
        )
        eigenvalues, eigenvectors = np.linalg.eigh(h_a1[coupling])
        exact_reduced = eigenvectors[:, 0]
        exact_physical = np.asarray(physical_a1_basis @ exact_reduced)
        exact_report = su2_resolved_entanglement(
            exact_physical, configurations, subsystem
        )
        target_metrics = _energy_variance(workload, target)
        carrier_metrics = _energy_variance(workload, carrier)
        target_audit = _physical_audit(
            target,
            schur_basis=schur_basis,
            configurations=configurations,
            coefficient_a1_basis=coefficient_a1_basis,
            exact_report=exact_report,
            subsystem=subsystem,
        )
        carrier_audit = _physical_audit(
            carrier,
            schur_basis=schur_basis,
            configurations=configurations,
            coefficient_a1_basis=coefficient_a1_basis,
            exact_report=exact_report,
            subsystem=subsystem,
        )

        trials = []
        states = []
        trained_models = []
        for seed in spec["seeds"]:
            training, state, trained = distill_carrier_anchored_sequential_cell_vit(
                labels,
                target,
                carrier,
                workload,
                seed=int(seed),
                **_architecture_kwargs(architecture),
            )
            audit = _physical_audit(
                state,
                schur_basis=schur_basis,
                configurations=configurations,
                coefficient_a1_basis=coefficient_a1_basis,
                exact_report=exact_report,
                subsystem=subsystem,
            )
            postprojected = _postprojected_a1_audit(
                state,
                coefficient_a1_basis=coefficient_a1_basis,
                physical_a1_basis=physical_a1_basis,
                hamiltonian_a1=h_a1[coupling],
                configurations=configurations,
                exact_report=exact_report,
                subsystem=subsystem,
            )
            trials.append(
                {**training, **audit, "postprojected_a1_diagnosis": postprojected}
            )
            states.append(state)
            trained_models.append(trained)

        energies = np.asarray([row["energy_total"] for row in trials])
        variances = np.asarray([row["variance_total"] for row in trials])
        fidelities = np.asarray([row["fidelity"] for row in trials])
        closures = np.asarray([row["a1_projection_norm"] for row in trials])
        trace_errors = np.asarray(
            [row["entanglement_error"]["trace_distance"] for row in trials]
        )
        spectrum_errors = np.asarray(
            [
                row["entanglement_error"]["resolved_spectrum_total_variation"]
                for row in trials
            ]
        )
        postprojected_energies = np.asarray(
            [row["postprojected_a1_diagnosis"]["energy_total"] for row in trials]
        )
        postprojected_variances = np.asarray(
            [row["postprojected_a1_diagnosis"]["variance_total"] for row in trials]
        )
        best_index = int(np.argmin(energies))
        best_models[coupling] = trained_models[best_index]
        tag = f"{coupling:g}"
        saved[f"coupling_{tag}_carrier_sequential"] = carrier
        saved[f"coupling_{tag}_target_filter_sequential"] = target
        saved[f"coupling_{tag}_compressed_best_sequential"] = states[best_index]

        gates = spec["gates"]
        coupling_gates = {
            "median_fidelity_gain_over_carrier": bool(
                np.median(fidelities)
                - float(abs(np.vdot(target, carrier)) ** 2)
                >= float(gates["minimum_median_fidelity_gain_over_carrier"])
            ),
            "median_energy_beats_carrier": bool(
                np.median(energies)
                < carrier_metrics["energy_total"]
                - float(gates["minimum_median_energy_gain_total"])
            ),
            "median_variance_beats_carrier": bool(
                np.median(variances) < carrier_metrics["variance_total"]
            ),
            "median_entanglement_trace_error_beats_carrier": bool(
                np.median(trace_errors)
                < carrier_audit["entanglement_error"]["trace_distance"]
            ),
            "median_resolved_spectrum_error_beats_carrier": bool(
                np.median(spectrum_errors)
                < carrier_audit["entanglement_error"][
                    "resolved_spectrum_total_variation"
                ]
            ),
            "minimum_a1_projection_norm": bool(
                np.min(closures) >= float(gates["minimum_a1_projection_norm"])
            ),
        }
        all_gate_rows.append(all(coupling_gates.values()))
        coupling_rows.append(
            {
                "j2_over_j1": coupling,
                "exact_energy_per_site": float(eigenvalues[0] / 16.0),
                "carrier_two_lanczos": {**carrier_metrics, **carrier_audit},
                "target_physical_filter": {**target_metrics, **target_audit},
                "target_recoupling": target_recoupling,
                "carrier_recoupling": carrier_recoupling,
                "trials": trials,
                "summary": {
                    "median_fidelity": float(np.median(fidelities)),
                    "median_fidelity_gain_over_carrier": float(
                        np.median(fidelities)
                        - abs(np.vdot(target, carrier)) ** 2
                    ),
                    "median_energy_total": float(np.median(energies)),
                    "median_energy_gain_over_carrier_total": float(
                        carrier_metrics["energy_total"] - np.median(energies)
                    ),
                    "median_variance_total": float(np.median(variances)),
                    "median_trace_distance": float(np.median(trace_errors)),
                    "median_resolved_spectrum_total_variation": float(
                        np.median(spectrum_errors)
                    ),
                    "minimum_a1_projection_norm": float(np.min(closures)),
                    "postprojected_median_energy_total": float(
                        np.median(postprojected_energies)
                    ),
                    "postprojected_median_variance_total": float(
                        np.median(postprojected_variances)
                    ),
                    "postprojection_repairs_median_variance_vs_carrier": bool(
                        np.median(postprojected_variances)
                        < carrier_metrics["variance_total"]
                    ),
                },
                "gates": coupling_gates,
                "passed": bool(all(coupling_gates.values())),
            }
        )

    if labels_reference is None:
        raise RuntimeError("no coupling was evaluated")
    saved["sequential_labels"] = labels_reference

    small_config = SequentialCellSchurViTConfig(
        rectangular_sequential_four_spin_layout(4, 4),
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
        mlp_ratio=int(architecture["mlp_ratio"]),
        q_conditioned_attention=True,
    )
    large_layout = rectangular_sequential_four_spin_layout(6, 6)
    large_config = SequentialCellSchurViTConfig(
        large_layout,
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
        mlp_ratio=int(architecture["mlp_ratio"]),
        q_conditioned_attention=True,
    )
    initialized_small = init_sequential_cell_schur_vit(
        jax.random.PRNGKey(0), small_config
    )
    initialized_large = init_sequential_cell_schur_vit(
        jax.random.PRNGKey(0), large_config
    )
    shape_transfer = _parameter_shapes(initialized_small) == _parameter_shapes(
        initialized_large
    )
    transferred = best_models[0.125]["model"]
    sample_count = int(spec["n36_sample_count"])
    samples = np.asarray(
        sample_sequential_cell_schur_vit(
            transferred,
            jax.random.PRNGKey(int(spec["n36_sample_seed"])),
            sample_count,
            large_config,
        )
    )
    for row in samples:
        validate_layout_state(large_layout, row, target_q=0)
    sample_actions = labels_to_sequential_actions(large_layout, samples)
    sampler_gate = bool(
        samples.shape == (sample_count, 35)
        and sample_actions.shape == (sample_count, 9, 3)
        and np.all(samples[:, -1] == 0)
    )
    saved["n36_transferred_residual_sampler_labels"] = samples

    carrier_rng = np.random.default_rng(int(spec["n36_carrier_probe_seed"]))
    carrier_probe_count = int(spec["n36_carrier_probe_count"])
    carrier_configurations = np.zeros((carrier_probe_count, 36), dtype=np.int8)
    for row in carrier_configurations:
        row[carrier_rng.choice(36, size=18, replace=False)] = 1
    carrier_orbit = build_n36_pi_zero_flux_orbit()
    carrier_probe = projected_pi_zero_flux_a1_amplitudes(
        carrier_configurations,
        orbit=carrier_orbit,
        microbatch_size=min(64, carrier_probe_count),
    )
    induced_audit = induced_representation_audit()
    external_n36_carrier_available = bool(
        np.all(np.isfinite(carrier_probe.amplitudes))
        and np.any(np.abs(carrier_probe.amplitudes) > 0.0)
        and int(induced_audit["projector_rank"]) == 1
        and float(induced_audit["projector_idempotency_error"]) < 1.0e-12
        and float(induced_audit["a1_vector_error"]) < 1.0e-12
    )
    saved["n36_projected_carrier_probe_configurations"] = carrier_configurations
    saved["n36_projected_carrier_probe_amplitudes"] = carrier_probe.amplitudes

    compression_passed = bool(all(all_gate_rows) and shape_transfer and sampler_gate)
    exact_a1_scalable = False
    carrier_residual_common_basis_interface = False
    n36_authorized = bool(
        compression_passed
        and exact_a1_scalable
        and external_n36_carrier_available
        and carrier_residual_common_basis_interface
    )
    _atomic_npz(state_path, saved)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "architecture": {
            "state": "normalized cell-sequential SU2 singlet residual added to a frozen two-Lanczos carrier",
            "attention": "configuration-dependent pair bias from local q equality, local-q separation, and intermediate-q separation",
            "magnetic_indices_seen_by_network": False,
            "carrier_nodes_can_change": True,
            "parameter_shapes_independent_of_lattice_size": shape_transfer,
        },
        "couplings": coupling_rows,
        "n36_preflight": {
            "parameter_shapes_match_n16": shape_transfer,
            "transferred_trained_residual_sample_count": sample_count,
            "sample_shape": list(samples.shape),
            "action_shape": list(sample_actions.shape),
            "sampler_gate_passed": sampler_gate,
            "exact_gamma_a1_projection_available_in_autoregressive_sampler": exact_a1_scalable,
            "matched_external_projected_carrier_available": external_n36_carrier_available,
            "carrier_residual_common_basis_interface_available": carrier_residual_common_basis_interface,
            "projected_carrier_probe_count": carrier_probe_count,
            "projected_carrier_nonzero_fraction": float(
                np.mean(np.abs(carrier_probe.amplitudes) > 0.0)
            ),
            "induced_carrier_a1_audit": {
                key: value
                for key, value in induced_audit.items()
                if key != "projector"
            },
            "energy_training_started": False,
        },
        "gates": {
            "n16_compression_passed_both_couplings": bool(all(all_gate_rows)),
            "n16_to_n36_parameter_shape_transfer": shape_transfer,
            "n36_transferred_residual_sampler": sampler_gate,
            "scalable_exact_gamma_a1": exact_a1_scalable,
            "n36_external_projected_carrier": external_n36_carrier_available,
            "n36_carrier_residual_common_basis_interface": carrier_residual_common_basis_interface,
            "n36_gpu_production_authorized": n36_authorized,
        },
        "blocking_reasons": [
            name
            for name, passed in {
                "n16_compression": bool(all(all_gate_rows)),
                "shape_transfer": shape_transfer,
                "n36_sampler": sampler_gate,
                "scalable_exact_gamma_a1": exact_a1_scalable,
                "n36_external_projected_carrier": external_n36_carrier_available,
                "n36_carrier_residual_common_basis_interface": carrier_residual_common_basis_interface,
            }.items()
            if not passed
        ],
        "decision": (
            "implement_scalable_exact_A1_and_carrier_residual_common_basis_before_energy_training"
            if compression_passed
            else "revise_configuration_dependent_residual_before_N36"
        ),
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "repeated_ed_performed": False,
        "upstream_sources": {
            "filter_result_sha256": filter_result["result_sha256"],
            "a1_result_sha256": a1_result["result_sha256"],
        },
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "This is a size-transferable SU2 residual-compression and sampling test. "
            "It is not an N36 variational-energy result: the exact spatial A1 projection "
            "and matched projected carrier are still missing at N36."
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


__all__ = ["TASK", "run_spin_entanglement_filter_scalable_bridge"]
