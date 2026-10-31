"""Reproducible exact artifact for six-site-core intercluster mechanisms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from su2_nqs.benchmarks.two_cluster_support_orientation import (
    BoundaryInterfaceResponse,
    CarrierUnionAudit,
    SupportOrientationCarrierReport,
    support_orientation_carrier_report,
)
from su2_nqs.benchmarks.two_cluster_virtual_processes import (
    CoveringResonanceReport,
    TwoClusterVirtualProcessReport,
    translated_covering_resonance_report,
    two_cluster_open_partition,
    two_cluster_virtual_process_report,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "two_cluster_virtual_processes"
CLAIM_BOUNDARY = (
    "Exact six-site/N=12 mechanism diagnostics plus a non-ED periodic 6x6 "
    "bond-ledger audit of translated fixed-orientation 2x3 interfaces; no "
    "N=36 state or ED, rotated 3x2 covering, thermodynamic, phase, scaling, "
    "or variational-energy advantage is claimed."
)
DEFAULT_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/"
    "spin_two_cluster_virtual_processes.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/"
    "spin_two_cluster_virtual_processes_v1"
)


def _validate(spec: dict[str, Any]) -> None:
    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
        or spec.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("invalid two-cluster artifact identity or claim boundary")
    if spec.get("open_perturbation") != {
        "rows": 2,
        "columns": 6,
        "periodic": False,
        "cluster_shape": [2, 3],
        "unperturbed_hamiltonian": "H_C1_plus_H_C2",
        "perturbation": "all_J1_J2_bonds_crossing_the_vertical_cluster_cut",
        "reported_cluster_spin_labels": "doubled_q",
    }:
        raise ValueError("the open perturbation contract changed")
    if spec.get("periodic_resonance") != {
        "rows": 2,
        "columns": 6,
        "periodic": True,
        "translation_orbit": "all_distinct_row_and_column_translations",
        "also_report": "column_only_three_covering_orbit",
        "parallel_edge_convention": "combine_wrapped_images_by_summing_couplings",
    }:
        raise ValueError("the periodic covering-resonance contract changed")
    if spec.get("full_coupling_carrier_test") != {
        "boundary_scale_lambda": 1.0,
        "parent_state": "cluster_ground_singlet_product",
        "triplet_pair_basis": "total_singlet_coupling_of_left_and_right_q2_multiplets",
        "rank_k_subspace": "all_k_squared_products_of_leading_left_and_right_carriers",
        "carrier_orderings": [
            "lowest_triplet_energy",
            "raw_boundary_response_schmidt",
            "resolvent_dressed_correction_schmidt",
            "complete_q2_pair_space",
        ],
    }:
        raise ValueError("the full-coupling carrier comparison contract changed")
    if spec.get("support_orientation_audit") != {
        "rows": 6,
        "columns": 6,
        "periodic": True,
        "cluster_shape": [2, 3],
        "translation_group_order": 36,
        "distinct_fixed_orientation_tilings": 6,
        "interface_families": [
            "direction_resolved",
            "physical_pair_total",
        ],
        "response": "resolvent_dressed_q2_left_right_singular_modes",
        "shared_bank": "occurrence_weighted_left_right_union",
        "excluded_coverings": "rotated_or_reflected_3x2",
    }:
        raise ValueError("the support/orientation audit contract changed")
    expected = {
        "square_2x6_j2_0p5": ("square", 0.5),
        "triangular_2x6_j2_0p125": ("triangular", 0.125),
    }
    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("exactly the square and triangular cases are required")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("each case must be an object")
        case_id = str(case.get("case_id", ""))
        if case_id not in expected or case_id in seen:
            raise ValueError("case ids must be the unique registered pair")
        seen.add(case_id)
        if (case.get("geometry"), float(case.get("j2_over_j1", np.nan))) != expected[
            case_id
        ]:
            raise ValueError("a registered case geometry or coupling changed")


def _array(array: np.ndarray) -> list[Any]:
    return np.asarray(array).tolist()


def _virtual_payload(report: TwoClusterVirtualProcessReport) -> dict[str, Any]:
    return {
        "cluster_energy": report.cluster_energy,
        "cluster_singlet_gap": report.cluster_singlet_gap,
        "intercluster_bonds": [
            {
                "full_sites": [bond.first, bond.second],
                "cluster_local_sites": [bond.first_local, bond.second_local],
                "coupling": bond.coupling,
                "shell": bond.shell,
            }
            for bond in report.intercluster_bonds
        ],
        "triplet_multiplet_count": len(report.triplet_energies),
        "triplet_energies": _array(report.triplet_energies),
        "response_amplitude_matrix": _array(report.response_amplitude_matrix),
        "action_norm_squared": report.action_norm_squared,
        "first_order_expectation": report.first_order_expectation,
        "second_order_energy": report.second_order_energy,
        "cluster_spin_sector_resolution": {
            "initial_q_pair": [0, 0],
            "one_boundary_action_q_pair": [2, 2],
            "q2_q2_norm_squared": report.q2_q2_norm_squared,
            "outside_q2_q2_norm_bound": report.outside_q2_q2_norm_bound,
            "rank_one_selection_residual": report.rank_one_selection_residual,
            "interpretation": (
                "One scalar boundary exchange acting on two cluster singlets "
                "creates one S=1 (q=2) multiplet on each cluster; q>=4 is "
                "absent at this order."
            ),
        },
        "low_energy_triplet_ordering": {
            "cumulative_action_norm_fractions": _array(
                report.cumulative_low_energy_triplet_norm_fractions
            ),
            "cumulative_second_order_energy_fractions": _array(
                report.cumulative_low_energy_triplet_e2_fractions
            ),
            "lowest_pair_norm_fraction": report.lowest_triplet_pair_norm_fraction,
            "lowest_pair_second_order_energy_fraction": (
                report.lowest_triplet_pair_e2_fraction
            ),
        },
        "raw_boundary_response_schmidt": {
            "rank": report.response_schmidt_rank,
            "singular_values": _array(report.response_singular_values),
            "cumulative_action_norm_fractions": _array(
                report.response_cumulative_norm_fractions
            ),
            "variational_second_order_energy_fractions": _array(
                report.raw_response_carrier_e2_fractions
            ),
            "leading_left_lowest_triplet_overlap": (
                report.leading_left_lowest_triplet_overlap
            ),
            "leading_right_lowest_triplet_overlap": (
                report.leading_right_lowest_triplet_overlap
            ),
            "leading_left_right_overlap": (
                report.response_leading_left_right_overlap
            ),
            "matrix_asymmetry_fraction": report.response_asymmetry_fraction,
        },
        "resolvent_dressed_correction_schmidt": {
            "correction_matrix": _array(report.correction_amplitude_matrix),
            "singular_values": _array(report.correction_singular_values),
            "cumulative_correction_norm_fractions": _array(
                report.correction_cumulative_norm_fractions
            ),
            "variational_second_order_energy_fractions": _array(
                report.correction_carrier_e2_fractions
            ),
            "truncated_svd_second_order_energy_fractions": _array(
                report.correction_truncated_svd_e2_fractions
            ),
            "leading_left_right_overlap": (
                report.correction_leading_left_right_overlap
            ),
        },
        "carrier_design_readout": (
            "Select geometry/support-aware resolvent-dressed boundary-response "
            "carriers, retaining distinct left and right singular spaces; the "
            "energy-lowest triplet alone is not the optimal compressed carrier."
        ),
        "open_full_coupling": {
            "boundary_scale_lambda": 1.0,
            "product_energy": report.open_product_energy,
            "exact_ground_energy": report.open_exact_ground_energy,
            "exact_coupling_energy_gain": report.open_coupling_energy_gain,
            "second_order_fraction_of_exact_coupling_gain": (
                report.second_order_open_gap_fraction
            ),
            "product_ground_fidelity": report.open_product_ground_fidelity,
            "left_cluster_core_probability": report.open_left_core_probability,
            "right_cluster_core_probability": report.open_right_core_probability,
            "cluster_pair_sector_q_values": _array(
                report.exact_ground_sector_q_values
            ),
            "equal_q_sector_weights": _array(
                report.exact_ground_equal_q_sector_weights
            ),
            "offdiagonal_q_sector_weight": (
                report.exact_ground_offdiagonal_q_weight
            ),
            "excited_q0_pair_weight": report.exact_ground_excited_q0_pair_weight,
            "response_matrix_physical_residual": (
                report.response_matrix_physical_residual
            ),
            "carrier_subspace_definition": (
                "Every rank-k arm contains |gg> and all k^2 total-singlet "
                "products of the leading k left and leading k right q=2 "
                "carrier vectors; energies are direct lambda=1 Rayleigh-Ritz "
                "values, not perturbative extrapolations."
            ),
            "carrier_results": [
                {
                    "carrier": row.carrier,
                    "left_carrier_count": row.left_carrier_count,
                    "right_carrier_count": row.right_carrier_count,
                    "triplet_pair_dimension": row.triplet_pair_dimension,
                    "total_subspace_dimension": row.total_subspace_dimension,
                    "ground_energy": row.ground_energy,
                    "recovered_exact_gap_fraction": (
                        row.recovered_exact_gap_fraction
                    ),
                }
                for row in report.full_coupling_carriers
            ],
        },
    }


def _resonance_payload(report: CoveringResonanceReport) -> dict[str, Any]:
    return {
        "boundary_convention": report.boundary_convention,
        "raw_translation_count": report.raw_translation_count,
        "all_distinct_translations": {
            "covering_count": report.distinct_covering_count,
            "subspace_rank": report.subspace_rank,
            "overlap_matrix": _array(report.overlap_matrix),
            "projected_hamiltonian": _array(report.projected_hamiltonian),
            "overlap_eigenvalues": _array(report.overlap_eigenvalues),
            "uniform_coherent_energy": report.uniform_coherent_energy,
            "generalized_ground_energy": report.generalized_ground_energy,
            "resonance_energy_gain": report.resonance_energy_gain,
            "recovered_exact_gap_fraction": report.recovered_exact_gap_fraction,
        },
        "column_only_orbit": {
            "covering_count": report.column_only_covering_count,
            "subspace_rank": report.column_only_subspace_rank,
            "generalized_ground_energy": report.column_only_ground_energy,
            "resonance_energy_gain": report.column_only_resonance_energy_gain,
            "recovered_exact_gap_fraction": (
                report.column_only_recovered_exact_gap_fraction
            ),
        },
        "single_covering_energy": report.single_covering_energy,
        "exact_ground_energy": report.exact_ground_energy,
        "interpretation": (
            "This globally singlet covering span quantifies a q=0 resonance "
            "mechanism separate from the one-exchange q=2 virtual pair."
        ),
    }


def _interface_payload(report: BoundaryInterfaceResponse) -> dict[str, Any]:
    rank = report.correction_schmidt_rank
    return {
        "interface_id": report.interface_id,
        "direction_labels": list(report.direction_labels),
        "occurrence_count_over_six_distinct_tilings": report.occurrence_count,
        "bond_count_per_occurrence": report.bond_count_per_occurrence,
        "coupling_matrix": _array(report.coupling_matrix),
        "action_norm_squared": report.action_norm_squared,
        "second_order_energy": report.second_order_energy,
        "correction_schmidt_rank": rank,
        "correction_singular_values": _array(report.correction_singular_values),
        "correction_left_modes": _array(report.correction_left_modes[:, :rank]),
        "correction_right_modes": _array(report.correction_right_modes[:, :rank]),
        "leading_left_right_overlap_squared": report.leading_left_right_overlap,
        "route_conditioned_second_order_energy_fractions": _array(
            report.route_conditioned_e2_fractions
        ),
    }


def _union_payload(report: CarrierUnionAudit) -> dict[str, Any]:
    return {
        "family": report.family,
        "unique_interface_count": report.unique_interface_count,
        "weighted_occurrence_count": report.weighted_occurrence_count,
        "mean_action_norm_squared_per_distinct_tiling": (
            report.mean_action_norm_squared_per_distinct_tiling
        ),
        "mean_second_order_energy_per_distinct_tiling": (
            report.mean_second_order_energy_per_distinct_tiling
        ),
        "union_definition": (
            "SVD of the occurrence-weighted concatenation of every dressed "
            "correction X and X.T; this keeps distinct left/right spaces."
        ),
        "union_singular_values": _array(report.union_singular_values),
        "union_rank": report.union_rank,
        "union_cumulative_weight_fractions": _array(
            report.union_cumulative_weight_fractions
        ),
        "route_k_union_ranks": _array(report.route_k_union_ranks),
        "route_conditioned_second_order_energy_fractions": _array(
            report.route_conditioned_e2_fractions
        ),
        "shared_bank_vectors": _array(report.shared_bank_vectors),
        "shared_bank_correction_norm_fractions": _array(
            report.shared_bank_correction_norm_fractions
        ),
        "shared_bank_second_order_energy_fractions": _array(
            report.shared_bank_e2_fractions
        ),
        "minimum_shared_carriers": {
            "90_percent_second_order_energy": (
                report.minimum_shared_carriers_for_90pct_e2
            ),
            "99_percent_second_order_energy": (
                report.minimum_shared_carriers_for_99pct_e2
            ),
            "99p9_percent_second_order_energy": (
                report.minimum_shared_carriers_for_99p9pct_e2
            ),
        },
        "threshold_count_scope": (
            "Minimum prefix length in the reported joint-covariance ordering; "
            "not a global optimization over every shared local subspace."
        ),
    }


def _orientation_payload(
    report: SupportOrientationCarrierReport,
) -> dict[str, Any]:
    return {
        "lattice_shape": list(report.lattice_shape),
        "cluster_shape": list(report.cluster_shape),
        "translation_group_order": report.translation_group_order,
        "distinct_tiling_count": report.distinct_tiling_count,
        "tiling_stabilizer_order": report.tiling_stabilizer_order,
        "representative_translation_shifts": [
            list(shift) for shift in report.representative_translation_shifts
        ],
        "direction_classes": [
            {
                "label": row.label,
                "shell": row.shell,
                "displacement": list(row.displacement),
                "coupling": row.coupling,
                "periodic_bond_count": row.periodic_bond_count,
            }
            for row in report.direction_classes
        ],
        "periodic_bond_reconstruction_error": (
            report.periodic_bond_reconstruction_error
        ),
        "direction_resolved": {
            "interfaces": [
                _interface_payload(row)
                for row in report.direction_resolved_interfaces
            ],
            "union": _union_payload(report.direction_resolved_union),
            "energy_scope": (
                "Support-design diagnostic: directions are evaluated "
                "separately, so cross-direction interference is omitted."
            ),
        },
        "physical_pair_total": {
            "interfaces": [
                _interface_payload(row)
                for row in report.physical_pair_total_interfaces
            ],
            "union": _union_payload(report.physical_pair_total_union),
            "energy_scope": (
                "All bonds joining the same block pair are summed before the "
                "response; this is the physical second-order pair ledger."
            ),
        },
        "design_readout": (
            "A different leading carrier for each interface is highly compact "
            "but their union is not. A shared local bank must be selected from "
            "the joint dressed left/right covariance, with geometry-dependent "
            "multiplicity fixed by the reported energy-recovery curve."
        ),
        "claim_boundary": report.claim_boundary,
    }


def run_two_cluster_virtual_processes(
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the two deterministic N=12 exact diagnostics and seal one artifact."""

    started = time.perf_counter()
    config_path = config_path.resolve()
    output = output.resolve()
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result exists; pass --overwrite: {result_path}")
    source_manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    runtime_identity = numerical_runtime_identity()
    case_payloads = []
    for case in spec["cases"]:
        geometry = str(case["geometry"])
        ratio = float(case["j2_over_j1"])
        partition = two_cluster_open_partition(geometry, ratio)
        virtual = two_cluster_virtual_process_report(geometry, ratio)
        resonance = translated_covering_resonance_report(geometry, ratio)
        orientation = support_orientation_carrier_report(geometry, ratio)
        case_payloads.append(
            {
                "case_id": case["case_id"],
                "geometry": geometry,
                "j2_over_j1": ratio,
                "open_partition_reconstruction_error": (
                    partition.reconstruction_error
                ),
                "virtual_pair": _virtual_payload(virtual),
                "covering_resonance": _resonance_payload(resonance),
                "support_orientation": _orientation_payload(orientation),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": source_manifest,
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "cases": case_payloads,
        "total_wall_seconds": time.perf_counter() - started,
        "hamiltonian_diagonalization": {
            "open_cluster_sites": 6,
            "periodic_torus_sites": 12,
            "n36_ed_performed": False,
        },
        "mechanism_summary": {
            "single_boundary_exchange": "two_cluster_q2_virtual_pair_only",
            "full_coupling_open_ground": (
                "about_80_percent_cluster_core_and_20_percent_q2_pair_weight"
            ),
            "q0_mechanism": "translated_covering_resonance",
            "recommended_carrier": (
                "resolvent_dressed_support_aware_left_right_response_modes"
            ),
            "orientation_result": (
                "interface_specific_leading_modes_are_compact_but_a_shared_"
                "translation_bank_requires_multiple_local_q2_carriers"
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    payload = run_two_cluster_virtual_processes(
        arguments.config,
        arguments.output,
        overwrite=arguments.overwrite,
    )
    print(json.dumps({
        "status": payload["status"],
        "result_sha256": payload["result_sha256"],
        "output": str((arguments.output / "results.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "TASK",
    "run_two_cluster_virtual_processes",
]
