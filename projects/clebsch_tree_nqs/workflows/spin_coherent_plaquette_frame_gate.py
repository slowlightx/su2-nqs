"""Preregistered exact-N16 gate for coherent multi-frame plaquette carriers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    graph_hamiltonian_coo,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.coherent_plaquette_frames import (
    coherent_plaquette_frame_columns,
    distinct_partition_orbit,
    internal_bond_coverage,
    nongeometric_site_permutation,
    scrambled_coherent_frame_columns,
)
from su2_nqs.benchmarks.four_cell_schur_frontier import SQUARE_N16_SUPPORTS
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.local_carrier_product_subspace import (
    coherent_square_plaquette_product,
    project_columns_by_index_table,
    square_a1_b1_product_columns,
)
from su2_nqs.benchmarks.triangular_state_manifold import (
    TRIANGULAR_N16_RHOMBUS_SUPPORTS,
    TRIANGULAR_N16_SUPERCELL,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds


TASK = "spin_coherent_plaquette_frame_gate"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid coherent-frame gate identity")
    scan = spec.get("angle_scan")
    gates = spec.get("gates")
    if not isinstance(scan, dict) or not isinstance(gates, dict):
        raise TypeError("angle_scan and gates must be objects")
    if int(scan.get("count", 0)) < 33:
        raise ValueError("angle_scan.count must be at least 33")
    if float(scan.get("minimum", 0.0)) >= float(scan.get("maximum", 0.0)):
        raise ValueError("angle scan bounds must be ordered")
    seeds = tuple(int(seed) for seed in spec.get("development_scramble_seeds", ()))
    if len(seeds) < 8 or len(set(seeds)) != len(seeds):
        raise ValueError("at least eight distinct development scrambles are required")
    for name in (
        "minimum_energy_win_fraction",
        "minimum_joint_energy_variance_win_fraction",
        "minimum_frame_bond_coverage",
        "maximum_projection_equivalence_error",
        "maximum_spin_raising_relative_norm",
        "maximum_gamma_a1_relative_residual",
    ):
        value = float(gates.get(name, np.nan))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"gates.{name} must be finite and nonnegative")
    if spec.get("remote_training_on_pass") is not False:
        raise ValueError("this local gate must not directly authorize remote training")


def _angles(spec: dict[str, object]) -> np.ndarray:
    scan = spec["angle_scan"]
    values = np.linspace(
        float(scan["minimum"]),
        float(scan["maximum"]),
        int(scan["count"]),
        endpoint=True,
        dtype=np.float64,
    )
    return np.unique(np.concatenate((values, np.asarray((0.0,), dtype=np.float64))))


def _coefficient_vector(
    labels: tuple[tuple[int, ...], ...], angle: float
) -> np.ndarray:
    local = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float64)
    return np.asarray(
        [np.prod(local[np.asarray(label, dtype=np.int64)]) for label in labels],
        dtype=np.float64,
    )


def _scan_columns(columns, labels, hamiltonian, angles) -> tuple[dict, np.ndarray]:
    values = np.asarray(columns, dtype=np.float64)
    action = np.asarray(hamiltonian @ values)
    gram = values.T @ values
    h1 = values.T @ action
    h2 = action.T @ action
    rows = []
    for angle in angles:
        coefficients = _coefficient_vector(labels, float(angle))
        norm2 = float(coefficients @ gram @ coefficients)
        if norm2 <= 1.0e-18:
            continue
        energy = float(coefficients @ h1 @ coefficients / norm2)
        second = float(coefficients @ h2 @ coefficients / norm2)
        rows.append(
            {
                "angle": float(angle),
                "energy_total": energy,
                "energy_per_site": energy / 16.0,
                "variance_total": max(second - energy * energy, 0.0),
                "variance_per_site": max(second - energy * energy, 0.0) / 16.0,
            }
        )
    if not rows:
        raise RuntimeError("every coherent-frame angle had zero projected norm")
    selected = min(rows, key=lambda row: (row["energy_total"], row["variance_total"]))
    state = coherent_square_plaquette_product(
        values, labels, float(selected["angle"])
    )
    return selected, state


def _case_data(case_id: str):
    if case_id == "square_n16_j2_0.5":
        cells = SQUARE_N16_SUPPORTS
        group = square_gamma_a1_group(4)
        bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.5, periodic=True)
        j1_bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.0, periodic=True)
    elif case_id == "triangular_n16_j2_0.125":
        cells = TRIANGULAR_N16_RHOMBUS_SUPPORTS
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_N16_SUPERCELL)
        bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N16_SUPERCELL, j1=1.0, j2=0.125
        )
        j1_bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N16_SUPERCELL, j1=1.0, j2=0.0
        )
    else:
        raise ValueError(f"unknown coherent-frame case: {case_id}")
    return cells, group, bonds, j1_bonds


def _run_case(
    case_id: str,
    *,
    basis,
    angles: np.ndarray,
    scramble_seeds: tuple[int, ...],
    gates: dict[str, object],
) -> dict[str, object]:
    cells, group, bonds, j1_bonds = _case_data(case_id)
    rows, columns, values, checked = graph_hamiltonian_coo(16, 0, bonds)
    if checked != basis:
        raise RuntimeError("coherent-frame and Hamiltonian bases disagree")
    hamiltonian = coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    coherent = coherent_plaquette_frame_columns(basis, cells, group)
    selected, state = _scan_columns(
        coherent.columns, coherent.labels, hamiltonian, angles
    )

    primary_columns, primary_labels = square_a1_b1_product_columns(basis, cells)
    primary_state = coherent_square_plaquette_product(
        primary_columns, primary_labels, float(selected["angle"])
    )
    indices = physical_permutation_index_table(basis, group)
    projected_primary = project_columns_by_index_table(
        primary_state[:, None], indices
    )[:, 0]
    projected_primary /= np.linalg.norm(projected_primary)
    overlap = np.vdot(projected_primary, state)
    if overlap.real < 0.0:
        projected_primary *= -1.0
    equivalence_error = float(np.linalg.norm(projected_primary - state))

    controls = []
    for seed in scramble_seeds:
        permutation = nongeometric_site_permutation(
            n_sites=16, group=group, seed=seed
        )
        scrambled_columns = scrambled_coherent_frame_columns(
            coherent, basis, group, permutation
        )
        scrambled_selected, _scrambled_state = _scan_columns(
            scrambled_columns, coherent.labels, hamiltonian, angles
        )
        controls.append(
            {
                "seed": seed,
                "site_permutation": list(permutation),
                "selected": scrambled_selected,
                "physical_wins_energy": bool(
                    selected["energy_total"] < scrambled_selected["energy_total"]
                ),
                "physical_wins_energy_and_variance": bool(
                    selected["energy_total"] < scrambled_selected["energy_total"]
                    and selected["variance_total"]
                    < scrambled_selected["variance_total"]
                ),
            }
        )

    partitions = distinct_partition_orbit(cells, group)
    primary_coverage = internal_bond_coverage((cells,), j1_bonds)
    frame_coverage = internal_bond_coverage(partitions, j1_bonds)
    energy_fraction = float(
        np.mean([row["physical_wins_energy"] for row in controls])
    )
    joint_fraction = float(
        np.mean([row["physical_wins_energy_and_variance"] for row in controls])
    )
    group_indices = physical_permutation_index_table(basis, group)
    spatial_residual = float(np.linalg.norm(np.mean(state[group_indices], axis=0) - state))
    spin_residual = float(spin_raising_relative_norm(state, basis))
    passed = bool(
        energy_fraction >= float(gates["minimum_energy_win_fraction"])
        and joint_fraction
        >= float(gates["minimum_joint_energy_variance_win_fraction"])
        and float(frame_coverage["coverage_fraction"])
        >= float(gates["minimum_frame_bond_coverage"])
        and equivalence_error
        <= float(gates["maximum_projection_equivalence_error"])
        and spin_residual <= float(gates["maximum_spin_raising_relative_norm"])
        and spatial_residual <= float(gates["maximum_gamma_a1_relative_residual"])
    )
    return {
        "case_id": case_id,
        "physical": selected,
        "frame_group_size": coherent.group_size,
        "distinct_partition_count": coherent.distinct_partition_count,
        "primary_j1_bond_coverage": primary_coverage,
        "coherent_frame_j1_bond_coverage": frame_coverage,
        "projection_equals_explicit_frame_sum_error": equivalence_error,
        "spin_raising_relative_norm": spin_residual,
        "gamma_a1_relative_residual": spatial_residual,
        "scrambled_controls": controls,
        "energy_win_fraction": energy_fraction,
        "joint_energy_variance_win_fraction": joint_fraction,
        "median_scrambled_energy_per_site": float(
            np.median([row["selected"]["energy_per_site"] for row in controls])
        ),
        "median_scrambled_variance_per_site": float(
            np.median([row["selected"]["variance_per_site"] for row in controls])
        ),
        "passed": passed,
    }


def run_spin_coherent_plaquette_frame_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the square/triangular exact-small coherent-frame falsifier."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    basis = fixed_m_spin_basis(16, 0)
    angles = _angles(spec)
    seeds = tuple(int(seed) for seed in spec["development_scramble_seeds"])
    cases = [
        _run_case(
            case_id,
            basis=basis,
            angles=angles,
            scramble_seeds=seeds,
            gates=spec["gates"],
        )
        for case_id in (
            "square_n16_j2_0.5",
            "triangular_n16_j2_0.125",
        )
    ]
    passed = bool(all(case["passed"] for case in cases))
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "construction": {
            "carrier": "coherent translated/rotated four-spin plaquette singlets",
            "local_trainable_coordinate": "one shared A1/B1 mixing angle",
            "terminal_sector": "exact S=0, Gamma-A1",
            "control": "one global non-geometric site permutation of the complete frame bank followed by the same A1 projector",
            "magnetic_indices_learned": False,
        },
        "cases": cases,
        "all_exact_n16_gates_passed": passed,
        "n36_support_preflight_allowed": passed,
        "remote_training_allowed": False,
        "no_many_body_diagonalization_performed": True,
        "decision": (
            "implement_n36_coherent_frame_support_preflight"
            if passed
            else "stop_coherent_frame_route_before_n36"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_coherent_plaquette_frame_gate"]
