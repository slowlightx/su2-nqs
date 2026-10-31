"""Exact-small gate for a size-transferable plaquette-RVB initialization.

The carrier has one local mixing angle independent of system size:
``tensor_cells(cos(theta)|A1> + sin(theta)|B1>)``.  A terminal Gamma-A1
projection resonates translated/rotated tilings.  The scan reuses the sealed
triangular N=16 A1 Hamiltonian and builds only the small square N=16 fixed-M
matrix; it never diagonalizes a many-body Hamiltonian.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.four_cell_schur_frontier import SQUARE_N16_SUPPORTS
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.local_carrier_product_subspace import (
    coherent_square_plaquette_product,
    project_columns_by_index_table,
    square_a1_b1_product_columns,
)
from su2_nqs.benchmarks.rvb_covering_frontier import prepare_rvb_frontier_case
from su2_nqs.benchmarks.triangular_state_manifold import (
    TRIANGULAR_N16_RHOMBUS_SUPPORTS,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity


TASK = "spin_scale_first_plaquette_carrier_gate"


def _load_contract(config_path: Path) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid plaquette-carrier gate identity")
    scan = spec.get("angle_scan")
    gates = spec.get("gates")
    if not isinstance(scan, dict) or not isinstance(gates, dict):
        raise TypeError("angle_scan and gates must be objects")
    count = scan.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 17:
        raise ValueError("angle_scan.count must be an integer of at least 17")
    for name in ("minimum", "maximum"):
        if not np.isfinite(float(scan.get(name, np.nan))):
            raise ValueError(f"angle_scan.{name} must be finite")
    if float(scan["minimum"]) >= float(scan["maximum"]):
        raise ValueError("angle scan bounds must be ordered")
    for name in (
        "maximum_energy_per_site",
        "maximum_variance_per_site",
        "minimum_gamma_a1_projection_weight",
        "maximum_spin_raising_relative_norm",
        "maximum_gamma_a1_residual",
    ):
        if not np.isfinite(float(gates.get(name, np.nan))):
            raise ValueError(f"gates.{name} must be finite")
    return spec


def _metrics(state: np.ndarray, hamiltonian, *, n_sites: int) -> dict[str, float]:
    vector = np.asarray(state, dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    action = np.asarray(hamiltonian @ vector)
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    variance = float(np.vdot(residual, residual).real)
    return {
        "energy_total": energy,
        "energy_per_site": energy / n_sites,
        "variance_total": variance,
        "variance_per_site": variance / n_sites,
    }


def _angles(spec: dict[str, object]) -> np.ndarray:
    scan = spec["angle_scan"]
    values = np.linspace(
        float(scan["minimum"]),
        float(scan["maximum"]),
        int(scan["count"]),
        endpoint=False,
        dtype=np.float64,
    )
    return np.unique(np.concatenate((values, np.asarray((0.0,), dtype=np.float64))))


def _scan_projected_columns(
    projected_columns: np.ndarray,
    labels: tuple[tuple[int, ...], ...],
    hamiltonian,
    angles: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for angle in angles:
        state = coherent_square_plaquette_product(
            projected_columns, labels, float(angle)
        )
        rows.append({"angle": float(angle), **_metrics(state, hamiltonian, n_sites=16)})
    selected = min(rows, key=lambda row: (row["energy_total"], row["variance_total"]))
    return selected, rows


def run_spin_scale_first_plaquette_carrier_gate(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Scan the one-angle carrier on square and triangular N=16 supports."""

    spec = _load_contract(config_path)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    basis = fixed_m_spin_basis(16, 0)
    angles = _angles(spec)

    square = prepare_rvb_frontier_case("square_n16_j2_0.5")
    if square.configurations != basis:
        raise RuntimeError("square carrier and Hamiltonian bases disagree")
    square_columns, square_labels = square_a1_b1_product_columns(
        basis, SQUARE_N16_SUPPORTS
    )
    square_indices = physical_permutation_index_table(basis, square.group)
    square_projected = project_columns_by_index_table(
        square_columns, square_indices
    )
    square_selected, square_scan = _scan_projected_columns(
        square_projected, square_labels, square.hamiltonian, angles
    )
    square_raw = coherent_square_plaquette_product(
        square_columns, square_labels, square_selected["angle"]
    )
    square_state = coherent_square_plaquette_product(
        square_projected, square_labels, square_selected["angle"]
    )
    square_projection_weight = float(
        np.linalg.norm(
            project_columns_by_index_table(square_raw[:, None], square_indices)[:, 0]
        )
        ** 2
    )
    square_spatial_residual = float(
        np.linalg.norm(np.mean(square_state[square_indices], axis=0) - square_state)
    )

    cache_path = Path(str(spec["triangular_cache"]))
    if not cache_path.is_absolute():
        cache_path = (config_path.parent / cache_path).resolve()
    with np.load(cache_path) as cache:
        triangular_phi = np.asarray(cache["physical_a1_basis"], dtype=np.float64)
        triangular_h = np.asarray(
            cache["coupling_0.125_h_a1"], dtype=np.float64
        )
    triangular_columns, triangular_labels = square_a1_b1_product_columns(
        basis, TRIANGULAR_N16_RHOMBUS_SUPPORTS
    )
    triangular_reduced_columns = triangular_phi.T @ triangular_columns
    triangular_selected, triangular_scan = _scan_projected_columns(
        triangular_reduced_columns,
        triangular_labels,
        triangular_h,
        angles,
    )
    triangular_raw = coherent_square_plaquette_product(
        triangular_columns, triangular_labels, triangular_selected["angle"]
    )
    triangular_reduced = coherent_square_plaquette_product(
        triangular_reduced_columns,
        triangular_labels,
        triangular_selected["angle"],
    )
    triangular_state = triangular_phi @ triangular_reduced
    triangular_projection_weight = float(
        np.linalg.norm(triangular_phi.T @ triangular_raw) ** 2
    )
    triangular_spatial_residual = float(
        np.linalg.norm(
            triangular_phi @ (triangular_phi.T @ triangular_state)
            - triangular_state
        )
    )

    gates = spec["gates"]
    cases = []
    for case_id, selected, scan_rows, state, projection_weight, spatial_residual in (
        (
            "square_n16_j2_0.5",
            square_selected,
            square_scan,
            square_state,
            square_projection_weight,
            square_spatial_residual,
        ),
        (
            "triangular_n16_j2_0.125",
            triangular_selected,
            triangular_scan,
            triangular_state,
            triangular_projection_weight,
            triangular_spatial_residual,
        ),
    ):
        spin_residual = spin_raising_relative_norm(state, basis)
        local_a1 = next(row for row in scan_rows if row["angle"] == 0.0)
        passed = bool(
            selected["energy_per_site"]
            <= float(gates["maximum_energy_per_site"])
            and selected["variance_per_site"]
            <= float(gates["maximum_variance_per_site"])
            and projection_weight
            >= float(gates["minimum_gamma_a1_projection_weight"])
            and spin_residual
            <= float(gates["maximum_spin_raising_relative_norm"])
            and spatial_residual <= float(gates["maximum_gamma_a1_residual"])
        )
        cases.append(
            {
                "case_id": case_id,
                "selected": selected,
                "local_a1_angle_zero": local_a1,
                "energy_gain_from_local_a1_per_site": float(
                    selected["energy_per_site"] - local_a1["energy_per_site"]
                ),
                "gamma_a1_projection_weight": projection_weight,
                "spin_raising_relative_norm": spin_residual,
                "gamma_a1_relative_residual": spatial_residual,
                "scan_count": len(scan_rows),
                "passed": passed,
            }
        )

    passed = bool(cases and all(case["passed"] for case in cases))
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "carrier": {
            "local_space": "four_spin_q0_A1_plus_B1",
            "form": "tensor_cells(cos(theta) A1 + sin(theta) B1)",
            "trainable_parameter_count": 1,
            "terminal_projection": "Gamma-A1",
            "magnetic_indices_learned": False,
        },
        "cases": cases,
        "all_carrier_gates_passed": passed,
        "n36_promotion_allowed": passed,
        "no_many_body_diagonalization_performed": True,
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_load_contract", "run_spin_scale_first_plaquette_carrier_gate"]
