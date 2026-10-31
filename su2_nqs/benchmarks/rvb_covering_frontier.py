"""Local state-manifold gate for a symmetry-built-in RVB representation.

The question is intentionally asked before introducing a Transformer: does
the complete nearest-neighbour valence-bond manifold contain a useful
``Gamma-A1`` direction beyond a competitive carrier's degree-two Hamiltonian
Krylov space?  If the exact Ritz oracle does not improve that nested control,
no learned parametrization of the same covering manifold can create a
variational win.

The registered cases are the full-D6 triangular ``N=12`` torus at
``J2/J1=0.125`` and the square ``4x4`` torus at ``J2/J1=0.5``.  They reuse the
existing physical carriers and do not diagonalize the many-body Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from .clebsch_tree_exact import (
    graph_hamiltonian_coo,
    physical_permutation_index_table,
)
from .fusion_spatial_bridge import spin_raising_relative_norm
from .rvb_covering_oracle import (
    CoveringRelationFeatures,
    DimerMatching,
    DimerMatchingOrbit,
    RVBOrbitColumns,
    RitzState,
    covering_relation_features,
    dimer_matching_orbits,
    enumerate_perfect_matchings,
    lowest_ritz_state,
    rvb_orbit_columns,
    two_lanczos_state,
)
from .xdiag_physical_seed_krylov import SMALL_CASES, physical_seed_amplitudes


TRIANGULAR_N12_SUPERCELL = ((6, 4), (0, 2))


@dataclass(frozen=True)
class RVBFrontierCaseData:
    """Reusable exact-small data for the oracle and learned covering gates."""

    name: str
    n_sites: int
    configurations: tuple[tuple[int, ...], ...]
    hamiltonian: object
    carrier: np.ndarray
    carrier_gamma_a1_projection_weight: float
    group: tuple[tuple[int, ...], ...]
    matchings: tuple[DimerMatching, ...]
    orbits: tuple[DimerMatchingOrbit, ...]
    orbit_columns: RVBOrbitColumns
    relation_features: CoveringRelationFeatures


def _normalize(vector: np.ndarray) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(state))
    if norm <= 1.0e-14 or not np.isfinite(norm):
        raise ValueError("state must have nonzero finite norm")
    return state / norm


def _state_metrics(state: np.ndarray, hamiltonian) -> dict[str, float]:
    vector = _normalize(state)
    action = np.asarray(hamiltonian @ vector)
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return {
        "energy_total": energy,
        "variance_total": float(np.vdot(residual, residual).real),
    }


def _ritz_metrics(result: RitzState, *, n_sites: int) -> dict[str, object]:
    singular = np.asarray(result.singular_values)
    return {
        "energy_total": result.energy,
        "energy_per_site": result.energy / n_sites,
        "variance_total": result.variance,
        "variance_per_site": result.variance / n_sites,
        "rank": result.rank,
        "column_count": int(result.coefficients.size),
        "smallest_retained_singular_value": float(singular[result.rank - 1]),
        "largest_singular_value": float(singular[0]),
    }


def _sector_metrics(state, configurations, group) -> dict[str, float]:
    vector = _normalize(state)
    indices = physical_permutation_index_table(configurations, group)
    projected = np.mean(vector[indices], axis=0)
    return {
        "spin_raising_relative_norm": spin_raising_relative_norm(
            vector, configurations
        ),
        "gamma_a1_relative_residual": float(
            np.linalg.norm(projected - vector) / np.linalg.norm(vector)
        ),
    }


def prepare_rvb_frontier_case(name: str) -> RVBFrontierCaseData:
    """Build one registered case without diagonalizing the Hamiltonian."""

    if name == "triangular_n12_j2_0.125":
        n_sites = 12
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_N12_SUPERCELL)
        matching_bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.0
        )
        hamiltonian_bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.125
        )
        carrier_case = SMALL_CASES["hnf6x2t4"]
    elif name == "square_n16_j2_0.5":
        n_sites = 16
        group = square_gamma_a1_group(4)
        matching_bonds = square_j1j2_bonds(
            4, 4, j1=1.0, j2=0.0, periodic=True
        )
        hamiltonian_bonds = square_j1j2_bonds(
            4, 4, j1=1.0, j2=0.5, periodic=True
        )
        carrier_case = SMALL_CASES["square4"]
    else:
        raise ValueError(f"unknown RVB frontier case: {name}")

    rows, columns, values, configurations = graph_hamiltonian_coo(
        n_sites, 0, hamiltonian_bonds
    )
    hamiltonian = coo_matrix(
        (values, (rows, columns)),
        shape=(len(configurations), len(configurations)),
    ).tocsr()
    carrier = physical_seed_amplitudes(
        np.asarray(configurations, dtype=np.int8), carrier_case
    )[:, 0]
    carrier = _normalize(carrier)
    permutation_indices = physical_permutation_index_table(configurations, group)
    projected_carrier = np.mean(carrier[permutation_indices], axis=0)
    carrier_projection_weight = float(np.vdot(projected_carrier, projected_carrier).real)
    if carrier_projection_weight <= 1.0e-14:
        raise RuntimeError("carrier has no weight in the registered Gamma-A1 sector")
    carrier = _normalize(projected_carrier)
    matchings = enumerate_perfect_matchings(n_sites, matching_bonds)
    orbits = dimer_matching_orbits(matchings, group)
    orbit_data = rvb_orbit_columns(configurations, orbits)
    nonzero_orbits = tuple(orbit for orbit in orbits if orbit.has_a1_vector)
    features = covering_relation_features(
        tuple(orbit.representative for orbit in nonzero_orbits),
        matching_bonds,
        group,
    )
    return RVBFrontierCaseData(
        name=name,
        n_sites=n_sites,
        configurations=configurations,
        hamiltonian=hamiltonian,
        carrier=carrier,
        carrier_gamma_a1_projection_weight=carrier_projection_weight,
        group=tuple(group),
        matchings=matchings,
        orbits=orbits,
        orbit_columns=orbit_data,
        relation_features=features,
    )


def _case(data: RVBFrontierCaseData) -> dict[str, object]:
    name = data.name
    n_sites = data.n_sites
    configurations = data.configurations
    hamiltonian = data.hamiltonian
    carrier = data.carrier
    group = data.group
    matchings = data.matchings
    orbits = data.orbits
    orbit_data = data.orbit_columns
    rvb_columns = np.asarray(orbit_data.columns, dtype=np.complex128)
    carrier_action = np.asarray(hamiltonian @ carrier)
    carrier_action2 = np.asarray(hamiltonian @ carrier_action)
    krylov_columns = np.column_stack((carrier, carrier_action, carrier_action2))

    two_lanczos = two_lanczos_state(carrier, hamiltonian)
    rvb_only = lowest_ritz_state(rvb_columns, hamiltonian)
    carrier_plus_rvb = lowest_ritz_state(
        np.column_stack((carrier, rvb_columns)), hamiltonian
    )
    krylov_plus_rvb = lowest_ritz_state(
        np.column_stack((krylov_columns, rvb_columns)), hamiltonian
    )

    carrier_row = _state_metrics(carrier, hamiltonian)
    carrier_row.update(
        {
            "energy_per_site": carrier_row["energy_total"] / n_sites,
            "variance_per_site": carrier_row["variance_total"] / n_sites,
        }
    )
    nonzero_orbits = tuple(orbit for orbit in orbits if orbit.has_a1_vector)
    incremental = np.asarray(krylov_plus_rvb.coefficients[3:])
    result = {
        "case": name,
        "n_sites": n_sites,
        "fixed_m_dimension": len(configurations),
        "space_group_order": len(group),
        "matching_count": len(matchings),
        "matching_orbit_count": len(orbits),
        "nonzero_a1_orbit_count": len(nonzero_orbits),
        "vanishing_a1_orbit_count": len(orbits) - len(nonzero_orbits),
        "orbit_sizes": [orbit.orbit_size for orbit in orbits],
        "orbit_raw_norms": orbit_data.raw_norms.tolist(),
        "carrier_gamma_a1_projection_weight": data.carrier_gamma_a1_projection_weight,
        "carrier": carrier_row,
        "two_lanczos": _ritz_metrics(two_lanczos, n_sites=n_sites),
        "rvb_a1_oracle": _ritz_metrics(rvb_only, n_sites=n_sites),
        "carrier_plus_rvb_a1_oracle": _ritz_metrics(
            carrier_plus_rvb, n_sites=n_sites
        ),
        "two_lanczos_plus_rvb_a1_oracle": _ritz_metrics(
            krylov_plus_rvb, n_sites=n_sites
        ),
        "rvb_increment_beyond_two_lanczos_per_site": (
            krylov_plus_rvb.energy - two_lanczos.energy
        )
        / n_sites,
        "carrier_rvb_minus_two_lanczos_per_site": (
            carrier_plus_rvb.energy - two_lanczos.energy
        )
        / n_sites,
        "two_lanczos_to_augmented_fidelity": float(
            abs(np.vdot(two_lanczos.state, krylov_plus_rvb.state)) ** 2
        ),
        "augmented_rvb_coefficient_norm": float(np.linalg.norm(incremental)),
        "sector_audit": {
            "carrier": _sector_metrics(carrier, configurations, group),
            "two_lanczos": _sector_metrics(
                two_lanczos.state, configurations, group
            ),
            "rvb_a1_oracle": _sector_metrics(
                rvb_only.state, configurations, group
            ),
            "two_lanczos_plus_rvb_a1_oracle": _sector_metrics(
                krylov_plus_rvb.state, configurations, group
            ),
        },
    }
    return result


def rvb_covering_frontier() -> dict[str, object]:
    """Run the triangular-N12 and square-N16 short-range RVB oracle gate."""

    triangular = _case(prepare_rvb_frontier_case("triangular_n12_j2_0.125"))
    square = _case(prepare_rvb_frontier_case("square_n16_j2_0.5"))
    maximum_sector_residual = max(
        value
        for case in (triangular, square)
        for audit in case["sector_audit"].values()
        for value in audit.values()
    )
    return {
        "architecture": {
            "spin_symmetry": "each covering is a product of exact oriented singlets",
            "spatial_symmetry": "equal signed coefficients on matching orbits",
            "forward_reynolds_sum_required": False,
            "neural_optimization_used": False,
        },
        "triangular_n12": triangular,
        "square_n16": square,
        "maximum_sector_residual": float(maximum_sector_residual),
        "exact_sector_gate_passed": maximum_sector_residual < 1.0e-10,
        "rvb_direction_beyond_two_lanczos_present_both_cases": all(
            case["rvb_increment_beyond_two_lanczos_per_site"] < -1.0e-10
            for case in (triangular, square)
        ),
        "claim_boundary": (
            "This is an exact-small state-manifold oracle. It can license an "
            "invariant covering Transformer, but cannot establish neural "
            "compression, large-size scaling, or a QSL phase."
        ),
    }


__all__ = [
    "RVBFrontierCaseData",
    "TRIANGULAR_N12_SUPERCELL",
    "prepare_rvb_frontier_case",
    "rvb_covering_frontier",
]
