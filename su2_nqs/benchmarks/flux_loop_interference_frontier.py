"""Exact-small gate for flux-sector interference plus local dimer resonance.

This benchmark tests a state-manifold change rather than another response
head on one carrier.  The common variational space contains a projected
pi/0-flux carrier, its two Hamiltonian Krylov descendants, and three fixed
paired-Z2 carriers.  A single physical extension then acts on each paired
carrier with the full-lattice A1 sum of opposite-edge singlet projectors.

Every column is an exact spin singlet in the Gamma-A1 spatial sector.  The
causal control actively scrambles the physical dimer-response columns by a
non-geometric site permutation and projects them back to the same exact
sector.  Hence any advantage cannot be attributed to extra columns, spin
leakage, or a larger spatial sector.  A degree-matched Hamiltonian Krylov
space remains the strict variational-frontier control.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import sparse

from ..lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from ..operators.square_a1_scalar import square_a1_scalar_token_connectivity
from .adaptive_projected_pair_carrier import batched_singlet_agp_amplitudes
from .clebsch_tree_exact import physical_permutation_index_table
from .four_cell_additive_carrier import _case_data as four_cell_case_data
from .fusion_spatial_bridge import spin_raising_relative_norm
from .paired_z2_carrier import paired_z2_kernel_bank
from .qsl_three_arm import triangular_pi_flux_operator_components
from .rvb_covering_frontier import prepare_rvb_frontier_case
from .rvb_covering_oracle import RitzState, lowest_ritz_state
from .scalar_operator_frontier import triangular_a1_resonance_operators
from .xdiag_physical_seed_krylov import SMALL_CASES


CaseName = Literal["triangular_n16_j2_0.125", "square_n16_j2_0.5"]


def _normalize(vector: np.ndarray) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state must have nonzero finite norm")
    return state / norm


def _ritz_row(result: RitzState, n_sites: int) -> dict[str, float | int]:
    return {
        "energy_total": result.energy,
        "energy_per_site": result.energy / n_sites,
        "variance_total": result.variance,
        "variance_per_site": result.variance / n_sites,
        "rank": result.rank,
        "column_count": int(result.coefficients.size),
    }


def _reynolds(vector: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(vector)[indices], axis=0)


def _paired_carriers(
    configurations: np.ndarray,
    geometry: Literal["triangular_n16", "square_n16"],
    permutation_indices: np.ndarray,
) -> tuple[np.ndarray, tuple[dict[str, float | int | str], ...]]:
    columns = []
    diagnostics = []
    for kernel in paired_z2_kernel_bank(geometry):
        raw = batched_singlet_agp_amplitudes(configurations, kernel.pairing)
        projected = _normalize(_reynolds(raw, permutation_indices))
        columns.append(projected)
        diagnostics.append(
            {
                "label": kernel.label,
                "quasiparticle_gap": kernel.quasiparticle_gap,
                "u_condition_number": kernel.u_condition_number,
                "continuous_igg_nullity": kernel.continuous_igg_nullity,
            }
        )
    return np.column_stack(columns), tuple(diagnostics)


def _square_token_matrix(
    configurations: np.ndarray,
    *,
    linear_size: int,
    label: str,
) -> sparse.csr_matrix:
    states = tuple(tuple(int(bit) for bit in row) for row in configurations)
    positions = {state: index for index, state in enumerate(states)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for old_index, state in enumerate(states):
        connected, elements = square_a1_scalar_token_connectivity(
            state, linear_size
        )[label]
        for candidate, value in zip(connected, elements, strict=True):
            rows.append(positions[candidate])
            columns.append(old_index)
            values.append(float(value))
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(states), len(states))
    ).tocsr()


def _active_site_permutation(
    vector: np.ndarray,
    configurations: np.ndarray,
    image: Sequence[int],
) -> np.ndarray:
    states = tuple(tuple(int(bit) for bit in row) for row in configurations)
    n_sites = len(states[0])
    checked = tuple(int(site) for site in image)
    if tuple(sorted(checked)) != tuple(range(n_sites)):
        raise ValueError("image must be a site permutation")
    positions = {state: index for index, state in enumerate(states)}
    result = np.zeros_like(np.asarray(vector), dtype=np.complex128)
    for old_index, state in enumerate(states):
        transformed = [0] * n_sites
        for old_site, new_site in enumerate(checked):
            transformed[new_site] = state[old_site]
        result[positions[tuple(transformed)]] += vector[old_index]
    return result


def _krylov_columns(
    carrier: np.ndarray, hamiltonian, count: int
) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    columns = [_normalize(carrier)]
    while len(columns) < count:
        columns.append(np.asarray(hamiltonian @ columns[-1]))
    return np.column_stack(columns)


def _sector_maximum(
    columns: np.ndarray,
    configurations: np.ndarray,
    permutation_indices: np.ndarray,
) -> dict[str, float]:
    matrix = np.asarray(columns, dtype=np.complex128)
    spin = []
    spatial = []
    for column in matrix.T:
        state = _normalize(column)
        spin.append(spin_raising_relative_norm(state, configurations))
        spatial.append(
            float(np.linalg.norm(_reynolds(state, permutation_indices) - state))
        )
    return {
        "maximum_spin_raising_relative_norm": float(max(spin, default=0.0)),
        "maximum_gamma_a1_relative_residual": float(max(spatial, default=0.0)),
    }


def _frontier_case_data(name: CaseName):
    if name == "triangular_n16_j2_0.125":
        data = triangular_pi_flux_operator_components(0.125)
        configurations = np.asarray(data.configurations, dtype=np.int8)
        group = triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))
        dimer = triangular_a1_resonance_operators(
            SMALL_CASES["square4"], data.integer_basis
        )["rhombus_dimer_resonance"]
        return (
            configurations,
            data.hamiltonian,
            _normalize(data.carrier),
            tuple(group),
            dimer,
            "triangular_n16",
        )
    if name == "square_n16_j2_0.5":
        data = prepare_rvb_frontier_case("square_n16_j2_0.5")
        configurations = np.asarray(data.configurations, dtype=np.int8)
        dimer = _square_token_matrix(
            configurations,
            linear_size=4,
            label="plaquette_dimer_resonance",
        )
        return (
            configurations,
            data.hamiltonian,
            _normalize(data.carrier),
            tuple(square_gamma_a1_group(4)),
            dimer,
            "square_n16",
        )
    raise ValueError(f"unknown flux-loop case: {name!r}")


def flux_loop_interference_case(
    name: CaseName,
    *,
    scramble_seeds: Sequence[int],
) -> dict[str, object]:
    """Run one exact-support physical-versus-scrambled causal gate."""

    seeds = tuple(int(seed) for seed in scramble_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("scramble_seeds must be distinct nonnegative integers")
    (
        configurations,
        hamiltonian,
        carrier,
        group,
        dimer_operator,
        geometry,
    ) = _frontier_case_data(name)
    n_sites = configurations.shape[1]
    permutation_indices = physical_permutation_index_table(configurations, group)
    carrier = _normalize(_reynolds(carrier, permutation_indices))
    paired, kernel_diagnostics = _paired_carriers(
        configurations, geometry, permutation_indices
    )
    krylov_three = _krylov_columns(carrier, hamiltonian, 3)
    phase_columns = np.column_stack((krylov_three, paired))
    physical_responses = np.column_stack(
        [np.asarray(dimer_operator @ paired[:, index]) for index in range(3)]
    )
    physical_columns = np.column_stack((phase_columns, physical_responses))

    two_lanczos = lowest_ritz_state(krylov_three, hamiltonian)
    phase = lowest_ritz_state(phase_columns, hamiltonian)
    physical = lowest_ritz_state(physical_columns, hamiltonian)
    matched_krylov = lowest_ritz_state(
        _krylov_columns(carrier, hamiltonian, physical.rank), hamiltonian
    )

    scramble_rows = []
    scrambled_sector_maxima = []
    for seed in seeds:
        image = tuple(np.random.default_rng(seed).permutation(n_sites).tolist())
        responses = np.column_stack(
            [
                _normalize(
                    _reynolds(
                        _active_site_permutation(
                            physical_responses[:, index], configurations, image
                        ),
                        permutation_indices,
                    )
                )
                for index in range(physical_responses.shape[1])
            ]
        )
        scrambled_sector_maxima.append(
            _sector_maximum(responses, configurations, permutation_indices)
        )
        result = lowest_ritz_state(
            np.column_stack((phase_columns, responses)), hamiltonian
        )
        scramble_rows.append(
            {
                "seed": seed,
                "site_permutation": list(image),
                **_ritz_row(result, n_sites),
            }
        )

    physical_energy = physical.energy / n_sites
    physical_variance = physical.variance / n_sites
    energy_wins = sum(
        physical_energy < float(row["energy_per_site"]) - 1.0e-12
        for row in scramble_rows
    )
    joint_wins = sum(
        physical_energy < float(row["energy_per_site"]) - 1.0e-12
        and physical_variance < float(row["variance_per_site"]) - 1.0e-12
        for row in scramble_rows
    )
    sector_audit = {
        "physical_columns": _sector_maximum(
            physical_columns, configurations, permutation_indices
        ),
        "scrambled_response_columns": {
            "maximum_spin_raising_relative_norm": max(
                row["maximum_spin_raising_relative_norm"]
                for row in scrambled_sector_maxima
            ),
            "maximum_gamma_a1_relative_residual": max(
                row["maximum_gamma_a1_relative_residual"]
                for row in scrambled_sector_maxima
            ),
        },
    }
    exact_sector = all(
        value < 1.0e-10
        for block in sector_audit.values()
        for value in block.values()
    )
    phase_improves = bool(
        phase.energy < two_lanczos.energy - 1.0e-11
        and phase.variance < two_lanczos.variance - 1.0e-11
    )
    response_improves = bool(
        physical.energy < phase.energy - 1.0e-11
        and physical.variance < phase.variance - 1.0e-11
    )
    scrambles_match_rank = all(
        int(row["rank"]) == physical.rank for row in scramble_rows
    )
    all_scrambles = bool(
        scrambles_match_rank
        and energy_wins == len(seeds)
        and joint_wins == len(seeds)
    )
    beats_krylov = bool(
        physical.energy < matched_krylov.energy - 1.0e-11
        and physical.variance < matched_krylov.variance - 1.0e-11
    )
    return {
        "case": name,
        "n_sites": n_sites,
        "fixed_m_dimension": len(configurations),
        "space_group_order": len(group),
        "paired_kernel_diagnostics": list(kernel_diagnostics),
        "carrier_overlap_matrix": np.asarray(
            np.column_stack((carrier, paired)).conj().T
            @ np.column_stack((carrier, paired))
        ).real.tolist(),
        "two_lanczos": _ritz_row(two_lanczos, n_sites),
        "cross_flux_phase_bank": _ritz_row(phase, n_sites),
        "phase_bank_plus_dimer_resonance": _ritz_row(physical, n_sites),
        "dimension_matched_krylov": _ritz_row(matched_krylov, n_sites),
        "phase_bank_gain_over_two_lanczos_per_site": (
            phase.energy - two_lanczos.energy
        )
        / n_sites,
        "dimer_gain_over_phase_bank_per_site": (
            physical.energy - phase.energy
        )
        / n_sites,
        "physical_minus_matched_krylov_per_site": (
            physical.energy - matched_krylov.energy
        )
        / n_sites,
        "scrambled_controls": scramble_rows,
        "physical_energy_wins_over_scramble": energy_wins,
        "physical_joint_energy_variance_wins_over_scramble": joint_wins,
        "scramble_count": len(seeds),
        "directional_all_win_null_probability": 0.5 ** len(seeds),
        "sector_audit": sector_audit,
        "gates": {
            "exact_s0_gamma_a1": exact_sector,
            "phase_bank_improves_two_lanczos_energy_and_variance": phase_improves,
            "dimer_resonance_improves_phase_bank_energy_and_variance": response_improves,
            "every_scramble_matches_physical_variational_rank": scrambles_match_rank,
            "physical_dimer_beats_every_scramble_energy_and_variance": all_scrambles,
            "physical_dimer_beats_dimension_matched_krylov_energy_and_variance": beats_krylov,
        },
    }


def triangular_n12_saturation_control() -> dict[str, object]:
    """Show why triangular N=12 cannot validate this carrier mechanism."""

    data = prepare_rvb_frontier_case("triangular_n12_j2_0.125")
    configurations = np.asarray(data.configurations, dtype=np.int8)
    indices = physical_permutation_index_table(configurations, data.group)
    paired, _diagnostics = _paired_carriers(
        configurations, "triangular_n12", indices
    )
    carrier = _normalize(_reynolds(data.carrier, indices))
    carrier_rank = int(
        np.linalg.matrix_rank(
            np.column_stack((carrier, paired)), tol=1.0e-11
        )
    )
    exact = _case_data_exact_singlet_a1_dimension()
    return {
        "case": "triangular_n12_j2_0.125",
        "carrier_bank_rank": carrier_rank,
        "exact_s0_gamma_a1_dimension": exact,
        "saturated": carrier_rank == exact,
        "interpretation": (
            "The four projected flux carriers span the entire S=0, Gamma-A1 "
            "space, so an exact N=12 energy would be a finite-sector saturation "
            "artifact rather than evidence for the dimer mechanism."
        ),
    }


def _case_data_exact_singlet_a1_dimension() -> int:
    data = _case_data_for_saturation()
    basis = np.asarray(data["singlet_basis"], dtype=np.complex128)
    indices = np.asarray(data["indices"], dtype=np.int32)
    projected = np.column_stack(
        [_reynolds(basis[:, index], indices) for index in range(basis.shape[1])]
    )
    return int(np.linalg.matrix_rank(projected, tol=1.0e-11))


def _case_data_for_saturation() -> dict[str, object]:
    # Kept behind a tiny wrapper to make the use of the existing exact Schur
    # reference explicit and easy to replace without touching the gate logic.
    return four_cell_case_data("triangular_n12_j2_0125")


def flux_loop_interference_frontier(
    *, scramble_seeds: Sequence[int]
) -> dict[str, object]:
    """Run the registered triangular/square confirmation suite."""

    triangular = flux_loop_interference_case(
        "triangular_n16_j2_0.125", scramble_seeds=scramble_seeds
    )
    square = flux_loop_interference_case(
        "square_n16_j2_0.5", scramble_seeds=scramble_seeds
    )
    mechanism = all(
        case["gates"][gate]
        for case in (triangular, square)
        for gate in (
            "exact_s0_gamma_a1",
            "phase_bank_improves_two_lanczos_energy_and_variance",
            "dimer_resonance_improves_phase_bank_energy_and_variance",
            "physical_dimer_beats_every_scramble_energy_and_variance",
        )
    )
    frontier = all(
        case["gates"][
            "physical_dimer_beats_dimension_matched_krylov_energy_and_variance"
        ]
        for case in (triangular, square)
    )
    return {
        "triangular_n12_saturation_control": triangular_n12_saturation_control(),
        "triangular_n16": triangular,
        "square_n16": square,
        "decision": {
            "cross_geometry_mechanism_gate_passed": mechanism,
            "variational_frontier_gate_passed": frontier,
            "n36_sampled_preflight_licensed": mechanism,
            "n36_gpu_production_licensed": mechanism and frontier,
            "branch": (
                "promote_to_n36_gpu_production"
                if mechanism and frontier
                else "promote_only_to_bounded_n36_sampled_preflight"
                if mechanism
                else "freeze_flux_loop_interference_manifold"
            ),
        },
    }


__all__ = [
    "flux_loop_interference_case",
    "flux_loop_interference_frontier",
    "triangular_n12_saturation_control",
]
