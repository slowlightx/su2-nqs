"""Symmetry-labelled loop-resonance screening on triangular ``4x4``.

The existing scalar Operator-ViT uses uniform ``A1`` sums of rhombus and
six-site loops.  This module asks a different physical question: does the
remaining variational residual live in *virtual* nontrivial spatial sectors?

Three orientation-resolved rhombus operators are decomposed into an ``E2``
doublet.  Both the even ring exchange and the opposite-edge singlet
resonance are closed back to ``A1`` as ``sum_mu O_mu^dagger O_mu``.  The
oriented ring current is an ``A2`` pseudoscalar and is closed as ``O^2``.
All elementary operators are SU(2) scalars; learned magnetic indices never
appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from ..lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from ..lattices.triangular_e2 import E2_ORIENTATION_FRAME, orientation_axis_permutation
from ..lattices.triangular_hnf import (
    C6,
    J1_POSITIVE,
    MX,
    hnf_site_permutation,
    hnf_sites,
    reduce_hnf,
)
from .clebsch_tree_exact import physical_permutation_index_table
from .qsl_three_arm import triangular_pi_flux_operator_components
from .scalar_operator_frontier import (
    _bond_sum_matrix,
    _cycle_matrix,
    triangular_a1_resonance_operators,
)
from .xdiag_physical_seed_krylov import (
    SMALL_CASES,
    spin_raising_relative_norm,
)


@dataclass(frozen=True)
class LabelledLoopOperators:
    """Orientation-resolved loop operators and their exact diagnostics."""

    e2_ring: tuple[sparse.csr_matrix, sparse.csr_matrix]
    e2_dimer: tuple[sparse.csr_matrix, sparse.csr_matrix]
    a2_current: sparse.csr_matrix
    a1_resonance: tuple[sparse.csr_matrix, ...]
    labels: tuple[str, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state must have finite nonzero norm")
    return value / norm


def _rhombus_pair_representation(operation) -> np.ndarray:
    """Return the real E2 matrix for the three unoriented rhombus axes."""

    pairs = ((0, 1), (0, 2), (1, 2))
    pair_position = {pair: index for index, pair in enumerate(pairs)}
    axis_map = orientation_axis_permutation(operation, J1_POSITIVE)
    mapping = tuple(
        pair_position[tuple(sorted((axis_map[first], axis_map[second])))]
        for first, second in pairs
    )
    permutation = np.zeros((3, 3), dtype=np.float64)
    for old, new in enumerate(mapping):
        permutation[new, old] = 1.0
    frame = E2_ORIENTATION_FRAME
    return frame @ permutation.T @ frame.T


def triangular_labelled_loop_operators() -> LabelledLoopOperators:
    """Build exact fixed-``Sz`` E2/A2 loop operators on triangular ``4x4``."""

    data = triangular_pi_flux_operator_components(0.125)
    case = SMALL_CASES["square4"]
    integer_basis = data.integer_basis
    sites = hnf_sites(case.supercell)
    positions = {site: index for index, site in enumerate(sites)}
    dimension = len(integer_basis)
    zero = sparse.csr_matrix((dimension, dimension), dtype=np.float64)
    identity = sparse.identity(dimension, dtype=np.float64, format="csr")
    rings = [zero.copy() for _ in range(3)]
    dimers = [zero.copy() for _ in range(3)]
    current = sparse.csr_matrix((dimension, dimension), dtype=np.complex128)
    bond_cache: dict[tuple[int, int], sparse.csr_matrix] = {}

    def singlet(first: int, second: int) -> sparse.csr_matrix:
        bond = tuple(sorted((int(first), int(second))))
        if bond not in bond_cache:
            bond_cache[bond] = _bond_sum_matrix(
                integer_basis, case.n_sites, (bond,)
            )
        return 0.25 * identity - bond_cache[bond]

    axis_pairs = ((0, 1), (0, 2), (1, 2))
    for orientation, (first_axis, second_axis) in enumerate(axis_pairs):
        first_step = J1_POSITIVE[first_axis]
        second_step = J1_POSITIVE[second_axis]
        determinant = (
            first_step[0] * second_step[1]
            - first_step[1] * second_step[0]
        )
        if determinant < 0:
            first_step, second_step = second_step, first_step
        for site in sites:
            coordinates = (
                site,
                reduce_hnf(
                    (site[0] + first_step[0], site[1] + first_step[1]),
                    case.supercell,
                ),
                reduce_hnf(
                    (
                        site[0] + first_step[0] + second_step[0],
                        site[1] + first_step[1] + second_step[1],
                    ),
                    case.supercell,
                ),
                reduce_hnf(
                    (site[0] + second_step[0], site[1] + second_step[1]),
                    case.supercell,
                ),
            )
            cycle = tuple(positions[value] for value in coordinates)
            if len(set(cycle)) != 4:
                raise RuntimeError("a registered rhombus aliases on the torus")
            forward = _cycle_matrix(integer_basis, case.n_sites, cycle)
            backward = _cycle_matrix(
                integer_basis, case.n_sites, tuple(reversed(cycle))
            )
            rings[orientation] = rings[orientation] + forward + backward
            current = current + (-0.5j) * (forward - backward)
            dimers[orientation] = dimers[orientation] + (
                singlet(cycle[2], cycle[3]) @ singlet(cycle[0], cycle[1])
                + singlet(cycle[3], cycle[0]) @ singlet(cycle[1], cycle[2])
            )

    def e2(values: Sequence[sparse.csr_matrix]):
        return tuple(
            sum(
                (
                    float(E2_ORIENTATION_FRAME[row, axis]) * values[axis]
                    for axis in range(3)
                ),
                start=zero.copy(),
            ).tocsr()
            for row in range(2)
        )

    a1 = triangular_a1_resonance_operators(case, integer_basis)
    return LabelledLoopOperators(
        e2_ring=e2(rings),
        e2_dimer=e2(dimers),
        a2_current=current.tocsr(),
        a1_resonance=tuple(a1.values()),
        labels=(
            "E2_rhombus_ring_return",
            "E2_rhombus_dimer_return",
            "A2_directed_ring_return",
        ),
    )


def _labelled_return_columns(
    state: np.ndarray,
    operators: LabelledLoopOperators,
) -> tuple[np.ndarray, ...]:
    vector = np.asarray(state)
    ring = sum(
        (operator @ (operator @ vector) for operator in operators.e2_ring),
        start=np.zeros_like(vector, dtype=np.complex128),
    )
    dimer = sum(
        (operator @ (operator @ vector) for operator in operators.e2_dimer),
        start=np.zeros_like(vector, dtype=np.complex128),
    )
    current = operators.a2_current @ (operators.a2_current @ vector)
    return np.asarray(ring), np.asarray(dimer), np.asarray(current)


def _orthonormal_span(columns: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(tuple(np.asarray(column) for column in columns))
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(1.0e-14, float(singular_values[0]) * 1.0e-12)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return left[:, :rank]


def _ritz_row(label: str, columns: Sequence[np.ndarray], hamiltonian) -> dict[str, object]:
    basis = _orthonormal_span(columns)
    projected = basis.conj().T @ (hamiltonian @ basis)
    projected = 0.5 * (projected + projected.conj().T)
    energies, vectors = np.linalg.eigh(projected)
    state = _normalize(basis @ vectors[:, 0])
    action = hamiltonian @ state
    residual = action - energies[0] * state
    return {
        "label": label,
        "rank": int(basis.shape[1]),
        "energy_total": float(energies[0].real),
        "energy_per_site": float(energies[0].real / 16.0),
        "variance_total": float(np.vdot(residual, residual).real),
    }


def _residual_capture(
    state: np.ndarray,
    directions: Sequence[np.ndarray],
    hamiltonian,
) -> float:
    vector = _normalize(state)
    energy = float(np.vdot(vector, hamiltonian @ vector).real)
    residual = np.asarray(hamiltonian @ vector - energy * vector)
    columns = [
        np.asarray(direction) - vector * np.vdot(vector, direction)
        for direction in directions
    ]
    basis = _orthonormal_span(columns)
    captured = basis @ (basis.conj().T @ residual)
    return float(
        np.vdot(captured, captured).real
        / max(float(np.vdot(residual, residual).real), 1.0e-30)
    )


def _covariance_diagnostics(
    state: np.ndarray,
    operators: LabelledLoopOperators,
    configurations: np.ndarray,
    integer_basis: tuple[int, ...],
) -> dict[str, object]:
    vector = _normalize(state)
    case = SMALL_CASES["square4"]
    basis = tuple(tuple(int(bit) for bit in row) for row in configurations)
    c6_index, mx_index = physical_permutation_index_table(
        basis,
        (
            hnf_site_permutation(case.supercell, operation=C6),
            hnf_site_permutation(case.supercell, operation=MX),
        ),
    )

    def e2_error(values, indices, representation):
        pulled = values[:, indices]
        first = np.einsum("ab,bn->an", representation, pulled)
        second = np.einsum("ab,bn->an", representation.T, pulled)
        return min(
            float(np.linalg.norm(values - first)),
            float(np.linalg.norm(values - second)),
        ) / max(float(np.linalg.norm(values)), 1.0e-30)

    first_ring = np.stack([operator @ vector for operator in operators.e2_ring])
    first_dimer = np.stack([operator @ vector for operator in operators.e2_dimer])
    current = np.asarray(operators.a2_current @ vector)
    returns = np.column_stack(_labelled_return_columns(vector, operators))
    group = triangular_hnf_gamma_a1_group(case.supercell)
    group_indices = physical_permutation_index_table(basis, group)
    a1_error = max(
        float(np.linalg.norm(returns[indices] - returns))
        / max(float(np.linalg.norm(returns)), 1.0e-30)
        for indices in group_indices
    )
    all_vectors = np.column_stack(
        (
            *(first_ring[index] for index in range(2)),
            *(first_dimer[index] for index in range(2)),
            current,
            *(returns[:, index] for index in range(3)),
        )
    )
    # The directed-current image is purely imaginary.  The independent exact
    # helper is real-valued, so certify every nonzero real/imaginary component
    # separately instead of silently discarding the latter.
    real_columns = []
    for index in range(all_vectors.shape[1]):
        for component in (all_vectors[:, index].real, all_vectors[:, index].imag):
            if np.linalg.norm(component) > 1.0e-13:
                real_columns.append(component)
    splus = spin_raising_relative_norm(
        np.column_stack(real_columns), integer_basis, 16
    )
    hermiticity = max(
        float(np.max(np.abs((operator - operator.getH()).data)))
        if (operator - operator.getH()).nnz
        else 0.0
        for operator in (*operators.e2_ring, *operators.e2_dimer, operators.a2_current)
    )
    positivity = {
        label: float(np.vdot(vector, returned).real)
        for label, returned in zip(
            operators.labels,
            _labelled_return_columns(vector, operators),
            strict=True,
        )
    }
    return {
        "e2_ring_covariance_max_relative_error": max(
            e2_error(first_ring, c6_index, _rhombus_pair_representation(C6)),
            e2_error(first_ring, mx_index, _rhombus_pair_representation(MX)),
        ),
        "e2_dimer_covariance_max_relative_error": max(
            e2_error(first_dimer, c6_index, _rhombus_pair_representation(C6)),
            e2_error(first_dimer, mx_index, _rhombus_pair_representation(MX)),
        ),
        "a2_covariance_max_relative_error": max(
            float(np.linalg.norm(current[c6_index] - current))
            / max(float(np.linalg.norm(current)), 1.0e-30),
            float(np.linalg.norm(current[mx_index] + current))
            / max(float(np.linalg.norm(current)), 1.0e-30),
        ),
        "returned_a1_max_relative_error": a1_error,
        "maximum_splus_relative_norm": float(np.max(splus)),
        "maximum_elementary_hermiticity_error": hermiticity,
        "return_expectations": positivity,
        "minimum_return_expectation": min(positivity.values()),
    }


def triangular_symmetry_labelled_loop_screen(
    coefficient_panel: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object]:
    """Screen E2/A2 loop returns on frozen SU2 Operator-ViT states."""

    coefficients = np.asarray(coefficient_panel, dtype=np.float64)
    if coefficients.ndim != 3 or coefficients.shape[1:] != (2, 5):
        raise ValueError("coefficient_panel must have shape (seeds,2,5)")
    data = triangular_pi_flux_operator_components(0.125)
    operators = triangular_labelled_loop_operators()
    hamiltonian = data.hamiltonian
    rows = []
    for seed_index, layer_coefficients in enumerate(coefficients):
        first, second = layer_coefficients
        weights = np.concatenate(
            (np.ones(1), first + second, np.outer(second, first).reshape(-1))
        )
        state = _normalize(data.component_matrix @ weights)
        h_columns = [state]
        for _ in range(8):
            h_columns.append(np.asarray(hamiltonian @ h_columns[-1]))
        labelled = _labelled_return_columns(state, operators)
        a1_directions = tuple(operator @ state for operator in operators.a1_resonance)
        base = h_columns[:2]
        panels = {
            "one_lanczos": base,
            "matched_degree4_krylov": h_columns[:5],
            "existing_A1_loop_bank": (*base, *a1_directions),
            "labelled_E2_A2_loop_bank": (*base, *labelled),
            "combined_A1_and_labelled_loops": (*base, *a1_directions, *labelled),
            "matched_degree7_krylov": h_columns[:8],
        }
        result_rows = {
            label: _ritz_row(label, columns, hamiltonian)
            for label, columns in panels.items()
        }
        rows.append(
            {
                "frozen_seed_index": seed_index,
                "input_energy_total": float(
                    np.vdot(state, hamiltonian @ state).real
                ),
                "ritz": result_rows,
                "labelled_residual_capture_fraction": _residual_capture(
                    state, labelled, hamiltonian
                ),
                "existing_a1_residual_capture_fraction": _residual_capture(
                    state, a1_directions, hamiltonian
                ),
                "certificate": _covariance_diagnostics(
                    state,
                    operators,
                    data.configurations,
                    data.integer_basis,
                ),
            }
        )

    labelled_beats_krylov = [
        row["ritz"]["labelled_E2_A2_loop_bank"]["energy_total"]
        < row["ritz"]["matched_degree4_krylov"]["energy_total"]
        for row in rows
    ]
    labelled_beats_a1 = [
        row["ritz"]["labelled_E2_A2_loop_bank"]["energy_total"]
        < row["ritz"]["existing_A1_loop_bank"]["energy_total"]
        for row in rows
    ]
    combined_beats_krylov = [
        row["ritz"]["combined_A1_and_labelled_loops"]["energy_total"]
        < row["ritz"]["matched_degree7_krylov"]["energy_total"]
        for row in rows
    ]
    exact_certificates = [
        max(
            row["certificate"]["e2_ring_covariance_max_relative_error"],
            row["certificate"]["e2_dimer_covariance_max_relative_error"],
            row["certificate"]["a2_covariance_max_relative_error"],
            row["certificate"]["returned_a1_max_relative_error"],
            row["certificate"]["maximum_splus_relative_norm"],
            row["certificate"]["maximum_elementary_hermiticity_error"],
        )
        < 1.0e-10
        and row["certificate"]["minimum_return_expectation"] > -1.0e-10
        for row in rows
    ]
    gates = {
        "all_exact_symmetry_certificates_pass": all(exact_certificates),
        "labelled_bank_beats_matched_krylov_all_frozen_seeds": all(
            labelled_beats_krylov
        ),
        "labelled_bank_beats_existing_A1_loops_all_frozen_seeds": all(
            labelled_beats_a1
        ),
        "combined_bank_beats_matched_krylov_all_frozen_seeds": all(
            combined_beats_krylov
        ),
        "labelled_residual_capture_exceeds_A1_all_frozen_seeds": all(
            row["labelled_residual_capture_fraction"]
            > row["existing_a1_residual_capture_fraction"]
            for row in rows
        ),
    }
    return {
        "geometry": "triangular_4x4_periodic",
        "j2_over_j1": 0.125,
        "frozen_state_count": len(rows),
        "response_labels": list(operators.labels),
        "rows": rows,
        "paired_labelled_vs_degree4_krylov": labelled_beats_krylov,
        "paired_labelled_vs_existing_A1_loops": labelled_beats_a1,
        "paired_combined_vs_degree7_krylov": combined_beats_krylov,
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "claim_boundary": (
            "This is an exact-support residual screen on three previously frozen "
            "SU2/A1 Operator-ViT states. It tests whether virtual E2/A2 loop "
            "excursions identify missing ground-state directions; it is not a "
            "fresh-seed training audit or a thermodynamic QSL result."
        ),
    }


__all__ = [
    "LabelledLoopOperators",
    "triangular_labelled_loop_operators",
    "triangular_symmetry_labelled_loop_screen",
]
