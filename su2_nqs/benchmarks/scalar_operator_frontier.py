"""Exact-small frontier for SU(2)-scalar operators on a physical carrier.

The purpose of this benchmark is architectural, not to replace large-system
VMC with exact diagonalization.  Starting from the exact-singlet, full-A1
projected pi/0-flux seed on the triangular 4x4 torus, it compares equal-
dimension response spaces:

* ordinary polynomial Lanczos, generated only by the target Hamiltonian; and
* shell-resolved scalar response, generated separately by the J1 and J2 bond
  orbits.

Every generator is a sum of ``S_i.S_j`` scalars over a complete space-group
orbit.  It therefore preserves both total SU(2) and ``Gamma-A1`` exactly.  A
shell-resolved gain over the same-dimensional Krylov space is the minimal
physics gate for replacing the finite operator bank by spatial attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from ..lattices.triangular_e2 import (
    E2_ORIENTATION_FRAME,
    J2_E2_ORIENTATION_FRAME,
)
from ..lattices.triangular_hnf import (
    J1_POSITIVE,
    J2_POSITIVE,
    hnf_sites,
    reduce_hnf,
)
from .xdiag_physical_seed_krylov import (
    SMALL_CASES,
    configuration_from_repository_integer,
    fixed_sz_sparse_hamiltonian,
    physical_seed_amplitudes,
    spin_raising_relative_norm,
)


@dataclass(frozen=True)
class ScalarOperatorRitzRow:
    """One variational subspace result in total and per-site units."""

    label: str
    dimension: int
    energy_total: float
    energy_per_site: float
    direct_residual_norm: float


def _orthonormal_span(columns: list[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(float(singular_values[0]) * 1.0e-12, 1.0e-14)
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 1:
        raise ValueError("operator response space has zero numerical rank")
    return u[:, :rank]


def _ritz(label: str, columns: list[np.ndarray], hamiltonian, n_sites: int):
    basis = _orthonormal_span(columns)
    projected = basis.T @ (hamiltonian @ basis)
    energies, vectors = np.linalg.eigh(0.5 * (projected + projected.T))
    state = basis @ vectors[:, 0]
    residual = np.linalg.norm(hamiltonian @ state - energies[0] * state)
    return ScalarOperatorRitzRow(
        label=label,
        dimension=int(basis.shape[1]),
        energy_total=float(energies[0]),
        energy_per_site=float(energies[0] / n_sites),
        direct_residual_norm=float(residual),
    )


def _bond_sum_matrix(
    integer_basis: tuple[int, ...],
    n_sites: int,
    bonds: tuple[tuple[int, int], ...],
) -> sparse.csr_matrix:
    """Return ``sum_(ij) S_i.S_j`` in the registered MSB-first basis."""

    positions = {state: index for index, state in enumerate(integer_basis)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(len(integer_basis), dtype=np.float64)
    for column, state in enumerate(integer_basis):
        for first, second in bonds:
            first_mask = 1 << (n_sites - 1 - first)
            second_mask = 1 << (n_sites - 1 - second)
            first_bit = bool(state & first_mask)
            second_bit = bool(state & second_mask)
            diagonal[column] += 0.25 if first_bit == second_bit else -0.25
            if first_bit != second_bit:
                rows.append(positions[state ^ first_mask ^ second_mask])
                columns.append(column)
                values.append(0.5)
    rows.extend(range(len(integer_basis)))
    columns.extend(range(len(integer_basis)))
    values.extend(float(value) for value in diagonal)
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(integer_basis), len(integer_basis)),
    ).tocsr()


def _orientation_operators(case, integer_basis, axes):
    sites = hnf_sites(case.supercell)
    positions = {site: index for index, site in enumerate(sites)}
    operators = []
    for displacement in axes:
        bonds = []
        for first, site in enumerate(sites):
            target = reduce_hnf(
                (site[0] + displacement[0], site[1] + displacement[1]),
                case.supercell,
            )
            second = positions[target]
            bonds.append(tuple(sorted((first, second))))
        unique = tuple(sorted(set(bonds)))
        if len(unique) != case.n_sites:
            raise RuntimeError("orientation shell aliases on the registered torus")
        operators.append(
            _bond_sum_matrix(integer_basis, case.n_sites, unique)
        )
    return tuple(operators)


def _e2_components(operators, frame):
    return tuple(
        sum(
            (float(frame[row, axis]) * operators[axis] for axis in range(3)),
            start=sparse.csr_matrix(operators[0].shape, dtype=np.float64),
        )
        for row in range(2)
    )


def _symmetrized_e2_scalar_action(left, right, vector):
    return sum(
        0.5 * (a @ (b @ vector) + b @ (a @ vector))
        for a, b in zip(left, right, strict=True)
    )


def _active_permutation_action(
    vector: np.ndarray,
    integer_basis: tuple[int, ...],
    n_sites: int,
    image: tuple[int, ...],
) -> np.ndarray:
    """Apply one active site permutation to a fixed-Sz state vector."""

    if tuple(sorted(image)) != tuple(range(n_sites)):
        raise ValueError("image must be a site permutation")
    positions = {state: index for index, state in enumerate(integer_basis)}
    result = np.zeros_like(vector, dtype=np.float64)
    for old_index, state in enumerate(integer_basis):
        transformed = 0
        for old_site, new_site in enumerate(image):
            bit = (state >> (n_sites - 1 - old_site)) & 1
            transformed |= bit << (n_sites - 1 - new_site)
        result[positions[transformed]] += vector[old_index]
    return result


def _active_permutation_matrix(
    integer_basis: tuple[int, ...],
    n_sites: int,
    image: tuple[int, ...],
) -> sparse.csr_matrix:
    """Return one active site permutation in the registered fixed-Sz basis."""

    if tuple(sorted(image)) != tuple(range(n_sites)):
        raise ValueError("image must be a site permutation")
    positions = {state: index for index, state in enumerate(integer_basis)}
    rows = np.empty(len(integer_basis), dtype=np.int64)
    for old_index, state in enumerate(integer_basis):
        transformed = 0
        for old_site, new_site in enumerate(image):
            bit = (state >> (n_sites - 1 - old_site)) & 1
            transformed |= bit << (n_sites - 1 - new_site)
        rows[old_index] = positions[transformed]
    columns = np.arange(len(integer_basis), dtype=np.int64)
    return sparse.coo_matrix(
        (np.ones(len(integer_basis), dtype=np.float64), (rows, columns)),
        shape=(len(integer_basis), len(integer_basis)),
    ).tocsr()


def _cycle_action(
    vector: np.ndarray,
    integer_basis: tuple[int, ...],
    n_sites: int,
    cycle: tuple[int, ...],
) -> np.ndarray:
    image = list(range(n_sites))
    for source, target in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        image[source] = target
    return _active_permutation_action(
        vector, integer_basis, n_sites, tuple(image)
    )


def _cycle_matrix(
    integer_basis: tuple[int, ...],
    n_sites: int,
    cycle: tuple[int, ...],
) -> sparse.csr_matrix:
    image = list(range(n_sites))
    for source, target in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        image[source] = target
    return _active_permutation_matrix(
        integer_basis, n_sites, tuple(image)
    )


def _bond_action(
    vector: np.ndarray,
    integer_basis: tuple[int, ...],
    n_sites: int,
    first: int,
    second: int,
) -> np.ndarray:
    """Apply one ``S_first.S_second`` without constructing a sparse matrix."""

    positions = {state: index for index, state in enumerate(integer_basis)}
    first_mask = 1 << (n_sites - 1 - first)
    second_mask = 1 << (n_sites - 1 - second)
    result = np.zeros_like(vector, dtype=np.float64)
    for index, state in enumerate(integer_basis):
        first_bit = bool(state & first_mask)
        second_bit = bool(state & second_mask)
        result[index] += (0.25 if first_bit == second_bit else -0.25) * vector[index]
        if first_bit != second_bit:
            result[index] += 0.5 * vector[
                positions[state ^ first_mask ^ second_mask]
            ]
    return result


def _singlet_projector_action(
    vector: np.ndarray,
    integer_basis: tuple[int, ...],
    n_sites: int,
    first: int,
    second: int,
) -> np.ndarray:
    return 0.25 * vector - _bond_action(
        vector, integer_basis, n_sites, first, second
    )


def triangular_a1_resonance_operators(
    case,
    integer_basis: tuple[int, ...],
) -> dict[str, sparse.csr_matrix]:
    """Build full-``A1`` SU(2)-scalar resonance operators.

    Each matrix is a uniform sum over the complete translation and orientation
    orbit on the registered triangular torus.  The ring and loop generators
    are permutation scalars.  The dimer generator is a sum of products of
    pair-singlet projectors on opposite rhombus edges.
    """

    sites = hnf_sites(case.supercell)
    positions = {site: index for index, site in enumerate(sites)}
    dimension = len(integer_basis)
    zero = sparse.csr_matrix((dimension, dimension), dtype=np.float64)
    identity = sparse.identity(dimension, dtype=np.float64, format="csr")
    ring = zero.copy()
    dimer = zero.copy()
    bond_cache: dict[tuple[int, int], sparse.csr_matrix] = {}

    def singlet(first: int, second: int) -> sparse.csr_matrix:
        bond = tuple(sorted((first, second)))
        if bond not in bond_cache:
            bond_cache[bond] = _bond_sum_matrix(
                integer_basis, case.n_sites, (bond,)
            )
        return 0.25 * identity - bond_cache[bond]

    for site in sites:
        for first_axis in range(3):
            for second_axis in range(first_axis + 1, 3):
                first_step = J1_POSITIVE[first_axis]
                second_step = J1_POSITIVE[second_axis]
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
                    raise RuntimeError("registered rhombus aliases on the torus")
                ring = ring + _cycle_matrix(
                    integer_basis, case.n_sites, cycle
                )
                ring = ring + _cycle_matrix(
                    integer_basis, case.n_sites, tuple(reversed(cycle))
                )
                first_dimer = singlet(cycle[2], cycle[3]) @ singlet(
                    cycle[0], cycle[1]
                )
                second_dimer = singlet(cycle[3], cycle[0]) @ singlet(
                    cycle[1], cycle[2]
                )
                dimer += first_dimer + second_dimer

    # Neighbours are ordered around a site in the repository's 120-degree
    # primitive coordinates.  Cycling the six neighbours is a nonlocal
    # singlet rearrangement while leaving the central spin untouched.
    neighbour_steps = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))
    hexagon = zero.copy()
    for site in sites:
        cycle = tuple(
            positions[
                reduce_hnf(
                    (site[0] + step[0], site[1] + step[1]),
                    case.supercell,
                )
            ]
            for step in neighbour_steps
        )
        if len(set(cycle)) != 6:
            raise RuntimeError("registered six-site loop aliases on the torus")
        hexagon = hexagon + _cycle_matrix(
            integer_basis, case.n_sites, cycle
        )
        hexagon = hexagon + _cycle_matrix(
            integer_basis, case.n_sites, tuple(reversed(cycle))
        )
    return {
        "rhombus_ring_exchange": ring.tocsr(),
        "rhombus_dimer_resonance": dimer.tocsr(),
        "six_site_loop_rearrangement": hexagon.tocsr(),
    }


def _rhombus_and_hexagon_responses(case, integer_basis, carrier):
    """Return full-A1 ring, dimer-resonance, and six-site loop actions."""

    operators = triangular_a1_resonance_operators(case, integer_basis)
    return tuple(operator @ carrier for operator in operators.values())


def triangular_pi_flux_scalar_operator_frontier(
    coupling: float = 0.125,
) -> dict[str, object]:
    """Return the triangular-4x4 scalar-response variational hierarchy."""

    if not np.isfinite(coupling) or coupling < 0.0:
        raise ValueError("coupling must be nonnegative and finite")
    case = SMALL_CASES["square4"]
    hamiltonian, integer_basis = fixed_sz_sparse_hamiltonian(case, coupling)
    h1, h1_basis = fixed_sz_sparse_hamiltonian(case, 0.0)
    h_at_one, h_at_one_basis = fixed_sz_sparse_hamiltonian(case, 1.0)
    if integer_basis != h1_basis or integer_basis != h_at_one_basis:
        raise RuntimeError("shell Hamiltonians use inconsistent basis orderings")
    h2 = h_at_one - h1
    configurations = np.asarray(
        [
            configuration_from_repository_integer(value, case.n_sites)
            for value in integer_basis
        ],
        dtype=np.int8,
    )
    carrier = np.asarray(
        physical_seed_amplitudes(configurations, case)[:, 0], dtype=np.float64
    )
    carrier /= np.linalg.norm(carrier)
    h_carrier = hamiltonian @ carrier
    h1_carrier = h1 @ carrier
    h2_carrier = h2 @ carrier
    h2_lanczos = hamiltonian @ h_carrier
    quadratic_shells = (
        h1 @ h1_carrier,
        0.5 * (h1 @ h2_carrier + h2 @ h1_carrier),
        h2 @ h2_carrier,
    )
    j1_orientations = _orientation_operators(case, integer_basis, J1_POSITIVE)
    j2_orientations = _orientation_operators(case, integer_basis, J2_POSITIVE)
    j1_e2 = _e2_components(j1_orientations, E2_ORIENTATION_FRAME)
    j2_e2 = _e2_components(j2_orientations, J2_E2_ORIENTATION_FRAME)
    e2_scalar_responses = (
        _symmetrized_e2_scalar_action(j1_e2, j1_e2, carrier),
        _symmetrized_e2_scalar_action(j1_e2, j2_e2, carrier),
        _symmetrized_e2_scalar_action(j2_e2, j2_e2, carrier),
    )
    ring_response, dimer_response, hexagon_response = (
        _rhombus_and_hexagon_responses(case, integer_basis, carrier)
    )
    resonance_responses = (
        ring_response,
        dimer_response,
        hexagon_response,
    )
    krylov = [carrier]
    for _ in range(8):
        krylov.append(hamiltonian @ krylov[-1])
    spaces = (
        ("bare_pi0", [carrier]),
        ("one_lanczos_H", [carrier, h_carrier]),
        ("shell_resolved_J1_J2", [carrier, h1_carrier, h2_carrier]),
        ("two_lanczos_H", [carrier, h_carrier, h2_lanczos]),
        (
            "shell_resolved_quadratic",
            [carrier, h1_carrier, h2_carrier, *quadratic_shells],
        ),
        (
            "e2_scalar_bank",
            [carrier, *e2_scalar_responses],
        ),
        (
            "krylov_degree_3",
            krylov[:4],
        ),
        (
            "krylov_degree_5",
            krylov[:6],
        ),
        (
            "krylov2_plus_e2_scalar_bank",
            [*krylov[:3], *e2_scalar_responses],
        ),
        (
            "resonance_operator_bank",
            [carrier, *resonance_responses],
        ),
        (
            "krylov2_plus_resonance_bank",
            [*krylov[:3], *resonance_responses],
        ),
        (
            "krylov_degree_8",
            krylov,
        ),
        (
            "krylov5_plus_resonance_bank",
            [*krylov[:6], *resonance_responses],
        ),
    )
    rows = tuple(
        _ritz(label, columns, hamiltonian, case.n_sites)
        for label, columns in spaces
    )
    exact_total = float(
        eigsh(
            hamiltonian,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1.0e-12,
        )[0]
    )
    scalar_columns = np.column_stack(
        (
            carrier,
            h_carrier,
            h1_carrier,
            h2_carrier,
            *quadratic_shells,
            *e2_scalar_responses,
            *resonance_responses,
        )
    )
    singlet_residuals = spin_raising_relative_norm(
        scalar_columns, integer_basis, case.n_sites
    )
    carrier_energy = float(np.vdot(carrier, h_carrier).real)
    carrier_variance = float(
        np.vdot(h_carrier, h_carrier).real - carrier_energy**2
    )
    by_label = {row.label: row for row in rows}
    shell = by_label["shell_resolved_J1_J2"]
    one_lanczos = by_label["one_lanczos_H"]
    two_lanczos = by_label["two_lanczos_H"]
    e2_bank = by_label["e2_scalar_bank"]
    krylov3 = by_label["krylov_degree_3"]
    krylov5 = by_label["krylov_degree_5"]
    joint_e2 = by_label["krylov2_plus_e2_scalar_bank"]
    resonance = by_label["resonance_operator_bank"]
    krylov3 = by_label["krylov_degree_3"]
    joint_resonance = by_label["krylov2_plus_resonance_bank"]
    krylov5 = by_label["krylov_degree_5"]
    deep_joint_resonance = by_label["krylov5_plus_resonance_bank"]
    krylov8 = by_label["krylov_degree_8"]
    return {
        "geometry": "triangular_4x4_periodic",
        "n_sites": case.n_sites,
        "fixed_sz_basis_dimension": len(integer_basis),
        "j2_over_j1": float(coupling),
        "carrier": "projected pi/0-flux S=0 Gamma-A1",
        "rows": tuple(row.__dict__ for row in rows),
        "exact_energy_total": exact_total,
        "exact_energy_per_site": exact_total / case.n_sites,
        "maximum_scalar_carrier_splus_relative_norm": float(
            np.max(singlet_residuals)
        ),
        "carrier_variance_total": carrier_variance,
        "carrier_variance_per_site": carrier_variance / case.n_sites,
        "one_lanczos_linear_descent_slope_per_site": (
            2.0 * carrier_variance / case.n_sites
        ),
        "same_dimension_gate": {
            "shell_dimension": shell.dimension,
            "two_lanczos_dimension": two_lanczos.dimension,
            "shell_minus_two_lanczos_energy_per_site": (
                shell.energy_per_site - two_lanczos.energy_per_site
            ),
            "shell_gain_beyond_one_lanczos_per_site": (
                shell.energy_per_site - one_lanczos.energy_per_site
            ),
            "shell_beats_same_dimension_two_lanczos": (
                shell.energy_per_site < two_lanczos.energy_per_site - 1.0e-12
            ),
        },
        "e2_scalar_gate": {
            "e2_bank_dimension": e2_bank.dimension,
            "degree3_krylov_dimension": krylov3.dimension,
            "e2_bank_minus_degree3_krylov_energy_per_site": (
                e2_bank.energy_per_site - krylov3.energy_per_site
            ),
            "joint_dimension": joint_e2.dimension,
            "degree5_krylov_dimension": krylov5.dimension,
            "joint_minus_degree5_krylov_energy_per_site": (
                joint_e2.energy_per_site - krylov5.energy_per_site
            ),
            "e2_bank_beats_matched_krylov": (
                e2_bank.dimension == krylov3.dimension
                and e2_bank.energy_per_site < krylov3.energy_per_site - 1.0e-12
            ),
            "e2_adds_after_matched_krylov": (
                joint_e2.dimension == krylov5.dimension
                and joint_e2.energy_per_site < krylov5.energy_per_site - 1.0e-12
            ),
        },
        "resonance_operator_gate": {
            "bank_dimension": resonance.dimension,
            "degree3_krylov_dimension": krylov3.dimension,
            "bank_minus_degree3_krylov_energy_per_site": (
                resonance.energy_per_site - krylov3.energy_per_site
            ),
            "joint_dimension": joint_resonance.dimension,
            "degree5_krylov_dimension": krylov5.dimension,
            "joint_minus_degree5_krylov_energy_per_site": (
                joint_resonance.energy_per_site - krylov5.energy_per_site
            ),
            "deep_joint_dimension": deep_joint_resonance.dimension,
            "degree8_krylov_dimension": krylov8.dimension,
            "deep_joint_minus_degree8_krylov_energy_per_site": (
                deep_joint_resonance.energy_per_site - krylov8.energy_per_site
            ),
            "bank_beats_matched_krylov": (
                resonance.dimension == krylov3.dimension
                and resonance.energy_per_site < krylov3.energy_per_site - 1.0e-12
            ),
            "adds_after_krylov2": (
                joint_resonance.dimension == krylov5.dimension
                and joint_resonance.energy_per_site
                < krylov5.energy_per_site - 1.0e-12
            ),
            "adds_after_krylov5": (
                deep_joint_resonance.dimension == krylov8.dimension
                and deep_joint_resonance.energy_per_site
                < krylov8.energy_per_site - 1.0e-12
            ),
            "carrier_labels": (
                "rhombus_ring_exchange",
                "rhombus_dimer_resonance",
                "six_site_loop_rearrangement",
            ),
        },
        "interpretation": (
            "A gain of shell-resolved J1/J2 response over the equal-dimensional "
            "two-Lanczos space licenses a spatially resolved SU2-scalar operator "
            "head; otherwise use ordinary Krylov as the stronger baseline."
        ),
    }


def triangular_scalar_operator_scan(
    couplings: Sequence[float],
) -> tuple[dict[str, object], ...]:
    """Run the exact-small carrier gate along a declared J2/J1 scan."""

    checked = tuple(float(value) for value in couplings)
    if not checked or any(not np.isfinite(value) or value < 0.0 for value in checked):
        raise ValueError("couplings must be a nonempty sequence of finite nonnegative values")
    if len(set(checked)) != len(checked):
        raise ValueError("couplings must be distinct")
    return tuple(triangular_pi_flux_scalar_operator_frontier(value) for value in checked)


__all__ = [
    "ScalarOperatorRitzRow",
    "triangular_a1_resonance_operators",
    "triangular_pi_flux_scalar_operator_frontier",
    "triangular_scalar_operator_scan",
]
