"""Independent exact-small oracle for the XDiag physical-seed adapter.

This module deliberately contains no XDiag bindings.  It builds amplitudes,
orbit sums, the fixed-``Sz`` Hamiltonian, and polynomial block-Krylov spaces
from repository conventions so an external XDiag result can be checked rather
than reproduced through a shared implementation.

XDiag packs site ``i`` into bit ``i``.  Repository integer basis labels pack
site zero into the most-significant bit.  The site-array amplitudes themselves
use the same site order; only integer serialization is reversed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy import sparse

from su2_nqs.benchmarks.exact_diagonalization import computational_basis
from su2_nqs.benchmarks.projected_singlet_agp import (
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    closed_shell_pairing_from_hopping,
    triangular_pi_zero_flux_hopping,
)
from su2_nqs.lattices.triangular_hnf import (
    hnf_point_group_permutations,
    hnf_sites,
    hnf_translation_permutations,
    triangular_hnf_j1j2_bonds,
    validate_hnf,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan


Supercell = tuple[tuple[int, int], tuple[int, int]]
Configuration = tuple[int, ...]


@dataclass(frozen=True)
class PhysicalSeedCase:
    """One exact-small geometry and its fixed physical seed ordering."""

    geometry_id: str
    supercell: Supercell
    seed_names: tuple[str, ...]
    site_order: tuple[int, ...]

    @property
    def n_sites(self) -> int:
        return self.supercell[0][0] * self.supercell[1][1]


SMALL_CASES: dict[str, PhysicalSeedCase] = {
    "square4": PhysicalSeedCase(
        geometry_id="square4",
        supercell=((4, 0), (0, 4)),
        seed_names=("pi0_DSL_A1", "stripe_tower_A1"),
        site_order=tuple(range(16)),
    ),
    "hnf6x2t4": PhysicalSeedCase(
        geometry_id="hnf6x2t4",
        supercell=((6, 4), (0, 2)),
        seed_names=("120_tower_A1",),
        site_order=tuple(range(12)),
    ),
    "hnf6x2t4_color": PhysicalSeedCase(
        geometry_id="hnf6x2t4_color",
        supercell=((6, 4), (0, 2)),
        seed_names=("120_tower_A1",),
        site_order=tuple(
            sorted(
                range(12),
                key=lambda site: (
                    sum(hnf_sites(((6, 4), (0, 2)))[site]) % 3,
                    site,
                ),
            )
        ),
    ),
}


def configuration_from_repository_integer(state: int, n_sites: int) -> Configuration:
    """Return site-ordered bits from a repository MSB-first integer."""

    if isinstance(state, bool) or not isinstance(state, Integral) or state < 0:
        raise ValueError("state must be a nonnegative integer")
    if not 0 < n_sites <= 64 or state >= 1 << n_sites:
        raise ValueError("state does not fit the requested positive site count")
    return tuple((int(state) >> (n_sites - 1 - site)) & 1 for site in range(n_sites))


def repository_integer_from_configuration(configuration: Sequence[int]) -> int:
    """Serialize site zero as the most-significant bit."""

    result = 0
    for bit in configuration:
        if int(bit) not in (0, 1):
            raise ValueError("configuration must be binary")
        result = (result << 1) | int(bit)
    return result


def xdiag_uint_from_configuration(configuration: Sequence[int]) -> int:
    """Pack site ``i`` into XDiag's internal bit ``i``."""

    result = 0
    for site, bit in enumerate(configuration):
        if int(bit) not in (0, 1):
            raise ValueError("configuration must be binary")
        result |= int(bit) << site
    return result


def configuration_from_xdiag_uint(state: int, n_sites: int) -> Configuration:
    """Convert an XDiag internal unsigned bit pattern to site order."""

    if isinstance(state, bool) or not isinstance(state, Integral) or state < 0:
        raise ValueError("state must be a nonnegative integer")
    if not 0 < n_sites <= 64 or state >= 1 << n_sites:
        raise ValueError("state does not fit the requested positive site count")
    return tuple((int(state) >> site) & 1 for site in range(n_sites))


def _active_transform(configuration: Configuration, image: Sequence[int]) -> Configuration:
    transformed = [0] * len(configuration)
    for old_site, new_site in enumerate(image):
        transformed[int(new_site)] = configuration[old_site]
    return tuple(transformed)


def ordered_gamma_a1_group(case: PhysicalSeedCase) -> tuple[tuple[int, ...], ...]:
    """Return active site maps after the case's deterministic relabeling."""

    old_group = triangular_hnf_gamma_a1_group(case.supercell)
    inverse = {old: new for new, old in enumerate(case.site_order)}
    return tuple(
        tuple(inverse[image[old]] for old in case.site_order) for image in old_group
    )


def generated_hnf_gamma_a1_toml(case: PhysicalSeedCase) -> str:
    """Generate a deterministic XDiag J1-J2/full-D6/A1 input."""

    matrix = validate_hnf(case.supercell)
    n_sites = matrix[0][0] * matrix[1][1]
    translations = hnf_translation_permutations(matrix)
    points = hnf_point_group_permutations(matrix)
    if len(points) != 12:
        raise ValueError("the registered adapter preflight requires full D6")
    group = ordered_gamma_a1_group(case)
    if len(group) != 12 * n_sites:
        raise RuntimeError("translation-by-D6 group has unexpected order")
    j1 = triangular_hnf_j1j2_bonds(matrix, j1=1.0, j2=0.0)
    j2 = triangular_hnf_j1j2_bonds(matrix, j1=0.0, j2=1.0)
    interactions: list[tuple[str, int, int]] = []
    inverse = {old: new for new, old in enumerate(case.site_order)}
    for label, shell in (("J1", j1), ("J2", j2)):
        for first, second, weight in shell:
            multiplicity = int(round(weight))
            if abs(weight - multiplicity) > 1.0e-14 or multiplicity < 1:
                raise ValueError(
                    "the exact-small XDiag input needs positive integer image "
                    "multiplicities"
                )
            interactions.extend(
                (
                    label,
                    min(inverse[int(first)], inverse[int(second)]),
                    max(inverse[int(first)], inverse[int(second)]),
                )
                for _ in range(multiplicity)
            )
    lines = ["# Generated by xdiag_physical_seed_krylov.py", ""]
    lines.append("Interactions = [")
    lines.extend(
        f"  ['{label}', 'SdotS', {first}, {second}],"
        for label, first, second in interactions
    )
    lines.extend(("]", "", "Symmetries = ["))
    lines.extend(
        "  [" + ", ".join(str(site) for site in image) + "]," for image in group
    )
    lines.extend(("]", "", "[Gamma.D6.A1]", "characters = ["))
    lines.extend("  [1.0, 0.0]," for _ in group)
    lines.extend(
        (
            "]",
            "allowed_symmetries = ["
            + ", ".join(str(index) for index in range(len(group)))
            + "]",
            "",
        )
    )
    # Keep these calls live as independent sanity checks on the two factors.
    if len(translations) != n_sites:
        raise RuntimeError("translation count mismatch")
    return "\n".join(lines)


def _configuration_sign(configurations: np.ndarray) -> np.ndarray:
    left_down = np.cumsum(1 - configurations, axis=1) - (1 - configurations)
    inversions = np.sum(configurations * left_down, axis=1)
    pair_count = configurations.shape[1] // 2
    parity = (inversions + pair_count * (pair_count - 1) // 2) % 2
    return np.where(parity == 0, 1.0, -1.0)


def _agp_amplitudes(configurations: np.ndarray, pairing: np.ndarray) -> np.ndarray:
    states = np.asarray(configurations, dtype=np.int8)
    pair_count = states.shape[1] // 2
    output = np.zeros(states.shape[0], dtype=np.float64)
    signs = _configuration_sign(states)
    for index, state in enumerate(states):
        up = np.flatnonzero(state == 1)
        down = np.flatnonzero(state == 0)
        if up.size == pair_count:
            output[index] = signs[index] * np.linalg.det(pairing[np.ix_(up, down)])
    return output


def pi0_dsl_square_components(
    configurations: np.ndarray, linear_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return closed-shell determinant columns and registered A1 coefficients."""

    if linear_size == 4:
        boundaries = (("A", "P"), ("P", "A"), ("A", "A"))
        coefficients = (1.0 / 3.0,) * 3
    elif linear_size == 6:
        boundaries = (("P", "P"), ("A", "P"), ("P", "A"))
        coefficients = (1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0)
    else:
        raise ValueError("pi0 induced orbit is registered only for L=4 or L=6")
    components: list[np.ndarray] = []
    for coefficient, (boundary_x, boundary_y) in zip(
        coefficients, boundaries, strict=True
    ):
        hopping = triangular_pi_zero_flux_hopping(
            linear_size, boundary_x=boundary_x, boundary_y=boundary_y
        )
        packed, gap = closed_shell_pairing_from_hopping(hopping)
        if gap <= 1.0e-10:
            raise RuntimeError("registered pi0 boundary is not closed shell")
        pairing = np.zeros((linear_size**2, linear_size**2), dtype=np.float64)
        upper = np.triu_indices(linear_size**2, 1)
        pairing[upper] = packed
        pairing[(upper[1], upper[0])] = packed
        components.append(_agp_amplitudes(configurations, pairing))
    return np.stack(components, axis=1), np.asarray(coefficients)


def _pi0_dsl_square_amplitudes(configurations: np.ndarray, linear_size: int) -> np.ndarray:
    components, coefficients = pi0_dsl_square_components(
        configurations, linear_size
    )
    return components @ coefficients


def _two_sublattice_singlet(
    configurations: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    states = np.asarray(configurations, dtype=np.int8)
    counts = tuple(int(np.sum(labels == label)) for label in (0, 1))
    if counts[0] != counts[1]:
        raise ValueError("two collective sublattices must have equal size")
    q = counts[0]
    output = np.zeros(states.shape[0], dtype=np.float64)
    for row, state in enumerate(states):
        n_up = [int(np.sum(state[labels == label])) for label in (0, 1)]
        m = [2 * value - q for value in n_up]
        output[row] = clebsch_gordan(q, m[0], q, m[1], 0, 0) / sqrt(
            comb(q, n_up[0]) * comb(q, n_up[1])
        )
    return output


def _stripe_square_amplitudes(configurations: np.ndarray, linear_size: int) -> np.ndarray:
    coordinates = tuple(
        (x, y) for y in range(linear_size) for x in range(linear_size)
    )
    labels = (
        np.asarray([x % 2 for x, _ in coordinates]),
        np.asarray([y % 2 for _, y in coordinates]),
        np.asarray([(x + y) % 2 for x, y in coordinates]),
    )
    return sum(
        (_two_sublattice_singlet(configurations, partition) for partition in labels),
        start=np.zeros(np.asarray(configurations).shape[0]),
    ) / 3.0


def _tower_120_hnf_amplitudes(
    configurations: np.ndarray, case: PhysicalSeedCase
) -> np.ndarray:
    states = np.asarray(configurations, dtype=np.int8)
    old_sites = hnf_sites(case.supercell)
    labels = np.asarray(
        [sum(old_sites[old_site]) % 3 for old_site in case.site_order]
    )
    counts = tuple(int(np.sum(labels == label)) for label in range(3))
    if len(set(counts)) != 1:
        raise ValueError("three collective sublattices must have equal size")
    q = counts[0]
    output = np.zeros(states.shape[0], dtype=np.float64)
    for row, state in enumerate(states):
        n_up = [int(np.sum(state[labels == label])) for label in range(3)]
        m = [2 * value - q for value in n_up]
        if sum(m) != 0:
            continue
        output[row] = (
            clebsch_gordan(q, m[0], q, m[1], q, m[0] + m[1])
            * clebsch_gordan(q, m[0] + m[1], q, m[2], 0, 0)
            / sqrt(comb(q, n_up[0]) * comb(q, n_up[1]) * comb(q, n_up[2]))
        )
    return output


def physical_seed_amplitudes(
    configurations: np.ndarray, case: PhysicalSeedCase
) -> np.ndarray:
    """Return columns in the C++ adapter's registered seed ordering."""

    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != case.n_sites:
        raise ValueError("configurations have the wrong shape")
    if np.any((states != 0) & (states != 1)):
        raise ValueError("configurations must be binary")
    if np.any(np.sum(states, axis=1) != case.n_sites // 2):
        raise ValueError("every configuration must be half filled")
    if case.geometry_id == "square4":
        return np.stack(
            (
                _pi0_dsl_square_amplitudes(states, 4),
                _stripe_square_amplitudes(states, 4),
            ),
            axis=1,
        )
    if case.geometry_id in {"hnf6x2t4", "hnf6x2t4_color"}:
        return _tower_120_hnf_amplitudes(states, case)[:, None]
    raise ValueError("unregistered exact-small geometry")


def fixed_sz_sparse_hamiltonian(
    case: PhysicalSeedCase, coupling: float
) -> tuple[sparse.csr_matrix, tuple[int, ...]]:
    """Build an independent sparse ``J1-J2`` Hamiltonian in repository order."""

    basis = computational_basis(case.n_sites, case.n_sites // 2)
    positions = {state: index for index, state in enumerate(basis)}
    bonds = triangular_hnf_j1j2_bonds(
        case.supercell, j1=1.0, j2=float(coupling)
    )
    inverse = {old: new for new, old in enumerate(case.site_order)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(len(basis), dtype=np.float64)
    for column, state in enumerate(basis):
        for old_first, old_second, weight in bonds:
            first = inverse[old_first]
            second = inverse[old_second]
            first_bit = (state >> (case.n_sites - 1 - first)) & 1
            second_bit = (state >> (case.n_sites - 1 - second)) & 1
            diagonal[column] += weight * (
                0.25 if first_bit == second_bit else -0.25
            )
            if first_bit != second_bit:
                flipped = state ^ (1 << (case.n_sites - 1 - first))
                flipped ^= 1 << (case.n_sites - 1 - second)
                rows.append(positions[flipped])
                columns.append(column)
                values.append(0.5 * weight)
    rows.extend(range(len(basis)))
    columns.extend(range(len(basis)))
    values.extend(float(value) for value in diagonal)
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    return matrix, basis


def spin_raising_relative_norm(
    amplitudes: np.ndarray,
    basis: Sequence[int],
    n_sites: int,
) -> np.ndarray:
    """Return ``||S+ psi||/||psi||`` for fixed-``Sz=0`` amplitude columns."""

    vectors = np.asarray(amplitudes, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    if vectors.shape[0] != len(basis):
        raise ValueError("amplitudes and basis have incompatible sizes")
    target = computational_basis(n_sites, n_sites // 2 + 1)
    positions = {state: index for index, state in enumerate(target)}
    raised = np.zeros((len(target), vectors.shape[1]), dtype=np.float64)
    for column, state in enumerate(basis):
        for site in range(n_sites):
            mask = 1 << (n_sites - 1 - site)
            if not state & mask:
                raised[positions[state | mask]] += vectors[column]
    denominator = np.linalg.norm(vectors, axis=0)
    if np.any(denominator <= 1.0e-300):
        raise ValueError("cannot certify a zero-norm state")
    return np.linalg.norm(raised, axis=0) / denominator


def explicit_a1_reduced_basis(
    case: PhysicalSeedCase,
    representatives_xdiag: Sequence[int],
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Return normalized orbit columns in the emitted XDiag ordering."""

    basis = computational_basis(case.n_sites, case.n_sites // 2)
    positions = {state: index for index, state in enumerate(basis)}
    group = ordered_gamma_a1_group(case)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    orbit_sizes: list[int] = []
    stabilizers: list[int] = []
    observed_orbits: set[frozenset[int]] = set()
    for column, raw_xdiag in enumerate(representatives_xdiag):
        configuration = configuration_from_xdiag_uint(
            int(raw_xdiag), case.n_sites
        )
        orbit_configurations = {
            _active_transform(configuration, image) for image in group
        }
        orbit = frozenset(
            repository_integer_from_configuration(state)
            for state in orbit_configurations
        )
        if orbit in observed_orbits:
            raise ValueError("emitted representatives repeat one orbit")
        observed_orbits.add(orbit)
        packed_orbit = {
            xdiag_uint_from_configuration(state) for state in orbit_configurations
        }
        if int(raw_xdiag) != min(packed_orbit):
            raise ValueError("emitted state is not the minimum XDiag uint in its orbit")
        orbit_size = len(orbit)
        if len(group) % orbit_size:
            raise RuntimeError("orbit size does not divide group order")
        stabilizer = len(group) // orbit_size
        orbit_sizes.append(orbit_size)
        stabilizers.append(stabilizer)
        value = 1.0 / sqrt(orbit_size)
        for state in orbit:
            row_indices.append(positions[state])
            column_indices.append(column)
            values.append(value)
    matrix = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(basis), len(representatives_xdiag)),
    ).tocsc()
    return matrix, np.asarray(orbit_sizes), np.asarray(stabilizers)


def polynomial_block_krylov(
    hamiltonian: np.ndarray,
    normalized_seeds: np.ndarray,
    maximum_depth: int,
    *,
    rank_tolerance: float = 1.0e-11,
) -> list[dict[str, object]]:
    """Independent two-pass-MGS polynomial block-Krylov reference."""

    hamiltonian = np.asarray(hamiltonian, dtype=np.float64)
    candidates = np.asarray(normalized_seeds, dtype=np.float64)
    q = np.zeros((hamiltonian.shape[0], 0), dtype=np.float64)

    def append(raw: np.ndarray) -> list[int]:
        nonlocal q
        added: list[int] = []
        for column in raw.T:
            vector = np.array(column, copy=True)
            unprojected_norm = np.linalg.norm(vector)
            for _ in range(2):
                vector -= q @ (q.T @ vector)
            norm = np.linalg.norm(vector)
            if unprojected_norm > 0.0 and norm > rank_tolerance * unprojected_norm:
                vector /= norm
                added.append(q.shape[1])
                q = np.column_stack((q, vector))
        return added

    frontier = append(candidates)
    if not frontier:
        raise ValueError("seed bank has zero numerical rank")
    rows: list[dict[str, object]] = []
    for depth in range(maximum_depth + 1):
        projected = q.T @ hamiltonian @ q
        energies, vectors = np.linalg.eigh(0.5 * (projected + projected.T))
        nroots = min(3, len(energies))
        ritz = q @ vectors[:, :nroots]
        residuals = np.linalg.norm(
            hamiltonian @ ritz - ritz * energies[None, :nroots], axis=0
        )
        rows.append(
            {
                "depth": depth,
                "basis_dimension": q.shape[1],
                "deflated": not frontier,
                "energies": energies[:nroots],
                "direct_residual_norms": residuals,
            }
        )
        if depth < maximum_depth and frontier:
            frontier = append(hamiltonian @ q[:, frontier])
    return rows
