"""Physics-facing exact-small diagnostics for triangular singlet states.

This module deliberately contains no model or optimizer logic.  It acts on a
normalized fixed-``M=0`` wavefunction and asks which spatial correlations are
present, on what support they become accurate, and which physical deformation
points toward the exact ground state.  Periodic triangular clusters use the
same 120-degree HNF convention as :mod:`su2_nqs.lattices.triangular_hnf`.

The error-channel routines use the exact-state correction
``|g>-|psi><psi|g>`` rather than ``(H-E)|psi>``.  The latter is identically a
linear combination of the Hamiltonian bond terms and is therefore a circular
diagnostic of the missing physics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from ..lattices.triangular_hnf import (
    J1_POSITIVE,
    J2_POSITIVE,
    IntMatrix,
    IntVector,
    hnf_point_group_permutations,
    hnf_sites,
    hnf_translation_permutations,
    reduce_hnf,
    validate_hnf,
)
from .local_indistinguishability import pure_state_partial_trace, trace_distance


def normalized_state(state: np.ndarray) -> np.ndarray:
    """Return a finite normalized complex vector."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError("state has zero numerical norm")
    return vector / norm


def fixed_sector_to_full_state(
    state: np.ndarray, basis: Sequence[Sequence[int]]
) -> np.ndarray:
    """Embed a fixed-sector state in the full computational bit basis."""

    vector = normalized_state(state)
    rows = np.asarray(tuple(tuple(int(bit) for bit in row) for row in basis), dtype=np.int8)
    if rows.ndim != 2 or rows.shape[0] != len(vector) or rows.shape[1] < 1:
        raise ValueError("basis must be a nonempty rectangular table matching state")
    if np.any((rows != 0) & (rows != 1)):
        raise ValueError("basis entries must be bits")
    powers = 2 ** np.arange(rows.shape[1] - 1, -1, -1, dtype=np.int64)
    indices = rows.astype(np.int64) @ powers
    if len(np.unique(indices)) != len(indices):
        raise ValueError("basis configurations must be distinct")
    result = np.zeros(2 ** rows.shape[1], dtype=np.complex128)
    result[indices] = vector
    return result


def pair_permutation_indices(
    basis: Sequence[Sequence[int]], first: int, second: int
) -> np.ndarray:
    """Return the fixed-sector lookup implementing the physical swap ``P_ij``."""

    configurations = tuple(tuple(int(bit) for bit in row) for row in basis)
    if not configurations or any(len(row) != len(configurations[0]) for row in configurations):
        raise ValueError("basis must be a nonempty rectangular table")
    n_sites = len(configurations[0])
    first, second = int(first), int(second)
    if first == second or not 0 <= first < n_sites or not 0 <= second < n_sites:
        raise ValueError("pair endpoints must be distinct valid sites")
    positions = {configuration: index for index, configuration in enumerate(configurations)}
    answer = np.empty(len(configurations), dtype=np.int32)
    for index, configuration in enumerate(configurations):
        swapped = list(configuration)
        swapped[first], swapped[second] = swapped[second], swapped[first]
        try:
            answer[index] = positions[tuple(swapped)]
        except KeyError as error:
            raise ValueError("basis is not closed under pair exchange") from error
    return answer


def exchange_action(state: np.ndarray, permutation_indices: np.ndarray) -> np.ndarray:
    """Apply ``S_i.S_j = P_ij/2-I/4`` from a precomputed swap lookup."""

    vector = np.asarray(state, dtype=np.complex128)
    indices = np.asarray(permutation_indices, dtype=np.int64)
    if vector.ndim != 1 or indices.shape != vector.shape:
        raise ValueError("state and permutation lookup must have the same vector shape")
    if np.any(indices < 0) or np.any(indices >= len(vector)):
        raise ValueError("permutation lookup lies outside the state")
    return 0.5 * vector[indices] - 0.25 * vector


def hnf_oriented_bond_families(
    supercell: IntMatrix,
    displacements: Sequence[IntVector] = J1_POSITIVE,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return one directed periodic bond family for every displacement.

    Periodic images remain distinct, as required for bond-centred Fourier
    sources on small tori.  For the registered N=12 and N=16 cells every
    family has exactly one bond per site and no self aliases.
    """

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    families = []
    for raw in displacements:
        displacement = (int(raw[0]), int(raw[1]))
        if displacement == (0, 0):
            raise ValueError("bond displacement must be nonzero")
        bonds = []
        for first, site in enumerate(sites):
            target = reduce_hnf(
                (site[0] + displacement[0], site[1] + displacement[1]), matrix
            )
            second = positions[target]
            if first == second:
                raise ValueError("bond displacement aliases a site to itself")
            bonds.append((first, second))
        families.append(tuple(bonds))
    if not families:
        raise ValueError("at least one displacement is required")
    return tuple(families)


def hnf_oriented_triangles(supercell: IntMatrix) -> tuple[tuple[int, int, int], ...]:
    """Return the two consistently oriented elementary triangles per site."""

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}

    def site_index(origin: IntVector, displacement: IntVector) -> int:
        return positions[
            reduce_hnf(
                (origin[0] + displacement[0], origin[1] + displacement[1]),
                matrix,
            )
        ]

    triangles = []
    for origin_index, origin in enumerate(sites):
        first = site_index(origin, (1, 0))
        second = site_index(origin, (0, 1))
        diagonal = site_index(origin, (1, 1))
        for triangle in ((origin_index, first, diagonal), (origin_index, diagonal, second)):
            if len(set(triangle)) != 3:
                raise ValueError("elementary triangle aliases on this torus")
            triangles.append(triangle)
    return tuple(triangles)


class FixedSzPairActions:
    """Cached pair-exchange actions in one deterministic fixed-Sz basis."""

    def __init__(self, basis: Sequence[Sequence[int]]):
        self.basis = tuple(tuple(int(bit) for bit in row) for row in basis)
        if not self.basis:
            raise ValueError("basis must be nonempty")
        self.n_sites = len(self.basis[0])
        self._indices: dict[tuple[int, int], np.ndarray] = {}

    def indices(self, first: int, second: int) -> np.ndarray:
        edge = tuple(sorted((int(first), int(second))))
        if edge[0] == edge[1]:
            raise ValueError("pair endpoints must be distinct")
        if edge not in self._indices:
            self._indices[edge] = pair_permutation_indices(
                self.basis, edge[0], edge[1]
            )
        return self._indices[edge]

    def pair(self, state: np.ndarray, first: int, second: int) -> np.ndarray:
        return exchange_action(state, self.indices(first, second))

    def weighted_bonds(
        self,
        state: np.ndarray,
        bonds: Sequence[tuple[int, int, complex]],
    ) -> np.ndarray:
        vector = np.asarray(state, dtype=np.complex128)
        answer = np.zeros_like(vector)
        for first, second, weight in bonds:
            value = complex(weight)
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError("bond weights must be finite")
            answer += value * self.pair(vector, int(first), int(second))
        return answer

    def chirality(
        self, state: np.ndarray, triangle: tuple[int, int, int]
    ) -> np.ndarray:
        """Apply ``i[B_ij,B_jk]`` for one oriented triangle."""

        first, middle, last = (int(site) for site in triangle)
        vector = np.asarray(state, dtype=np.complex128)
        forward = self.pair(self.pair(vector, middle, last), first, middle)
        backward = self.pair(self.pair(vector, first, middle), middle, last)
        return 1.0j * (forward - backward)


def bond_center_phase_actions(
    state: np.ndarray,
    actions: FixedSzPairActions,
    supercell: IntMatrix,
    momentum: tuple[float, float],
    displacements: Sequence[IntVector] = J1_POSITIVE,
) -> np.ndarray:
    """Return columns ``D_delta(k)|psi>`` for periodic bond-centred sources."""

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    families = hnf_oriented_bond_families(matrix, displacements)
    kx, ky = float(momentum[0]), float(momentum[1])
    if not all(math.isfinite(value) for value in (kx, ky)):
        raise ValueError("momentum must be finite")
    columns = []
    for displacement, bonds in zip(displacements, families, strict=True):
        weighted = []
        dx, dy = int(displacement[0]), int(displacement[1])
        for site, (first, second) in zip(sites, bonds, strict=True):
            phase = np.exp(
                2.0j
                * np.pi
                * (kx * (site[0] + 0.5 * dx) + ky * (site[1] + 0.5 * dy))
            )
            weighted.append((first, second, phase))
        columns.append(actions.weighted_bonds(state, weighted))
    return np.column_stack(columns)


def uniform_chirality_action(
    state: np.ndarray,
    actions: FixedSzPairActions,
    supercell: IntMatrix,
) -> np.ndarray:
    """Apply the sum over all consistently oriented elementary chiralities."""

    answer = np.zeros_like(np.asarray(state, dtype=np.complex128))
    for triangle in hnf_oriented_triangles(supercell):
        answer += actions.chirality(state, triangle)
    return answer


def _center_action(state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, complex]:
    vector = normalized_state(state)
    image = np.asarray(action, dtype=np.complex128)
    if image.shape != vector.shape:
        raise ValueError("operator action must match state")
    mean = complex(np.vdot(vector, image))
    return image - mean * vector, mean


def triangular_periodic_observable_panel(
    state: np.ndarray,
    basis: Sequence[Sequence[int]],
    supercell: IntMatrix,
    *,
    pair_actions: FixedSzPairActions | None = None,
) -> dict[str, object]:
    """Evaluate direction, dimer, nematic, and chirality diagnostics."""

    vector = normalized_state(state)
    matrix = validate_hnf(supercell)
    n_sites = len(hnf_sites(matrix))
    if len(basis) != len(vector) or len(basis[0]) != n_sites:
        raise ValueError("basis/state do not match the supercell")
    actions = FixedSzPairActions(basis) if pair_actions is None else pair_actions
    if actions.n_sites != n_sites:
        raise ValueError("cached pair actions do not match the supercell")
    families = hnf_oriented_bond_families(matrix)
    orientation_actions = np.column_stack(
        [
            actions.weighted_bonds(
                vector, tuple((first, second, 1.0) for first, second in family)
            )
            for family in families
        ]
    )
    orientation_means = np.asarray(
        [np.vdot(vector, orientation_actions[:, index]) for index in range(3)]
    )
    second_families = hnf_oriented_bond_families(matrix, J2_POSITIVE)
    second_actions = np.column_stack(
        [
            actions.weighted_bonds(
                vector, tuple((first, second, 1.0) for first, second in family)
            )
            for family in second_families
        ]
    )
    second_means = np.asarray(
        [np.vdot(vector, second_actions[:, index]) for index in range(3)]
    )
    nematic = (
        (2.0 * orientation_actions[:, 0] - orientation_actions[:, 1] - orientation_actions[:, 2])
        / math.sqrt(6.0),
        (orientation_actions[:, 1] - orientation_actions[:, 2]) / math.sqrt(2.0),
    )
    nematic_rows = []
    for image in nematic:
        centered, mean = _center_action(vector, image)
        nematic_rows.append(
            {
                "mean_per_site": float(mean.real / n_sites),
                "mean_imaginary_per_site": float(mean.imag / n_sites),
                "connected_second_per_site_squared": float(
                    np.vdot(centered, centered).real / n_sites**2
                ),
            }
        )

    dimer: dict[str, object] = {}
    for name, momentum in (
        ("M1", (0.5, 0.0)),
        ("M2", (0.0, 0.5)),
        ("M3", (0.5, 0.5)),
    ):
        sources = bond_center_phase_actions(vector, actions, matrix, momentum)
        source_means = vector.conj() @ sources
        centered_sources = sources - vector[:, None] * source_means[None, :]
        covariance = centered_sources.conj().T @ centered_sources / n_sites**2
        covariance = 0.5 * (covariance + covariance.conj().T)
        values, vectors = np.linalg.eigh(covariance)
        leading = vectors[:, -1]
        pivot = int(np.argmax(np.abs(leading)))
        leading *= np.exp(-1.0j * np.angle(leading[pivot]))
        dimer[name] = {
            "momentum_fraction": list(momentum),
            "orientation_mean_real_per_site": (source_means.real / n_sites).tolist(),
            "orientation_mean_imaginary_per_site": (source_means.imag / n_sites).tolist(),
            "orientation_covariance_real": covariance.real.tolist(),
            "orientation_covariance_imaginary": covariance.imag.tolist(),
            "trace_strength": float(np.trace(covariance).real),
            "leading_strength": float(values[-1].real),
            "leading_orientation_real": leading.real.tolist(),
            "leading_orientation_imaginary": leading.imag.tolist(),
        }

    chirality_action = uniform_chirality_action(vector, actions, matrix)
    chirality_centered, chirality_mean = _center_action(vector, chirality_action)
    triangle_count = len(hnf_oriented_triangles(matrix))
    return {
        "orientation_bond_energy_per_site": [
            float(value.real / n_sites) for value in orientation_means
        ],
        "orientation_bond_energy_imaginary_per_site": [
            float(value.imag / n_sites) for value in orientation_means
        ],
        "orientation_spread_per_site": float(
            np.ptp(orientation_means.real) / n_sites
        ),
        "second_neighbor_orientation_bond_energy_per_site": [
            float(value.real / n_sites) for value in second_means
        ],
        "second_neighbor_orientation_bond_energy_imaginary_per_site": [
            float(value.imag / n_sites) for value in second_means
        ],
        "second_neighbor_orientation_spread_per_site": float(
            np.ptp(second_means.real) / n_sites
        ),
        "bond_nematic_components": nematic_rows,
        "bond_nematic_connected_total": float(
            sum(row["connected_second_per_site_squared"] for row in nematic_rows)
        ),
        "dimer_M": dimer,
        "chirality_mean_per_triangle": float(chirality_mean.real / triangle_count),
        "chirality_mean_imaginary_per_triangle": float(
            chirality_mean.imag / triangle_count
        ),
        "chirality_connected_per_triangle_squared": float(
            np.vdot(chirality_centered, chirality_centered).real / triangle_count**2
        ),
        "chirality_connected_per_triangle": float(
            np.vdot(chirality_centered, chirality_centered).real / triangle_count
        ),
    }


def _apply_permutation_to_support(
    support: Sequence[int], permutation: Sequence[int]
) -> tuple[int, ...]:
    return tuple(sorted(int(permutation[int(site)]) for site in support))


def triangular_cluster_supports(
    supercell: IntMatrix,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Return translation/point-group complete local support families.

    The registered families have sizes 2, 3, 4, 6, and ``N/2``.  The six-site
    and half-torus shapes are generated from one seed by every retained point
    operation and translation, so no lattice orientation is privileged.
    """

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    n_sites = len(sites)
    if n_sites % 2:
        raise ValueError("half-torus supports require even volume")
    families = hnf_oriented_bond_families(matrix)
    bonds = {tuple(sorted(edge)) for family in families for edge in family}
    triangles = {tuple(sorted(value)) for value in hnf_oriented_triangles(matrix)}

    rhombi: set[tuple[int, ...]] = set()
    for origin in sites:
        for first_index in range(len(J1_POSITIVE)):
            for second_index in range(first_index + 1, len(J1_POSITIVE)):
                first = J1_POSITIVE[first_index]
                second = J1_POSITIVE[second_index]
                coordinates = (
                    origin,
                    reduce_hnf((origin[0] + first[0], origin[1] + first[1]), matrix),
                    reduce_hnf((origin[0] + second[0], origin[1] + second[1]), matrix),
                    reduce_hnf(
                        (
                            origin[0] + first[0] + second[0],
                            origin[1] + first[1] + second[1],
                        ),
                        matrix,
                    ),
                )
                support = tuple(sorted(positions[value] for value in coordinates))
                if len(set(support)) == 4:
                    rhombi.add(support)

    translations = tuple(image for _, image in hnf_translation_permutations(matrix))
    points = tuple(image for _, _, image in hnf_point_group_permutations(matrix))

    def orbit(seed: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        result = set()
        for point in points:
            rotated = _apply_permutation_to_support(seed, point)
            for translation in translations:
                result.add(_apply_permutation_to_support(rotated, translation))
        return tuple(sorted(result))

    # These seeds are used only on the registered 4x4 physical-state audit.
    horizontal = int(matrix[0][0])
    vertical = int(matrix[1][1])
    if (horizontal, matrix[0][1], vertical) != (4, 0, 4):
        raise ValueError("six-site and half-torus supports are registered on 4x4 HNF")
    six_seed = tuple(
        positions[(x, y)] for y in range(2) for x in range(3)
    )
    half_seed = tuple(
        positions[(x, y)] for y in range(2) for x in range(4)
    )
    return {
        "bond_2": tuple(sorted(bonds)),
        "triangle_3": tuple(sorted(triangles)),
        "rhombus_4": tuple(sorted(rhombi)),
        "block_2x3_6": orbit(six_seed),
        "half_torus_8": orbit(half_seed),
    }


def _von_neumann_entropy(rdm: np.ndarray) -> float:
    values = np.linalg.eigvalsh(0.5 * (rdm + rdm.conj().T)).real
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("RDM has zero trace")
    values /= total
    positive = values[values > 1.0e-15]
    return float(-np.dot(positive, np.log(positive)))


def local_rdm_error_profile(
    state: np.ndarray,
    exact_state: np.ndarray,
    n_sites: int,
    support_families: Mapping[str, Sequence[Sequence[int]]],
    *,
    reference_rdms: Mapping[str, Sequence[np.ndarray]] | None = None,
) -> dict[str, object]:
    """Compare local RDMs to an exact state from two sites to half a torus."""

    vector = normalized_state(state)
    exact = normalized_state(exact_state)
    if vector.shape != (2**int(n_sites),) or exact.shape != vector.shape:
        raise ValueError("states must be full computational vectors")
    result: dict[str, object] = {}
    for name, raw_supports in support_families.items():
        supports = tuple(tuple(sorted(int(site) for site in support)) for support in raw_supports)
        if not supports:
            raise ValueError("support families must be nonempty")
        distances = []
        entropy_errors = []
        cached = None if reference_rdms is None else reference_rdms.get(str(name))
        if cached is not None and len(cached) != len(supports):
            raise ValueError("cached reference RDM count does not match supports")
        for support_index, support in enumerate(supports):
            reference = (
                pure_state_partial_trace(exact, n_sites, support)
                if cached is None
                else np.asarray(cached[support_index], dtype=np.complex128)
            )
            candidate = pure_state_partial_trace(vector, n_sites, support)
            distances.append(trace_distance(candidate, reference))
            entropy_errors.append(_von_neumann_entropy(candidate) - _von_neumann_entropy(reference))
        values = np.asarray(distances, dtype=np.float64)
        entropy = np.asarray(entropy_errors, dtype=np.float64)
        result[str(name)] = {
            "support_size": len(supports[0]),
            "orbit_size": len(supports),
            "trace_distance_mean": float(np.mean(values)),
            "trace_distance_rms": float(np.sqrt(np.mean(values**2))),
            "trace_distance_max": float(np.max(values)),
            "entropy_error_mean": float(np.mean(entropy)),
            "entropy_error_rms": float(np.sqrt(np.mean(entropy**2))),
        }
    return result


def local_rdm_reference_cache(
    exact_state: np.ndarray,
    n_sites: int,
    support_families: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, tuple[np.ndarray, ...]]:
    """Precompute exact local RDMs for repeated same-support comparisons."""

    exact = normalized_state(exact_state)
    if exact.shape != (2**int(n_sites),):
        raise ValueError("exact_state must be a full computational vector")
    return {
        str(name): tuple(
            pure_state_partial_trace(
                exact,
                int(n_sites),
                tuple(sorted(int(site) for site in support)),
            )
            for support in supports
        )
        for name, supports in support_families.items()
    }


def rhombus_resonance_actions(
    state: np.ndarray,
    actions: FixedSzPairActions,
    supercell: IntMatrix,
) -> np.ndarray:
    """Return three sums of competing opposite-edge dimer products.

    For directions ``a,b`` on a rhombus, the source is
    ``B(r,r+a)B(r+b,r+a+b)-B(r,r+b)B(r+a,r+a+b)``.  Each product joins
    disjoint bonds and is Hermitian.  The three columns cover all unordered
    pairs of nearest-neighbour orientations.
    """

    vector = np.asarray(state, dtype=np.complex128)
    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    columns = []
    for first_index in range(len(J1_POSITIVE)):
        for second_index in range(first_index + 1, len(J1_POSITIVE)):
            first_step = J1_POSITIVE[first_index]
            second_step = J1_POSITIVE[second_index]
            image = np.zeros_like(vector)
            for origin_index, origin in enumerate(sites):
                first = positions[
                    reduce_hnf(
                        (origin[0] + first_step[0], origin[1] + first_step[1]),
                        matrix,
                    )
                ]
                second = positions[
                    reduce_hnf(
                        (origin[0] + second_step[0], origin[1] + second_step[1]),
                        matrix,
                    )
                ]
                diagonal = positions[
                    reduce_hnf(
                        (
                            origin[0] + first_step[0] + second_step[0],
                            origin[1] + first_step[1] + second_step[1],
                        ),
                        matrix,
                    )
                ]
                first_covering = actions.pair(
                    actions.pair(vector, second, diagonal), origin_index, first
                )
                second_covering = actions.pair(
                    actions.pair(vector, first, diagonal), origin_index, second
                )
                image += first_covering - second_covering
            columns.append(image)
    return np.column_stack(columns)


def triangular_error_channel_actions(
    state: np.ndarray,
    basis: Sequence[Sequence[int]],
    supercell: IntMatrix,
    *,
    pair_actions: FixedSzPairActions | None = None,
) -> dict[str, np.ndarray]:
    """Build centered Gamma-A1, time-reversal-even physical sources.

    A Gamma-A1 ground-state error is exactly orthogonal to a bare M-point
    dimer, nematic doublet, or time-reversal-odd chirality.  The dictionary
    therefore contains their invariant composites ``sum_M D_M^dagger D_M``,
    ``N_1^2+N_2^2``, and ``Chi^2``.  This makes a zero overlap a physical
    statement rather than a symmetry selection rule.
    """

    vector = normalized_state(state)
    matrix = validate_hnf(supercell)
    actions = FixedSzPairActions(basis) if pair_actions is None else pair_actions
    if actions.n_sites != len(hnf_sites(matrix)):
        raise ValueError("cached pair actions do not match the supercell")
    families = hnf_oriented_bond_families(matrix)
    orientation = np.column_stack(
        [
            actions.weighted_bonds(
                vector, tuple((first, second, 1.0) for first, second in family)
            )
            for family in families
        ]
    )
    def orientation_apply(candidate: np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
        return sum(
            (
                float(coefficient)
                * actions.weighted_bonds(
                    candidate,
                    tuple((first, second, 1.0) for first, second in family),
                )
                for coefficient, family in zip(coefficients, families, strict=True)
            ),
            start=np.zeros_like(candidate),
        )

    first_nematic = (2.0, -1.0, -1.0)
    second_nematic = (0.0, 1.0, -1.0)
    n1 = orientation_apply(vector, first_nematic) / math.sqrt(6.0)
    n2 = orientation_apply(vector, second_nematic) / math.sqrt(2.0)
    nematic_scalar = (
        orientation_apply(n1, first_nematic) / math.sqrt(6.0)
        + orientation_apply(n2, second_nematic) / math.sqrt(2.0)
    )[:, None]

    dimer_scalar = np.zeros_like(vector)
    sites = hnf_sites(matrix)
    for momentum in ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):
        sources = bond_center_phase_actions(vector, actions, matrix, momentum)
        for orientation_index, (displacement, family) in enumerate(
            zip(J1_POSITIVE, families, strict=True)
        ):
            dx, dy = displacement
            adjoint_bonds = []
            for site, (first, second) in zip(sites, family, strict=True):
                phase = np.exp(
                    -2.0j
                    * np.pi
                    * (
                        momentum[0] * (site[0] + 0.5 * dx)
                        + momentum[1] * (site[1] + 0.5 * dy)
                    )
                )
                adjoint_bonds.append((first, second, phase))
            dimer_scalar += actions.weighted_bonds(
                sources[:, orientation_index], adjoint_bonds
            )
    dimer_scalar = dimer_scalar[:, None]

    chirality_once = uniform_chirality_action(vector, actions, matrix)
    chirality_scalar = uniform_chirality_action(
        chirality_once, actions, matrix
    )[:, None]
    rhombus_scalar = np.sum(
        rhombus_resonance_actions(vector, actions, matrix), axis=1
    )[:, None]
    j2_families = hnf_oriented_bond_families(matrix, J2_POSITIVE)
    j2_shell = sum(
        (
            actions.weighted_bonds(
                vector,
                tuple((first, second, 1.0) for first, second in family),
            )
            for family in j2_families
        ),
        start=np.zeros_like(vector),
    )[:, None]

    def centered(columns: np.ndarray) -> np.ndarray:
        values = np.asarray(columns, dtype=np.complex128)
        return values - vector[:, None] * (vector.conj() @ values)[None, :]

    return {
        "j2_shell_deformation": centered(j2_shell),
        "bond_nematic_scalar": centered(nematic_scalar),
        "dimer_M_star_scalar": centered(dimer_scalar),
        "chirality_squared": centered(chirality_scalar),
        "rhombus_resonance_A1": centered(rhombus_scalar),
    }


def _orthonormal_column_span(
    columns: np.ndarray, *, relative_tolerance: float = 1.0e-10
) -> np.ndarray:
    values = np.asarray(columns, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("columns must be a matrix")
    if values.shape[1] == 0:
        return np.empty((values.shape[0], 0), dtype=np.complex128)
    left, singular, _right = np.linalg.svd(values, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.empty((values.shape[0], 0), dtype=np.complex128)
    keep = singular > relative_tolerance * singular[0]
    return left[:, keep]


def error_channel_tomography(
    state: np.ndarray,
    exact_state: np.ndarray,
    hamiltonian,
    channel_actions: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Resolve the exact wavefunction error and Ritz gain by physical channel."""

    vector = normalized_state(state)
    exact = normalized_state(exact_state)
    if exact.shape != vector.shape or hamiltonian.shape != (len(vector), len(vector)):
        raise ValueError("state, exact_state, and Hamiltonian dimensions must agree")
    overlap = complex(np.vdot(vector, exact))
    if abs(overlap) > 0.0:
        exact = exact * np.exp(-1.0j * np.angle(overlap))
    overlap = complex(np.vdot(vector, exact))
    correction = exact - overlap * vector
    correction_norm = float(np.linalg.norm(correction))
    energy = float(np.vdot(vector, hamiltonian @ vector).real)
    exact_energy = float(np.vdot(exact, hamiltonian @ exact).real)
    residual = np.asarray(hamiltonian @ vector - energy * vector)
    rows: dict[str, object] = {}
    all_columns = []
    for raw_name, raw_columns in channel_actions.items():
        name = str(raw_name)
        columns = np.asarray(raw_columns, dtype=np.complex128)
        if columns.ndim == 1:
            columns = columns[:, None]
        if columns.ndim != 2 or columns.shape[0] != len(vector):
            raise ValueError("every channel action must match the state dimension")
        columns = columns - vector[:, None] * (vector.conj() @ columns)[None, :]
        basis = _orthonormal_column_span(columns)
        projected = basis @ (basis.conj().T @ correction) if basis.shape[1] else np.zeros_like(correction)
        fraction = (
            float(np.vdot(projected, projected).real / correction_norm**2)
            if correction_norm > 1.0e-14
            else 0.0
        )
        augmented = _orthonormal_column_span(np.column_stack((vector, basis)))
        projected_h = augmented.conj().T @ (hamiltonian @ augmented)
        ritz_energy = float(np.linalg.eigvalsh(0.5 * (projected_h + projected_h.conj().T))[0].real)
        denominator = max(energy - exact_energy, 1.0e-15)
        rows[name] = {
            "rank": int(basis.shape[1]),
            "exact_error_fraction": fraction,
            "ritz_energy": ritz_energy,
            "energy_error_recovery": float(np.clip((energy - ritz_energy) / denominator, 0.0, 1.0)),
        }
        all_columns.append(columns)
    combined = np.column_stack(all_columns) if all_columns else np.empty((len(vector), 0))
    combined_basis = _orthonormal_column_span(combined)
    combined_projection = (
        combined_basis @ (combined_basis.conj().T @ correction)
        if combined_basis.shape[1]
        else np.zeros_like(correction)
    )
    augmented = _orthonormal_column_span(np.column_stack((vector, combined_basis)))
    projected_h = augmented.conj().T @ (hamiltonian @ augmented)
    combined_ritz = float(
        np.linalg.eigvalsh(0.5 * (projected_h + projected_h.conj().T))[0].real
    )
    correction_residual_cosine = 0.0
    if correction_norm > 1.0e-14 and np.linalg.norm(residual) > 1.0e-14:
        correction_residual_cosine = float(
            abs(np.vdot(correction, residual))
            / (correction_norm * np.linalg.norm(residual))
        )
    return {
        "fidelity": float(abs(overlap) ** 2),
        "exact_correction_norm": correction_norm,
        "energy": energy,
        "exact_energy": exact_energy,
        "variance": float(np.vdot(residual, residual).real),
        "residual_exact_correction_abs_cosine": correction_residual_cosine,
        "channels": rows,
        "combined": {
            "rank": int(combined_basis.shape[1]),
            "exact_error_fraction": (
                float(np.vdot(combined_projection, combined_projection).real / correction_norm**2)
                if correction_norm > 1.0e-14
                else 0.0
            ),
            "ritz_energy": combined_ritz,
            "energy_error_recovery": float(
                np.clip(
                    (energy - combined_ritz) / max(energy - exact_energy, 1.0e-15),
                    0.0,
                    1.0,
                )
            ),
        },
    }


__all__ = [
    "FixedSzPairActions",
    "bond_center_phase_actions",
    "error_channel_tomography",
    "exchange_action",
    "fixed_sector_to_full_state",
    "hnf_oriented_bond_families",
    "hnf_oriented_triangles",
    "local_rdm_error_profile",
    "local_rdm_reference_cache",
    "normalized_state",
    "pair_permutation_indices",
    "rhombus_resonance_actions",
    "triangular_cluster_supports",
    "triangular_error_channel_actions",
    "triangular_periodic_observable_panel",
    "uniform_chirality_action",
]
