"""Exact two-cluster diagnostics for the six-site singlet core.

The calculation in this module has a deliberately narrow physical scope.  Two
open ``2 x 3`` clusters tile an open ``2 x 6`` strip.  The unperturbed
Hamiltonian is the sum of the two isolated-cluster Hamiltonians and ``V`` is
*all* ``J1-J2`` bonds crossing their common vertical cut.  Starting from the
product of the two exact cluster ground singlets, a single boundary exchange
is a rank-one tensor on each cluster.  The Wigner-Eckart selection rule then
restricts ``V |0,0>`` exactly to an ``S_C=1`` multiplet on both clusters.

Two complementary quantities are exposed:

* a spectral second-order decomposition in the nine six-spin triplet
  multiplets, including the response-Schmidt rank of the boundary action;
* a separate, finite-coupling ``q=0`` resonance diagnostic obtained by
  diagonalizing the full periodic ``2 x 6`` Hamiltonian in the span of all
  translated two-cluster product coverings.

The latter is not a thermodynamic estimate.  With only two rows, periodic
canonicalization combines distinct lattice images that connect the same pair;
the returned report states that thin-torus convention explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np

from ..frontier.triangle_patch_physical_init import (
    triangle_patch_cluster_reference,
)
from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.triangular_j1j2 import triangular_j1j2_bonds
from .exact_diagonalization import (
    computational_basis,
    heisenberg_sz_matrix,
    spin_squared_matrix,
)


Geometry = Literal["square", "triangular"]
SpinBond = tuple[int, int, float]


@dataclass(frozen=True)
class InterclusterBond:
    """One open-strip bond joining the two canonical ``2 x 3`` clusters."""

    first: int
    second: int
    first_local: int
    second_local: int
    coupling: float
    shell: Literal["J1", "J2"]


@dataclass(frozen=True)
class TwoClusterOpenPartition:
    """Audited open ``2 x 6`` decomposition ``H = H_C1 + H_C2 + V``."""

    geometry: Geometry
    j2_over_j1: float
    cluster_sites: tuple[tuple[int, ...], tuple[int, ...]]
    cluster_bonds: tuple[SpinBond, ...]
    full_bonds: tuple[SpinBond, ...]
    intercluster_bonds: tuple[InterclusterBond, ...]
    reconstruction_error: float


@dataclass(frozen=True)
class FullCouplingCarrierResult:
    """One ``lambda=1`` variational subspace on the open ``2 x 6`` strip."""

    carrier: str
    left_carrier_count: int
    right_carrier_count: int
    triplet_pair_dimension: int
    total_subspace_dimension: int
    ground_energy: float
    recovered_exact_gap_fraction: float


@dataclass(frozen=True)
class TwoClusterVirtualProcessReport:
    """Exact weak-boundary response of two six-spin ground singlets.

    All spin labels are doubled integers.  ``triplet_energies`` contains one
    representative (the ``M=0`` member) of each of the nine ``q=2``
    multiplets.  ``response_amplitude_matrix[a,b]`` is the normalized
    total-singlet amplitude

    ``< (t_a t_b) S_tot=0 | V | g g >``.

    Its Frobenius norm squared is therefore ``||V|gg>||^2``.
    """

    geometry: Geometry
    j2_over_j1: float
    cluster_energy: float
    cluster_singlet_gap: float
    intercluster_bonds: tuple[InterclusterBond, ...]
    triplet_energies: np.ndarray
    response_amplitude_matrix: np.ndarray
    response_singular_values: np.ndarray
    response_schmidt_rank: int
    response_cumulative_norm_fractions: np.ndarray
    action_norm_squared: float
    q2_q2_norm_squared: float
    outside_q2_q2_norm_bound: float
    rank_one_selection_residual: float
    first_order_expectation: float
    second_order_energy: float
    cumulative_low_energy_triplet_norm_fractions: np.ndarray
    cumulative_low_energy_triplet_e2_fractions: np.ndarray
    lowest_triplet_pair_norm_fraction: float
    lowest_triplet_pair_e2_fraction: float
    leading_left_lowest_triplet_overlap: float
    leading_right_lowest_triplet_overlap: float
    response_leading_left_right_overlap: float
    response_asymmetry_fraction: float
    correction_amplitude_matrix: np.ndarray
    correction_singular_values: np.ndarray
    correction_cumulative_norm_fractions: np.ndarray
    raw_response_carrier_e2_fractions: np.ndarray
    correction_carrier_e2_fractions: np.ndarray
    correction_truncated_svd_e2_fractions: np.ndarray
    correction_leading_left_right_overlap: float
    open_product_energy: float
    open_exact_ground_energy: float
    open_coupling_energy_gain: float
    second_order_open_gap_fraction: float
    open_product_ground_fidelity: float
    open_left_core_probability: float
    open_right_core_probability: float
    exact_ground_sector_q_values: np.ndarray
    exact_ground_equal_q_sector_weights: np.ndarray
    exact_ground_offdiagonal_q_weight: float
    exact_ground_excited_q0_pair_weight: float
    response_matrix_physical_residual: float
    full_coupling_carriers: tuple[FullCouplingCarrierResult, ...]


@dataclass(frozen=True)
class CoveringResonanceReport:
    """Exact ``q=0`` covering-resonance audit on a periodic ``2 x 6`` torus."""

    geometry: Geometry
    j2_over_j1: float
    boundary_convention: str
    raw_translation_count: int
    distinct_covering_count: int
    subspace_rank: int
    overlap_matrix: np.ndarray
    projected_hamiltonian: np.ndarray
    overlap_eigenvalues: np.ndarray
    single_covering_energy: float
    uniform_coherent_energy: float
    generalized_ground_energy: float
    resonance_energy_gain: float
    exact_ground_energy: float
    recovered_exact_gap_fraction: float
    column_only_covering_count: int
    column_only_subspace_rank: int
    column_only_ground_energy: float
    column_only_resonance_energy_gain: float
    column_only_recovered_exact_gap_fraction: float


@dataclass(frozen=True)
class ClusterQ2ResponseBasis:
    """Exact six-site ``q=2`` response data in a fixed local convention.

    ``sz_transition_matrix[a, i]`` is
    ``<t_a,m=0|S_i^z|g>`` in row-major local-site order.  It is the only
    cluster-local ingredient needed to turn an arbitrary intercluster
    coupling matrix ``J[i,j]`` into the reduced singlet-pair response
    ``sqrt(3) Z J Z.T``.  Exposing this small exact object lets companion
    support/orientation audits reuse the same eigenbasis and Condon-Shortley
    convention as the canonical open-cut benchmark.
    """

    geometry: Geometry
    j2_over_j1: float
    cluster_energy: float
    triplet_energies: np.ndarray
    triplet_gaps: np.ndarray
    sz_transition_matrix: np.ndarray


_CLUSTER_SITES = (
    (0, 1, 2, 6, 7, 8),
    (3, 4, 5, 9, 10, 11),
)


def _checked_geometry(geometry: str) -> Geometry:
    if geometry not in {"square", "triangular"}:
        raise ValueError("geometry must be 'square' or 'triangular'")
    return geometry


def _checked_ratio(j2_over_j1: float) -> float:
    ratio = float(j2_over_j1)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("j2_over_j1 must be finite and nonnegative")
    return ratio


def _geometry_bonds(
    geometry: Geometry,
    rows: int,
    columns: int,
    *,
    j1: float,
    j2: float,
    periodic: bool,
) -> tuple[SpinBond, ...]:
    generator = (
        square_j1j2_bonds if geometry == "square" else triangular_j1j2_bonds
    )
    return generator(
        rows,
        columns,
        j1=float(j1),
        j2=float(j2),
        periodic=periodic,
    )


def _bond_dictionary(bonds: tuple[SpinBond, ...]) -> dict[tuple[int, int], float]:
    return {(int(first), int(second)): float(coupling) for first, second, coupling in bonds}


@lru_cache(maxsize=None)
def _open_partition_cached(
    geometry: Geometry, j2_over_j1: float
) -> TwoClusterOpenPartition:
    geometry = _checked_geometry(geometry)
    ratio = _checked_ratio(j2_over_j1)
    full = _geometry_bonds(
        geometry, 2, 6, j1=1.0, j2=ratio, periodic=False
    )
    j1_pairs = set(
        _bond_dictionary(
            _geometry_bonds(geometry, 2, 6, j1=1.0, j2=0.0, periodic=False)
        )
    )
    j2_pairs = set(
        _bond_dictionary(
            _geometry_bonds(geometry, 2, 6, j1=0.0, j2=1.0, periodic=False)
        )
    )
    site_sets = tuple(set(sites) for sites in _CLUSTER_SITES)
    local_maps = tuple(
        {site: local for local, site in enumerate(sites)}
        for sites in _CLUSTER_SITES
    )
    internal: list[list[SpinBond]] = [[], []]
    crossing: list[InterclusterBond] = []
    for first, second, coupling in full:
        pair = (first, second)
        if first in site_sets[0] and second in site_sets[0]:
            internal[0].append(
                (local_maps[0][first], local_maps[0][second], coupling)
            )
        elif first in site_sets[1] and second in site_sets[1]:
            internal[1].append(
                (local_maps[1][first], local_maps[1][second], coupling)
            )
        else:
            if first in site_sets[1]:
                first, second = second, first
            if first not in site_sets[0] or second not in site_sets[1]:
                raise RuntimeError("the two clusters do not partition the open strip")
            in_j1, in_j2 = pair in j1_pairs, pair in j2_pairs
            if in_j1 == in_j2:
                raise RuntimeError("an open-strip boundary bond has ambiguous shell")
            crossing.append(
                InterclusterBond(
                    first=first,
                    second=second,
                    first_local=local_maps[0][first],
                    second_local=local_maps[1][second],
                    coupling=float(coupling),
                    shell="J1" if in_j1 else "J2",
                )
            )

    reference = triangle_patch_cluster_reference(geometry, ratio)
    canonical_internal = tuple(sorted(reference.bonds))
    if tuple(sorted(internal[0])) != canonical_internal:
        raise RuntimeError("left cluster does not reproduce the canonical open core")
    if tuple(sorted(internal[1])) != canonical_internal:
        raise RuntimeError("right cluster does not reproduce the canonical open core")

    reconstructed: dict[tuple[int, int], float] = {}
    for cluster, local_bonds in zip(_CLUSTER_SITES, internal, strict=True):
        for first, second, coupling in local_bonds:
            pair = tuple(sorted((cluster[first], cluster[second])))
            reconstructed[pair] = reconstructed.get(pair, 0.0) + coupling
    for bond in crossing:
        pair = tuple(sorted((bond.first, bond.second)))
        reconstructed[pair] = reconstructed.get(pair, 0.0) + bond.coupling
    target = _bond_dictionary(full)
    all_pairs = set(target) | set(reconstructed)
    error = max(
        (abs(target.get(pair, 0.0) - reconstructed.get(pair, 0.0)) for pair in all_pairs),
        default=0.0,
    )
    if error > 1.0e-14:
        raise RuntimeError("H_C1 + H_C2 + V does not reconstruct the open graph")
    return TwoClusterOpenPartition(
        geometry=geometry,
        j2_over_j1=ratio,
        cluster_sites=_CLUSTER_SITES,
        cluster_bonds=canonical_internal,
        full_bonds=tuple(full),
        intercluster_bonds=tuple(crossing),
        reconstruction_error=float(error),
    )


def two_cluster_open_partition(
    geometry: Geometry, j2_over_j1: float
) -> TwoClusterOpenPartition:
    """Return the canonical open ``2 x 6`` two-cluster bond partition."""

    return _open_partition_cached(
        _checked_geometry(geometry), _checked_ratio(j2_over_j1)
    )


def _simultaneous_h_s2_basis(
    hamiltonian: np.ndarray,
    spin_squared: np.ndarray,
    *,
    energy_tolerance: float = 2.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simultaneously diagonalize two commuting real symmetric matrices."""

    energies, vectors = np.linalg.eigh(hamiltonian)
    resolved_vectors: list[np.ndarray] = []
    resolved_energies: list[float] = []
    resolved_q: list[int] = []
    start = 0
    while start < len(energies):
        stop = start + 1
        while (
            stop < len(energies)
            and abs(float(energies[stop] - energies[start])) <= energy_tolerance
        ):
            stop += 1
        block = vectors[:, start:stop]
        s2_values, rotations = np.linalg.eigh(block.T @ spin_squared @ block)
        for offset, s2_value in enumerate(s2_values):
            vector = np.asarray(block @ rotations[:, offset], dtype=np.float64)
            q = int(round(math.sqrt(max(0.0, 1.0 + 4.0 * float(s2_value))) - 1.0))
            exact_s2 = q * (q + 2) / 4.0
            if abs(float(s2_value) - exact_s2) > 2.0e-8:
                raise RuntimeError("failed to resolve a doubled total-spin label")
            resolved_vectors.append(vector)
            resolved_energies.append(float(energies[start]))
            resolved_q.append(q)
        start = stop
    order = sorted(
        range(len(resolved_energies)),
        key=lambda index: (resolved_energies[index], resolved_q[index]),
    )
    return (
        np.asarray([resolved_energies[index] for index in order]),
        np.asarray([resolved_q[index] for index in order], dtype=np.int32),
        np.column_stack([resolved_vectors[index] for index in order]),
    )


def _cluster_m0_spectrum(
    geometry: Geometry, ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]:
    reference = triangle_patch_cluster_reference(geometry, ratio)
    hamiltonian, basis = heisenberg_sz_matrix(6, reference.bonds, n_up=3)
    spin_squared, spin_basis = spin_squared_matrix(6, n_up=3)
    if basis != spin_basis:
        raise RuntimeError("the H and S^2 bases disagree")
    energies, q_values, vectors = _simultaneous_h_s2_basis(
        hamiltonian, spin_squared
    )
    full_positions = {state: index for index, state in enumerate(range(64))}
    ground = np.asarray(
        [reference.computational_state[full_positions[state]] for state in basis],
        dtype=np.float64,
    )
    ground /= np.linalg.norm(ground)
    residual = np.linalg.norm(hamiltonian @ ground - reference.energy * ground)
    if residual > 2.0e-12:
        raise RuntimeError("the cluster core is not the M=0 ground eigenstate")
    return energies, q_values, vectors, basis, ground, spin_squared


@lru_cache(maxsize=None)
def _cluster_q2_response_basis_cached(
    geometry: Geometry, j2_over_j1: float
) -> ClusterQ2ResponseBasis:
    reference = triangle_patch_cluster_reference(geometry, j2_over_j1)
    energies, q_values, vectors, basis, ground, _ = _cluster_m0_spectrum(
        geometry, j2_over_j1
    )
    selected = q_values == 2
    triplet_energies = np.asarray(energies[selected], dtype=np.float64)
    triplets = np.asarray(vectors[:, selected], dtype=np.float64)
    if triplets.shape != (20, 9):
        raise RuntimeError("six spins must contain nine triplet multiplets")
    bits = np.asarray(
        [
            [(state >> (5 - site)) & 1 for site in range(6)]
            for state in basis
        ],
        dtype=np.int8,
    )
    local_sz = bits.astype(np.float64) - 0.5
    transitions = np.asarray(
        triplets.T @ (ground[:, None] * local_sz), dtype=np.float64
    )
    gaps = np.asarray(triplet_energies - reference.energy, dtype=np.float64)
    if np.any(gaps <= 0.0):
        raise RuntimeError("the cluster triplet gaps must be positive")
    for array in (triplet_energies, gaps, transitions):
        array.setflags(write=False)
    return ClusterQ2ResponseBasis(
        geometry=geometry,
        j2_over_j1=j2_over_j1,
        cluster_energy=reference.energy,
        triplet_energies=triplet_energies,
        triplet_gaps=gaps,
        sz_transition_matrix=transitions,
    )


def cluster_q2_response_basis(
    geometry: Geometry, j2_over_j1: float
) -> ClusterQ2ResponseBasis:
    """Return the canonical exact ``2 x 3`` ground-to-triplet response basis."""

    return _cluster_q2_response_basis_cached(
        _checked_geometry(geometry), _checked_ratio(j2_over_j1)
    )


def _carrier_subspace_e2_fractions(
    source: np.ndarray,
    denominators: np.ndarray,
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
) -> np.ndarray:
    """Variational E2 recovery in nested left/right carrier subspaces.

    At depth ``k`` the correction is allowed to be any matrix in
    ``span(left[:,:k]) x span(right[:,:k])``.  The coefficient matrix is
    optimized with the *full* diagonal resolvent metric rather than by merely
    truncating the source.  This is the relevant measure for choosing a
    support-aware carrier bank.
    """

    exact_lowering = float(np.sum(np.square(np.abs(source)) / denominators))
    fractions = []
    for count in range(1, source.shape[0] + 1):
        basis = [
            np.outer(left_vectors[:, left], right_vectors[:, right])
            for left in range(count)
            for right in range(count)
        ]
        force = np.asarray(
            [np.vdot(vector, source).real for vector in basis],
            dtype=np.float64,
        )
        metric = np.asarray(
            [
                [np.sum(first * denominators * second) for second in basis]
                for first in basis
            ],
            dtype=np.float64,
        )
        recovered = float(force @ np.linalg.solve(metric, force))
        fractions.append(recovered / exact_lowering)
    return np.asarray(fractions, dtype=np.float64)


def carrier_subspace_e2_fractions(
    source: np.ndarray,
    denominators: np.ndarray,
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
) -> np.ndarray:
    """Return nested variational ``E2`` recovery for local carrier bases.

    This public wrapper is shared by exact-small support/orientation audits;
    the arrays use the nine ``q=2`` multiplets of one six-site cluster and do
    not contain magnetic indices.
    """

    source = np.asarray(source, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    left_vectors = np.asarray(left_vectors, dtype=np.float64)
    right_vectors = np.asarray(right_vectors, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("source must be a square matrix")
    size = source.shape[0]
    if denominators.shape != source.shape:
        raise ValueError("denominators must match source")
    if left_vectors.shape != (size, size) or right_vectors.shape != (size, size):
        raise ValueError("left and right carrier bases must be square and complete")
    if np.any(denominators <= 0.0):
        raise ValueError("all excitation denominators must be positive")
    return _carrier_subspace_e2_fractions(
        source, denominators, left_vectors, right_vectors
    )


def _triplet_multiplet_states(
    triplets_m0: np.ndarray,
    m0_basis: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return consistently phased ``m=+1,0,-1`` members of each triplet."""

    zero = np.zeros((64, triplets_m0.shape[1]), dtype=np.float64)
    for position, state in enumerate(m0_basis):
        zero[state, :] = triplets_m0[position, :]

    plus = np.zeros_like(zero)
    minus = np.zeros_like(zero)
    for state in range(64):
        row = zero[state, :]
        if not np.any(row):
            continue
        for site in range(6):
            mask = 1 << (5 - site)
            if state & mask:
                minus[state ^ mask, :] += row / np.sqrt(2.0)
            else:
                plus[state ^ mask, :] += row / np.sqrt(2.0)
    for member in (plus, zero, minus):
        error = np.linalg.norm(member.T @ member - np.eye(9))
        if error > 3.0e-12:
            raise RuntimeError("failed to construct orthonormal triplet multiplets")
    return plus, zero, minus


def _pair_coefficient_matrix_to_fixed_state(
    coefficients: np.ndarray,
    multiplets: tuple[np.ndarray, np.ndarray, np.ndarray],
    basis: tuple[int, ...],
) -> np.ndarray:
    """Map a nine-by-nine reduced triplet-pair singlet to the N=12 basis."""

    plus, zero, minus = multiplets
    cluster_matrix = (
        plus @ coefficients @ minus.T
        - zero @ coefficients @ zero.T
        + minus @ coefficients @ plus.T
    ) / np.sqrt(3.0)
    answer = np.empty(len(basis), dtype=np.float64)
    for position, state in enumerate(basis):
        local_indices = []
        for sites in _CLUSTER_SITES:
            index = 0
            for site in sites:
                index = (index << 1) | ((state >> (11 - site)) & 1)
            local_indices.append(index)
        answer[position] = cluster_matrix[local_indices[0], local_indices[1]]
    return answer


def _fixed_state_to_cluster_matrix(
    state: np.ndarray, basis: tuple[int, ...]
) -> np.ndarray:
    matrix = np.zeros((64, 64), dtype=np.float64)
    for amplitude, configuration in zip(state, basis, strict=True):
        local_indices = []
        for sites in _CLUSTER_SITES:
            index = 0
            for site in sites:
                index = (index << 1) | ((configuration >> (11 - site)) & 1)
            local_indices.append(index)
        matrix[local_indices[0], local_indices[1]] = amplitude
    return matrix


def _full_coupling_carrier_result(
    name: str,
    hamiltonian: np.ndarray,
    product: np.ndarray,
    multiplets: tuple[np.ndarray, np.ndarray, np.ndarray],
    basis: tuple[int, ...],
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    count: int,
    *,
    product_energy: float,
    exact_energy: float,
) -> FullCouplingCarrierResult:
    pair_states = [
        _pair_coefficient_matrix_to_fixed_state(
            np.outer(left_vectors[:, left], right_vectors[:, right]),
            multiplets,
            basis,
        )
        for left in range(count)
        for right in range(count)
    ]
    columns = np.column_stack([product, *pair_states])
    gram_error = np.linalg.norm(columns.T @ columns - np.eye(columns.shape[1]))
    if gram_error > 5.0e-12:
        raise RuntimeError("the full-coupling carrier basis is not orthonormal")
    reduced = columns.T @ hamiltonian @ columns
    energy = float(np.linalg.eigvalsh(reduced)[0])
    exact_gap = product_energy - exact_energy
    return FullCouplingCarrierResult(
        carrier=name,
        left_carrier_count=count,
        right_carrier_count=count,
        triplet_pair_dimension=count * count,
        total_subspace_dimension=1 + count * count,
        ground_energy=energy,
        recovered_exact_gap_fraction=(product_energy - energy) / exact_gap,
    )


@lru_cache(maxsize=None)
def _virtual_report_cached(
    geometry: Geometry, j2_over_j1: float
) -> TwoClusterVirtualProcessReport:
    geometry = _checked_geometry(geometry)
    ratio = _checked_ratio(j2_over_j1)
    partition = _open_partition_cached(geometry, ratio)
    reference = triangle_patch_cluster_reference(geometry, ratio)
    energies, q_values, vectors, basis, ground, spin_squared = _cluster_m0_spectrum(
        geometry, ratio
    )
    triplet_mask = q_values == 2
    triplet_energies = np.asarray(energies[triplet_mask], dtype=np.float64)
    triplets = np.asarray(vectors[:, triplet_mask], dtype=np.float64)
    if triplets.shape != (20, 9):
        raise RuntimeError("six spins must contain nine triplet multiplets")

    bits = np.asarray(
        [
            [(state >> (5 - site)) & 1 for site in range(6)]
            for state in basis
        ],
        dtype=np.int8,
    )
    local_sz = bits.astype(np.float64) - 0.5
    # Z[a,i] = <t_a,m=0| S_i^z |g>.  Wigner-Eckart then supplies the
    # normalized two-triplet singlet amplitude sqrt(3) * Z J Z^T.
    transitions = triplets.T @ (ground[:, None] * local_sz)
    boundary_coupling = np.zeros((6, 6), dtype=np.float64)
    for bond in partition.intercluster_bonds:
        boundary_coupling[bond.first_local, bond.second_local] += bond.coupling
    amplitude = np.sqrt(3.0) * transitions @ boundary_coupling @ transitions.T
    weights = np.square(np.abs(amplitude))
    action_norm = float(np.sum(weights))
    gaps = triplet_energies - reference.energy
    denominators = gaps[:, None] + gaps[None, :]
    if np.any(denominators <= 0.0):
        raise RuntimeError("the triplet excitation denominators must be positive")
    e2_terms = weights / denominators
    e2_magnitude = float(np.sum(e2_terms))
    second_order = -e2_magnitude

    cumulative_norm = np.asarray(
        [float(np.sum(weights[:count, :count]) / action_norm) for count in range(1, 10)]
    )
    cumulative_e2 = np.asarray(
        [float(np.sum(e2_terms[:count, :count]) / e2_magnitude) for count in range(1, 10)]
    )
    singular_values = np.linalg.svd(amplitude, compute_uv=False)
    left, singular_values_again, right_adjoint = np.linalg.svd(
        amplitude, full_matrices=False
    )
    if not np.allclose(singular_values, singular_values_again, atol=1.0e-14):
        raise RuntimeError("inconsistent response singular values")
    threshold = 2.0e-12 * max(1.0, float(singular_values[0]))
    rank = int(np.sum(singular_values > threshold))
    cumulative_svd = np.cumsum(np.square(singular_values)) / action_norm
    raw_carrier_e2 = _carrier_subspace_e2_fractions(
        amplitude,
        denominators,
        left,
        right_adjoint.T,
    )

    # The exact first-order wavefunction correction has coefficients
    # X_ab = A_ab / (Delta_a + Delta_b).  Its SVD is therefore a
    # resolvent-dressed carrier ordering, distinct from the raw boundary
    # response SVD whenever the multiplet gaps are nonuniform.
    correction = amplitude / denominators
    correction_left, correction_singular, correction_right_adjoint = np.linalg.svd(
        correction, full_matrices=False
    )
    correction_cumulative = (
        np.cumsum(np.square(correction_singular))
        / float(np.sum(np.square(correction_singular)))
    )
    correction_carrier_e2 = _carrier_subspace_e2_fractions(
        amplitude,
        denominators,
        correction_left,
        correction_right_adjoint.T,
    )
    correction_truncated_e2 = []
    for count in range(1, 10):
        truncated = (
            correction_left[:, :count]
            * correction_singular[None, :count]
        ) @ correction_right_adjoint[:count, :]
        lowering = (
            2.0 * float(np.sum(amplitude * truncated))
            - float(np.sum(denominators * np.square(truncated)))
        )
        correction_truncated_e2.append(lowering / e2_magnitude)
    correction_truncated_e2_array = np.asarray(
        correction_truncated_e2, dtype=np.float64
    )

    identity = np.eye(len(basis))
    selection_residual = max(
        float(np.linalg.norm((spin_squared - 2.0 * identity) @ (local_sz[:, site] * ground)))
        for site in range(6)
    )
    # <g|S_i|g>=0 for a singlet, so the factorized first-order expectation
    # vanishes.  Keep the evaluated number as an audit rather than hard-code it.
    local_magnetizations = ground @ (local_sz * ground[:, None])
    first_order = float(
        sum(
            bond.coupling
            * local_magnetizations[bond.first_local]
            * local_magnetizations[bond.second_local]
            for bond in partition.intercluster_bonds
        )
    )
    outside_bound = max(
        0.0,
        12.0 * selection_residual * selection_residual
        * float(np.sum(np.square(boundary_coupling))),
    )
    asymmetry = float(
        np.linalg.norm(amplitude - amplitude.T) / np.linalg.norm(amplitude)
    )

    # Move from the weak-boundary audit to a lambda=1 open-strip check.  The
    # exact ground state is used only at N=12.  This asks both whether the
    # cluster core remains recognizable and whether the perturbatively chosen
    # carrier ordering survives at full coupling.
    open_hamiltonian, open_basis = heisenberg_sz_matrix(
        12, partition.full_bonds, n_up=6
    )
    open_values, open_vectors = np.linalg.eigh(open_hamiltonian)
    open_exact_energy = float(open_values[0])
    open_ground = np.asarray(open_vectors[:, 0], dtype=np.float64)
    product = _cluster_product_state(reference.computational_state, open_basis)
    product_energy = float(product @ (open_hamiltonian @ product))
    if abs(product_energy - 2.0 * reference.energy) > 3.0e-13:
        raise RuntimeError("the open product energy is not twice the cluster energy")
    open_gain = product_energy - open_exact_energy
    product_fidelity = float(abs(np.vdot(product, open_ground)) ** 2)
    open_cluster_matrix = _fixed_state_to_cluster_matrix(open_ground, open_basis)
    local_ground = reference.computational_state
    left_core_probability = float(
        np.linalg.norm(local_ground @ open_cluster_matrix) ** 2
    )
    right_core_probability = float(
        np.linalg.norm(open_cluster_matrix @ local_ground) ** 2
    )

    full_spin_squared, full_spin_basis = spin_squared_matrix(6)
    if full_spin_basis != tuple(range(64)):
        raise RuntimeError("the full cluster S^2 basis order changed")
    sector_values, sector_vectors = np.linalg.eigh(full_spin_squared)
    sector_q_values = np.asarray((0, 2, 4, 6), dtype=np.int32)
    sector_projectors = []
    for q in sector_q_values:
        target = q * (q + 2) / 4.0
        selected = np.abs(sector_values - target) < 2.0e-10
        sector_projectors.append(
            sector_vectors[:, selected] @ sector_vectors[:, selected].T
        )
    equal_sector_weights = np.asarray(
        [
            float(
                np.linalg.norm(
                    projector @ open_cluster_matrix @ projector.T
                )
                ** 2
            )
            for projector in sector_projectors
        ],
        dtype=np.float64,
    )
    offdiagonal_weight = float(
        sum(
            np.linalg.norm(
                left_projector @ open_cluster_matrix @ right_projector.T
            )
            ** 2
            for left_index, left_projector in enumerate(sector_projectors)
            for right_index, right_projector in enumerate(sector_projectors)
            if left_index != right_index
        )
    )
    if abs(float(np.sum(equal_sector_weights)) + offdiagonal_weight - 1.0) > 3.0e-12:
        raise RuntimeError("the cluster-spin sectors do not resolve the open ground state")
    excited_q0_pair_weight = float(equal_sector_weights[0] - product_fidelity)

    multiplets = _triplet_multiplet_states(triplets, basis)
    open_action = open_hamiltonian @ product
    physical_response = np.empty((9, 9), dtype=np.float64)
    for left_index in range(9):
        for right_index in range(9):
            coefficients = np.zeros((9, 9), dtype=np.float64)
            coefficients[left_index, right_index] = 1.0
            pair_state = _pair_coefficient_matrix_to_fixed_state(
                coefficients, multiplets, open_basis
            )
            physical_response[left_index, right_index] = pair_state @ open_action
    response_residual = float(
        min(
            np.linalg.norm(physical_response - amplitude),
            np.linalg.norm(physical_response + amplitude),
        )
        / np.linalg.norm(amplitude)
    )
    if response_residual > 3.0e-12:
        raise RuntimeError("the reduced response matrix failed its physical audit")

    identity_triplets = np.eye(9)
    full_coupling_carriers = (
        _full_coupling_carrier_result(
            "lowest_triplet_pair",
            open_hamiltonian,
            product,
            multiplets,
            open_basis,
            identity_triplets,
            identity_triplets,
            1,
            product_energy=product_energy,
            exact_energy=open_exact_energy,
        ),
        *(
            _full_coupling_carrier_result(
                f"raw_response_schmidt_k{count}",
                open_hamiltonian,
                product,
                multiplets,
                open_basis,
                left,
                right_adjoint.T,
                count,
                product_energy=product_energy,
                exact_energy=open_exact_energy,
            )
            for count in (1, 2)
        ),
        *(
            _full_coupling_carrier_result(
                f"resolvent_dressed_schmidt_k{count}",
                open_hamiltonian,
                product,
                multiplets,
                open_basis,
                correction_left,
                correction_right_adjoint.T,
                count,
                product_energy=product_energy,
                exact_energy=open_exact_energy,
            )
            for count in (1, 2)
        ),
        _full_coupling_carrier_result(
            "complete_q2_pair_space",
            open_hamiltonian,
            product,
            multiplets,
            open_basis,
            identity_triplets,
            identity_triplets,
            9,
            product_energy=product_energy,
            exact_energy=open_exact_energy,
        ),
    )

    for array in (
        triplet_energies,
        amplitude,
        singular_values,
        cumulative_svd,
        cumulative_norm,
        cumulative_e2,
        correction,
        correction_singular,
        correction_cumulative,
        raw_carrier_e2,
        correction_carrier_e2,
        correction_truncated_e2_array,
        sector_q_values,
        equal_sector_weights,
    ):
        array.setflags(write=False)
    return TwoClusterVirtualProcessReport(
        geometry=geometry,
        j2_over_j1=ratio,
        cluster_energy=reference.energy,
        cluster_singlet_gap=reference.singlet_gap,
        intercluster_bonds=partition.intercluster_bonds,
        triplet_energies=triplet_energies,
        response_amplitude_matrix=amplitude,
        response_singular_values=singular_values,
        response_schmidt_rank=rank,
        response_cumulative_norm_fractions=cumulative_svd,
        action_norm_squared=action_norm,
        q2_q2_norm_squared=action_norm,
        outside_q2_q2_norm_bound=outside_bound,
        rank_one_selection_residual=selection_residual,
        first_order_expectation=first_order,
        second_order_energy=second_order,
        cumulative_low_energy_triplet_norm_fractions=cumulative_norm,
        cumulative_low_energy_triplet_e2_fractions=cumulative_e2,
        lowest_triplet_pair_norm_fraction=float(cumulative_norm[0]),
        lowest_triplet_pair_e2_fraction=float(cumulative_e2[0]),
        leading_left_lowest_triplet_overlap=float(abs(left[0, 0]) ** 2),
        leading_right_lowest_triplet_overlap=float(abs(right_adjoint[0, 0]) ** 2),
        response_leading_left_right_overlap=float(
            abs(np.vdot(left[:, 0], right_adjoint[0, :])) ** 2
        ),
        response_asymmetry_fraction=asymmetry,
        correction_amplitude_matrix=correction,
        correction_singular_values=correction_singular,
        correction_cumulative_norm_fractions=correction_cumulative,
        raw_response_carrier_e2_fractions=raw_carrier_e2,
        correction_carrier_e2_fractions=correction_carrier_e2,
        correction_truncated_svd_e2_fractions=correction_truncated_e2_array,
        correction_leading_left_right_overlap=float(
            abs(
                np.vdot(
                    correction_left[:, 0], correction_right_adjoint[0, :]
                )
            )
            ** 2
        ),
        open_product_energy=product_energy,
        open_exact_ground_energy=open_exact_energy,
        open_coupling_energy_gain=open_gain,
        second_order_open_gap_fraction=e2_magnitude / open_gain,
        open_product_ground_fidelity=product_fidelity,
        open_left_core_probability=left_core_probability,
        open_right_core_probability=right_core_probability,
        exact_ground_sector_q_values=sector_q_values,
        exact_ground_equal_q_sector_weights=equal_sector_weights,
        exact_ground_offdiagonal_q_weight=offdiagonal_weight,
        exact_ground_excited_q0_pair_weight=excited_q0_pair_weight,
        response_matrix_physical_residual=response_residual,
        full_coupling_carriers=full_coupling_carriers,
    )


def two_cluster_virtual_process_report(
    geometry: Geometry, j2_over_j1: float
) -> TwoClusterVirtualProcessReport:
    """Return the exact ``S_C=1`` virtual-pair decomposition."""

    return _virtual_report_cached(
        _checked_geometry(geometry), _checked_ratio(j2_over_j1)
    )


def _cluster_product_state(
    local_ground: np.ndarray,
    basis: tuple[int, ...],
) -> np.ndarray:
    amplitudes = np.empty(len(basis), dtype=np.float64)
    for position, state in enumerate(basis):
        local_indices = []
        for sites in _CLUSTER_SITES:
            index = 0
            for site in sites:
                index = (index << 1) | ((state >> (11 - site)) & 1)
            local_indices.append(index)
        amplitudes[position] = (
            local_ground[local_indices[0]] * local_ground[local_indices[1]]
        )
    norm = np.linalg.norm(amplitudes)
    if abs(norm - 1.0) > 2.0e-13:
        raise RuntimeError("the embedded two-cluster product is not normalized")
    return amplitudes


def _translated_state(
    state: np.ndarray,
    basis: tuple[int, ...],
    *,
    row_shift: int,
    column_shift: int,
) -> np.ndarray:
    positions = {configuration: index for index, configuration in enumerate(basis)}
    translated = np.zeros_like(state)
    for old_position, configuration in enumerate(basis):
        new_configuration = 0
        for site in range(12):
            bit = (configuration >> (11 - site)) & 1
            row, column = divmod(site, 6)
            new_site = ((row + row_shift) % 2) * 6 + (column + column_shift) % 6
            new_configuration |= bit << (11 - new_site)
        translated[positions[new_configuration]] = state[old_position]
    return translated


def _distinct_translation_orbit(
    base_state: np.ndarray,
    basis: tuple[int, ...],
    *,
    row_shifts: tuple[int, ...] = (0, 1),
) -> tuple[np.ndarray, ...]:
    distinct: list[np.ndarray] = []
    for row_shift in row_shifts:
        if row_shift not in {0, 1}:
            raise ValueError("row shifts on the 2x6 torus must be zero or one")
        for column_shift in range(6):
            candidate = _translated_state(
                base_state,
                basis,
                row_shift=row_shift,
                column_shift=column_shift,
            )
            duplicate = False
            for existing in distinct:
                overlap = np.vdot(existing, candidate)
                if abs(abs(overlap) - 1.0) <= 2.0e-12:
                    duplicate = True
                    break
            if not duplicate:
                distinct.append(candidate)
    return tuple(distinct)


def _covering_subspace_energy(
    hamiltonian: np.ndarray,
    orbit: tuple[np.ndarray, ...],
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return generalized and uniform energies of one nonorthogonal orbit."""

    columns = np.column_stack(orbit)
    overlap = columns.T @ columns
    projected_h = columns.T @ hamiltonian @ columns
    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    keep = overlap_values > 2.0e-11 * max(1.0, float(overlap_values[-1]))
    rank = int(np.sum(keep))
    orthonormalizer = (
        columns
        @ overlap_vectors[:, keep]
        @ np.diag(1.0 / np.sqrt(overlap_values[keep]))
    )
    reduced_h = orthonormalizer.T @ hamiltonian @ orthonormalizer
    generalized_ground = float(np.linalg.eigvalsh(reduced_h)[0])
    uniform = np.sum(columns, axis=1)
    uniform_norm = float(np.vdot(uniform, uniform).real)
    if uniform_norm <= 1.0e-14:
        raise RuntimeError("the uniform covering superposition vanished")
    uniform_energy = float(uniform @ (hamiltonian @ uniform) / uniform_norm)
    return (
        generalized_ground,
        rank,
        overlap,
        projected_h,
        overlap_values,
        uniform_energy,
    )


@lru_cache(maxsize=None)
def _covering_resonance_cached(
    geometry: Geometry, j2_over_j1: float
) -> CoveringResonanceReport:
    geometry = _checked_geometry(geometry)
    ratio = _checked_ratio(j2_over_j1)
    reference = triangle_patch_cluster_reference(geometry, ratio)
    periodic_bonds = _geometry_bonds(
        geometry, 2, 6, j1=1.0, j2=ratio, periodic=True
    )
    hamiltonian, basis = heisenberg_sz_matrix(
        12, periodic_bonds, n_up=6
    )
    base = _cluster_product_state(reference.computational_state, basis)
    orbit = _distinct_translation_orbit(base, basis)
    (
        generalized_ground,
        rank,
        overlap,
        projected_h,
        overlap_values,
        uniform_energy,
    ) = _covering_subspace_energy(hamiltonian, orbit)
    column_orbit = _distinct_translation_orbit(
        base, basis, row_shifts=(0,)
    )
    (
        column_ground,
        column_rank,
        _,
        _,
        _,
        _,
    ) = _covering_subspace_energy(hamiltonian, column_orbit)
    single = float(base @ (hamiltonian @ base))
    exact_ground = float(np.linalg.eigvalsh(hamiltonian)[0])
    gap = single - exact_ground
    recovered = (
        (single - generalized_ground) / gap if gap > 1.0e-13 else float("nan")
    )
    for array in (overlap, projected_h, overlap_values):
        array.setflags(write=False)
    return CoveringResonanceReport(
        geometry=geometry,
        j2_over_j1=ratio,
        boundary_convention=(
            "periodic 2x6 thin torus; distinct wrapped images joining the "
            "same site pair are combined by summing their couplings"
        ),
        raw_translation_count=12,
        distinct_covering_count=len(orbit),
        subspace_rank=rank,
        overlap_matrix=overlap,
        projected_hamiltonian=projected_h,
        overlap_eigenvalues=overlap_values,
        single_covering_energy=single,
        uniform_coherent_energy=uniform_energy,
        generalized_ground_energy=generalized_ground,
        resonance_energy_gain=single - generalized_ground,
        exact_ground_energy=exact_ground,
        recovered_exact_gap_fraction=float(recovered),
        column_only_covering_count=len(column_orbit),
        column_only_subspace_rank=column_rank,
        column_only_ground_energy=column_ground,
        column_only_resonance_energy_gain=single - column_ground,
        column_only_recovered_exact_gap_fraction=(
            (single - column_ground) / gap if gap > 1.0e-13 else float("nan")
        ),
    )


def translated_covering_resonance_report(
    geometry: Geometry, j2_over_j1: float
) -> CoveringResonanceReport:
    """Return the periodic ``2 x 6`` translated-covering resonance report."""

    return _covering_resonance_cached(
        _checked_geometry(geometry), _checked_ratio(j2_over_j1)
    )


__all__ = [
    "ClusterQ2ResponseBasis",
    "CoveringResonanceReport",
    "FullCouplingCarrierResult",
    "InterclusterBond",
    "TwoClusterOpenPartition",
    "TwoClusterVirtualProcessReport",
    "carrier_subspace_e2_fractions",
    "cluster_q2_response_basis",
    "translated_covering_resonance_report",
    "two_cluster_open_partition",
    "two_cluster_virtual_process_report",
]
