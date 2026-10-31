"""Exact support/orientation audit for six-site ``q=2`` carrier banks.

The production ``6 x 6`` triangle-patch model starts from fixed-orientation
``2 x 3`` blocks and restores translations.  This module asks a deliberately
smaller, exact question: how many *local* six-site triplet carriers are needed
to represent the resolvent-dressed response to every boundary support seen by
that translated tiling?

No 36-site state is diagonalized.  The only eigensystem is the open six-site
cluster.  Periodic ``6 x 6`` bonds are used as a ledger that enumerates
interfaces.  The 36 translation-group elements contain only six distinct
block partitions (a six-element stabilizer permutes the blocks), and the
report keeps both counts explicit.

For a directed interface coupling ``J[i,j]`` and
``Z[a,i] = <t_a,m=0|S_i^z|g>``, the exact singlet-pair source and first-order
correction are

``A = sqrt(3) Z J Z.T`` and ``X_ab = A_ab/(Delta_a+Delta_b)``.

The left and right singular vectors of ``X`` are generally different.  A
shared orientation-agnostic local bank is therefore obtained from the leading
eigenvectors of ``sum_w (X X.T + X.T X)`` over all interface occurrences.
The module reports both direction-resolved supports and the physical
pair-total interactions obtained after summing all bonds between two blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.triangular_j1j2 import triangular_j1j2_bonds
from .two_cluster_virtual_processes import (
    carrier_subspace_e2_fractions,
    cluster_q2_response_basis,
)


Geometry = Literal["square", "triangular"]
InterfaceFamily = Literal["direction_resolved", "physical_pair_total"]

_ROWS = 6
_COLUMNS = 6
_CLUSTER_ROWS = 2
_CLUSTER_COLUMNS = 3


@dataclass(frozen=True)
class BoundaryDirectionClass:
    """One positive displacement representative of a bond shell."""

    label: str
    shell: Literal["J1", "J2"]
    displacement: tuple[int, int]
    coupling: float
    periodic_bond_count: int


@dataclass(frozen=True)
class BoundaryInterfaceResponse:
    """Exact six-site response to one deduplicated boundary support."""

    interface_id: str
    direction_labels: tuple[str, ...]
    occurrence_count: int
    bond_count_per_occurrence: int
    coupling_matrix: np.ndarray
    response_amplitude_matrix: np.ndarray
    action_norm_squared: float
    second_order_energy: float
    correction_amplitude_matrix: np.ndarray
    correction_singular_values: np.ndarray
    correction_schmidt_rank: int
    correction_left_modes: np.ndarray
    correction_right_modes: np.ndarray
    leading_left_right_overlap: float
    route_conditioned_e2_fractions: np.ndarray


@dataclass(frozen=True)
class CarrierUnionAudit:
    """Compression of several interface-specific left/right response spaces.

    Threshold carrier counts refer to prefixes of the reported joint-covariance
    ordering.  They are not claims of a globally optimized minimum over the
    Grassmannian of all shared local subspaces.
    """

    family: InterfaceFamily
    unique_interface_count: int
    weighted_occurrence_count: int
    mean_action_norm_squared_per_distinct_tiling: float
    mean_second_order_energy_per_distinct_tiling: float
    union_singular_values: np.ndarray
    union_rank: int
    union_cumulative_weight_fractions: np.ndarray
    route_k_union_ranks: np.ndarray
    route_conditioned_e2_fractions: np.ndarray
    shared_bank_vectors: np.ndarray
    shared_bank_correction_norm_fractions: np.ndarray
    shared_bank_e2_fractions: np.ndarray
    minimum_shared_carriers_for_90pct_e2: int
    minimum_shared_carriers_for_99pct_e2: int
    minimum_shared_carriers_for_99p9pct_e2: int


@dataclass(frozen=True)
class SupportOrientationCarrierReport:
    """Translation-orbit interface ledger and exact local carrier audit."""

    geometry: Geometry
    j2_over_j1: float
    lattice_shape: tuple[int, int]
    cluster_shape: tuple[int, int]
    translation_group_order: int
    distinct_tiling_count: int
    tiling_stabilizer_order: int
    representative_translation_shifts: tuple[tuple[int, int], ...]
    direction_classes: tuple[BoundaryDirectionClass, ...]
    periodic_bond_reconstruction_error: float
    direction_resolved_interfaces: tuple[BoundaryInterfaceResponse, ...]
    physical_pair_total_interfaces: tuple[BoundaryInterfaceResponse, ...]
    direction_resolved_union: CarrierUnionAudit
    physical_pair_total_union: CarrierUnionAudit
    claim_boundary: str


@dataclass
class _InterfaceAccumulator:
    direction_labels: tuple[str, ...]
    coupling_matrix: np.ndarray
    bond_count: int
    occurrence_count: int = 1


def _checked_geometry(geometry: str) -> Geometry:
    if geometry not in {"square", "triangular"}:
        raise ValueError("geometry must be 'square' or 'triangular'")
    return geometry


def _checked_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("j2_over_j1 must be finite and nonnegative")
    return ratio


def _direction_specs(
    geometry: Geometry, ratio: float
) -> tuple[tuple[str, Literal["J1", "J2"], tuple[int, int], float], ...]:
    if geometry == "square":
        return (
            ("J1_horizontal_dc", "J1", (0, 1), 1.0),
            ("J1_vertical_dr", "J1", (1, 0), 1.0),
            ("J2_diagonal_dr_plus_dc", "J2", (1, 1), ratio),
            ("J2_diagonal_dr_minus_dc", "J2", (1, -1), ratio),
        )
    return (
        ("J1_a_dr", "J1", (1, 0), 1.0),
        ("J1_b_dc", "J1", (0, 1), 1.0),
        ("J1_c_dr_plus_dc", "J1", (1, 1), 1.0),
        ("J2_dr_minus_dc", "J2", (1, -1), ratio),
        ("J2_dr_plus_2dc", "J2", (1, 2), ratio),
        ("J2_2dr_plus_dc", "J2", (2, 1), ratio),
    )


def _directional_periodic_bonds(
    geometry: Geometry, ratio: float
) -> tuple[
    tuple[tuple[int, int, str, float], ...],
    tuple[BoundaryDirectionClass, ...],
    float,
]:
    records: list[tuple[int, int, str, float]] = []
    classes: list[BoundaryDirectionClass] = []
    for label, shell, (dr, dc), coupling in _direction_specs(geometry, ratio):
        count = 0
        if coupling != 0.0:
            for row in range(_ROWS):
                for column in range(_COLUMNS):
                    target_row = (row + dr) % _ROWS
                    target_column = (column + dc) % _COLUMNS
                    first = row * _COLUMNS + column
                    second = target_row * _COLUMNS + target_column
                    if first == second:
                        raise RuntimeError("a registered direction produced a self bond")
                    records.append((first, second, label, float(coupling)))
                    count += 1
        classes.append(
            BoundaryDirectionClass(
                label=label,
                shell=shell,
                displacement=(dr, dc),
                coupling=float(coupling),
                periodic_bond_count=count,
            )
        )

    observed: dict[tuple[int, int], float] = {}
    for first, second, _, coupling in records:
        pair = tuple(sorted((first, second)))
        observed[pair] = observed.get(pair, 0.0) + coupling
    generator = (
        square_j1j2_bonds if geometry == "square" else triangular_j1j2_bonds
    )
    expected = {
        (first, second): coupling
        for first, second, coupling in generator(
            _ROWS,
            _COLUMNS,
            j1=1.0,
            j2=ratio,
            periodic=True,
        )
    }
    pairs = set(observed) | set(expected)
    error = max(
        (
            abs(observed.get(pair, 0.0) - expected.get(pair, 0.0))
            for pair in pairs
        ),
        default=0.0,
    )
    if error > 1.0e-14:
        raise RuntimeError("direction classes do not reconstruct the periodic graph")
    return tuple(records), tuple(classes), float(error)


def _tiling(
    row_shift: int, column_shift: int
) -> tuple[tuple[tuple[int, ...], ...], dict[int, tuple[int, int]]]:
    clusters: list[tuple[int, ...]] = []
    site_map: dict[int, tuple[int, int]] = {}
    for block_row in range(_ROWS // _CLUSTER_ROWS):
        for block_column in range(_COLUMNS // _CLUSTER_COLUMNS):
            sites = []
            cluster_index = len(clusters)
            for local_row in range(_CLUSTER_ROWS):
                for local_column in range(_CLUSTER_COLUMNS):
                    row = (
                        row_shift + block_row * _CLUSTER_ROWS + local_row
                    ) % _ROWS
                    column = (
                        column_shift
                        + block_column * _CLUSTER_COLUMNS
                        + local_column
                    ) % _COLUMNS
                    site = row * _COLUMNS + column
                    local_index = local_row * _CLUSTER_COLUMNS + local_column
                    if site in site_map:
                        raise RuntimeError("translated clusters overlap")
                    site_map[site] = (cluster_index, local_index)
                    sites.append(site)
            clusters.append(tuple(sites))
    if len(site_map) != _ROWS * _COLUMNS:
        raise RuntimeError("translated clusters do not tile the lattice")
    return tuple(clusters), site_map


def _distinct_tilings() -> tuple[
    tuple[tuple[int, int], ...],
    tuple[dict[int, tuple[int, int]], ...],
    int,
]:
    representatives: dict[
        tuple[tuple[int, ...], ...], tuple[tuple[int, int], dict[int, tuple[int, int]]]
    ] = {}
    for row_shift in range(_ROWS):
        for column_shift in range(_COLUMNS):
            clusters, site_map = _tiling(row_shift, column_shift)
            signature = tuple(sorted(tuple(sorted(cluster)) for cluster in clusters))
            representatives.setdefault(
                signature, ((row_shift, column_shift), site_map)
            )
    shifts = tuple(value[0] for value in representatives.values())
    maps = tuple(value[1] for value in representatives.values())
    group_order = _ROWS * _COLUMNS
    if group_order % len(maps) != 0:
        raise RuntimeError("the tiling orbit does not divide the translation group")
    return shifts, maps, group_order // len(maps)


def _matrix_key(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix, dtype=np.float64).tobytes()


def _canonical_unordered_matrix(matrix: np.ndarray) -> np.ndarray:
    direct = np.ascontiguousarray(matrix, dtype=np.float64)
    transposed = np.ascontiguousarray(matrix.T, dtype=np.float64)
    return direct if direct.tobytes() <= transposed.tobytes() else transposed


def _register_interface(
    registry: dict[tuple[tuple[str, ...], bytes], _InterfaceAccumulator],
    *,
    direction_labels: tuple[str, ...],
    matrix: np.ndarray,
    bond_count: int,
) -> None:
    key = (direction_labels, _matrix_key(matrix))
    existing = registry.get(key)
    if existing is None:
        registry[key] = _InterfaceAccumulator(
            direction_labels=direction_labels,
            coupling_matrix=np.asarray(matrix, dtype=np.float64),
            bond_count=bond_count,
        )
        return
    if existing.bond_count != bond_count:
        raise RuntimeError("a deduplicated interface changed its bond count")
    existing.occurrence_count += 1


def _interface_ledgers(
    directional_bonds: tuple[tuple[int, int, str, float], ...],
    tiling_maps: tuple[dict[int, tuple[int, int]], ...],
) -> tuple[tuple[_InterfaceAccumulator, ...], tuple[_InterfaceAccumulator, ...]]:
    direction_registry: dict[
        tuple[tuple[str, ...], bytes], _InterfaceAccumulator
    ] = {}
    pair_registry: dict[
        tuple[tuple[str, ...], bytes], _InterfaceAccumulator
    ] = {}
    for site_map in tiling_maps:
        direction_groups: dict[
            tuple[int, int, str], tuple[np.ndarray, int]
        ] = {}
        pair_groups: dict[
            tuple[int, int], tuple[np.ndarray, int, set[str]]
        ] = {}
        for first, second, label, coupling in directional_bonds:
            first_cluster, first_local = site_map[first]
            second_cluster, second_local = site_map[second]
            if first_cluster == second_cluster:
                continue

            direction_key = (first_cluster, second_cluster, label)
            if direction_key not in direction_groups:
                direction_groups[direction_key] = (np.zeros((6, 6)), 0)
            direction_matrix, direction_count = direction_groups[direction_key]
            direction_matrix[first_local, second_local] += coupling
            direction_groups[direction_key] = (
                direction_matrix,
                direction_count + 1,
            )

            lower, upper = sorted((first_cluster, second_cluster))
            pair_key = (lower, upper)
            if pair_key not in pair_groups:
                pair_groups[pair_key] = (np.zeros((6, 6)), 0, set())
            pair_matrix, pair_count, labels = pair_groups[pair_key]
            if first_cluster == lower:
                pair_matrix[first_local, second_local] += coupling
            else:
                pair_matrix[second_local, first_local] += coupling
            labels.add(label)
            pair_groups[pair_key] = (pair_matrix, pair_count + 1, labels)

        for (_, _, label), (matrix, count) in direction_groups.items():
            _register_interface(
                direction_registry,
                direction_labels=(label,),
                matrix=matrix,
                bond_count=count,
            )
        for matrix, count, labels in pair_groups.values():
            canonical = _canonical_unordered_matrix(matrix)
            _register_interface(
                pair_registry,
                direction_labels=tuple(sorted(labels)),
                matrix=canonical,
                bond_count=count,
            )

    def ordered(
        registry: dict[tuple[tuple[str, ...], bytes], _InterfaceAccumulator]
    ) -> tuple[_InterfaceAccumulator, ...]:
        return tuple(
            sorted(
                registry.values(),
                key=lambda row: (
                    row.direction_labels,
                    row.bond_count,
                    _matrix_key(row.coupling_matrix),
                ),
            )
        )

    return ordered(direction_registry), ordered(pair_registry)


def _interface_response(
    geometry: Geometry,
    ratio: float,
    family: InterfaceFamily,
    index: int,
    accumulator: _InterfaceAccumulator,
) -> BoundaryInterfaceResponse:
    basis = cluster_q2_response_basis(geometry, ratio)
    transitions = basis.sz_transition_matrix
    response = np.asarray(
        np.sqrt(3.0)
        * transitions
        @ accumulator.coupling_matrix
        @ transitions.T,
        dtype=np.float64,
    )
    denominators = basis.triplet_gaps[:, None] + basis.triplet_gaps[None, :]
    correction = np.asarray(response / denominators, dtype=np.float64)
    left, singular_values, right_adjoint = np.linalg.svd(
        correction, full_matrices=False
    )
    right = np.asarray(right_adjoint.T, dtype=np.float64)
    threshold = 2.0e-12 * max(1.0, float(singular_values[0]))
    rank = int(np.sum(singular_values > threshold))
    lowering = float(np.sum(np.square(response) / denominators))
    route_fractions = carrier_subspace_e2_fractions(
        response, denominators, left, right
    )
    for array in (
        accumulator.coupling_matrix,
        response,
        correction,
        singular_values,
        left,
        right,
        route_fractions,
    ):
        array.setflags(write=False)
    prefix = "dir" if family == "direction_resolved" else "pair"
    return BoundaryInterfaceResponse(
        interface_id=f"{prefix}_{index:02d}",
        direction_labels=accumulator.direction_labels,
        occurrence_count=accumulator.occurrence_count,
        bond_count_per_occurrence=accumulator.bond_count,
        coupling_matrix=accumulator.coupling_matrix,
        response_amplitude_matrix=response,
        action_norm_squared=float(np.sum(np.square(response))),
        second_order_energy=-lowering,
        correction_amplitude_matrix=correction,
        correction_singular_values=singular_values,
        correction_schmidt_rank=rank,
        correction_left_modes=left,
        correction_right_modes=right,
        leading_left_right_overlap=float(abs(np.vdot(left[:, 0], right[:, 0])) ** 2),
        route_conditioned_e2_fractions=route_fractions,
    )


def _numerical_rank(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if len(singular) == 0:
        return 0
    return int(np.sum(singular > 2.0e-12 * max(1.0, float(singular[0]))))


def _minimum_count(curve: np.ndarray, threshold: float) -> int:
    passing = np.flatnonzero(curve >= threshold - 2.0e-13)
    return int(passing[0] + 1) if len(passing) else int(len(curve))


def _union_audit(
    geometry: Geometry,
    ratio: float,
    family: InterfaceFamily,
    interfaces: tuple[BoundaryInterfaceResponse, ...],
    distinct_tiling_count: int,
) -> CarrierUnionAudit:
    if not interfaces:
        raise RuntimeError("the translated tiling has no intercluster interfaces")
    weighted_blocks = []
    for interface in interfaces:
        scale = math.sqrt(interface.occurrence_count)
        weighted_blocks.extend(
            (
                scale * interface.correction_amplitude_matrix,
                scale * interface.correction_amplitude_matrix.T,
            )
        )
    stacked = np.concatenate(weighted_blocks, axis=1)
    common_vectors, union_singular, _ = np.linalg.svd(
        stacked, full_matrices=False
    )
    threshold = 2.0e-12 * max(1.0, float(union_singular[0]))
    union_rank = int(np.sum(union_singular > threshold))
    cumulative = np.cumsum(np.square(union_singular)) / float(
        np.sum(np.square(union_singular))
    )

    route_union_ranks = []
    for count in range(1, 10):
        columns = []
        for interface in interfaces:
            retained = min(count, interface.correction_schmidt_rank)
            columns.extend(
                (
                    interface.correction_left_modes[:, :retained],
                    interface.correction_right_modes[:, :retained],
                )
            )
        route_union_ranks.append(_numerical_rank(np.concatenate(columns, axis=1)))

    exact_lowerings = np.asarray(
        [
            -interface.second_order_energy * interface.occurrence_count
            for interface in interfaces
        ],
        dtype=np.float64,
    )
    total_lowering = float(np.sum(exact_lowerings))
    weighted_action_norm = sum(
        interface.occurrence_count * interface.action_norm_squared
        for interface in interfaces
    )
    route_e2 = sum(
        weight * interface.route_conditioned_e2_fractions
        for weight, interface in zip(exact_lowerings, interfaces, strict=True)
    ) / total_lowering

    basis = cluster_q2_response_basis(geometry, ratio)
    denominators = basis.triplet_gaps[:, None] + basis.triplet_gaps[None, :]
    shared_e2 = np.zeros(9, dtype=np.float64)
    shared_norm = np.zeros(9, dtype=np.float64)
    total_correction_norm = sum(
        interface.occurrence_count
        * float(np.sum(np.square(interface.correction_amplitude_matrix)))
        for interface in interfaces
    )
    for interface, exact_weight in zip(
        interfaces, exact_lowerings, strict=True
    ):
        fractions = carrier_subspace_e2_fractions(
            interface.response_amplitude_matrix,
            denominators,
            common_vectors,
            common_vectors,
        )
        shared_e2 += exact_weight * fractions
        for count in range(1, 10):
            vectors = common_vectors[:, :count]
            projected = (
                vectors.T
                @ interface.correction_amplitude_matrix
                @ vectors
            )
            shared_norm[count - 1] += (
                interface.occurrence_count
                * float(np.sum(np.square(projected)))
            )
    shared_e2 /= total_lowering
    shared_norm /= total_correction_norm

    for array in (
        union_singular,
        cumulative,
        common_vectors,
        shared_e2,
        shared_norm,
    ):
        array.setflags(write=False)
    ranks = np.asarray(route_union_ranks, dtype=np.int32)
    ranks.setflags(write=False)
    route_e2 = np.asarray(route_e2, dtype=np.float64)
    route_e2.setflags(write=False)
    return CarrierUnionAudit(
        family=family,
        unique_interface_count=len(interfaces),
        weighted_occurrence_count=sum(
            interface.occurrence_count for interface in interfaces
        ),
        mean_action_norm_squared_per_distinct_tiling=(
            weighted_action_norm / distinct_tiling_count
        ),
        mean_second_order_energy_per_distinct_tiling=(
            -total_lowering / distinct_tiling_count
        ),
        union_singular_values=union_singular,
        union_rank=union_rank,
        union_cumulative_weight_fractions=cumulative,
        route_k_union_ranks=ranks,
        route_conditioned_e2_fractions=route_e2,
        shared_bank_vectors=common_vectors,
        shared_bank_correction_norm_fractions=shared_norm,
        shared_bank_e2_fractions=shared_e2,
        minimum_shared_carriers_for_90pct_e2=_minimum_count(shared_e2, 0.90),
        minimum_shared_carriers_for_99pct_e2=_minimum_count(shared_e2, 0.99),
        minimum_shared_carriers_for_99p9pct_e2=_minimum_count(shared_e2, 0.999),
    )


@lru_cache(maxsize=None)
def _support_orientation_carrier_report_cached(
    geometry: Geometry, ratio: float
) -> SupportOrientationCarrierReport:
    directional_bonds, direction_classes, reconstruction_error = (
        _directional_periodic_bonds(geometry, ratio)
    )
    shifts, maps, stabilizer = _distinct_tilings()
    if len(maps) != _CLUSTER_ROWS * _CLUSTER_COLUMNS:
        raise RuntimeError("the fixed 2x3 tiling must have six translated partitions")
    direction_rows, pair_rows = _interface_ledgers(directional_bonds, maps)
    direction_interfaces = tuple(
        _interface_response(
            geometry, ratio, "direction_resolved", index, accumulator
        )
        for index, accumulator in enumerate(direction_rows)
    )
    pair_interfaces = tuple(
        _interface_response(
            geometry, ratio, "physical_pair_total", index, accumulator
        )
        for index, accumulator in enumerate(pair_rows)
    )
    direction_union = _union_audit(
        geometry,
        ratio,
        "direction_resolved",
        direction_interfaces,
        len(maps),
    )
    pair_union = _union_audit(
        geometry,
        ratio,
        "physical_pair_total",
        pair_interfaces,
        len(maps),
    )
    return SupportOrientationCarrierReport(
        geometry=geometry,
        j2_over_j1=ratio,
        lattice_shape=(_ROWS, _COLUMNS),
        cluster_shape=(_CLUSTER_ROWS, _CLUSTER_COLUMNS),
        translation_group_order=_ROWS * _COLUMNS,
        distinct_tiling_count=len(maps),
        tiling_stabilizer_order=stabilizer,
        representative_translation_shifts=shifts,
        direction_classes=direction_classes,
        periodic_bond_reconstruction_error=reconstruction_error,
        direction_resolved_interfaces=direction_interfaces,
        physical_pair_total_interfaces=pair_interfaces,
        direction_resolved_union=direction_union,
        physical_pair_total_union=pair_union,
        claim_boundary=(
            "Exact six-site q=2 response audit over the six distinct translations "
            "of the fixed-orientation 2x3 tiling on a periodic 6x6 bond ledger. "
            "It does not include 3x2 rotated/reflected cluster coverings, point-"
            "group projection, a trained N=36 state, or N=36 exact diagonalization."
        ),
    )


def support_orientation_carrier_report(
    geometry: Geometry, j2_over_j1: float
) -> SupportOrientationCarrierReport:
    """Return the exact translated-interface carrier audit."""

    return _support_orientation_carrier_report_cached(
        _checked_geometry(geometry), _checked_ratio(j2_over_j1)
    )


__all__ = [
    "BoundaryDirectionClass",
    "BoundaryInterfaceResponse",
    "CarrierUnionAudit",
    "SupportOrientationCarrierReport",
    "support_orientation_carrier_report",
]
