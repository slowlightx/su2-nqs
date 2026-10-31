"""Fail-closed numerical and checkpoint helpers for the D1-T C0 teacher.

The production Hamiltonian action is supplied by the pinned XDiag executable.
This module contains no N=36 execution entry point.  Its dense-array routines
are independent exact-small/synthetic references for the registered
incremental rank-three block-Krylov algorithm and its same-span null control.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np


ArrayAction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BlockLowdinResult:
    """A block increment and the spectral diagnostics used to accept it."""

    vectors: np.ndarray
    projected_gram_eigenvalues: np.ndarray
    unprojected_largest_gram_eigenvalue: float
    retained_rank: int
    orthonormality_error: float


@dataclass(frozen=True)
class TeacherArmDepth:
    """Basis-invariant diagnostics for one arm at one completed depth."""

    depth: int
    accepted_dimension: int
    frontier_rank: int
    energies: np.ndarray
    direct_residual_norms: np.ndarray
    residual_intervals: np.ndarray
    seed_subspace_weights: np.ndarray
    carrier_overlap_squares: np.ndarray
    selected_adjacent_gaps: np.ndarray
    projected_cluster_ids: np.ndarray
    projected_root_boundary_gap: float | None
    orthonormality_error: float
    projected_hamiltonian_antihermitian_error: float
    projected_hamiltonian_relative_antihermitian_error: float
    ritz_vectors: np.ndarray
    ritz_coefficients: np.ndarray


@dataclass(frozen=True)
class ProjectedClusterPartition:
    """An independently assigned partition of the sorted selected Ritz roots."""

    adjacent_gaps: np.ndarray
    cluster_ids: np.ndarray
    clusters: tuple[tuple[int, ...], ...]
    root_boundary_gap: float | None


@dataclass(frozen=True)
class ClusterInvariant:
    """Rotation-invariant data for one complete selected Ritz cluster."""

    indices: tuple[int, ...]
    projector_basis: np.ndarray
    residual_singular_values: np.ndarray
    residual_frobenius_norm: float
    residual_operator_norm: float
    seed_projector_eigenvalues: np.ndarray
    seed_projector_trace: float
    fixed_physical_carrier_weights: np.ndarray
    splus_singular_values: np.ndarray | None
    splus_operator_norm: float | None
    raw_label: str


@dataclass(frozen=True)
class ClusterComparison:
    """Same-span discrepancies with no within-cluster vector matching."""

    projector_operator_norm_distance: float
    residual_spectrum_max_difference: float
    residual_frobenius_norm_difference: float
    residual_operator_norm_difference: float
    seed_spectrum_max_difference: float
    seed_trace_difference: float
    carrier_weight_max_difference: float
    splus_spectrum_max_difference: float | None
    splus_operator_norm_difference: float | None


@dataclass(frozen=True)
class TeacherDepth:
    """Physical/null diagnostics and exact work accounting at one depth."""

    depth: int
    physical: TeacherArmDepth
    null: TeacherArmDepth
    common_projected_clusters: tuple[tuple[int, ...], ...]
    physical_cluster_invariants: tuple[ClusterInvariant, ...]
    null_cluster_invariants: tuple[ClusterInvariant, ...]
    cluster_comparisons: tuple[ClusterComparison, ...]
    full_span_projector_frobenius_error: float
    ritz_subspace_projector_frobenius_error: float
    physical_splus_operator_norm: float | None
    null_splus_operator_norm: float | None
    physical_splus_ritz_norms: np.ndarray | None
    null_splus_ritz_norms: np.ndarray | None
    physical_batched_h_actions: int
    null_batched_h_actions: int
    total_batched_h_actions: int
    physical_scalar_equivalent_h_mvm: int
    null_scalar_equivalent_h_mvm: int
    total_scalar_equivalent_h_mvm: int


@dataclass(frozen=True)
class TeacherTrajectory:
    """The complete bounded C0 trajectory and cached physical/null states."""

    depths: tuple[TeacherDepth, ...]
    physical_q: np.ndarray
    physical_hq: np.ndarray
    physical_frontier: np.ndarray
    physical_h_frontier: np.ndarray
    null_q: np.ndarray
    null_hq: np.ndarray
    null_frontier: np.ndarray
    null_h_frontier: np.ndarray
    physical_q0: np.ndarray
    normalized_carriers: np.ndarray


def validate_c0_scale_contract(config: Mapping[str, Any]) -> float:
    """Recompute and validate the C0 Hamiltonian scale before any action."""

    try:
        cell = config["cell"]
        j1_edges = int(cell["j1_edge_count"])
        j2_edges = int(cell["j2_edge_count"])
        j1 = float(cell["j1"])
        j2 = float(cell["j2_over_j1"])
        registered = float(cell["scale_exact"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("C0 scale fields are missing or malformed") from error
    if j1_edges != 108 or j2_edges != 108 or j1 != 1.0 or j2 != 0.125:
        raise ValueError("C0 cell edge/coupling identity changed")
    recomputed = 0.75 * (j1_edges * abs(j1) + j2_edges * abs(j2))
    if recomputed != 91.125 or registered != recomputed:
        raise ValueError(
            "C0 Hamiltonian scale mismatch; rejected before basis/action"
        )
    return recomputed


@dataclass(frozen=True)
class OwnershipAssessment:
    """Prospectively classified shallow three-root ownership outcome."""

    status: str
    lower_bounds: np.ndarray | None
    conservative_complement_gaps: np.ndarray | None
    reason: str
    evidence: str = "none"
    target_cluster_ids: np.ndarray | None = None
    excluded_interval_distances: tuple[np.ndarray, ...] | None = None
    residual_intervals: np.ndarray | None = None


@dataclass(frozen=True)
class PartitionedOwnershipAssessment:
    """Rotation-safe ownership for an ordered mixture of Ritz clusters."""

    status: str
    evaluation_depth: int
    conditional_on_registered_exact_window: bool
    cluster_records: tuple[dict[str, Any], ...]
    reason: str


@dataclass(frozen=True)
class ExactAnchorIntervals:
    """Two-seed residual-expanded singlet intervals and merged clusters."""

    centers: np.ndarray
    half_widths: np.ndarray
    intervals: np.ndarray
    cluster_ids: np.ndarray
    complete_through_fourth: bool


def construct_exact_anchor_intervals(
    seed_energies: np.ndarray,
    seed_direct_residuals: np.ndarray,
    consensus_centers: np.ndarray,
    *,
    matching_tolerance: float,
    degeneracy_tolerance: float = 1.0e-10,
    complete_through_fourth: bool,
) -> ExactAnchorIntervals:
    """Expand two-seed anchor values before any ownership calculation.

    The half width is

    ``max_seed(|E_seed-center| + rho_seed) + matching_tolerance``.

    Thus the fourth-singlet lower boundary cannot silently omit its own
    residual, reseed spread, or the registered M=0/M=1 matching tolerance.
    Adjacent expanded intervals separated by at most ``degeneracy_tolerance``
    are merged into one complete cluster.
    """

    values = np.asarray(seed_energies, dtype=np.float64)
    residuals = np.asarray(seed_direct_residuals, dtype=np.float64)
    centers = np.asarray(consensus_centers, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] < 1:
        raise ValueError("seed energies must have shape (2, number_of_singlets)")
    if residuals.shape != values.shape or centers.shape != (values.shape[1],):
        raise ValueError("anchor energy, residual and center shapes disagree")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(residuals)) or np.any(~np.isfinite(centers)):
        raise FloatingPointError("anchor inputs must be finite")
    if np.any(residuals < 0.0):
        raise ValueError("anchor direct residuals must be nonnegative")
    if (
        not np.isfinite(matching_tolerance)
        or matching_tolerance < 0.0
        or not np.isfinite(degeneracy_tolerance)
        or degeneracy_tolerance < 0.0
    ):
        raise ValueError("anchor matching/degeneracy tolerance is invalid")
    if not isinstance(complete_through_fourth, (bool, np.bool_)):
        raise ValueError("complete_through_fourth must be an explicit boolean")

    half_widths = np.max(np.abs(values - centers[None, :]) + residuals, axis=0)
    half_widths += matching_tolerance
    intervals = np.column_stack((centers - half_widths, centers + half_widths))
    if np.any(np.diff(centers) < 0.0):
        raise ValueError("consensus singlets must be supplied in spectral order")

    cluster_ids = np.empty(len(centers), dtype=np.int64)
    cluster = 0
    cluster_ids[0] = cluster
    cluster_upper = float(intervals[0, 1])
    for index in range(1, len(centers)):
        if float(intervals[index, 0]) - cluster_upper > degeneracy_tolerance:
            cluster += 1
        cluster_ids[index] = cluster
        cluster_upper = max(cluster_upper, float(intervals[index, 1]))
    return ExactAnchorIntervals(
        centers=np.asarray(centers, dtype=np.float64),
        half_widths=np.asarray(half_widths, dtype=np.float64),
        intervals=np.asarray(intervals, dtype=np.float64),
        cluster_ids=cluster_ids,
        complete_through_fourth=bool(complete_through_fourth),
    )


def assess_three_root_ownership(
    ritz_energies: np.ndarray,
    direct_residual_norms: np.ndarray,
    per_ritz_splus_norms: np.ndarray,
    exact_singlet_intervals: np.ndarray,
    *,
    exact_cluster_ids: np.ndarray,
    target_cluster_ids: np.ndarray,
    anchor_complete_through_fourth: bool,
    direct_target_weights: np.ndarray | None = None,
    minimum_owned_weight: float = 0.9,
    evaluation_depth: int = 2,
) -> OwnershipAssessment:
    """Assess ownership against explicit exact-singlet intervals/clusters.

    ``exact_singlet_intervals`` has one ``[lower, upper]`` row for every
    certified singlet, in spectral order.  Production intervals are unions of
    the two-seed ``energy +/- direct residual`` intervals (and any registered
    classification tolerance); a consensus energy is never treated as an
    exact point.  ``exact_cluster_ids`` labels complete degenerate clusters,
    while the three ``target_cluster_ids`` prospectively assign each Ritz
    vector to one such cluster.

    The conservative complement gap is the minimum *interval distance* from
    ``[mu-rho, mu+rho]`` to every certified cluster outside the assigned
    target.  This includes lower excluded roots, not only the fourth level.
    A weak residual-gap lower bound is only ``NOT_CERTIFIED_BY_BOUND``: it is
    not evidence that the physical weight is small.  A physical shallow FAIL
    is returned only when an independently measured target-cluster overlap is
    supplied and is at most the frozen threshold.
    """

    if evaluation_depth in (0, 1):
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUATED_BY_DESIGN",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="formal N36 ownership is evaluated only at depth two",
        )
    if evaluation_depth != 2:
        raise ValueError("formal C0 ownership depth must be 0, 1, or 2")
    energies = np.asarray(ritz_energies, dtype=np.float64)
    residuals = np.asarray(direct_residual_norms, dtype=np.float64)
    splus = np.asarray(per_ritz_splus_norms, dtype=np.float64)
    intervals = np.asarray(exact_singlet_intervals, dtype=np.float64)
    cluster_ids = np.asarray(exact_cluster_ids)
    targets = np.asarray(target_cluster_ids)
    if energies.shape != (3,) or residuals.shape != (3,) or splus.shape != (3,):
        raise ValueError("ownership requires exactly three Ritz diagnostics")
    if np.any(~np.isfinite(energies)) or np.any(~np.isfinite(residuals)) or np.any(~np.isfinite(splus)):
        raise FloatingPointError("ownership diagnostics must be finite")
    if np.any(residuals < 0.0) or np.any(splus < 0.0):
        raise ValueError("residual and S+ norms must be nonnegative")
    if (
        intervals.ndim != 2
        or intervals.shape[1:] != (2,)
        or np.any(~np.isfinite(intervals))
        or np.any(intervals[:, 0] > intervals[:, 1])
    ):
        raise ValueError("exact singlet intervals must be finite [lower,upper] rows")
    if cluster_ids.ndim != 1 or cluster_ids.shape[0] != intervals.shape[0]:
        raise ValueError("exact_cluster_ids must label every singlet interval")
    if targets.shape != (3,):
        raise ValueError("target_cluster_ids must assign exactly three Ritz roots")
    if (
        cluster_ids.dtype.kind not in "iu"
        or targets.dtype.kind not in "iu"
        or np.any(cluster_ids < 0)
        or np.any(targets < 0)
    ):
        raise ValueError("cluster identifiers must be nonnegative integers")
    if not isinstance(anchor_complete_through_fourth, (bool, np.bool_)):
        raise ValueError("anchor completeness must be an explicit boolean")
    if not np.isfinite(minimum_owned_weight) or not 0.0 < minimum_owned_weight < 1.0:
        raise ValueError("minimum_owned_weight must lie in (0,1)")

    if not anchor_complete_through_fourth or intervals.shape[0] < 4:
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="a complete lowest-spectrum anchor through the fourth singlet is required",
        )

    # A cluster must occupy one contiguous spectral run.  Between different
    # clusters, certified intervals must be strictly ordered and disjoint.
    seen_clusters: set[int] = set()
    prior_cluster = int(cluster_ids[0])
    seen_clusters.add(prior_cluster)
    for index in range(1, len(cluster_ids)):
        cluster = int(cluster_ids[index])
        if cluster != prior_cluster:
            if cluster in seen_clusters:
                raise ValueError("one exact cluster appears in disjoint spectral runs")
            if intervals[index - 1, 1] >= intervals[index, 0]:
                return OwnershipAssessment(
                    status="OWNERSHIP_NOT_EVALUABLE",
                    lower_bounds=None,
                    conservative_complement_gaps=None,
                    reason="distinct exact-cluster intervals overlap or touch",
                )
            seen_clusters.add(cluster)
            prior_cluster = cluster

    if int(cluster_ids[2]) == int(cluster_ids[3]):
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="one complete exact cluster crosses the root-3/root-4 boundary",
        )

    registered_targets = cluster_ids[:3]
    if not np.array_equal(targets, registered_targets):
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="target assignment is not the registered lowest-three cluster ordering",
        )

    cluster_intervals: dict[int, np.ndarray] = {}
    for cluster in np.unique(cluster_ids):
        rows = intervals[cluster_ids == cluster]
        cluster_intervals[int(cluster)] = np.array(
            [np.min(rows[:, 0]), np.max(rows[:, 1])], dtype=np.float64
        )

    # Direct overlap is an independent vector/subspace measurement.  It needs
    # the same complete-cluster definition and prospective assignment, but it
    # must not be hidden behind a residual-gap failure: that certificate can
    # be inconclusive even when the measured overlap is near one.
    if direct_target_weights is not None:
        if len(np.unique(targets)) != len(targets):
            return OwnershipAssessment(
                status="OWNERSHIP_NOT_EVALUABLE",
                lower_bounds=None,
                conservative_complement_gaps=None,
                reason=(
                    "multiroot direct ownership requires a complete-cluster "
                    "principal-angle task, not per-column weights"
                ),
                target_cluster_ids=np.asarray(targets, dtype=np.int64),
            )
        direct = np.asarray(direct_target_weights, dtype=np.float64)
        if (
            direct.shape != (3,)
            or np.any(~np.isfinite(direct))
            or np.any(direct < 0.0)
            or np.any(direct > 1.0 + 1.0e-12)
        ):
            raise ValueError("direct target-cluster weights must be three finite probabilities")
        direct = np.minimum(direct, 1.0)
        if np.any(direct <= minimum_owned_weight):
            return OwnershipAssessment(
                status="DIRECT_OVERLAP_FAIL",
                lower_bounds=None,
                conservative_complement_gaps=None,
                reason="a direct target-cluster overlap is at or below the frozen threshold",
                evidence="direct_target_cluster_overlap",
                target_cluster_ids=np.asarray(targets, dtype=np.int64),
            )
        return OwnershipAssessment(
            status="DIRECT_OVERLAP_PASS",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="all direct target-cluster overlaps are strictly above threshold",
            evidence="direct_target_cluster_overlap",
            target_cluster_ids=np.asarray(targets, dtype=np.int64),
        )

    if len(np.unique(targets)) != len(targets):
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason=(
                "a selected target cluster contains multiple Ritz roots; "
                "individual residual bounds cannot label vectors inside it"
            ),
            target_cluster_ids=np.asarray(targets, dtype=np.int64),
        )

    ritz_intervals = np.stack((energies - residuals, energies + residuals), axis=1)
    gaps = np.empty(3, dtype=np.float64)
    all_excluded_distances: list[np.ndarray] = []
    for root, target in enumerate(targets):
        target_interval = cluster_intervals[int(target)]
        ritz_interval = ritz_intervals[root]
        # Unique assignment requires overlap with its target interval.  Merely
        # being nearest to a point estimate is not enough.
        target_overlap = min(ritz_interval[1], target_interval[1]) - max(
            ritz_interval[0], target_interval[0]
        )
        if target_overlap < 0.0:
            return OwnershipAssessment(
                status="OWNERSHIP_NOT_EVALUABLE",
                lower_bounds=None,
                conservative_complement_gaps=None,
                reason=f"Ritz root {root} is not interval-consistent with its target cluster",
            )
        excluded_distances: list[float] = []
        for cluster, exact_interval in cluster_intervals.items():
            if cluster == int(target):
                continue
            if ritz_interval[1] < exact_interval[0]:
                distance = float(exact_interval[0] - ritz_interval[1])
            elif exact_interval[1] < ritz_interval[0]:
                distance = float(ritz_interval[0] - exact_interval[1])
            else:
                distance = 0.0
            excluded_distances.append(distance)
        distances_array = np.asarray(excluded_distances, dtype=np.float64)
        all_excluded_distances.append(distances_array)
        gaps[root] = float(np.min(distances_array, initial=np.inf)) if distances_array.size else 0.0
        if not gaps[root] > 0.0:
            return OwnershipAssessment(
                status="OWNERSHIP_NOT_EVALUABLE",
                lower_bounds=None,
                conservative_complement_gaps=None,
                reason=f"Ritz root {root} overlaps an excluded exact cluster",
            )

    bounds = 1.0 - np.square(residuals / gaps) - 0.5 * np.square(splus)
    if np.all(bounds > minimum_owned_weight):
        return OwnershipAssessment(
            status="OWNERSHIP_BOUND_PASS",
            lower_bounds=np.asarray(bounds, dtype=np.float64),
            conservative_complement_gaps=np.asarray(gaps, dtype=np.float64),
            reason="all conservative residual-gap ownership bounds pass",
            evidence="residual_interval_gap_bound",
            target_cluster_ids=np.asarray(targets, dtype=np.int64),
            excluded_interval_distances=tuple(all_excluded_distances),
            residual_intervals=ritz_intervals,
        )
    return OwnershipAssessment(
        status="OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE",
        lower_bounds=np.asarray(bounds, dtype=np.float64),
        conservative_complement_gaps=np.asarray(gaps, dtype=np.float64),
        reason=(
            "the rigorous residual-gap bound is inconclusive; it is not a "
            "measurement of small target weight"
        ),
        evidence="residual_interval_gap_bound",
        target_cluster_ids=np.asarray(targets, dtype=np.int64),
        excluded_interval_distances=tuple(all_excluded_distances),
        residual_intervals=ritz_intervals,
    )


def assess_complete_cluster_ownership(
    projected_cluster_energies: np.ndarray,
    residual_block_operator_norm: float,
    splus_cluster_operator_norm: float,
    exact_cluster_intervals: np.ndarray,
    *,
    target_cluster_index: int,
    anchor_complete_through_fourth: bool,
    evaluation_depth: int,
    direct_minimum_principal_weight: float | None = None,
    minimum_owned_weight: float = 0.9,
) -> OwnershipAssessment:
    """Assess one complete multiroot cluster with rotation-invariant norms.

    The direct mode consumes the *minimum* squared principal-angle cosine.
    Otherwise the registered block-residual/Splus operator-norm lower bound is
    used.  Formal depths zero and one are unconditionally by-design labels.
    """

    if evaluation_depth in (0, 1):
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUATED_BY_DESIGN",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="formal N36 ownership is evaluated only at depth two",
        )
    if evaluation_depth != 2:
        raise ValueError("formal C0 ownership depth must be 0, 1, or 2")
    energies = np.asarray(projected_cluster_energies, dtype=np.float64)
    intervals = np.asarray(exact_cluster_intervals, dtype=np.float64)
    if energies.ndim != 1 or energies.size < 2 or np.any(~np.isfinite(energies)):
        raise ValueError("complete multiroot cluster energies are invalid")
    if np.any(np.diff(energies) < 0.0):
        raise ValueError("cluster energies must be supplied in sorted order")
    if (
        intervals.ndim != 2
        or intervals.shape[1:] != (2,)
        or intervals.shape[0] < 2
        or np.any(~np.isfinite(intervals))
        or np.any(intervals[:, 0] > intervals[:, 1])
    ):
        raise ValueError("complete exact-cluster intervals are invalid")
    if (
        isinstance(target_cluster_index, bool)
        or not isinstance(target_cluster_index, int)
        or not 0 <= target_cluster_index < intervals.shape[0]
    ):
        raise ValueError("target_cluster_index is invalid")
    if (
        not np.isfinite(residual_block_operator_norm)
        or residual_block_operator_norm < 0.0
        or not np.isfinite(splus_cluster_operator_norm)
        or splus_cluster_operator_norm < 0.0
        or not np.isfinite(minimum_owned_weight)
        or not 0.0 < minimum_owned_weight < 1.0
    ):
        raise ValueError("cluster ownership norms/threshold are invalid")
    if not anchor_complete_through_fourth:
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="registered first-four exact-cluster window is incomplete",
        )

    radius = float(residual_block_operator_norm)
    ritz_interval = np.asarray(
        [float(energies[0] - radius), float(energies[-1] + radius)],
        dtype=np.float64,
    )
    target = intervals[target_cluster_index]
    if min(ritz_interval[1], target[1]) < max(ritz_interval[0], target[0]):
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            lower_bounds=None,
            conservative_complement_gaps=None,
            reason="Ritz-cluster residual interval misses its assigned exact cluster",
            residual_intervals=ritz_interval[None, :],
        )

    distances: list[float] = []
    for index, excluded in enumerate(intervals):
        if index == target_cluster_index:
            continue
        distances.append(
            max(
                0.0,
                float(ritz_interval[0] - excluded[1]),
                float(excluded[0] - ritz_interval[1]),
            )
        )
    distance_array = np.asarray(distances, dtype=np.float64)

    if direct_minimum_principal_weight is not None:
        direct = float(direct_minimum_principal_weight)
        if not np.isfinite(direct) or direct < 0.0 or direct > 1.0 + 1.0e-12:
            raise ValueError("minimum principal-angle weight must be in [0,1]")
        status = (
            "DIRECT_OVERLAP_PASS"
            if direct > minimum_owned_weight
            else "DIRECT_OVERLAP_FAIL"
        )
        return OwnershipAssessment(
            status=status,
            lower_bounds=np.asarray([min(direct, 1.0)], dtype=np.float64),
            conservative_complement_gaps=None,
            reason="complete-cluster minimum principal-angle weight was measured",
            evidence="direct_complete_cluster_minimum_principal_weight",
            target_cluster_ids=np.asarray([target_cluster_index], dtype=np.int64),
            excluded_interval_distances=(distance_array,),
            residual_intervals=ritz_interval[None, :],
        )

    delta = float(np.min(distance_array, initial=np.inf)) if distance_array.size else 0.0
    if not delta > 0.0:
        return OwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            lower_bounds=None,
            conservative_complement_gaps=np.asarray([delta]),
            reason="Ritz-cluster interval touches an excluded exact cluster",
            evidence="cluster_residual_interval_gap_bound",
            target_cluster_ids=np.asarray([target_cluster_index], dtype=np.int64),
            excluded_interval_distances=(distance_array,),
            residual_intervals=ritz_interval[None, :],
        )
    bound = 1.0 - (radius / delta) ** 2 - 0.5 * splus_cluster_operator_norm**2
    status = (
        "OWNERSHIP_BOUND_PASS"
        if bound > minimum_owned_weight
        else "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE"
    )
    return OwnershipAssessment(
        status=status,
        lower_bounds=np.asarray([bound], dtype=np.float64),
        conservative_complement_gaps=np.asarray([delta], dtype=np.float64),
        reason=(
            "complete-cluster residual/Splus operator-norm bound passes"
            if status == "OWNERSHIP_BOUND_PASS"
            else "complete-cluster bound is conservative and inconclusive"
        ),
        evidence="cluster_residual_interval_gap_bound",
        target_cluster_ids=np.asarray([target_cluster_index], dtype=np.int64),
        excluded_interval_distances=(distance_array,),
        residual_intervals=ritz_interval[None, :],
    )


def assess_partitioned_three_root_ownership(
    ritz_energies: np.ndarray,
    direct_residual_norms: np.ndarray,
    per_ritz_splus_norms: np.ndarray,
    projected_clusters: tuple[tuple[int, ...], ...],
    residual_block_operator_norms: np.ndarray,
    splus_cluster_operator_norms: np.ndarray,
    exact_singlet_intervals: np.ndarray,
    *,
    exact_cluster_ids: np.ndarray,
    anchor_complete_through_fourth: bool,
    evaluation_depth: int = 2,
    minimum_owned_weight: float = 0.9,
) -> PartitionedOwnershipAssessment:
    """Assess the ordered lowest-three partition without labeling degeneracies.

    Singleton projected clusters use their direct Ritz residual and Splus
    norm.  Multiroot clusters use only the complete residual-block and Splus
    operator norms.  The ordered projected cluster cardinalities must exactly
    equal those of the complete exact clusters in the first-three window.
    Every distance to every excluded exact cluster is retained, including
    zero.  A zero distance is not filtered away.
    """

    if evaluation_depth in (0, 1):
        return PartitionedOwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUATED_BY_DESIGN",
            evaluation_depth=evaluation_depth,
            conditional_on_registered_exact_window=True,
            cluster_records=(),
            reason="formal N36 ownership is evaluated only at depth two",
        )
    if evaluation_depth != 2:
        raise ValueError("formal C0 ownership depth must be 0, 1, or 2")
    energies = np.asarray(ritz_energies, dtype=np.float64)
    residuals = np.asarray(direct_residual_norms, dtype=np.float64)
    per_splus = np.asarray(per_ritz_splus_norms, dtype=np.float64)
    block_residuals = np.asarray(residual_block_operator_norms, dtype=np.float64)
    block_splus = np.asarray(splus_cluster_operator_norms, dtype=np.float64)
    intervals = np.asarray(exact_singlet_intervals, dtype=np.float64)
    exact_ids = np.asarray(exact_cluster_ids)
    if energies.shape != (3,) or residuals.shape != (3,) or per_splus.shape != (3,):
        raise ValueError("partitioned ownership requires three Ritz roots")
    if (
        np.any(~np.isfinite(energies))
        or np.any(~np.isfinite(residuals))
        or np.any(~np.isfinite(per_splus))
        or np.any(residuals < 0.0)
        or np.any(per_splus < 0.0)
        or np.any(np.diff(energies) < 0.0)
    ):
        raise ValueError("partitioned Ritz diagnostics are invalid")
    if (
        intervals.ndim != 2
        or intervals.shape[1:] != (2,)
        or intervals.shape[0] < 4
        or np.any(~np.isfinite(intervals))
        or np.any(intervals[:, 0] > intervals[:, 1])
        or exact_ids.shape != (intervals.shape[0],)
        or exact_ids.dtype.kind not in "iu"
        or np.any(exact_ids < 0)
    ):
        raise ValueError("exact interval/cluster table is invalid")
    if (
        not np.isfinite(minimum_owned_weight)
        or minimum_owned_weight != 0.9
        or not isinstance(anchor_complete_through_fourth, (bool, np.bool_))
    ):
        raise ValueError("ownership threshold/completeness identity changed")
    if not projected_clusters or tuple(
        index for cluster in projected_clusters for index in cluster
    ) != (0, 1, 2) or any(
        not cluster or tuple(sorted(set(cluster))) != cluster
        for cluster in projected_clusters
    ):
        raise ValueError("projected clusters must be an ordered partition of roots 0,1,2")
    if block_residuals.shape != (len(projected_clusters),) or block_splus.shape != (
        len(projected_clusters),
    ) or np.any(~np.isfinite(block_residuals)) or np.any(~np.isfinite(block_splus)) or np.any(block_residuals < 0.0) or np.any(block_splus < 0.0):
        raise ValueError("one finite nonnegative block norm is required per projected cluster")
    if not anchor_complete_through_fourth:
        return PartitionedOwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            evaluation_depth=2,
            conditional_on_registered_exact_window=True,
            cluster_records=(),
            reason="registered exact window is not complete through the fourth singlet",
        )

    # Exact cluster labels must occur in one contiguous spectral run.  Build
    # the complete union interval for each such run and retain its order.
    exact_order: list[int] = []
    exact_rows: dict[int, list[int]] = {}
    for index, raw_cluster in enumerate(exact_ids):
        cluster = int(raw_cluster)
        if cluster not in exact_rows:
            exact_order.append(cluster)
            exact_rows[cluster] = []
        elif exact_order[-1] != cluster:
            raise ValueError("one exact cluster occurs in disjoint spectral runs")
        exact_rows[cluster].append(index)
    exact_cluster_intervals: dict[int, np.ndarray] = {}
    for cluster in exact_order:
        rows = intervals[exact_rows[cluster]]
        exact_cluster_intervals[cluster] = np.asarray(
            [np.min(rows[:, 0]), np.max(rows[:, 1])], dtype=np.float64
        )
    for left, right in zip(exact_order, exact_order[1:], strict=False):
        if (
            exact_cluster_intervals[right][0]
            - exact_cluster_intervals[left][1]
            <= 1.0e-10
        ):
            raise ValueError("distinct exact clusters are within the frozen merge tolerance")
    if int(exact_ids[2]) == int(exact_ids[3]):
        return PartitionedOwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            evaluation_depth=2,
            conditional_on_registered_exact_window=True,
            cluster_records=(),
            reason="an exact cluster crosses the root-3/root-4 boundary",
        )
    target_exact_order: list[int] = []
    for cluster in exact_ids[:3]:
        value = int(cluster)
        if not target_exact_order or target_exact_order[-1] != value:
            target_exact_order.append(value)
    exact_cardinalities = [int(np.count_nonzero(exact_ids[:3] == cluster)) for cluster in target_exact_order]
    projected_cardinalities = [len(cluster) for cluster in projected_clusters]
    if projected_cardinalities != exact_cardinalities:
        return PartitionedOwnershipAssessment(
            status="OWNERSHIP_NOT_EVALUABLE",
            evaluation_depth=2,
            conditional_on_registered_exact_window=True,
            cluster_records=(),
            reason="projected/exact ordered complete-cluster cardinalities differ",
        )

    records: list[dict[str, Any]] = []
    statuses: list[str] = []
    for ordinal, (projected, exact_target) in enumerate(
        zip(projected_clusters, target_exact_order, strict=True)
    ):
        if len(projected) == 1:
            root = projected[0]
            radius = float(residuals[root])
            splus_norm = float(per_splus[root])
            certificate_kind = "singleton_residual_interval_gap_bound"
        else:
            radius = float(block_residuals[ordinal])
            splus_norm = float(block_splus[ordinal])
            certificate_kind = "complete_cluster_residual_interval_gap_bound"
        ritz_interval = np.asarray(
            [energies[projected[0]] - radius, energies[projected[-1]] + radius],
            dtype=np.float64,
        )
        target_interval = exact_cluster_intervals[exact_target]
        target_intersection = float(
            min(ritz_interval[1], target_interval[1])
            - max(ritz_interval[0], target_interval[0])
        )
        excluded_ids: list[int] = []
        excluded_intervals: list[list[float]] = []
        distances: list[float] = []
        for exact_cluster in exact_order:
            if exact_cluster == exact_target:
                continue
            excluded = exact_cluster_intervals[exact_cluster]
            excluded_ids.append(exact_cluster)
            excluded_intervals.append(excluded.tolist())
            distances.append(
                max(
                    0.0,
                    float(ritz_interval[0] - excluded[1]),
                    float(excluded[0] - ritz_interval[1]),
                )
            )
        delta = float(min(distances)) if distances else 0.0
        lower_bound: float | None = None
        residual_ratio_squared: float | None = None
        spin_weight_bound = 0.5 * splus_norm**2
        if target_intersection < 0.0 or delta <= 0.0:
            status = "OWNERSHIP_NOT_EVALUABLE"
            reason = (
                "Ritz interval does not overlap the assigned target cluster"
                if target_intersection < 0.0
                else "Ritz interval touches or overlaps an excluded exact cluster"
            )
        else:
            residual_ratio_squared = (radius / delta) ** 2
            lower_bound = 1.0 - residual_ratio_squared - spin_weight_bound
            status = (
                "OWNERSHIP_BOUND_PASS"
                if lower_bound > minimum_owned_weight
                else "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE"
            )
            reason = (
                "registered conditional lower bound passes"
                if status == "OWNERSHIP_BOUND_PASS"
                else "registered conditional bound is conservative and inconclusive"
            )
        statuses.append(status)
        records.append(
            {
                "projected_cluster_ordinal": ordinal,
                "projected_root_indices": list(projected),
                "projected_cluster_cardinality": len(projected),
                "exact_target_cluster_id": exact_target,
                "exact_target_interval": target_interval.tolist(),
                "ritz_residual_interval": ritz_interval.tolist(),
                "excluded_exact_cluster_ids": excluded_ids,
                "excluded_exact_cluster_intervals": excluded_intervals,
                "excluded_interval_distances_including_zero": distances,
                "delta": delta,
                "residual_norm_used": radius,
                "splus_operator_norm_used": splus_norm,
                "residual_ratio_squared": residual_ratio_squared,
                "splus_spin_weight_upper_bound": spin_weight_bound,
                "owned_weight_lower_bound": lower_bound,
                "certificate_kind": certificate_kind,
                "status": status,
                "reason": reason,
            }
        )
    if "OWNERSHIP_NOT_EVALUABLE" in statuses:
        aggregate = "OWNERSHIP_NOT_EVALUABLE"
        aggregate_reason = "at least one complete target cluster is not evaluable"
    elif "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE" in statuses:
        aggregate = "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE"
        aggregate_reason = "at least one conservative cluster bound is inconclusive"
    else:
        aggregate = "OWNERSHIP_BOUND_PASS"
        aggregate_reason = "every complete target-cluster bound strictly exceeds 0.9"
    return PartitionedOwnershipAssessment(
        status=aggregate,
        evaluation_depth=2,
        conditional_on_registered_exact_window=True,
        cluster_records=tuple(records),
        reason=aggregate_reason,
    )


_PARTITIONED_OWNERSHIP_KEYS = {
    "schema_version",
    "stage",
    "evaluation_depth",
    "conditional_on_registered_exact_window",
    "minimum_owned_weight_strictly_greater_than",
    "raw_per_ritz_label",
    "status",
    "reason",
    "clusters",
}
_PARTITIONED_OWNERSHIP_CLUSTER_KEYS = {
    "projected_cluster_ordinal",
    "projected_root_indices",
    "projected_cluster_cardinality",
    "exact_target_cluster_id",
    "exact_target_interval",
    "ritz_residual_interval",
    "excluded_exact_cluster_ids",
    "excluded_exact_cluster_intervals",
    "excluded_interval_distances_including_zero",
    "delta",
    "residual_norm_used",
    "splus_operator_norm_used",
    "residual_ratio_squared",
    "splus_spin_weight_upper_bound",
    "owned_weight_lower_bound",
    "certificate_kind",
    "status",
    "reason",
}


def validate_partitioned_ownership_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the exact formal v8 ownership JSON schema."""

    value = _exact_mapping_keys(
        payload, _PARTITIONED_OWNERSHIP_KEYS, "partitioned ownership payload"
    )
    allowed = {
        "OWNERSHIP_BOUND_PASS",
        "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE",
        "OWNERSHIP_NOT_EVALUABLE",
        "OWNERSHIP_NOT_EVALUATED_BY_DESIGN",
    }
    depth = value["evaluation_depth"]
    if (
        value["schema_version"] != 8
        or value["stage"] != "P9v2-D1-T-C0-ownership-v8"
        or isinstance(depth, bool)
        or depth not in (0, 1, 2)
        or value["conditional_on_registered_exact_window"] is not True
        or value["minimum_owned_weight_strictly_greater_than"] != 0.9
        or value["raw_per_ritz_label"]
        != "BASIS_DEPENDENT_RAW_NON_GATING_NON_OWNERSHIP"
        or value["status"] not in allowed
        or not isinstance(value["reason"], str)
        or not value["reason"]
        or not isinstance(value["clusters"], list)
    ):
        raise ValueError("partitioned ownership payload identity is invalid")
    if depth < 2:
        if value["status"] != "OWNERSHIP_NOT_EVALUATED_BY_DESIGN" or value["clusters"]:
            raise ValueError("depth-zero/one ownership payload must be empty by design")
        return dict(value)
    if value["status"] == "OWNERSHIP_NOT_EVALUATED_BY_DESIGN":
        raise ValueError("depth-two ownership cannot use the by-design label")
    clusters = value["clusters"]
    if not clusters:
        if value["status"] != "OWNERSHIP_NOT_EVALUABLE":
            raise ValueError("an empty depth-two ownership payload is only NOT_EVALUABLE")
        return dict(value)
    flattened: list[int] = []
    statuses: list[str] = []
    for ordinal, raw in enumerate(clusters):
        record = _exact_mapping_keys(
            raw,
            _PARTITIONED_OWNERSHIP_CLUSTER_KEYS,
            f"ownership cluster {ordinal}",
        )
        roots = record["projected_root_indices"]
        if (
            record["projected_cluster_ordinal"] != ordinal
            or not isinstance(roots, list)
            or not roots
            or any(isinstance(item, bool) or not isinstance(item, int) for item in roots)
            or roots != sorted(set(roots))
            or record["projected_cluster_cardinality"] != len(roots)
            or isinstance(record["exact_target_cluster_id"], bool)
            or not isinstance(record["exact_target_cluster_id"], int)
            or record["exact_target_cluster_id"] < 0
            or not isinstance(record["reason"], str)
            or not record["reason"]
        ):
            raise ValueError("ownership cluster identity/cardinality is invalid")
        flattened.extend(roots)
        for key in ("exact_target_interval", "ritz_residual_interval"):
            interval = _finite_json_array(record[key], f"ownership cluster {ordinal}.{key}", ndim=1)
            if interval.shape != (2,) or interval[0] > interval[1]:
                raise ValueError("ownership interval is not one ordered [lower,upper] pair")
        excluded_ids = record["excluded_exact_cluster_ids"]
        excluded_intervals = _finite_json_array(
            record["excluded_exact_cluster_intervals"],
            f"ownership cluster {ordinal}.excluded intervals",
            ndim=2,
        )
        distances = _finite_json_array(
            record["excluded_interval_distances_including_zero"],
            f"ownership cluster {ordinal}.excluded distances",
            ndim=1,
        )
        if (
            not isinstance(excluded_ids, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in excluded_ids)
            or len(set(excluded_ids)) != len(excluded_ids)
            or excluded_intervals.shape != (len(excluded_ids), 2)
            or distances.shape != (len(excluded_ids),)
            or np.any(excluded_intervals[:, 0] > excluded_intervals[:, 1])
            or np.any(distances < 0.0)
        ):
            raise ValueError("excluded exact-cluster distance table is invalid")
        delta = _finite_json_scalar(record["delta"], "ownership delta")
        radius = _finite_json_scalar(record["residual_norm_used"], "ownership residual norm")
        splus = _finite_json_scalar(record["splus_operator_norm_used"], "ownership Splus norm")
        spin_term = _finite_json_scalar(
            record["splus_spin_weight_upper_bound"], "ownership Splus bound"
        )
        if min(delta, radius, splus, spin_term) < 0.0:
            raise ValueError("ownership bound terms must be nonnegative")
        _assert_close(
            delta,
            float(np.min(distances)) if distances.size else 0.0,
            "ownership delta",
            1.0e-12,
        )
        _assert_close(spin_term, 0.5 * splus**2, "ownership Splus bound", 1.0e-12)
        kind = record["certificate_kind"]
        expected_kind = (
            "singleton_residual_interval_gap_bound"
            if len(roots) == 1
            else "complete_cluster_residual_interval_gap_bound"
        )
        if kind != expected_kind:
            raise ValueError("ownership certificate kind violates projected cardinality")
        status = record["status"]
        if status not in {
            "OWNERSHIP_BOUND_PASS",
            "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE",
            "OWNERSHIP_NOT_EVALUABLE",
        }:
            raise ValueError("ownership cluster status is invalid")
        statuses.append(status)
        ratio = record["residual_ratio_squared"]
        bound = record["owned_weight_lower_bound"]
        if status == "OWNERSHIP_NOT_EVALUABLE":
            if ratio is not None or bound is not None:
                raise ValueError("NOT_EVALUABLE cluster cannot manufacture a bound")
        else:
            ratio_value = _finite_json_scalar(ratio, "ownership residual ratio")
            bound_value = _finite_json_scalar(bound, "ownership lower bound")
            if delta <= 0.0:
                raise ValueError("a certified cluster requires positive Delta")
            _assert_close(ratio_value, (radius / delta) ** 2, "ownership residual ratio", 1.0e-12)
            _assert_close(bound_value, 1.0 - ratio_value - spin_term, "ownership lower bound", 1.0e-12)
            if (status == "OWNERSHIP_BOUND_PASS") != (bound_value > 0.9):
                raise ValueError("ownership bound status/threshold mismatch")
    if flattened != [0, 1, 2]:
        raise ValueError("ownership clusters do not partition the first three roots")
    expected_status = (
        "OWNERSHIP_NOT_EVALUABLE"
        if "OWNERSHIP_NOT_EVALUABLE" in statuses
        else "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE"
        if "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE" in statuses
        else "OWNERSHIP_BOUND_PASS"
    )
    if value["status"] != expected_status:
        raise ValueError("aggregate ownership status violates registered priority")
    return dict(value)


def serialize_partitioned_ownership(
    assessment: PartitionedOwnershipAssessment,
) -> dict[str, Any]:
    """Return the exact JSON object accepted for the formal v8 depth output."""

    allowed = {
        "OWNERSHIP_BOUND_PASS",
        "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE",
        "OWNERSHIP_NOT_EVALUABLE",
        "OWNERSHIP_NOT_EVALUATED_BY_DESIGN",
    }
    if not isinstance(assessment, PartitionedOwnershipAssessment) or assessment.status not in allowed:
        raise ValueError("partitioned ownership assessment/status is invalid")
    if assessment.evaluation_depth not in (0, 1, 2):
        raise ValueError("ownership evaluation depth is invalid")
    if assessment.evaluation_depth < 2 and (
        assessment.status != "OWNERSHIP_NOT_EVALUATED_BY_DESIGN"
        or assessment.cluster_records
    ):
        raise ValueError("depth-zero/one ownership must be empty and by-design")
    if assessment.evaluation_depth == 2 and assessment.status == "OWNERSHIP_NOT_EVALUATED_BY_DESIGN":
        raise ValueError("depth-two ownership cannot use the by-design label")
    payload = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-ownership-v8",
        "evaluation_depth": assessment.evaluation_depth,
        "conditional_on_registered_exact_window": assessment.conditional_on_registered_exact_window,
        "minimum_owned_weight_strictly_greater_than": 0.9,
        "raw_per_ritz_label": "BASIS_DEPENDENT_RAW_NON_GATING_NON_OWNERSHIP",
        "status": assessment.status,
        "reason": assessment.reason,
        "clusters": [dict(record) for record in assessment.cluster_records],
    }
    # Reject accidental NaN/Infinity or non-JSON values before the runner uses
    # this object to replace CPP-K's explicit pending placeholder.
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return validate_partitioned_ownership_payload(payload)


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if np.any(~np.isfinite(result)):
        raise FloatingPointError(f"{name} contains nonfinite entries")
    return result


def _orthonormality_error(vectors: np.ndarray) -> float:
    gram = vectors.conj().T @ vectors
    return float(np.max(np.abs(gram - np.eye(gram.shape[0])), initial=0.0))


def symmetric_block_lowdin(
    candidates: np.ndarray,
    basis: np.ndarray | None = None,
    *,
    relative_rank_tolerance: float = 1.0e-11,
    minimum_gram_eigenvalue_ratio: float = 1.0e-12,
    required_rank: int | None = None,
) -> BlockLowdinResult:
    """Two-pass project and symmetrically Löwdin-orthogonalize one block.

    Rank is tested against the largest eigenvalue of the *unprojected* block
    Gram matrix.  This prevents a pure roundoff residual from normalizing
    itself into a manufactured Krylov direction.
    """

    raw = _finite_matrix(candidates, "candidates")
    if not np.isfinite(relative_rank_tolerance) or not (
        0.0 < relative_rank_tolerance < 1.0
    ):
        raise ValueError("relative_rank_tolerance must lie in (0,1)")
    if not np.isfinite(minimum_gram_eigenvalue_ratio) or not (
        0.0 < minimum_gram_eigenvalue_ratio < 1.0
    ):
        raise ValueError("minimum_gram_eigenvalue_ratio must lie in (0,1)")
    if required_rank is not None and (
        isinstance(required_rank, bool)
        or not isinstance(required_rank, int)
        or required_rank < 1
    ):
        raise ValueError("required_rank must be a positive integer")

    if basis is None:
        existing = np.empty((raw.shape[0], 0), dtype=np.complex128)
    else:
        existing = np.asarray(basis, dtype=np.complex128)
        if existing.ndim != 2 or existing.shape[0] != raw.shape[0]:
            raise ValueError("basis and candidates dimensions differ")
        if np.any(~np.isfinite(existing)):
            raise FloatingPointError("basis contains nonfinite entries")

    unprojected_gram = 0.5 * (raw.conj().T @ raw + (raw.conj().T @ raw).conj().T)
    unprojected_values = np.linalg.eigvalsh(unprojected_gram)
    largest = float(max(unprojected_values[-1], 0.0))
    if not largest > 0.0:
        raise ValueError("candidate block has zero rank")

    projected = raw.copy()
    for _ in range(2):
        if existing.shape[1]:
            projected -= existing @ (existing.conj().T @ projected)
    gram = 0.5 * (projected.conj().T @ projected + (projected.conj().T @ projected).conj().T)
    values, vectors = np.linalg.eigh(gram)
    threshold = relative_rank_tolerance**2 * largest
    retained = values > threshold
    rank = int(np.count_nonzero(retained))
    if required_rank is not None and rank != required_rank:
        raise ValueError(
            f"block rank {rank} differs from required_rank={required_rank}"
        )
    if rank < 1:
        raise ValueError("candidate block has no accepted directions")
    accepted_values = values[retained]
    ratio = float(accepted_values[0] / accepted_values[-1])
    if ratio < minimum_gram_eigenvalue_ratio:
        raise ValueError(
            "accepted block Gram eigenvalue ratio is below the frozen gate"
        )
    q = projected @ (
        vectors[:, retained] @ np.diag(accepted_values ** -0.5)
    )

    # The cleanup is itself block-Löwdin: two more projections followed by a
    # symmetric normalization.  No columnwise modified Gram-Schmidt is used.
    for _ in range(2):
        if existing.shape[1]:
            q -= existing @ (existing.conj().T @ q)
    cleanup_gram = 0.5 * (q.conj().T @ q + (q.conj().T @ q).conj().T)
    cleanup_values, cleanup_vectors = np.linalg.eigh(cleanup_gram)
    if np.any(cleanup_values <= 0.0):
        raise FloatingPointError("Löwdin cleanup lost positive definiteness")
    q = q @ (
        cleanup_vectors
        @ np.diag(cleanup_values ** -0.5)
        @ cleanup_vectors.conj().T
    )
    error = _orthonormality_error(q)
    if existing.shape[1]:
        error = max(
            error,
            float(np.max(np.abs(existing.conj().T @ q), initial=0.0)),
        )
    return BlockLowdinResult(
        vectors=np.asarray(q, dtype=np.complex128),
        projected_gram_eigenvalues=np.asarray(values, dtype=np.float64),
        unprojected_largest_gram_eigenvalue=largest,
        retained_rank=rank,
        orthonormality_error=error,
    )


def _inspect_arm(
    *,
    depth: int,
    q: np.ndarray,
    hq: np.ndarray,
    frontier: np.ndarray,
    q0: np.ndarray,
    normalized_carriers: np.ndarray,
    target_roots: int,
    maximum_relative_antihermitian_error: float,
) -> TeacherArmDepth:
    raw_projected = q.conj().T @ hq
    anti = float(
        np.max(np.abs(raw_projected - raw_projected.conj().T), initial=0.0)
    )
    scale = max(
        float(np.max(np.abs(raw_projected), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    relative_anti = anti / scale
    if (
        not np.isfinite(relative_anti)
        or relative_anti > maximum_relative_antihermitian_error
    ):
        raise FloatingPointError(
            "projected Hamiltonian violates the frozen Hermiticity gate: "
            f"relative_error={relative_anti:.6e}"
        )
    projected = 0.5 * (raw_projected + raw_projected.conj().T)
    values, small_vectors = np.linalg.eigh(projected)
    if len(values) < target_roots:
        raise ValueError("accepted basis is smaller than the target root count")
    partition = partition_selected_ritz_clusters(
        values,
        depth=depth,
        target_roots=target_roots,
        tau_internal=1.0e-10,
    )
    coefficients = small_vectors[:, :target_roots]
    roots = q @ coefficients
    h_roots = hq @ coefficients
    energies = np.asarray(values[:target_roots], dtype=np.float64)
    residual = h_roots - roots * energies[None, :]
    residual_norms = np.asarray(np.linalg.norm(residual, axis=0), dtype=np.float64)
    seed_weights = np.asarray(
        np.sum(np.abs(q0.conj().T @ roots) ** 2, axis=0), dtype=np.float64
    )
    carrier_weights = np.asarray(
        np.abs(normalized_carriers.conj().T @ roots) ** 2,
        dtype=np.float64,
    )
    return TeacherArmDepth(
        depth=depth,
        accepted_dimension=int(q.shape[1]),
        frontier_rank=int(frontier.shape[1]),
        energies=energies,
        direct_residual_norms=residual_norms,
        residual_intervals=np.stack(
            (energies - residual_norms, energies + residual_norms), axis=1
        ),
        seed_subspace_weights=seed_weights,
        carrier_overlap_squares=carrier_weights,
        selected_adjacent_gaps=partition.adjacent_gaps,
        projected_cluster_ids=partition.cluster_ids,
        projected_root_boundary_gap=partition.root_boundary_gap,
        orthonormality_error=_orthonormality_error(q),
        projected_hamiltonian_antihermitian_error=anti,
        projected_hamiltonian_relative_antihermitian_error=relative_anti,
        ritz_vectors=np.asarray(roots, dtype=np.complex128),
        ritz_coefficients=np.asarray(coefficients, dtype=np.complex128),
    )


def projector_frobenius_distance(
    first: np.ndarray, second: np.ndarray
) -> float:
    """Return ``||P_first-P_second||_F`` without materializing projectors."""

    left = _finite_matrix(first, "first subspace")
    right = _finite_matrix(second, "second subspace")
    if left.shape[0] != right.shape[0]:
        raise ValueError("subspaces have different ambient dimensions")
    left_error = _orthonormality_error(left)
    right_error = _orthonormality_error(right)
    if max(left_error, right_error) > 1.0e-10:
        raise ValueError(
            "projector distance requires orthonormal input columns; "
            f"errors=({left_error:.3e},{right_error:.3e})"
        )
    overlap = left.conj().T @ right
    left_outside_right = left - right @ overlap.conj().T
    right_outside_left = right - left @ overlap
    squared = float(
        np.linalg.norm(left_outside_right, "fro") ** 2
        + np.linalg.norm(right_outside_left, "fro") ** 2
    )
    return float(np.sqrt(squared))


def projector_operator_norm_distance(
    first: np.ndarray, second: np.ndarray
) -> float:
    """Return a stable complete-projector operator-norm distance.

    The residual form avoids subtracting nearly equal principal-angle cosines,
    which otherwise creates a spurious numerical floor for identical spans.
    Both inputs must already be orthonormal and have equal rank.
    """

    left = _finite_matrix(first, "first cluster subspace")
    right = _finite_matrix(second, "second cluster subspace")
    if left.shape != right.shape:
        raise ValueError("cluster subspaces must have equal ambient shape and rank")
    left_error = _orthonormality_error(left)
    right_error = _orthonormality_error(right)
    if max(left_error, right_error) > 1.0e-10:
        raise ValueError("cluster projector distance requires orthonormal inputs")
    left_residual = left - right @ (right.conj().T @ left)
    right_residual = right - left @ (left.conj().T @ right)
    return float(
        max(
            np.linalg.norm(left_residual, ord=2),
            np.linalg.norm(right_residual, ord=2),
        )
    )


def partition_selected_ritz_clusters(
    projected_energies: np.ndarray,
    *,
    depth: int,
    target_roots: int = 3,
    tau_internal: float = 1.0e-10,
) -> ProjectedClusterPartition:
    """Partition the first three sorted Ritz roots independently in one arm.

    At depths one and two the fourth projected value is mandatory and the
    selected/unselected gap must be strictly larger than ``tau_internal``.
    Depth zero is the registered three-dimensional exception.
    """

    values = np.asarray(projected_energies, dtype=np.float64)
    if values.ndim != 1 or values.size < target_roots:
        raise ValueError("not enough projected Ritz values")
    if np.any(~np.isfinite(values)) or np.any(np.diff(values) < 0.0):
        raise FloatingPointError("projected Ritz values must be finite and sorted")
    if depth not in (0, 1, 2) or target_roots != 3:
        raise ValueError("the frozen C0 partition uses depths 0/1/2 and three roots")
    if not np.isfinite(tau_internal) or tau_internal != 1.0e-10:
        raise ValueError("the frozen physical-energy cluster threshold is 1e-10")
    selected = values[:target_roots]
    gaps = np.diff(selected)
    if np.any(~np.isfinite(gaps)) or np.any(gaps < 0.0):
        raise FloatingPointError("selected-root gap is ambiguous or nonfinite")
    boundary: float | None = None
    if depth in (1, 2):
        if values.size < target_roots + 1:
            raise ValueError("depth one/two requires a projected fourth root")
        boundary = float(values[target_roots] - values[target_roots - 1])
        if not np.isfinite(boundary) or not boundary > tau_internal:
            raise ValueError("projected root3/root4 boundary is not isolated")
    ids = np.zeros(target_roots, dtype=np.int64)
    cluster = 0
    for index, gap in enumerate(gaps, start=1):
        if float(gap) > tau_internal:
            cluster += 1
        ids[index] = cluster
    clusters = tuple(
        tuple(int(i) for i in np.flatnonzero(ids == label))
        for label in range(cluster + 1)
    )
    return ProjectedClusterPartition(
        adjacent_gaps=np.asarray(gaps, dtype=np.float64),
        cluster_ids=ids,
        clusters=clusters,
        root_boundary_gap=boundary,
    )


def require_common_projected_partition(
    physical: ProjectedClusterPartition,
    null: ProjectedClusterPartition,
) -> tuple[tuple[int, ...], ...]:
    """Fail closed unless two independently built selected partitions agree."""

    if not np.array_equal(physical.cluster_ids, null.cluster_ids):
        raise ValueError("physical/null internal projected-cluster partitions differ")
    if physical.clusters != null.clusters:
        raise ValueError("physical/null projected-cluster cardinalities differ")
    return physical.clusters


def compute_cluster_invariant(
    *,
    q: np.ndarray,
    hq: np.ndarray,
    ritz_coefficients: np.ndarray,
    cluster_indices: tuple[int, ...],
    q0: np.ndarray,
    normalized_physical_carriers: np.ndarray,
    raised_q: np.ndarray | None = None,
) -> ClusterInvariant:
    """Compute all frozen rotation-invariant diagnostics for one cluster.

    ``raised_q`` is ``Splus @ q`` already produced by the registered Splus
    action.  No Hamiltonian or Splus action is performed by this function.
    """

    basis = _finite_matrix(q, "q")
    h_basis = _finite_matrix(hq, "hq")
    coeff = _finite_matrix(ritz_coefficients, "ritz_coefficients")
    seed = _finite_matrix(q0, "q0")
    carriers = _finite_matrix(
        normalized_physical_carriers, "normalized physical carriers"
    )
    if h_basis.shape != basis.shape or coeff.shape[0] != basis.shape[1]:
        raise ValueError("Q/HQ/Ritz coefficient shapes disagree")
    if seed.shape[0] != basis.shape[0] or carriers.shape[0] != basis.shape[0]:
        raise ValueError("seed/carrier ambient dimension differs from Q")
    if max(_orthonormality_error(basis), _orthonormality_error(seed)) > 1.0e-10:
        raise ValueError("Q and Q0 must be orthonormal")
    carrier_norm_error = float(
        np.max(np.abs(np.sum(np.abs(carriers) ** 2, axis=0) - 1.0), initial=0.0)
    )
    if carrier_norm_error > 1.0e-10:
        raise ValueError("the fixed physical carriers must be normalized")
    if not cluster_indices or tuple(sorted(set(cluster_indices))) != cluster_indices:
        raise ValueError("cluster indices must be one nonempty sorted unique tuple")
    if cluster_indices[-1] >= coeff.shape[1]:
        raise ValueError("cluster index exceeds selected Ritz coefficients")

    yc = coeff[:, cluster_indices]
    psi = basis @ yc
    if _orthonormality_error(psi) > 1.0e-10:
        raise ValueError("selected complete cluster is not orthonormal")
    hpsi = h_basis @ yc
    ac = psi.conj().T @ hpsi
    residual = hpsi - psi @ ac
    residual_singular = np.sort(np.linalg.svd(residual, compute_uv=False))
    seed_gram = psi.conj().T @ seed @ (seed.conj().T @ psi)
    seed_gram = 0.5 * (seed_gram + seed_gram.conj().T)
    seed_spectrum = np.sort(np.linalg.eigvalsh(seed_gram))
    carrier_weights = np.asarray(
        np.sum(np.abs(psi.conj().T @ carriers) ** 2, axis=0),
        dtype=np.float64,
    )
    splus_singular: np.ndarray | None = None
    splus_operator: float | None = None
    if raised_q is not None:
        raised = _finite_matrix(raised_q, "raised_q")
        if raised.shape[1] != basis.shape[1]:
            raise ValueError("raised Q has the wrong number of columns")
        raised_cluster = raised @ yc
        splus_singular = np.sort(np.linalg.svd(raised_cluster, compute_uv=False))
        splus_operator = float(splus_singular[-1])
    return ClusterInvariant(
        indices=cluster_indices,
        projector_basis=np.asarray(psi, dtype=np.complex128),
        residual_singular_values=np.asarray(residual_singular, dtype=np.float64),
        residual_frobenius_norm=float(np.linalg.norm(residual, ord="fro")),
        residual_operator_norm=float(residual_singular[-1]),
        seed_projector_eigenvalues=np.asarray(seed_spectrum, dtype=np.float64),
        seed_projector_trace=float(np.sum(seed_spectrum)),
        fixed_physical_carrier_weights=carrier_weights,
        splus_singular_values=(
            None if splus_singular is None else np.asarray(splus_singular, dtype=np.float64)
        ),
        splus_operator_norm=splus_operator,
        raw_label="BASIS_DEPENDENT_RAW_NON_GATING_NON_OWNERSHIP",
    )


def compare_cluster_invariants(
    physical: ClusterInvariant,
    null: ClusterInvariant,
) -> ClusterComparison:
    """Compare complete cluster invariants without pairing Ritz columns."""

    if physical.indices != null.indices:
        raise ValueError("physical/null cluster indices differ")
    if physical.residual_singular_values.shape != null.residual_singular_values.shape:
        raise ValueError("physical/null residual cluster rank differs")
    if physical.seed_projector_eigenvalues.shape != null.seed_projector_eigenvalues.shape:
        raise ValueError("physical/null seed cluster rank differs")
    if physical.fixed_physical_carrier_weights.shape != null.fixed_physical_carrier_weights.shape:
        raise ValueError("physical/null fixed carrier bank differs")
    left_splus = physical.splus_singular_values
    right_splus = null.splus_singular_values
    if (left_splus is None) != (right_splus is None):
        raise ValueError("Splus cluster diagnostics are missing in one arm")
    splus_spectrum_difference: float | None = None
    splus_operator_difference: float | None = None
    if left_splus is not None and right_splus is not None:
        if left_splus.shape != right_splus.shape:
            raise ValueError("physical/null Splus cluster rank differs")
        splus_spectrum_difference = float(np.max(np.abs(left_splus - right_splus)))
        splus_operator_difference = abs(
            float(physical.splus_operator_norm) - float(null.splus_operator_norm)
        )
    return ClusterComparison(
        projector_operator_norm_distance=projector_operator_norm_distance(
            physical.projector_basis, null.projector_basis
        ),
        residual_spectrum_max_difference=float(
            np.max(
                np.abs(
                    physical.residual_singular_values
                    - null.residual_singular_values
                ),
                initial=0.0,
            )
        ),
        residual_frobenius_norm_difference=abs(
            physical.residual_frobenius_norm - null.residual_frobenius_norm
        ),
        residual_operator_norm_difference=abs(
            physical.residual_operator_norm - null.residual_operator_norm
        ),
        seed_spectrum_max_difference=float(
            np.max(
                np.abs(
                    physical.seed_projector_eigenvalues
                    - null.seed_projector_eigenvalues
                ),
                initial=0.0,
            )
        ),
        seed_trace_difference=abs(
            physical.seed_projector_trace - null.seed_projector_trace
        ),
        carrier_weight_max_difference=float(
            np.max(
                np.abs(
                    physical.fixed_physical_carrier_weights
                    - null.fixed_physical_carrier_weights
                ),
                initial=0.0,
            )
        ),
        splus_spectrum_max_difference=splus_spectrum_difference,
        splus_operator_norm_difference=splus_operator_difference,
    )


def _checked_action(action: ArrayAction, vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(action(vectors), dtype=np.complex128)
    if result.shape != vectors.shape:
        raise ValueError("Hamiltonian action changed the block shape")
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("Hamiltonian action returned nonfinite values")
    return result


def restricted_splus_diagnostics(
    action: ArrayAction | None,
    basis: np.ndarray,
    ritz_coefficients: np.ndarray,
) -> tuple[float | None, np.ndarray | None]:
    """Return the restricted S+ operator norm and every Ritz-vector norm.

    A Frobenius average is intentionally not returned: it can dilute a single
    contaminated direction and is insufficient for a subspace-purity gate.
    """

    if action is None:
        return None, None
    raised = np.asarray(action(basis), dtype=np.complex128)
    if raised.ndim != 2 or raised.shape[1] != basis.shape[1]:
        raise ValueError("S+ action returned an invalid block shape")
    if np.any(~np.isfinite(raised)):
        raise FloatingPointError("S+ action returned nonfinite values")
    gram = 0.5 * (raised.conj().T @ raised + (raised.conj().T @ raised).conj().T)
    largest = float(max(np.linalg.eigvalsh(gram)[-1], 0.0))
    operator_norm = float(np.sqrt(largest))
    raised_roots = raised @ np.asarray(ritz_coefficients, dtype=np.complex128)
    ritz_norms = np.asarray(np.linalg.norm(raised_roots, axis=0), dtype=np.float64)
    return operator_norm, ritz_norms


def restricted_splus_from_raised(
    raised_basis: np.ndarray,
    ritz_coefficients: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Diagnose a previously computed restricted S+ block without a new action."""

    raised = _finite_matrix(raised_basis, "raised_basis")
    coefficients = _finite_matrix(ritz_coefficients, "ritz_coefficients")
    if raised.shape[1] != coefficients.shape[0]:
        raise ValueError("raised basis and Ritz coefficients have incompatible shapes")
    singular = np.linalg.svd(raised, compute_uv=False)
    operator_norm = float(singular[0])
    raised_roots = raised @ coefficients
    ritz_norms = np.asarray(np.linalg.norm(raised_roots, axis=0), dtype=np.float64)
    return operator_norm, ritz_norms


def c0_physical_null_trajectory(
    hamiltonian_action: ArrayAction,
    physical_seed_vectors: np.ndarray,
    *,
    maximum_depth: int = 2,
    target_roots: int = 3,
    hamiltonian_scale: float = 1.0,
    relative_rank_tolerance: float = 1.0e-11,
    minimum_gram_eigenvalue_ratio: float = 1.0e-12,
    maximum_relative_antihermitian_error: float = 1.0e-10,
    null_rotation: np.ndarray | None = None,
    total_splus_action: ArrayAction | None = None,
) -> TeacherTrajectory:
    """Run the exact registered physical/null C0 algorithm on dense arrays.

    Physical and null independently execute the full pipeline.  For the
    registered full-rank depth-two cell this is three rank-three batched
    actions per arm: six calls and eighteen scalar-equivalent MVMs in total.
    """

    seeds = _finite_matrix(physical_seed_vectors, "physical_seed_vectors")
    rank = seeds.shape[1]
    if rank != 3 or target_roots != 3 or maximum_depth != 2:
        raise ValueError("the frozen C0 cell requires rank=roots=3 and depth=2")
    if not np.isfinite(hamiltonian_scale) or not hamiltonian_scale > 0.0:
        raise ValueError("hamiltonian_scale must be finite and positive")
    if null_rotation is None:
        rotation = np.asarray(
            [[1.0, 2.0, 2.0], [2.0, 1.0, -2.0], [-2.0, 2.0, -1.0]],
            dtype=np.complex128,
        ) / 3.0
    else:
        rotation = np.asarray(null_rotation, dtype=np.complex128)
    if rotation.shape != (3, 3):
        raise ValueError("null rotation must be 3-by-3")
    if np.max(np.abs(rotation.conj().T @ rotation - np.eye(3))) > 1.0e-14:
        raise ValueError("null rotation is not orthogonal/unitary")

    carrier_norms = np.linalg.norm(seeds, axis=0)
    if np.any(carrier_norms <= 0.0):
        raise ValueError("a frozen carrier has zero norm")
    normalized_carriers = seeds / carrier_norms[None, :]
    physical_initial = symmetric_block_lowdin(
        seeds,
        relative_rank_tolerance=relative_rank_tolerance,
        minimum_gram_eigenvalue_ratio=minimum_gram_eigenvalue_ratio,
        required_rank=rank,
    ).vectors
    null_seeds = seeds @ rotation
    null_initial = symmetric_block_lowdin(
        null_seeds,
        relative_rank_tolerance=relative_rank_tolerance,
        minimum_gram_eigenvalue_ratio=minimum_gram_eigenvalue_ratio,
        required_rank=rank,
    ).vectors

    physical_q = physical_initial
    physical_hq = np.empty((seeds.shape[0], 0), dtype=np.complex128)
    physical_frontier = physical_initial
    physical_h_frontier = np.empty_like(physical_frontier)
    null_q = null_initial
    null_hq = np.empty((seeds.shape[0], 0), dtype=np.complex128)
    null_frontier = null_initial
    null_h_frontier = np.empty_like(null_frontier)
    physical_q0 = physical_initial
    null_q0 = null_initial
    depths: list[TeacherDepth] = []
    physical_actions = 0
    null_actions = 0
    physical_mvm = 0
    null_mvm = 0

    for depth in range(maximum_depth + 1):
        physical_h_frontier = _checked_action(
            hamiltonian_action, physical_frontier
        )
        physical_actions += 1
        physical_mvm += physical_frontier.shape[1]
        null_h_frontier = _checked_action(hamiltonian_action, null_frontier)
        null_actions += 1
        null_mvm += null_frontier.shape[1]
        physical_hq = np.column_stack((physical_hq, physical_h_frontier))
        null_hq = np.column_stack((null_hq, null_h_frontier))

        physical_row = _inspect_arm(
            depth=depth,
            q=physical_q,
            hq=physical_hq,
            frontier=physical_frontier,
            q0=physical_q0,
            normalized_carriers=normalized_carriers,
            target_roots=target_roots,
            maximum_relative_antihermitian_error=(
                maximum_relative_antihermitian_error
            ),
        )
        null_row = _inspect_arm(
            depth=depth,
            q=null_q,
            hq=null_hq,
            frontier=null_frontier,
            q0=null_q0,
            normalized_carriers=normalized_carriers,
            target_roots=target_roots,
            maximum_relative_antihermitian_error=(
                maximum_relative_antihermitian_error
            ),
        )
        physical_partition = ProjectedClusterPartition(
            adjacent_gaps=physical_row.selected_adjacent_gaps,
            cluster_ids=physical_row.projected_cluster_ids,
            clusters=tuple(
                tuple(int(i) for i in np.flatnonzero(physical_row.projected_cluster_ids == label))
                for label in range(int(np.max(physical_row.projected_cluster_ids)) + 1)
            ),
            root_boundary_gap=physical_row.projected_root_boundary_gap,
        )
        null_partition = ProjectedClusterPartition(
            adjacent_gaps=null_row.selected_adjacent_gaps,
            cluster_ids=null_row.projected_cluster_ids,
            clusters=tuple(
                tuple(int(i) for i in np.flatnonzero(null_row.projected_cluster_ids == label))
                for label in range(int(np.max(null_row.projected_cluster_ids)) + 1)
            ),
            root_boundary_gap=null_row.projected_root_boundary_gap,
        )
        common_clusters = require_common_projected_partition(
            physical_partition, null_partition
        )

        physical_raised: np.ndarray | None = None
        null_raised: np.ndarray | None = None
        if total_splus_action is not None and depth in (0, 2):
            physical_raised = np.asarray(
                total_splus_action(physical_q), dtype=np.complex128
            )
            null_raised = np.asarray(total_splus_action(null_q), dtype=np.complex128)
            if (
                physical_raised.ndim != 2
                or null_raised.ndim != 2
                or physical_raised.shape[1] != physical_q.shape[1]
                or null_raised.shape[1] != null_q.shape[1]
                or np.any(~np.isfinite(physical_raised))
                or np.any(~np.isfinite(null_raised))
            ):
                raise ValueError("Splus action returned an invalid physical/null block")
            splus_physical = restricted_splus_from_raised(
                physical_raised, physical_row.ritz_coefficients
            )
            splus_null = restricted_splus_from_raised(
                null_raised, null_row.ritz_coefficients
            )
        else:
            splus_physical = (None, None)
            splus_null = (None, None)

        physical_invariants = tuple(
            compute_cluster_invariant(
                q=physical_q,
                hq=physical_hq,
                ritz_coefficients=physical_row.ritz_coefficients,
                cluster_indices=cluster,
                q0=physical_q0,
                normalized_physical_carriers=normalized_carriers,
                raised_q=physical_raised,
            )
            for cluster in common_clusters
        )
        null_invariants = tuple(
            compute_cluster_invariant(
                q=null_q,
                hq=null_hq,
                ritz_coefficients=null_row.ritz_coefficients,
                cluster_indices=cluster,
                q0=null_q0,
                normalized_physical_carriers=normalized_carriers,
                raised_q=null_raised,
            )
            for cluster in common_clusters
        )
        comparisons = tuple(
            compare_cluster_invariants(left, right)
            for left, right in zip(physical_invariants, null_invariants, strict=True)
        )
        depths.append(
            TeacherDepth(
                depth=depth,
                physical=physical_row,
                null=null_row,
                common_projected_clusters=common_clusters,
                physical_cluster_invariants=physical_invariants,
                null_cluster_invariants=null_invariants,
                cluster_comparisons=comparisons,
                full_span_projector_frobenius_error=(
                    projector_frobenius_distance(physical_q, null_q)
                ),
                ritz_subspace_projector_frobenius_error=(
                    projector_frobenius_distance(
                        physical_row.ritz_vectors, null_row.ritz_vectors
                    )
                ),
                physical_splus_operator_norm=splus_physical[0],
                null_splus_operator_norm=splus_null[0],
                physical_splus_ritz_norms=splus_physical[1],
                null_splus_ritz_norms=splus_null[1],
                physical_batched_h_actions=physical_actions,
                null_batched_h_actions=null_actions,
                total_batched_h_actions=physical_actions + null_actions,
                physical_scalar_equivalent_h_mvm=physical_mvm,
                null_scalar_equivalent_h_mvm=null_mvm,
                total_scalar_equivalent_h_mvm=physical_mvm + null_mvm,
            )
        )
        if depth == maximum_depth:
            break
        physical_next = symmetric_block_lowdin(
            physical_h_frontier / hamiltonian_scale,
            physical_q,
            relative_rank_tolerance=relative_rank_tolerance,
            minimum_gram_eigenvalue_ratio=minimum_gram_eigenvalue_ratio,
            required_rank=rank,
        ).vectors
        null_next = symmetric_block_lowdin(
            null_h_frontier / hamiltonian_scale,
            null_q,
            relative_rank_tolerance=relative_rank_tolerance,
            minimum_gram_eigenvalue_ratio=minimum_gram_eigenvalue_ratio,
            required_rank=rank,
        ).vectors
        physical_q = np.column_stack((physical_q, physical_next))
        null_q = np.column_stack((null_q, null_next))
        physical_frontier = physical_next
        null_frontier = null_next

    return TeacherTrajectory(
        depths=tuple(depths),
        physical_q=physical_q,
        physical_hq=physical_hq,
        physical_frontier=physical_frontier,
        physical_h_frontier=physical_h_frontier,
        null_q=null_q,
        null_hq=null_hq,
        null_frontier=null_frontier,
        null_h_frontier=null_h_frontier,
        physical_q0=physical_q0,
        normalized_carriers=normalized_carriers,
    )


def reconstruct_incremental_q_hq(
    generations: tuple[dict[str, np.ndarray], ...],
    *,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct cumulative Q/HQ in RAM from v8 F/HF increments."""

    if arm not in {"PHYSICAL", "NULL"} or not generations:
        raise ValueError("arm must be PHYSICAL or NULL and generations nonempty")
    q_blocks: list[np.ndarray] = []
    hq_blocks: list[np.ndarray] = []
    ambient: int | None = None
    for depth, generation in enumerate(generations):
        f_name = f"{arm}_F_{depth}"
        hf_name = f"{arm}_HF_{depth}"
        if f_name not in generation or hf_name not in generation:
            raise ValueError("incremental generation is missing F/HF")
        f_block = _finite_matrix(generation[f_name], f_name)
        hf_block = _finite_matrix(generation[hf_name], hf_name)
        if f_block.shape != hf_block.shape or f_block.shape[1] != 3:
            raise ValueError("incremental F/HF shapes differ")
        if ambient is None:
            ambient = f_block.shape[0]
        if f_block.shape[0] != ambient:
            raise ValueError("incremental ambient dimension changed")
        q_blocks.append(f_block)
        hq_blocks.append(hf_block)
    q = np.column_stack(q_blocks)
    hq = np.column_stack(hq_blocks)
    if _orthonormality_error(q) > 1.0e-10:
        raise ValueError("reconstructed Q is not orthonormal")
    return q, hq


_ARM_DIAGNOSTIC_KEYS = {
    "all_projected_energies",
    "energies",
    "energies_per_site",
    "direct_residual_norms",
    "residual_intervals",
    "seed_subspace_weights",
    "carrier_overlap_squares",
    "per_ritz_label",
    "selected_adjacent_gaps",
    "projected_cluster_ids",
    "orthonormality_error",
    "projected_h_antihermitian_error",
    "projected_h_relative_antihermitian_error",
    "projected_third_fourth_gap",
    "projected_lowest_three_isolated",
    "projected_hamiltonian",
}
_CLUSTER_DIAGNOSTIC_KEYS = {
    "indices",
    "cardinality",
    "residual_singular_values",
    "residual_frobenius_norm",
    "residual_operator_norm",
    "seed_projector_eigenvalues",
    "seed_projector_trace",
    "fixed_physical_carrier_weights",
    "splus_evaluated",
    "splus_singular_values",
    "splus_operator_norm",
    "splus_spin_weight_upper_bound",
    "raw_per_ritz_label",
}
_CLUSTER_COMPARISON_KEYS = {
    "projector_operator_norm_distance",
    "residual_spectrum_max_difference",
    "residual_frobenius_norm_difference",
    "residual_operator_norm_difference",
    "seed_spectrum_max_difference",
    "seed_trace_difference",
    "carrier_weight_max_difference",
    "splus_evaluated",
    "splus_spectrum_max_difference",
    "splus_operator_norm_difference",
}
_NORMAL_DIAGNOSTIC_KEYS = {
    "schema_version",
    "stage",
    "config_sha256",
    "xdiag_commit",
    "scope",
    "run_id",
    "geometry",
    "nsites",
    "nup",
    "coupling",
    "hamiltonian_scale_recomputed",
    "depth",
    "accepted_dimension",
    "frontier_rank",
    "attempt_identity",
    "physical",
    "null",
    "full_span_projector_frobenius_error",
    "ritz_subspace_projector_frobenius_error",
    "splus_evaluated",
    "physical_splus_restricted_operator_norm",
    "physical_splus_per_ritz_norms",
    "null_splus_restricted_operator_norm",
    "null_splus_per_ritz_norms",
    "physical_splus_spin_weight_upper_bound",
    "null_splus_spin_weight_upper_bound",
    "projected_cluster_common_partition",
    "physical_cluster_invariants",
    "null_cluster_invariants",
    "cluster_comparisons",
    "ownership_status",
    "ownership",
    "physical_batched_h_actions",
    "null_batched_h_actions",
    "total_batched_h_actions",
    "physical_scalar_equivalent_h_mvm",
    "null_scalar_equivalent_h_mvm",
    "total_committed_scalar_MVM",
    "exact_small_probe_setup_h_actions_disclosed",
    "total_splus_committed_batched_actions_this_depth",
    "total_splus_semantic_replay_batched_actions",
    "total_splus_scalar_columns_this_depth",
    "frontier_lowdin_orthonormality_error",
    "null_frontier_lowdin_orthonormality_error",
    "attempted_actual_scalar_MVM",
    "attempted_started_batched_actions",
    "attempted_completed_batched_actions",
    "attempted_abandoned_batched_actions",
    "attempted_retried_batched_actions",
    "attempted_splus_started_batched_actions",
    "attempted_splus_completed_batched_actions",
    "attempted_splus_abandoned_batched_actions",
    "observed_openmp_team_size",
    "process_max_rss_gib",
    "process_wall_seconds",
}


def validate_c0_normal_attempt_telemetry(
    payload: Mapping[str, Any],
    *,
    expected_attempt_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the exact normal CPP-K schema and its per-invocation ledger.

    This intentionally does not replace numerical checkpoint replay.  It is
    the cheap pre-commit schema/attempt check used immediately when the child
    exits, before the runner may publish any generation marker.
    """

    value = _exact_mapping_keys(
        payload, _NORMAL_DIAGNOSTIC_KEYS, "normal CPP-K diagnostics"
    )
    expected_attempt = _exact_mapping_keys(
        expected_attempt_identity,
        _ATTEMPT_IDENTITY_FIELDS,
        "expected CPP-K attempt identity",
    )
    observed_attempt = _exact_mapping_keys(
        value["attempt_identity"],
        _ATTEMPT_IDENTITY_FIELDS,
        "CPP-K attempt identity",
    )
    if dict(observed_attempt) != dict(expected_attempt):
        raise ValueError("normal CPP-K telemetry belongs to another attempt")
    for key, item in observed_attempt.items():
        if not isinstance(item, str) or not item:
            raise ValueError(f"normal CPP-K attempt identity is empty at {key}")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(observed_attempt["runtime_manifest_sha256"])
    ) is None:
        raise ValueError("normal CPP-K runtime-manifest SHA is malformed")
    depth = value["depth"]
    if (
        value["schema_version"] != 8
        or value["stage"] != "P9v2-D1-T-C0-production-v8"
        or value["config_sha256"]
        != "4cc274699563c6867cef84f11e746b503af5d74c9d2c59ad9add19b620762ab8"
        or value["scope"] != "FORMAL_N36_C0_V8"
        or value["geometry"]
        != "triangular_J1J2_6x6_periodic_axial_torus"
        or value["nsites"] != 36
        or value["nup"] != 18
        or not isinstance(value["run_id"], str)
        or not value["run_id"]
        or not isinstance(value["xdiag_commit"], str)
        or not value["xdiag_commit"]
        or isinstance(depth, bool)
        or depth not in (0, 1, 2)
    ):
        raise ValueError("normal CPP-K attempt identity/stage is invalid")
    _assert_close(value["coupling"], 0.125, "normal CPP-K coupling", 0.0)
    _assert_close(
        value["hamiltonian_scale_recomputed"],
        91.125,
        "normal CPP-K Hamiltonian scale",
        0.0,
    )
    if (
        isinstance(value["accepted_dimension"], bool)
        or value["accepted_dimension"] != 3 * (depth + 1)
        or isinstance(value["frontier_rank"], bool)
        or value["frontier_rank"] != 3
    ):
        raise ValueError("normal CPP-K dimension/frontier ledger mismatch")
    exact_counters = {
        "attempted_started_batched_actions": 2,
        "attempted_completed_batched_actions": 2,
        "attempted_abandoned_batched_actions": 0,
        "attempted_retried_batched_actions": 0,
        "attempted_actual_scalar_MVM": 6,
        "attempted_splus_started_batched_actions": 2 if depth < 2 else 4,
        "attempted_splus_completed_batched_actions": 2 if depth < 2 else 4,
        "attempted_splus_abandoned_batched_actions": 0,
    }
    for key, expected in exact_counters.items():
        if isinstance(value[key], bool) or value[key] != expected:
            raise ValueError(f"normal CPP-K attempt ledger mismatch at {key}")
    cumulative = {
        "physical_batched_h_actions": depth + 1,
        "null_batched_h_actions": depth + 1,
        "total_batched_h_actions": 2 * (depth + 1),
        "physical_scalar_equivalent_h_mvm": 3 * (depth + 1),
        "null_scalar_equivalent_h_mvm": 3 * (depth + 1),
        "total_committed_scalar_MVM": 6 * (depth + 1),
        "exact_small_probe_setup_h_actions_disclosed": 0,
        "total_splus_committed_batched_actions_this_depth": (
            2 if depth in (0, 2) else 0
        ),
        "total_splus_semantic_replay_batched_actions": 2 if depth > 0 else 0,
        "total_splus_scalar_columns_this_depth": (
            6 * (depth + 1) if depth in (0, 2) else 0
        ),
        "observed_openmp_team_size": 40,
    }
    for key, expected in cumulative.items():
        if isinstance(value[key], bool) or value[key] != expected:
            raise ValueError(f"normal CPP-K committed ledger mismatch at {key}")
    expected_ownership = (
        "OWNERSHIP_REQUIRES_RUNNER_EXACT_INTERVAL_EVALUATION"
        if depth == 2
        else "OWNERSHIP_NOT_EVALUATED_BY_DESIGN"
    )
    if (
        value["ownership_status"] != expected_ownership
        or not isinstance(value["ownership"], Mapping)
        or value["ownership"].get("status") != expected_ownership
    ):
        raise ValueError("normal precommit CPP-K ownership state is invalid")
    rss = _finite_json_scalar(value["process_max_rss_gib"], "CPP-K MaxRSS")
    wall = _finite_json_scalar(value["process_wall_seconds"], "CPP-K wall time")
    if rss < 0.0 or wall < 0.0:
        raise ValueError("normal CPP-K resource telemetry is negative")
    return {
        "stage": value["stage"],
        "attempted_h_batched_actions": 2,
        "attempted_actual_scalar_MVM": 6,
        "attempted_splus_started_batched_actions": exact_counters[
            "attempted_splus_started_batched_actions"
        ],
        "attempted_splus_completed_batched_actions": exact_counters[
            "attempted_splus_completed_batched_actions"
        ],
        "attempted_splus_abandoned_batched_actions": 0,
        "seed_rows_started": 0,
        "seed_rows_completed": 0,
        "seed_rows_abandoned": 0,
        "determinant_calls_started": 0,
        "determinant_calls_completed": 0,
        "determinant_calls_abandoned": 0,
        "basis_initializations_started": 0,
        "basis_initializations_completed": 0,
        "basis_initializations_abandoned": 0,
        "cpp_max_rss_gib": rss,
    }


def _exact_mapping_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} has missing or extra fields")
    return value


def _finite_json_array(value: Any, label: str, *, ndim: int | None = None) -> np.ndarray:
    def contains_boolean(item: Any) -> bool:
        if isinstance(item, (bool, np.bool_)):
            return True
        if isinstance(item, np.ndarray):
            return item.dtype.kind == "b" or any(
                contains_boolean(element) for element in item.flat
            )
        if isinstance(item, (list, tuple)):
            return any(contains_boolean(element) for element in item)
        return False

    if contains_boolean(value):
        raise ValueError(f"{label} contains a boolean in a numeric field")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a real numeric array") from error
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{label} has the wrong rank")
    if np.any(~np.isfinite(array)):
        raise FloatingPointError(f"{label} contains nonfinite values")
    return array


def _finite_json_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a numeric scalar, not boolean")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not np.isfinite(scalar):
        raise FloatingPointError(f"{label} is nonfinite")
    return scalar


def _assert_close(
    observed: Any,
    expected: Any,
    label: str,
    tolerance: float,
) -> None:
    left = _finite_json_array(observed, label)
    right = np.asarray(expected)
    if np.any(~np.isfinite(right)):
        raise FloatingPointError(f"recomputed {label} is nonfinite")
    if left.shape != right.shape:
        raise ValueError(f"{label} shape differs from semantic replay")
    discrepancy = float(np.max(np.abs(left - right), initial=0.0))
    if discrepancy > tolerance:
        raise ValueError(
            f"{label} differs from semantic replay: {discrepancy:.6e}"
        )


def _assert_json_semantic_close(
    observed: Any,
    expected: Any,
    *,
    label: str,
    tolerance: float,
) -> None:
    """Compare a strict JSON tree, allowing tolerance only for real numbers."""

    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{label} mapping schema differs from replay")
        for key in expected:
            _assert_json_semantic_close(
                observed[key], expected[key],
                label=f"{label}.{key}", tolerance=tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{label} list cardinality differs from replay")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _assert_json_semantic_close(
                left, right, label=f"{label}[{index}]", tolerance=tolerance
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected or type(observed) is not type(expected):
            raise ValueError(f"{label} categorical value differs from replay")
        return
    _assert_close(observed, expected, label, tolerance)


def _replay_arm_diagnostics(
    stored: Any,
    replay: TeacherArmDepth,
    *,
    q: np.ndarray,
    hq: np.ndarray,
    nsites: int,
    singleton_indices: set[int],
    tolerance: float,
    label: str,
) -> None:
    arm = _exact_mapping_keys(stored, _ARM_DIAGNOSTIC_KEYS, label)
    raw_projected = q.conj().T @ hq
    projected = 0.5 * (raw_projected + raw_projected.conj().T)
    all_energies = np.linalg.eigvalsh(projected)
    _assert_close(arm["all_projected_energies"], all_energies, f"{label}.all_projected_energies", tolerance)
    _assert_close(arm["energies"], replay.energies, f"{label}.energies", tolerance)
    _assert_close(arm["energies_per_site"], replay.energies / float(nsites), f"{label}.energies_per_site", tolerance)
    stored_residuals = _finite_json_array(
        arm["direct_residual_norms"], f"{label}.direct_residual_norms", ndim=1
    )
    stored_seed = _finite_json_array(
        arm["seed_subspace_weights"], f"{label}.seed_subspace_weights", ndim=1
    )
    stored_carrier = _finite_json_array(
        arm["carrier_overlap_squares"], f"{label}.carrier_overlap_squares", ndim=2
    )
    if stored_residuals.shape != (3,) or stored_seed.shape != (3,) or stored_carrier.shape != (3, 3):
        raise ValueError(f"{label} raw per-Ritz diagnostic shapes differ from rank three")
    # Inside a multiroot cluster an eigensolver may choose another rotation.
    # Those raw columns are deliberately non-gating.  Cardinality-one roots
    # are uniquely defined and therefore must replay to the strict tolerance.
    for index in singleton_indices:
        _assert_close(stored_residuals[index], replay.direct_residual_norms[index], f"{label}.direct_residual_norms[{index}]", tolerance)
        _assert_close(stored_seed[index], replay.seed_subspace_weights[index], f"{label}.seed_subspace_weights[{index}]", tolerance)
        _assert_close(stored_carrier[:, index], replay.carrier_overlap_squares[:, index], f"{label}.carrier_overlap_squares[:,{index}]", tolerance)
    intervals = _finite_json_array(
        arm["residual_intervals"], f"{label}.residual_intervals", ndim=2
    )
    if intervals.shape != (3, 2):
        raise ValueError(f"{label}.residual_intervals must have shape (3,2)")
    _assert_close(
        intervals,
        np.column_stack(
            (
                _finite_json_array(arm["energies"], f"{label}.energies") - stored_residuals,
                _finite_json_array(arm["energies"], f"{label}.energies") + stored_residuals,
            )
        ),
        f"{label}.residual_intervals_internal",
        tolerance,
    )
    if arm["per_ritz_label"] != "BASIS_DEPENDENT_RAW_NON_GATING_NON_OWNERSHIP":
        raise ValueError(f"{label} raw per-Ritz label is missing")
    _assert_close(arm["selected_adjacent_gaps"], replay.selected_adjacent_gaps, f"{label}.selected_adjacent_gaps", tolerance)
    stored_cluster_ids = arm["projected_cluster_ids"]
    if (
        not isinstance(stored_cluster_ids, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in stored_cluster_ids)
        or stored_cluster_ids != replay.projected_cluster_ids.tolist()
    ):
        raise ValueError(f"{label}.projected_cluster_ids differ from replay")
    _assert_close(arm["orthonormality_error"], replay.orthonormality_error, f"{label}.orthonormality_error", tolerance)
    _assert_close(arm["projected_h_antihermitian_error"], replay.projected_hamiltonian_antihermitian_error, f"{label}.projected_h_antihermitian_error", tolerance)
    _assert_close(arm["projected_h_relative_antihermitian_error"], replay.projected_hamiltonian_relative_antihermitian_error, f"{label}.projected_h_relative_antihermitian_error", tolerance)
    expected_gap = -1.0 if replay.projected_root_boundary_gap is None else replay.projected_root_boundary_gap
    _assert_close(arm["projected_third_fourth_gap"], expected_gap, f"{label}.projected_third_fourth_gap", tolerance)
    if arm["projected_lowest_three_isolated"] is not True:
        raise ValueError(f"{label}.projected_lowest_three_isolated must be true")
    _assert_close(arm["projected_hamiltonian"], projected.real, f"{label}.projected_hamiltonian", tolerance)


def _replay_cluster_without_splus(
    stored: Any,
    replay: ClusterInvariant,
    *,
    tolerance: float,
    label: str,
) -> None:
    cluster = _exact_mapping_keys(stored, _CLUSTER_DIAGNOSTIC_KEYS, label)
    indices = cluster["indices"]
    if (
        not isinstance(indices, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in indices)
        or indices != list(replay.indices)
        or isinstance(cluster["cardinality"], bool)
        or cluster["cardinality"] != len(replay.indices)
    ):
        raise ValueError(f"{label} index/cardinality mismatch")
    _assert_close(cluster["residual_singular_values"], replay.residual_singular_values, f"{label}.residual_singular_values", tolerance)
    _assert_close(cluster["residual_frobenius_norm"], replay.residual_frobenius_norm, f"{label}.residual_frobenius_norm", tolerance)
    _assert_close(cluster["residual_operator_norm"], replay.residual_operator_norm, f"{label}.residual_operator_norm", tolerance)
    _assert_close(cluster["seed_projector_eigenvalues"], replay.seed_projector_eigenvalues, f"{label}.seed_projector_eigenvalues", tolerance)
    _assert_close(cluster["seed_projector_trace"], replay.seed_projector_trace, f"{label}.seed_projector_trace", tolerance)
    _assert_close(cluster["fixed_physical_carrier_weights"], replay.fixed_physical_carrier_weights, f"{label}.fixed_physical_carrier_weights", tolerance)
    if cluster["raw_per_ritz_label"] != "BASIS_DEPENDENT_RAW_NON_GATING_NON_OWNERSHIP":
        raise ValueError(f"{label} raw diagnostic label is missing")


def _validate_stored_splus(
    diagnostics: Mapping[str, Any],
    *,
    depth: int,
    common_clusters: tuple[tuple[int, ...], ...],
    tolerance: float,
) -> None:
    evaluated = depth in (0, 2)
    if diagnostics["splus_evaluated"] is not evaluated:
        raise ValueError("stored Splus depth registration mismatch")
    arm_data: dict[str, tuple[float, np.ndarray]] = {}
    cluster_data: dict[str, list[tuple[np.ndarray, float]]] = {}
    for arm in ("physical", "null"):
        operator = _finite_json_scalar(
            diagnostics[f"{arm}_splus_restricted_operator_norm"],
            f"{arm} Splus restricted operator norm",
        )
        per_ritz = _finite_json_array(
            diagnostics[f"{arm}_splus_per_ritz_norms"],
            f"{arm} Splus per-Ritz norms",
            ndim=1,
        )
        bound = _finite_json_scalar(
            diagnostics[f"{arm}_splus_spin_weight_upper_bound"],
            f"{arm} Splus spin-weight bound",
        )
        if per_ritz.shape != (3,):
            raise ValueError(f"{arm} Splus per-Ritz vector must have length three")
        if evaluated:
            if operator < 0.0 or np.any(per_ritz < 0.0) or operator > 1.0e-10:
                raise ValueError(f"{arm} restricted Splus purity gate failed")
            _assert_close(bound, 0.5 * operator**2, f"{arm} Splus spin-weight bound", tolerance)
            if np.any(per_ritz > operator + tolerance):
                raise ValueError(f"{arm} per-Ritz Splus norm exceeds restricted norm")
        elif operator != -1.0 or not np.array_equal(per_ritz, -np.ones(3)) or bound != -1.0:
            raise ValueError("depth-one Splus sentinel schema mismatch")
        arm_data[arm] = (operator, per_ritz)

        clusters = diagnostics[f"{arm}_cluster_invariants"]
        if not isinstance(clusters, list) or len(clusters) != len(common_clusters):
            raise ValueError(f"{arm} Splus cluster cardinality mismatch")
        values: list[tuple[np.ndarray, float]] = []
        for ordinal, (stored, indices) in enumerate(zip(clusters, common_clusters, strict=True)):
            cluster = _exact_mapping_keys(
                stored, _CLUSTER_DIAGNOSTIC_KEYS, f"{arm} cluster {ordinal}"
            )
            if cluster["splus_evaluated"] is not evaluated:
                raise ValueError(f"{arm} cluster Splus registration mismatch")
            singular = _finite_json_array(
                cluster["splus_singular_values"],
                f"{arm} cluster {ordinal} Splus spectrum",
                ndim=1,
            )
            cluster_operator = _finite_json_scalar(
                cluster["splus_operator_norm"],
                f"{arm} cluster {ordinal} Splus operator norm",
            )
            cluster_bound = _finite_json_scalar(
                cluster["splus_spin_weight_upper_bound"],
                f"{arm} cluster {ordinal} Splus bound",
            )
            if evaluated:
                if singular.shape != (len(indices),) or np.any(singular < 0.0):
                    raise ValueError("registered Splus cluster spectrum shape/sign mismatch")
                _assert_close(cluster_operator, singular[-1], f"{arm} cluster {ordinal} Splus operator", tolerance)
                _assert_close(cluster_bound, 0.5 * cluster_operator**2, f"{arm} cluster {ordinal} Splus bound", tolerance)
                if cluster_operator > operator + tolerance:
                    raise ValueError("cluster Splus norm exceeds full restricted norm")
                if len(indices) == 1:
                    _assert_close(per_ritz[indices[0]], singular[0], f"{arm} singleton Splus", tolerance)
            elif singular.size or cluster_operator != -1.0 or cluster_bound != -1.0:
                raise ValueError("depth-one cluster Splus sentinel mismatch")
            values.append((singular, cluster_operator))
        cluster_data[arm] = values

    if evaluated and abs(arm_data["physical"][0] - arm_data["null"][0]) > 1.0e-10:
        raise ValueError("physical/null restricted Splus norms differ")
    comparisons = diagnostics["cluster_comparisons"]
    for ordinal, comparison in enumerate(comparisons):
        item = _exact_mapping_keys(
            comparison, _CLUSTER_COMPARISON_KEYS, f"cluster comparison {ordinal}"
        )
        if item["splus_evaluated"] is not evaluated:
            raise ValueError("cluster comparison Splus registration mismatch")
        p_spectrum, p_operator = cluster_data["physical"][ordinal]
        n_spectrum, n_operator = cluster_data["null"][ordinal]
        expected_spectrum = (
            float(np.max(np.abs(p_spectrum - n_spectrum), initial=0.0))
            if evaluated
            else -1.0
        )
        expected_operator = abs(p_operator - n_operator) if evaluated else -1.0
        _assert_close(item["splus_spectrum_max_difference"], expected_spectrum, f"cluster comparison {ordinal} Splus spectrum", tolerance)
        _assert_close(item["splus_operator_norm_difference"], expected_operator, f"cluster comparison {ordinal} Splus operator", tolerance)
        if evaluated and max(expected_spectrum, expected_operator) > 1.0e-10:
            raise ValueError("physical/null complete-cluster Splus gate failed")
        if evaluated and len(common_clusters[ordinal]) == 1:
            root = common_clusters[ordinal][0]
            if abs(arm_data["physical"][1][root] - arm_data["null"][1][root]) > 1.0e-10:
                raise ValueError("physical/null singleton Splus gate failed")


def replay_and_compare_checkpoint_semantics(
    states: tuple[dict[str, np.ndarray], ...],
    markers: tuple[dict[str, Any], ...],
    *,
    hamiltonian_scale: float = 91.125,
    nsites: int = 36,
    nup: int = 18,
    coupling: float = 0.125,
    expected_scope: str = "FORMAL_N36_C0_V8",
    expected_geometry: str = "triangular_J1J2_6x6_periodic_axial_torus",
    expected_ambient_dimension: int = 21_029_820,
    ownership_context: Mapping[str, Any] | None = None,
    tolerance: float = 1.0e-12,
) -> tuple[dict[str, Any], ...]:
    """Fresh-process replay of every persisted numerical C0 semantic.

    Only the four increment files per arm (plus generation-zero physical
    carriers) are trusted after their hash chain has been checked.  This
    routine reconstructs Q/HQ, frontiers, projected spectra, direct residuals,
    internal clusters and all physical/null invariants, then compares the
    immutable normal telemetry field-by-field.  Raw columns inside a multiroot
    cluster remain explicitly non-gating.

    Raised-Q is intentionally not checkpointed.  Therefore the actual total
    Splus action is replayed by each fresh CPP-K process; here its normal
    telemetry is tied to the registered depth, restricted/operator bounds,
    complete-cluster spectra and singleton relations without pretending that
    a stored diagnostic independently recreates the action.
    """

    if not np.isfinite(tolerance) or tolerance != 1.0e-12:
        raise ValueError("the frozen checkpoint replay tolerance is exactly 1e-12")
    if hamiltonian_scale != 91.125 or nsites != 36 or nup != 18 or coupling != 0.125:
        raise ValueError("formal v8 replay identity changed")
    if len(states) != len(markers) or not 1 <= len(states) <= 3:
        raise ValueError("checkpoint replay requires matching depths zero through two")
    if (
        isinstance(expected_ambient_dimension, bool)
        or not isinstance(expected_ambient_dimension, int)
        or expected_ambient_dimension < 1
    ):
        raise ValueError("expected ambient dimension must be a positive integer")
    for depth, generation in enumerate(states):
        for name, raw in generation.items():
            array = np.asarray(raw)
            if array.ndim != 2 or array.shape[0] != expected_ambient_dimension:
                raise ValueError(
                    f"depth {depth} state {name} has the wrong ambient dimension"
                )
    carriers = _finite_matrix(states[0].get("PHYSICAL_CARRIERS"), "physical carriers")
    if carriers.shape[1] != 3:
        raise ValueError("the fixed physical carrier bank must have rank three")
    carrier_norm_error = float(
        np.max(np.abs(np.sum(np.abs(carriers) ** 2, axis=0) - 1.0), initial=0.0)
    )
    if carrier_norm_error > 1.0e-10:
        raise ValueError("fixed physical carriers are not separately normalized")
    carrier_span = symmetric_block_lowdin(
        carriers,
        np.empty((carriers.shape[0], 0), dtype=np.complex128),
        required_rank=3,
    ).vectors

    reports: list[dict[str, Any]] = []
    for depth in range(len(states)):
        prefix = states[: depth + 1]
        physical_q, physical_hq = reconstruct_incremental_q_hq(prefix, arm="PHYSICAL")
        null_q, null_hq = reconstruct_incremental_q_hq(prefix, arm="NULL")
        physical_q0 = _finite_matrix(states[0]["PHYSICAL_F_0"], "physical Q0")
        null_q0 = _finite_matrix(states[0]["NULL_F_0"], "null Q0")
        physical_frontier = _finite_matrix(
            states[depth][f"PHYSICAL_F_{depth}"], "physical frontier"
        )
        null_frontier = _finite_matrix(
            states[depth][f"NULL_F_{depth}"], "null frontier"
        )
        if depth == 0:
            physical_novelty = projector_operator_norm_distance(
                physical_frontier, carrier_span
            )
            null_novelty = projector_operator_norm_distance(
                null_frontier, carrier_span
            )
        else:
            prior_physical_q, _ = reconstruct_incremental_q_hq(
                states[:depth], arm="PHYSICAL"
            )
            prior_null_q, _ = reconstruct_incremental_q_hq(
                states[:depth], arm="NULL"
            )
            expected_physical = symmetric_block_lowdin(
                states[depth - 1][f"PHYSICAL_HF_{depth - 1}"] / hamiltonian_scale,
                prior_physical_q,
                required_rank=3,
            ).vectors
            expected_null = symmetric_block_lowdin(
                states[depth - 1][f"NULL_HF_{depth - 1}"] / hamiltonian_scale,
                prior_null_q,
                required_rank=3,
            ).vectors
            physical_novelty = projector_operator_norm_distance(
                physical_frontier, expected_physical
            )
            null_novelty = projector_operator_norm_distance(
                null_frontier, expected_null
            )
        if max(physical_novelty, null_novelty) > tolerance:
            raise ValueError("frontier does not replay the registered two-pass Lowdin span")

        physical = _inspect_arm(
            depth=depth,
            q=physical_q,
            hq=physical_hq,
            frontier=physical_frontier,
            q0=physical_q0,
            normalized_carriers=carriers,
            target_roots=3,
            maximum_relative_antihermitian_error=1.0e-10,
        )
        null = _inspect_arm(
            depth=depth,
            q=null_q,
            hq=null_hq,
            frontier=null_frontier,
            q0=null_q0,
            normalized_carriers=carriers,
            target_roots=3,
            maximum_relative_antihermitian_error=1.0e-10,
        )
        p_partition = partition_selected_ritz_clusters(
            np.linalg.eigvalsh(0.5 * (physical_q.conj().T @ physical_hq + physical_hq.conj().T @ physical_q)),
            depth=depth,
        )
        n_partition = partition_selected_ritz_clusters(
            np.linalg.eigvalsh(0.5 * (null_q.conj().T @ null_hq + null_hq.conj().T @ null_q)),
            depth=depth,
        )
        common = require_common_projected_partition(p_partition, n_partition)
        singletons = {cluster[0] for cluster in common if len(cluster) == 1}
        physical_clusters = tuple(
            compute_cluster_invariant(
                q=physical_q,
                hq=physical_hq,
                ritz_coefficients=physical.ritz_coefficients,
                cluster_indices=cluster,
                q0=physical_q0,
                normalized_physical_carriers=carriers,
            )
            for cluster in common
        )
        null_clusters = tuple(
            compute_cluster_invariant(
                q=null_q,
                hq=null_hq,
                ritz_coefficients=null.ritz_coefficients,
                cluster_indices=cluster,
                q0=null_q0,
                normalized_physical_carriers=carriers,
            )
            for cluster in common
        )
        comparisons = tuple(
            compare_cluster_invariants(left, right)
            for left, right in zip(physical_clusters, null_clusters, strict=True)
        )
        span_error = projector_frobenius_distance(physical_q, null_q)
        ritz_error = projector_frobenius_distance(
            physical.ritz_vectors, null.ritz_vectors
        )
        if (
            span_error > 1.0e-10
            or ritz_error > 1.0e-10
            or np.max(np.abs(physical.energies - null.energies)) > 1.0e-10
        ):
            raise ValueError("physical/null replayed same-span gate failed")

        diagnostics = _exact_mapping_keys(
            markers[depth].get("diagnostics"),
            _NORMAL_DIAGNOSTIC_KEYS,
            f"depth {depth} normal diagnostics",
        )
        stable = markers[depth].get("stable_identity")
        if not isinstance(stable, Mapping):
            raise ValueError("checkpoint marker stable identity is absent")
        exact_identity = {
            "schema_version": 8,
            "stage": "P9v2-D1-T-C0-production-v8",
            "config_sha256": stable.get("config_sha256"),
            "xdiag_commit": stable.get("xdiag_commit"),
            "scope": expected_scope,
            "run_id": stable.get("run_id"),
            "geometry": expected_geometry,
            "nsites": nsites,
            "nup": nup,
            "depth": depth,
        }
        for key, expected in exact_identity.items():
            if isinstance(expected, int) and isinstance(diagnostics[key], bool):
                raise ValueError(f"depth {depth} diagnostic identity has boolean at {key}")
            if diagnostics[key] != expected:
                raise ValueError(f"depth {depth} diagnostic identity differs at {key}")
        attempt_identity = _exact_mapping_keys(
            diagnostics["attempt_identity"],
            _ATTEMPT_IDENTITY_FIELDS,
            f"depth {depth} diagnostic attempt identity",
        )
        if dict(attempt_identity) != markers[depth].get("attempt_identity"):
            raise ValueError("normal diagnostics are not bound to marker attempt")
        _assert_close(diagnostics["coupling"], coupling, "coupling", tolerance)
        _assert_close(diagnostics["hamiltonian_scale_recomputed"], hamiltonian_scale, "Hamiltonian scale", tolerance)
        if diagnostics["accepted_dimension"] != 3 * (depth + 1) or diagnostics["frontier_rank"] != 3:
            raise ValueError("accepted dimension/frontier rank ledger mismatch")
        _replay_arm_diagnostics(
            diagnostics["physical"], physical, q=physical_q, hq=physical_hq,
            nsites=nsites, singleton_indices=singletons,
            tolerance=tolerance, label=f"depth {depth} physical",
        )
        _replay_arm_diagnostics(
            diagnostics["null"], null, q=null_q, hq=null_hq,
            nsites=nsites, singleton_indices=singletons,
            tolerance=tolerance, label=f"depth {depth} null",
        )
        stored_physical_clusters = diagnostics["physical_cluster_invariants"]
        stored_null_clusters = diagnostics["null_cluster_invariants"]
        stored_comparisons = diagnostics["cluster_comparisons"]
        if not all(isinstance(value, list) and len(value) == len(common) for value in (stored_physical_clusters, stored_null_clusters, stored_comparisons)):
            raise ValueError("complete-cluster diagnostic cardinality mismatch")
        for ordinal, (stored_p, stored_n, stored_c, replay_p, replay_n, replay_c) in enumerate(
            zip(stored_physical_clusters, stored_null_clusters, stored_comparisons, physical_clusters, null_clusters, comparisons, strict=True)
        ):
            _replay_cluster_without_splus(stored_p, replay_p, tolerance=tolerance, label=f"depth {depth} physical cluster {ordinal}")
            _replay_cluster_without_splus(stored_n, replay_n, tolerance=tolerance, label=f"depth {depth} null cluster {ordinal}")
            comparison = _exact_mapping_keys(stored_c, _CLUSTER_COMPARISON_KEYS, f"depth {depth} cluster comparison {ordinal}")
            for key in (
                "projector_operator_norm_distance",
                "residual_spectrum_max_difference",
                "residual_frobenius_norm_difference",
                "residual_operator_norm_difference",
                "seed_spectrum_max_difference",
                "seed_trace_difference",
                "carrier_weight_max_difference",
            ):
                _assert_close(comparison[key], getattr(replay_c, key), f"depth {depth} cluster comparison {ordinal}.{key}", tolerance)
                if _finite_json_scalar(comparison[key], key) > 1.0e-10:
                    raise ValueError("physical/null complete-cluster invariant gate failed")
        _assert_close(diagnostics["full_span_projector_frobenius_error"], span_error, "full-span projector distance", tolerance)
        _assert_close(diagnostics["ritz_subspace_projector_frobenius_error"], ritz_error, "three-root projector distance", tolerance)
        if diagnostics["projected_cluster_common_partition"] is not True:
            raise ValueError("common projected-cluster partition verdict is not true")
        _validate_stored_splus(
            diagnostics, depth=depth, common_clusters=common, tolerance=tolerance
        )
        expected_ownership = (
            {"OWNERSHIP_NOT_EVALUATED_BY_DESIGN"}
            if depth < 2
            else {
                "OWNERSHIP_BOUND_PASS",
                "OWNERSHIP_BOUND_FAIL_CERTIFICATE_INCONCLUSIVE",
                "OWNERSHIP_NOT_EVALUABLE",
                "DIRECT_OVERLAP_PASS",
                "DIRECT_OVERLAP_FAIL",
            }
        )
        if diagnostics["ownership_status"] not in expected_ownership:
            raise ValueError("formal ownership status is absent, pending, or out of depth")
        ownership_payload = validate_partitioned_ownership_payload(
            diagnostics["ownership"]
        )
        if (
            ownership_payload["evaluation_depth"] != depth
            or ownership_payload["status"] != diagnostics["ownership_status"]
        ):
            raise ValueError("ownership object/status/depth binding mismatch")
        if depth == 2:
            required_ownership_context = {
                "exact_singlet_intervals",
                "exact_cluster_ids",
                "anchor_complete_through_fourth",
            }
            if (
                not isinstance(ownership_context, Mapping)
                or set(ownership_context) != required_ownership_context
                or type(
                    ownership_context["anchor_complete_through_fourth"]
                ) is not bool
            ):
                raise ValueError(
                    "depth-two replay requires the hash-bound exact-anchor context"
                )
            replayed_assessment = assess_partitioned_three_root_ownership(
                physical.energies,
                physical.direct_residual_norms,
                np.asarray(
                    diagnostics["physical_splus_per_ritz_norms"],
                    dtype=np.float64,
                ),
                common,
                np.asarray(
                    [cluster.residual_operator_norm for cluster in physical_clusters],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        _finite_json_scalar(
                            stored["splus_operator_norm"],
                            "stored physical-cluster Splus operator norm",
                        )
                        for stored in stored_physical_clusters
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    ownership_context["exact_singlet_intervals"],
                    dtype=np.float64,
                ),
                exact_cluster_ids=np.asarray(
                    ownership_context["exact_cluster_ids"]
                ),
                anchor_complete_through_fourth=ownership_context[
                    "anchor_complete_through_fourth"
                ],
                evaluation_depth=2,
                minimum_owned_weight=0.9,
            )
            replayed_ownership = serialize_partitioned_ownership(
                replayed_assessment
            )
            _assert_json_semantic_close(
                ownership_payload,
                replayed_ownership,
                label="depth-two ownership",
                tolerance=tolerance,
            )
        integer_ledger = {
            "physical_batched_h_actions": depth + 1,
            "null_batched_h_actions": depth + 1,
            "total_batched_h_actions": 2 * (depth + 1),
            "physical_scalar_equivalent_h_mvm": 3 * (depth + 1),
            "null_scalar_equivalent_h_mvm": 3 * (depth + 1),
            "total_committed_scalar_MVM": 6 * (depth + 1),
            "exact_small_probe_setup_h_actions_disclosed": 0,
            "total_splus_committed_batched_actions_this_depth": 2 if depth in (0, 2) else 0,
            "total_splus_semantic_replay_batched_actions": 2 if depth > 0 else 0,
            "total_splus_scalar_columns_this_depth": 6 * (depth + 1) if depth in (0, 2) else 0,
            "attempted_actual_scalar_MVM": 6,
            "attempted_started_batched_actions": 2,
            "attempted_completed_batched_actions": 2,
            "attempted_abandoned_batched_actions": 0,
            "attempted_retried_batched_actions": 0,
            "attempted_splus_started_batched_actions": 2 if depth < 2 else 4,
            "attempted_splus_completed_batched_actions": 2 if depth < 2 else 4,
            "attempted_splus_abandoned_batched_actions": 0,
            "observed_openmp_team_size": 40,
        }
        for key, expected in integer_ledger.items():
            if isinstance(diagnostics[key], bool) or diagnostics[key] != expected:
                raise ValueError(f"depth {depth} work ledger mismatch at {key}")
        physical_lowdin = max(
            _orthonormality_error(physical_frontier),
            float(np.max(np.abs(physical_q[:, :-3].conj().T @ physical_frontier), initial=0.0)) if depth else 0.0,
        )
        null_lowdin = max(
            _orthonormality_error(null_frontier),
            float(np.max(np.abs(null_q[:, :-3].conj().T @ null_frontier), initial=0.0)) if depth else 0.0,
        )
        _assert_close(diagnostics["frontier_lowdin_orthonormality_error"], physical_lowdin, "physical frontier Lowdin error", tolerance)
        _assert_close(diagnostics["null_frontier_lowdin_orthonormality_error"], null_lowdin, "null frontier Lowdin error", tolerance)
        rss = _finite_json_scalar(diagnostics["process_max_rss_gib"], "process MaxRSS")
        wall = _finite_json_scalar(diagnostics["process_wall_seconds"], "process wall time")
        if rss < 0.0 or rss > 89.6 or wall < 0.0 or wall > 18000.0:
            raise ValueError("per-depth resource telemetry exceeds the registered envelope")
        reports.append(
            {
                "depth": depth,
                "accepted_dimension": 3 * (depth + 1),
                "physical_frontier_projector_replay_error": physical_novelty,
                "null_frontier_projector_replay_error": null_novelty,
                "full_span_projector_frobenius_error": span_error,
                "ritz_subspace_projector_frobenius_error": ritz_error,
                "common_projected_clusters": common,
                "ownership_status": diagnostics["ownership_status"],
                "total_committed_scalar_MVM": 6 * (depth + 1),
            }
        )
    return tuple(reports)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_unique(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key rejected: {key}")
            result[key] = value
        return result

    payload = json.loads(
        Path(path).read_text(),
        object_pairs_hook=hook,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant rejected: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint JSON must be an object")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_checkpoint_latest(root: Path, payload: Mapping[str, Any]) -> None:
    """Publish LATEST with its crash partial confined to markers/."""

    root = Path(root)
    markers = root / "markers"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".LATEST.json.", suffix=".partial", dir=markers
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, root / "LATEST.json")
        _fsync_directory(markers)
        _fsync_directory(root)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


_STABLE_CHECKPOINT_IDENTITY_FIELDS = {
    "config_sha256",
    "source_manifest_sha256",
    "input_sha256",
    "seed_fill_binary_sha256",
    "block_krylov_binary_sha256",
    "xdiag_commit",
    "run_id",
    "prepared_sha256",
    "full_seed_fill_done_sha256",
}
_ATTEMPT_IDENTITY_FIELDS = {
    "attempt_id",
    "slurm_job_id",
    "node_list",
    "allocation_id",
    "runtime_manifest_sha256",
    "allocation_shape",
}


def _require_identity(
    identity: Mapping[str, str], required: set[str], label: str
) -> dict[str, str]:
    if set(identity) != required or any(
        not isinstance(value, str) or not value for value in identity.values()
    ):
        raise ValueError(f"{label} fields differ from the frozen v8 schema")
    for key, value in identity.items():
        if key.endswith("_sha256") and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} contains malformed SHA256 at {key}")
    if "attempt_id" in identity and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity["attempt_id"]
    ) is None:
        raise ValueError(f"{label} contains unsafe attempt ID")
    return dict(identity)


def _generation_array_names(generation: int) -> set[str]:
    names = {
        f"PHYSICAL_F_{generation}",
        f"PHYSICAL_HF_{generation}",
        f"NULL_F_{generation}",
        f"NULL_HF_{generation}",
    }
    if generation == 0:
        names.add("PHYSICAL_CARRIERS")
    return names


def _write_raw_real64(path: Path, values: np.ndarray) -> tuple[list[int], str]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("v8 state arrays must be dimension-by-three matrices")
    if np.iscomplexobj(array) and np.any(np.imag(array) != 0.0):
        raise ValueError("v8 state payloads are strictly real64")
    real = np.asarray(np.real(array), dtype="<f8", order="F")
    if np.any(~np.isfinite(real)):
        raise FloatingPointError("v8 state payload contains nonfinite values")
    with path.open("wb") as handle:
        handle.write(real.tobytes(order="F"))
        handle.flush()
        os.fsync(handle.fileno())
    return [int(real.shape[0]), 3], sha256_file(path)


def _read_raw_real64(path: Path, shape: list[int]) -> np.ndarray:
    if (
        len(shape) != 2
        or shape[1] != 3
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 1
    ):
        raise ValueError("invalid v8 state shape")
    expected_bytes = shape[0] * 3 * 8
    if path.stat().st_size != expected_bytes:
        raise ValueError("v8 state payload byte count mismatch")
    values = np.fromfile(path, dtype="<f8")
    result = values.reshape((shape[0], 3), order="F")
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("checkpoint contains nonfinite state")
    return np.asarray(result, dtype=np.float64)


def commit_checkpoint_generation(
    root: Path,
    generation: int,
    *,
    arrays: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
    stable_identity: Mapping[str, str],
    attempt_identity: Mapping[str, str],
    prior_semantic_replay: Callable[[tuple[dict[str, np.ndarray], ...], tuple[dict[str, Any], ...]], None] | None = None,
) -> dict[str, Any]:
    """Durably commit one v8 increment-only paired-arm generation."""

    if generation not in (0, 1, 2) or isinstance(generation, bool):
        raise ValueError("v8 checkpoint generation must be depth 0, 1, or 2")
    stable = _require_identity(
        stable_identity, _STABLE_CHECKPOINT_IDENTITY_FIELDS, "stable identity"
    )
    attempt = _require_identity(
        attempt_identity, _ATTEMPT_IDENTITY_FIELDS, "attempt identity"
    )
    expected_arrays = _generation_array_names(generation)
    if set(arrays) != expected_arrays:
        raise ValueError("incremental checkpoint has missing or extra state arrays")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("checkpoint diagnostics must be a mapping")

    root = Path(root)
    generations = root / "generations"
    markers = root / "markers"
    generations.mkdir(parents=True, exist_ok=True)
    markers.mkdir(parents=True, exist_ok=True)
    final_dir = generations / f"generation-{generation:04d}"
    final_marker = markers / f"generation-{generation:04d}.json"
    if final_dir.exists() or final_marker.exists():
        raise FileExistsError("checkpoint generation is immutable")
    if generation:
        previous = markers / f"generation-{generation - 1:04d}.json"
        if not previous.is_file():
            raise ValueError("previous generation marker is missing")
        validate_checkpoint_chain(
            root,
            through_generation=generation - 1,
            expected_stable_identity=stable,
            semantic_replay=prior_semantic_replay,
            require_semantic_replay=True,
        )
        parent_sha: str | None = sha256_file(previous)
    else:
        parent_sha = None

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".generation-{generation:04d}.",
            suffix=".partial",
            dir=generations,
        )
    )
    shapes: dict[str, list[int]] = {}
    files: dict[str, str] = {}
    try:
        for name in sorted(expected_arrays):
            filename = f"{name}.bin"
            shape, digest = _write_raw_real64(temporary / filename, arrays[name])
            shapes[filename] = shape
            files[filename] = digest
        _fsync_directory(temporary)
        os.replace(temporary, final_dir)
        _fsync_directory(generations)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    marker = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-checkpoint-generation-v8",
        "generation": generation,
        "stable_identity": stable,
        "attempt_identity": attempt,
        "parent_marker_sha256": parent_sha,
        "files_sha256": files,
        "file_shapes": shapes,
        "diagnostics": dict(diagnostics),
    }
    _atomic_json(final_marker, marker)
    marker_sha = sha256_file(final_marker)
    _atomic_checkpoint_latest(
        root,
        {
            "schema_version": 8,
            "stage": "P9v2-D1-T-C0-checkpoint-latest-v8",
            "generation": generation,
            "generation_marker": str(final_marker.relative_to(root)),
            "generation_marker_sha256": marker_sha,
        },
    )
    return marker


def validate_checkpoint_chain(
    root: Path,
    *,
    through_generation: int,
    expected_stable_identity: Mapping[str, str],
    semantic_replay: Callable[[tuple[dict[str, np.ndarray], ...], tuple[dict[str, Any], ...]], None] | None = None,
    require_semantic_replay: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Validate the full v8 hash chain and optionally replay every semantic."""

    if through_generation not in (0, 1, 2) or isinstance(through_generation, bool):
        raise ValueError("through_generation must be depth 0, 1, or 2")
    stable = _require_identity(
        expected_stable_identity,
        _STABLE_CHECKPOINT_IDENTITY_FIELDS,
        "stable identity",
    )
    if require_semantic_replay and semantic_replay is None:
        raise ValueError("formal semantic replay callback is mandatory")
    root = Path(root)
    marker_root = root / "markers"
    payload_root = root / "generations"
    if (
        root.is_symlink()
        or not root.is_dir()
        or marker_root.is_symlink()
        or not marker_root.is_dir()
        or payload_root.is_symlink()
        or not payload_root.is_dir()
    ):
        raise FileNotFoundError("checkpoint marker/generation directories are missing")
    root_entries = {entry.name for entry in root.iterdir()}
    if not {"generations", "markers"} <= root_entries or not root_entries <= {
        "generations", "markers", "LATEST.json"
    }:
        raise ValueError("checkpoint root has an unknown or missing entry")
    if "LATEST.json" in root_entries and (
        (root / "LATEST.json").is_symlink()
        or not (root / "LATEST.json").is_file()
    ):
        raise ValueError("checkpoint LATEST is not a regular file")
    committed_markers: set[int] = set()
    for entry in marker_root.iterdir():
        if entry.is_symlink():
            raise ValueError("checkpoint marker entry may not be a symlink")
        if entry.is_file() and (
            re.fullmatch(
                r"\.generation-\d{4}\.json\.[A-Za-z0-9_.-]+\.partial",
                entry.name,
            )
            or re.fullmatch(
                r"\.LATEST\.json\.[A-Za-z0-9_.-]+\.partial", entry.name
            )
        ):
            continue
        match = re.fullmatch(r"generation-(\d{4})\.json", entry.name)
        if not entry.is_file() or match is None:
            raise ValueError("unexpected checkpoint marker entry")
        committed_markers.add(int(match.group(1)))
    committed_payloads: set[int] = set()
    for entry in payload_root.iterdir():
        if entry.is_symlink():
            raise ValueError("checkpoint payload entry may not be a symlink")
        if entry.is_dir() and re.fullmatch(
            r"\.generation-\d{4}\.[A-Za-z0-9_.-]+\.partial", entry.name
        ):
            continue
        match = re.fullmatch(r"generation-(\d{4})", entry.name)
        if not entry.is_dir() or match is None:
            raise ValueError("unexpected checkpoint generation entry")
        committed_payloads.add(int(match.group(1)))
    exact_committed = set(range(through_generation + 1))
    if committed_markers != exact_committed or committed_payloads != exact_committed:
        raise ValueError("checkpoint chain has gap/fork/future/orphan")
    previous_sha: str | None = None
    markers: list[dict[str, Any]] = []
    states: list[dict[str, np.ndarray]] = []
    for generation in range(through_generation + 1):
        marker_path = root / "markers" / f"generation-{generation:04d}.json"
        payload_dir = root / "generations" / f"generation-{generation:04d}"
        if not marker_path.is_file() or not payload_dir.is_dir():
            raise ValueError("checkpoint chain has a marker/payload gap")
        marker = _read_json_unique(marker_path)
        required_marker_keys = {
            "schema_version",
            "stage",
            "generation",
            "stable_identity",
            "attempt_identity",
            "parent_marker_sha256",
            "files_sha256",
            "file_shapes",
            "diagnostics",
        }
        if set(marker) != required_marker_keys:
            raise ValueError("checkpoint marker has extra or missing fields")
        if (
            marker["schema_version"] != 8
            or marker["stage"] != "P9v2-D1-T-C0-checkpoint-generation-v8"
            or marker["generation"] != generation
            or marker["stable_identity"] != stable
            or marker["parent_marker_sha256"] != previous_sha
        ):
            raise ValueError("checkpoint marker identity/parent mismatch")
        _require_identity(
            marker["attempt_identity"], _ATTEMPT_IDENTITY_FIELDS, "attempt identity"
        )
        files = marker["files_sha256"]
        shapes = marker["file_shapes"]
        expected_files = {f"{name}.bin" for name in _generation_array_names(generation)}
        if not isinstance(files, dict) or set(files) != expected_files or set(shapes) != expected_files:
            raise ValueError("checkpoint payload manifest differs from v8 allowlist")
        payload_children = list(payload_dir.iterdir())
        actual_names = {path.name for path in payload_children}
        if (
            payload_dir.is_symlink()
            or actual_names != expected_files
            or any(path.is_symlink() or not path.is_file() for path in payload_children)
        ):
            raise ValueError("checkpoint payload has missing or extra files")
        generation_state: dict[str, np.ndarray] = {}
        for filename in sorted(expected_files):
            path = payload_dir / filename
            if sha256_file(path) != files[filename]:
                raise ValueError(f"checkpoint hash mismatch: {filename}")
            generation_state[filename.removesuffix(".bin")] = _read_raw_real64(
                path, shapes[filename]
            )
        if len({tuple(value.shape) for value in generation_state.values()}) != 1:
            raise ValueError("paired checkpoint state shapes differ")
        previous_sha = sha256_file(marker_path)
        markers.append(marker)
        states.append(generation_state)
    if semantic_replay is not None:
        semantic_replay(tuple(states), tuple(markers))
    return tuple(markers)


def load_checkpoint_generation(
    root: Path,
    generation: int | None = None,
    *,
    expected_stable_identity: Mapping[str, str],
    semantic_replay: Callable[[tuple[dict[str, np.ndarray], ...], tuple[dict[str, Any], ...]], None] | None = None,
    require_semantic_replay: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one increment only after the complete historical chain validates."""

    root = Path(root)
    latest_path = root / "LATEST.json"
    if not latest_path.is_file():
        raise FileNotFoundError("checkpoint LATEST is missing")
    latest = _read_json_unique(latest_path)
    if set(latest) != {
        "schema_version", "stage", "generation", "generation_marker", "generation_marker_sha256"
    } or latest.get("schema_version") != 8:
        raise ValueError("checkpoint LATEST schema mismatch")
    latest_generation = int(latest["generation"])
    selected = latest_generation if generation is None else generation
    if selected != latest_generation:
        raise ValueError("v8 load cannot bypass later committed generations")
    chain = validate_checkpoint_chain(
        root,
        through_generation=selected,
        expected_stable_identity=expected_stable_identity,
        semantic_replay=semantic_replay,
        require_semantic_replay=require_semantic_replay,
    )
    marker_path = root / "markers" / f"generation-{selected:04d}.json"
    expected_marker_path = f"markers/generation-{selected:04d}.json"
    if latest["stage"] != "P9v2-D1-T-C0-checkpoint-latest-v8" or latest["generation_marker"] != expected_marker_path:
        raise ValueError("LATEST stage/path mismatch")
    if latest["generation_marker_sha256"] != sha256_file(marker_path):
        raise ValueError("LATEST marker hash mismatch")
    payload_dir = root / "generations" / f"generation-{selected:04d}"
    marker = chain[-1]
    arrays = {
        filename.removesuffix(".bin"): _read_raw_real64(
            payload_dir / filename, marker["file_shapes"][filename]
        )
        for filename in marker["files_sha256"]
    }
    return arrays, marker


def promote_lagging_latest(
    root: Path,
    *,
    expected_stable_identity: Mapping[str, str],
    semantic_replay: Callable[[tuple[dict[str, np.ndarray], ...], tuple[dict[str, Any], ...]], None],
) -> dict[str, Any]:
    """Promote only one complete valid v8 chain; recognized partials are ignored."""

    root = Path(root)
    latest_path = root / "LATEST.json"
    latest: dict[str, Any] | None = None
    if latest_path.is_file():
        latest = _read_json_unique(latest_path)
        if set(latest) != {
            "schema_version", "stage", "generation", "generation_marker",
            "generation_marker_sha256",
        } or latest.get("schema_version") != 8 or latest.get("stage") != "P9v2-D1-T-C0-checkpoint-latest-v8":
            raise ValueError("checkpoint LATEST schema/stage mismatch")
        current = int(latest.get("generation", -1))
    else:
        current = -1
    marker_dir = root / "markers"
    payload_root = root / "generations"
    marker_indices: set[int] = set()
    for path in marker_dir.iterdir():
        if path.is_symlink():
            raise ValueError("checkpoint promotion refuses a symlink marker")
        if path.is_file() and (
            re.fullmatch(r"\.generation-\d{4}\.json\.[A-Za-z0-9_.-]+\.partial", path.name)
            or re.fullmatch(r"\.LATEST\.json\.[A-Za-z0-9_.-]+\.partial", path.name)
        ):
            continue
        match = re.fullmatch(r"generation-(\d{4})\.json", path.name)
        if not path.is_file() or match is None:
            raise ValueError("unexpected checkpoint marker entry")
        marker_indices.add(int(match.group(1)))
    payload_indices: set[int] = set()
    for path in payload_root.iterdir():
        if path.is_symlink():
            raise ValueError("checkpoint promotion refuses a symlink payload")
        if path.is_dir() and re.fullmatch(r"\.generation-\d{4}\.[A-Za-z0-9_.-]+\.partial", path.name):
            continue
        match = re.fullmatch(r"generation-(\d{4})", path.name)
        if not path.is_dir() or match is None:
            raise ValueError("unexpected checkpoint payload entry")
        payload_indices.add(int(match.group(1)))
    if not marker_indices or marker_indices != payload_indices:
        raise ValueError("checkpoint marker/payload fork or orphan")
    maximum = max(marker_indices)
    if maximum > 2 or marker_indices != set(range(maximum + 1)):
        raise ValueError("checkpoint chain has a gap or out-of-contract depth")
    if current > maximum:
        raise ValueError("LATEST is ahead of the valid chain")
    validate_checkpoint_chain(
        root,
        through_generation=maximum,
        expected_stable_identity=expected_stable_identity,
        semantic_replay=semantic_replay,
        require_semantic_replay=True,
    )
    if current >= 0:
        assert latest is not None
        current_marker = marker_dir / f"generation-{current:04d}.json"
        if latest.get("generation_marker") != f"markers/generation-{current:04d}.json":
            raise ValueError("current LATEST marker path mismatch")
        if latest.get("generation_marker_sha256") != sha256_file(current_marker):
            raise ValueError("current LATEST hash mismatch")
    if current == maximum and latest is not None:
        return latest
    final_marker = marker_dir / f"generation-{maximum:04d}.json"
    promoted = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-checkpoint-latest-v8",
        "generation": maximum,
        "generation_marker": str(final_marker.relative_to(root)),
        "generation_marker_sha256": sha256_file(final_marker),
    }
    _atomic_checkpoint_latest(root, promoted)
    return promoted


def quarantine_unmarked_checkpoint_tail(
    root: Path, *, recovery_id: str
) -> Path | None:
    """Make one post-payload/pre-marker crash window explicitly uncommitted.

    The function never deletes or accepts the orphan.  It only renames the
    unique next-depth final-name payload directory to the recognized partial
    namespace so both arm actions must be retried and disclosed.
    """

    if re.fullmatch(r"[A-Za-z0-9_.-]+", recovery_id) is None:
        raise ValueError("recovery_id contains unsafe characters")
    root = Path(root)
    markers_root = root / "markers"
    payload_root = root / "generations"
    if not markers_root.is_dir() or not payload_root.is_dir():
        raise FileNotFoundError("checkpoint marker/generation directories are missing")
    markers: set[int] = set()
    payloads: set[int] = set()
    for path in markers_root.iterdir():
        if path.is_symlink():
            raise ValueError("checkpoint recovery refuses a symlink marker")
        if path.is_file() and (
            re.fullmatch(r"\.generation-\d{4}\.json\.[A-Za-z0-9_.-]+\.partial", path.name)
            or re.fullmatch(r"\.LATEST\.json\.[A-Za-z0-9_.-]+\.partial", path.name)
        ):
            continue
        match = re.fullmatch(r"generation-(\d{4})\.json", path.name)
        if not path.is_file() or match is None:
            raise ValueError("unexpected checkpoint marker entry")
        markers.add(int(match.group(1)))
    for path in payload_root.iterdir():
        if path.is_symlink():
            raise ValueError("checkpoint recovery refuses a symlink payload")
        if path.is_dir() and re.fullmatch(r"\.generation-\d{4}\.[A-Za-z0-9_.-]+\.partial", path.name):
            continue
        match = re.fullmatch(r"generation-(\d{4})", path.name)
        if not path.is_dir() or match is None:
            raise ValueError("unexpected checkpoint payload entry")
        payloads.add(int(match.group(1)))
    if markers - payloads:
        raise ValueError("published checkpoint marker lacks its payload")
    orphans = payloads - markers
    if not orphans:
        return None
    expected = len(markers)
    if markers != set(range(expected)) or orphans != {expected} or expected > 2:
        raise ValueError("checkpoint mismatch is not one recoverable tail")
    source = payload_root / f"generation-{expected:04d}"
    target = payload_root / f".generation-{expected:04d}.{recovery_id}.partial"
    if target.exists():
        raise FileExistsError("checkpoint recovery partial already exists")
    os.replace(source, target)
    _fsync_directory(payload_root)
    return target
