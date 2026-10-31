"""Wavefunction-free transport diagnostics for boundary-channel gauges.

Successive left-associated autoregressive cuts move one physical site from the
hidden block into the revealed prefix.  This module compares two exact numeric
representations of the crossing coupling matrix ``J_LR``:

* the minimal singular-vector gauge, whose channel basis may rotate; and
* the deterministic hidden-endpoint gauge ``S_i . sum_j J_ij S_j``.

Only graph couplings and doubled-label rank metadata are used.  No model,
amplitude, conditional suffix, or observable enters these diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary_operator_channels import BoundaryRecouplingPlan


def _site_rows(
    values: np.ndarray, sites: np.ndarray, selected: tuple[int, ...]
) -> np.ndarray:
    position = {int(site): index for index, site in enumerate(sites)}
    concrete = np.asarray(values, dtype=np.float64)
    indices = np.asarray([position[site] for site in selected], dtype=np.int64)
    return concrete[indices, :]


def _orthonormal_span(
    values: np.ndarray, tolerance: float
) -> tuple[np.ndarray, int]:
    concrete = np.asarray(values, dtype=np.float64)
    if concrete.ndim != 2:
        raise ValueError("subspace values must be a matrix")
    if concrete.size == 0:
        return np.zeros((concrete.shape[0], 0), dtype=np.float64), 0
    left, singular, _ = np.linalg.svd(concrete, full_matrices=False)
    threshold = tolerance * max(1.0, float(singular[0]) if len(singular) else 0.0)
    rank = int(np.sum(singular > threshold))
    return left[:, :rank], rank


@dataclass(frozen=True)
class SubspaceTransport:
    """Principal-angle and rectangular Procrustes comparison on common sites."""

    common_sites: tuple[int, ...]
    old_restricted_rank: int
    new_restricted_rank: int
    principal_cosines: np.ndarray
    principal_angles_degrees: np.ndarray
    shared_procrustes_overlap: float
    dimension_penalized_procrustes_overlap: float
    normalized_procrustes_residual: float


def persistent_subspace_transport(
    old_vectors: np.ndarray,
    old_sites: np.ndarray,
    new_vectors: np.ndarray,
    new_sites: np.ndarray,
    *,
    rank_tolerance: float = 1.0e-12,
) -> SubspaceTransport:
    """Compare column spans after restricting both gauges to common endpoints."""

    if rank_tolerance < 0.0:
        raise ValueError("rank_tolerance must be non-negative")
    old_site_set = set(map(int, old_sites))
    new_site_set = set(map(int, new_sites))
    common = tuple(sorted(old_site_set & new_site_set))
    old = _site_rows(np.asarray(old_vectors), np.asarray(old_sites), common)
    new = _site_rows(np.asarray(new_vectors), np.asarray(new_sites), common)
    old_basis, old_rank = _orthonormal_span(old, rank_tolerance)
    new_basis, new_rank = _orthonormal_span(new, rank_tolerance)
    if old_rank == 0 or new_rank == 0:
        cosines = np.zeros(0, dtype=np.float64)
    else:
        cosines = np.linalg.svd(old_basis.T @ new_basis, compute_uv=False)
        cosines = np.clip(cosines, 0.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    shared_dimension = min(old_rank, new_rank)
    union_dimension = max(old_rank, new_rank)
    shared_overlap = (
        float(np.sum(cosines)) / shared_dimension if shared_dimension else 1.0
    )
    penalized_overlap = (
        float(np.sum(cosines)) / union_dimension if union_dimension else 1.0
    )
    denominator = old_rank + new_rank
    residual = (
        float(
            np.sqrt(
                max(0.0, old_rank + new_rank - 2.0 * float(np.sum(cosines)))
                / denominator
            )
        )
        if denominator
        else 0.0
    )
    return SubspaceTransport(
        common_sites=common,
        old_restricted_rank=old_rank,
        new_restricted_rank=new_rank,
        principal_cosines=cosines,
        principal_angles_degrees=angles,
        shared_procrustes_overlap=shared_overlap,
        dimension_penalized_procrustes_overlap=penalized_overlap,
        normalized_procrustes_residual=residual,
    )


@dataclass(frozen=True)
class SvdChannelMatch:
    """Gauge-invariant matching of individual rank-one SVD factors."""

    old_channel: int
    new_channel: int
    hidden_overlap: float
    prefix_overlap: float
    product_overlap: float


def _normalized_columns(values: np.ndarray) -> np.ndarray:
    concrete = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(concrete, axis=0)
    result = np.zeros_like(concrete)
    active = norms > 1.0e-14
    result[:, active] = concrete[:, active] / norms[active]
    return result


def match_svd_channels(
    old_plan: BoundaryRecouplingPlan,
    new_plan: BoundaryRecouplingPlan,
) -> tuple[SvdChannelMatch, ...]:
    """Match channels using both persistent hidden and prefix endpoints."""

    from scipy.optimize import linear_sum_assignment

    common_hidden = tuple(
        sorted(set(map(int, old_plan.hidden_sites)) & set(map(int, new_plan.hidden_sites)))
    )
    common_prefix = tuple(
        sorted(set(map(int, old_plan.prefix_sites)) & set(map(int, new_plan.prefix_sites)))
    )
    old_left = _normalized_columns(
        _site_rows(old_plan.left_vectors, old_plan.hidden_sites, common_hidden)
    )
    new_left = _normalized_columns(
        _site_rows(new_plan.left_vectors, new_plan.hidden_sites, common_hidden)
    )
    old_right = _normalized_columns(
        _site_rows(old_plan.right_vectors, old_plan.prefix_sites, common_prefix)
    )
    new_right = _normalized_columns(
        _site_rows(new_plan.right_vectors, new_plan.prefix_sites, common_prefix)
    )
    if old_plan.channel_rank == 0 or new_plan.channel_rank == 0:
        return ()
    hidden_overlap = np.abs(old_left.T @ new_left)
    prefix_overlap = np.abs(old_right.T @ new_right)
    score = hidden_overlap * prefix_overlap
    old_indices, new_indices = linear_sum_assignment(-score)
    return tuple(
        SvdChannelMatch(
            old_channel=int(old),
            new_channel=int(new),
            hidden_overlap=float(hidden_overlap[old, new]),
            prefix_overlap=float(prefix_overlap[old, new]),
            product_overlap=float(score[old, new]),
        )
        for old, new in sorted(zip(old_indices, new_indices, strict=True))
    )


@dataclass(frozen=True)
class EndpointChannelUpdate:
    """One persistent hidden-endpoint row under a cut advance."""

    hidden_site: int
    common_prefix_error: float
    added_prefix_sites: tuple[int, ...]
    added_coefficient_norm: float
    old_extended_norm: float
    new_norm: float
    cosine_overlap: float


def endpoint_gauge_matrix(plan: BoundaryRecouplingPlan) -> np.ndarray:
    """Reconstruct ``J_LR`` from deterministic hidden-endpoint rows."""

    result = np.zeros_like(plan.coupling_matrix)
    hidden_position = {
        int(site): position for position, site in enumerate(plan.hidden_sites)
    }
    for site, coefficients in zip(
        plan.endpoint_hidden_sites,
        plan.endpoint_right_coefficients,
        strict=True,
    ):
        result[hidden_position[int(site)]] = coefficients
    return result


def endpoint_channel_updates(
    old_plan: BoundaryRecouplingPlan,
    new_plan: BoundaryRecouplingPlan,
) -> tuple[EndpointChannelUpdate, ...]:
    """Compare stable endpoint rows and isolate newly appended coefficients."""

    old_position = {
        int(site): position
        for position, site in enumerate(old_plan.endpoint_hidden_sites)
    }
    new_position = {
        int(site): position
        for position, site in enumerate(new_plan.endpoint_hidden_sites)
    }
    old_prefix_position = {
        int(site): position for position, site in enumerate(old_plan.prefix_sites)
    }
    new_prefix_position = {
        int(site): position for position, site in enumerate(new_plan.prefix_sites)
    }
    common_prefix = tuple(
        sorted(set(old_prefix_position) & set(new_prefix_position))
    )
    added_prefix = tuple(sorted(set(new_prefix_position) - set(old_prefix_position)))
    updates = []
    for site in sorted(set(old_position) & set(new_position)):
        old = old_plan.endpoint_right_coefficients[old_position[site]]
        new = new_plan.endpoint_right_coefficients[new_position[site]]
        common_error = max(
            (
                abs(
                    float(old[old_prefix_position[prefix_site]])
                    - float(new[new_prefix_position[prefix_site]])
                )
                for prefix_site in common_prefix
            ),
            default=0.0,
        )
        extended = np.zeros(len(new_plan.prefix_sites), dtype=np.float64)
        for prefix_site in common_prefix:
            extended[new_prefix_position[prefix_site]] = old[
                old_prefix_position[prefix_site]
            ]
        added = np.asarray(
            [new[new_prefix_position[prefix_site]] for prefix_site in added_prefix]
        )
        old_norm = float(np.linalg.norm(extended))
        new_norm = float(np.linalg.norm(new))
        cosine = (
            float(np.dot(extended, new) / (old_norm * new_norm))
            if old_norm > 0.0 and new_norm > 0.0
            else 1.0
        )
        updates.append(
            EndpointChannelUpdate(
                hidden_site=site,
                common_prefix_error=float(common_error),
                added_prefix_sites=added_prefix,
                added_coefficient_norm=float(np.linalg.norm(added)),
                old_extended_norm=old_norm,
                new_norm=new_norm,
                cosine_overlap=cosine,
            )
        )
    return tuple(updates)


@dataclass(frozen=True)
class SvdTruncationAudit:
    requested_rank: int
    retained_rank: int
    cumulative_squared_weight: float
    relative_frobenius_error: float
    maximum_coupling_error: float


def svd_truncation_audit(
    plan: BoundaryRecouplingPlan, requested_rank: int
) -> SvdTruncationAudit:
    """Measure a fixed-rank approximation of the physical coupling matrix."""

    if isinstance(requested_rank, bool) or not isinstance(requested_rank, int):
        raise TypeError("requested_rank must be an integer")
    if requested_rank < 0:
        raise ValueError("requested_rank must be non-negative")
    retained = min(requested_rank, plan.channel_rank)
    if retained:
        approximation = (
            plan.left_vectors[:, :retained]
            * plan.singular_values[None, :retained]
        ) @ plan.right_vectors[:, :retained].T
    else:
        approximation = np.zeros_like(plan.coupling_matrix)
    total_squared = float(np.sum(plan.coupling_matrix**2))
    retained_squared = float(np.sum(plan.singular_values[:retained] ** 2))
    error = plan.coupling_matrix - approximation
    return SvdTruncationAudit(
        requested_rank=requested_rank,
        retained_rank=retained,
        cumulative_squared_weight=(
            retained_squared / total_squared if total_squared else 1.0
        ),
        relative_frobenius_error=(
            float(np.linalg.norm(error)) / np.sqrt(total_squared)
            if total_squared
            else 0.0
        ),
        maximum_coupling_error=float(np.max(np.abs(error), initial=0.0)),
    )


@dataclass(frozen=True)
class BoundaryGaugeTransport:
    """Complete numeric transport audit for one successive-cut step."""

    n_sites: int
    old_decision: int
    new_decision: int
    migrating_sites: tuple[int, ...]
    old_svd_rank: int
    new_svd_rank: int
    svd_rank_births: int
    svd_rank_deaths: int
    hidden_subspace: SubspaceTransport
    prefix_subspace: SubspaceTransport
    svd_channel_matches: tuple[SvdChannelMatch, ...]
    old_endpoint_sites: tuple[int, ...]
    new_endpoint_sites: tuple[int, ...]
    endpoint_births: tuple[int, ...]
    endpoint_deaths: tuple[int, ...]
    endpoint_updates: tuple[EndpointChannelUpdate, ...]
    common_coupling_error: float


def compile_boundary_gauge_transport(
    old_plan: BoundaryRecouplingPlan,
    new_plan: BoundaryRecouplingPlan,
    *,
    rank_tolerance: float = 1.0e-12,
) -> BoundaryGaugeTransport:
    """Compile wavefunction-free gauge transport between consecutive cuts."""

    if old_plan.n_sites != new_plan.n_sites:
        raise ValueError("successive plans must have the same site count")
    if new_plan.decision != old_plan.decision + 1:
        raise ValueError("plans must describe consecutive decisions")
    removed_hidden = tuple(
        sorted(set(map(int, old_plan.hidden_sites)) - set(map(int, new_plan.hidden_sites)))
    )
    added_prefix = tuple(
        sorted(set(map(int, new_plan.prefix_sites)) - set(map(int, old_plan.prefix_sites)))
    )
    if not removed_hidden or added_prefix != removed_hidden:
        raise ValueError(
            "a successive left cut must migrate the same hidden sites into the prefix"
        )

    common_hidden = tuple(map(int, new_plan.hidden_sites))
    common_prefix = tuple(map(int, old_plan.prefix_sites))
    old_hidden_position = {
        int(site): position for position, site in enumerate(old_plan.hidden_sites)
    }
    new_hidden_position = {
        int(site): position for position, site in enumerate(new_plan.hidden_sites)
    }
    old_prefix_position = {
        int(site): position for position, site in enumerate(old_plan.prefix_sites)
    }
    new_prefix_position = {
        int(site): position for position, site in enumerate(new_plan.prefix_sites)
    }
    common_error = max(
        (
            abs(
                float(
                    old_plan.coupling_matrix[
                        old_hidden_position[first], old_prefix_position[second]
                    ]
                )
                - float(
                    new_plan.coupling_matrix[
                        new_hidden_position[first], new_prefix_position[second]
                    ]
                )
            )
            for first in common_hidden
            for second in common_prefix
        ),
        default=0.0,
    )

    old_endpoints = tuple(map(int, old_plan.endpoint_hidden_sites))
    new_endpoints = tuple(map(int, new_plan.endpoint_hidden_sites))
    old_endpoint_set = set(old_endpoints)
    new_endpoint_set = set(new_endpoints)
    return BoundaryGaugeTransport(
        n_sites=old_plan.n_sites,
        old_decision=old_plan.decision,
        new_decision=new_plan.decision,
        migrating_sites=removed_hidden,
        old_svd_rank=old_plan.channel_rank,
        new_svd_rank=new_plan.channel_rank,
        svd_rank_births=max(0, new_plan.channel_rank - old_plan.channel_rank),
        svd_rank_deaths=max(0, old_plan.channel_rank - new_plan.channel_rank),
        hidden_subspace=persistent_subspace_transport(
            old_plan.left_vectors,
            old_plan.hidden_sites,
            new_plan.left_vectors,
            new_plan.hidden_sites,
            rank_tolerance=rank_tolerance,
        ),
        prefix_subspace=persistent_subspace_transport(
            old_plan.right_vectors,
            old_plan.prefix_sites,
            new_plan.right_vectors,
            new_plan.prefix_sites,
            rank_tolerance=rank_tolerance,
        ),
        svd_channel_matches=match_svd_channels(old_plan, new_plan),
        old_endpoint_sites=old_endpoints,
        new_endpoint_sites=new_endpoints,
        endpoint_births=tuple(sorted(new_endpoint_set - old_endpoint_set)),
        endpoint_deaths=tuple(sorted(old_endpoint_set - new_endpoint_set)),
        endpoint_updates=endpoint_channel_updates(old_plan, new_plan),
        common_coupling_error=float(common_error),
    )
