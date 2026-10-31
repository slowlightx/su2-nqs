"""Resolvent-dressed interface carriers for the strict triangle-patch tree.

This opt-in initializer turns the exact two-cluster mechanism audit into a
minimal scalable ansatz change.  On the open ``2 x 6`` reference cut, a
boundary exchange acting on two six-site singlets enters only
``q_left=q_right=2``.  In the complete nine-dimensional triplet multiplicity
space, the first-order response solves

``K X + X K = A``,

where ``K=H_cluster-E_ground`` and ``A`` is the reduced boundary action.
The leading left and right singular carriers of ``X`` are generally
different.  We orthonormalize their two-dimensional span, install that basis
as the two local ``q=2`` feature channels, and return fixed left/right frames
which reconstruct the two dressed carriers when the six-site support is
consumed by the upper tree.

No global parameter bank is enlarged.  For ``feature_dim=2`` the trainable
parameter count is exactly unchanged.  The physical input is one canonical
horizontal interface; other interface orientations remain an explicit next
step rather than being hidden behind generic random channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..benchmarks.exact_diagonalization import heisenberg_sz_matrix
from ..benchmarks.two_cluster_virtual_processes import two_cluster_open_partition
from ..models.triangle_patch_clebsch_tree import (
    TrianglePatchClebschTreePlan,
    triangle_patch_pair_layout,
    triangle_patch_parameter_count,
)
from ..models.triangle_patch_interface_routing import (
    TrianglePatchInterfaceRouting,
)
from .triangle_patch_physical_init import triangle_patch_cluster_reference
from .triangle_patch_physical_multiplet_init import (
    _cluster_multiplet_basis,
    init_triangle_patch_cluster_multiplets,
)


jax.config.update("jax_enable_x64", True)


Geometry = Literal["square", "triangular"]


@dataclass(frozen=True)
class TrianglePatchBoundaryResponseReference:
    """Canonical rank-two local carrier data for one cluster interface."""

    geometry: Geometry
    j2_over_j1: float
    response_amplitude_matrix: np.ndarray
    correction_amplitude_matrix: np.ndarray
    correction_singular_values: np.ndarray
    left_coefficients: np.ndarray
    right_coefficients: np.ndarray
    carrier_basis: np.ndarray
    left_coordinates: np.ndarray
    right_coordinates: np.ndarray
    left_right_overlap: float
    carrier_basis_orthonormality_error: float
    sylvester_residual: float
    span_reconstruction_residual: float
    triplet_basis_orthonormality_error: float


def _checked_geometry(geometry: str) -> Geometry:
    if geometry not in {"square", "triangular"}:
        raise ValueError("geometry must be 'square' or 'triangular'")
    return geometry


def _checked_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("j2_over_j1 must be finite and nonnegative")
    return ratio


def _lower_highest_triplets(highest: np.ndarray) -> np.ndarray:
    """Map normalized ``q=2,m2=2`` columns to ``m2=0`` columns."""

    if highest.shape[0] != 64:
        raise ValueError("six-site highest-weight states require 64 rows")
    zero = np.zeros_like(highest)
    for state in range(64):
        row = highest[state]
        if not np.any(row):
            continue
        for site in range(6):
            mask = 1 << (5 - site)
            if state & mask:
                zero[state ^ mask] += row / np.sqrt(2.0)
    return zero


def _canonical_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, singular, right_adjoint = np.linalg.svd(matrix, full_matrices=False)
    right = right_adjoint.T
    for column in range(left.shape[1]):
        pivot = int(np.argmax(np.abs(left[:, column])))
        if left[pivot, column] < 0.0:
            left[:, column] *= -1.0
            right[:, column] *= -1.0
    return left, singular, right


def _canonical_two_carrier_basis(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    first = np.array(left, copy=True)
    pivot = int(np.argmax(np.abs(first)))
    if first[pivot] < 0.0:
        first *= -1.0
        left *= -1.0
    residual = right - first * float(first @ right)
    norm = float(np.linalg.norm(residual))
    if norm <= 1.0e-10:
        raise RuntimeError("left/right response carriers do not span two dimensions")
    second = residual / norm
    pivot = int(np.argmax(np.abs(second)))
    if second[pivot] < 0.0:
        second *= -1.0
    basis = np.column_stack((first, second))
    left_coordinates = basis.T @ left
    right_coordinates = basis.T @ right
    reconstruction = max(
        float(np.linalg.norm(basis @ left_coordinates - left)),
        float(np.linalg.norm(basis @ right_coordinates - right)),
    )
    return basis, left_coordinates, right_coordinates, reconstruction


@lru_cache(maxsize=None)
def _boundary_response_cached(
    geometry: Geometry, ratio: float
) -> TrianglePatchBoundaryResponseReference:
    geometry = _checked_geometry(geometry)
    ratio = _checked_ratio(ratio)
    cluster = triangle_patch_cluster_reference(geometry, ratio)
    partition = two_cluster_open_partition(geometry, ratio)
    highest, channels = _cluster_multiplet_basis(geometry, 2)
    if highest.shape != (64, 9) or len(channels) != 9:
        raise RuntimeError("the six-site triplet multiplicity must be nine")
    zero = _lower_highest_triplets(highest)
    triplet_error = max(
        float(np.linalg.norm(highest.T @ highest - np.eye(9))),
        float(np.linalg.norm(zero.T @ zero - np.eye(9))),
    )
    if triplet_error > 3.0e-12:
        raise RuntimeError("canonical triplet Schur members are not orthonormal")

    hamiltonian, basis = heisenberg_sz_matrix(6, cluster.bonds)
    if basis != tuple(range(64)):
        raise RuntimeError("the local Hamiltonian basis order changed")
    reduced = highest.T @ hamiltonian @ highest
    reduced = 0.5 * (reduced + reduced.T)
    gap_operator = reduced - cluster.energy * np.eye(9)
    gap_values, gap_vectors = np.linalg.eigh(gap_operator)
    if np.min(gap_values) <= 0.0:
        raise RuntimeError("the triplet resolvent is not positive")

    states = np.arange(64, dtype=np.int64)
    bits = np.asarray(
        [[(state >> (5 - site)) & 1 for site in range(6)] for state in states],
        dtype=np.int8,
    )
    local_sz = bits.astype(np.float64) - 0.5
    transitions = zero.T @ (cluster.computational_state[:, None] * local_sz)
    boundary = np.zeros((6, 6), dtype=np.float64)
    for bond in partition.intercluster_bonds:
        boundary[bond.first_local, bond.second_local] += bond.coupling
    response = np.sqrt(3.0) * transitions @ boundary @ transitions.T

    eigen_response = gap_vectors.T @ response @ gap_vectors
    denominators = gap_values[:, None] + gap_values[None, :]
    correction = gap_vectors @ (eigen_response / denominators) @ gap_vectors.T
    sylvester = gap_operator @ correction + correction @ gap_operator - response
    sylvester_residual = float(np.linalg.norm(sylvester) / np.linalg.norm(response))
    if sylvester_residual > 2.0e-12:
        raise RuntimeError("the dressed boundary response failed its resolvent equation")

    left, singular, right = _canonical_svd(correction)
    leading_left = np.asarray(left[:, 0], dtype=np.float64)
    leading_right = np.asarray(right[:, 0], dtype=np.float64)
    carrier_basis, left_coordinates, right_coordinates, span_residual = (
        _canonical_two_carrier_basis(leading_left, leading_right)
    )
    basis_error = float(
        np.linalg.norm(carrier_basis.T @ carrier_basis - np.eye(2))
    )
    if basis_error > 2.0e-12 or span_residual > 2.0e-12:
        raise RuntimeError("failed to build the two-carrier interface basis")

    arrays = (
        response,
        correction,
        singular,
        leading_left,
        leading_right,
        carrier_basis,
        left_coordinates,
        right_coordinates,
    )
    for array in arrays:
        array.setflags(write=False)
    return TrianglePatchBoundaryResponseReference(
        geometry=geometry,
        j2_over_j1=ratio,
        response_amplitude_matrix=response,
        correction_amplitude_matrix=correction,
        correction_singular_values=singular,
        left_coefficients=leading_left,
        right_coefficients=leading_right,
        carrier_basis=carrier_basis,
        left_coordinates=left_coordinates,
        right_coordinates=right_coordinates,
        left_right_overlap=float(leading_left @ leading_right),
        carrier_basis_orthonormality_error=basis_error,
        sylvester_residual=sylvester_residual,
        span_reconstruction_residual=span_residual,
        triplet_basis_orthonormality_error=triplet_error,
    )


def triangle_patch_boundary_response_reference(
    geometry: Geometry, j2_over_j1: float
) -> TrianglePatchBoundaryResponseReference:
    """Return a defensive copy of the canonical dressed carrier data."""

    reference = _boundary_response_cached(
        _checked_geometry(geometry), _checked_ratio(j2_over_j1)
    )
    copied = {}
    for name in (
        "response_amplitude_matrix",
        "correction_amplitude_matrix",
        "correction_singular_values",
        "left_coefficients",
        "right_coefficients",
        "carrier_basis",
        "left_coordinates",
        "right_coordinates",
    ):
        array = np.array(getattr(reference, name), copy=True)
        array.setflags(write=False)
        copied[name] = array
    return TrianglePatchBoundaryResponseReference(
        geometry=reference.geometry,
        j2_over_j1=reference.j2_over_j1,
        **copied,
        left_right_overlap=reference.left_right_overlap,
        carrier_basis_orthonormality_error=reference.carrier_basis_orthonormality_error,
        sylvester_residual=reference.sylvester_residual,
        span_reconstruction_residual=reference.span_reconstruction_residual,
        triplet_basis_orthonormality_error=reference.triplet_basis_orthonormality_error,
    )


def _rotation_with_selected_second_row(coordinates: np.ndarray) -> np.ndarray:
    selected = np.asarray(coordinates, dtype=np.float64)
    if selected.shape != (2,):
        raise ValueError("the minimal interface route requires two coordinates")
    selected = selected / np.linalg.norm(selected)
    return np.asarray(
        [[selected[1], -selected[0]], [selected[0], selected[1]]],
        dtype=np.float64,
    )


def triangle_patch_boundary_response_routing(
    plan: TrianglePatchClebschTreePlan,
    *,
    j2_over_j1: float,
) -> TrianglePatchInterfaceRouting:
    """Build fixed frames selecting the dressed carrier on each input leg."""

    if not isinstance(plan, TrianglePatchClebschTreePlan):
        raise TypeError("plan must be a TrianglePatchClebschTreePlan")
    if plan.feature_dim != 2:
        raise ValueError("the minimal boundary-response route requires feature_dim=2")
    if plan.maximum_intermediate_q < 2:
        raise ValueError("the boundary-response route requires retained q=2")
    reference = _boundary_response_cached(plan.geometry, _checked_ratio(j2_over_j1))
    return TrianglePatchInterfaceRouting(
        routed_q=2,
        support_size=6,
        left_rotation=_rotation_with_selected_second_row(
            reference.left_coordinates
        ),
        right_rotation=_rotation_with_selected_second_row(
            reference.right_coordinates
        ),
    )


def init_triangle_patch_boundary_response_carriers(
    key: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    *,
    j2_over_j1: float,
    background_scale: float = 0.05,
    route_gate_scale: float = 0.05,
) -> dict[str, tuple[jax.Array, ...]]:
    """Install a rank-two dressed ``q=2`` bank; leave all other data intact."""

    if not isinstance(plan, TrianglePatchClebschTreePlan):
        raise TypeError("plan must be a TrianglePatchClebschTreePlan")
    if plan.feature_dim != 2:
        raise ValueError("the minimal boundary-response bank requires feature_dim=2")
    ratio = _checked_ratio(j2_over_j1)
    params = init_triangle_patch_cluster_multiplets(
        key,
        plan,
        j2_over_j1=ratio,
        background_scale=background_scale,
        route_gate_scale=route_gate_scale,
    )
    reference = _boundary_response_cached(plan.geometry, ratio)
    layout = triangle_patch_pair_layout(plan)
    positions = [index for index, block in enumerate(layout) if block.q_parent == 2]
    dimension = sum(
        layout[index].left_width * layout[index].right_width for index in positions
    )
    if dimension != reference.carrier_basis.shape[0]:
        raise RuntimeError("q=2 patch layout and response basis disagree")
    arrays = [np.array(block, copy=True) for block in params["patch_pair"]]
    offset = 0
    for position in positions:
        block = arrays[position]
        size = block.shape[1] * block.shape[2]
        for feature in range(2):
            block[feature] = reference.carrier_basis[
                offset : offset + size, feature
            ].reshape(block.shape[1:])
        offset += size
    result = {
        "patch_pair": tuple(jnp.asarray(block) for block in arrays),
        "reduced": params["reduced"],
        "root": params["root"],
        "patch_gates": params["patch_gates"],
        "reduced_gates": params["reduced_gates"],
    }
    if triangle_patch_parameter_count(result) != triangle_patch_parameter_count(params):
        raise RuntimeError("the boundary-response bank changed the parameter count")
    return result


__all__ = [
    "TrianglePatchBoundaryResponseReference",
    "init_triangle_patch_boundary_response_carriers",
    "triangle_patch_boundary_response_reference",
    "triangle_patch_boundary_response_routing",
]
