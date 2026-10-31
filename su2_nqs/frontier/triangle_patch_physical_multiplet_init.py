"""Opt-in exact low-energy multiplet carriers for the triangle-patch tree.

The sealed short-range-singlet initializer intentionally remains untouched.
This module builds an independent complete two-triangle Schur oracle on one
open ``2 x 3`` cluster.  In each doubled total-spin sector ``q=0,2,4,6`` it
diagonalizes the physical cluster Hamiltonian in a highest-weight basis.  The
resulting reduced coefficients determine the whole SU(2) multiplet through
the repository's fixed Condon--Shortley tensors.

The opt-in initializer first calls the sealed initializer, then replaces only
its local feature-channel-one patch bank.  Thus the established ground
singlet, upper tree, root, route gates, and random-key consumption are
unchanged.  The installed carrier is the first excited singlet for ``q=0``
and the lowest multiplet for every retained ``q=2,4,6`` sector.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..benchmarks.exact_diagonalization import (
    heisenberg_sz_matrix,
    spin_squared_matrix,
)
from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.triangular_j1j2 import triangular_j1j2_bonds
from ..models.triangle_patch_clebsch_tree import (
    TrianglePatchClebschTreePlan,
    triangle_patch_pair_layout,
)
from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.local_schur import cell_schur_spec, cell_schur_tensor
from ..su2.quantum_numbers import coupling_range, magnetic_numbers, validate_q
from .triangle_patch_physical_init import init_triangle_patch_cluster_singlet


jax.config.update("jax_enable_x64", True)


Geometry = Literal["square", "triangular"]


@dataclass(frozen=True)
class TrianglePatchClusterMultipletReference:
    """One exact open-``2 x 3`` total-spin multiplet at highest weight."""

    geometry: Geometry
    j2_over_j1: float
    total_q: int
    eigen_index: int
    energy: float
    gap_above: float | None
    multiplicity_dimension: int
    bonds: tuple[tuple[int, int, float], ...]
    coefficients: np.ndarray
    highest_weight_state: np.ndarray
    basis_orthonormality_error: float
    eigen_residual: float
    spin_squared_residual: float


def _checked_geometry(geometry: str) -> Geometry:
    if geometry not in {"square", "triangular"}:
        raise ValueError("geometry must be 'square' or 'triangular'")
    return geometry


def _checked_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("j2_over_j1 must be finite and nonnegative")
    return ratio


def _patch_local_sites(geometry: Geometry) -> tuple[tuple[int, int, int], ...]:
    # Canonical local row-major order is (a,b,c,d,e,f), matching the sealed
    # initializer and the registered geometry patches.
    if geometry == "square":
        return ((0, 1, 3), (2, 4, 5))
    return ((0, 3, 4), (1, 2, 5))


def _bits_to_index(bits: np.ndarray) -> np.ndarray:
    return (
        4 * bits[:, 0].astype(np.int64)
        + 2 * bits[:, 1].astype(np.int64)
        + bits[:, 2].astype(np.int64)
    )


def _cluster_bonds(
    geometry: Geometry,
    ratio: float,
) -> tuple[tuple[int, int, float], ...]:
    bonds = (
        square_j1j2_bonds(2, 3, j1=1.0, j2=ratio, periodic=False)
        if geometry == "square"
        else triangular_j1j2_bonds(2, 3, j1=1.0, j2=ratio, periodic=False)
    )
    return tuple(bonds)


@lru_cache(maxsize=None)
def _cluster_multiplet_basis(
    geometry: Geometry,
    total_q: int,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    """Return the complete highest-weight Schur basis for one spin sector."""

    checked_geometry = _checked_geometry(geometry)
    checked_q = validate_q(total_q, name="total_q")
    if checked_q not in (0, 2, 4, 6):
        raise ValueError("six spin-1/2 sites support total_q in {0,2,4,6}")
    states = np.arange(64, dtype=np.int64)
    bits = np.asarray(
        [[(state >> (5 - site)) & 1 for site in range(6)] for state in states],
        dtype=np.int32,
    )
    left_sites, right_sites = _patch_local_sites(checked_geometry)
    left_index = _bits_to_index(bits[:, left_sites])
    right_index = _bits_to_index(bits[:, right_sites])
    table = cell_schur_tensor("triangular_triangle")
    spec = cell_schur_spec("triangular_triangle")
    columns: list[np.ndarray] = []
    channels: list[tuple[int, int, int, int]] = []
    for q_left in spec.q_values:
        for q_right in spec.q_values:
            if checked_q not in coupling_range(q_left, q_right):
                continue
            right_m_values = set(magnetic_numbers(q_right))
            for left_a in range(spec.multiplicity(q_left)):
                for right_a in range(spec.multiplicity(q_right)):
                    column = np.zeros(64, dtype=np.float64)
                    for left_m in magnetic_numbers(q_left):
                        right_m = checked_q - left_m
                        if right_m not in right_m_values:
                            continue
                        column += (
                            table[left_index, q_left, left_m + 3, left_a]
                            * table[
                                right_index,
                                q_right,
                                right_m + 3,
                                right_a,
                            ]
                            * clebsch_gordan(
                                q_left,
                                left_m,
                                q_right,
                                right_m,
                                checked_q,
                                checked_q,
                            )
                        )
                    columns.append(column)
                    channels.append((q_left, q_right, left_a, right_a))
    basis = np.column_stack(columns)
    expected_multiplicity = {0: 5, 2: 9, 4: 5, 6: 1}[checked_q]
    if basis.shape != (64, expected_multiplicity):
        raise RuntimeError("the complete triangle carriers lost a spin sector")
    error = float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1])))
    if error > 2.0e-12:
        raise RuntimeError("a local multiplet Schur basis is not orthonormal")
    basis.setflags(write=False)
    return basis, tuple(channels)


@lru_cache(maxsize=None)
def _multiplet_reference_cached(
    geometry: Geometry,
    ratio: float,
    total_q: int,
    eigen_index: int,
) -> TrianglePatchClusterMultipletReference:
    checked_geometry = _checked_geometry(geometry)
    checked_ratio = _checked_ratio(ratio)
    checked_q = validate_q(total_q, name="total_q")
    if checked_q not in (0, 2, 4, 6):
        raise ValueError("six spin-1/2 sites support total_q in {0,2,4,6}")
    if isinstance(eigen_index, bool) or not isinstance(eigen_index, Integral):
        raise TypeError("eigen_index must be an integer")
    level = int(eigen_index)
    if level < 0:
        raise ValueError("eigen_index must be nonnegative")
    basis, _ = _cluster_multiplet_basis(checked_geometry, checked_q)
    if level >= basis.shape[1]:
        raise ValueError("eigen_index exceeds the spin-sector multiplicity")
    bonds = _cluster_bonds(checked_geometry, checked_ratio)
    hamiltonian, integer_basis = heisenberg_sz_matrix(6, bonds)
    spin_squared, spin_basis = spin_squared_matrix(6)
    if integer_basis != spin_basis or integer_basis != tuple(range(64)):
        raise RuntimeError("a local exact operator changed computational-basis order")
    reduced = basis.T @ hamiltonian @ basis
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = np.linalg.eigh(reduced)
    coefficients = np.asarray(vectors[:, level], dtype=np.float64)
    pivot = int(np.argmax(np.abs(coefficients)))
    if coefficients[pivot] < 0.0:
        coefficients *= -1.0
    state = np.asarray(basis @ coefficients, dtype=np.float64)
    energy = float(values[level])
    eigen_residual = float(np.linalg.norm(hamiltonian @ state - energy * state))
    spin_eigenvalue = checked_q * (checked_q + 2) / 4.0
    spin_residual = float(
        np.linalg.norm(spin_squared @ state - spin_eigenvalue * state)
    )
    gram_error = float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1])))
    if (
        eigen_residual > 3.0e-12
        or spin_residual > 3.0e-12
        or abs(np.linalg.norm(state) - 1.0) > 2.0e-13
    ):
        raise RuntimeError("failed to construct an exact local spin multiplet")
    coefficients.setflags(write=False)
    state.setflags(write=False)
    gap_above = (
        float(values[level + 1] - values[level])
        if level + 1 < len(values)
        else None
    )
    return TrianglePatchClusterMultipletReference(
        geometry=checked_geometry,
        j2_over_j1=checked_ratio,
        total_q=checked_q,
        eigen_index=level,
        energy=energy,
        gap_above=gap_above,
        multiplicity_dimension=int(basis.shape[1]),
        bonds=bonds,
        coefficients=coefficients,
        highest_weight_state=state,
        basis_orthonormality_error=gram_error,
        eigen_residual=eigen_residual,
        spin_squared_residual=spin_residual,
    )


def triangle_patch_cluster_multiplet_reference(
    geometry: Geometry,
    j2_over_j1: float,
    total_q: int,
    *,
    eigen_index: int = 0,
) -> TrianglePatchClusterMultipletReference:
    """Return a defensive copy of one exact local eigenmultiplet.

    The computational state is the highest-weight member ``m2=total_q``.
    ``coefficients`` follow deterministic
    ``(q_left,q_right,left_a,right_a)`` order and are independent of the
    magnetic member.
    """

    checked_q = validate_q(total_q, name="total_q")
    if checked_q not in (0, 2, 4, 6):
        raise ValueError("six spin-1/2 sites support total_q in {0,2,4,6}")
    if isinstance(eigen_index, bool) or not isinstance(eigen_index, Integral):
        raise TypeError("eigen_index must be an integer")
    level = int(eigen_index)
    if level < 0:
        raise ValueError("eigen_index must be nonnegative")
    reference = _multiplet_reference_cached(
        _checked_geometry(geometry),
        _checked_ratio(j2_over_j1),
        checked_q,
        level,
    )
    coefficients = np.array(reference.coefficients, copy=True)
    state = np.array(reference.highest_weight_state, copy=True)
    coefficients.setflags(write=False)
    state.setflags(write=False)
    return TrianglePatchClusterMultipletReference(
        geometry=reference.geometry,
        j2_over_j1=reference.j2_over_j1,
        total_q=reference.total_q,
        eigen_index=reference.eigen_index,
        energy=reference.energy,
        gap_above=reference.gap_above,
        multiplicity_dimension=reference.multiplicity_dimension,
        bonds=reference.bonds,
        coefficients=coefficients,
        highest_weight_state=state,
        basis_orthonormality_error=reference.basis_orthonormality_error,
        eigen_residual=reference.eigen_residual,
        spin_squared_residual=reference.spin_squared_residual,
    )


def init_triangle_patch_cluster_multiplets(
    key: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    *,
    j2_over_j1: float,
    background_scale: float = 0.05,
    route_gate_scale: float = 0.05,
) -> dict[str, tuple[jax.Array, ...]]:
    """Replace the sealed initializer's random local floor by multiplets.

    Feature channel zero remains the exact local ground singlet.  Feature
    channel one becomes the first excited singlet at ``q=0`` and the lowest
    multiplet in each retained ``q=2,4,6`` sector.  All non-patch banks are
    returned bit-for-bit from the sealed initializer.
    """

    params = init_triangle_patch_cluster_singlet(
        key,
        plan,
        j2_over_j1=j2_over_j1,
        background_scale=background_scale,
        route_gate_scale=route_gate_scale,
    )
    ratio = _checked_ratio(j2_over_j1)
    patch_layout = triangle_patch_pair_layout(plan)
    retained_parent_q = {layout.q_parent for layout in patch_layout}
    if 2 not in retained_parent_q:
        raise ValueError("physical multiplet carriers require retained q=2")
    patch_arrays = [np.array(block, copy=True) for block in params["patch_pair"]]
    positions = {layout.key: index for index, layout in enumerate(patch_layout)}
    for total_q in sorted(retained_parent_q & {0, 2, 4, 6}):
        level = 1 if total_q == 0 else 0
        reference = _multiplet_reference_cached(
            plan.geometry, ratio, total_q, level
        )
        _, channels = _cluster_multiplet_basis(plan.geometry, total_q)
        if len(channels) != reference.coefficients.size:
            raise RuntimeError("multiplet coefficients and Schur ledger disagree")
        for coefficient, (q_left, q_right, left_a, right_a) in zip(
            reference.coefficients, channels, strict=True
        ):
            position = positions.get((q_left, q_right, total_q))
            if position is None:
                raise RuntimeError("the plan lost a required local multiplet block")
            patch_arrays[position][1, left_a, right_a] = coefficient
    return {
        "patch_pair": tuple(jnp.asarray(block) for block in patch_arrays),
        "reduced": params["reduced"],
        "root": params["root"],
        "patch_gates": params["patch_gates"],
        "reduced_gates": params["reduced_gates"],
    }


__all__ = [
    "TrianglePatchClusterMultipletReference",
    "init_triangle_patch_cluster_multiplets",
    "triangle_patch_cluster_multiplet_reference",
]
