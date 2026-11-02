"""Exact character sectors for monomial fusion-tree symmetry actions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..operators.tree_automorphism import TreeFrameIsomorphismPlan
from ..su2.fusion_tree import FusionTree, FusionTreeState


def _sparse():
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("tree symmetry sectors require scipy") from error
    return sparse


@dataclass(frozen=True)
class TranslationCharacterBasis:
    """An orthonormal fixed-momentum basis in one total-spin tree sector."""

    tree: FusionTree
    target_q: int
    linear_size: int
    momentum: tuple[int, int]
    full_states: tuple[FusionTreeState, ...]
    representatives: tuple[FusionTreeState, ...]
    columns: object
    audit: dict[str, float | int]


def _apply_power(
    plan: TreeFrameIsomorphismPlan,
    state: FusionTreeState,
    exponent: int,
) -> tuple[FusionTreeState, float]:
    result, phase = state, 1.0
    for _ in range(exponent):
        result, local_phase = plan.apply(result)
        phase *= local_phase
    return result, phase


def translation_character_basis(
    tree: FusionTree,
    target_q: int,
    tx: TreeFrameIsomorphismPlan,
    ty: TreeFrameIsomorphismPlan,
    linear_size: int,
    momentum: tuple[int, int],
    *,
    zero_tolerance: float = 1.0e-13,
) -> TranslationCharacterBasis:
    """Build the ``(kx,ky)`` orbit basis of commuting support-one translations.

    Momentum integers label ``2*pi*(kx,ky)/L`` and obey
    ``T_mu |k> = exp(-2*pi*i*k_mu/L) |k>``.  Columns are obtained by applying
    the finite-group character projector to one representative per monomial
    orbit.  No dense Hilbert-dimension-square projector is formed.
    """

    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, Integral)
        or linear_size < 1
    ):
        raise ValueError("linear_size must be a positive integer")
    linear_size = int(linear_size)
    if tx.source_tree != tree or tx.target_tree != tree:
        raise ValueError("tx must be an automorphism of the supplied tree")
    if ty.source_tree != tree or ty.target_tree != tree:
        raise ValueError("ty must be an automorphism of the supplied tree")
    if zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be positive")
    kx, ky = (int(momentum[0]) % linear_size, int(momentum[1]) % linear_size)
    states = tree.enumerate_states(int(target_q))
    positions = {state.labels: index for index, state in enumerate(states)}
    visited: set[int] = set()
    representatives = []
    rows, columns, values = [], [], []
    discarded_orbits = 0
    maximum_orbit_size = 0
    for seed_index, seed in enumerate(states):
        if seed_index in visited:
            continue
        orbit_indices: set[int] = set()
        projected: dict[int, complex] = {}
        for shift_x in range(linear_size):
            shifted_x, phase_x = _apply_power(tx, seed, shift_x)
            for shift_y in range(linear_size):
                shifted, phase_y = _apply_power(ty, shifted_x, shift_y)
                index = positions[shifted.labels]
                orbit_indices.add(index)
                character = np.exp(
                    2.0j
                    * np.pi
                    * (kx * shift_x + ky * shift_y)
                    / linear_size
                )
                projected[index] = projected.get(index, 0.0j) + (
                    character * phase_x * phase_y / linear_size**2
                )
        visited.update(orbit_indices)
        maximum_orbit_size = max(maximum_orbit_size, len(orbit_indices))
        norm = float(np.sqrt(sum(abs(value) ** 2 for value in projected.values())))
        if norm <= zero_tolerance:
            discarded_orbits += 1
            continue
        column = len(representatives)
        representatives.append(seed)
        for row, value in projected.items():
            normalized = value / norm
            if abs(normalized) > zero_tolerance:
                rows.append(row)
                columns.append(column)
                values.append(normalized)

    sparse = _sparse()
    basis = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(states), len(representatives)),
        dtype=np.complex128,
    ).tocsr()
    gram = (basis.conj().T @ basis).toarray()
    orthonormality_error = float(
        np.max(np.abs(gram - np.eye(len(representatives))), initial=0.0)
    )
    tx_matrix = _monomial_plan_matrix(tx, states)
    ty_matrix = _monomial_plan_matrix(ty, states)
    tx_eigenvalue = np.exp(-2.0j * np.pi * kx / linear_size)
    ty_eigenvalue = np.exp(-2.0j * np.pi * ky / linear_size)
    tx_residual = tx_matrix @ basis - tx_eigenvalue * basis
    ty_residual = ty_matrix @ basis - ty_eigenvalue * basis
    audit = {
        "orthonormality_error": orthonormality_error,
        "tx_eigen_residual": float(
            np.max(np.abs(tx_residual.data), initial=0.0)
        ),
        "ty_eigen_residual": float(
            np.max(np.abs(ty_residual.data), initial=0.0)
        ),
        "full_dimension": len(states),
        "sector_dimension": len(representatives),
        "discarded_incompatible_orbits": discarded_orbits,
        "maximum_orbit_size": maximum_orbit_size,
        "stored_nnz": int(basis.nnz),
    }
    return TranslationCharacterBasis(
        tree=tree,
        target_q=int(target_q),
        linear_size=linear_size,
        momentum=(kx, ky),
        full_states=states,
        representatives=tuple(representatives),
        columns=basis,
        audit=audit,
    )


def _monomial_plan_matrix(
    plan: TreeFrameIsomorphismPlan,
    states: tuple[FusionTreeState, ...],
):
    sparse = _sparse()
    positions = {state.labels: index for index, state in enumerate(states)}
    rows, columns, values = [], [], []
    for column, state in enumerate(states):
        mapped, phase = plan.apply(state)
        rows.append(positions[mapped.labels])
        columns.append(column)
        values.append(phase)
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(states), len(states))
    ).tocsr()


__all__ = ["TranslationCharacterBasis", "translation_character_basis"]
