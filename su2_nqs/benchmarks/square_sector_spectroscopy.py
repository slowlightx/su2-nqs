"""Sparse exact sector minima and one-dimensional space-group labels on square tori."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.square_symmetry import square_momentum_label, square_site_generators
from ..operators.permutation import (
    apply_site_permutation_fusion,
    heisenberg_graph_fusion_matrix,
)
from ..su2.fusion_path import count_paths


@dataclass(frozen=True)
class SquareSectorMinimum:
    """Lowest state of one exact doubled-spin sector on a square torus."""

    linear_size: int
    target_q: int
    j2_over_j1: float
    energy: float
    state: np.ndarray
    next_level_gap: float
    momentum: tuple[int, int]
    momentum_label: str
    little_group_irrep: str
    label: str
    audit: dict[str, float]


def square_sector_shells(linear_size: int, target_q: int):
    """Build sparse unit-coupling J1 and J2 Hamiltonian shells."""

    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, Integral)
        or linear_size < 2
    ):
        raise ValueError("linear_size must be an integer at least two")
    n_sites = int(linear_size) ** 2
    if isinstance(target_q, bool) or not isinstance(target_q, Integral):
        raise TypeError("target_q must be a doubled-integer label")
    target_q = int(target_q)
    if count_paths(n_sites, target_q) == 0:
        raise ValueError("target_q is unreachable for this square")
    h1 = heisenberg_graph_fusion_matrix(
        n_sites,
        square_j1j2_bonds(
            int(linear_size), int(linear_size), j1=1.0, j2=0.0, periodic=True
        ),
        target_q=target_q,
    ).tocsr()
    h2 = heisenberg_graph_fusion_matrix(
        n_sites,
        square_j1j2_bonds(
            int(linear_size), int(linear_size), j1=0.0, j2=1.0, periodic=True
        ),
        target_q=target_q,
    ).tocsr()
    return h1, h2


def _symmetry_character(
    state: np.ndarray,
    image: tuple[int, ...],
    *,
    target_q: int,
) -> tuple[complex, float]:
    transformed = apply_site_permutation_fusion(
        state, image, target_q=target_q
    )
    character = complex(np.vdot(state, transformed))
    residual = float(np.linalg.norm(transformed - character * state))
    return character, residual


def _nearest_root(character: complex, order: int) -> tuple[int, float]:
    roots = np.exp(-2.0j * np.pi * np.arange(order) / order)
    errors = np.abs(roots - character)
    index = int(np.argmin(errors))
    return index, float(errors[index])


def _sign_character(character: complex) -> tuple[int, float]:
    signs = np.asarray((1.0, -1.0), dtype=np.complex128)
    errors = np.abs(signs - character)
    index = int(np.argmin(errors))
    return (1 if index == 0 else -1), float(errors[index])


def _unique_state_square_label(
    state: np.ndarray,
    *,
    linear_size: int,
    target_q: int,
) -> tuple[tuple[int, int], str, str, dict[str, float]]:
    generators = square_site_generators(linear_size)
    tx, tx_residual = _symmetry_character(
        state, generators["tx"], target_q=target_q
    )
    ty, ty_residual = _symmetry_character(
        state, generators["ty"], target_q=target_q
    )
    nx, tx_root_error = _nearest_root(tx, linear_size)
    ny, ty_root_error = _nearest_root(ty, linear_size)
    momentum = (nx, ny)
    momentum_name = square_momentum_label(momentum, linear_size)

    c4, c4_residual = _symmetry_character(
        state, generators["c4"], target_q=target_q
    )
    mx, mx_residual = _symmetry_character(
        state, generators["mx"], target_q=target_q
    )
    c4_sign, c4_sign_error = _sign_character(c4)
    mx_sign, mx_sign_error = _sign_character(mx)
    if momentum_name in ("Gamma", "M"):
        irreps = {
            (1, 1): "A1",
            (1, -1): "A2",
            (-1, 1): "B1",
            (-1, -1): "B2",
        }
        irrep = irreps[(c4_sign, mx_sign)]
        little_group_error = max(c4_sign_error, mx_sign_error)
    else:
        raise ValueError(
            "a unique square-torus state outside Gamma/M requires a degenerate-star analysis"
        )
    audit = {
        "tx_eigen_residual": tx_residual,
        "ty_eigen_residual": ty_residual,
        "c4_eigen_residual": c4_residual,
        "mx_eigen_residual": mx_residual,
        "tx_root_error": tx_root_error,
        "ty_root_error": ty_root_error,
        "little_group_character_error": little_group_error,
        "maximum_space_group_error": max(
            tx_residual,
            ty_residual,
            c4_residual,
            mx_residual,
            tx_root_error,
            ty_root_error,
            little_group_error,
        ),
    }
    return momentum, momentum_name, irrep, audit


def lowest_square_sector_minimum(
    h1,
    h2,
    *,
    linear_size: int,
    target_q: int,
    j2_over_j1: float,
    low_modes: int = 8,
) -> SquareSectorMinimum:
    """Solve and label a nondegenerate lowest state in one square sector."""

    if isinstance(j2_over_j1, bool) or not isinstance(j2_over_j1, Real):
        raise TypeError("j2_over_j1 must be a real scalar")
    ratio = float(j2_over_j1)
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("j2_over_j1 must be finite and nonnegative")
    dimension = count_paths(linear_size * linear_size, target_q)
    if h1.shape != (dimension, dimension) or h2.shape != (dimension, dimension):
        raise ValueError("square shell shape does not match the requested sector")
    if low_modes < 2 and dimension > 1:
        raise ValueError("low_modes must be at least two for a nontrivial sector")
    hamiltonian = (h1 + ratio * h2).tocsr()
    difference = (hamiltonian - hamiltonian.getH()).tocsr()
    hermiticity_error = float(np.max(np.abs(difference.data), initial=0.0))

    if dimension <= low_modes:
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
    else:
        from scipy.sparse.linalg import eigsh

        initial = np.linspace(1.0, 2.0, dimension, dtype=np.float64)
        initial /= np.linalg.norm(initial)
        values, vectors = eigsh(
            hamiltonian,
            k=low_modes,
            which="SA",
            tol=1.0e-12,
            maxiter=30000,
            v0=initial,
        )
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]
    values = np.asarray(values, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    state = vectors[:, 0]
    state /= np.linalg.norm(state)
    residuals = np.asarray(
        [
            np.linalg.norm(hamiltonian @ vectors[:, index] - values[index] * vectors[:, index])
            for index in range(values.size)
        ],
        dtype=np.float64,
    )
    degeneracy_tolerance = 2.0e-10 + 20.0 * (
        residuals[0] + (residuals[1] if dimension > 1 else 0.0)
    )
    next_gap = float(values[1] - values[0]) if dimension > 1 else float("inf")
    if dimension > 1 and next_gap <= degeneracy_tolerance:
        raise ValueError(
            "lowest state is degenerate; use a closed multiplet decomposition"
        )
    momentum, momentum_name, irrep, symmetry_audit = _unique_state_square_label(
        state,
        linear_size=linear_size,
        target_q=target_q,
    )
    audit = {
        "hermiticity_error": hermiticity_error,
        "ground_eigen_residual": float(residuals[0]),
        "maximum_retained_eigen_residual": float(np.max(residuals)),
        "ground_degeneracy_tolerance": float(degeneracy_tolerance),
        **symmetry_audit,
    }
    return SquareSectorMinimum(
        linear_size=linear_size,
        target_q=target_q,
        j2_over_j1=ratio,
        energy=float(values[0]),
        state=state,
        next_level_gap=next_gap,
        momentum=momentum,
        momentum_label=momentum_name,
        little_group_irrep=irrep,
        label=f"{momentum_name}.{irrep}",
        audit=audit,
    )
