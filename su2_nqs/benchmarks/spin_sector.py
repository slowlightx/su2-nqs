"""Spin-sector diagnostics for fixed-magnetization wavefunctions.

The SU(2) fusion representation fixes total spin by construction.  This module
provides the independent fixed-``Sz`` reference needed to quantify what a
generic computational-basis state still has to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..operators.sz_graph_sparse import heisenberg_sz_graph_sparse


@dataclass(frozen=True)
class SpinPurityMetrics:
    """Total-spin diagnostics for one normalized fixed-``Sz`` state."""

    normalization: float
    spin_squared_expectation: float
    singlet_weight: float
    non_singlet_weight: float
    projected_singlet_residual: float


@dataclass(frozen=True)
class SingletMultiplicityProjection:
    """Normalized multiplicity-space projection of a fixed-``Sz`` state."""

    state: np.ndarray
    singlet_weight: float
    isometry_error: float


def project_sz_to_singlet_multiplicity(
    state: np.ndarray,
    fusion_to_sz: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> SingletMultiplicityProjection:
    """Project a fixed-``Sz`` state through an audited fusion-basis isometry.

    ``fusion_to_sz`` has fixed-``Sz`` configurations as rows and singlet
    fusion paths as columns.  The returned state is normalized in the
    multiplicity basis; ``singlet_weight`` is the norm before that final
    normalization.
    """

    vector = np.asarray(state, dtype=np.complex128)
    transform = np.asarray(fusion_to_sz, dtype=np.complex128)
    if vector.ndim != 1 or transform.ndim != 2 or transform.shape[0] != len(vector):
        raise ValueError("state and fusion_to_sz dimensions do not agree")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    normalization = float(np.vdot(vector, vector).real)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("state must have finite positive norm")
    identity = np.eye(transform.shape[1], dtype=np.complex128)
    isometry_error = float(
        np.max(np.abs(transform.conj().T @ transform - identity), initial=0.0)
    )
    if isometry_error > tolerance:
        raise ValueError("fusion_to_sz columns are not orthonormal")
    coefficients = transform.conj().T @ (vector / np.sqrt(normalization))
    singlet_weight = float(np.vdot(coefficients, coefficients).real)
    if not np.isfinite(singlet_weight) or singlet_weight <= tolerance:
        raise ValueError("state has no numerically resolvable singlet component")
    return SingletMultiplicityProjection(
        state=coefficients / np.sqrt(singlet_weight),
        singlet_weight=singlet_weight,
        isometry_error=isometry_error,
    )


def spin_squared_sz_sparse(n: int, *, n_up: int | None = None):
    """Return sparse total ``S^2`` in deterministic fixed-``Sz`` basis order.

    The construction uses ``S^2 = 3N/4 + 2 sum_{i<j} S_i.S_j`` and therefore
    remains independent of the coupled-basis implementation.
    """

    if n < 1:
        raise ValueError("N must be positive")
    if n_up is None:
        n_up = n // 2
    if not 0 <= n_up <= n:
        raise ValueError("n_up must lie between zero and N")
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("sparse spin-sector diagnostics require scipy") from error
    pair_bonds = tuple(
        (first, second, 1.0)
        for first in range(n)
        for second in range(first + 1, n)
    )
    pair_sum, basis = heisenberg_sz_graph_sparse(n, pair_bonds, n_up=n_up)
    identity = sparse.identity(len(basis), format="csr", dtype=np.float64)
    return (0.75 * n * identity + 2.0 * pair_sum).tocsr(), basis


def singlet_projector_action(
    state: np.ndarray,
    spin_squared: Any,
    *,
    maximum_spin: int,
) -> np.ndarray:
    """Apply the exact spectral polynomial for the total-spin singlet.

    In an even-spin fixed-``Sz=0`` sector, the allowed total spins are the
    integers ``S=0,...,maximum_spin``.  Hence

    ``P_0 = product_{S=1}^{maximum_spin} (I-S^2/[S(S+1)])``.
    """

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or spin_squared.shape != (len(vector), len(vector)):
        raise ValueError("state and spin_squared dimensions do not agree")
    if maximum_spin < 0:
        raise ValueError("maximum_spin must be nonnegative")
    projected = vector.copy()
    for total_spin in range(1, maximum_spin + 1):
        eigenvalue = float(total_spin * (total_spin + 1))
        projected = projected - np.asarray(spin_squared @ projected) / eigenvalue
    return projected


def spin_purity_metrics(
    state: np.ndarray,
    spin_squared: Any,
    *,
    maximum_spin: int,
    tolerance: float = 1.0e-10,
) -> SpinPurityMetrics:
    """Measure ``<S^2>`` and singlet/non-singlet weights.

    ``tolerance`` is used only to reject impossible numerical values.  It does
    not clip a physically nonzero spin contamination.
    """

    vector = np.asarray(state, dtype=np.complex128)
    normalization = float(np.vdot(vector, vector).real)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("state must have finite positive norm")
    normalized = vector / np.sqrt(normalization)
    action = np.asarray(spin_squared @ normalized)
    expectation = complex(np.vdot(normalized, action))
    if abs(expectation.imag) > tolerance:
        raise RuntimeError("Hermitian S^2 expectation acquired an imaginary part")
    singlet = singlet_projector_action(
        normalized,
        spin_squared,
        maximum_spin=maximum_spin,
    )
    singlet_weight = float(np.vdot(singlet, singlet).real)
    if not -tolerance <= singlet_weight <= 1.0 + tolerance:
        raise RuntimeError("singlet-projector weight lies outside [0,1]")
    singlet_weight = min(1.0, max(0.0, singlet_weight))
    residual = float(np.linalg.norm(spin_squared @ singlet))
    return SpinPurityMetrics(
        normalization=normalization,
        spin_squared_expectation=float(expectation.real),
        singlet_weight=singlet_weight,
        non_singlet_weight=1.0 - singlet_weight,
        projected_singlet_residual=residual,
    )
