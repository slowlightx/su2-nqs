"""Nonlinear SU(2)-reduced local tomography from sampled moment matrices.

The statistical layer in :mod:`su2_nqs.vmc.transition_moments` returns
``M[o,b,a]=<psi_b|O_o|psi_a>``.  This module maps the identity, pair, and
oriented-chirality matrices to all local blocks of a two-root subspace.  It
does not normalize traces, clip eigenvalues, or project density matrices to
the positive cone: raw violations are part of the Monte Carlo audit.

The compact real vector returned by :func:`two_root_tomography_vector` is
intended as a nonlinear callback to the joint block bootstrap, so every trace
norm and QDDM singular value is reconstructed inside every replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from su2_nqs.benchmarks.local_indistinguishability import (
    local_code_decoupling_defect,
    local_code_qddm_singular_values,
    trace_distance,
    transition_trace_norm,
    three_spin_su2_transition,
    two_spin_su2_transition,
)
from su2_nqs.vmc.transition_moments import hermitize_transition_moments


TOMOGRAPHY_VECTOR_NAMES = (
    "diagonal_trace_distance",
    "transition_trace_norm",
    "decoupling_defect",
    "qddm_singular_value_1",
    "qddm_singular_value_2",
    "qddm_singular_value_3",
    "overlap_null_absolute",
    "root0_trace_error_absolute",
    "root1_trace_error_absolute",
    "root0_raw_hermiticity_error",
    "root1_raw_hermiticity_error",
    "root0_minimum_eigenvalue",
    "root1_minimum_eigenvalue",
    "choi_minimum_eigenvalue",
    "maximum_operator_antihermiticity",
)


@dataclass(frozen=True)
class TwoRootLocalTomography:
    """Transition-complete local blocks, code metrics, and raw audits."""

    region_size: Literal[2, 3]
    root_indices: tuple[int, int]
    rho00: np.ndarray
    rho11: np.ndarray
    rho01: np.ndarray
    diagonal_trace_distance: float
    transition_trace_norm: float
    decoupling_defect: float
    qddm_singular_values: np.ndarray
    overlap_null_absolute: float
    root_traces: np.ndarray
    root_trace_errors: np.ndarray
    raw_diagonal_hermiticity_errors: np.ndarray
    diagonal_minimum_eigenvalues: np.ndarray
    choi_minimum_eigenvalue: float
    maximum_operator_antihermiticity: float


def _validate_indices(
    moments: np.ndarray,
    identity_index: int,
    pair_indices: Sequence[int],
    chirality_index: int | None,
    roots: tuple[int, int],
) -> tuple[np.ndarray, int, tuple[int, ...], int | None, tuple[int, int]]:
    values = np.asarray(moments, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("moments must have shape (operator, state, state)")
    if np.any(~np.isfinite(values)):
        raise ValueError("moments must be finite")
    identity = int(identity_index)
    pairs = tuple(int(value) for value in pair_indices)
    chirality = None if chirality_index is None else int(chirality_index)
    root_pair = (int(roots[0]), int(roots[1]))
    indices = (identity, *pairs) + (() if chirality is None else (chirality,))
    if any(index < 0 or index >= values.shape[0] for index in indices):
        raise ValueError("an operator index lies outside moments")
    if (
        root_pair[0] < 0
        or root_pair[1] < 0
        or root_pair[0] >= root_pair[1]
        or root_pair[1] >= values.shape[1]
    ):
        raise ValueError("roots must be an increasing valid pair")
    if len(set(indices)) != len(indices):
        raise ValueError("identity, pair, and chirality indices must be distinct")
    return values, identity, pairs, chirality, root_pair


def _raw_hermiticity_error(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))


def _tomography(
    moments: np.ndarray,
    *,
    identity_index: int,
    pair_indices: Sequence[int],
    chirality_index: int | None,
    roots: tuple[int, int],
) -> TwoRootLocalTomography:
    values, identity, pairs, chirality, (first, second) = _validate_indices(
        moments, identity_index, pair_indices, chirality_index, roots
    )
    if chirality is None:
        if len(pairs) != 1:
            raise ValueError("two-spin tomography needs exactly one pair operator")

        def reconstruct(matrix: np.ndarray, bra: int, ket: int) -> np.ndarray:
            return two_spin_su2_transition(
                matrix[identity, bra, ket], matrix[pairs[0], bra, ket]
            )

        region_size: Literal[2, 3] = 2
    else:
        if len(pairs) != 3:
            raise ValueError("three-spin tomography needs exactly three pair operators")

        def reconstruct(matrix: np.ndarray, bra: int, ket: int) -> np.ndarray:
            return three_spin_su2_transition(
                matrix[identity, bra, ket],
                tuple(matrix[index, bra, ket] for index in pairs),
                matrix[chirality, bra, ket],
            )

        region_size = 3

    # Audit the raw unbiased matrices before exploiting exact Hermiticity of
    # the registered local operators as a variance-reducing projection.
    raw00 = reconstruct(values, first, first)
    raw11 = reconstruct(values, second, second)
    hermitian_values = hermitize_transition_moments(values)
    rho00 = reconstruct(hermitian_values, first, first)
    rho11 = reconstruct(hermitian_values, second, second)
    rho01 = reconstruct(hermitian_values, second, first)
    rho00 = np.asarray(0.5 * (rho00 + rho00.conj().T), dtype=np.complex128)
    rho11 = np.asarray(0.5 * (rho11 + rho11.conj().T), dtype=np.complex128)
    choi = np.block([[rho00, rho01], [rho01.conj().T, rho11]])
    qddm = local_code_qddm_singular_values(rho00, rho11, rho01)
    traces = np.asarray((np.trace(rho00), np.trace(rho11)), dtype=np.complex128)
    trace_errors = np.abs(traces - 1.0)
    diagonal_minima = np.asarray(
        (np.min(np.linalg.eigvalsh(rho00)), np.min(np.linalg.eigvalsh(rho11))),
        dtype=np.float64,
    )
    operator_antihermiticity = float(
        np.max(
            np.abs(values - values.conj().transpose(0, 2, 1)), initial=0.0
        )
    )
    return TwoRootLocalTomography(
        region_size=region_size,
        root_indices=(first, second),
        rho00=rho00,
        rho11=rho11,
        rho01=np.asarray(rho01, dtype=np.complex128),
        diagonal_trace_distance=trace_distance(rho00, rho11),
        transition_trace_norm=transition_trace_norm(rho01),
        decoupling_defect=local_code_decoupling_defect(rho00, rho11, rho01),
        qddm_singular_values=qddm,
        overlap_null_absolute=float(
            abs(hermitian_values[identity, second, first])
        ),
        root_traces=traces,
        root_trace_errors=np.asarray(trace_errors, dtype=np.float64),
        raw_diagonal_hermiticity_errors=np.asarray(
            (_raw_hermiticity_error(raw00), _raw_hermiticity_error(raw11)),
            dtype=np.float64,
        ),
        diagonal_minimum_eigenvalues=diagonal_minima,
        choi_minimum_eigenvalue=float(np.min(np.linalg.eigvalsh(choi))),
        maximum_operator_antihermiticity=operator_antihermiticity,
    )


def two_root_two_spin_tomography(
    moments: np.ndarray,
    *,
    identity_index: int = 0,
    pair_index: int = 1,
    roots: tuple[int, int] = (0, 1),
) -> TwoRootLocalTomography:
    """Reconstruct one two-spin region from identity and pair matrices."""

    return _tomography(
        moments,
        identity_index=identity_index,
        pair_indices=(pair_index,),
        chirality_index=None,
        roots=roots,
    )


def two_root_three_spin_tomography(
    moments: np.ndarray,
    *,
    identity_index: int = 0,
    pair_indices: Sequence[int] = (1, 2, 3),
    chirality_index: int = 4,
    roots: tuple[int, int] = (0, 1),
) -> TwoRootLocalTomography:
    """Reconstruct one oriented three-spin region from five moment matrices."""

    return _tomography(
        moments,
        identity_index=identity_index,
        pair_indices=pair_indices,
        chirality_index=chirality_index,
        roots=roots,
    )


def two_root_tomography_vector(
    moments: np.ndarray,
    *,
    region_size: Literal[2, 3],
    identity_index: int = 0,
    pair_indices: Sequence[int] = (1, 2, 3),
    chirality_index: int = 4,
    roots: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Return bootstrap-facing nonlinear metrics in ``TOMOGRAPHY_VECTOR_NAMES`` order."""

    if int(region_size) == 2:
        if len(tuple(pair_indices)) < 1:
            raise ValueError("two-spin tomography needs a pair index")
        result = two_root_two_spin_tomography(
            moments,
            identity_index=identity_index,
            pair_index=int(tuple(pair_indices)[0]),
            roots=roots,
        )
    elif int(region_size) == 3:
        result = two_root_three_spin_tomography(
            moments,
            identity_index=identity_index,
            pair_indices=pair_indices,
            chirality_index=chirality_index,
            roots=roots,
        )
    else:
        raise ValueError("region_size must be two or three")
    return np.asarray(
        (
            result.diagonal_trace_distance,
            result.transition_trace_norm,
            result.decoupling_defect,
            *result.qddm_singular_values,
            result.overlap_null_absolute,
            *result.root_trace_errors,
            *result.raw_diagonal_hermiticity_errors,
            *result.diagonal_minimum_eigenvalues,
            result.choi_minimum_eigenvalue,
            result.maximum_operator_antihermiticity,
        ),
        dtype=np.float64,
    )


def local_diagonal_rdm_from_moments(
    moments: np.ndarray,
    *,
    root: int,
    region_size: Literal[2, 3],
    identity_index: int = 0,
    pair_indices: Sequence[int] = (1, 2, 3),
    chirality_index: int = 4,
) -> np.ndarray:
    """Reconstruct one Hermitian diagonal RDM without trace/PSD projection."""

    values = hermitize_transition_moments(moments)
    index = int(root)
    if index < 0 or index >= values.shape[1]:
        raise ValueError("root lies outside moments")
    pairs = tuple(int(value) for value in pair_indices)
    if int(region_size) == 2:
        if len(pairs) < 1:
            raise ValueError("two-spin tomography needs a pair index")
        rdm = two_spin_su2_transition(
            values[int(identity_index), index, index], values[pairs[0], index, index]
        )
    elif int(region_size) == 3:
        if len(pairs) != 3:
            raise ValueError("three-spin tomography needs three pair indices")
        rdm = three_spin_su2_transition(
            values[int(identity_index), index, index],
            tuple(values[pair, index, index] for pair in pairs),
            values[int(chirality_index), index, index],
        )
    else:
        raise ValueError("region_size must be two or three")
    return np.asarray(0.5 * (rdm + rdm.conj().T), dtype=np.complex128)


__all__ = [
    "TOMOGRAPHY_VECTOR_NAMES",
    "TwoRootLocalTomography",
    "local_diagonal_rdm_from_moments",
    "two_root_three_spin_tomography",
    "two_root_tomography_vector",
    "two_root_two_spin_tomography",
]
