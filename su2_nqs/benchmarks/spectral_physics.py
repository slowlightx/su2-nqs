"""Exact finite-size response diagnostics in SU(2) multiplicity space.

The routines in this module are deliberately physics-facing.  They turn a
rank-one spin operator into a matrix between two total-spin multiplicity
sectors and evaluate zero-temperature linear responses without constructing
magnetic-index copies of either sector.  Angular momenta are doubled
integers throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np

from ..su2.fusion_path import FusionPath, enumerate_paths
from ..su2.quantum_numbers import validate_q
from ..su2.reduced_vector import (
    appended_spin_vector_source_coefficient,
    old_block_vector_advance_coefficient,
)


def _sparse():
    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError("spectral response diagnostics require scipy") from error
    return sparse


def _sparse_linalg():
    try:
        from scipy.sparse import linalg
    except ImportError as error:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError("spectral response diagnostics require scipy") from error
    return linalg


def _extended_paths(path: FusionPath) -> tuple[FusionPath, ...]:
    q = path[-1]
    lower = (path + (q - 1,),) if q > 0 else ()
    return lower + (path + (q + 1,),)


def _can_reach(q: int, remaining: int, target_q: int) -> bool:
    return (
        q >= 0
        and abs(q - target_q) <= remaining
        and (remaining - abs(q - target_q)) % 2 == 0
    )


@dataclass(frozen=True)
class SectorVectorOperator:
    """Reduced ``q_op=2`` matrix between fixed total-spin sectors.

    ``matrix`` contains reduced matrix elements
    ``<new_path||sum_i c_i S_i||old_path>`` and follows the project-wide
    orientation ``matrix[new, old]``.  In particular, for ``q_old=0`` and
    ``q_new=2``, the physical ``mu=0`` matrix is ``matrix / sqrt(3)``.
    """

    n_sites: int
    q_new: int
    q_old: int
    new_paths: tuple[FusionPath, ...]
    old_paths: tuple[FusionPath, ...]
    matrix: object


def compile_sector_spin_vector(
    endpoint_weights: Sequence[complex],
    *,
    q_new: int,
    q_old: int,
    zero_tolerance: float = 2.0e-14,
) -> SectorVectorOperator:
    """Compile ``sum_i c_i S_i`` directly between two fixed spin sectors.

    Unlike the all-irrep reference table, this recurrence discards a prefix
    pair as soon as either side can no longer reach its requested final spin.
    It is therefore suitable for the N=16 singlet-to-triplet response used by
    the finite-size physics audit while retaining the same exact numeric CG
    coefficients.
    """

    try:
        raw_weights = tuple(endpoint_weights)
    except TypeError as error:
        raise TypeError("endpoint_weights must be a finite numeric sequence") from error
    if not raw_weights:
        raise ValueError("endpoint_weights must contain at least one site")
    weights = np.asarray(raw_weights, dtype=np.complex128)
    if weights.ndim != 1 or np.any(~np.isfinite(weights)):
        raise ValueError("endpoint_weights must be a finite one-dimensional sequence")
    validate_q(q_new, name="q_new")
    validate_q(q_old, name="q_old")
    if abs(q_new - q_old) not in (0, 2) or q_new + q_old == 0:
        raise ValueError("q_op=2 requires an admissible non-scalar sector pair")
    n_sites = len(weights)
    new_paths = enumerate_paths(n_sites, q_new)
    old_paths = enumerate_paths(n_sites, q_old)
    if not new_paths or not old_paths:
        raise ValueError("one requested total-spin sector is unreachable")
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")

    initial: FusionPath = (1,)
    entries: dict[tuple[FusionPath, FusionPath], complex] = {}
    if weights[0] != 0.0:
        entries[(initial, initial)] = math.sqrt(1.5) * weights[0]
    prefixes: tuple[FusionPath, ...] = (initial,)

    for site, endpoint_weight in enumerate(weights[1:], start=1):
        remaining = n_sites - site - 1
        accumulated: dict[tuple[FusionPath, FusionPath], complex] = {}
        for (prefix_new, prefix_old), old_value in entries.items():
            for new in _extended_paths(prefix_new):
                if not _can_reach(new[-1], remaining, q_new):
                    continue
                for old in _extended_paths(prefix_old):
                    if not _can_reach(old[-1], remaining, q_old):
                        continue
                    coefficient = old_block_vector_advance_coefficient(
                        prefix_new[-1], prefix_old[-1], new[-1], old[-1]
                    )
                    if coefficient:
                        key = (new, old)
                        accumulated[key] = (
                            accumulated.get(key, 0.0j) + coefficient * old_value
                        )

        next_prefixes: list[FusionPath] = []
        for prefix in prefixes:
            extensions = _extended_paths(prefix)
            next_prefixes.extend(extensions)
            if endpoint_weight == 0.0:
                continue
            for new in extensions:
                if not _can_reach(new[-1], remaining, q_new):
                    continue
                for old in extensions:
                    if not _can_reach(old[-1], remaining, q_old):
                        continue
                    coefficient = appended_spin_vector_source_coefficient(
                        prefix[-1], new[-1], old[-1]
                    )
                    if coefficient:
                        key = (new, old)
                        accumulated[key] = (
                            accumulated.get(key, 0.0j)
                            + coefficient * endpoint_weight
                        )
        prefixes = tuple(next_prefixes)
        entries = {
            key: value
            for key, value in accumulated.items()
            if abs(value) > zero_tolerance
        }

    new_positions = {path: index for index, path in enumerate(new_paths)}
    old_positions = {path: index for index, path in enumerate(old_paths)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for (new, old), value in entries.items():
        if new[-1] != q_new or old[-1] != q_old:
            continue
        rows.append(new_positions[new])
        columns.append(old_positions[old])
        values.append(value)
    sparse = _sparse()
    matrix = sparse.coo_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, columns)),
        shape=(len(new_paths), len(old_paths)),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return SectorVectorOperator(
        n_sites=n_sites,
        q_new=q_new,
        q_old=q_old,
        new_paths=new_paths,
        old_paths=old_paths,
        matrix=matrix,
    )


@dataclass(frozen=True)
class ZeroTemperatureResponse:
    """Solution of a projected zero-temperature response equation."""

    source_norm_squared: float
    static_response: float
    fidelity_response: float
    harmonic_gap: float
    relative_linear_residual: float
    iterations_info: int
    removed_null_amplitude: complex
    solution: np.ndarray


def zero_temperature_response(
    hamiltonian,
    ground_energy: float,
    source: np.ndarray,
    *,
    null_vector: np.ndarray | None = None,
    null_shift: float = 1.0,
    relative_tolerance: float = 1.0e-11,
    maximum_iterations: int | None = None,
) -> ZeroTemperatureResponse:
    """Solve ``(H-E0)^-1 source`` with an optional ground-state projection.

    The returned ``static_response`` is ``2 <b|(H-E0)^-1|b>`` and the
    ``fidelity_response`` is ``||(H-E0)^-1 b||^2``.  No volume normalization
    is applied here.  When source and ground state share a symmetry sector,
    pass the normalized ground state as ``null_vector``; its component is
    removed and the null direction is shifted only to make CG nonsingular.
    """

    linalg = _sparse_linalg()
    vector = np.asarray(source, dtype=np.complex128)
    if vector.ndim != 1 or vector.shape[0] != hamiltonian.shape[0]:
        raise ValueError("source shape must match the square Hamiltonian")
    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square")
    if not np.isfinite(ground_energy) or relative_tolerance <= 0.0:
        raise ValueError("ground_energy and relative_tolerance must be valid")
    if null_shift <= 0.0:
        raise ValueError("null_shift must be positive")

    removed = 0.0j
    null = None
    if null_vector is not None:
        null = np.asarray(null_vector, dtype=np.complex128)
        if null.shape != vector.shape:
            raise ValueError("null_vector shape must match source")
        norm = float(np.real(np.vdot(null, null)))
        if norm <= 0.0:
            raise ValueError("null_vector must have positive norm")
        null = null / math.sqrt(norm)
        removed = complex(np.vdot(null, vector))
        vector = vector - removed * null

    def apply(value: np.ndarray) -> np.ndarray:
        result = hamiltonian @ value - ground_energy * value
        if null is not None:
            result = result + null_shift * null * np.vdot(null, value)
        return np.asarray(result, dtype=np.complex128)

    operator = linalg.LinearOperator(
        hamiltonian.shape, matvec=apply, rmatvec=apply, dtype=np.complex128
    )
    diagonal = np.asarray(hamiltonian.diagonal(), dtype=np.complex128) - ground_energy
    if null is not None:
        diagonal = diagonal + null_shift * np.abs(null) ** 2
    scale = np.maximum(np.real(diagonal), 1.0e-12)
    preconditioner = linalg.LinearOperator(
        hamiltonian.shape,
        matvec=lambda value: np.asarray(value) / scale,
        rmatvec=lambda value: np.asarray(value) / scale,
        dtype=np.complex128,
    )
    solution, info = linalg.cg(
        operator,
        vector,
        M=preconditioner,
        rtol=relative_tolerance,
        atol=0.0,
        maxiter=maximum_iterations,
    )
    residual = apply(solution) - vector
    source_norm = float(np.real(np.vdot(vector, vector)))
    residual_norm = float(np.linalg.norm(residual))
    relative_residual = residual_norm / max(math.sqrt(source_norm), 1.0e-300)
    overlap = float(np.real(np.vdot(vector, solution)))
    if info != 0 or relative_residual > max(20.0 * relative_tolerance, 1.0e-9):
        raise RuntimeError(
            f"projected response solve failed: info={info}, "
            f"relative residual={relative_residual:.3e}"
        )
    return ZeroTemperatureResponse(
        source_norm_squared=source_norm,
        static_response=2.0 * overlap,
        fidelity_response=float(np.real(np.vdot(solution, solution))),
        harmonic_gap=source_norm / overlap if overlap > 0.0 else math.inf,
        relative_linear_residual=relative_residual,
        iterations_info=int(info),
        removed_null_amplitude=removed,
        solution=np.asarray(solution),
    )


def low_energy_spectral_weights(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    source: np.ndarray,
    *,
    ground_energy: float,
    minimum_gap: float = 1.0e-10,
) -> list[dict[str, float | int]]:
    """Return ordered low-energy weights of a response source."""

    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    source = np.asarray(source, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape != (len(source), len(values)):
        raise ValueError("eigenvector columns and eigenvalues must agree")
    rows: list[dict[str, float | int]] = []
    for index, energy in enumerate(values):
        gap = float(energy - ground_energy)
        if gap <= minimum_gap:
            continue
        weight = float(abs(np.vdot(vectors[:, index], source)) ** 2)
        rows.append(
            {
                "mode": int(index),
                "energy": float(energy),
                "gap": gap,
                "weight": weight,
                "static_response_weight": 2.0 * weight / gap,
            }
        )
    return rows


__all__ = [
    "SectorVectorOperator",
    "ZeroTemperatureResponse",
    "compile_sector_spin_vector",
    "low_energy_spectral_weights",
    "zero_temperature_response",
]
