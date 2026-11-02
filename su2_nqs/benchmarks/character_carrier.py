"""Deterministic exact-small coordinate bases for cyclic characters.

The one-site translation of a left-associated SU(2) fusion basis is generally
dense, so its character sector is not a disjoint orbit quotient of fusion-path
labels.  This module therefore constructs an orthonormal *character carrier*
from projected columns.  It is an exact reference construction, not a
scalable orbit sampler.

Only the translation representation, group period, and character enter the
carrier construction.  A Hamiltonian may be compressed afterwards, which
makes accidental Hamiltonian or ground-state leakage structurally impossible
at the constructor boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np


def _maximum_absolute(values: np.ndarray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


@dataclass(frozen=True)
class CharacterCarrier:
    """An orthonormal basis for one cyclic-translation character.

    ``columns`` is the rectangular isometry ``Q``.  Its columns span the
    projector range and obey ``T Q = character_eigenvalue * Q``.
    ``pivot_indices`` refer only to the input basis of the supplied
    translation matrix; their selection never uses a Hamiltonian.
    """

    period: int
    momentum_integer: int
    character_eigenvalue: complex
    projector: np.ndarray
    columns: np.ndarray
    pivot_indices: tuple[int, ...]
    pivot_residuals: tuple[float, ...]
    audit: Mapping[str, float | int]

    @property
    def full_dimension(self) -> int:
        return int(self.columns.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.columns.shape[1])


@dataclass(frozen=True)
class CompressedCharacterOperator:
    """One operator represented in a :class:`CharacterCarrier`."""

    matrix: np.ndarray
    audit: Mapping[str, float]


def cyclic_character_projector(
    translation: np.ndarray,
    period: int,
    momentum_integer: int,
    *,
    representation_tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, complex, dict[str, float]]:
    """Return ``P_k`` for ``T|k> = exp(-2*pi*i*k/N)|k>``.

    The input is validated as a unitary representation of the cyclic group.
    The returned projector uses the same momentum convention as
    :func:`su2_nqs.benchmarks.sector_spectroscopy.cyclic_momentum_projectors`.
    """

    operator = np.asarray(translation, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("translation must be a nonempty square matrix")
    if operator.shape[0] < 1:
        raise ValueError("translation must be a nonempty square matrix")
    if (
        isinstance(period, bool)
        or not isinstance(period, Integral)
        or int(period) < 1
    ):
        raise ValueError("period must be a positive integer")
    if isinstance(momentum_integer, bool) or not isinstance(
        momentum_integer, Integral
    ):
        raise TypeError("momentum_integer must be an integer")
    if not np.isfinite(representation_tolerance) or representation_tolerance <= 0:
        raise ValueError("representation_tolerance must be positive")

    period = int(period)
    momentum = int(momentum_integer) % period
    dimension = operator.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    unitarity_error = _maximum_absolute(operator.conj().T @ operator - identity)
    closure_error = _maximum_absolute(
        np.linalg.matrix_power(operator, period) - identity
    )
    if unitarity_error > representation_tolerance:
        raise ValueError("translation is not unitary at the requested tolerance")
    if closure_error > representation_tolerance:
        raise ValueError("translation does not close at the requested period")

    projector = np.zeros_like(operator)
    power = identity.copy()
    for shift in range(period):
        phase = np.exp(2.0j * np.pi * momentum * shift / period)
        projector += phase * power / period
        power = power @ operator
    projector = 0.5 * (projector + projector.conj().T)
    eigenvalue = complex(np.exp(-2.0j * np.pi * momentum / period))
    audit = {
        "translation_unitarity_error": unitarity_error,
        "translation_closure_error": closure_error,
        "projector_hermiticity_error": _maximum_absolute(
            projector - projector.conj().T
        ),
        "projector_idempotency_error": _maximum_absolute(
            projector @ projector - projector
        ),
        "projector_character_residual": _maximum_absolute(
            operator @ projector - eigenvalue * projector
        ),
    }
    return projector, eigenvalue, audit


def _deterministic_projector_pivots(
    projector: np.ndarray,
    rank: int,
    *,
    pivot_tolerance: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Choose independent projector columns by residual diagonal pivoting."""

    selected: list[int] = []
    residuals: list[float] = []
    orthonormal: list[np.ndarray] = []
    base_diagonal = np.real(np.diag(projector))
    for _ in range(rank):
        if orthonormal:
            temporary = np.column_stack(orthonormal)
            residual_diagonal = base_diagonal - np.sum(
                np.abs(temporary) ** 2, axis=1
            )
        else:
            residual_diagonal = base_diagonal.copy()
        residual_diagonal = np.maximum(residual_diagonal, 0.0)
        if selected:
            residual_diagonal[np.asarray(selected, dtype=np.int64)] = -np.inf
        maximum = float(np.max(residual_diagonal))
        if not np.isfinite(maximum) or maximum <= pivot_tolerance:
            raise RuntimeError("projector columns lost rank during pivot selection")
        # np.argmax gives the lowest input-basis index on an exact tie.
        pivot = int(np.argmax(residual_diagonal))
        vector = projector[:, pivot].copy()
        if orthonormal:
            temporary = np.column_stack(orthonormal)
            # Two passes make the rank-revealing step stable for small sectors.
            vector -= temporary @ (temporary.conj().T @ vector)
            vector -= temporary @ (temporary.conj().T @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= np.sqrt(pivot_tolerance):
            raise RuntimeError("selected projector pivot is numerically dependent")
        vector /= norm
        anchor = int(np.argmax(np.abs(vector)))
        if abs(vector[anchor]) > 0.0:
            vector *= np.conj(vector[anchor]) / abs(vector[anchor])
        selected.append(pivot)
        residuals.append(maximum)
        orthonormal.append(vector)
    return tuple(selected), tuple(residuals)


def build_cyclic_character_carrier(
    translation: np.ndarray,
    period: int,
    momentum_integer: int,
    *,
    representation_tolerance: float = 1.0e-10,
    projector_tolerance: float = 1.0e-10,
    pivot_tolerance: float = 1.0e-13,
) -> CharacterCarrier:
    """Build a deterministic Hamiltonian-independent character carrier.

    Projected input-basis columns are selected by deterministic residual
    pivoting.  If ``C = P[:, pivots]`` and ``G = C^dagger C``, the returned
    isometry is the symmetric orthonormalization ``Q = C G^(-1/2)`` (followed
    only by deterministic column phases).  Thus ``Q Q^dagger = P`` without a
    state-dependent projection denominator.
    """

    if not np.isfinite(projector_tolerance) or projector_tolerance <= 0:
        raise ValueError("projector_tolerance must be positive")
    if not np.isfinite(pivot_tolerance) or pivot_tolerance <= 0:
        raise ValueError("pivot_tolerance must be positive")
    projector, eigenvalue, representation_audit = cyclic_character_projector(
        translation,
        period,
        momentum_integer,
        representation_tolerance=representation_tolerance,
    )
    eigenvalues = np.linalg.eigvalsh(projector)
    distance_to_projector_spectrum = float(
        np.max(np.minimum(np.abs(eigenvalues), np.abs(eigenvalues - 1.0)))
    )
    if distance_to_projector_spectrum > projector_tolerance:
        raise ValueError("cyclic average is not a numerical orthogonal projector")
    rank = int(np.count_nonzero(eigenvalues > 0.5))
    if rank < 1:
        raise ValueError("the requested cyclic character has empty support")
    trace_rank_error = abs(float(np.trace(projector).real) - rank)
    if trace_rank_error > projector_tolerance:
        raise ValueError("projector trace does not agree with its numerical rank")

    pivots, residuals = _deterministic_projector_pivots(
        projector, rank, pivot_tolerance=pivot_tolerance
    )
    projected_columns = projector[:, np.asarray(pivots, dtype=np.int64)]
    gram = projected_columns.conj().T @ projected_columns
    gram = 0.5 * (gram + gram.conj().T)
    gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(gram)
    if float(np.min(gram_eigenvalues)) <= pivot_tolerance:
        raise RuntimeError("selected projected columns do not span the character")
    inverse_square_root = (
        gram_eigenvectors
        @ np.diag(1.0 / np.sqrt(gram_eigenvalues))
        @ gram_eigenvectors.conj().T
    )
    columns = projected_columns @ inverse_square_root
    # Fix otherwise irrelevant U(1) phases for deterministic hashes/artifacts.
    for column in range(rank):
        anchor = int(np.argmax(np.abs(columns[:, column])))
        value = columns[anchor, column]
        if abs(value) > 0.0:
            columns[:, column] *= np.conj(value) / abs(value)

    translation_array = np.asarray(translation, dtype=np.complex128)
    identity = np.eye(rank, dtype=np.complex128)
    audit: dict[str, float | int] = {
        **representation_audit,
        "full_dimension": int(projector.shape[0]),
        "carrier_dimension": rank,
        "projector_spectrum_error": distance_to_projector_spectrum,
        "projector_trace_rank_error": trace_rank_error,
        "carrier_isometry_error": _maximum_absolute(
            columns.conj().T @ columns - identity
        ),
        "carrier_projector_reconstruction_error": _maximum_absolute(
            columns @ columns.conj().T - projector
        ),
        "carrier_character_residual": _maximum_absolute(
            translation_array @ columns - eigenvalue * columns
        ),
        "selected_gram_condition_number": float(
            np.max(gram_eigenvalues) / np.min(gram_eigenvalues)
        ),
        "selected_gram_minimum_eigenvalue": float(np.min(gram_eigenvalues)),
        "selected_gram_maximum_eigenvalue": float(np.max(gram_eigenvalues)),
        "minimum_selected_pivot_residual": float(min(residuals)),
    }
    return CharacterCarrier(
        period=int(period),
        momentum_integer=int(momentum_integer) % int(period),
        character_eigenvalue=eigenvalue,
        projector=projector,
        columns=columns,
        pivot_indices=pivots,
        pivot_residuals=residuals,
        audit=audit,
    )


def compress_character_operator(
    operator: np.ndarray,
    translation: np.ndarray,
    carrier: CharacterCarrier,
    *,
    hermiticity_tolerance: float = 1.0e-10,
) -> CompressedCharacterOperator:
    """Compress a Hermitian operator after the carrier has been constructed."""

    matrix = np.asarray(operator, dtype=np.complex128)
    translation_array = np.asarray(translation, dtype=np.complex128)
    expected = (carrier.full_dimension, carrier.full_dimension)
    if matrix.shape != expected or translation_array.shape != expected:
        raise ValueError("operator and translation must match the carrier dimension")
    if not np.isfinite(hermiticity_tolerance) or hermiticity_tolerance <= 0:
        raise ValueError("hermiticity_tolerance must be positive")
    hermiticity_error = _maximum_absolute(matrix - matrix.conj().T)
    if hermiticity_error > hermiticity_tolerance:
        raise ValueError("operator must be Hermitian")

    columns = carrier.columns
    projector = carrier.projector
    compressed = columns.conj().T @ matrix @ columns
    compressed = 0.5 * (compressed + compressed.conj().T)
    applied = matrix @ columns
    reconstructed_action = columns @ compressed
    projected_full = projector @ matrix @ projector

    # An independent spectral coordinate basis comes from diagonalizing P,
    # rather than from the selected projected columns.
    projector_values, projector_vectors = np.linalg.eigh(projector)
    spectral_columns = projector_vectors[:, projector_values > 0.5]
    independent_block = spectral_columns.conj().T @ matrix @ spectral_columns
    independent_block = 0.5 * (independent_block + independent_block.conj().T)
    spectrum_error = _maximum_absolute(
        np.linalg.eigvalsh(compressed) - np.linalg.eigvalsh(independent_block)
    )
    audit = {
        "operator_hermiticity_error": hermiticity_error,
        "operator_translation_commutator_error": _maximum_absolute(
            matrix @ translation_array - translation_array @ matrix
        ),
        "carrier_invariant_action_error": _maximum_absolute(
            applied - reconstructed_action
        ),
        "projected_operator_reconstruction_error": _maximum_absolute(
            columns @ compressed @ columns.conj().T - projected_full
        ),
        "compressed_hermiticity_error": _maximum_absolute(
            compressed - compressed.conj().T
        ),
        "independent_projector_spectrum_error": spectrum_error,
    }
    return CompressedCharacterOperator(matrix=compressed, audit=audit)


def lift_normalized_character_state(
    coefficients: np.ndarray,
    carrier: CharacterCarrier,
    *,
    normalization_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Lift normalized carrier coefficients to the original Hilbert space."""

    values = np.asarray(coefficients, dtype=np.complex128)
    if values.shape != (carrier.dimension,):
        raise ValueError("coefficient vector does not match the carrier dimension")
    norm = float(np.vdot(values, values).real)
    if not np.isfinite(norm) or abs(norm - 1.0) > normalization_tolerance:
        raise ValueError("carrier coefficients must be normalized")
    return carrier.columns @ values


__all__ = [
    "CharacterCarrier",
    "CompressedCharacterOperator",
    "build_cyclic_character_carrier",
    "compress_character_operator",
    "cyclic_character_projector",
    "lift_normalized_character_state",
]
