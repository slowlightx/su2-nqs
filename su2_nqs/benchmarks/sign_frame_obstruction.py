"""Exact-small diagnostics for site-local Marshall sign frames.

The routines in this module do not define a physical observable and do not
attempt to cure a generic sign problem.  They quantify the obstruction that
remains when a real antiferromagnetic Heisenberg Hamiltonian is restricted to
site-factorized ``Z2`` rephasings.  The same convention is also used to audit
Born-weighted ground-state signs in a fixed-``Sz`` computational basis.

Bits follow the project convention: bit zero is spin down, bit one is spin up,
and site zero is the most significant bit whenever configurations are encoded
as integers elsewhere in the repository.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math

import numpy as np


Bond = tuple[int, int, float]


@dataclass(frozen=True)
class BornSignStatistics:
    """Born-weighted residual-sign data after applying one diagonal gauge."""

    positive_weight: float
    negative_weight: float
    zero_weight: float
    coherence: float
    violation: float


@dataclass(frozen=True)
class GaugeMinimum:
    """Minimum of a gauge cost and the lexicographically ordered minimizers."""

    value: float
    indices: tuple[int, ...]
    bitstrings: tuple[str, ...]
    bitstring_sha256: str


def aggregate_antiferromagnetic_bonds(
    n_sites: int,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
) -> tuple[Bond, ...]:
    """Validate and aggregate nonnegative unordered exchange bonds.

    Zero-weight bonds are discarded after aggregation.  Negative exchange is
    deliberately outside this diagnostic: for mixed signed couplings the
    desired edge phase and hence the frustration indicator must be generalized
    explicitly rather than inferred silently.
    """

    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    combined: dict[tuple[int, int], float] = {}
    for raw in bonds:
        if len(raw) == 2:
            first, second = raw
            coupling = 1.0
        elif len(raw) == 3:
            first, second, coupling = raw
        else:
            raise ValueError(f"invalid bond: {raw!r}")
        first, second = int(first), int(second)
        coupling = float(coupling)
        if not math.isfinite(coupling) or coupling < 0.0:
            raise ValueError("Marshall-obstruction bonds must have finite J >= 0")
        if first > second:
            first, second = second, first
        if not 0 <= first < second < n_sites:
            raise ValueError(
                f"bond sites must satisfy 0 <= i < j < n_sites: {raw!r}"
            )
        combined[(first, second)] = combined.get((first, second), 0.0) + coupling
    result = tuple(
        (first, second, coupling)
        for (first, second), coupling in sorted(combined.items())
        if coupling > 0.0
    )
    if not result:
        raise ValueError("at least one positive exchange bond is required")
    return result


def enumerate_site_z2_gauges(n_sites: int) -> np.ndarray:
    """Return all gauges with ``z[0] = +1`` in lexicographic ``-1,+1`` order.

    Fixing site zero removes the global redundancy in a fixed-magnetization
    sector.  No equivalence with a full-Hilbert-space gauge is implied.
    """

    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    count = 1 << (n_sites - 1)
    gauges = np.empty((count, n_sites), dtype=np.int8)
    gauges[:, 0] = 1
    for row in range(count):
        for offset in range(n_sites - 1):
            bit = (row >> (n_sites - offset - 2)) & 1
            gauges[row, offset + 1] = 1 if bit else -1
    return gauges


def gauge_bitstring(gauge: Sequence[int]) -> str:
    """Encode a ``+/-1`` site gauge deterministically as a binary string."""

    values = tuple(int(value) for value in gauge)
    if not values or any(value not in (-1, 1) for value in values):
        raise ValueError("a gauge must contain only -1 and +1")
    return "".join("1" if value == 1 else "0" for value in values)


def gauge_bitstring_hash(bitstrings: Sequence[str]) -> str:
    """Return a stable SHA256 for an ordered gauge-bitstring list."""

    payload = "\n".join(str(value) for value in bitstrings).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def configuration_gauge_diagonal(
    basis: Sequence[Sequence[int]],
    gauge: Sequence[int],
) -> np.ndarray:
    """Return ``prod_{i: bit_i=0} z_i`` for every basis configuration."""

    gauge_array = np.asarray(gauge, dtype=np.int8)
    if gauge_array.ndim != 1 or gauge_array.size < 2:
        raise ValueError("gauge must be a one-dimensional site vector")
    if np.any((gauge_array != -1) & (gauge_array != 1)):
        raise ValueError("gauge entries must be -1 or +1")
    configurations = np.asarray(basis, dtype=np.int8)
    if configurations.ndim != 2 or configurations.shape[1] != gauge_array.size:
        raise ValueError("basis configurations and gauge size do not match")
    if np.any((configurations != 0) & (configurations != 1)):
        raise ValueError("basis configurations must contain only bits")
    selected = np.where(configurations == 0, gauge_array[None, :], 1)
    return np.prod(selected, axis=1, dtype=np.int8)


def all_configuration_gauge_diagonals(
    basis: Sequence[Sequence[int]],
    gauges: np.ndarray,
) -> np.ndarray:
    """Vectorized computational diagonals for a collection of site gauges."""

    gauge_array = np.asarray(gauges, dtype=np.int8)
    configurations = np.asarray(basis, dtype=np.int8)
    if gauge_array.ndim != 2 or configurations.ndim != 2:
        raise ValueError("gauges and basis must both be rank-two arrays")
    if gauge_array.shape[1] != configurations.shape[1]:
        raise ValueError("gauges and configurations must have the same site count")
    if np.any((gauge_array != -1) & (gauge_array != 1)):
        raise ValueError("gauge entries must be -1 or +1")
    if np.any((configurations != 0) & (configurations != 1)):
        raise ValueError("basis configurations must contain only bits")
    result = np.ones(
        (gauge_array.shape[0], configurations.shape[0]), dtype=np.int8
    )
    for site in range(configurations.shape[1]):
        result *= np.where(
            configurations[None, :, site] == 0,
            gauge_array[:, None, site],
            1,
        ).astype(np.int8)
    return result


def edge_obstruction_costs(
    gauges: np.ndarray,
    bonds: Sequence[Bond],
    *,
    weight_power: int,
) -> np.ndarray:
    """Return the positive-edge fraction for every gauge.

    ``weight_power=1`` is the normalized off-diagonal L1 obstruction
    ``Phi1``.  ``weight_power=2`` is the independently minimized squared
    Frobenius fraction ``Phi2``.
    """

    if weight_power not in (1, 2):
        raise ValueError("weight_power must be one or two")
    gauge_array = np.asarray(gauges, dtype=np.int8)
    if gauge_array.ndim != 2:
        raise ValueError("gauges must be a rank-two array")
    if np.any((gauge_array != -1) & (gauge_array != 1)):
        raise ValueError("gauge entries must be -1 or +1")
    numerator = np.zeros(gauge_array.shape[0], dtype=np.float64)
    denominator = 0.0
    for first, second, coupling in bonds:
        if not 0 <= first < second < gauge_array.shape[1]:
            raise ValueError("bond endpoint lies outside the gauge")
        coupling = float(coupling)
        if not math.isfinite(coupling) or coupling < 0.0:
            raise ValueError("edge weights must be finite and nonnegative")
        weight = coupling ** weight_power
        denominator += weight
        numerator += weight * (gauge_array[:, first] == gauge_array[:, second])
    if not denominator > 0.0:
        raise ValueError("bond weight denominator must be positive")
    return numerator / denominator


def gauge_minimum(
    costs: Sequence[float],
    gauges: np.ndarray,
    *,
    tie_tolerance: float,
) -> GaugeMinimum:
    """Collect all cost minimizers using a frozen absolute tie tolerance."""

    if tie_tolerance < 0.0 or not math.isfinite(tie_tolerance):
        raise ValueError("tie_tolerance must be finite and nonnegative")
    values = np.asarray(costs, dtype=np.float64)
    gauge_array = np.asarray(gauges, dtype=np.int8)
    if values.ndim != 1 or values.size != gauge_array.shape[0]:
        raise ValueError("one cost is required for every gauge")
    if not np.all(np.isfinite(values)):
        raise ValueError("gauge costs must be finite")
    minimum = float(np.min(values))
    indices = tuple(
        int(index)
        for index in np.flatnonzero(np.abs(values - minimum) <= tie_tolerance)
    )
    bitstrings = tuple(gauge_bitstring(gauge_array[index]) for index in indices)
    if tuple(sorted(bitstrings)) != bitstrings:
        raise RuntimeError("gauge minimizers are not in lexicographic order")
    return GaugeMinimum(
        value=minimum,
        indices=indices,
        bitstrings=bitstrings,
        bitstring_sha256=gauge_bitstring_hash(bitstrings),
    )


def matrix_positive_offdiagonal_fractions(
    matrix: object,
    gauge_diagonals: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute L1 and squared-Frobenius positive off-diagonal fractions.

    The calculation starts from the actual fixed-``Sz`` matrix support and is
    therefore an independent closure of :func:`edge_obstruction_costs` for the
    antiferromagnetic Heisenberg graph.  Both orientations of a Hermitian
    matrix element may be present; they cancel between numerator and
    denominator in the normalized fractions.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if hasattr(matrix, "tocoo"):
        coo = matrix.tocoo(copy=True)
        if len(coo.shape) != 2 or coo.shape[0] != coo.shape[1]:
            raise ValueError("matrix must be square")
        matrix_dimension = int(coo.shape[0])
        coo.sum_duplicates()
        rows = np.asarray(coo.row)
        columns = np.asarray(coo.col)
        values = np.asarray(coo.data, dtype=np.complex128)
    else:
        dense = np.asarray(matrix, dtype=np.complex128)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError("matrix must be square")
        matrix_dimension = int(dense.shape[0])
        rows, columns = np.nonzero(dense)
        values = dense[rows, columns]
    diagonal = np.asarray(gauge_diagonals, dtype=np.int8)
    if diagonal.ndim != 2:
        raise ValueError("gauge_diagonals must be rank two")
    if diagonal.shape[1] != matrix_dimension:
        raise ValueError("matrix dimension and gauge diagonals do not match")
    if np.any((diagonal != -1) & (diagonal != 1)):
        raise ValueError("gauge diagonals must contain only -1 and +1")
    off_diagonal = rows != columns
    rows, columns, values = rows[off_diagonal], columns[off_diagonal], values[off_diagonal]
    if values.size == 0:
        raise ValueError("matrix has no off-diagonal support")
    maximum_imaginary = float(np.max(np.abs(np.imag(values)), initial=0.0))
    real_values = np.real(values)
    l1_denominator = float(np.sum(np.abs(real_values)))
    l2_denominator = float(np.sum(np.square(np.abs(real_values))))
    if l1_denominator <= 0.0 or l2_denominator <= 0.0:
        raise ValueError("matrix off-diagonal denominator must be positive")
    l1 = np.empty(diagonal.shape[0], dtype=np.float64)
    l2 = np.empty(diagonal.shape[0], dtype=np.float64)
    for start in range(0, diagonal.shape[0], chunk_size):
        stop = min(start + chunk_size, diagonal.shape[0])
        signs = diagonal[start:stop, rows] * diagonal[start:stop, columns]
        transformed = signs * real_values[None, :]
        positive = np.maximum(transformed, 0.0)
        l1[start:stop] = np.sum(positive, axis=1) / l1_denominator
        l2[start:stop] = np.sum(np.square(positive), axis=1) / l2_denominator
    return l1, l2, maximum_imaginary


def phase_frame_cost(angles: Sequence[float], bonds: Sequence[Bond]) -> float:
    """Return the fixed-frame chordal exchange-phase mismatch.

    The normalized cost is ``sum J(1+cos(theta_i-theta_j))/2sum J``.
    It is a classical exchange-frame descriptor.  For angles restricted to
    zero and pi it equals ``Phi1``; for a 120-degree frame the transformed
    Hamiltonian is generally complex and this value must not be called a
    positive-off-diagonal or stoquasticity fraction.
    """

    values = np.asarray(angles, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("angles must be a finite one-dimensional vector")
    numerator = 0.0
    denominator = 0.0
    for first, second, coupling in bonds:
        if not 0 <= first < second < values.size:
            raise ValueError("bond endpoint lies outside the phase frame")
        coupling = float(coupling)
        if not math.isfinite(coupling) or coupling < 0.0:
            raise ValueError("edge weights must be finite and nonnegative")
        numerator += coupling * (
            1.0 + math.cos(float(values[first] - values[second]))
        ) / 2.0
        denominator += coupling
    if denominator <= 0.0:
        raise ValueError("bond weight denominator must be positive")
    return float(numerator / denominator)


def named_order_frames(
    lattice: str,
    rows: int,
    columns: int,
) -> Mapping[str, np.ndarray]:
    """Return preregistered square or triangular local ``Sz`` rotations."""

    if lattice not in ("square", "triangular"):
        raise ValueError("lattice must be square or triangular")
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    coordinates = tuple(divmod(site, columns) for site in range(rows * columns))
    if lattice == "square":
        return {
            "neel": np.asarray(
                [math.pi * ((row + column) % 2) for row, column in coordinates]
            ),
            "stripe_x": np.asarray(
                [math.pi * (column % 2) for row, column in coordinates]
            ),
            "stripe_y": np.asarray(
                [math.pi * (row % 2) for row, column in coordinates]
            ),
        }
    colors = np.asarray([(row + column) % 3 for row, column in coordinates])
    return {
        "stripe_x": np.asarray(
            [math.pi * (column % 2) for row, column in coordinates]
        ),
        "stripe_y": np.asarray(
            [math.pi * (row % 2) for row, column in coordinates]
        ),
        "stripe_xy": np.asarray(
            [math.pi * ((row + column) % 2) for row, column in coordinates]
        ),
        "120_plus": 2.0 * math.pi * colors / 3.0,
        "120_minus": -2.0 * math.pi * colors / 3.0,
    }


def z2_gauge_from_angles(
    angles: Sequence[float],
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Convert a zero/pi frame to its normalized ``z[0]=+1`` gauge."""

    values = np.asarray(angles, dtype=np.float64)
    phases = np.exp(1.0j * values)
    if float(np.max(np.abs(np.imag(phases)), initial=0.0)) > tolerance:
        raise ValueError("the frame is not Z2-valued")
    gauge = np.where(np.real(phases) >= 0.0, 1, -1).astype(np.int8)
    gauge *= gauge[0]
    return gauge


def site_frame_overlap(first: Sequence[int], second: Sequence[int]) -> float:
    """Absolute normalized overlap of two real site gauges."""

    left = np.asarray(first, dtype=np.int8)
    right = np.asarray(second, dtype=np.int8)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("site gauges must have the same one-dimensional shape")
    if np.any((left != -1) & (left != 1)) or np.any((right != -1) & (right != 1)):
        raise ValueError("site gauges must contain only -1 and +1")
    return float(abs(np.dot(left.astype(np.float64), right.astype(np.float64))) / left.size)


def born_sign_statistics(
    aligned_real_state: Sequence[float],
    gauge_diagonal: Sequence[int],
    *,
    zero_tolerance: float,
) -> BornSignStatistics:
    """Return ``W+``, ``W-``, ``W0``, coherence, and minority weight.

    ``W0`` contains the Born mass of coefficients whose absolute amplitude is
    at most the frozen tolerance.  ``violation=min(W+,W-)`` is insensitive to
    the global sign of the eigenvector and is not a configuration fraction.
    """

    if zero_tolerance < 0.0 or not math.isfinite(zero_tolerance):
        raise ValueError("zero_tolerance must be finite and nonnegative")
    state = np.asarray(aligned_real_state, dtype=np.float64)
    diagonal = np.asarray(gauge_diagonal, dtype=np.int8)
    if state.ndim != 1 or diagonal.shape != state.shape:
        raise ValueError("state and gauge diagonal must be matching vectors")
    if np.any((diagonal != -1) & (diagonal != 1)):
        raise ValueError("gauge diagonal entries must be -1 or +1")
    norm = float(np.dot(state, state))
    if not norm > 0.0:
        raise ValueError("state norm must be positive")
    probabilities = np.square(state) / norm
    transformed = diagonal * state
    positive = transformed > zero_tolerance
    negative = transformed < -zero_tolerance
    zero = ~(positive | negative)
    w_plus = float(np.sum(probabilities[positive]))
    w_minus = float(np.sum(probabilities[negative]))
    w_zero = float(np.sum(probabilities[zero]))
    return BornSignStatistics(
        positive_weight=w_plus,
        negative_weight=w_minus,
        zero_weight=w_zero,
        coherence=abs(w_plus - w_minus),
        violation=min(w_plus, w_minus),
    )


def all_born_sign_statistics(
    aligned_real_state: Sequence[float],
    gauge_diagonals: np.ndarray,
    *,
    zero_tolerance: float,
) -> Mapping[str, np.ndarray]:
    """Vectorized Born sign statistics for every site gauge."""

    state = np.asarray(aligned_real_state, dtype=np.float64)
    diagonals = np.asarray(gauge_diagonals, dtype=np.int8)
    if diagonals.ndim != 2 or state.ndim != 1 or diagonals.shape[1] != state.size:
        raise ValueError("state and gauge diagonals have incompatible shapes")
    if np.any((diagonals != -1) & (diagonals != 1)):
        raise ValueError("gauge diagonal entries must be -1 or +1")
    if zero_tolerance < 0.0 or not math.isfinite(zero_tolerance):
        raise ValueError("zero_tolerance must be finite and nonnegative")
    norm = float(np.dot(state, state))
    if not norm > 0.0:
        raise ValueError("state norm must be positive")
    probabilities = np.square(state) / norm
    nonzero = np.abs(state) > zero_tolerance
    zero_weight = float(np.sum(probabilities[~nonzero]))
    signed_weights = probabilities * np.sign(state) * nonzero
    signed_sums = diagonals.astype(np.float64) @ signed_weights
    active_weight = max(0.0, 1.0 - zero_weight)
    positive = np.clip((active_weight + signed_sums) / 2.0, 0.0, 1.0)
    negative = np.clip((active_weight - signed_sums) / 2.0, 0.0, 1.0)
    return {
        "positive_weight": positive,
        "negative_weight": negative,
        "zero_weight": np.full(diagonals.shape[0], zero_weight),
        "coherence": np.abs(signed_sums),
        "violation": np.minimum(positive, negative),
    }


def align_state_real(
    state: Sequence[complex],
) -> tuple[np.ndarray, float, float]:
    """Align the largest coefficient positive and return real/imaginary audits."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("state must be a nonempty vector")
    norm = float(np.linalg.norm(vector))
    if not norm > 0.0:
        raise ValueError("state norm must be positive")
    vector = vector / norm
    pivot = int(np.argmax(np.abs(vector)))
    phase = np.angle(vector[pivot])
    aligned = vector * np.exp(-1.0j * phase)
    if np.real(aligned[pivot]) < 0.0:
        aligned = -aligned
    imaginary_norm = float(np.linalg.norm(np.imag(aligned)))
    return np.asarray(np.real(aligned), dtype=np.float64), imaginary_norm, norm


def optimal_edge_frequencies(
    gauges: np.ndarray,
    minimizer_indices: Sequence[int],
    bonds: Sequence[Bond],
) -> np.ndarray:
    """Uniform-optimum frequency that each bond remains positive."""

    gauge_array = np.asarray(gauges, dtype=np.int8)
    indices = np.asarray(tuple(int(value) for value in minimizer_indices), dtype=int)
    if indices.size == 0:
        raise ValueError("at least one minimizer is required")
    selected = gauge_array[indices]
    return np.asarray(
        [
            np.mean(selected[:, first] == selected[:, second], dtype=np.float64)
            for first, second, _ in bonds
        ],
        dtype=np.float64,
    )


__all__ = [
    "Bond",
    "BornSignStatistics",
    "GaugeMinimum",
    "aggregate_antiferromagnetic_bonds",
    "align_state_real",
    "all_born_sign_statistics",
    "all_configuration_gauge_diagonals",
    "born_sign_statistics",
    "configuration_gauge_diagonal",
    "edge_obstruction_costs",
    "enumerate_site_z2_gauges",
    "gauge_bitstring",
    "gauge_bitstring_hash",
    "gauge_minimum",
    "matrix_positive_offdiagonal_fractions",
    "named_order_frames",
    "optimal_edge_frequencies",
    "phase_frame_cost",
    "site_frame_overlap",
    "z2_gauge_from_angles",
]
