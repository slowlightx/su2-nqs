"""Exact matrix-support forensic checks for the T3-S1 corrective replay.

This module is deliberately separate from ``sign_frame_obstruction``.  It
decodes the actual fixed-``Sz`` sparse Hamiltonian support back into exchanged
site pairs, recovers decimal exchange weights from the matrix entries, and
uses :class:`fractions.Fraction` plus integer transition counts.  It therefore
does not inherit the reduction-order issue that correctly caused the original
T3-S1 run to fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from .sign_frame_obstruction import (
    gauge_bitstring,
    gauge_bitstring_hash,
)


Edge = tuple[int, int]


@dataclass(frozen=True)
class DecodedExchangeSupport:
    """Canonical off-diagonal support decoded into registered exchange edges."""

    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    edges: tuple[Edge, ...]
    edge_indices: np.ndarray
    recovered_weights: tuple[Fraction, ...]
    multiplicities: tuple[int, ...]
    expected_oriented_multiplicity: int
    maximum_imaginary_value: float
    maximum_exchange_value_error: float
    reverse_missing_count: int
    duplicate_directed_count: int


@dataclass(frozen=True)
class ExactFractionMinimum:
    """Exact minimum and deterministic gauge identity."""

    value: Fraction
    indices: tuple[int, ...]
    bitstrings: tuple[str, ...]
    bitstring_sha256: str


def decimal_fraction(value: str | int | float) -> Fraction:
    """Convert a registered decimal literal to an exact nonnegative fraction."""

    fraction = Fraction(str(value))
    if fraction < 0:
        raise ValueError("exchange weights must be nonnegative")
    return fraction


def fraction_text(value: Fraction) -> str:
    """Serialize a fraction canonically without converting through float."""

    return f"{value.numerator}/{value.denominator}"


def expected_oriented_multiplicity(n_sites: int, n_up: int) -> int:
    """Number of directed fixed-sector flips contributed by one exchange edge."""

    if n_sites < 2 or not 1 <= n_up <= n_sites - 1:
        raise ValueError("the fixed sector must contain both spin values")
    return 2 * math.comb(n_sites - 2, n_up - 1)


def decode_exchange_support(
    matrix: object,
    basis: Sequence[Sequence[int]],
    expected_edge_weights: Mapping[Edge, Fraction],
) -> DecodedExchangeSupport:
    """Decode actual sparse off-diagonal entries into two-site spin exchanges.

    The input is first canonicalized with ``sum_duplicates`` and
    ``eliminate_zeros``.  Every surviving off-diagonal entry must be real and
    positive, connect two fixed-``N_up`` bitstrings by ``01 <-> 10``, belong to
    exactly one registered active edge, have its reverse entry, and equal
    ``J_e/2``.  The recovered weight is obtained independently as
    ``Fraction(str(2*H_rc))`` before comparison with the graph registration.
    """

    try:
        canonical = matrix.tocsr(copy=True)
    except AttributeError as error:
        raise ValueError("matrix must provide a SciPy sparse interface") from error
    canonical.sum_duplicates()
    canonical.eliminate_zeros()
    if canonical.shape[0] != canonical.shape[1]:
        raise ValueError("matrix must be square")
    configurations = np.asarray(basis, dtype=np.int8)
    if configurations.ndim != 2 or configurations.shape[0] != canonical.shape[0]:
        raise ValueError("basis and matrix dimensions differ")
    if np.any((configurations != 0) & (configurations != 1)):
        raise ValueError("basis must contain only bits")
    sector_counts = np.sum(configurations, axis=1)
    if sector_counts.size == 0 or np.any(sector_counts != sector_counts[0]):
        raise ValueError("basis is not a fixed-N_up sector")
    n_sites = int(configurations.shape[1])
    n_up = int(sector_counts[0])

    expected = {
        tuple(sorted((int(edge[0]), int(edge[1])))): decimal_fraction(weight)
        for edge, weight in expected_edge_weights.items()
    }
    if len(expected) != len(expected_edge_weights) or not expected:
        raise ValueError("registered active edges must be nonempty and unique")
    edges = tuple(sorted(expected))
    edge_to_index = {edge: index for index, edge in enumerate(edges)}

    coo = canonical.tocoo(copy=False)
    off_diagonal = coo.row != coo.col
    rows = np.asarray(coo.row[off_diagonal], dtype=np.int64)
    columns = np.asarray(coo.col[off_diagonal], dtype=np.int64)
    complex_values = np.asarray(coo.data[off_diagonal], dtype=np.complex128)
    maximum_imaginary = float(
        np.max(np.abs(np.imag(complex_values)), initial=0.0)
    )
    values = np.asarray(np.real(complex_values), dtype=np.float64)
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("all decoded antiferromagnetic off-diagonal values must be positive")

    directed_pairs = tuple(zip(rows.tolist(), columns.tolist(), strict=True))
    duplicate_directed_count = len(directed_pairs) - len(set(directed_pairs))
    directed_set = set(directed_pairs)
    reverse_missing_count = sum(
        (column, row) not in directed_set for row, column in directed_pairs
    )

    edge_indices = np.empty(values.size, dtype=np.int32)
    recovered_by_edge: dict[Edge, Fraction] = {}
    maximum_value_error = 0.0
    for position, (row, column, value) in enumerate(
        zip(rows, columns, values, strict=True)
    ):
        left = configurations[int(row)]
        right = configurations[int(column)]
        differing = np.flatnonzero(left != right)
        if differing.size != 2:
            raise ValueError("off-diagonal support does not flip exactly two sites")
        first, second = (int(differing[0]), int(differing[1]))
        if not (
            left[first] == right[second]
            and left[second] == right[first]
            and left[first] != left[second]
        ):
            raise ValueError("off-diagonal support is not a 01 <-> 10 exchange")
        edge = (first, second)
        if edge not in edge_to_index:
            raise ValueError(f"matrix support contains unregistered edge {edge}")
        edge_indices[position] = edge_to_index[edge]
        recovered = decimal_fraction(2.0 * float(value))
        previous = recovered_by_edge.setdefault(edge, recovered)
        if previous != recovered:
            raise ValueError("one edge has inconsistent recovered matrix weights")
        registered = expected[edge]
        if recovered != registered:
            raise ValueError("matrix-recovered and registered edge weights differ")
        expected_value = 0.5 * float(registered)
        maximum_value_error = max(
            maximum_value_error, abs(float(value) - expected_value)
        )

    if set(recovered_by_edge) != set(edges):
        raise ValueError("one or more registered active edges are absent from support")
    multiplicities = tuple(
        int(np.count_nonzero(edge_indices == index)) for index in range(len(edges))
    )
    expected_multiplicity = expected_oriented_multiplicity(n_sites, n_up)
    recovered_weights = tuple(recovered_by_edge[edge] for edge in edges)
    return DecodedExchangeSupport(
        rows=rows,
        columns=columns,
        values=values,
        edges=edges,
        edge_indices=edge_indices,
        recovered_weights=recovered_weights,
        multiplicities=multiplicities,
        expected_oriented_multiplicity=expected_multiplicity,
        maximum_imaginary_value=maximum_imaginary,
        maximum_exchange_value_error=float(maximum_value_error),
        reverse_missing_count=int(reverse_missing_count),
        duplicate_directed_count=int(duplicate_directed_count),
    )


def positive_transition_counts(
    support: DecodedExchangeSupport,
    gauges: np.ndarray,
    gauge_diagonals: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, int]:
    """Count actual positive matrix transitions and audit ``D_rD_c=z_iz_j``."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    site_gauges = np.asarray(gauges, dtype=np.int8)
    diagonals = np.asarray(gauge_diagonals, dtype=np.int8)
    if site_gauges.ndim != 2 or diagonals.ndim != 2:
        raise ValueError("gauges and gauge diagonals must be rank two")
    if site_gauges.shape[0] != diagonals.shape[0]:
        raise ValueError("one configuration diagonal is required per gauge")
    if np.any((site_gauges != -1) & (site_gauges != 1)) or np.any(
        (diagonals != -1) & (diagonals != 1)
    ):
        raise ValueError("all gauge values must be +/-1")
    if support.rows.size and int(
        max(np.max(support.rows), np.max(support.columns))
    ) >= diagonals.shape[1]:
        raise ValueError("support and configuration diagonals differ")

    edge_first = np.asarray([edge[0] for edge in support.edges], dtype=np.int32)
    edge_second = np.asarray([edge[1] for edge in support.edges], dtype=np.int32)
    transition_first = edge_first[support.edge_indices]
    transition_second = edge_second[support.edge_indices]
    counts = np.zeros((site_gauges.shape[0], len(support.edges)), dtype=np.int32)
    mismatch_count = 0
    edge_masks = [
        support.edge_indices == edge_index
        for edge_index in range(len(support.edges))
    ]
    for start in range(0, site_gauges.shape[0], chunk_size):
        stop = min(start + chunk_size, site_gauges.shape[0])
        actual = (
            diagonals[start:stop, support.rows]
            * diagonals[start:stop, support.columns]
        )
        expected = (
            site_gauges[start:stop, transition_first]
            * site_gauges[start:stop, transition_second]
        )
        mismatch_count += int(np.count_nonzero(actual != expected))
        for edge_index, mask in enumerate(edge_masks):
            counts[start:stop, edge_index] = np.count_nonzero(
                actual[:, mask] > 0, axis=1
            )
    return counts, mismatch_count


def exact_graph_costs(
    gauges: np.ndarray,
    edges: Sequence[Edge],
    weights: Sequence[Fraction],
    *,
    weight_power: int,
) -> tuple[Fraction, ...]:
    """Exact graph-side Phi1/Phi2 for every site gauge."""

    if weight_power not in (1, 2):
        raise ValueError("weight_power must be one or two")
    site_gauges = np.asarray(gauges, dtype=np.int8)
    if site_gauges.ndim != 2 or len(edges) != len(weights):
        raise ValueError("gauge, edge, and weight dimensions differ")
    powered = tuple(decimal_fraction(value) ** weight_power for value in weights)
    denominator = sum(powered, Fraction(0))
    if denominator <= 0:
        raise ValueError("graph denominator must be positive")
    answers = []
    for gauge in site_gauges:
        numerator = sum(
            (
                weight
                for (first, second), weight in zip(edges, powered, strict=True)
                if gauge[first] == gauge[second]
            ),
            Fraction(0),
        )
        answers.append(numerator / denominator)
    return tuple(answers)


def exact_matrix_costs(
    positive_counts: np.ndarray,
    multiplicities: Sequence[int],
    recovered_weights: Sequence[Fraction],
    *,
    weight_power: int,
) -> tuple[Fraction, ...]:
    """Exact matrix-side Phi1/Phi2 from integer actual-support counts."""

    if weight_power not in (1, 2):
        raise ValueError("weight_power must be one or two")
    counts = np.asarray(positive_counts, dtype=np.int64)
    multiplicity = tuple(int(value) for value in multiplicities)
    if counts.ndim != 2 or counts.shape[1] != len(multiplicity):
        raise ValueError("positive counts and multiplicities differ")
    if len(recovered_weights) != len(multiplicity):
        raise ValueError("one recovered weight is required per edge")
    powered = tuple(
        decimal_fraction(value) ** weight_power for value in recovered_weights
    )
    denominator = sum(
        (count * weight for count, weight in zip(multiplicity, powered, strict=True)),
        Fraction(0),
    )
    if denominator <= 0:
        raise ValueError("matrix denominator must be positive")
    answers = []
    for row in counts:
        numerator = sum(
            (
                int(count) * weight
                for count, weight in zip(row, powered, strict=True)
            ),
            Fraction(0),
        )
        answers.append(numerator / denominator)
    return tuple(answers)


def exact_fraction_minimum(
    costs: Sequence[Fraction], gauges: np.ndarray
) -> ExactFractionMinimum:
    """Return the exact minimum without a floating tie tolerance."""

    values = tuple(Fraction(value) for value in costs)
    site_gauges = np.asarray(gauges, dtype=np.int8)
    if not values or len(values) != site_gauges.shape[0]:
        raise ValueError("one exact cost is required per gauge")
    minimum = min(values)
    indices = tuple(index for index, value in enumerate(values) if value == minimum)
    bitstrings = tuple(gauge_bitstring(site_gauges[index]) for index in indices)
    return ExactFractionMinimum(
        value=minimum,
        indices=indices,
        bitstrings=bitstrings,
        bitstring_sha256=gauge_bitstring_hash(bitstrings),
    )


def stable_fsum_matrix_costs(
    support: DecodedExchangeSupport,
    positive_counts: np.ndarray,
    *,
    weight_power: int,
) -> np.ndarray:
    """Stable secondary matrix fractions from the same actual COO support.

    The audited site gauge gives a uniform sign to all transitions belonging
    to one edge.  Each edge block is therefore summed with ``math.fsum`` and a
    second ``math.fsum`` combines precisely the selected actual-support blocks.
    """

    if weight_power not in (1, 2):
        raise ValueError("weight_power must be one or two")
    counts = np.asarray(positive_counts, dtype=np.int64)
    multiplicities = np.asarray(support.multiplicities, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != len(support.edges):
        raise ValueError("positive counts and support differ")
    if np.any((counts != 0) & (counts != multiplicities[None, :])):
        raise ValueError("a site gauge must give one uniform sign per edge block")
    edge_sums = []
    for edge_index in range(len(support.edges)):
        values = support.values[support.edge_indices == edge_index]
        if weight_power == 1:
            edge_sums.append(math.fsum(abs(float(value)) for value in values))
        else:
            edge_sums.append(
                math.fsum(float(value) * float(value) for value in values)
            )
    denominator = math.fsum(edge_sums)
    return np.asarray(
        [
            math.fsum(
                edge_sums[index]
                for index, count in enumerate(row)
                if count == multiplicities[index]
            )
            / denominator
            for row in counts
        ],
        dtype=np.float64,
    )


def naive_ratio_forward_error_bound(term_count: int) -> Mapping[str, float]:
    """Conservative double-precision sequential-sum ratio error estimate."""

    if term_count < 1:
        raise ValueError("term_count must be positive")
    unit_roundoff = float(np.finfo(np.float64).eps / 2.0)
    gamma = term_count * unit_roundoff / (1.0 - term_count * unit_roundoff)
    ratio_bound = 2.0 * gamma / (1.0 - gamma)
    return {
        "term_count": int(term_count),
        "float64_epsilon": float(np.finfo(np.float64).eps),
        "unit_roundoff": unit_roundoff,
        "gamma_n": float(gamma),
        "conservative_ratio_absolute_error_bound": float(ratio_bound),
    }


__all__ = [
    "DecodedExchangeSupport",
    "Edge",
    "ExactFractionMinimum",
    "decimal_fraction",
    "decode_exchange_support",
    "exact_fraction_minimum",
    "exact_graph_costs",
    "exact_matrix_costs",
    "expected_oriented_multiplicity",
    "fraction_text",
    "naive_ratio_forward_error_bound",
    "positive_transition_counts",
    "stable_fsum_matrix_costs",
]
