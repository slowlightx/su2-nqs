"""Independent N=36 amplitude oracle for the XDiag seed-fill canary.

The production C++ executable obtains representatives in XDiag's bit order
and explicitly maps them to repository row-major sites before evaluation.
This module reproduces that conversion independently and evaluates direct
induced-parton and collective-spin formulas.  It contains no XDiag bindings
and no model-operator code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from math import comb, factorial, sqrt
from numbers import Integral

import numpy as np


N_SITES = 36
GROUP_ORDER = 432
XDIAG_TO_REPOSITORY = (
    0,
    21,
    8,
    29,
    16,
    31,
    3,
    18,
    11,
    26,
    13,
    34,
    7,
    28,
    15,
    30,
    2,
    23,
    10,
    25,
    12,
    33,
    5,
    20,
    14,
    35,
    1,
    22,
    9,
    24,
    17,
    32,
    4,
    19,
    6,
    27,
)
SEED_NAMES = ("pi0_DSL_A1", "120_tower_A1", "stripe_tower_A1")


def _triangular_pi_zero_hopping(anti_x: bool, anti_y: bool) -> np.ndarray:
    hopping = np.zeros((N_SITES, N_SITES), dtype=np.float64)
    for y in range(6):
        for x in range(6):
            source = y * 6 + x
            for dx, dy in ((1, 0), (0, 1), (1, 1)):
                target = ((y + dy) % 6) * 6 + (x + dx) % 6
                value = 1.0 if (dx, dy) == (1, 0) else (-1.0) ** x
                if anti_x and x + dx >= 6:
                    value *= -1.0
                if anti_y and y + dy >= 6:
                    value *= -1.0
                hopping[source, target] = value
                hopping[target, source] = value
    return hopping


@lru_cache(maxsize=1)
def _pi_zero_pairings() -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for anti_x, anti_y in ((False, False), (True, False), (False, True)):
        eigenvalues, eigenvectors = np.linalg.eigh(
            _triangular_pi_zero_hopping(anti_x, anti_y)
        )
        if eigenvalues[18] - eigenvalues[17] <= 1.0e-10:
            raise RuntimeError("registered pi0 boundary is not closed shell")
        occupied = eigenvectors[:, :18]
        result.append(occupied @ occupied.T)
    return tuple(result)


def _configuration_sign(states: np.ndarray) -> np.ndarray:
    left_down = np.cumsum(1 - states, axis=1) - (1 - states)
    inversions = np.sum(states * left_down, axis=1)
    parity = (inversions + 18 * 17 // 2) % 2
    return np.where(parity == 0, 1.0, -1.0)


def _dsl_amplitudes(states: np.ndarray) -> np.ndarray:
    output = np.zeros(states.shape[0], dtype=np.float64)
    coefficients = (1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0)
    signs = _configuration_sign(states)
    for row, state in enumerate(states):
        up = np.flatnonzero(state == 1)
        down = np.flatnonzero(state == 0)
        output[row] = signs[row] * sum(
            coefficient * np.linalg.det(pairing[np.ix_(up, down)])
            for coefficient, pairing in zip(
                coefficients, _pi_zero_pairings(), strict=True
            )
        )
    return output


def _valid_m(q: int, m: int) -> bool:
    return q >= 0 and abs(m) <= q and (q - m) % 2 == 0


def _cg(q1: int, m1: int, q2: int, m2: int, q: int, m: int) -> float:
    if (
        not _valid_m(q1, m1)
        or not _valid_m(q2, m2)
        or not _valid_m(q, m)
        or m1 + m2 != m
        or abs(q1 - q2) > q
        or q > q1 + q2
        or (q1 + q2 + q) % 2
    ):
        return 0.0

    def fact(value: int) -> int:
        return factorial(value) if value >= 0 else 0

    a = (q1 + q2 - q) // 2
    b = (q1 - m1) // 2
    c = (q2 + m2) // 2
    d = (q - q2 + m1) // 2
    e = (q - q1 - m2) // 2
    lower = max(0, -d, -e)
    upper = min(a, b, c)
    if lower > upper:
        return 0.0
    triangle = (
        (q + 1)
        * fact((q + q1 - q2) // 2)
        * fact((q - q1 + q2) // 2)
        * fact((q1 + q2 - q) // 2)
        / fact((q1 + q2 + q) // 2 + 1)
    )
    magnetic = (
        fact((q + m) // 2)
        * fact((q - m) // 2)
        * fact((q1 - m1) // 2)
        * fact((q1 + m1) // 2)
        * fact((q2 - m2) // 2)
        * fact((q2 + m2) // 2)
    )
    series = 0.0
    for k in range(lower, upper + 1):
        denominator = (
            fact(k)
            * fact(a - k)
            * fact(b - k)
            * fact(c - k)
            * fact(d + k)
            * fact(e + k)
        )
        series += (-1.0 if k % 2 else 1.0) / denominator
    return sqrt(triangle * magnetic) * series


def _two_sublattice_singlet(
    states: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    output = np.zeros(states.shape[0], dtype=np.float64)
    q = int(np.sum(labels == 0))
    if q != int(np.sum(labels == 1)):
        raise RuntimeError("stripe partition is not balanced")
    for row, state in enumerate(states):
        n_up = [int(np.sum(state[labels == label])) for label in (0, 1)]
        m = [2 * value - q for value in n_up]
        output[row] = _cg(q, m[0], q, m[1], 0, 0) / sqrt(
            comb(q, n_up[0]) * comb(q, n_up[1])
        )
    return output


def _stripe_amplitudes(states: np.ndarray) -> np.ndarray:
    coordinates = tuple((x, y) for y in range(6) for x in range(6))
    partitions = (
        np.asarray([x % 2 for x, _ in coordinates]),
        np.asarray([y % 2 for _, y in coordinates]),
        np.asarray([(x + y) % 2 for x, y in coordinates]),
    )
    return sum(
        (_two_sublattice_singlet(states, labels) for labels in partitions),
        start=np.zeros(states.shape[0]),
    ) / 3.0


def _tower_120_amplitudes(states: np.ndarray) -> np.ndarray:
    labels = np.asarray([(x + y) % 3 for y in range(6) for x in range(6)])
    q = int(np.sum(labels == 0))
    if any(int(np.sum(labels == label)) != q for label in range(3)):
        raise RuntimeError("120 partition is not balanced")
    output = np.zeros(states.shape[0], dtype=np.float64)
    for row, state in enumerate(states):
        n_up = [int(np.sum(state[labels == label])) for label in range(3)]
        m = [2 * value - q for value in n_up]
        if sum(m) != 0:
            continue
        output[row] = (
            _cg(q, m[0], q, m[1], q, m[0] + m[1])
            * _cg(q, m[0] + m[1], q, m[2], 0, 0)
            / sqrt(comb(q, n_up[0]) * comb(q, n_up[1]) * comb(q, n_up[2]))
        )
    return output


def repository_configurations_from_xdiag_uints(
    representatives: Sequence[int],
    *,
    site_map: Sequence[int] = XDIAG_TO_REPOSITORY,
) -> np.ndarray:
    """Decode XDiag integers and reorder sites into repository row-major order."""

    mapping = tuple(int(site) for site in site_map)
    if len(mapping) != N_SITES or sorted(mapping) != list(range(N_SITES)):
        raise ValueError("site_map must be a permutation of 0..35")
    states = np.zeros((len(representatives), N_SITES), dtype=np.int8)
    limit = 1 << N_SITES
    for row, raw in enumerate(representatives):
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise TypeError("representatives must contain integers")
        value = int(raw)
        if value < 0 or value >= limit:
            raise ValueError("representative does not fit 36 bits")
        for xdiag_site, repository_site in enumerate(mapping):
            states[row, repository_site] = (value >> xdiag_site) & 1
    if np.any(np.sum(states, axis=1) != N_SITES // 2):
        raise ValueError("every P2 representative must have n_up=18")
    return states


def n36_physical_seed_amplitudes_from_xdiag_uints(
    representatives: Sequence[int],
    *,
    site_map: Sequence[int] = XDIAG_TO_REPOSITORY,
) -> np.ndarray:
    """Return DSL, 120-tower and stripe-tower amplitudes for XDiag states."""

    states = repository_configurations_from_xdiag_uints(
        representatives, site_map=site_map
    )
    dsl = _dsl_amplitudes(states)
    tower_120 = _tower_120_amplitudes(states)
    stripe = _stripe_amplitudes(states)
    result = np.stack((dsl, tower_120, stripe), axis=1)
    if result.shape != (len(representatives), len(SEED_NAMES)):
        raise RuntimeError("independent N36 seed oracle returned a wrong shape")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("independent N36 seed oracle is nonfinite")
    return np.asarray(result, dtype=np.float64)


def held_out_amplitude_errors(
    records: Sequence[Mapping[str, object]],
    *,
    site_map: Sequence[int] = XDIAG_TO_REPOSITORY,
    group_order: int = GROUP_ORDER,
) -> dict[str, object]:
    """Compare emitted C++ held-out records with the independent Python oracle."""

    if not records:
        raise ValueError("held-out records must be nonempty")
    ordinals = [int(record["representative_ordinal"]) for record in records]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
        raise ValueError("held-out records must have unique sorted ordinals")
    integers = [int(record["representative_xdiag_uint"]) for record in records]
    expected_physical = n36_physical_seed_amplitudes_from_xdiag_uints(
        integers, site_map=site_map
    )
    observed_physical = np.asarray(
        [record["physical_amplitudes"] for record in records], dtype=np.float64
    )
    observed_reduced = np.asarray(
        [record["reduced_amplitudes"] for record in records], dtype=np.float64
    )
    norms = np.asarray(
        [record["representative_norm"] for record in records], dtype=np.float64
    )
    if observed_physical.shape != expected_physical.shape:
        raise ValueError("held-out physical-amplitude shape mismatch")
    if observed_reduced.shape != expected_physical.shape:
        raise ValueError("held-out reduced-amplitude shape mismatch")
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("held-out representative norms must be finite and positive")
    expected_reduced = (
        np.sqrt(float(group_order)) * expected_physical / norms[:, None]
    )

    physical_delta = observed_physical - expected_physical
    reduced_delta = observed_reduced - expected_reduced

    def relative(delta: np.ndarray, reference: np.ndarray) -> float:
        denominator = float(np.linalg.norm(reference))
        if denominator <= 0.0:
            raise ValueError("held-out reference has zero norm")
        return float(np.linalg.norm(delta) / denominator)

    return {
        "representative_ordinals": ordinals,
        "record_count": len(records),
        "maximum_physical_amplitude_absolute_error": float(
            np.max(np.abs(physical_delta))
        ),
        "physical_amplitude_relative_l2_error": relative(
            physical_delta, expected_physical
        ),
        "reduced_amplitude_relative_l2_error": relative(
            reduced_delta, expected_reduced
        ),
    }
