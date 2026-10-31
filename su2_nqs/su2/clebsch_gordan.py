"""Clebsch-Gordan coefficients in the Condon-Shortley convention.

The numerical implementation is an exact finite factorial sum over doubled
integer labels.  It has no SymPy dependency and is safe to use while building
offline coefficient tables.  Neural-network/JAX runtime code consumes plain
arrays and never invokes a symbolic algebra system.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Iterable

from .quantum_numbers import is_admissible_triangle, validate_m, validate_q


def _fact(n: int) -> int:
    return math.factorial(n) if n >= 0 else 0


@lru_cache(maxsize=None)
def clebsch_gordan(q1: int, m1: int, q2: int, m2: int, q: int, m: int) -> float:
    """Return ``<q1,m1; q2,m2 | q,m>`` for doubled integer labels."""

    validate_q(q1, name="q1")
    validate_q(q2, name="q2")
    validate_q(q, name="q")
    validate_m(q1, m1, name="m1")
    validate_m(q2, m2, name="m2")
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError(f"m must be an integer doubled projection, got {m!r}")
    if abs(m) > q or (q - m) % 2:
        return 0.0
    if m1 + m2 != m or not is_admissible_triangle(q1, q2, q):
        return 0.0

    a = (q1 + q2 - q) // 2
    b = (q1 - m1) // 2
    c = (q2 + m2) // 2
    d = (q - q2 + m1) // 2
    e = (q - q1 - m2) // 2
    k_min = max(0, -d, -e)
    k_max = min(a, b, c)
    if k_min > k_max:
        return 0.0

    triangle_prefactor = (
        (q + 1)
        * _fact((q + q1 - q2) // 2)
        * _fact((q - q1 + q2) // 2)
        * _fact((q1 + q2 - q) // 2)
        / _fact((q1 + q2 + q) // 2 + 1)
    )
    magnetic_prefactor = (
        _fact((q + m) // 2)
        * _fact((q - m) // 2)
        * _fact((q1 - m1) // 2)
        * _fact((q1 + m1) // 2)
        * _fact((q2 - m2) // 2)
        * _fact((q2 + m2) // 2)
    )
    series = 0.0
    for k in range(k_min, k_max + 1):
        denominator = (
            _fact(k)
            * _fact(a - k)
            * _fact(b - k)
            * _fact(c - k)
            * _fact(d + k)
            * _fact(e + k)
        )
        series += (-1.0 if k % 2 else 1.0) / denominator
    return math.sqrt(triangle_prefactor * magnetic_prefactor) * series


def spin_half_cg(q_left: int, m_left: int, m_phys: int, q_total: int) -> float:
    """Specialized convenience wrapper for coupling one spin-1/2."""

    return clebsch_gordan(
        q_left,
        m_left,
        1,
        m_phys,
        q_total,
        m_left + m_phys,
    )


def precompute_cg_table(
    couplings: Iterable[tuple[int, int, int]],
) -> dict[str, float]:
    """Build a JSON-serializable coefficient table offline."""

    table: dict[str, float] = {}
    for q1, q2, q in couplings:
        if not is_admissible_triangle(q1, q2, q):
            raise ValueError(f"inadmissible coupling ({q1}, {q2}) -> {q}")
        for m1 in range(-q1, q1 + 1, 2):
            for m2 in range(-q2, q2 + 1, 2):
                m = m1 + m2
                if abs(m) <= q and (q - m) % 2 == 0:
                    key = f"{q1},{m1},{q2},{m2},{q},{m}"
                    table[key] = clebsch_gordan(q1, m1, q2, m2, q, m)
    return table


def save_cg_table(path: str | Path, table: dict[str, float]) -> None:
    """Write a deterministic offline CG table."""

    Path(path).write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")


def load_cg_table(path: str | Path) -> dict[tuple[int, ...], float]:
    """Load an offline table using tuple keys convenient for runtime lookup."""

    raw = json.loads(Path(path).read_text())
    return {tuple(int(value) for value in key.split(",")): float(coef) for key, coef in raw.items()}
