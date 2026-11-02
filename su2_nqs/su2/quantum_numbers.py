"""Utilities for doubled-integer SU(2) quantum numbers.

Every public function in this module accepts ``q = 2j`` and ``m = 2m_j``.
This keeps parity checks exact and prevents floating-point half-integers from
leaking into Hilbert-space labels.
"""

from __future__ import annotations

from numbers import Integral


def validate_q(q: int, *, name: str = "q") -> int:
    """Return a validated non-negative doubled angular momentum."""

    if isinstance(q, bool) or not isinstance(q, Integral):
        raise TypeError(f"{name} must be an integer doubled spin, got {q!r}")
    if q < 0:
        raise ValueError(f"{name} must be non-negative, got {q}")
    return int(q)


def validate_m(q: int, m: int, *, name: str = "m") -> int:
    """Validate a doubled magnetic quantum number for doubled spin ``q``."""

    validate_q(q)
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError(f"{name} must be an integer doubled projection, got {m!r}")
    if abs(m) > q or (q - m) % 2:
        raise ValueError(f"{name}={m} is not a projection of q={q}")
    return int(m)


def magnetic_numbers(q: int) -> tuple[int, ...]:
    """Return ``-q, -q+2, ..., q``."""

    validate_q(q)
    return tuple(range(-q, q + 1, 2))


def is_admissible_triangle(q1: int, q2: int, q3: int) -> bool:
    """Whether three doubled spins satisfy the SU(2) triangle rules."""

    if any(isinstance(q, bool) or not isinstance(q, Integral) or q < 0 for q in (q1, q2, q3)):
        return False
    return abs(q1 - q2) <= q3 <= q1 + q2 and (q1 + q2 + q3) % 2 == 0


def coupling_range(q1: int, q2: int) -> tuple[int, ...]:
    """All doubled spins in the tensor product ``q1 (x) q2``."""

    validate_q(q1, name="q1")
    validate_q(q2, name="q2")
    return tuple(range(abs(q1 - q2), q1 + q2 + 1, 2))
