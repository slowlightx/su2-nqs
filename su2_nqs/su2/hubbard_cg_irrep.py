"""Local Hubbard irreps for physical-basis Clebsch--Gordan networks.

The four physical tokens keep charge and doubled-spin quantum numbers
explicit.  Magnetic projections are structural inputs to numerical CG
coefficients; learned reduced tensors must never depend on ``m2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


EMPTY = 0
UP = 1
DOWN = 2
DOUBLON = 3
HUBBARD_TOKENS = (EMPTY, UP, DOWN, DOUBLON)


@dataclass(frozen=True)
class HubbardLocalIrrep:
    """Charge and spin labels of one physical onsite basis state."""

    token: int
    charge: int
    q: int
    m2: int
    reduced_label: int


_LOCAL_IRREPS = (
    HubbardLocalIrrep(EMPTY, 0, 0, 0, 0),
    HubbardLocalIrrep(UP, 1, 1, 1, 1),
    HubbardLocalIrrep(DOWN, 1, 1, -1, 1),
    HubbardLocalIrrep(DOUBLON, 2, 0, 0, 2),
)


def validate_hubbard_token(token: int) -> int:
    """Return a checked physical token in ``{0,1,2,3}``."""

    if isinstance(token, bool) or not isinstance(token, Integral):
        raise TypeError(f"token must be an integer, got {token!r}")
    token = int(token)
    if token not in HUBBARD_TOKENS:
        raise ValueError(f"token must be one of {HUBBARD_TOKENS}, got {token}")
    return token


def hubbard_local_irrep(token: int) -> HubbardLocalIrrep:
    """Map one physical token to its fixed onsite ``(charge,q,m2)`` labels."""

    return _LOCAL_IRREPS[validate_hubbard_token(token)]


def token_quantum_number_tables() -> tuple[tuple[int, ...], ...]:
    """Return immutable token-indexed charge, q, m2 and reduced-label tables."""

    return tuple(
        tuple(getattr(irrep, field) for irrep in _LOCAL_IRREPS)
        for field in ("charge", "q", "m2", "reduced_label")
    )


__all__ = [
    "DOUBLON",
    "DOWN",
    "EMPTY",
    "HUBBARD_TOKENS",
    "HubbardLocalIrrep",
    "UP",
    "hubbard_local_irrep",
    "token_quantum_number_tables",
    "validate_hubbard_token",
]
