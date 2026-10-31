"""Left-associated fusion paths for spin-1/2 degrees of freedom."""

from __future__ import annotations

from math import comb
from numbers import Integral
from typing import Iterable, Sequence

from .quantum_numbers import validate_q

FusionPath = tuple[int, ...]
FUSION_STEPS = (-1, 1)


def _validate_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("N must be an integer")
    if n < 1:
        raise ValueError("N must be positive")
    return int(n)


def _target_is_reachable(q: int, remaining: int, target_q: int) -> bool:
    return q >= 0 and abs(q - target_q) <= remaining and (remaining - abs(q - target_q)) % 2 == 0


def is_valid_path(path: Sequence[int], target_q: int | None = None) -> bool:
    """Check a spin-1/2 left-fusion path.

    ``path[k]`` is the doubled total spin after fusing the first ``k+1``
    physical spins.  Consequently every non-empty path starts at ``1`` and
    neighboring labels differ by exactly one.
    """

    try:
        raw_values = tuple(path)
    except TypeError:
        return False
    if any(isinstance(q, bool) or not isinstance(q, Integral) for q in raw_values):
        return False
    values = tuple(int(q) for q in raw_values)
    if not values or values[0] != 1:
        return False
    if any(q < 0 for q in values):
        return False
    if any(abs(right - left) != 1 for left, right in zip(values, values[1:])):
        return False
    if target_q is not None:
        if isinstance(target_q, bool) or not isinstance(target_q, Integral) or target_q < 0:
            return False
        if values[-1] != target_q:
            return False
    return True


def allowed_next_steps(prefix: Sequence[int], n: int, target_q: int) -> tuple[int, ...]:
    """Return exactly valid next fusion increments.

    The physical first spin is fixed.  For an empty prefix this function
    returns ``(1,)`` to initialize the path; all later returned values are the
    autoregressive increments ``-1`` or ``+1``.  A step is retained only if
    the target remains reachable in the number of spins left.
    """

    _validate_n(n)
    validate_q(target_q, name="target_q")
    raw_prefix = tuple(prefix)
    if raw_prefix and not is_valid_path(raw_prefix):
        return ()
    prefix = tuple(int(value) for value in raw_prefix)
    if len(prefix) >= n:
        return ()
    if not prefix:
        remaining = n - 1
        return (1,) if _target_is_reachable(1, remaining, target_q) else ()
    if not is_valid_path(prefix):
        return ()

    remaining = n - len(prefix) - 1
    return tuple(
        step
        for step in FUSION_STEPS
        if _target_is_reachable(prefix[-1] + step, remaining, target_q)
    )


def next_step_mask(prefix: Sequence[int], n: int, target_q: int) -> tuple[bool, bool]:
    """Boolean mask in the canonical action order ``(-1, +1)``."""

    allowed = allowed_next_steps(prefix, n, target_q)
    return (-1 in allowed, 1 in allowed)


def enumerate_paths(n: int, target_q: int) -> tuple[FusionPath, ...]:
    """Enumerate every valid path ending at ``target_q``."""

    _validate_n(n)
    validate_q(target_q, name="target_q")
    if not allowed_next_steps((), n, target_q):
        return ()

    paths: list[FusionPath] = []

    def visit(prefix: FusionPath) -> None:
        if len(prefix) == n:
            if prefix[-1] == target_q:
                paths.append(prefix)
            return
        for step in allowed_next_steps(prefix, n, target_q):
            visit(prefix + (prefix[-1] + step,))

    visit((1,))
    return tuple(paths)


def count_paths(n: int, target_q: int) -> int:
    """Exact multiplicity of total doubled spin ``target_q``.

    For ``S = target_q / 2`` this is

    ``C(N, N/2-S) - C(N, N/2-S-1)``.
    """

    _validate_n(n)
    validate_q(target_q, name="target_q")
    if target_q > n or (n - target_q) % 2:
        return 0
    k = (n - target_q) // 2
    lower = comb(n, k - 1) if k >= 1 else 0
    return comb(n, k) - lower


def paths_from_steps(steps: Iterable[int]) -> FusionPath:
    """Convert ``N-1`` increments to a path with implicit first label one."""

    steps = tuple(steps)
    path = [1]
    for step in steps:
        if step not in FUSION_STEPS:
            raise ValueError(f"fusion steps must be +/-1, got {step}")
        path.append(path[-1] + step)
    result = tuple(path)
    if not is_valid_path(result):
        raise ValueError(f"steps do not form a valid path: {steps!r}")
    return result


def steps_from_path(path: Sequence[int]) -> tuple[int, ...]:
    """Return the ``+/-1`` autoregressive tokens of a valid path."""

    if not is_valid_path(path):
        raise ValueError(f"invalid fusion path: {path!r}")
    return tuple(right - left for left, right in zip(path, path[1:]))
