"""Fixed-magnetization computational basis helpers."""

from __future__ import annotations

from itertools import combinations


def enumerate_sz_configs(n: int, n_up: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate bit strings with exactly ``n_up`` up spins."""

    if n < 1:
        raise ValueError("N must be positive")
    if not 0 <= n_up <= n:
        raise ValueError("n_up must lie between zero and N")
    answers = []
    for positions in combinations(range(n), n_up):
        bits = [0] * n
        for position in positions:
            bits[position] = 1
        answers.append(tuple(bits))
    return tuple(answers)

