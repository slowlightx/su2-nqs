"""Reference graph Heisenberg Hamiltonians in the computational basis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..benchmarks.exact_diagonalization import heisenberg_sz_matrix


def heisenberg_graph_matrix(
    n: int,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    *,
    n_up: int | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build the small-system graph Hamiltonian used by exact benchmarks."""

    return heisenberg_sz_matrix(n, bonds, n_up=n_up)

