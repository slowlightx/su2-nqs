"""Open-boundary Heisenberg connectivity in a fixed-S fusion-path basis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..su2.fusion_path import FusionPath, enumerate_paths, is_valid_path
from ..su2.wigner6j import pair_heisenberg_block
from .pair_spin import pair_spin_energy


def get_conn(
    path: Sequence[int],
    couplings: Sequence[float] | None = None,
) -> tuple[tuple[FusionPath, ...], np.ndarray]:
    """Return paths connected to ``path`` and their summed matrix elements."""

    path = tuple(path)
    if not is_valid_path(path):
        raise ValueError(f"invalid fusion path: {path!r}")
    n = len(path)
    if couplings is None:
        couplings = (1.0,) * max(0, n - 1)
    else:
        couplings = tuple(float(value) for value in couplings)
        if len(couplings) != n - 1:
            raise ValueError("an open N-site chain requires N-1 couplings")

    connected: defaultdict[FusionPath, float] = defaultdict(float)
    if n == 1:
        connected[path] = 0.0
    else:
        connected[path] += couplings[0] * pair_spin_energy(path[1])

    # Bond k joins zero-based sites k and k+1.  Only q after site k changes.
    for bond in range(1, n - 1):
        q_left = path[bond - 1]
        q_mid = path[bond]
        q_total = path[bond + 1]
        block, channels = pair_heisenberg_block(q_left, q_total)
        try:
            column = channels.index(q_mid)
        except ValueError as error:
            raise ValueError(f"path contains an inadmissible local channel: {path!r}") from error
        for row, q_new in enumerate(channels):
            value = couplings[bond] * block[row, column]
            if abs(value) > 1e-15:
                new_path = path[:bond] + (q_new,) + path[bond + 1 :]
                connected[new_path] += float(value)

    items = sorted(
        ((candidate, value) for candidate, value in connected.items() if abs(value) > 1e-14),
        key=lambda item: item[0],
    )
    return tuple(candidate for candidate, _ in items), np.asarray(
        [value for _, value in items], dtype=np.float64
    )


def connected_configs(
    path: Sequence[int],
    couplings: Sequence[float] | None = None,
) -> tuple[tuple[FusionPath, ...], np.ndarray]:
    """Compatibility name for :func:`get_conn` used in the design notes."""

    return get_conn(path, couplings)


@dataclass(frozen=True)
class HeisenbergChain:
    """Open-boundary chain represented directly in one total-spin sector."""

    n: int
    target_q: int = 0
    couplings: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if self.couplings is None:
            object.__setattr__(self, "couplings", (1.0,) * max(0, self.n - 1))
        elif len(self.couplings) != self.n - 1:
            raise ValueError("an open N-site chain requires N-1 couplings")
        else:
            object.__setattr__(self, "couplings", tuple(float(value) for value in self.couplings))

    @property
    def basis(self) -> tuple[FusionPath, ...]:
        return enumerate_paths(self.n, self.target_q)

    def get_conn(self, path: Sequence[int]) -> tuple[tuple[FusionPath, ...], np.ndarray]:
        path = tuple(path)
        if not is_valid_path(path, self.target_q) or len(path) != self.n:
            raise ValueError("path does not belong to this chain sector")
        return get_conn(path, self.couplings)

    def to_dense(self) -> np.ndarray:
        basis = self.basis
        positions = {path: position for position, path in enumerate(basis)}
        matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
        for column, path in enumerate(basis):
            connected, elements = self.get_conn(path)
            for candidate, element in zip(connected, elements):
                matrix[positions[candidate], column] += element
        return matrix
