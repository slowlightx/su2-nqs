"""Open Heisenberg chain connectivity in a fixed-magnetization bit basis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..su2.sz_basis import enumerate_sz_configs

SpinConfiguration = tuple[int, ...]
GraphBond = tuple[int, int, float]


def get_conn_sz(
    configuration: Sequence[int],
    couplings: Sequence[float] | None = None,
) -> tuple[tuple[SpinConfiguration, ...], np.ndarray]:
    """Return bit configurations connected by an OBC Heisenberg chain."""

    configuration = tuple(configuration)
    if not configuration or any(bit not in (0, 1) for bit in configuration):
        raise ValueError("configuration must be a non-empty binary sequence")
    n = len(configuration)
    if couplings is None:
        couplings = (1.0,) * (n - 1)
    elif len(couplings) != n - 1:
        raise ValueError("an open N-site chain requires N-1 couplings")
    connected: defaultdict[SpinConfiguration, float] = defaultdict(float)
    for bond, coupling in enumerate(couplings):
        left, right = configuration[bond], configuration[bond + 1]
        connected[configuration] += float(coupling) * (0.25 if left == right else -0.25)
        if left != right:
            flipped = list(configuration)
            flipped[bond], flipped[bond + 1] = right, left
            connected[tuple(flipped)] += 0.5 * float(coupling)
    items = sorted(connected.items(), key=lambda item: item[0])
    return tuple(item[0] for item in items), np.asarray([item[1] for item in items])


@dataclass(frozen=True)
class SzHeisenbergChain:
    n: int
    n_up: int | None = None
    couplings: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if self.n_up is None:
            object.__setattr__(self, "n_up", self.n // 2)
        if not 0 <= self.n_up <= self.n:
            raise ValueError("n_up must lie between zero and N")
        if self.couplings is None:
            object.__setattr__(self, "couplings", (1.0,) * (self.n - 1))
        elif len(self.couplings) != self.n - 1:
            raise ValueError("an open N-site chain requires N-1 couplings")

    @property
    def basis(self) -> tuple[SpinConfiguration, ...]:
        return enumerate_sz_configs(self.n, self.n_up)

    def get_conn(
        self, configuration: Sequence[int]
    ) -> tuple[tuple[SpinConfiguration, ...], np.ndarray]:
        configuration = tuple(configuration)
        if len(configuration) != self.n or sum(configuration) != self.n_up:
            raise ValueError("configuration does not belong to this Sz sector")
        return get_conn_sz(configuration, self.couplings)

    def to_dense(self) -> np.ndarray:
        basis = self.basis
        positions = {configuration: position for position, configuration in enumerate(basis)}
        matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
        for column, configuration in enumerate(basis):
            connected, elements = self.get_conn(configuration)
            for candidate, element in zip(connected, elements):
                matrix[positions[candidate], column] += element
        return matrix


def get_conn_sz_graph(
    configuration: Sequence[int],
    bonds: Sequence[GraphBond],
) -> tuple[tuple[SpinConfiguration, ...], np.ndarray]:
    """Return fixed-magnetization connectivity for an arbitrary spin graph."""

    configuration = tuple(configuration)
    if not configuration or any(bit not in (0, 1) for bit in configuration):
        raise ValueError("configuration must be a non-empty binary sequence")
    n = len(configuration)
    connected: defaultdict[SpinConfiguration, float] = defaultdict(float)
    for bond in bonds:
        if len(bond) != 3:
            raise ValueError("graph bonds must contain two sites and a coupling")
        first, second, coupling = bond
        if first == second or not 0 <= first < n or not 0 <= second < n:
            raise ValueError(f"invalid graph bond: {bond!r}")
        left, right = configuration[first], configuration[second]
        connected[configuration] += float(coupling) * (
            0.25 if left == right else -0.25
        )
        if left != right:
            flipped = list(configuration)
            flipped[first], flipped[second] = right, left
            connected[tuple(flipped)] += 0.5 * float(coupling)
    items = sorted(
        (
            (candidate, value)
            for candidate, value in connected.items()
            if abs(value) > 1.0e-14
        ),
        key=lambda item: item[0],
    )
    return tuple(item[0] for item in items), np.asarray(
        [item[1] for item in items], dtype=np.float64
    )


@dataclass(frozen=True)
class SzHeisenbergGraph:
    """Arbitrary graph Heisenberg connectivity in a fixed-Sz bit basis."""

    n: int
    bonds: tuple[GraphBond, ...]
    n_up: int | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("N must be positive")
        if self.n_up is None:
            object.__setattr__(self, "n_up", self.n // 2)
        if not 0 <= self.n_up <= self.n:
            raise ValueError("n_up must lie between zero and N")
        normalized = []
        for bond in self.bonds:
            if len(bond) != 3:
                raise ValueError("graph bonds must contain two sites and a coupling")
            first, second, coupling = bond
            if first == second or not 0 <= first < self.n or not 0 <= second < self.n:
                raise ValueError(f"invalid graph bond: {bond!r}")
            normalized.append((int(first), int(second), float(coupling)))
        object.__setattr__(self, "bonds", tuple(normalized))

    @property
    def basis(self) -> tuple[SpinConfiguration, ...]:
        return enumerate_sz_configs(self.n, self.n_up)

    def get_conn(
        self, configuration: Sequence[int]
    ) -> tuple[tuple[SpinConfiguration, ...], np.ndarray]:
        """Return combined nonzero entries of one Hamiltonian column."""

        configuration = tuple(configuration)
        if len(configuration) != self.n or sum(configuration) != self.n_up:
            raise ValueError("configuration does not belong to this Sz sector")
        return get_conn_sz_graph(configuration, self.bonds)

    def to_dense(self) -> np.ndarray:
        """Return the graph Hamiltonian in deterministic bit-string order."""

        basis = self.basis
        positions = {
            configuration: position
            for position, configuration in enumerate(basis)
        }
        matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
        for column, configuration in enumerate(basis):
            connected, elements = self.get_conn(configuration)
            for candidate, element in zip(connected, elements):
                matrix[positions[candidate], column] += element
        return matrix
