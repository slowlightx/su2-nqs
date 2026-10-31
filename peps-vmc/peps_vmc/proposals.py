"""Exactly known, full-support IID proposals, independent of PEPS parameters.

A frozen finite mixture of product-singlet covering probabilities is mixed
with uniform fixed-M=0 configurations. An optional H-exchange pushforward
improves support near connections. There is no Markov-chain equilibration or
unknown PEPS normalization in this proposal; importance concentration still
has to be measured and may be poor.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


def _validate_coverings(coverings: np.ndarray, n_sites: int) -> None:
    if (coverings.ndim != 3 or len(coverings) < 1 or coverings.shape[1:] != (n_sites//2, 2)
            or not np.issubdtype(coverings.dtype, np.integer)
            or any(not np.array_equal(np.sort(row.reshape(-1)), np.arange(n_sites)) for row in coverings)):
        raise ValueError("each covering must pair every physical site exactly once")


def torus_coverings(rows: int, columns: int, *, count: int = 48, seed: int = 0) -> np.ndarray:
    """Freeze nearest-neighbor triangular coverings using local rhombus flips.

    Correlation in constructing this finite library does not correlate later
    IID configuration draws: their exactly known law is its finite mixture.
    """
    if rows < 4 or columns < 4 or rows % 2 or columns % 2 or count < 6:
        raise ValueError("even torus lengths>=4 and at least six coverings required")
    n = rows*columns
    directions = ((1, 0), (0, 1), (1, 1))
    def site(r, c): return (r % rows)*columns+(c % columns)
    library = []
    for dr, dc in directions:
        cycles = []; visited = set()
        for start in range(n):
            if start in visited: continue
            cycle = []; r, c = divmod(start, columns)
            while site(r, c) not in visited:
                v = site(r, c); visited.add(v); cycle.append(v); r += dr; c += dc
            if len(cycle) % 2: raise ValueError("covering direction has an odd periodic cycle")
            cycles.append(cycle)
        for shift in (0, 1):
            pairs = [tuple(sorted((cycle[(k+shift) % len(cycle)], cycle[(k+shift+1) % len(cycle)])))
                     for cycle in cycles for k in range(0, len(cycle), 2)]
            library.append(tuple(sorted(pairs)))
    rng = np.random.default_rng(seed)
    while len(library) < count:
        pairs = set(library[int(rng.integers(len(library)))])
        for _ in range(12*n):
            r, c = divmod(int(rng.integers(n)), columns)
            ia, ib = rng.choice(3, 2, replace=False); a, b = directions[ia], directions[ib]
            vertices = (site(r, c), site(r+a[0], c+a[1]),
                        site(r+a[0]+b[0], c+a[1]+b[1]), site(r+b[0], c+b[1]))
            x, y, z, t = vertices
            first = {tuple(sorted((x, y))), tuple(sorted((z, t)))}
            second = {tuple(sorted((x, t))), tuple(sorted((y, z)))}
            if first <= pairs: pairs = (pairs-first) | second
            elif second <= pairs: pairs = (pairs-second) | first
        library.append(tuple(sorted(pairs)))
    result = np.asarray(library[:count], dtype=np.int32)
    _validate_coverings(result, n)
    return result


@dataclass(frozen=True)
class CoveringBridgeProposal:
    """Known normalized density over physical zero-magnetization bit strings."""

    coverings: np.ndarray
    bonds: np.ndarray
    exchange_probabilities: np.ndarray
    epsilon: float = .25

    def __post_init__(self) -> None:
        if (not isinstance(self.coverings, np.ndarray) or not isinstance(self.bonds, np.ndarray)
                or not isinstance(self.exchange_probabilities, np.ndarray)):
            raise TypeError("explicit numeric NumPy arrays required")
        n = self.coverings.shape[1]*2 if self.coverings.ndim == 3 else 0
        if n < 2 or n % 2: raise ValueError("positive even site count required")
        _validate_coverings(self.coverings, n)
        if (self.bonds.ndim != 2 or self.bonds.shape[1] != 2 or len(self.bonds) < 1
                or not np.issubdtype(self.bonds.dtype, np.integer)
                or np.any(self.bonds < 0) or np.any(self.bonds >= n)
                or np.any(self.bonds[:, 0] == self.bonds[:, 1])):
            raise ValueError("valid exchange bonds required")
        p = self.exchange_probabilities
        if (p.shape != (len(self.bonds),) or not np.isfinite(p).all() or np.any(p < 0)
                or not np.isclose(p.sum(), 1., atol=1e-14, rtol=1e-14)):
            raise ValueError("normalized nonnegative exchange probabilities required")
        if not 0 < self.epsilon < 1: raise ValueError("strict uniform support fraction required")
        for name in ("coverings", "bonds", "exchange_probabilities"):
            frozen = np.array(getattr(self, name), copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def n_sites(self) -> int:
        return 2*self.coverings.shape[1]

    def _rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        if (rows.ndim != 2 or rows.shape[1] != self.n_sites
                or not np.isin(rows, (0, 1)).all() or not np.all(rows.sum(1) == self.n_sites//2)):
            raise ValueError("physical binary rows in fixed doubled M=0 required")
        return rows

    def base_density(self, rows: np.ndarray) -> np.ndarray:
        """Exact dimer-mixture plus uniform probability, not a PEPS amplitude."""
        rows = self._rows(rows)
        allowed = np.all(rows[:, self.coverings[:, :, 0]] != rows[:, self.coverings[:, :, 1]], axis=-1)
        return ((1-self.epsilon)*allowed.mean(1)*2.**(-self.n_sites//2)
                + self.epsilon/comb(self.n_sites, self.n_sites//2))

    def density(self, rows: np.ndarray) -> np.ndarray:
        """Exact half-identity/half-exchange pushforward, including null swaps."""
        rows = self._rows(rows); result = []
        for row in rows:
            changed = np.broadcast_to(row, (len(self.bonds), self.n_sites)).copy()
            ids = np.arange(len(self.bonds)); a, b = self.bonds.T
            changed[ids, a], changed[ids, b] = row[b], row[a]
            push = self.base_density(changed) @ self.exchange_probabilities
            result.append(.5*self.base_density(row[None])[0]+.5*push)
        return np.asarray(result)

    def draw(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """IID constructive samples and their exact strictly positive density."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive sample count required")
        rows = np.zeros((count, self.n_sites), dtype=np.int32)
        for row in rows:
            if rng.random() < self.epsilon:
                row[rng.choice(self.n_sites, self.n_sites//2, replace=False)] = 1
            else:
                covering = self.coverings[int(rng.integers(len(self.coverings)))]
                values = rng.integers(2, size=self.n_sites//2, dtype=np.int32)
                row[covering[:, 0]] = values; row[covering[:, 1]] = 1-values
            if rng.random() < .5:
                a, b = self.bonds[int(rng.choice(len(self.bonds), p=self.exchange_probabilities))]
                row[a], row[b] = row[b], row[a]
        return rows, self.density(rows)
