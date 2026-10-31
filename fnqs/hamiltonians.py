"""Pauli Ising super-Hamiltonian and independent finite-system references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import all_bits, validate_size

COUPLING_NAMES = ("j1", "j2", "jx", "h")


@dataclass(frozen=True)
class IsingCouplings:
    """H = -j1 sum ZZ(i,i+1) -j2 sum ZZ(i,i+2) -jx sum XX -h sum X.

    X,Z are Pauli matrices. Each sum includes all n origins, even when a small
    periodic ring repeats an unordered pair (notably the n=4 distance-2 term).
    """

    j1: float = 1.0
    j2: float = 0.0
    jx: float = 0.0
    h: float = 1.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.vector())):
            raise ValueError("couplings must be finite real numbers")

    def vector(self) -> np.ndarray:
        """Return the documented ordering (j1,j2,jx,h)."""
        return np.asarray((self.j1, self.j2, self.jx, self.h), dtype=np.float64)

    def tokens(self) -> tuple[tuple[str, int, float], ...]:
        """Operator/range/coefficient metadata, including zero interactions.

        The first model uses a fixed coupling vector; these tokens document the
        future Hamiltonian encoder contract and do not imply token generalization.
        """
        return (("ZZ", 1, -self.j1), ("ZZ", 2, -self.j2),
                ("XX", 1, -self.jx), ("X", 0, -self.h))


def connectivity(bits: np.ndarray, couplings: IsingCouplings) -> tuple[np.ndarray, np.ndarray]:
    """Return unique new configurations and H[new,old], summing repeats."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be a binary vector")
    bits = bits.astype(np.int32)
    n = len(bits)
    validate_size(n)
    z = 2 * bits - 1
    old = 0
    for bit in bits:
        old = (old << 1) | int(bit)
    entries = {old: -couplings.j1 * float(z @ np.roll(z, -1))
               - couplings.j2 * float(z @ np.roll(z, -2))}
    for site in range(n):
        mask = 1 << (n - 1 - site)
        single = old ^ mask
        pair = single ^ (1 << (n - 1 - ((site + 1) % n)))
        entries[single] = entries.get(single, 0.0) - couplings.h
        entries[pair] = entries.get(pair, 0.0) - couplings.jx
    indices = sorted(k for k, value in entries.items() if value != 0)
    new_bits = np.asarray([[(label >> (n - 1 - site)) & 1 for site in range(n)]
                           for label in indices], dtype=np.int32).reshape(-1,n)
    return new_bits, np.asarray([entries[k] for k in indices], dtype=np.float64)


def dense_hamiltonian(n: int, couplings: IsingCouplings) -> np.ndarray:
    """Build a dense exact reference, limited to n <= 10."""
    validate_size(n)
    if n > 10:
        raise ValueError("dense reference restricted to n <= 10")
    basis = all_bits(n)
    matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
    weights = 1 << np.arange(n - 1, -1, -1)
    for old, bits in enumerate(basis):
        new, values = connectivity(bits, couplings)
        matrix[new @ weights, old] = values
    return matrix


def ground_state(n: int, couplings: IsingCouplings) -> tuple[float, np.ndarray, float]:
    """Return E0, normalized state, residual via optional SciPy sparse ED.

    n <= 14 is an explicit guard against accidental large ED. No reference
    state is silently reused as a neural-network output.
    """
    validate_size(n)
    if n > 14:
        raise ValueError("ED is restricted to n <= 14")
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh

    basis = all_bits(n)
    weights = 1 << np.arange(n - 1, -1, -1)
    rows, cols, values = [], [], []
    for old, bits in enumerate(basis):
        new, coefficients = connectivity(bits, couplings)
        rows.extend(new @ weights)
        cols.extend([old] * len(coefficients))
        values.extend(coefficients)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(basis), len(basis))).tocsr()
    energies, states = eigsh(matrix, k=1, which="SA", tol=1e-12,
                             v0=np.ones(len(basis)) / np.sqrt(len(basis)))
    state = states[:, 0]
    state *= np.sign(state[np.argmax(np.abs(state))])
    residual = float(np.linalg.norm(matrix @ state - energies[0] * state))
    return float(energies[0]), state, residual
