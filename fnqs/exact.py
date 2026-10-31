"""Finite-PBC Jordan-Wigner TFIM oracle, with no exponentially large arrays.

Even n, j1>0, h>0: the physical ground state is in the even parity sector with
antiperiodic fermion momenta. The degenerate h=0 endpoint is deliberately
excluded. Energies and susceptibility are totals, not per-site quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import validate_size


@dataclass(frozen=True)
class TFIMReference:
    """Free-fermion reference for -j1 sum ZZ -h sum X, Pauli operators."""

    n: int
    h: float
    j1: float = 1.0

    def __post_init__(self) -> None:
        validate_size(self.n)
        if not np.isfinite(self.h) or not np.isfinite(self.j1) or min(self.h, self.j1) <= 0:
            raise ValueError("reference requires finite j1>0 and h>0")

    @property
    def momenta(self) -> np.ndarray:
        return (2 * np.arange(self.n) + 1) * np.pi / self.n

    @property
    def dispersion(self) -> np.ndarray:
        k = self.momenta
        return np.hypot(self.h - self.j1 * np.cos(k), self.j1 * np.sin(k))

    def energy(self) -> float:
        """Return the total ground-state energy (quasiparticle energy is 2e_k)."""
        return -float(self.dispersion.sum())

    def fidelity_susceptibility(self) -> float:
        """Total chi_hh: F^2(h,h+dh) = 1 - chi_hh dh^2 + O(dh^3)."""
        k = self.momenta[:self.n // 2]
        e2 = self.dispersion[:self.n // 2] ** 2
        return float(0.25 * np.sum((self.j1 * np.sin(k) / e2) ** 2))

    def contraction(self, distances: np.ndarray) -> np.ndarray:
        """Real Majorana contraction G_d; G_1=1 at the ordered endpoint."""
        d = np.asarray(distances)[..., None]
        k, e = self.momenta, self.dispersion
        return np.mean(((self.j1 * np.cos(k) - self.h) * np.cos(d * k)
                        + self.j1 * np.sin(k) * np.sin(d * k)) / e, axis=-1)

    def correlation(self, r: int) -> float:
        """Return <Z_i Z_{i+r}> from the finite-chain Toeplitz determinant."""
        if not isinstance(r, (int, np.integer)) or not 0 <= r < self.n:
            raise ValueError("r must be an integer in [0,n)")
        if r == 0:
            return 1.0
        r = min(r, self.n - r)
        indices = np.arange(r)
        from scipy.linalg import det
        return float(det(self.contraction(indices[None, :] - indices[:, None] + 1)))

    def entropy(self, ell: int, order: float = 2.0) -> float:
        """Interval Renyi entropy; order=1 selects von Neumann entropy.

        PBC CFT coefficient is c/6*(1+1/order); S2 therefore has slope c/4,
        while von Neumann has slope c/3. Logs are natural.
        """
        if not isinstance(ell, (int, np.integer)) or not 1 <= ell < self.n:
            raise ValueError("ell must be an integer in [1,n)")
        if not np.isfinite(order) or order <= 0:
            raise ValueError("order must be positive and finite")
        indices = np.arange(ell)
        block = self.contraction(indices[None, :] - indices[:, None])
        singular = np.clip(np.linalg.svd(block, compute_uv=False), 0, 1)
        p = (1 + singular) / 2
        if order == 1:
            from scipy.special import xlogy
            return -float(np.sum(xlogy(p, p) + xlogy(1 - p, 1 - p)))
        return float(np.sum(np.log(p ** order + (1 - p) ** order)) / (1 - order))
