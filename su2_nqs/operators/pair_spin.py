"""Two-spin scalar-product eigenvalues."""

from __future__ import annotations


def pair_spin_energy(pair_q: int) -> float:
    """Eigenvalue of ``S_i.S_j`` for two spin-1/2s with doubled spin q."""

    if pair_q == 0:
        return -0.75
    if pair_q == 2:
        return 0.25
    raise ValueError("two spin-1/2s can fuse only to pair_q=0 or pair_q=2")

