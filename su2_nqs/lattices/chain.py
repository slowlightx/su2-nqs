"""One-dimensional chain geometry."""

from __future__ import annotations


def chain_bonds(n: int, *, periodic: bool = False) -> tuple[tuple[int, int], ...]:
    if n < 1:
        raise ValueError("N must be positive")
    bonds = [(site, site + 1) for site in range(n - 1)]
    if periodic and n > 2:
        bonds.append((0, n - 1))
    return tuple(bonds)


def dimerized_chain_couplings(
    n: int,
    *,
    coupling: float = 1.0,
    dimerization: float = 0.0,
) -> tuple[float, ...]:
    """Return open-chain couplings ``J_b=J*(1+(-1)^b delta)``.

    Bond zero is the strong bond for positive ``dimerization``.  The helper
    deliberately returns only the ``N-1`` coupling values expected by
    :class:`su2_nqs.operators.heisenberg_chain.HeisenbergChain`; boundary
    conditions remain an explicit choice at the lattice/operator layer.
    """

    if n < 1:
        raise ValueError("N must be positive")
    if abs(float(dimerization)) > 1.0:
        raise ValueError("dimerization must lie in [-1, 1]")
    scale = float(coupling)
    delta = float(dimerization)
    return tuple(scale * (1.0 + (-1.0) ** bond * delta) for bond in range(n - 1))
