"""Independent finite-size Z2 sectors and ground-state coupling response.

P=product X_i. Sector representatives have bit zero at site 0 (MSB),
and |b;p>=(|b>+p|complement b>)/sqrt(2), p in (-1,+1).
No thermodynamic class is inferred from these finite-size references.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, eigsh

from .basis import all_bits, validate_size
from .hamiltonians import IsingCouplings


def _validate_sector(n: int, parity: int) -> None:
    validate_size(n)
    if n > 14:
        raise ValueError("sector ED restricted to n <= 14")
    if isinstance(parity, (bool, np.bool_)) or not isinstance(parity, (int, np.integer)) or parity not in (-1, 1):
        raise ValueError("parity must be the integer -1 or +1")


@lru_cache(maxsize=16)
def _sector_terms(n: int, parity: int) -> tuple[csr_matrix, ...]:
    half = 1 << (n - 1)
    labels = np.arange(half)
    z = 2 * all_bits(n)[:half] - 1
    zz = diags(-np.sum(z * np.roll(z, -1, axis=1), axis=1), dtype=np.float64).tocsr()
    zz2 = diags(-np.sum(z * np.roll(z, -2, axis=1), axis=1), dtype=np.float64).tocsr()
    masks = 1 << np.arange(n - 1, -1, -1)

    def flips(changes: np.ndarray) -> csr_matrix:
        new = labels[:, None] ^ changes[None, :]
        complemented = new >= half
        rows = np.where(complemented, (1 << n) - 1 - new, new)
        values = -np.where(complemented, parity, 1).astype(np.float64)
        cols = np.broadcast_to(labels[:, None], rows.shape)
        matrix = coo_matrix((values.ravel(), (rows.ravel(), cols.ravel())), shape=(half, half)).tocsr()
        matrix.sum_duplicates()
        return matrix

    return zz, zz2, flips(masks ^ np.roll(masks, -1)), flips(masks)


def parity_hamiltonian(n: int, couplings: IsingCouplings, parity: int) -> csr_matrix:
    """Pauli PBC H[new,old] in the p=±1 sector, even 4<=n<=14."""
    _validate_sector(n, parity)
    terms = _sector_terms(n, int(parity))
    return sum((value * term for value, term in zip(couplings.vector(), terms, strict=True)),
               csr_matrix(terms[0].shape, dtype=np.float64))


def lift_parity_state(state: np.ndarray, n: int, parity: int) -> np.ndarray:
    """Lift sector amplitudes to MSB-first computational labels; preserve norm."""
    _validate_sector(n, parity)
    state = np.asarray(state)
    half = 1 << (n - 1)
    if state.shape != (half,) or not np.isfinite(state).all():
        raise ValueError("state must be a finite sector vector")
    lifted = np.zeros(1 << n, dtype=np.result_type(state, np.float64))
    lifted[:half] = state / np.sqrt(2)
    lifted[-1:half - 1:-1] = parity * state / np.sqrt(2)
    return lifted


@dataclass(frozen=True)
class SectorReference:
    """Finite-size ground state, parity-resolved gaps and total h-QGT."""

    n: int
    energy: float
    same_parity_gap: float
    opposite_parity_gap: float
    chi_total: float
    state: np.ndarray
    state_residual: float
    spectrum_residual: float
    response_relative_residual: float
    response_gauge_error: float


def parity_reference(n: int, couplings: IsingCouplings,
                     linear_tolerance: float = 1e-11) -> SectorReference:
    """Solve parity sectors and the projected ground-state derivative in h.

    (H-E0+|0><0|)|d_h 0> = -(d_h H-<d_h H>)|0>.
    Total chi is ||d_h 0||² after imposing <0|d_h 0>=0.
    The positive-amplitude setting requires h>0 and jx>=0; the odd gap
    is retained separately and is not the response equation's same-parity gap.
    """
    _validate_sector(n, 1)
    if couplings.h <= 0 or couplings.jx < 0:
        raise ValueError("reference requires h>0 and jx>=0")
    if not np.isfinite(linear_tolerance) or not 0 < linear_tolerance < 1:
        raise ValueError("linear_tolerance must be finite and between zero and one")
    even, odd = (parity_hamiltonian(n, couplings, p) for p in (1, -1))
    # Uniform starts would impose extra momentum restrictions and could miss
    # the true first excited level; deterministic generic starts span them.
    initial = np.random.default_rng(n).normal(size=even.shape[0])
    energies, states = eigsh(even, k=2, which="SA", tol=1e-12, v0=initial)
    order = np.argsort(energies); energies, states = energies[order], states[:, order]
    odd_values, odd_states = eigsh(odd, k=1, which="SA", tol=1e-12,
                                  v0=np.random.default_rng(n + 1).normal(size=odd.shape[0]))
    energy = float(energies[0]); psi = states[:, 0]
    psi *= np.sign(psi[np.argmax(np.abs(psi))])
    residuals = [float(np.linalg.norm(even @ states[:, i] - energies[i] * states[:, i])) for i in range(2)]
    residuals.append(float(np.linalg.norm(odd @ odd_states[:, 0] - odd_values[0] * odd_states[:, 0])))
    if max(residuals) > 1e-9 or odd_values[0] < energy - 1e-9:
        raise FloatingPointError("sector eigensolver or even ground-state check failed")
    derivative = _sector_terms(n, 1)[3]
    source = derivative @ psi
    source -= psi * (psi @ source)
    operator = LinearOperator(even.shape,
        matvec=lambda v: even @ v - energy * v + psi * (psi @ v), dtype=np.float64)
    diagonal = even.diagonal() - energy + psi**2
    preconditioner = LinearOperator(even.shape, matvec=lambda v: v / diagonal, dtype=np.float64)
    if np.linalg.norm(source) < 1e-13:
        response = np.zeros_like(psi); relative_residual = 0.
    else:
        response, info = cg(operator, -source, M=preconditioner, rtol=linear_tolerance,
                            atol=0., maxiter=10000)
        response -= psi * (psi @ response)
        relative_residual = float(np.linalg.norm(operator @ response + source) / np.linalg.norm(source))
        if info != 0 or not np.isfinite(relative_residual) or relative_residual > max(1e-8, 100 * linear_tolerance):
            raise FloatingPointError("ground-state response solve did not converge")
    return SectorReference(n, energy, float(energies[1] - energy),
        float(odd_values[0] - energy), float(response @ response),
        lift_parity_state(psi, n, 1), residuals[0], max(residuals),
        relative_residual, float(abs(psi @ response)))
