"""Exact small-support SU(2) projection and physical-cut diagnostics.

No Hamiltonian eigensolver is used. Angular-momentum labels ``q`` are doubled.
The dense projector is deliberately restricted to validation-scale N <= 16.
"""
from __future__ import annotations

import numpy as np


def singlet_projector_matrix(spin_squared: np.ndarray, *, n_sites: int) -> np.ndarray:
    """Spectral polynomial P(q=0) for even N in an invariant m2=0 basis.

    Apply factors from largest q downwards to avoid amplifying high-spin
    roundoff. No eigenvectors, fitting, or Hamiltonian information enter P.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 2 or n_sites > 16 or n_sites % 2:
        raise ValueError("validation projector requires even integer 2 <= N <= 16")
    s2 = np.asarray(spin_squared, dtype=np.complex128)
    if s2.ndim != 2 or s2.shape[0] != s2.shape[1] or not np.isfinite(s2).all():
        raise ValueError("S2 must be a finite square matrix")
    if not np.allclose(s2, s2.conj().T, atol=1e-11, rtol=0):
        raise ValueError("S2 must be Hermitian")
    projector = np.eye(len(s2), dtype=np.complex128)
    for q in range(n_sites, 0, -2):
        projector = projector - (s2 @ projector)/(q*(q+2)/4)
    if np.linalg.norm(s2@projector) > 1e-8 or np.linalg.norm(projector@projector-projector) > 1e-9:
        raise ValueError("S2 does not have the required integer-spin spectrum")
    return projector


def singlet_isometry(spin_squared: np.ndarray, *, n_sites: int) -> np.ndarray:
    """Orthonormal columns spanning the polynomial projector's range.

    Factored application Q (Q† psi) avoids normalizing tiny out-of-sector
    roundoff when a raw ViT starts close to a high-spin uniform state. QR is
    performed on P0 only, never on the Hamiltonian or a target wavefunction.
    """
    from scipy.linalg import qr
    projector = singlet_projector_matrix(spin_squared, n_sites=n_sites)
    rank = int(round(np.trace(projector).real))
    if rank < 1:
        raise ValueError("selected spatial sector contains no singlet")
    columns = qr(projector, mode="economic", pivoting=True)[0][:, :rank]
    # One fixed spectral-refinement pass removes range-extraction roundoff,
    # before the final orthogonalization. P0^2=P0 analytically.
    for q in range(n_sites, 0, -2):
        columns = columns - np.asarray(spin_squared)@columns/(q*(q+2)/4)
    columns = qr(columns, mode="economic")[0]
    if (np.max(abs(columns@columns.conj().T-projector)) > 1e-10
            or np.max(abs(np.asarray(spin_squared)@columns)) > 1e-10):
        raise ValueError("singlet range factorization failed")
    return columns


def physical_cut_metrics(state: np.ndarray, basis: np.ndarray, sites: tuple[int, ...]) -> dict:
    """Full physical reduced density matrix, valid also for nonsinglet states.

    Report local q-sector probabilities, not a singlet-only entropy formula.
    The test cuts are four spin-1/2 sites; no global fusion partition is used.
    """
    bits = np.asarray(basis, dtype=int)
    psi = np.asarray(state, dtype=np.complex128)
    if len(sites) != 4 or len(set(sites)) != 4 or any(i < 0 or i >= bits.shape[1] for i in sites):
        raise ValueError("select four distinct physical sites")
    rest = tuple(i for i in range(bits.shape[1]) if i not in sites)
    row = bits[:, sites] @ (1 << np.arange(3, -1, -1))
    col = bits[:, rest] @ (1 << np.arange(len(rest)-1, -1, -1))
    amplitude = np.zeros((16, 2**len(rest)), dtype=complex)
    amplitude[row, col] = psi/np.linalg.norm(psi)
    rho = amplitude @ amplitude.conj().T
    # Independent physical Pauli-matrix construction of block total spin.
    pauli = (np.array([[0,1],[1,0]]), np.array([[0,1j],[-1j,0]]), np.diag([-1,1]))
    totals = []
    for sigma in pauli:
        total = np.zeros((16,16), dtype=complex)
        for i in range(4):
            matrix = np.ones((1,1))
            for j in range(4):
                matrix = np.kron(matrix, sigma/2 if i == j else np.eye(2))
            total += matrix
        totals.append(total)
    s2 = sum(s@s for s in totals)
    probabilities = {}
    for q in (0,2,4):
        p = np.eye(16)
        for other in (0,2,4):
            if other != q:
                p = p@(s2-other*(other+2)/4*np.eye(16))/(q*(q+2)/4-other*(other+2)/4)
        probabilities[str(q)] = float(np.trace(rho@p).real)
    eigenvalues = np.linalg.eigvalsh(rho)
    positive = eigenvalues[eigenvalues > 1e-15]
    return dict(sites=list(sites), q_probabilities=probabilities,
                entropy_nats=float(-np.sum(positive*np.log(positive))),
                su2_commutator_max=float(max(np.linalg.norm(rho@s-s@rho) for s in totals)))
