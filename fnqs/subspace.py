"""Small-system fidelity-selected snapshot control inspired by NFS.

This is an independent diagnostic, not a neural result or a Kibble-Zurek
simulation. The fixed overlap matrix is whitened with an explicit rank cutoff.
"""

from __future__ import annotations

import numpy as np

from .observables import normalized_state


def select_snapshots(states: np.ndarray, infidelity: float = 0.05) -> list[int]:
    """Greedily select ordered path snapshots, preserving both endpoints."""
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 2 or len(states) < 2 or not 0 < infidelity < 1:
        raise ValueError("need >=2 ordered snapshots and 0<threshold<1")
    states = np.stack([normalized_state(s) for s in states])
    selected = [0]
    for index in range(1, len(states)):
        if 1 - abs(np.vdot(states[selected[-1]], states[index])) ** 2 >= infidelity:
            selected.append(index)
    if selected[-1] != len(states) - 1:
        selected.append(len(states) - 1)
    return selected


def project_subspace(states: np.ndarray, operator: np.ndarray, cutoff: float = 1e-10) -> dict:
    """Project reusable operators and solve a rank-filtered generalized Ritz problem."""
    states, operator = np.asarray(states, dtype=np.complex128), np.asarray(operator, dtype=np.complex128)
    if states.ndim != 2 or len(states) < 1 or operator.shape != (states.shape[1], states.shape[1]) or not 0 < cutoff < 1:
        raise ValueError("incompatible states/operator or invalid cutoff")
    states = np.stack([normalized_state(s) for s in states])
    basis = states.T
    overlap = basis.conj().T @ basis
    projected = basis.conj().T @ operator @ basis
    eigenvalues, vectors = np.linalg.eigh(overlap)
    keep = eigenvalues > cutoff * eigenvalues[-1]
    whitening = vectors[:, keep] / np.sqrt(eigenvalues[keep])
    reduced = whitening.conj().T @ projected @ whitening
    energies = np.linalg.eigvalsh(reduced)
    return {"overlap": overlap, "operator": projected, "whitening": whitening,
            "rank": int(keep.sum()), "energies": energies}
