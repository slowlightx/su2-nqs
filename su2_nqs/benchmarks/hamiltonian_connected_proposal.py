"""Hamiltonian-connected proposals on a finite fusion-path graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sparse():
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("connected proposals require scipy") from error
    return sparse


@dataclass(frozen=True)
class HamiltonianConnectedKernel:
    """Column-stochastic one-hop kernel on nonzero off-diagonal H edges."""

    transition: object
    degrees: np.ndarray

    @property
    def state_count(self) -> int:
        return int(len(self.degrees))

    def distribution(self, probabilities: np.ndarray) -> np.ndarray:
        """Push a normalized source distribution through one H-graph hop."""

        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (self.state_count,):
            raise ValueError("probabilities have the wrong state count")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("probabilities must be finite and nonnegative")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("probabilities must have positive total")
        result = np.asarray(self.transition @ (values / total)).reshape(-1)
        return result / np.sum(result)

    def sample(
        self, source_indices: np.ndarray, generator: np.random.Generator
    ) -> np.ndarray:
        """Draw one uniformly selected connected state per source index."""

        sources = np.asarray(source_indices)
        if sources.ndim != 1 or not np.issubdtype(sources.dtype, np.integer):
            raise TypeError("source_indices must be a one-dimensional integer array")
        if np.any(sources < 0) or np.any(sources >= self.state_count):
            raise ValueError("source index lies outside the kernel")
        matrix = self.transition.tocsc()
        draws = np.empty(len(sources), dtype=np.int32)
        for position, source in enumerate(sources):
            start, stop = matrix.indptr[source : source + 2]
            candidates = matrix.indices[start:stop]
            probabilities = matrix.data[start:stop]
            draws[position] = int(generator.choice(candidates, p=probabilities))
        return draws


def build_hamiltonian_connected_kernel(
    hamiltonian: object,
    *,
    absolute_tolerance: float = 1.0e-14,
) -> HamiltonianConnectedKernel:
    """Build a uniform-neighbor transition from a Hermitian Hamiltonian.

    Isolated vertices receive a self-loop.  Edge magnitudes are deliberately
    discarded: the proposal resolves reachable histories without encoding a
    variational or teacher-dependent coefficient.
    """

    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and nonnegative")
    sparse = _sparse()
    matrix = sparse.csc_matrix(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("hamiltonian must be a nonempty square matrix")
    hermiticity = matrix - matrix.conj().T
    if hermiticity.nnz and np.max(np.abs(hermiticity.data)) > 1.0e-11:
        raise ValueError("hamiltonian must be Hermitian")
    matrix = matrix.copy()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    if matrix.nnz:
        keep = np.abs(matrix.data) > absolute_tolerance
        matrix.data = keep.astype(np.float64)
        matrix.eliminate_zeros()
    degrees = np.asarray(matrix.sum(axis=0)).reshape(-1).astype(np.int32)
    isolated = np.flatnonzero(degrees == 0)
    if len(isolated):
        matrix = matrix.tolil()
        matrix[isolated, isolated] = 1.0
        matrix = matrix.tocsc()
        degrees[isolated] = 1
    transition = matrix @ sparse.diags(1.0 / degrees)
    normalization_error = float(
        np.max(np.abs(np.asarray(transition.sum(axis=0)).reshape(-1) - 1.0))
    )
    if normalization_error > 2.0e-14:
        raise RuntimeError("connected transition is not column stochastic")
    return HamiltonianConnectedKernel(
        transition=transition.tocsr(), degrees=degrees
    )


def connected_mixture_probabilities(
    base_probabilities: np.ndarray,
    kernel: HamiltonianConnectedKernel,
    *,
    uniform_fraction: float,
    connected_fraction: float,
) -> np.ndarray:
    """Return ``Born + uniform + one-hop`` proposal probabilities."""

    if (
        not np.isfinite(uniform_fraction)
        or not np.isfinite(connected_fraction)
        or uniform_fraction < 0.0
        or connected_fraction < 0.0
        or uniform_fraction + connected_fraction >= 1.0
    ):
        raise ValueError("proposal fractions must be nonnegative and sum below one")
    base = np.asarray(base_probabilities, dtype=np.float64)
    if base.shape != (kernel.state_count,):
        raise ValueError("base probabilities have the wrong state count")
    if np.any(~np.isfinite(base)) or np.any(base < 0.0) or np.sum(base) <= 0.0:
        raise ValueError("base probabilities must be finite and nonnegative")
    base = base / np.sum(base)
    born_fraction = 1.0 - uniform_fraction - connected_fraction
    proposal = (
        born_fraction * base
        + uniform_fraction / kernel.state_count
        + connected_fraction * kernel.distribution(base)
    )
    return proposal / np.sum(proposal)


__all__ = [
    "HamiltonianConnectedKernel",
    "build_hamiltonian_connected_kernel",
    "connected_mixture_probabilities",
]
