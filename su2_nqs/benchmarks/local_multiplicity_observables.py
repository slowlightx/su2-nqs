"""Basis-invariant local readouts for exact-small multiplicity carriers.

The routines here operate on a normalized physical wavefunction stored on an
arbitrary binary basis (usually the fixed-``M=0`` basis).  They first form the
physical reduced density matrix of a registered disjoint cell and only then
change to the exact local Schur basis.  Summing over the magnetic partners of
a fixed doubled spin ``q`` leaves a density matrix on the multiplicity space.

This order matters: raw neural features and individual multiplicity
coordinates are gauge dependent, whereas the block weight and eigenvalues of
the multiplicity density matrix are invariant under a unitary change of local
multiplicity basis.  In a point-group-fixed basis, such as square
``(A1,B1)``, the diagonal occupations and coherence are additionally useful
covariant physical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from ..su2.local_schur import (
    CellKind,
    cell_schur_spec,
    cell_schur_tensor,
)
from ..su2.quantum_numbers import magnetic_numbers


@dataclass(frozen=True)
class MultiplicityBlockDensity:
    """One magnetic-traced local ``q`` block.

    ``matrix`` is unnormalized: its trace is the probability that the cell
    has total doubled spin ``q``.  ``conditional_matrix`` has unit trace when
    that probability is nonzero.  ``eigenvalues`` are sorted descending and
    are therefore invariant under a multiplicity-gauge rotation.
    """

    q: int
    block_weight: float
    matrix: np.ndarray
    conditional_matrix: np.ndarray
    eigenvalues: np.ndarray
    purity: float
    entropy: float


@dataclass(frozen=True)
class LocalMultiplicityDensity:
    """Complete exact local Schur decomposition of one physical-cell RDM."""

    kind: CellKind
    cell: tuple[int, ...]
    state_norm_error: float
    physical_trace_error: float
    schur_trace_error: float
    blocks: tuple[MultiplicityBlockDensity, ...]

    def block(self, q: int) -> MultiplicityBlockDensity:
        """Return one registered doubled-spin block."""

        for block in self.blocks:
            if block.q == q:
                return block
        raise KeyError(q)


def _validated_basis(
    basis: Sequence[Sequence[int]], *, n_sites: int
) -> np.ndarray:
    values = np.asarray(basis, dtype=np.int8)
    if values.ndim != 2 or values.shape[1] != n_sites:
        raise ValueError(f"basis must have shape (dimension, {n_sites})")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("basis must contain only physical spin bits")
    weights = 1 << np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    indices = values @ weights
    if len(np.unique(indices)) != len(indices):
        raise ValueError("basis configurations must be duplicate free")
    return values


def _full_physical_state(
    amplitudes: Sequence[complex], basis: np.ndarray
) -> tuple[np.ndarray, float]:
    vector = np.asarray(amplitudes, dtype=np.complex128)
    if vector.ndim != 1 or vector.shape[0] != basis.shape[0]:
        raise ValueError("amplitudes and basis dimension disagree")
    if not np.all(np.isfinite(vector)):
        raise ValueError("amplitudes must be finite")
    norm = float(np.real(np.vdot(vector, vector)))
    if norm <= 1.0e-28:
        raise ValueError("wavefunction has zero numerical norm")
    normalized = vector / np.sqrt(norm)
    n_sites = basis.shape[1]
    weights = 1 << np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    indices = basis @ weights
    full = np.zeros(1 << n_sites, dtype=np.complex128)
    full[indices] = normalized
    return full, abs(norm - 1.0)


def _schur_columns(kind: CellKind) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    spec = cell_schur_spec(kind)
    table = cell_schur_tensor(kind)
    labels: list[tuple[int, int, int]] = []
    columns = []
    for q in spec.q_values:
        for m2 in magnetic_numbers(q):
            for multiplicity in range(spec.multiplicity(q)):
                labels.append((q, m2, multiplicity))
                columns.append(table[:, q, m2 + spec.cell_size, multiplicity])
    return np.column_stack(columns), tuple(labels)


def local_multiplicity_density(
    amplitudes: Sequence[complex],
    basis: Sequence[Sequence[int]],
    *,
    cell: Sequence[int],
    kind: CellKind,
) -> LocalMultiplicityDensity:
    """Trace a physical state to one cell and resolve all local ``q`` blocks.

    The state is normalized internally.  Sites in ``cell`` are ordered: that
    order must match the local square/triangle convention used to construct
    the model.  Sites outside the cell are traced in their original order.
    """

    spec = cell_schur_spec(kind)
    raw_cell = tuple(cell)
    if len(raw_cell) != spec.cell_size:
        raise ValueError(f"{kind} requires a cell of size {spec.cell_size}")
    if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw_cell):
        raise TypeError("cell site labels must be integers")
    checked_cell = tuple(int(site) for site in raw_cell)
    if len(set(checked_cell)) != len(checked_cell):
        raise ValueError("cell sites must be distinct")

    vector = np.asarray(amplitudes)
    if vector.ndim != 1:
        raise ValueError("amplitudes must be a vector")
    raw_basis = np.asarray(basis)
    if raw_basis.ndim != 2:
        raise ValueError("basis must be rank two")
    n_sites = raw_basis.shape[1]
    if any(site < 0 or site >= n_sites for site in checked_cell):
        raise ValueError("cell site lies outside the physical system")
    states = _validated_basis(basis, n_sites=n_sites)
    full, norm_error = _full_physical_state(vector, states)

    environment = tuple(site for site in range(n_sites) if site not in checked_cell)
    tensor = full.reshape((2,) * n_sites)
    matrix = np.transpose(tensor, checked_cell + environment).reshape(
        1 << spec.cell_size, -1
    )
    physical_rdm = matrix @ matrix.conj().T
    physical_trace_error = abs(float(np.trace(physical_rdm).real) - 1.0)

    schur_vectors, labels = _schur_columns(kind)
    schur_rdm = schur_vectors.conj().T @ physical_rdm @ schur_vectors
    schur_trace_error = abs(float(np.trace(schur_rdm).real) - 1.0)
    blocks = []
    for q in spec.q_values:
        multiplicity = spec.multiplicity(q)
        reduced = np.zeros((multiplicity, multiplicity), dtype=np.complex128)
        for m2 in magnetic_numbers(q):
            indices = [
                labels.index((q, m2, index)) for index in range(multiplicity)
            ]
            reduced += schur_rdm[np.ix_(indices, indices)]
        reduced = (reduced + reduced.conj().T) / 2.0
        weight = float(np.trace(reduced).real)
        conditional = reduced / weight if weight > 1.0e-14 else np.zeros_like(reduced)
        eigenvalues = np.linalg.eigvalsh(conditional)[::-1]
        eigenvalues = np.maximum(eigenvalues.real, 0.0)
        if eigenvalues.sum() > 0.0:
            eigenvalues /= eigenvalues.sum()
        positive = eigenvalues[eigenvalues > 1.0e-15]
        blocks.append(
            MultiplicityBlockDensity(
                q=q,
                block_weight=weight,
                matrix=reduced,
                conditional_matrix=conditional,
                eigenvalues=eigenvalues,
                purity=float(np.sum(eigenvalues**2)),
                entropy=float(-np.sum(positive * np.log(positive))),
            )
        )
    if abs(sum(block.block_weight for block in blocks) - 1.0) > 3.0e-12:
        raise RuntimeError("local Schur block weights do not close to one")
    return LocalMultiplicityDensity(
        kind=kind,
        cell=checked_cell,
        state_norm_error=norm_error,
        physical_trace_error=physical_trace_error,
        schur_trace_error=schur_trace_error,
        blocks=tuple(blocks),
    )


def local_multiplicity_density_json(
    density: LocalMultiplicityDensity,
) -> dict[str, object]:
    """Return a JSON-ready representation without hiding complex coherences."""

    def complex_matrix(values: np.ndarray) -> list[list[list[float]]]:
        return [
            [[float(value.real), float(value.imag)] for value in row]
            for row in values
        ]

    return {
        "kind": density.kind,
        "cell": list(density.cell),
        "state_norm_error_before_internal_normalization": density.state_norm_error,
        "physical_trace_error": density.physical_trace_error,
        "schur_trace_error": density.schur_trace_error,
        "blocks": [
            {
                "q": block.q,
                "block_weight": block.block_weight,
                "matrix": complex_matrix(block.matrix),
                "conditional_matrix": complex_matrix(block.conditional_matrix),
                "multiplicity_eigenvalues": block.eigenvalues.tolist(),
                "multiplicity_purity": block.purity,
                "multiplicity_entropy": block.entropy,
            }
            for block in density.blocks
        ],
    }


__all__ = [
    "LocalMultiplicityDensity",
    "MultiplicityBlockDensity",
    "local_multiplicity_density",
    "local_multiplicity_density_json",
]
