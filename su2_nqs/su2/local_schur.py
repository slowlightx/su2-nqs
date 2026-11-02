"""Complete exact local Schur transforms for small spin-half cells.

This is group-theory data, independent of neural models, Hamiltonians, and
benchmarks.  Tables use doubled ``q,m`` labels and computational basis bits
``0=down, 1=up`` with site zero most significant.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from .fusion_tree import FusionTree
from .quantum_numbers import magnetic_numbers, validate_m
from .recoupling import tree_basis_matrix


CellKind = Literal["square_plaquette", "triangular_triangle"]


@dataclass(frozen=True)
class CellSchurSpec:
    """Static complete local Schur decomposition for one cell kind."""

    kind: CellKind
    cell_size: int
    q_values: tuple[int, ...]
    multiplicities: tuple[int, ...]
    maximum_multiplicity: int
    symmetry_content: tuple[str, ...]

    def multiplicity(self, q: int) -> int:
        if q not in self.q_values:
            return 0
        return self.multiplicities[self.q_values.index(q)]


@lru_cache(maxsize=None)
def cell_schur_spec(kind: CellKind) -> CellSchurSpec:
    """Return complete local q sectors and multiplicities."""

    if kind == "square_plaquette":
        return CellSchurSpec(
            kind,
            4,
            (0, 2, 4),
            (2, 3, 1),
            3,
            ("q0:A1+B1", "q2:B2+E", "q4:A1"),
        )
    if kind == "triangular_triangle":
        return CellSchurSpec(
            kind,
            3,
            (1, 3),
            (2, 1),
            2,
            ("q1:E", "q3:A1"),
        )
    raise ValueError(f"unknown cell kind: {kind!r}")


def _local_basis_vectors(kind: CellKind, q: int, m2: int) -> np.ndarray:
    """Return exact computational-basis columns for a fixed local ``q,m``."""

    spec = cell_schur_spec(kind)
    if q not in spec.q_values:
        raise ValueError("q is absent from the requested cell Schur decomposition")
    validate_m(q, m2, name="m2")
    tree = (
        FusionTree.balanced(4)
        if kind == "square_plaquette"
        else FusionTree.left_associated(3)
    )
    vectors, states = tree_basis_matrix(tree, q, m2)
    if len(states) != spec.multiplicity(q):
        raise RuntimeError("local fusion-state count disagrees with Schur multiplicity")
    if kind == "square_plaquette" and q == 0:
        # Rotate [(01)_0(23)_0]_0 and [(01)_1(23)_1]_0 into local D4 A1/B1.
        rotation = np.asarray(
            (
                (np.sqrt(3.0) / 2.0, 0.5),
                (0.5, -np.sqrt(3.0) / 2.0),
            ),
            dtype=np.float64,
        )
        vectors = vectors @ rotation
    return vectors


@lru_cache(maxsize=None)
def cell_schur_tensor(kind: CellKind) -> np.ndarray:
    """Return ``U[s,q,m+cell_size,a]=<s|q,m,a>`` as a read-only table."""

    spec = cell_schur_spec(kind)
    table = np.zeros(
        (
            1 << spec.cell_size,
            spec.cell_size + 1,
            2 * spec.cell_size + 1,
            spec.maximum_multiplicity,
        ),
        dtype=np.float64,
    )
    complete_columns: list[np.ndarray] = []
    for q in spec.q_values:
        multiplicity = spec.multiplicity(q)
        for m2 in magnetic_numbers(q):
            vectors = np.asarray(_local_basis_vectors(kind, q, m2))
            if np.max(np.abs(vectors.imag), initial=0.0) > 2.0e-13:
                raise RuntimeError("local Schur basis unexpectedly became complex")
            real = np.asarray(vectors.real, dtype=np.float64)
            table[:, q, m2 + spec.cell_size, :multiplicity] = real
            complete_columns.extend(real[:, index] for index in range(multiplicity))
    unitary = np.column_stack(complete_columns)
    identity = np.eye(1 << spec.cell_size)
    if unitary.shape != identity.shape:
        raise RuntimeError("local Schur transform is incomplete")
    if max(
        np.linalg.norm(unitary.T @ unitary - identity),
        np.linalg.norm(unitary @ unitary.T - identity),
    ) > 2.0e-12:
        raise RuntimeError("local Schur transform is not unitary")
    table.setflags(write=False)
    return table


def cell_schur_unitarity_error(kind: CellKind) -> float:
    """Return the largest Gram/completeness error of the local Schur table."""

    spec = cell_schur_spec(kind)
    table = cell_schur_tensor(kind)
    columns = []
    for q in spec.q_values:
        for m2 in magnetic_numbers(q):
            for a in range(spec.multiplicity(q)):
                columns.append(table[:, q, m2 + spec.cell_size, a])
    unitary = np.column_stack(columns)
    identity = np.eye(1 << spec.cell_size)
    return float(
        max(
            np.linalg.norm(unitary.T @ unitary - identity),
            np.linalg.norm(unitary @ unitary.T - identity),
        )
    )


__all__ = [
    "CellKind",
    "CellSchurSpec",
    "cell_schur_spec",
    "cell_schur_tensor",
    "cell_schur_unitarity_error",
]
