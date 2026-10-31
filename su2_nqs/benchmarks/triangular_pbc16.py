"""Exact geometry/operator invariants for the 4x4 triangular PBC pilot."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


ROWS = 4
COLUMNS = 4
N = ROWS * COLUMNS
TRANSLATIONS = ((1, 0), (0, 1), (1, 1))


def translate_site(site: int, dr: int, dc: int) -> int:
    """Translate a row-major 4x4 site by an oblique lattice displacement."""

    if not 0 <= site < N:
        raise ValueError("site must lie on the 4x4 torus")
    row, column = divmod(site, COLUMNS)
    return ((row + dr) % ROWS) * COLUMNS + (column + dc) % COLUMNS


def geometry_audit(
    bonds: tuple[tuple[int, int, float], ...],
) -> dict[str, object]:
    """Check coordination and primitive-translation invariance of one shell."""

    weights = {(first, second): coupling for first, second, coupling in bonds}
    coordination: defaultdict[int, float] = defaultdict(float)
    for (first, second), coupling in weights.items():
        coordination[first] += coupling
        coordination[second] += coupling

    translation_mismatches: dict[str, int] = {}
    for dr, dc in TRANSLATIONS:
        translated = {}
        for (first, second), coupling in weights.items():
            edge = tuple(
                sorted((translate_site(first, dr, dc), translate_site(second, dr, dc)))
            )
            translated[edge] = coupling
        translation_mismatches[f"{dr},{dc}"] = len(
            set(weights.items()).symmetric_difference(set(translated.items()))
        )
    values = np.asarray([coordination[site] for site in range(N)])
    return {
        "bond_count": len(bonds),
        "expected_bond_count": 3 * N,
        "minimum_coordination": float(np.min(values)),
        "maximum_coordination": float(np.max(values)),
        "translation_mismatches": translation_mismatches,
    }


def operator_audit(matrix) -> dict[str, object]:
    """Return sparse storage and Hermiticity diagnostics for a shell operator."""

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be square")
    difference = (matrix - matrix.getH()).tocsr()
    maximum = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    csr_bytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return {
        "shape": list(matrix.shape),
        "dimension": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "csr_bytes": csr_bytes,
        "maximum_hermiticity_residual": maximum,
    }
