"""Complete fixed-support singlet center and doubled-q Schmidt compression.

The two physical regions may be large; only their complete retained SU(2)
multiplets are used. The Hamiltonian is the projection of all supplied
internal and cross-region bonds, never a physical H-squared operator.
"""
from __future__ import annotations

from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from ..su2.quantum_numbers import validate_q
from .su2_sector_merge import ReducedBlockData, compact_reduced_block


class SingletCutOperator:
    """Dense reduced H on all coefficients C[q][a,b] of (A_q,E_q)->0.

    Each coefficient multiplies a normalized magnetic singlet, so its
    ordinary Frobenius norm already includes every magnetic component.
    The retained physical supports are fixed, not the coefficient matrix.
    """

    def __init__(self, left: ReducedBlockData, right: ReducedBlockData,
                 cross_bonds: Sequence[tuple[int, int, float]], *,
                 maximum_dimension: int = 4096):
        if (isinstance(maximum_dimension, bool) or not isinstance(maximum_dimension, Integral)
                or maximum_dimension < 1):
            raise ValueError('positive integer center dimension bound required')
        self.left = compact_reduced_block(left)
        self.right = compact_reduced_block(right)
        if set(left.q_widths) != set(right.q_widths):
            raise ValueError('matching doubled-q singlet sectors required')
        self.slices = {}; cursor = 0
        for q in sorted(left.q_widths):
            count = left.q_widths[q]*right.q_widths[q]
            self.slices[q] = slice(cursor, cursor+count); cursor += count
        self.dimension = cursor
        if not 0 < cursor <= maximum_dimension:
            raise ValueError('complete singlet center exceeds dimension bound')
        weighted = {}
        for i, j, value in cross_bonds:
            if (isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < left.n_sites
                    or isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < right.n_sites
                    or not np.isreal(value) or not np.isfinite(value)):
                raise ValueError('finite real cross bond with local physical sites required')
            row = weighted.setdefault(int(i), {})
            for key, matrix in self.right.site_reduced_spin[j].items():
                row[key] = row.get(key, 0.)+float(value)*matrix
        matrix = np.zeros((cursor, cursor), dtype=complex)
        for q, sl in self.slices.items():
            dl, dr = left.q_widths[q], right.q_widths[q]
            matrix[sl, sl] = (np.kron(left.h_by_q[q], np.eye(dr))
                              +np.kron(np.eye(dl), right.h_by_q[q]))
        for qo, out in self.slices.items():
            for qi, incoming in self.slices.items():
                if abs(qo-qi) > 2 or qo+qi < 2:
                    continue
                # Wigner--Eckart convention of ReducedBlockData. Both R
                # factors map qi->qo; the second ket index uses transpose.
                factor = -np.sqrt((qo+1)/(qi+1))
                for i, row in weighted.items():
                    a = self.left.site_reduced_spin[i].get((qo, qi))
                    b = row.get((qo, qi))
                    if a is not None and b is not None:
                        matrix[out, incoming] += factor*np.kron(a, b)
        self.hermiticity_error = float(np.max(abs(matrix-matrix.conj().T)))
        if self.hermiticity_error > 2e-9:
            raise ArithmeticError('complete reduced singlet Hamiltonian is not Hermitian')
        matrix = (matrix+matrix.conj().T)/2
        self.matrix = matrix.real if np.max(abs(matrix.imag)) < 1e-14 else matrix

    def encode(self, coefficients: Mapping[int, np.ndarray]) -> np.ndarray:
        """Flatten complete doubled-q ket coefficient matrices."""
        if set(coefficients) != set(self.slices):
            raise ValueError('all singlet doubled-q sectors required')
        values = []
        for q in self.slices:
            c = np.asarray(coefficients[q])
            if c.shape != (self.left.q_widths[q], self.right.q_widths[q]) or not np.isfinite(c).all():
                raise ValueError('finite matching singlet coefficient matrices required')
            values.append(c.ravel())
        return np.concatenate(values)

    def decode(self, vector: np.ndarray) -> dict[int, np.ndarray]:
        """Restore normalized-singlet coefficient matrix shapes."""
        v = np.asarray(vector)
        if v.shape != (self.dimension,) or not np.isfinite(v).all():
            raise ValueError('finite complete center vector required')
        return {q: v[sl].reshape(self.left.q_widths[q], self.right.q_widths[q])
                for q, sl in self.slices.items()}


def compress_singlet_center(coefficients: Mapping[int, np.ndarray],
        widths: Mapping[int, int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """q-SVD one SAME center, keeping whole multiplets and normalizing.

    Returns left rotations and new ket coefficients in the unchanged right
    frame. No q+1 weight multiplies singular values of normalized singlets.
    Consequently the global raw squared overlap equals retained norm.
    """
    if set(coefficients) != set(widths):
        raise ValueError('complete matching doubled-q widths required')
    rotations = {}; compressed = {}; spectra = {}; before = 0.; after = 0.
    for q, value in coefficients.items():
        validate_q(q)
        c = np.asarray(value)
        width = widths[q]
        if (c.ndim != 2 or not np.isfinite(c).all() or isinstance(width, bool)
                or not isinstance(width, Integral) or not 1 <= width <= min(c.shape)):
            raise ValueError('finite matrix and admissible positive same-q rank required')
        u, s, vh = np.linalg.svd(c, full_matrices=False)
        rotations[q] = u[:, :width]
        compressed[q] = s[:width, None]*vh[:width]
        spectra[q] = (s*s).tolist()
        before += float(np.vdot(c, c).real); after += float(np.sum(s[:width]**2))
    if abs(before-1) > 2e-10 or after <= 0:
        raise ValueError('normalized nonzero global singlet center required')
    compressed = {q: c/np.sqrt(after) for q, c in compressed.items()}
    return rotations, compressed, dict(retained_norm_squared=after,
        discarded_norm_squared=max(0., before-after), normalized_raw_fidelity_squared=after,
        squared_singular_values_by_q=spectra, magnetic_multiplets_kept_complete=True,
        raw_fidelity_is_not_projected_A1_fidelity=True)
