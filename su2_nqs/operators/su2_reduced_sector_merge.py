"""Scalar SU(2) merge action entirely in multiplicity-pair coordinates.

Boundary vector operators use ReducedBlockData's convention
<qo,ao,mo|T_mu|qi,ai,mi> = CG(qi,mi;2,mu|qo,mo) R[qo,qi][ao,ai].
The finite magnetic CG sums are cached offline. Matvec never expands a
D_left by D_right magnetic product, nor forms its Hamiltonian matrix.
All q and m labels are doubled integers in the established CG convention.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.quantum_numbers import coupling_range
from .su2_sector_merge import (
    ReducedBlockData, _checked_bonds, _integer, _tolerance,
    compact_reduced_block,
)


def scalar_vector_recoupling(ql_out: int, qr_out: int,
                            ql_in: int, qr_in: int, target_q: int) -> float:
    """CG-summed coefficient of R_left X R_right.T for T_left dot T_right.

    Both pair bases are normalized CG states with parent doubled m=target_q.
    The scalar product is sum_mu (-1)^(mu/2) T_mu(left) T_-mu(right),
    with doubled mu=-2,0,2. No Wigner--Eckart normalization is inferred.
    """
    for q in (ql_out, qr_out, ql_in, qr_in, target_q):
        _integer(q, "doubled q", 0)
    return _scalar_vector_recoupling_cached(int(ql_out), int(qr_out), int(ql_in), int(qr_in), int(target_q))


@lru_cache(maxsize=32768)
def _scalar_vector_recoupling_cached(ql_out: int, qr_out: int,
                                     ql_in: int, qr_in: int, target_q: int) -> float:
    if (target_q not in coupling_range(ql_out, qr_out)
            or target_q not in coupling_range(ql_in, qr_in)):
        return 0.
    if ql_out not in coupling_range(ql_in, 2) or qr_out not in coupling_range(qr_in, 2):
        return 0.
    result = 0.
    for ml_in in range(-ql_in, ql_in+1, 2):
        mr_in = target_q-ml_in
        if abs(mr_in) > qr_in or (qr_in-mr_in) % 2:
            continue
        ket = clebsch_gordan(ql_in, ml_in, qr_in, mr_in, target_q, target_q)
        for mu in (-2, 0, 2):
            ml_out, mr_out = ml_in+mu, mr_in-mu
            if abs(ml_out) > ql_out or abs(mr_out) > qr_out:
                continue
            bra = clebsch_gordan(ql_out, ml_out, qr_out, mr_out, target_q, target_q)
            left = clebsch_gordan(ql_in, ml_in, 2, mu, ql_out, ml_out)
            right = clebsch_gordan(qr_in, mr_in, 2, -mu, qr_out, mr_out)
            result += (-1. if mu else 1.)*bra*ket*left*right
    return result


class ReducedSectorMergeOperator:
    """Highest-m scalar H without magnetic product-vector expansion.

    Channel order is identical to SectorMergeOperator. The constructor's
    optional ``_prepared`` argument is accepted for the existing solver's
    compact-boundary integration, but no prepared magnetic matrices are used.
    This class intentionally has no magnetic ``expand``/``project`` API;
    use only compact boundary propagation with it.
    """

    def __init__(self, left: ReducedBlockData, right: ReducedBlockData,
                 cross_bonds: Sequence[tuple[int, int, float]], target_q: int, *,
                 memory_limit_bytes: int = 512*1024**2, tolerance: float = 2e-10,
                 _prepared=None):
        self.tolerance = _tolerance(tolerance)
        self.memory_limit_bytes = _integer(memory_limit_bytes, "memory_limit_bytes")
        self.target_q = _integer(target_q, "target_q", 0)
        self.left = compact_reduced_block(left, tolerance=self.tolerance)
        self.right = compact_reduced_block(right, tolerance=self.tolerance)
        self.cross_bonds = _checked_bonds(left, right, cross_bonds)
        self.channels = tuple((ql, a, qr, b)
            for ql, dl in sorted(left.q_widths.items())
            for qr, dr in sorted(right.q_widths.items()) if self.target_q in coupling_range(ql, qr)
            for a in range(dl) for b in range(dr))
        self.dimension = len(self.channels)
        if not self.dimension:
            raise ValueError("target doubled-q sector is absent from the retained child product")
        all_arrays = [*left.h_by_q.values(), *right.h_by_q.values(),
            *(a for child in (self.left, self.right) for site in child.site_reduced_spin for a in site.values())]
        real = all(not np.any(np.asarray(a).imag) for a in all_arrays)
        self.dtype = np.dtype(np.float64 if real else np.complex128)
        cast = lambda a: np.asarray(np.asarray(a).real if real else a, dtype=self.dtype)
        self._left_h = {q: cast(a) for q, a in self.left.h_by_q.items()}
        self._right_h = {q: cast(a) for q, a in self.right.h_by_q.items()}
        self._slices = {}; cursor = 0
        for ql, dl in sorted(left.q_widths.items()):
            for qr, dr in sorted(right.q_widths.items()):
                if self.target_q in coupling_range(ql, qr):
                    self._slices[ql, qr] = (slice(cursor, cursor+dl*dr), dl, dr)
                    cursor += dl*dr
        weighted_right = {}
        for i, j, strength in self.cross_bonds:
            if strength == 0:
                continue
            row = weighted_right.setdefault(i, {})
            for key, matrix in self.right.site_reduced_spin[j].items():
                row[key] = row.get(key, 0.)+strength*cast(matrix)
        self._terms = []
        bytes_used = sum(a.nbytes for a in all_arrays)
        temporary_right_bytes = sum(a.nbytes for row in weighted_right.values() for a in row.values())
        workspace = self.dimension*16*8
        if bytes_used+workspace+temporary_right_bytes > self.memory_limit_bytes:
            raise MemoryError("reduced scalar merge exceeds the explicit memory budget")
        for (lo, ro), (out, _, _) in self._slices.items():
            for (li, ri), (incoming, _, _) in self._slices.items():
                factor = scalar_vector_recoupling(lo, ro, li, ri, self.target_q)
                if factor == 0:
                    continue
                aa, bb = [], []
                for site, right_rows in weighted_right.items():
                    a = self.left.site_reduced_spin[site].get((lo, li))
                    b = right_rows.get((ro, ri))
                    if a is not None and b is not None:
                        aa.append(cast(a)); bb.append(cast(b))
                if aa:
                    stack_bytes = sum(x.nbytes for x in aa+bb)
                    # Batched left/X/right products and their reduction;
                    # reserve complex128 even for a real input vector.
                    term_workspace = len(aa)*aa[0].shape[0]*(bb[0].shape[1]+bb[0].shape[0])*16*2
                    workspace = max(workspace, self.dimension*16*8+term_workspace)
                    if bytes_used+stack_bytes+workspace+temporary_right_bytes > self.memory_limit_bytes:
                        raise MemoryError("reduced scalar merge exceeds the explicit memory budget")
                    a, b = np.stack(aa), factor*np.stack(bb)
                    bytes_used += a.nbytes+b.nbytes
                    self._terms.append((out, incoming, (lo, ro), (li, ri), a, b))
        if bytes_used+workspace > self.memory_limit_bytes:
            raise MemoryError("reduced scalar merge exceeds the explicit memory budget")
        self.matvec_count = 0; self.matvec_seconds = 0.
        self.diagnostics = dict(target_q=self.target_q, reduced_dimension=self.dimension,
            product_dimension=left.dimension*right.dimension,
            magnetic_product_vectors_constructed=False, physical_basis_constructed=False,
            dense_product_H_constructed=False, scalar_CG_sum_terms=len(self._terms),
            estimated_reduced_storage_bytes=bytes_used, estimated_reduced_workspace_bytes=workspace,
            temporary_boundary_aggregation_bytes=temporary_right_bytes,
            estimated_product_workspace_bytes=0, exact_selection_rule_zero_max=0.,
            CG_convention="Condon-Shortley finite doubled-integer sum", compact_boundary_only=True)

    def matvec(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply all internal/cross bonds to normalized reduced pair channels."""
        started = perf_counter()
        values = np.asarray(coefficients)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("finite reduced coefficients of the declared dimension required")
        result = np.zeros(self.dimension, dtype=np.result_type(values.dtype, self.dtype))
        blocks = {key: values[sl].reshape(dl, dr) for key, (sl, dl, dr) in self._slices.items()}
        for (ql, qr), (sl, dl, dr) in self._slices.items():
            x = blocks[ql, qr]
            result[sl] = (self._left_h[ql] @ x+x @ self._right_h[qr].T).ravel()
        for out, _, out_key, in_key, a, b in self._terms:
            value = np.matmul(np.matmul(a, blocks[in_key]), np.swapaxes(b, -1, -2)).sum(axis=0)
            result[out] += value.ravel()
        self.matvec_count += 1; self.matvec_seconds += perf_counter()-started
        return result

    def linear_operator(self) -> LinearOperator:
        """SciPy Hermitian matrix-free operator over reduced multiplicities."""
        return LinearOperator((self.dimension, self.dimension), matvec=self.matvec,
                              rmatvec=self.matvec, dtype=self.dtype)

    def dense_reference(self, *, max_dimension: int = 64) -> np.ndarray:
        """Small-sector oracle, bounded independently of magnetic product size."""
        if self.dimension > min(_integer(max_dimension, "max_dimension"), 128):
            raise ValueError("dense sector reference exceeds its explicit small-system bound")
        eye = np.eye(self.dimension)
        return np.column_stack([self.matvec(eye[:, i]) for i in range(self.dimension)])


__all__ = ["scalar_vector_recoupling", "ReducedSectorMergeOperator"]
