"""Matrix-free SU(2) coarse merges without a physical or product-H basis.

All angular-momentum labels are doubled integers. Boundary spin operators act
on retained child multiplets, ordered (q, multiplicity, m), with increasing m.
For each target q only its highest magnetic component m=q is diagonalized.
Fixed CG coefficients expand a reduced vector into a D_left by D_right array;
Hamiltonian action uses sparse child operators. No (D_left*D_right)^2 matrix,
nor a 2**N physical basis, is constructed.

The resulting scalar Hamiltonian is the projection into the retained child
space. Its eigenvalue residual is NOT the physical energy variance. Discarded
child-space contributions require the full physical H action elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.quantum_numbers import coupling_range
from .su2_block_transfer import ProjectedBlock


@dataclass(frozen=True)
class ReducedBlockData:
    """Basis-free SU(2) child block; complete multiplets are always retained.

    Projected site vectors assemble NEW cross-block couplings. Multiplying
    them does not recover internal two-site correlations across discarded
    intermediate states; the internal scalar Hamiltonian is stored separately.
    """

    n_sites: int
    q_widths: Mapping[int, int]
    h_by_q: Mapping[int, np.ndarray]
    site_spin: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...] = ()
    site_reduced_spin: tuple[Mapping[tuple[int, int], np.ndarray], ...] | None = None

    # In compact form <q_o,a_o,m_o|T_mu|q_i,a_i,m_i> equals
    # CG(q_i,m_i;2,mu|q_o,m_o) R[(q_o,q_i)][a_o,a_i].
    # Spherical components: T_0=S_z, T_{+2}=-S_+/sqrt(2),
    # T_{-2}=S_-/sqrt(2). Only one representation may be populated.

    @property
    def labels(self) -> tuple[tuple[int, int, int], ...]:
        """Deterministic (doubled q, multiplicity index, doubled m) order."""
        return tuple((int(q), a, m) for q, d in sorted(self.q_widths.items())
                     for a in range(int(d)) for m in range(-int(q), int(q) + 1, 2))

    @property
    def dimension(self) -> int:
        return sum((int(q) + 1) * int(d) for q, d in self.q_widths.items())


@dataclass(frozen=True)
class SectorMergeResult:
    """Retained reduced maps and optional recursively reusable block data."""

    energies: Mapping[int, np.ndarray]
    reduced_maps: Mapping[int, np.ndarray]
    channels: Mapping[int, tuple[tuple[int, int, int, int], ...]]
    eigen_residuals: Mapping[int, np.ndarray]
    block: ReducedBlockData | None
    diagnostics: Mapping[str, object]
    warnings: tuple[str, ...]


class SectorMergeSolveError(RuntimeError):
    """Bounded eigensolve failed; diagnostics preserve work already incurred."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _tolerance(value: float) -> float:
    if isinstance(value, (bool, complex)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError('tolerance must be finite and positive')
    return float(value)


def _labels(q_widths: Mapping[int, int], n_sites: int, *,
            max_dimension: int | None = None) -> tuple[tuple[int, int, int], ...]:
    _integer(n_sites, 'n_sites')
    if not q_widths:
        raise ValueError('at least one retained doubled-q sector is required')
    for q, d in q_widths.items():
        _integer(q, 'q', 0)
        _integer(d, 'multiplicity')
        if q > n_sites or (q - n_sites) % 2:
            raise ValueError('doubled q must have the block site parity and not exceed n_sites')
    dimension = sum((int(q)+1)*int(d) for q, d in q_widths.items())
    if max_dimension is not None and dimension > _integer(max_dimension, 'max_dimension'):
        raise ValueError('retained child dimension exceeds explicit validation bound')
    return tuple((int(q), a, m) for q, d in sorted(q_widths.items())
                 for a in range(int(d)) for m in range(-int(q), int(q) + 1, 2))


def _generators(labels: tuple[tuple[int, int, int], ...]) -> tuple[np.ndarray, sparse.csr_matrix]:
    pos = {label: i for i, label in enumerate(labels)}
    rows, cols, values = [], [], []
    for col, (q, a, m) in enumerate(labels):
        if m < q:
            rows.append(pos[q, a, m + 2]); cols.append(col)
            values.append(np.sqrt((q - m) * (q + m + 2)) / 2)
    return np.asarray([m / 2 for _, _, m in labels]), sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(labels), len(labels)))


def _hermitian(matrix: np.ndarray, shape: tuple[int, int], name: str, tol: float) -> np.ndarray:
    a = np.asarray(matrix)
    if (a.shape != shape or not np.all(np.isfinite(a)) or
            np.max(abs(a - a.conj().T)) > tol * max(1., float(np.max(abs(a))))):
        raise ValueError(f'{name} must be finite and Hermitian with shape {shape}')
    return a


def validate_reduced_block(block: ReducedBlockData, *, tolerance: float = 2e-10,
                           max_dimension: int = 4096) -> dict[str, float]:
    """Check full total-spin generators and local vector covariance.

    Projected individual sites need not obey their original onsite spin
    commutators. Their SUM must be the known total-spin generators, and each
    individual boundary operator must transform as a vector under that sum.
    """
    tol = _tolerance(tolerance)
    labels = _labels(block.q_widths, block.n_sites, max_dimension=max_dimension)
    d = len(labels)
    if set(block.h_by_q) != set(block.q_widths):
        raise ValueError('scalar reduced Hamiltonian must cover precisely the retained q sectors')
    for q, width in block.q_widths.items():
        _hermitian(block.h_by_q[q], (width, width), f'H[q={q}]', tol)
    if block.site_reduced_spin is not None:
        return _validate_compact_spin(block, tol)
    if len(block.site_spin) != block.n_sites:
        raise ValueError('one vector operator per physical child site is required')
    jz, jp = _generators(labels)
    total_z = np.zeros((d, d), complex); total_p = np.zeros((d, d), complex)
    error = 0.; scale = 1.
    for site in block.site_spin:
        if len(site) != 3:
            raise ValueError('site_spin must contain x, y, z components')
        x, y, z = (_hermitian(s, (d, d), 'site spin', tol) for s in site)
        p = x + 1j * y; minus = p.conj().T
        total_z += z; total_p += p
        checks = ((jz[:, None] - jz[None, :]) * z,
                  (jz[:, None] - jz[None, :]) * p - p,
                  jp @ z - (jp.T @ z.T).T + p,
                  jp @ p - (jp.T @ p.T).T,
                  jp @ minus - (jp.T @ minus.T).T - 2*z)
        error = max(error, *(float(np.max(abs(value))) for value in checks))
        scale = max(scale, float(np.max(abs(x))), float(np.max(abs(y))), float(np.max(abs(z))))
    total_error = max(float(np.max(abs(total_z - np.diag(jz)))),
                      float(np.max(abs(total_p - jp.toarray()))))
    if max(error, total_error) > tol * scale:
        raise ValueError('boundary operators violate total-spin identity or rank-one SU(2) covariance')
    return dict(total_spin_error=total_error, vector_covariance_error=error,
                retained_dimension=float(d))


def _validate_compact_spin(block: ReducedBlockData, tol: float) -> dict[str, float]:
    """Rank-one covariance is fixed by CG; check reduced adjoints and total J."""
    if block.site_spin or len(block.site_reduced_spin) != block.n_sites:
        raise ValueError('compact blocks require only one reduced vector operator per site')
    total = {}; error = 0.; scale = 1.
    for site in block.site_reduced_spin:
        for key, value in site.items():
            if (not isinstance(key, tuple) or len(key) != 2
                    or any(isinstance(q, bool) or not isinstance(q, Integral) for q in key)):
                raise ValueError('reduced spin keys must be doubled (q_out,q_in) pairs')
            qo, qi = key
            if (qo not in block.q_widths or qi not in block.q_widths
                    or abs(qo-qi) > 2 or qo+qi < 2 or (qo-qi) % 2):
                raise ValueError('reduced spin block violates rank-one charge selection')
            value = np.asarray(value)
            shape = (block.q_widths[qo], block.q_widths[qi])
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError('reduced spin matrix has invalid shape or nonfinite entries')
            reverse = np.asarray(site.get((qi, qo), np.zeros(shape[::-1])))
            factor = (-1.)**((qi-qo)//2)*np.sqrt((qo+1)/(qi+1))
            error = max(error, float(np.max(abs(reverse-factor*value.conj().T))))
            scale = max(scale, float(np.max(abs(value))))
            total[key] = total.get(key, np.zeros(shape, complex)) + value
    total_error = 0.
    for qo, do in block.q_widths.items():
        for qi, di in block.q_widths.items():
            expected = np.sqrt(qo*(qo+2))/2*np.eye(do) if qo == qi else np.zeros((do, di))
            total_error = max(total_error, float(np.max(abs(total.get((qo, qi), 0.)-expected))))
    if max(error, total_error) > tol*scale:
        raise ValueError('compact boundary operators violate reduced Hermiticity or total-spin identity')
    return dict(total_spin_error=total_error, vector_covariance_error=0.,
                reduced_hermiticity_error=error, retained_dimension=float(block.dimension))


def compact_reduced_block(block: ReducedBlockData, *, tolerance: float = 2e-10) -> ReducedBlockData:
    """Convert dense boundary spins to exact rank-one multiplicity matrices.

    The conversion is a change of storage, not a further subspace truncation.
    All three input vector components are validated before extracting from S_z.
    """
    tol = _tolerance(tolerance)
    validate_reduced_block(block, tolerance=tol)
    if block.site_reduced_spin is not None:
        return block
    positions = {label: i for i, label in enumerate(block.labels)}
    reduced = []
    for _, _, z in block.site_spin:
        site = {}
        for qo, do in sorted(block.q_widths.items()):
            for qi, di in sorted(block.q_widths.items()):
                if abs(qo-qi) > 2 or qo+qi < 2:
                    continue
                value = np.zeros((do, di), complex); norm = 0.
                for m in range(-min(qo, qi), min(qo, qi)+1, 2):
                    cg = clebsch_gordan(qi, m, 2, 0, qo, m)
                    rows = [positions[qo, a, m] for a in range(do)]
                    cols = [positions[qi, a, m] for a in range(di)]
                    value += cg*z[np.ix_(rows, cols)]; norm += cg*cg
                if norm == 0.:
                    raise ArithmeticError('rank-one reduced matrix has no nonzero S_z component')
                site[qo, qi] = value/norm
        reduced.append(site)
    out = ReducedBlockData(block.n_sites, dict(block.q_widths), block.h_by_q,
                           site_reduced_spin=tuple(reduced))
    validate_reduced_block(out, tolerance=tol)
    return out


def reduced_block_from_arrays(n_sites: int, labels: Sequence[tuple[int, int, int]],
                              hamiltonian: np.ndarray,
                              site_spin: Sequence[Sequence[np.ndarray]], *,
                              tolerance: float = 2e-10) -> ReducedBlockData:
    """Adapt stored reduced operators, rejecting magnetic splitting/order errors."""
    tol = _tolerance(tolerance)
    labels = tuple(tuple(x) for x in labels)
    if len(labels) > 4096:
        raise ValueError('retained child dimension exceeds explicit adapter bound 4096')
    for row in labels:
        if len(row) != 3 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in row):
            raise ValueError('labels must be doubled integer (q,a,m) triples')
    widths = {q: max(a for charge, a, _ in labels if charge == q) + 1 for q, _, _ in labels}
    if labels != _labels(widths, n_sites, max_dimension=4096):
        raise ValueError('labels must contain complete multiplets in canonical q,a,m order')
    d = len(labels)
    if d > 4096:
        raise ValueError('retained child dimension exceeds explicit adapter bound 4096')
    h = _hermitian(hamiltonian, (d, d), 'projected H', tol)
    pos = {label: i for i, label in enumerate(labels)}
    blocks = {}; reconstructed = np.zeros_like(h)
    for q, width in widths.items():
        ids = [pos[q, a, q] for a in range(width)]
        value = np.array(h[np.ix_(ids, ids)], copy=True)
        blocks[int(q)] = value
        for m in range(-q, q + 1, 2):
            ids = [pos[q, a, m] for a in range(width)]
            reconstructed[np.ix_(ids, ids)] = value
    if np.max(abs(h - reconstructed)) > tol * max(1., float(np.max(abs(h)))):
        raise ValueError('projected H is not a scalar common to all magnetic components')
    spin = tuple(tuple(np.array(a, copy=True) for a in site) for site in site_spin)
    out = ReducedBlockData(int(n_sites), widths, blocks, spin)
    validate_reduced_block(out, tolerance=tol)
    return out


def reduced_block_from_projected(block: ProjectedBlock, *, tolerance: float = 2e-10) -> ReducedBlockData:
    """Drop the physical basis after adapting a validated small-block reference."""
    return reduced_block_from_arrays(block.isometry.n_sites, block.labels,
                                     block.hamiltonian, block.site_spin, tolerance=tolerance)


class _LazySpinComponents:
    """Sparse magnetic matrices are materialized only at queried boundary sites."""

    def __init__(self, owner: _SparseBlock, component: int):
        self.owner = owner; self.component = component

    def __len__(self) -> int:
        return self.owner.block.n_sites

    def __getitem__(self, site: int) -> sparse.csr_matrix:
        if not 0 <= site < len(self):
            raise IndexError(site)
        return self.owner.compact_site_matrices(site)[self.component]


def _input_storage_bytes(block: ReducedBlockData) -> int:
    spin = (sum(np.asarray(a).nbytes for site in block.site_spin for a in site)
            if block.site_reduced_spin is None else
            sum(np.asarray(a).nbytes for site in block.site_reduced_spin for a in site.values()))
    return spin + sum(np.asarray(h).nbytes for h in block.h_by_q.values())


class _SparseBlock:
    def __init__(self, block: ReducedBlockData, tol: float):
        self.validation = validate_reduced_block(block, tolerance=tol)
        self.block = block
        self.labels = block.labels
        self.positions = {x: i for i, x in enumerate(self.labels)}
        self.dimension = len(self.labels)
        rows, cols, values = [], [], []
        for q, h in block.h_by_q.items():
            for m in range(-q, q + 1, 2):
                for a, b in np.ndindex(h.shape):
                    rows.append(self.positions[q, a, m]); cols.append(self.positions[q, b, m])
                    values.append(h[a, b])
        self.h = sparse.csr_matrix((values, (rows, cols)), shape=(self.dimension,)*2)
        self.input_storage_bytes = _input_storage_bytes(block)
        self._compact_cache = {}
        if block.site_reduced_spin is not None:
            self.selection_error = 0.
            self.is_real = (not np.any(self.h.data.imag)
                and all(not np.any(np.asarray(a).imag) for site in block.site_reduced_spin for a in site.values()))
            if self.is_real:
                self.h = self.h.real.tocsr()
            self.h.eliminate_zeros()
            self.z = _LazySpinComponents(self, 0)
            self.p = _LazySpinComponents(self, 1)
            self.minus = _LazySpinComponents(self, 2)
            return
        charges = np.asarray([q for q, _, _ in self.labels])
        magnetic = np.asarray([m for _, _, m in self.labels])
        vector_allowed = ((abs(charges[:, None] - charges[None, :]) <= 2)
                          & (charges[:, None] + charges[None, :] >= 2))
        z_allowed = vector_allowed & (magnetic[:, None] == magnetic[None, :])
        p_allowed = vector_allowed & (magnetic[:, None] == magnetic[None, :] + 2)
        self.z = []; self.p = []; self.minus = []; self.selection_error = 0.
        for x, y, z in block.site_spin:
            plus = x + 1j*y
            self.selection_error = max(self.selection_error,
                float(np.max(abs(np.where(z_allowed, 0., z)))),
                float(np.max(abs(np.where(p_allowed, 0., plus)))))
            scale = max(1., float(np.max(abs(z))), float(np.max(abs(plus))))
            if self.selection_error > tol*scale:
                raise ValueError('boundary vector violates exact magnetic/triangle selection rules')
            self.z.append(sparse.csr_matrix(np.where(z_allowed, z, 0.)))
            self.p.append(sparse.csr_matrix(np.where(p_allowed, plus, 0.)))
            self.minus.append(self.p[-1].conj().T.tocsr())
        matrices = [self.h, *self.z, *self.p, *self.minus]
        self.is_real = all(not np.any(a.data.imag) for a in matrices)
        if self.is_real:
            self.h = self.h.real.tocsr()
            self.z = [a.real.tocsr() for a in self.z]
            self.p = [a.real.tocsr() for a in self.p]
            self.minus = [a.real.tocsr() for a in self.minus]
        for a in [self.h, *self.z, *self.p, *self.minus]:
            a.eliminate_zeros()
    @property
    def storage_bytes(self) -> int:
        matrices = ([self.h, *(a for triple in self._compact_cache.values() for a in triple)]
                    if self.block.site_reduced_spin is not None else
                    [self.h, *self.z, *self.p, *self.minus])
        return sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in matrices)

    def compact_site_matrices(self, site: int) -> tuple[sparse.csr_matrix, ...]:
        if site in self._compact_cache:
            return self._compact_cache[site]
        matrices = []
        for mu, factor in ((0, 1.), (2, -np.sqrt(2.))):
            rows, cols, data = [], [], []
            for (qo, qi), value in self.block.site_reduced_spin[site].items():
                aa, bb = np.nonzero(value)
                for mi in range(-qi, qi+1, 2):
                    mo = mi+mu
                    if abs(mo) > qo:
                        continue
                    cg = clebsch_gordan(qi, mi, 2, mu, qo, mo)*factor
                    if cg == 0.:
                        continue
                    rows.extend(self.positions[qo, int(a), mo] for a in aa)
                    cols.extend(self.positions[qi, int(b), mi] for b in bb)
                    data.extend(cg*np.asarray(value)[aa, bb])
            matrix = sparse.csr_matrix((data, (rows, cols)), shape=(self.dimension,)*2)
            if self.is_real:
                matrix = matrix.real.tocsr()
            matrix.eliminate_zeros(); matrices.append(matrix)
        matrices.append(matrices[1].conj().T.tocsr())
        self._compact_cache[site] = tuple(matrices)
        return self._compact_cache[site]


def _precheck_memory(left: ReducedBlockData, right: ReducedBlockData, limit: int) -> None:
    """Reject unreasonable metadata/workspaces before vector-validation allocates."""
    dimensions = [len(_labels(b.q_widths, b.n_sites, max_dimension=4096)) for b in (left, right)]
    inputs = sum(_input_storage_bytes(b) for b in (left, right))
    if dimensions[0]*dimensions[1]*16*8 + max(dimensions)**2*16*12 + inputs > limit:
        raise MemoryError('input/validation/product workspace exceeds the explicit memory budget')


def _embedding_storage_bound(left: ReducedBlockData, right: ReducedBlockData, q: int,
                             all_m: bool) -> int:
    # CSR indices are conservatively charged as int64; exact CG zeros can only
    # reduce the cache. Reserve every m needed by boundary propagation together.
    nnz = sum(dl*dr*(min(ql, qr)+1) for ql, dl in left.q_widths.items()
              for qr, dr in right.q_widths.items() if q in coupling_range(ql, qr))
    return (q+1 if all_m else 1)*(nnz*16+(left.dimension*right.dimension+1)*8)


def _checked_bonds(left: ReducedBlockData, right: ReducedBlockData,
                    cross_bonds: Sequence[tuple[int, int, float]]) -> tuple[tuple[int, int, float], ...]:
    bonds = []
    for i, j, strength in cross_bonds:
        if (isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < left.n_sites
                or isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < right.n_sites
                or np.iscomplexobj(strength) or not np.isfinite(strength)):
            raise ValueError('cross bonds require valid child site indices and finite real strength')
        bonds.append((int(i), int(j), float(strength)))
    return tuple(bonds)


def _prepare_boundary_sites(prepared: tuple[_SparseBlock, _SparseBlock],
                             bonds: Sequence[tuple[int, int, float]], limit: int,
                             all_sites: bool = False) -> None:
    needed = (set(range(prepared[0].block.n_sites)), set(range(prepared[1].block.n_sites))) if all_sites else (
        {i for i, _, g in bonds if g}, {j for _, j, g in bonds if g})
    reserve = 0
    for child, sites in zip(prepared, needed, strict=True):
        if child.block.site_reduced_spin is None:
            continue
        for site in sites-set(child._compact_cache):
            nnz_bound = sum(np.count_nonzero(value)*(min(qo, qi)+1)
                            for (qo, qi), value in child.block.site_reduced_spin[site].items())
            reserve += 3*(nnz_bound*24+(child.dimension+1)*8)
    base = (prepared[0].dimension*prepared[1].dimension*16*8
            + max(b.dimension for b in prepared)**2*16*12
            + sum(b.storage_bytes+b.input_storage_bytes for b in prepared))
    if base+reserve > limit:
        raise MemoryError('boundary sparse expansion exceeds the explicit memory budget')
    for child, sites in zip(prepared, needed, strict=True):
        for site in sites:
            child.z[site]


class SectorMergeOperator:
    """Highest-m matrix-free scalar H for one total doubled-q sector."""

    def __init__(self, left: ReducedBlockData, right: ReducedBlockData,
                 cross_bonds: Sequence[tuple[int, int, float]], target_q: int, *,
                 memory_limit_bytes: int = 512 * 1024**2, tolerance: float = 2e-10,
                 _prepared: tuple[_SparseBlock, _SparseBlock] | None = None):
        self.tolerance = _tolerance(tolerance)
        self.memory_limit_bytes = _integer(memory_limit_bytes, 'memory_limit_bytes')
        self.target_q = _integer(target_q, 'target_q', 0)
        _precheck_memory(left, right, self.memory_limit_bytes)
        self.left, self.right = _prepared if _prepared is not None else (
            _SparseBlock(left, self.tolerance), _SparseBlock(right, self.tolerance))
        self.shape_product = (self.left.dimension, self.right.dimension)
        self.dtype = np.dtype(np.float64 if self.left.is_real and self.right.is_real else np.complex128)
        self.channels = tuple((ql, a, qr, b) for ql, dl in sorted(left.q_widths.items())
            for qr, dr in sorted(right.q_widths.items()) if self.target_q in coupling_range(ql, qr)
            for a in range(dl) for b in range(dr))
        self.dimension = len(self.channels)
        if not self.dimension:
            raise ValueError('target doubled-q sector is absent from the retained child product')
        self.cross_bonds = _checked_bonds(left, right, cross_bonds)
        _prepare_boundary_sites((self.left, self.right), self.cross_bonds, self.memory_limit_bytes)
        product_dimension = self.left.dimension * self.right.dimension
        # Reserve eight complex product arrays even if the input happens to be
        # real, plus sparse operators; eigensolver/output reserves are added later.
        workspace = product_dimension * 16 * 8
        storage = self.left.storage_bytes + self.right.storage_bytes
        input_storage = self.left.input_storage_bytes + self.right.input_storage_bytes
        validation_workspace = max(self.shape_product)**2*16*12
        if workspace + storage + input_storage + validation_workspace > self.memory_limit_bytes:
            raise MemoryError('matrix-free product-vector workspace exceeds the explicit memory budget')
        self._embeddings: dict[int, sparse.csr_matrix] = {}
        self.matvec_count = 0; self.matvec_seconds = 0.
        self.diagnostics: dict[str, object] = dict(target_q=self.target_q,
            reduced_dimension=self.dimension, product_dimension=product_dimension,
            estimated_product_workspace_bytes=workspace, child_sparse_operator_bytes=storage,
            resident_input_operator_bytes=input_storage,
            reserved_validation_workspace_bytes=validation_workspace,
            forbidden_dense_product_H_bytes=product_dimension**2*16,
            exact_selection_rule_zero_max=max(self.left.selection_error, self.right.selection_error),
            physical_basis_constructed=False, dense_product_H_constructed=False)
        self._embedding(self.target_q)

    def _embedding(self, m: int) -> sparse.csr_matrix:
        if isinstance(m, bool) or not isinstance(m, Integral) or abs(m) > self.target_q or (m-self.target_q) % 2:
            raise ValueError('m must be a valid doubled magnetic projection for target_q')
        m = int(m)
        if m in self._embeddings:
            return self._embeddings[m]
        rows, cols, data = [], [], []
        for k, (ql, al, qr, ar) in enumerate(self.channels):
            for ml in range(-ql, ql + 1, 2):
                mr = m - ml
                if abs(mr) > qr or (mr-qr) % 2:
                    continue
                coefficient = clebsch_gordan(ql, ml, qr, mr, self.target_q, m)
                if coefficient != 0.:
                    rows.append(self.left.positions[ql, al, ml]*self.right.dimension
                                + self.right.positions[qr, ar, mr])
                    cols.append(k); data.append(coefficient)
        value = sparse.csr_matrix((data, (rows, cols)),
                                  shape=(np.prod(self.shape_product), self.dimension))
        gram_diag = np.asarray(value.power(2).sum(axis=0)).ravel()
        if np.max(abs(gram_diag - 1.)) > self.tolerance:
            raise ArithmeticError('CG sector embedding lost normalization')
        # Distinct channels have disjoint product labels, so their Gram
        # off-diagonal entries vanish without a dense transform multiplication.
        self._embeddings[m] = value
        self.diagnostics['CG_normalization_error'] = max(
            float(self.diagnostics.get('CG_normalization_error', 0.)), float(np.max(abs(gram_diag-1.))))
        self.diagnostics['cached_embedding_bytes'] = sum(
            c.data.nbytes+c.indices.nbytes+c.indptr.nbytes for c in self._embeddings.values())
        if (self.diagnostics['estimated_product_workspace_bytes']
                + self.diagnostics['child_sparse_operator_bytes']
                + self.diagnostics['resident_input_operator_bytes']
                + self.diagnostics['reserved_validation_workspace_bytes']
                + self.diagnostics['cached_embedding_bytes'] > self.memory_limit_bytes):
            raise MemoryError('CG embedding cache exceeds the explicit memory budget')
        return value

    def expand(self, coefficients: np.ndarray, m: int | None = None) -> np.ndarray:
        """Expand one reduced vector into a product array, without physical sites."""
        values = np.asarray(coefficients)
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise ValueError('finite reduced coefficients of the declared sector dimension required')
        return np.asarray(self._embedding(self.target_q if m is None else m) @ values).reshape(self.shape_product)

    def project(self, product_array: np.ndarray, m: int | None = None) -> np.ndarray:
        """Adjoint CG projection of one product array in a chosen magnetic slice."""
        array = np.asarray(product_array)
        if array.shape != self.shape_product or not np.all(np.isfinite(array)):
            raise ValueError('finite array with the retained child product shape required')
        return np.asarray(self._embedding(self.target_q if m is None else m).conj().T @ array.ravel())

    def product_action(self, value: np.ndarray) -> np.ndarray:
        """Apply the compressed child Hamiltonian to a D_left by D_right array."""
        x = np.asarray(value)
        if x.shape != self.shape_product or not np.all(np.isfinite(x)):
            raise ValueError('finite retained child product array required')
        result = self.left.h @ x + (self.right.h @ x.T).T
        for i, j, strength in self.cross_bonds:
            if strength == 0.:
                continue
            result = result + strength * (
                self.left.z[i] @ (self.right.z[j] @ x.T).T
                + .5*(self.left.p[i] @ (self.right.minus[j] @ x.T).T
                      + self.left.minus[i] @ (self.right.p[j] @ x.T).T))
        return result

    def matvec(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply the target-q Hamiltonian to one reduced vector."""
        start = perf_counter()
        result = self.project(self.product_action(self.expand(coefficients)))
        self.matvec_count += 1; self.matvec_seconds += perf_counter() - start
        return result

    def linear_operator(self) -> LinearOperator:
        """SciPy Hermitian operator; no dense sector or product matrix is cached."""
        return LinearOperator((self.dimension, self.dimension), matvec=self.matvec,
                              rmatvec=self.matvec, dtype=self.dtype)

    def dense_reference(self, *, max_dimension: int = 64) -> np.ndarray:
        """Explicit small-sector test oracle, guarded independently of product size."""
        if self.dimension > min(_integer(max_dimension, 'max_dimension'), 128):
            raise ValueError('dense sector reference exceeds its explicit small-system bound')
        eye = np.eye(self.dimension)
        return np.column_stack([self.matvec(eye[:, i]) for i in range(self.dimension)])


def _canonical_eigenvectors(values: np.ndarray, vectors: np.ndarray, tol: float
                            ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Fix phases and degenerate bases using low-rank projector actions only."""
    columns = []; groups = []; start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and abs(values[stop]-values[start]) <= tol*max(1., abs(values[start])):
            stop += 1
        v = np.asarray(vectors[:, start:stop], dtype=np.complex128)
        local = []
        for row in range(len(v)):
            candidate = v @ v[row].conj()
            for _ in range(2):
                for old in local:
                    candidate -= old*np.vdot(old, candidate)
            norm = np.linalg.norm(candidate)
            if norm < 1e-10:
                continue
            candidate /= norm
            # Symmetry can make several entries exactly equal in magnitude;
            # roundoff must not choose different phase anchors across equivalent
            # dense/compact contraction paths.
            magnitudes = abs(candidate)
            pivot = int(np.flatnonzero(magnitudes >= np.max(magnitudes)*(1.-1e-12))[0])
            candidate *= np.conj(candidate[pivot])/abs(candidate[pivot])
            local.append(candidate)
            if len(local) == stop-start:
                break
        if len(local) != stop-start:
            raise ArithmeticError('retained energy eigenspace lost rank')
        columns.extend(local); groups.append((start, stop)); start = stop
    return np.column_stack(columns), tuple(groups)


def _propagate_boundary(operators: Mapping[int, SectorMergeOperator],
                        maps: Mapping[int, np.ndarray], energies: Mapping[int, np.ndarray],
                        left: ReducedBlockData, right: ReducedBlockData,
                        tolerance: float) -> ReducedBlockData:
    widths = {q: w.shape[1] for q, w in maps.items()}
    labels = _labels(widths, left.n_sites+right.n_sites)
    positions = {v: i for i, v in enumerate(labels)}
    dim = len(labels)
    z = [np.zeros((dim, dim), complex) for _ in range(left.n_sites+right.n_sites)]
    plus = [np.zeros((dim, dim), complex) for _ in z]
    sample = next(iter(operators.values()))
    for column, (q, a, m) in enumerate(labels):
        x = operators[q].expand(maps[q][:, a], m)
        for i in range(len(z)):
            is_left = i < left.n_sites
            site = i if is_left else i-left.n_sites
            child = sample.left if is_left else sample.right
            actions = (child.z[site] @ x if is_left else (child.z[site] @ x.T).T,
                       child.p[site] @ x if is_left else (child.p[site] @ x.T).T)
            for matrix, action, new_m in zip((z[i], plus[i]), actions, (m, m+2), strict=True):
                for new_q, transform in maps.items():
                    if (abs(new_m) > new_q or (new_m-new_q) % 2 or abs(new_q-q) > 2
                            or new_q+q < 2):
                        continue
                    entries = transform.conj().T @ operators[new_q].project(action, new_m)
                    for new_a, value in enumerate(entries):
                        matrix[positions[new_q, new_a, new_m], column] = value
    spin = tuple(((p+p.conj().T)/2, (p-p.conj().T)/(2j), zz)
                 for p, zz in zip(plus, z, strict=True))
    out = ReducedBlockData(left.n_sites+right.n_sites, widths,
                           {q: np.diag(energies[q]) for q in widths}, spin)
    validate_reduced_block(out, tolerance=tolerance)
    return out


@lru_cache(maxsize=16384)
def _boundary_recoupling(ql_out: int, qr_out: int, ql_in: int, qr_in: int,
                         q_out: int, q_in: int, side: int) -> float:
    """Exact rank-one recoupling factor in the declared reduced convention.

    Use highest magnetic states on both parent legs. This is a finite CG sum,
    independent of multiplicity widths and physical block sizes.
    """
    if side == 0 and qr_out != qr_in or side == 1 and ql_out != ql_in:
        return 0.
    mu = q_out-q_in
    denominator = clebsch_gordan(q_in, q_in, 2, mu, q_out, q_out)
    if denominator == 0.:
        raise ArithmeticError('rank-one parent CG coefficient unexpectedly vanished')
    value = 0.
    for ml_in in range(-ql_in, ql_in+1, 2):
        mr_in = q_in-ml_in
        if abs(mr_in) > qr_in:
            continue
        ml_out = ml_in+mu if side == 0 else ml_in
        mr_out = mr_in if side == 0 else mr_in+mu
        if abs(ml_out) > ql_out or abs(mr_out) > qr_out:
            continue
        rank_one = (clebsch_gordan(ql_in, ml_in, 2, mu, ql_out, ml_out) if side == 0
                    else clebsch_gordan(qr_in, mr_in, 2, mu, qr_out, mr_out))
        value += (clebsch_gordan(ql_out, ml_out, qr_out, mr_out, q_out, q_out)
                  *clebsch_gordan(ql_in, ml_in, qr_in, mr_in, q_in, q_in)*rank_one)
    return value/denominator


def _propagate_compact_boundary(operators: Mapping[int, SectorMergeOperator],
                                maps: Mapping[int, np.ndarray], energies: Mapping[int, np.ndarray],
                                left: ReducedBlockData, right: ReducedBlockData,
                                tolerance: float) -> ReducedBlockData:
    """Recouple reduced site vectors directly; never form magnetic D_out² arrays."""
    left = compact_reduced_block(left, tolerance=tolerance)
    right = compact_reduced_block(right, tolerance=tolerance)
    children = (left, right)
    widths = {q: w.shape[1] for q, w in maps.items()}
    pieces = {}
    for q, w in maps.items():
        pairs = {}
        cursor = 0
        for ql, dl in sorted(left.q_widths.items()):
            for qr, dr in sorted(right.q_widths.items()):
                if q not in coupling_range(ql, qr):
                    continue
                pairs[ql, qr] = w[cursor:cursor+dl*dr].reshape(dl, dr, w.shape[1])
                cursor += dl*dr
        if cursor != operators[q].dimension:
            raise ArithmeticError('reduced merge map is inconsistent with its channel order')
        pieces[q] = pairs
    sites = []
    for side, child in enumerate(children):
        source_keys = set().union(*(s.keys() for s in child.site_reduced_spin))
        sources = {key: np.stack([s.get(key, np.zeros((child.q_widths[key[0]], child.q_widths[key[1]])))
                                 for s in child.site_reduced_spin]) for key in source_keys}
        output = [{} for _ in range(child.n_sites)]
        for qo, do in widths.items():
            for qi, di in widths.items():
                if abs(qo-qi) > 2 or qo+qi < 2:
                    continue
                result = np.zeros((child.n_sites, do, di), complex)
                for (lo, ro), wo in pieces[qo].items():
                    for (li, ri), wi in pieces[qi].items():
                        if (side == 0 and ro != ri) or (side == 1 and lo != li):
                            continue
                        key = (lo, li) if side == 0 else (ro, ri)
                        reduced = sources.get(key)
                        if reduced is None:
                            continue
                        factor = _boundary_recoupling(lo, ro, li, ri, qo, qi, side)
                        if factor == 0.:
                            continue
                        if side == 0:
                            result += factor*np.einsum('abx,sac,cby->sxy', wo.conj(), reduced, wi, optimize=True)
                        else:
                            result += factor*np.einsum('abx,sbc,acy->sxy', wo.conj(), reduced, wi, optimize=True)
                for site, value in zip(output, result, strict=True):
                    site[qo, qi] = value
        sites.extend(output)
    result = ReducedBlockData(left.n_sites+right.n_sites, widths,
                              {q: np.diag(energies[q]) for q in widths},
                              site_reduced_spin=tuple(sites))
    validate_reduced_block(result, tolerance=tolerance)
    return result


def solve_sector_merge(left: ReducedBlockData, right: ReducedBlockData,
                        cross_bonds: Sequence[tuple[int, int, float]], *,
                        retain: Mapping[int, int], tol: float = 1e-9, maxiter: int = 1000,
                        seed: int = 0, propagate_operators: bool = True,
                        max_output_dimension: int = 128,
                        memory_limit_bytes: int = 512*1024**2,
                        dense_reference_limit: int = 64,
                        validation_tolerance: float = 2e-10,
                        compact_boundary: bool = False,
                        krylov_dimension: int | None = None) -> SectorMergeResult:
    """Find low multiplets in requested q sectors and optionally merge again.

    ARPACK residuals and matvec counts are reported for each q. The returned
    maps act only on reduced child channels. They do not assert that the full
    physical state has zero variance or has the global lattice A1 symmetry.
    ``compact_boundary=True`` stores exact rank-one multiplicity matrices and
    propagates them by CG recoupling, without dense magnetic output operators.
    ``krylov_dimension`` optionally controls ARPACK's ``ncv``. ``None`` keeps
    the existing solver default. A requested dimension is capped at each
    sector's size; it must exceed the computed eigenpair count by at least
    one for real arithmetic, or two for complex arithmetic. This option has
    no effect on the bounded dense-reference path. Memory estimates use the
    actual capped dimension, not the requested value.
    """
    tol = _tolerance(tol); validation_tolerance = _tolerance(validation_tolerance)
    _integer(maxiter, 'maxiter'); _integer(max_output_dimension, 'max_output_dimension')
    _integer(memory_limit_bytes, 'memory_limit_bytes'); _integer(seed, 'seed', 0)
    if krylov_dimension is not None:
        krylov_dimension = _integer(krylov_dimension, 'krylov_dimension', 2)
    dense_reference_limit = _integer(dense_reference_limit, 'dense_reference_limit', 0)
    if dense_reference_limit > 128:
        raise ValueError('dense reference fallback must stay at dimension 128 or below')
    if not retain:
        raise ValueError('explicit nonempty total-q retention is required')
    for q, d in retain.items():
        _integer(q, 'q', 0); _integer(d, 'retained multiplicity')
    output_dimension = sum((q+1)*d for q, d in retain.items())
    if propagate_operators and output_dimension > max_output_dimension:
        raise ValueError('propagated boundary dimension exceeds its explicit output bound')
    start = perf_counter()
    _precheck_memory(left, right, memory_limit_bytes)
    prepared = (_SparseBlock(left, validation_tolerance), _SparseBlock(right, validation_tolerance))
    cross_bonds = _checked_bonds(left, right, cross_bonds)
    _prepare_boundary_sites(prepared, cross_bonds, memory_limit_bytes,
                             all_sites=propagate_operators and not compact_boundary)
    # Unlike a single operator's incremental estimate, this reserve covers ALL
    # retained sectors and magnetic slices simultaneously, plus all output maps.
    dimensions = {q: sum(dl*dr for ql, dl in left.q_widths.items() for qr, dr in right.q_widths.items()
                         if q in coupling_range(ql, qr)) for q in retain}
    embedding_reserve = sum(_embedding_storage_bound(left, right, q, propagate_operators and not compact_boundary) for q in retain)
    map_reserve = sum(dimensions[q]*count*16 for q, count in retain.items())
    compact_entries = sum(do*di for qo, do in retain.items() for qi, di in retain.items()
                          if abs(qo-qi) <= 2 and qo+qi >= 2)
    output_reserve = (compact_entries*16*(2*(left.n_sites+right.n_sites)+4) if compact_boundary else
                      output_dimension**2*16*(6*(left.n_sites+right.n_sites)+12)) if propagate_operators else 0
    common_reserve = (left.dimension*right.dimension*16*8
        + sum(b.storage_bytes+b.input_storage_bytes for b in prepared)
        + max(left.dimension, right.dimension)**2*16*12
        + embedding_reserve + map_reserve + output_reserve)
    if common_reserve > memory_limit_bytes:
        raise MemoryError('combined sector caches/output workspace exceeds the explicit memory budget')
    peak_reserve = common_reserve
    operators = {}; energies = {}; maps = {}; residuals = {}; records = {}; warnings = []
    rng = np.random.default_rng(seed)
    for q, count in sorted(retain.items()):
        op = SectorMergeOperator(left, right, cross_bonds, q, memory_limit_bytes=memory_limit_bytes,
                                 tolerance=validation_tolerance, _prepared=prepared)
        operators[q] = op
        if count > op.dimension:
            raise ValueError('requested multiplicity exceeds the available total-q dimension')
        op.diagnostics['krylov_dimension_requested'] = krylov_dimension
        op.diagnostics['krylov_dimension'] = None
        op.diagnostics['computed_eigenpair_count'] = None
        sector_start = perf_counter()
        if op.dimension <= dense_reference_limit:
            estimate = common_reserve + op.dimension**2*16*8
            if estimate > memory_limit_bytes:
                raise MemoryError('dense reference workspace exceeds the explicit memory budget')
            peak_reserve = max(peak_reserve, estimate)
            dense = op.dense_reference(max_dimension=max(1, dense_reference_limit))
            values, vectors = np.linalg.eigh(dense)
            solver = 'bounded_dense_reference'
        else:
            if count >= op.dimension-1:
                raise ValueError('large sector requests must leave space for matrix-free eigsh')
            eigenvalue_limit = op.dimension-2 if op.dtype.kind == 'c' else op.dimension-1
            if count > eigenvalue_limit:
                raise ValueError('complex matrix-free ARPACK requires at least two unrequested dimensions')
            k = min(count+2, eigenvalue_limit)
            if k == count:
                warnings.append(f'q={q} has no extra computed eigenpair; retention-boundary degeneracy is unverified.')
            ncv = min(op.dimension, max(2*k+1, 20) if krylov_dimension is None
                      else krylov_dimension)
            minimum_ncv = k + (2 if op.dtype.kind == 'c' else 1)
            if ncv < minimum_ncv:
                raise ValueError(f'krylov_dimension is too small for q={q}: '
                                 f'ARPACK requires at least {minimum_ncv} for k={k}')
            op.diagnostics['krylov_dimension'] = ncv
            op.diagnostics['computed_eigenpair_count'] = k
            estimate = common_reserve + op.dimension*ncv*16*4 + op.dimension*k*16*8
            op.diagnostics['estimated_solve_workspace_bytes'] = estimate
            peak_reserve = max(peak_reserve, estimate)
            if estimate > memory_limit_bytes:
                raise MemoryError('ARPACK/output workspace exceeds the explicit memory budget')
            initial = rng.normal(size=op.dimension)
            if op.dtype.kind == 'c':
                initial = initial+1j*rng.normal(size=op.dimension)
            try:
                values, vectors = eigsh(op.linear_operator(), k=k, which='SA', tol=tol,
                                        maxiter=maxiter, ncv=ncv, v0=initial)
            except ArpackNoConvergence as error:
                raise SectorMergeSolveError('bounded matrix-free sector eigensolve did not converge',
                    dict(target_q=q, converged_eigenpairs=len(error.eigenvalues),
                         requested=count, matvec_count=op.matvec_count,
                         matvec_seconds=op.matvec_seconds, maxiter=maxiter,
                         krylov_dimension_requested=krylov_dimension,
                         krylov_dimension=ncv, computed_eigenpair_count=k,
                         estimated_solve_workspace_bytes=estimate)) from error
            order = np.argsort(values); values, vectors = values[order], vectors[:, order]
            solver = 'matrix_free_eigsh'
        canonical, groups = _canonical_eigenvectors(values, vectors, max(10*tol, validation_tolerance))
        if any(a < count < b for a, b in groups):
            warnings.append(f'q={q} retention cuts a degenerate energy subspace; individual retained axes are conventional.')
        w = canonical[:, :count]
        hw = np.column_stack([op.matvec(w[:, a]) for a in range(count)])
        e = np.real(np.einsum('ia,ia->a', w.conj(), hw))
        errors = np.linalg.norm(hw-w*e[None, :], axis=0)
        if np.max(errors) > max(100*tol, 1e-8)*max(1., float(np.max(abs(e)))):
            raise SectorMergeSolveError('retained eigenpair residual exceeds declared tolerance',
                dict(target_q=q, residuals=errors.tolist(), matvec_count=op.matvec_count,
                     matvec_seconds=op.matvec_seconds))
        energies[q] = e; maps[q] = w; residuals[q] = errors
        records[q] = dict(op.diagnostics, solver=solver, matvec_count=op.matvec_count,
                          matvec_seconds=op.matvec_seconds, total_seconds=perf_counter()-sector_start,
                          eigen_residual_max=float(np.max(errors)), degenerate_groups=groups)
    propagation_start = perf_counter()
    propagate = _propagate_compact_boundary if compact_boundary else _propagate_boundary
    block = (propagate(operators, maps, energies, left, right,
                       max(validation_tolerance, 1e-9)) if propagate_operators else None)
    warnings.append('Eigen-residuals are within the retained child space; full physical H variance and global A1 are not established.')
    diagnostics = dict(sectors=records, total_seconds=perf_counter()-start,
        boundary_propagation_seconds=perf_counter()-propagation_start,
        boundary_operators_propagated=propagate_operators, output_dimension=output_dimension,
        compact_boundary_operators=bool(compact_boundary and propagate_operators),
        krylov_dimension_requested=krylov_dimension,
        input_validation=[prepared[0].validation, prepared[1].validation],
        combined_embedding_reserve_bytes=embedding_reserve,
        retained_map_reserve_bytes=map_reserve, output_operator_reserve_bytes=output_reserve,
        estimated_total_peak_bytes=peak_reserve,
        memory_estimate_scope='resident input operators (dense or reduced) + sparse children + all sector/magnetic caches + retained maps + output/validation and solver workspace; excludes unrelated caller data and Python allocator/runtime overhead',
        physical_basis_constructed=False, dense_product_H_constructed=False)
    return SectorMergeResult(energies, maps, {q: op.channels for q, op in operators.items()},
                             residuals, block, diagnostics, tuple(warnings))


def project_sector_maps(left: ReducedBlockData, right: ReducedBlockData,
                         cross_bonds: Sequence[tuple[int, int, float]], *,
                         maps: Mapping[int, np.ndarray], compact_boundary: bool = True,
                         max_output_dimension: int = 128,
                         memory_limit_bytes: int = 512*1024**2,
                         tolerance: float = 2e-10) -> ReducedBlockData:
    """Project arbitrary orthonormal reduced maps, keeping full reduced H blocks.

    This is the same-width copy/random-control interface. Unlike an eigensolve,
    its H[q] generally has off-diagonal multiplicity entries; discarding those
    would change the physical Hamiltonian in all subsequent merges.
    """
    tol = _tolerance(tolerance)
    limit = _integer(memory_limit_bytes, 'memory_limit_bytes')
    output_limit = _integer(max_output_dimension, 'max_output_dimension')
    if not maps:
        raise ValueError('nonempty reduced maps are required')
    output_dimension = 0; normalized = {}
    for q, value in maps.items():
        _integer(q, 'q', 0)
        w = np.asarray(value)
        if w.ndim != 2 or not np.all(np.isfinite(w)) or not w.shape[1]:
            raise ValueError('each reduced map must be a finite nonempty matrix')
        output_dimension += (q+1)*w.shape[1]
        if output_dimension > output_limit:
            raise ValueError('propagated boundary dimension exceeds its explicit output bound')
        if np.max(abs(w.conj().T @ w-np.eye(w.shape[1]))) > tol:
            raise ValueError('reduced maps must have orthonormal multiplicity columns')
        normalized[q] = w
    _precheck_memory(left, right, limit)
    prepared = (_SparseBlock(left, tol), _SparseBlock(right, tol))
    bonds = _checked_bonds(left, right, cross_bonds)
    _prepare_boundary_sites(prepared, bonds, limit, all_sites=not compact_boundary)
    output_entries = sum(a.shape[1]*b.shape[1] for qo, a in normalized.items() for qi, b in normalized.items()
                         if abs(qo-qi) <= 2 and qo+qi >= 2)
    output_bytes = (output_entries*16*(2*(left.n_sites+right.n_sites)+4) if compact_boundary else
                    output_dimension**2*16*(6*(left.n_sites+right.n_sites)+12))
    estimate = (left.dimension*right.dimension*16*8 + max(left.dimension, right.dimension)**2*16*12
        + sum(b.storage_bytes+b.input_storage_bytes for b in prepared) + output_bytes
        + sum(_embedding_storage_bound(left, right, q, not compact_boundary) for q in normalized)
        + sum(w.nbytes*3 for w in normalized.values()))
    if estimate > limit:
        raise MemoryError('combined map-projection workspace exceeds the explicit memory budget')
    operators = {}; h_by_q = {}
    for q, w in normalized.items():
        op = SectorMergeOperator(left, right, bonds, q, memory_limit_bytes=limit,
                                 tolerance=tol, _prepared=prepared)
        if w.shape[0] != op.dimension:
            raise ValueError('map rows must equal the declared target-q channel dimension')
        hw = np.column_stack([op.matvec(w[:, a]) for a in range(w.shape[1])])
        h = w.conj().T @ hw
        _hermitian(h, h.shape, 'projected reduced H', tol)
        h_by_q[q] = (h+h.conj().T)/2
        operators[q] = op
    propagate = _propagate_compact_boundary if compact_boundary else _propagate_boundary
    zero_energies = {q: np.zeros(w.shape[1]) for q, w in normalized.items()}
    block = propagate(operators, normalized, zero_energies, left, right, tol)
    result = ReducedBlockData(block.n_sites, block.q_widths, h_by_q,
                              block.site_spin, block.site_reduced_spin)
    validate_reduced_block(result, tolerance=tol)
    return result
