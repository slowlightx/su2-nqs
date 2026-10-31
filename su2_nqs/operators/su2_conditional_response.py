"""Coherent conditional boundary response of one raw SU(2) tree cut.

The saved state has topology ``((A,B)C,D)0``. Its actual root and C tensors
are recoupled to ``(A,(B,D))0``; a local RDM square root is not substituted
for that purification. One-site compact operators are exact only inside
their retained input/output spaces. Consequently this module computes the
response projected into the retained A-child-product and B/D bath spaces,
NOT the full physical bath leakage or a Reynolds-projected A1 state.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds

from ..su2.quantum_numbers import coupling_range, validate_q
from ..su2.wigner6j import f_move
from .su2_sector_merge import ReducedBlockData, _boundary_recoupling, compact_reduced_block

Channels = tuple[tuple[int, int, int, int], ...]


def _channels(left: Mapping[int, int], right: Mapping[int, int], q: int) -> Channels:
    return tuple((ql, a, qr, b) for ql, dl in sorted(left.items())
                 for qr, dr in sorted(right.items()) if q in coupling_range(ql, qr)
                 for a in range(dl) for b in range(dr))


def _pieces(left: Mapping[int, int], right: Mapping[int, int], q: int,
            matrix: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    out = {}; cursor = 0
    for ql, dl in sorted(left.items()):
        for qr, dr in sorted(right.items()):
            if q in coupling_range(ql, qr):
                out[ql, qr] = matrix[cursor:cursor+dl*dr].reshape(dl, dr, -1)
                cursor += dl*dr
    if cursor != matrix.shape[0]:
        raise ValueError("map rows disagree with deterministic coupled-channel order")
    return out


def _checked_maps(left: Mapping[int, int], right: Mapping[int, int],
                  maps: Mapping[int, np.ndarray], tolerance: float) -> dict[int, np.ndarray]:
    result = {}
    for q, matrix in maps.items():
        validate_q(q)
        value = np.asarray(matrix, dtype=np.complex128)
        if (value.ndim != 2 or value.shape[1] < 1
                or value.shape[0] != len(_channels(left, right, q))
                or not np.all(np.isfinite(value))):
            raise ValueError("finite nonempty reduced map with legal doubled-q rows required")
        if np.max(abs(value.conj().T @ value-np.eye(value.shape[1]))) > tolerance:
            raise ValueError("conditional response requires canonical isometric child maps")
        result[q] = value
    if not result:
        raise ValueError("at least one reduced map required")
    return result


def root_coefficient_matrices(root_map: np.ndarray, root_channels: Channels) -> dict[int, np.ndarray]:
    """Unpack normalized-root singlet coefficients (no magnetic degeneracy).

    ``root_channels`` uses existing ``(q_left,a,q_right,b)`` ordering and
    ``root_map`` has one singlet output. The caller's isometric child bases
    make these ordinary coefficient matrices, not RDMs or Schmidt square roots.
    """
    value = np.asarray(root_map)
    if value.shape != (len(root_channels), 1) or not np.all(np.isfinite(value)):
        raise ValueError("finite single-output root map required")
    result = {}
    for q, a, qr, b in root_channels:
        validate_q(q); validate_q(qr)
        if (q != qr or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0
                           for x in (a, b))):
            raise ValueError("singlet root requires equal doubled child spins")
        dl = 1+max(x[1] for x in root_channels if x[0] == q)
        dr = 1+max(x[3] for x in root_channels if x[0] == q)
        result.setdefault(q, np.zeros((dl, dr), complex))
    seen = set()
    for coefficient, (q, a, _, b) in zip(value[:, 0], root_channels, strict=True):
        if (q, a, b) in seen:
            raise ValueError("duplicate root channel")
        seen.add((q, a, b)); result[q][a, b] = coefficient
    if len(seen) != sum(v.size for v in result.values()):
        raise ValueError("root channels must include every multiplicity pair")
    return result


def _cut_coefficients(a: ReducedBlockData, b: ReducedBlockData, d: ReducedBlockData,
                      parent_maps: Mapping[int, np.ndarray], root: Mapping[int, np.ndarray],
                      tolerance: float) -> dict[int, np.ndarray]:
    pieces = {p: _pieces(a.q_widths, b.q_widths, p, w) for p, w in parent_maps.items()}
    gamma = {}
    for p, coefficient in root.items():
        validate_q(p)
        v = np.asarray(coefficient, complex)
        if (p not in parent_maps or p not in d.q_widths
                or v.shape != (parent_maps[p].shape[1], d.q_widths[p])
                or not np.all(np.isfinite(v))):
            raise ValueError("root coefficients do not match retained parent/environment widths")
        gamma[p] = v
    norm = sum(float(np.vdot(v, v).real) for v in gamma.values())
    if not np.isfinite(norm) or abs(norm-1.) > tolerance:
        raise ValueError("normalized canonical raw-root coefficients required")
    result = {}
    for qa, da in sorted(a.q_widths.items()):
        columns = []
        for qb, db in sorted(b.q_widths.items()):
            for p, dd in sorted(d.q_widths.items()):
                if qa not in coupling_range(qb, p):
                    continue
                block = np.zeros((da, db, dd), complex)
                if p in gamma and (qa, qb) in pieces[p]:
                    f, left_channels, right_channels = f_move(qa, qb, p, 0)
                    phase = f[right_channels.index(qa), left_channels.index(p)]
                    block = phase*np.einsum("abc,cd->abd", pieces[p][qa, qb], gamma[p], optimize=True)
                columns.append(block.reshape(da, -1))
        result[qa] = np.concatenate(columns, axis=1) if columns else np.zeros((da, 0), complex)
    error = abs(sum(float(np.vdot(c, c).real) for c in result.values())-1.)
    if error > tolerance:
        raise ArithmeticError("singlet recoupling failed raw normalization")
    return result


def _premerge_vector(left: ReducedBlockData, right: ReducedBlockData,
                     source_maps: Mapping[int, np.ndarray], site: int,
                     qo: int, qi: int) -> np.ndarray:
    """Reduced site vector from retained source to untruncated child product."""
    source = _pieces(left.q_widths, right.q_widths, qi, source_maps[qi])
    side = int(site >= left.n_sites)
    child = right if side else left
    local = site-left.n_sites if side else site
    vector = child.site_reduced_spin[local]
    rows = []
    for lo, dl in sorted(left.q_widths.items()):
        for ro, dr in sorted(right.q_widths.items()):
            if qo not in coupling_range(lo, ro):
                continue
            value = np.zeros((dl, dr, source_maps[qi].shape[1]), complex)
            for (li, ri), w in source.items():
                if (not side and ro != ri) or (side and lo != li):
                    continue
                key = (ro, ri) if side else (lo, li)
                r = vector.get(key)
                if r is None:
                    continue
                factor = _boundary_recoupling(lo, ro, li, ri, qo, qi, side)
                if side:
                    value += factor*np.einsum("ob,aby->aoy", r, w, optimize=True)
                else:
                    value += factor*np.einsum("oa,aby->oby", r, w, optimize=True)
            rows.append(value.reshape(dl*dr, -1))
    return np.concatenate(rows, axis=0)


def _bath_action(c: np.ndarray, b: ReducedBlockData, d: ReducedBlockData,
                 qo: int, qi: int, weighted: tuple[dict, dict]) -> np.ndarray:
    # C has A-source rows. Reinterpret its columns as coupled B/D channels.
    source = {key: v.transpose(2, 0, 1) for key, v in
              _pieces(b.q_widths, d.q_widths, qi, c.T).items()}
    rows = []
    for lo, dl in sorted(b.q_widths.items()):
        for ro, dr in sorted(d.q_widths.items()):
            if qo not in coupling_range(lo, ro):
                continue
            value = np.zeros((c.shape[0], dl, dr), complex)
            for (li, ri), w in source.items():
                if ro == ri and (lo, li) in weighted[0]:
                    f = _boundary_recoupling(lo, ro, li, ri, qo, qi, 0)
                    value += f*np.einsum("ob,abd->aod", weighted[0][lo, li], w, optimize=True)
                if lo == li and (ro, ri) in weighted[1]:
                    f = _boundary_recoupling(lo, ro, li, ri, qo, qi, 1)
                    value += f*np.einsum("od,abd->abo", weighted[1][ro, ri], w, optimize=True)
            rows.append(value.reshape(c.shape[0], -1))
    return np.concatenate(rows, axis=1)


@dataclass(frozen=True)
class ConditionalResponseFactor:
    """Thin-product factor Y with multiplicity density Y Y^dagger.

    No factor ``q+1`` is applied to this density: each coefficient multiplies
    a normalized A/environment magnetic singlet. Its eigenvalues already
    include all magnetic components. ``terms`` are A @ Z, never a dense H.
    """
    target_q: int
    target_channels: Channels
    bath_channels: Channels
    terms: tuple[tuple[np.ndarray, np.ndarray], ...]
    source_coefficients: Mapping[int, np.ndarray]
    diagnostics: Mapping[str, object]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.target_channels), len(self.bath_channels)

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply Y to one bath-channel vector without materializing Y."""
        v = np.asarray(vector)
        if v.shape != (self.shape[1],) or not np.all(np.isfinite(v)):
            raise ValueError("finite bath vector of the declared dimension required")
        return sum((a @ (z @ v) for a, z in self.terms), np.zeros(self.shape[0], complex))

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply Y^dagger in exactly the same coupled coordinate frame."""
        v = np.asarray(vector)
        if v.shape != (self.shape[0],) or not np.all(np.isfinite(v)):
            raise ValueError("finite target vector of the declared dimension required")
        return sum((z.conj().T @ (a.conj().T @ v) for a, z in self.terms),
                   np.zeros(self.shape[1], complex))

    def as_linear_operator(self) -> LinearOperator:
        """Return a complex128 SciPy operator for bounded thin SVD."""
        return LinearOperator(self.shape,
            matvec=lambda v: self.matvec(np.asarray(v).reshape(-1)),
            rmatvec=lambda v: self.rmatvec(np.asarray(v).reshape(-1)),
            matmat=self.matmat, rmatmat=self.rmatmat, dtype=np.complex128)

    def matmat(self, vectors: np.ndarray) -> np.ndarray:
        """Apply Y to a batch of bath vectors, retaining the same thin factors."""
        v = np.asarray(vectors)
        if v.ndim != 2 or v.shape[0] != self.shape[1] or not np.all(np.isfinite(v)):
            raise ValueError("finite bath-column matrix of the declared dimension required")
        return sum((a @ (z @ v) for a, z in self.terms), np.zeros((self.shape[0], v.shape[1]), complex))

    def rmatmat(self, vectors: np.ndarray) -> np.ndarray:
        """Apply Y^dagger to a batch of target vectors."""
        v = np.asarray(vectors)
        if v.ndim != 2 or v.shape[0] != self.shape[0] or not np.all(np.isfinite(v)):
            raise ValueError("finite target-column matrix of the declared dimension required")
        return sum((z.conj().T @ (a.conj().T @ v) for a, z in self.terms),
                   np.zeros((self.shape[1], v.shape[1]), complex))

    def to_dense(self, *, maximum_entries: int = 2_000_000) -> np.ndarray:
        """Materialize only Y, under an explicit entry bound, never Y Y^dagger."""
        if (isinstance(maximum_entries, bool) or not isinstance(maximum_entries, Integral)
                or maximum_entries < 1 or np.prod(self.shape) > maximum_entries):
            raise MemoryError("response-factor materialization exceeds the explicit bound")
        return sum((a @ z for a, z in self.terms), np.zeros(self.shape, complex))


def make_conditional_cut_response(*, a_children: tuple[ReducedBlockData, ReducedBlockData],
        a_source: ReducedBlockData, a_maps: Mapping[int, np.ndarray], b: ReducedBlockData,
        d: ReducedBlockData, parent_maps: Mapping[int, np.ndarray],
        root_coefficients: Mapping[int, np.ndarray],
        cross_bonds_ab: Sequence[tuple[int, int, float]],
        cross_bonds_ad: Sequence[tuple[int, int, float]], target_q: int = 2,
        center: bool = True, memory_limit_bytes: int = 512*1024**2,
        tolerance: float = 2e-10) -> ConditionalResponseFactor:
    """Build the entangled conditional response of the actual saved RAW state.

    A is represented by its two retained children; its target sector is NOT
    projected back into the old A width. B/D outputs ARE retained, an explicit
    controlled scope limitation. All incoming ``q=target_q-2,target_q,+2``
    terms, sites, Schmidt columns and both boundary branches sum coherently.
    When ``center=True``, subtract the exact retained-space cut expectation
    times the saved state; this is not a compressed-H-squared variance. The
    A source supplies layout metadata only: its cached site operators are
    deliberately NOT used. The mean is reconstructed from the same child
    operators and maps that define Y, avoiding an inconsistent cached gauge.
    ``memory_limit_bytes`` bounds retained thin-factor allocations, not total
    process RSS, caller-owned inputs, validation or SVD workspaces.
    """
    started = perf_counter(); validate_q(target_q)
    if (not isinstance(center, bool) or isinstance(memory_limit_bytes, bool)
            or not isinstance(memory_limit_bytes, Integral) or memory_limit_bytes < 1
            or not np.isfinite(tolerance) or not 0 < tolerance < 1e-6):
        raise ValueError("valid center flag, positive memory budget and small tolerance required")
    left, right = (compact_reduced_block(x, tolerance=tolerance) for x in a_children)
    a, b, d = (compact_reduced_block(x, tolerance=tolerance) for x in (a_source, b, d))
    if left.n_sites+right.n_sites != a.n_sites:
        raise ValueError("A source does not match its child site count")
    maps = _checked_maps(left.q_widths, right.q_widths, a_maps, tolerance)
    if {q: w.shape[1] for q, w in maps.items()} != dict(a.q_widths):
        raise ValueError("A source widths do not match the source maps")
    parents = _checked_maps(a.q_widths, b.q_widths, parent_maps, tolerance)
    coefficients = _cut_coefficients(a, b, d, parents, root_coefficients, tolerance)
    target_channels = _channels(left.q_widths, right.q_widths, target_q)
    bath_channels = _channels(b.q_widths, d.q_widths, target_q)
    if not target_channels or not bath_channels:
        raise ValueError("target doubled-q absent from A product or B/D complement")
    # Aggregate all bonds before contraction, preserving their coherent sum.
    weights: dict[int, tuple[dict, dict]] = {}
    for side, (bath, bonds) in enumerate(((b, cross_bonds_ab), (d, cross_bonds_ad))):
        for i, j, strength in bonds:
            if (isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < a.n_sites
                    or isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < bath.n_sites
                    or np.iscomplexobj(strength) or not np.isfinite(strength)):
                raise ValueError("cut bonds require local site indices and finite real strengths")
            pair = weights.setdefault(int(i), ({}, {}))
            for key, value in bath.site_reduced_spin[j].items():
                pair[side][key] = pair[side].get(key, 0.) + float(strength)*value
    source_qs = tuple(q for q in maps if abs(q-target_q) <= 2 and q+target_q >= 2
                      and (q-target_q) % 2 == 0)
    # Reserve factor terms before allocation. Complex128 is used even for real
    # states. A larger active width should use matvecs, not a dense target H.
    reserve = sum(len(weights)*maps[q].shape[1]*(len(target_channels)+len(bath_channels))*16
                  for q in source_qs)
    reserve += sum(c.nbytes for c in coefficients.values())
    reserve += sum(v.nbytes for pair in weights.values() for row in pair for v in row.values())
    if reserve > memory_limit_bytes:
        raise MemoryError("conditional response thin factors exceed the explicit memory budget")
    terms = []
    for qi in source_qs:
        factor = -np.sqrt((target_q+1)/(qi+1))
        for site, weighted in weights.items():
            ra = _premerge_vector(left, right, maps, site, target_q, qi)
            z = _bath_action(coefficients[qi], b, d, target_q, qi, weighted)
            terms.append((factor*ra, z))
    expectation = 0j
    if center:
        for qo, co in coefficients.items():
            for qi, ci in coefficients.items():
                if abs(qo-qi) > 2 or qo+qi < 2:
                    continue
                factor = -np.sqrt((qo+1)/(qi+1))
                for site, weighted in weights.items():
                    premerge = _premerge_vector(left, right, maps, site, qo, qi)
                    ra = maps[qo].conj().T @ premerge
                    z = _bath_action(ci, b, d, qo, qi, weighted)
                    expectation += factor*np.vdot(co, ra @ z)
        if abs(expectation.imag) > tolerance*max(1., abs(expectation.real)):
            raise ArithmeticError("real boundary Hamiltonian acquired a complex expectation")
        if target_q in coefficients:
            terms.append((-float(expectation.real)*maps[target_q], coefficients[target_q]))
    actual_bytes = sum(a.nbytes+z.nbytes for a, z in terms)
    if actual_bytes > memory_limit_bytes:
        raise MemoryError("conditional response factors exceeded their runtime memory budget")
    diagnostics = dict(target_q=target_q, coherent_source_q=list(source_qs),
        source_q_weights={q: float(np.vdot(c, c).real) for q, c in coefficients.items()},
        boundary_inside_sites=sorted(weights), center=center,
        cutmean_source="same A child operators and maps as the response; cached A-source site operators ignored",
        cut_expectation_real=float(expectation.real) if center else None,
        cut_expectation_imaginary=float(expectation.imag) if center else None,
        retained_factor_bytes=actual_bytes, estimated_factor_reserve_bytes=reserve,
        memory_budget_scope="stored thin factors; excludes caller-owned inputs, validation/centering temporaries and later SVD workspace",
        dense_factor_bytes=len(target_channels)*len(bath_channels)*16,
        factor_terms=len(terms), seconds=perf_counter()-started,
        scope="P_A-child-product tensor P_B tensor P_D times (Hcut-mean) psi" if center else
              "P_A-child-product tensor P_B tensor P_D times Hcut psi",
        physical_bath_leakage_included=False, complete_A1=False,
        magnetic_density_factor_applied=False, full_physical_variance=False,
        physical_2_to_N_basis_constructed=False, dense_H_constructed=False)
    return ConditionalResponseFactor(target_q, target_channels, bath_channels,
                                     tuple(terms), coefficients, diagnostics)


@dataclass(frozen=True)
class NestedResponseExpansion:
    """A strictly nested multiplicity basis; every old column is protected."""
    target_q: int
    frame: np.ndarray
    added_singular_values: np.ndarray
    diagnostics: Mapping[str, object]


def nested_response_expansion(target_q: int, old_frame: np.ndarray,
        response: ConditionalResponseFactor, *, additional: int = 2, seed: int = 0,
        tolerance: float = 1e-10, maximum_iterations: int = 300) -> NestedResponseExpansion:
    """Append nonzero conditional-response directions without rotating old ones.

    No occupied direction is removed, so the original RAW state is feasible
    after exact zero-extension of its ancestors. This is an initialization
    guarantee, not an energy win or a statement about subsequent optimization.
    """
    validate_q(target_q)
    old = np.asarray(old_frame, complex)
    if (target_q != response.target_q or old.ndim != 2 or old.shape[1] < 1 or old.shape[0] != response.shape[0]
            or not np.all(np.isfinite(old)) or isinstance(additional, bool)
            or not isinstance(additional, Integral) or additional < 1
            or old.shape[1]+additional > old.shape[0]
            or additional >= min(response.shape) or isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, Integral) or maximum_iterations < 1 or not np.isfinite(tolerance)
            or not 0 < tolerance < 1e-6):
        raise ValueError("legal same-q source frame and bounded positive expansion required")
    if np.max(abs(old.conj().T @ old-np.eye(old.shape[1]))) > tolerance:
        raise ValueError("old frame must be canonical before nested expansion")
    def project(v):
        return v-old @ (old.conj().T @ v)
    real_coordinates = not np.any(old.imag) and all(not np.any(x.imag) for term in response.terms for x in term)
    def checked_action(v, adjoint=False):
        x = np.asarray(v).reshape(-1)
        y = response.rmatvec(project(x)) if adjoint else project(response.matvec(x))
        if real_coordinates:
            if np.any(y.imag):
                raise ArithmeticError("declared exact-real factors produced complex action")
            return y.real
        return y
    op = LinearOperator(response.shape, dtype=np.float64 if real_coordinates else np.complex128,
        matvec=checked_action, rmatvec=lambda v: checked_action(v, adjoint=True))
    if np.prod(response.shape) <= 262144:
        dense = project(response.to_dense(maximum_entries=262144))
        u, values, _ = np.linalg.svd(dense.real if real_coordinates else dense, full_matrices=False)
        u, values = u[:, :additional], values[:additional]
    else:
        u, values, _ = svds(op, k=additional, which="LM", tol=tolerance,
                            maxiter=maximum_iterations, random_state=seed)
        order = np.argsort(values)[::-1]; u, values = u[:, order], values[order]
    if len(values) != additional or np.min(values) <= tolerance*max(1., float(np.max(values))):
        raise ValueError("not enough nonzero new response directions at the fixed rank")
    new, _ = np.linalg.qr(project(project(u)), mode="reduced")
    for column in range(new.shape[1]):
        pivot = int(np.argmax(abs(new[:, column])))
        if real_coordinates:
            new[:, column] *= -1 if new[pivot, column].real < 0 else 1
        else:
            new[:, column] *= np.exp(-1j*np.angle(new[pivot, column]))
    frame = np.column_stack((old, new))
    error = float(np.max(abs(frame.conj().T @ frame-np.eye(frame.shape[1]))))
    if error > 10*tolerance or not np.array_equal(frame[:, :old.shape[1]], old):
        raise ArithmeticError("nested response expansion lost the protected old subspace")
    return NestedResponseExpansion(target_q, frame, values,
        dict(old_width=old.shape[1], new_width=frame.shape[1], additional=additional,
             real_singular_solver_used=real_coordinates,
             Gram_error=error, old_columns_bitwise_preserved=True,
             exact_old_state_feasible_after_ancestor_zero_extension=True,
             old_state_energy_and_variance_unchanged_before_updates=True,
             projected_A1_quality_improvement_not_established=True))
