"""Bounded offline SU(2) block transfer and Hamiltonian-informed merging.

This dense reference constructs open block multiplets, never a new production
wavefunction or a replacement for full-H VMC. All charges/projections are doubled
integers. A block has ``U[q][physical, multiplicity, m_index]`` with ascending
``m=-q,-q+2,...,q`` and site zero the most significant physical bit.

Projected first moments are exact within the chosen product subspace. Squaring
the projected Hamiltonian generally does NOT give the physical second moment.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.quantum_numbers import coupling_range


@dataclass(frozen=True)
class BlockIsometry:
    """Computational-basis open SU(2) multiplets; ``q=2j`` throughout."""

    n_sites: int
    sectors: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class ProjectedBlock:
    """An orthonormal block, its scalar H and every local vector operator.

    ``labels`` orders the retained columns as ``(q,a,m)``. ``site_spin[i]``
    contains Cartesian (Sx,Sy,Sz), including transitions between q sectors.
    """

    isometry: BlockIsometry
    basis: np.ndarray
    labels: tuple[tuple[int, int, int], ...]
    hamiltonian: np.ndarray
    site_spin: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    diagnostics: Mapping[str, float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MergeResult:
    """Low-energy complete multiplets and reproducible reduced-space maps.

    Each source channel is ``(q_left,a_left,q_right,a_right)``. Columns of
    ``reduced_maps[q]`` are shared by ALL magnetic components. Degenerate
    energy projectors include whole clusters, including discarded clusters.
    """

    block: ProjectedBlock
    energies: Mapping[int, np.ndarray]
    reduced_maps: Mapping[int, np.ndarray]
    channels: Mapping[int, tuple[tuple[int, int, int, int], ...]]
    retained_projectors: Mapping[int, np.ndarray]
    energy_projectors: Mapping[int, tuple[np.ndarray, ...]]
    diagnostics: Mapping[str, float]
    warnings: tuple[str, ...]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _valid_tolerance(tolerance: float) -> float:
    if (isinstance(tolerance, (bool, complex)) or not np.isscalar(tolerance)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('tolerance must be finite and positive')
    return float(tolerance)


def _apply_spin(vectors: np.ndarray, n_sites: int, site: int,
                component: str) -> np.ndarray:
    rows = np.arange(1 << n_sites)
    mask = 1 << (n_sites - 1 - site)
    up = (rows & mask) != 0
    if component == 'z':
        return (up.astype(float) - .5)[:, None] * vectors
    raising = np.zeros_like(vectors, dtype=np.complex128)
    raising[rows[~up] | mask] = vectors[~up]
    if component == '+':
        return raising
    lowering = np.zeros_like(raising)
    lowering[rows[up] ^ mask] = vectors[up]
    if component == '-':
        return lowering
    return (raising + lowering) / 2 if component == 'x' else (raising - lowering) / (2j)


def _validate_multiplets(block: BlockIsometry, *, orthonormal: bool,
                         tolerance: float = 2e-10) -> dict[str, float]:
    n = _positive_integer(block.n_sites, 'n_sites')
    if n > 12:
        raise ValueError('offline block reference is bounded to at most 12 physical sites')
    if not block.sectors:
        raise ValueError('at least one nonempty doubled-q sector is required')
    magnetization = np.asarray([2 * i.bit_count() - n for i in range(1 << n)])
    magnetic_error = 0.
    ladder_error = 0.
    for q, values in block.sectors.items():
        if (isinstance(q, bool) or not isinstance(q, Integral) or q < 0
                or q > n or (q - n) % 2):
            raise ValueError('sector keys must be legal doubled integer spins for this block')
        u = np.asarray(values, dtype=np.complex128)
        if (u.ndim != 3 or u.shape[0] != 1 << n or u.shape[1] < 1
                or u.shape[2] != q + 1 or not np.all(np.isfinite(u))):
            raise ValueError('U[q] must have finite shape (2**n, d_q, q+1)')
        scale = float(np.linalg.norm(u))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError('each multiplet sector must have finite nonzero norm')
        for k, m in enumerate(range(-int(q), int(q) + 1, 2)):
            magnetic_error = max(magnetic_error, float(np.linalg.norm(
                (magnetization - m)[:, None] * u[:, :, k])) / scale)
            raised = sum((_apply_spin(u[:, :, k], n, i, '+') for i in range(n)),
                         np.zeros_like(u[:, :, k]))
            expected = (np.sqrt((q - m) * (q + m + 2)) / 2 * u[:, :, k + 1]
                        if k < q else np.zeros_like(raised))
            ladder_error = max(ladder_error, float(np.linalg.norm(raised - expected)) / scale)
    if max(magnetic_error, ladder_error) > tolerance:
        raise ValueError('input magnetic columns are not common Condon-Shortley SU(2) multiplets')
    if sum(np.asarray(u).shape[1] * (q + 1) for q, u in block.sectors.items()) > 1024:
        raise ValueError('offline retained-basis reference dimension exceeds 1024')
    basis, _ = _flatten(block)
    gram_error = float(np.max(abs(basis.conj().T @ basis - np.eye(basis.shape[1]))))
    if orthonormal and gram_error > tolerance:
        raise ValueError('block columns must be orthonormal; canonicalize reduced maps first')
    return dict(magnetic_error=magnetic_error, ladder_error=ladder_error, gram_error=gram_error)


def _flatten(block: BlockIsometry) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    columns = []
    labels = []
    for q in sorted(block.sectors):
        u = np.asarray(block.sectors[q], dtype=np.complex128)
        for a in range(u.shape[1]):
            for k, m in enumerate(range(-q, q + 1, 2)):
                columns.append(u[:, a, k])
                labels.append((int(q), a, m))
    return np.column_stack(columns), tuple(labels)


def canonicalize_block(n_sites: int, sectors: Mapping[int, np.ndarray], *,
                       tolerance: float = 2e-10
                       ) -> tuple[BlockIsometry, dict[int, np.ndarray], dict[str, float]]:
    """QR arbitrary full-rank reduced maps, using ONE gauge for every m.

    Returns ``(block, R_by_q, diagnostics)`` with ``old_U[m] = new_U[m] @ R``.
    No magnetic component is independently normalized or truncated.
    """
    tolerance = _valid_tolerance(tolerance)
    original = BlockIsometry(n_sites, sectors)
    before = _validate_multiplets(original, orthonormal=False, tolerance=tolerance)
    canonical = {}
    gauges = {}
    replay = 0.
    for q in sorted(sectors):
        u = np.asarray(sectors[q], dtype=np.complex128)
        _, r = np.linalg.qr(u[:, :, -1], mode='reduced')
        if (r.shape[0] != r.shape[1] or np.linalg.matrix_rank(r, tol=1e-13 *
                float(np.linalg.norm(r, 2))) != r.shape[0]):
            raise ValueError('reduced maps must have full column rank')
        phase = np.diag(r) / np.abs(np.diag(r))
        r = phase.conj()[:, None] * r
        values = np.stack([np.linalg.solve(r.T, u[:, :, k].T).T
                           for k in range(q + 1)], axis=-1)
        replay = max(replay, float(np.max(abs(np.einsum('pak,ab->pbk', values, r) - u))))
        canonical[int(q)] = values
        gauges[int(q)] = r
    block = BlockIsometry(int(n_sites), canonical)
    after = _validate_multiplets(block, orthonormal=True, tolerance=tolerance)
    return block, gauges, dict(after, original_gram_error=before['gram_error'], replay_error=replay)


def _scalar_blocks(matrix: np.ndarray, labels: tuple[tuple[int, int, int], ...], *,
                   tolerance: float = 2e-10) -> tuple[dict[int, np.ndarray], float]:
    by_q = {}
    reconstructed = np.zeros_like(matrix)
    positions = {label: i for i, label in enumerate(labels)}
    for q in sorted({x[0] for x in labels}):
        d = max(a for charge, a, _ in labels if charge == q) + 1
        arrays = [matrix[np.ix_([positions[q, a, m] for a in range(d)],
                                [positions[q, a, m] for a in range(d)])]
                  for m in range(-q, q + 1, 2)]
        reduced = sum(arrays) / (q + 1)
        by_q[q] = (reduced + reduced.conj().T) / 2
        for m in range(-q, q + 1, 2):
            ids = [positions[q, a, m] for a in range(d)]
            reconstructed[np.ix_(ids, ids)] = by_q[q]
    error = float(np.max(abs(matrix - reconstructed)))
    if error > tolerance * max(1., float(np.max(abs(matrix)))):
        raise ValueError('Hamiltonian is not SU(2) scalar in the retained multiplets')
    return by_q, error


def _validate_projected_block(block: ProjectedBlock, tolerance: float) -> None:
    """Reject stale/manually inconsistent public projected-block containers.

    Local spin matrices must be the physical-site projections, not merely
    arbitrary rank-one operators with the same SU(2) covariance. This bounded
    oracle checks their meaning directly; it does not require their truncated
    onsite commutators to close an SU(2) algebra.
    """
    _validate_multiplets(block.isometry, orthonormal=True, tolerance=tolerance)
    expected, labels = _flatten(block.isometry)
    if (block.labels != labels or np.shape(block.basis) != expected.shape
            or not np.allclose(block.basis, expected, atol=tolerance, rtol=0)):
        raise ValueError('projected basis/labels do not match the block isometry')
    d = len(labels)
    h = np.asarray(block.hamiltonian)
    if (h.shape != (d, d) or not np.all(np.isfinite(h))
            or not np.allclose(h, h.conj().T, atol=tolerance, rtol=0)):
        raise ValueError('projected H must be finite and Hermitian with the correct dimension')
    _scalar_blocks(h, labels, tolerance=tolerance)
    if len(block.site_spin) != block.isometry.n_sites:
        raise ValueError('one physical-site vector operator is required for every block site')
    for i, operators in enumerate(block.site_spin):
        if len(operators) != 3:
            raise ValueError('site spin must include Cartesian x, y, z components')
        for component, operator in zip(('x', 'y', 'z'), operators, strict=True):
            s = np.asarray(operator)
            if (s.shape != (d, d) or not np.all(np.isfinite(s))
                    or not np.allclose(s, s.conj().T, atol=tolerance, rtol=0)):
                raise ValueError('site spin matrices must be finite and Hermitian with the correct dimension')
            physical = expected.conj().T @ _apply_spin(expected, block.isometry.n_sites, i, component)
            if not np.allclose(s, physical, atol=tolerance, rtol=0):
                raise ValueError('site spin matrix does not equal its declared physical-site projection')


def project_block(block: BlockIsometry, hamiltonian: np.ndarray, *,
                  max_dimension: int = 1024, tolerance: float = 2e-10) -> ProjectedBlock:
    """Project a bounded physical H and all site-spin components offline.

    The reported ``physical_H2_minus_projected_H_squared_norm`` measures the
    exact omitted-space second-moment contribution for THIS input H.
    """
    tolerance = _valid_tolerance(tolerance)
    n = _positive_integer(block.n_sites, 'n_sites')
    dimension = 1 << n
    if dimension > min(_positive_integer(max_dimension, 'max_dimension'), 1024):
        raise ValueError('dense physical-H projection is limited to dimension 1024')
    diagnostics = _validate_multiplets(block, orthonormal=True, tolerance=tolerance)
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if (h.shape != (dimension, dimension) or not np.all(np.isfinite(h))
            or not np.allclose(h, h.conj().T, atol=tolerance, rtol=0)):
        raise ValueError('finite Hermitian physical Hamiltonian required')
    # PHP can be accidentally scalar in a singlet-only subspace even when H
    # itself breaks SU(2). Check the supplied physical H before projection.
    scalar_defect = 0.
    for component, adjoint in (('z', 'z'), ('+', '-')):
        acted_left = sum((_apply_spin(h, block.n_sites, i, component)
                          for i in range(block.n_sites)), np.zeros_like(h))
        acted_right = sum((_apply_spin(h.conj().T, block.n_sites, i, adjoint).conj().T
                           for i in range(block.n_sites)), np.zeros_like(h))
        scalar_defect = max(scalar_defect, float(np.max(abs(acted_left - acted_right))))
    if scalar_defect > tolerance * max(1., float(np.max(abs(h)))):
        raise ValueError('physical Hamiltonian is not SU(2) scalar before projection')
    basis, labels = _flatten(block)
    hb = h @ basis
    effective = basis.conj().T @ hb
    _, scalar_error = _scalar_blocks(effective, labels, tolerance=tolerance)
    spin = tuple(tuple(basis.conj().T @ _apply_spin(basis, block.n_sites, i, c)
                       for c in ('x', 'y', 'z')) for i in range(block.n_sites))
    leakage = hb - basis @ effective
    diagnostics.update(scalar_error=scalar_error, physical_scalar_error=scalar_defect,
                       H_leakage_norm=float(np.linalg.norm(leakage)),
                       physical_H2_minus_projected_H_squared_norm=float(np.linalg.norm(
                           leakage.conj().T @ leakage)))
    warnings = (() if basis.shape[1] == dimension else (
        'Truncated block: (P H P)^2 is not P H^2 P; use the physical H action for variance.',))
    return ProjectedBlock(block, basis, labels, effective, spin, diagnostics, warnings)


def _projector_basis(projector: np.ndarray, rank: int) -> np.ndarray:
    columns = []
    for i in range(len(projector)):
        value = projector[:, i].copy()
        for _ in range(2):
            for old in columns:
                value -= old * np.vdot(old, value)
        norm = np.linalg.norm(value)
        if norm < 1e-9:
            continue
        value /= norm
        pivot = int(np.argmax(abs(value)))
        value *= np.conj(value[pivot]) / abs(value[pivot])
        columns.append(value)
        if len(columns) == rank:
            break
    if len(columns) != rank:
        raise ArithmeticError('degenerate eigenspace projector lost numerical rank')
    return np.column_stack(columns)


def _deterministic_eigh(h: np.ndarray, tolerance: float
                       ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[int, ...]]:
    energies, vectors = np.linalg.eigh(h)
    groups = []
    start = 0
    while start < len(energies):
        stop = start + 1
        while (stop < len(energies) and abs(energies[stop] - energies[start]) <=
               tolerance * max(1., abs(float(energies[start])))):
            stop += 1
        groups.append((start, stop))
        start = stop
    projectors = tuple(vectors[:, a:b] @ vectors[:, a:b].conj().T for a, b in groups)
    canonical = np.column_stack([_projector_basis(p, b - a)
                                 for p, (a, b) in zip(projectors, groups, strict=True)])
    values = np.real(np.diag(canonical.conj().T @ h @ canonical))
    return values, canonical, projectors, tuple(b for _, b in groups)


def merge_projected_blocks(left: ProjectedBlock, right: ProjectedBlock,
                           cross_bonds: Sequence[tuple[int, int, float]], *,
                           retain: Mapping[int, int], max_effective_dimension: int = 512,
                           max_physical_sites: int = 12, tolerance: float = 2e-10
                           ) -> MergeResult:
    """Merge using fixed CG and actual interblock Heisenberg bonds.

    All retained charge sectors include their entire magnetic multiplet. The
    ``omitted_child_product_H_norm`` only measures loss inside the child product
    space; it can vanish while the full physical energy variance is nonzero.
    This bounded dense reference rejects production-size tensor products BEFORE
    allocating them. ``retain`` is explicit, so low-energy selection cannot
    silently remove all q>0 channels of a future internal edge.
    """
    tolerance = _valid_tolerance(tolerance)
    dl, dr = len(left.labels), len(right.labels)
    n = (_positive_integer(left.isometry.n_sites, 'left.n_sites')
         + _positive_integer(right.isometry.n_sites, 'right.n_sites'))
    if dl * dr > min(_positive_integer(max_effective_dimension, 'max_effective_dimension'), 512):
        raise ValueError('dense product effective space exceeds the bounded reference limit 512')
    if n > min(_positive_integer(max_physical_sites, 'max_physical_sites'), 12):
        raise ValueError('merged computational basis exceeds the bounded reference limit 12 sites')
    if not retain:
        raise ValueError('explicit nonempty doubled-q retention is required')
    _validate_projected_block(left, tolerance)
    _validate_projected_block(right, tolerance)
    h = np.kron(left.hamiltonian, np.eye(dr)) + np.kron(np.eye(dl), right.hamiltonian)
    for i, j, strength in cross_bonds:
        if (isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < left.isometry.n_sites
                or isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < right.isometry.n_sites
                or not np.isfinite(strength) or np.iscomplexobj(strength)):
            raise ValueError('cross bonds use valid local site indices and finite real strength')
        h += float(strength) * sum((np.kron(a, b) for a, b in
                                    zip(left.site_spin[i], right.site_spin[j], strict=True)),
                                   np.zeros_like(h))
    channels = {}
    for ql, ul in sorted(left.isometry.sectors.items()):
        for qr, ur in sorted(right.isometry.sectors.items()):
            for q in coupling_range(ql, qr):
                channels.setdefault(q, []).extend((ql, a, qr, b)
                    for a in range(ul.shape[1]) for b in range(ur.shape[1]))
    channels = {int(q): tuple(rows) for q, rows in sorted(channels.items())}
    lp, rp = ({v: i for i, v in enumerate(x.labels)} for x in (left, right))
    columns = []
    labels = []
    for q, rows in channels.items():
        for a, (ql, al, qr, ar) in enumerate(rows):
            for m in range(-q, q + 1, 2):
                column = np.zeros(dl * dr, dtype=np.complex128)
                for ml in range(-ql, ql + 1, 2):
                    mr = m - ml
                    if abs(mr) <= qr and (qr - mr) % 2 == 0:
                        column[lp[ql, al, ml] * dr + rp[qr, ar, mr]] = clebsch_gordan(
                            ql, ml, qr, mr, q, m)
                columns.append(column)
                labels.append((q, a, m))
    coupling = np.column_stack(columns)
    cg_error = float(np.max(abs(coupling.conj().T @ coupling - np.eye(dl * dr))))
    if cg_error > tolerance:
        raise ArithmeticError('fixed CG product-to-coupled transform is not unitary')
    reduced, scalar_error = _scalar_blocks(coupling.conj().T @ h @ coupling,
                                            tuple(labels), tolerance=tolerance)
    positions = {v: i for i, v in enumerate(labels)}
    energies, maps, projectors, energy_projectors = {}, {}, {}, {}
    warnings = list(left.warnings + right.warnings)
    transform_columns = []
    output_labels = []
    for q in sorted(retain):
        if isinstance(q, bool) or not isinstance(q, Integral) or q not in reduced:
            raise ValueError('requested doubled-q sector is unavailable in the child spaces')
        count = _positive_integer(retain[q], 'retained multiplicity')
        if count > len(reduced[q]):
            raise ValueError('retained multiplicity exceeds the available q space')
        values, vectors, clusters, ends = _deterministic_eigh(reduced[q], tolerance)
        maps[q] = vectors[:, :count]
        energies[q] = values[:count]
        projectors[q] = maps[q] @ maps[q].conj().T
        energy_projectors[q] = clusters
        if count not in ends:
            warnings.append(f'q={q} retention cuts a degenerate energy subspace; its chosen basis is conventional.')
        for a in range(count):
            for m in range(-q, q + 1, 2):
                ids = [positions[q, k, m] for k in range(len(reduced[q]))]
                transform_columns.append(coupling[:, ids] @ maps[q][:, a])
                output_labels.append((int(q), a, m))
    transform = np.column_stack(transform_columns)
    basis = np.kron(left.basis, right.basis) @ transform
    sectors = {}
    offset = 0
    for q in sorted(retain):
        count = int(retain[q])
        sectors[int(q)] = basis[:, offset:offset + count * (q + 1)].reshape(1 << n, count, q + 1)
        offset += count * (q + 1)
    block = BlockIsometry(n, sectors)
    diagnostics = _validate_multiplets(block, orthonormal=True, tolerance=tolerance)
    effective = transform.conj().T @ h @ transform
    spin = tuple(tuple(transform.conj().T @ np.kron(s, np.eye(dr)) @ transform
                       for s in site) for site in left.site_spin) + tuple(
        tuple(transform.conj().T @ np.kron(np.eye(dl), s) @ transform
              for s in site) for site in right.site_spin)
    omitted = h @ transform - transform @ effective
    diagnostics.update(cg_unitarity_error=cg_error, scalar_error=scalar_error,
                       effective_product_dimension=float(dl * dr),
                       omitted_child_product_H_norm=float(np.linalg.norm(omitted)))
    warnings.append('Merged H is P H P: its squared matrix omits discarded-space physical H^2 contributions.')
    warnings = tuple(dict.fromkeys(warnings))
    projected = ProjectedBlock(block, basis, tuple(output_labels), effective, spin, diagnostics, warnings)
    return MergeResult(projected, energies, maps, channels, projectors, energy_projectors,
                       diagnostics, warnings)
