"""Physical region readouts from fixed-support amplitude-query caches.

This estimator applies to ANY fixed-M wavefunction, not only SU(2) states.
It includes off-diagonal coherences by evaluating all region configurations
with the same local doubled projection. Configuration frequencies alone are
not a reduced density matrix. No PSD repair, twirling or spin-sector repair
is performed, and cached energy-audit rows do not become a new audit.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def region_completion_rows(rows: np.ndarray, sites: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return unique full configurations and [record,region-index] query IDs.

    Site zero is the most significant bit. A query ID of -1 denotes an EXACT
    fixed-M selection-rule zero, never an empirical amplitude floor. Region
    sizes are deliberately bounded to eight spins for this local workflow.
    """
    rows = np.asarray(rows)
    if (rows.ndim != 2 or not len(rows) or not np.issubdtype(rows.dtype, np.integer)
            or not np.isin(rows, (0, 1)).all() or not np.all(rows.sum(1) == rows[0].sum())):
        raise ValueError('nonempty integer bit configurations in one fixed doubled-M sector required')
    if (not 1 <= len(sites) <= 8 or len(set(sites)) != len(sites)
            or any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s < rows.shape[1] for s in sites)):
        raise ValueError('one to eight distinct physical sites in explicit order required')
    size = 1 << len(sites)
    local = ((np.arange(size)[:, None] >> np.arange(len(sites)-1, -1, -1)) & 1).astype(rows.dtype)
    lookup = np.full((len(rows), size), -1, np.int64)
    chunks = []; ids = []; offset = 0
    for i, row in enumerate(rows):
        allowed = np.flatnonzero(local.sum(1) == row[list(sites)].sum())
        expanded = np.repeat(row[None], len(allowed), axis=0)
        expanded[:, list(sites)] = local[allowed]
        chunks.append(expanded); ids.append((i, allowed, offset)); offset += len(allowed)
    unique, inverse = np.unique(np.concatenate(chunks), axis=0, return_inverse=True)
    for i, allowed, offset in ids:
        lookup[i, allowed] = inverse[offset:offset+len(allowed)]
    return unique, lookup


def reconstruct_region_queries(values: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """Expand unique amplitudes, retaining exact fixed-M selection-rule zeros."""
    values, lookup = np.asarray(values), np.asarray(lookup)
    if (values.ndim != 1 or lookup.ndim != 2 or not np.issubdtype(lookup.dtype, np.integer)
            or not np.isfinite(values).all() or np.any(lookup < -1) or np.any(lookup >= len(values))):
        raise ValueError('finite unique amplitudes and valid query IDs required')
    result = np.zeros(lookup.shape, dtype=values.dtype)
    mask = lookup >= 0
    result[mask] = values[lookup[mask]]
    return result


def cached_region_readout(
    rows: np.ndarray, sites: tuple[int, ...], amplitudes: np.ndarray,
    density: np.ndarray, operators: dict[str, np.ndarray], *, groups: int = 32,
) -> tuple[dict, np.ndarray]:
    """Node-safe self-normalized importance readout with delete-group SEs.

    ``amplitudes[i,b]`` is psi(b, environment_i), on ONE common amplitude
    scale; ``density[i]`` is the exact original full-configuration proposal.
    Every original record is retained, including exact target nodes. No H
    products enter. The returned RDM is explicitly Hermitian but may be
    statistically nonpositive; its eigenvalues are NOT a certified spectrum.
    Equal contiguous groups must comprise independent IID records.
    """
    rows = np.asarray(rows); phi = np.asarray(amplitudes, np.complex128)
    q = np.asarray(density)
    if (rows.ndim != 2 or not np.issubdtype(rows.dtype, np.integer) or not np.isin(rows, (0, 1)).all()
            or not 1 <= len(sites) <= 8 or len(set(sites)) != len(sites)
            or any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s < rows.shape[1] for s in sites)
            or phi.shape != (len(rows), 1 << len(sites)) or not np.isfinite(phi).all()
            or q.shape != (len(rows),) or np.iscomplexobj(q) or not np.isfinite(q).all() or np.any(q <= 0)
            or isinstance(groups, bool) or not isinstance(groups, Integral) or groups < 4
            or len(rows) % groups):
        raise ValueError('finite region amplitudes, positive fixed proposal and >=4 equal IID groups required')
    packed = rows[:, sites] @ (1 << np.arange(len(sites)-1, -1, -1))
    parent = phi[np.arange(len(rows)), packed]
    mass = abs(parent)**2 / q
    if mass.sum() <= 0:
        raise ValueError('sampled target norm is zero')
    names = tuple(operators); numerators = []
    for name in names:
        op = np.asarray(operators[name])
        if op.shape != (phi.shape[1], phi.shape[1]) or not np.isfinite(op).all():
            raise ValueError('finite physical matrices in explicit region order required')
        action = np.einsum('ij,ij->i', op[packed], phi)
        numerators.append(parent.conj() * action / q)
    values = np.stack(numerators, axis=1)
    point = values.sum(0) / mass.sum()
    ids = np.repeat(np.arange(groups), len(rows)//groups)
    if any(mass[ids != g].sum() <= 0 for g in range(groups)):
        raise FloatingPointError('a delete-group target norm is zero; no finite jackknife is defined')
    leave = np.array([values[ids != g].sum(0) / mass[ids != g].sum() for g in range(groups)])
    se = np.sqrt((groups-1)/groups * np.sum(abs(leave-leave.mean(0))**2, axis=0))
    raw = np.zeros((phi.shape[1], phi.shape[1]), complex)
    np.add.at(raw, packed, parent[:, None] * phi.conj() / q[:, None])
    raw /= mass.sum(); hermitian = (raw+raw.conj().T)/2
    return dict(records=len(rows), independent_groups=int(groups),
        observables={name:dict(real=float(point[i].real), imaginary=float(point[i].imag),
            group_jackknife_SE=float(se[i])) for i, name in enumerate(names)},
        normalized_RDM_trace_real=float(np.trace(hermitian).real),
        RDM_antihermitian_frobenius=float(np.linalg.norm(raw-raw.conj().T)),
        minimum_Hermitian_RDM_eigenvalue=float(np.linalg.eigvalsh(hermitian)[0]),
        norm_effective_contributions=float(mass.sum()**2/(mass@mass)),
        maximum_norm_record_fraction=float(mass.max()/mass.sum()),
        exact_target_nodes=int(np.count_nonzero(parent == 0)),
        PSD_repair=False, SU2_twirl=False, physical_RDM_eigenspectrum_certified=False,
        cached_rows_not_new_independent_audit=True), hermitian
