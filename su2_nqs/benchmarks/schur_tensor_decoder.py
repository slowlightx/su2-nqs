"""Exact-small linear decoder from local Schur tensors and fixed CG merges.

No spin/H projector or eigensolver. This prepares numeric coefficients outside
JAX; all learned nonlinearities belong to the separate reduced-label model.
"""
from __future__ import annotations

import numpy as np
from ..su2.local_schur import cell_schur_spec, cell_schur_tensor, CellKind
from ..su2.local_schur_fusion import four_cell_schur_states
from ..models.spin_clebsch_tree import spin_cg_tensor


def four_cell_tensor_decoder_rows(
    bits: np.ndarray, kind: CellKind, cells: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """D[physical row, reduced configuration] for four complete Schur cells.

    Computational 0=down,1=up, site0 MSB. A fixed physical row fixes EVERY
    subtree magnetic projection; multiply the local Schur entries and three
    CG tensors in ((cell0,cell1),(cell2,cell3))->q=0. No magnetic leg is omitted.
    q and m are doubled integers. Enumeration is validation-only.
    """
    spec = cell_schur_spec(kind); size = spec.cell_size
    cells = tuple(tuple(cell) for cell in cells)
    if (len(cells) != 4 or any(len(c) != size for c in cells)
            or sorted(site for c in cells for site in c) != list(range(4*size))):
        raise ValueError('four complete, disjoint cells must cover sites exactly')
    b = np.asarray(bits)
    if b.ndim != 2 or b.shape[1] != 4*size or not np.isin(b, (0, 1)).all():
        raise ValueError('expected computational bit rows with site0 most significant')
    states = four_cell_schur_states(kind)
    q = np.asarray([s.cell_q for s in states]); a = np.asarray([s.cell_a for s in states])
    pair = np.asarray([s.pair_q for s in states])
    schur = cell_schur_tensor(kind); cg = spin_cg_tensor(size, size)
    local = b[:, np.asarray(cells)]
    m = np.sum(2*local.astype(int)-1, axis=-1)
    packed = local@(1 << np.arange(size-1, -1, -1))
    values = np.ones((len(b), len(states)))
    for i in range(4):
        values *= schur[packed[:, i, None], q[None, :, i], m[:, i, None]+size, a[None, :, i]]
    values *= cg[q[None, :, 0], m[:, 0, None]+size, q[None, :, 1], m[:, 1, None]+size, pair[None]]
    values *= cg[q[None, :, 2], m[:, 2, None]+size, q[None, :, 3], m[:, 3, None]+size, pair[None]]
    left_m = m[:, 0]+m[:, 1]
    # Exact Condon--Shortley root CG(q,m;q,-m|0,0). Other rows vanish.
    allowed = ((m.sum(axis=1)[:, None] == 0) & (abs(left_m[:, None]) <= pair[None])
               & ((pair[None]-left_m[:, None]) % 2 == 0))
    root = (-1.)**((pair[None]-left_m[:, None])//2)/np.sqrt(pair[None]+1.)
    return values*np.where(allowed, root, 0.)


__all__ = ['four_cell_tensor_decoder_rows']
