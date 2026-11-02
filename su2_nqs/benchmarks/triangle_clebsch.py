"""Exact LOCAL nine-spin references for triangular ClebschTree initialization.

This is not a global ED solver or a ViT component. A local Heisenberg filter
acts on complete multiplicity spaces before the declared finite-width map.
All spin labels are doubled integers; physical bit zero means down.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from ..lattices.triangular_j1j2 import triangular_j1j2_bonds
from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.triangle_blocks import triangle_block_tables, real_c3_axes


def nine_site_hamiltonians(length: int, j2_over_j1: float) -> tuple[dict, ...]:
    """Return each actual cover block's H in COMPLETE fixed-q multiplicities.

    The physical reference is H_ij=J*(P_ij/2-I/4). No magnetic component is
    optimized, and no local invariant/singlet projection is imposed.
    """
    if isinstance(j2_over_j1, bool) or not isfinite(j2_over_j1):
        raise ValueError('finite real J2/J1 required')
    tree = triangular_schur_hierarchy(length)
    bonds = triangular_j1j2_bonds(length, length, j2=j2_over_j1, periodic=True)
    data = triangle_block_tables()
    bits = (np.arange(512)[:, None] >> np.arange(8, -1, -1)) & 1
    magnetic = (2*bits-1).sum(1)
    powers = 1 << np.arange(8, -1, -1)
    result = []
    for block in tree.blocks:
        ids = {s:i for i,s in enumerate(block.sites)}
        internal = [(ids[a], ids[b], float(j)) for a,b,j in bonds if a in ids and b in ids]
        h = np.zeros((512, 512), dtype=np.float64)
        for a,b,j in internal:
            swapped = bits.copy(); swapped[:, [a,b]] = bits[:, [b,a]]
            h[swapped @ powers, np.arange(512)] += .5*j
            h[np.arange(512), np.arange(512)] -= .25*j
        reduced = {}
        for q,u in data['nine'].items():
            high = u*(magnetic == q)[:, None]
            reduced[q] = high.T @ h @ high
        result.append(dict(sites=block.sites, internal_bonds=internal, reduced=reduced))
    return tuple(result)


def local_filter_maps(length: int, j2_over_j1: float, beta: float) -> tuple[dict, ...]:
    """Return exp[-beta*(H_block-Emin)] in each doubled-q block.

    beta is a preregistered INITIALIZATION scale, not a learned parameter.
    A common energy shift across q preserves relative spin-sector weights.
    The operation preserves SU(2), real time reversal and physical local C3.
    """
    if isinstance(beta, bool) or not isfinite(beta) or beta < 0:
        raise ValueError('finite nonnegative local initialization beta required')
    result = []
    for block in nine_site_hamiltonians(length, j2_over_j1):
        eigen = {q:np.linalg.eigh(h) for q,h in block['reduced'].items()}
        ground = min(float(e[0][0]) for e in eigen.values())
        maps = {q:(v*np.exp(-beta*(e-ground))) @ v.T for q,(e,v) in eigen.items()}
        result.append(maps)
    return tuple(result)


def local_spectrum_summary(length: int, j2_over_j1: float) -> list[dict]:
    """Resolve the isolated-block spectrum by doubled spin and C3 irrep.

    E is a two-dimensional REAL rotation irrep, not an assumption about a
    bulk chiral phase. These isolated blocks omit inter-block bonds.
    """
    rotations = triangle_block_tables()['rotations']
    result = []
    for block in nine_site_hamiltonians(length, j2_over_j1):
        sectors = {}
        for q,h in block['reduced'].items():
            r = rotations[q]; a,e = real_c3_axes(r)
            ea = np.linalg.eigvalsh(a.T @ h @ a)
            ee = np.linalg.eigvalsh(e.T @ h @ e) if e.shape[1] else np.array([])
            sectors[q] = dict(A_energies=ea.tolist(), E_pair_energies=ee[::2].tolist(),
                E_partner_splitting_max=float(np.max(abs(ee[::2]-ee[1::2]))) if len(ee) else 0.,
                commutator_max=float(np.max(abs(h @ r-r @ h))))
        result.append(dict(sites=block['sites'], internal_bonds=block['internal_bonds'],
                           sectors=sectors, isolated_block_not_bulk=True))
    return result
