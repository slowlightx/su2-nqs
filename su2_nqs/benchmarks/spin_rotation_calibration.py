"""Exact symmetry-readout controls, not a finite-size QSL classifier.

The positive *family* is the published kagome nearest-neighbour RVB PEPS
(Schuch et al., PRB 86, 115108; Poilblanc et al., PRB 86, 014404).
Its tiny periodic representatives only validate amplitudes and symmetry.
Neither four winding labels nor a partial-rotation zero certifies topology.
All angular momenta/projections below are doubled integer labels.
"""
from __future__ import annotations

from itertools import permutations
from math import comb
from numbers import Integral
from typing import Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan


def kagome_rvb_local_tensors() -> dict[str, np.ndarray]:
    """Published simplex/site tensors in virtual q=1 plus q=0 space.

    Virtual indices 0/1 are physical down/up, index 2 is the q=0 vacuum.
    Triangle legs (A,B,C) are counterclockwise; oriented singlets run
    clockwise A->C->B->A, agreeing with Condon--Shortley CG phases.
    P[s,a,b] = <s|P|a,b>.  These are unnormalized PEPS tensors, not a state.
    """
    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    for p in permutations(range(3)):
        inversions = sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
        epsilon[p] = (-1)**inversions
    epsilon[2, 2, 2] = 1
    site = np.zeros((2, 3, 3), dtype=np.complex128)
    for s in (0, 1):
        site[s, s, 2] = site[s, 2, s] = 1
    return dict(epsilon=epsilon, site=site,
                virtual_q=np.array([1, 1, 0], dtype=np.int32),
                virtual_m=np.array([-1, 1, 0], dtype=np.int32),
                J=np.diag([-1., -1., 1.]))


def kagome_clockwise_edges(rows: int, columns: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return directed (source,target,row-cut,column-cut) NN torus edges.

    Site index is 3*(row*columns+column)+sublattice (A=0,B=1,C=2).
    The two cut parities count crossings of a chosen torus seam modulo two.
    They are covering winding labels, not orthogonal physical sectors.
    Only 12/18-site exact calibration geometries are admitted here.
    """
    if (any(isinstance(x, bool) or not isinstance(x, Integral) for x in (rows, columns))
            or rows < 2 or columns < 2 or 3*rows*columns not in (12, 18)):
        raise ValueError('bounded kagome torus requires 12 or 18 sites, dimensions >=2')
    def index(r: int, c: int, s: int) -> int:
        return 3*((r % rows)*columns+c % columns)+s
    edges = []
    def append(r: int, c: int, s: int, dr: int, dc: int, t: int) -> None:
        r, c = r % rows, c % columns
        edges.append((index(r, c, s), index(r+dr, c+dc, t),
                      ((r+dr)//rows) % 2, ((c+dc)//columns) % 2))
    for r in range(rows):
        for c in range(columns):
            # Up triangle A(r,c),B(r,c),C(r,c), clockwise arrows.
            append(r, c, 0, 0, 0, 2)
            append(r, c, 2, 0, 0, 1)
            append(r, c, 1, 0, 0, 0)
            # Down triangle A(r,c),B(r,c-1),C(r-1,c).
            append(r, c, 0, -1, 0, 2)
            append(r-1, c, 2, 1, -1, 1)
            append(r, c-1, 1, 0, 1, 0)
    if len({tuple(sorted(e[:2])) for e in edges}) != len(edges):
        raise RuntimeError('geometry has duplicate physical bonds')
    return tuple(edges)


def aklt_local_tensors() -> dict[str, np.ndarray]:
    """AKLT spin-1 SPT false positive with genuine virtual half-spin edges.

    Two virtual q=1 irreps are symmetrized into physical q=2 with CS CG;
    adjacent virtual legs couple to q=0 on a singlet bond.  The half-spin
    virtual SO3 projective sign matches a spinon, but this 1D Haldane SPT
    is NOT intrinsic 2D topological order. This is a tensor identity control,
    not an AKLT bulk state used as a triangle Hamiltonian competitor.
    """
    site = np.zeros((3,2,2),dtype=np.complex128)
    bond = np.zeros((2,2),dtype=np.complex128)
    for a,ma in enumerate((-1,1)):
        for b,mb in enumerate((-1,1)):
            bond[a,b] = clebsch_gordan(1,ma,1,mb,0,0)
            for c,m in enumerate((-2,0,2)):
                site[c,a,b] = clebsch_gordan(1,ma,1,mb,2,m)
    return dict(site=site,singlet_bond=bond,
                physical_q=np.full(3,2,dtype=np.int32),
                physical_m=np.array([-2,0,2],dtype=np.int32),
                virtual_q=np.array([1,1],dtype=np.int32),
                virtual_m=np.array([-1,1],dtype=np.int32))


def _perfect_matchings(n_sites: int, edges: Sequence[tuple[int, int, int, int]]) -> tuple[tuple[int, ...], ...]:
    adjacency = [[] for _ in range(n_sites)]
    for e, (i, j, _, _) in enumerate(edges):
        adjacency[i].append((j, e)); adjacency[j].append((i, e))
    result = []
    def visit(mask: int, chosen: tuple[int, ...]) -> None:
        if not mask:
            result.append(chosen); return
        i = (mask & -mask).bit_length()-1
        for j, edge in sorted(adjacency[i]):
            if mask & (1 << j):
                visit(mask ^ (1 << i) ^ (1 << j), chosen+(edge,))
    visit((1 << n_sites)-1, ())
    return tuple(result)


def singlet_covering_state(n_sites: int, directed_pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Normalized product of directed CS singlets, bit 0 down, site 0 MSB.

    Each pair is q=1 x q=1 -> q=0.  Every physical site must occur once.
    This short-range-entangled state is an explicit non-QSL control.
    """
    if (isinstance(n_sites, bool) or not isinstance(n_sites, Integral)
            or not 2 <= n_sites <= 18 or n_sites % 2):
        raise ValueError('even n_sites between 2 and 18 required')
    pairs = tuple(tuple(p) for p in directed_pairs)
    flat = [i for p in pairs for i in p]
    if (any(len(p) != 2 for p in pairs)
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in flat)
            or sorted(flat) != list(range(n_sites))):
        raise ValueError('directed pairs must cover every site exactly once')
    out = np.zeros(2**n_sites, dtype=np.complex128)
    for packed_pairs in range(2**len(pairs)):
        packed, amplitude = 0, 1.
        for a, (i, j) in enumerate(pairs):
            bit = (packed_pairs >> a) & 1
            packed |= bit << (n_sites-1-i)
            packed |= (1-bit) << (n_sites-1-j)
            amplitude *= clebsch_gordan(1, 2*bit-1, 1, 1-2*bit, 0, 0)
        out[packed] = amplitude
    return out


def kagome_rvb_reference(rows: int = 2, columns: int = 2) -> dict[str, object]:
    """Exact clockwise NN-RVB covers grouped by four winding parities.

    Physical singlet-cover states are nonorthogonal. Return their actual
    normalized Gram matrix and also the raw sum (each covering normalized).
    The known SET family assignment is external literature evidence; this
    routine makes NO bulk-QSL/cusp claim from these tiny representatives.
    """
    edges = kagome_clockwise_edges(rows, columns)
    n = 3*rows*columns
    covers = _perfect_matchings(n, edges)
    raw = np.zeros((2**n, 4), dtype=np.complex128)
    counts = np.zeros(4, dtype=np.int32)
    for cover in covers:
        wr = sum(edges[e][2] for e in cover) % 2
        wc = sum(edges[e][3] for e in cover) % 2
        sector = 2*wr+wc
        raw[:, sector] += singlet_covering_state(n, [edges[e][:2] for e in cover])
        counts[sector] += 1
    norms = np.linalg.norm(raw, axis=0)
    if np.min(norms) <= 1e-13:
        raise RuntimeError('a winding-covering sum vanished')
    states = raw/norms
    total_raw = raw.sum(axis=1)
    total = total_raw/np.linalg.norm(total_raw)
    return dict(n_sites=n, rows=rows, columns=columns, edges=edges, covers=covers,
                covering_counts=counts, raw_sector_states=raw,
                states=states, gram=states.conj().T@states,
                total_state=total, total_raw_state=total_raw,
                finite_representative=True, intrinsic_topology_certified=False,
                QSL_claim=False, cusp_claim=False)


def pure_state_rdm(state: np.ndarray, sites: Sequence[int]) -> np.ndarray:
    """Actual partial trace in explicit ordered site list; no symmetry repair."""
    state = np.asarray(state, dtype=np.complex128)
    sites = tuple(sites)
    if (state.ndim != 1 or len(state) < 2 or len(state) & (len(state)-1)
            or not np.isfinite(state).all() or abs(np.vdot(state, state)-1) > 1e-10):
        raise ValueError('normalized finite power-of-two pure state required')
    n = len(state).bit_length()-1
    if (not sites or len(set(sites)) != len(sites)
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in sites)
            or min(sites) < 0 or max(sites) >= n):
        raise ValueError('distinct valid sites required')
    order = sites+tuple(i for i in range(n) if i not in sites)
    amplitudes = state.reshape((2,)*n).transpose(order).reshape(2**len(sites), -1)
    return amplitudes@amplitudes.conj().T


def manifold_region_algebra(states: np.ndarray, sites: Sequence[int]) -> dict[str, object]:
    """Test the COMPLETE local operator algebra within a finite state span.

    Columns of states need not be orthogonal. After actual Gram whitening,
    every transition RDM is compared with delta_ab times the average RDM.
    A zero defect is equivalent to P O_A P = c(O_A) P for every local O_A.
    This is a local-error-detection test, not by itself intrinsic topology.
    Diagonal-only comparisons or equal expectations of chosen singlet
    observables could hide off-diagonal local distinguishability.
    """
    states = np.asarray(states, dtype=np.complex128)
    sites = tuple(sites)
    if (states.ndim != 2 or not 1 <= states.shape[1] <= 8
            or states.shape[0] < 2 or states.shape[0] & (states.shape[0]-1)
            or not np.isfinite(states).all()):
        raise ValueError('finite power-of-two state columns required')
    n = states.shape[0].bit_length()-1
    if (not 1 <= len(sites) <= 6 or len(set(sites)) != len(sites)
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in sites)
            or min(sites) < 0 or max(sites) >= n):
        raise ValueError('distinct valid local region of 1..6 sites required')
    gram = states.conj().T@states
    eigenvalues, columns = np.linalg.eigh(gram)
    if np.min(eigenvalues) <= 1e-12:
        raise ValueError('state span must have nonsingular actual Gram matrix')
    orthogonal = states@(columns/np.sqrt(eigenvalues))
    order = sites+tuple(i for i in range(n) if i not in sites)
    matrices = [orthogonal[:,a].reshape((2,)*n).transpose(order).reshape(2**len(sites),-1)
                for a in range(states.shape[1])]
    transitions = np.array([[a@b.conj().T for b in matrices] for a in matrices])
    average = sum(transitions[a,a] for a in range(len(matrices)))/len(matrices)
    defect = transitions-np.eye(len(matrices))[:,:,None,None]*average
    return dict(gram=gram, gram_eigenvalues=eigenvalues,
                orthonormality_error=float(np.linalg.norm(orthogonal.conj().T@orthogonal-np.eye(len(matrices)))),
                transition_rdms=transitions, average_rdm=average,
                local_algebra_defect_frobenius=float(np.linalg.norm(defect)),
                maximum_transition_defect=float(np.max(np.linalg.norm(defect,axis=(-2,-1)))),
                diagonal_only=False, all_local_operator_matrix_units_tested=True,
                QSL_claim=False)


def lieb_mattis_singlet(n_sites: int, sublattice: Sequence[int]) -> np.ndarray:
    """SU2-restored ordered negative: maximal A/B spins coupled to singlet.

    Each equal sublattice has doubled spin q=n_sites/2.  This is the exact
    singlet projection of a two-sublattice Neel product, not a QSL; its
    same-sublattice pair correlation is +1/4 at arbitrary distance.
    """
    if (isinstance(n_sites, bool) or not isinstance(n_sites, Integral)
            or not 2 <= n_sites <= 18 or n_sites % 2):
        raise ValueError('even n_sites between 2 and 18 required')
    a = tuple(sublattice)
    if (len(a) != n_sites//2 or len(set(a)) != len(a)
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in a)
            or min(a) < 0 or max(a) >= n_sites):
        raise ValueError('valid equal-size sublattice required')
    bits = (np.arange(2**n_sites)[:, None] >> np.arange(n_sites-1, -1, -1)) & 1
    n_a = bits[:, a].sum(axis=1)
    n_total = bits.sum(axis=1)
    q = n_sites//2
    amplitudes = np.zeros(2**n_sites, dtype=np.complex128)
    for up in range(q+1):
        m = 2*up-q
        amplitudes[(n_total == q) & (n_a == up)] = (
            clebsch_gordan(q, m, q, -m, 0, 0)/comb(q, up))
    return amplitudes


def boundary_even_charge_rotation(length: int, vacuum_weight: float, angles: np.ndarray) -> dict[str, object]:
    """Analytic q=1 plus q=0 even-charge boundary MECHANISM calibration.

    rho_L is the normalized parity-even projection of tensor-product local
    rho = p|vac><vac| + (1-p)I_spin/2, J=(-1,-1,+1). Then W(2pi)=J
    and the trace is (lambda_plus**L+lambda_minus**L)/(1+(2p-1)**L).
    For p>1/2 all finite curves stay positive and smooth; the large-L rate
    has a finite-height pi cusp from dominant-branch switching.
    This boundary toy has NO independently defined 2D bulk. Dimers/SPT/cats
    can share virtual properties: it is expressly NOT a known-QSL control.
    Its 2pi center is even virtual q=1 count, NOT physical-region site parity.
    """
    angles = np.asarray(angles)
    if (isinstance(length, bool) or not isinstance(length, Integral) or length < 1
            or isinstance(vacuum_weight, bool) or not np.isfinite(vacuum_weight)
            or not .5 < vacuum_weight < 1 or angles.ndim != 1 or not len(angles)
            or angles.dtype.kind not in 'fiu' or not np.isfinite(angles).all()):
        raise ValueError('positive integer L, 1/2<p<1 and finite real angle vector required')
    p = float(vacuum_weight)
    plus = p+(1-p)*np.cos(angles/2)
    minus = p-(1-p)*np.cos(angles/2)
    # Stable exact logarithmic sum, not a floor or removal of actual zeros.
    log_z = np.logaddexp(length*np.log(plus), length*np.log(minus))-np.log1p((2*p-1)**length)
    return dict(z=np.exp(log_z), log_z=log_z, rate=-log_z/length,
                lambda_plus=plus, lambda_minus=minus,
                asymptotic_rate=-np.log(np.maximum(plus, minus)),
                boundary_length=int(length), vacuum_weight=p,
                virtual_center_even_parity=True, physical_spin_region=False,
                known_bulk_positive=False, QSL_claim=False)
