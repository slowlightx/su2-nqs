"""Same-region physical readout distances and empirical refinement checks.

Labels use q=2S. A refinement difference is NOT a certified state error.
The caller must check matching physical sites, Hamiltonian and boundaries
before interpreting a comparison as evidence for competing states.
"""
from __future__ import annotations

from numbers import Integral
import hashlib
import json
from typing import Sequence

import numpy as np

from .tensor_network_readout import physical_spin_block_readout


def physical_problem_signature(
    n_sites: int, bonds: Sequence[tuple[int,int,float]], *, geometry: str, boundary: str,
) -> str:
    """Hash a labelled Heisenberg graph, geometry and PHYSICAL boundary.

    Combine repeated unordered edges, preserving their total coupling. No
    site relabelling is inferred. Parton boundary twists are NOT physical
    spin-Hamiltonian boundary conditions and must not replace ``boundary``.
    """
    if isinstance(n_sites,bool) or not isinstance(n_sites,Integral) or n_sites<2:
        raise ValueError('n_sites must be an integer >=2')
    if not isinstance(geometry,str) or not geometry or not isinstance(boundary,str) or not boundary:
        raise ValueError('geometry and physical boundary must be explicit')
    edges = {}
    for i,j,w in bonds:
        if (any(isinstance(s,bool) or not isinstance(s,Integral) for s in (i,j)) or
                not 0<=i<n_sites or not 0<=j<n_sites or i==j or
                np.iscomplexobj(w) or not np.isfinite(w)):
            raise ValueError('bonds require distinct valid integer sites and finite real couplings')
        edge = tuple(sorted((int(i),int(j))))
        edges[edge] = edges.get(edge,0.)+float(w)
    payload = dict(n_sites=int(n_sites),geometry=geometry,boundary=boundary,
                   bonds=[(i,j,w) for (i,j),w in sorted(edges.items()) if w!=0.])
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def physical_state_similarity(state_a: np.ndarray, state_b: np.ndarray) -> dict:
    """Compare aligned physical amplitudes modulo overall norm and phase.

    The caller must align the SAME basis first. This is not a proof of phase
    identity; two finite-volume near-ground vectors can agree while different
    thermodynamic continuations remain possible. No symmetry projection.
    """
    vectors = []
    for state in (state_a,state_b):
        v = np.asarray(state,dtype=np.complex128)
        if v.ndim!=1 or not np.isfinite(v).all() or np.linalg.norm(v)<=1e-14:
            raise ValueError('states must be finite nonzero vectors')
        vectors.append(v/np.linalg.norm(v))
    a,b = vectors
    if a.shape!=b.shape:
        raise ValueError('states must have the same aligned physical support')
    overlap = np.vdot(a,b)
    return dict(fidelity=float(abs(overlap)**2),
                phase_aligned_distance=float(np.linalg.norm(a-b*np.exp(-1j*np.angle(overlap)))))


def regional_readout_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> dict:
    """Compare physical RDMs without symmetry repair, including broken SU(2).

    Return trace distance and total variation of p(q). Multiplicity spectra
    are compared ONLY if BOTH actual inputs admit invariant factorization.
    This does not infer a phase or identify two states as competing minima.
    """
    a = np.asarray(rho_a, dtype=np.complex128)
    b = np.asarray(rho_b, dtype=np.complex128)
    if a.shape != b.shape:
        raise ValueError('RDMs must have the same physical region dimension')
    ra, rb = physical_spin_block_readout(a), physical_spin_block_readout(b)
    trace = .5 * float(np.abs(np.linalg.eigvalsh(a-b)).sum())
    p_tv = .5 * sum(abs(ra['p_q'][q]-rb['p_q'][q]) for q in ra['p_q'])
    result = dict(physical_trace_distance=trace, p_q_total_variation=p_tv,
                  both_strict_su2_eligible=ra['strict_su2_eligible'] and rb['strict_su2_eligible'],
                  measurement_contractivity_error=max(0., p_tv-trace))
    if result['both_strict_su2_eligible']:
        sa, sb = ra['invariant_readout'], rb['invariant_readout']
        result['multiplicity_spectrum_total_variation'] = .5 * sum(
            float(np.abs(np.asarray(sa['sectors'][q]['unnormalized_multiplicity_spectrum'])-
                         np.asarray(sb['sectors'][q]['unnormalized_multiplicity_spectrum'])).sum())
            for q in sa['sectors'])
    return result


def compare_with_refinement(
    rho_a: np.ndarray, rho_b: np.ndarray, *,
    refinements_a: Sequence[np.ndarray], refinements_b: Sequence[np.ndarray],
) -> dict:
    """Compare a signal with EACH endpoint's declared refinement changes.

    For each metric use the maximum endpoint-to-refinement distance and
    their SUM as a conservative empirical resolution reference. Empty
    histories never pass. This is not a statistical confidence interval or
    a bound on finite-D/chi bias; converging to the wrong state can pass it.
    """
    distance = regional_readout_distance(rho_a, rho_b)
    histories = [
        [regional_readout_distance(endpoint, rho) for rho in refinements]
        for endpoint, refinements in ((rho_a, refinements_a), (rho_b, refinements_b))
    ]
    complete = all(histories)
    metrics = ['physical_trace_distance', 'p_q_total_variation']
    if ('multiplicity_spectrum_total_variation' in distance and
            all('multiplicity_spectrum_total_variation' in r for h in histories for r in h)):
        metrics.append('multiplicity_spectrum_total_variation')
    envelopes = {}
    for key in metrics:
        delta = [max((r[key] for r in h), default=None) for h in histories]
        envelope = sum(delta) if complete else None
        envelopes[key] = dict(
            difference=distance[key], refinement_a=delta[0], refinement_b=delta[1],
            refinement_sum=envelope,
            empirical_margin=distance[key]-envelope if complete else None,
            exceeds_declared_refinement=bool(complete and distance[key] > envelope+1e-12),
        )
    return dict(distance=distance, refinement_histories_available=bool(complete),
                metrics=envelopes,
                warning='Refinement differences are empirical, not certified errors. Matching H/BC, ansatz bias and sampling errors require separate checks; no size/phase release follows automatically.')


__all__ = ['physical_problem_signature', 'physical_state_similarity',
           'regional_readout_distance', 'compare_with_refinement']
