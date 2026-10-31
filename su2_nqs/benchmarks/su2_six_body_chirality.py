"""Frozen six-spin chirality scalar and its exact block-spin resolution.

T=(16/3) chi_A chi_B is the physical multiplicity tau_y product, with zero
action on either three-spin quartet. Unlike kappa_A.kappa_B (four-local),
T has Pauli weight exactly six. Linear Hilbert--Schmidt independence does
not prove that low-order data plus positivity cannot bound its expectation.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, combinations_with_replacement

import numpy as np

from .singlet_physics_readout import scalar_chirality_matrix
from .spin_operator_information import spin_dot_matrix
from .spin_partition_readout import _joint_projectors
from .su2_multiplicity_intervention import compact6_intervention_operators


@lru_cache(maxsize=1)
def six_body_chirality_operators() -> dict[str, np.ndarray]:
    """Physical64x64 operators; A=(0,3,4), B=(1,5,2), doubled q_total=0/2."""
    chi_a=scalar_chirality_matrix(6,(0,3,4))
    chi_b=scalar_chirality_matrix(6,(1,5,2))
    product=chi_a@chi_b
    total=(16/3)*product
    projectors=dict(_joint_projectors(6,(0,3,4)))
    result=dict(T=total,chi_A_chi_B=product,
        T_qtotal0=projectors[1,1,0]@total@projectors[1,1,0],
        T_qtotal2=projectors[1,1,2]@total@projectors[1,1,2],
        kappa_A_dot_B=compact6_intervention_operators()['vector_chirality_A_dot_B'])
    for matrix in result.values(): matrix.setflags(write=False)
    return result


def six_body_operator_diagnostics() -> dict[str, float | str]:
    """Exact bounded algebra and declared low-order operator-space checks."""
    bank=six_body_chirality_operators();t=bank['T'];parts=[bank['T_qtotal0'],bank['T_qtotal2']]
    joint=dict(_joint_projectors(6,(0,3,4)))
    partials=[np.trace(t.reshape((2,)*12),axis1=site,axis2=site+6).reshape(32,32)
              for site in range(6)]
    pairs=[spin_dot_matrix(6,i,j) for i,j in combinations(range(6),2)]
    low=[np.eye(64),*pairs,*[(a@b+b@a)/2 for a,b in combinations_with_replacement(pairs,2)]]
    return dict(
        hermiticity_frobenius=float(np.linalg.norm(t-t.conj().T)),
        decomposition_error=float(np.linalg.norm(t-sum(parts))),
        kappa_spin_weighted_identity_error=float(np.linalg.norm(bank['kappa_A_dot_B']-(-9/4*parts[0]+3/4*parts[1]))),
        T_squared_support_error=float(np.linalg.norm(t@t-joint[1,1,0]-joint[1,1,2])),
        norm_frobenius=float(np.linalg.norm(t)),
        max_one_site_partial_trace_norm=float(max(map(np.linalg.norm,partials))),
        max_HS_overlap_declared_low_order=float(max(abs(np.trace(t@op))/64 for op in low)),
        declared_low_order_columns=len(low),
        max_HS_overlap_joint_spin_projectors=float(max(abs(np.trace(t@op))/64 for op in joint.values())),
        conclusion='Every single-site partial trace is zero, hence Pauli weight6 only and HS orthogonality to every <=5-local operator. This is operator-linear nonredundancy, not a theorem of unconstrained expectation values at fixed low-order data.')
