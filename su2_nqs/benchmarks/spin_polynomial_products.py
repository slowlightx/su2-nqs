"""Exact bounded H/H² terminal plans for physical spin carriers, without ED.

Reuse the scalar-word sparse plan and batched JAX executor. Connectivity is
real symmetric Heisenberg H[new,old]; a second expansion sums all repeated
terminals BEFORE evaluation. No small-coefficient clipping or carrier ratios.
This host preparation is for fixed analytic-carrier force data, not the
neural training critical path or a new scalable Hamiltonian implementation.
"""
from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
import numpy as np

from ..operators.spin_config import SpinBond,heisenberg_spin_connectivity
from .operator_word_jax import ScalarWordActionPlan


def build_spin_h2_plan(configurations: np.ndarray,bonds: Sequence[SpinBond]
                       ) -> tuple[ScalarWordActionPlan,dict]:
    """Three outputs per center: C, HC, H²C; 0=down, site zero MSB."""
    start=perf_counter();states=np.asarray(configurations)
    if (states.ndim!=2 or not len(states) or not states.shape[1]
            or not np.issubdtype(states.dtype,np.integer) or np.any((states!=0)&(states!=1))):
        raise ValueError('nonempty integer physical-bit matrix required')
    if not bonds:raise ValueError('nonempty finite real bonds required')
    # Existing connectivity validates indices; additionally exclude silent
    # noninteger index coercions and nonfinite couplings at this API boundary.
    for bond in bonds:
        if len(bond) not in (2,3) or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in bond[:2]):
            raise ValueError('integer bond endpoints required')
        if len(bond)==3 and not np.isfinite(bond[2]):raise ValueError('finite coupling required')
    union={};positions=[];outputs=[];coefficients=[];requests=0;connect_calls=0
    for row,x in enumerate(states):
        center=tuple(map(int,x));cache={}
        def connect(state):
            nonlocal connect_calls
            if state not in cache:
                cache[state]=heisenberg_spin_connectivity(state,bonds);connect_calls+=1
            return cache[state]
        frontier={center:1.}
        for power in range(3):
            requests+=len(frontier)
            for state,c in frontier.items():
                if c==0:continue # exact arithmetic zero only, no threshold filter
                if state not in union:union[state]=len(union)
                positions.append(union[state]);outputs.append(3*row+power);coefficients.append(c)
            if power<2:
                following={}
                for state,c in frontier.items():
                    connected,values=connect(state)
                    for target,v in zip(connected,values,strict=True):
                        following[target]=following.get(target,0.)+c*float(v)
                frontier=following
    plan=ScalarWordActionPlan(np.asarray(tuple(union),np.int8),np.asarray(positions,np.int32),
        np.asarray(outputs,np.int32),np.asarray(coefficients,float),len(states),3,2,requests)
    return plan,dict(centers=len(states),powers=[0,1,2],unique_terminal_queries=len(union),
        sparse_entries=len(coefficients),connectivity_calls=connect_calls,
        host_plan_seconds=perf_counter()-start,small_coefficient_clipping=False,
        full_H_applications_per_center=2,H2_not_ED=True)


__all__=['build_spin_h2_plan']
