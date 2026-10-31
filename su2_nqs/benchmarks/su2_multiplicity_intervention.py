"""Physical, gauge-independent SU(2)-scalar interventions on regional RDMs.

Pair-spin q=0/2 measurements resolve internal fusion channels of three spins.
All outcomes are retained. No virtual tensor coordinates, twirl, postselection,
spatial projection, or assertion about energy outside the RDM support occurs.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.linalg import expm

from .spin_operator_information import spin_dot_matrix
from .singlet_physics_readout import scalar_chirality_matrix
from .spin_partition_readout import _joint_projectors
from ..su2.fusion_tree import FusionTree
from ..su2.recoupling import tree_basis_matrix


@lru_cache(maxsize=120)
def _physical_spin(n: int, site: int, component: int) -> np.ndarray:
    single = (np.array([[0,1],[1,0]], complex)/2,
              np.array([[0,1j],[-1j,0]], complex)/2, np.diag([-.5,.5]))
    op=np.ones((1,1),complex)
    for i in range(n): op=np.kron(op,single[component] if i==site else np.eye(2))
    return op


def triangle_vector_chirality(n: int, triangle: Sequence[int]) -> tuple[np.ndarray,...]:
    """Oriented sum S_i cross S_j+S_j cross S_k+S_k cross S_i."""
    side=tuple(triangle)
    if (isinstance(n,bool) or not isinstance(n,Integral) or not 3<=n<=6
            or len(side)!=3 or len(set(side))!=3 or any(isinstance(i,(bool,np.bool_))
            or not isinstance(i,Integral) or not 0<=i<n for i in side)):
        raise ValueError('three distinct integer sites on 3..6 spins required')
    return tuple(sum(_physical_spin(n,side[t],(c+1)%3)@_physical_spin(n,side[(t+1)%3],(c+2)%3)
        -_physical_spin(n,side[t],(c+2)%3)@_physical_spin(n,side[(t+1)%3],(c+1)%3)
        for t in range(3)) for c in range(3))


def triangle_multiplicity_identity() -> dict:
    """Verify kappa=c tau_y tensor S in the fixed ((01),2) CS basis.

    a=0,1 labels doubled pair spin0,2; magnetic order is m=-1,+1.
    Also verify physical kappa=4 chi S_total, including absent quartet blocks.
    """
    tree=FusionTree.left_associated(3)
    lookup={}
    for q in (1,3):
        for m in range(-q,q+1,2):
            columns,states=tree_basis_matrix(tree,q,m)
            for a in range(len(states)): lookup[q,m,a]=columns[:,a]
    doublet=np.column_stack([lookup[1,m,a] for a in range(2) for m in (-1,1)])
    quartet=np.column_stack([lookup[3,m,0] for m in (-3,-1,1,3)])
    tau_y=np.array([[0,-1j],[1j,0]])
    spins=[sum(_physical_spin(3,i,c) for i in range(3)) for c in range(3)]
    small=[np.array([[0,1],[1,0]])/2,np.array([[0,1j],[-1j,0]])/2,np.diag([-.5,.5])]
    kappas=triangle_vector_chirality(3,(0,1,2))
    chi=scalar_chirality_matrix(3,(0,1,2))
    targets=[np.kron(tau_y,s) for s in small]
    c=float(sum(np.vdot(t,doublet.conj().T@k@doublet).real for t,k in zip(targets,kappas))
            /sum(np.vdot(t,t).real for t in targets))
    return dict(coefficient_c=c,expected_absolute_c=float(np.sqrt(3)),
        doublet_tensor_identity_error=max(float(np.linalg.norm(doublet.conj().T@k@doublet-c*t)) for k,t in zip(kappas,targets)),
        quartet_internal_error=max(float(np.linalg.norm(quartet.conj().T@k@quartet)) for k in kappas),
        doublet_quartet_error=max(float(np.linalg.norm(doublet.conj().T@k@quartet)) for k in kappas),
        physical_kappa_equals_4chiS_error=max(float(np.linalg.norm(k-4*chi@s)) for k,s in zip(kappas,spins)),
        convention='Condon-Shortley ((0,1),2); a=0/1 corresponds q_pair=0/2; m=-1,+1; tau_y=[[0,-i],[i,0]]')


def pair_spin_projectors(n: int, pair: Sequence[int]) -> dict[int, np.ndarray]:
    """Complete physical pair-spin projectors keyed by doubled q=0,2."""
    if isinstance(n, bool) or not isinstance(n, Integral) or not 2 <= n <= 6:
        raise ValueError('integer region size 2..6 required')
    pair = tuple(pair)
    if (len(pair) != 2 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
            or not 0 <= i < n for i in pair) or pair[0] == pair[1]):
        raise ValueError('two distinct integer physical sites required')
    dot = spin_dot_matrix(int(n), *map(int, pair))
    identity = np.eye(2**n)
    return {0: identity/4-dot, 2: 3*identity/4+dot}


def pair_spin_dephase(rho: np.ndarray, pair: Sequence[int]) -> tuple[np.ndarray, dict[int, float]]:
    """Nonselective q=0/2 measurement; return sum_q P_q rho P_q and probabilities.

    On a three-spin block the two j=1/2 multiplicities are distinguished by
    the chosen physical pair spin. All q=3/2 states remain in the triplet
    pair outcome. This preserves regional and block total-spin weights.
    """
    rho = np.asarray(rho, np.complex128)
    n = rho.shape[0].bit_length()-1 if rho.ndim == 2 else 0
    if (rho.shape != (2**n, 2**n) or not 2 <= n <= 6 or not np.isfinite(rho).all()
            or np.linalg.norm(rho-rho.conj().T) > 1e-9 or abs(np.trace(rho)-1) > 1e-9):
        raise ValueError('finite Hermitian normalized 2..6-site RDM required')
    projectors = pair_spin_projectors(n, pair)
    return (sum(p@rho@p for p in projectors.values()),
            {q: float(np.trace(rho@p).real) for q, p in projectors.items()})


def chirality_unitary(n: int, triangle: Sequence[int], angle: float) -> np.ndarray:
    """Physical SU(2)-scalar exp(-i angle S_i.(S_j cross S_k))."""
    if not np.isfinite(angle):
        raise ValueError('finite angle required')
    triangle = tuple(triangle)
    if (len(triangle) != 3 or len(set(triangle)) != 3 or
            any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                or not 0 <= i < n for i in triangle)):
        raise ValueError('three distinct integer physical sites required')
    return expm(-1j*float(angle)*scalar_chirality_matrix(n, triangle))


@lru_cache(maxsize=1)
def compact6_intervention_operators() -> dict[str, np.ndarray]:
    """Frozen supported operators on (0,0),(0,1),(0,2),(1,0),(1,1),(1,2).

    A=(0,3,4), B=(1,5,2) are disjoint oriented elementary triangles.
    Regional K/M form factors are NOT full-system structure factors.
    """
    n = 6
    coordinates = np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    a, b = (0, 3, 4), (1, 5, 2)
    dots = {(i, j): spin_dot_matrix(n, i, j) for i in range(n) for j in range(i+1, n)}
    get = lambda i, j: 3*np.eye(64)/4 if i == j else dots[tuple(sorted((i, j)))]
    result = {}
    for squared_distance, name in ((1, 'H_J1_supported'), (3, 'H_J2_supported')):
        result[name] = sum(op for (i, j), op in dots.items()
            if (lambda d: d[0]**2+d[1]**2-d[0]*d[1])(coordinates[i]-coordinates[j]) == squared_distance)
    result['triangle_Casimir_energy'] = sum(get(side[i], side[(i+1)%3]) for side in (a,b) for i in range(3))
    result['S_A_dot_S_B'] = sum(get(i,j) for i in a for j in b)
    for name, k in [('K', (1/3,1/3)), ('M1', (.5,0)), ('M2', (0,.5)), ('M3', (.5,.5))]:
        phase = 2*np.pi*(coordinates@np.asarray(k))
        result['regional_'+name+'_scalar'] = sum(np.cos(phase[i]-phase[j])*get(i,j)
            for i in range(n) for j in range(n))/n
    kappas = [triangle_vector_chirality(n,side) for side in (a,b)]
    result['vector_chirality_A_dot_B'] = sum(x@y for x,y in zip(*kappas))
    for labels, op in _joint_projectors(6,a):
        result['P_'+('_'.join(map(str,labels)))] = op
        if labels in ((1,1,0),(1,1,2)):
            result['vector_chirality_qtotal'+str(labels[-1])] = op@result['vector_chirality_A_dot_B']@op
    for op in result.values(): op.setflags(write=False)
    return result


def intervention_observables(rho: np.ndarray) -> dict[str, float]:
    """Linear actual-RDM supported observables; no interpretation as full H."""
    matrix = np.asarray(rho,complex)
    if matrix.shape != (64,64): raise ValueError('compact six-site matrix required')
    values = {k: np.trace(matrix@v) for k,v in compact6_intervention_operators().items()}
    if max(abs(v.imag) for v in values.values()) > 1e-9:
        raise ValueError('nonreal Hermitian expectations')
    return {k: float(v.real) for k,v in values.items()}
