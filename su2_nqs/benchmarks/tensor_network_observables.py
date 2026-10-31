"""Physical regional spin/dimer and cyclic weights, including broken symmetry.

These are measurement operators on 1..8 spin-half sites, not a variational
fusion-path implementation. q=2S; bits down/up; site zero is MSB. Joint q,k
probabilities exist even for broken symmetry, but are NOT entanglement levels.
"""
from __future__ import annotations

from itertools import combinations
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from .regional_su2_readout import regional_schur_columns
from .tensor_network_readout import physical_spin_block_readout


def regional_physical_observables(rho: np.ndarray, *, cyclic: bool = False) -> dict:
    """Read all pair spins and disjoint connected dimers; optionally a full cycle.

    A full cycle follows the supplied physical site ordering. Its weights are
    traces of commuting projectors P_q P_k; no SU2/spatial twirl is performed.
    Only invariant rho permits interpretation as a symmetry-labelled spectrum.
    """
    rho=np.asarray(rho,dtype=np.complex128)
    block=physical_spin_block_readout(rho)
    d=len(rho); n=d.bit_length()-1
    bits=((np.arange(d)[:,None]>>np.arange(n-1,-1,-1))&1)
    operators=[]; pairs=list(combinations(range(n),2)); means=[]
    for i,j in pairs:
        op=np.diag((bits[:,i]-.5)*(bits[:,j]-.5)).astype(complex)
        for old,row in enumerate(bits):
            if row[i]!=row[j]:
                op[old^((1<<(n-1-i))|(1<<(n-1-j))),old]=.5
        operators.append(op); means.append(float(np.trace(rho@op).real))
    connected=[]
    for a,b in combinations(range(len(pairs)),2):
        if set(pairs[a]).isdisjoint(pairs[b]):
            value=np.einsum('ij,jk,ki->',rho,operators[a],operators[b],optimize=True)
            connected.append(dict(bonds=[list(pairs[a]),list(pairs[b])],
                value=float(value.real-means[a]*means[b]),imaginary_error=float(abs(value.imag))))
    spin_from_pairs=3*n/4+2*sum(means)
    result=dict(spin_pairs=[dict(sites=list(pair),value=value) for pair,value in zip(pairs,means)],
        disjoint_connected_dimers=connected,
        spin_squared_pair_closure_error=float(abs(spin_from_pairs-block['spin_squared_mean'])))
    if cyclic:
        image=np.roll(bits,1,axis=1)@2**np.arange(n-1,-1,-1)
        t=np.zeros((d,d),complex); t[image,np.arange(d)]=1
        u,labels=regional_schur_columns(n)
        powers=[np.linalg.matrix_power(t,j) for j in range(n)]
        projectors=[sum(np.exp(-2j*np.pi*k*j/n)*powers[j] for j in range(n))/n for k in range(n)]
        joint={}; error=0.
        for q in sorted({label[0] for label in labels}):
            ids=[i for i,label in enumerate(labels) if label[0]==q]
            p=u[:,ids]@u[:,ids].conj().T
            values=[]
            for pk in projectors:
                value=np.trace(rho@p@pk)
                error=max(error,float(abs(value.imag)),float(max(0.,-value.real)))
                values.append(float(value.real))
            joint[str(q)]=values
            error=max(error,abs(sum(values)-block['p_q'][str(q)]))
        spatial_defect=float(np.linalg.norm(rho@t-t@rho))
        result['cyclic']=dict(order=n,p_q_k=joint,probability_closure_error=error,
            spatial_commutator_frobenius=spatial_defect,
            symmetry_spectrum_eligible=block['strict_su2_eligible'] and spatial_defect<=1e-9,
            warning='Projector probabilities remain measurable for broken symmetry. No entanglement-spectrum or bulk-momentum interpretation without invariance.')
    return result


def _finite_measurement_environment(psi: Any) -> tuple[Any, complex]:
    """Contract actual finite MPS boundaries; do not assume canonical forms.

    Saved optimization boundaries can declare A/B forms while their Schmidt
    data have finite truncation inconsistency. The MPS convenience methods
    assume exact canonicality. Full environments retain the physical state
    and divide by its contracted norm, without canonicalizing the parent.
    """
    if psi.bc != 'finite':
        raise ValueError('explicit finite MPS boundaries required')
    from tenpy.networks.mps import MPSEnvironment
    environment = MPSEnvironment(psi, psi)
    norm = complex(environment.full_contraction(0))
    if (not np.isfinite(norm) or norm.real <= 0 or
            abs(norm.imag) > 1e-10*norm.real):
        raise ValueError('finite MPS contracted norm must be positive and real')
    return environment, norm


def tenpy_spin_correlations(psi: Any, rows: int, columns: int) -> tuple[dict,np.ndarray]:
    """Measure finite-MPS global S² and K/M structure factors on a 120° torus.

    S(k)=sum_ij exp[ik.(r_i-r_j)] <S_i.S_j>/N. This is a finite-size
    structure factor, NOT an infinite-volume magnetic order parameter.
    """
    if (any(isinstance(x,bool) or not isinstance(x,Integral) or x<1 for x in (rows,columns))
            or rows*columns!=psi.L):
        raise ValueError('rows and columns must match the finite MPS')
    environment,norm = _finite_measurement_environment(psi)
    c=np.asarray(environment.correlation_function('Sz','Sz',hermitian=True),complex)
    c+=.5*np.asarray(environment.correlation_function('Sp','Sm',hermitian=True),complex)
    c+=.5*np.asarray(environment.correlation_function('Sm','Sp',hermitian=True),complex)
    c/=norm
    imaginary=float(np.max(abs(c.imag))); c=c.real
    coordinates=np.array([divmod(i,columns) for i in range(psi.L)])
    # Repository metric has NN (1,0),(0,1),(1,1), so K=(1/3,1/3).
    momenta={'Gamma':(0.,0.),'K':(1/3,1/3),'minus_K':(-1/3,-1/3),
             'M1':(.5,0.),'M2':(0.,.5),'M3':(.5,.5)}
    values={}
    for name,k in momenta.items():
        phase=np.exp(2j*np.pi*(coordinates@np.array(k)))
        values[name]=float(np.vdot(phase,c@phase).real/psi.L)
    sz=np.asarray(environment.expectation_value('Sz')).real/norm.real
    return dict(spin_squared=float(c.sum()),total_sz=float(sz.sum()),
        maximum_local_sz=float(np.max(abs(sz))),structure_factors=values,
        maximum_imaginary_correlation=imaginary,
        diagonal_spin_error=float(np.max(abs(np.diag(c)-.75))),
        contracted_norm_squared=norm.real,
        contraction='full finite left/right environments; normalized by actual contraction',
        warning='Finite PBC MPS; no singlet projection and no thermodynamic order extrapolation.'),c


def tenpy_dimer_correlations(psi: Any, bonds: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrized connected <{B_a,B_b}/2> - <B_a><B_b>, B=Si.Sj.

    Operator order on shared sites is retained. This includes variance on
    the diagonal and does not assume real amplitudes or a singlet input.
    """
    edges = [tuple(b) for b in bonds]
    if (not edges or any(len(b)!=2 or b[0]==b[1] or
            any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<psi.L for i in b)
            for b in edges)):
        raise ValueError('dimer bonds require two distinct valid integer sites')
    terms = [('Sz','Sz',1.), ('Sp','Sm',.5), ('Sm','Sp',.5)]
    environment,norm = _finite_measurement_environment(psi)
    def bond(i, j):
        return sum(w*environment.expectation_value_term([(a,i),(b,j)], autoJW=False)
                   for a,b,w in terms)/norm
    means = np.asarray([bond(*b) for b in edges], complex)
    products = np.empty((len(edges),len(edges)), complex)
    for a,(i,j) in enumerate(edges):
        for b in range(a,len(edges)):
            k,l = edges[b]
            ab = sum(w*v*environment.expectation_value_term([(x,i),(y,j),(z,k),(t,l)],autoJW=False)
                     for x,y,w in terms for z,t,v in terms)/norm
            value = (ab+np.conj(ab))/2
            products[a,b] = products[b,a] = value
    return means, products - np.outer(means,means)
