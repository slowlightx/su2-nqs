"""Finite SU(2) x C3/D3 ternary intertwiners, with explicit conventions.

This is offline local group-theory data. A cycle is the active physical
permutation ``old 0->1, 1->2, 2->0``; it sends product labels (a,b,c) to
(c,a,b). Magnetic labels are doubled integers. Learned synthesis matrices
have input-basis rows and output-feature columns: ``R_in W = W R_out``.
Their physical-amplitude contraction uses W, not an unmarked conjugate.

The representation here describes a common rotation of three input ports
and their local frames. It is NOT independent invariance of each triangle,
and does not automatically give a lattice-A1 state on a non-invariant cover.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from numbers import Integral
import numpy as np

from .clebsch_gordan import clebsch_gordan
from .local_schur import cell_schur_tensor
from .quantum_numbers import coupling_range, is_admissible_triangle, magnetic_numbers, validate_m, validate_q
from .wigner6j import f_move


@dataclass(frozen=True)
class C3Multiplicity:
    """Whole D3 irreps at one doubled spin: A1, A2, and conjugate E pairs.

    E contributes two reduced features, with ell=(+1,-1), and the same
    number of opposite chiralities is retained by construction. The total
    virtual dimension of this sector is (q+1)*(a1+a2+2*e_pairs).
    """

    q: int
    a1: int=0
    a2: int=0
    e_pairs: int=0

    def __post_init__(self) -> None:
        validate_q(self.q)
        for name in ('a1','a2','e_pairs'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.width<1:
            raise ValueError('a retained spin sector needs at least one complete irrep')

    @property
    def width(self) -> int:
        return int(self.a1+self.a2+2*self.e_pairs)

    @property
    def virtual_dimension(self) -> int:
        return (int(self.q)+1)*self.width

    @property
    def ell(self) -> tuple[int,...]:
        return (0,)*(self.a1+self.a2)+(1,-1)*self.e_pairs

    def rotation(self) -> np.ndarray:
        return np.diag(np.exp(2j*np.pi*np.asarray(self.ell)/3))

    def reflection(self) -> np.ndarray:
        result=np.eye(self.width,dtype=complex)
        result[self.a1:self.a1+self.a2,self.a1:self.a1+self.a2]*=-1
        for start in range(self.a1+self.a2,self.width,2):
            result[start:start+2,start:start+2]=[[0,1],[1,0]]
        return result

    def time_reversal_multiplicity(self) -> np.ndarray:
        """The real matrix multiplying K; physical magnetic T is separate."""
        result=np.eye(self.width,dtype=complex)
        for start in range(self.a1+self.a2,self.width,2):
            result[start:start+2,start:start+2]=[[0,1],[1,0]]
        return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result=np.array(array,copy=True)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def ternary_cycle_block(q0: int,q1: int,q2: int,total_q: int) -> tuple[np.ndarray,tuple[int,...],tuple[int,...]]:
    """Reduced active cycle from ((q0,q1)a,q2)Q to ((q2,q0)b,q1)Q.

    Two existing F moves and ordinary SU(2) exchange phases account for the
    arbitrary charge triple. Source and target q12 channels are returned
    explicitly. Merely cycling q12 or feature labels would be incorrect.
    """
    for q in (q0,q1,q2,total_q):
        validate_q(q)
    old,source,pair=f_move(q0,q1,q2,total_q)
    new,target,new_pair=f_move(q0,q2,q1,total_q)
    if pair!=new_pair:
        raise RuntimeError('swapped pair channels changed their deterministic order')
    pair_phase=np.asarray([(-1.)**((q1+q2-q)//2) for q in pair])
    first_phase=np.asarray([(-1.)**((q0+q2-q)//2) for q in target])
    matrix=first_phase[:,None]*(new.T @ (pair_phase[:,None]*old))
    return _readonly(matrix),source,target


def ternary_coupled_basis(q0: int,q1: int,q2: int,total_q: int,m: int) -> tuple[np.ndarray,tuple[int,...]]:
    """Small local columns <m0,m1,m2 | Q,m,q12>, Condon--Shortley phase."""
    for q in (q0,q1,q2,total_q):
        validate_q(q)
    validate_m(total_q,m)
    channels=tuple(q for q in coupling_range(q0,q1) if is_admissible_triangle(q,q2,total_q))
    if not channels:
        raise ValueError('empty ternary coupling sector')
    rows=list(product(magnetic_numbers(q0),magnetic_numbers(q1),magnetic_numbers(q2)))
    result=np.zeros((len(rows),len(channels)),dtype=np.float64)
    for row,(m0,m1,m2) in enumerate(rows):
        if m0+m1+m2==m:
            for column,q12 in enumerate(channels):
                if abs(m0+m1)<=q12:
                    result[row,column]=(clebsch_gordan(q0,m0,q1,m1,q12,m0+m1)
                        *clebsch_gordan(q12,m0+m1,q2,m2,total_q,m))
    return result,channels


@lru_cache(maxsize=1)
def triangle_chiral_schur() -> tuple[np.ndarray,tuple[tuple[int,int,int],...]]:
    """Complete columns <physical bits | q,m,ell> for three spin halves.

    q=1 retains ell=+1 and -1. q=3 retains ell=0. In each q=1 magnetic
    block the reflection P01 swaps the two chiral columns with no extra
    phase; time reversal swaps them and applies (-1)^((q-m)/2) on m.
    This is a unitary basis change, not additional variational capacity.
    """
    table=cell_schur_tensor('triangular_triangle')
    rotation,channels,_=ternary_cycle_block(1,1,1,1)
    omega=np.exp(2j*np.pi/3)
    projector=(np.eye(2)+omega.conjugate()*rotation+omega.conjugate()**2*(rotation @ rotation))/3
    # q12=2 has even reflection. Anchoring on it makes P01 exchange +,-.
    positive=projector[:,channels.index(2)]
    positive/=np.linalg.norm(positive)
    change=np.column_stack([positive,positive.conj()])
    columns=[]
    labels=[]
    for q in (1,3):
        for m in magnetic_numbers(q):
            block=table[:,q,m+3,:2 if q==1 else 1]
            transformed=block @ change if q==1 else block.astype(complex)
            for i,ell in enumerate((1,-1) if q==1 else (0,)):
                columns.append(transformed[:,i]);labels.append((q,m,ell))
    return _readonly(np.column_stack(columns)),tuple(labels)


# Channel order: q0,q1,q2,q12,a0,a1,a2. There is no magnetic index here.
ReducedChannel=tuple[int,int,int,int,int,int,int]


@dataclass(frozen=True)
class TernaryC3Representation:
    """Offline complete reduced representation for one ternary total q."""

    total_q: int
    children: tuple[C3Multiplicity,...]
    channels: tuple[ReducedChannel,...]
    rotation: np.ndarray
    reflection: np.ndarray
    time_reversal_multiplicity: np.ndarray

    def irrep_multiplicities(self) -> tuple[int,int,int]:
        size=len(self.channels)
        trivial=(size+2*np.trace(self.rotation).real)/3
        a1=(trivial+np.trace(self.reflection).real)/2
        a2=(trivial-np.trace(self.reflection).real)/2
        pairs=(size-trivial)/2
        values=np.asarray([a1,a2,pairs])
        integers=np.rint(values).astype(int)
        if np.max(abs(values-integers))>1e-10 or np.min(integers)<0:
            raise RuntimeError('D3 character counting failed')
        return tuple(int(x) for x in integers)


@lru_cache(maxsize=None)
def ternary_c3_representation(children: tuple[C3Multiplicity,...],total_q: int) -> TernaryC3Representation:
    """Combine all ordered q triples, their F moves, and child C3 frames.

    The dense matrices are an offline development/reference convenience,
    bounded to 512 reduced channels. A production evaluator should apply
    these same sparse charge-orbit/F blocks directly to reduced tensors.
    All three ordered cyclic images of unequal child charges are retained.
    """
    validate_q(total_q)
    if (not children or any(not isinstance(x,C3Multiplicity) for x in children)
            or len({x.q for x in children})!=len(children)):
        raise ValueError('one complete C3Multiplicity per distinct child q required')
    lookup={x.q:x for x in children}
    q_values=tuple(sorted(lookup))
    channels=[]
    for q0,q1,q2 in product(q_values,repeat=3):
        for q12 in coupling_range(q0,q1):
            if is_admissible_triangle(q12,q2,total_q):
                channels.extend((q0,q1,q2,q12,a0,a1,a2)
                    for a0,a1,a2 in product(range(lookup[q0].width),range(lookup[q1].width),range(lookup[q2].width)))
    if not channels or len(channels)>512:
        raise ValueError('local reference needs 1..512 reduced channels; use sparse blocks beyond that')
    channels=tuple(channels)
    positions={channel:i for i,channel in enumerate(channels)}
    size=len(channels)
    rotation=np.zeros((size,size),dtype=complex)
    reflection=np.zeros_like(rotation)
    time_reverse=np.zeros_like(rotation)
    for column,(q0,q1,q2,q12,a0,a1,a2) in enumerate(channels):
        block,old,new=ternary_cycle_block(q0,q1,q2,total_q)
        phase=np.exp(2j*np.pi*(lookup[q0].ell[a0]+lookup[q1].ell[a1]+lookup[q2].ell[a2])/3)
        for row,q20 in enumerate(new):
            target=(q2,q0,q1,q20,a2,a0,a1)
            rotation[positions[target],column]+=phase*block[row,old.index(q12)]
        reflected=[];reversed_features=[];reflection_phase=(-1.)**((q0+q1-q12)//2)
        for q,a in ((q0,a0),(q1,a1),(q2,a2)):
            s=lookup[q].reflection()[:,a]
            t=lookup[q].time_reversal_multiplicity()[:,a]
            target=int(np.flatnonzero(abs(s)>0)[0])
            reflected.append(target);reflection_phase*=s[target]
            reversed_features.append(int(np.flatnonzero(abs(t)>0)[0]))
        target=(q1,q0,q2,q12,reflected[1],reflected[0],reflected[2])
        reflection[positions[target],column]=reflection_phase
        target=(q0,q1,q2,q12,*reversed_features)
        time_reverse[positions[target],column]=1.
    return TernaryC3Representation(int(total_q),children,channels,
        _readonly(rotation),_readonly(reflection),_readonly(time_reverse))


def project_ternary_synthesis(raw: np.ndarray,source: TernaryC3Representation,
    target: C3Multiplicity,*,reflection: bool=False,time_reversal: bool=True) -> np.ndarray:
    """Project synthesis W onto SU2 x C3, optionally D3 and physical T.

    W has shape (source multiplicity,target width). Parent and child spaces
    carry the same total q. On physical wavefunctions its columns contract
    with the fixed CG structural columns. No magnetic-index NN is involved.
    The safe default is C3+T; reflection needs an explicit D3 opt-in and a
    geometric frame certificate before being interpreted as physical.
    """
    values=np.asarray(raw,dtype=np.complex128)
    if target.q!=source.total_q or values.shape!=(len(source.channels),target.width):
        raise ValueError('synthesis shape or doubled total q differs from its representation')
    if not np.isfinite(values).all():
        raise ValueError('finite synthesis coefficients required')
    r,ro=source.rotation,target.rotation()
    result=(values+r @ values @ ro.conj().T+(r @ r) @ values @ (ro @ ro).conj().T)/3
    if reflection:
        result=(result+source.reflection @ result @ target.reflection().conj().T)/2
    if time_reversal:
        result=(result+source.time_reversal_multiplicity @ result.conj()
                @ target.time_reversal_multiplicity().conj().T)/2
    return result


def ternary_intertwiner_basis(source: TernaryC3Representation,target: C3Multiplicity,
    *,reflection: bool=False,time_reversal: bool=True) -> tuple[np.ndarray,...]:
    """Independent real coordinates for C3, optionally D3 and T, synthesis.

    Removes redundant projected raw parameters, avoiding an artificial SR
    null space and misleading parameter-count comparison. Local offline
    image SVD is bounded to 512 real raw coordinates.
    By default only C3 and T are imposed; D3 requires reflection=True.
    """
    shape=(len(source.channels),target.width)
    if 2*np.prod(shape)>512:
        raise ValueError('offline image basis exceeds 512 real raw coordinates')
    columns=[]
    for coefficient in (1.,1j):
        for i in range(int(np.prod(shape))):
            raw=np.zeros(shape,dtype=complex);raw.flat[i]=coefficient
            projected=project_ternary_synthesis(raw,source,target,
                reflection=reflection,time_reversal=time_reversal).reshape(-1)
            columns.append(np.concatenate([projected.real,projected.imag]))
    image=np.column_stack(columns)
    u,s,_=np.linalg.svd(image,full_matrices=False)
    rank=int(np.count_nonzero(s>1e-11))
    n=int(np.prod(shape))
    return tuple(_readonly((u[:n,i]+1j*u[n:,i]).reshape(shape)) for i in range(rank))
