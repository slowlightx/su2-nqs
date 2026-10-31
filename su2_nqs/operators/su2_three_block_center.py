"""Exact raw-tree three-block singlet center, without a magnetic product tensor.

The physical blocks A, B and E remain fixed. Coefficients describe
``((A B) q, E q) -> 0`` and can be factorized into a new AB isometry and
the existing singlet root. Pair interactions are evaluated by the tested
two-block operator, with signed permutations between three-spin singlets.
No physical full-H variance or coherent spatial projection is implied.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from numbers import Integral
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.quantum_numbers import coupling_range
from .su2_sector_merge import ReducedBlockData, SectorMergeOperator


@lru_cache(maxsize=1024)
def _singlet_pair_sign(qa: int, qb: int, qe: int, pair: str) -> float:
    """Overlap of AB/E and AE/B or BE/A singlets in the SAME A,B,E order."""
    if pair == 'AB':
        return 1.
    if pair not in ('AE', 'BE') or qe not in coupling_range(qa, qb):
        raise ValueError('legal three-block singlet charges and pair required')
    value = 0.
    for ma in range(-qa, qa+1, 2):
        for mb in range(-qb, qb+1, 2):
            me = -ma-mb
            if abs(me)>qe or (me-qe)%2:
                continue
            original = (clebsch_gordan(qa,ma,qb,mb,qe,-me)
                        *clebsch_gordan(qe,-me,qe,me,0,0))
            alternative = (clebsch_gordan(qa,ma,qe,me,qb,-mb)
                           *clebsch_gordan(qb,-mb,qb,mb,0,0) if pair=='AE' else
                           clebsch_gordan(qb,mb,qe,me,qa,-ma)
                           *clebsch_gordan(qa,-ma,qa,ma,0,0))
            value += original*alternative
    if abs(abs(value)-1.)>2e-12:
        raise ArithmeticError('three-block singlet recoupling is not a signed permutation')
    return float(np.sign(value))


class ThreeBlockSingletOperator:
    """Exact H on an unfactorized AB/root center with environment E fixed.

    All labels use doubled q. ``channels`` are ``(qa,a,qb,b,qe,e)``.
    Input block H already includes its own internal bonds. Every interblock
    physical bond must occur in exactly one supplied AB/AE/BE list.
    """

    def __init__(self, a: ReducedBlockData, b: ReducedBlockData, environment: ReducedBlockData,
                 bonds_ab: Sequence[tuple[int,int,float]],
                 bonds_ae: Sequence[tuple[int,int,float]],
                 bonds_be: Sequence[tuple[int,int,float]], *,
                 memory_limit_bytes: int = 768*1024**2, max_dimension: int = 100_000):
        if (isinstance(max_dimension,bool) or not isinstance(max_dimension,Integral) or max_dimension<1
                or isinstance(memory_limit_bytes,bool) or not isinstance(memory_limit_bytes,Integral)
                or memory_limit_bytes<1):
            raise ValueError('positive integer center and memory bounds required')
        self.blocks=(a,b,environment)
        center_dimension=sum(da*db*de for qe,de in environment.q_widths.items()
            for qa,da in a.q_widths.items() for qb,db in b.q_widths.items()
            if qe in coupling_range(qa,qb))
        if not 0<center_dimension<=max_dimension:
            raise ValueError('three-block singlet center absent or above explicit dimension bound')
        self.channels=tuple((qa,ia,qb,ib,qe,ie)
            for qe,de in sorted(environment.q_widths.items())
            for qa,da in sorted(a.q_widths.items()) for qb,db in sorted(b.q_widths.items())
            if qe in coupling_range(qa,qb)
            for ia in range(da) for ib in range(db) for ie in range(de))
        self.dimension=len(self.channels)
        if self.dimension!=center_dimension:raise ArithmeticError('center channel count changed')
        lookup={c:i for i,c in enumerate(self.channels)}
        self.q_indices={q:np.asarray([i for i,c in enumerate(self.channels) if c[4]==q],dtype=int)
                        for q in sorted(environment.q_widths)
                        if any(c[4]==q for c in self.channels)}
        zero=lambda x:replace(x,h_by_q={q:np.zeros_like(h) for q,h in x.h_by_q.items()})
        self.pairs={}; estimated_bytes=self.dimension*16*16
        for name,left,right,spectator,bonds in (
            ('AB',a,b,environment,bonds_ab),('AE',zero(a),zero(environment),b,bonds_ae),
            ('BE',zero(b),zero(environment),a,bonds_be)):
            prepared=None; rows={}
            for q,width in sorted(spectator.q_widths.items()):
                if not any(q in coupling_range(ql,qr) for ql in left.q_widths for qr in right.q_widths):
                    continue
                op=SectorMergeOperator(left,right,bonds,q,memory_limit_bytes=memory_limit_bytes,
                                       _prepared=prepared)
                if prepared is None:
                    prepared=(op.left,op.right)
                    estimated_bytes += sum(x.storage_bytes+x.input_storage_bytes for x in prepared)
                indices=[]; signs=[]
                for ql,il,qr,ir in op.channels:
                    canonical=[(ql,il,qr,ir,q,k) if name=='AB' else
                               (ql,il,q,k,qr,ir) if name=='AE' else
                               (q,k,ql,il,qr,ir) for k in range(width)]
                    indices.append([lookup[c] for c in canonical])
                    signs.append(_singlet_pair_sign(canonical[0][0],canonical[0][2],canonical[0][4],name))
                index=np.asarray(indices,dtype=np.int64); sign=np.asarray(signs)[:,None]
                estimated_bytes += index.nbytes+sign.nbytes+int(op.diagnostics['cached_embedding_bytes'])
                rows[q]=(op,index,sign)
            self.pairs[name]=rows
        # Pair operations are serial; reserve the largest pair workspace once.
        estimated_bytes += max(int(op.diagnostics['estimated_product_workspace_bytes'])+
                               int(op.diagnostics['reserved_validation_workspace_bytes'])
                               for rows in self.pairs.values() for op,_,_ in rows.values())
        if estimated_bytes>memory_limit_bytes:
            raise MemoryError('three-block cached pair operators exceed the memory bound')
        self.matvec_count=0;self.matvec_seconds=0.
        self.diagnostics=dict(reduced_dimension=self.dimension,
            estimated_resident_plus_serial_workspace_bytes=estimated_bytes,
            full_magnetic_three_block_tensor_constructed=False,
            full_H_variance_known=False,full_A1_energy_known=False,
            pair_sector_counts={name:len(rows) for name,rows in self.pairs.items()})

    def matvec(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply all block internals and AB/AE/BE boundary bonds exactly."""
        tick=perf_counter();c=np.asarray(coefficients)
        if c.shape!=(self.dimension,) or not np.isfinite(c).all():
            raise ValueError('finite coefficient vector in the declared center required')
        dtype=np.result_type(c.dtype,*[op.dtype for rows in self.pairs.values() for op,_,_ in rows.values()],
                             *[np.asarray(h).dtype for h in self.blocks[2].h_by_q.values()])
        result=np.zeros(self.dimension,dtype=dtype)
        for rows in self.pairs.values():
            for op,indices,signs in rows.values():
                columns=c[indices]*signs
                transformed=np.column_stack([op.matvec(columns[:,i]) for i in range(columns.shape[1])])
                result[indices] += transformed*signs
        # AB already includes A and B internal H; E was zeroed in AE/BE.
        environment=self.blocks[2]
        for q,indices in self.q_indices.items():
            width=environment.q_widths[q]
            matrix=c[indices].reshape(-1,width)
            result[indices] += (matrix@np.asarray(environment.h_by_q[q]).T).ravel()
        self.matvec_count+=1;self.matvec_seconds+=perf_counter()-tick
        return result

    def encode(self, upper_maps: Mapping[int,np.ndarray],
               root_coefficients: Mapping[int,np.ndarray]) -> np.ndarray:
        """Combine an AB intertwiner and normalized root into the linear center."""
        if set(upper_maps)!=set(self.q_indices) or set(root_coefficients)!=set(self.q_indices):
            raise ValueError('maps and root must cover exactly the available environment sectors')
        dtype=np.result_type(*[np.asarray(v).dtype for v in upper_maps.values()],
                             *[np.asarray(v).dtype for v in root_coefficients.values()])
        c=np.empty(self.dimension,dtype=dtype)
        for q,indices in self.q_indices.items():
            n=len(indices)//self.blocks[2].q_widths[q]
            w=np.asarray(upper_maps[q]);r=np.asarray(root_coefficients[q])
            if (w.ndim!=2 or w.shape[0]!=n or r.shape!=(w.shape[1],self.blocks[2].q_widths[q])
                    or not np.isfinite(w).all() or not np.isfinite(r).all()):
                raise ValueError('upper/root shapes must match the declared doubled-q center')
            c[indices]=(w@r).ravel()
        return c

    def factor(self, coefficients: np.ndarray, *, widths: Mapping[int,int]) -> tuple[dict,dict,dict]:
        """Exact q-QR back into unchanged-width upper/root tensors; no truncation."""
        c=np.asarray(coefficients)
        if c.shape!=(self.dimension,) or not np.isfinite(c).all() or set(widths)!=set(self.q_indices):
            raise ValueError('finite center and complete fixed widths required')
        maps={};roots={};error=0.
        for q,indices in self.q_indices.items():
            d=self.blocks[2].q_widths[q];n=len(indices)//d;width=widths[q]
            if isinstance(width,bool) or not isinstance(width,Integral) or not d<=width<=n:
                raise ValueError('exact factorization needs upper width >= fixed environment width')
            matrix=c[indices].reshape(n,d)
            # QR has exactly d columns. If the old bond was wider, append a
            # deterministic orthogonal completion without physical root weight.
            basis,r=np.linalg.qr(matrix,mode='reduced')
            for k in range(d):
                if abs(r[k,k])>0:
                    phase=r[k,k]/abs(r[k,k]);basis[:,k]*=phase;r[k,:]*=np.conj(phase)
            if width>d:
                for k in range(n):
                    v=np.eye(1,n,k,dtype=basis.dtype).ravel()
                    for _ in range(2):v-=basis@(basis.conj().T@v)
                    if np.linalg.norm(v)>1e-10:
                        basis=np.column_stack((basis,v/np.linalg.norm(v)))
                        if basis.shape[1]==width:break
                r=np.vstack((r,np.zeros((width-d,d),dtype=r.dtype)))
            maps[q]=basis;roots[q]=r
            error=max(error,float(np.max(abs(basis@r-matrix))))
        if error>2e-10*max(1.,float(np.linalg.norm(c))):
            raise ArithmeticError('upper/root factorization did not preserve the optimized center')
        return maps,roots,dict(maximum_factorization_error=error,no_singular_value_truncation=True)


def parent_anchored_center_ritz(operator: ThreeBlockSingletOperator, parent: np.ndarray, *,
                               krylov_vectors: int = 3,
                               initial_product: np.ndarray | None = None) -> tuple[np.ndarray,dict]:
    """One bounded exact-H joint-center Krylov step, always retaining the parent."""
    if isinstance(krylov_vectors,bool) or not isinstance(krylov_vectors,Integral) or not 1<=krylov_vectors<=8:
        raise ValueError('one to eight frozen parent-anchored Krylov vectors required')
    c=np.asarray(parent)
    if c.shape!=(operator.dimension,) or not np.isfinite(c).all() or np.linalg.norm(c)==0:
        raise ValueError('finite nonzero parent center required')
    c=c/np.linalg.norm(c);vectors=[];products=[];candidate=c
    cached=None if initial_product is None else np.asarray(initial_product)
    if cached is not None and (cached.shape!=c.shape or not np.isfinite(cached).all()):
        raise ValueError('cached product must be H acting on the NORMALIZED same parent')
    start=operator.matvec_count;tick=perf_counter()
    for iteration in range(krylov_vectors):
        vectors.append(candidate)
        h=cached if iteration==0 and cached is not None else operator.matvec(candidate)
        products.append(h)
        v=h.copy()
        for _ in range(2):
            for b in vectors:v-=b*np.vdot(b,v)
        norm=np.linalg.norm(v)
        if norm<2e-13*max(1.,np.linalg.norm(h)):break
        candidate=v/norm
    basis=np.column_stack(vectors);hb=np.column_stack(products)
    little=basis.conj().T@hb
    hermiticity=float(np.max(abs(little-little.conj().T)))
    if hermiticity>2e-9:raise ArithmeticError('joint-center effective H lost Hermiticity')
    vals,vecs=np.linalg.eigh((little+little.conj().T)/2)
    optimized=basis@vecs[:,0];hop=hb@vecs[:,0]
    overlap=np.vdot(c,optimized)
    if abs(overlap)>0:
        phase=np.conj(overlap)/abs(overlap);optimized*=phase;hop*=phase
    before=float(np.vdot(c,products[0]).real);after=float(np.vdot(optimized,hop).real)
    if after>before+2e-10:raise ArithmeticError('parent-feasible center Ritz energy increased')
    return optimized,dict(raw_energy_before=before,raw_energy_after=after,
        center_krylov_vectors_requested=krylov_vectors,center_krylov_vectors_used=len(vectors),
        center_H_matvecs=operator.matvec_count-start,
        previously_evaluated_parent_H_products=int(cached is not None),seconds=perf_counter()-tick,
        reduced_center_residual_squared=float(np.linalg.norm(hop-after*optimized)**2),
        physical_full_H_variance_known=False,full_A1_energy_known=False,
        parent_retained_in_trial_space=True,hermiticity_error=hermiticity,
        parent_center_overlap_squared=float(abs(np.vdot(c,optimized))**2))
