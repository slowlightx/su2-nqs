"""Exact low-rank evaluation of scalar swap words on closed-shell carriers.

A disjoint up/down exchange replaces one row in each occupied-spin Slater
matrix. With r exchanges the physical spin amplitude is (-1)^r times two
r-by-r determinant ratios times the original amplitude. No division by the
coherent carrier is made. Ill-conditioned component matrices fall back to
the direct amplitude evaluator; carrier nodes are never floored.
"""

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from su2_nqs.operators.projected_permutations import disjoint_swap_masks, evaluate_swap_bank


class ClosedShellSwapEvaluator:
    """Batched real closed-shell scalar-word actions; up to three swaps/word."""

    def __init__(self, orbitals: np.ndarray, carrier_weights: np.ndarray, permutations: np.ndarray,
                 coefficients: np.ndarray, direct_amplitudes: Callable[[np.ndarray],np.ndarray], *,
                 chunk_size: int=4096, condition_limit: float=1e9) -> None:
        v=np.asarray(orbitals,dtype=float)
        p=np.asarray(permutations)
        c=np.asarray(coefficients,dtype=float)
        w=np.asarray(carrier_weights,dtype=float)
        if v.ndim!=3 or v.shape[1]!=2*v.shape[2] or w.shape!=(len(v),) or p.ndim!=2 or p.shape[1]!=v.shape[1] or c.ndim!=2 or len(c)!=len(p):
            raise ValueError("incompatible closed-shell carrier/bank shapes")
        if chunk_size<1 or condition_limit<=1 or not all(np.isfinite(x).all() for x in (v,c,w)):
            raise ValueError("invalid numeric inputs")
        self.masks=disjoint_swap_masks(p)
        pairs=[[(i,int(j)) for i,j in enumerate(row) if i<j] for row in p]
        if max(map(len,pairs))>3:
            raise ValueError("at most three disjoint swaps per word")
        self.left=np.zeros((len(p),3),dtype=np.int32)
        self.right=np.zeros_like(self.left)
        for row,items in enumerate(pairs):
            for k,(i,j) in enumerate(items):
                self.left[row,k],self.right[row,k]=i,j
        self.v,self.w,self.c,self.direct=v,w,c,direct_amplitudes
        self.chunk_size,self.condition_limit=chunk_size,condition_limit

        @jax.jit
        def kernel(state,up_slot,down_slot,up_ratio,down_ratio,base,left,right,coeff):
            active=state[left]!=state[right]
            old_up=jnp.where(state[left]==1,left,right)
            old_down=jnp.where(state[left]==0,left,right)
            # All padding and like-spin exchanges become identity rows/columns.
            up_indices=up_slot[old_up]
            down_indices=down_slot[old_down]
            u=up_ratio[:,old_down[:,:,None],up_indices[:,None,:]]
            d=down_ratio[:,old_up[:,:,None],down_indices[:,None,:]]
            keep=active[:,:,None]&active[:,None,:]
            u=jnp.where(keep[None],u,jnp.eye(3))
            d=jnp.where(keep[None],d,jnp.eye(3))
            def det3(x):
                return (x[...,0,0]*(x[...,1,1]*x[...,2,2]-x[...,1,2]*x[...,2,1])
                        -x[...,0,1]*(x[...,1,0]*x[...,2,2]-x[...,1,2]*x[...,2,0])
                        +x[...,0,2]*(x[...,1,0]*x[...,2,1]-x[...,1,1]*x[...,2,0]))
            sign=jnp.where(jnp.sum(active,axis=1)%2==0,1.,-1.)
            amplitudes=sign*jnp.sum(base[:,None]*det3(u)*det3(d),axis=0)
            return amplitudes@coeff
        self.kernel=kernel

    def __call__(self, configurations: np.ndarray) -> tuple[np.ndarray,dict]:
        """Return exact bank images and fallback/work accounting."""
        states=np.asarray(configurations)
        n=self.v.shape[1]; half=n//2
        if states.ndim!=2 or states.shape[1]!=n or np.any((states!=0)&(states!=1)) or np.any(states.sum(axis=1)!=half):
            raise ValueError("half-filled bit configurations required")
        outputs=[]; fallback=0; max_condition=0.
        for state in states:
            up=np.flatnonzero(state); down=np.flatnonzero(1-state)
            u,d=self.v[:,up],self.v[:,down]
            condition=float(max(np.max(np.linalg.cond(u)),np.max(np.linalg.cond(d))))
            max_condition=max(max_condition,condition)
            if not np.isfinite(condition) or condition>self.condition_limit:
                value,_=evaluate_swap_bank(state[None],self.direct,self.masks,self.c)
                outputs.append(value[0]);fallback+=1
                continue
            up_ratio=self.v@np.linalg.inv(u)
            down_ratio=self.v@np.linalg.inv(d)
            # Spin-ordering parity and the singlet-Pfaffian parity cancel
            # their constant half*(half-1)/2 terms.
            parity=int(np.sum(up))%2
            base=(((-1.)**parity)*np.linalg.det(u)*np.linalg.det(d))*self.w
            up_slot=np.zeros(n,dtype=np.int32);down_slot=np.zeros(n,dtype=np.int32)
            up_slot[up]=np.arange(half);down_slot[down]=np.arange(half)
            total=np.zeros(self.c.shape[1])
            for start in range(0,len(self.c),self.chunk_size):
                size=min(self.chunk_size,len(self.c)-start)
                left=np.zeros((self.chunk_size,3),dtype=np.int32);right=np.zeros_like(left)
                coeff=np.zeros((self.chunk_size,self.c.shape[1]))
                left[:size]=self.left[start:start+size];right[:size]=self.right[start:start+size];coeff[:size]=self.c[start:start+size]
                total+=np.asarray(self.kernel(state,up_slot,down_slot,up_ratio,down_ratio,base,left,right,coeff))
            outputs.append(total)
        return np.asarray(outputs),{"sample_count":len(states),"word_count":len(self.c),"direct_fallback_count":fallback,"maximum_component_condition":max_condition}
