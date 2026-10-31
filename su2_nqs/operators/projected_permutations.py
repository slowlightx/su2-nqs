"""Exact trivial-space-group projection of scalar permutation operators.

For an A1 carrier, P_G O |psi> = (|G|^-1 sum_g U_g O U_g^-1)|psi>.
Compile the conjugate operator once, combine identical site permutations,
and never insert a projector between the factors defining O. Site zero is
the most significant bit. No projection is approximated by sampled images.
"""

from collections.abc import Callable

import numpy as np


def reynolds_permutation_bank(permutations: np.ndarray, coefficients: np.ndarray, group: np.ndarray) -> tuple[np.ndarray,np.ndarray,dict]:
    """Return the complete group-averaged bank; check group closure exactly."""
    p,g=np.asarray(permutations),np.asarray(group)
    c=np.asarray(coefficients,dtype=float)
    if p.ndim!=2 or g.ndim!=2 or g.shape[1]!=p.shape[1] or c.ndim!=2 or len(c)!=len(p):
        raise ValueError("incompatible bank/group shapes")
    n=p.shape[1]
    if not np.issubdtype(p.dtype,np.integer) or not np.issubdtype(g.dtype,np.integer):
        raise TypeError("integer site maps required")
    if not np.all(np.sort(p,axis=1)==np.arange(n)) or not np.all(np.sort(g,axis=1)==np.arange(n)) or not np.isfinite(c).all():
        raise ValueError("invalid permutation or coefficients")
    members={tuple(row) for row in g}
    if len(members)!=len(g) or tuple(range(n)) not in members:
        raise ValueError("group needs distinct elements and identity")
    for left in g:
        if any(tuple(row) not in members for row in left[g]):
            raise ValueError("group is not closed")
    conjugates=np.concatenate([image[p[:,np.argsort(image)]] for image in g],axis=0)
    unique,inverse=np.unique(conjugates,axis=0,return_inverse=True)
    weights=np.tile(c,(len(g),1))/len(g)
    combined=np.column_stack([np.bincount(inverse,weights=weights[:,i],minlength=len(unique)) for i in range(c.shape[1])])
    # Only discard exact zero rows; no tolerance-dependent operator pruning.
    active=np.any(combined!=0,axis=1)
    return unique[active],combined[active],{"group_order":len(g),"raw_terms":len(conjugates),"unique_terms":int(np.sum(active))}


def disjoint_swap_masks(permutations: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Compile involutive site maps as disjoint swaps of MSB-first bits."""
    p=np.asarray(permutations)
    if p.ndim!=2 or not 1<=p.shape[1]<=64 or not np.issubdtype(p.dtype,np.integer):
        raise ValueError("integer site maps of size 1..64 required")
    n=p.shape[1]
    if not np.all(np.sort(p,axis=1)==np.arange(n)) or not np.all(np.take_along_axis(p,p,axis=1)==np.arange(n)):
        raise ValueError("only permutations made of disjoint swaps are supported")
    swaps=[[(i,int(j)) for i,j in enumerate(row) if i<j] for row in p]
    width=max(1,max(map(len,swaps)))
    first=np.zeros((len(p),width),dtype=np.uint64)
    second=np.zeros_like(first)
    for row,pairs in enumerate(swaps):
        for k,(i,j) in enumerate(pairs):
            first[row,k]=np.uint64(1)<<np.uint64(n-1-i)
            second[row,k]=np.uint64(1)<<np.uint64(n-1-j)
    return first,second


def evaluate_swap_bank(states: np.ndarray, amplitude_fn: Callable[[np.ndarray],np.ndarray], masks: tuple[np.ndarray,np.ndarray], coefficients: np.ndarray, *, sample_batch_size: int=1) -> tuple[np.ndarray,dict]:
    """Evaluate a bank of disjoint-swap words with exact per-batch deduplication."""
    s=np.asarray(states)
    first,second=masks
    c=np.asarray(coefficients)
    if s.ndim!=2 or len(s)<1 or not 1<=s.shape[1]<=64 or np.any((s!=0)&(s!=1)) or sample_batch_size<1:
        raise ValueError("nonempty bit samples and positive batch size required")
    if first.shape!=second.shape or first.ndim!=2 or c.ndim!=2 or len(c)!=len(first):
        raise ValueError("invalid masks/coefficient shapes")
    shifts=np.arange(s.shape[1]-1,-1,-1,dtype=np.uint64)
    powers=np.left_shift(np.uint64(1),shifts)
    result=[]; requests=unique_count=0
    for start in range(0,len(s),sample_batch_size):
        packed=np.sum(s[start:start+sample_batch_size].astype(np.uint64)*powers,axis=1)
        words=packed[:,None,None]
        different=((words&first)!=0)!=((words&second)!=0)
        flips=np.bitwise_xor.reduce(np.where(different,first^second,np.uint64(0)),axis=2)
        targets=packed[:,None]^flips
        unique,inverse=np.unique(targets,return_inverse=True)
        configs=((unique[:,None]>>shifts)&1).astype(np.int8)
        amplitudes=np.asarray(amplitude_fn(configs))
        if amplitudes.shape!=(len(unique),) or not np.isfinite(amplitudes).all():
            raise FloatingPointError("invalid carrier amplitudes")
        result.append(amplitudes[inverse].reshape(targets.shape)@c)
        requests+=targets.size; unique_count+=len(unique)
    return np.concatenate(result),{"raw_terminal_requests":requests,"unique_terminal_requests":unique_count}
