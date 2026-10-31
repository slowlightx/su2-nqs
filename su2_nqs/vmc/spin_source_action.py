"""Exact physical spin-source states on an already coherent singlet carrier.

No CT tensor or production default is modified.  From a genuine singlet,
``O_k=sum_i phase_i S_i^z/sqrt(N)`` creates doubled ``(q,m2)=(2,0)``
unless it annihilates the carrier.  The returned H action includes BOTH
``[H,O]psi`` and ``O Hpsi``; an approximate carrier is never treated as an
eigenstate.  Full support is supplied separately by orbit/bridge/uniform Q.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np


def make_spin_source_products(
    query_factory: Callable, images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], probabilities: Sequence[float],
    phases: np.ndarray, *, n_sites: int, projection_chunk: int = 4,
) -> tuple[Callable, dict]:
    """Build shared exact ``psi,Hpsi,Qorbit,Qpush`` for parent and spin seeds.

    ``query_factory(edges)(bank,bits)`` returns raw psi and swapped psi in the
    supplied edge order.  ``phases`` has shape ``[channels,N]``.  Output psi
    and Hpsi have shape ``[batch,1+channels]``, parent first.  Qorbit/Qpush
    refer to the NORMALIZED RAW BANK, not the coherent target.  Use a positive
    full-support proposal to integrate second moments at target/carrier nodes.
    The factory does not infer strict SU(2) from arbitrary input amplitudes.
    """
    if (isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2
            or isinstance(projection_chunk, bool) or not isinstance(projection_chunk, Integral)
            or projection_chunk < 1):
        raise ValueError('positive integer site count/chunk required')
    maps=np.asarray(images); phase=np.asarray(phases, np.complex128)
    if (maps.ndim != 2 or maps.shape[1] != n_sites or not len(maps)
            or not np.issubdtype(maps.dtype,np.integer)
            or not np.all(np.sort(maps,axis=1)==np.arange(n_sites))
            or len(maps)%projection_chunk):
        raise ValueError('complete physical permutation rows and dividing chunk required')
    if phase.ndim != 2 or phase.shape[1] != n_sites or not len(phase) or not np.isfinite(phase).all():
        raise ValueError('finite complex source coefficient matrix required')
    if not bonds:
        raise ValueError('nonempty exchange graph required')
    edges=np.asarray([(a,b) for a,b,_ in bonds])
    coupling=np.asarray([c for _,_,c in bonds],float); prob=np.asarray(probabilities,float)
    if (edges.dtype.kind not in 'iu' or np.any(edges<0) or np.any(edges>=n_sites)
            or np.any(edges[:,0]==edges[:,1]) or not np.isfinite(coupling).all()
            or prob.shape != coupling.shape or not np.isfinite(prob).all() or np.any(prob<0)
            or not np.isclose(prob.sum(),1.,rtol=0,atol=1e-13)):
        raise ValueError('legal exchange edges, couplings and normalized proposal weights required')
    combined={}
    for edge,c,p in zip(edges,coupling,prob,strict=True):
        key=tuple(sorted(map(int,edge))); old=combined.get(key,(0.,0.))
        combined[key]=(old[0]+c,old[1]+p)
    edges=np.asarray(sorted(combined),np.int32)
    coupling=np.asarray([combined[tuple(e)][0] for e in edges])
    prob=np.asarray([combined[tuple(e)][1] for e in edges])
    if np.any((coupling != 0)&(prob<=0)):
        raise ValueError('positive exchange-bridge support required for every H edge')
    inverse=np.argsort(maps,axis=1)
    image_edges=np.sort(inverse[:,edges],axis=-1)
    queries,reverse=np.unique(image_edges.reshape(-1,2),axis=0,return_inverse=True)
    reindex=reverse.reshape(len(maps),len(edges))
    query=query_factory(queries.astype(np.int32))
    groups=jnp.asarray(maps.reshape(-1,projection_chunk,n_sites))
    indices=jnp.asarray(reindex.reshape(-1,projection_chunk,len(edges)))
    ed,hw,pw,coeff=map(jnp.asarray,(edges,coupling,prob,phase))

    def one(bank,bits):
        # Samplers may store uint8 bits; signed differences must not wrap.
        bits=jnp.asarray(bits,dtype=jnp.int32)
        def image_value(image,index):
            center,changed=query(bank,bits[image])
            return jnp.concatenate((jnp.asarray(center)[None],changed[index]))
        def partial(items):
            image,index=items; v=jax.vmap(image_value)(image,index)
            return jnp.sum(v,axis=0),jnp.sum(abs(v)**2,axis=0)
        coherent,power=jax.lax.map(jax.checkpoint(partial),(groups,indices))
        coherent=jnp.sum(coherent,axis=0)/len(maps)
        power=jnp.sum(power,axis=0)/len(maps)
        parent,swapped=coherent[0],coherent[1:]
        source=coeff@(bits-.5)/np.sqrt(n_sites)
        # O(T_ab x)-O(x), using PHYSICAL rather than image-frame sites.
        difference=(bits[ed[:,1]]-bits[ed[:,0]])[None,:]
        changed_source=source[:,None]+difference*(coeff[:,ed[:,0]]-coeff[:,ed[:,1]])/np.sqrt(n_sites)
        phi=jnp.concatenate((jnp.asarray(parent)[None],source*parent))
        changed=jnp.concatenate((swapped[None,:],changed_source*swapped[None,:]),axis=0)
        unlike=bits[ed[:,0]] != bits[ed[:,1]]
        diagonal=jnp.sum(hw*jnp.where(unlike,-.25,.25))
        hphi=diagonal*phi+jnp.sum(.5*hw[None,:]*jnp.where(unlike[None,:],changed,0.),axis=1)
        return phi,hphi,power[0],jnp.dot(pw,power[1:])

    products=jax.jit(lambda bank,rows:jax.lax.map(lambda x:one(bank,x),rows))
    return products,dict(n_sites=int(n_sites),channels=len(phase),images=len(maps),
        unique_physical_exchange_edges=len(edges),unique_raw_exchange_queries=len(queries),
        both_commutator_and_carrier_residual=True,carrier_eigenstate_assumed=False,
        second_moment_requires_full_support=True,strict_spin_requires_singlet_carrier=True)
