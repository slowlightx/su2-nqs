"""Matrix-free total-spin action on fixed-magnetization physical amplitudes.

Bits use 0=down, 1=up. ``target_m2=2M`` is an integer. No basis enumeration,
spin projector, symbolic table or Hamiltonian construction is required.
"""
from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import jax
import jax.numpy as jnp


def make_spin_squared_ratios(
    log_amplitude: Callable, *, n_sites: int, target_m2: int = 0,
    connector_batch_size: int = 4,
) -> Callable:
    """Return JIT batched ``(S^2 psi)(x)/psi(x)``; undefined at carrier nodes.

    S2(x,x)=N/2+M^2; each unlike-spin exchange has coefficient one.
    Packing up/down sites gives exactly n_up*n_down connectors per center,
    rather than evaluating all N(N-1)/2 pairs (including unchanged states).
    Runtime invalid configurations return NaN, never repaired samples.
    The complex local estimator need not be positive or real pointwise.
    """
    for name,value in (("n_sites",n_sites),("target_m2",target_m2),
                       ("connector_batch_size",connector_batch_size)):
        if isinstance(value,bool) or not isinstance(value,Integral):
            raise TypeError(f"{name} must be an integer")
    n_sites, target_m2 = int(n_sites),int(target_m2)
    if n_sites < 1 or abs(target_m2)>n_sites or (n_sites+target_m2)%2:
        raise ValueError("invalid N or doubled magnetization")
    if connector_batch_size < 1:
        raise ValueError("connector batch size must be positive")
    n_up=(n_sites+target_m2)//2
    n_down=n_sites-n_up

    def one(params, bits):
        if bits.shape != (n_sites,):
            raise ValueError("configuration shape disagrees with N")
        valid=jnp.all((bits==0)|(bits==1)) & (jnp.sum(bits)==n_up)
        diagonal=n_sites/2+target_m2**2/4
        if n_up==0 or n_down==0:
            return jnp.where(valid,diagonal+0j,jnp.nan+0j)
        up=jnp.nonzero(bits==1,size=n_up)[0]
        down=jnp.nonzero(bits==0,size=n_down)[0]
        pairs=jnp.stack((jnp.repeat(up,n_down),jnp.tile(down,n_up)),axis=1)
        center=log_amplitude(params,bits)

        def exchange(edge):
            swapped=bits.at[edge[0]].set(0).at[edge[1]].set(1)
            return jnp.exp(log_amplitude(params,swapped)-center)

        offdiagonal=jnp.sum(jax.lax.map(exchange,pairs,batch_size=connector_batch_size))
        return jnp.where(valid,diagonal+offdiagonal,jnp.nan+0j)

    return jax.jit(jax.vmap(one,in_axes=(None,0)))
