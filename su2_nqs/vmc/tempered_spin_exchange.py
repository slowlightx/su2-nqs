"""Replica exchange for a frozen fixed-magnetization Born distribution.

Replica beta samples |psi(x)|**(2*beta); beta=0 is uniform within the fixed
spin sector. This changes the Markov kernel, never the neural architecture.
Only beta=1 observations may estimate the physical Born distribution.
"""
from collections.abc import Callable, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from .physical_exchange import spin_exchange_proposals


def replica_swap_log_ratio(beta_left: float, beta_right: float,
                           log_weight_left: jax.Array,
                           log_weight_right: jax.Array) -> jax.Array:
    """Log Hastings ratio for swapping two replicas; log_weight=2 Re log psi."""
    return (beta_left-beta_right)*(log_weight_right-log_weight_left)


def make_tempered_spin_exchange(log_amplitude: Callable, betas: Sequence[float],
                                *, proposals: int, swap_every: int=8) -> Callable:
    """Return a JIT frozen-parameter replica-exchange transition.

    Input states have shape (replicas, independent_ladders, sites); lineage
    labels have shape (replicas, independent_ladders). Each replica attempts
    exactly `proposals` symmetric site exchanges, with adjacent even/odd swap
    sweeps every `swap_every` attempts. State exchange preserves each chain's
    doubled magnetization M2. The caller must initialize the correct sector.
    Diagnostics count nontrivial local moves and replica-swap acceptance.
    """
    b=np.asarray(betas,dtype=float)
    if b.ndim!=1 or len(b)<2 or not np.isfinite(b).all() or b[0]!=0 or b[-1]!=1 or not np.all(np.diff(b)>0):
        raise ValueError('betas must increase strictly from zero to one')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (proposals,swap_every)) or proposals%swap_every:
        raise ValueError('positive integer proposals must be divisible by swap_every')
    beta=jnp.asarray(b)[:,None];replicas=len(b)
    batched=jax.vmap(log_amplitude,in_axes=(None,0))

    @jax.jit
    def sample(params,key,states,lineages):
        if states.ndim!=3 or states.shape[0]!=replicas or lineages.shape!=states.shape[:2]:
            raise ValueError('replica state/lineage shape mismatch')
        chains=states.shape[1];shape=states.shape
        weights=2*batched(params,states.reshape(-1,shape[-1])).real.reshape(shape[:2])
        moves=jnp.zeros(replicas);swaps=jnp.zeros(replicas-1)
        def block(_,carry):
            rng,x,w,tags,moved,swapped=carry
            def local_step(_,inner):
                rng,x,w,moved=inner
                rng,proposal_key,accept_key=jax.random.split(rng,3)
                candidate=spin_exchange_proposals(proposal_key,x.reshape(-1,shape[-1])).reshape(shape)
                cw=2*batched(params,candidate.reshape(-1,shape[-1])).real.reshape(shape[:2])
                log_ratio=jnp.where(beta==0,0.,beta*(cw-w))
                accept=jnp.log(jax.random.uniform(accept_key,shape[:2]))<jnp.minimum(0.,log_ratio)
                changed=jnp.any(candidate!=x,axis=-1)
                return (rng,jnp.where(accept[...,None],candidate,x),jnp.where(accept,cw,w),
                        moved+jnp.sum(accept&changed,axis=1))
            rng,x,w,moved=jax.lax.fori_loop(0,swap_every,local_step,(rng,x,w,moved))
            for parity in (0,1):
                for left in range(parity,replicas-1,2):
                    right=left+1
                    rng,accept_key=jax.random.split(rng)
                    ratio=replica_swap_log_ratio(b[left],b[right],w[left],w[right])
                    accept=jnp.log(jax.random.uniform(accept_key,(chains,)))<jnp.minimum(0.,ratio)
                    xl,xr=x[left],x[right];wl,wr=w[left],w[right];tl,tr=tags[left],tags[right]
                    x=x.at[left].set(jnp.where(accept[:,None],xr,xl)).at[right].set(jnp.where(accept[:,None],xl,xr))
                    w=w.at[left].set(jnp.where(accept,wr,wl)).at[right].set(jnp.where(accept,wl,wr))
                    tags=tags.at[left].set(jnp.where(accept,tr,tl)).at[right].set(jnp.where(accept,tl,tr))
                    swapped=swapped.at[left].add(jnp.sum(accept))
            return rng,x,w,tags,moved,swapped
        key,states,_,lineages,moves,swaps=jax.lax.fori_loop(
            0,proposals//swap_every,block,(key,states,weights,lineages,moves,swaps))
        return states,key,lineages,dict(local_move_fraction=moves/(chains*proposals),
            swap_acceptance=swaps/(chains*(proposals//swap_every)))
    return sample
