"""Independent-replica SWAP estimator for positive-amplitude NQS Renyi-2."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def swap_ratios(log_amplitude: Callable[[jax.Array], jax.Array], first: jax.Array,
                second: jax.Array, ell: int) -> jax.Array:
    """Replica SWAP ratios; both replicas must be independent Born draws.

    Returns one positive ratio per pair. This estimator's expectation is
    Tr(rho_A^2); log of a finite-sample mean is biased. Sampling uncertainty
    must be reported separately and checked as the replica count increases.
    """
    first,second=jnp.asarray(first),jnp.asarray(second)
    if first.ndim!=2 or first.shape!=second.shape or not isinstance(ell,int) or not 1<=ell<first.shape[1]:
        raise ValueError("need matching replica batches and 1<=ell<n")
    swapped_first=jnp.concatenate((second[:,:ell],first[:,ell:]),axis=1)
    swapped_second=jnp.concatenate((first[:,:ell],second[:,ell:]),axis=1)
    return jnp.exp(log_amplitude(swapped_first)+log_amplitude(swapped_second)
                   -log_amplitude(first)-log_amplitude(second))


def summarize_swap(ratios: np.ndarray) -> dict:
    """Estimate S2 and its delta-method SE, retaining raw SWAP statistics."""
    ratios=np.asarray(ratios,dtype=np.float64)
    if ratios.ndim!=1 or len(ratios)<2 or not np.all(np.isfinite(ratios)) or np.any(ratios<0) or ratios.mean()<=0:
        raise ValueError("need >=2 finite nonnegative ratios with positive mean")
    mean=float(ratios.mean());se=float(ratios.std(ddof=1)/np.sqrt(len(ratios)))
    return {"swap_mean":mean,"swap_se":se,"renyi2":-float(np.log(mean)),
            "renyi2_se_delta_method":se/mean,"replica_pairs":len(ratios),
            "relative_se":se/mean,"mean_in_physical_range":0<mean<=1}
