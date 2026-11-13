"""Exact two-stage Metropolis screening with a frozen positive surrogate.

A cheap first stage may reject proposals without querying the physical target.
The second stage restores detailed balance for the full target; neither a
partial symmetry sum nor a learned approximation becomes the physical state.
The surrogate must stay fixed for each homogeneous chain segment. Efficiency
requires measuring effective samples per total cost, not query savings alone.
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np


def delayed_log_acceptance(log_surrogate_ratio: np.ndarray,
                           log_target_ratio: np.ndarray,
                           log_reverse_forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stage log probabilities for any reversible proposal pair.

    Ratios are trial/current; ``log_reverse_forward`` is log T(old|new)/T(new|old).
    Strict positivity of both densities is required; normalizers cancel.
    """
    s, t, r = np.broadcast_arrays(log_surrogate_ratio, log_target_ratio, log_reverse_forward)
    if not all(np.isfinite(x).all() for x in (s,t,r)):
        raise ValueError('finite log ratios of positive densities required')
    return np.minimum(0.,s+r), np.minimum(0.,t-s)


def screened_metropolis(log_current_target: np.ndarray,
                        log_current_surrogate: np.ndarray,
                        log_trial_surrogate: np.ndarray,
                        log_reverse_forward: np.ndarray,
                        evaluate_trial_target: Callable[[np.ndarray], np.ndarray],
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Screen a batch; callback evaluates full target ONLY at surviving indices.

    This function does not generate proposals or alter quantum labels. Returns
    accepted mask, updated full log scores, and exact query/work accounting.
    The caller updates configurations and surrogate scores with the same mask.
    """
    old, cheap, trial, ratio = map(lambda x: np.asarray(x,float),
                                  (log_current_target,log_current_surrogate,
                                   log_trial_surrogate,log_reverse_forward))
    if old.ndim != 1 or any(x.shape != old.shape for x in (cheap,trial,ratio)):
        raise ValueError('matching one-dimensional chain arrays required')
    if not all(np.isfinite(x).all() for x in (old,cheap,trial,ratio)):
        raise ValueError('finite strictly positive full and surrogate densities required')
    first = np.log(rng.random(len(old))) < np.minimum(0.,trial-cheap+ratio)
    indices = np.flatnonzero(first)
    full = np.asarray(evaluate_trial_target(indices),float) if len(indices) else np.empty(0)
    if full.shape != indices.shape or not np.isfinite(full).all():
        raise ValueError('target callback must return one finite log score per survivor')
    take = np.log(rng.random(len(indices))) < np.minimum(0.,full-old[indices]-(trial[indices]-cheap[indices]))
    accepted = np.zeros(len(old),bool); accepted[indices[take]] = True
    next_target = old.copy(); next_target[indices[take]] = full[take]
    return accepted,next_target,dict(proposals=len(old),full_target_queries=len(indices),
                                    accepted=int(accepted.sum()),surrogate_queries=len(old))
