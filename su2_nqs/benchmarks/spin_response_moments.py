"""Node-faithful energy and total-spin moments of diagonal spin-source states.

Use physical bits (0=down, site zero MSB). A uniformly selected involution
from identity and ALL spin transpositions pushes parent Born samples to Q.
Q covers both H O|psi> and S_tot^2 O|psi>, even at exact parent nodes.
No approximate-parent first-moment correction is discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Sequence

import numpy as np


def _bits(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states)
    if (x.ndim != 2 or len(x) == 0 or x.shape[1] < 2 or x.dtype.kind not in 'biu'
            or not np.isin(x, (0, 1)).all() or np.any(x.sum(1) != x[0].sum())):
        raise ValueError('nonempty fixed-magnetization physical bit matrix required')
    return x


def transpose_born_samples(states: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Draw Q exactly: one uniform identity/all-pairs swap per parent Born draw.

    At half filling Q >= [1+2*C(N/2,2)]/[1+C(N,2)] p. This is not an MCMC
    equilibration step and has no burn-in; validity assumes the input Born law.
    """
    x = _bits(states).copy(); pairs = list(combinations(range(x.shape[1]), 2))
    choices = generator.integers(len(pairs)+1, size=len(x))
    for k, choice in enumerate(choices):
        if choice:
            i, j = pairs[choice-1]; x[k, [i, j]] = x[k, [j, i]]
    return x


@dataclass(frozen=True)
class SwapMomentPlan:
    """Unique amplitude queries and identity/lexicographic-pair inverse map."""
    configurations: np.ndarray
    indices: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    n_sites: int


def swap_moment_plan(states: np.ndarray) -> SwapMomentPlan:
    """Deduplicate the complete one-swap neighborhood; never drop self-swaps."""
    x = _bits(states); n = x.shape[1]
    pairs = tuple(combinations(range(n), 2))
    rows = np.repeat(x[:, None, :], len(pairs)+1, axis=1)
    for k, (i, j) in enumerate(pairs, 1):
        rows[:, k, [i, j]] = x[:, [j, i]]
    unique, inverse = np.unique(rows.reshape(-1, n), axis=0, return_inverse=True)
    return SwapMomentPlan(unique, inverse.reshape(len(x), -1), pairs, n)


def spin_source_moments(plan: SwapMomentPlan, log_amplitudes: np.ndarray,
                        bonds: Sequence[tuple[int, int, float]],
                        sources: dict[str, np.ndarray], *, energy_shift: float = 0.,
                        target_q: int = 2) -> dict[str, np.ndarray]:
    """Importance moments of psi and O psi, with O=sum_i phase_i S_i^z/sqrt(N).

    Input samples must follow Q, not p. Returned unnormalized moments average
    to physical norm, <H-shift>, ||(H-shift)phi||^2, <S^2>, and
    ||[S^2-q(q+2)/4]phi||^2. q=2S is doubled; the parent uses q=0.
    A response energy variance is not a excitation-gap estimator.
    """
    n = plan.n_sites
    if (isinstance(target_q, bool) or not isinstance(target_q, Integral) or target_q < 0
            or target_q > n or target_q % 2 != n % 2 or not np.isfinite(energy_shift)):
        raise ValueError('valid doubled target_q and finite energy shift required')
    if not sources or 'parent' in sources:
        raise ValueError('nonempty source bank; parent is reserved')
    for phase in sources.values():
        if np.asarray(phase).shape != (n,) or not np.isfinite(phase).all():
            raise ValueError('one finite source coefficient per physical site required')
    logs = np.asarray(log_amplitudes, complex)
    if (logs.shape != (len(plan.configurations),) or np.isnan(logs).any()
            or np.isposinf(logs.real).any() or not np.isfinite(logs.imag).all()):
        raise ValueError('finite phases and finite or negative-infinite log magnitudes required')
    log = logs[plan.indices]; maximum = np.max(log.real, axis=1)
    if not np.isfinite(maximum).all():
        raise ValueError('Q has zero support on a requested record')
    amplitudes = np.exp(log-maximum[:, None])
    density = np.mean(abs(amplitudes)**2, axis=1)
    hc = np.zeros(len(plan.pairs)+1); hc[0] = -energy_shift
    lookup = {p:i+1 for i,p in enumerate(plan.pairs)}
    for i,j,coupling in bonds:
        if (isinstance(i,bool) or isinstance(j,bool) or not isinstance(i,Integral)
                or not isinstance(j,Integral) or not 0 <= i < n or not 0 <= j < n
                or i == j or not np.isfinite(coupling)):
            raise ValueError('valid distinct physical bond sites and finite exchange required')
        hc[lookup[tuple(sorted((i,j)))]] += .5*coupling
        hc[0] -= .25*coupling
    spin = np.ones(len(hc)); spin[0] = n*(4-n)/4
    output = {}; parent_h = amplitudes@hc
    for name, phase in [('parent', None), *sources.items()]:
        values = (np.ones(len(plan.configurations)) if phase is None else
                  (plan.configurations-.5)@np.asarray(phase)/np.sqrt(n))
        phi = amplitudes*values[plan.indices]; center = phi[:,0]
        hphi = phi@hc; sphi = phi@spin
        # Separate genuine operator action from the approximate parent's
        # residual: (H-shift)Opsi = [H,O]psi + O(H-shift)psi.
        weighted_parent = values[plan.indices[:,0]]*parent_h
        commutator = hphi-weighted_parent
        casimir = 0. if phase is None else target_q*(target_q+2)/4
        moments = dict(norm=abs(center)**2, first=(center.conj()*hphi).real,
                       first_imag=(center.conj()*hphi).imag, second=abs(hphi)**2,
                       spin=(center.conj()*sphi).real,
                       spin_defect=abs(sphi-casimir*center)**2,
                       commutator_first=(center.conj()*commutator).real,
                       commutator_second=abs(commutator)**2,
                       parent_residual_first=(center.conj()*weighted_parent).real,
                       parent_residual_second=abs(weighted_parent)**2,
                       commutator_parent_cross=2*(commutator.conj()*weighted_parent).real)
        output.update({name+'__'+k: v/density for k,v in moments.items()})
    return output
