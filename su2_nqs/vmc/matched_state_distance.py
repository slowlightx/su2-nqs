"""Energy-blind ray calibration and common-parent candidate contrasts.

Distances are empirical sine Fubini--Study distances, not certified global
bounds. Independent audit samples must not be used by the ray calibration.
Both candidates in a contrast are reweighted from the SAME parent ensemble.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np
from .correlated_audit import correlated_summary


def ratio_distance(log_ratios: np.ndarray) -> float:
    """Return the gauge-invariant empirical sine FS distance on parent draws."""
    r = np.asarray(log_ratios, dtype=complex).ravel()
    if not len(r) or not np.isfinite(r).all():
        raise ValueError('nonempty finite log ratios required')
    z = np.exp(r-r.real.max())
    return float(np.sqrt(max(0., 1-min(1., abs(z.mean())**2/np.mean(abs(z)**2)))))


def match_ray_distance(log_ratio_at_scale: Callable[[float], np.ndarray], *,
                       target: float, bracket_scales: Sequence[float],
                       tolerance: float = 1e-6, iterations: int = 24) -> dict:
    """Find the first sampled bracket and bisect; never inspect any energies.

    The caller supplies a fixed direction from a frozen parent. No assumption
    of global monotonicity is made. Only the first grid crossing is refined;
    failure to meet tolerance inside this bounded budget is reported, not
    repaired by expanding the bracket or consulting an audit ensemble.
    """
    grid = np.asarray(bracket_scales, dtype=float)
    if (not 0 < target < 1 or not 0 < tolerance < target or iterations < 1
            or grid.ndim != 1 or len(grid) < 2 or grid[0] != 0
            or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0)):
        raise ValueError('positive target/tolerance and increasing grid from zero required')
    history = []

    def evaluate(scale: float) -> tuple[float, float]:
        d = ratio_distance(log_ratio_at_scale(float(scale)))
        history.append(dict(scale=float(scale), distance=d))
        return float(scale), d

    low, dlow = evaluate(0.)
    if dlow > tolerance:
        raise ValueError('ray origin is not the frozen parent')
    bracket = None
    for value in grid[1:]:
        high, dhigh = evaluate(value)
        if dhigh >= target:
            bracket = (low, high)
            break
        low, dlow = high, dhigh
    if bracket is not None:
        low, high = bracket
        for _ in range(iterations):
            if min(abs(h['distance']-target) for h in history) <= tolerance:
                break
            mid, distance = evaluate((low+high)/2)
            if distance < target:
                low = mid
            else:
                high = mid
    best = min(history, key=lambda h: abs(h['distance']-target))
    return dict(**best, target=float(target), error=abs(best['distance']-target),
                matched=abs(best['distance']-target) <= tolerance,
                evaluations=len(history), history=history)


def candidate_contrast(log_ratio_a: np.ndarray, energy_a: np.ndarray,
                       log_ratio_b: np.ndarray, energy_b: np.ndarray, *,
                       n_sites: int, reference_weights: np.ndarray | None = None) -> dict:
    """B minus A using separate self-normalizations against a common parent.

    In particular, A's local energies cannot be treated as an unweighted
    parent-energy ensemble: samples were drawn from the original parent.
    Optional exact parent weights are for exhaustive small-system tests.
    """
    a = correlated_summary(log_ratio_a, np.zeros_like(energy_a), energy_a,
                           n_sites=n_sites, reference_weights=reference_weights)
    b = correlated_summary(log_ratio_b, np.zeros_like(energy_b), energy_b,
                           n_sites=n_sites, reference_weights=reference_weights)
    return dict(energy_per_site=b['candidate_energy_per_site']-a['candidate_energy_per_site'],
                variance_per_site=b['candidate_variance_per_site']-a['candidate_variance_per_site'],
                sampled_fs_sine=b['sampled_fs_sine']-a['sampled_fs_sine'])


def paired_candidate_intervals(log_ratio_a: np.ndarray, energy_a: np.ndarray,
                               log_ratio_b: np.ndarray, energy_b: np.ndarray, *,
                               n_sites: int, seed: int, replicates: int = 4000,
                               family_tests: int = 24) -> dict:
    """Whole-chain bootstrap, paired across both directions and all snapshots.

    Arrays have shape (streams, snapshots, chains), each stream targeting
    the same parent state. Tail omission, non-equilibrium and finite-sample
    reweighting bias are not repaired by the confidence intervals.
    """
    arrays = [np.asarray(x) for x in (log_ratio_a, energy_a, log_ratio_b, energy_b)]
    shape = arrays[0].shape
    if (len(shape) != 3 or shape[-1] < 2 or any(x.shape != shape for x in arrays)
            or not all(np.isfinite(x).all() for x in arrays)
            or n_sites < 1 or replicates < 2 or family_tests < 1):
        raise ValueError('matching finite stream/time/chain arrays and valid budget required')
    point = candidate_contrast(*arrays, n_sites=n_sites)
    s, _, c = shape
    indices = np.random.default_rng(seed).integers(c, size=(replicates, s, c))
    boot = []
    for start in range(0, replicates, 64):
        idx = indices[start:start+64]
        la, ea, lb, eb = [np.stack([a[j][:, idx[:, j]].transpose(1, 0, 2)
                                    for j in range(s)], axis=1).reshape(len(idx), -1)
                         for a in arrays]
        moments = []
        for lr, e in ((la, ea), (lb, eb)):
            z = np.exp(lr-lr.real.max(axis=1, keepdims=True))
            w = abs(z)**2
            fs = np.sqrt(np.maximum(0., 1-np.minimum(1., abs(z.mean(axis=1))**2/w.mean(axis=1))))
            w /= w.sum(axis=1, keepdims=True)
            m = np.sum(w*e, axis=1)
            v = np.sum(w*abs(e-m[:, None])**2, axis=1)
            moments.append(np.stack((m.real/n_sites, v/n_sites, fs), axis=1))
        boot.append(moments[1]-moments[0])
    boot = np.concatenate(boot)
    alpha = .05/family_tests
    return {key: dict(difference=point[key], bootstrap_se=float(boot[:, i].std(ddof=1)),
                      paired_chain_95=np.quantile(boot[:, i], [.025, .975]).tolist(),
                      family_adjusted_interval=np.quantile(boot[:, i], [alpha/2, 1-alpha/2]).tolist(),
                      family_confidence=1-alpha)
            for i, key in enumerate(('energy_per_site', 'variance_per_site', 'sampled_fs_sine'))}
