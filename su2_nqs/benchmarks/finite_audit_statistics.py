"""Paired finite-state uncertainty and explicit baseline-tail accounting."""

import numpy as np

from .finite_scalar_dressing import finite_step_moments

FIELDS = ("psi", "direction", "hpsi", "hdirection", "proposal_squared")
METRICS = ("norm_ratio", "energy_total", "variance_total", "energy_change_total", "variance_change_total", "carrier_fidelity")


def paired_chain_summary(data: dict, chain_ids: np.ndarray, epsilon: float, *, replicates: int, seed: int) -> dict:
    """Bootstrap complete persistent chains and report leave-one-chain sensitivity."""
    chains = np.asarray(chain_ids)
    if chains.shape != np.asarray(data["psi"]).shape or replicates < 2:
        raise ValueError("matching chain identifiers and at least two resamples required")
    groups = [np.flatnonzero(chains == c) for c in np.unique(chains)]
    if len(groups) < 2:
        raise ValueError("at least two chains required")
    def moments(ix):
        return finite_step_moments(*(np.asarray(data[k])[ix] for k in FIELDS),epsilon)
    full = moments(np.arange(len(chains)))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(replicates):
        ix = np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        m = moments(ix);boot.append([m[k] for k in METRICS])
    boot = np.asarray(boot)
    full["paired_chain_bootstrap_se"] = dict(zip(METRICS,map(float,boot.std(axis=0,ddof=1))))
    full["bootstrap_percentile_95"] = {k:np.quantile(boot[:,i],[.025,.975]).tolist() for i,k in enumerate(METRICS)}
    leave = [moments(chains != c) for c in np.unique(chains)]
    full["leave_one_chain_out"] = {k:{"minimum":min(m[k] for m in leave),"maximum":max(m[k] for m in leave)} for k in ("energy_change_total","variance_change_total")}
    full["sample_count"],full["independent_chain_count"] = len(chains),len(groups)
    return full


def residual_tail_diagnostics(data: dict, epsilon: float, *, reference_energy: float, thresholds: np.ndarray) -> dict:
    """Decompose actual paired changes by pre-existing carrier-residual strata.

    No trimming is used for the physical estimate. Deletions below are only
    a sensitivity diagnostic and do not define an alternative reported state.
    """
    p,d,h,hd,q = (np.asarray(data[k]) for k in FIELDS)
    cut = np.asarray(thresholds)
    if cut.ndim != 1 or np.any(np.diff(cut)<0):
        raise ValueError("ordered tail thresholds required")
    x,y = p+epsilon*d,h+epsilon*hd
    w0,w1 = abs(p)**2/q,abs(x)**2/q
    z0,z1 = w0.sum(),w1.sum()
    e0,e1 = np.sum(p.conj()*h/q).real/z0,np.sum(x.conj()*y/q).real/z1
    r0,r1 = abs(h-e0*p)**2/q,abs(y-e1*x)**2/q
    de = (x.conj()*y/q).real/z1-(p.conj()*h/q).real/z0
    dv = r1/z1-r0/z0
    score = abs(h-reference_energy*p)**2/q
    labels = np.searchsorted(cut,score,side="right")
    strata = []
    for i in range(len(cut)+1):
        mask = labels==i
        strata.append({"stratum":i,"count":int(mask.sum()),"carrier_variance_contribution":float(r0[mask].sum()/z0),
                       "energy_change_contribution":float(de[mask].sum()),"variance_change_contribution":float(dv[mask].sum())})
    top_count = max(1,int(np.ceil(.01*len(p))))
    keep = np.ones(len(p),bool);keep[np.argsort(score)[-top_count:]]=False
    trimmed = finite_step_moments(*(np.asarray(data[k])[keep] for k in FIELDS),epsilon)
    return {"strata":strata,"reference_energy":float(reference_energy),"thresholds":cut.tolist(),
            "largest_one_percent_carrier_variance_fraction":float(np.sort(r0)[-top_count:].sum()/r0.sum()),
            "largest_single_carrier_variance_fraction":float(r0.max()/r0.sum()),
            "tail_deletion_diagnostic_only":{"removed_count":top_count,"energy_change_total":trimmed["energy_change_total"],"variance_change_total":trimmed["variance_change_total"]}}
