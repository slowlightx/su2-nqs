"""Fixed-parent correlated sampling; no sampler, optimizer or model decisions.

The input configurations must sample the parent Born distribution. Biased
diagnostic configurations must never be passed as an energy-audit ensemble.
Finite-sample self-normalization and missing-support bias are not removed by
the paired whole-chain bootstrap implemented here.
"""
from __future__ import annotations
import numpy as np


def correlated_summary(log_ratios: np.ndarray, parent_energies: np.ndarray,
                       candidate_energies: np.ndarray, *, n_sites: int,
                       reference_weights: np.ndarray | None = None) -> dict:
    """Estimate candidate moments and physical distances on parent samples.

    ``log_ratios=log(psi_candidate/psi_parent)``; both energies are full-H
    local ratios at the same configurations. Optional exact reference weights
    support small-system validation, not an empirical correction for mixing.
    """
    r=np.asarray(log_ratios,dtype=complex).ravel()
    e0=np.asarray(parent_energies,dtype=complex).ravel();e1=np.asarray(candidate_energies,dtype=complex).ravel()
    if n_sites<1 or not len(r) or r.shape!=e0.shape or r.shape!=e1.shape or not all(np.isfinite(v).all() for v in (r,e0,e1)):
        raise ValueError('finite matching ratios/energies and positive site count required')
    q=np.ones(len(r)) if reference_weights is None else np.asarray(reference_weights,dtype=float).ravel()
    if q.shape!=r.shape or not np.isfinite(q).all() or np.any(q<0) or q.sum()<=0:
        raise ValueError('reference weights must be finite, nonnegative, with positive sum')
    q=q/q.sum();shift=r.real.max();amplitude=np.exp(r-shift);w=abs(amplitude)**2
    z=q@w;p=q*w/z;m0=q@e0;m1=p@e1
    v0=q@abs(e0-m0)**2;v1=p@abs(e1-m1)**2
    log_z=float(np.log(z)+2*shift)
    return dict(parent_energy_per_site=float(m0.real/n_sites),candidate_energy_per_site=float(m1.real/n_sites),
        parent_variance_per_site=float(v0/n_sites),candidate_variance_per_site=float(v1/n_sites),
        delta_energy_per_site=float((m1-m0).real/n_sites),delta_variance_per_site=float((v1-v0)/n_sites),
        candidate_imaginary_energy_per_site=float(m1.imag/n_sites),
        sampled_fs_sine=float(np.sqrt(max(0.,1-min(1.,abs(q@amplitude)**2/z)))),
        reweighting_ess_fraction=float(z*z/(q@(w*w))),
        sampled_forward_kl=float(log_z-q@(2*r.real)),log_norm_ratio_estimate=log_z,
        maximum_normalized_weight=float(p.max()),
        top_one_percent_weight_share=float(np.sort(p)[-max(1,int(np.ceil(len(p)*.01))):].sum()))


def paired_chain_intervals(log_ratios: np.ndarray, parent_energies: np.ndarray,
                           candidate_energies: np.ndarray, *, n_sites: int,
                           seed: int, replicates: int=4000, family_tests: int=16) -> dict:
    """Resample whole chains within each stream, paired across both states.

    Inputs have shape (independent streams, time snapshots, chains). Streams
    may use different stationary kernels targeting the SAME parent state.
    No ratio clipping, energy clipping, chain removal or bootstrap selection.
    """
    arrays=[np.asarray(x) for x in (log_ratios,parent_energies,candidate_energies)]
    shape=arrays[0].shape
    if len(shape)!=3 or shape[-1]<2 or any(x.shape!=shape for x in arrays) or replicates<2 or family_tests<1:
        raise ValueError('matching stream/time/chain arrays and valid bootstrap budget required')
    if not all(np.isfinite(x).all() for x in arrays):raise ValueError('nonfinite correlated audit')
    rng=np.random.default_rng(seed);s,_,c=shape;indices=rng.integers(c,size=(replicates,s,c));boot=[]
    for start in range(0,replicates,64):
        idx=indices[start:start+64]
        lr,e0,e1=[np.stack([a[j][:,idx[:,j]].transpose(1,0,2) for j in range(s)],axis=1).reshape(len(idx),-1) for a in arrays]
        w=np.exp(2*(lr.real-lr.real.max(axis=1,keepdims=True)));w/=w.sum(axis=1,keepdims=True)
        m0=e0.mean(axis=1);m1=np.sum(w*e1,axis=1)
        v0=np.mean(abs(e0-m0[:,None])**2,axis=1);v1=np.sum(w*abs(e1-m1[:,None])**2,axis=1)
        boot.append(np.stack(((m1-m0).real/n_sites,(v1-v0)/n_sites),axis=1))
    boot=np.concatenate(boot);alpha=.05/family_tests
    point=correlated_summary(*arrays,n_sites=n_sites)
    return {key:dict(difference=point['delta_'+key],bootstrap_se=float(boot[:,i].std(ddof=1)),
                    paired_chain_95=np.quantile(boot[:,i],[.025,.975]).tolist(),
                    family_adjusted_interval=np.quantile(boot[:,i],[alpha/2,1-alpha/2]).tolist(),
                    family_confidence=1-alpha)
            for i,key in enumerate(('energy_per_site','variance_per_site'))}
