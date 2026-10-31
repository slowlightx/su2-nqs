"""Small-system exhaustive and large-system Born audits for pilot endpoints."""
from __future__ import annotations
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from fnqs.basis import all_bits
from fnqs.diagnostics.response import response_records, bootstrap_response, summarize_response
from fnqs.hamiltonians import IsingCouplings
from fnqs.observables import correlations_from_state, sampled_correlations
from fnqs.sectors import parity_reference
from fnqs.variants.size_features import log_psi, SizeFeatureConfig
from .engine import draw


def action(vector: np.ndarray, n: int, g: np.ndarray) -> np.ndarray:
    """Apply all Pauli terms in MSB integer order; guard exhaustive n<=14."""
    bits = all_bits(n)
    v, g = np.asarray(vector), np.asarray(g, dtype=float)
    if v.shape != (1 << n,) or g.shape != (4,) or not np.isfinite(v).all() or not np.isfinite(g).all():
        raise ValueError("invalid exhaustive vector/couplings")
    z = 2*bits-1
    result = (-g[0]*np.sum(z*np.roll(z, -1, axis=1), axis=1)
              -g[1]*np.sum(z*np.roll(z, -2, axis=1), axis=1))*v
    labels = np.arange(1 << n)
    for site in range(n):
        mask = 1 << (n-1-site)
        pair = mask ^ (1 << (n-1-(site+1)%n))
        result -= g[3]*v[labels^mask]+g[2]*v[labels^pair]
    return result


@partial(jax.jit, static_argnames=("config",))
def amplitudes(params, bits, g, config):
    return jax.jvp(lambda x: log_psi(params, bits, x, config), (g,), (jnp.array([0.,0.,0.,1.]),))


@partial(jax.jit, static_argnames=("config",))
def sampled_records(params, bits, g, config):
    return response_records(params, bits, g, lambda p,b,x: log_psi(p,b,x,config))


def reference(n: int, g: list[float]) -> dict:
    """Independent parity-resolved ED/linear response, with original guards."""
    ref = parity_reference(n, IsingCouplings(*g))
    return {"energy_total": ref.energy, "chi_total": ref.chi_total,
            "c_half": float(correlations_from_state(ref.state, n)[n//2]),
            "state": ref.state, "state_residual": ref.state_residual,
            "response_relative_residual": ref.response_relative_residual}


def exact_audit(params: dict, n: int, g: list[float], config: SizeFeatureConfig,
                ref: dict) -> dict:
    """Evaluate all Born rows; no reference labels enter optimization."""
    bits = all_bits(n)
    logs, score = amplitudes(params, jnp.asarray(bits), jnp.asarray(g), config)
    psi, score = np.exp(np.asarray(logs)), np.asarray(score)
    probability = psi**2
    normalization = float(probability.sum())
    if abs(normalization-1) > 1e-10:
        raise RuntimeError("exhaustive normalization failed")
    rows = {"score": score, "energy": action(psi,n,g)/psi,
            "h_score": action(psi*score,n,g)/psi,
            "source": action(psi,n,[0,0,0,1])/psi, "fd_scores": score[:,None]}
    result = summarize_response(rows, probability)
    corr = float(correlations_from_state(psi,n)[n//2])
    result.update(n=n, couplings=g, normalization_error=abs(normalization-1),
                  state_infidelity=float(max(0.,1-abs(np.vdot(ref["state"],psi))**2)),
                  chi_relative_error=abs(result["chi_total"]/ref["chi_total"]-1),
                  energy_per_site_error=abs(result["energy_total"]-ref["energy_total"])/n,
                  c_half=corr, c_half_absolute_error=abs(corr-ref["c_half"]),
                  reference_chi_total=ref["chi_total"], born_samples=0, basis_rows=len(bits),
                  hamiltonian_actions=2, source_actions=1, finite_difference_measured=False)
    return result


def born_audit(params: dict, n: int, g: list[float], config: SizeFeatureConfig,
               seed: int, samples: int, ref: dict, archive: Path | None = None) -> dict:
    """Paired-row bootstrap retains response/chi and half-ring correlation SE."""
    bits = draw(params,jax.random.PRNGKey(seed),samples,n,jnp.asarray(g),config)
    chunks = [sampled_records(params,bits[i:i+128],jnp.asarray(g),config)
              for i in range(0,samples,128)]
    rows = {k: np.concatenate([np.asarray(c[k]) for c in chunks]) for k in chunks[0]}
    if archive is not None:
        archive.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(archive,bits=np.asarray(bits),couplings=np.asarray(g),**rows)
    result = bootstrap_response(rows, seed+100000, 200)
    corr, se = sampled_correlations(np.asarray(bits))
    result.update(n=n,couplings=g,chi_relative_error=abs(result["chi_total"]/ref["chi_total"]-1),
                  chi_relative_error_plus_2se=(abs(result["chi_total"]-ref["chi_total"])
                        +2*result["bootstrap_se"]["chi_total"])/ref["chi_total"],
                  c_half=float(corr[n//2]),c_half_se=float(se[n//2]),
                  c_half_absolute_error_plus_2se=abs(corr[n//2]-ref["c_half"])+2*se[n//2],
                  energy_per_site_error=abs(result["energy_total"]-ref["energy_total"])/n,
                  energy_per_site_se=float(np.std(rows["energy"],ddof=1)/np.sqrt(samples)/n),
                  reference_chi_total=ref["chi_total"],born_samples=samples,basis_rows=0,
                  hamiltonian_actions=2,source_actions=1, audit_seed=seed)
    return result
