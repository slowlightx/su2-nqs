"""Independent Born-row audits with physical h derivatives and joint errors."""
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from fnqs.models import GRUConfig
from fnqs.training import local_energies
from fnqs.pilots.measurements import amplitudes
from fnqs.variants.size_features import SizeFeatureConfig, log_psi


def correlation_rows(bits: np.ndarray) -> np.ndarray:
    """Return translation-averaged Pauli ZZ at every periodic separation."""
    b=np.asarray(bits)
    if b.ndim!=2 or len(b)<2 or b.shape[1]<4 or not np.all((b==0)|(b==1)):
        raise ValueError('at least two binary configurations required')
    z=2*b.astype(np.float64)-1
    return np.stack([np.mean(z*np.roll(z,-r,axis=1),axis=1) for r in range(b.shape[1])],axis=1)


def joint_statistics(score: np.ndarray, corr: np.ndarray) -> dict:
    """Estimate chi and the covariance of chi/C using their row influences.

    Rows are independent draws from the fixed physical Born distribution.
    chi uses the covariance convention 1/M. Its influence is (s-mean(s))^2;
    the supplied joint covariance is the covariance of estimators, not spins.
    """
    s,c=np.asarray(score,dtype=float),np.asarray(corr,dtype=float)
    if s.ndim!=1 or c.ndim!=2 or len(s)!=len(c) or len(s)<2 or not np.isfinite(s).all() or not np.isfinite(c).all():
        raise ValueError('finite matching independent rows required')
    square=(s-s.mean())**2
    joint=np.column_stack((square,c));cov=np.cov(joint,rowvar=False,ddof=1)/len(s)
    return dict(chi_total=float(square.mean()),score_mean=float(s.mean()),correlations=c.mean(axis=0),
                joint_covariance=cov,chi_se_influence=float(np.sqrt(max(0.,cov[0,0]))),
                correlation_se=np.sqrt(np.maximum(0.,np.diag(cov)[1:])))


@partial(jax.jit,static_argnames=('config',))
def energy_rows(params: dict,bits: jax.Array,g: jax.Array,config: SizeFeatureConfig) -> jax.Array:
    """Use the independently tested original local-energy path for legacy GRU."""
    if config.size_features!='legacy':raise ValueError('frozen confirmation requires legacy features')
    return local_energies(params,bits,g,GRUConfig(config.hidden_dim,config.geometry_scale))


@partial(jax.jit,static_argnames=('config',))
def fd_rows(params: dict,bits: jax.Array,g: jax.Array,config: SizeFeatureConfig) -> jax.Array:
    """Two centered differences of the physical h at fixed raw parameters."""
    values=[]
    for step in (.001,.0005):
        values.append((log_psi(params,bits,g.at[3].add(step),config)-
                       log_psi(params,bits,g.at[3].add(-step),config))/(2*step))
    return jnp.stack(values,axis=1)


def summarize_rows(bits: np.ndarray,score: np.ndarray,energy: np.ndarray,fd: np.ndarray,
                   bootstrap_seed: int,bootstrap: int=200) -> dict:
    """Preserve all C(r) errors and bootstrap chi, including their covariance."""
    c=correlation_rows(bits);stats=joint_statistics(score,c)
    e,f=np.asarray(energy,dtype=float),np.asarray(fd,dtype=float)
    if e.ndim!=1 or len(e)<2 or len(e)>len(bits) or f.ndim!=2 or f.shape[1]!=2 or len(f)<2 or len(f)>len(bits):
        raise ValueError('energy/FD must be nonempty prefixes of Born rows')
    if not np.isfinite(e).all() or not np.isfinite(f).all() or type(bootstrap)!=int or bootstrap<2:
        raise ValueError('finite rows and at least two bootstrap replicates required')
    s=np.asarray(score);subset=s[:len(f)];centered=subset-subset.mean()
    denominator=float(np.mean(centered**2))
    if denominator<=0:raise ValueError('physical score variance must be positive')
    difference=f-f.mean(axis=0)-centered[:,None]
    rng=np.random.default_rng(bootstrap_seed)
    replicates=np.array([np.var(s[rng.integers(len(s),size=len(s))]) for _ in range(bootstrap)])
    boot=float(replicates.std(ddof=1))
    return {**stats,'chi_se_bootstrap':boot,'chi_se':max(boot,stats['chi_se_influence']),
        'energy_total':float(e.mean()),'energy_se':float(e.std(ddof=1)/np.sqrt(len(e))),
        'energy_variance':float(e.var()),'energy_rows':len(e),'fd_rows':len(f),
        'fd_score_relative_rms_error':np.sqrt(np.mean(difference**2,axis=0)/denominator),
        'chi_bootstrap_replicates':replicates,'samples':len(s)}
