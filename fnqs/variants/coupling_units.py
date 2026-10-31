"""Equivalent GRU coordinates adapted to a narrow physical h interval.

The physical model remains the original normalized GRU. An optimizer may
work with x_h=(h-center*j1)/scale, then convert its parameters back before
saving or measuring. Isotropic SR damping and clipping depend on these
coordinates; this is an optimization intervention, not extra expressivity.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from fnqs.variants.size_features import SizeFeatureConfig,log_psi,sample


@dataclass(frozen=True)
class CouplingUnits:
    """Finite affine h units; center and scale are fixed, unlearned values."""
    center: float=1.
    scale: float=.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.center) or not math.isfinite(self.scale) or self.scale<=0:
            raise ValueError('finite center and positive finite scale required')


def encoded(g: jax.Array, units: CouplingUnits) -> jax.Array:
    """Encode physical (j1,j2,jx,h); pure model code contains no Hamiltonian."""
    g=jnp.asarray(g,dtype=jnp.float64)
    if g.shape!=(4,):raise ValueError('four physical couplings required')
    return g.at[3].set((g[3]-units.center*g[0])/units.scale)


def to_units(params: dict, units: CouplingUnits) -> dict:
    """Exact affine parameter map preserving psi at every physical coupling."""
    if params['context'].ndim!=2 or params['context'].shape[0]!=6 or params['w'].shape[0]!=11:
        raise ValueError('original GRU parameter layout required')
    result=dict(params)
    c=params['context'];w=params['w']
    result['context']=c.at[0].set(c[0]+units.center*c[3]).at[3].set(units.scale*c[3])
    result['w']=w.at[7].set(w[7]+units.center*w[10]).at[10].set(units.scale*w[10])
    return result


def to_raw(params: dict, units: CouplingUnits) -> dict:
    """Inverse map; saved raw parameters use the original physical API."""
    if params['context'].ndim!=2 or params['context'].shape[0]!=6 or params['w'].shape[0]!=11:
        raise ValueError('original GRU parameter layout required')
    result=dict(params)
    c=params['context'];w=params['w']
    result['context']=c.at[0].set(c[0]-units.center*c[3]/units.scale).at[3].set(c[3]/units.scale)
    result['w']=w.at[7].set(w[7]-units.center*w[10]/units.scale).at[10].set(w[10]/units.scale)
    return result


def physical_log_psi(params: dict,bits: jax.Array,g: jax.Array,config: SizeFeatureConfig,units: CouplingUnits) -> jax.Array:
    """Physical-coupling amplitude from optimizer-coordinate parameters."""
    return log_psi(params,bits,encoded(g,units),config)


def physical_sample(params: dict,key: jax.Array,count: int,n: int,g: jax.Array,config: SizeFeatureConfig,units: CouplingUnits) -> jax.Array:
    """Sample the identical Born distribution in affine coordinates."""
    return sample(params,key,count,n,encoded(g,units),config)
