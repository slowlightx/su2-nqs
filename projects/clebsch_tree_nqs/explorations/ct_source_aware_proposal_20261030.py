"""Known positive spin-source tilts of an independently normalized IID base.

This proposal changes no physical state. Bits 0/1 mean doubled m=-1/+1.
The exact rejection law is q=Qbase*w/Z, with w=eta+|F(bits)|². Z is generally
unknown. ``relative_density`` returns Qbase*w, NEVER a normalized density.
Z cancels in self-normalized energies, variances, overlaps, FS and amplitude
SR; absolute target norms and unbiased normalized-density claims do not follow.
"""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PositiveSpinSourceTilt:
    """A bounded known scalar factor over physical fixed doubled m=0 rows."""

    coefficients: np.ndarray
    floor: float = .1
    factor_squared_bound: float | None = None

    def __post_init__(self) -> None:
        c=np.asarray(self.coefficients,np.complex128)
        if (c.ndim!=1 or len(c)<2 or len(c)%2 or not np.isfinite(c).all()
                or not np.isfinite(self.floor) or self.floor<=0):
            raise ValueError("finite even coefficient vector and strict positive floor required")
        universal=(.5*np.sum(abs(c)))**2
        bound=universal if self.factor_squared_bound is None else float(self.factor_squared_bound)
        if not np.isfinite(bound) or bound<=0:
            raise ValueError("finite positive proved source-factor bound required")
        c=np.array(c,copy=True);c.setflags(write=False)
        object.__setattr__(self,"coefficients",c)
        object.__setattr__(self,"factor_squared_bound",float(bound))

    @property
    def n_sites(self) -> int:
        return len(self.coefficients)

    def _rows(self, rows: np.ndarray) -> np.ndarray:
        rows=np.asarray(rows)
        if (rows.ndim!=2 or rows.shape[1]!=self.n_sites or not np.issubdtype(rows.dtype,np.integer)
                or not np.isin(rows,(0,1)).all() or not np.all(rows.sum(axis=1)==self.n_sites//2)):
            raise ValueError("integer binary rows with exactly doubled m=0 required")
        return rows

    def factor(self, rows: np.ndarray) -> np.ndarray:
        """Return F=sum_i coefficient_i*S_i^z, not a wavefunction amplitude."""
        rows=self._rows(rows)
        return (rows-.5)@self.coefficients

    def weight(self, rows: np.ndarray) -> np.ndarray:
        """Return eta+|F|², strictly positive even at physical source nodes."""
        values=self.floor+abs(self.factor(rows))**2
        if np.any(values>self.envelope):
            raise ArithmeticError("declared analytic source bound violated; no clipping")
        return values

    @property
    def envelope(self) -> float:
        """Known rejection ceiling with outward floating-point roundoff margin."""
        return float((self.floor+self.factor_squared_bound)*(1+16*np.finfo(float).eps))

    def relative_density(self, base_density: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Return q_tilde=Qbase*w with UNKNOWN common normalizer Z."""
        rows=self._rows(rows);q=np.asarray(base_density,np.float64)
        if q.shape!=(len(rows),) or not np.isfinite(q).all() or np.any(q<=0):
            raise ValueError("exact positive normalized base density required")
        return q*self.weight(rows)

    @property
    def lower_density_ratio_bound(self) -> float:
        """Proved q/Qbase lower bound, using Z<=envelope rather than estimated Z."""
        return self.floor/self.envelope

    def draw(self, base_draw: Callable[[int,int],tuple[np.ndarray,dict]], count: int,
             rng: np.random.Generator, *, batch_size: int=256,
             maximum_candidates: int=100_000) -> tuple[np.ndarray,dict[str,Any]]:
        """Construct IID rejection samples without estimating or using Z.

        ``base_draw`` must generate IID normalized Qbase rows. Accepted rows
        are independent; excess final-batch acceptances are discarded. Restore
        the outer Generator state and identical batching to replay a draw.
        """
        if (isinstance(count,bool) or not isinstance(count,int) or count<1
                or isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1
                or maximum_candidates<count):
            raise ValueError("positive count/batch and sufficient bounded candidates required")
        before=deepcopy(rng.bit_generator.state);kept=[];used=0;total_accepted=0;seeds=[]
        while sum(len(x) for x in kept)<count:
            if used+batch_size>maximum_candidates:
                raise TimeoutError("registered rejection-candidate cap reached")
            seed=int(rng.integers(0,2**32,dtype=np.uint32));seeds.append(seed)
            rows,_=base_draw(batch_size,seed);rows=self._rows(rows)
            if len(rows)!=batch_size:
                raise ValueError("base callback returned wrong batch size")
            weights=self.weight(rows)
            accepted=rng.random(batch_size)<weights/self.envelope
            kept.append(rows[accepted]);used+=batch_size;total_accepted+=int(accepted.sum())
        result=np.concatenate(kept)[:count]
        return result,dict(records=count,base_candidates=used,all_batch_acceptances=total_accepted,
            empirical_acceptance=total_accepted/used,base_batch_seeds=seeds,
            rng_before=before,rng_after=deepcopy(rng.bit_generator.state),
            source_factor_squared_bound=self.factor_squared_bound,rejection_envelope=self.envelope,
            positive_floor=self.floor,normalizer_known=False,
            exact_IID_rejection_law="q=Qbase*(eta+|F|^2)/Z",Z_estimate_not_used_for_density=True)


def triangular_K_tilt(length: int=6, *, floor: float=.1) -> PositiveSpinSourceTilt:
    """Exact K=(1/3,1/3) guide on a balanced even triangular LxL torus.

    For N=36 the three colors each contain12 sites. Integer enumeration of
    allowed color occupations (a,b,c), a+b+c=18, proves max|F|²=108/36=3.
    This is independent of the physical parent and every measured sample.
    """
    if isinstance(length,bool) or not isinstance(length,int) or length<6 or length%6:
        raise ValueError("positive torus length divisible by6 required")
    n=length**2;colour_size=n//3;maximum=0
    for a in range(colour_size+1):
        for b in range(colour_size+1):
            c=n//2-a-b
            if 0<=c<=colour_size:
                maximum=max(maximum,a*a+b*b+c*c-a*b-b*c-c*a)
    colour=np.asarray([(i//length+i%length)%3 for i in range(n)])
    coefficients=np.exp(2j*np.pi*colour/3)/sqrt(n)
    return PositiveSpinSourceTilt(coefficients,floor,maximum/n)
