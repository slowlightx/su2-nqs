"""Free correlation exponents and finite-grid susceptibility peak checks."""
from __future__ import annotations
import numpy as np


def eta_fit(sizes: list[int],correlations: list[np.ndarray],r_min: int=4,
            correction: bool=False) -> dict:
    """Fit free eta with a separate amplitude for each size, using PBC chords.

    Optional correction permits eta(L)=eta+b/L. No Ising exponent is fixed.
    This descriptive fit does not provide a sampling confidence interval.
    """
    if len(sizes)!=len(correlations) or not sizes or len(set(sizes))!=len(sizes) or type(r_min)!=int or r_min<1 or any(type(n)!=int or n<4 or n%2 for n in sizes):
        raise ValueError('distinct sizes, matching correlations and positive r_min required')
    blocks=[];values=[]
    for i,(n,c) in enumerate(zip(sizes,correlations)):
        c=np.asarray(c,dtype=float);r=np.arange(r_min,n//2+1)
        if c.shape!=(n,) or len(r)<3 or np.any(c[r]<=0) or not np.isfinite(c).all():
            raise ValueError('positive full correlations with at least three fit distances required')
        x=np.log(n/np.pi*np.sin(np.pi*r/n));amplitudes=np.zeros((len(r),len(sizes)));amplitudes[:,i]=1
        cols=[-x[:,None]]
        if correction:cols.append(-x[:,None]/n)
        blocks.append(np.concatenate([*cols,amplitudes],axis=1));values.append(np.log(c[r]))
    design=np.concatenate(blocks);y=np.concatenate(values)
    coefficients,_,rank,_=np.linalg.lstsq(design,y,rcond=None)
    if rank!=design.shape[1]:raise ValueError('correlation exponent fit is rank deficient')
    return dict(eta=float(coefficients[0]),correction=float(coefficients[1]) if correction else None,
                rms=float(np.sqrt(np.mean((y-design@coefficients)**2))),points=len(y),r_min=r_min,
                correction_L_inverse=correction,free_exponent=True,statistical_interval=False)


def local_peak(fields: np.ndarray,chi: np.ndarray,window: int=7) -> dict:
    """Fit a local quadratic around a discrete maximum and report bracketing.

    Field coordinates are centered before least squares. A nonconcave or
    out-of-neighborhood vertex falls back to the grid maximum, with a flag.
    """
    h,y=np.asarray(fields,dtype=float),np.asarray(chi,dtype=float)
    if h.ndim!=1 or y.shape!=h.shape or len(h)<5 or np.any(np.diff(h)<=0) or np.any(y<=0) or not np.isfinite(h+y).all():
        raise ValueError('ordered finite positive peak observations required')
    if type(window)!=int or window<5 or window%2!=1 or window>len(h):raise ValueError('odd local window >=5 within the grid required')
    index=int(np.argmax(y));bracketed=0<index<len(h)-1
    start=max(0,min(index-window//2,len(h)-window));stop=min(len(h),start+window)
    x=h[start:stop]-h[index];a,b,c=np.polyfit(x,y[start:stop],2)
    vertex=float(h[index]-b/(2*a)) if a<0 else float(h[index])
    accepted=bracketed and a<0 and h[start]<=vertex<=h[stop-1]
    return dict(h_peak=vertex if accepted else float(h[index]),grid_peak=float(h[index]),
        bracketed=bool(bracketed),quadratic_vertex_accepted=bool(accepted),window=window,
        local_curvature=float(2*a),fit_rms=float(np.sqrt(np.mean((a*x*x+b*x+c-y[start:stop])**2))))
