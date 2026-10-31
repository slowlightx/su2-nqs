"""Two-phase finite-state audit with known within-pool inclusion probabilities.

Stratified SRS without replacement estimates *raw correction moments*, not
ratios. Cheap carrier moments are evaluated on the complete phase-one pool.
Independent pools account for both sampling stages in the final uncertainty.
"""

import numpy as np
from scipy.stats import t as student_t

NAMES = ("norm_ratio", "carrier_energy_total", "energy_total", "carrier_variance_total",
         "variance_total", "energy_change_total", "variance_change_total")


def select_strata(labels: np.ndarray, allocation: list[int | None], *, seed: int) -> dict:
    """Draw every nonempty stratum with positive known inclusion probability."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=len(allocation)):
        raise ValueError("nonempty integer stratum labels required")
    rng = np.random.default_rng(seed)
    chosen, probabilities, strata = [], [], []
    for h, target in enumerate(allocation):
        if target is not None and (not isinstance(target,int) or isinstance(target,bool) or target<2):
            raise ValueError("allocation must be at least two or None for census")
        population = np.flatnonzero(labels==h); count=len(population)
        size=count if target is None else min(target,count)
        selected=np.sort(rng.choice(population,size,replace=False)) if size else np.empty(0,dtype=int)
        chosen.extend(selected);probabilities.extend([size/count]*size if count else [])
        strata.append(dict(stratum=h,population_count=count,sample_count=size,
                           inclusion_probability=float(size/count) if count else None,census=size==count))
    return dict(indices=np.asarray(chosen,dtype=int),inclusion_probability=np.asarray(probabilities),strata=strata)


def carrier_moments(psi: np.ndarray, hpsi: np.ndarray, q: np.ndarray, reference_energy: float) -> np.ndarray:
    """Per-configuration norm and first/second centered-H moments."""
    p,h,q = map(np.asarray,(psi,hpsi,q))
    if p.ndim!=1 or h.shape!=p.shape or q.shape!=p.shape or np.any(q<=0) or not all(np.isfinite(x).all() for x in (p,h,q)):
        raise ValueError("matching finite vectors and positive proposal required")
    residual=h-reference_energy*p
    return np.column_stack((abs(p)**2/q,(p.conj()*residual).real/q,abs(residual)**2/q))


def correction_moments(psi: np.ndarray, direction: np.ndarray, hpsi: np.ndarray,
                       hdirection: np.ndarray, q: np.ndarray, epsilon: float,
                       reference_energy: float) -> np.ndarray:
    """Exact quadratic differences of raw moments at the fixed finite step."""
    p,d,h,hd,q=map(np.asarray,(psi,direction,hpsi,hdirection,q))
    carrier_moments(p,h,q,reference_energy)
    if d.shape!=p.shape or hd.shape!=p.shape or not np.isfinite(d).all() or not np.isfinite(hd).all() or not np.isfinite(epsilon):
        raise ValueError("matching finite direction/H images required")
    r=h-reference_energy*p;rd=hd-reference_energy*d
    return np.column_stack(((2*epsilon*(p.conj()*d).real+epsilon**2*abs(d)**2)/q,
        (epsilon*(p.conj()*rd+d.conj()*r).real+epsilon**2*(d.conj()*rd).real)/q,
        (2*epsilon*(r.conj()*rd).real+epsilon**2*abs(rd)**2)/q))


def stratified_correction(values: np.ndarray, selected_labels: np.ndarray, population_counts: np.ndarray) -> dict:
    """Unbiased pool-mean correction and conditional SRS finite-population covariance."""
    values=np.asarray(values,dtype=float);labels=np.asarray(selected_labels);counts=np.asarray(population_counts)
    if values.ndim!=2 or labels.shape!=(len(values),) or counts.ndim!=1 or not np.issubdtype(labels.dtype,np.integer) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0) or np.any(labels<0) or np.any(labels>=len(counts)) or counts.sum()<=0 or not np.isfinite(values).all():
        raise ValueError("invalid stratified moment data")
    mean=np.zeros(values.shape[1]);cov=np.zeros((values.shape[1],values.shape[1]));rows=[]
    for h,count in enumerate(counts):
        v=values[labels==h];size=len(v)
        if size>count or (count>0 and size==0) or (count>size and size<2):
            raise ValueError("every populated stratum needs a census or >=2 sampled values")
        weight=count/counts.sum()
        contribution=weight*v.mean(axis=0) if size else np.zeros(values.shape[1])
        mean+=contribution
        if size<count:
            cov+=weight**2*(1-size/count)*np.atleast_2d(np.cov(v,rowvar=False,ddof=1))/size
        rows.append(dict(stratum=h,population_count=int(count),sample_count=size,weighted_moment_contribution=contribution.tolist()))
    return dict(mean=mean,covariance=cov,strata=rows)


def physical_values_and_jacobian(moments: np.ndarray, reference_energy: float) -> tuple[np.ndarray,np.ndarray]:
    """Convert [Z,B,C,delta Z,delta B,delta C] to actual finite-state observables."""
    t=np.asarray(moments,dtype=float)
    if t.shape!=(6,) or not np.isfinite(t).all():raise ValueError("six finite moments required")
    a,b,c,u,v,w=t;a1,b1,c1=a+u,b+v,c+w
    if min(a,a1)<=0:raise ValueError("estimated physical norms must be positive")
    e0,e1=b/a,b1/a1;v0,v1=c/a-e0**2,c1/a1-e1**2
    ge0=np.array([-b/a**2,1/a,0,0,0,0]);ge1=np.array([-b1/a1**2,1/a1,0,-b1/a1**2,1/a1,0])
    gv0=np.array([-c/a**2+2*b*b/a**3,-2*b/a**2,1/a,0,0,0])
    gv1=np.tile([-c1/a1**2+2*b1*b1/a1**3,-2*b1/a1**2,1/a1],2)
    values=np.array([a1/a,reference_energy+e0,reference_energy+e1,v0,v1,e1-e0,v1-v0])
    jac=np.vstack(([-u/a**2,0,0,1/a,0,0],ge0,ge1,gv0,gv1,ge1-ge0,gv1-gv0))
    return values,jac


def summarize_independent_pools(moments: np.ndarray, conditional_covariances: np.ndarray,
                                reference_energy: float) -> dict:
    """Replicate-pool delta-method SE with Student-t intervals; no double counting.

    Between-pool covariance already contains phase-one MCMC and phase-two
    selection noise. Conditional design uncertainty is reported separately,
    never added to that total again. Eight pools still give approximate CIs.
    """
    t=np.asarray(moments);conditional=np.asarray(conditional_covariances)
    if t.ndim!=2 or t.shape[1]!=6 or len(t)<3 or conditional.shape!=(len(t),6,6):
        raise ValueError("at least three independent six-moment pools required")
    average=t.mean(axis=0);values,jac=physical_values_and_jacobian(average,reference_energy)
    covariance=np.cov(t,rowvar=False,ddof=1)/len(t)
    design=conditional.sum(axis=0)/len(t)**2
    se=np.sqrt(np.maximum(np.diag(jac@covariance@jac.T),0))
    design_se=np.sqrt(np.maximum(np.diag(jac@design@jac.T),0))
    critical=float(student_t.ppf(.975,len(t)-1))
    leave=np.asarray([physical_values_and_jacobian(np.delete(t,i,axis=0).mean(axis=0),reference_energy)[0] for i in range(len(t))])
    return dict(zip(NAMES,map(float,values))) | dict(
        independent_pool_count=len(t),degrees_of_freedom=len(t)-1,t_critical_95=critical,
        independent_pool_se=dict(zip(NAMES,map(float,se))),
        conditional_design_se=dict(zip(NAMES,map(float,design_se))),
        student_t_95={k:[float(y-critical*s),float(y+critical*s)] for k,y,s in zip(NAMES,values,se)},
        leave_one_pool_out={k:[float(leave[:,i].min()),float(leave[:,i].max())] for i,k in enumerate(NAMES)},
        mean_moments=average.tolist(),covariance_of_mean=covariance.tolist(),conditional_covariance_of_mean=design.tolist())
