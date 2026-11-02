"""Node-safe common-proposal calibration of actual finite parameter candidates.

No linear-wavefunction assumption is made. The caller supplies amplitudes of
every actual candidate, and independently establishes proposal support. These
finite-sample decisions are not population-radius or convergence certificates.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np


def _groups(count: int, groups: int) -> np.ndarray:
    if (isinstance(groups,bool) or not isinstance(groups,Integral) or groups<4
            or count%groups or count//groups<1):
        raise ValueError('at least four equal independent groups required')
    return np.repeat(np.arange(groups),count//groups)


def _fs(phi: np.ndarray, density: np.ndarray) -> float:
    """Stable finite angle without subtracting a nearly unit fidelity."""
    scale=max(float(np.max(abs(phi))),np.finfo(float).tiny)
    x=phi/scale*np.sqrt(np.max(density)/density)[:,None]
    z=float(np.vdot(x[:,0],x[:,0]).real)
    if z<=0 or np.vdot(x[:,1],x[:,1]).real<=0:
        raise FloatingPointError('zero observed parent/candidate norm')
    cross=np.vdot(x[:,0],x[:,1]);residual=x[:,1]-x[:,0]*(cross/z)
    return float(np.arctan2(np.linalg.norm(residual),abs(cross)/np.sqrt(z)))


def common_proposal_FS(phi: np.ndarray, density: np.ndarray, *, groups: int=32) -> dict:
    """Paired finite FS with delete-group uncertainty, retaining parent nodes."""
    p,q=np.asarray(phi),np.asarray(density)
    if (p.ndim!=2 or p.shape[1]!=2 or q.shape!=(len(p),) or np.iscomplexobj(q)
            or not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(q<=0)):
        raise ValueError('finite paired amplitudes and positive real common Q required')
    labels=_groups(len(p),groups);point=_fs(p,q)
    leave=np.array([_fs(p[labels!=g],q[labels!=g]) for g in range(groups)])
    se=float(np.sqrt((groups-1)/groups*np.sum((leave-leave.mean())**2)))
    mass=abs(p)**2/q[:,None];total=mass.sum(0)
    return dict(FS_radians=point,group_jackknife_SE=se,nominal_upper=point+2*se,
        norm_estimates=mass.mean(0).tolist(),norm_effective_contributions=(total**2/np.sum(mass**2,axis=0)).tolist(),
        maximum_norm_contribution=(mass.max(0)/total).tolist(),parent_node_records=int(np.sum(p[:,0]==0)),
        candidate_mass_at_parent_nodes=float(mass[p[:,0]==0,1].sum()/total[1]),
        records=len(p),independent_groups=groups,no_wavefunction_division=True,
        nominal_not_rigorous_population_interval=True)


def select_finite_grid(phi: np.ndarray, density: np.ndarray, fractions: np.ndarray, *,
    split: int, groups: int=32, target: float=.001, maximum: float=.005,
) -> tuple[int,dict]:
    """Choose once on the fit half, then guard once; failure returns parent.

    Columns are ACTUAL nonlinear finite states in decreasing fraction order;
    the last column is the unchanged parent at zero. The independent guard
    cannot choose another fraction. No Hamiltonian data are accepted here.
    """
    p,q,grid=np.asarray(phi),np.asarray(density),np.asarray(fractions)
    if (p.ndim!=2 or q.shape!=(len(p),) or np.iscomplexobj(q) or not np.isfinite(p).all()
            or not np.isfinite(q).all() or np.any(q<=0) or np.iscomplexobj(grid)
            or grid.shape!=(p.shape[1],) or len(grid)<2
            or not np.isfinite(grid).all() or grid[0]!=1 or grid[-1]!=0
            or np.any(np.diff(grid)>=0) or isinstance(split,bool)
            or not isinstance(split,Integral) or not 0<split<len(p)
            or not np.isfinite([target,maximum]).all() or not 0<target<maximum<np.pi/2):
        raise ValueError('actual decreasing finite-state grid[1,...,0], split and physical angles required')
    parent=len(grid)-1;fitting=[]
    for i in range(parent):
        fitting.append(common_proposal_FS(p[:split][:,[parent,i]],q[:split],groups=groups))
    eligible=[i for i,d in enumerate(fitting) if 1e-12<d['FS_radians']<=target]
    chosen=eligible[0] if eligible else parent
    report=dict(selection=fitting,selection_records=split,guard_records=len(p)-split,
        proposed_index=chosen,parent_index=parent,guard_used_to_select_smaller_fraction=False,
        amplitude_only=True,finite_wavefunction_linearity_assumed=False)
    if chosen==parent:
        return parent,dict(report,accepted=False,reason='no_nonzero_registered_fraction_within_fit_radius',selected_fraction=0.)
    try:
        guard=common_proposal_FS(p[split:][:,[parent,chosen]],q[split:],groups=groups)
        ratio=guard['FS_radians']/fitting[chosen]['FS_radians']
        accepted=guard['nominal_upper']<=maximum and .5<=ratio<=2
    except (FloatingPointError,ArithmeticError) as error:
        return parent,dict(report,accepted=False,reason='invalid_guard',error=str(error),selected_fraction=0.)
    return (chosen if accepted else parent),dict(report,guard=guard,guard_to_selection_FS_ratio=ratio,
        accepted=bool(accepted),reason='passed' if accepted else 'independent_geometry_guard_failed',
        selected_fraction=float(grid[chosen]) if accepted else 0.)


def bilateral_direction_moments(psi: np.ndarray, hpsi: np.ndarray, dpsi: np.ndarray,
    hdpsi: np.ndarray, density: np.ndarray, *, groups: int=128, n_sites: int=36,
) -> dict:
    """Both sides of the real quotient derivative under a covering proposal.

    Parent plus its H-exchange bridge covers these FIRST-order integrands.
    No claim is made about squared-tangent norm, tangent metric or curvature.
    H acts on the complete tangent before projection/averaging, never PHP.
    """
    p,h,d,hd,q=map(np.asarray,(psi,hpsi,dpsi,hdpsi,density))
    if (p.ndim!=1 or any(x.shape!=p.shape for x in (h,d,hd,q)) or np.iscomplexobj(q)
            or any(not np.isfinite(x).all() for x in (p,h,d,hd,q)) or np.any(q<=0)
            or isinstance(n_sites,bool) or not isinstance(n_sites,Integral) or n_sites<1):
        raise ValueError('finite complete products, positive Q, and integer site count required')
    labels=_groups(len(p),groups)
    # Fixed common scaling avoids underflow without changing any quotient.
    scale=max(float(np.max(abs(p))),np.finfo(float).tiny)
    p,h,d,hd=[x/scale*np.sqrt(np.max(q)/q) for x in (p,h,d,hd)]
    def value(mask):
        a,b,c,e=(x[mask] for x in (p,h,d,hd));z=float(np.vdot(a,a).real)
        if z<=0:raise FloatingPointError('zero observed parent norm')
        energy=np.vdot(a,b)/z
        first=np.vdot(c,b-energy.real*a).real/z/n_sites
        second=(np.vdot(a,e)-energy.real*np.vdot(a,c)).real/z/n_sites
        return np.array([first,second,first+second,energy.real/n_sites,energy.imag/n_sites])
    point=value(np.ones(len(p),bool));leave=np.array([value(labels!=g) for g in range(groups)])
    se=np.sqrt((groups-1)/groups*np.sum((leave-leave.mean(0))**2,axis=0))
    return dict(metric_order=['g1_per_site','g2_per_site','bilateral_slope_per_site','parent_energy_per_site','parent_imaginary_energy_per_site'],
        values=point.tolist(),independent_group_jackknife_SE=se.tolist(),
        first_order_node_safe=True,global_tangent_metric_certified=False,
        records=len(p),independent_groups=groups,finite_sample_not_population_stationarity=True)
