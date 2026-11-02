"""Bounded transport into a retracted physical manifold, without new heads.

Joined coefficients may be complex, but the returned columns have REAL
parameters. Finite differences include the entire user-supplied retraction.
Geometry uses node-safe projected physical amplitudes and fixed positive Q.
This diagnoses a finite-cache tangent; it is not a global FS-optimal split.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np

from .importance_tangent import tangent_system
from .joint_linear_step import linear_moment_slopes
from .joint_moments import normalized_common_moment_direction


def retracted_real_columns(
    entries: np.ndarray, directions: np.ndarray, retract: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray, parent: np.ndarray, q: np.ndarray, *, amplitude_scale: float,
    probe_fs: float = 1e-4, maximum_relative_defect: float = .1,
) -> tuple[np.ndarray, dict]:
    """Transport d and i*d with central probes at h and h/2 (four calls each).

    h is calibrated to the UNRETRACTED physical FS speed, not coordinate
    norm. Retain only columns with stable half-step differences and a small
    two-sided defect. Non-smooth rank-opening directions are reported, never
    falsely labelled differentiable. Whole-q truncation remains the callback's
    responsibility. The returned matrix contains REAL-parameter derivatives,
    normalized to unit supplied-cache FS speed; no amplitude ratios or H calls.
    """
    z, d, f, p, density = map(np.asarray, (entries, directions, features, parent, q))
    if (z.ndim != 1 or d.ndim != 2 or not d.shape[1] or d.shape[0] != len(z)
            or f.shape != (len(p),len(z)) or p.ndim != 1
            or not np.isfinite([amplitude_scale,probe_fs,maximum_relative_defect]).all()
            or amplitude_scale <= 0 or probe_fs <= 0 or not 0 < maximum_relative_defect < 1
            or any(not np.isfinite(v).all() for v in (z,d,f,p,density))):
        raise ValueError('finite compatible entries/directions/features and positive controls required')
    # Validates Q, nodes, and the nonzero physical parent before any callback.
    tangent_system(p,np.zeros_like(p),amplitude_scale*f@d,density)
    center = np.asarray(retract(z),complex)
    if center.shape != z.shape or not np.isfinite(center).all():
        raise ValueError('retraction must return finite entries of the same shape')

    def physical(v):
        return tangent_system(p,np.zeros_like(p),(amplitude_scale*f@v)[:,None],density)[2][:,0]

    records, retained = [], []
    for index, direction in enumerate(np.c_[d,1j*d].T):
        original = physical(direction); speed = float(np.linalg.norm(original))
        row = dict(input_column=index,original_fs_speed=speed,retained=False)
        if speed <= 1e-12:
            row['status']='physical null direction';records.append(row);continue
        h = probe_fs/speed
        plus, minus, half_plus, half_minus = [np.asarray(retract(z+alpha*h*direction),complex)
                                            for alpha in (1.,-1.,.5,-.5)]
        if any(v.shape!=z.shape or not np.isfinite(v).all() for v in (plus,minus,half_plus,half_minus)):
            raise ValueError('retraction returned invalid probe entries')
        coarse=(plus-minus)/(2*h);fine=(half_plus-half_minus)/h
        v, vf = physical(coarse),physical(fine)
        size = float(np.linalg.norm(vf))
        relative=float(np.linalg.norm(v-vf)/max(size,1e-30))
        defect=float(np.linalg.norm(physical((half_plus+half_minus-2*center)/h))/max(size,1e-30))
        row.update(coordinate_probe=h,retracted_fs_speed=size,half_step_relative_defect=relative,
            two_sided_relative_defect=defect,
            physical_direction_cosine=float(np.vdot(original,vf).real/max(speed*size,1e-30)),
            center_physical_replay_error=float(np.linalg.norm(physical(center-z))))
        if size>1e-12 and max(relative,defect)<=maximum_relative_defect:
            retained.append(fine/size);row.update(retained=True,status='stable finite-cache retracted column')
        else:row['status']='null or non-smooth/curved at the fixed probe scale'
        records.append(row)
    columns=np.column_stack(retained) if retained else np.zeros((len(z),0),complex)
    return columns,dict(columns=records,input_complex_columns=d.shape[1],retained_real_columns=len(retained),
        probe_fs=probe_fs,maximum_relative_defect=maximum_relative_defect,
        maximum_retraction_calls=1+8*d.shape[1],
        no_H_calls=True,physical_metric_scope='supplied projected amplitudes on known positive-Q records',
        no_new_model_parameters=True,not_a_stationarity_certificate=True)


def joint_real_coefficients(
    psi: np.ndarray, hpsi: np.ndarray, tangent: np.ndarray, h_tangent: np.ndarray,
    q: np.ndarray, blocks: Sequence[np.ndarray], *, n_sites: int, damping: float = .1,
) -> tuple[np.ndarray, dict]:
    """Common E/R descent for REAL retracted coordinates, not complex doubling."""
    t, ht = np.asarray(tangent,complex),np.asarray(h_tangent,complex)
    linear_moment_slopes(psi,hpsi,t,ht,q,n_sites)
    if not 1<=len(blocks)<=3:raise ValueError('one to three declared blocks required')
    gradients,names=[],[]
    for i,mask in enumerate(blocks):
        m=np.asarray(mask)
        if m.dtype!=bool or m.shape!=np.asarray(psi).shape or not m.any():
            raise ValueError('nonempty boolean control blocks required')
        row=linear_moment_slopes(np.asarray(psi)[m],np.asarray(hpsi)[m],t[m],ht[m],np.asarray(q)[m],n_sites)
        gradients.extend((row['energy'],row['variance']));names.extend((f'block{i}.energy',f'block{i}.variance'))
    d,info=normalized_common_moment_direction(psi,hpsi,t,q,np.asarray(gradients),names,damping=damping)
    info.update(real_parameter_columns=t.shape[1],complex_doubling=False,no_new_H_calls=True)
    return -d,info


__all__=['retracted_real_columns','joint_real_coefficients']
