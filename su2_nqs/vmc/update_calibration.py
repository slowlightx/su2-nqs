"""Independent calibration of a FROZEN physical update, without refitting it.

Positive Q, complex amplitudes, TOTAL H products and actual parameter JVPs.
These finite-chain screens are not population or equilibration guarantees.
The residual-response bound is a first-order Gauss--Newton trust proxy, not
the Hessian of variance; every finite proposal still needs full-H evaluation.
"""
from __future__ import annotations

import numpy as np

from .coherent_tangent import coherent_tangent_diagnostics
from .residual_curvature import residual_response_gram


def independent_update_calibration(
    psi: np.ndarray, hpsi: np.ndarray, delta: np.ndarray, hdelta: np.ndarray,
    q: np.ndarray, groups: np.ndarray, n_sites: int, *, maximum_alpha: float,
    response_safety: float = .5,
) -> dict:
    """Screen a fixed orientation and bound alpha using new full/parity folds.

    Nothing here changes the direction or learns parameters. Require summed
    two-sided energy AND residual slopes <0 on full/even/odd quadrature; the
    individual coherent terms need not have the same sign. A failed noisy
    screen is not proof of globally uphill physics or insufficient capacity.
    The population-Hermitian 2*forward estimate is also reported, not silently
    substituted for the exact finite-Q derivative. Whole-chain delete-one SE
    is descriptive and does not certify chain mixing or unseen rare states.

    Bound alpha by safety*(-dR)/Gram(k), where k is the centered normalized
    residual response. This bounds the positive quadratic proxy, NOT all
    normalization/energy/nonlinear curvature terms. Null directions cannot
    pass. Gauge, Q normalization and direction-unit invariance are preserved.
    """
    p, hp, d, hd, density = (np.asarray(v) for v in (psi, hpsi, delta, hdelta, q))
    ids = np.asarray(groups)
    if (ids.shape != p.shape or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)):
        raise ValueError('one nonnegative integer chain label per record required')
    labels, counts = np.unique(ids, return_counts=True)
    if len(labels) < 4 or len(set(counts)) != 1 or counts.min() < 2:
        raise ValueError('at least four equally represented chains, >=2 records each required')
    if min(sum(labels % 2 == k) for k in (0, 1)) < 2:
        raise ValueError('each predetermined parity fold needs at least two chains')
    if (isinstance(maximum_alpha, bool) or not np.isfinite(maximum_alpha) or maximum_alpha <= 0
            or isinstance(response_safety, bool) or not np.isfinite(response_safety)
            or not 0 < response_safety <= .5):
        raise ValueError('positive finite alpha and response safety in (0,.5] required')
    masks = dict(full=np.ones(len(p), bool), half_0=ids % 2 == 0, half_1=ids % 2 == 1)
    rows = {}
    bounds = []
    for name, mask in masks.items():
        row = coherent_tangent_diagnostics(p[mask], hp[mask], d[mask], hd[mask], density[mask], n_sites)
        _, response, _ = residual_response_gram(p[mask], hp[mask], d[mask, None], hd[mask, None], density[mask], n_sites)
        cost = max(float(response[0, 0]), 0.)
        bound = min(float(maximum_alpha), response_safety * (-row['variance_slope_per_site']) / cost) if cost > 0 else float(maximum_alpha)
        row.update(residual_response_per_site=cost, response_alpha_bound=max(bound, 0.))
        rows[name] = row
        bounds.append(max(bound, 0.))
    keys = ('energy_slope_per_site', 'variance_slope_per_site')
    leave = []
    for label in labels:
        mask = ids != label
        row = coherent_tangent_diagnostics(p[mask], hp[mask], d[mask], hd[mask], density[mask], n_sites)
        leave.append([row[k] for k in keys])
    leave = np.asarray(leave)
    se = np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(axis=0))**2, axis=0))
    checks = dict(energy_descent_all_folds=all(r[keys[0]] < 0 for r in rows.values()),
                  residual_descent_all_folds=all(r[keys[1]] < 0 for r in rows.values()),
                  nonzero_tangent_all_folds=all(r['tangent_fs_speed'] > 1e-12 for r in rows.values()))
    eligible = bool(all(checks.values()))
    return dict(eligible=eligible, checks=checks, rejected_by=[k for k, v in checks.items() if not v],
                regions=rows, maximum_alpha=min(bounds) if eligible else 0.,
                chain_jackknife_se=dict(zip(keys, map(float, se))),
                response_safety=float(response_safety), fixed_direction_not_refitted=True,
                calibration_used_for_step_selection=True, calibration_is_not_untouched_audit=True,
                response_bound_is_not_full_Hessian_or_certificate=True,
                noisy_rejection_is_not_architecture_no_go=True)


__all__ = ['independent_update_calibration']
