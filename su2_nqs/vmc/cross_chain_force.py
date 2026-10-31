"""One fixed-parent, two-fold physical-force diagnostic on positive-Q data.

Fit REAL tangent coefficients on one chain fold only, freeze them, and report
complete two-sided energy/variance slopes and FS speed on the other fold.
There is no state update, sample generation, tuning, or population guarantee.
Leave-one-chain errors are CONDITIONAL on the fixed training coefficients and
columns; they do not include training uncertainty or tangent-selection bias.
"""
from __future__ import annotations

import numpy as np

from .carrier_protection import physical_moments
from .importance_tangent import tangent_system
from .joint_linear_step import linear_moment_slopes
from .joint_moments import residual_moment_cotangents
from .retraction_transport import joint_real_coefficients


def _measure(p, hp, t, ht, q, coefficients, n_sites):
    delta, hdelta = t @ coefficients, ht @ coefficients
    slopes = linear_moment_slopes(p, hp, delta[:, None], hdelta[:, None], q, n_sites)
    centered = tangent_system(p, hp, t, q)[2] @ coefficients
    speed = float(np.linalg.norm(centered))
    result = {k: float(v[0]) for k, v in slopes.items()}
    result['fs_speed'] = speed
    for key in ('energy', 'variance'):
        result[key + '_per_unit_fs'] = result[key] / speed if speed > 1e-12 else None
    return result


def _leverage(p, hp, t, ht, q, coefficients, n_sites):
    delta, hdelta = t @ coefficients, ht @ coefficients
    norm = np.sum(abs(p)**2 / q)
    energy = np.sum(p.conj() * hp / q) / norm
    contributions = {
        'energy': ((delta.conj()*hp+p.conj()*hdelta-energy*2*(p.conj()*delta).real)
                   / (q*norm*n_sites)).real,
        'fs_power': abs(tangent_system(p, hp, t, q)[2] @ coefficients)**2,
    }
    wp, wh, _ = residual_moment_cotangents(p, hp, q, n_sites)
    contributions['variance'] = (wp*delta + wh*hdelta).real
    report = {}
    for name, values in contributions.items():
        magnitude = np.sort(abs(values))[::-1]
        total = float(magnitude.sum())
        report[name] = dict(signed_sum=float(values.sum()), absolute_sum=total,
            maximum_absolute_fraction=float(magnitude[0]/total) if total else 0.,
            top_four_absolute_fraction=float(magnitude[:4].sum()/total) if total else 0.)
    return report


def cross_chain_calibration(
    psi: np.ndarray, hpsi: np.ndarray, tangent: np.ndarray, h_tangent: np.ndarray,
    q: np.ndarray, groups: np.ndarray, folds: np.ndarray, *, n_sites: int,
    damping: float = .1, finite_difference_step: float = 1e-5,
) -> dict:
    """Fit fold0 -> test fold1 and reverse ONCE, with no optimization loop.

    Tangent columns represent real coordinates of an already symmetry-valid
    physical state (angular-momentum labels remain doubled integers upstream).
    groups are chain identifiers; a chain may never straddle the two folds.
    Q normalization cancels. Both coherences and wavefunction nodes are kept.
    Report a conditional held-fold jackknife and linear-state finite-difference
    checks, not a finite-step retracted candidate, untouched audit, or update.
    """
    p, hp, t, ht, density = (np.asarray(psi, complex), np.asarray(hpsi, complex),
        np.asarray(tangent, complex), np.asarray(h_tangent, complex), np.asarray(q))
    linear_moment_slopes(p, hp, t, ht, density, n_sites)
    chain, fold = np.asarray(groups), np.asarray(folds)
    if (chain.shape != p.shape or fold.shape != p.shape
            or chain.dtype.kind not in 'iu' or fold.dtype.kind not in 'iu'
            or not np.array_equal(np.unique(fold), [0, 1])
            or not np.isfinite(finite_difference_step) or finite_difference_step <= 0):
        raise ValueError('integer chain ids, two integer folds, and positive FD step required')
    for group in np.unique(chain):
        if np.unique(fold[chain == group]).size != 1:
            raise ValueError('a chain must not straddle folds')
    if any(np.unique(chain[fold == f]).size < 3 for f in (0, 1)):
        raise ValueError('at least three complete chains per fold required')
    records = []
    for training_fold in (0, 1):
        train, held = fold == training_fold, fold != training_fold
        coefficients, geometry = joint_real_coefficients(p[train], hp[train], t[train], ht[train],
            density[train], [np.ones(train.sum(), bool)], n_sites=n_sites, damping=damping)
        train_result = _measure(p[train], hp[train], t[train], ht[train], density[train], coefficients, n_sites)
        test_result = _measure(p[held], hp[held], t[held], ht[held], density[held], coefficients, n_sites)
        # Freeze coefficients: no refit or sign/weight selection on held chains.
        deletions = []
        for group in np.unique(chain[held]):
            m = held & (chain != group)
            deletions.append(_measure(p[m], hp[m], t[m], ht[m], density[m], coefficients, n_sites))
        errors = {}
        for key in ('energy', 'variance', 'fs_speed', 'energy_per_unit_fs', 'variance_per_unit_fs'):
            values = [v[key] for v in deletions]
            if any(v is None for v in values):
                errors[key] = None
            else:
                values = np.asarray(values)
                errors[key] = float(np.sqrt((len(values)-1)/len(values)*np.sum((values-values.mean())**2)))
        delta, hdelta = t[held] @ coefficients, ht[held] @ coefficients
        eps = finite_difference_step
        plus = physical_moments(p[held]+eps*delta, hp[held]+eps*hdelta, density[held], n_sites)
        minus = physical_moments(p[held]-eps*delta, hp[held]-eps*hdelta, density[held], n_sites)
        fd_errors = {k: abs((plus[k+'_per_site']-minus[k+'_per_site'])/(2*eps)-test_result[k])
                     for k in ('energy', 'variance')}
        flags = dict(energy_point_down=test_result['energy'] < 0,
            variance_point_nonworse=test_result['variance'] <= 0,
            energy_below_minus_two_conditional_SE=test_result['energy'] < -2*errors['energy'])
        records.append(dict(training_fold=training_fold, held_fold=1-training_fold,
            training_chains=int(np.unique(chain[train]).size), held_chains=int(np.unique(chain[held]).size),
            coefficients=coefficients.tolist(), training_geometry=geometry, training=train_result,
            held=test_result, conditional_chain_jackknife_se=errors,
            fs_speed_transfer=test_result['fs_speed']/train_result['fs_speed'] if train_result['fs_speed'] > 1e-12 else None,
            held_contribution_concentration=_leverage(p[held], hp[held], t[held], ht[held], density[held], coefficients, n_sites),
            linear_state_finite_difference_error=fd_errors, flags=flags))
    return dict(folds=records, real_parameter_columns=t.shape[1],
        held_joint_point_descent_both=all(v['flags']['energy_point_down'] and v['flags']['variance_point_nonworse'] for v in records),
        held_energy_two_SE_nonworse_variance_both=all(all(v['flags'].values()) for v in records),
        conditional_errors_exclude_training_and_selection_uncertainty=True,
        records_may_be_previously_seen=True, no_parameter_update=True, no_new_samples=True,
        not_an_untouched_audit=True, not_a_stationarity_certificate=True,
        multiple_columns_or_edges_are_not_population_proof=True)


__all__ = ['cross_chain_calibration']
