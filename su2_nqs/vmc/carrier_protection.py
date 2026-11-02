"""Node-safe physical-update and carrier-quality constraints on fixed Q data.

These are optimizer controls, NOT untouched validation or rigorous Hilbert-
space bounds. Every moment uses TOTAL Hpsi; wavefunction nodes are retained.
No architecture, Hamiltonian, sampler, symbolic work or spin labels are owned
by this module. Strict symmetry must already hold for every candidate state.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .cutset_training import weighted_state_distance


@dataclass(frozen=True)
class CarrierProtection:
    """Declared finite-quadrature bounds; energy/variance are per site."""
    maximum_step_fs: float = .01
    maximum_carrier_fs: float = .04
    minimum_ess_fraction: float = .6
    maximum_variance_ratio: float = 1.1
    energy_tolerance_per_site: float = 0.
    minimum_nonzero_fs: float = 1e-12

    def __post_init__(self) -> None:
        values = (self.maximum_step_fs, self.maximum_carrier_fs, self.minimum_ess_fraction,
                  self.maximum_variance_ratio, self.energy_tolerance_per_site, self.minimum_nonzero_fs)
        if not all(np.isfinite(v) and not isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError('protection bounds must be finite real numbers')
        if not 0 < self.maximum_step_fs <= self.maximum_carrier_fs < np.pi/2:
            raise ValueError('need 0 < step FS <= carrier FS < pi/2')
        if not 0 < self.minimum_ess_fraction <= 1 or self.maximum_variance_ratio < 1:
            raise ValueError('invalid ESS or variance envelope')
        if not 0 <= self.energy_tolerance_per_site or not 0 <= self.minimum_nonzero_fs < self.maximum_step_fs:
            raise ValueError('invalid energy tolerance or null-step threshold')


def physical_moments(psi: np.ndarray, hpsi: np.ndarray, q: np.ndarray, n_sites: int) -> dict:
    """Fixed-Q normalized full-H energy/variance, without dividing by psi.

    Hpsi must be TOTAL Hpsi, not Hpsi/N. Q may have an unknown normalization;
    the reported norm is ||psi||^2/Z_Q, not an absolute wavefunction norm.
    The imaginary energy is preserved; it is not forced to zero.
    """
    p, hp, density = np.asarray(psi, complex), np.asarray(hpsi, complex), np.asarray(q, float)
    if p.ndim != 1 or not len(p) or hp.shape != p.shape or density.shape != p.shape:
        raise ValueError('psi, Hpsi, Q must be matching nonempty vectors')
    if not np.isfinite(p).all() or not np.isfinite(hp).all() or not np.isfinite(density).all() or np.any(density <= 0):
        raise ValueError('finite amplitudes/products and strictly positive Q required')
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError('positive integer site count required')
    weights = abs(p)**2/density; norm = float(weights.sum())
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('state needs a positive finite quadrature norm')
    energy = np.sum(p.conj()*hp/density)/norm
    variance = float(np.sum(abs(hp-energy*p)**2/density)/norm/n_sites)
    return dict(energy_per_site=float(energy.real/n_sites), imaginary_energy_per_site=float(energy.imag/n_sites),
        variance_per_site=variance, norm_over_Q_partition=float(norm/len(p)),
        weight_ess_fraction=float(norm**2/(len(p)*np.sum(weights**2))),
        exact_node_samples=int(np.sum(p == 0)), unknown_Q_normalizer_cancels=True)


def physical_distance(parent: np.ndarray, candidate: np.ndarray, q: np.ndarray) -> dict:
    """Existing weighted-Gram FS angle, stable for exactly identical/gauge rays."""
    result = weighted_state_distance(parent, candidate, q)
    before, after = np.asarray(parent)/np.sqrt(q), np.asarray(candidate)/np.sqrt(q)
    norm = np.vdot(before, before).real
    overlap = np.vdot(before, after)
    perpendicular = after-before*(overlap/norm)
    result['fs'] = float(np.arctan2(np.linalg.norm(perpendicular), abs(overlap)/np.sqrt(norm)))
    return result


def carrier_quality_gate(carrier: np.ndarray, hcarrier: np.ndarray, parent: np.ndarray,
                         hparent: np.ndarray, candidate: np.ndarray, hcandidate: np.ndarray,
                         q: np.ndarray, groups: np.ndarray, n_sites: int,
                         policy: CarrierProtection) -> dict:
    """Require finite nonzero capped updates and non-worsening control energy.

    Candidate must not exceed carrier OR parent energy on this Q control set;
    variance is bounded by a declared multiple of carrier variance. Decisions
    use point estimates, not a confidence certificate. Reuse makes this a
    training/acceptance set. Paired leave-chain errors are descriptive only.
    """
    c = physical_moments(carrier, hcarrier, q, n_sites)
    old = physical_moments(parent, hparent, q, n_sites)
    new = physical_moments(candidate, hcandidate, q, n_sites)
    ids = np.asarray(groups)
    if ids.shape != np.asarray(q).shape or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('one integer chain label required per record')
    labels, counts = np.unique(ids, return_counts=True)
    if len(labels) < 4 or len(set(counts)) != 1:
        raise ValueError('need at least four equal independent control chains')
    step = physical_distance(parent, candidate, q)
    total = physical_distance(carrier, candidate, q)
    de = new['energy_per_site']-c['energy_per_site']
    dp = new['energy_per_site']-old['energy_per_site']
    dv = new['variance_per_site']-c['variance_per_site']
    leave = []
    for label in labels:
        keep = ids != label
        cm = physical_moments(np.asarray(carrier)[keep], np.asarray(hcarrier)[keep], np.asarray(q)[keep], n_sites)
        nm = physical_moments(np.asarray(candidate)[keep], np.asarray(hcandidate)[keep], np.asarray(q)[keep], n_sites)
        leave.append((nm['energy_per_site']-cm['energy_per_site'], nm['variance_per_site']-cm['variance_per_site']))
    leave = np.asarray(leave)
    se = np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(axis=0))**2, axis=0))
    bounds = dict(nonzero_update=step['fs'] > policy.minimum_nonzero_fs,
        step_distance=step['fs'] <= policy.maximum_step_fs*(1+1e-8),
        carrier_distance=total['fs'] <= policy.maximum_carrier_fs,
        importance_concentration=new['weight_ess_fraction'] >= policy.minimum_ess_fraction,
        energy_vs_carrier=de <= policy.energy_tolerance_per_site,
        energy_vs_parent=dp <= policy.energy_tolerance_per_site,
        variance_envelope=new['variance_per_site'] <= policy.maximum_variance_ratio*c['variance_per_site'])
    return dict(accepted=bool(all(bounds.values())), checks=bounds,
        rejected_by=[name for name, passed in bounds.items() if not passed],
        carrier=c, parent=old, candidate=new, step_distance=step, carrier_distance=total,
        delta_energy_vs_carrier=de, delta_energy_vs_parent=dp, delta_variance_vs_carrier=dv,
        paired_chain_jackknife_se=dict(energy_per_site=float(se[0]), variance_per_site=float(se[1])),
        control_used_for_training=True, finite_sample_global_protection_certified=False)


__all__ = ['CarrierProtection', 'physical_moments', 'physical_distance', 'carrier_quality_gate']
