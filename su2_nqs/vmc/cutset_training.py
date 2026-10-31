"""Node-safe fixed-proposal training and paired independent-chain audits.

``qtilde`` is a positive, FIXED density, possibly unnormalized. Samples must
target ``qtilde / sum(qtilde)``. Its unknown partition function cancels in
normalized energy, overlap, and variance ratios. These finite-sample ratios
are biased; chain bootstrap errors do not certify equilibration or tail
coverage. No routine below requires division by a wavefunction amplitude.
"""
from __future__ import annotations

from itertools import combinations
from numbers import Integral
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .transition_moments import estimate_transition_moments, joint_block_bootstrap


def _positive_proposal(qtilde: np.ndarray, count: int) -> np.ndarray:
    proposal = np.asarray(qtilde, dtype=np.float64)
    if proposal.shape != (count,) or not np.isfinite(proposal).all() or np.any(proposal <= 0):
        raise ValueError('qtilde must have one finite strictly positive value per sample')
    return proposal


def weighted_diagonal_loss(psi: jax.Array, qtilde: jax.Array,
                           diagonal: jax.Array) -> jax.Array:
    """Differentiate the fixed-Q self-normalized diagonal expectation.

    Pass ``3 Hz`` for a strict-singlet energy objective. Q is stop-gradient:
    differentiating the sampler density would change this quadrature. At an
    exact node, direct amplitude differentiation remains well defined.
    Concrete invalid proposals raise; invalid traced proposals return NaN.
    """
    amplitude = jnp.asarray(psi)
    proposal = jax.lax.stop_gradient(jnp.asarray(qtilde, dtype=jnp.float64))
    values = jnp.asarray(diagonal, dtype=jnp.float64)
    if amplitude.ndim != 1 or amplitude.shape[0] < 1 or proposal.shape != amplitude.shape or values.shape != amplitude.shape:
        raise ValueError('psi, qtilde, diagonal must be matching nonempty vectors')
    if not isinstance(proposal, jax.core.Tracer):
        _positive_proposal(np.asarray(proposal), amplitude.shape[0])
    weight = jnp.real(amplitude.conj() * amplitude) / proposal
    norm = jnp.sum(weight)
    valid = (jnp.all(jnp.isfinite(amplitude)) & jnp.all(jnp.isfinite(proposal))
             & jnp.all(proposal > 0) & jnp.all(jnp.isfinite(values)) & (norm > 0))
    return jnp.where(valid, jnp.sum(weight * values) / norm, jnp.nan)


def weighted_state_distance(old: np.ndarray, new: np.ndarray,
                            qtilde: np.ndarray) -> dict:
    """Finite FS angle from a common positive quadrature, including old nodes.

    ``ess_fraction`` is candidate-weight concentration, NOT autocorrelation
    ESS. Opening old nodes is allowed; reverse KL would instead diverge.
    """
    before = np.asarray(old, dtype=np.complex128)
    after = np.asarray(new, dtype=np.complex128)
    if before.ndim != 1 or len(before) < 1 or after.shape != before.shape:
        raise ValueError('old and new amplitudes must be matching nonempty vectors')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('amplitudes must be finite')
    proposal = _positive_proposal(qtilde, len(before))
    old_weights, new_weights = abs(before)**2/proposal, abs(after)**2/proposal
    old_norm, new_norm = old_weights.sum(), new_weights.sum()
    if min(old_norm, new_norm) <= 0:
        raise ValueError('both states need positive quadrature norm')
    overlap = np.sum(before.conj()*after/proposal)
    normalized = float(np.clip(abs(overlap)/np.sqrt(old_norm*new_norm), 0., 1.))
    return dict(fs=float(np.arccos(normalized)), normalized_overlap=normalized,
                norm_ratio=float(new_norm/old_norm),
                ess_fraction=float(new_norm**2/(len(before)*np.sum(new_weights**2))),
                max_weight_fraction=float(new_weights.max()/new_norm),
                old_ess_fraction=float(old_norm**2/(len(before)*np.sum(old_weights**2))),
                old_max_weight_fraction=float(old_weights.max()/old_norm),
                opened_node_samples=int(np.sum((before == 0) & (after != 0))),
                weight_ess_is_not_autocorrelation_ess=True)


def paired_chain_audit(
    psi: np.ndarray, hpsi: np.ndarray, q: np.ndarray, groups: np.ndarray,
    n_sites: int, bootstrap_replicates: int, seed: int, arm_names: Sequence[str],
) -> dict:
    """Joint full-H moments and paired changes, resampling whole equal chains.

    Arrays have shape (draw, arm); every arm uses exactly the same Q draws.
    Chains must be independent and equally long. Within-chain dependence is
    retained. Imaginary energy is preserved, and |Hpsi|² at psi=0 is included.
    Norm output is only norm/Z_Q or a norm ratio, never absolute norm.
    """
    amplitudes = np.asarray(psi, dtype=np.complex128)
    actions = np.asarray(hpsi, dtype=np.complex128)
    if amplitudes.ndim != 2 or min(amplitudes.shape) < 1 or actions.shape != amplitudes.shape:
        raise ValueError('psi and hpsi need matching nonempty (draw, arm) shapes')
    if not np.isfinite(amplitudes).all() or not np.isfinite(actions).all():
        raise ValueError('all amplitudes and H actions must be finite')
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError('n_sites must be a positive integer')
    names = tuple(arm_names)
    if (len(names) != amplitudes.shape[1] or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError('arm_names must uniquely name every arm')
    count, arms = amplitudes.shape
    proposal = _positive_proposal(q, count)
    chain_ids = np.asarray(groups)
    if chain_ids.shape != (count,) or not np.issubdtype(chain_ids.dtype, np.integer):
        raise ValueError('groups must contain one integer chain label per sample')
    labels, counts = np.unique(chain_ids, return_counts=True)
    if len(labels) < 4 or len(set(counts)) != 1:
        raise ValueError('need at least four equally long independent chains')
    states = np.concatenate((amplitudes, actions), axis=1)
    estimate = estimate_transition_moments(states, states[:, None, :], proposal)
    metrics = ('energy_per_site', 'imaginary_energy_per_site',
               'second_moment_total', 'variance_per_site', 'norm_ratio_to_parent')
    pairs = tuple(combinations(range(arms), 2))
    output_names = tuple(f'{name}.{metric}' for name in names for metric in metrics)
    output_names += tuple(f'{names[new]}-minus-{names[old]}.{metric}'
                          for old, new in pairs for metric in ('energy_per_site', 'variance_per_site'))

    def transform(moment):
        gram = moment[0]
        norms = np.diag(gram)[:arms].real
        if np.any(norms <= 0):
            raise FloatingPointError('an audit replicate has zero state norm')
        energy = np.asarray([gram[a, arms+a]/norms[a] for a in range(arms)])
        second = np.diag(gram)[arms:].real/norms
        variance = (second-abs(energy)**2)/n_sites
        values = np.column_stack((energy.real/n_sites, energy.imag/n_sites,
                                  second, variance, norms/norms[0])).reshape(-1).tolist()
        for old, new in pairs:
            values.extend(((energy[new].real-energy[old].real)/n_sites,
                           variance[new]-variance[old]))
        return np.asarray(values, dtype=np.float64)

    point = transform(estimate.moments)
    bootstrap = joint_block_bootstrap(
        estimate, chain_ids, bootstrap_resamples=bootstrap_replicates, seed=seed,
        nonlinear_transform=transform, nonlinear_names=output_names)
    leave = np.asarray([transform(estimate.observations[chain_ids != label].mean(axis=0))
                        for label in labels])
    jackknife = np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(axis=0))**2, axis=0))
    summaries = {
        name: dict(value=float(point[index]), chain_jackknife_se=float(jackknife[index]),
                   chain_bootstrap_se=float(bootstrap.nonlinear.standard_errors[index]),
                   chain_bootstrap_95=bootstrap.nonlinear.percentile_interval[index].tolist(),
                   leave_one_chain_out_range=[float(leave[:, index].min()), float(leave[:, index].max())])
        for index, name in enumerate(output_names)
    }
    per_arm = {}
    for a, name in enumerate(names):
        entry = {metric: summaries[f'{name}.{metric}'] for metric in metrics}
        z = abs(amplitudes[:, a])**2/proposal
        energy = estimate.moments[0, a, arms+a]/z.mean()
        residual = abs(actions[:, a]-energy*amplitudes[:, a])**2/proposal
        stable_variance = float(residual.sum()/z.sum()/n_sites)
        if abs(stable_variance-entry['variance_per_site']['value']) > 1e-10*(1+stable_variance):
            raise FloatingPointError('Gram and node-safe residual variance disagree')
        contribution = residual.sum()
        entry.update(norm_over_proposal_partition=float(z.mean()),
                     weight_ess_fraction=float(z.sum()**2/(count*np.sum(z**2))),
                     max_weight_fraction=float(z.max()/z.sum()),
                     stable_residual_variance_per_site=stable_variance,
                     max_residual_sample_fraction=float(residual.max()/contribution) if contribution > 0 else 0.,
                     max_residual_chain_fraction=float(max(residual[chain_ids == label].sum() for label in labels)/contribution) if contribution > 0 else 0.,
                     exact_node_samples=int(np.sum(amplitudes[:, a] == 0)),
                     node_H2_contribution_fraction=float(np.sum(abs(actions[amplitudes[:, a] == 0, a])**2/proposal[amplitudes[:, a] == 0])
                                                         /np.sum(abs(actions[:, a])**2/proposal)) if np.any(actions[:, a] != 0) else 0.)
        per_arm[name] = entry
    paired = {
        f'{names[new]}-minus-{names[old]}': {
            metric: summaries[f'{names[new]}-minus-{names[old]}.{metric}']
            for metric in ('energy_per_site', 'variance_per_site')}
        for old, new in pairs}
    return dict(arms=per_arm, paired=paired, sample_count=count,
                independent_chains=len(labels), samples_per_chain=int(counts[0]),
                uncertainty_unit='whole independent chain; within-chain samples not IID',
                ratio_estimates_finite_sample_biased=True,
                proposal_normalizer_known=False,
                raw_norm_interpretation='||psi||^2 / Z_Q; only relative norms are physical without Z_Q',
                imaginary_energy_forced_zero=False, equilibration_certified=False,
                unseen_tail_coverage_certified=False,
                joint_metric_names=list(output_names),
                joint_chain_bootstrap_covariance=bootstrap.nonlinear.covariance.tolist())
