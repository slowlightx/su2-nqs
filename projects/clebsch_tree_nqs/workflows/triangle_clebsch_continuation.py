"""Bounded same-H native ClebschTree continuation and serial live widening.

Reuse the verified physical sampler, cached complete-H action, prepared SR,
checkpoint/replay and independent audit.  No attention, new head, carrier,
energy-selected checkpoint or changed CG/spatial convention is introduced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
import signal
import socket
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_expansion import initialize_live_triangular_expansion
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout, mixed_parameter_count
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains, run_exchange_sweeps
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
from su2_nqs.vmc.physical_step_calibration import make_fs_calibrator
from su2_nqs.vmc.replay_validation import compare_replay
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
from . import triangle_optimizer_comparison as shared
from .spin_triangular_clebsch_train import checked_arity_endpoint

ROOT = Path(__file__).resolve().parents[3]
SAME_WIDTH_TASK = 'native_triangle_same_width_continuation_v3'
QUALITY_TASK = 'native_triangle_GCNN_quality_v4'
OPTIMIZED_TASK = 'native_triangle_calibrated_direct_SR_v5'
WIDE_PRODUCTION_TASK = 'native_triangle_wide_production_SR_v6'
DIRECTION_TASK = 'native_triangle_fixed_parent_direction_SR_v7'
EVIDENCE_GUARDED_TASK = 'native_triangle_evidence_guarded_regularized_SR_v8'
DIRECTION_TASKS = (DIRECTION_TASK, EVIDENCE_GUARDED_TASK)


def validate_direction_plan(plan: dict) -> None:
    """Freeze one width28 direction check before a32-update continuation.

    This is a new protocol, not a reinterpretation of an old widening job.
    Parent parameters/moments/RNG are retained; the256-chain training stream
    is explicitly reinitialized independently from the confirmation stream.
    """
    evidence_guarded = plan['task'] == EVIDENCE_GUARDED_TASK
    control = ('frozen_parent_regularized_SR_evidence_guarded_endpoint' if evidence_guarded
               else 'frozen_parent_direction_then_same_width_SR')
    required = {'registration.json', 'results.json', 'terminal_audit.json',
                'checkpoint.pkl', 'process_replay.json'}
    if (plan['model'] != dict(length=6, multiplicity_cap=28, maximum_q=10)
            or plan['parent']['model'] != plan['model']
            or plan['parent']['kind'] != 'continuation'
            or plan['arity'] != ('ternary_joint' if plan['j2_over_j1'] == 0 else 'binary_truncated')
            or plan['j2_over_j1'] not in (0., .125) or plan['binary_rank'] != 8
            or plan.get('optimization_control') != control
            or plan.get('continuation_profile') != 'inherit_parameters_optimizer_explicit_resampled_chains'
            or plan.get('reset_optimizer') is not False
            or plan['pilot_steps'] != 0 or plan['training']['steps'] != 32):
        raise ValueError('direction protocol requires unchanged width28 parent, zero committed pilot and32 updates')
    if (set(plan['parent'].get('sealed_files', {})) != required
            or any(not isinstance(v, str) or len(v) != 64 for v in plan['parent']['sealed_files'].values())
            or plan.get('runtime_contract', {}).get('policy') != 'pin_first_pilot_numerical_runtime'
            or not plan['runtime_contract'].get('node')):
        raise ValueError('direction protocol requires sealed audited parent and pinned numerical runtime')
    if (plan.get('sr_solver') != 'dual_force' or plan.get('sr_use_adam') is not False
            or plan.get('sr_tangent_fs_radius') != .003 or plan.get('parameter_chunk_size') != 8192
            or plan['learning_rate'] != .03 or plan['damping'] != (.5 if evidence_guarded else .05)
            or plan['maximum_fs_sine'] != .005 or plan['minimum_ess_fraction'] != .98):
        raise ValueError('fixed direct-SR recipe and physical guard may not change')
    expected = dict(training=(4096, 256), calibration=(2048, 128),
                    direction_check=(2048, 128), audit=(16384, 128))
    for section, counts in expected.items():
        if (plan[section]['samples'], plan[section]['chains']) != counts:
            raise ValueError('direction protocol records/chains differ: '+section)
    cal = plan['calibration']; confirm = plan['direction_check']
    policy = ('reject_resolved_ascent_allow_unresolved' if evidence_guarded
              else 'require_resolved_descent')
    if confirm.get('release_policy', 'require_resolved_descent') != policy:
        raise ValueError('direction evidence policy must match its explicit protocol version')
    if (cal.get('target_fs_sine') != .001 or cal.get('relative_tolerance') != .02
            or cal.get('max_factor') != 2. or cal.get('bracket_steps') != 8
            or cal.get('bisection_steps') != 12 or cal.get('minimum_ess_fraction') != .98
            or confirm.get('minimum_fs_sine') != 1e-12 or confirm.get('maximum_fs_sine') != .005
            or confirm.get('minimum_ess_fraction') != .98 or confirm.get('slope_standard_errors') != 2.
            or confirm.get('parent_energy_consistency_standard_errors') != 3.):
        raise ValueError('fit-only radius and one-shot independent slope/FS gate are frozen')
    integers = [plan[k] for k in ('projection_chunk', 'action_chunk', 'score_chunk', 'halvings')]
    for section in expected:
        integers += [plan[section][k] for k in ('samples', 'chains', 'refresh_proposals', 'spacing_proposals')]
    integers += [plan['training']['initial_burn_proposals'], cal['initial_burn_proposals'],
                 cal['bracket_steps'],cal['bisection_steps'],plan['training']['steps'],
                 *plan['model'].values(),plan['binary_rank'],plan['budget']['maximum_peak_device_bytes']]
    integers += [plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds')]
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in integers):
        raise ValueError('positive integer work/burn/memory/time counts required')
    maximum_hours = 24
    if evidence_guarded:
        train_hours = 17 if plan['j2_over_j1'] == 0 else 19
        maximum_hours = 4+2+train_hours
        if any(plan['budget'][phase+'_seconds'] > hours*3600
               for phase, hours in (('pilot', 4), ('replay', 2), ('train', train_hours))):
            raise ValueError('v8 phase allocation exceeds its frozen4/2/17-or19-hour bound')
    if (plan['budget']['maximum_peak_device_bytes'] > 70*1024**3
            or sum(plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds')) > maximum_hours*3600):
        raise ValueError('direction continuation exceeds its70GiB/A100-hour bound')
    seeds = plan['seeds']
    if (set(seeds) != {'calibration', 'audit', 'expansion', 'training', 'direction'}
            or len(set(seeds.values())) != len(seeds)
            or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < 2**32 for v in seeds.values())):
        raise ValueError('direction, new training, audit and inherited-protocol seeds must be distinct uint32')
    if not math.isfinite(plan['tolerance']) or not 0 < plan['tolerance'] <= 2e-11:
        raise ValueError('physical/replay tolerance cannot be weakened')
    deadline = datetime.fromisoformat(plan['campaign_deadline_utc'])
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds() != 0:
        raise ValueError('direction continuation requires a UTC deadline')


def validate_plan(plan: dict) -> None:
    """Validate released widening or bounded full-state same-width continuation."""
    if plan['task'] in DIRECTION_TASKS:
        validate_direction_plan(plan)
        return
    same_width = plan['task'] == SAME_WIDTH_TASK
    quality = plan['task'] == QUALITY_TASK
    optimized = plan['task'] == OPTIMIZED_TASK
    wide = plan['task'] == WIDE_PRODUCTION_TASK
    calibrated = optimized or wide
    revision = plan['task'] == 'native_triangle_sameH_stable_width_continuation_v2' or same_width or quality or calibrated
    if (plan['task'] not in ('native_triangle_sameH_serial_width_continuation',
                             'native_triangle_sameH_stable_width_continuation_v2', SAME_WIDTH_TASK, QUALITY_TASK,
                             OPTIMIZED_TASK, WIDE_PRODUCTION_TASK)
            or plan['model'].get('length') != 6 or plan['model'].get('maximum_q') != 10
            or plan['model'].get('multiplicity_cap') not in ((28, 32) if wide else
                                                           (18, 20, 24) if revision else (16, 18, 20))
            or plan['arity'] not in shared.ARITIES or plan['binary_rank'] != 8
            or plan['j2_over_j1'] not in (0., .125)):
        raise ValueError('undeclared physical model, width, arity, H or binary rank')
    cap = plan['model']['multiplicity_cap']
    parent = plan['parent']; old = parent['model']
    repair = revision and parent['kind'] == 'checkpoint_replay_rebase'
    control = plan.get('optimization_control') == 'same_parent_same_width_Adam_reset'
    if calibrated:
        required = {'registration.json', 'results.json', 'terminal_audit.json',
                    'checkpoint.pkl', 'process_replay.json'}
        cal = plan['calibration']
        profile_ok = ((old['multiplicity_cap'] == 20 and cap in (28, 32)
                       and plan['training']['steps'] == 128
                       and plan.get('optimization_control') == 'single_wide_production_direct_SR'
                       and plan.get('reset_optimizer') is True
                       and plan['arity'] == ('ternary_joint' if plan['j2_over_j1'] == 0 else 'binary_truncated'))
                      if wide else (cap == old['multiplicity_cap'] and cap in (20, 24)
                       and 2 <= plan['training']['steps'] <= 64
                       and plan.get('optimization_control') == 'matched_empirical_FS_direct_SR'))
        if (parent['kind'] != 'continuation' or not profile_ok
                or plan.get('sr_use_adam') is not False or plan.get('sr_tangent_fs_radius') != .003
                or plan.get('parameter_chunk_size') != 8192
                or not isinstance(plan.get('reset_optimizer'), bool)
                or set(parent.get('sealed_files', {})) != required
                or any(not isinstance(v, str) or len(v) != 64 for v in parent['sealed_files'].values())
                or plan.get('runtime_contract', {}).get('policy') != 'pin_first_pilot_numerical_runtime'
                or not plan['runtime_contract'].get('node')
                or cal.get('target_fs_sine') != .001
                or cal.get('relative_tolerance') != .02 or cal.get('max_factor') != 2.
                or cal.get('bracket_steps') != 8 or cal.get('bisection_steps') != 12
                or cal.get('minimum_ess_fraction') != .98):
            raise ValueError('calibrated profile requires its sealed width/step contract, direct SR and fit-only empirical FS calibration')
        deadline = datetime.fromisoformat(plan['campaign_deadline_utc'])
        if deadline.utcoffset() is None or deadline.utcoffset().total_seconds() != 0:
            raise ValueError('calibrated campaign requires an explicit UTC deadline')
        if wide and (plan['audit']['samples'], plan['audit']['chains']) != (65536, 256):
            raise ValueError('wide production requires its frozen65536-record independent endpoint audit')
        if wide and plan['budget']['maximum_peak_device_bytes'] > 70*1024**3:
            raise ValueError('wide production keeps70GiB of the A10080 allocation as its measured peak ceiling')
    if quality:
        required = {'registration.json', 'results.json', 'terminal_audit.json',
                    'checkpoint.pkl', 'process_replay.json'}
        if (parent['kind'] != 'continuation' or old['multiplicity_cap'] != 20
                or cap not in (20, 24) or not 2 <= plan['training']['steps'] <= 64
                or plan.get('optimization_control') != 'paired_capacity_reset_direct_SR'
                or plan.get('sr_use_adam') is not False or plan.get('sr_tangent_fs_radius') != .003
                or plan.get('parameter_chunk_size') != 8192
                or set(parent.get('sealed_files', {})) != required
                or any(not isinstance(v, str) or len(v) != 64 for v in parent['sealed_files'].values())
                or plan.get('runtime_contract', {}).get('policy') != 'pin_first_pilot_numerical_runtime'
                or not plan['runtime_contract'].get('node')):
            raise ValueError('quality release requires sealed width20 parent, live20/24 inclusion and paired direct-SR reset')
    if same_width:
        if (parent['kind'] != 'continuation' or control or cap != old['multiplicity_cap']
                or plan.get('continuation_profile') != 'inherit_full_optimizer_sampler_rng'
                or not 2 <= plan['training']['steps'] <= 64):
            raise ValueError('same-width continuation must inherit the complete fixed audited state')
        required = {'registration.json', 'results.json', 'terminal_audit.json',
                    'checkpoint.pkl', 'process_replay.json'}
        if (set(parent.get('sealed_files', {})) != required
                or any(not isinstance(v, str) or len(v) != 64
                       for v in parent['sealed_files'].values())):
            raise ValueError('same-width parent requires immutable endpoint and replay file seals')
        contract = plan.get('runtime_contract', {})
        if (contract.get('policy') != 'pin_first_pilot_numerical_runtime'
                or not isinstance(contract.get('node'), str) or not contract['node']):
            raise ValueError('same-width replay requires a pinned node and pilot runtime fingerprint')
    if control and (not revision or cap != 18 or parent['kind'] != 'continuation'
                    or plan['stage'] != 'w18_control'):
        raise ValueError('capacity control must keep audited width18 and explicitly reset Adam')
    if same_width or quality or calibrated:
        previous_cap = cap
        if quality or wide: previous_cap = 20
    elif revision:
        previous_cap = 18 if repair or control else {18: 16, 20: 18, 24: 20}[cap]
    else:
        previous_cap = {16: 16, 18: 16, 20: 18}[cap]
    if (old['length'] != 6 or old['maximum_q'] != 10
            or old['multiplicity_cap'] != previous_cap):
        raise ValueError('parent width differs from the registered continuation/widening profile')
    if parent['kind'] not in (('continuation', 'checkpoint_replay_rebase') if revision else
                              ('legacy_arity', 'optimizer_comparison', 'continuation')):
        raise ValueError('unknown parent checkpoint contract')
    if revision and plan.get('sr_solver') != 'dual_force':
        raise ValueError('stable width revision explicitly registers dual_force arithmetic')
    if repair and (cap != 18 or plan['case'] != 'j0125_binary_truncated'
                   or plan['j2_over_j1'] != .125 or parent['absolute_completed_steps'] != 254):
        raise ValueError('only the sealed two-update width18 failed replay may rebase')
    if (plan['training']['samples'], plan['training']['chains']) != ((4096, 64) if quality or calibrated else (2048, 64)):
        raise ValueError('training records/chains differ from registered release')
    if (plan['calibration']['samples'], plan['calibration']['chains']) != (512, 64):
        raise ValueError('independent physical-step guard needs512 records/64 chains')
    audit_counts = (plan['audit']['samples'], plan['audit']['chains'])
    if (audit_counts not in ((32768, 256), (65536, 256), (262144, 256)) if calibrated else
            audit_counts != ((65536, 256) if quality else (16384, 128))):
        raise ValueError('untouched audit records/chains differ from registered release')
    if (plan['pilot_steps'] != 2 or not (same_width or quality or calibrated) and
            plan['training']['steps'] != (14 if repair else 32 if cap == 16 else 16)):
        raise ValueError('fixed32/16/16 endpoints with two-step pilots required')
    if (plan['learning_rate'] != (.03 if quality or calibrated else 1e-4)
            or (plan['damping'] not in (.05, .5) if optimized else plan['damping'] != .05)):
        raise ValueError('same preregistered SR+Adam recipe required')
    if plan['maximum_fs_sine'] != .005 or plan['minimum_ess_fraction'] != .98:
        raise ValueError('common train/independent guard cannot be weakened')
    integers = [plan['pilot_steps'], plan['training']['steps'], plan['projection_chunk'], plan['action_chunk'], plan['score_chunk'],
        plan['halvings'], *plan['model'].values(), plan['budget']['maximum_peak_device_bytes']]
    for section in ('training', 'calibration', 'audit'):
        row = plan[section]
        integers += [row[k] for k in ('samples', 'chains', 'refresh_proposals', 'spacing_proposals')]
        if row['samples'] % row['chains']:
            raise ValueError('records must divide into chain labels')
    integers += [plan['calibration']['initial_burn_proposals']]
    if calibrated:
        integers += [plan['calibration'][key] for key in ('bracket_steps', 'bisection_steps')]
    integers += [plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds')]
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in integers):
        raise ValueError('record/work/time counts must be positive integers')
    if any(not math.isfinite(plan[k]) or plan[k] <= 0 for k in ('tolerance', 'new_feature_scale')):
        raise ValueError('positive finite tolerance and initialization scale required')
    if plan['tolerance'] > 2e-11:
        raise ValueError('physical exactness tolerance cannot be weakened')
    seeds = list(plan['seeds'].values())
    if (set(plan['seeds']) != {'calibration', 'audit', 'expansion'} or len(set(seeds)) != 3
            or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < 2**32 for v in seeds)):
        raise ValueError('three distinct uint32 streams required')
    ceiling = 96*3600 if wide else 36*3600 if optimized else 24*3600 if quality else 8*3600 if same_width else 6*3600
    if sum(plan['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds')) > ceiling:
        raise ValueError('one serial stage exceeds its registered A100-hour ceiling')


def numerical_runtime_fingerprint() -> dict:
    """Read the ACTUAL executing runtime; first remote pilot seals this value.

    Node, CUDA backend/compiler runtime, package versions and arithmetic flags
    must agree in independent replay and later continuation processes. Local
    CPU test versions are never inserted into GPU plans.
    """
    versions = {}
    for name in ('jax', 'jaxlib', 'numpy', 'jax-cuda12-plugin', 'jax-cuda12-pjrt',
                 'jax-cuda13-plugin', 'jax-cuda13-pjrt'):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    devices = jax.local_devices()
    return dict(node=socket.gethostname().split('.')[0], python=platform.python_version(),
                package_versions=versions, platform=jax.default_backend(),
                x64=bool(jax.config.jax_enable_x64),
                devices=[dict(kind=d.device_kind, platform=d.platform,
                              platform_version=str(getattr(d.client, 'platform_version', 'unknown')),
                              runtime_type=str(getattr(d.client, 'runtime_type', 'unknown')))
                         for d in devices],
                environment={k: os.environ.get(k) for k in (
                    'XLA_FLAGS', 'JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'JAX_ENABLE_X64',
                    'JAX_DEFAULT_MATMUL_PRECISION', 'NVIDIA_TF32_OVERRIDE',
                    'XLA_PYTHON_CLIENT_PREALLOCATE', 'OPENBLAS_NUM_THREADS',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})


def direction_check_moments(psi: object, hpsi: object, dpsi: object, dhpsi: object,
                            candidate_psi: object, *, chains: int, n_sites: int = 36) -> dict:
    """One frozen direction under parent Born samples, time-major chains.

    Hermiticity gives dE=2 Re[<H dpsi/psi>-<Hpsi/psi><dpsi/psi>].
    Unlike the score-covariance estimator, this includes first-order paths
    through exact parent nodes because H acts on the COMPLETE tangent.
    The finite-step FS/ESS remains parent-support sampled, not a guarantee
    about newly opened node weight or global equilibration. Delete-one whole
    chains recomputes the nonlinear product of means in the slope estimate.
    """
    if (isinstance(chains, bool) or not isinstance(chains, Integral) or chains < 2
            or isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1):
        raise ValueError('at least two chain labels and positive integer site count required')
    arrays = [np.asarray(v, np.complex128).reshape(-1) for v in (psi,hpsi,dpsi,dhpsi,candidate_psi)]
    if len({len(v) for v in arrays}) != 1 or not len(arrays[0]) or len(arrays[0]) % chains:
        raise ValueError('matching nonempty time-major arrays must divide into chains')
    if not all(np.isfinite(v).all() for v in arrays) or np.any(arrays[0] == 0):
        return dict(finite=False, samples=len(arrays[0]), chains=int(chains),
                    reason='nonfinite_values_or_sampled_parent_node')
    a,h,d,hd,b = [v.reshape(-1, chains) for v in arrays]
    e,o,ho,ratio = h/a,d/a,hd/a,b/a
    if not all(np.isfinite(v).all() for v in (e,o,ho,ratio)):
        return dict(finite=False, samples=int(a.size), chains=int(chains), reason='nonfinite_ratios')
    mean_e,mean_o,mean_ho = e.mean(),o.mean(),ho.mean()
    slope = 2*np.real(mean_ho-mean_e*mean_o)/n_sites
    chain_e,chain_o,chain_ho = e.mean(0),o.mean(0),ho.mean(0)
    leave_e=(chain_e.sum()-chain_e)/(chains-1)
    leave_o=(chain_o.sum()-chain_o)/(chains-1)
    leave_ho=(chain_ho.sum()-chain_ho)/(chains-1)
    leave_slopes=2*np.real(leave_ho-leave_e*leave_o)/n_sites
    se=float(np.sqrt((chains-1)/chains*np.sum((leave_slopes-leave_slopes.mean())**2)))
    weight=abs(ratio)**2; weight_mean=float(weight.mean())
    if not math.isfinite(weight_mean) or weight_mean <= 0:
        return dict(finite=False, samples=int(a.size), chains=int(chains), reason='zero_candidate_support')
    fs=float(np.sqrt(max(0.,1.-abs(ratio.mean())**2/weight_mean)))
    ess=float(weight_mean**2/np.mean(weight**2))
    score_cov=2*np.real(np.mean(o.conj()*e)-mean_o.conjugate()*mean_e)/n_sites
    parent_se=float(chain_e.real.std(ddof=1)/np.sqrt(chains)/n_sites)
    half=len(e)//2
    if half:
        half_difference=(e[-half:].mean(0)-e[:half].mean(0)).real/n_sites
        half_drift=float(half_difference.mean())
        half_se=float(half_difference.std(ddof=1)/np.sqrt(chains))
    else:
        half_drift=half_se=None
    values=dict(energy_slope_per_site=float(slope),energy_slope_chain_jackknife_SE=se,
        score_covariance_slope_per_site=float(score_cov),
        hermitian_minus_score_slope_per_site=float(slope-score_cov),
        parent_energy_per_site=float(mean_e.real)/n_sites,
        parent_energy_chain_SE=parent_se,
        parent_imaginary_energy_per_site=float(mean_e.imag)/n_sites,
        sampled_fs_sine=fs,reweighting_ess_fraction=ess,
        finite_FS_relative_to_fit_target=fs/.001,
        finite_FS_in_half_to_double_target=bool(.0005 <= fs <= .002))
    return dict(**values,finite=all(math.isfinite(v) for v in values.values()),
        samples=int(a.size),chains=int(chains),records_per_chain=len(a),
        leave_one_chain_slopes_per_site=leave_slopes.tolist(),
        chain_parent_energy_per_site=(chain_e.real/n_sites).tolist(),
        parent_half_window_drift_per_site=half_drift,parent_half_window_drift_chain_SE=half_se,
        candidate_sampled_exact_nodes=int(np.sum(b == 0)),
        estimator='Hermitian_complete_H_JVP_parent_Born',
        mixing_certified=False,finite_FS_node_complete=False,
        confirmation_used_to_fit_direction_or_radius=False)


def with_parent_energy_reference(moments: dict, reference: dict) -> dict:
    """Attach a predeclared same-checkpoint reference, never a lower-energy state."""
    if (not all(math.isfinite(reference[k]) for k in ('energy_per_site','energy_chain_SE'))
            or reference['energy_chain_SE'] < 0):
        raise ValueError('finite reference energy and nonnegative uncertainty required')
    out=dict(moments)
    if moments.get('finite',False):
        delta=moments['parent_energy_per_site']-reference['energy_per_site']
        combined=math.hypot(moments['parent_energy_chain_SE'],reference['energy_chain_SE'])
        out['parent_energy_reference_check']=dict(reference=reference,
            difference_per_site=float(delta),combined_independent_chain_SE=combined,
            standardized_difference=(float(delta/combined) if combined > 0 else 0. if delta==0 else None),
            agreement_does_not_certify_equilibrium=True)
    return out


def frozen_parent_energy_reference(plan: dict) -> dict:
    """Verify the same width28 parent audit, never the best-result plot point."""
    path=Path(plan['parent']['directory'])/'terminal_audit.json'
    digest=file_sha256(path)
    if digest != plan['parent']['sealed_files']['terminal_audit.json']:
        raise ValueError('frozen parent energy-reference audit changed')
    audit=json.loads(path.read_text())
    if any(not math.isfinite(audit[k]) for k in ('energy_per_site','energy_chain_SE')) or audit['energy_chain_SE'] < 0:
        raise ValueError('finite same-parent energy and nonnegative chain uncertainty required')
    return dict(energy_per_site=audit['energy_per_site'],energy_chain_SE=audit['energy_chain_SE'],
                audit_sha256=digest,path=str(path))


def direction_check_gate(moments: dict, settings: dict) -> dict:
    """One-shot confirmation; never rescale/refit a direction using these data."""
    finite=bool(moments.get('finite',False))
    upper=(moments['energy_slope_per_site']+settings['slope_standard_errors']*
           moments['energy_slope_chain_jackknife_SE']) if finite else None
    checks=dict(finite=finite,
        slope_upper_bound_negative=bool(finite and upper < 0),
        finite_step_FS_safe=bool(finite and settings['minimum_fs_sine'] <= moments['sampled_fs_sine']
                                <= settings['maximum_fs_sine']),
        reweighting_ESS_safe=bool(finite and moments['reweighting_ess_fraction'] >= settings['minimum_ess_fraction']))
    if 'parent_energy_consistency_standard_errors' in settings:
        parent=moments.get('parent_energy_reference_check',{})
        checks['same_parent_energy_consistent']=bool(finite and parent and
            abs(parent['difference_per_site']) <= settings['parent_energy_consistency_standard_errors']*
            parent['combined_independent_chain_SE'])
    return dict(passed=all(checks.values()),checks=checks,slope_upper_bound_per_site=upper,
        thresholds={k:settings[k] for k in ('minimum_fs_sine','maximum_fs_sine',
                    'minimum_ess_fraction','slope_standard_errors','parent_energy_consistency_standard_errors') if k in settings},
        one_frozen_direction_one_confirmation=True,global_convergence_certified=False)


def evidence_guarded_direction_check(moments: dict, settings: dict) -> dict:
    """Release a fixed endpoint unless the initial slope resolves ascent.

    This v8 optimization policy is not a descent certificate. An unresolved
    two-standard-error interval may release the preregistered32-step endpoint,
    retaining all finite-state, sampled-FS, ESS and parent-energy safeguards.
    The independent terminal audit determines the observed energy quality.
    No direction, radius, or threshold is fitted using these records.
    """
    if moments.get('finite', False) and (
            not math.isfinite(moments['energy_slope_per_site'])
            or not math.isfinite(moments['energy_slope_chain_jackknife_SE'])
            or moments['energy_slope_chain_jackknife_SE'] < 0):
        moments = dict(moments, finite=False)
    strict = direction_check_gate(moments, settings)
    finite = strict['checks']['finite']
    lower = (moments['energy_slope_per_site']-settings['slope_standard_errors']*
             moments['energy_slope_chain_jackknife_SE']) if finite else None
    descent = strict['checks']['slope_upper_bound_negative']
    ascent = bool(finite and lower > 0)
    checks = {key: value for key, value in strict['checks'].items()
              if key != 'slope_upper_bound_negative'}
    checks['no_resolved_ascent'] = bool(finite and not ascent)
    return dict(strict, passed=all(checks.values()), checks=checks,
                slope_lower_bound_per_site=lower,
                slope_evidence=('nonfinite' if not finite else 'confirmed_descent' if descent
                                else 'resolved_ascent' if ascent else 'unresolved'),
                confirmed_descent=descent,
                release_policy='reject_resolved_ascent_allow_unresolved',
                unresolved_is_not_descent_evidence=True,
                endpoint_energy_gain_confirmed=False)


def registered_direction_gate(plan: dict, moments: dict) -> dict:
    """Select the frozen gate by protocol identity, never by observed results."""
    if plan['task'] == DIRECTION_TASK:
        return direction_check_gate(moments, plan['direction_check'])
    if plan['task'] == EVIDENCE_GUARDED_TASK:
        return evidence_guarded_direction_check(moments, plan['direction_check'])
    raise ValueError('unknown fixed-parent direction protocol')


def shrink_physical_update(parent, proposed, train_rows, cal_rows, evaluate,
                           *, maximum_fs_sine: float, minimum_ess_fraction: float, halvings: int):
    """Shrink only, using BOTH sample sets and no candidate-H/energy queries.

    Independent samples may only reject or shrink an already proposed update.
    They can never amplify it to match a target radius.  This is not a claim
    that finite-sample FS estimates on different chains become identical.
    """
    theta, unravel = ravel_pytree(parent); delta = ravel_pytree(proposed)[0]-theta
    if not np.isfinite(np.asarray(delta)).all() or not np.any(np.asarray(delta) != 0):
        return parent, dict(accepted=False, applied_factor=0., attempted_factor=0., amplitude_evaluations=0,
                            training={}, calibration={}, reason='zero_or_nonfinite_direction')
    old_train, old_cal = evaluate(parent, train_rows), evaluate(parent, cal_rows)
    last = None
    for k in range(halvings):
        factor = .5**k; candidate = unravel(theta+factor*delta)
        new_train, new_cal = evaluate(candidate, train_rows), evaluate(candidate, cal_rows)
        train = sampled_overlap_diagnostics(jnp.log((new_train/old_train).astype(jnp.complex128)))
        cal = sampled_overlap_diagnostics(jnp.log((new_cal/old_cal).astype(jnp.complex128)))
        records = dict(training=jax.tree.map(lambda v: np.asarray(v).item(), train),
                       calibration=jax.tree.map(lambda v: np.asarray(v).item(), cal))
        finite = all(np.isfinite(np.asarray(v)).all() for v in (new_train, new_cal, theta+factor*delta))
        nonzero = bool(np.all(np.asarray(new_train) != 0) and np.all(np.asarray(new_cal) != 0))
        ok = finite and nonzero and all(
            np.isfinite(row['sampled_fs_sine']) and row['sampled_fs_sine'] <= maximum_fs_sine
            and row['reweighting_ess_fraction'] >= minimum_ess_fraction for row in records.values())
        ratios = (new_train/old_train, new_cal/old_cal)
        spread = [float(jnp.std(v)/jnp.maximum(jnp.sqrt(jnp.mean(jnp.abs(v)**2)), 1e-300)) for v in ratios]
        live = (max(records['training']['sampled_fs_sine'], records['calibration']['sampled_fs_sine']) > 1e-12
                and max(spread) > 1e-12)
        last = dict(accepted=bool(ok and live), applied_factor=factor if ok and live else 0.,
                    attempted_factor=factor, amplitude_evaluations=2+2*(k+1),
                    normalized_ratio_spread_training=spread[0], normalized_ratio_spread_calibration=spread[1], **records)
        if ok and live:
            return candidate, last
    return parent, last


def runtime(plan: dict, template: object) -> dict:
    """Original native real tensors, full coherent A1 and complete same-H SR."""
    cfg = MixedClebschConfig(**plan['model']); arity = plan['arity']
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=plan['j2_over_j1'], periodic=True))
    edges = tuple((a, b) for a, b, _ in triangular_j1j2_bonds(6, 6, j2=0., periodic=True))
    bank = lambda p: arity_clebsch_bank(p, cfg, arity, plan['binary_rank'])
    prepare = jax.jit(lambda p: prepare_feature_lut_bank(bank(p), cfg))
    projected = make_feature_lut_projected_amplitude(cfg, images, chunk=plan['projection_chunk'])
    cached = make_cached_projected_heisenberg_action(cfg, images, bonds, projection_chunk=plan['projection_chunk'])

    @jax.jit
    def action(prepared, rows):
        def one(row):
            psi, hp = cached(prepared, row[None]); return jnp.stack((psi[0], hp[0]))
        values = jax.lax.map(one, rows, batch_size=plan['action_chunk'])
        return values[:, 0], values[:, 1]

    update = make_physical_sr_adam(lambda p, x: projected(prepare(p), x), template,
        damping=plan['damping'], score_chunk=plan['score_chunk'], maximum_fs_sine=plan['maximum_fs_sine'],
        minimum_ess_fraction=plan['minimum_ess_fraction'], prepare=prepare, prepared_amplitude=projected,
        sr_solver=plan.get('sr_solver', 'woodbury_gradient'),
        **({key: plan[key] for key in ('parameter_chunk_size', 'sr_use_adam', 'sr_tangent_fs_radius')}
           if plan['task'] in (QUALITY_TASK, OPTIMIZED_TASK, WIDE_PRODUCTION_TASK, *DIRECTION_TASKS) else {}))

    calibrated = None
    if plan['task'] in (OPTIMIZED_TASK, WIDE_PRODUCTION_TASK, *DIRECTION_TASKS):
        calibrated = make_fs_calibrator(prepare,
            lambda b, x: jnp.log(projected(b, x).astype(jnp.complex128)), template,
            batch_size=plan['score_chunk'], **{key: plan['calibration'][key] for key in (
                'target_fs_sine', 'relative_tolerance', 'max_factor', 'bracket_steps',
                'bisection_steps', 'minimum_ess_fraction')})
    _, unravel = ravel_pytree(template)

    @jax.jit
    def normalized_calibration_direction(parent, proposed, rows):
        theta, _ = ravel_pytree(parent); delta = ravel_pytree(proposed)[0]-theta
        def logs(values):
            prepared = prepare(unravel(values))
            return jax.lax.map(lambda x: jnp.log(jnp.abs(projected(prepared, x))),
                               rows, batch_size=plan['score_chunk'])
        _, tangent = jax.jvp(logs, (theta,), (delta,))
        norm = jnp.std(tangent)
        valid = jnp.isfinite(norm) & (norm > 1e-14)
        scale = jnp.where(valid, plan['calibration'].get('target_fs_sine', .001)/norm, 0.)
        return unravel(theta+scale*delta), norm, valid

    @jax.jit
    def evaluate(p, rows):
        prepared = prepare(p)
        return jax.lax.map(lambda x: projected(prepared, x), rows, batch_size=plan['score_chunk'])

    train_sampler = shared.make_record_sampler(projected, plan['training'], edges)
    cal_sampler = shared.make_record_sampler(projected, plan['calibration'], edges)

    def transition(state):
        p, moments, key, states, ck, cx = state; tick = perf_counter(); prepared = prepare(p)
        key, states, rows = train_sampler(prepared, key, states)
        ck, cx, cal_rows = cal_sampler(prepared, ck, cx)
        jax.block_until_ready(rows); sampling = perf_counter()-tick; tick = perf_counter()
        psi, hp = action(prepared, rows); e = jax.block_until_ready(hp/psi)
        if not np.isfinite(np.asarray(e)).all():
            raise FloatingPointError('nonfinite full-H ratios; no clipping or node floor')
        h_seconds = perf_counter()-tick; tick = perf_counter()
        proposed, next_moments, sr = update(p, moments, rows, e, plan['learning_rate'], warm=True)
        fitting = {}
        guard_rows = cal_rows
        valid_fit = True
        if calibrated is not None:
            # Split by independent CHAIN, never by successive snapshots. Only
            # the first half fits a scalar radius; the other half can only
            # veto/shrink. No candidate H or audit enters this choice.
            shaped = cal_rows.reshape(-1, plan['calibration']['chains'], 36)
            fit_chains = plan['calibration']['chains']//2
            fit_rows = shaped[:, :fit_chains].reshape(-1, 36)
            guard_rows = shaped[:, fit_chains:].reshape(-1, 36)
            normalized, tangent, valid = normalized_calibration_direction(p, proposed, fit_rows)
            proposed, fitted = calibrated(p, normalized, fit_rows)
            valid_fit = bool(valid) and bool(fitted['accepted'])
            fitting = dict(raw_fit_tangent=float(tangent),
                fit=jax.tree.map(lambda v: np.asarray(v).item(), fitted),
                fitted_chains=fit_chains, held_out_guard_chains=fit_chains)
        chosen, guard = shrink_physical_update(p, proposed, rows, guard_rows, evaluate,
            maximum_fs_sine=plan['maximum_fs_sine'], minimum_ess_fraction=plan['minimum_ess_fraction'],
            halvings=plan['halvings'])
        accepted = bool(sr['accepted']) and valid_fit and guard['accepted']
        if not accepted:
            chosen, next_moments = p, moments
        jax.block_until_ready(chosen)
        info = dict(accepted=accepted, training_energy_per_site=float(jnp.mean(e))/36,
            training_variance_per_site=float(jnp.var(e))/36, full_H_calls=len(rows),
            training_rows_sha256=pytree_sha256(rows), calibration_rows_sha256=pytree_sha256(cal_rows),
            sr=jax.tree.map(lambda v: np.asarray(v).item(), sr), physical_guard=guard)
        if calibrated is not None:
            final_fit = sampled_overlap_diagnostics(jnp.log((evaluate(chosen, fit_rows)/
                evaluate(p, fit_rows)).astype(jnp.complex128)))
            info['empirical_step_calibration'] = dict(**fitting,
                final_fit=jax.tree.map(lambda v: np.asarray(v).item(), final_fit),
                held_out_used_only_as_veto_or_shrink=True,
                global_FS_or_equilibrium_certificate=False)
        return (chosen, next_moments, key, states, ck, cx), info, dict(
            sampling_seconds=sampling, H_seconds=h_seconds, proposal_and_guard_seconds=perf_counter()-tick)

    def direction_confirmation(parent, candidate):
        """Fresh parent-Born chains, no fitting and no optimizer mutation."""
        spec=plan['direction_check']
        key,init=jax.random.split(jax.random.PRNGKey(plan['seeds']['direction']))
        states=initialize_spin_exchange_chains(init,n_chains=spec['chains'],n_sites=36,n_up=18)
        key,states,rows=shared.make_record_sampler(projected,spec,edges)(prepare(parent),key,states)
        delta=jax.tree.map(lambda new,old:new-old,candidate,parent)
        # Differentiating the entire coherent full-H action retains tangent
        # contributions from connected configurations where parent psi=0.
        query=jax.jit(lambda p,d,x:jax.jvp(lambda z:action(prepare(z),x),(p,),(d,)))
        (psi,hp),(dp,hdp)=jax.block_until_ready(query(parent,delta,rows))
        new_psi=jax.block_until_ready(evaluate(candidate,rows))
        moments=direction_check_moments(psi,hp,dp,hdp,new_psi,chains=spec['chains'])
        payload=dict(bits=np.asarray(rows),psi=np.asarray(psi),hpsi=np.asarray(hp),
            dpsi=np.asarray(dp),dhpsi=np.asarray(hdp),candidate_psi=np.asarray(new_psi),
            terminal_states=np.asarray(states),terminal_rng=np.asarray(key),
            chain_labels=np.tile(np.arange(spec['chains']),spec['samples']//spec['chains']))
        return moments,payload

    return dict(cfg=cfg, bank=bank, prepare=prepare, projected=projected, action=action,
                transition=transition, images=images, bonds=bonds, edges=edges,
                direction_confirmation=direction_confirmation)


def _sealed_json(path: Path) -> tuple[dict, str]:
    row = json.loads(path.read_text()); seal = row['sha256']
    if canonical_sha256({k: v for k, v in row.items() if k != 'sha256'}) != seal:
        raise ValueError('parent registration is not self-consistent')
    return row, seal


def load_parent(plan: dict) -> tuple[tuple, dict]:
    """Verify actual future endpoints, never predict unknown checkpoint hashes."""
    req = plan['parent']; directory = Path(req['directory']); kind = req['kind']
    if plan['task'] in (SAME_WIDTH_TASK, QUALITY_TASK, OPTIMIZED_TASK, WIDE_PRODUCTION_TASK, *DIRECTION_TASKS):
        for name, digest in req['sealed_files'].items():
            if file_sha256(directory/name) != digest:
                raise ValueError('same-width audited parent artifact changed: '+name)
        old_replay = json.loads((directory/'process_replay.json').read_text())
        if old_replay.get('status') != 'passed' or old_replay.get('optimizer_updates_committed') != 0:
            raise ValueError('same-width parent must have passed process-isolated replay')
    if kind == 'checkpoint_replay_rebase':
        # This imports a physically committed checkpoint, NOT a successful old
        # replay or audited endpoint. Preserve the old failure and all moments
        # and RNG; the new arithmetic must pass its own independent replay.
        for name, digest in req['sealed_files'].items():
            if file_sha256(directory/name) != digest:
                raise ValueError('failed-pilot source artifact changed')
        reg, seal = _sealed_json(directory/'registration.json')
        source_plan = json.loads(Path(req['plan_path']).read_text())
        if (canonical_sha256(source_plan) != reg['plan_sha256']
                or any(reg.get(k) != v for k, v in req['expected_identity'].items())):
            raise ValueError('failed-pilot source plan or identity changed')
        pilot=json.loads((directory/'pilot.json').read_text())
        replay=json.loads((directory/'process_replay.json').read_text())
        checkpoint_sha=file_sha256(directory/'checkpoint.pkl')
        if (pilot['status'] != 'passed' or replay['status'] != 'failed'
                or pilot['registration_sha256'] != seal or replay['registration_sha256'] != seal
                or pilot['checkpoint_sha256'] != checkpoint_sha
                or replay['checkpoint_sha256'] != checkpoint_sha
                or replay['optimizer_updates_committed'] != 0
                or replay['report']['passed'] is not False
                or not all(r['bitwise_exact'] for r in replay['report']['leaves'] if r['discrete'])):
            raise ValueError('failed-pilot discrete trajectory or checkpoint seals disagree')
        state,meta=shared.restore_state(directory/'checkpoint.pkl',seal)
        if (meta['completed_steps'] != 2 or len(meta['history']) != 2
                or meta['absolute_completed_steps'] != req['absolute_completed_steps']
                or int(state[1][0]) != meta['optimizer_base_step']+2
                or np.shape(state[3]) != (64,36)):
            raise ValueError('failed-pilot full optimizer/RNG endpoint differs')
        return state,dict(directory=str(directory),kind=kind,model=req['model'],
            registration_sha256=seal,checkpoint_sha256=checkpoint_sha,
            params_sha256=pytree_sha256(state[0]),optimizer_sha256=pytree_sha256(state[1]),
            old_failed_replay_sha256=file_sha256(directory/'process_replay.json'),
            old_replay_gate_passed=False,absolute_completed_steps=meta['absolute_completed_steps'],
            optimizer_policy='inherit_all_Adam_moments_and_RNG_with_new_equivalent_dual_force_arithmetic')
    if kind == 'legacy_arity':
        spec = dict(model=req['model'], arm=plan['arity'], j2_over_j1=plan['j2_over_j1'],
            arity_comparison=dict(binary_rank=plan['binary_rank']), arity_endpoint_parent=req,
            endpoint_protocol='same_H_full_state_continuation', additional_steps=plan['training']['steps'],
            steps=req['completed_steps']+plan['training']['steps'])
        _, _, endpoint = checked_arity_endpoint(MixedClebschConfig(**req['model']), spec)
        cp = endpoint['saved']; meta = cp['metadata']; seal = req['registration_sha256']
        state = (cp['params'], cp['optimizer_state'], jnp.asarray(meta['rng']),
                 jnp.asarray(meta['sampler_states']), None, None)
        absolute = meta['completed_steps']
    else:
        reg, seal = _sealed_json(directory/'registration.json')
        if any(reg.get(k) != v for k, v in req['expected_identity'].items()):
            raise ValueError('future parent registration identity differs')
        if kind == 'continuation':
            source_plan = json.loads(Path(req['plan_path']).read_text())
            if canonical_sha256(source_plan) != reg['plan_sha256']:
                raise ValueError('serial parent plan is not the preregistered previous stage')
            if plan['task'] in DIRECTION_TASKS and set(plan['seeds'].values()) & set(source_plan['seeds'].values()):
                raise ValueError('new direction/training/audit streams may not reuse the parent registered seeds')
        result = json.loads((directory/'results.json').read_text())
        audit = json.loads((directory/'terminal_audit.json').read_text())
        path = directory/'checkpoint.pkl'
        if (result['status'] != 'fixed_steps_complete' or result['registration_sha256'] != seal
                or result['checkpoint_sha256'] != file_sha256(path)
                or result['terminal_audit'] != audit or not audit['independent_from_training_and_calibration']
                or audit['params_sha256'] != result['params_sha256']):
            raise ValueError('future parent result/checkpoint/audit mismatch')
        state, meta = shared.restore_state(path, seal)
        if (result['params_sha256'] != pytree_sha256(state[0])
                or result['completed_steps'] != meta['completed_steps']
                or len(meta['history']) != meta['completed_steps']):
            raise ValueError('future parent endpoint counter/parameters mismatch')
        if kind == 'optimizer_comparison':
            if (reg['plan_sha256'] != 'ab2c3642bfdddb3ad5ddf343ed67807711991267175c7f2d5262c1411e7810c9'
                    or reg['arity'] != plan['arity'] or reg['optimizer'] != 'sr_adam'
                    or req['model'] != dict(length=6, multiplicity_cap=16, maximum_q=10)
                    or plan['j2_over_j1'] != .125 or meta['completed_steps'] != 20
                    or result['parent_completed_steps'] != 200 or int(state[1][0]) != 220):
                raise ValueError('only the preregistered SR+Adam220 endpoint may continue')
            absolute = 220
        else:
            if (reg['model'] != req['model'] or reg['arity'] != plan['arity']
                    or reg['j2_over_j1'] != plan['j2_over_j1'] or reg['case'] != plan['case']
                    or result['absolute_completed_steps'] != meta['absolute_completed_steps']
                    or int(state[1][0]) != meta['optimizer_base_step']+meta['completed_steps']):
                raise ValueError('serial parent changed model/H/arity/history')
            absolute = meta['absolute_completed_steps']
    if absolute != req['absolute_completed_steps']:
        raise ValueError('parent is not the fixed preregistered lifetime endpoint')
    if np.shape(state[3]) != (64, 36):
        raise ValueError('parent must retain exactly64 native36-spin chains')
    return state, dict(directory=str(directory), kind=kind, model=req['model'],
        registration_sha256=seal, checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'),
        params_sha256=pytree_sha256(state[0]), optimizer_sha256=pytree_sha256(state[1]),
        audit_sha256=file_sha256(directory/'terminal_audit.json'), absolute_completed_steps=absolute)


def expansion_preflight(plan: dict, parent_state: tuple, expanded: object, rt: dict) -> dict:
    """Verify actual projected inclusion/full-H and a nongauge new-gate JVP."""
    old_plan = dict(plan, model=plan['parent']['model']); old_rt = runtime(old_plan, parent_state[0])
    rows = jnp.asarray(parent_state[3][:2]); old_prepared = old_rt['prepare'](parent_state[0])
    new_prepared = rt['prepare'](expanded)
    old_p, old_h = jax.block_until_ready(old_rt['action'](old_prepared, rows))
    new_p, new_h = jax.block_until_ready(rt['action'](new_prepared, rows))
    relative = lambda a, b: float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(b), 1e-300))
    errors = dict(amplitude_relative_error=relative(new_p, old_p), full_H_relative_error=relative(new_h, old_h))
    old_layout = mixed_clebsch_layout(MixedClebschConfig(**old_plan['model']))
    old_shapes = [{key: g['shape'][1:] for g in row['groups'] for key in g['keys']} for row in old_layout]
    tangent = jax.tree.map(jnp.zeros_like, expanded); top = []
    for previous, row in zip(old_shapes, mixed_clebsch_layout(rt['cfg']), strict=True):
        groups = []
        for group in row['groups']:
            pieces = []
            for key in group['keys']:
                oo, ol, ore = previous[key]; shape = group['shape'][1:]
                a, l, r = jnp.indices(shape)
                mask = (a < oo)&((l >= ol)|(r >= ore))
                pieces.append(jnp.where(mask, jnp.cos(a+2*l+3*r), 0.)/np.sqrt(shape[-1]*shape[-2]))
            groups.append(jnp.stack(pieces))
        top.append(tuple(groups))
    tangent['top'] = tuple(top)
    tangent_rows = jnp.asarray(parent_state[3][:8])
    def query(p):
        prepared = rt['prepare'](p)
        return jax.vmap(lambda x: rt['projected'](prepared, x))(tangent_rows)
    derivative = jax.jit(lambda p, d: jax.jvp(query, (p,), (d,))[1])(expanded, tangent)
    tangent_psi = jax.jit(query)(expanded)
    perpendicular = derivative-tangent_psi*jnp.vdot(tangent_psi, derivative)/jnp.vdot(tangent_psi, tangent_psi)
    strength = float(jnp.linalg.norm(perpendicular)/jnp.maximum(jnp.linalg.norm(tangent_psi), 1e-300))
    if (not all(math.isfinite(v) and v <= plan['tolerance'] for v in errors.values())
            or not math.isfinite(strength) or strength <= 1e-10):
        raise ArithmeticError('live width inclusion changed the state/H or opened only dead/gauge gates')
    return dict(status='passed', errors=errors, new_gate_nongauge_relative_tangent=strength, full_H_centers=4)


def checked_training_release(output: Path, metadata: dict, plan: dict, registration: str) -> str:
    """Require the actual sealed fixture replay, not merely a passed string."""
    pilot = json.loads((output/'pilot.json').read_text())
    replay = json.loads((output/'process_replay.json').read_text())
    initial = file_sha256(output/'checkpoint.pkl') if metadata['completed_steps'] == 2 else metadata.get('pilot_checkpoint_sha256')
    fixture = file_sha256(output/'next_transition_fixture.pkl'); preflight = file_sha256(output/'preflight.json')
    if (pilot['status'] != 'passed' or pilot.get('registration_sha256') != registration
            or pilot['checkpoint_sha256'] != initial or pilot['fixture_sha256'] != fixture
            or pilot['preflight_sha256'] != preflight or replay['status'] != 'passed'
            or replay.get('registration_sha256') != registration or replay['checkpoint_sha256'] != initial
            or replay.get('fixture_sha256') != fixture or replay.get('preflight_sha256') != preflight
            or replay.get('optimizer_updates_committed') != 0):
        raise ValueError('successful measured pilot and sealed independent replay required')
    if plan['task'] == WIDE_PRODUCTION_TASK:
        validate_wide_pilot_cost(plan, pilot)
    report = replay.get('report', {})
    if (report.get('protocol') != 'exact_discrete_relative_float_v1' or report.get('passed') is not True
            or report.get('tolerance') != plan['tolerance'] or not report.get('leaves')):
        raise ValueError('complete numerical replay report is missing or uses a different protocol')
    for row in report['leaves']:
        if row.get('passed') is not True or row.get('finite') is not True:
            raise ValueError('replay contains failed/nonfinite leaves')
        if row['discrete']:
            if row['bitwise_exact'] is not True:
                raise ValueError('replay changed a discrete optimizer/RNG/sampling decision')
        else:
            rel = row.get('relative_vector_error'); absolute = row.get('absolute_error')
            diagnostic = (row.get('natural_unit_scalar_diagnostic') and row['shape'] == []
                          and row['path'].startswith('[1]'))
            if not (rel is not None and math.isfinite(rel) and rel <= plan['tolerance'] or
                    diagnostic and absolute is not None and math.isfinite(absolute) and absolute <= plan['tolerance']):
                raise ValueError('replay numerical errors exceed the preregistered tolerance')
    return initial


def validate_wide_pilot_cost(plan: dict, pilot: dict) -> None:
    """Require actual width28/32 timing/memory before its126-update release."""
    warm = pilot.get('warm_step_seconds')
    peak = pilot.get('peak_device_bytes')
    expected = (warm*(plan['training']['steps']-2+
        plan['audit']['samples']/plan['training']['samples'])+900.
        if isinstance(warm, (int, float)) and math.isfinite(warm) else math.inf)
    if (not isinstance(warm, (int, float)) or not math.isfinite(warm) or warm <= 0
            or not isinstance(peak, int) or peak < (1 if plan['require_gpu'] else 0)
            or peak > plan['budget']['maximum_peak_device_bytes']
            or expected > plan['budget']['train_seconds']
            or pilot.get('forecast_remaining_seconds') != expected
            or pilot.get('maximum_remaining_seconds') != plan['budget']['train_seconds']
            or pilot.get('completed_steps') != 2):
        raise ValueError('wide production pilot lacks measured full-step cost/memory within its fixed budget')


def continuation_work_counts(plan: dict, history: list[dict]) -> dict[str, int]:
    """Count executed H centers, retaining the registered train/audit sizes."""
    return dict(full_H_training_centers=sum(int(row['full_H_calls']) for row in history),
                fixture_and_replay_H_centers=2*int(plan['training']['samples']),
                terminal_audit_H_centers=int(plan['audit']['samples']),
                calibration_H_centers=0)


def wide_production_time_guard(plan: dict, phase: str, history: list[dict],
                               elapsed_seconds: float, *, now: datetime | None = None,
                               audit: bool = False) -> dict:
    """Reserve measured work and checkpoint time before a bounded wide step.

    The saved state is always the last committed update. This is an elapsed
    time guard, not a new energy/variance selection rule. A pilot has no
    measured first-step cost; its Slurm time limit and signal handler remain
    the first cold-compile bound. Later steps use the largest of the last
    three measured transitions plus25% headroom and120s for saving.
    """
    now = datetime.now(timezone.utc) if now is None else now
    deadline = datetime.fromisoformat(plan['campaign_deadline_utc'])
    phase_left = plan['budget'][phase+'_seconds']-elapsed_seconds
    campaign_left = (deadline-now).total_seconds()
    warm = max((float(row['seconds']) for row in history[-3:]), default=0.)
    work_units = plan['audit']['samples']/plan['training']['samples'] if audit else 1.
    reserve = 1.25*warm*work_units+120.
    available = min(phase_left, campaign_left)
    return dict(can_start=bool(math.isfinite(available) and available > reserve),
                available_seconds=available, reserved_seconds=reserve,
                measured_step_seconds=warm, action='audit' if audit else 'transition')


def initialize_resampled_chains(plan: dict, parent: tuple, rt: dict) -> tuple[tuple, dict]:
    """Explicitly fork new chain populations, retaining original RNG provenance.

    Increasing chain counts is NOT bitwise continuation of the64-chain
    sampler. Parameters and optimizer are unchanged; the old streams remain
    in the immutable parent and this record. New streams have registered
    independent seeds/burn and are subsequently checkpointed in full.
    """
    prepared=rt['prepare'](parent[0]); values=[]
    for section,seed_name in (('training','training'),('calibration','calibration')):
        spec=plan[section]; key,init=jax.random.split(jax.random.PRNGKey(plan['seeds'][seed_name]))
        states=initialize_spin_exchange_chains(init,n_chains=spec['chains'],n_sites=36,n_up=18)
        states,_,key,_=jax.jit(lambda k,x:run_exchange_sweeps(prepared,k,x,
            amplitude_fn=rt['projected'],particle_kind='spin',
            sweeps=spec['initial_burn_proposals'],exchange_edges=rt['edges']))(key,states)
        jax.block_until_ready(states);values.extend((key,states))
    state=(parent[0],parent[1],*values)
    record=dict(policy='explicit_new_sampler_populations_not_bitwise_parent_continuation',
        parent_params_sha256=pytree_sha256(parent[0]),parent_optimizer_sha256=pytree_sha256(parent[1]),
        parent_sampler_rng_sha256=pytree_sha256(parent[2:]),
        parent_training_rng=np.asarray(parent[2]).tolist(),parent_training_states=np.asarray(parent[3]).tolist(),
        parent_calibration_rng=np.asarray(parent[4]).tolist(),parent_calibration_states=np.asarray(parent[5]).tolist(),
        new_sampler_rng_sha256=pytree_sha256(state[2:]),
        training_seed=plan['seeds']['training'],calibration_seed=plan['seeds']['calibration'],
        training_chains=plan['training']['chains'],calibration_chains=plan['calibration']['chains'],
        training_initial_burn_proposals=plan['training']['initial_burn_proposals'],
        calibration_initial_burn_proposals=plan['calibration']['initial_burn_proposals'],
        parameters_and_optimizer_inherited_bitwise=True,mixing_certified=False)
    return state,record


def direction_gate_replay_matches(expected: dict, actual: dict, *, tolerance: float) -> bool:
    """Keep gate decisions exact while comparing derived bounds numerically.

    Only the two slope bounds are floating replay diagnostics. Their strict
    relative tolerance is the existing physical replay tolerance, without an
    absolute floor. Thresholds, classifications, keys and all other fields
    remain exact. This does not recompute or change an acceptance decision.
    """
    if not math.isfinite(tolerance) or not 0 < tolerance <= 2e-11:
        raise ValueError('gate replay tolerance must be positive and at most2e-11')
    if not isinstance(actual, dict) or set(expected) != set(actual):
        return False
    bounds = {'slope_upper_bound_per_site', 'slope_lower_bound_per_site'} & set(expected)
    if canonical_sha256({k: v for k, v in expected.items() if k not in bounds}) != canonical_sha256(
            {k: v for k, v in actual.items() if k not in bounds}):
        return False
    for key in bounds:
        left, right = expected[key], actual[key]
        if left is None:
            if right is not None:
                return False
            continue
        if (isinstance(left, bool) or isinstance(right, bool)
                or not isinstance(left, (int, float)) or not isinstance(right, (int, float))
                or not math.isfinite(left) or not math.isfinite(right)):
            return False
        scale = max(abs(left), abs(right))
        if scale and abs(left/scale-right/scale) > tolerance:
            return False
    return True


def checked_direction_release(output: Path, meta: dict, plan: dict, registration: str,
                              *, require_replay: bool = True) -> dict:
    """Bind the one-shot direction gate to its frozen parent and exact fixture."""
    pilot=json.loads((output/'pilot.json').read_text())
    source_hash=(file_sha256(output/'checkpoint.pkl') if meta['completed_steps']==0
                 else meta.get('pilot_checkpoint_sha256'))
    direction_path=output/'direction_check.json'
    raw_path=output/'direction_check.npz'
    direction=json.loads(direction_path.read_text())
    if (pilot.get('status') != 'passed' or pilot.get('completed_steps') != 0
            or pilot.get('registration_sha256') != registration
            or pilot.get('checkpoint_sha256') != source_hash
            or pilot.get('fixture_sha256') != file_sha256(output/'next_transition_fixture.pkl')
            or pilot.get('preflight_sha256') != file_sha256(output/'preflight.json')
            or pilot.get('direction_report_sha256') != file_sha256(direction_path)
            or pilot.get('direction_records_sha256') != file_sha256(raw_path)
            or direction.get('registration_sha256') != registration
            or direction.get('parent_checkpoint_sha256') != source_hash
            or direction.get('seed') != plan['seeds']['direction']
            or direction.get('optimizer_updates_committed') != 0
            or direction.get('training_direction_accepted') is not True
            or direction.get('records_sha256') != file_sha256(raw_path)
            or direction.get('gate') != registered_direction_gate(plan,direction['moments'])
            or direction['gate']['passed'] is not True):
        raise ValueError('sealed successful fixed-parent direction gate required')
    with np.load(raw_path) as raw:
        recomputed=direction_check_moments(*[raw[k] for k in ('psi','hpsi','dpsi','dhpsi','candidate_psi')],
                                           chains=plan['direction_check']['chains'])
    recomputed=with_parent_energy_reference(recomputed,frozen_parent_energy_reference(plan))
    if canonical_sha256(recomputed) != canonical_sha256(direction['moments']):
        raise ValueError('direction statistics disagree with their immutable raw records')
    warm=pilot.get('warm_step_seconds');peak=pilot.get('peak_device_bytes')
    forecast=(warm*(plan['training']['steps']+plan['audit']['samples']/plan['training']['samples'])+900.
              if isinstance(warm,(int,float)) and math.isfinite(warm) else math.inf)
    if (not isinstance(warm,(int,float)) or warm <= 0 or not math.isfinite(warm)
            or not isinstance(peak,int) or peak < (1 if plan['require_gpu'] else 0)
            or peak > plan['budget']['maximum_peak_device_bytes']
            or pilot.get('forecast_remaining_seconds') != forecast
            or forecast > plan['budget']['train_seconds']):
        raise ValueError('direction pilot lacks actual cost/memory headroom for the fixed continuation')
    if pilot.get('cold_warm_same_state_replay',{}).get('passed') is not True:
        raise ValueError('pilot must measure a replayed warm transition, not multiply compile cost')
    if require_replay:
        replay=json.loads((output/'process_replay.json').read_text())
        report=replay.get('report',{})
        if (replay.get('status') != 'passed' or replay.get('optimizer_updates_committed') != 0
                or replay.get('registration_sha256') != registration
                or replay.get('checkpoint_sha256') != source_hash
                or replay.get('fixture_sha256') != pilot['fixture_sha256']
                or replay.get('direction_report_sha256') != pilot['direction_report_sha256']
                or replay.get('independently_recomputed_gate',{}).get('passed') is not True
                or not direction_gate_replay_matches(direction['gate'],
                    replay.get('independently_recomputed_gate'), tolerance=plan['tolerance'])
                or report.get('protocol') != 'exact_discrete_relative_float_v1'
                or report.get('tolerance') != plan['tolerance'] or report.get('passed') is not True
                or not report.get('leaves') or not all(row.get('passed') and row.get('finite') for row in report['leaves'])):
            raise ValueError('independent full transition/direction process replay required')
        for row in report['leaves']:
            if row['discrete']:
                if row.get('bitwise_exact') is not True:
                    raise ValueError('direction replay changed a discrete optimizer or RNG decision')
            else:
                relative=row.get('relative_vector_error');absolute=row.get('absolute_error')
                diagnostic=(row.get('natural_unit_scalar_diagnostic') and row['shape']==[]
                            and row['path'].startswith('[0][1]'))
                if not (relative is not None and math.isfinite(relative) and relative <= plan['tolerance'] or
                        diagnostic and absolute is not None and math.isfinite(absolute) and absolute <= plan['tolerance']):
                    raise ValueError('direction replay numerical errors exceed registered tolerance')
    return pilot


def run_direction_phases(plan: dict, phase: str, state: tuple, meta: dict, rt: dict,
                         output: Path, registration: str, lineage: dict,
                         stop: list[bool], phase_started: float) -> dict:
    """Zero-update direction pilot/replay, then32 fixed accepted updates."""
    def pause() -> dict:
        report=dict(status='safe_pause',completed_steps=meta['completed_steps'],
                    absolute_completed_steps=meta['absolute_completed_steps'])
        atomic_json(output/'progress.json',report);return report

    def reserve(history: list[dict], *, audit: bool = False) -> bool:
        note=wide_production_time_guard(plan,phase,history,perf_counter()-phase_started,audit=audit)
        if not note['can_start']:
            stop[0]=True;atomic_json(output/'time_guard.json',dict(phase=phase,**note))
        return bool(note['can_start'])

    if phase in ('pilot','replay'):
        if meta['completed_steps'] != 0:
            raise ValueError('direction pilot/replay must preserve the frozen zero-update parent')
        if phase=='pilot' and (output/'pilot.json').exists():
            # One confirmation only: never retry a failed gate with another draw.
            old=json.loads((output/'pilot.json').read_text())
            if old.get('registration_sha256') != registration:
                raise ValueError('existing one-shot pilot belongs to another registration')
            return old
        if phase=='pilot' and any((output/name).exists() for name in ('direction_check.json','direction_check.npz')):
            raise ValueError('partial one-shot confirmation exists; preserve it rather than redraw or overwrite')
        if phase=='replay':
            checked_direction_release(output,meta,plan,registration,require_replay=False)
            if (output/'process_replay.json').exists():
                checked_direction_release(output,meta,plan,registration,require_replay=True)
                return dict(status='replay_passed',optimizer_updates_committed=0,cached_verified_replay=True)
        if stop[0] or not reserve([]):return pause()
        if not (output/'preflight.json').exists():
            atomic_json(output/'preflight.json',shared.preflight(rt,state[0],state[3],plan))
        tick=perf_counter();proposed,info,timing=rt['transition'](state)
        step_seconds=perf_counter()-tick;warm_replay=None;cold_seconds=step_seconds
        if phase=='pilot':
            # Compile once, then replay precisely the same frozen state/RNG.
            # Do not multiply cold compilation time by32 future updates.
            cold_fixture=shared.transition_fixture(proposed,info)
            if stop[0] or not reserve([dict(seconds=step_seconds)]):return pause()
            tick=perf_counter();proposed,info,timing=rt['transition'](state)
            step_seconds=perf_counter()-tick
            warm_replay=compare_replay(cold_fixture,shared.transition_fixture(proposed,info),
                tolerance=plan['tolerance'],diagnostic_prefixes=('[1]',))
            if not warm_replay['passed']:
                atomic_json(output/'cold_warm_replay_failure.json',warm_replay)
                raise ArithmeticError('same-parent cold/warm transition changed the frozen direction')
        if stop[0] or not reserve([dict(seconds=2*step_seconds)]):return pause()
        moments,payload=rt['direction_confirmation'](state[0],proposed[0])
        moments=with_parent_energy_reference(moments,frozen_parent_energy_reference(plan))
        gate=registered_direction_gate(plan,moments)
        # Numeric arrays only in replay fixture; no wall-clock values or paths.
        fixture=(shared.transition_fixture(proposed,info),payload)
        if phase=='replay':
            expected=load_checkpoint(output/'next_transition_fixture.pkl')['params']
            report=compare_replay(expected,fixture,tolerance=plan['tolerance'],diagnostic_prefixes=('[0][1]',))
            replay=dict(status='passed' if report['passed'] and gate['passed'] else 'failed',report=report,
                optimizer_updates_committed=0,registration_sha256=registration,
                checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
                fixture_sha256=file_sha256(output/'next_transition_fixture.pkl'),
                direction_report_sha256=file_sha256(output/'direction_check.json'),
                independently_recomputed_gate=gate)
            atomic_json(output/'process_replay.json',replay)
            if replay['status'] != 'passed':raise ArithmeticError('full direction/sampler/H/optimizer/RNG replay failed')
            return dict(status='replay_passed',optimizer_updates_committed=0)
        save_checkpoint(output/'next_transition_fixture.pkl',fixture)
        np.savez_compressed(output/'direction_check.npz',**payload)
        report=dict(registration_sha256=registration,parent_checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
            parent_params_sha256=pytree_sha256(state[0]),candidate_params_sha256=pytree_sha256(proposed[0]),
            seed=plan['seeds']['direction'],moments=moments,gate=gate,optimizer_updates_committed=0,
            training_direction_accepted=bool(info['accepted']),direction_info=info,
            records_sha256=file_sha256(output/'direction_check.npz'),
            full_H_parent_centers=plan['direction_check']['samples'],
            full_H_JVP_tangent_centers=plan['direction_check']['samples'],
            direction_or_radius_refitted_from_confirmation=False)
        atomic_json(output/'direction_check.json',report)
        peak=max(int((d.memory_stats() or {}).get('peak_bytes_in_use',0)) for d in jax.local_devices())
        forecast=step_seconds*(plan['training']['steps']+plan['audit']['samples']/plan['training']['samples'])+900.
        passed=bool(info['accepted'] and gate['passed'] and forecast <= plan['budget']['train_seconds']
                    and peak <= plan['budget']['maximum_peak_device_bytes']
                    and (not plan['require_gpu'] or peak > 0) and not stop[0])
        pilot=dict(status='passed' if passed else 'failed',completed_steps=0,optimizer_updates_committed=0,
            registration_sha256=registration,checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
            fixture_sha256=file_sha256(output/'next_transition_fixture.pkl'),
            preflight_sha256=file_sha256(output/'preflight.json'),
            direction_report_sha256=file_sha256(output/'direction_check.json'),
            direction_records_sha256=file_sha256(output/'direction_check.npz'),
            warm_step_seconds=step_seconds,transition_timing=timing,peak_device_bytes=peak,
            cold_transition_seconds=cold_seconds,cold_warm_same_state_replay=warm_replay,
            forecast_remaining_seconds=forecast,maximum_remaining_seconds=plan['budget']['train_seconds'],
            phase_seconds=perf_counter()-phase_started,gate=gate,
            failure_reasons=[name for name,ok in dict(direction_accepted=bool(info['accepted']),
                confirmation=gate['passed'],cost=forecast<=plan['budget']['train_seconds'],
                memory=peak<=plan['budget']['maximum_peak_device_bytes'],not_interrupted=not stop[0]).items() if not ok])
        atomic_json(output/'pilot.json',pilot)
        return pilot
    if phase != 'train':raise ValueError('unknown direction continuation phase')
    pilot=checked_direction_release(output,meta,plan,registration)
    meta['pilot_checkpoint_sha256']=pilot['checkpoint_sha256']
    while meta['completed_steps'] < plan['training']['steps'] and not stop[0]:
        cost_history=meta['history'] or [dict(seconds=pilot['warm_step_seconds'])]
        if not reserve(cost_history):break
        tick=perf_counter();next_state,info,timing=rt['transition'](state)
        if not info['accepted']:
            atomic_json(output/'rejected_step.json',info)
            raise RuntimeError('rejected physical update; previous accepted state preserved')
        state=next_state;meta['completed_steps']+=1;meta['absolute_completed_steps']+=1
        meta['history'].append(dict(step=meta['completed_steps'],seconds=perf_counter()-tick,**timing,**info))
        shared.save_state(output/'checkpoint.pkl',state,meta)
        atomic_json(output/'progress.json',dict(status='running',completed_steps=meta['completed_steps'],
            absolute_completed_steps=meta['absolute_completed_steps'],last=meta['history'][-1]))
        print(json.dumps(meta['history'][-1]),flush=True)
    if stop[0] or meta['completed_steps'] != plan['training']['steps']:return pause()
    if not reserve(meta['history'],audit=True):return pause()
    audit=shared.terminal_audit(rt,state[0],plan['audit'],plan['seeds']['audit'],output)
    result=dict(status='fixed_steps_complete',completed_steps=meta['completed_steps'],
        absolute_completed_steps=meta['absolute_completed_steps'],optimizer_policy=meta['optimizer_policy'],
        parameter_count=mixed_parameter_count(state[0]),history=meta['history'],terminal_audit=audit,
        params_sha256=pytree_sha256(state[0]),checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
        registration_sha256=registration,parent=lineage,
        full_H_training_centers=sum(int(x['full_H_calls']) for x in meta['history']),
        uncommitted_direction_and_replay_training_H_centers=3*plan['training']['samples'],
        direction_and_replay_parent_H_centers=2*plan['direction_check']['samples'],
        direction_and_replay_tangent_H_JVP_centers=2*plan['direction_check']['samples'],
        terminal_audit_H_centers=plan['audit']['samples'],preflight_H_centers=4,
        fixed_endpoint_no_energy_selection=True,direction_gate_sha256=file_sha256(output/'direction_check.json'),
        calibration_H_centers=0,sampler_branch_sha256=file_sha256(output/'sampler_branch.json'))
    if plan['task'] == EVIDENCE_GUARDED_TASK:
        result.update(initial_slope_evidence=pilot['gate']['slope_evidence'],
                      initial_descent_confirmed=pilot['gate']['confirmed_descent'],
                      terminal_audit_is_quality_decision=True)
    atomic_json(output/'results.json',result)
    atomic_json(output/'progress.json',dict(status='fixed_steps_complete',completed_steps=meta['completed_steps'],
        absolute_completed_steps=meta['absolute_completed_steps']))
    return result


def run(plan: dict, phase: str, plan_sha256: str) -> dict:
    """Two accepted updates + isolated replay, then a fixed audited endpoint."""
    phase_started = perf_counter()
    validate_plan(plan)
    if canonical_sha256(plan) != plan_sha256:
        raise ValueError('registered stage plan changed')
    if plan['task'] in (OPTIMIZED_TASK, WIDE_PRODUCTION_TASK, *DIRECTION_TASKS) and datetime.now(timezone.utc) >= datetime.fromisoformat(
            plan['campaign_deadline_utc']):
        raise RuntimeError('six-day campaign deadline reached; no new numerical work')
    jax.config.update('jax_enable_x64', True)
    if plan['require_gpu'] and jax.default_backend() != 'gpu':
        raise RuntimeError('allocated GPU required')
    if plan['task'] in DIRECTION_TASKS and plan['require_gpu']:
        devices=jax.local_devices()
        if len(devices) != 1 or 'A100' not in devices[0].device_kind:
            raise RuntimeError('direction continuation requires exactly one allocated A100')
    parent, lineage = load_parent(plan); output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_sha256=plan_sha256, case=plan['case'], stage=plan['stage'], arity=plan['arity'],
        model=plan['model'], j2_over_j1=plan['j2_over_j1'], binary_rank=plan['binary_rank'], parent=lineage,
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    if plan['task'] in (SAME_WIDTH_TASK, QUALITY_TASK, OPTIMIZED_TASK, WIDE_PRODUCTION_TASK, *DIRECTION_TASKS):
        fingerprint = numerical_runtime_fingerprint()
        if fingerprint['node'] != plan['runtime_contract']['node']:
            raise ValueError('same-width continuation must stay on its pinned numerical node')
        identity['numerical_runtime'] = fingerprint
    reg = canonical_sha256(identity); registered = dict(identity, sha256=reg)
    if (output/'registration.json').exists() and json.loads((output/'registration.json').read_text()) != registered:
        raise ValueError('immutable stage registration changed')
    atomic_json(output/'registration.json', registered)
    widening = plan['model']['multiplicity_cap'] != plan['parent']['model']['multiplicity_cap']
    p = parent[0]
    if widening:
        p, expansion = initialize_live_triangular_expansion(MixedClebschConfig(**plan['parent']['model']), p,
            MixedClebschConfig(**plan['model']), jax.random.PRNGKey(plan['seeds']['expansion']),
            mode=plan['arity'], binary_rank=plan['binary_rank'], new_feature_scale=plan['new_feature_scale'])
    rt = runtime(plan, p)
    if (output/'checkpoint.pkl').exists():
        state, meta = shared.restore_state(output/'checkpoint.pkl', reg)
    else:
        branch_record=None
        reset_for_capacity_control = plan.get('optimization_control') in (
            'same_parent_same_width_Adam_reset', 'paired_capacity_reset_direct_SR')
        reset_optimizer = reset_for_capacity_control or (
            plan['task'] in (OPTIMIZED_TASK, WIDE_PRODUCTION_TASK) and plan['reset_optimizer'])
        moments = _manual_adam_init(p) if widening or reset_optimizer else parent[1]
        ck, cx = parent[4:]
        if plan['task'] in DIRECTION_TASKS:
            state,branch_record=initialize_resampled_chains(plan,parent,rt)
            ck,cx=state[4:]
        elif ck is None:
            ck, ci = jax.random.split(jax.random.PRNGKey(plan['seeds']['calibration']))
            cx = initialize_spin_exchange_chains(ci, n_chains=64, n_sites=36, n_up=18)
            prepared = rt['prepare'](p)
            cx, _, ck, _ = jax.jit(lambda key, x: run_exchange_sweeps(prepared, key, x,
                amplitude_fn=rt['projected'], particle_kind='spin',
                sweeps=plan['calibration']['initial_burn_proposals'], exchange_edges=rt['edges']))(ck, cx)
        if plan['task'] not in DIRECTION_TASKS:
            state = (p, moments, *parent[2:4], ck, cx)
        meta = dict(registration_sha256=reg, completed_steps=0, history=[],
            parent_absolute_completed_steps=lineage['absolute_completed_steps'],
            absolute_completed_steps=lineage['absolute_completed_steps'], optimizer_base_step=int(moments[0]),
            optimizer_policy=('reset_for_new_dimensions' if widening else
                              'explicit_reset_matching_wider_capacity_arm' if reset_for_capacity_control else
                              'explicit_reset_for_empirical_FS_direct_SR' if reset_optimizer else
                              'inherit_full_SR_Adam_moments'))
        if branch_record is not None:
            atomic_json(output/'sampler_branch.json',branch_record)
            meta['sampler_branch_sha256']=file_sha256(output/'sampler_branch.json')
            meta['parent_sampler_rng_sha256']=branch_record['parent_sampler_rng_sha256']
            meta['sampler_trajectory_inherited_bitwise']=False
            meta['optimizer_policy']='inherit_optimizer_counter_and_moments_direct_SR'
        if reset_for_capacity_control:
            atomic_json(output/'capacity_control_initialization.json',dict(
                same_parent_parameters_bitwise=pytree_sha256(p)==lineage['params_sha256'],
                parent_params_sha256=lineage['params_sha256'],params_sha256=pytree_sha256(p),
                reset_Adam_counter_and_moments=True,training_and_calibration_chains_and_RNG_inherited=True,
                matched_wider_arm=plan['matched_wider_arm']))
        elif reset_optimizer:
            # A same-width optimization screen resets old moments without a
            # paired capacity arm. Keep that generic provenance independent
            # of the legacy widening contract and its matched_wider_arm key.
            atomic_json(output/'optimizer_reset_initialization.json',dict(
                optimization_control=plan['optimization_control'],
                optimizer_policy=meta['optimizer_policy'],
                same_parent_parameters_bitwise=pytree_sha256(p)==lineage['params_sha256'],
                parent_params_sha256=lineage['params_sha256'],params_sha256=pytree_sha256(p),
                reset_Adam_counter_and_moments=True,
                training_and_calibration_chains_and_RNG_inherited=True))
        if widening:
            atomic_json(output/'expansion.json', dict(expansion,
                physical_preflight=expansion_preflight(plan, parent, p, rt)))
        shared.save_state(output/'checkpoint.pkl', state, meta)
    if (meta['completed_steps'] != len(meta['history'])
            or meta['absolute_completed_steps'] != meta['parent_absolute_completed_steps']+meta['completed_steps']
            or int(state[1][0]) != meta['optimizer_base_step']+meta['completed_steps']):
        raise ValueError('optimizer/lifetime/history counters disagree')
    if widening and json.loads((output/'expansion.json').read_text())['physical_preflight']['status'] != 'passed':
        raise ValueError('live projected inclusion preflight required')
    stop = [False]
    def request_stop(*_): stop[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM): signal.signal(sig, request_stop)

    if plan['task'] in DIRECTION_TASKS:
        if (np.shape(state[3]) != (256,36) or np.shape(state[5]) != (128,36)
                or meta.get('sampler_branch_sha256') != file_sha256(output/'sampler_branch.json')):
            raise ValueError('explicit new training/calibration sampler branch is incomplete or changed')
        return run_direction_phases(plan,phase,state,meta,rt,output,reg,lineage,stop,phase_started)

    def time_to_start(*, audit: bool = False) -> bool:
        if plan['task'] != WIDE_PRODUCTION_TASK:
            return True
        note = wide_production_time_guard(plan, phase, meta['history'],
            perf_counter()-phase_started, audit=audit)
        if not note['can_start']:
            stop[0] = True
            atomic_json(output/'time_guard.json', dict(phase=phase,
                completed_steps=meta['completed_steps'], **note))
        return bool(note['can_start'])

    def paused() -> dict:
        answer = dict(status='safe_pause', completed_steps=meta['completed_steps'],
                      absolute_completed_steps=meta['absolute_completed_steps'])
        atomic_json(output/'progress.json', answer)
        return answer

    if phase == 'replay':
        pilot = json.loads((output/'pilot.json').read_text())
        if (pilot['status'] != 'passed' or pilot['checkpoint_sha256'] != file_sha256(output/'checkpoint.pkl')
                or pilot.get('registration_sha256') != reg
                or pilot['fixture_sha256'] != file_sha256(output/'next_transition_fixture.pkl')
                or pilot['preflight_sha256'] != file_sha256(output/'preflight.json')):
            raise ValueError('passed unmodified pilot/fixture required')
        if plan['task'] == WIDE_PRODUCTION_TASK:
            validate_wide_pilot_cost(plan, pilot)
        if stop[0] or not time_to_start():
            return paused()
        expected = load_checkpoint(output/'next_transition_fixture.pkl')['params']
        new, info, _ = rt['transition'](state)
        report = compare_replay(expected, shared.transition_fixture(new, info),
            tolerance=plan['tolerance'], diagnostic_prefixes=('[1]',))
        atomic_json(output/'process_replay.json', dict(status='passed' if report['passed'] else 'failed',
            report=report, checkpoint_sha256=file_sha256(output/'checkpoint.pkl'), optimizer_updates_committed=0,
            registration_sha256=reg, fixture_sha256=file_sha256(output/'next_transition_fixture.pkl'),
            preflight_sha256=file_sha256(output/'preflight.json')))
        if not report['passed']:
            raise ArithmeticError('isolated full sampler/H/optimizer/RNG replay failed')
        return dict(status='replay_passed')
    if phase not in ('pilot', 'train'):
        raise ValueError('unknown phase')
    if phase == 'pilot' and not (output/'preflight.json').exists():
        atomic_json(output/'preflight.json', shared.preflight(rt, state[0], state[3], plan))
    if phase == 'train':
        meta['pilot_checkpoint_sha256'] = checked_training_release(output, meta, plan, reg)
    target = plan['pilot_steps'] if phase == 'pilot' else plan['training']['steps']; started = perf_counter()
    while meta['completed_steps'] < target and not stop[0]:
        if plan['task'] in (OPTIMIZED_TASK, WIDE_PRODUCTION_TASK) and datetime.now(timezone.utc) >= datetime.fromisoformat(
                plan['campaign_deadline_utc']):
            stop[0] = True
            break
        if not time_to_start():
            break
        tick = perf_counter(); state, info, timing = rt['transition'](state)
        if not info['accepted']:
            atomic_json(output/'rejected_step.json', info)
            raise RuntimeError('rejected/zero physical update; previous parent checkpoint preserved')
        meta['completed_steps'] += 1; meta['absolute_completed_steps'] += 1
        meta['history'].append(dict(step=meta['completed_steps'], seconds=perf_counter()-tick, **timing, **info))
        shared.save_state(output/'checkpoint.pkl', state, meta)
        atomic_json(output/'progress.json', dict(status='running', completed_steps=meta['completed_steps'],
            absolute_completed_steps=meta['absolute_completed_steps'], last=meta['history'][-1]))
        print(json.dumps(meta['history'][-1]), flush=True)
    if stop[0] or meta['completed_steps'] != target:
        return paused()
    if phase == 'pilot':
        if not time_to_start():
            return paused()
        new, info, _ = rt['transition'](state)
        save_checkpoint(output/'next_transition_fixture.pkl', shared.transition_fixture(new, info))
        warm = meta['history'][-1]['seconds']
        forecast = warm*(plan['training']['steps']-2+plan['audit']['samples']/plan['training']['samples'])+900.
        peak = max(int((d.memory_stats() or {}).get('peak_bytes_in_use', 0)) for d in jax.local_devices())
        passed = bool(info['accepted'] and forecast <= plan['budget']['train_seconds']
                      and peak <= plan['budget']['maximum_peak_device_bytes']
                      and (plan['task'] != WIDE_PRODUCTION_TASK or not plan['require_gpu'] or peak > 0))
        report = dict(status='passed' if passed else 'failed', completed_steps=2,
            registration_sha256=reg,
            checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
            fixture_sha256=file_sha256(output/'next_transition_fixture.pkl'),
            preflight_sha256=file_sha256(output/'preflight.json'), warm_step_seconds=warm,
            forecast_remaining_seconds=forecast, peak_device_bytes=peak,
            maximum_remaining_seconds=plan['budget']['train_seconds'])
        atomic_json(output/'pilot.json', report)
        if not passed:
            raise RuntimeError('pilot cannot release fixed remaining time/memory/update budget')
        return report
    if not time_to_start(audit=True):
        return paused()
    audit = shared.terminal_audit(rt, state[0], plan['audit'], plan['seeds']['audit'], output)
    result = dict(status='fixed_steps_complete', completed_steps=target,
        absolute_completed_steps=meta['absolute_completed_steps'], optimizer_policy=meta['optimizer_policy'],
        parameter_count=mixed_parameter_count(state[0]), history=meta['history'], terminal_audit=audit,
        params_sha256=pytree_sha256(state[0]), checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
        registration_sha256=reg, parent=lineage,
        **continuation_work_counts(plan, meta['history']),
        preflight_H_centers=4+(4 if widening else 0),
        covariance_convention='F=Cov(real logabs scores);g=2Cov(scores,E)',
        last_session_seconds=perf_counter()-started, fixed_endpoint_no_energy_selection=True)
    atomic_json(output/'results.json', result)
    atomic_json(output/'progress.json', dict(status='fixed_steps_complete', completed_steps=target,
        absolute_completed_steps=meta['absolute_completed_steps']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--phase', choices=('pilot', 'replay', 'train'), required=True)
    args = parser.parse_args()
    answer = run(json.loads(args.plan.read_text()), args.phase, args.plan_sha256)
    print(json.dumps({k: v for k, v in answer.items() if k != 'history'}), flush=True)
    if answer['status'] == 'safe_pause':
        raise SystemExit(75)
    if answer['status'] == 'failed':
        raise SystemExit(2)
