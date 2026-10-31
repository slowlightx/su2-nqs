"""Bounded continuation of native finite-width triangular ClebschTree states.

This reuses the verified native sampler, complete coherent A1/H action and
original-parameter ridge-SR transition. It never retile-transfers an N36 state,
changes a charge budget, inserts attention, or resets an inherited optimizer.
Each arity has an immutable parent and its own isolated pilot/replay/endpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_dictionary_vmc_local import (
    make_record_sampler, summarize_born_audit,
)
from projects.clebsch_tree_nqs.workflows.triangle_n144_train import make_training_transition
from projects.clebsch_tree_nqs.workflows.triangle_scale_cost import balanced_block_probe_states
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_parameter_count, mixed_clebsch_layout
from su2_nqs.su2.triangle_blocks import nine_site_output
from su2_nqs.models.triangular_clebsch_expansion import initialize_live_triangular_expansion
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.replay_validation import compare_replay
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('binary_truncated', 'ternary_joint')
DIAGNOSTIC_PREFIXES = ("[1]['last_diagnostics']", '[6]', '[7]')
RESUME_PROFILE = 'saved_pilot_budget_extension_v1'
COMPLETED_PROFILE = 'audited_parent_continuation_v2'
SAME_WIDTH_PROFILE = 'audited_same_width_global18_v3'


def continuation_runtime() -> dict:
    """Record the accelerator/compiler and scheduler node used for replay."""
    return dict(numerical_runtime_identity(), scheduler_node=os.environ.get('SLURMD_NODENAME'),
                device_kinds=[device.device_kind for device in jax.local_devices()])


def validate_plan(spec: dict) -> None:
    """Restrict this release to one fixed-width N144 native continuation."""
    if spec.get('training_release') is not True:
        raise ValueError('explicit main-agent training release required')
    profile = spec.get('release_profile')
    completed = profile in (COMPLETED_PROFILE, SAME_WIDTH_PROFILE)
    allowed_models = ([dict(length=12, multiplicity_cap=w, maximum_q=10) for w in (16, 20)]
                      if completed else [dict(length=12, multiplicity_cap=16, maximum_q=10)])
    if (spec['model'] not in allowed_models
            or spec['arm'] not in ARMS or spec['binary_rank'] != 8
            or spec['j2_over_j1'] not in (0., .125)):
        raise ValueError('this release fixes N144, registered width16/20 profiles, rank8 and native arities')
    integers = [spec['training'][k] for k in ('steps', 'chains', 'records_per_chain',
        'initial_burn_proposals', 'refresh_proposals', 'proposals_between')]
    integers += [spec['audit'][k] for k in ('chains', 'records_per_chain',
        'burn_proposals', 'proposals_between')]
    integers += [spec[k] for k in ('projection_chunk', 'action_chunk', 'pilot_steps')]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in integers):
        raise ValueError('positive integer counts required')
    if (spec['training']['steps'] not in ((8, 16) if completed else (8,)) or spec['pilot_steps'] != 2
            or spec['training']['chains'] != 8 or spec['training']['records_per_chain'] != (32 if completed else 16)
            or spec['audit']['chains']*spec['audit']['records_per_chain'] != 512):
        raise ValueError('fixed eight-update, 128-training/512-audit release required')
    if not 0 < spec['tolerance'] <= 2e-11:
        raise ValueError('strict replay tolerance required')
    block = spec['sr'].get('parameter_chunk_size')
    if block is not None and (isinstance(block, bool) or not isinstance(block, int) or not 1 <= block <= 32768):
        raise ValueError('SR parameter chunk must be an integer between1 and32768')
    if profile not in (None, RESUME_PROFILE, COMPLETED_PROFILE, SAME_WIDTH_PROFILE):
        raise ValueError('unknown native continuation release profile')
    if profile is None and 'resume_from' in spec:
        raise ValueError('saved-pilot adoption requires its explicit budget-extension profile')
    if profile == RESUME_PROFILE and not spec.get('resume_from'):
        raise ValueError('budget extension must resume a sealed saved pilot, not cold-start')
    if profile == COMPLETED_PROFILE:
        source_model = spec['parent']['model']
        if (source_model != dict(length=12, multiplicity_cap=16, maximum_q=10)
                or spec['parent']['global_optimizer_steps'] != 10
                or spec['parent']['arm'] != spec['arm']
                or spec['parent']['j2_over_j1'] != spec['j2_over_j1']
                or 'resume_from' in spec or not isinstance(spec.get('expansion_seed'), int)
                or isinstance(spec.get('expansion_seed'), bool) or not 0 <= spec['expansion_seed'] < 2**32):
            raise ValueError('completed-parent profile requires sealed native global10 parent and live same-cover expansion')
    if profile == SAME_WIDTH_PROFILE:
        parent = spec['parent']
        if (spec['model'] != dict(length=12, multiplicity_cap=20, maximum_q=10)
                or parent['model'] != spec['model'] or parent['global_optimizer_steps'] != 18
                or parent['arm'] != spec['arm'] or parent['j2_over_j1'] != spec['j2_over_j1']
                or spec['training']['steps'] != 8 or 'resume_from' in spec
                or 'expansion_seed' in spec or spec['sr'].get('parameter_chunk_size') != 8192):
            raise ValueError('same-width profile requires sealed global18 width20 parent and bounded8-step block-SR continuation')
    maximum_hours = 40 if completed else (8 if profile is None else 13)
    phases = spec['phase_seconds']
    if (set(phases) != {'pilot', 'replay', 'train'}
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in phases.values())
            or sum(phases.values()) > maximum_hours*3600):
        raise ValueError('aggregate per-arm allocation exceeds its explicit release')
    seed = spec['audit']['seed']
    if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
        raise ValueError('audit requires a preregistered uint32 seed')


def checked_parent(spec: dict) -> tuple[dict, dict]:
    """Verify the actual native optimized state, not its N36 initialization."""
    request = spec['parent']
    directory = Path(request['directory'])
    if spec.get('release_profile') in (COMPLETED_PROFILE, SAME_WIDTH_PROFILE):
        return checked_completed_parent(spec)
    for key, filename in (('registration', 'registration.json'), ('results', 'results.json')):
        if file_sha256(directory/filename) != request[key+'_sha256']:
            raise ValueError('parent '+key+' seal mismatch')
    registration = json.loads((directory/'registration.json').read_text())
    if registration['sha256'] != canonical_sha256({
            k: v for k, v in registration.items() if k != 'sha256'}):
        raise ValueError('parent canonical registration changed')
    cfg = registration['config']
    if any(cfg[k] != spec[k] for k in ('model', 'binary_rank', 'j2_over_j1')):
        raise ValueError('parent Hamiltonian/charge/arity family mismatch')
    results = json.loads((directory/'results.json').read_text())
    arm = spec['arm']
    record = results['arms'][arm]
    checkpoint = directory/arm/'checkpoint.pkl'
    if (results['status'] != 'completed' or record['status'] != 'completed'
            or record['completed_steps'] != 2 or record['accepted_updates'] != 2
            or file_sha256(checkpoint) != request['checkpoint_sha256']
            or request['checkpoint_sha256'] != record['checkpoint_sha256']):
        raise ValueError('completed two-update native parent required')
    saved = load_checkpoint(checkpoint)
    metadata = saved['metadata']
    if (metadata['registration_sha256'] != registration['sha256'] or metadata['arm'] != arm
            or metadata['params_sha256'] != request['params_sha256']
            or pytree_sha256(saved['params']) != request['params_sha256']
            or metadata['optimizer_sha256'] != pytree_sha256(saved['optimizer_state'])
            or metadata['sampler_rng_sha256'] != pytree_sha256((
                jnp.asarray(metadata['sampler_states']), jnp.asarray(metadata['rng'])))
            or int(saved['optimizer_state']['completed_steps']) != 2
            or int(saved['optimizer_state']['accepted_updates']) != 2
            or len(metadata['history']) != 2
            or np.asarray(metadata['sampler_states']).shape != (8, 144)
            or float(saved['optimizer_state']['damping']) != spec['sr']['damping']):
        raise ValueError('native parameter/optimizer/sampler/RNG/counter seal mismatch')
    audit = json.loads((directory/arm/'audit.json').read_text())
    # The older N144 audit seals the effective physical bank, NOT the native
    # binary factor parameters. Compare the correct representation explicitly.
    # A binary fold performs floating einsums: its bit hash is backend-specific.
    # Seal the original GPU-audited bank here; replay physical amplitudes below
    # with the unchanged relative tolerance instead of requiring CPU/GPU bits.
    if (audit != record['terminal_audit'] or audit['params_sha256'] != request['audited_bank_sha256']
            or audit['registration_sha256'] != registration['sha256']
            or audit.get('independent_of_training') is not True
            or file_sha256(directory/arm/'audit_samples.npz') != audit['raw_samples_sha256']):
        raise ValueError('parent native arity and audited physical-bank identity differ')
    if spec['audit']['seed'] in [cfg['seeds']['training'], cfg['seeds']['initial_audit'],
                               *cfg['seeds']['final_audits'].values()]:
        raise ValueError('audit RNG must be independent of all parent streams')
    return saved, registration


def checked_completed_parent(spec: dict) -> tuple[dict, dict]:
    """Read a completed native N144 endpoint, with parameters and raw audit sealed."""
    request = spec['parent']; directory = Path(request['directory'])
    for key, name in (('registration', 'registration.json'), ('results', 'results.json'),
                      ('checkpoint', 'checkpoint.pkl'), ('audit', 'terminal_audit.json'),
                      ('raw_samples', 'terminal_audit_samples.npz')):
        if file_sha256(directory/name) != request[key+'_sha256']:
            raise ValueError('completed parent '+key+' file changed')
    registration = json.loads((directory/'registration.json').read_text())
    if registration['sha256'] != canonical_sha256({k:v for k,v in registration.items() if k != 'sha256'}):
        raise ValueError('completed parent canonical registration changed')
    cfg = registration['config']
    if any(cfg[k] != spec['parent'][k] for k in ('model', 'arm', 'j2_over_j1')) or cfg['binary_rank'] != spec['binary_rank']:
        raise ValueError('completed parent model/H/arity differs')
    if spec.get('release_profile') == SAME_WIDTH_PROFILE:
        training_keys = ('chains', 'records_per_chain', 'initial_burn_proposals',
                         'refresh_proposals', 'proposals_between')
        if (any(cfg[k] != spec[k] for k in ('sr', 'sampler', 'projection_chunk', 'action_chunk'))
                or any(cfg['training'][k] != spec['training'][k] for k in training_keys)):
            raise ValueError('same-width continuation must retain parent optimizer, sampler and execution options')
    result = json.loads((directory/'results.json').read_text())
    audit = json.loads((directory/'terminal_audit.json').read_text())
    saved = load_checkpoint(directory/'checkpoint.pkl'); meta=saved['metadata']; opt=saved['optimizer_state']
    if (result['status'] != 'completed' or result['global_optimizer_steps'] != request['global_optimizer_steps']
            or result['terminal_audit'] != audit or not audit['untouched'] or not audit['fixed_endpoint']
            or audit['checkpoint_sha256'] != request['checkpoint_sha256']
            or audit['raw_samples_sha256'] != request['raw_samples_sha256']
            or meta['registration_sha256'] != registration['sha256'] or meta['arm'] != spec['arm']
            or audit['params_sha256'] != request['params_sha256']
            or meta['params_sha256'] != request['params_sha256']
            or pytree_sha256(saved['params']) != request['params_sha256']
            or pytree_sha256(opt) != meta['optimizer_sha256']
            or pytree_sha256((jnp.asarray(meta['sampler_states']),jnp.asarray(meta['rng']))) != meta['sampler_rng_sha256']
            or int(opt['completed_steps']) != request['global_optimizer_steps']
            or int(opt['accepted_updates']) != request['global_optimizer_steps']
            or np.asarray(meta['sampler_states']).shape != (spec['training']['chains'], 144)
            or spec['audit']['seed'] == cfg['audit']['seed']):
        raise ValueError('completed parent endpoint/audit/optimizer/RNG identity mismatch')
    return saved, registration


def initialize_completed_parent(spec: dict, saved: dict) -> tuple[dict, dict]:
    """Keep the audited parent ray and native sampler/RNG; start a new stage ledger.

    Widening changes the SR coordinate dimension but stateless SR has no Adam
    moments to transfer. The counters remain lifetime counters; old diagnostic
    arrays are not represented as optimizer data for the larger tangent space.
    """
    source = MixedClebschConfig(**spec['parent']['model'])
    target = MixedClebschConfig(**spec['model'])
    params = saved['params']
    report = dict(same_width=True, exact_same_physical_state=True)
    if target.multiplicity_cap != source.multiplicity_cap:
        params, report = initialize_live_triangular_expansion(source, params, target,
            jax.random.PRNGKey(spec['expansion_seed']), mode=spec['arm'], binary_rank=spec['binary_rank'])
    same_width_continuation = spec.get('release_profile') == SAME_WIDTH_PROFILE
    optimizer = (saved['optimizer_state'] if same_width_continuation else
                 {k:saved['optimizer_state'][k] for k in ('completed_steps','accepted_updates','damping')})
    if float(optimizer['damping']) != spec['sr']['damping']:
        raise ValueError('completed-parent continuation must retain registered SR damping')
    parent_history = saved['metadata']['new_history']
    if same_width_continuation:
        parent_history = saved['metadata'].get('inherited_history', []) + parent_history
    meta = dict(saved['metadata'], history=[], new_history=[], completed_new_steps=0,
                optimizer_base_step=spec['parent']['global_optimizer_steps'],
                parent_history=parent_history)
    if same_width_continuation:
        report.update(optimizer_reset=False, optimizer_sha256=pytree_sha256(optimizer),
                      sampler_reset=False, rng_reset=False, expansion_performed=False)
    return dict(params=params, optimizer_state=optimizer, metadata=meta), report


def added_coordinate_mask(source: MixedClebschConfig, target: MixedClebschConfig, params: dict) -> dict:
    """Mark newly opened reduced slots, independent of their numerical values."""
    if source.length != target.length or source.maximum_q != target.maximum_q:
        raise ValueError('new-slot accounting needs an unchanged cover and doubled charges')
    mask = jax.tree.map(lambda v: jnp.zeros(v.shape, dtype=bool), params)
    mask['local'] = tuple(tuple(jnp.arange(w.shape[1])[None, :] >=
        nine_site_output(q, source.multiplicity_cap).width for q,w in zip(target.local_q,node,strict=True))
        for node in params['local'])
    # Broadcast local masks across all complete incoming Schur channels.
    mask['local'] = tuple(tuple(jnp.broadcast_to(m, w.shape) for m,w in zip(node,weights,strict=True))
                          for node,weights in zip(mask['local'],params['local'],strict=True))
    rows=[]
    for old, new in zip(mixed_clebsch_layout(source),mixed_clebsch_layout(target),strict=True):
        shapes={key:group['shape'][1:] for group in old['groups'] for key in group['keys']}
        groups=[]
        for group in new['groups']:
            bits=np.ones(group['shape'],dtype=bool)
            for index,key in enumerate(group['keys']):
                bits[(index,)+tuple(slice(v) for v in shapes[key])]=False
            groups.append(jnp.asarray(bits))
        rows.append(tuple(groups))
    mask['top']=tuple(rows)
    return mask


def checked_checkpoint(path: Path, registration_sha: str, parent_sha: str, *, optimizer_base_step: int = 2) -> dict:
    """Load an atomic continuation with explicit inherited/global counters."""
    saved = load_checkpoint(path)
    meta, opt = saved['metadata'], saved['optimizer_state']
    n = meta['completed_new_steps']
    if (meta['registration_sha256'] != registration_sha or meta['parent_checkpoint_sha256'] != parent_sha
            or meta['params_sha256'] != pytree_sha256(saved['params'])
            or meta['optimizer_sha256'] != pytree_sha256(opt)
            or meta['sampler_rng_sha256'] != pytree_sha256((
                jnp.asarray(meta['sampler_states']), jnp.asarray(meta['rng'])))
            or len(meta['new_history']) != n or int(opt['completed_steps']) != optimizer_base_step+n
            or int(opt['accepted_updates']) != optimizer_base_step+n):
        raise ValueError('continuation checkpoint/counter identity mismatch')
    return saved


def checked_saved_pilot(spec: dict) -> dict:
    """Adopt only the sealed, valid two-update pilot stopped by its time budget.

    Parameters, optimizer, chains and RNG are inherited exactly. Old failures
    remain immutable: the new budget/source registration requires a new pilot
    fixture and an independent process replay, never relabels an old gate.
    """
    if spec.get('release_profile') != RESUME_PROFILE:
        raise ValueError('explicit saved-pilot continuation release required')
    request = spec['resume_from']; directory = Path(request['directory'])
    for name in ('registration', 'results', 'checkpoint', 'pilot'):
        suffix = '.pkl' if name == 'checkpoint' else '.json'
        if file_sha256(directory/(name+suffix)) != request[name+'_sha256']:
            raise ValueError('saved pilot '+name+' seal mismatch')
    registration = json.loads((directory/'registration.json').read_text())
    if registration['sha256'] != canonical_sha256({
            k: v for k, v in registration.items() if k != 'sha256'}):
        raise ValueError('saved pilot canonical registration changed')
    previous = registration['config']
    fixed_fields = ('model', 'j2_over_j1', 'arm', 'binary_rank', 'projection_chunk',
        'action_chunk', 'tolerance', 'require_gpu', 'parent', 'training', 'pilot_steps',
        'sampler', 'sr')
    if any(previous[k] != spec[k] for k in fixed_fields):
        raise ValueError('saved-pilot budget extension may not alter physics or training')
    if ({k: v for k, v in previous['audit'].items() if k != 'seed'} !=
            {k: v for k, v in spec['audit'].items() if k != 'seed'}
            or previous['audit']['seed'] == spec['audit']['seed']):
        raise ValueError('same audit size and a fresh preregistered audit seed required')
    pilot = json.loads((directory/'pilot.json').read_text())
    result = json.loads((directory/'results.json').read_text())
    if (pilot.get('status') != 'budget_failed' or pilot.get('memory_gate_passed') is not True
            or pilot.get('registration_sha256') != registration['sha256']
            or pilot.get('checkpoint_sha256') != request['checkpoint_sha256']
            or not np.isfinite(pilot.get('remaining_train_forecast_seconds', np.nan))
            or pilot['remaining_train_forecast_seconds'] <= previous['phase_seconds']['train']
            or result.get('status') != 'failed' or result.get('phase') != 'pilot'
            or result.get('completed_new_steps') != 2
            or result.get('error') != "RuntimeError('measured continuation forecast or memory exceeds preregistered allocation')"):
        raise ValueError('only a memory-safe pilot stopped solely by its time budget may resume')
    saved = checked_checkpoint(directory/'checkpoint.pkl', registration['sha256'],
        spec['parent']['checkpoint_sha256'])
    history = saved['metadata']['new_history']
    if (saved['metadata']['completed_new_steps'] != 2 or result['history'] != history
            or any(not row['SR']['accepted'] or not 0 < row['SR']['sampled_fs_sine'] <=
                spec['sr']['maximum_overlap_distance'] or not np.isfinite(row['seconds'])
                or row['seconds'] <= 0 for row in history)):
        raise ValueError('two accepted, nonzero, sealed physical pilot updates required')
    return saved


def remaining_work_forecast(history: list[dict], *, target_steps: int,
        audit_centers: int, training_centers: int) -> float:
    """Count cold work once; inherited warm timings are not spent again.

    Retain two warm-equivalent units for compilation/audit slack and a15%
    margin. The original two-update history excludes its first compiled step.
    """
    if (len(history) < 2 or target_steps < len(history)
            or audit_centers < 1 or training_centers < 1):
        raise ValueError('two measured steps and positive remaining-work sizes required')
    warm = max(float(row['seconds']) for row in history[1:])
    if not np.isfinite(warm) or warm <= 0:
        raise ValueError('finite positive warm timing required')
    units = target_steps-len(history)+audit_centers/training_centers+2
    return warm*units*1.15


def phase_has_room(history: list[dict], elapsed: float, limit: float, *, units: float = 1.) -> bool:
    """Pause before starting an update/audit that is unlikely to fit safely."""
    if not np.isfinite(elapsed) or elapsed < 0 or not np.isfinite(limit) or limit <= 0 or units <= 0:
        raise ValueError('finite positive phase budget required')
    measured = [float(row['seconds']) for row in history[-3:]]
    if any(not np.isfinite(x) or x <= 0 for x in measured):
        raise ValueError('invalid recorded timing')
    reserve = max(1800., (max(measured)*1.25*units+300.) if measured else 1800.)
    return elapsed < limit-reserve


def validate_committed_update(old_params: object, changed: tuple, new_step: int, *,
        maximum_fs_sine: float = .01, minimum_ess_fraction: float = .98, optimizer_base_step: int = 2) -> None:
    """Reject zero/rejected updates and mismatched counters before persistence."""
    params, optimizer, _, _, _, energies, _, diagnostic = changed
    fs = float(diagnostic['sampled_fs_sine'])
    ess = float(diagnostic['reweighting_ess_fraction'])
    if (not bool(diagnostic['accepted'])
            or int(optimizer['completed_steps']) != optimizer_base_step+new_step
            or int(optimizer['accepted_updates']) != optimizer_base_step+new_step
            or not np.isfinite(np.asarray(energies)).all()
            or any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))
            or not np.isfinite(fs) or not 0 < fs <= maximum_fs_sine
            or not np.isfinite(ess) or not minimum_ess_fraction <= ess <= 1.
            or pytree_sha256(params) == pytree_sha256(old_params)):
        raise ArithmeticError('rejected, nonfinite, zero or miscounted update cannot complete training')


def checked_training_release(output: Path, saved: dict, registration_sha: str) -> str:
    """Bind training/resume to the actual independently replayed pilot state."""
    pilot = json.loads((output/'pilot.json').read_text())
    replay = json.loads((output/'process_replay.json').read_text())
    report = replay.get('replay_validation', {})
    if (pilot.get('status') != 'passed' or replay.get('status') != 'passed'
            or pilot.get('registration_sha256') != registration_sha
            or replay.get('registration_sha256') != registration_sha
            or replay.get('checkpoint_sha256') != pilot.get('checkpoint_sha256')
            or replay.get('fixture_sha256') != pilot.get('fixture_sha256')
            or file_sha256(output/'next_transition_fixture.pkl') != pilot.get('fixture_sha256')
            or report.get('passed') is not True or report.get('protocol') != 'exact_discrete_relative_float_v1'
            or not 0 < report.get('tolerance', 1) <= 2e-11
            or not report.get('leaves') or any(x.get('passed') is not True or x.get('finite') is not True
                or (x.get('bitwise_exact') is not True if x.get('discrete') else
                    not np.isfinite(x.get('relative_vector_error', np.inf))
                    or not (x['relative_vector_error'] <= report['tolerance']
                        or x.get('natural_unit_scalar_diagnostic') is True and x.get('shape') == []
                        and x.get('absolute_error', np.inf) <= report['tolerance']))
                for x in report['leaves'])):
        raise ValueError('sealed same-registration pilot/full-transition replay required')
    n = saved['metadata']['completed_new_steps']
    if ((n == 2 and file_sha256(output/'checkpoint.pkl') != pilot['checkpoint_sha256'])
            or n > 2 and saved['metadata'].get('pilot_checkpoint_sha256') != pilot['checkpoint_sha256']
            or n < 2):
        raise ValueError('training checkpoint does not descend from replayed pilot')
    return pilot['checkpoint_sha256']


def run(config_path: Path, *, phase: str) -> dict:
    """Execute two updates, independent-process replay, or the fixed endpoint."""
    if phase not in ('pilot', 'replay', 'train'):
        raise ValueError('unknown phase')
    started = perf_counter()  # Include parent validation/preparation/compilation.
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text()); validate_plan(spec)
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('actual A100 GPU required')
    parent, parent_registration = checked_parent(spec)
    completed_parent = spec.get('release_profile') in (COMPLETED_PROFILE, SAME_WIDTH_PROFILE)
    optimizer_base_step = spec['parent']['global_optimizer_steps'] if completed_parent else 2
    training_centers = spec['training']['chains']*spec['training']['records_per_chain']
    initialization = None
    if completed_parent:
        parent, initialization = initialize_completed_parent(spec, parent)
    output = Path(spec['output'])
    if not output.exists():
        if phase != 'pilot':
            raise FileNotFoundError('pilot must run before replay/training')
        output.mkdir(parents=True)
        sources = transitive_local_source_manifest([Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'],
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
        registration = dict(config=spec, config_sha256=file_sha256(config_path), sources=sources,
            parent_registration_sha256=parent_registration['sha256'], runtime=(
                continuation_runtime() if completed_parent else numerical_runtime_identity()),
            created_utc=datetime.now(timezone.utc).isoformat())
        registration['sha256'] = canonical_sha256(registration)
        atomic_json(output/'registration.json', registration)
        result = dict(status='registered', completed_new_steps=0, history=[],
            exact_S0=True, full_A1_images=1728, arity=spec['arm'], new_architecture=False,
            parent_is_competitive=False, equilibrium_certified=False,
            parent_checkpoint_sha256=spec['parent']['checkpoint_sha256'])
        if completed_parent:
            result.update(initialization=initialization, width_changed=spec['model'] != spec['parent']['model'],
                new_architecture=spec['model'] != spec['parent']['model'],
                optimizer_base_step=optimizer_base_step, optimizer='stateless_ridge_SR',
                optimizer_moments_transferred=False, sampler_and_rng_inherited=True)
        atomic_json(output/'results.json', result)
    else:
        registration = json.loads((output/'registration.json').read_text())
        if (registration['sha256'] != canonical_sha256({k: v for k, v in registration.items() if k != 'sha256'})
                or registration['config_sha256'] != file_sha256(config_path)
                or registration['config'] != spec or any(file_sha256(ROOT/p) != h
                    for p, h in registration['sources'].items())):
            raise ValueError('frozen continuation source/config changed')
        if completed_parent and registration['runtime'] != continuation_runtime():
            raise ValueError('continuation replay accelerator/compiler/node changed')
        result = json.loads((output/'results.json').read_text())
    model = MixedClebschConfig(**spec['model'])
    maps = np.asarray(triangular_hnf_gamma_a1_group(((12, 0), (0, 12))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(12, 12, j2=spec['j2_over_j1'], periodic=True))
    if len(maps) != 1728 or len(bonds) != (432 if spec['j2_over_j1'] == 0 else 864):
        raise ArithmeticError('full A1/H identity changed')
    amplitude = make_feature_lut_projected_amplitude(model, maps, chunk=spec['projection_chunk'])
    action = make_cached_projected_heisenberg_action(model, maps, bonds,
        projection_chunk=spec['projection_chunk'])
    prepare = lambda p: prepare_feature_lut_bank(arity_clebsch_bank(
        p, model, spec['arm'], spec['binary_rank']), model)

    @jax.jit
    def evaluate(prepared, rows):
        def one(row):
            psi, hp = action(prepared, row[None])
            return jnp.stack((psi[0], hp[0]))
        values = jax.lax.map(one, rows, batch_size=spec['action_chunk'])
        return values[:, 0], values[:, 1]

    transition = make_training_transition(prepare, amplitude, evaluate, parent['params'], n_sites=144,
        training=spec['training'], sampler_options=spec['sampler'],
        exchange_edges=tuple((a, b) for a, b, weight in bonds if weight != 0), sr_options=spec['sr'])
    new_slots = (added_coordinate_mask(MixedClebschConfig(**spec['parent']['model']), model, parent['params'])
                 if completed_parent else None)
    @jax.jit
    def participation(previous, candidate):
        delta=jax.tree.map(lambda a,b:b-a,previous,candidate)
        total=sum(jnp.vdot(v,v).real for v in jax.tree.leaves(delta))
        newly=sum(jnp.sum(jnp.where(m,abs(v)**2,0.)) for m,v in
                  zip(jax.tree.leaves(new_slots),jax.tree.leaves(delta),strict=True))
        return jnp.sqrt(newly),jnp.sqrt(newly/jnp.maximum(total,1e-300))
    # A tiny frozen-audit replay binds the native checkpoint to the actual
    # physical state evaluated by the old run, including binary folding and A1.
    raw_parent_path = (Path(spec['parent']['directory'])/'terminal_audit_samples.npz' if completed_parent
                       else Path(spec['parent']['directory'])/spec['arm']/'audit_samples.npz')
    with np.load(raw_parent_path) as raw_parent:
        parent_rows = np.asarray(raw_parent['bits']).reshape(-1, 144)[:2]
        parent_reference = np.asarray(raw_parent['psi']).reshape(-1)[:2]
    parent_prepared = jax.jit(prepare)(parent['params'])
    parent_actual = np.asarray(jax.jit(jax.vmap(amplitude, in_axes=(None, 0)))(
        parent_prepared, jnp.asarray(parent_rows))).astype(parent_reference.dtype)
    parent_report = compare_replay(parent_reference, parent_actual, tolerance=spec['tolerance'])
    if not parent_report['passed']:
        raise ArithmeticError('frozen audited physical parent amplitude differs')
    result['parent_physical_amplitude_replay'] = parent_report
    checkpoint = output/'checkpoint.pkl'
    if checkpoint.exists():
        saved = checked_checkpoint(checkpoint, registration['sha256'], spec['parent']['checkpoint_sha256'],
                                   optimizer_base_step=optimizer_base_step)
    elif spec.get('resume_from'):
        saved = checked_saved_pilot(spec)
        metadata = dict(saved['metadata'], registration_sha256=registration['sha256'],
            pilot_checkpoint_sha256=None, adopted_saved_pilot=spec['resume_from'])
        save_checkpoint(checkpoint, saved['params'], optimizer_state=saved['optimizer_state'], metadata=metadata)
        saved = checked_checkpoint(checkpoint, registration['sha256'], spec['parent']['checkpoint_sha256'])
        result['adopted_saved_pilot'] = dict(spec['resume_from'], optimizer_reset=False,
            sampler_reset=False, rng_reset=False, inherited_new_steps=2,
            old_failed_gate_is_not_passed=True)
    else:
        saved = parent
    params, optimizer = saved['params'], saved['optimizer_state']
    states, key = jnp.asarray(saved['metadata']['sampler_states']), jnp.asarray(saved['metadata']['rng'])
    history = saved['metadata'].get('new_history', [])
    inherited_phase_steps = len(history)
    pilot_checkpoint_sha = (checked_training_release(output, saved, registration['sha256'])
                            if phase == 'train' else None)
    if result['completed_new_steps'] > len(history):
        raise ValueError('result ledger is ahead of atomic checkpoint')
    result.update(history=history, completed_new_steps=len(history))
    stop = []
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda signum, _: stop.append(signum))
    def persist(status):
        result.update(status=status, history=history, completed_new_steps=len(history),
            global_optimizer_steps=int(optimizer['completed_steps']),
            phase=phase, phase_seconds=perf_counter()-started)
        atomic_json(output/'results.json', result)
    try:
        if phase == 'replay':
            pilot = json.loads((output/'pilot.json').read_text())
            fixture_path = output/'next_transition_fixture.pkl'
            if (pilot['status'] != 'passed' or file_sha256(checkpoint) != pilot['checkpoint_sha256']
                    or file_sha256(fixture_path) != pilot['fixture_sha256']
                    or pilot.get('registration_sha256') != registration['sha256']):
                raise ValueError('pilot replay seals changed')
            actual = transition(params, optimizer, states, key)
            report = compare_replay(load_checkpoint(fixture_path)['params'], actual,
                tolerance=spec['tolerance'], diagnostic_prefixes=DIAGNOSTIC_PREFIXES)
            atomic_json(output/'process_replay.json', dict(status='passed' if report['passed'] else 'failed',
                replay_validation=report, optimizer_updates_committed=0,
                checkpoint_sha256=pilot['checkpoint_sha256'], fixture_sha256=pilot['fixture_sha256'],
                registration_sha256=registration['sha256'], phase_seconds=perf_counter()-started,
                full_H_centers=training_centers))
            if not report['passed']:
                raise ArithmeticError('native full transition process replay differs')
            return report
        limit = 2 if phase == 'pilot' else spec['training']['steps']
        for step in range(len(history), limit):
            if stop or not phase_has_room(history, perf_counter()-started, spec['phase_seconds'][phase]):
                persist('safe_pause'); return result
            tick = perf_counter()
            changed = transition(params, optimizer, states, key)
            validate_committed_update(params, changed, step+1,
                maximum_fs_sine=spec['sr']['maximum_overlap_distance'],
                minimum_ess_fraction=spec['sr']['minimum_ess_fraction'], optimizer_base_step=optimizer_base_step)
            candidate, next_optimizer, next_states, next_key, rows, energies, sampling, diagnostic = changed
            activated = (participation(params,candidate) if completed_parent else None)
            params, optimizer, states, key = candidate, next_optimizer, next_states, next_key
            plain = lambda tree: jax.tree.map(lambda x: np.asarray(x).tolist(), tree)
            history.append(dict(step=step+1, global_step=int(optimizer['completed_steps']),
                seconds=perf_counter()-tick, training_energy_per_site=float(energies.real.mean()/144),
                variance_per_site=float(jnp.mean(abs(energies-energies.mean())**2)/144),
                SR=plain(diagnostic), sampling=plain(sampling), full_H_centers=training_centers))
            if activated is not None:
                history[-1]['new_coordinate_update_norm']=float(activated[0])
                history[-1]['new_coordinate_fraction_of_update_norm']=float(activated[1])
            save_checkpoint(checkpoint, params, optimizer_state=optimizer, metadata=dict(
                registration_sha256=registration['sha256'], parent_checkpoint_sha256=spec['parent']['checkpoint_sha256'],
                params_sha256=pytree_sha256(params), optimizer_sha256=pytree_sha256(optimizer),
                sampler_rng_sha256=pytree_sha256((states, key)), sampler_states=np.asarray(states), rng=np.asarray(key),
                arm=spec['arm'], completed_new_steps=len(history), new_history=history,
                pilot_checkpoint_sha256=pilot_checkpoint_sha,
                inherited_history=parent['metadata'].get('parent_history', parent['metadata'].get('history', [])),
                optimizer_base_step=optimizer_base_step, fixed_terminal_new_step=spec['training']['steps']))
            persist('training')
            print(json.dumps(history[-1]), flush=True)
        if stop:
            persist('safe_pause'); return result
        if phase == 'pilot':
            if not phase_has_room(history, perf_counter()-started, spec['phase_seconds']['pilot']):
                persist('safe_pause'); return result
            expected = transition(params, optimizer, states, key)
            if stop:
                persist('safe_pause'); return result
            save_checkpoint(output/'next_transition_fixture.pkl', expected, metadata=dict(uncommitted=True))
            forecast = remaining_work_forecast(history, target_steps=spec['training']['steps'],
                audit_centers=spec['audit']['chains']*spec['audit']['records_per_chain'],
                training_centers=spec['training']['chains']*spec['training']['records_per_chain'])
            memory = [device.memory_stats() for device in jax.local_devices()
                      if device.memory_stats() is not None]
            peak = max((item.get('peak_bytes_in_use', 0) for item in memory), default=0)
            passed = forecast < spec['phase_seconds']['train'] and 0 < peak < 70*2**30
            atomic_json(output/'pilot.json', dict(status='passed' if passed else 'budget_failed',
                registration_sha256=registration['sha256'],
                checkpoint_sha256=file_sha256(checkpoint), fixture_sha256=file_sha256(output/'next_transition_fixture.pkl'),
                remaining_train_forecast_seconds=forecast, measured_two_update_seconds=[h['seconds'] for h in history],
                parameter_count=mixed_parameter_count(params),
                full_H_centers=(len(history)-inherited_phase_steps+1)*training_centers,
                inherited_history_H_centers=sum(h['full_H_centers'] for h in history[:inherited_phase_steps]),
                this_phase_committed_updates=len(history)-inherited_phase_steps,
                uncommitted_fixture_H_centers=training_centers,
                timing_includes_parent_validation_and_compilation=True,
                device_memory=memory, memory_gate_passed=0 < peak < 70*2**30,
                phase_seconds=perf_counter()-started))
            persist('pilot_passed' if passed else 'budget_failed')
            if not passed:
                raise RuntimeError('measured continuation forecast or memory exceeds preregistered allocation')
            return result
        options = spec['audit']
        if stop or not phase_has_room(history, perf_counter()-started,
                spec['phase_seconds']['train'], units=options['chains']*options['records_per_chain']/training_centers):
            persist('safe_pause'); return result
        sampler = make_record_sampler(amplitude, records=options['records_per_chain'],
            burn_proposals=options['burn_proposals'], proposals_between=options['proposals_between'],
            exchange_edges=tuple((a, b) for a, b, weight in bonds if weight != 0), **spec['sampler'])
        audit_states = jnp.asarray(balanced_block_probe_states(model, count=options['chains'], seed=options['seed']+1))
        bank = jax.jit(prepare)(params)
        audit_states, audit_key, rows, info = jax.block_until_ready(
            sampler(bank, jax.random.PRNGKey(options['seed']), audit_states))
        psi, hp = jax.block_until_ready(evaluate(bank, rows.reshape(-1, 144)))
        if np.any(np.asarray(psi) == 0) or not np.isfinite(np.asarray(hp/psi)).all():
            raise FloatingPointError('invalid untouched audit energies; no clipping/flooring')
        energies = np.asarray(hp/psi).reshape(rows.shape[:2])
        raw = output/'terminal_audit_samples.npz'
        np.savez_compressed(raw, bits=np.asarray(rows), psi=np.asarray(psi), Hpsi=np.asarray(hp),
            local_energy=energies, rng=np.asarray(audit_key), states=np.asarray(audit_states))
        record = summarize_born_audit(energies, np.asarray(rows), n_sites=144)
        record.update(params_sha256=pytree_sha256(params), checkpoint_sha256=file_sha256(checkpoint),
            seed=options['seed'], raw_samples_sha256=file_sha256(raw), untouched=True, fixed_endpoint=True)
        atomic_json(output/'terminal_audit.json', record)
        result['terminal_audit'] = record
        persist('completed')
        return result
    except BaseException as error:
        result['error'] = repr(error); persist('failed'); raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('pilot', 'replay', 'train'), required=True)
    args = parser.parse_args()
    record = run(args.config, phase=args.phase)
    print(json.dumps(record, indent=2))
    if record.get('status') == 'safe_pause':
        raise SystemExit(75)
