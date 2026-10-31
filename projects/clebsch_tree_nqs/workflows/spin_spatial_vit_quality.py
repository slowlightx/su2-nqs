"""One resumable full Spatial-ViT family and its frozen quality/readout series.

No carrier, small head, SU(2) decoder, N36 Hilbert enumeration or ED. Every
network parameter trains with the existing real-parameter complex SR solver.
Spatial projection stages and quality milestones are fixed before training.
Audit streams are fresh and never enter training/step/checkpoint selection.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.spatial_vit_physics import (
    make_physical_spatial_vit_log, make_scalar_structure_factors,
    born_conditional_regional_matrices,
    scalar_structure_query_plan, scalar_structure_records,
)
from su2_nqs.benchmarks.su2_physics_protocol import (
    multiplicity_operator_data, rotational_symmetry_defect,
    rotational_defect_chain_audit,
)
from su2_nqs.benchmarks.spatial_vit_sampling import (
    make_log_exchange_sampler, make_full_heisenberg_ratios, projected_log_function,
)
from su2_nqs.benchmarks.spin_partition_readout import _spin_projectors, spin_partition_readout
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
from su2_nqs.benchmarks.singlet_physics_readout import compact6_channel_operators
from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit,
    fixed_sz_spatial_vit_parameter_count, fixed_sz_spatial_vit_log_amplitude,
)
from su2_nqs.models.literature_spatial_vit import (
    LiteratureSpatialViTConfig, init_literature_spatial_vit,
    literature_spatial_vit_log_amplitude,
)
from su2_nqs.benchmarks.spatial_vit_sampling import patch_translation_cosets
from su2_nqs.models.spatial_vit_transfer import transfer_spatial_vit_size
from su2_nqs.models.spatial_vit_capacity import expand_spatial_vit_capacity
from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint
from su2_nqs.vmc.dual_sampled_sr import make_dual_sampled_sr_update
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.regional_density import (
    regional_query_plan, regional_density_summary, regional_observable_summary,
)
from .spin import _bonds, _groups
from .spin_full_vit_pilot import audit_statistics

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_spatial_vit_quality.json'
CONTINUATION_DEFAULT = DEFAULT.with_name('spin_spatial_vit_quality_continuation.json')
STABLE_CONTINUATION_DEFAULT = DEFAULT.with_name('spin_spatial_vit_quality_stable_continuation.json')


def sealed_json(path: Path, payload: dict) -> dict:
    """Atomically write a new record with its canonical content digest."""
    result = dict(payload)
    result['result_sha256'] = canonical_sha256(result)
    atomic_json(path, result)
    return result


def read_sealed(path: Path) -> dict:
    """Verify one local numerical record; no historical result is re-signed."""
    record = json.loads(path.read_text())
    if record['result_sha256'] != canonical_sha256(
            {k: v for k, v in record.items() if k != 'result_sha256'}):
        raise ValueError('changed record: '+str(path))
    return record


def validate_spec(spec: dict) -> FixedSzSpatialViTConfig:
    """Validate the preregistered finite-Hamiltonian and disjoint RNG contract."""
    if spec['task'] != 'spin_spatial_vit_quality' or not spec['preregistered_before_training']:
        raise ValueError('registered full Spatial-ViT protocol required')
    family = spec.get('model_family', 'legacy')
    if family not in {'legacy', 'literature'}:
        raise ValueError('unknown model family')
    if spec.get('capacity_expansion') is not None and (
            spec['capacity_expansion'] != 'state_preserving_mlp_depth'
            or family != 'literature' or spec.get('repair_parameters_from') is None):
        raise ValueError('capacity expansion requires an explicit literature-model repair parent')
    config = (LiteratureSpatialViTConfig if family == 'literature' else FixedSzSpatialViTConfig)(**spec['model'])
    if not isinstance(spec.get('spin_inversion_even', False), bool):
        raise ValueError('spin inversion must be an explicit boolean')
    if family == 'legacy' and spec.get('spin_inversion_even', False):
        raise ValueError('legacy state cannot silently acquire spin-inversion projection')
    if config.target_m2 != 0:
        raise ValueError('this quality series uses doubled magnetization m=0')
    convention = spec.get('coordinate_convention', 'physical120')
    if convention not in {'physical120', 'legacy_checkpoint'}:
        raise ValueError('explicit physical120 or legacy_checkpoint convention required')
    if spec.get('initial_checkpoint') is not None:
        if family != 'legacy':
            raise ValueError('legacy checkpoint cannot silently become an author-aligned state')
        if convention != 'legacy_checkpoint':
            raise ValueError('archived checkpoint requires its unchanged legacy coordinate convention')
        if any(s['projection'] != 'gamma_a1' for s in spec['stages']):
            raise ValueError('continuation must preserve the parent full-A1 physical state')
    seeds = [spec['initialization_seed'], spec['training_seed'], *spec['audit']['seeds']]
    if len(set(seeds)) != len(seeds) or any(isinstance(x, bool) or not isinstance(x, int) for x in seeds):
        raise ValueError('integer initialization, training and audit seeds must be disjoint')
    stages = spec['stages']
    if (not stages or any(s['projection'] not in {'none', 'translation', 'gamma_a1'}
                         or isinstance(s['steps'], bool) or s['steps'] < 1 for s in stages)):
        raise ValueError('nonempty fixed positive projection-stage budgets required')
    allowed_sr = set(spec['sr'])
    for stage in stages:
        if set(stage.get('sr', {})) - allowed_sr:
            raise ValueError('stage SR overrides must use registered solver fields')
    terminal = sum(s['steps'] for s in stages)
    milestones = spec['quality_milestones']
    if milestones != sorted(set(milestones)) or milestones[0] != 0 or milestones[-1] != terminal:
        raise ValueError('sorted quality milestones must include initial and fixed terminal steps')
    for section, fields in [('sampling', ['chains', 'snapshots_per_update', 'proposals_between_snapshots', 'burn_proposals']),
                            ('audit', ['chains', 'snapshots', 'proposals_between_snapshots', 'burn_proposals', 'query_batch'])]:
        if any(isinstance(spec[section][k], bool) or spec[section][k] < 1 for k in fields):
            raise ValueError('positive sample and burn-in budgets required')
    if min(spec['audit']['chains'], spec['audit']['snapshots'], len(spec['audit']['seeds'])) < 2:
        raise ValueError('at least two streams, chains and snapshots required for uncertainty')
    if (not np.isfinite(spec['j2_over_j1']) or spec['maximum_training_seconds'] <= 0
            or spec['maximum_audit_seconds_per_checkpoint'] <= 0):
        raise ValueError('finite Hamiltonian and positive time caps required')
    trust = spec.get('netket', {}).get('sampled_trust')
    expected_devices = spec.get('netket', {}).get('expected_device_count')
    if expected_devices is not None and (
            isinstance(expected_devices, bool) or not isinstance(expected_devices, int)
            or expected_devices < 1):
        raise ValueError('expected_device_count must be a positive integer')
    if trust is not None:
        if (set(trust) != {'maximum_fs_sine', 'minimum_ess_fraction', 'halvings'}
                or not isinstance(trust['halvings'], int) or isinstance(trust['halvings'], bool)
                or not 1 <= trust['halvings'] <= 8
                or not np.isfinite(trust['maximum_fs_sine'])
                or not 0 < trust['maximum_fs_sine'] < 1
                or not np.isfinite(trust['minimum_ess_fraction'])
                or not 0 < trust['minimum_ess_fraction'] <= 1):
            raise ValueError('invalid preregistered sampled physical trust bound')
    if spec['momenta'].get('Gamma') != [0, 0]:
        raise ValueError('Gamma=(0,0) is required to measure total S squared')
    for vector in spec['momenta'].values():
        k = np.asarray(vector, float)
        periods = np.array([config.rows, config.columns])
        if k.shape != (2,) or not np.isfinite(k).all() or not np.allclose(k*periods, np.rint(k*periods), atol=1e-12, rtol=0):
            raise ValueError('each momentum must be allowed by the actual physical torus')
    for region in spec['regions'].values():
        sites = region['sites']
        if (not 1 <= len(sites) <= 8 or len(set(sites)) != len(sites)
                or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < config.n_sites for i in sites)):
            raise ValueError('valid fixed physical region sites required')
    return config


def model_log(spec: dict, config: FixedSzSpatialViTConfig,
              group: np.ndarray | None) -> tuple[Callable, int]:
    """Select the declared physical amplitude, never silently remap a checkpoint."""
    if spec.get('model_family', 'legacy') == 'literature':
        if spec.get('coordinate_convention', 'physical120') != 'physical120':
            raise ValueError('literature model uses physical120 coordinates directly')
        images = (np.arange(config.n_sites)[None] if group is None
                  else patch_translation_cosets(group, config))
        image_array = jnp.asarray(images)
        parity = spec.get('spin_inversion_even', False)
        def projected(p, x):
            inputs = x[image_array]
            if parity:
                inputs = jnp.concatenate((inputs, 1-inputs), axis=0)
            logs = jax.vmap(lambda y: literature_spatial_vit_log_amplitude(p, y, config))(inputs)
            maximum = jnp.max(logs.real)
            shift = jax.lax.stop_gradient(jnp.where(jnp.isfinite(maximum), maximum, 0.))
            value = shift+jnp.log(jnp.mean(jnp.exp(logs-shift)))
            return jnp.where(jnp.isfinite(maximum), value, -jnp.inf+0j)
        return projected, len(images)*(2 if parity else 1)
    if spec.get('coordinate_convention', 'physical120') == 'physical120':
        return make_physical_spatial_vit_log(config, group)
    if group is None:
        return lambda p, x: fixed_sz_spatial_vit_log_amplitude(p, x, config), 1
    return projected_log_function(config, group)


def initial_model(spec: dict) -> tuple[object, dict]:
    """Load trusted sealed full-network weights; SR has no inherited momentum.

    Historical parameters and files are immutable. A fresh sampler/RNG is
    used for the new experiment; normalization, projection and coordinates
    remain EXACTLY those of the parent. Doubled magnetization stays m=0.
    """
    config = validate_spec(spec)
    size_parent = spec.get('size_warm_start_from')
    if size_parent is not None:
        if spec.get('model_family') != 'literature' or any(spec.get(k) is not None for k in (
                'initial_checkpoint', 'repair_parameters_from', 'coupling_warm_start_from',
                'netket_resume_from', 'netket_continue_from', 'continuation_from')):
            raise ValueError('size transfer requires one literature-model parent')
        staged = spec.get('size_transfer_allow_staged_projection', False)
        if not isinstance(staged, bool):
            raise ValueError('staged size-transfer projection requires an explicit boolean')
        if ((not staged and any(s['projection'] != 'gamma_a1' for s in spec['stages']))
                or spec['stages'][-1]['projection'] != 'gamma_a1'
                or spec['netket']['initialization'] != 'balanced_spin_half'):
            raise ValueError('size transfer requires declared projection schedule and new exact-support chains')
        directory = ROOT/size_parent['directory']
        registration = read_sealed(directory/'registration.json')
        old = registration['config']
        for name in ('model_family', 'coordinate_convention', 'spin_inversion_even', 'j2_over_j1'):
            if old.get(name) != spec.get(name):
                raise ValueError('size-transfer physical convention changed: '+name)
        relative = Path('quality')/f'step{size_parent["completed_steps"]:05d}.pkl'
        checkpoint = directory/relative
        if (relative != Path(size_parent['checkpoint_path'])
                or file_sha256(checkpoint) != size_parent['checkpoint_sha256']
                or registration['result_sha256'] != size_parent['registration_sha256']):
            raise ValueError('size-transfer immutable parent changed')
        for name, digest in registration['source_manifest'].items():
            if file_sha256(directory/'source_snapshots'/name) != digest:
                raise ValueError('size-transfer frozen source changed')
        for name in ('su2_nqs/models/fixed_sz_spatial_vit.py',
                     'su2_nqs/models/literature_spatial_vit.py',
                     'projects/clebsch_tree_nqs/workflows/spin.py'):
            if file_sha256(ROOT/name) != registration['source_manifest'][name]:
                raise ValueError('size-transfer backbone implementation changed: '+name)
        saved = load_checkpoint(checkpoint)
        params = saved['params']['model']
        if (pytree_sha256(params) != size_parent['params_sha256']
                or saved['metadata']['registration_sha256'] != registration['result_sha256']
                or saved['metadata']['completed_steps'] != size_parent['completed_steps']
                or checkpoint_projection(saved['metadata']) != 'gamma_a1'):
            raise ValueError('size-transfer parent metadata/parameters changed')
        source = LiteratureSpatialViTConfig(**old['model'])
        size_rule = spec.get('size_transfer_rule', 'normalized_bilinear')
        resized = transfer_spatial_vit_size(params, source, config, rule=size_rule)
        template = init_literature_spatial_vit(jax.random.PRNGKey(spec['initialization_seed']), config)
        expected, structure = jax.tree.flatten(template)
        actual, observed = jax.tree.flatten(resized)
        if structure != observed or any(np.shape(a) != np.shape(b)
                or not np.isfinite(np.asarray(a)).all() for a, b in zip(actual, expected, strict=True)):
            raise ValueError('size-transfer target weights invalid')
        return resized, dict(**size_parent, inherited_projection=spec['stages'][0]['projection'],
            parent_projection='gamma_a1', staged_projection_is_explicit=staged,
            size_transfer_is_new_state=True, source_model=old['model'], target_model=spec['model'],
            transfer_rule=('periodic_bilinear_normalized_patch_coordinates'
                           if size_rule == 'normalized_bilinear' else 'folded_effective_spatial_kernel'),
            sampler_reinitialized=True, optimizer_and_work_counters_reset=True,
            exact_parent_wavefunction_reproduction_claim=False)
    repair = spec.get('repair_parameters_from')
    if repair is not None:
        # A DIFFERENT registered optimizer/sampler protocol on an immutable
        # same-H quality state. Not an identical-trajectory resume, and not
        # selection among independent-audit outcomes.
        if any(spec.get(k) is not None for k in ('initial_checkpoint', 'coupling_warm_start_from',
                'netket_resume_from', 'netket_continue_from', 'continuation_from')):
            raise ValueError('repair requires exactly one immutable parent')
        parent = ROOT/repair['directory']
        registration = read_sealed(parent/'registration.json')
        if registration['result_sha256'] != repair['registration_sha256']:
            raise ValueError('repair parent registration changed')
        expansion = spec.get('capacity_expansion')
        if expansion is not None and (
                expansion != 'state_preserving_mlp_depth' or spec.get('model_family') != 'literature'
                or any(s['projection'] != 'gamma_a1' for s in spec['stages'])):
            raise ValueError('capacity expansion requires explicit same-H full-A1 literature ViT')
        for key in ('model', 'model_family', 'coordinate_convention', 'spin_inversion_even', 'j2_over_j1'):
            if key == 'model' and expansion is not None:
                continue  # The inclusion below validates every unchanged architecture field.
            if registration['config'].get(key) != spec.get(key):
                raise ValueError('same-H repair changed physical model: '+key)
        relative = Path(repair['checkpoint_path'])
        if relative != Path('quality')/f'step{repair["completed_steps"]:05d}.pkl':
            raise ValueError('repair requires an immutable quality checkpoint')
        checkpoint = parent/relative
        if file_sha256(checkpoint) != repair['checkpoint_sha256']:
            raise ValueError('repair parent checkpoint changed')
        for name, digest in registration['source_manifest'].items():
            if file_sha256(parent/'source_snapshots'/name) != digest:
                raise ValueError('repair parent frozen source changed')
        for name in ('su2_nqs/models/fixed_sz_spatial_vit.py', 'su2_nqs/models/literature_spatial_vit.py',
                     'su2_nqs/benchmarks/spatial_vit_sampling.py',
                     'projects/clebsch_tree_nqs/workflows/spin.py'):
            if file_sha256(ROOT/name) != registration['source_manifest'][name]:
                raise ValueError('repair backbone implementation changed: '+name)
        saved = load_checkpoint(checkpoint)
        params = saved['params']['model']
        if (pytree_sha256(params) != repair['params_sha256']
                or saved['metadata']['registration_sha256'] != registration['result_sha256']
                or saved['metadata']['completed_steps'] != repair['completed_steps']):
            raise ValueError('repair parent metadata/parameters changed')
        expansion_info = {}
        if expansion is not None:
            source_config = LiteratureSpatialViTConfig(**registration['config']['model'])
            old_params = params
            params = expand_spatial_vit_capacity(params, source_config, config,
                jax.random.PRNGKey(spec['initialization_seed']))
            rng = np.random.default_rng(spec['initialization_seed'])
            probe = jnp.asarray(np.stack([rng.permutation(
                [0]*(config.n_sites//2)+[1]*(config.n_sites//2)) for _ in range(128)]))
            old_logs = jax.jit(jax.vmap(lambda x: literature_spatial_vit_log_amplitude(
                old_params, x, source_config)))(probe)
            new_logs = jax.jit(jax.vmap(lambda x: literature_spatial_vit_log_amplitude(
                params, x, config)))(probe)
            error = float(jnp.max(jnp.abs(jnp.exp(new_logs-old_logs)-1)))
            if not np.isfinite(error) or error > 2e-11:
                raise ValueError('capacity expansion changed physical amplitudes')
            expansion_info = dict(capacity_expanded=True,
                source_model=registration['config']['model'], target_model=spec['model'],
                initial_raw_amplitude_ratio_max_error=error,
                parent_parameter_count=fixed_sz_spatial_vit_parameter_count(old_params),
                expanded_parameter_count=fixed_sz_spatial_vit_parameter_count(params),
                old_parameters_preserved=True, new_coordinates_initialized=True)
        template = init_literature_spatial_vit(jax.random.PRNGKey(spec['initialization_seed']), config)
        leaves, structure = jax.tree_util.tree_flatten(params)
        expected, template_structure = jax.tree_util.tree_flatten(template)
        if (structure != template_structure or any(np.shape(a) != np.shape(b)
                or np.asarray(a).dtype != np.asarray(b).dtype or not np.isfinite(np.asarray(a)).all()
                for a, b in zip(leaves, expected))):
            raise ValueError('repair parent weights do not match the complete network')
        return params, dict(**repair, reinitialized_parameters=False,
            inherited_projection=checkpoint_projection(saved['metadata']),
            sampler_reinitialized=True, optimizer_and_work_counters_reset=True,
            repair_is_new_experiment=True, **expansion_info)
    transfer = spec.get('coupling_warm_start_from')
    if transfer is not None:
        if any(spec.get(key) is not None for key in
               ('initial_checkpoint', 'netket_resume_from', 'netket_continue_from', 'continuation_from')):
            raise ValueError('coupling transfer is not optimizer continuation')
        directory = ROOT/transfer['directory']
        registration = read_sealed(directory/'registration.json')
        trained = read_sealed(directory/'training.json')
        old = registration['config']
        for key in ('model', 'model_family', 'coordinate_convention', 'spin_inversion_even'):
            if old.get(key) != spec.get(key):
                raise ValueError('coupling transfer changed physical backbone: '+key)
        if (spec.get('model_family') != 'literature'
                or any(s['projection'] != 'gamma_a1' for s in spec['stages'])
                or old['j2_over_j1'] != transfer['parent_j2_over_j1']
                or old['j2_over_j1'] == spec['j2_over_j1']
                or registration['result_sha256'] != transfer['registration_sha256']
                or trained['registration_sha256'] != registration['result_sha256']
                or trained['completed_steps'] != transfer['completed_steps']
                or trained['status'] != 'fixed_steps_complete'):
            raise ValueError('coupling-transfer parent/projection identity changed')
        relative = Path(transfer['checkpoint_path'])
        if relative != Path('quality')/f'step{transfer["completed_steps"]:05d}.pkl':
            raise ValueError('coupling transfer requires a fixed quality checkpoint')
        for name, digest in registration['source_manifest'].items():
            if file_sha256(directory/'source_snapshots'/name) != digest:
                raise ValueError('coupling-transfer frozen source changed')
        for name in ('su2_nqs/models/fixed_sz_spatial_vit.py',
                     'su2_nqs/models/literature_spatial_vit.py',
                     'su2_nqs/benchmarks/spatial_vit_sampling.py',
                     'projects/clebsch_tree_nqs/workflows/spin.py'):
            if file_sha256(ROOT/name) != registration['source_manifest'][name]:
                raise ValueError('coupling-transfer backbone implementation changed: '+name)
        checkpoint = directory/relative
        if file_sha256(checkpoint) != transfer['checkpoint_sha256']:
            raise ValueError('coupling-transfer checkpoint changed')
        saved = load_checkpoint(checkpoint)
        params = saved['params']['model']
        digest = pytree_sha256(params)
        if (digest != trained['final_params_sha256'] or digest != transfer['params_sha256']
                or saved['metadata']['registration_sha256'] != registration['result_sha256']
                or saved['metadata']['completed_steps'] != transfer['completed_steps']
                or checkpoint_projection(saved['metadata']) != 'gamma_a1'):
            raise ValueError('coupling-transfer parameter/metadata mismatch')
        template = init_literature_spatial_vit(jax.random.PRNGKey(spec['initialization_seed']), config)
        leaves, structure = jax.tree_util.tree_flatten(params)
        expected, template_structure = jax.tree_util.tree_flatten(template)
        if (structure != template_structure or
                any(np.shape(a) != np.shape(b) or np.asarray(a).dtype != np.asarray(b).dtype
                    or not np.isfinite(np.asarray(a)).all() for a, b in zip(leaves, expected))):
            raise ValueError('coupling-transfer weights do not match the full model')
        return params, dict(**transfer, inherited_projection='gamma_a1',
            reinitialized_parameters=False, sampler_reinitialized=True,
            inherited_optimizer_momentum=False, optimizer_and_work_counters_reset=True,
            new_hamiltonian_j2_over_j1=spec['j2_over_j1'])
    parent = spec.get('initial_checkpoint')
    if parent is None:
        initializer = (init_literature_spatial_vit if spec.get('model_family', 'legacy') == 'literature'
                       else init_fixed_sz_spatial_vit)
        return initializer(jax.random.PRNGKey(spec['initialization_seed']), config), {}
    if not jax.config.x64_enabled:
        raise RuntimeError('enable x64 BEFORE unpickling archived JAX arrays')
    paths = {key: ROOT/parent[key] for key in ('checkpoint', 'result', 'training_result')}
    if any(not p.resolve().is_relative_to(ROOT) for p in paths.values()):
        raise ValueError('trusted parent paths must remain in the repository')
    if file_sha256(paths['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('parent checkpoint changed')
    result, trained = read_sealed(paths['result']), read_sealed(paths['training_result'])
    if (result['result_sha256'] != parent['result_sha256']
            or trained['result_sha256'] != parent['training_result_sha256']
            or result['model_config'] != asdict(config)
            or result['j2_over_j1'] != spec['j2_over_j1']
            or trained['checkpoint_sha256'] != parent['checkpoint_sha256']):
        raise ValueError('parent family/Hamiltonian/configuration identity changed')
    saved = load_checkpoint(paths['checkpoint'])
    params = saved['params']['model']
    digest = pytree_sha256(params)
    if (digest != parent['params_sha256'] or digest != trained['final_params_sha256']
            or int(saved['optimizer_state']['step']) != trained['completed_steps']
            or fixed_sz_spatial_vit_parameter_count(params) != result['parameter_count']):
        raise ValueError('parent parameter/optimizer replay mismatch')
    if any(np.asarray(v).dtype != np.dtype('float64') or not np.isfinite(v).all()
           for v in jax.tree_util.tree_leaves(params)):
        raise ValueError('parent must have finite float64 weights')
    return params, dict(**parent, inherited_projection='gamma_a1',
        reinitialized_parameters=False, sampler_reinitialized=True,
        inherited_optimizer_momentum=False, parent_local_SR_steps=trained['completed_steps'])


def checkpoint_projection(metadata: dict) -> str:
    """A loaded A1 parent is already projected at quality step zero."""
    return (metadata['history'][-1]['projection'] if metadata['history'] else
            metadata.get('parent', {}).get('inherited_projection', 'none'))


def recovery_state(spec: dict, registration: dict) -> tuple[dict, dict]:
    """Resume a sealed interrupted trajectory without resetting its budget/RNG.

    A separately sealed numerical solver revision may change ONLY the ridge
    linear solver, not physical amplitudes, samples, stages, trust thresholds,
    seeds, audits or total budgets. Original sources and records stay immutable.
    No independent audit value is used in this decision.
    """
    recovery = spec['continuation_from']
    if (not np.isfinite(recovery['charged_overhead_seconds'])
            or recovery['charged_overhead_seconds'] < 0
            or not isinstance(recovery['discarded_and_replay_full_H_configs'], int)
            or recovery['discarded_and_replay_full_H_configs'] < 0):
        raise ValueError('nonnegative recovery overhead must be charged to the original budget')
    prior_output = (ROOT/recovery['output']).resolve()
    if not prior_output.is_relative_to(ROOT):
        raise ValueError('recovery paths must remain in the repository')
    prior_registration = read_sealed(prior_output/'registration.json')
    progress = read_sealed(prior_output/'progress.json')
    if (prior_registration['result_sha256'] != recovery['registration_sha256']
            or progress['result_sha256'] != recovery['progress_sha256']
            or file_sha256(prior_output/'checkpoint.pkl') != recovery['checkpoint_sha256']):
        raise ValueError('interrupted trajectory identity changed')
    old, new = dict(prior_registration['config']), dict(spec)
    old.pop('output'); new.pop('output'); new.pop('continuation_from')
    new['sr'] = dict(new['sr'])
    new['sr'].pop('linear_solver')
    if old != new or spec['sr']['linear_solver'] != 'svd':
        raise ValueError('numerical recovery must preserve the physical training protocol')
    saved = load_checkpoint(prior_output/'checkpoint.pkl')
    state, meta = saved['params'], dict(saved['metadata'])
    if (meta['registration_sha256'] != prior_registration['result_sha256']
            or meta['completed_steps'] != progress['completed_steps']
            or meta['history'] != progress['history']
            or int(saved['optimizer_state']['step']) != meta['completed_steps']
            or pytree_sha256(state['model']) != progress['final_params_sha256']):
        raise ValueError('interrupted state/optimizer replay mismatch')
    meta['registration_sha256'] = registration['result_sha256']
    meta['numerical_recovery'] = recovery
    # Charge discarded/replayed work conservatively inside the ORIGINAL cap.
    meta['active_training_seconds'] += recovery['charged_overhead_seconds']
    meta['additional_full_H_calls'] = recovery['discarded_and_replay_full_H_configs']
    meta['status'] = 'running'
    return state, meta


def stage_at(spec: dict, completed_steps: int) -> tuple[int, str]:
    """The NEXT update's predeclared projector; transitions do not depend on energy."""
    boundary = 0
    for index, stage in enumerate(spec['stages']):
        boundary += stage['steps']
        if completed_steps < boundary:
            return index, stage['projection']
    return len(spec['stages'])-1, spec['stages'][-1]['projection']


def make_collector(log_fn, *, snapshots: int, proposals: int):
    """Return persistent-chain snapshots with exact fixed-Sz exchange balance."""
    sample = make_log_exchange_sampler(log_fn, proposals)
    @jax.jit
    def collect(params, key, states):
        def step(carry, _):
            x, rng = carry
            changed, rng, accepted, moved = sample(params, rng, x)
            return (changed, rng), (changed, accepted, moved)
        (states, key), (records, accepts, moves) = jax.lax.scan(
            step, (states, key), None, length=snapshots)
        return states, key, records, accepts, moves
    return collect


def register(config_path: Path, output: Path, *,
             additional_entry_paths: tuple[Path, ...] = ()) -> tuple[dict, dict]:
    """Pin protocol and numerical source snapshots before any parameter training."""
    spec = json.loads(config_path.read_text())
    validate_spec(spec)
    if spec.get('backend') == 'netket_ntk':
        additional_entry_paths = (*additional_entry_paths, Path(__file__).with_name('spin_spatial_vit_netket.py'))
    if (not any(spec.get(k) is not None for k in ('netket_resume_from', 'netket_continue_from'))
            and any(spec.get(k) is not None for k in
                    ('initial_checkpoint', 'coupling_warm_start_from', 'repair_parameters_from',
                     'size_warm_start_from'))):
        initial_model(spec)  # Verify the immutable parent before registering or resuming.
    # A complete NetKet continuation/recovery validates its immediate parent
    # below. Retained initial/warm-start declarations describe ancestry, not
    # a request to initialize again from that older state's weights/RNG.
    if spec.get('continuation_from') is not None:
        recovery_state(spec, {'result_sha256': 'pre-registration-validation'})
    if spec.get('netket_resume_from') is not None:
        from .spin_spatial_vit_netket import recovery_state as netket_recovery_state
        netket_recovery_state(spec, 'pre-registration-validation')
    if spec.get('netket_continue_from') is not None:
        if spec.get('netket_resume_from') is not None:
            raise ValueError('declare one parent: continuation or runtime recovery, not both')
        from .spin_spatial_vit_netket import continuation_state as netket_continuation_state
        netket_continuation_state(spec, 'pre-registration-validation')
    sources = transitive_local_source_manifest((Path(__file__), *additional_entry_paths), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    path = output/'registration.json'
    if path.exists():
        registration = read_sealed(path)
        if registration['config'] != spec or registration['source_manifest'] != sources:
            raise ValueError('protocol/numerical source changed; start a separately registered run')
        return spec, registration
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refuse to register over unrelated files')
    output.mkdir(parents=True, exist_ok=True)
    for relative in sources:
        destination = output/'source_snapshots'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
    registration = sealed_json(path, dict(config=spec,
        config_sha256=canonical_sha256(spec), source_manifest=sources,
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        state_family='full configuration-dependent factored Spatial-ViT, no carrier/residual',
        symmetry='exact fixed doubled magnetization m=0; SU(2) NOT imposed',
        spatial_projection='fixed staged schedule; all frozen audits use complete Gamma-A1',
        coordinate_map=('physical120 directly in literature model' if spec.get('model_family') == 'literature' else
                        'archived legacy input unchanged; NO column reflection or normalization change'
                        if spec.get('coordinate_convention') == 'legacy_checkpoint' else
                        'physical triangular120 metric -> model60 metric by column reflection'),
        hamiltonian_diagonalization_performed=False, remote_submitted=False))
    return spec, registration


def _snapshot(output: Path, state: dict, metadata: dict) -> None:
    step = metadata['completed_steps']
    path = output/'quality'/f'step{step:05d}.pkl'
    if not path.exists():
        save_checkpoint(path, state, metadata=metadata)


def train(config_path: Path, output: Path, *, resume: bool = False,
          until_step: int | None = None, session_seconds: float = 1800.) -> dict:
    """Fixed-budget all-parameter SR; checkpoints include sampler RNG/history.

    A session cap pauses an unfinished run; it never labels a partial trajectory
    converged. Resuming keeps the original TOTAL update/time budgets. The SR
    solver has no momentum state; its count and damping protocol are saved.
    """
    spec, registration = register(config_path, output)
    config = validate_spec(spec)
    maximum = sum(s['steps'] for s in spec['stages'])
    limit = maximum if until_step is None else until_step
    if (not isinstance(limit, int) or not 0 <= limit <= maximum
            or not np.isfinite(session_seconds) or session_seconds <= 0):
        raise ValueError('session budget must lie inside registered total budget')
    checkpoint = output/'checkpoint.pkl'
    if (output/'training.json').exists():
        return read_sealed(output/'training.json')
    sampler_spec = spec['sampling']
    if checkpoint.exists():
        if not resume:
            raise FileExistsError('use --resume for an existing training checkpoint')
        saved = load_checkpoint(checkpoint)
        state, meta = saved['params'], saved['metadata']
        if (meta['registration_sha256'] != registration['result_sha256']
                or meta['completed_steps'] != len(meta['history'])
                or int(saved['optimizer_state']['step']) != meta['completed_steps']):
            raise ValueError('checkpoint identity/count mismatch')
    elif spec.get('continuation_from') is not None:
        state, meta = recovery_state(spec, registration)
    else:
        params, parent_record = initial_model(spec)
        init, rng = jax.random.split(jax.random.PRNGKey(spec['training_seed']))
        states = initialize_spin_exchange_chains(init, n_chains=sampler_spec['chains'],
            n_sites=config.n_sites, n_up=config.target_n_up)
        state = dict(model=params, chains=states, key=rng)
        meta = dict(registration_sha256=registration['result_sha256'], completed_steps=0,
            history=[], active_training_seconds=0., compile_seconds=0., compiled_stages=[],
            current_stage=-1, warmup_full_H_calls=0, training_full_H_calls=0,
            initial_params_sha256=pytree_sha256(params), status='running',
            model_config=asdict(config), parent=parent_record)
        _snapshot(output, state, meta)
    session_start = perf_counter()
    groups = _groups(config.geometry, config.rows, config.columns)
    bonds = _bonds(config.geometry, config.rows, config.columns, spec['j2_over_j1'])
    kernels = {}

    def save():
        save_checkpoint(checkpoint, state, optimizer_state=dict(step=meta['completed_steps']), metadata=meta)
        sealed_json(output/'progress.json', dict(**meta, parameter_count=fixed_sz_spatial_vit_parameter_count(state['model']),
            final_params_sha256=pytree_sha256(state['model']), runtime=numerical_runtime_identity()))

    save()
    while meta['completed_steps'] < limit:
        if (perf_counter()-session_start >= session_seconds
                or meta['active_training_seconds'] >= spec['maximum_training_seconds']):
            break
        index, projection = stage_at(spec, meta['completed_steps'])
        if index not in kernels:
            group = None if projection == 'none' else groups[projection]
            log_fn, images = model_log(spec, config, group)
            collector = make_collector(log_fn, snapshots=sampler_spec['snapshots_per_update'],
                proposals=sampler_spec['proposals_between_snapshots'])
            local = make_full_heisenberg_ratios(log_fn, bonds)
            solver = dict(spec['sr'], **spec['stages'][index].get('sr', {}))
            update = make_dual_sampled_sr_update(log_fn, state['model'], **solver)
            thermal = make_log_exchange_sampler(log_fn, sampler_spec['burn_proposals'])
            tick = perf_counter()
            _, _, warm_records, _, _ = collector(state['model'], state['key'], state['chains'])
            warm_x = warm_records.reshape((-1, config.n_sites))
            warm_e = local(state['model'], warm_x)
            jax.block_until_ready(update(state['model'], warm_x, warm_e))
            jax.block_until_ready(thermal(state['model'], state['key'], state['chains']))
            elapsed = perf_counter()-tick
            meta['compile_seconds'] += elapsed
            meta['warmup_full_H_calls'] += len(warm_x)
            meta['compiled_stages'].append(dict(stage=index, projection=projection, images=images,
                compile_and_warm_seconds=elapsed, records_per_update=len(warm_x), sr=solver))
            kernels[index] = (collector, local, update, thermal, images)
            print('compiled', projection, 'images', images, 'seconds', round(elapsed, 3), flush=True)
            save()
            if perf_counter()-session_start >= session_seconds:
                break
        collector, local, update, thermal, images = kernels[index]
        tick = perf_counter()
        if meta['current_stage'] != index:
            x, rng, _, _ = thermal(state['model'], state['key'], state['chains'])
            state = dict(model=state['model'], chains=x, key=rng)
            meta['current_stage'] = index
        x, rng, records, accepts, moves = collector(state['model'], state['key'], state['chains'])
        samples = records.reshape((-1, config.n_sites))
        energies = local(state['model'], samples)
        proposed, diagnostics = update(state['model'], samples, energies)
        jax.block_until_ready((proposed, diagnostics, energies))
        def failed_attempt(reason: str):
            # Never leave a stopped process labelled running or hide spent H work.
            meta['status'] = reason
            meta['active_training_seconds'] += perf_counter()-tick
            meta['additional_full_H_calls'] = meta.get('additional_full_H_calls', 0)+len(samples)
            meta.setdefault('failed_attempts', []).append(dict(
                next_step=meta['completed_steps']+1, reason=reason,
                relative_linear_residual=float(diagnostics['relative_linear_residual']),
                physical_update_installed=False))
            save()
        if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves((proposed, diagnostics, energies))):
            failed_attempt('nonfinite_proposal_last_valid_checkpoint_preserved')
            raise FloatingPointError(meta['status'])
        if float(diagnostics['relative_linear_residual']) > 1e-7:
            failed_attempt('solver_guard_last_valid_checkpoint_preserved')
            raise FloatingPointError('SR solve residual failed; no proposal installed')
        state = dict(model=proposed, chains=x, key=rng)
        mean = np.asarray(energies).mean()
        elapsed = perf_counter()-tick
        meta['completed_steps'] += 1
        meta['active_training_seconds'] += elapsed
        meta['training_full_H_calls'] += len(samples)
        row = dict(step=meta['completed_steps'], projection=projection, images=images,
            stage=index, damping=float(spec['stages'][index].get('sr', {}).get('damping', spec['sr']['damping'])),
            energy_before_update_per_site=float(mean.real/config.n_sites),
            variance_before_update_per_site=float(np.mean(abs(np.asarray(energies)-mean)**2)/config.n_sites),
            nontrivial_acceptance=float(np.mean(moves)), seconds=elapsed,
            gradient_norm=float(diagnostics['gradient_norm']),
            sampled_fs_sine=float(diagnostics['sampled_fs_sine']),
            reweighting_ess_fraction=float(diagnostics['reweighting_ess_fraction']),
            relative_linear_residual=float(diagnostics['relative_linear_residual']),
            accepted=bool(diagnostics['accepted']))
        meta['history'].append(row)
        save()
        if meta['completed_steps'] in spec['quality_milestones']:
            _snapshot(output, state, meta)
        if row['step'] == 1 or row['step'] % 16 == 0:
            print(row, flush=True)
    finished = meta['completed_steps'] == maximum
    meta['status'] = ('fixed_steps_complete' if finished else
        'total_time_cap' if meta['active_training_seconds'] >= spec['maximum_training_seconds'] else 'paused_session_budget')
    save()
    if finished or meta['status'] == 'total_time_cap':
        _snapshot(output, state, meta)
        return sealed_json(output/'training.json', dict(**meta,
            final_params_sha256=pytree_sha256(state['model']),
            parameter_count=fixed_sz_spatial_vit_parameter_count(state['model']),
            all_parameters_eligible_for_training=True, carrier=None,
            strict_su2=False, converged=False, variational_win_claimed=False,
            runtime=numerical_runtime_identity()))
    return read_sealed(output/'progress.json')


def _batched_logs(log_batch, params, bits, batch: int) -> np.ndarray:
    values = []
    for offset in range(0, len(bits), batch):
        x = bits[offset:offset+batch]
        padded = np.pad(x, ((0, batch-len(x)), (0, 0)), mode='edge')
        values.append(np.asarray(log_batch(params, jnp.asarray(padded)))[:len(x)])
    return np.concatenate(values)


def _energy_stream(spec: dict, config, params: dict, destination: Path, step: int,
                   original_seed: int, local, thermal, collector, bounded,
                   registration_sha256: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """One immutable Born stream; reusable by energy-only and physical audits."""
    a = spec['audit']; seed = original_seed+1009*step
    path = destination/f'stream{seed}.npz'
    manifest = path.with_suffix('.json')
    identity = pytree_sha256(params)
    if path.exists():
        stream = read_sealed(manifest)
        if (stream['raw_sha256'] != file_sha256(path) or stream['params_sha256'] != identity
                or stream['seed'] != seed
                or stream.get('registration_sha256', registration_sha256) != registration_sha256):
            raise ValueError('audit stream identity changed')
        with np.load(path) as z:
            x, e = z['states'], z['energies']
    else:
        bounded()
        init, rng = jax.random.split(jax.random.PRNGKey(seed))
        states = initialize_spin_exchange_chains(init, n_chains=a['chains'],
            n_sites=config.n_sites, n_up=config.target_n_up)
        states, rng, _, _ = thermal(params, rng, states)
        _, _, x, _, moves = collector(params, rng, states)
        x = np.asarray(x)
        bounded()
        e = _batched_logs(local, params, x.reshape((-1, config.n_sites)),
                          a['query_batch']).reshape(x.shape[:2])
        if not np.isfinite(e).all():
            raise FloatingPointError('nonfinite fresh audit; no endpoint selection')
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, states=x, energies=e, moves=np.asarray(moves))
        temporary.replace(path)
        stream = sealed_json(manifest, dict(seed=seed, params_sha256=identity,
            registration_sha256=registration_sha256,
            raw_sha256=file_sha256(path), records=int(e.size), full_H_calls=int(e.size)))
    if x.shape != (a['snapshots'], a['chains'], config.n_sites) or e.shape != x.shape[:2]:
        raise ValueError('cached audit shape changed')
    if not np.isfinite(e).all() or stream['full_H_calls'] != e.size:
        raise ValueError('invalid cached full-H audit')
    return x, e, stream


def audit_energy(config_path: Path, output: Path, *, stream_index: int | None = None,
                 aggregate_only: bool = False, step: int | None = None) -> dict:
    """Frozen terminal energy/variance audit; streams may run on separate GPUs.

    Uses the original preregistered seeds/burn/work. It does not resample or
    optimize, and its cached configurations remain usable for later RDM work.
    The CPU-only aggregator refuses incomplete streams or a partial endpoint.
    """
    spec, registration = register(config_path, output)
    config = validate_spec(spec); a = spec['audit']
    terminal = sum(stage['steps'] for stage in spec['stages'])
    if step is None or step == terminal:
        step = terminal
        training = read_sealed(output/'training.json')
        if training['completed_steps'] != step or training['status'] != 'fixed_steps_complete':
            raise ValueError('energy audit requires the fixed terminal endpoint')
    elif step not in spec['quality_milestones']:
        raise ValueError('energy audit requires a preregistered frozen milestone')
    else:
        training = read_sealed(output/'progress.json')
        if training['completed_steps'] < step:
            raise ValueError('energy audit milestone has not been reached')
    checkpoint = output/'quality'/f'step{step:05d}.pkl'
    saved = load_checkpoint(checkpoint); params = saved['params']['model']
    if (saved['metadata']['registration_sha256'] != registration['result_sha256']
            or saved['metadata']['completed_steps'] != step
            or (step == terminal and pytree_sha256(params) != training['final_params_sha256'])):
        raise ValueError('frozen terminal checkpoint changed')
    if aggregate_only and stream_index is not None:
        raise ValueError('aggregation and stream selection are mutually exclusive')
    if stream_index is not None and not 0 <= stream_index < len(a['seeds']):
        raise ValueError('invalid independent stream index')
    destination = output/'audits'/f'step{step:05d}'
    destination.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    def bounded():
        if aggregate_only:
            raise FileNotFoundError('aggregation never generates missing audit samples')
        if perf_counter()-started > spec['maximum_audit_seconds_per_checkpoint']:
            raise TimeoutError('audit time cap; completed streams retained')
    group = _groups(config.geometry, config.rows, config.columns)['gamma_a1']
    log_fn, images = model_log(spec, config, group)
    local = make_full_heisenberg_ratios(log_fn, _bonds(config.geometry, config.rows, config.columns, spec['j2_over_j1']))
    thermal = make_log_exchange_sampler(log_fn, a['burn_proposals'])
    collector = make_collector(log_fn, snapshots=a['snapshots'], proposals=a['proposals_between_snapshots'])
    indices = range(len(a['seeds'])) if stream_index is None else [stream_index]
    energies, reports = [], []
    for index in indices:
        _, e, report = _energy_stream(spec, config, params, destination, step,
            a['seeds'][index], local, thermal, collector, bounded, registration['result_sha256'])
        energies.append(e); reports.append(report)
    if stream_index is not None:
        return reports[0]  # Worker writes only its own immutable stream.
    energy = audit_statistics(np.asarray(energies), config.n_sites,
        bootstrap_seed=a['bootstrap_seed']+step, replicates=a['bootstrap_replicates'])
    result = sealed_json(destination/'energy.json', dict(step=step,
        registration_sha256=registration['result_sha256'], checkpoint_sha256=file_sha256(checkpoint),
        params_sha256=pytree_sha256(params), audit_projection='gamma_a1', image_count=images,
        strict_su2=False, energy=energy, streams=reports, full_H_calls=int(np.asarray(energies).size),
        frozen_before_audit=True, audit_never_used_for_training=True,
        fixed_terminal_endpoint=bool(step == terminal),
        seconds=perf_counter()-started, physical_readouts_pending=True,
        competitive_accuracy_certified=False, physical_phase_claim=False))
    print('untouched terminal energy/variance', json.dumps(energy), flush=True)
    return result


def audit(config_path: Path, output: Path, *, steps: list[int] | None = None) -> dict:
    """Fresh full-A1 Born streams and coherent physical readouts of saved milestones.

    Raw-stage checkpoints are explicitly labelled projection AFTER training.
    Scalar K/M and S^2 include off-diagonal ratios; 3*SzSz is not used. Saved
    raw RDMs and whole-chain replicates are preserved without SU2 repair.
    """
    spec, registration = register(config_path, output)
    config = validate_spec(spec)
    folders = sorted((output/'quality').glob('step*.pkl'))
    selected = [int(p.stem[4:]) for p in folders] if steps is None else steps
    total_started = perf_counter()
    a = spec['audit']
    group = _groups(config.geometry, config.rows, config.columns)['gamma_a1']
    log_fn, images = model_log(spec, config, group)
    log_batch = jax.jit(jax.vmap(log_fn, in_axes=(None, 0)))
    local = make_full_heisenberg_ratios(log_fn, _bonds(config.geometry, config.rows, config.columns, spec['j2_over_j1']))
    thermal = make_log_exchange_sampler(log_fn, a['burn_proposals'])
    collector = make_collector(log_fn, snapshots=a['snapshots'], proposals=a['proposals_between_snapshots'])
    coordinates = np.array([divmod(i, config.columns) for i in range(config.n_sites)])
    momenta = spec['momenta']
    structure = make_scalar_structure_factors(log_fn, coordinates, momenta)
    summaries = {}
    for step in selected:
        checkpoint = output/'quality'/f'step{step:05d}.pkl'
        saved = load_checkpoint(checkpoint)
        if saved['metadata']['registration_sha256'] != registration['result_sha256']:
            raise ValueError('quality checkpoint protocol changed')
        params = saved['params']['model']
        destination = output/'audits'/f'step{step:05d}'
        destination.mkdir(parents=True, exist_ok=True)
        if (destination/'results.json').exists():
            result = read_sealed(destination/'results.json')
            if result['checkpoint_sha256'] != file_sha256(checkpoint):
                raise ValueError('audited checkpoint changed')
            summaries[str(step)] = result
            continue
        started = perf_counter()
        def bounded():
            if perf_counter()-started > spec['maximum_audit_seconds_per_checkpoint']:
                raise TimeoutError('audit cap; completed streams/readouts retained, no new tuning')
            if perf_counter()-total_started > spec.get('maximum_total_audit_seconds', np.inf):
                raise TimeoutError('total audit cap; completed outputs retained, no new tuning')
        records, energies, stream_reports = [], [], []
        for original_seed in a['seeds']:
            x, e, stream = _energy_stream(spec, config, params, destination, step,
                original_seed, local, thermal, collector, bounded, registration['result_sha256'])
            records.append(x); energies.append(e); stream_reports.append(stream)
        values = np.asarray(energies)
        bits = np.asarray(records).reshape((-1, config.n_sites))
        chain_ids = np.concatenate([np.tile(np.arange(a['chains'])+s*a['chains'], a['snapshots'])
                                    for s in range(len(a['seeds']))])
        energy = audit_statistics(values, config.n_sites, bootstrap_seed=a['bootstrap_seed']+step,
            replicates=a['bootstrap_replicates'])
        structure_path = destination/'structure.npz'
        if structure_path.exists():
            if read_sealed(structure_path.with_suffix('.json'))['raw_sha256'] != file_sha256(structure_path):
                raise ValueError('cached structure-factor records changed')
            with np.load(structure_path) as z:
                scalar, zz = z['scalar'], z['zz']
        else:
            bounded()
            scalar_parts, zz_parts = [], []
            for offset in range(0, len(bits), a['query_batch']):
                bounded()
                x = bits[offset:offset+a['query_batch']]
                pad = np.pad(x, ((0, a['query_batch']-len(x)), (0, 0)), mode='edge')
                sf, diagonal = structure(params, jnp.asarray(pad))
                scalar_parts.append(np.asarray(sf)[:len(x)]); zz_parts.append(np.asarray(diagonal)[:len(x)])
            scalar, zz = np.concatenate(scalar_parts), np.concatenate(zz_parts)
            np.savez_compressed(structure_path, scalar=scalar, zz=zz, chain_ids=chain_ids)
            sealed_json(structure_path.with_suffix('.json'), dict(raw_sha256=file_sha256(structure_path),
                params_sha256=pytree_sha256(params), records=len(bits)))
        def chain_summary(v):
            means = np.array([v[chain_ids == c].mean() for c in np.unique(chain_ids)])
            return dict(value=float(v.mean().real), imaginary=float(v.mean().imag),
                chain_SE=float(means.real.std(ddof=1)/np.sqrt(len(means))))
        sf_report = {name: dict(scalar=chain_summary(scalar[:, i]), zz=chain_summary(zz[:, i]))
                     for i, name in enumerate(momenta)}
        region_reports = {}
        for name, cut in spec['regions'].items():
            path = destination/(name+'.json')
            if path.exists():
                region_reports[name] = read_sealed(path)
                if region_reports[name]['raw_sha256'] != file_sha256(path.with_suffix('.npz')):
                    raise ValueError('cached coherent RDM changed')
                continue
            bounded()
            plan = regional_query_plan(bits, cut['sites'])
            logs = _batched_logs(log_batch, params, plan.configurations, a['query_batch'])
            matrices = born_conditional_regional_matrices(plan, logs)
            summary = regional_density_summary(matrices, chain_ids)
            rho = summary['rho']
            block = physical_spin_block_readout(rho)
            operators = {f'p_q{q}': p for q, p in _spin_projectors(len(cut['sites']))}
            if len(cut['sites']) == 6:
                split = joint_spin_operator_split(6, cut['left'], (3, 3, 0))
                operators.update(joint_q3_full=split.full, joint_q3_low=split.low, joint_q3_high=split.high)
                if name == 'compact6':
                    operators.update(compact6_channel_operators())
            observables = {label: regional_observable_summary(summary, op) for label, op in operators.items()}
            raw = destination/(name+'.npz')
            np.savez_compressed(raw, rho=rho, jackknife_rhos=summary['jackknife_rhos'],
                per_record_matrices=matrices, sites=cut['sites'], chain_ids=chain_ids)
            region_reports[name] = sealed_json(path, dict(sites=cut['sites'],
                raw_sha256=file_sha256(raw), diagnostics=block,
                joint_spin=spin_partition_readout(rho, cut['left']), observables=observables,
                unique_regional_amplitude_queries=len(plan.configurations),
                strict_multiplicity_spectrum_reported=False,
                warning='Raw coherent RDM; fixed-Sz ViT is not strict SU2. No twirl or eigenvalue repair.'))
        replay = load_checkpoint(checkpoint)['params']['model']
        probe = jnp.asarray(bits[:a['query_batch']])
        replay_error = float(np.max(abs(np.asarray(log_batch(params, probe))-np.asarray(log_batch(replay, probe)))))
        reference = np.asarray(log_batch(params, probe))
        symmetry_error = 0.
        for permutation in group[::max(1, len(group)//12)]:
            transformed = np.asarray(log_batch(params, probe[:, jnp.asarray(permutation)]))
            symmetry_error = max(symmetry_error, float(np.max(abs(np.exp(transformed-reference)-1))))
        result = sealed_json(destination/'results.json', dict(step=step,
            registration_sha256=registration['result_sha256'], checkpoint_sha256=file_sha256(checkpoint),
            params_sha256=pytree_sha256(params), training_projection=checkpoint_projection(saved['metadata']),
            audit_projection='gamma_a1',
            raw_stage_projection_after_training=checkpoint_projection(saved['metadata']) != 'gamma_a1',
            image_count=images, strict_su2=False, energy=energy, structure_factors=sf_report,
            total_spin_squared=chain_summary(config.n_sites*scalar[:, list(momenta).index('Gamma')]),
            regions=region_reports, streams=stream_reports, seconds=perf_counter()-started,
            checkpoint_log_replay_error=replay_error, sampled_A1_relative_error=symmetry_error,
            full_H_calls=int(values.size), full_scalar_pair_operator_calls=int(len(bits)*config.n_sites*(config.n_sites-1)//2),
            additional_observable_work_counted_separately=True,
            competitive_accuracy_certified=False, physical_phase_claim=False))
        summaries[str(step)] = result
        print('audit', step, energy, flush=True)
    return sealed_json(output/'quality_summary.json', dict(registration_sha256=registration['result_sha256'],
        checkpoints={k: v['result_sha256'] for k, v in summaries.items()},
        fresh_audits_not_used_for_training=True, competitive_accuracy_certified=False))


def cached_quality_identity(parent: Path, step: int, *, allow_registered_milestone: bool = False) -> tuple[dict, dict, Path]:
    """Verify a terminal state or an explicitly requested, audited milestone.

    Milestones use immutable checkpoint/audit seals, never the mutable latest
    training record. They remain milestones, not a relabelled terminal run.
    This read-only extension does not change the trained amplitude definition.
    """
    frozen = read_sealed(parent/'registration.json')
    spec = frozen['config']
    if not isinstance(allow_registered_milestone, bool):
        raise ValueError('registered milestone permission must be an explicit boolean')
    if allow_registered_milestone:
        if (not isinstance(step, int) or isinstance(step, bool) or step <= 0
                or step not in spec['quality_milestones']):
            raise ValueError('cached milestone must be a registered frozen endpoint')
        identity_path = parent/'audits'/f'step{step:05d}'/'results.json'
        identity = read_sealed(identity_path)
        checkpoint = parent/'quality'/f'step{step:05d}.pkl'
        if (identity['step'] != step or identity['audit_projection'] != 'gamma_a1'
                or identity['training_projection'] != 'gamma_a1'
                or identity['registration_sha256'] != frozen['result_sha256']
                or identity['checkpoint_sha256'] != file_sha256(checkpoint)):
            raise ValueError('cached milestone checkpoint/full-A1 audit identity changed')
        identity = dict(identity, final_params_sha256=identity['params_sha256'])
    else:
        identity_path = parent/'training.json'
        identity = read_sealed(identity_path)
        if (step != sum(stage['steps'] for stage in spec['stages'])
                or identity['completed_steps'] != step
                or identity['status'] != 'fixed_steps_complete'):
            raise ValueError('only the declared complete frozen endpoint is allowed')
    return frozen, identity, identity_path


def audit_cached_physics(config_path: Path, output: Path | None = None) -> dict:
    """Bounded coherent physics of ONE frozen endpoint using only cached chains.

    Does not call register() on, modify, or re-sign the training run. No new
    sampler, H action, optimization, SU2 twirl, or RDM eigenvalue repair.
    The separately sealed readout records its own sources and verifies the
    frozen physical-amplitude definitions, checkpoint and every cache seal.
    """
    settings = json.loads(config_path.read_text())
    if settings['task'] != 'cached_spatial_vit_physics':
        raise ValueError('explicit cached-physics protocol required')
    parent = ROOT/Path(settings['parent'])
    destination = output or ROOT/settings['output']
    if destination.exists():
        raise FileExistsError('fresh immutable physics output required')
    step = settings['step']
    frozen, training, identity_path = cached_quality_identity(parent, step,
        allow_registered_milestone=settings.get('allow_registered_milestone', False))
    spec = frozen['config']; config = validate_spec(spec)
    a = spec['audit']
    if frozen['result_sha256'] != settings['parent_registration_sha256']:
        raise ValueError('only the declared complete frozen endpoint is allowed')
    checkpoint = parent/'quality'/f'step{step:05d}.pkl'
    if file_sha256(checkpoint) != settings['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    if len(frozen['source_manifest']) != settings['frozen_source_count']:
        raise ValueError('frozen source count changed')
    for relative, digest in frozen['source_manifest'].items():
        if file_sha256(parent/'source_snapshots'/relative) != digest:
            raise ValueError('frozen numerical source changed: '+relative)
    # These functions are the exact declared state definition. New readout
    # code may extend this file, but must not change their AST or model arrays.
    amplitude_ast = {}
    relative = str(Path(__file__).relative_to(ROOT))
    old_ast = ast.parse((parent/'source_snapshots'/relative).read_text())
    new_ast = ast.parse(Path(__file__).read_text())
    for name in ('validate_spec', 'model_log'):
        pick = lambda tree: next(node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name)
        old = ast.dump(pick(old_ast), include_attributes=False)
        new = ast.dump(pick(new_ast), include_attributes=False)
        if old != new:
            raise ValueError('physical-amplitude definition changed: '+name)
        amplitude_ast[name] = canonical_sha256(old)
    for relative in ('su2_nqs/models/fixed_sz_spatial_vit.py',
                     'su2_nqs/models/literature_spatial_vit.py',
                     'su2_nqs/benchmarks/spatial_vit_sampling.py',
                     'projects/clebsch_tree_nqs/workflows/spin.py'):
        if file_sha256(ROOT/relative) != frozen['source_manifest'][relative]:
            raise ValueError('frozen amplitude dependency changed: '+relative)
    portable_inputs = []
    if settings.get('frozen_parameter_artifact'):
        # Read ALL frozen model weights without deserializing an optional
        # NetKet sampler. This artifact NEVER restores optimizer/sampler state.
        from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
        artifact = ROOT/settings['frozen_parameter_artifact']
        metadata_path = artifact.with_name('metadata.json')
        expected = dict(purpose='Read-only physical analysis of a frozen trained model; not initialization or a resumable optimizer checkpoint',
            step=step, checkpoint_sha256=settings['checkpoint_sha256'],
            params_sha256=settings['params_sha256'], registration_sha256=frozen['result_sha256'])
        template = (init_literature_spatial_vit if spec.get('model_family') == 'literature'
                    else init_fixed_sz_spatial_vit)(jax.random.PRNGKey(0), config)
        params = load_parameter_artifact(artifact, template, expected_identity=expected)
        exported = json.loads(metadata_path.read_text())
        if exported['checkpoint_sha256'] != file_sha256(checkpoint):
            raise ValueError('portable metadata/checkpoint seal differs')
        saved = dict(params=dict(model=params), metadata=exported['metadata'])
        portable_inputs = [artifact, artifact.with_suffix('.npz'), metadata_path]
    else:
        saved = load_checkpoint(checkpoint); params = saved['params']['model']
    identity = pytree_sha256(params)
    if (identity != settings['params_sha256'] or identity != training['final_params_sha256']
            or saved['metadata']['registration_sha256'] != frozen['result_sha256']
            or checkpoint_projection(saved['metadata']) != 'gamma_a1'):
        raise ValueError('frozen final A1 parameters/stage changed')
    indices = settings['snapshot_indices']
    if (indices != sorted(set(indices)) or len(indices) < 2
            or any(not isinstance(i, int) or isinstance(i, bool)
                   or not 0 <= i < a['snapshots'] for i in indices)):
        raise ValueError('fixed distinct cache snapshot indices required')
    records, groups, cache_inputs, stream_ids = [], [], {}, []
    cached = parent/'audits'/f'step{step:05d}'
    for stream_index, original_seed in enumerate(a['seeds']):
        seed = original_seed+1009*step
        path = cached/f'stream{seed}.npz'; manifest = path.with_suffix('.json')
        stream = read_sealed(manifest)
        if (stream['seed'] != seed or stream['params_sha256'] != identity
                or stream['registration_sha256'] != frozen['result_sha256']
                or stream['raw_sha256'] != file_sha256(path)
                or stream['raw_sha256'] != settings['stream_sha256'][str(seed)]):
            raise ValueError('independent frozen audit cache changed')
        with np.load(path) as data:
            states = data['states']
            if states.shape != (a['snapshots'], a['chains'], config.n_sites):
                raise ValueError('cached chain shape changed')
            records.append(states[np.asarray(indices)])
        groups.append(np.tile(np.arange(a['chains'])+stream_index*a['chains'], len(indices)))
        cache_inputs[str(path)] = file_sha256(path)
        cache_inputs[str(manifest)] = file_sha256(manifest)
        stream_ids.append(seed)
    bits = np.asarray(records).reshape((-1, config.n_sites))
    chain_ids = np.concatenate(groups)
    if not np.isin(bits, (0, 1)).all() or not (bits.sum(axis=1) == config.target_n_up).all():
        raise ValueError('cached samples left the physical Sz support')
    if len(bits) != settings['expected_records'] or len(np.unique(chain_ids)) != settings['expected_chains']:
        raise ValueError('declared complete cached-chain record count changed')
    if (not isinstance(settings['query_batch'], int) or settings['query_batch'] < 1
            or not 0 < settings['compute_seconds'] <= 1800):
        raise ValueError('bounded query batch and at most thirty-minute compute cap required')
    if settings.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('readout GPU allocation required; no login-node computation')
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    destination.mkdir(parents=True, exist_ok=False)
    for name in sources:
        target = destination/'source_snapshots'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    audit_identity = cached/('results.json' if settings.get('allow_registered_milestone', False) else 'energy.json')
    input_paths = [checkpoint, parent/'registration.json', identity_path,
                   audit_identity, config_path, *portable_inputs]
    inputs = dict(cache_inputs, **{str(path): file_sha256(path) for path in input_paths})
    registration = sealed_json(destination/'registration.json', dict(config=settings,
        inputs_sha256=inputs, source_manifest=sources,
        frozen_parent_source_manifest=frozen['source_manifest'],
        amplitude_definition_AST_sha256=amplitude_ast,
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        params_sha256=identity, checkpoint_sha256=file_sha256(checkpoint),
        audit_seeds=stream_ids, fixed_snapshot_indices=indices,
        whole_chains=len(np.unique(chain_ids)), records_per_chain=len(indices),
        new_training=0, new_samples=0, new_H_calls=0,
        no_phase_or_strict_entanglement_spectrum_claim=True))
    start = perf_counter(); batch = settings['query_batch']
    ledger = dict(projected_amplitude_queries=0, padded_projected_amplitude_queries=0)
    def bounded():
        if perf_counter()-start >= settings['compute_seconds']:
            raise TimeoutError('hard cached-readout cap: no extension or partial physics claim')
    group = _groups(config.geometry, config.rows, config.columns)['gamma_a1']
    log_fn, images = model_log(spec, config, group)
    log_batch = jax.jit(jax.vmap(log_fn, in_axes=(None, 0)))
    def query(configurations):
        values = []
        for offset in range(0, len(configurations), batch):
            bounded(); x = configurations[offset:offset+batch]
            padded = np.pad(x, ((0, batch-len(x)), (0, 0)), mode='edge')
            values.append(np.asarray(log_batch(params, jnp.asarray(padded)))[:len(x)])
            ledger['projected_amplitude_queries'] += len(x)
            ledger['padded_projected_amplitude_queries'] += len(padded)
        return np.concatenate(values)
    coordinates = np.array([divmod(i, config.columns) for i in range(config.n_sites)])
    cut = settings.get('region')
    try:
        if 'regions' in settings:
            if cut is not None or not settings['regions']:
                raise ValueError('regional-only audit needs regions and no legacy region')
            from .spin_su2_analysis import cached_regional_readout, _problem_signature
            problem = settings['problem']
            signature, bonds = _problem_signature(problem,
                maximum_region_sites=8 if settings.get('raw_regional_only',False) else 6)
            if (problem['geometry'], problem['rows'], problem['columns'], problem['j2']) != (
                    config.geometry, config.rows, config.columns, spec['j2_over_j1']):
                raise ValueError('regional problem differs from frozen physical model')
            from su2_nqs.benchmarks.readout_quality import physical_problem_signature
            if signature != physical_problem_signature(config.n_sites,
                    _bonds(config.geometry, config.rows, config.columns, spec['j2_over_j1']),
                    geometry=f'{config.geometry}_{config.rows}x{config.columns}_row_major',
                    boundary='PBC_both_physical'):
                raise ValueError('regional labelled physical Hamiltonian differs')
            plans = {}
            for name, region_cut in settings['regions'].items():
                if (region_cut['sites'] != problem['regions'][name]
                        or region_cut['left'] != problem['left_sites'][name]
                        or region_cut.get('cyclic', False) != (name in problem.get('cyclic_regions', []))
                        or region_cut.get('compact_triangular_2x3', False) != (name in problem.get('compact_triangular_2x3', []))):
                    raise ValueError('regional geometry/partition/channel routing differs')
                plans[name] = regional_query_plan(bits, region_cut['sites'])
            configurations, inverse = np.unique(np.concatenate(
                [plan.configurations for plan in plans.values()]), axis=0, return_inverse=True)
            tick = perf_counter(); logs = query(configurations); query_seconds = perf_counter()-tick
            regions, arrays, offset = {}, dict(states=bits, chain_ids=chain_ids), 0
            for name, plan in plans.items():
                bounded()
                regional_logs = logs[inverse[offset:offset+len(plan.configurations)]]
                offset += len(plan.configurations)
                matrices = born_conditional_regional_matrices(plan, regional_logs)
                if settings.get('raw_regional_only', False):
                    summary = regional_density_summary(matrices, chain_ids)
                    raw_region = dict(rho=summary['rho'], jackknife_rhos=summary['jackknife_rhos'],
                        per_record_matrices=matrices, chain_ids=chain_ids)
                    report = dict(sites=settings['regions'][name]['sites'],
                        raw_only=True, strict_multiplicity_spectrum_reported=False,
                        input_repaired=False,
                        diagnostics={k:v for k,v in summary.items() if k not in ('rho','jackknife_rhos')})
                else:
                    report, raw_region = cached_regional_readout(matrices, chain_ids, settings['regions'][name])
                regions[name] = report
                arrays.update({name+'__'+key: value for key, value in raw_region.items()})
                arrays[name+'__region_sites'] = np.array(plan.sites)
            raw = destination/'physics.npz'; np.savez_compressed(raw, **arrays)
            bounded()
            if (pytree_sha256(params) != identity
                    or {path:file_sha256(Path(path)) for path in inputs} != inputs):
                raise AssertionError('frozen state/cache mutated during regional readout')
            return sealed_json(destination/'results.json', dict(
                registration_sha256=registration['result_sha256'], params_sha256=identity,
                checkpoint_sha256=file_sha256(checkpoint), step=step, raw_sha256=file_sha256(raw),
                physical_problem_signature=signature, bonds=bonds,
                model_parameter_count=sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)),
                runtime=numerical_runtime_identity(),
                regions=regions, image_count=images, ledger=ledger,
                unique_combined_queries=len(configurations),
                unique_regional_queries={k:len(v.configurations) for k,v in plans.items()},
                main_query_seconds=query_seconds, compute_seconds=perf_counter()-start,
                independent_chains=len(np.unique(chain_ids)), records=len(bits), records_per_chain=len(indices),
                new_training=0, new_samples=0, new_H_calls=0, strict_SU2=False, full_A1=True,
                strict_multiplicity_spectrum_reported=False, physical_phase_claim=False,
                readout_used_for_training_or_selection=False, mixing_certified=False,
                warning='Whole-chain jackknife includes off-diagonal coherences and nonlinear conditional readouts. Cached Born-law equilibrium and unseen-tail bias are not certified. No SU2 repair or phase assignment.'))
        timing_plan = scalar_structure_query_plan(bits[:settings['timing_records']])
        tick = perf_counter(); timing_logs = query(timing_plan.configurations)
        timing_seconds = perf_counter()-tick
        # Existing all-pair JAX estimator is an independent implementation
        # replay on TWO fixed records only, not additional H or Born samples.
        scalar_kernel = make_scalar_structure_factors(log_fn, coordinates, spec['momenta'])
        tick = perf_counter(); reference, diagonal = scalar_kernel(params, jnp.asarray(bits[:2]))
        reference, diagonal = np.asarray(reference), np.asarray(diagonal)
        ledger['independent_replay_projected_amplitude_queries'] = 2*(1+config.n_sites*(config.n_sites-1)//2)
        scalar_timing, zz_timing = scalar_structure_records(timing_plan, timing_logs, coordinates, spec['momenta'])
        replay_error = float(np.max(abs(scalar_timing[:2]-reference)))
        if replay_error > 2e-10 or np.max(abs(zz_timing[:2]-diagonal)) > 2e-12:
            raise AssertionError('coalesced full-scalar replay disagrees; no readout interpreted')
        timing = sealed_json(destination/'timing.json', dict(records=settings['timing_records'],
            unique_projected_queries=len(timing_plan.configurations),
            compile_and_first_seconds=timing_seconds,
            independent_scalar_replay_seconds=perf_counter()-tick,
            independent_scalar_replay_maximum_absolute_error=replay_error,
            image_count=images, completed_before_full_readout=True))
        bounded()
        pairs = scalar_structure_query_plan(bits)
        region = regional_query_plan(bits, cut['sites'])
        configurations, inverse = np.unique(np.concatenate((pairs.configurations, region.configurations)),
                                            axis=0, return_inverse=True)
        tick = perf_counter(); logs = query(configurations); query_seconds = perf_counter()-tick
        scalar, zz = scalar_structure_records(pairs, logs[inverse[:len(pairs.configurations)]],
                                               coordinates, spec['momenta'])
        matrices = born_conditional_regional_matrices(region, logs[inverse[len(pairs.configurations):]])
        summary = regional_density_summary(matrices, chain_ids)
        rho = summary['rho']; physics, sigma = multiplicity_operator_data(rho)
        physics['physical'].pop('invariant_readout', None)  # Never manufacture a strict ES.
        rotation = rotational_symmetry_defect(rho)
        noise = rotational_defect_chain_audit(matrices, chain_ids)
        ids = np.unique(chain_ids)
        means = np.stack([scalar[chain_ids == c].mean(axis=0) for c in ids])
        zmeans = np.stack([zz[chain_ids == c].mean(axis=0) for c in ids])
        rng = np.random.default_rng(settings['bootstrap_seed'])
        resamples = rng.integers(0, len(ids), (settings['bootstrap_replicates'], len(ids)))
        def chain_report(values):
            return dict(value=float(values.mean().real), imaginary=float(values.mean().imag),
                whole_chain_SE=float(values.real.std(ddof=1)/np.sqrt(len(values))),
                imaginary_whole_chain_SE=float(values.imag.std(ddof=1)/np.sqrt(len(values))),
                bootstrap_95=tuple(map(float, np.percentile(values.real[resamples].mean(axis=1), [2.5,97.5]))))
        sf = {name:dict(scalar=chain_report(means[:, i]), zz=chain_report(zmeans[:, i]))
              for i, name in enumerate(spec['momenta'])}
        gamma = list(spec['momenta']).index('Gamma')
        global_s2 = chain_report(config.n_sites*means[:, gamma])
        parity_bound = None
        if config.n_sites % 4 == 0 and spec.get('spin_inversion_even', False):
            parity_bound = dict(non_singlet_weight_upper_bound_estimate=global_s2['value']/6,
                estimated_upper_bound_whole_chain_SE=global_s2['whole_chain_SE']/6,
                bootstrap_95=tuple(value/6 for value in global_s2['bootstrap_95']),
                exact_state_inequality='P(S>0)<=<S²>/6 for bit-flip-even Sz0 at N divisible by four',
                scope='Only the true expectation obeys the rigorous inequality. This finite-chain estimate/CI is not a rigorous probability certificate; no clipping of negative estimates.')
        operators = {f'p_q{q}': p for q, p in _spin_projectors(len(cut['sites']))}
        observables = {name:regional_observable_summary(summary, op) for name,op in operators.items()}
        raw = destination/'physics.npz'
        np.savez_compressed(raw, states=bits, chain_ids=chain_ids, scalar=scalar, zz=zz,
            rho=rho, jackknife_rhos=summary['jackknife_rhos'], per_record_matrices=matrices,
            region_sites=cut['sites'], **{f'sigma_q{q}':matrix for q,matrix in sigma.items()})
        bounded()
        if pytree_sha256(params) != identity or file_sha256(checkpoint) != settings['checkpoint_sha256']:
            raise AssertionError('frozen endpoint mutated during readout')
        current_inputs = {path:file_sha256(Path(path)) for path in inputs}
        if current_inputs != inputs:
            raise AssertionError('cached source/endpoint/audit input changed')
        return sealed_json(destination/'results.json', dict(registration_sha256=registration['result_sha256'],
            params_sha256=identity, step=step,
            model_parameter_count=sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)),
            checkpoint_sha256=file_sha256(checkpoint), raw_sha256=file_sha256(raw),
            registered_milestone_readout=settings.get('allow_registered_milestone', False),
            runtime=numerical_runtime_identity(), source_training_projection='gamma_a1',
            strict_SU2=False, full_A1=True, spin_inversion_even=spec.get('spin_inversion_even',False),
            structure_factor_definition='S(k)=sum_ij exp[2pi i k.(r_i-r_j)]<S_i.S_j>/N',
            structure_factors=sf, scalar_whole_chain_covariance_of_means=(np.cov(means.real,rowvar=False)/len(ids)).tolist(),
            total_spin_squared=global_s2, even_spin_non_singlet_bound=parity_bound,
            region=dict(sites=cut['sites'], conditional_coherent_Born_estimator=True,
                diagnostics={key:value for key,value in summary.items() if key not in ('rho','jackknife_rhos')},
                multiplicity_operator_data=physics, rotational_audit=rotation, chain_noise_audit=noise,
                spin_weight_observables=observables, joint_spin=spin_partition_readout(rho,cut['left']),
                strict_multiplicity_spectrum_reported=False, input_repaired=False),
            timing=timing, image_count=images, ledger=ledger,
            unique_pair_amplitude_queries=len(pairs.configurations),
            unique_regional_amplitude_queries=len(region.configurations),
            unique_combined_queries=len(configurations), main_query_seconds=query_seconds,
            compute_seconds=perf_counter()-start, new_training=0,new_samples=0,new_H_calls=0,
            independent_chains=len(ids),records=len(bits),records_per_chain=len(indices),
            readout_used_for_training_or_selection=False, mixing_certified=False,
            competitive_accuracy_certified=False, physical_phase_claim=False,
            warning='Whole chains are uncertainty units; reused configurations inherit prior Born-equilibration/tail limits. Regional invariant-algebra data are legal without strict SU2, not physical SU2 entanglement levels.'))
    except BaseException as error:
        sealed_json(destination/'incomplete.json', dict(registration_sha256=registration['result_sha256'],
            exception_type=type(error).__name__,message=str(error),ledger=ledger,
            compute_seconds=perf_counter()-start,complete_physics_reported=False,
            new_H_calls=0,new_samples=0,new_training=0))
        raise


def materialize_cached_plan(plan: dict) -> dict:
    """Fill future seals, not choices, after a fixed endpoint/audit completes.

    Region shapes, snapshot indices, budgets and bootstrap seeds are already
    fixed in the submitted plan. No energy or readout value selects them.
    """
    parent = ROOT/plan['parent']
    step = plan['step']
    milestone = plan.get('allow_registered_milestone', False)
    reg, training, identity_path = cached_quality_identity(parent, step,
        allow_registered_milestone=milestone)
    spec = reg['config']
    if (spec['model']['geometry'] != plan['problem']['geometry']
            or spec['model']['rows'] != plan['problem']['rows']
            or spec['model']['columns'] != plan['problem']['columns']
            or spec['j2_over_j1'] != plan['problem']['j2']):
        raise ValueError('cached-plan endpoint or physical problem changed')
    checkpoint = parent/'quality'/f'step{step:05d}.pkl'
    if training['registration_sha256'] != reg['result_sha256']:
        raise ValueError('cached-plan training belongs to another registration')
    audit = parent/'audits'/f'step{step:05d}'
    energy = read_sealed(audit/('results.json' if milestone else 'energy.json'))
    if energy['params_sha256'] != training['final_params_sha256']:
        raise ValueError('cached-plan energy audit belongs to another state')
    streams = {}
    for seed in spec['audit']['seeds']:
        actual = seed+1009*step
        path = audit/f'stream{actual}.npz'
        record = read_sealed(path.with_suffix('.json'))
        if (record['params_sha256'] != training['final_params_sha256']
                or record['raw_sha256'] != file_sha256(path)):
            raise ValueError('cached-plan audit stream changed')
        streams[str(actual)] = record['raw_sha256']
    settings = dict(plan, task='cached_spatial_vit_physics',
        parent_registration_sha256=reg['result_sha256'], checkpoint_sha256=file_sha256(checkpoint),
        params_sha256=training['final_params_sha256'], frozen_source_count=len(reg['source_manifest']),
        stream_sha256=streams, expected_chains=len(streams)*spec['audit']['chains'],
        expected_records=len(streams)*spec['audit']['chains']*len(plan['snapshot_indices']))
    settings.pop('materialized_config', None)
    return settings


def run_cached_plan(path: Path) -> dict:
    """Run an immutable submitted readout plan using the existing estimator."""
    plan = json.loads(path.read_text())
    settings = materialize_cached_plan(plan)
    target = ROOT/plan['materialized_config']
    if target.exists():
        if json.loads(target.read_text()) != settings:
            raise ValueError('cached-plan materialized seals changed')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(target, settings)
    return audit_cached_physics(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'audit', 'audit-energy', 'campaign', 'alignment', 'cached-physics', 'cached-plan'))
    parser.add_argument('--config', type=Path, default=STABLE_CONTINUATION_DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--until-step', type=int)
    parser.add_argument('--session-seconds', type=float, default=1800.)
    parser.add_argument('--steps', type=int, nargs='+')
    parser.add_argument('--stream-index', type=int)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    output = args.output or ROOT/spec['output']
    if args.phase == 'cached-plan':
        run_cached_plan(args.config)
        return
    if args.phase == 'cached-physics':
        audit_cached_physics(args.config, output)
        return
    if args.phase == 'audit-energy':
        if args.steps is not None and len(args.steps) != 1:
            parser.error('energy-only audit accepts one frozen milestone')
        audit_energy(args.config, output, stream_index=args.stream_index, aggregate_only=args.aggregate_only,
                     step=None if args.steps is None else args.steps[0])
        return
    if args.phase == 'alignment':
        alignment_benchmark(args.config, output)
        return
    if args.phase in {'train', 'campaign'}:
        result = train(args.config, output, resume=args.resume, until_step=args.until_step,
                       session_seconds=args.session_seconds)
        if args.phase == 'campaign' and result['status'] in {'fixed_steps_complete', 'total_time_cap'}:
            selected = args.steps
            if selected is None:
                available = sorted(int(p.stem[4:]) for p in (output/'quality').glob('step*.pkl'))
                selected = [available[-1], *available[:-1]]  # Terminal audit first, no endpoint selection.
            audit(args.config, output, steps=selected)
    else:
        audit(args.config, output, steps=args.steps)


def alignment_benchmark(config_path: Path, output: Path) -> dict:
    """Fixed-work reference-capacity costs, with common arrays for LN control.

    Benchmark samples are random fixed-m Born-sector configurations, NOT a
    trained/audited state. These timings establish actual operation cost and
    correctness, not achievable precision. Each timing synchronizes JAX.
    GPU production must measure its own samples/sec and complete-update time.
    """
    spec = json.loads(config_path.read_text())
    if spec['task'] != 'spatial_vit_literature_alignment':
        raise ValueError('alignment protocol required')
    if (output/'benchmark.json').exists():
        return read_sealed(output/'benchmark.json')
    output.mkdir(parents=True, exist_ok=True)
    record = sealed_json(output/'benchmark_registration.json', dict(config=spec,
        config_sha256=canonical_sha256(spec), source_manifest=transitive_local_source_manifest(
            (Path(__file__),), root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')),
        registered_at_utc=datetime.now(timezone.utc).isoformat(), no_precision_claim=True))
    from dataclasses import replace
    measurements = []
    for capacity in spec['capacities']:
        config = LiteratureSpatialViTConfig(**dict(spec['model'], **capacity['model']))
        params = init_literature_spatial_vit(jax.random.PRNGKey(spec['initialization_seed']), config)
        states = initialize_spin_exchange_chains(jax.random.PRNGKey(spec['timing_seed']),
            n_chains=spec['timing_records'], n_sites=config.n_sites, n_up=config.target_n_up)
        groups = _groups(config.geometry, config.rows, config.columns)
        bonds = _bonds(config.geometry, config.rows, config.columns, spec['j2_over_j1'])
        row = dict(name=capacity['name'], config=asdict(config),
                   parameter_count=fixed_sz_spatial_vit_parameter_count(params),
                   params_sha256=pytree_sha256(params), operations={})
        for projection in spec['timing_projections']:
            log_fn, images = model_log(dict(model_family='literature', spin_inversion_even=True),
                                      config, None if projection == 'none' else groups[projection])
            log_batch = jax.jit(jax.vmap(log_fn, in_axes=(None, 0)))
            local = make_full_heisenberg_ratios(log_fn, bonds)
            sampler = make_log_exchange_sampler(log_fn, spec['timing_proposals'])
            gradient = jax.jit(jax.grad(lambda p, x: jnp.mean(log_batch(p, x).real)))
            operations = {
                'amplitude': lambda: log_batch(params, states),
                'full_H': lambda: local(params, states),
                'mean_log_reverse_gradient': lambda: gradient(params, states),
                'sampler': lambda: sampler(params, jax.random.PRNGKey(3), states)}
            stats = dict(image_count=images, records=len(states), proposals=spec['timing_proposals'])
            for name, operation in operations.items():
                tick = perf_counter(); value = operation(); jax.block_until_ready(value)
                first = perf_counter()-tick
                timings = []
                for _ in range(spec['timing_repeats']):
                    tick = perf_counter(); jax.block_until_ready(operation()); timings.append(perf_counter()-tick)
                if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(value)):
                    raise FloatingPointError('nonfinite reference benchmark')
                stats[name] = dict(compile_and_first_seconds=first,
                    median_seconds=float(np.median(timings)), repeats_seconds=timings)
            row['operations'][projection] = stats
            print('reference cost', capacity['name'], projection, stats, flush=True)
        # Pooling-only ablation: EXACTLY identical arrays, output norms and capacity.
        control = replace(config, pooling='tokenwise_control')
        raw, _ = model_log(dict(model_family='literature'), config, None)
        other, _ = model_log(dict(model_family='literature'), control, None)
        delta = np.asarray(jax.vmap(lambda x: raw(params, x)-other(params, x))(states))
        row['pooling_control'] = dict(identical_parameter_hash=pytree_sha256(params),
            real_log_difference_std=float(delta.real.std()),
            imaginary_log_difference_std=float(delta.imag.std()),
            same_initial_physical_state=False, only_pooling_order_changes=True)
        measurements.append(row)
        # Each capacity is independently persisted before releasing compiled executables.
        sealed_json(output/(capacity['name']+'.json'), row)
        jax.clear_caches()
    return sealed_json(output/'benchmark.json', dict(
        registration_sha256=record['result_sha256'], runtime=numerical_runtime_identity(),
        measurements=measurements, gpu_timing_measured=jax.default_backend() == 'gpu',
        full_training_update_timed=False, precision_or_phase_claim=False))


if __name__ == '__main__':
    main()
