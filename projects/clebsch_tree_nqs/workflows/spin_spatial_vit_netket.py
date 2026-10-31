"""Optional mature NTK-SR backend for the EXISTING author-aligned Spatial-ViT.

No new ansatz/head/carrier. NetKet owns sampling, JAX H action and NTK-SR;
the exact same pure-JAX physical log amplitude and local audit workflow are
reused. Fixed Sz/parity/spatial projection are NOT exact SU(2). Only trusted
locally produced pickle checkpoints may be loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import importlib.metadata
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import pytree_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity, file_sha256
from su2_nqs.models.fixed_sz_spatial_vit import fixed_sz_spatial_vit_parameter_count
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from .spin import _bonds, _groups
from .spin_spatial_vit_quality import (
    ROOT, validate_spec, initial_model, model_log, register, read_sealed,
    sealed_json, stage_at, _snapshot,
)

jax.config.update('jax_enable_x64', True)
NETKET_VERSION = '3.22.3'


def balanced_spin_half_initial_states(key: jax.Array, chains: int, n_sites: int,
                                      dtype: object = jnp.int8) -> jax.Array:
    """Generate uniform legal doubled-m=0 configurations, never repair samples.

    A uniform permutation of N/2 entries -1 and N/2 entries +1 gives each
    fixed-Sz spin-half configuration the same (N/2)!² multiplicity. NetKet
    inputs are doubled magnetic labels; the physical model maps them to bits.
    This is an initial-state law, not a claim of Born-law equilibration.
    """
    if (isinstance(chains, bool) or not isinstance(chains, int) or chains < 1
            or isinstance(n_sites, bool) or not isinstance(n_sites, int)
            or n_sites < 2 or n_sites % 2):
        raise ValueError('positive chains and even spin-half N required')
    base = jnp.concatenate((-jnp.ones(n_sites//2, dtype=dtype),
                            jnp.ones(n_sites//2, dtype=dtype)))
    keys = jax.random.split(key, chains)
    return jax.vmap(lambda rng: jax.random.permutation(rng, base))(keys)


def uniform_exchange_proposal(key: jax.Array, spins: jax.Array) -> jax.Array:
    """Symmetric all-pair proposal on doubled magnetic labels, including null moves.

    Both gathered values come from the OLD row. Each ordered distinct pair
    has probability 1/[N(N-1)]; the inverse has exactly the same probability.
    Thus no Hastings correction or state-dependent cluster mask is needed.
    """
    if spins.ndim != 2 or spins.shape[1] < 2:
        raise ValueError('a batch of at least two-site spin configurations is required')
    n = spins.shape[1]
    a, b = jax.random.split(key)
    i = jax.random.randint(a, (len(spins),), 0, n)
    short_j = jax.random.randint(b, (len(spins),), 0, n-1)
    j = short_j+(short_j >= i)
    si = jnp.take_along_axis(spins, i[:, None], axis=1)
    sj = jnp.take_along_axis(spins, j[:, None], axis=1)
    sites = jnp.arange(n)[None, :]
    return jnp.where(sites == i[:, None], sj,
                     jnp.where(sites == j[:, None], si, spins))


def physical_hamiltonian(spec: dict):
    """Same physical graph, NO Marshall sign transformation; S=Pauli/2."""
    import netket as nk
    config = validate_spec(spec)
    bonds = _bonds(config.geometry, config.rows, config.columns, spec['j2_over_j1'])
    weights = sorted({float(w) for _, _, w in bonds})
    graph = nk.graph.Graph(edges=[(i, j, weights.index(float(w))) for i, j, w in bonds],
                          n_nodes=config.n_sites)
    hilbert = nk.hilbert.Spin(s=.5, N=config.n_sites, total_sz=0)
    # NetKet's Heisenberg uses Pauli dot products. Dividing J by4 here,
    # not only the plotted energy, also aligns SR step/gradient units.
    h = nk.operator.Heisenberg(hilbert, graph, J=[w/4 for w in weights],
                              sign_rule=[False]*len(weights), dtype=np.float64,
                              cls=nk.operator.LocalOperatorJax)
    return hilbert, graph, h


def make_driver(spec: dict, params: dict, projection: str, *, step: int = 0):
    """NetKet ±1 inputs map explicitly to repository down/up bits0/1."""
    config = validate_spec(spec)
    expected_devices = spec.get('netket', {}).get('expected_device_count')
    if expected_devices is not None and jax.device_count() != expected_devices:
        raise RuntimeError(
            f'registered device count is {expected_devices}, observed {jax.device_count()}; '
            'refusing to change the validated sampling/SR execution layout')
    import netket as nk
    import optax
    if nk.__version__ != NETKET_VERSION:
        raise ValueError('use the tested pinned NetKet backend')
    group = None if projection == 'none' else _groups(config.geometry, config.rows, config.columns)[projection]
    log_fn, images = model_log(spec, config, group)
    def apply(variables, spins):
        bits = ((jnp.asarray(spins)+1)/2).astype(jnp.int32)
        return jax.vmap(log_fn, in_axes=(None, 0))(variables['params'], bits)
    hilbert, graph, h = physical_hamiltonian(spec)
    sampling = spec['sampling']; backend = spec['netket']
    if backend.get('version',NETKET_VERSION)!=NETKET_VERSION or backend.get('momentum') is not None:
        raise ValueError('only the pinned, momentum-free SR protocol is tested')
    initialization = backend.get('initialization', 'native')
    kernel = backend.get('exchange_kernel', 'native')
    if kernel not in ('native', 'uniform_all_pairs'):
        raise ValueError('unvalidated exchange kernel')
    if initialization in ('balanced_spin_half', 'parent_configurations'):
        parent_spins = None
        if initialization == 'parent_configurations':
            # Parameters are verified separately by initial_model. Keep ALL
            # declared parent chains, not a favorable subset; use the NEW
            # experiment's RNG/log probabilities/optimizer/work counters.
            declaration = spec.get('coupling_warm_start_from') or spec['repair_parameters_from']
            checkpoint = ROOT/declaration['directory']/declaration['checkpoint_path']
            if file_sha256(checkpoint) != declaration['checkpoint_sha256']:
                raise ValueError('initial parent configuration checkpoint changed')
            parent_spins = np.array(load_checkpoint(checkpoint)['params']['netket_sampler_state'].σ, copy=True)
            if (parent_spins.shape != (sampling['chains'], config.n_sites)
                    or not np.isin(parent_spins, [-1, 1]).all()
                    or not (parent_spins.sum(-1) == 0).all()):
                raise ValueError('declared parent chains are not complete legal Sz=0 configurations')
        # A separately registered initialization-only revision. Retain the
        # identical ExchangeRule.transition; override random_state BEFORE
        # generation instead of clipping/replacing illegal production samples.
        class ExactSpinHalfExchangeRule(nk.sampler.rules.ExchangeRule):
            def random_state(self, sampler, machine, parameters, sampler_state, key):
                return balanced_spin_half_initial_states(
                    key, sampler.n_batches, config.n_sites, sampler.dtype)
        class UniformSpinHalfExchangeRule(ExactSpinHalfExchangeRule):
            def transition(self, sampler, machine, parameters, state, key, spins):
                return uniform_exchange_proposal(key, spins), None
        rule_type = UniformSpinHalfExchangeRule if kernel == 'uniform_all_pairs' else ExactSpinHalfExchangeRule
        rule = rule_type(graph=graph, d_max=config.n_sites)
        sampler = nk.sampler.MetropolisSampler(hilbert, rule,
            n_chains=sampling['chains'], sweep_size=sampling['proposals_between_snapshots'])
    elif initialization == 'native':
        if kernel != 'native':
            raise ValueError('uniform exchange requires an explicit exact-support initial law')
        sampler = nk.sampler.MetropolisExchange(hilbert, graph=graph, d_max=config.n_sites,
            n_chains=sampling['chains'], sweep_size=sampling['proposals_between_snapshots'])
    else:
        raise ValueError('unvalidated NetKet initialization law')
    vs = nk.vqs.MCState(sampler, apply_fun=apply, variables={'params': params},
        n_samples=sampling['chains']*sampling['snapshots_per_update'],
        n_discard_per_chain=int(np.ceil(sampling['burn_proposals']/sampling['proposals_between_snapshots'])),
        chunk_size=backend['chunk_size'], seed=spec['initialization_seed'],
        sampler_seed=spec['training_seed'])
    if backend.get('replicate_initial_params', False) and jax.device_count() > 1:
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('S',))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        vs.parameters = jax.tree_util.tree_map(
            lambda value: jax.device_put(np.array(value, copy=True), replicated), params)
    if initialization == 'parent_configurations':
        # Do NOT hide a checkpoint array in a JIT-static rule closure. Install
        # all verified parent rows in the native template's global sharding;
        # retain the NEW experiment RNG. MCState.sample recomputes log_prob.
        template = vs.sampler_state.σ
        native_spins = jax.device_put(np.array(parent_spins, dtype=template.dtype, copy=True), template.sharding)
        vs.sampler_state = vs.sampler_state.replace(σ=native_spins)
        vs.reset()
    # Repository convention: (2 Re Cov + lambda I)d = 2 Re force,
    # delta theta = -eta d. NetKet 3.22.3 uses Re Cov and twice the
    # force, hence returns 2d at diag_shift=lambda/2. Halve its SGD rate
    # to preserve the registered PHYSICAL update, not just its direction.
    stage_index, _ = stage_at(spec, step)
    solver = dict(spec['sr'], **spec['stages'][stage_index].get('sr', {}))
    driver = nk.driver.VMC_SR(h, optax.sgd(solver['step_size']/2),
        variational_state=vs, diag_shift=solver['damping']/2,
        mode='complex', use_ntk=True, on_the_fly=backend['on_the_fly'],
        chunk_size_bwd=backend['chunk_size_bwd'], momentum=None)
    driver._step_count = step
    return driver, h, log_fn, images


def bounded_physical_direction(driver, direction: object, samples: jax.Array,
                               spec: dict | None = None) -> tuple[object, dict]:
    """Bound a nonlinear sampled state distance without additional H calls.

    First passing fixed halving, never energy/audit-selected backtracking.
    This is a training-sample bound, NOT an out-of-sample certificate. A failed
    attempt keeps the parameters but still consumes the registered step/H work.
    """
    trust = None if spec is None else spec['netket'].get('sampled_trust')
    if trust is None:
        return direction, {}
    stage_index, _ = stage_at(spec, int(driver.step_count))
    solver = dict(spec['sr'], **spec['stages'][stage_index].get('sr', {}))
    learning_rate = solver['step_size']/2
    from netket.jax import apply_chunked
    old = driver.state.parameters
    flat_samples = samples.reshape((-1, samples.shape[-1]))
    apply = apply_chunked(driver.state._apply_fun, in_axes=(None, 0), chunk_size=driver.state.chunk_size)
    before = apply({'params': old}, flat_samples)
    last = {}
    for k in range(trust['halvings']):
        scale = 2.**(-k)
        candidate = jax.tree_util.tree_map(
            lambda p, d: p-learning_rate*scale*d, old, direction)
        ratios = apply({'params': candidate}, flat_samples)-before
        diagnostics = sampled_overlap_diagnostics(ratios)
        jax.block_until_ready(diagnostics)
        last = {name: float(value) for name, value in diagnostics.items()}
        if (bool(jnp.isfinite(ratios).all())
                and last['sampled_fs_sine'] <= trust['maximum_fs_sine']
                and last['reweighting_ess_fraction'] >= trust['minimum_ess_fraction']):
            return jax.tree_util.tree_map(lambda d: scale*d, direction), dict(
                **last, update_scale=scale, trust_evaluations=k+1, update_accepted=True)
    return jax.tree_util.tree_map(jnp.zeros_like, direction), dict(
        sampled_fs_sine=0., reweighting_ess_fraction=1., update_scale=0.,
        trust_evaluations=trust['halvings'], update_accepted=False,
        last_rejected_fs_sine=last.get('sampled_fs_sine'))


def driver_payload(driver) -> tuple[dict, dict]:
    """Completed-step checkpoint: parameters, complete sampler RNG and SR state."""
    state = dict(model=driver.state.parameters, netket_sampler_state=driver.state.sampler_state)
    optimizer = dict(step=int(driver.step_count), optimizer=driver._optimizer_state,
                     old_updates=driver._old_updates, loss_stats_online=driver._loss_stats_online,
                     netket_version=NETKET_VERSION)
    return state, optimizer


def restore_driver(driver, saved: dict) -> None:
    """Restore only our pinned-version, full-state checkpoints, not weights alone."""
    state, optimizer = saved['params'], saved['optimizer_state']
    if optimizer['netket_version'] != NETKET_VERSION:
        raise ValueError('optimizer version changed')
    parameters = state['model']
    if jax.device_count() > 1:
        # A pickle restores JAX leaves on one device. Native SR returns
        # replicated parameters; restore that layout explicitly as well as
        # sample partitions. This changes layout, never numerical values.
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('S',))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        parameters = jax.tree_util.tree_map(
            lambda value: jax.device_put(np.array(value, copy=True), replicated), parameters)
        jax.block_until_ready(parameters)
    driver.state.parameters = parameters
    sampler_state = state['netket_sampler_state']
    if jax.device_count() > 1:
        # Preserve global configurations/RNG values, but adopt the sampler's
        # native multi-device layout rather than the pickle's single device.
        def place(value, template):
            if isinstance(template, jax.Array):
                if np.shape(value) != template.shape:
                    raise ValueError('global sampler shape changed during GPU recovery')
                if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                    data=np.array(jax.random.key_data(value),copy=True)
                    host_value=jax.random.wrap_key_data(jnp.asarray(data),impl=jax.random.key_impl(value))
                else:
                    host_value=np.array(value,copy=True)
                return jax.device_put(host_value, template.sharding)
            return value
        sampler_state = jax.tree_util.tree_map(place, sampler_state, driver.state.sampler_state)
        jax.block_until_ready(sampler_state)
        for original, placed in zip(jax.tree_util.tree_leaves(state['netket_sampler_state']),
                                    jax.tree_util.tree_leaves(sampler_state)):
            if isinstance(original, jax.Array) and jax.dtypes.issubdtype(original.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(jax.random.key_data(original),jax.random.key_data(placed))
            else:
                np.testing.assert_array_equal(original,placed)
    driver.state.sampler_state = sampler_state
    driver._optimizer_state = optimizer['optimizer']
    driver._old_updates = optimizer['old_updates']
    driver._loss_stats_online = optimizer['loss_stats_online']
    driver._step_count = int(optimizer['step'])
    driver.state.reset()


def recovery_state(spec: dict, registration_sha256: str) -> dict:
    """Recover a sealed NetKet trajectory into a separately registered output.

    Only runtime chunks, equivalent NTK construction, output and an explicitly
    declared nonnegative wall-time extension may change. The fixed step count,
    physical/model/SR/sampling/audit fields and cumulative counters are retained.
    Original source snapshots, records and checkpoint are never re-signed.
    """
    declaration = spec['netket_resume_from']
    parent = ROOT/Path(declaration['directory'])
    progress = read_sealed(parent/'progress.json')
    registration = read_sealed(parent/'registration.json')
    checkpoint = parent/'checkpoint.pkl'
    for label, actual in (('checkpoint_sha256', file_sha256(checkpoint)),
                          ('registration_sha256', registration['result_sha256']),
                          ('progress_sha256', progress['result_sha256'])):
        if declaration[label] != actual:
            raise ValueError('recovery parent seal changed: '+label)
    if progress['checkpoint_sha256'] != declaration['checkpoint_sha256']:
        raise ValueError('parent progress/checkpoint mismatch')
    for relative, expected in registration['source_manifest'].items():
        if file_sha256(parent/'source_snapshots'/relative) != expected:
            raise ValueError('parent frozen runtime snapshot changed')
    old = copy.deepcopy(registration['config'])
    new = copy.deepcopy(spec)
    new.pop('netket_resume_from')
    old.pop('output'); new.pop('output')
    extension = declaration.get('time_budget_extension_seconds', 0)
    if (isinstance(extension, bool) or not isinstance(extension, (int, float))
            or not np.isfinite(extension) or extension < 0
            or new['maximum_training_seconds'] != old['maximum_training_seconds']+extension):
        raise ValueError('runtime recovery time cap requires an exact declared nonnegative extension')
    if ('time_budget_extension_seconds' in declaration
            and (progress.get('status') != 'paused_session_budget'
                 or (parent/'training.json').exists())):
        raise ValueError('declared time extension requires a safely paused unfinished parent')
    new['maximum_training_seconds'] = old['maximum_training_seconds']
    # Chunking only changes evaluation/reduction geometry, not the estimator
    # or SR equation. Any other protocol change needs a different experiment.
    for key in ('chunk_size', 'chunk_size_bwd', 'on_the_fly'):
        old['netket'].pop(key, None); new['netket'].pop(key, None)
    if old != new:
        raise ValueError('recovery changed more than runtime NTK/chunks/output')
    saved = load_checkpoint(checkpoint)
    meta = saved['metadata']
    if (meta['completed_steps'] != declaration['completed_steps']
            or saved['optimizer_state']['step'] != meta['completed_steps']
            or len(meta['history']) != meta['completed_steps']
            or meta['registration_sha256'] != registration['result_sha256']
            or pytree_sha256(saved['params']['model']) != progress['final_params_sha256']):
        raise ValueError('parent parameters/count/optimizer do not match the seal')
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(saved['params']['model'])):
        raise ValueError('nonfinite parent parameters')
    saved = copy.deepcopy(saved)
    saved['metadata']['registration_sha256'] = registration_sha256
    saved['metadata']['recovery_parent'] = dict(**declaration,
        parameters_reinitialized=False, sampler_reinitialized=False,
        physics_and_SR_unchanged=True, global_work_budget_inherited=True,
        parent_runtime_chunks=registration['config']['netket'],
        recovery_runtime_chunks=spec['netket'],
        parent_training_time_cap_seconds=registration['config']['maximum_training_seconds'],
        recovery_training_time_cap_seconds=spec['maximum_training_seconds'],
        fixed_step_budget_unchanged=True)
    return saved


def validate_continuation_protocol(parent_spec: dict, spec: dict) -> None:
    """Allow a declared full-A1 extension of a frozen, audited boundary.

    Model, Hamiltonian, SR and GLOBAL samples per update stay unchanged.
    A new authorized time cap must be declared as an explicit nonnegative
    extension; cumulative work/time counters are never reset. New audit
    seeds and milestones are preregistered before new training. This is a
    new experiment, not mutation of the parent's seal.
    """
    declaration = spec['netket_continue_from']
    additional = declaration['additional_steps']
    if isinstance(additional, bool) or not isinstance(additional, int) or not 1 <= additional <= 1000:
        raise ValueError('continuation needs a declared 1–1000 additional steps')
    old = copy.deepcopy(parent_spec)
    new = copy.deepcopy(spec)
    for value in (old, new):
        value.pop('output', None)
        value.pop('netket_resume_from', None)
        value.pop('netket_continue_from', None)
    time_extension = declaration.get('time_budget_extension_seconds', 0)
    if (isinstance(time_extension, bool) or not isinstance(time_extension, (int, float))
            or not np.isfinite(time_extension) or time_extension < 0
            or new['maximum_training_seconds'] != old['maximum_training_seconds']+time_extension):
        raise ValueError('continuation time cap requires an exact nonnegative declared extension')
    new['maximum_training_seconds'] = old['maximum_training_seconds']
    old_stages = old.pop('stages')
    new_stages = new.pop('stages')
    parent_steps = sum(stage['steps'] for stage in old_stages)
    milestone = declaration.get('from_registered_milestone', False)
    if not isinstance(milestone, bool):
        raise ValueError('registered milestone continuation must be an explicit boolean')
    if milestone:
        boundary = declaration['completed_steps']
        if (isinstance(boundary, bool) or not isinstance(boundary, int)
                or boundary not in old['quality_milestones']
                or not 0 < boundary < parent_steps):
            raise ValueError('a positive nonterminal registered milestone is required')
        # Preserve EVERY executed projection and stage-specific SR setting.
        # Only the unexecuted tail is replaced by the newly authorized work.
        executed, remaining = [], boundary
        for stage in old_stages:
            if not remaining:
                break
            prefix = dict(stage, steps=min(stage['steps'], remaining))
            executed.append(prefix)
            remaining -= prefix['steps']
        old_stages, parent_steps = executed, boundary
    extension = new_stages[len(old_stages):]
    if (old_stages[-1]['projection'] != 'gamma_a1'
            or declaration['completed_steps'] != parent_steps
            or new_stages[:len(old_stages)] != old_stages or not extension
            or any(stage != dict(projection='gamma_a1', steps=stage['steps'])
                   or stage['steps'] < 1 for stage in extension)
            or sum(stage['steps'] for stage in extension) != additional):
        raise ValueError('continuation must append only full-A1 steps to the completed parent')
    old.pop('quality_milestones'); new.pop('quality_milestones')
    old_audit_seeds = old['audit'].pop('seeds')
    new_audit_seeds = new['audit'].pop('seeds')
    if set(old_audit_seeds) & set(new_audit_seeds):
        raise ValueError('continuation requires fresh independent audit seeds')
    old['audit'].pop('bootstrap_seed', None)
    new['audit'].pop('bootstrap_seed', None)
    if old != new:
        raise ValueError('continuation changed model, physical/SR/sampling protocol or inherited budgets')
    validate_spec(spec)


def continuation_state(spec: dict, registration_sha256: str) -> dict:
    """Read a sealed immutable quality checkpoint without changing its parent."""
    declaration = spec['netket_continue_from']
    parent = ROOT/Path(declaration['directory'])
    relative = Path(declaration['checkpoint_path'])
    if relative.is_absolute() or '..' in relative.parts or relative != Path('quality')/f'step{declaration["completed_steps"]:05d}.pkl':
        raise ValueError('continuation must use the declared immutable quality checkpoint')
    checkpoint = parent/relative
    if not checkpoint.resolve().is_relative_to(parent.resolve()):
        raise ValueError('continuation checkpoint escapes the parent directory')
    registration = read_sealed(parent/'registration.json')
    progress = read_sealed(parent/'progress.json')
    validate_continuation_protocol(registration['config'], spec)
    for label, actual in (('checkpoint_sha256', file_sha256(checkpoint)),
                          ('registration_sha256', registration['result_sha256']),
                          ('progress_sha256', progress['result_sha256'])):
        if declaration[label] != actual:
            raise ValueError('continuation parent seal changed: '+label)
    for relative_source, digest in registration['source_manifest'].items():
        if file_sha256(parent/'source_snapshots'/relative_source) != digest:
            raise ValueError('continuation parent frozen runtime changed')
    if declaration.get('from_registered_milestone', False):
        if progress.get('status') != 'paused_session_budget':
            raise ValueError('milestone parent must be safely paused; no concurrent writer')
        audit = read_sealed(parent/'audits'/f'step{declaration["completed_steps"]:05d}'/'results.json')
        if (declaration.get('audit_sha256') != audit['result_sha256']
                or audit.get('step') != declaration['completed_steps']
                or audit.get('registration_sha256') != registration['result_sha256']
                or audit.get('checkpoint_sha256') != declaration['checkpoint_sha256']
                or audit.get('params_sha256') != progress['final_params_sha256']
                or audit.get('training_projection') != 'gamma_a1'
                or audit.get('audit_projection') != 'gamma_a1'):
            raise ValueError('milestone continuation requires its completed frozen full-A1 audit')
    saved = load_checkpoint(checkpoint)
    meta = saved['metadata']
    steps = declaration['completed_steps']
    if (meta['completed_steps'] != steps or saved['optimizer_state']['step'] != steps
            or len(meta['history']) != steps or progress['completed_steps'] != steps
            or meta['registration_sha256'] != registration['result_sha256']
            or pytree_sha256(saved['params']['model']) != progress['final_params_sha256']):
        raise ValueError('continuation parent parameters/count/optimizer do not match the seal')
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(saved['params']['model'])):
        raise ValueError('nonfinite continuation parent parameters')
    saved = copy.deepcopy(saved)
    saved['metadata']['registration_sha256'] = registration_sha256
    saved['metadata']['continuation_parent'] = dict(**declaration,
        parameters_reinitialized=False, sampler_reinitialized=False,
        physics_and_SR_unchanged=True, inherited_work_and_time_counters=True,
        audit_seeds_changed_before_training=True,
        parent_training_time_cap_seconds=registration['config']['maximum_training_seconds'],
        continuation_training_time_cap_seconds=spec['maximum_training_seconds'])
    return saved


def full_update(driver, *, spec: dict | None = None) -> dict:
    """One complete production update; never install nonfinite parameters.

    Same reset/compute/update/count order as the pinned driver's run loop.
    No independent audit or energy-selected backtracking enters this update.
    """
    driver.reset_step()
    # A production-size sharded sampler must preserve the actual Hilbert
    # support, not only shapes/finite scores. Validate before ANY H action.
    samples = np.asarray(driver.state.samples)
    if not np.isin(samples, [-1, 1]).all() or not (samples.sum(axis=-1) == 0).all():
        raise FloatingPointError('sampler left spin-1/2 Sz=0 support; no H action/update installed')
    stats, direction = driver.compute_loss_and_update()
    jax.block_until_ready(direction)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(direction)):
        raise FloatingPointError('nonfinite SR direction; last checkpoint retained')
    energy, variance = complex(stats.mean), float(stats.variance)
    if not np.isfinite([energy.real, energy.imag, variance]).all():
        raise FloatingPointError('nonfinite training moments; last checkpoint retained')
    # Capture layout BEFORE parameter installation invalidates MCState's
    # sample cache. Accessing .samples afterwards would silently draw a
    # second batch, advance RNG and change the registered trajectory.
    sampled = driver.state.samples
    sample_layout = dict(sample_sharding=str(sampled.sharding),
        unique_local_sample_partitions=len({str(shard.index) for shard in sampled.addressable_shards}))
    end_spins = np.asarray(driver.state.sampler_state.σ)
    if not np.isin(end_spins, [-1, 1]).all() or not (end_spins.sum(axis=-1) == 0).all():
        raise FloatingPointError('sampler storage corrupted during H/SR; update not installed')
    direction, trust_diagnostics = bounded_physical_direction(driver, direction, sampled, spec)
    driver.update_parameters(direction)
    jax.block_until_ready(driver.state.parameters)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(driver.state.parameters)):
        raise FloatingPointError('nonfinite proposal; last checkpoint retained')
    driver._step_count += 1
    return dict(energy=energy.real, imaginary=energy.imag, variance=variance,
                error_of_mean=float(stats.error_of_mean), **sample_layout, **trust_diagnostics)


def train(config_path: Path, output: Path, *, until_step: int | None = None,
          session_seconds: float = 126000., resume: bool = False) -> dict:
    """Bounded complete-network run; stage schedule/RNG/total budgets survive resume."""
    if importlib.metadata.version('netket') != NETKET_VERSION:
        raise ValueError('pinned environment required')
    spec, reg = register(config_path, output, additional_entry_paths=(Path(__file__),))
    config = validate_spec(spec)
    if spec.get('backend') != 'netket_ntk' or spec.get('model_family') != 'literature':
        raise ValueError('explicit author-aligned optional-backend config required')
    if spec.get('require_gpu', False) and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU allocation required; do not silently consume login CPU')
    maximum = sum(s['steps'] for s in spec['stages'])
    limit = maximum if until_step is None else until_step
    if not isinstance(limit, int) or not 0 <= limit <= maximum or session_seconds <= 0:
        raise ValueError('invalid fixed training/session endpoint')
    if (output/'training.json').exists():
        return read_sealed(output/'training.json')
    checkpoint = output/'checkpoint.pkl'
    if checkpoint.exists():
        if not resume:
            raise FileExistsError('use explicit --resume')
        saved = load_checkpoint(checkpoint); meta = saved['metadata']
        if (meta['registration_sha256'] != reg['result_sha256'] or
                meta['completed_steps'] != saved['optimizer_state']['step']):
            raise ValueError('checkpoint registration/count mismatch')
        params = saved['params']['model']
    elif spec.get('netket_resume_from') is not None or spec.get('netket_continue_from') is not None:
        parent_loader = continuation_state if spec.get('netket_continue_from') is not None else recovery_state
        saved = parent_loader(spec, reg['result_sha256'])
        params = saved['params']['model']
        meta = saved['metadata']
        save_checkpoint(output/'quality'/f'step{meta["completed_steps"]:05d}.pkl',
                        saved['params'], optimizer_state=saved['optimizer_state'], metadata=meta)
    else:
        params, parent = initial_model(spec); saved = None
        meta = dict(registration_sha256=reg['result_sha256'], completed_steps=0, history=[],
            active_training_seconds=0., compile_seconds=0., current_stage=-1,
            model_config=asdict(config), parent=parent, backend='netket_ntk',
            training_full_H_calls=0, warmup_full_H_calls=0, status='running',
            initial_params_sha256=pytree_sha256(params))
        _snapshot(output, dict(model=params), meta)
    requested = [False]
    def stop(*_): requested[0] = True
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    driver, current = None, None; started = perf_counter()
    def save(status='running'):
        state, optimizer = driver_payload(driver)
        meta['status'] = status
        save_checkpoint(checkpoint, state, optimizer_state=optimizer, metadata=meta)
        sealed_json(output/'progress.json', dict(**meta, parameter_count=fixed_sz_spatial_vit_parameter_count(state['model']),
            final_params_sha256=pytree_sha256(state['model']), checkpoint_sha256=file_sha256(checkpoint),
            netket_version=NETKET_VERSION, runtime=numerical_runtime_identity()))
    try:
        while meta['completed_steps'] < limit:
            if requested[0] or perf_counter()-started >= session_seconds or meta['active_training_seconds'] >= spec['maximum_training_seconds']:
                break
            index, projection = stage_at(spec, meta['completed_steps'])
            if index != current:
                previous = saved if driver is None else dict(zip(('params', 'optimizer_state'), driver_payload(driver)))
                driver, h, _, images = make_driver(spec, params, projection, step=meta['completed_steps'])
                if previous is not None:
                    restore_driver(driver, previous)
                current = index; meta['current_stage'] = index
                save()
            tick = perf_counter()
            moments = full_update(driver, spec=spec)
            seconds = perf_counter()-tick
            # First update of each process/stage includes compilation; record
            # actual wall, not a claimed accelerator-only/compile-free speed.
            params = driver.state.parameters
            meta['completed_steps'] = int(driver.step_count)
            meta['active_training_seconds'] += seconds
            meta['recovery_gpu_device_seconds'] = meta.get('recovery_gpu_device_seconds', 0.)+seconds*jax.device_count()
            records = spec['sampling']['chains']*spec['sampling']['snapshots_per_update']
            meta['training_full_H_calls'] += records
            memory = [device.memory_stats() or {} for device in jax.local_devices()]
            row = dict(step=meta['completed_steps'], stage=index, projection=projection,
                images=images, seconds_including_compile=seconds,
                energy_before_update_per_site=moments['energy']/config.n_sites,
                imaginary_energy_before_update_per_site=moments['imaginary']/config.n_sites,
                variance_before_update_per_site=moments['variance']/config.n_sites,
                gpu_peak_bytes=max(int(m.get('peak_bytes_in_use', 0)) for m in memory),
                local_gpu_peak_bytes=[int(m.get('peak_bytes_in_use', 0)) for m in memory],
                device_count=jax.device_count(),
                sample_sharding=moments['sample_sharding'],
                unique_local_sample_partitions=moments['unique_local_sample_partitions'],
                samples=records, full_H_calls=records)
            row.update({k: v for k, v in moments.items() if k in
                ('sampled_fs_sine', 'reweighting_ess_fraction', 'update_scale',
                 'trust_evaluations', 'update_accepted', 'last_rejected_fs_sine')})
            meta['history'].append(row)
            save()
            if meta['completed_steps'] in spec['quality_milestones']:
                state, optimizer = driver_payload(driver)
                quality = output/'quality'/f'step{meta["completed_steps"]:05d}.pkl'
                if not quality.exists():
                    save_checkpoint(quality, state, optimizer_state=optimizer, metadata=meta)
            if row['step'] == 1 or row['step'] % 20 == 0:
                print(json.dumps(row), flush=True)
        if driver is None:
            if limit == 0 and meta['completed_steps'] == 0 and saved is None:
                # Freeze the exact imported state for a target-H audit before
                # creating any sampler or optimizer. A later normal invocation
                # starts the registered trajectory from these same weights.
                return sealed_json(output/'progress.json', dict(meta,
                    status='paused_session_budget', initial_frozen_only=True,
                    parameter_count=fixed_sz_spatial_vit_parameter_count(params),
                    final_params_sha256=pytree_sha256(params),
                    netket_version=NETKET_VERSION, runtime=numerical_runtime_identity()))
            raise ValueError('no update session started')
        status = ('fixed_steps_complete' if meta['completed_steps'] == maximum else
                  'total_time_cap' if meta['active_training_seconds'] >= spec['maximum_training_seconds'] else 'paused_session_budget')
        save(status)
        if status != 'paused_session_budget':
            state, optimizer = driver_payload(driver)
            quality = output/'quality'/f'step{meta["completed_steps"]:05d}.pkl'
            if not quality.exists():
                save_checkpoint(quality, state, optimizer_state=optimizer, metadata=meta)
            return sealed_json(output/'training.json', dict(**meta,
                parameter_count=fixed_sz_spatial_vit_parameter_count(params),
                final_params_sha256=pytree_sha256(params), runtime=numerical_runtime_identity(),
                strict_su2=False, converged=False, variational_win_claimed=False))
        return read_sealed(output/'progress.json')
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--until-step', type=int)
    parser.add_argument('--session-seconds', type=float, default=126000.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--require-endpoint', action='store_true',
        help='safe pause exits2 so a dependent task cannot treat it as complete')
    parser.add_argument('--require-sample-partitions', type=int, default=1,
        help='fail closed unless the completed production batch actually spans this many GPUs')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    result = train(args.config, ROOT/spec['output'], until_step=args.until_step,
                   session_seconds=args.session_seconds, resume=args.resume)
    print(json.dumps(dict(status=result['status'], completed_steps=result['completed_steps'])), flush=True)
    expected = args.until_step if args.until_step is not None else sum(s['steps'] for s in spec['stages'])
    if args.require_endpoint and result['completed_steps'] != expected:
        raise SystemExit(2)
    if not result['history']:
        if result['completed_steps'] != 0 or not result.get('initial_frozen_only'):
            raise RuntimeError('missing optimization history')
        raise SystemExit(0)
    last = result['history'][-1]
    if last.get('unique_local_sample_partitions', 1) < args.require_sample_partitions:
        raise RuntimeError('visible GPUs did not actually partition the production samples')
    if args.require_sample_partitions > 1:
        if last['gpu_peak_bytes'] >= 75*1024**3:
            raise RuntimeError('insufficient multi-GPU memory margin; checkpoint preserved')
        print('verified native sample sharding:', json.dumps({k:last[k] for k in
            ('samples','full_H_calls','device_count','sample_sharding',
             'unique_local_sample_partitions','local_gpu_peak_bytes','seconds_including_compile')}),flush=True)
