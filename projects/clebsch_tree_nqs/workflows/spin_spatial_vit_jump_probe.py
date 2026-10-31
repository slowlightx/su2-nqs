"""Bounded, no-production-output diagnosis of a restored full Spatial-ViT.

Compare identical physical configurations across independent H-action kernels,
and compare the trust-region candidate with the actual installed SGD update.
All steps and extra H calls are diagnostic; no parent checkpoint is modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from .spin import _bonds, _groups
from .spin_spatial_vit_quality import initial_model, model_log, stage_at, validate_spec
from .spin_spatial_vit_netket import (
    make_driver, restore_driver, bounded_physical_direction, full_update, driver_payload,
)


def max_tree_error(first: object, second: object) -> float:
    """Maximum absolute leaf discrepancy; reject structural/shape mismatch."""
    if jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(second):
        raise ValueError('parameter structure changed during installation')
    error = 0.
    for left, right in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape:
            raise ValueError('parameter shape changed during installation')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError('nonfinite parameter installation')
        error = max(error, float(np.max(np.abs(a-b))))
    return error


def single_device_evaluate(function: Callable, params: object, bits: np.ndarray,
                           device: jax.Device) -> np.ndarray:
    """Evaluate the independent oracle outside NetKet's active global mesh.

    Explicit device_put alone is insufficient: JAX 0.11 also checks the
    thread-local context mesh installed by NetKet. The context manager restores
    that production mesh after the independent, host-copied evaluation.
    """
    single_mesh = jax.sharding.Mesh(np.asarray([device]), ('oracle',))
    single_sharding = jax.sharding.SingleDeviceSharding(device)
    with jax.set_mesh(single_mesh):
        local_params = jax.tree_util.tree_map(
            lambda value: jax.device_put(np.array(value, copy=True), single_sharding), params)
        local_bits = jax.device_put(np.array(bits, copy=True), single_sharding)
        return np.asarray(function(local_params, local_bits))


def replica_diagnostics(tree: object) -> dict:
    """Compare actual buffers sharing an index, not just the global array view.

    A declared replicated JAX array can have divergent device buffers while
    ``np.asarray(array)`` only observes its first replica. Never use this
    inspection to silently choose a training gradient or repair its values.
    """
    leaves = []
    maximum = 0.
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(value, jax.Array):
            continue
        seen = {}
        comparisons = []
        layout = []
        for shard in value.addressable_shards:
            index = str(shard.index)
            data = np.asarray(shard.data)
            layout.append(dict(device=str(shard.device), index=index, shape=list(data.shape)))
            if not np.isfinite(data).all():
                raise FloatingPointError('nonfinite actual device buffer')
            if index in seen:
                first_device, reference = seen[index]
                error = float(np.max(np.abs(data-reference)))
                maximum = max(maximum, error)
                comparisons.append(dict(first_device=first_device,
                    other_device=str(shard.device), maximum_absolute_error=error))
            else:
                seen[index] = (str(shard.device), data)
        leaves.append(dict(path=jax.tree_util.keystr(path),
            declared_sharding=str(value.sharding), addressable_shards=layout,
            same_index_comparisons=comparisons))
    return dict(maximum_replica_error=maximum, leaves=leaves)


def replicate_host_candidate(candidate: object) -> object:
    """Explicitly replicate an already host-defined *diagnostic* candidate.

    This does not establish that its first-replica SR direction was correct.
    It is an optional intervention isolating buffer/layout failure only.
    """
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('S',))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(np.array(value, copy=True), replicated), candidate)


def _numeric_payload(tree: object) -> object:
    """Expose typed PRNG keys as their exact key data for checkpoint comparison."""
    def convert(value):
        if isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            return np.asarray(jax.random.key_data(value))
        return value
    return jax.tree_util.tree_map(convert, tree)


def resume_replay(config_path: Path, checkpoint: Path, expected_checkpoint: Path,
                  output: Path, *, maximum_seconds: float = 900.) -> dict:
    """Replay two real production updates and compare the sealed complete endpoint.

    The existing production reset/SR/update order is reused without the oracle
    calls of the instrumented jump probe. No checkpoint or audit is overwritten.
    This compares parameters, full sampler/RNG and optimizer state, not weights
    alone. Only the explicitly validated single-device protocol is admitted.
    """
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(config_path.read_text())
    validate_spec(spec)
    if spec['netket'].get('expected_device_count') != 1 or jax.device_count() != 1:
        raise ValueError('checkpoint replay requires the registered single-device protocol')
    if maximum_seconds <= 0 or (spec.get('require_gpu') and jax.default_backend() != 'gpu'):
        raise ValueError('positive bounded GPU replay budget required')
    start, expected = load_checkpoint(checkpoint), load_checkpoint(expected_checkpoint)
    step = int(start['optimizer_state']['step'])
    if int(expected['optimizer_state']['step']) != step+2:
        raise ValueError('exactly two production updates must separate replay checkpoints')
    _, projection = stage_at(spec, step)
    driver, _, _, _ = make_driver(spec, start['params']['model'], projection, step=step)
    restore_driver(driver, start)
    started = perf_counter()
    rows = []
    for _ in range(2):
        if perf_counter()-started >= maximum_seconds:
            raise TimeoutError('bounded checkpoint replay exceeded its budget')
        rows.append(full_update(driver, spec=spec))
    state, optimizer = driver_payload(driver)
    errors = dict(
        parameters=max_tree_error(state['model'], expected['params']['model']),
        sampler_and_RNG=max_tree_error(_numeric_payload(state['netket_sampler_state']),
                                      _numeric_payload(expected['params']['netket_sampler_state'])),
        optimizer=max_tree_error(_numeric_payload({k:v for k,v in optimizer.items() if k!='netket_version'}),
                                 _numeric_payload({k:v for k,v in expected['optimizer_state'].items() if k!='netket_version'})),
    )
    # Floating log probabilities and SR reductions may differ across separately
    # compiled GPU processes. Separate exact stochastic state from roundoff;
    # do not call a floating log_prob difference an RNG/configuration change.
    same_configs=np.array_equal(state['netket_sampler_state'].σ,expected['params']['netket_sampler_state'].σ)
    same_rng=np.array_equal(jax.random.key_data(state['netket_sampler_state'].rng),
                           jax.random.key_data(expected['params']['netket_sampler_state'].rng))
    config=validate_spec(spec);group=_groups(config.geometry,config.rows,config.columns)[projection]
    log_fn,_=model_log(spec,config,group)
    bits=((np.asarray(start['params']['netket_sampler_state'].σ)+1)/2).astype(np.int32)
    logs=jax.vmap(log_fn,in_axes=(None,0))
    discrepancy=np.asarray(logs(state['model'],bits)-logs(expected['params']['model'],bits))
    physical_log_rms=float(np.sqrt(np.mean(abs(discrepancy-discrepancy.mean())**2)))
    optimizer_relative=max(float(np.max(abs(np.asarray(a)-np.asarray(b))/(1+abs(np.asarray(b)))))
        for a,b in zip(jax.tree_util.tree_leaves(_numeric_payload({k:v for k,v in optimizer.items() if k!='netket_version'})),
                       jax.tree_util.tree_leaves(_numeric_payload({k:v for k,v in expected['optimizer_state'].items() if k!='netket_version'}))))
    strict_v1=errors['parameters'] <= 1e-11 and errors['sampler_and_RNG'] == 0 and errors['optimizer'] <= 1e-9
    passed=(same_configs and same_rng and errors['parameters'] <= 1e-9
        and errors['sampler_and_RNG'] <= 1e-7 and optimizer_relative <= 1e-8
        and physical_log_rms <= 1e-7)
    result = dict(status='passed' if passed else 'replay_mismatch',
        config_sha256=file_sha256(config_path), start_checkpoint_sha256=file_sha256(checkpoint),
        expected_checkpoint_sha256=file_sha256(expected_checkpoint),
        start_step=step, completed_step=int(driver.step_count), maximum_absolute_errors=errors,
        actual_production_updates=rows, full_H_calls=2*spec['sampling']['chains']*spec['sampling']['snapshots_per_update'],
        exact_configurations=same_configs,exact_RNG=same_rng,physical_log_ratio_centered_RMS=physical_log_rms,
        optimizer_relative_max_error=optimizer_relative,
        bitwise_auxiliary_v1_passed=strict_v1,
        replay_validation='v2: exact configurations/RNG; parameters1e-9, floating sampler auxiliaries1e-7, elementwise mixed-relative optimizer1e-8, centered physical log-ratio RMS1e-7. Original v1 failure records retained.',
        seconds=perf_counter()-started, production=False, source_sha256=file_sha256(Path(__file__)))
    atomic_json(output,result)
    if not passed:
        raise RuntimeError('full checkpoint replay mismatch; continuation remains gated')
    return result


def directional_statistics(scores: np.ndarray, energies: np.ndarray,
                           chain_ids: np.ndarray, n_sites: int) -> dict:
    """Energy tangent and FS speed, including whole-chain sampling uncertainty.

    Scores are d log(psi)/dt for a REAL parameter direction; the derivative is
    2 Re Cov(score,E_local). This does NOT predict finite-step energy or the
    derivative of the Hamiltonian variance. Uniform amplitude/phase gauges
    drop out. Every delete-chain sample recenters BOTH factors.
    """
    scores, energies, chain_ids = map(np.asarray, (scores, energies, chain_ids))
    if (scores.ndim != 1 or scores.shape != energies.shape or scores.shape != chain_ids.shape
            or not np.isfinite(scores).all() or not np.isfinite(energies).all()
            or isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 1):
        raise ValueError('finite paired one-dimensional scores, energies and chain labels required')
    groups = np.unique(chain_ids)
    if len(groups) < 2:
        raise ValueError('at least two independent chains required')
    def point(mask):
        v, e = scores[mask], energies[mask]
        v, e = v-v.mean(), e-e.mean()
        speed = float(np.sqrt(np.mean(abs(v)**2)))
        slope = float(2*np.mean(v.conj()*e).real/n_sites)
        return dict(energy_slope_per_site=slope, FS_speed=speed,
            energy_slope_per_site_per_FS=None if speed == 0 else slope/speed)
    measured = point(np.ones(len(scores), bool))
    deleted = [point(chain_ids != g) for g in groups]
    errors = {}
    for key in measured:
        if measured[key] is not None and all(x[key] is not None for x in deleted):
            values = np.array([x[key] for x in deleted])
            errors[key] = float(np.sqrt((len(groups)-1)/len(groups)*np.sum((values-values.mean())**2)))
    return dict(point=measured, whole_chain_SE=errors, independent_chains=len(groups),
                samples=len(scores), variance_derivative_measured=False)


def optimizer_conditions(config_path: Path, checkpoint: Path, audit_path: Path,
                         output: Path, *, maximum_seconds: float = 2700.) -> dict:
    """Three damping directions at ONE frozen parent and matched physical radius.

    A single fresh production-sized batch defines all three SR solves; whole
    chains from the already frozen parent audit assess tangent generalization.
    Those audit records become DEVELOPMENT data, not an untouched child audit.
    No parameter update, checkpoint or production state is installed. This can
    find residual descent but cannot certify stationarity of all directions.
    """
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(config_path.read_text()); config = validate_spec(spec)
    if spec['netket'].get('expected_device_count') != 1 or jax.device_count() != 1:
        raise ValueError('optimizer comparison requires the verified one-GPU path')
    saved = load_checkpoint(checkpoint); params = saved['params']['model']
    step = int(saved['optimizer_state']['step']); _, projection = stage_at(spec, step)
    if projection != 'gamma_a1':
        raise ValueError('unchanged full A1 parent required')
    audit = json.loads(audit_path.read_text())
    if (audit['checkpoint_sha256'] != file_sha256(checkpoint)
            or audit['params_sha256'] != pytree_sha256(params) or audit['step'] != step):
        raise ValueError('audit and fixed parent differ')
    record = dict(status='registered', config_sha256=file_sha256(config_path),
        checkpoint_sha256=file_sha256(checkpoint), audit_sha256=file_sha256(audit_path),
        step=step, damping_factors=[.1,1.,10.], target_FS_sine=.005,
        actual_Fs_matching_tolerance_fraction=.02, maximum_seconds=maximum_seconds,
        audit_used_for_development=True, new_state_generated=False,
        stationarity_certified=False, full_H_calls=0, rows=[], source_sha256=file_sha256(Path(__file__)))
    atomic_json(output,record); started=perf_counter()
    def budget():
        if perf_counter()-started > maximum_seconds:
            raise TimeoutError('fixed optimizer-condition budget reached; partial diagnostic retained')
    # Fixed eight evenly spaced snapshots per stream, ALL original chains.
    hold_spins=[]; hold_e=[]; hold_ids=[]; hashes={}; offset=0
    for stream in audit['streams']:
        path=audit_path.parent/f"stream{stream['seed']}.npz"
        if file_sha256(path)!=stream['raw_sha256']:
            raise ValueError('fixed parent audit stream changed')
        hashes[str(path)]=stream['raw_sha256']
        with np.load(path) as z:
            indices=np.arange(0,len(z['states']),16)
            x=z['states'][indices]; e=z['energies'][indices]
            hold_spins.append((2*x-1).reshape(-1,config.n_sites))
            hold_e.append(e.reshape(-1))
            hold_ids.append(np.broadcast_to(np.arange(x.shape[1])+offset,x.shape[:2]).reshape(-1))
            offset+=x.shape[1]
    hold_spins=jnp.asarray(np.concatenate(hold_spins)); hold_e=np.concatenate(hold_e); hold_ids=np.concatenate(hold_ids)
    reference_params_sha=pytree_sha256(params); sampled=None; train_energy=None
    from netket.jax import apply_chunked
    for factor in record['damping_factors']:
        budget(); current=deepcopy(spec)
        index,_=stage_at(current,step)
        solver=dict(current['sr'],**current['stages'][index].get('sr',{}))
        solver['damping']*=factor; current['stages'][index]['sr']=solver
        driver,_,_,_=make_driver(current,params,projection,step=step)
        restore_driver(driver,saved); driver.reset_step()
        if sampled is not None:
            driver.state._samples=sampled  # Same legal batch, not a second sampler draw.
        else:
            sampled=driver.state.samples
        if not np.isin(np.asarray(sampled),[-1,1]).all() or not (np.asarray(sampled).sum(-1)==0).all():
            raise ValueError('fixed batch left spin-half Sz=0 support')
        stats,direction=driver.compute_loss_and_update();jax.block_until_ready(direction)
        record['full_H_calls']+=int(np.prod(sampled.shape[:-1]))
        if pytree_sha256(driver.state.parameters)!=reference_params_sha:
            raise RuntimeError('fixed-parent solve changed parameters')
        energies=np.asarray(driver.state.local_estimators(driver._ham).data).reshape(-1)
        record['full_H_calls']+=int(np.prod(sampled.shape[:-1]))
        if train_energy is not None:
            np.testing.assert_allclose(energies,train_energy,atol=1e-11,rtol=0)
        train_energy=energies
        flat=sampled.reshape(-1,config.n_sites)
        apply=apply_chunked(driver.state._apply_fun,in_axes=(None,0),chunk_size=driver.state.chunk_size)
        log=lambda p,x:apply({'params':p},x)
        negative=jax.tree_util.tree_map(lambda d:-d,direction)
        train_scores=np.asarray(jax.jvp(lambda p:log(p,flat),(params,),(negative,))[1])
        train_ids=np.repeat(np.arange(sampled.shape[0]),sampled.shape[1])
        train=directional_statistics(train_scores,energies,train_ids,config.n_sites)
        speed=train['point']['FS_speed']
        if not np.isfinite(speed) or speed<=0:
            raise ValueError('no finite nonzero SR tangent')
        scale=record['target_FS_sine']/speed; before=log(params,flat)
        for _ in range(4):
            budget()
            candidate=jax.tree_util.tree_map(lambda p,d:p+scale*d,params,negative)
            ratios=log(candidate,flat)-before
            distance={k:float(v) for k,v in sampled_overlap_diagnostics(ratios).items()}
            if not np.isfinite(np.asarray(ratios)).all() or distance['sampled_fs_sine']<=0:
                raise ValueError('nonfinite/zero finite physical update')
            if abs(distance['sampled_fs_sine']/record['target_FS_sine']-1)<.02:break
            scale*=record['target_FS_sine']/distance['sampled_fs_sine']
        hold_scores=np.asarray(jax.jvp(lambda p:log(p,hold_spins),(params,),(negative,))[1])
        development=directional_statistics(hold_scores,hold_e,hold_ids,config.n_sites)
        hold_ratios=log(candidate,hold_spins)-log(params,hold_spins)
        hold_distance={k:float(v) for k,v in sampled_overlap_diagnostics(hold_ratios).items()}
        record['rows'].append(dict(damping=solver['damping'],factor=factor,
            training_energy_per_site=float(complex(stats.mean).real/config.n_sites),
            train_tangent=train,development_tangent=development,
            matched_scale=scale,training_distance=distance,development_distance=hold_distance,
            radius_matched=abs(distance['sampled_fs_sine']/record['target_FS_sine']-1)<.02))
        record.update(status='partial',seconds=perf_counter()-started,audit_streams_sha256=hashes)
        atomic_json(output,record)
    record.update(status='complete',seconds=perf_counter()-started)
    atomic_json(output,record)
    return record


def run(config_path: Path, checkpoint: Path | None, output: Path, *, steps: int = 2,
        oracle_rows: int = 8, maximum_seconds: float = 1500.,
        freeze_parameters: bool = False, replicate_candidate: bool = False) -> dict:
    """Run at most two instrumented updates from immutable weights/full state."""
    if steps not in (0, 1, 2) or not 1 <= oracle_rows <= 32 or maximum_seconds <= 0:
        raise ValueError('bounded diagnostic needs 0–2 steps and 1–32 oracle rows')
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(config_path.read_text()); config = validate_spec(spec)
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('this full-size diagnosis requires an allocated GPU')
    saved = None if checkpoint is None else load_checkpoint(checkpoint)
    params = initial_model(spec)[0] if saved is None else saved['params']['model']
    step = 0 if saved is None else int(saved['optimizer_state']['step'])
    _, projection = stage_at(spec, step)
    driver, h, _, images = make_driver(spec, params, projection, step=step)
    if saved is not None:
        restore_driver(driver, saved)
    single_device = jax.devices()[0]
    # This independent invocation is deliberately on ONE device. It does not
    # inherit the production sample-axis mesh or the NTK estimator layout.
    # Rebuild its closed-over image/edge arrays there as well as evaluating it
    # there; otherwise those constants can retain the production context mesh.
    with jax.set_mesh(jax.sharding.Mesh(np.asarray([single_device]), ('oracle',))):
        group = None if projection == 'none' else _groups(
            config.geometry, config.rows, config.columns)[projection]
        oracle_log_fn, oracle_images = model_log(spec, config, group)
        if oracle_images != images:
            raise ValueError('independent oracle changed the physical projector')
        reference_h = make_full_heisenberg_ratios(oracle_log_fn,
            _bonds(config.geometry, config.rows, config.columns, spec['j2_over_j1']))
        reference_log = jax.jit(jax.vmap(oracle_log_fn, in_axes=(None, 0)))
    record = dict(config_sha256=file_sha256(config_path),
        checkpoint_sha256=None if checkpoint is None else file_sha256(checkpoint),
        initial_params_sha256=pytree_sha256(driver.state.parameters),
        checkpoint_step=step, projection=projection, images=images,
        devices=[str(d) for d in jax.devices()], rows=[], production=False,
        extra_full_H_calls=0, maximum_seconds=maximum_seconds,
        parameters_frozen=freeze_parameters, replicate_candidate=replicate_candidate,
        replica_repair_is_diagnostic_not_validated_SR=replicate_candidate,
        initial_parameter_replication=replica_diagnostics(driver.state.parameters))
    started = perf_counter()

    def persist():
        record['elapsed_seconds'] = perf_counter()-started
        atomic_json(output, record)
        print(json.dumps(record['rows'][-1] if record['rows'] else record), flush=True)

    for iteration in range(max(1, steps)):
        if perf_counter()-started >= maximum_seconds:
            record['status'] = 'bounded_time_cap'; persist(); return record
        driver.reset_step()
        replicas_before_sampling = replica_diagnostics(driver.state.parameters)
        sampled = driver.state.samples
        spins = np.asarray(sampled)
        if not np.isin(spins, [-1, 1]).all() or not (spins.sum(-1) == 0).all():
            raise FloatingPointError('sampler support violation before H action')
        flat_spins = spins.reshape((-1, config.n_sites))
        # Cover the whole flattened batch, including both GPU sample shards.
        indices = np.linspace(0, len(flat_spins)-1, oracle_rows, dtype=int)
        probe_spins = flat_spins[indices]
        bits = ((probe_spins+1)//2).astype(np.int32)
        old = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), driver.state.parameters)
        oracle_log = single_device_evaluate(reference_log, old, bits, single_device)
        native_log = np.asarray(driver.state.log_value(jnp.asarray(probe_spins)))
        # NetKet 3.22.3 returns LocalEstimators, not a NumPy/JAX array.
        # Its documented .data axes match MCState.samples' first two axes.
        estimator = driver.state.local_estimators(h)
        estimator_data = np.asarray(estimator.data)
        if estimator_data.shape != spins.shape[:-1]:
            raise ValueError('local-energy axes disagree with sampled configurations')
        local_e = estimator_data.reshape(-1)
        oracle_e = single_device_evaluate(reference_h, old, bits, single_device)
        record['extra_full_H_calls'] += len(flat_spins)+oracle_rows
        sigma = np.asarray(driver.state.sampler_state.σ)
        sigma_log = np.asarray(driver.state.log_value(jnp.asarray(sigma)))
        row = dict(step_before=int(driver.step_count),
            parameter_replicas_before_sampling=replicas_before_sampling,
            parameter_replicas_before_SR=replica_diagnostics(driver.state.parameters),
            samples_sha256=pytree_sha256(spins), params_sha256=pytree_sha256(old),
            measured_energy_per_site=float(local_e.real.mean()/config.n_sites),
            measured_variance_per_site=float(np.mean(abs(local_e-local_e.mean())**2)/config.n_sites),
            fixed_rows_native_oracle_log_error=float(np.max(abs(native_log-oracle_log))),
            fixed_rows_native_oracle_H_error=float(np.max(abs(local_e[indices]-oracle_e))),
            fixed_rows_native_H_real=local_e[indices].real.tolist(),
            fixed_rows_oracle_H_real=oracle_e.real.tolist(),
            sampler_stored_log_probability_error=float(np.max(abs(
                np.asarray(driver.state.sampler_state.log_prob)-2*sigma_log.real))),
            sampled_log_real_quantiles=np.quantile(sigma_log.real,[0,.25,.5,.75,1]).tolist())
        record['rows'].append(row); persist()
        if steps == 0:
            break
        if freeze_parameters:
            # Advance only the sampler/RNG, with precisely the same physical
            # wavefunction. This separates sampling/kernel drift from updates.
            row['parameters_unchanged_error'] = max_tree_error(old,driver.state.parameters)
            driver.state.reset()
            continue
        stats, direction = driver.compute_loss_and_update()
        jax.block_until_ready(direction)
        row['SR_direction_replication'] = replica_diagnostics(direction)
        row['SR_energy_per_site'] = float(complex(stats.mean).real/config.n_sites)
        row['SR_variance_per_site'] = float(stats.variance/config.n_sites)
        row['parameters_mutated_by_SR_error'] = max_tree_error(old, driver.state.parameters)
        direction, trust = bounded_physical_direction(driver, direction, sampled, spec)
        row['trust_direction_replication'] = replica_diagnostics(direction)
        row['trust'] = trust
        index, _ = stage_at(spec, int(driver.step_count))
        solver = dict(spec['sr'], **spec['stages'][index].get('sr', {}))
        expected = jax.tree_util.tree_map(lambda p,d: p-solver['step_size']*.5*np.asarray(d), old, direction)
        candidate_log = single_device_evaluate(reference_log, expected, bits, single_device)
        driver.update_parameters(direction)
        jax.block_until_ready(driver.state.parameters)
        row['raw_installed_parameter_replication'] = replica_diagnostics(driver.state.parameters)
        if replicate_candidate:
            driver.state.parameters = replicate_host_candidate(expected)
            jax.block_until_ready(driver.state.parameters)
        row['final_installed_parameter_replication'] = replica_diagnostics(driver.state.parameters)
        row['candidate_installed_parameter_error'] = max_tree_error(expected, driver.state.parameters)
        installed_log = np.asarray(driver.state.log_value(jnp.asarray(probe_spins)))
        row['candidate_installed_fixed_log_error'] = float(np.max(abs(candidate_log-installed_log)))
        row['fixed_rows_actual_overlap'] = {k: float(v) for k,v in sampled_overlap_diagnostics(jnp.asarray(installed_log-oracle_log)).items()}
        row['actual_parameter_max_change'] = max_tree_error(old, driver.state.parameters)
        driver._step_count += 1
        persist()
    record['status'] = 'diagnostic_complete'; persist()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--oracle-rows', type=int, default=8)
    parser.add_argument('--maximum-seconds', type=float, default=1500.)
    parser.add_argument('--freeze-parameters', action='store_true')
    parser.add_argument('--replicate-candidate', action='store_true',
        help='diagnostic only: explicitly replicate the host expected candidate after installation')
    parser.add_argument('--compare-checkpoint', type=Path,
        help='replay exactly two existing production updates and compare the full saved endpoint')
    parser.add_argument('--optimizer-audit',type=Path,
        help='bounded three-damping fixed-parent SR/tangent comparison using existing audit records')
    args = parser.parse_args()
    if args.optimizer_audit is not None:
        if args.checkpoint is None or args.compare_checkpoint is not None:
            parser.error('optimizer comparison needs one fixed checkpoint, not a replay')
        optimizer_conditions(args.config,args.checkpoint,args.optimizer_audit,args.output,
                             maximum_seconds=args.maximum_seconds)
    elif args.compare_checkpoint is not None:
        if args.checkpoint is None or args.freeze_parameters or args.replicate_candidate:
            parser.error('checkpoint comparison needs a starting checkpoint and no layout/filter interventions')
        resume_replay(args.config,args.checkpoint,args.compare_checkpoint,args.output,
                      maximum_seconds=args.maximum_seconds)
    else:
        run(args.config,args.checkpoint,args.output,steps=args.steps,
            oracle_rows=args.oracle_rows,maximum_seconds=args.maximum_seconds,
            freeze_parameters=args.freeze_parameters, replicate_candidate=args.replicate_candidate)
