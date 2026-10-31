"""Bounded full-H SR of the existing finite-Schur neural/free parameterizations.

Single physical state convention, centered additive strong carrier, independent
paired endpoint audit. No Hamiltonian AD, architecture change, or new head.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.spatial_vit_sampling import make_log_exchange_sampler
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, init_finite_schur_network, neural_reduced_entries,
    neural_reduced_bank, free_reduced_bank, make_finite_schur_amplitude,
    centered_additive_amplitudes)
from su2_nqs.models.finite_schur_decoder import init_layer_standardized_free_tensors, shifted_neural_reduced_bank
from su2_nqs.models.schur_cutset_compact import make_full_a1_batch
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.cutset_training import paired_chain_audit, weighted_state_distance
from su2_nqs.vmc.dual_sampled_sr import make_dual_sampled_sr_update
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint
from su2_nqs.vmc.checkpoint import load_checkpoint
from .spin import _groups, _bonds
from .spin_decoder_cost import ROOT, physical_config, peak_rss_bytes
from .spin_schur_carrier_control import draw, batched

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_optimize.json'


def centered_products(carrier: jax.Array, hcarrier: jax.Array, residual: jax.Array,
                      hresidual: jax.Array, reference: jax.Array,
                      hreference: jax.Array, scale: float) -> tuple[jax.Array, jax.Array]:
    """Combine physical amplitudes and TOTAL H products, including residual nodes."""
    return carrier+scale*(residual-reference), hcarrier+scale*(hresidual-hreference)


def validate_spec(spec: dict) -> None:
    """Reject mismatched budgets/unsupported carriers before any computation."""
    if (spec['geometry'], spec['rows'], spec['columns']) != ('triangular', 6, 6):
        raise ValueError('this bounded stage uses the independently certified N36 carrier')
    if len(set(spec['seeds'].values())) != len(spec['seeds']):
        raise ValueError('scale/calibration/training/audit/uncertainty seeds must be distinct')
    for name in ('steps', 'chains', 'records_per_step', 'H_microbatch'):
        if isinstance(spec[name], bool) or not isinstance(spec[name], int) or spec[name] < 1:
            raise ValueError('budgets must be positive integers')
    for count in (spec['chains']*spec['records_per_step'],
                  spec['audit']['chains']*spec['audit']['records']):
        if count % spec['H_microbatch']:
            raise ValueError('complete microbatches required; no silent sample dropping')
    if spec['audit']['chains'] < 4:
        raise ValueError('paired chain audit requires at least four independent chains')
    initialization=spec.get('free_initialization')
    if initialization is not None:
        if (initialization != 'independent_uniform_layer_standardized'
                or 'initialization' not in spec['seeds']
                or isinstance(spec.get('initialization_calibration_rows'),bool)
                or not isinstance(spec.get('initialization_calibration_rows'),int)
                or spec['initialization_calibration_rows'] < 2):
            raise ValueError('explicit independent initializer, seed and probe budget required')
    if not isinstance(spec.get('neural_tensor_offset',False),bool):
        raise ValueError('neural tensor offset flag must be boolean')
    if spec.get('neural_tensor_offset',False) and initialization is None:
        raise ValueError('neural tensor offset requires the declared independent matrix initializer')


def registration(spec: dict) -> dict:
    """Seal numerical inputs and the complete fixed-endpoint protocol."""
    validate_spec(spec)
    parent = ROOT/spec['parent']
    names = ('results.json', 'network.json', 'network.npz', 'free_tensors.json', 'free_tensors.npz')
    record = dict(config=spec, inputs={str((parent/n).relative_to(ROOT)): file_sha256(parent/n) for n in names},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    if spec.get('carrier_checkpoint') is not None:
        path = ROOT/spec['carrier_checkpoint']['path']
        digest = file_sha256(path)
        if digest != spec['carrier_checkpoint']['sha256']:
            raise ValueError('strong carrier checkpoint changed')
        record['carrier_checkpoint_sha256'] = digest
    record['registration_sha256'] = canonical_sha256(record)
    return record


def setup(spec: dict):
    """Replay the identical initial bank; carrier and projection are unchanged."""
    n = spec['rows']*spec['columns']
    cfg = FiniteSchurDecoderConfig(replace(physical_config(spec['rows'], spec['columns'],
        spec['multiplicity_width']), maximum_intermediate_q=spec['maximum_q']),
        spec['rows']//2, spec['columns']//2, spec['embedding_dim'], spec['depth'])
    parent = ROOT/spec['parent']
    template = init_finite_schur_network(jax.random.PRNGKey(0), cfg)
    network = load_parameter_artifact(parent/'network.json', template)
    entries = neural_reduced_entries(network, cfg)
    canonical_free = {'entries': jnp.stack((entries.real, entries.imag), axis=1)}
    stored_free = load_parameter_artifact(parent/'free_tensors.json', canonical_free)
    # GPU fusion across a saved/reloaded eager/JIT boundary can change the
    # recomputed generator by a few ulps. Validate the saved numeric bank, then
    # canonicalize the FREE initializer to these very same regenerated entries.
    # Do not ask rounded artifact bytes to define two different starting states.
    difference = np.linalg.norm(np.asarray(stored_free['entries']-canonical_free['entries']))
    norm = np.linalg.norm(np.asarray(canonical_free['entries']))
    if not np.isfinite(difference) or difference > 2e-12*max(norm, 1e-300):
        raise ValueError('saved free bank differs from regenerated neural bank')
    free = canonical_free
    if pytree_sha256(neural_reduced_bank(network, cfg)) != pytree_sha256(free_reduced_bank(free, cfg)):
        raise ValueError('initial physical tensor banks must be identical')
    # A NEW, explicit experiment may test a literature-style independent W
    # initializer. Its residual bank differs from the neural one; both physical
    # states still start EXACTLY at C by centering against their OWN reference.
    if spec.get('free_initialization') is not None:
        key=jax.random.PRNGKey(spec['seeds']['initialization'])
        rows=initialize_spin_exchange_chains(jax.random.fold_in(key,1),
            n_chains=spec['initialization_calibration_rows'],n_sites=n,n_up=n//2)
        free,_=init_layer_standardized_free_tensors(key,cfg,np.asarray(rows))
    projector = PhysicalSpatialProjector(_groups(spec['geometry'], spec['rows'], spec['columns'])['gamma_a1'])
    raw = make_finite_schur_amplitude(cfg)
    projected = jax.jit(make_full_a1_batch(raw, projector, chunk_size=spec['projection_chunk']))
    bonds = _bonds(spec['geometry'], spec['rows'], spec['columns'], spec['j2_over_j1'])
    plan, evaluate, _ = make_projected_heisenberg_products(raw, projector, bonds, chunk_size=spec['H_chunk'])
    if spec.get('carrier_checkpoint') is None:
        orbit = build_n36_pi_zero_flux_orbit()
        carrier = jax.jit(lambda x: projected_pi_zero_flux_a1_amplitudes_jax(x, orbit=orbit))
    else:
        from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
        from .spin_adaptive_pair_carrier import _pair_dynamic_functions
        path = ROOT/spec['carrier_checkpoint']['path']
        if file_sha256(path) != spec['carrier_checkpoint']['sha256']:
            raise ValueError('strong singlet carrier checkpoint changed')
        saved = load_checkpoint(path)
        pairing = jnp.asarray(saved['params']['pairing_real'], dtype=jnp.float64)
        if pairing.shape != (n*(n-1)//2,) or not np.isfinite(np.asarray(pairing)).all():
            raise ValueError('invalid strong singlet pairing')
        agp = ProjectedSingletAGPConfig(n, tuple(tuple(map(int, row))
            for row in projector.site_permutations))
        fn = _pair_dynamic_functions(agp, amplitude_scale=1., tile_size=2)['amplitude']
        carrier = jax.jit(lambda x: fn(pairing, x))
    generators = dict(network=jax.jit(lambda p: neural_reduced_bank(p, cfg)),
                      free=jax.jit(lambda p: free_reduced_bank(p, cfg)))
    if spec.get('neural_tensor_offset',False):
        offset=free['entries'][:,0]+1j*free['entries'][:,1]
        generators['network']=jax.jit(lambda p:shifted_neural_reduced_bank(p,cfg,
            offset_entries=offset,reference_entries=entries))
    return n, dict(network=network, free=free), generators, projected, carrier, bonds, jax.jit(plan), jax.jit(evaluate)


def json_scalars(values: dict) -> dict:
    return {k: np.asarray(v).item() for k, v in values.items()}


def check_resources(spec: dict, start: float) -> None:
    if perf_counter()-start > spec['wall_limit_seconds_per_process']:
        raise TimeoutError('registered wall limit exceeded')
    if peak_rss_bytes() > spec['peak_rss_limit_gib']*2**30:
        raise MemoryError('registered RSS limit exceeded')


def proposal_spec(spec: dict, stage: str) -> dict:
    return {**spec[stage], 'uniform_floor_relative_to_pilot_mean': spec['uniform_floor_relative_to_pilot_mean'],
            'bridge_fraction': spec['bridge_fraction']}


def prepare(spec: dict, reg: dict, output: Path) -> dict:
    """Calibrate only normalization units, without energy-dependent selection."""
    if (output/'prepared.json').exists():
        raise FileExistsError('preparation already sealed')
    n, params, generators, projected, carrier, bonds, _, _ = setup(spec)
    states = initialize_spin_exchange_chains(jax.random.PRNGKey(spec['seeds']['scale']),
        n_chains=spec['scale_uniform_samples'], n_sites=n, n_up=n//2)
    c = batched(carrier, np.asarray(states), 32)
    cscale = float(np.sqrt(np.mean(abs(c)**2)))
    if not np.isfinite(cscale) or cscale <= 0:
        raise FloatingPointError('nonpositive carrier normalization unit')
    data, info = draw(proposal_spec(spec, 'calibration'), n, carrier, bonds, cscale,
                      spec['seeds']['calibration'], spec['calibration']['records'])
    reference = generators['network'](params['network'])
    phi = batched(lambda x: projected(reference, x), data['states'], 8)
    scale = float(np.sqrt(np.sum(abs(data['carrier'])**2/data['q'])/np.sum(abs(phi)**2/data['q'])))
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError('invalid residual normalization unit')
    np.savez_compressed(output/'calibration.npz', **data, reference_values=phi, scale_states=np.asarray(states))
    record = dict(registration_sha256=reg['registration_sha256'], cscale=cscale, residual_scale=scale,
        calibration_info=info, normalization_is_not_global_certification=True,
        calibration_energy_used=False, initial_bank_sha256=pytree_sha256(reference),
        calibration_sha256=file_sha256(output/'calibration.npz'))
    if spec.get('free_initialization') is not None:
        free_reference=generators['free'](params['free'])
        free_phi=batched(lambda x:projected(free_reference,x),data['states'],8)
        free_scale=float(np.sqrt(np.sum(abs(data['carrier'])**2/data['q'])/
                                   np.sum(abs(free_phi)**2/data['q'])))
        if not np.isfinite(free_scale) or free_scale <= 0:
            raise FloatingPointError('invalid independently initialized residual norm')
        record.update(residual_scale_by_arm=dict(network=scale,free=free_scale),
            reference_bank_sha256_by_arm=dict(network=pytree_sha256(reference),
                                             free=pytree_sha256(free_reference)),
            free_initialization=spec['free_initialization'],
            residual_banks_deliberately_different=not spec.get('neural_tensor_offset',False),
            common_initial_tensor_offset=spec.get('neural_tensor_offset',False),initial_physical_carrier_same=True)
    atomic_json(output/'prepared.json', record)
    print('normalization units sealed', cscale, scale, flush=True)
    return record


def full_products(bank, reference, states, carrier_action, plan, evaluate, *, scale: float,
                  microbatch: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bounded device-only residual work plus analytic carrier, no H/phi shortcut."""
    values, products, counts = [], [], []
    start = perf_counter()
    for offset in range(0, len(states), microbatch):
        x = jnp.asarray(states[offset:offset+microbatch], dtype=jnp.int32)
        prepared = plan(x)
        hp, phi, count, padded = evaluate(bank, prepared)
        hr, ref, _, _ = evaluate(reference, prepared)
        c, hc = carrier_action(x)
        psi, hpsi = centered_products(c, hc, phi, hp, ref, hr, scale)
        values.append(np.asarray(psi)); products.append(np.asarray(hpsi))
        counts.append((int(count), int(padded), int(prepared.inverse.size)))
    psi, hpsi = np.concatenate(values), np.concatenate(products)
    if not np.isfinite(psi).all() or not np.isfinite(hpsi).all():
        raise FloatingPointError('nonfinite full-H products; no filtering/clipping')
    return psi, hpsi, dict(seconds=perf_counter()-start, centers=len(states),
        residual_H_banks=2, unique_raw_queries_per_bank=sum(row[0] for row in counts),
        padded_raw_queries_per_bank=sum(row[1] for row in counts),
        unmerged_image_queries_per_bank=sum(row[2] for row in counts))


def train(spec: dict, reg: dict, output: Path, arm: str, *, until_step: int | None = None) -> dict:
    """Exactly the registered number of attempts; resume counters, never select endpoints."""
    folder = output/arm; folder.mkdir(exist_ok=True)
    limit = spec['steps'] if until_step is None else until_step
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= spec['steps']:
        raise ValueError('endpoint outside registered total step budget')
    if (folder/'endpoint.json').exists():
        raise FileExistsError('fixed endpoint already sealed')
    prepared = json.loads((output/'prepared.json').read_text())
    n, initial, generators, projected, raw_carrier, bonds, plan, evaluate = setup(spec)
    params = initial[arm]; generator = generators[arm]
    reference = generators['network'](initial['network'])
    if spec.get('free_initialization') is not None:
        reference=generators[arm](initial[arm])
    residual_scale=prepared.get('residual_scale_by_arm',{}).get(arm,prepared['residual_scale'])
    carrier = lambda x: raw_carrier(x)/prepared['cscale']
    amplitude = centered_additive_amplitudes(projected, carrier, reference, scale=residual_scale)
    log_bank = lambda bank, x: jnp.log(amplitude(bank, x[None])[0])
    log_params = lambda p, x: log_bank(generator(p), x)
    update = make_dual_sampled_sr_update(log_params, params, **spec['SR'])
    sampler = make_log_exchange_sampler(log_bank, spec['decorrelation_proposals_per_record'])
    burn = make_log_exchange_sampler(log_bank, spec['burn_proposals_per_chain'])
    carrier_work = make_amplitude_heisenberg_action(lambda _, x: carrier(x[None])[0], bonds, n_sites=n)
    carrier_action = lambda x: carrier_work({}, x)
    identity = dict(registration_sha256=reg['registration_sha256'], arm=arm,
                    prepared_sha256=file_sha256(output/'prepared.json'))
    checkpoint = folder/'checkpoint.pkl'; history = []; completed = 0
    seed = spec['seeds']['training']; key = jax.random.PRNGKey(seed)
    states = initialize_spin_exchange_chains(jax.random.fold_in(key, 1),
        n_chains=spec['chains'], n_sites=n, n_up=n//2)
    burnt = False
    if checkpoint.exists():
        saved = load_resumable_checkpoint(checkpoint, expected_run_identity=identity)
        params, history, completed = saved['params'], saved['history'], saved['completed_steps']
        key, states, burnt = (saved['optimizer_state'][k] for k in ('key', 'states', 'burnt'))
    def persist():
        save_resumable_checkpoint(checkpoint, params, optimizer_state=dict(key=key, states=states, burnt=burnt),
            completed_steps=len(history), next_sample_index=len(history)*spec['chains']*spec['records_per_step'],
            run_identity=identity, history=history)
    persist()
    for step in range(completed, limit):
        tick = perf_counter(); bank = generator(params)
        if not burnt:
            states, key, _, _ = burn(bank, key, states); jax.block_until_ready(states); burnt = True
        samples, acceptances, movements = [], [], []
        for _ in range(spec['records_per_step']):
            states, key, acceptance, movement = sampler(bank, key, states)
            samples.append(np.asarray(states)); acceptances.append(float(acceptance)); movements.append(float(movement))
        samples = np.concatenate(samples); sampling_seconds = perf_counter()-tick
        psi, hpsi, work = full_products(bank, reference, samples, carrier_action, plan, evaluate,
            scale=residual_scale, microbatch=spec['H_microbatch'])
        if np.any(psi == 0):
            raise FloatingPointError('Born training sample at exact node; fail closed')
        energy = hpsi/psi
        tick = perf_counter(); params, diagnostics = update(params, jnp.asarray(samples), jnp.asarray(energy))
        jax.block_until_ready(params); sr_seconds = perf_counter()-tick
        row = dict(step=step+1, energy_per_site=float(energy.real.mean()/n),
            imaginary_energy_per_site=float(energy.imag.mean()/n),
            variance_per_site=float(np.mean(abs(energy-energy.mean())**2)/n),
            acceptance=float(np.mean(acceptances)), movement=float(np.mean(movements)),
            sampling_seconds=sampling_seconds, full_H=work, SR_seconds=sr_seconds,
            diagnostics=json_scalars(diagnostics), parameter_sha256=pytree_sha256(params))
        history.append(row); persist()
        if peak_rss_bytes() > spec['peak_rss_limit_gib']*2**30:
            raise MemoryError('registered RSS limit exceeded; checkpoint retained')
        np.savez_compressed(folder/f'training_step_{step+1:02d}.npz', states=samples, psi=psi, hpsi=hpsi)
        atomic_json(folder/'progress.json', dict(arm=arm, completed_steps=len(history), history=history,
            peak_rss_bytes=peak_rss_bytes(), registration_sha256=reg['registration_sha256']))
        print(arm, step+1, 'E/N', row['energy_per_site'], 'var/N', row['variance_per_site'],
              'FS', row['diagnostics']['sampled_fs_sine'], 'seconds', sampling_seconds+work['seconds']+sr_seconds, flush=True)
    if limit != spec['steps']:
        return dict(arm=arm, completed_steps=len(history), fixed_terminal_steps=spec['steps'],
                    status='paused_registered_milestone', checkpoint_sha256=file_sha256(checkpoint))
    save_parameter_artifact(folder/'endpoint.json', params, identity=identity)
    result = dict(arm=arm, registration_sha256=reg['registration_sha256'], fixed_attempts=len(history),
        full_H_centers=sum(h['full_H']['centers'] for h in history), history=history,
        endpoint_sha256=file_sha256(folder/'endpoint.json'), peak_rss_bytes=peak_rss_bytes(),
        endpoint_selected_by_energy=False, early_training_is_not_equilibrium_certified=True)
    atomic_json(folder/'results.json', result)
    return result


def audit(spec: dict, reg: dict, output: Path) -> dict:
    """Fresh common defensive-Q samples after both fixed endpoints are sealed."""
    if (output/'audit.json').exists():
        raise FileExistsError('untouched audit already consumed')
    endpoints = {arm: file_sha256(output/arm/'endpoint.json') for arm in ('network', 'free')}
    n, initial, generators, projected, raw_carrier, bonds, plan, evaluate = setup(spec)
    prepared = json.loads((output/'prepared.json').read_text()); scale = prepared['residual_scale']
    cscale = prepared['cscale']; reference = generators['network'](initial['network'])
    data, info = draw(proposal_spec(spec, 'audit'), n, raw_carrier, bonds, cscale,
        spec['seeds']['audit'], spec['audit']['records'])
    # draw() stores HC per site; products/audit require TOTAL HC and Hpsi.
    c, hc = data['carrier'], n*data['hcarrier']
    positions = {tuple(map(int, x)): i for i, x in enumerate(data['states'])}
    def cached_carrier_action(x):
        indices = [positions[tuple(map(int, row))] for row in np.asarray(x)]
        return c[indices], hc[indices]
    psi = [c]; hpsi = [hc]; works = {}; distances = {}
    for arm in ('network', 'free'):
        params = load_parameter_artifact(output/arm/'endpoint.json', initial[arm])
        bank = generators[arm](params)
        arm_reference=(generators[arm](initial[arm]) if spec.get('free_initialization') is not None else reference)
        arm_scale=prepared.get('residual_scale_by_arm',{}).get(arm,scale)
        p, hp, work = full_products(bank, arm_reference, data['states'], cached_carrier_action, plan, evaluate,
            scale=arm_scale, microbatch=spec['H_microbatch'])
        psi.append(p); hpsi.append(hp); works[arm] = work
        distances[arm] = weighted_state_distance(c, p, data['q'])
        print('audit', arm, work, flush=True)
    psi = np.stack(psi, axis=1); hpsi = np.stack(hpsi, axis=1)
    metrics = paired_chain_audit(psi, hpsi, data['q'], data['groups'], n,
        spec['audit']['bootstrap_replicates'], spec['seeds']['bootstrap'], ('carrier', 'network', 'free'))
    np.savez_compressed(output/'audit.npz', **data, psi=psi, hpsi=hpsi)
    result = dict(registration_sha256=reg['registration_sha256'], endpoint_seals=endpoints,
        audit_sha256=file_sha256(output/'audit.npz'), metrics=metrics, distances=distances,
        sampling=info, full_H_work=works, training_samples_per_arm=spec['steps']*spec['chains']*spec['records_per_step'],
        audit_used_for_training=False, architecture_unchanged=True, new_ED=False, remote_submitted=False,
        peak_rss_bytes=peak_rss_bytes())
    atomic_json(output/'audit.json', result)
    print(json.dumps(metrics['paired'], indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--stage', choices=('register', 'prepare', 'train', 'audit'), required=True)
    parser.add_argument('--arm', choices=('network', 'free'))
    parser.add_argument('--until-step', type=int)
    args = parser.parse_args(); spec = json.loads(args.config.read_text()); validate_spec(spec)
    output = ROOT/spec['output']; output.mkdir(parents=True, exist_ok=True)
    reg = registration(spec)
    if args.stage == 'register':
        if (output/'registration.json').exists(): raise FileExistsError('registration exists')
        atomic_json(output/'registration.json', reg); print(reg['registration_sha256']); return
    if reg != json.loads((output/'registration.json').read_text()):
        raise ValueError('numerical registration mismatch')
    if args.stage == 'train' and args.arm is None: parser.error('--arm required')
    start = perf_counter()
    def alarm(_signal, _frame): raise TimeoutError('registered wall limit reached')
    previous = signal.signal(signal.SIGALRM, alarm); signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage == 'prepare': result = prepare(spec, reg, output)
        elif args.stage == 'train': result = train(spec, reg, output, args.arm, until_step=args.until_step)
        else: result = audit(spec, reg, output)
        check_resources(spec, start)
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',
            dict(wall_seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes(), complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',
            dict(type=type(error).__name__, message=str(error), wall_seconds=perf_counter()-start))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__': main()
