"""Off-training FS calibration and carrier-protected finite-Schur optimization.

Extends the sealed existing runner through its reusable setup/product/audit
interfaces. It changes only the optimizer, never the physical state manifold.
Old sources/checkpoints remain immutable; the control set is not called audit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.spatial_vit_sampling import make_log_exchange_sampler
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import centered_additive_amplitudes
from su2_nqs.vmc.carrier_protection import CarrierProtection, carrier_quality_gate, physical_moments
from su2_nqs.vmc.cutset_training import weighted_state_distance
from su2_nqs.vmc.dual_sampled_sr import make_dual_sampled_sr_update
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.matched_state_step import matched_state_step
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint
from .spin_decoder_cost import ROOT, peak_rss_bytes
from .spin_schur_carrier_control import batched
from .spin_finite_schur_optimize import setup, validate_spec, full_products, json_scalars, check_resources, audit

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_protected.json'


def registration(spec: dict) -> dict:
    """Seal unchanged physical parents, reused controls and fresh audit seeds."""
    validate_spec(spec); CarrierProtection(**spec['protection'])
    parent, previous = ROOT/spec['parent'], ROOT/spec['previous']
    names = [parent/name for name in ('results.json', 'network.json', 'network.npz', 'free_tensors.json', 'free_tensors.npz')]
    names += [previous/name for name in ('registration.json', 'prepared.json', 'calibration.npz')]
    names += [previous/arm/name for arm in ('network', 'free') for name in ('endpoint.json', 'endpoint.npz')]
    old_spec = json.loads((previous/'registration.json').read_text())['config']
    if spec['seeds']['audit'] in old_spec['seeds'].values() or spec['seeds']['training'] in old_spec['seeds'].values():
        raise ValueError('new training and untouched audit need fresh seeds')
    record = dict(config=spec, inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in names},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    record['registration_sha256'] = canonical_sha256(record)
    return record


def interpolate(parent: object, proposal: object, alpha: float) -> object:
    """Same frozen parameter direction; alpha is an optimizer step, not a new head."""
    return jax.tree.map(lambda a, b: a+alpha*(b-a), parent, proposal)


def prepare(spec: dict, reg: dict, output: Path) -> dict:
    """Reuse fixed Q/known HC, cache reference products, reject prior failed drift."""
    if (output/'prepared.json').exists(): raise FileExistsError('preparation already sealed')
    previous = ROOT/spec['previous']; old = json.loads((previous/'prepared.json').read_text())
    with np.load(previous/'calibration.npz') as archive:
        data = {k: archive[k] for k in archive.files}
    n, initial, generators, projected, _, _, plan, evaluate = setup(spec)
    reference = generators['network'](initial['network'])
    values, products, counts = [], [], []
    tick = perf_counter()
    for offset in range(0, len(data['states']), spec['H_microbatch']):
        hp, p, count, padded = evaluate(reference, plan(jnp.asarray(data['states'][offset:offset+spec['H_microbatch']])))
        values.append(np.asarray(p)); products.append(np.asarray(hp)); counts.append((int(count), int(padded)))
    phi, hphi = np.concatenate(values), np.concatenate(products)
    data['reference_values'] = phi; data['reference_H_total'] = hphi
    reference_work = dict(seconds=perf_counter()-tick, centers=len(phi), residual_H_banks=1,
                          unique_raw_queries=sum(c[0] for c in counts), padded_raw_queries=sum(c[1] for c in counts))
    old_distances = {}
    for arm in ('network', 'free'):
        params = load_parameter_artifact(previous/arm/'endpoint.json', initial[arm])
        p = data['carrier']+old['residual_scale']*(batched(lambda x: projected(generators[arm](params), x), data['states'], 8)-phi)
        d = weighted_state_distance(data['carrier'], p, data['q'])
        old_distances[arm] = dict(distance=d, rejected_by_carrier_radius=d['fs'] > spec['protection']['maximum_carrier_fs'])
    if not all(v['rejected_by_carrier_radius'] for v in old_distances.values()):
        raise ValueError('registered cache preflight did not intercept the prior failed drift; preserve result, do not train')
    np.savez_compressed(output/'control.npz', **data)
    record = dict(registration_sha256=reg['registration_sha256'], cscale=old['cscale'],
        residual_scale=old['residual_scale'], control_sha256=file_sha256(output/'control.npz'),
        control_records=len(phi), control_used_for_training=True, original_Q_seed=spec['seeds']['calibration'],
        reference_work=reference_work, prior_failed_geometry=old_distances,
        initial_control_moments=physical_moments(data['carrier'], n*data['hcarrier'], data['q'], n),
        no_new_control_samples=True, old_audit_not_reused=True)
    atomic_json(output/'prepared.json', record)
    print('Control cache sealed; previous endpoint FS:', {a:v['distance']['fs'] for a,v in old_distances.items()}, flush=True)
    return record


def train(spec: dict, reg: dict, output: Path, arm: str) -> dict:
    """Fixed attempts with separate physical controls, rollback and complete restart state."""
    folder = output/arm; folder.mkdir(exist_ok=True)
    if (folder/'endpoint.json').exists(): raise FileExistsError('fixed endpoint already sealed')
    start = perf_counter(); prepared = json.loads((output/'prepared.json').read_text())
    with np.load(output/'control.npz') as archive: data = {k:archive[k] for k in archive.files}
    n, initial, generators, projected, raw_carrier, bonds, plan, evaluate = setup(spec)
    params = initial[arm]; generator = generators[arm]; reference = generators['network'](initial['network'])
    scale = prepared['residual_scale']; carrier = lambda x: raw_carrier(x)/prepared['cscale']
    amplitude = centered_additive_amplitudes(projected, carrier, reference, scale=scale)
    log_bank = lambda bank, x: jnp.log(amplitude(bank, x[None])[0])
    update = make_dual_sampled_sr_update(lambda p,x:log_bank(generator(p),x), params, **spec['SR'])
    sampler = make_log_exchange_sampler(log_bank, spec['decorrelation_proposals_per_record'])
    burn = make_log_exchange_sampler(log_bank, spec['burn_proposals_per_chain'])
    carrier_work = make_amplitude_heisenberg_action(lambda _,x:carrier(x[None])[0], bonds, n_sites=n)
    carrier_action = lambda x:carrier_work({},x)
    policy = CarrierProtection(**spec['protection'])
    # Exact reuse of state-independent plans; every image/bond multiplicity remains.
    plans = [plan(jnp.asarray(data['states'][i:i+spec['H_microbatch']]))
             for i in range(0, len(data['states']), spec['H_microbatch'])]
    jax.block_until_ready(plans)
    c, hc = data['carrier'], n*data['hcarrier']; control_psi, control_hp = c.copy(), hc.copy()
    key = jax.random.PRNGKey(spec['seeds']['training'])
    states = initialize_spin_exchange_chains(jax.random.fold_in(key,1), n_chains=spec['chains'], n_sites=n, n_up=n//2)
    history = []; burnt = False
    identity = dict(registration_sha256=reg['registration_sha256'], arm=arm,
                    prepared_sha256=file_sha256(output/'prepared.json'))
    checkpoint = folder/'checkpoint.pkl'
    if checkpoint.exists():
        saved = load_resumable_checkpoint(checkpoint, expected_run_identity=identity)
        params, history = saved['params'], saved['history']
        opt = saved['optimizer_state']; key, states, burnt = opt['key'],opt['states'],opt['burnt']
        control_psi, control_hp = opt['control_psi'],opt['control_hp']
    def persist() -> None:
        save_resumable_checkpoint(checkpoint, params, optimizer_state=dict(key=key,states=states,burnt=burnt,
            control_psi=control_psi,control_hp=control_hp), completed_steps=len(history),
            next_sample_index=len(history)*spec['chains']*spec['records_per_step'],
            run_identity=identity,history=history)
    persist()
    for step in range(len(history), spec['steps']):
        tick = perf_counter(); bank = generator(params)
        if not burnt:
            states,key,_,_ = burn(bank,key,states); jax.block_until_ready(states); burnt=True
        samples=[]; acceptance=[]; movement=[]
        for _ in range(spec['records_per_step']):
            states,key,a,m = sampler(bank,key,states)
            samples.append(np.asarray(states)); acceptance.append(float(a)); movement.append(float(m))
        samples=np.concatenate(samples); sampling_seconds=perf_counter()-tick
        psi,hpsi,work = full_products(bank,reference,samples,carrier_action,plan,evaluate,
            scale=scale,microbatch=spec['H_microbatch'])
        if np.any(psi==0) or not np.isfinite(hpsi/psi).all(): raise FloatingPointError('invalid Born local energy')
        energies=hpsi/psi
        tick=perf_counter(); proposal,diag=update(params,jnp.asarray(samples),jnp.asarray(energies))
        jax.block_until_ready(proposal); sr_seconds=perf_counter()-tick
        tick=perf_counter()
        def candidate(alpha: float) -> np.ndarray:
            candidate_bank=generator(interpolate(params,proposal,alpha))
            return c+scale*(batched(lambda x:projected(candidate_bank,x),data['states'],8)-data['reference_values'])
        matched=matched_state_step(candidate,control_psi,data['q'],policy.maximum_step_fs,1.,
            spec['matching_bisections'],relative_tolerance=spec['matching_relative_tolerance'])
        proposed=interpolate(params,proposal,matched.alpha); proposed_bank=generator(proposed)
        physical_matching_seconds=perf_counter()-tick
        tick=perf_counter(); hp_rows=[]; p_rows=[]; query_count=0
        for prepared_plan in plans:
            h,p,count,_=evaluate(proposed_bank,prepared_plan)
            hp_rows.append(np.asarray(h));p_rows.append(np.asarray(p));query_count+=int(count)
        cp=c+scale*(np.concatenate(p_rows)-data['reference_values'])
        chp=hc+scale*(np.concatenate(hp_rows)-data['reference_H_total'])
        control_H_seconds=perf_counter()-tick
        replay_error=float(np.max(abs(cp-matched.amplitudes)))
        if replay_error > 2e-10*max(1.,float(np.max(abs(cp)))):
            raise FloatingPointError('amplitude/H-plan control state replay mismatch')
        gate=carrier_quality_gate(c,hc,control_psi,control_hp,cp,chp,data['q'],data['groups'],n,policy)
        accepted=bool(gate['accepted'] and diag['accepted'])
        np.savez_compressed(folder/f'control_step_{step+1:02d}.npz',parent_psi=control_psi,parent_hpsi=control_hp,
                            candidate_psi=cp,candidate_hpsi=chp)
        save_parameter_artifact(folder/f'candidate_step_{step+1:02d}.json',proposed,
            identity={**identity,'attempt':step+1,'kind':'candidate_not_committed_until_quality_gate'},
            overwrite=True)  # deterministic recovery of an interrupted, not-yet-checkpointed attempt
        if accepted: params=proposed;control_psi=cp;control_hp=chp
        row=dict(step=step+1,accepted=accepted,raw_SR=json_scalars(diag),matching=matched.as_dict(),quality_gate=gate,
            training_energy_per_site=float(energies.real.mean()/n),training_variance_per_site=float(np.mean(abs(energies-energies.mean())**2)/n),
            sampling_seconds=sampling_seconds,SR_seconds=sr_seconds,physical_matching_seconds=physical_matching_seconds,
            acceptance=float(np.mean(acceptance)),movement=float(np.mean(movement)),full_H=work,
            control_H=dict(seconds=control_H_seconds,centers=len(c),residual_H_banks=1,
                frozen_reference_reused=True,unique_raw_queries=query_count),
            amplitude_H_plan_replay_max=replay_error,parameter_sha256=pytree_sha256(params))
        history.append(row);persist()
        np.savez_compressed(folder/f'training_step_{step+1:02d}.npz',states=samples,psi=psi,hpsi=hpsi)
        atomic_json(folder/'progress.json',dict(arm=arm,history=history,completed_steps=len(history),
            accepted_steps=sum(r['accepted'] for r in history),peak_rss_bytes=peak_rss_bytes()))
        print(arm,step+1,'accept',accepted,'control DeltaE/N',gate['delta_energy_vs_carrier'],
            'FS step/carrier',gate['step_distance']['fs'],gate['carrier_distance']['fs'],
            'reject',gate['rejected_by'],flush=True)
        check_resources(spec,start)
    save_parameter_artifact(folder/'endpoint.json',params,identity=identity)
    result=dict(arm=arm,registration_sha256=reg['registration_sha256'],fixed_attempts=len(history),
        accepted_steps=sum(r['accepted'] for r in history),training_H_centers=sum(r['full_H']['centers'] for r in history),
        control_H_centers=sum(r['control_H']['centers'] for r in history),history=history,
        endpoint_sha256=file_sha256(folder/'endpoint.json'),endpoint_selected_by_energy=False,
        control_is_optimizer_data_not_audit=True,initial_parameter_sha256=pytree_sha256(initial[arm]),
        final_parameter_sha256=pytree_sha256(params),peak_rss_bytes=peak_rss_bytes())
    atomic_json(folder/'results.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=('register','prepare','train','audit'),required=True)
    parser.add_argument('--arm',choices=('network','free'))
    args=parser.parse_args();spec=json.loads(args.config.read_text());reg=registration(spec)
    output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists(): raise FileExistsError('registration exists')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg != json.loads((output/'registration.json').read_text()): raise ValueError('registration mismatch')
    if args.stage=='train' and args.arm is None: parser.error('--arm required')
    start=perf_counter()
    def alarm(_signal,_frame): raise TimeoutError('registered wall limit reached')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='prepare': prepare(spec,reg,output)
        elif args.stage=='train': train(spec,reg,output,args.arm)
        else: audit(spec,reg,output)
        check_resources(spec,start)
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',
            dict(wall_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',
            dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__': main()
