"""Fixed-parent Born vs node-safe importance SR, reusing sealed finite-Schur APIs.

One attempted step per method/parameterization, same physical radius and
carrier protection. Diagnose alignment with -(H-E)C before actual full-H
evaluation; neither high residual energy nor in-cache fitting is a verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import centered_additive_amplitudes
from su2_nqs.vmc.carrier_protection import CarrierProtection, carrier_quality_gate, physical_distance
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.importance_tangent import (make_amplitude_jacobian,
    importance_sr_direction, direction_diagnostics)
from su2_nqs.vmc.matched_state_step import matched_state_step
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint
from .spin_decoder_cost import ROOT, peak_rss_bytes
from .spin_schur_carrier_control import batched, draw
from .spin_finite_schur_optimize import setup, validate_spec, full_products, check_resources, proposal_spec
from .spin_finite_schur_protected import interpolate

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_direction.json'


def registration(spec: dict) -> dict:
    """Seal prior known training data, physical parents and untouched audit seed."""
    validate_spec(spec); CarrierProtection(**spec['protection'])
    if spec['methods'] != ['born', 'importance'] or spec['steps'] != 1:
        raise ValueError('one fixed attempt for exactly two declared methods required')
    if spec['direction_damping'] <= 0 or spec['direction_step_size'] <= 0:
        raise ValueError('positive fixed damping/step required')
    previous, parent = ROOT/spec['previous'], ROOT/spec['parent']
    old = json.loads((previous/'registration.json').read_text())['config']
    if spec['seeds']['audit'] in old['seeds'].values():
        raise ValueError('fresh untouched audit seed required')
    paths = [parent/name for name in ('results.json', 'network.json', 'network.npz', 'free_tensors.json', 'free_tensors.npz')]
    paths += [previous/name for name in ('registration.json', 'prepared.json', 'control.npz')]
    paths += [previous/arm/'training_step_01.npz' for arm in ('network', 'free')]
    record = dict(config=spec, inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    record['registration_sha256'] = canonical_sha256(record)
    return record


def candidate_products(bank, data: dict, plans: list, evaluate: object, scale: float,
                       n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """One candidate bank; immutable reference and TOTAL HC already cached."""
    start = perf_counter(); values, products, counts = [], [], []
    for plan in plans:
        hp, p, count, padded = evaluate(bank, plan)
        values.append(np.asarray(p)); products.append(np.asarray(hp)); counts.append((int(count), int(padded)))
    psi = data['carrier']+scale*(np.concatenate(values)-data['reference_values'])
    hpsi = n*data['hcarrier']+scale*(np.concatenate(products)-data['reference_H_total'])
    return psi, hpsi, dict(seconds=perf_counter()-start, centers=len(psi), residual_H_banks=1,
        cached_reference_and_HC=True, unique_raw_queries=sum(c[0] for c in counts),
        padded_raw_queries=sum(c[1] for c in counts))


def prepare(spec: dict, reg: dict, output: Path) -> dict:
    """Reuse control products/units; no new H action, audit read or samples."""
    if (output/'prepared.json').exists(): raise FileExistsError('preparation sealed')
    previous = ROOT/spec['previous']; prep = json.loads((previous/'prepared.json').read_text())
    result = dict(registration_sha256=reg['registration_sha256'], cscale=prep['cscale'],
        residual_scale=prep['residual_scale'], control_sha256=file_sha256(previous/'control.npz'),
        reference_H_reused=True, control_used_for_training=True, old_audit_not_read=True,
        new_training_samples=0, new_training_H_centers=0)
    atomic_json(output/'prepared.json', result)
    return result


def train(spec: dict, reg: dict, output: Path, arm: str) -> dict:
    """Two independent fixed-parent proposals; complete joint parameters, no continuation."""
    folder = output/arm; folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists(): raise FileExistsError('fixed direction comparison sealed')
    # Partial artifact recovery must not silently overwrite a completed proposal.
    if any(folder.glob('*/candidate.json')): raise FileExistsError('partial proposals retained; inspect before recovery')
    start = perf_counter(); prep = json.loads((output/'prepared.json').read_text())
    previous = ROOT/spec['previous']
    with np.load(previous/'control.npz') as d: data = {k:d[k] for k in d.files}
    with np.load(previous/arm/'training_step_01.npz') as d: born = {k:d[k] for k in d.files}
    n, initial, generators, projected, raw_carrier, _, plan, evaluate = setup(spec)
    params, generator = initial[arm], generators[arm]
    reference = generators['network'](initial['network']); scale = prep['residual_scale']
    carrier = lambda x: raw_carrier(x)/prep['cscale']
    amplitude = centered_additive_amplitudes(projected, carrier, reference, scale=scale)
    amplitude_params = lambda p,x: amplitude(generator(p), x[None])[0]
    jacobian = make_amplitude_jacobian(amplitude_params, params, chunk_size=spec['jacobian_chunk'])
    tick = perf_counter()
    jac = np.asarray(jacobian(params, jnp.asarray(data['states'])))
    bj = np.asarray(jacobian(params, jnp.asarray(born['states'])))
    jac_seconds = perf_counter()-tick
    # A cross-chain prediction, not another model or selectable regularizer.
    ids = data['groups']; crossfold = []
    for parity in (0, 1):
        fit = ids % 2 == parity; held = ~fit
        direction, fitting = importance_sr_direction(data['carrier'][fit], n*data['hcarrier'][fit],
            jac[fit], data['q'][fit], damping=spec['direction_damping'])
        prediction = direction_diagnostics(data['carrier'][held], n*data['hcarrier'][held],
                                           jac[held], data['q'][held], direction)
        crossfold.append(dict(fit_chains=np.unique(ids[fit]).tolist(), held_chains=np.unique(ids[held]).tolist(),
                              fit=fitting, held=prediction, used_to_select_direction=False))
    if np.any(born['psi'] == 0): raise FloatingPointError('old Born cache unexpectedly has carrier nodes')
    directions = {}; diagnostics = {}
    directions['born'], diagnostics['born'] = importance_sr_direction(born['psi'], born['hpsi'], bj,
        abs(born['psi'])**2, damping=spec['direction_damping'])
    directions['importance'], diagnostics['importance'] = importance_sr_direction(data['carrier'],
        n*data['hcarrier'], jac, data['q'], damping=spec['direction_damping'])
    for method in spec['methods']:
        diagnostics[method]['common_control_prediction'] = direction_diagnostics(data['carrier'],
            n*data['hcarrier'], jac, data['q'], directions[method])
    np.savez_compressed(folder/'tangents.npz', control_jacobian=jac, born_jacobian=bj,
        born_direction=directions['born'], importance_direction=directions['importance'])
    plans = [plan(jnp.asarray(data['states'][i:i+spec['H_microbatch']]))
             for i in range(0,len(ids),spec['H_microbatch'])]
    jax.block_until_ready(plans)
    flat, unravel = ravel_pytree(params); policy = CarrierProtection(**spec['protection']); rows = {}
    for method in spec['methods']:
        target = unravel(flat-spec['direction_step_size']*jnp.asarray(directions[method]))
        destination = folder/method; destination.mkdir(exist_ok=True)
        tick = perf_counter()
        def candidate(alpha):
            bank = generator(interpolate(params, target, alpha))
            return data['carrier']+scale*(batched(lambda x:projected(bank,x),data['states'],8)-data['reference_values'])
        matching = matched_state_step(candidate, data['carrier'], data['q'], policy.maximum_step_fs, 1.,
            spec['matching_bisections'], relative_tolerance=spec['matching_relative_tolerance'])
        proposed = interpolate(params, target, matching.alpha)
        matching_seconds = perf_counter()-tick
        p,hp,work = candidate_products(generator(proposed),data,plans,evaluate,scale,n)
        replay = float(np.max(abs(p-matching.amplitudes)))
        if replay > 2e-10*max(1.,float(np.max(abs(p)))): raise FloatingPointError('candidate H-plan replay failed')
        gate = carrier_quality_gate(data['carrier'],n*data['hcarrier'],data['carrier'],n*data['hcarrier'],
            p,hp,data['q'],ids,n,policy)
        accepted = bool(matching.matched and gate['accepted'])
        identity = dict(registration_sha256=reg['registration_sha256'],arm=arm,method=method)
        save_parameter_artifact(destination/'candidate.json',proposed,identity={**identity,'kind':'raw_fixed_proposal'})
        committed = proposed if accepted else params
        save_parameter_artifact(destination/'endpoint.json',committed,identity={**identity,'kind':'protected_endpoint'})
        np.savez_compressed(destination/'control.npz',psi=p,hpsi=hp)
        row = dict(method=method,accepted=accepted,diagnostics=diagnostics[method],matching=matching.as_dict(),
            quality_gate=gate,control_H=work,matching_seconds=matching_seconds,amplitude_H_plan_replay_max=replay,
            candidate_sha256=file_sha256(destination/'candidate.json'),endpoint_sha256=file_sha256(destination/'endpoint.json'),
            initial_parameter_sha256=pytree_sha256(params),final_parameter_sha256=pytree_sha256(committed))
        save_resumable_checkpoint(destination/'checkpoint.pkl',committed,
            optimizer_state=dict(candidate_params=proposed,direction=directions[method],
                                 control_psi=p if accepted else data['carrier'],
                                 control_hpsi=hp if accepted else n*data['hcarrier']),
            completed_steps=1,next_sample_index=0,run_identity=identity,history=[row])
        rows[method] = row
        atomic_json(destination/'results.json',row)
        print(arm,method,'accepted',accepted,'control DeltaE/N',gate['delta_energy_vs_carrier'],
            'DeltaVar/N',gate['delta_variance_vs_carrier'],'FS',matching.distance['fs'],
            'crossfold cos',[x['held']['descent_alignment_cosine'] for x in crossfold],flush=True)
        check_resources(spec,start)
    result = dict(registration_sha256=reg['registration_sha256'],arm=arm,methods=rows,crossfold=crossfold,
        control_fit_records=len(ids),born_fit_records=len(born['psi']),jacobian_seconds=jac_seconds,
        tangent_sha256=file_sha256(folder/'tangents.npz'),control_H_centers=sum(x['control_H']['centers'] for x in rows.values()),
        new_training_H_centers=0,new_training_samples=0,damping_selected_by_outcome=False,
        fixed_parent=True,peak_rss_bytes=peak_rss_bytes())
    atomic_json(folder/'results.json',result)
    return result


def audit(spec: dict, reg: dict, output: Path) -> dict:
    """Untouched COMMON Q for all raw proposals, including rejected ones."""
    if (output/'audit.json').exists(): raise FileExistsError('audit already consumed')
    names = [f'{arm}_{method}' for arm in ('network','free') for method in spec['methods']]
    seals = {name:file_sha256(output/name.split('_')[0]/name.split('_')[1]/'candidate.json') for name in names}
    n, initial, generators, _, raw_carrier, bonds, plan, evaluate = setup(spec)
    prep = json.loads((output/'prepared.json').read_text()); reference = generators['network'](initial['network'])
    data, info = draw(proposal_spec(spec,'audit'),n,raw_carrier,bonds,prep['cscale'],spec['seeds']['audit'],spec['audit']['records'])
    c,hc = data['carrier'], n*data['hcarrier']; positions = {tuple(map(int,x)):i for i,x in enumerate(data['states'])}
    def action(x):
        indices = [positions[tuple(map(int,row))] for row in np.asarray(x)]
        return c[indices],hc[indices]
    psi,hpsi = [c],[hc]; works,distances,decisions = {},{},{}
    for name in names:
        arm,method = name.split('_'); destination=output/arm/method
        params=load_parameter_artifact(destination/'candidate.json',initial[arm])
        p,hp,work=full_products(generators[arm](params),reference,data['states'],action,plan,evaluate,
            scale=prep['residual_scale'],microbatch=spec['H_microbatch'])
        psi.append(p);hpsi.append(hp);works[name]=work
        distances[name]=physical_distance(c,p,data['q'])
        decisions[name]=json.loads((destination/'results.json').read_text())['accepted']
        print('untouched candidate',name,work['seconds'],flush=True)
    psi,hpsi = np.stack(psi,axis=1),np.stack(hpsi,axis=1)
    metrics=paired_chain_audit(psi,hpsi,data['q'],data['groups'],n,spec['audit']['bootstrap_replicates'],
        spec['seeds']['bootstrap'],('carrier',*names))
    np.savez_compressed(output/'audit.npz',**data,psi=psi,hpsi=hpsi)
    result=dict(registration_sha256=reg['registration_sha256'],candidate_seals=seals,metrics=metrics,
        audit_sha256=file_sha256(output/'audit.npz'),distances=distances,accepted_before_audit=decisions,
        raw_candidates_include_rejections=True,committed_rejections_equal_carrier=True,
        audit_used_for_training=False,full_H_work=works,sampling=info,architecture_unchanged=True,
        new_ED=False,remote_submitted=False,peak_rss_bytes=peak_rss_bytes())
    atomic_json(output/'audit.json',result)
    print(json.dumps(metrics['paired'],indent=2),flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=('register','prepare','train','audit'),required=True)
    parser.add_argument('--arm',choices=('network','free'))
    args=parser.parse_args();spec=json.loads(args.config.read_text());reg=registration(spec)
    output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists(): raise FileExistsError('registration sealed')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg != json.loads((output/'registration.json').read_text()): raise ValueError('registration mismatch')
    if args.stage=='train' and args.arm is None: parser.error('--arm required')
    start=perf_counter()
    def alarm(_signal,_frame): raise TimeoutError('bounded process limit exceeded')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='prepare':prepare(spec,reg,output)
        elif args.stage=='train':train(spec,reg,output,args.arm)
        else:audit(spec,reg,output)
        check_resources(spec,start)
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',
                    dict(wall_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',
                    dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start))
        raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
