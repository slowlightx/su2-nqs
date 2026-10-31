"""Fixed-endpoint, matched physical-step comparison of two SU2 estimators.

Reuses the sealed model, sampler, carrier, node-safe H and paired audit.
Only the estimator changes; all backbone parameters receive full gradients.
N64 is conditional on the frozen N36 audit gate, never on training energy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.schur_cutset_attention import make_branch_bank
from su2_nqs.vmc.cutset_training import weighted_state_distance, paired_chain_audit
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.matched_state_step import matched_state_step
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint
from su2_nqs.vmc.singlet_carrier_control import carrier_control_evaluation
from .spin_decoder_cost import ROOT, peak_rss_bytes
from .spin_schur_carrier_control import setup, draw, batched

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_control_training.json'


def registration(spec: dict) -> dict:
    """Seal parents, known metric sets, protocol and all numerical dependencies."""
    interface = json.loads((ROOT/spec['interface_config']).read_text())
    parent = ROOT/interface['parent']/'triangular'
    paths = [ROOT/spec['interface_config'], parent/'results.json', parent/'parameters.json', parent/'parameters.npz']
    for case in spec['cases']:
        known = ROOT/spec['known_metric_parent']/f"n{case['length']**2}"
        paths += [known/'results.json', known/'audit_values.npz']
    result = dict(config=spec, inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def combine_development_metric(fresh: dict, known: dict) -> dict:
    """Equal normalized-Gram mixture; not an unbiased new sampling density.

    Inputs share physical amplitude units but may have different unnormalized
    Q. Each corpus contributes half the empirical carrier norm. Only used for
    step distance, NEVER energy training or untouched expectation values.
    """
    combined = {key:np.concatenate((fresh[key], known[key])) for key in ('states', 'carrier', 'reference')}
    weights = []
    for data in (fresh, known):
        q = np.asarray(data['q'], dtype=float); c = np.asarray(data['carrier'])
        if np.any(q <= 0) or not np.isfinite(q).all():
            raise ValueError('metric Q must be positive and finite')
        norm = np.sum(abs(c)**2/q)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('metric carrier has zero/nonfinite norm')
        weights.append(2*q*norm)
    combined['q'] = np.concatenate(weights)
    combined['corpus'] = np.concatenate((np.zeros(len(fresh['q']), dtype=int), np.ones(len(known['q']), dtype=int)))
    return combined


def ray_residual_concentration(carrier: np.ndarray, psi: np.ndarray, q: np.ndarray, groups: np.ndarray) -> dict:
    """Physical change modulo the carrier ray, not raw additive parametrization."""
    norm = np.sum(abs(carrier)**2/q)
    overlap = np.sum(carrier.conj()*psi/q)/norm
    residual = abs(psi-overlap*carrier)**2/q
    total = residual.sum()
    return dict(max_sample_fraction=float(residual.max()/total) if total else 0.,
                max_chain_fraction=float(max(residual[groups == g].sum() for g in np.unique(groups))/total) if total else 0.)


def expansion_gate(audit: dict, training: dict, distances: dict, residuals: dict, spec: dict) -> dict:
    """Frozen operational N64 release rule; distinct from an estimator-win claim."""
    g = spec['gate']; paired = audit['paired']; cv = paired['control-minus-carrier']
    delta = paired['control-minus-diagonal']['energy_per_site']
    checks = dict(
        resolved_control_energy_gain=cv['energy_per_site']['chain_bootstrap_95'][1] < g['energy_bootstrap95_upper_below'],
        control_energy_point_not_worse_than_diagonal=delta['value'] <= g['control_minus_diagonal_energy_point_max'],
        operational_variance_safety=cv['variance_per_site']['value']+g['variance_jackknife_safety_sigma']*cv['variance_per_site']['chain_jackknife_se']
            <= g['variance_parent_fraction_margin']*audit['arms']['carrier']['variance_per_site']['value'],
        all_attempted_steps_matched_and_accepted=all(row['accepted_pair'] for row in training['history']))
    fs = []
    for arm in spec['arms']:
        metric = training['final_metric_distance'][arm]['fs']; observed = distances[arm]['fs']; fs.append(observed)
        checks[arm+'_metric_transfer'] = metric > 0 and observed**2/metric**2 <= g['max_audit_to_metric_parent_fs_squared_ratio']
        checks[arm+'_norm_concentration'] = audit['arms'][arm]['weight_ess_fraction'] >= spec['minimum_norm_weight_ess_fraction']
        checks[arm+'_ray_residual_not_one_chain'] = residuals[arm]['max_chain_fraction'] <= g['max_ray_residual_chain_fraction']
        checks[arm+'_H_residual_not_one_chain'] = audit['arms'][arm]['max_residual_chain_fraction'] <= g['max_H_residual_chain_fraction']
    checks['audit_between_arm_distance_match'] = min(fs) > 0 and max(fs)/min(fs) <= g['max_audit_between_arm_fs_ratio']
    return dict(checks={key:bool(value) for key,value in checks.items()}, passed=bool(all(checks.values())),
        CV_beats_diagonal_energy_95=bool(delta['chain_bootstrap_95'][1] < 0),
        variance_rule='delta + 1JKSE <= 10% parent variance; operational safety, not a variance win',
        production_authorized=False)


def run(path: Path, length: int, stage: str, resume: bool = False) -> None:
    spec = json.loads(path.read_text()); case = next(c for c in spec['cases'] if c['length'] == length)
    output = ROOT/spec['output']; registered = json.loads((output/'registration.json').read_text())
    if registered != registration(spec):
        raise ValueError('registered numerical source/input/protocol changed')
    if length == 8:
        prerequisite = output/'n36/audit/results.json'
        if not prerequisite.exists() or not json.loads(prerequisite.read_text())['gate']['passed']:
            raise PermissionError('N64 requires the complete preregistered N36 audit gate')
    folder = output/f'n{length**2}'; dest = folder/stage; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'results.json').exists():
        raise FileExistsError('completed stage is sealed')
    if (dest/'progress.json').exists() and not resume:
        raise FileExistsError('interrupted stage needs explicit --resume, not silent restart')
    if resume and stage == 'prepare':
        raise ValueError('preparation is bounded once; use a new registration if interrupted')
    previous_progress = json.loads((dest/'progress.json').read_text()) if resume and (dest/'progress.json').exists() else None
    identity = dict(registration_sha256=registered['registration_sha256'], length=length, stage=stage)
    start = perf_counter(); prior_seconds = 0.; ledger = {}
    def progress(event: str, **extra):
        seconds = prior_seconds+perf_counter()-start
        atomic_json(dest/'progress.json', dict(event=event, seconds=seconds, peak_rss_bytes=peak_rss_bytes(), ledger=ledger, **extra))
        print(length, stage, event, round(seconds,2), 's', round(peak_rss_bytes()/2**30,2), 'GiB', flush=True)
        if peak_rss_bytes() > spec['peak_rss_limit_gib']*2**30:
            raise MemoryError('registered RSS cap')
        if seconds > spec[stage+'_wall_seconds']:
            raise TimeoutError('cumulative stage wall cap')
    def alarm(_s, _f): raise TimeoutError('registered stage wall cap')
    old_alarm = signal.signal(signal.SIGALRM, alarm); signal.alarm(spec[stage+'_wall_seconds'])
    try:
        initial, cfg, residual, carrier, bonds, group_order = setup(spec, length)
        generate = jax.jit(lambda p: make_branch_bank(p, cfg)); evaluate = jax.jit(residual)
        def values(bank, states, key):
            result = batched(lambda x: evaluate(bank,x), states, spec['microbatch'])
            ledger[key] = ledger.get(key,0)+len(states)
            return result
        initial_bank = generate(initial)
        progress('setup')
        if stage == 'prepare':
            rng = np.random.default_rng(spec['scale_seed']+case['seed_offset'])
            uniform = np.zeros((spec['uniform_scale_samples'],length**2),dtype=np.int8)
            for row in uniform: row[rng.choice(length**2,length**2//2,replace=False)] = 1
            cscale = float(np.sqrt(np.mean(abs(batched(carrier,uniform,32))**2)))
            if not np.isfinite(cscale) or cscale <= 0: raise FloatingPointError('carrier scaling')
            corpora = {}; info = {}
            for label in ('calibration','training'):
                data, info[label] = draw(spec,length**2,carrier,bonds,cscale,spec[label+'_seed']+case['seed_offset'],spec[label+'_blocks'])
                data['reference'] = values(initial_bank,data['states'],'reference_amplitude_centers')
                corpora[label] = data; np.savez_compressed(dest/(label+'.npz'),**data)
                progress(label+'_saved')
            known_folder = ROOT/spec['known_metric_parent']/f'n{length**2}'
            known = dict(np.load(known_folder/'audit_values.npz'))
            known_scale = json.loads((known_folder/'results.json').read_text())['carrier_scale']
            known['carrier'] = known['carrier']*known_scale/cscale
            known['reference'] = values(initial_bank,known['states'],'reference_amplitude_centers')
            metric = combine_development_metric(corpora['calibration'],known)
            scale = float(np.sqrt(np.sum(abs(metric['reference'])**2/metric['q'])/
                                  np.sum(abs(metric['carrier'])**2/metric['q'])))
            if not np.isfinite(scale) or scale <= 0: raise FloatingPointError('residual scaling')
            np.savez_compressed(dest/'metric.npz',**metric)
            save_parameter_artifact(dest/'initial.json',initial,identity=identity)
            result = dict(identity=identity, carrier_scale=cscale, residual_scale=scale, sampling=info,
                initial_sha256=pytree_sha256(initial), group_order=group_order,
                parameter_count=sum(x.size for x in jax.tree.leaves(initial)), ledger=ledger,
                seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes())
            result['files'] = {p.name:file_sha256(p) for p in dest.glob('*.npz')}
            atomic_json(dest/'results.json',result)
        elif stage == 'train':
            prepared = json.loads((folder/'prepare/results.json').read_text())
            if prepared['initial_sha256'] != pytree_sha256(initial): raise ValueError('initial state changed')
            for name,digest in prepared['files'].items():
                if file_sha256(folder/'prepare'/name) != digest: raise ValueError('prepared corpus changed')
            data = dict(np.load(folder/'prepare/training.npz')); metric = dict(np.load(folder/'prepare/metric.npz'))
            scale = prepared['residual_scale']; params = {arm:initial for arm in spec['arms']}
            m = jax.tree.map(jnp.zeros_like,params); v = jax.tree.map(jnp.zeros_like,params); history = []
            if resume:
                saved = load_resumable_checkpoint(dest/'checkpoint.pkl',expected_run_identity=identity)
                params = saved['params']; m,v = saved['optimizer_state']; history = saved['history']
                if saved['metadata']['prepared_files'] != prepared['files']: raise ValueError('checkpoint corpus changed')
                ledger.update(saved['metadata']['ledger']); prior_seconds = saved['metadata']['active_seconds']
                if previous_progress is not None:
                    prior_seconds = max(prior_seconds,previous_progress['seconds'])
                    for key,value in previous_progress['ledger'].items(): ledger[key] = max(ledger.get(key,0),value)
                remaining = spec['train_wall_seconds']-prior_seconds-(perf_counter()-start)
                if remaining <= 0: raise TimeoutError('cumulative training budget exhausted')
                signal.alarm(max(1,int(remaining)))
            def state(p,corpus,key):
                return corpus['carrier']+(values(generate(p),corpus['states'],key)-corpus['reference'])/scale
            oldmetric = {arm:state(params[arm],metric,arm+'_metric_centers') for arm in spec['arms']}
            def micro_objective(bank,x,source):
                return 2*jnp.real(jnp.vdot(residual(bank,x)/scale,jax.lax.stop_gradient(source)))
            tick = perf_counter()
            bank_gradient = jax.jit(jax.grad(micro_objective)).lower(initial_bank,
                data['states'][:spec['microbatch']],jnp.zeros(spec['microbatch'],dtype=jnp.complex128)).compile()
            _,warm_pullback = jax.vjp(generate,initial)
            jax.block_until_ready(warm_pullback(jax.tree.map(jnp.zeros_like,initial_bank)))
            shared_compile_seconds = perf_counter()-tick
            progress('shared_gradient_compile_complete',shared_compile_seconds=shared_compile_seconds)
            arm_seconds = {arm:sum(row['arms'][arm]['total_seconds'] for row in history) for arm in spec['arms']}
            for step in range(len(history),case['steps']):
                candidates = {}; candidates_metric = {}; rows = {}
                for arm in spec['arms']:
                    tick = perf_counter(); bank,pullback = jax.vjp(generate,params[arm])
                    psi = data['carrier']+(values(bank,data['states'],arm+'_train_amplitude_centers')-data['reference'])/scale
                    norm = float(np.sum(abs(psi)**2/data['q']))
                    if arm == 'control':
                        evaluated = carrier_control_evaluation(psi,data['carrier'],data['hcarrier'],data['q'],data['diagonal'])
                        energy = float(evaluated.energy); force = np.asarray(evaluated.residual)
                    else:
                        energy = float(np.sum(abs(psi)**2/data['q']*data['diagonal'])/norm)
                        force = (data['diagonal']-energy)*psi
                    source = force/(data['q']*norm); gb = jax.tree.map(jnp.zeros_like,bank)
                    for i in range(0,len(psi),spec['microbatch']):
                        sl = slice(i,i+spec['microbatch'])
                        derivative = bank_gradient(bank,data['states'][sl],source[sl])
                        gb = jax.tree.map(lambda a,b:a+b,gb,derivative)
                    gradient = pullback(gb)[0]; jax.block_until_ready(gradient)
                    ledger[arm+'_gradient_centers'] = ledger.get(arm+'_gradient_centers',0)+len(psi)
                    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
                        raise FloatingPointError('nonfinite full gradient')
                    m[arm] = jax.tree.map(lambda a,b:.9*a+.1*b,m[arm],gradient)
                    v[arm] = jax.tree.map(lambda a,b:.999*a+.001*b*b,v[arm],gradient)
                    direction = jax.tree.map(lambda a,b:spec['learning_rate']*(a/(1-.9**(step+1)))/
                        (jnp.sqrt(b/(1-.999**(step+1)))+1e-8),m[arm],v[arm])
                    gradient_seconds = perf_counter()-tick
                    def candidate(alpha):
                        p = jax.tree.map(lambda a,b:a-alpha*b,params[arm],direction)
                        return state(p,metric,arm+'_metric_centers')
                    match = matched_state_step(candidate,oldmetric[arm],metric['q'],case['step_fs'],
                        spec['max_step_alpha'],spec['bisection_steps'],relative_tolerance=spec['relative_matching_tolerance'])
                    candidates[arm] = jax.tree.map(lambda a,b:a-match.alpha*b,params[arm],direction)
                    candidates_metric[arm] = match.amplitudes
                    parent_distance = weighted_state_distance(metric['carrier'],match.amplitudes,metric['q'])
                    rows[arm] = dict(training_objective_before=energy,match=match.as_dict(),parent_distance=parent_distance,
                        gradient_seconds=gradient_seconds,total_seconds=perf_counter()-tick,
                        gradient_norm_by_block={key:float(np.sqrt(sum(np.sum(np.asarray(x)**2) for x in jax.tree.leaves(gradient[key])))) for key in gradient})
                    arm_seconds[arm] += rows[arm]['total_seconds']
                    progress(f'step_{step+1}_{arm}_matched',radius=match.distance['fs'],matched=match.matched)
                radii = [rows[arm]['match']['distance']['fs'] for arm in spec['arms']]
                accepted = all(rows[a]['match']['matched'] and rows[a]['parent_distance']['fs'] <= case['parent_fs_cap']
                    and rows[a]['parent_distance']['ess_fraction'] >= spec['minimum_norm_weight_ess_fraction'] for a in spec['arms'])
                accepted = bool(accepted and min(radii) > 0 and max(radii)/min(radii) <= spec['maximum_matched_step_ratio'])
                if accepted: params = candidates; oldmetric = candidates_metric
                history.append(dict(step=step+1,accepted_pair=accepted,arms=rows))
                save_resumable_checkpoint(dest/'checkpoint.pkl',jax.device_get(params),optimizer_state=jax.device_get((m,v)),
                    completed_steps=step+1,next_sample_index=0,run_identity=identity,history=history,
                    metadata=dict(active_seconds=prior_seconds+perf_counter()-start,ledger=ledger,
                                  prepared_files=prepared['files'],fixed_endpoint=True))
                progress(f'step_{step+1}_sealed',accepted_pair=accepted)
            distances = {};changes = {};constituents = {}
            for arm in spec['arms']:
                if not (dest/(arm+'.json')).exists():
                    save_parameter_artifact(dest/(arm+'.json'),params[arm],identity={**identity,'arm':arm})
                replay = load_parameter_artifact(dest/(arm+'.json'),initial,expected_identity={**identity,'arm':arm})
                if pytree_sha256(replay) != pytree_sha256(params[arm]): raise AssertionError('endpoint replay')
                distances[arm] = weighted_state_distance(metric['carrier'],oldmetric[arm],metric['q'])
                constituents[arm] = {name:weighted_state_distance(metric['carrier'][metric['corpus'] == k],
                    oldmetric[arm][metric['corpus'] == k],metric['q'][metric['corpus'] == k])
                    for k,name in enumerate(('fresh','known'))}
                changes[arm] = {key:float(np.sqrt(sum(np.sum((np.asarray(a)-np.asarray(b))**2) for a,b in
                    zip(jax.tree.leaves(params[arm][key]),jax.tree.leaves(initial[key]),strict=True)))) for key in initial}
            np.savez_compressed(dest/'metric_values.npz',**{a:oldmetric[a] for a in spec['arms']})
            atomic_json(dest/'results.json',dict(identity=identity,history=history,final_metric_distance=distances,
                parameter_changes=changes,endpoint_sha256={a:pytree_sha256(params[a]) for a in spec['arms']},
                constituent_metric_distance=constituents,shared_gradient_compile_seconds=shared_compile_seconds,
                arm_seconds=arm_seconds,ledger=ledger,seconds=prior_seconds+perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),
                fixed_endpoint=True,untouched_audit_generated=False))
        else:
            prepared = json.loads((folder/'prepare/results.json').read_text())
            if prepared['initial_sha256'] != pytree_sha256(initial): raise ValueError('initial state changed')
            for name,digest in prepared['files'].items():
                if file_sha256(folder/'prepare'/name) != digest: raise ValueError('prepared corpus changed')
            training = json.loads((folder/'train/results.json').read_text()); scale = prepared['residual_scale']
            endpoints = {a:load_parameter_artifact(folder/'train'/(a+'.json'),initial,
                expected_identity={**identity,'stage':'train','arm':a}) for a in spec['arms']}
            for arm in spec['arms']:
                if pytree_sha256(endpoints[arm]) != training['endpoint_sha256'][arm]: raise ValueError('endpoint hash disagrees with training')
            inputs = {str(p.relative_to(folder)):file_sha256(p) for p in
                [folder/'prepare/results.json',folder/'train/results.json',folder/'train/metric_values.npz']+
                [folder/'train'/(a+'.'+ext) for a in spec['arms'] for ext in ('json','npz')]}
            seal = dict(identity=identity,files=inputs)
            if (dest/'inputs.json').exists():
                if json.loads((dest/'inputs.json').read_text()) != seal: raise ValueError('audit endpoints changed')
            else: atomic_json(dest/'inputs.json',seal)
            if resume:
                if previous_progress is None: raise ValueError('no previous audit progress to resume')
                prior_seconds = previous_progress['seconds']
                ledger.update(previous_progress['ledger'])
                remaining = spec['audit_wall_seconds']-prior_seconds-(perf_counter()-start)
                if remaining <= 0: raise TimeoutError('cumulative audit cap')
                signal.alarm(max(1,int(remaining)))
            if (dest/'samples.npz').exists():
                data = dict(np.load(dest/'samples.npz')); info = json.loads((dest/'sampling.json').read_text())
                if info['samples_sha256'] != file_sha256(dest/'samples.npz'): raise ValueError('audit samples changed')
            else:
                data,info = draw(spec,length**2,carrier,bonds,prepared['carrier_scale'],spec['audit_seed']+case['seed_offset'],case['audit_blocks'])
                np.savez_compressed(dest/'samples.npz',**data)
                info['samples_sha256'] = file_sha256(dest/'samples.npz'); atomic_json(dest/'sampling.json',info)
            progress('fresh_samples_saved_after_endpoint_seal')
            action = make_amplitude_heisenberg_action(lambda b,x:residual(b,x[None])[0]/scale,bonds,n_sites=length**2)
            def hvalues(bank,label):
                file = dest/(label+'_H.npz'); amps=[]; hs=[]; done=0
                bank_hash = pytree_sha256(bank)
                if file.exists():
                    saved = dict(np.load(file)); amps=[saved['amplitudes']]; hs=[saved['h_amplitudes']]; done=len(amps[0])
                    if str(saved['samples_sha256']) != info['samples_sha256'] or str(saved['bank_sha256']) != bank_hash:
                        raise ValueError('partial H belongs to other samples/bank')
                    if hs[0].shape != amps[0].shape or done > len(data['states']) or not np.isfinite(amps[0]).all() or not np.isfinite(hs[0]).all():
                        raise ValueError('invalid partial H checkpoint')
                for i in range(done,len(data['states']),spec['H_microbatch']):
                    a,h = map(np.asarray,action(bank,data['states'][i:i+spec['H_microbatch']]))
                    amps.append(a); hs.append(h); ledger['residual_H_centers']=ledger.get('residual_H_centers',0)+len(a)
                    np.savez_compressed(file,amplitudes=np.concatenate(amps),h_amplitudes=np.concatenate(hs),
                                        samples_sha256=info['samples_sha256'],bank_sha256=bank_hash)
                    progress(label+f'_full_H_{i+len(a)}')
                return np.concatenate(amps),np.concatenate(hs)
            changed = {a:training['endpoint_sha256'][a] != prepared['initial_sha256'] for a in spec['arms']}
            allpsi = [data['carrier']]; allh = [data['hcarrier']*length**2]
            if any(changed.values()): ref,href = hvalues(initial_bank,'reference')
            for arm in spec['arms']:
                if changed[arm]:
                    val,hval = hvalues(generate(endpoints[arm]),arm)
                    allpsi.append(data['carrier']+val-ref); allh.append(data['hcarrier']*length**2+hval-href)
                else: allpsi.append(data['carrier']); allh.append(data['hcarrier']*length**2)
            psi = np.column_stack(allpsi); hp = np.column_stack(allh)
            np.savez_compressed(dest/'values.npz',**data,psi=psi,hpsi=hp)
            report = paired_chain_audit(psi,hp,data['q'],data['groups'],n_sites=length**2,
                bootstrap_replicates=spec['bootstrap_replicates'],seed=spec['bootstrap_seed']+case['seed_offset'],arm_names=['carrier']+spec['arms'])
            distances = {a:weighted_state_distance(psi[:,0],psi[:,k+1],data['q']) for k,a in enumerate(spec['arms'])}
            residuals = {a:ray_residual_concentration(psi[:,0],psi[:,k+1],data['q'],data['groups']) for k,a in enumerate(spec['arms'])}
            report.update(identity=identity,sampling=info,distances=distances,ray_residual_concentration=residuals,
                gate=expansion_gate(report,training,distances,residuals,spec),ledger=ledger,
                carrier_H_centers=len(data['states']),seconds=prior_seconds+perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),
                untouched_fixed_endpoint=True,files={p.name:file_sha256(p) for p in dest.glob('*.npz')})
            atomic_json(dest/'results.json',report)
            print(json.dumps(dict(paired=report['paired'],gate=report['gate']),indent=2),flush=True)
        progress('complete')
    except Exception as error:
        atomic_json(dest/'failure.json',dict(error=repr(error),seconds=prior_seconds+perf_counter()-start,ledger=ledger))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','prepare','train','audit'))
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--length',type=int,choices=(6,8),default=6)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); spec=json.loads(args.config.read_text())
    if args.stage == 'register':
        path=ROOT/spec['output']/'registration.json'
        if path.exists(): raise FileExistsError(path)
        atomic_json(path,registration(spec)); print(path)
    else: run(args.config,args.length,args.stage,args.resume)


if __name__ == '__main__': main()
