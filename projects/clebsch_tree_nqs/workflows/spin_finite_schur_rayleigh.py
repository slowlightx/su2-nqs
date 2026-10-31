"""One full-Rayleigh natural-gradient attempt with actual worst-chain FS matching.

Reuse the sealed architecture, carrier, control cache and amplitude Jacobian.
Transpose the existing H plan for the missing reverse gradient. No new head,
old-audit read, damping search, trajectory extension or remote computation.
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
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import FiniteSchurDecoderConfig, make_finite_schur_amplitude
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.full_rayleigh import (reverse_rayleigh_weights, full_rayleigh_gradient,
    full_rayleigh_sr_direction, common_rayleigh_sr_direction)
from su2_nqs.vmc.coherent_tangent import coherent_tangent_diagnostics
from su2_nqs.vmc.carrier_protection import CarrierProtection, carrier_quality_gate, physical_distance, physical_moments
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from su2_nqs.vmc.cutset_training import paired_chain_audit
from .spin_decoder_cost import ROOT, physical_config, peak_rss_bytes
from .spin_finite_schur_optimize import setup, validate_spec, check_resources, full_products, proposal_spec
from .spin_finite_schur_direction import candidate_products
from .spin_finite_schur_jvp import tangent_records
from .spin_schur_carrier_control import batched, draw

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_rayleigh.json'


def chain_regions(groups: np.ndarray) -> dict[str, np.ndarray]:
    """Full cache, predetermined parity halves and EVERY individual chain."""
    ids = np.asarray(groups)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('one integer chain label per record required')
    labels, counts = np.unique(ids, return_counts=True)
    if len(labels) != 8 or len(set(counts)) != 1 or counts.min() < 2:
        raise ValueError('exactly eight equally represented chains required')
    return dict(full=np.ones(len(ids), bool), half_0=ids%2 == 0, half_1=ids%2 == 1,
                **{f'chain_{int(label)}': ids == label for label in labels})


def total_slope_gate(records: dict) -> dict:
    """Only summed coherent slopes must descend; raw FS ratio is diagnostic."""
    rows = [records['full'], *records['halves']]
    checks = dict(total_descent_full_and_each_half=all(r['energy_slope_per_site'] < 0 for r in rows),
                  nonzero_physical_direction=all(r['tangent_fs_speed'] > 1e-12 for r in rows))
    speeds = [r['tangent_fs_speed'] for r in records['halves']]
    return dict(eligible=bool(all(checks.values())), checks=checks,
        rejected_by=[k for k, v in checks.items() if not v],
        half_speed_ratio=max(speeds)/min(speeds) if min(speeds) > 0 else None,
        individual_coherence_signs_not_a_physical_gate=True,
        raw_speed_ratio_not_a_gate_actual_FS_is_calibrated=True)


def registration(spec: dict) -> dict:
    """Freeze all numerical rules/inputs before reverse-gradient evaluation."""
    validate_spec(spec); CarrierProtection(**spec['protection'])
    if spec['steps'] != 1 or spec['direction_damping'] != .1 or spec['matching_bisections'] != 12:
        raise ValueError('one attempt, fixed .1 damping and13 amplitude probes required')
    if (spec['maximum_q'], spec['multiplicity_width'], spec['embedding_dim'], spec['depth']) != (4,2,16,2):
        raise ValueError('sealed architecture must remain unchanged')
    source, previous, parent, jvp = (ROOT/spec[k] for k in ('direction_source','previous','parent','jvp_source'))
    old = json.loads((source/'registration.json').read_text())['config']
    recent = json.loads((jvp/'registration.json').read_text())['config']
    keys=('geometry','rows','columns','j2_over_j1','multiplicity_width','maximum_q',
          'embedding_dim','depth','projection_chunk','H_chunk','H_microbatch','parent','previous')
    if any(spec[key]!=old[key] for key in keys) or recent['direction_source']!=spec['direction_source']:
        raise ValueError('sealed physical definition and source cache must remain identical')
    if spec['seeds']['audit'] in {*old['seeds'].values(), *recent['seeds'].values()}:
        raise ValueError('fresh conditional audit seed required')
    paths = [source/name for name in ('registration.json','prepared.json')]+[previous/'control.npz']
    paths += [source/arm/'tangents.npz' for arm in ('network','free')]
    paths += [jvp/arm/name for arm in ('network','free') for name in ('products.npz','results.json')]
    paths += [parent/name for name in ('results.json','network.json','network.npz','free_tensors.json','free_tensors.npz')]
    result = dict(config=spec, inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def train(spec: dict, reg: dict, output: Path, arm: str) -> dict:
    """Complete joint-parameter gradient, fixed direction, at most ONE proposal."""
    folder = output/arm; folder.mkdir(exist_ok=True)
    if any(folder.iterdir()): raise FileExistsError('completed/partial artifacts retained; do not overwrite')
    start = perf_counter(); units = json.loads((ROOT/spec['direction_source']/'prepared.json').read_text())
    scale = units['residual_scale']
    with np.load(ROOT/spec['previous']/'control.npz') as d: data = {k:d[k] for k in d.files}
    with np.load(ROOT/spec['direction_source']/arm/'tangents.npz') as d: cached = {k:d[k] for k in d.files}
    jac = cached['control_jacobian']; regions = chain_regions(data['groups'])
    n, initial, generators, projected, _, _, plan, evaluate = setup(spec)
    params, generator = initial[arm], generators[arm]
    flat, unravel = ravel_pytree(params); bank, bank_vjp = jax.vjp(generator, params)
    cfg = FiniteSchurDecoderConfig(replace(physical_config(6,6,2), maximum_intermediate_q=4),3,3,16,2)
    pullback = make_projected_product_pullback(make_finite_schur_amplitude(cfg), n, chunk_size=spec['H_chunk'])
    reverse_halves, terms_halves, half_work = [], [], []
    c, hc, q = data['carrier'], n*data['hcarrier'], data['q']
    for parity in (0,1):
        mask = regions[f'half_{parity}']; states = data['states'][mask]
        weights = scale*reverse_rayleigh_weights(c[mask], q[mask])
        gradient_bank = jax.tree.map(jnp.zeros_like, bank); value = 0.; counts=[]; tick=perf_counter()
        for offset in range(0,len(states),spec['H_microbatch']):
            prepared = plan(jnp.asarray(states[offset:offset+spec['H_microbatch']]))
            coeff = jnp.asarray(weights[offset:offset+spec['H_microbatch']])
            val, derivative, count, padded = pullback(bank, prepared, jnp.zeros_like(coeff), coeff)
            gradient_bank = jax.tree.map(lambda a,b:a+b, gradient_bank, derivative)
            value += float(val); counts.append((int(count),int(padded)))
        reverse, _ = ravel_pytree(bank_vjp(gradient_bank)[0]); reverse=np.asarray(reverse)
        wanted = float(np.real(np.sum(weights*data['reference_H_total'][mask])))
        replay = abs(value-wanted)/max(1.,abs(wanted))
        if replay > 2e-10: raise FloatingPointError('transposed-plan objective replay fails')
        terms = full_rayleigh_gradient(c[mask],hc[mask],jac[mask],q[mask],reverse)
        reverse_halves.append(reverse); terms_halves.append(terms)
        half_work.append(dict(centers=len(states),seconds=perf_counter()-tick,
            unique_raw_queries=sum(x[0] for x in counts),padded_raw_queries=sum(x[1] for x in counts),
            objective_replay_relative_error=replay,explicit_chunked_pullback=True))
        check_resources(spec,start)
    norm = sum(t['norm'] for t in terms_halves)
    reverse = sum(t['norm']*g for t,g in zip(terms_halves,reverse_halves))/norm
    terms = full_rayleigh_gradient(c,hc,jac,q,reverse)
    # Replay against previous independent full-H JVPs; no H work or audit read.
    old = json.loads((ROOT/spec['jvp_source']/arm/'results.json').read_text())
    replay = {}
    with np.load(ROOT/spec['jvp_source']/arm/'products.npz') as d:
        for name,row in old['records'].items():
            vector = -d[name+'_direction']
            errors = [abs(terms['gradient']@vector/n-row['diagnostic']['full']['energy_slope_per_site'])]
            errors += [abs(t['gradient']@vector/n-r['energy_slope_per_site'])
                       for t,r in zip(terms_halves,row['diagnostic']['halves'])]
            if max(errors) > 2e-9: raise FloatingPointError('full-gradient dot direction disagrees with independent old JVP')
            replay[name] = max(errors)
    pooled_direction, pooled_sr = full_rayleigh_sr_direction(c,hc,jac,q,terms['gradient'],damping=spec['direction_damping'])
    objective_gradients=np.stack([terms['gradient'],*[t['gradient'] for t in terms_halves]])
    direction, sr = common_rayleigh_sr_direction(c,hc,jac,q,objective_gradients,damping=spec['direction_damping'])
    pooled_sr['predicted_half_energy_slopes_per_site']=[float(-t['gradient']@pooled_direction/n) for t in terms_halves]
    fold = []
    for parity in (0,1):
        mask=regions[f'half_{parity}']; other=1-parity
        fd, fi=full_rayleigh_sr_direction(c[mask],hc[mask],jac[mask],q[mask],terms_halves[parity]['gradient'],damping=.1)
        fold.append(dict(fit_half=parity,fit=fi,other_half_energy_slope_per_site=float(-terms_halves[other]['gradient']@fd/n),
                         diagnostic_only=True,used_to_select_direction=False))
    dp = unravel(-jnp.asarray(direction)); _, dbank = jax.jvp(generator,(params,),(dp,))
    jvp = make_projected_product_jvp(evaluate); plans=[]; delta=[]; hdelta=[]; jvp_work=[]; tick=perf_counter()
    for offset in range(0,len(q),spec['H_microbatch']):
        prepared=plan(jnp.asarray(data['states'][offset:offset+spec['H_microbatch']]))
        plans.append(prepared)
        _,_,count,padded,dh,d=jvp(bank,dbank,prepared)
        delta.append(np.asarray(d)*scale); hdelta.append(np.asarray(dh)*scale)
        jvp_work.append((int(count),int(padded)))
    delta,hdelta=np.concatenate(delta),np.concatenate(hdelta); jvp_seconds=perf_counter()-tick
    tangent_error=float(np.linalg.norm(delta+jac@direction)/max(np.linalg.norm(delta),1e-30))
    record=tangent_records(data,delta,hdelta,n)
    slope_error=abs(record['full']['energy_slope_per_site']+terms['gradient']@direction/n)
    if tangent_error>2e-9 or slope_error>2e-9: raise FloatingPointError('new complete-direction JVP/gradient identity fails')
    # Independent actual-parameter finite differences on the FIRST four centers.
    eps=spec['finite_difference_step']/max(1.,np.linalg.norm(direction));m=spec['H_microbatch']
    plus=evaluate(generator(unravel(flat-eps*jnp.asarray(direction))),plans[0])
    minus=evaluate(generator(unravel(flat+eps*jnp.asarray(direction))),plans[0])
    fd_p=np.asarray(plus[1]-minus[1])*scale/(2*eps); fd_h=np.asarray(plus[0]-minus[0])*scale/(2*eps)
    fd_errors=dict(amplitude=float(np.linalg.norm(fd_p-delta[:m])/max(np.linalg.norm(delta[:m]),1e-30)),
                   H_amplitude=float(np.linalg.norm(fd_h-hdelta[:m])/max(np.linalg.norm(hdelta[:m]),1e-30)))
    if max(fd_errors.values())>2e-6: raise FloatingPointError('actual-parameter finite difference fails')
    eligibility=total_slope_gate(record); policy=CarrierProtection(**spec['protection'])
    trial=dict(raw_candidate_constructed=False,accepted=False,cache_accepted=False,eligibility=eligibility,
               candidate_H_centers=0,final_parameter_sha256=pytree_sha256(params))
    matching=None; candidate_work=None
    if eligibility['eligible']:
        # Linear speed is only a starting bracket; EVERY probe measures nonlinear distances.
        speed=max([record['full']['tangent_fs_speed'],*[r['tangent_fs_speed'] for r in record['halves']],
                   *[r['tangent_fs_speed'] for r in record['chains'].values()]])
        maximum=min(spec['direction_step_size'],2*policy.maximum_step_fs/max(speed,1e-30))
        def candidate(alpha):
            b=generator(unravel(flat-alpha*jnp.asarray(direction)))
            return c+scale*(batched(lambda x:projected(b,x),data['states'],8)-data['reference_values'])
        matched=matched_chain_state_step(candidate,c,q,regions,policy.maximum_step_fs,maximum,spec['matching_bisections'],
                                         relative_tolerance=spec['matching_relative_tolerance'])
        proposed=unravel(flat-matched.alpha*jnp.asarray(direction))
        p,hp,candidate_work=candidate_products(generator(proposed),data,plans,evaluate,scale,n)
        replay_error=float(np.max(abs(p-matched.amplitudes)))
        if replay_error>2e-10*max(1.,float(np.max(abs(p)))): raise FloatingPointError('matched/H amplitude replay fails')
        gates=[]
        for name in ('full','half_0','half_1'):
            mask=regions[name]
            gates.append(carrier_quality_gate(c[mask],hc[mask],c[mask],hc[mask],p[mask],hp[mask],q[mask],data['groups'][mask],n,policy))
        cache_accepted=bool(matched.search['matched'] and all(g['accepted'] for g in gates))
        save_parameter_artifact(folder/'candidate.json',proposed,identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,kind='raw_full_Rayleigh_proposal'))
        np.savez_compressed(folder/'candidate_control.npz',psi=p,hpsi=hp)
        matching=matched.as_dict()
        trial.update(raw_candidate_constructed=True,cache_accepted=cache_accepted,matching=matching,quality_gates=gates,
            candidate_H_centers=len(p),candidate_parameter_sha256=pytree_sha256(proposed),
            candidate_sha256=file_sha256(folder/'candidate.json'),amplitude_H_replay_error=replay_error)
    arrays=dict(full_gradient=terms['gradient'],full_forward=terms['forward'],full_reverse=terms['reverse'],
                reverse_raw=reverse,direction=direction,pooled_direction=pooled_direction,delta=delta,hdelta=hdelta,control_jacobian=jac)
    for parity,t in enumerate(terms_halves):
        arrays[f'half_{parity}_gradient']=t['gradient'];arrays[f'half_{parity}_reverse_raw']=reverse_halves[parity]
    np.savez_compressed(folder/'products.npz',**arrays)
    result=dict(registration_sha256=reg['registration_sha256'],arm=arm,trial=trial,SR=sr,pooled_SR=pooled_sr,crossfold=fold,
        tangent=record,old_JVP_slope_replay_errors=replay,pullback_work=half_work,
        JVP_H_centers=len(q),JVP_seconds=jvp_seconds,JVP_unique_raw_queries=sum(x[0] for x in jvp_work),
        finite_difference_H_centers=2*m,finite_difference_relative_errors=fd_errors,
        cached_Jacobian_relative_error=tangent_error,new_JVP_slope_abs_error=slope_error,candidate_work=candidate_work,
        product_sha256=file_sha256(folder/'products.npz'),new_training_samples=0,control_used_for_training=True,
        old_audit_not_read=True,architecture_unchanged=True,peak_rss_bytes=peak_rss_bytes())
    # No commit before the single fresh conditional audit. This checkpoint is the protected parent.
    identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='proposal')
    save_gated_checkpoint(folder/'proposal_checkpoint.pkl',params,optimizer_state=dict(direction=direction),run_identity=identity,decision=trial)
    atomic_json(folder/'results.json',result)
    print(arm,'full/half slopes',[record['full']['energy_slope_per_site'],*[r['energy_slope_per_site'] for r in record['halves']]],
          'raw speed ratio',eligibility['half_speed_ratio'],'cache accepted',trial['cache_accepted'],flush=True)
    return result


def finalize(spec: dict, reg: dict, output: Path) -> dict:
    """One fresh common audit only if a cache-passing proposal exists; no refit."""
    if (output/'results.json').exists(): raise FileExistsError('final endpoints sealed')
    rows={arm:json.loads((output/arm/'results.json').read_text()) for arm in ('network','free')}
    n,initial,generators,_,raw_carrier,bonds,plan,evaluate=setup(spec)
    units=json.loads((ROOT/spec['direction_source']/'prepared.json').read_text());scale=units['residual_scale']
    constructed=[arm for arm,r in rows.items() if r['trial']['raw_candidate_constructed']]
    passing=[arm for arm,r in rows.items() if r['trial']['cache_accepted']]
    accepted={arm:False for arm in rows}; audit_result=None
    if passing:
        data,info=draw(proposal_spec(spec,'audit'),n,raw_carrier,bonds,units['cscale'],spec['seeds']['audit'],spec['audit']['records'])
        c,hc=data['carrier'],n*data['hcarrier'];positions={tuple(map(int,x)):i for i,x in enumerate(data['states'])}
        def action(x):
            ids=[positions[tuple(map(int,row))] for row in np.asarray(x)]
            return c[ids],hc[ids]
        values,products=[c],[hc];works={};gates={};distances={}
        reference=generators['network'](initial['network']);regions=chain_regions(data['groups']);policy=CarrierProtection(**spec['protection'])
        for arm in constructed:
            params=load_parameter_artifact(output/arm/'candidate.json',initial[arm])
            p,hp,work=full_products(generators[arm](params),reference,data['states'],action,plan,evaluate,
                scale=scale,microbatch=spec['H_microbatch'])
            values.append(p);products.append(hp);works[arm]=work
            distances[arm]={name:physical_distance(c[mask],p[mask],data['q'][mask]) for name,mask in regions.items()}
            gate=carrier_quality_gate(c,hc,c,hc,p,hp,data['q'],data['groups'],n,policy)
            gate['control_used_for_training']=False;gate['fresh_audit_used_for_acceptance']=True
            gates[arm]=gate
            accepted[arm]=bool(arm in passing and gate['accepted'] and
                max(d['fs'] for d in distances[arm].values())<=policy.maximum_step_fs*(1+1e-8))
        values,products=np.stack(values,axis=1),np.stack(products,axis=1)
        metrics=paired_chain_audit(values,products,data['q'],data['groups'],n,spec['audit']['bootstrap_replicates'],spec['seeds']['bootstrap'],['carrier',*constructed])
        np.savez_compressed(output/'audit.npz',**data,psi=values,hpsi=products)
        audit_result=dict(metrics=metrics,gates=gates,distances=distances,full_H_work=works,sampling=info,
            audit_sha256=file_sha256(output/'audit.npz'),audit_used_for_gradients=False,audit_used_for_acceptance=True,
            raw_candidates_include_cache_rejections=True,no_post_audit_refit=True)
        atomic_json(output/'audit.json',audit_result)
    for arm,row in rows.items():
        folder=output/arm;params=initial[arm]
        if accepted[arm]: params=load_parameter_artifact(folder/'candidate.json',initial[arm])
        decision={**row['trial'],'accepted':accepted[arm],'final_parameter_sha256':pytree_sha256(params),
                  'fresh_audit_performed':audit_result is not None,'initial_parameter_sha256':pytree_sha256(initial[arm])}
        identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='final')
        save_parameter_artifact(folder/'endpoint.json',params,identity=identity)
        with np.load(folder/'products.npz') as d:direction=d['direction']
        save_gated_checkpoint(folder/'checkpoint.pkl',params,optimizer_state=dict(direction=direction),run_identity=identity,decision=decision)
        atomic_json(folder/'decision.json',decision)
    result=dict(registration_sha256=reg['registration_sha256'],accepted_updates=sum(accepted.values()),accepted=accepted,
        cache_passing=passing,constructed=constructed,new_training_samples=0,
        new_audit_records=spec['audit']['chains']*spec['audit']['records'] if audit_result else 0,
        new_audit_H_centers=sum(w['centers'] for w in audit_result['full_H_work'].values()) if audit_result else 0,
        audit_skip_reason=None if audit_result else 'no proposal passed fixed cache energy/variance and actual chain-FS bounds',
        new_ED=False,remote_submitted=False,architecture_unchanged=True)
    atomic_json(output/'results.json',result);print(json.dumps(result,indent=2),flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--stage',choices=('register','train','finalize'),required=True)
    parser.add_argument('--arm',choices=('network','free'));args=parser.parse_args()
    spec=json.loads(args.config.read_text());reg=registration(spec);output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists():raise FileExistsError('registration sealed')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg!=json.loads((output/'registration.json').read_text()):raise ValueError('numerical registration mismatch')
    if args.stage=='train' and args.arm is None:parser.error('--arm required')
    start=perf_counter()
    def alarm(_signal,_frame):raise TimeoutError('registered process limit exceeded')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='train':train(spec,reg,output,args.arm)
        else:finalize(spec,reg,output)
        check_resources(spec,start)
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',dict(wall_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start));raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
