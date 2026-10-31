"""Full-H JVP/two-sided coherence gate before unchanged finite-Schur updates.

Reuses sealed directions, parent banks and exact H plans. Crossfold directions
are diagnostic only. No new fit, sign selection or audit-driven continuation.
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

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact,save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,numerical_runtime_identity,transitive_local_source_manifest
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.coherent_tangent import coherent_tangent_diagnostics,CoherentTangentPolicy,cross_chain_tangent_gate
from su2_nqs.vmc.importance_tangent import importance_sr_direction
from su2_nqs.vmc.carrier_protection import CarrierProtection,carrier_quality_gate,physical_moments,physical_distance
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.matched_state_step import matched_state_step
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint
from .spin_decoder_cost import ROOT,peak_rss_bytes
from .spin_finite_schur_optimize import setup,validate_spec,full_products,check_resources,proposal_spec
from .spin_finite_schur_protected import interpolate
from .spin_finite_schur_direction import candidate_products
from .spin_schur_carrier_control import batched,draw

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_jvp.json'


def registration(spec: dict) -> dict:
    """Seal control-only numerical inputs and rules BEFORE evaluating H tangents."""
    validate_spec(spec);CoherentTangentPolicy(**spec['tangent_policy']);CarrierProtection(**spec['protection'])
    if spec['methods']!=['born','importance'] or spec['steps']!=1 or spec['finite_difference_microbatches']!=1:
        raise ValueError('fixed two methods, one attempt and one FD microbatch required')
    source,control,parent=ROOT/spec['direction_source'],ROOT/spec['previous'],ROOT/spec['parent']
    old=json.loads((source/'registration.json').read_text())['config']
    if spec['seeds']['audit'] in old['seeds'].values():raise ValueError('fresh conditional audit seed required')
    paths=[source/name for name in ('registration.json','prepared.json')]
    paths += [source/arm/name for arm in ('network','free') for name in ('results.json','tangents.npz')]
    paths += [control/'control.npz']
    paths += [parent/name for name in ('results.json','network.json','network.npz','free_tensors.json','free_tensors.npz')]
    result=dict(config=spec,inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(),sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    result['registration_sha256']=canonical_sha256(result)
    return result


def tangent_records(data: dict, delta: np.ndarray, hdelta: np.ndarray, n: int) -> dict:
    """Whole, predetermined chain halves and descriptive individual-chain moments."""
    c,hc,q,ids=data['carrier'],n*data['hcarrier'],data['q'],data['groups']
    diagnostic=lambda mask:coherent_tangent_diagnostics(c[mask],hc[mask],delta[mask],hdelta[mask],q[mask],n)
    return dict(full=diagnostic(np.ones(len(q),bool)),
        halves=[diagnostic(ids%2==parity) for parity in (0,1)],
        chains={str(int(label)):diagnostic(ids==label) for label in np.unique(ids)})


def probe(spec: dict,reg: dict,output: Path,arm: str) -> dict:
    """Bounded JVP validation and conditional updates; no new training data."""
    folder=output/arm;folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists() or any(folder.glob('*/endpoint.json')):
        raise FileExistsError('completed/partial endpoints kept; do not overwrite')
    start=perf_counter();source=ROOT/spec['direction_source']
    units=json.loads((source/'prepared.json').read_text());scale=units['residual_scale']
    with np.load(ROOT/spec['previous']/'control.npz') as d:data={k:d[k] for k in d.files}
    with np.load(source/arm/'tangents.npz') as d:cached={k:d[k] for k in d.files}
    n,initial,generators,projected,_,_,plan,evaluate=setup(spec)
    params=initial[arm];generator=generators[arm];flat,unravel=ravel_pytree(params)
    bank=generator(params);jvp=make_projected_product_jvp(evaluate)
    plans=[plan(jnp.asarray(data['states'][i:i+spec['H_microbatch']]))
           for i in range(0,len(data['q']),spec['H_microbatch'])]
    jax.block_until_ready(plans)
    directions={m:cached[m+'_direction'] for m in spec['methods']}
    for parity in (0,1):
        mask=data['groups']%2==parity
        directions[f'fit_{parity}'],_=importance_sr_direction(data['carrier'][mask],n*data['hcarrier'][mask],
            cached['control_jacobian'][mask],data['q'][mask],damping=spec['direction_damping'])
    policy=CoherentTangentPolicy(**spec['tangent_policy']);records={};arrays={}
    for name,direction in directions.items():
        dp=unravel(-jnp.asarray(direction)) # Frozen descent orientation, NOT refitted or flipped.
        _,dbank=jax.jvp(generator,(params,),(dp,))
        tick=perf_counter();dh_rows=[];d_rows=[];p_rows=[];h_rows=[];queries=[]
        for prepared in plans:
            h,p,count,padded,dh,d=jvp(bank,dbank,prepared)
            dh_rows.append(np.asarray(dh)*scale);d_rows.append(np.asarray(d)*scale)
            p_rows.append(np.asarray(p));h_rows.append(np.asarray(h));queries.append((int(count),int(padded)))
        delta,hdelta=np.concatenate(d_rows),np.concatenate(dh_rows)
        jvp_seconds=perf_counter()-tick
        expected=-cached['control_jacobian']@direction
        amplitude_error=float(np.linalg.norm(delta-expected)/max(np.linalg.norm(expected),1e-30))
        if amplitude_error>2e-9:raise FloatingPointError('H kernel JVP disagrees with cached amplitude Jacobian')
        values=tangent_records(data,delta,hdelta,n)
        gate=cross_chain_tangent_gate(values['halves'],policy)
        row=dict(name=name,diagnostic=values,eligibility=gate,diagnostic_only=name.startswith('fit_'),
            amplitude_jacobian_relative_error=amplitude_error,JVP_H=dict(seconds=jvp_seconds,centers=len(delta),
                primal_plus_tangent=True,unique_raw_queries=sum(x[0] for x in queries),
                padded_raw_queries=sum(x[1] for x in queries)),direction_parameter_norm=float(np.linalg.norm(direction)))
        # The global directions have a fixed actual-param finite-difference check.
        if name in spec['methods']:
            eps=spec['finite_difference_step']/max(1.,np.linalg.norm(direction));prepared=plans[0]
            tick=perf_counter();plus=evaluate(generator(unravel(flat-eps*jnp.asarray(direction))),prepared)
            minus=evaluate(generator(unravel(flat+eps*jnp.asarray(direction))),prepared)
            fd_h=np.asarray(plus[0]-minus[0])*scale/(2*eps)
            fd_p=np.asarray(plus[1]-minus[1])*scale/(2*eps);m=spec['H_microbatch']
            errors=dict(amplitude=float(np.linalg.norm(fd_p-delta[:m])/max(np.linalg.norm(delta[:m]),1e-30)),
                H_amplitude=float(np.linalg.norm(fd_h-hdelta[:m])/max(np.linalg.norm(hdelta[:m]),1e-30)))
            if max(errors.values())>2e-6:raise FloatingPointError('actual parameter H/psi finite difference fails')
            p0,h0=np.concatenate(p_rows)[:m],np.concatenate(h_rows)[:m]
            c,hc,q=data['carrier'][:m],n*data['hcarrier'][:m],data['q'][:m]
            pm=c+scale*(np.asarray(plus[1])-p0);hm=hc+scale*(np.asarray(plus[0])-h0)
            pp=c+scale*(np.asarray(minus[1])-p0);hp=hc+scale*(np.asarray(minus[0])-h0)
            em=physical_moments(pm,hm,q,n);ep=physical_moments(pp,hp,q,n)
            slopes=coherent_tangent_diagnostics(c,hc,delta[:m],hdelta[:m],q,n)
            observed=(em['energy_per_site']-ep['energy_per_site'])/(2*eps)
            error=abs(observed-slopes['energy_slope_per_site'])
            if error>2e-6*max(1.,abs(slopes['energy_slope_per_site'])):raise FloatingPointError('two-sided Rayleigh derivative fails')
            row['finite_difference']=dict(relative_errors=errors,energy_slope_per_site=observed,
                two_sided_slope_abs_error=error,epsilon=float(eps),forward_H_centers=2*m,
                seconds=perf_counter()-tick)
        arrays[name+'_delta']=delta;arrays[name+'_hdelta']=hdelta;arrays[name+'_direction']=direction
        records[name]=row
        print(arm,name,'two-sided half slopes',[r['energy_slope_per_site'] for r in values['halves']],
            'speed ratio',gate['speed_ratio'],'eligible',gate['eligible'],'reject',gate['rejected_by'],flush=True)
        check_resources(spec,start)
    np.savez_compressed(folder/'products.npz',**arrays)
    protection=CarrierProtection(**spec['protection']);updates={}
    for method in spec['methods']:
        destination=folder/method;destination.mkdir(exist_ok=True)
        eligibility=records[method]['eligibility'];committed=params
        trial=dict(accepted=False,raw_candidate_constructed=False,eligibility=eligibility,
                   candidate_H_centers=0,source_direction_unchanged=True)
        if eligibility['eligible']:
            direction=directions[method];target=unravel(flat-spec['direction_step_size']*jnp.asarray(direction))
            full_speed=records[method]['diagnostic']['full']['tangent_fs_speed']
            max_speed=max(r['tangent_fs_speed'] for r in records[method]['diagnostic']['halves'])
            radius=protection.maximum_step_fs*min(1.,full_speed/max_speed)
            def candidate(alpha):
                b=generator(interpolate(params,target,alpha))
                return data['carrier']+scale*(batched(lambda x:projected(b,x),data['states'],8)-data['reference_values'])
            match=matched_state_step(candidate,data['carrier'],data['q'],radius,1.,spec['matching_bisections'],
                                     relative_tolerance=spec['matching_relative_tolerance'])
            proposed=interpolate(params,target,match.alpha)
            p,hp,work=candidate_products(generator(proposed),data,plans,evaluate,scale,n)
            decisions=[]
            for mask in (np.ones(len(p),bool),data['groups']%2==0,data['groups']%2==1):
                c,hc=data['carrier'][mask],n*data['hcarrier'][mask]
                decisions.append(carrier_quality_gate(c,hc,c,hc,p[mask],hp[mask],data['q'][mask],data['groups'][mask],n,protection))
            accepted=bool(match.matched and all(g['accepted'] for g in decisions))
            if accepted:committed=proposed
            trial.update(accepted=accepted,raw_candidate_constructed=True,matching=match.as_dict(),
                         quality_gates_full_even_odd=decisions,candidate_H=work,candidate_H_centers=work['centers'])
            save_parameter_artifact(destination/'candidate.json',proposed,
                identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,method=method,kind='gated_candidate'))
            np.savez_compressed(destination/'control.npz',psi=p,hpsi=hp)
        identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,method=method)
        save_parameter_artifact(destination/'endpoint.json',committed,identity=identity)
        trial['endpoint_sha256']=file_sha256(destination/'endpoint.json')
        trial['initial_parameter_sha256']=pytree_sha256(params);trial['final_parameter_sha256']=pytree_sha256(committed)
        save_resumable_checkpoint(destination/'checkpoint.pkl',committed,
            optimizer_state=dict(frozen_direction=directions[method],pre_update_gate=eligibility),
            completed_steps=int(trial['raw_candidate_constructed']),next_sample_index=0,run_identity=identity,history=[trial])
        updates[method]=trial
    result=dict(registration_sha256=reg['registration_sha256'],arm=arm,records=records,updates=updates,
        products_sha256=file_sha256(folder/'products.npz'),JVP_H_centers=sum(r['JVP_H']['centers'] for r in records.values()),
        finite_difference_H_centers=sum(r.get('finite_difference',{}).get('forward_H_centers',0) for r in records.values()),
        candidate_H_centers=sum(u['candidate_H_centers'] for u in updates.values()),
        new_training_samples=0,consumed_audit_read=False,architecture_unchanged=True,peak_rss_bytes=peak_rss_bytes())
    atomic_json(folder/'results.json',result)
    return result


def finalize(spec: dict,reg: dict,output: Path) -> dict:
    """Audit accepted updates only; do NOT sample when every endpoint is carrier."""
    if (output/'results.json').exists():raise FileExistsError('finalized decision sealed')
    runs={a:json.loads((output/a/'results.json').read_text()) for a in ('network','free')}
    accepted=[(a,m) for a,r in runs.items() for m,u in r['updates'].items() if u['accepted']]
    result=dict(registration_sha256=reg['registration_sha256'],accepted_updates=len(accepted),
        accepted_names=[a+'_'+m for a,m in accepted],new_audit_records=0,new_audit_H_centers=0,
        new_ED=False,remote_submitted=False)
    if accepted:
        n,initial,generators,_,carrier,bonds,plan,evaluate=setup(spec)
        units=json.loads((ROOT/spec['direction_source']/'prepared.json').read_text());reference=generators['network'](initial['network'])
        data,info=draw(proposal_spec(spec,'audit'),n,carrier,bonds,units['cscale'],spec['seeds']['audit'],spec['audit']['records'])
        c,hc=data['carrier'],n*data['hcarrier'];positions={tuple(map(int,x)):i for i,x in enumerate(data['states'])}
        def action(x):
            idx=[positions[tuple(map(int,row))] for row in np.asarray(x)]
            return c[idx],hc[idx]
        psi,hpsi=[c],[hc];names=['carrier'];works={};distances={}
        for arm,method in accepted:
            params=load_parameter_artifact(output/arm/method/'endpoint.json',initial[arm])
            p,hp,work=full_products(generators[arm](params),reference,data['states'],action,plan,evaluate,
                                  scale=units['residual_scale'],microbatch=spec['H_microbatch'])
            name=arm+'_'+method;psi.append(p);hpsi.append(hp);names.append(name);works[name]=work
            distances[name]=physical_distance(c,p,data['q'])
        psi,hpsi=np.stack(psi,axis=1),np.stack(hpsi,axis=1)
        metrics=paired_chain_audit(psi,hpsi,data['q'],data['groups'],n,spec['audit']['bootstrap_replicates'],spec['seeds']['bootstrap'],names)
        np.savez_compressed(output/'audit.npz',**data,psi=psi,hpsi=hpsi)
        atomic_json(output/'audit.json',dict(metrics=metrics,distances=distances,full_H_work=works,sampling=info,
            audit_sha256=file_sha256(output/'audit.npz'),audit_used_for_training=False))
        result.update(new_audit_records=len(c),new_audit_H_centers=sum(w['centers'] for w in works.values()))
    else:result['audit_skipped_reason']='all unchanged endpoints equal original carrier; do not append same-family sampling'
    atomic_json(output/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--stage',choices=('register','probe','finalize'),required=True)
    parser.add_argument('--arm',choices=('network','free'));args=parser.parse_args()
    spec=json.loads(args.config.read_text());reg=registration(spec);output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists():raise FileExistsError('registration sealed')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg!=json.loads((output/'registration.json').read_text()):raise ValueError('numerical registration mismatch')
    if args.stage=='probe' and args.arm is None:parser.error('--arm required')
    start=perf_counter()
    def alarm(_signal,_frame):raise TimeoutError('registered process limit exceeded')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='probe':probe(spec,reg,output,args.arm)
        else:finalize(spec,reg,output)
        check_resources(spec,start)
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',dict(wall_seconds=perf_counter()-start,
            peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start))
        raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
