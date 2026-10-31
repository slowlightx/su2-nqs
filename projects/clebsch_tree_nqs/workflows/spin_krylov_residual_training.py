"""Fixed-budget full-parameter neural/free residual training at frozen K2.

Reuse the existing decoder, H plans/pullbacks, FS/SR, checkpoints and audits.
Known optimizer records never become validation; audit cannot alter endpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.operator_word_jax import evaluate_scalar_word_action_plan_numpy
from su2_nqs.benchmarks.spin_power_bridge import build_bounded_spin_power_plan
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact,load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,numerical_runtime_identity,transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import FiniteSchurDecoderConfig,make_finite_schur_amplitude
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products,carrier_span_diagnostics
from su2_nqs.vmc.carrier_protection import physical_moments,physical_distance
from su2_nqs.vmc.cross_sample_geometry import observable_common_direction
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.joint_linear_step import linear_moment_slopes
from su2_nqs.vmc.joint_moments import residual_moment_cotangents
from su2_nqs.vmc.importance_tangent import make_amplitude_jacobian
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.rayleigh_cotangents import energy_moment_cotangents
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint,load_resumable_checkpoint
from .spin_finite_schur_optimize import setup
from .spin_decoder_cost import physical_config,peak_rss_bytes
from .spin_schur_carrier_control import batched,draw
from . import spin_krylov_carrier_bridge as frozen

jax.config.update('jax_enable_x64',True)
ROOT=frozen.ROOT
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_krylov_residual_training.json'
ARMS=('network','free')


def loaded(path: Path) -> dict:
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def register(spec: dict,output: Path) -> None:
    """Seal the strong parent, immutable architecture/data, bounds and audit seed."""
    if (spec['maximum_attempts_per_arm']!=8 or spec['audit']['chains']*spec['audit']['records']!=128
            or (spec['rows'],spec['columns'],spec['multiplicity_width'],spec['maximum_q'],
                spec['embedding_dim'],spec['depth'],spec['j2_over_j1'])!=(6,6,2,4,16,2,.125)
            or sum(spec['stage_seconds'].values())!=spec['wall_limit_total_seconds']):
        raise ValueError('registered same-architecture local budget required')
    bridge=ROOT/spec['bridge']; old=frozen.read(bridge/'registration.json');frozen.verify(old)
    paths=[CONFIG,ROOT/spec['known_source'],ROOT/spec['units_source'],
           bridge/'registration.json',bridge/'results.json']
    paths += [ROOT/spec['parent']/name for name in ('results.json','network.json','network.npz','free_tensors.json','free_tensors.npz')]
    reg=dict(config=spec,runtime=numerical_runtime_identity(),
        K2_coefficients=old['polynomial_coefficients'],inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    reg['registration_sha256']=canonical_sha256(reg)
    output.mkdir(exist_ok=False);atomic_json(output/'registration.json',reg)
    atomic_json(output/'budget.json',dict(seconds=0.,completed_stages=[]))
    print('sealed K2/full neural/free training and untouched audit',reg['registration_sha256'],flush=True)


def power_products(states: np.ndarray,carrier,bonds,spec: dict,folder: Path,check) -> tuple[np.ndarray,dict]:
    """Checkpoint exact analytic C..H³C per row; no clipped small terms."""
    folder.mkdir(exist_ok=True);start=perf_counter();rows=[];work=[];batch=spec['carrier_amplitude_batch']
    def amplitude(x):
        check(); pad=np.pad(x,((0,batch-len(x)),(0,0)),mode='edge')
        return np.asarray(carrier(jnp.asarray(pad)))[:len(x)]
    for i,state in enumerate(states):
        path=folder/f'row{i:03d}.npz';meta=folder/f'row{i:03d}.json'
        if path.exists() and meta.exists():
            saved=loaded(path)
            if not np.array_equal(saved['state'],state):raise ValueError('power checkpoint state mismatch')
            rows.append(saved['powers']);work.append(frozen.read(meta));continue
        tick=perf_counter();check()
        plan,record=build_bounded_spin_power_plan(state[None],bonds,maximum_power=3,
            maximum_expansion_entries=spec['maximum_expansion_entries'],maximum_terminal_queries=spec['maximum_terminal_queries'])
        values=evaluate_scalar_word_action_plan_numpy(plan,amplitude,amplitude_batch_size=batch)[0]
        np.savez_compressed(path,state=state,powers=values)
        record['seconds']=perf_counter()-tick;atomic_json(meta,record);rows.append(values);work.append(record)
        if (i+1)%16==0:print('carrier powers',i+1,'/',len(states),round(perf_counter()-start,2),'s',flush=True)
    return np.stack(rows),dict(seconds=perf_counter()-start,centers=len(states),
        exact_algebraic_H_applications=3*len(states),unique_terminal_queries=sum(w['unique_terminal_queries'] for w in work),
        host_plan_seconds=sum(w['host_plan_seconds'] for w in work),per_row=work)


def prepare(spec: dict,reg: dict,output: Path,check) -> None:
    if (output/'prepared.json').exists():raise ValueError('preparation already complete')
    known=loaded(ROOT/spec['known_source']);units=frozen.read(ROOT/spec['units_source'])
    n,initial,generators,projected,carrier,bonds,_,_=setup(spec)
    if len(known['states'])!=64 or len(np.unique(known['groups']))!=8:
        raise ValueError('the same64 known mixture records required')
    powers,work=power_products(known['states'],carrier,bonds,spec,output/'known_powers',check)
    powers/=units['cscale']
    np.testing.assert_allclose(powers[:,:3],known['base_products'],rtol=2e-8,atol=2e-8)
    k,hk=polynomial_carrier_products(powers,reg['K2_coefficients'])
    reference=generators['network'](initial['network'])
    phi=batched(lambda x:projected(reference,x),known['states'],8)
    np.testing.assert_allclose(phi,known['reference_values'],rtol=2e-8,atol=1e-16)
    scale=float(np.sqrt(np.sum(abs(k)**2/known['q'])/np.sum(abs(phi)**2/known['q'])))
    if not np.isfinite(scale) or scale<=0:raise ValueError('invalid K2-relative residual units')
    data={k:v for k,v in known.items() if k not in ('scale_states','base_products','carrier','hcarrier')}
    data.update(carrier=k,hcarrier=hk/n,base_products=powers,reference_values=phi)
    np.savez_compressed(output/'known.npz',**data)
    record=dict(registration_sha256=reg['registration_sha256'],cscale=units['cscale'],residual_scale=scale,
        norm_units_only_no_energy_fit=True,initial_bank_sha256=pytree_sha256(reference),
        known_sha256=file_sha256(output/'known.npz'),carrier_work=work,
        initial_parameter_counts={a:int(ravel_pytree(initial[a])[0].size) for a in ARMS},
        known_data_not_validation=True,architecture_unchanged=True)
    atomic_json(output/'prepared.json',record)
    print('K2 and reference sealed; parameter counts',record['initial_parameter_counts'],flush=True)


def accept_candidate(p: np.ndarray,h: np.ndarray,old_p: np.ndarray,old_h: np.ndarray,
                     data: dict,spec: dict,matching: dict) -> dict:
    """Fixed known-data controls, not an independent statistical certificate."""
    fit=data['groups']%2==0;control=~fit;metrics={}
    flags={}
    for label,mask in (('fit',fit),('control',control)):
        before=physical_moments(old_p[mask],old_h[mask],data['q'][mask],36)
        after=physical_moments(p[mask],h[mask],data['q'][mask],36)
        anchor=physical_moments(data['carrier'][mask],36*data['hcarrier'][mask],data['q'][mask],36)
        metrics[label]=dict(before=before,after=after,K2=anchor)
        flags[label+'_energy']=after['energy_per_site']<before['energy_per_site']-1e-12 if label=='fit' else after['energy_per_site']<=before['energy_per_site']+spec['known_control_energy_tolerance_per_site']
        flags[label+'_variance']=after['variance_per_site']<=spec['known_control_variance_ratio_to_K2']*anchor['variance_per_site']+1e-12
    distance=physical_distance(data['carrier'],p,data['q'])
    flags.update(nonzero_step=matching['alpha']>0 and matching['distances']['full']['fs']>1e-12,
        physical_step_cap=all(v['fs']<=spec['target_step_FS']*(1+1e-8) for v in matching['distances'].values()),
        carrier_distance=distance['fs']<=spec['maximum_carrier_FS'],
        importance=distance['ess_fraction']>=spec['minimum_ESS_fraction'])
    return dict(accepted=all(flags.values()),checks=flags,metrics=metrics,
        carrier_distance=distance,known_control_not_validation=True)


def train(spec: dict,reg: dict,output: Path,arm: str,check) -> None:
    folder=output/arm;folder.mkdir(exist_ok=True)
    if (folder/'endpoint.json').exists():raise ValueError('endpoint already frozen')
    prep=frozen.read(output/'prepared.json');data=loaded(output/'known.npz')
    if file_sha256(output/'known.npz')!=prep['known_sha256']:raise ValueError('known data changed')
    n,initial,generators,projected,_,_,plan,evaluate=setup(spec)
    params=initial[arm];generator=generators[arm];scale=prep['residual_scale']
    reference=generators['network'](initial['network'])
    if pytree_sha256(reference)!=prep['initial_bank_sha256']:raise ValueError('reference changed')
    flat,unravel=ravel_pytree(params)
    jac_fn=make_amplitude_jacobian(lambda p,x:scale*projected(generator(p),x[None])[0],params,chunk_size=spec['jacobian_chunk'])
    cfg=FiniteSchurDecoderConfig(replace(physical_config(6,6,2),maximum_intermediate_q=4),3,3,16,2)
    pullback=make_projected_product_pullback(make_finite_schur_amplitude(cfg),n,chunk_size=spec['H_chunk'])
    multiple=jax.jit(jax.vmap(pullback,in_axes=(None,None,0,0)))
    fit=data['groups']%2==0
    all_plans=[plan(jnp.asarray(data['states'][i:i+4])) for i in range(0,64,4)]
    fit_plans=[plan(jnp.asarray(data['states'][fit][i:i+4])) for i in range(0,32,4)]
    jax.block_until_ready(all_plans);jax.block_until_ready(fit_plans)
    from .spin_finite_schur_direction import candidate_products
    regions=dict(full=np.ones(64,bool),fit=fit,control=~fit,
        **{f'chain{g}':data['groups']==g for g in np.unique(data['groups'])})
    identity=dict(registration_sha256=reg['registration_sha256'],arm=arm)
    checkpoint=folder/'checkpoint.pkl';history=[];accepted=0;reason='fixed_budget_complete';raw_params=params
    p,h=data['carrier'].copy(),n*data['hcarrier'].copy()
    if checkpoint.exists():
        saved=load_resumable_checkpoint(checkpoint,expected_run_identity=identity)
        params,history=saved['params'],saved['history']
        st=saved['optimizer_state'];p,h,accepted,raw_params=st['psi'],st['hpsi'],st['accepted'],st['raw_params']
    def persist():
        save_resumable_checkpoint(checkpoint,params,optimizer_state=dict(psi=p,hpsi=h,accepted=accepted,raw_params=raw_params),
            completed_steps=len(history),next_sample_index=0,run_identity=identity,history=history)
    persist()
    try:
        for step in range(len(history),spec['maximum_attempts_per_arm']):
            check();tick=perf_counter();bank,vjp=jax.vjp(generator,params)
            jac=np.asarray(jac_fn(params,jnp.asarray(data['states'])))
            we,he,_=energy_moment_cotangents(p[fit],h[fit],data['q'][fit],n)
            wr,hr,_=residual_moment_cotangents(p[fit],h[fit],data['q'][fit],n)
            wp,wh=scale*np.stack((we,wr)),scale*np.stack((he,hr))
            bank_grad=jax.tree.map(lambda b:jnp.zeros((2,*b.shape),b.dtype),bank)
            counts=[];value=np.zeros(2)
            for i,prepared in enumerate(fit_plans):
                check();vals,g,c,padded=multiple(bank,prepared,jnp.asarray(wp[:,4*i:4*i+4]),jnp.asarray(wh[:,4*i:4*i+4]))
                bank_grad=jax.tree.map(lambda a,b:a+b,bank_grad,g);value+=np.asarray(vals)
                counts.append((int(np.asarray(c)[0]),int(np.asarray(padded)[0])))
            gs=np.stack([np.asarray(ravel_pytree(vjp(jax.tree.map(lambda b:b[k],bank_grad))[0])[0]) for k in range(2)])
            direction,info=observable_common_direction(p,jac,data['q'],gs,['fit.energy','fit.variance'],
                damping=spec['direction_damping'],relative_cutoff=spec['relative_cutoff'])
            row=dict(attempt=step+1,gradient_seconds=perf_counter()-tick,SR=info,
                full_parameter_count=len(flat),gradient_objectives=2,gradient_H_centers=64,
                pullback_raw_queries=2*sum(c[0] for c in counts),pullback_padded_queries=2*sum(c[1] for c in counts))
            if not info['common_descent'] or not np.any(direction):
                reason='zero_common_direction_skip_identical_retries';row.update(accepted=False,stop_reason=reason)
                history.append(row);persist();break
            replay=None
            if step==0:
                _,delta_bank=jax.jvp(generator,(params,),(unravel(jnp.asarray(direction)),))
                jvp=make_projected_product_jvp(evaluate);tp=[];th=[]
                for prepared in fit_plans:
                    _,_,_,_,dh,dp=jvp(bank,delta_bank,prepared);tp.append(np.asarray(dp));th.append(np.asarray(dh))
                dp,dh=scale*np.concatenate(tp),scale*np.concatenate(th)
                slopes=linear_moment_slopes(p[fit],h[fit],dp[:,None],dh[:,None],data['q'][fit],n)
                want=np.array([slopes['energy'][0],slopes['variance'][0]])
                replay=float(np.max(abs(want-gs@direction)/np.maximum(1.,abs(want))))
                if replay>2e-8:raise ValueError('full gradient/H-JVP slope replay failed')
                row['gradient_H_JVP_replay_relative_defect']=replay;row['verification_H_centers']=64
                row['outside_K2_polynomial_span']=carrier_span_diagnostics(p[fit],h[fit],dp,dh,
                    data['base_products'][fit,:3],data['base_products'][fit,1:4],data['q'][fit],n)
            flat_now,_=ravel_pytree(params)
            def candidate(alpha):
                b=generator(unravel(flat_now-alpha*jnp.asarray(direction)))
                return data['carrier']+scale*(batched(lambda x:projected(b,x),data['states'],8)-data['reference_values'])
            match=matched_chain_state_step(candidate,p,data['q'],regions,spec['target_step_FS'],.1,
                spec['matching_bisections'],relative_tolerance=spec['matching_relative_tolerance'])
            raw_params=unravel(flat_now-match.alpha*jnp.asarray(direction))
            cp,ch,work=candidate_products(generator(raw_params),data,all_plans,evaluate,scale,n)
            defect=float(np.max(abs(cp-match.amplitudes))/max(1.,float(np.max(abs(cp)))))
            if defect>2e-8:raise ValueError('candidate projected-H amplitude replay failed')
            gate=accept_candidate(cp,ch,p,h,data,spec,match.as_dict())
            row.update(accepted=gate['accepted'],gate=gate,matching=match.as_dict(),candidate_H=work,
                candidate_amplitude_replay_relative_defect=defect,candidate_parameter_sha256=pytree_sha256(raw_params))
            np.savez_compressed(folder/f'attempt{step+1:02d}.npz',psi=cp,hpsi=ch,direction=direction,
                energy_gradient=gs[0],variance_gradient=gs[1])
            if gate['accepted']:
                params,p,h=raw_params,cp,ch;accepted+=1
            history.append(row);persist()
            print(arm,'attempt',step+1,'accepted',gate['accepted'],
                'FIT E',gate['metrics']['fit']['after']['energy_per_site'],
                'CONTROL E',gate['metrics']['control']['after']['energy_per_site'],flush=True)
            if not gate['accepted']:
                reason='rejected_fixed_candidate_skip_identical_retries';break
    except TimeoutError:
        reason='time_cap_last_completed_checkpoint_retained'
        # A partially evaluated candidate is NEVER the retained/evaluated raw state.
        saved=load_resumable_checkpoint(checkpoint,expected_run_identity=identity)
        params,history=saved['params'],saved['history'];st=saved['optimizer_state']
        p,h,accepted,raw_params=st['psi'],st['hpsi'],st['accepted'],st['raw_params']
    signal.alarm(0)
    save_parameter_artifact(folder/'endpoint.json',params,identity={**identity,'kind':'fixed_endpoint'})
    save_parameter_artifact(folder/'last_raw.json',raw_params,identity={**identity,'kind':'last_completed_raw_candidate'})
    result=dict(registration_sha256=reg['registration_sha256'],arm=arm,history=history,attempts=len(history),
        accepted_updates=accepted,unused_attempts=spec['maximum_attempts_per_arm']-len(history),stop_reason=reason,
        initial_parameter_sha256=pytree_sha256(initial[arm]),endpoint_parameter_sha256=pytree_sha256(params),
        endpoint_sha256=file_sha256(folder/'endpoint.json'),raw_sha256=file_sha256(folder/'last_raw.json'),
        initial_parameter_count=len(flat),full_parameters_eligible=True,new_training_samples=0,
        audit_state_kind='endpoint' if accepted else 'raw_rejected' if pytree_sha256(raw_params)!=pytree_sha256(initial[arm]) else 'K2_identity',
        strict_SU2_and_complete_A1=True,architecture_unchanged=True,peak_rss_bytes=peak_rss_bytes())
    atomic_json(folder/'training.json',result)


def audit(spec: dict,reg: dict,output: Path,check) -> None:
    if (output/'audit.json').exists():raise ValueError('audit already consumed')
    prep=frozen.read(output/'prepared.json');boundaries={a:frozen.read(output/a/'training.json') for a in ARMS}
    boundary=dict(registration_sha256=reg['registration_sha256'],states=boundaries,
        audit_seed=spec['audit_seed'],endpoints_sealed_before_draw=True,audit_used_for_training=False)
    if (output/'audit_registration.json').exists():
        if frozen.read(output/'audit_registration.json')!=boundary:raise ValueError('audit boundary changed')
    else:atomic_json(output/'audit_registration.json',boundary)
    for arm,row in boundaries.items():
        if file_sha256(output/arm/'endpoint.json')!=row['endpoint_sha256'] or file_sha256(output/arm/'last_raw.json')!=row['raw_sha256']:
            raise ValueError('endpoint/raw parameters changed before audit')
    n,initial,generators,_,carrier,bonds,plan,evaluate=setup(spec)
    reference=generators['network'](initial['network']);q_path=output/'audit_draws.npz'
    if q_path.exists():data=loaded(q_path);sampling=frozen.read(output/'sampling.json')
    else:
        proposal={**spec['audit'],'uniform_floor_relative_to_pilot_mean':spec['uniform_floor_relative_to_pilot_mean'],
            'bridge_fraction':spec['bridge_fraction']}
        data,sampling=draw(proposal,n,carrier,bonds,prep['cscale'],spec['audit_seed'],spec['audit']['records'])
        np.savez_compressed(q_path,**data);atomic_json(output/'sampling.json',sampling)
    check();powers,pwork=power_products(data['states'],carrier,bonds,spec,output/'audit_powers',check)
    powers/=prep['cscale'];k,hk=polynomial_carrier_products(powers,reg['K2_coefficients'])
    names=['K2'];ps=[k];hs=[hk];works={};distance={};metrics_by_arm={}
    plans=[plan(jnp.asarray(data['states'][i:i+4])) for i in range(0,128,4)]
    hp0=[];phi0=[];tick=perf_counter()
    for prepared in plans:
        check();hp,phi,_,_=evaluate(reference,prepared);hp0.append(np.asarray(hp));phi0.append(np.asarray(phi))
    data.update(carrier=k,hcarrier=hk/n,reference_values=np.concatenate(phi0),reference_H_total=np.concatenate(hp0))
    reference_seconds=perf_counter()-tick
    from .spin_finite_schur_direction import candidate_products
    for arm,row in boundaries.items():
        kind=row['audit_state_kind'];path=output/arm/('endpoint.json' if kind=='endpoint' else 'last_raw.json')
        params=load_parameter_artifact(path,initial[arm]);name=arm+'_'+kind
        if kind=='K2_identity':p,h=k.copy(),hk.copy();work=dict(seconds=0.,centers=0,identical_K2_reused=True)
        else:
            check();p,h,work=candidate_products(generators[arm](params),data,plans,evaluate,prep['residual_scale'],n)
        names.append(name);ps.append(p);hs.append(h);works[name]=work
        distance[name]=dict(full=physical_distance(k,p,data['q']),
            per_chain={str(g):physical_distance(k[data['groups']==g],p[data['groups']==g],data['q'][data['groups']==g]) for g in np.unique(data['groups'])})
        # Protected no-update endpoints are exactly K2, not the rejected raw state.
        metrics_by_arm[arm]=dict(evaluated_state=name,endpoint_is_K2=row['accepted_updates']==0)
        print('audit full H',name,round(work['seconds'],2),'s',flush=True)
    p,h=np.column_stack(ps),np.column_stack(hs)
    metrics=paired_chain_audit(p,h,data['q'],data['groups'],n,spec['audit']['bootstrap_replicates'],spec['bootstrap_seed'],names)
    np.savez_compressed(output/'audit.npz',**{k:v for k,v in data.items() if k not in ('carrier','hcarrier')},psi=p,hpsi=h,powers=powers)
    resolved=[];compatible=[]
    for arm,role in metrics_by_arm.items():
        name=role['evaluated_state'];pair=metrics['paired'][name+'-minus-K2'];e,r=pair['energy_per_site'],pair['variance_per_site']
        if not role['endpoint_is_K2'] and e['value']<-2*e['chain_jackknife_se'] and r['value']<=0:resolved.append(arm)
        if e['value']<=2*e['chain_jackknife_se'] and r['value']<=2*r['chain_jackknife_se']:compatible.append(name)
    result=dict(registration_sha256=reg['registration_sha256'],boundary=boundary,metrics=metrics,
        arm_roles=metrics_by_arm,distances=distance,carrier_power_work=pwork,residual_H_work=works,
        reference_H_centers=128,reference_H_seconds=reference_seconds,sampling=sampling,
        resolved_endpoint_energy_variance_wins=resolved,statistically_compatible_candidates=compatible,
        audit_used_for_training=False,endpoints_changed_after_audit=False,new_audit_records=128,
        known_optimizer_records=64,matched_architecture_and_data_not_equal_parameter_count=True,
        arbitrary_configuration_dependent_ViT=False,new_ED=False,GPU_hours=0,
        endpoint_seals_unchanged=all(file_sha256(output/a/'endpoint.json')==boundaries[a]['endpoint_sha256'] for a in ARMS),
        no_automatic_production=True,peak_rss_bytes=peak_rss_bytes())
    result['result_sha256']=canonical_sha256(result);atomic_json(output/'audit.json',result)
    print('independent paired changes', {k:v for k,v in metrics['paired'].items()},flush=True)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('register','prepare','train','audit'),required=True)
    parser.add_argument('--arm',choices=ARMS);args=parser.parse_args()
    spec=frozen.read(CONFIG);output=ROOT/spec['output']
    if args.stage=='register':register(spec,output);return
    reg=frozen.read(output/'registration.json');frozen.verify(reg)
    if spec!=reg['config']:raise ValueError('configuration changed')
    stage=args.arm if args.stage=='train' else args.stage
    if stage is None:parser.error('--arm required for training')
    budget=frozen.read(output/'budget.json')
    if stage in budget['completed_stages']:raise ValueError('bounded stage already completed')
    base=budget['seconds'];start=perf_counter();cap=min(spec['stage_seconds'][stage],spec['wall_limit_total_seconds']-base)
    if cap<=0:raise TimeoutError('cumulative budget exhausted')
    last_budget_write=[start-1.]
    def check():
        now=perf_counter();seconds=base+now-start
        if now-last_budget_write[0]>=.5:
            atomic_json(output/'budget.json',dict(seconds=seconds,completed_stages=budget['completed_stages'],active_stage=stage))
            last_budget_write[0]=now
        if now-start>cap:raise TimeoutError('stage time cap')
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30:raise MemoryError('memory cap')
    def alarm(_s,_f):raise TimeoutError('bounded stage time cap')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(cap)))
    done=False
    try:
        if stage=='prepare':prepare(spec,reg,output,check)
        elif stage in ARMS:train(spec,reg,output,stage,check)
        else:audit(spec,reg,output,check)
        done=True
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        seconds=base+perf_counter()-start
        completed=budget['completed_stages']+([stage] if done else [])
        atomic_json(output/'budget.json',dict(seconds=seconds,completed_stages=completed))
        atomic_json(output/(stage+'_execution.json'),dict(seconds=perf_counter()-start,total_seconds=seconds,
            complete=done,peak_rss_bytes=peak_rss_bytes()))


if __name__=='__main__':main()
