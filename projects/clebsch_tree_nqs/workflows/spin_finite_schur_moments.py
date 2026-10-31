"""One energy/residual-moment common direction, unchanged state and H kernels.

Reuse full energy gradients, plans, pullbacks, physical-step search, carrier
protection and final audit/checkpoint writer from the sealed Rayleigh gate.
Only the residual cotangents and six-objective direction are new.
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
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import FiniteSchurDecoderConfig, make_finite_schur_amplitude
from su2_nqs.vmc.joint_moments import residual_moment_cotangents, normalized_common_moment_direction
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.coherent_tangent import coherent_tangent_diagnostics
from su2_nqs.vmc.carrier_protection import CarrierProtection, carrier_quality_gate, physical_moments
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from .spin_decoder_cost import ROOT, physical_config, peak_rss_bytes
from .spin_finite_schur_optimize import setup, check_resources
from .spin_finite_schur_direction import candidate_products
from .spin_finite_schur_jvp import tangent_records
from .spin_finite_schur_rayleigh import registration as energy_registration, chain_regions, finalize
from .spin_schur_carrier_control import batched

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_moments.json'
OBJECTIVES=('E_full','E_half_0','E_half_1','R_full','R_half_0','R_half_1')


def registration(spec: dict) -> dict:
    """Seal joint descent and STRICT no-variance-increase rules before work."""
    result=energy_registration(spec);result.pop('registration_sha256')
    if spec['objectives']!=list(OBJECTIVES) or spec['protection']['maximum_variance_ratio']!=1.:
        raise ValueError('six fixed objectives and STRICT nonincreasing residual moment required')
    source=ROOT/spec['moment_source'];old=json.loads((source/'registration.json').read_text())['config']
    if spec['seeds']['audit'] in old['seeds'].values():raise ValueError('fresh joint-moment audit seed required')
    if any(spec[k]!=old[k] for k in ('parent','previous','direction_source','jvp_source','maximum_q','j2_over_j1')):
        raise ValueError('same physical parent and sealed energy cache required')
    paths=[source/'registration.json']+[source/arm/name for arm in ('network','free') for name in ('products.npz','results.json')]
    result['inputs'].update({str(p.relative_to(ROOT)):file_sha256(p) for p in paths})
    result['sources']=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    result['registration_sha256']=canonical_sha256(result)
    return result


def joint_slope_gate(records: dict, common: dict) -> dict:
    """Both actual JVP slopes descend on full cache AND each fixed half."""
    rows=[records['full'],*records['halves']]
    checks=dict(common_direction=common['common_descent'],
        energy_descent_full_and_each_half=all(r['energy_slope_per_site']<0 for r in rows),
        residual_descent_full_and_each_half=all(r['variance_slope_per_site']<0 for r in rows),
        nonzero_physical_direction=all(r['tangent_fs_speed']>1e-12 for r in rows))
    return dict(eligible=bool(all(checks.values())),checks=checks,
                rejected_by=[k for k,v in checks.items() if not v],finite_cache_not_population_certificate=True)


def train(spec: dict, reg: dict, output: Path, arm: str) -> dict:
    """One complete joint direction and at most one measured physical proposal."""
    folder=output/arm;folder.mkdir(exist_ok=True)
    if any(folder.iterdir()):raise FileExistsError('completed/partial artifacts preserved')
    start=perf_counter();source=ROOT/spec['moment_source']
    units=json.loads((ROOT/spec['direction_source']/'prepared.json').read_text());scale=units['residual_scale']
    with np.load(ROOT/spec['previous']/'control.npz') as d:data={k:d[k] for k in d.files}
    with np.load(source/arm/'products.npz') as d:cached={k:d[k] for k in d.files}
    n,initial,generators,projected,_,_,plan,evaluate=setup(spec)
    params,generator=initial[arm],generators[arm];flat,unravel=ravel_pytree(params)
    bank,bank_vjp=jax.vjp(generator,params);c,hc,q=data['carrier'],n*data['hcarrier'],data['q']
    jac=cached['control_jacobian'];regions=chain_regions(data['groups'])
    cfg=FiniteSchurDecoderConfig(replace(physical_config(6,6,2),maximum_intermediate_q=4),3,3,16,2)
    scalar=make_projected_product_pullback(make_finite_schur_amplitude(cfg),n,chunk_size=spec['H_chunk'])
    pullback=jax.jit(jax.vmap(scalar,in_axes=(None,None,0,0)))
    gp,gh,full_info=residual_moment_cotangents(c,hc,q,n)
    r_halves=[];r_full_parts=[];work=[];half_infos=[]
    for parity in (0,1):
        mask=regions[f'half_{parity}'];states=data['states'][mask]
        lp,lh,info=residual_moment_cotangents(c[mask],hc[mask],q[mask],n);half_infos.append(info)
        wp=scale*np.stack((lp,gp[mask]));wh=scale*np.stack((lh,gh[mask]))
        total=jax.tree.map(lambda v:jnp.zeros((2,*v.shape),v.dtype),bank)
        value=np.zeros(2);counts=[];tick=perf_counter()
        for offset in range(0,len(states),spec['H_microbatch']):
            prepared=plan(jnp.asarray(states[offset:offset+spec['H_microbatch']]))
            sl=slice(offset,offset+spec['H_microbatch'])
            val,derivative,count,padded=pullback(bank,prepared,jnp.asarray(wp[:,sl]),jnp.asarray(wh[:,sl]))
            total=jax.tree.map(lambda a,b:a+b,total,derivative);value+=np.asarray(val)
            counts.append((int(count[0]),int(padded[0])))
        gradients=[]
        for rhs in (0,1):
            leaves=jax.tree.map(lambda v:v[rhs],total)
            g,_=ravel_pytree(bank_vjp(leaves)[0]);gradients.append(np.asarray(g))
        wanted=np.real(wp@data['reference_values'][mask]+wh@data['reference_H_total'][mask])
        replay=float(np.max(abs(value-wanted)/np.maximum(1.,abs(wanted))))
        if replay>2e-10:raise FloatingPointError('two-RHS residual transpose objective replay fails')
        r_halves.append(gradients[0]);r_full_parts.append(gradients[1])
        work.append(dict(centers=len(states),cotangent_RHS=2,scalar_pullback_center_equivalents=2*len(states),
            seconds=perf_counter()-tick,unique_raw_queries_per_RHS=sum(x[0] for x in counts),
            padded_raw_queries_per_RHS=sum(x[1] for x in counts),objective_replay_relative_error=replay))
        check_resources(spec,start)
    energies=np.stack((cached['full_gradient'],cached['half_0_gradient'],cached['half_1_gradient']))/n
    residuals=np.stack((sum(r_full_parts),*r_halves));gradients=np.concatenate((energies,residuals))
    # Replay all saved Hdelta directions without reading an old audit or new H work.
    replay={};old=json.loads((ROOT/spec['jvp_source']/arm/'results.json').read_text())
    with np.load(ROOT/spec['jvp_source']/arm/'products.npz') as d:
        for name,row in old['records'].items():
            expected=np.array([row['diagnostic']['full']['variance_slope_per_site'],*[r['variance_slope_per_site'] for r in row['diagnostic']['halves']]])
            error=float(np.max(abs(-residuals@d[name+'_direction']-expected)))
            if error>2e-8:raise FloatingPointError('residual gradient disagrees with old independent JVP')
            replay[name]=error
    previous=json.loads((source/arm/'results.json').read_text())
    old_expected=np.array([previous['tangent']['full']['variance_slope_per_site'],*[r['variance_slope_per_site'] for r in previous['tangent']['halves']]])
    error=float(np.max(abs(-residuals@cached['direction']-old_expected)))
    if error>2e-8:raise FloatingPointError('residual gradient disagrees with prior energy-only JVP')
    replay['energy_only_common']=error
    direction,common=normalized_common_moment_direction(c,hc,jac,q,gradients,OBJECTIVES,damping=.1)
    _,dbank=jax.jvp(generator,(params,),(unravel(-jnp.asarray(direction)),))
    jvp=make_projected_product_jvp(evaluate);plans=[];deltas=[];hdeltas=[];tick=perf_counter()
    for offset in range(0,len(q),spec['H_microbatch']):
        prepared=plan(jnp.asarray(data['states'][offset:offset+spec['H_microbatch']]))
        plans.append(prepared);_,_,_,_,dh,d=jvp(bank,dbank,prepared)
        deltas.append(np.asarray(d)*scale);hdeltas.append(np.asarray(dh)*scale)
    delta,hdelta=np.concatenate(deltas),np.concatenate(hdeltas);jvp_seconds=perf_counter()-tick
    records=tangent_records(data,delta,hdelta,n)
    measured=np.array([records['full']['energy_slope_per_site'],*[r['energy_slope_per_site'] for r in records['halves']],
                       records['full']['variance_slope_per_site'],*[r['variance_slope_per_site'] for r in records['halves']]])
    slope_error=float(np.max(abs(measured+gradients@direction)))
    tangent_error=float(np.linalg.norm(delta+jac@direction)/max(np.linalg.norm(delta),1e-30))
    if slope_error>2e-8 or tangent_error>2e-9:raise FloatingPointError('new JVP/complete-gradient identity fails')
    # Actual-parameter FD, FIRST FOUR centers only, same H work for E and R.
    eps=spec['finite_difference_step']/max(1.,np.linalg.norm(direction));m=spec['H_microbatch']
    plus=evaluate(generator(unravel(flat-eps*jnp.asarray(direction))),plans[0])
    minus=evaluate(generator(unravel(flat+eps*jnp.asarray(direction))),plans[0])
    fd_p=np.asarray(plus[1]-minus[1])*scale/(2*eps);fd_h=np.asarray(plus[0]-minus[0])*scale/(2*eps)
    fd_errors=dict(amplitude=float(np.linalg.norm(fd_p-delta[:m])/max(np.linalg.norm(delta[:m]),1e-30)),
        H_amplitude=float(np.linalg.norm(fd_h-hdelta[:m])/max(np.linalg.norm(hdelta[:m]),1e-30)))
    points=[]
    for output_tuple in (plus,minus):
        p=c[:m]+scale*(np.asarray(output_tuple[1])-data['reference_values'][:m])
        hp=hc[:m]+scale*(np.asarray(output_tuple[0])-data['reference_H_total'][:m])
        points.append(physical_moments(p,hp,q[:m],n))
    reference=coherent_tangent_diagnostics(c[:m],hc[:m],delta[:m],hdelta[:m],q[:m],n)
    fd_slopes={}
    for moment,slope in (('energy_per_site','energy_slope_per_site'),('variance_per_site','variance_slope_per_site')):
        observed=(points[0][moment]-points[1][moment])/(2*eps)
        difference=abs(observed-reference[slope]);fd_slopes[slope]=dict(observed=observed,absolute_error=difference)
        if difference>2e-6*max(1.,abs(reference[slope])):raise FloatingPointError('complete physical-moment FD fails')
    if max(fd_errors.values())>2e-6:raise FloatingPointError('actual-parameter amplitude/H FD fails')
    gate=joint_slope_gate(records,common);policy=CarrierProtection(**spec['protection'])
    trial=dict(raw_candidate_constructed=False,accepted=False,cache_accepted=False,eligibility=gate,
               candidate_H_centers=0,final_parameter_sha256=pytree_sha256(params))
    candidate_work=None
    if gate['eligible']:
        speed=max([records['full']['tangent_fs_speed'],*[r['tangent_fs_speed'] for r in records['halves']],
                   *[r['tangent_fs_speed'] for r in records['chains'].values()]])
        maximum=min(spec['direction_step_size'],2*policy.maximum_step_fs/max(speed,1e-30))
        def candidate(alpha):
            b=generator(unravel(flat-alpha*jnp.asarray(direction)))
            return c+scale*(batched(lambda x:projected(b,x),data['states'],8)-data['reference_values'])
        match=matched_chain_state_step(candidate,c,q,regions,policy.maximum_step_fs,maximum,12,relative_tolerance=spec['matching_relative_tolerance'])
        proposed=unravel(flat-match.alpha*jnp.asarray(direction))
        p,hp,candidate_work=candidate_products(generator(proposed),data,plans,evaluate,scale,n)
        discrepancy=float(np.max(abs(p-match.amplitudes)))
        if discrepancy>2e-10*max(1.,float(np.max(abs(p)))):raise FloatingPointError('matched/H amplitude replay fails')
        gates=[]
        for name in ('full','half_0','half_1'):
            mask=regions[name]
            gates.append(carrier_quality_gate(c[mask],hc[mask],c[mask],hc[mask],p[mask],hp[mask],q[mask],data['groups'][mask],n,policy))
        cache=bool(match.search['matched'] and all(g['accepted'] for g in gates))
        save_parameter_artifact(folder/'candidate.json',proposed,identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,kind='raw_joint_moment_proposal'))
        np.savez_compressed(folder/'candidate_control.npz',psi=p,hpsi=hp)
        trial.update(raw_candidate_constructed=True,cache_accepted=cache,matching=match.as_dict(),quality_gates=gates,
            candidate_H_centers=len(p),candidate_parameter_sha256=pytree_sha256(proposed),
            candidate_sha256=file_sha256(folder/'candidate.json'),amplitude_H_replay_error=discrepancy)
    np.savez_compressed(folder/'products.npz',energy_gradients=energies,residual_gradients=residuals,
        direction=direction,delta=delta,hdelta=hdelta,control_jacobian=jac)
    result=dict(registration_sha256=reg['registration_sha256'],arm=arm,trial=trial,common=common,
        tangent=records,moment_parent=full_info,moment_half_parents=half_infos,pullback_work=work,
        old_JVP_residual_replay_errors=replay,new_JVP_slope_abs_error=slope_error,cached_Jacobian_relative_error=tangent_error,
        JVP_H_centers=len(q),JVP_seconds=jvp_seconds,finite_difference_H_centers=2*m,
        finite_difference_relative_errors=fd_errors,finite_difference_slopes_first_four=fd_slopes,candidate_work=candidate_work,
        product_sha256=file_sha256(folder/'products.npz'),new_training_samples=0,control_used_for_training=True,
        old_audit_not_read=True,architecture_unchanged=True,peak_rss_bytes=peak_rss_bytes())
    save_gated_checkpoint(folder/'proposal_checkpoint.pkl',params,optimizer_state=dict(direction=direction),
        run_identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='proposal'),decision=trial)
    atomic_json(folder/'results.json',result)
    print(arm,'common',common['common_descent'],'six measured slopes',measured.tolist(),'cache accepted',trial['cache_accepted'],flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--stage',choices=('register','train','finalize'),required=True)
    parser.add_argument('--arm',choices=('network','free'));args=parser.parse_args()
    spec=json.loads(args.config.read_text());reg=registration(spec);output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists():raise FileExistsError('registration sealed')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg!=json.loads((output/'registration.json').read_text()):raise ValueError('numeric registration mismatch')
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
