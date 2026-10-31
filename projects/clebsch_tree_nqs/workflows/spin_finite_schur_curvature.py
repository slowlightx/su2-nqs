"""One residual-curvature-aware update in a sealed six-tangent span.

Same state, carrier, decoder, H kernel and actual chain-FS cap. Reuse saved
full gradients/H tangents, then independently validate ONE new direction.
No new head, fitting extension, old-audit input, damping/radius/weight sweep.
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
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.vmc.residual_curvature import residual_curvature_direction
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.carrier_protection import CarrierProtection,carrier_quality_gate,physical_moments
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from .spin_decoder_cost import ROOT,peak_rss_bytes
from .spin_finite_schur_optimize import setup,check_resources
from .spin_finite_schur_moments import registration as moment_registration,OBJECTIVES,joint_slope_gate
from .spin_finite_schur_rayleigh import chain_regions,finalize
from .spin_finite_schur_jvp import tangent_records
from .spin_finite_schur_direction import candidate_products
from .spin_schur_carrier_control import batched

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_curvature.json'
BASIS=('born','importance','fit_0','fit_1','energy_common','joint_common')


def registration(spec: dict) -> dict:
    """Seal one fixed response metric and six SAVED parameter-update columns."""
    result=moment_registration(spec);result.pop('registration_sha256')
    if spec['basis']!=list(BASIS) or spec['response_weight']!=1. or spec['residual_floor']!=1e-12:
        raise ValueError('fixed six-tangent basis and response metric required')
    source=ROOT/spec['curvature_source'];old=json.loads((source/'registration.json').read_text())['config']
    if spec['seeds']['audit'] in old['seeds'].values():raise ValueError('fresh curvature audit seed required')
    for key in ('parent','previous','direction_source','jvp_source','moment_source','maximum_q','j2_over_j1','protection'):
        if spec[key]!=old[key]:raise ValueError('same physical parent and STRICT joint protection required')
    paths=[source/'registration.json']+[source/arm/'products.npz' for arm in ('network','free')]
    result['inputs'].update({str(p.relative_to(ROOT)):file_sha256(p) for p in paths})
    result['sources']=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    result['registration_sha256']=canonical_sha256(result)
    return result


def saved_tangent_span(spec: dict, arm: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    """Same physical parent; saved SR d has update -d. No audit/candidate read."""
    dirs,ds,hds=[],[],[]
    with np.load(ROOT/spec['jvp_source']/arm/'products.npz') as p:
        for name in BASIS[:4]:
            dirs.append(-p[name+'_direction']);ds.append(p[name+'_delta']);hds.append(p[name+'_hdelta'])
    for key in ('moment_source','curvature_source'):
        with np.load(ROOT/spec[key]/arm/'products.npz') as p:
            dirs.append(-p['direction']);ds.append(p['delta']);hds.append(p['hdelta'])
    with np.load(ROOT/spec['curvature_source']/arm/'products.npz') as p:cache={k:p[k] for k in p.files}
    return np.stack(dirs,axis=1),np.stack(ds,axis=1),np.stack(hds,axis=1),cache


def train(spec: dict, reg: dict, output: Path, arm: str) -> dict:
    """Rotate the update at unchanged actual FS; do not shrink the old direction."""
    folder=output/arm;folder.mkdir(exist_ok=True)
    if any(folder.iterdir()):raise FileExistsError('completed/partial artifacts preserved')
    start=perf_counter();units=json.loads((ROOT/spec['direction_source']/'prepared.json').read_text());scale=units['residual_scale']
    with np.load(ROOT/spec['previous']/'control.npz') as p:data={k:p[k] for k in p.files}
    b,d,hd,cache=saved_tangent_span(spec,arm);n,initial,generators,projected,_,_,plan,evaluate=setup(spec)
    params,generator=initial[arm],generators[arm];flat,unravel=ravel_pytree(params);bank=generator(params)
    c,hc,q=data['carrier'],n*data['hcarrier'],data['q'];regions=chain_regions(data['groups'])
    masks=[regions[k] for k in ('full','half_0','half_1')]
    gradients=np.concatenate((cache['energy_gradients'],cache['residual_gradients']))
    projected_gradients=gradients@b
    coeff,common=residual_curvature_direction(c,hc,d,hd,b,q,masks,projected_gradients,OBJECTIVES,n,
        damping=spec['direction_damping'],response_weight=spec['response_weight'],residual_floor=spec['residual_floor'])
    update=b@coeff;direction=-update;expected_delta=d@coeff;expected_hdelta=hd@coeff
    _,dbank=jax.jvp(generator,(params,),(unravel(jnp.asarray(update)),))
    jvp=make_projected_product_jvp(evaluate);plans=[];ds=[];hds=[];tick=perf_counter()
    for offset in range(0,len(q),spec['H_microbatch']):
        prepared=plan(jnp.asarray(data['states'][offset:offset+spec['H_microbatch']]))
        plans.append(prepared);_,_,_,_,dh,delta=jvp(bank,dbank,prepared)
        ds.append(np.asarray(delta)*scale);hds.append(np.asarray(dh)*scale)
    delta,hdelta=np.concatenate(ds),np.concatenate(hds);seconds=perf_counter()-tick
    errors=dict(amplitude=float(np.linalg.norm(delta-expected_delta)/max(np.linalg.norm(expected_delta),1e-30)),
        H_amplitude=float(np.linalg.norm(hdelta-expected_hdelta)/max(np.linalg.norm(expected_hdelta),1e-30)))
    if max(errors.values())>2e-9:raise FloatingPointError('new complete JVP disagrees with saved-span linearity')
    records=tangent_records(data,delta,hdelta,n)
    slopes=np.array([records['full']['energy_slope_per_site'],*[r['energy_slope_per_site'] for r in records['halves']],
        records['full']['variance_slope_per_site'],*[r['variance_slope_per_site'] for r in records['halves']]])
    slope_error=float(np.max(abs(slopes-gradients@update)))
    if slope_error>2e-8:raise FloatingPointError('six slopes disagree with complete gradients')
    eps=spec['finite_difference_step']/max(1.,np.linalg.norm(update));m=spec['H_microbatch']
    plus=evaluate(generator(unravel(flat+eps*jnp.asarray(update))),plans[0])
    minus=evaluate(generator(unravel(flat-eps*jnp.asarray(update))),plans[0])
    fd_errors=dict(amplitude=float(np.linalg.norm(np.asarray(plus[1]-minus[1])*scale/(2*eps)-delta[:m])/max(np.linalg.norm(delta[:m]),1e-30)),
        H_amplitude=float(np.linalg.norm(np.asarray(plus[0]-minus[0])*scale/(2*eps)-hdelta[:m])/max(np.linalg.norm(hdelta[:m]),1e-30)))
    if max(fd_errors.values())>2e-6:raise FloatingPointError('new actual-parameter FD fails')
    gate=joint_slope_gate(records,common);policy=CarrierProtection(**spec['protection'])
    trial=dict(raw_candidate_constructed=False,accepted=False,cache_accepted=False,eligibility=gate,
        candidate_H_centers=0,final_parameter_sha256=pytree_sha256(params));work=None;ray=[]
    if gate['eligible']:
        speed=max([records['full']['tangent_fs_speed'],*[r['tangent_fs_speed'] for r in records['halves']],
            *[r['tangent_fs_speed'] for r in records['chains'].values()]])
        maximum=min(spec['direction_step_size'],2*policy.maximum_step_fs/max(speed,1e-30))
        def candidate(alpha):
            bank_new=generator(unravel(flat+alpha*jnp.asarray(update)))
            return c+scale*(batched(lambda x:projected(bank_new,x),data['states'],8)-data['reference_values'])
        match=matched_chain_state_step(candidate,c,q,regions,policy.maximum_step_fs,maximum,12,
            relative_tolerance=spec['matching_relative_tolerance'])
        proposed=unravel(flat+match.alpha*jnp.asarray(update))
        p,hp,work=candidate_products(generator(proposed),data,plans,evaluate,scale,n)
        discrepancy=float(np.max(abs(p-match.amplitudes)))
        if discrepancy>2e-10*max(1.,float(np.max(abs(p)))):raise FloatingPointError('matched amplitude/H replay fails')
        gates=[]
        for mask in masks:
            parent=physical_moments(c[mask],hc[mask],q[mask],n)
            predicted=physical_moments(c[mask]+match.alpha*delta[mask],hc[mask]+match.alpha*hdelta[mask],q[mask],n)
            ray.append(dict(delta_energy=predicted['energy_per_site']-parent['energy_per_site'],
                delta_residual=predicted['variance_per_site']-parent['variance_per_site']))
            gates.append(carrier_quality_gate(c[mask],hc[mask],c[mask],hc[mask],p[mask],hp[mask],q[mask],data['groups'][mask],n,policy))
        ray_pass=all(r['delta_energy']<=0 and r['delta_residual']<=0 for r in ray)
        cache_pass=bool(match.search['matched'] and ray_pass and all(g['accepted'] for g in gates))
        save_parameter_artifact(folder/'candidate.json',proposed,identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,kind='raw_residual_trust_proposal'))
        np.savez_compressed(folder/'candidate_control.npz',psi=p,hpsi=hp)
        trial.update(raw_candidate_constructed=True,cache_accepted=cache_pass,matching=match.as_dict(),quality_gates=gates,
            tangent_ray_moment_gate=ray_pass,candidate_H_centers=len(p),candidate_parameter_sha256=pytree_sha256(proposed),
            candidate_sha256=file_sha256(folder/'candidate.json'),amplitude_H_replay_error=discrepancy)
    np.savez_compressed(folder/'products.npz',direction=direction,coefficients=coeff,parameter_basis=b,
        delta=delta,hdelta=hdelta,energy_gradients=cache['energy_gradients'],residual_gradients=cache['residual_gradients'],
        control_jacobian=cache['control_jacobian'])
    result=dict(registration_sha256=reg['registration_sha256'],arm=arm,trial=trial,common=common,
        tangent=records,tangent_ray_at_matched_step=ray,saved_span_JVP_relative_errors=errors,
        new_JVP_slope_abs_error=slope_error,finite_difference_relative_errors=fd_errors,
        JVP_H_centers=len(q),JVP_seconds=seconds,finite_difference_H_centers=2*m,candidate_work=work,
        reused_derivative_H_centers=6*len(q),new_pullback_centers=0,new_training_samples=0,
        product_sha256=file_sha256(folder/'products.npz'),control_used_for_training=True,
        old_audit_not_read=True,architecture_unchanged=True,peak_rss_bytes=peak_rss_bytes())
    save_gated_checkpoint(folder/'proposal_checkpoint.pkl',params,optimizer_state=dict(direction=direction),
        run_identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='proposal'),decision=trial)
    atomic_json(folder/'results.json',result);check_resources(spec,start)
    print(arm,'common',common['common_descent'],'six slopes',slopes.tolist(),'cache',trial['cache_accepted'],flush=True)
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
