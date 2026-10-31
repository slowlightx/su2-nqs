"""Independent step calibration before any fixed-budget convergence trial.

Same parent, frozen curvature direction, exact SU2/full A1 and original H.
FIT cache is already seen; new CALIBRATION changes only step eligibility and
length. A separate conditional untouched CONFIRMATION cannot refit anything.
An unsuccessful direction does not earn more training by shrinking to zero.
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
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.update_calibration import independent_update_calibration
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.carrier_protection import CarrierProtection, carrier_quality_gate, physical_moments
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from .spin_decoder_cost import ROOT, peak_rss_bytes
from .spin_finite_schur_optimize import setup, check_resources, proposal_spec
from .spin_finite_schur_curvature import registration as curvature_registration
from .spin_finite_schur_rayleigh import chain_regions, finalize as confirm
from .spin_finite_schur_direction import candidate_products
from .spin_schur_carrier_control import batched, draw

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_reliability.json'


def registration(spec: dict) -> dict:
    """Freeze data roles, direction, conservative rules and conditional budget."""
    result = curvature_registration(spec); result.pop('registration_sha256')
    source = ROOT/spec['fixed_direction_source']
    old = json.loads((source/'registration.json').read_text())['config']
    if any(spec[k] != old[k] for k in ('parent','previous','maximum_q','multiplicity_width','embedding_dim','depth','protection')):
        raise ValueError('same architecture, physical parent and protection required')
    if (spec['seeds']['independent_calibration'] in old['seeds'].values()
            or spec['seeds']['audit'] in old['seeds'].values()):
        raise ValueError('distinct fresh calibration and confirmation seeds required')
    if spec['response_safety'] != .5 or spec['convergence_budget'] != dict(
        maximum_attempts=8, training_records_per_attempt=64, calibration_records_per_attempt=64,
        endpoint_audit_records=128, maximum_cpu_wall_seconds=7200):
        raise ValueError('fixed response safety and eight-attempt conditional budget required')
    cfg = spec['independent_calibration']
    if (cfg['chains'],cfg['records']) != (8,16):
        raise ValueError('eight independent calibration chains with sixteen records required')
    paths = [source/'registration.json']+[source/arm/'products.npz' for arm in ('network','free')]
    result['inputs'].update({str(p.relative_to(ROOT)):file_sha256(p) for p in paths})
    result['sources'] = transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def calibrate(spec: dict, reg: dict, output: Path) -> None:
    """Draw NEW positive-Q calibration, never read a consumed old audit."""
    if (output/'calibration.npz').exists():raise FileExistsError('calibration preserved')
    n,_,_,_,carrier,bonds,_,_=setup(spec)
    units=json.loads((ROOT/spec['direction_source']/'prepared.json').read_text())
    data,info=draw(proposal_spec(spec,'independent_calibration'),n,carrier,bonds,units['cscale'],
        spec['seeds']['independent_calibration'],spec['independent_calibration']['records'])
    np.savez_compressed(output/'calibration.npz',**data)
    atomic_json(output/'calibration.json',dict(registration_sha256=reg['registration_sha256'],
        calibration_sha256=file_sha256(output/'calibration.npz'),sampling=info,
        records=len(data['q']),used_for_gradients=False,used_for_step_selection=True,
        is_untouched_confirmation=False,old_audits_not_read=True))
    print('independent calibration sealed',len(data['q']),flush=True)


def combined_regions(old_groups: np.ndarray, new_groups: np.ndarray) -> dict[str,np.ndarray]:
    """Union, each dataset/fold and all sixteen chains; no phase-blind gaps."""
    size=len(old_groups)+len(new_groups)
    regions=dict(full=np.ones(size,bool))
    for prefix,groups,offset in (('fit',old_groups,0),('calibration',new_groups,len(old_groups))):
        for name,mask in chain_regions(groups).items():
            full=np.zeros(size,bool);full[offset:offset+len(groups)]=mask
            regions[prefix+'_'+name]=full
    return regions


def proposal(spec: dict, reg: dict, output: Path, arm: str) -> dict:
    """Screen frozen orientation BEFORE amplitude probes or a new finite H state."""
    folder=output/arm;folder.mkdir(exist_ok=True)
    if any(folder.iterdir()):raise FileExistsError('completed/partial arm preserved')
    start=perf_counter();units=json.loads((ROOT/spec['direction_source']/'prepared.json').read_text())
    scale=units['residual_scale']
    with np.load(ROOT/spec['previous']/'control.npz') as p:fit={k:p[k] for k in p.files}
    with np.load(output/'calibration.npz') as p:new={k:p[k] for k in p.files}
    with np.load(ROOT/spec['fixed_direction_source']/arm/'products.npz') as p:
        direction=p['direction'];old_delta=p['delta'];old_hdelta=p['hdelta']
    n,initial,generators,projected,_,_,plan,evaluate=setup(spec)
    params,generator=initial[arm],generators[arm];flat,unravel=ravel_pytree(params)
    bank,dbank=jax.jvp(generator,(params,),(unravel(-jnp.asarray(direction)),))
    jvp=make_projected_product_jvp(evaluate);plans=[];phi=[];hphi=[];ds=[];hds=[];counts=[]
    tick=perf_counter()
    for offset in range(0,len(new['q']),spec['H_microbatch']):
        prepared=plan(jnp.asarray(new['states'][offset:offset+spec['H_microbatch']]))
        plans.append(prepared);hp,p,count,padded,hd,d=jvp(bank,dbank,prepared)
        phi.append(np.asarray(p));hphi.append(np.asarray(hp));ds.append(np.asarray(d)*scale);hds.append(np.asarray(hd)*scale)
        counts.append((int(count),int(padded)))
    delta,hdelta=np.concatenate(ds),np.concatenate(hds);jvp_seconds=perf_counter()-tick
    new['reference_values'],new['reference_H_total']=np.concatenate(phi),np.concatenate(hphi)
    # Independent physical/parameter derivative check on FIRST four new centers.
    eps=spec['finite_difference_step']/max(1.,np.linalg.norm(direction));m=spec['H_microbatch']
    plus=evaluate(generator(unravel(flat-eps*jnp.asarray(direction))),plans[0])
    minus=evaluate(generator(unravel(flat+eps*jnp.asarray(direction))),plans[0])
    errors=dict(amplitude=float(np.linalg.norm(scale*np.asarray(plus[1]-minus[1])/(2*eps)-delta[:m])/max(np.linalg.norm(delta[:m]),1e-30)),
        H_amplitude=float(np.linalg.norm(scale*np.asarray(plus[0]-minus[0])/(2*eps)-hdelta[:m])/max(np.linalg.norm(hdelta[:m]),1e-30)))
    if max(errors.values())>2e-6:raise FloatingPointError('new calibration actual-parameter FD fails')
    old_guard=independent_update_calibration(fit['carrier'],n*fit['hcarrier'],old_delta,old_hdelta,
        fit['q'],fit['groups'],n,maximum_alpha=spec['direction_step_size'],response_safety=spec['response_safety'])
    guard=independent_update_calibration(new['carrier'],n*new['hcarrier'],delta,hdelta,new['q'],new['groups'],n,
        maximum_alpha=spec['direction_step_size'],response_safety=spec['response_safety'])
    gate=dict(eligible=bool(old_guard['eligible'] and guard['eligible']),fit=old_guard,independent_calibration=guard)
    trial=dict(raw_candidate_constructed=False,accepted=False,cache_accepted=False,
        eligibility=gate,candidate_H_centers=0,final_parameter_sha256=pytree_sha256(params))
    work=None;ray=[]
    if gate['eligible']:
        fields=('states','q','carrier','hcarrier','reference_values','reference_H_total')
        combined={k:np.concatenate((fit[k],new[k])) for k in fields}
        combined['groups']=np.r_[fit['groups'],new['groups']+8]
        regions=combined_regions(fit['groups'],new['groups'])
        c,hc,q=combined['carrier'],n*combined['hcarrier'],combined['q']
        policy=CarrierProtection(**spec['protection'])
        maximum=min(old_guard['maximum_alpha'],guard['maximum_alpha'])
        def candidate(alpha):
            b=generator(unravel(flat-alpha*jnp.asarray(direction)))
            return c+scale*(batched(lambda x:projected(b,x),combined['states'],8)-combined['reference_values'])
        match=matched_chain_state_step(candidate,c,q,regions,policy.maximum_step_fs,maximum,12,
            relative_tolerance=spec['matching_relative_tolerance'])
        proposed=unravel(flat-match.alpha*jnp.asarray(direction))
        fit_plans=[plan(jnp.asarray(fit['states'][i:i+m])) for i in range(0,len(fit['q']),m)]
        p,hp,work=candidate_products(generator(proposed),combined,fit_plans+plans,evaluate,scale,n)
        discrepancy=float(np.max(abs(p-match.amplitudes)))
        if discrepancy>2e-10*max(1.,float(np.max(abs(p)))):raise FloatingPointError('calibrated amplitude/H replay fails')
        all_delta,all_hdelta=np.r_[old_delta,delta],np.r_[old_hdelta,hdelta]
        gates=[]
        for prefix in ('fit','calibration'):
            for name in ('full','half_0','half_1'):
                mask=regions[prefix+'_'+name]
                before=physical_moments(c[mask],hc[mask],q[mask],n)
                after=physical_moments(c[mask]+match.alpha*all_delta[mask],hc[mask]+match.alpha*all_hdelta[mask],q[mask],n)
                ray.append(dict(region=prefix+'_'+name,delta_energy=after['energy_per_site']-before['energy_per_site'],
                    delta_residual=after['variance_per_site']-before['variance_per_site']))
                gates.append(carrier_quality_gate(c[mask],hc[mask],c[mask],hc[mask],p[mask],hp[mask],q[mask],combined['groups'][mask],n,policy))
        # Response-limited steps may deliberately underfill FS; never claim equal distance.
        nonnull=max(d['fs'] for d in match.distances.values())>policy.minimum_nonzero_fs
        capped=max(d['fs'] for d in match.distances.values())<=policy.maximum_step_fs*(1+1e-8)
        ray_pass=all(r['delta_energy']<=0 and r['delta_residual']<=0 for r in ray)
        passing=bool(nonnull and capped and ray_pass and all(g['accepted'] for g in gates))
        save_parameter_artifact(folder/'candidate.json',proposed,identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,kind='independently_calibrated_proposal'))
        np.savez_compressed(folder/'candidate_control.npz',psi=p,hpsi=hp)
        trial.update(raw_candidate_constructed=True,cache_accepted=passing,matching=match.as_dict(),quality_gates=gates,
            tangent_ray_moment_gate=ray_pass,candidate_H_centers=len(p),candidate_parameter_sha256=pytree_sha256(proposed),
            candidate_sha256=file_sha256(folder/'candidate.json'),amplitude_H_replay_error=discrepancy,
            response_limited_maximum_alpha=maximum,underfilled_FS_not_an_equal_distance_comparison=not match.search['matched'])
    np.savez_compressed(folder/'products.npz',direction=direction,delta=delta,hdelta=hdelta,
        reference_values=new['reference_values'],reference_H_total=new['reference_H_total'])
    result=dict(registration_sha256=reg['registration_sha256'],arm=arm,trial=trial,fit_guard=old_guard,
        calibration_guard=guard,JVP_H_centers=len(new['q']),JVP_seconds=jvp_seconds,
        JVP_unique_raw_queries=sum(r[0] for r in counts),JVP_padded_raw_queries=sum(r[1] for r in counts),
        finite_difference_H_centers=2*m,finite_difference_relative_errors=errors,candidate_work=work,
        tangent_ray_at_calibrated_step=ray,new_training_samples=0,new_calibration_records=len(new['q']),
        new_pullback_centers=0,direction_refitted=False,old_audit_not_read=True,architecture_unchanged=True,
        product_sha256=file_sha256(folder/'products.npz'),peak_rss_bytes=peak_rss_bytes())
    save_gated_checkpoint(folder/'proposal_checkpoint.pkl',params,optimizer_state=dict(direction=direction),
        run_identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='proposal'),decision=trial)
    atomic_json(folder/'results.json',result);check_resources(spec,start)
    print(arm,'calibration slopes',[(r['energy_slope_per_site'],r['variance_slope_per_site']) for r in guard['regions'].values()],
        'eligible',gate['eligible'],'finite candidate',trial['raw_candidate_constructed'],flush=True)
    return result


def finalize(spec: dict, reg: dict, output: Path) -> dict:
    """Reuse the sealed fresh-confirmation/checkpoint code; no post-audit refit."""
    result=confirm(spec,reg,output)
    release=dict(registration_sha256=reg['registration_sha256'],budget=spec['convergence_budget'],
        eligible_arms=[a for a,v in result['accepted'].items() if v],
        released=bool(result['accepted_updates']),training_executed=False,
        reason='requires separately sealed full-gradient fixed-endpoint trajectory' if result['accepted_updates'] else
            'no independently confirmed nonzero update; undertraining cannot yet be diagnosed',
        no_trajectory_extension=True,no_best_endpoint_selection=True)
    atomic_json(output/'convergence_release.json',release)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=('register','calibrate','proposal','finalize'),required=True)
    parser.add_argument('--arm',choices=('network','free'));args=parser.parse_args()
    spec=json.loads(args.config.read_text());reg=registration(spec);output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists():raise FileExistsError('registration sealed')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg!=json.loads((output/'registration.json').read_text()):raise ValueError('numerical registration mismatch')
    if args.stage=='proposal' and args.arm is None:parser.error('--arm required')
    start=perf_counter()
    def alarm(_signal,_frame):raise TimeoutError('registered process limit exceeded')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='calibrate':calibrate(spec,reg,output)
        elif args.stage=='proposal':proposal(spec,reg,output,args.arm)
        else:finalize(spec,reg,output)
        check_resources(spec,start)
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',dict(wall_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start));raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
