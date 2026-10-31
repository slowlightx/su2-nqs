"""Population E/R forces and a spectral control; unchanged neural architecture.

Reuse the previous bounded proposal implementation, full H, JVP, actual-FS
search, carrier protection, checkpoint and paired audit. New work is analytic
H²C and force construction from explicitly KNOWN blocks, not more of an old
trajectory or an adaptive gate relaxation.
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

from su2_nqs.benchmarks.spin_polynomial_products import build_spin_h2_plan
from su2_nqs.benchmarks.operator_word_jax import make_fixed_capacity_scalar_word_kernel,evaluate_fixed_capacity_scalar_word_kernel
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.vmc.population_moment_forces import population_moment_forces,imaginary_time_energy_certificate
from su2_nqs.vmc.cross_sample_geometry import equal_mass_geometry,observable_common_direction
from su2_nqs.vmc.importance_tangent import make_amplitude_jacobian,tangent_system
from su2_nqs.vmc.carrier_protection import CarrierProtection,carrier_quality_gate,physical_distance
from su2_nqs.vmc.update_calibration import independent_update_calibration
from su2_nqs.vmc.coherent_tangent import coherent_tangent_diagnostics
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from .spin_decoder_cost import ROOT,peak_rss_bytes
from .spin_finite_schur_optimize import setup,registration as base_registration,check_resources,full_products,proposal_spec
from .spin_finite_schur_generalization import METHODS,inputs,calibrate,propose,finalize
from .spin_finite_schur_rayleigh import chain_regions
from .spin_schur_carrier_control import batched,draw

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_population.json'


def loaded(path: Path) -> dict:
    """Read a fixed numeric artifact without changing it."""
    with np.load(path) as a:return {k:a[k] for k in a.files}


def known_blocks(spec: dict) -> list[dict]:
    """Previously seen FIT64/GEO128/CAL128 are now FORCE data explicitly."""
    paths=(ROOT/spec['fit_source']/'control.npz',ROOT/spec['geometry_source']/'calibration.npz',ROOT/spec['previous']/'calibration.npz')
    return [{k:v for k,v in loaded(path).items() if k in ('states','q','carrier','hcarrier','groups')} for path in paths]


def registration(spec: dict) -> dict:
    """Seal fixed model, data roles, spectral risk policy and bounded work."""
    reg=base_registration(spec);reg.pop('registration_sha256');CarrierProtection(**spec['protection'])
    old=json.loads((ROOT/spec['previous']/'registration.json').read_text())['config']
    keys=('parent','geometry','rows','columns','j2_over_j1','multiplicity_width','maximum_q',
        'embedding_dim','depth','projection_chunk','H_chunk','H_microbatch','protection')
    if any(spec[k]!=old[k] for k in keys):raise ValueError('same state/kernel and neural protection required')
    if (spec['methods']!=list(METHODS) or spec['steps']!=1 or spec['primary_method']!='hermitian'
            or spec['relative_cutoff']!=1e-6 or spec['direction_damping']!=.1
            or spec['H2_terminal_capacities']!=[16384,32768] or spec['H2_entry_capacity']!=32768
            or spec['H2_amplitude_batch']!=32 or any(s in old['seeds'].values() for s in spec['seeds'].values())):
        raise ValueError('fixed population-force rules and fresh seeds required')
    paths=[ROOT/spec['fit_source']/'control.npz',ROOT/spec['units_source']/'prepared.json',
        ROOT/spec['geometry_source']/'calibration.npz',ROOT/spec['previous']/'registration.json',ROOT/spec['previous']/'calibration.npz']
    for arm in ('network','free'):
        paths += [ROOT/spec['force_source']/arm/'products.npz']
        paths += [ROOT/spec['previous']/arm/p for p in ('geometry.npz','geometry.json','reference.npz')]
    if 'reuse_h2_source' in spec:
        paths += [ROOT/spec['reuse_h2_source']/p for p in ('known_products.npz','known_products.json',
            'registration.json','workflow_source_snapshot.py')]
    reg['inputs'].update({str(p.relative_to(ROOT)):file_sha256(p) for p in paths})
    reg['sources']=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    reg['registration_sha256']=canonical_sha256(reg);return reg


def carrier_h2(spec: dict,states: np.ndarray,carrier,bonds,cscale: float,start: float
               ) -> tuple[np.ndarray,list[dict]]:
    """Reuse fixed-capacity batched executor; topology-only capacity selection."""
    fn=lambda x:carrier(x)/cscale
    kernels={cap:make_fixed_capacity_scalar_word_kernel(fn,n_sites=states.shape[1],amplitude_batch_size=32,
        terminal_capacity=cap,entry_capacity=spec['H2_entry_capacity'],output_count=3) for cap in spec['H2_terminal_capacities']}
    products=[];work=[]
    for i,x in enumerate(states):
        plan,info=build_spin_h2_plan(x[None],bonds)
        cap=next((v for v in kernels if len(plan.terminal_configurations)<=v),None)
        if cap is None:raise MemoryError('carrier H² topology exceeds both registered capacities')
        values,execution=evaluate_fixed_capacity_scalar_word_kernel(kernels[cap],plan)
        products.append(values[0]);work.append({**info,**execution});check_resources(spec,start)
        if (i+1)%32==0:print('H² centers',i+1,'/',len(states),flush=True)
    return np.asarray(products),work


def prepare(spec: dict,reg: dict,output: Path) -> None:
    """No sampling: exact C/HC/H²C on the 320 KNOWN force configurations."""
    if (output/'known_products.npz').exists():raise FileExistsError('known products sealed')
    start=perf_counter();blocks=known_blocks(spec);n,_,_,_,carrier,bonds,_,_=setup(spec)
    units=json.loads((ROOT/spec['units_source']/'prepared.json').read_text())
    all_states=np.concatenate([b['states'] for b in blocks])
    if 'reuse_h2_source' in spec:
        source=ROOT/spec['reuse_h2_source'];old=json.loads((source/'known_products.json').read_text())
        if old['product_sha256']!=file_sha256(source/'known_products.npz'):raise ValueError('cached H² seal changed')
        products=loaded(source/'known_products.npz')['products'];work=[]
    else:products,work=carrier_h2(spec,all_states,carrier,bonds,units['cscale'],start)
    expected=np.column_stack((np.concatenate([b['carrier'] for b in blocks]),n*np.concatenate([b['hcarrier'] for b in blocks])))
    error=float(np.max(abs(products[:,:2]-expected)))
    if error>2e-9*max(1.,float(np.max(abs(expected)))):raise FloatingPointError('analytic carrier C/HC replay fails')
    np.savez_compressed(output/'known_products.npz',products=products)
    atomic_json(output/'known_products.json',dict(registration_sha256=reg['registration_sha256'],product_sha256=file_sha256(output/'known_products.npz'),
        centers=len(all_states),work=work,C_HC_replay_absolute_error=error,new_samples=0,
        data_roles='ALL320 are previously seen force data, not validation',
        reused_H2_source=spec.get('reuse_h2_source'),new_H2_centers=0 if 'reuse_h2_source' in spec else len(all_states),
        peak_rss_bytes=peak_rss_bytes()))


def geometry(spec: dict,reg: dict,output: Path,arm: str) -> None:
    """Keep the old192 geometry; amend forces and perform cross-block consensus."""
    start=perf_counter();folder=output/arm;folder.mkdir()
    blocks=known_blocks(spec);fit,geo,cache,units=inputs(spec,arm)
    previous=loaded(ROOT/spec['previous']/arm/'geometry.npz');known=loaded(output/'known_products.npz')['products']
    _,initial,generators,projected,_,_,_,_=setup(spec);params,generator=initial[arm],generators[arm]
    fn=lambda p,x:units['residual_scale']*projected(generator(p),x[None])[0]
    jac=np.asarray(make_amplitude_jacobian(fn,params,chunk_size=4)(params,jnp.asarray(blocks[2]['states'])))
    js=[cache['control_jacobian'],previous['jacobian'],jac]
    gp,gj,gq=equal_mass_geometry([b['carrier'] for b in blocks[:2]],js[:2],[b['q'] for b in blocks[:2]])
    pp,pj,pq=equal_mass_geometry([b['carrier'] for b in blocks],js,[b['q'] for b in blocks])
    hp,h2=known[:,1],known[:,2];masks=np.concatenate([b['groups'] for b in blocks]);pforces=[];details={}
    for title,indices,p,j,q,h,h_sq in [
        ('fit',np.arange(64),blocks[0]['carrier'],js[0],blocks[0]['q'],hp[:64],h2[:64]),
        ('pooled',np.arange(320),pp,pj,pq,hp,h2)]:
        pair=[];regions=[np.ones(len(p),bool),masks[indices]%2==0,masks[indices]%2==1]
        for mask in regions:
            e,r,info=population_moment_forces(p[mask],h[mask],h_sq[mask],j[mask],q[mask],36)
            pair.append((e,r));details.setdefault(title,[]).append(info)
        pforces.append(np.stack([v[0] for v in pair]+[v[1] for v in pair]))
    consensus=[];offset=0
    for i,b in enumerate(blocks):
        end=offset+len(b['q']);e,r,info=population_moment_forces(b['carrier'],hp[offset:end],h2[offset:end],js[i],b['q'],36)
        consensus.append((e,r));details['block_'+str(i)]=info;offset=end
    forces=[previous['hermitian_gradients'],pforces[0],pforces[1],np.stack([v[0] for v in consensus]+[v[1] for v in consensus])]
    finite=np.r_[cache['energy_gradients'],cache['residual_gradients']];fit_a,_,_,_=tangent_system(fit['carrier'],36*fit['hcarrier'],js[0],fit['q'])
    variants={};records={}
    for method,g in zip(METHODS,forces,strict=True):
        names=[f'{kind}_{i}' for kind in ('E','R') for i in range(3)]
        d,info=observable_common_direction(gp,gj,gq,g,names,damping=.1,relative_cutoff=1e-6)
        speed=float(np.linalg.norm(fit_a@d)/np.sqrt(2))
        if info['common_descent'] and speed>1e-12:d/=speed
        else:d=np.zeros_like(d);info['common_descent']=False
        variants[method+'_direction']=d;variants[method+'_gradients']=g
        records[method]=dict(solve=info,force_kind=spec['method_meanings'][method],raw_fit_FS_speed=speed,
            finite_fit_slopes=(-finite@d).tolist(),force_fit_slopes=(-g@d).tolist(),normalization='unit SAME FIT-full FS')
    replay=float(np.linalg.norm(variants['fit_direction']-previous['hermitian_direction'])/max(np.linalg.norm(previous['hermitian_direction']),1e-30))
    if replay>2e-9:raise FloatingPointError('previous primary control fails replay')
    known_ref=loaded(ROOT/spec['previous']/arm/'reference.npz')['reference_values']
    np.savez_compressed(folder/'geometry.npz',jacobian=previous['jacobian'],known_jacobian=jac,
        reference_values=previous['reference_values'],known_reference_values=known_ref,
        finite_gradients=finite,**variants)
    atomic_json(folder/'geometry.json',dict(registration_sha256=reg['registration_sha256'],arm=arm,methods=records,
        geometry_sha256=file_sha256(folder/'geometry.npz'),new_Jacobian_records=128,geometry_unchanged=True,
        known_force_records=320,previous_primary_replay_relative_error=replay,population_diagnostics=details,
        force_H2_sha256=file_sha256(output/'known_products.npz'),full_parameter_count=js[0].shape[1],
        parent_parameter_sha256=pytree_sha256(params),peak_rss_bytes=peak_rss_bytes()))
    print(arm,'common force directions',[records[m]['solve']['common_descent'] for m in METHODS],flush=True);check_resources(spec,start)


def proposals(spec: dict,reg: dict,output: Path,arm: str) -> None:
    """Reuse full-H proposals, then additionally protect prior-CAL force block."""
    propose(spec,reg,output,arm)
    blocks=known_blocks(spec);folder=output/arm;saved=loaded(folder/'geometry.npz');b=blocks[2]
    _,initial,generators,projected,*_=setup(spec)
    for method in METHODS:
        path=folder/method/'results.json';row=json.loads(path.read_text());trial=row['trial']
        if trial['raw_candidate_constructed']:
            candidate=load_parameter_artifact(folder/method/'candidate.json',initial[arm])
            scale=json.loads((ROOT/spec['units_source']/'prepared.json').read_text())['residual_scale']
            values=b['carrier']+scale*(batched(lambda x:projected(generators[arm](candidate),x),b['states'],8)-saved['known_reference_values'])
            distance={k:physical_distance(b['carrier'][mask],values[mask],b['q'][mask]) for k,mask in chain_regions(b['groups']).items()}
            trial['extra_known_force_block_distances']=distance
            trial['cache_accepted']=bool(trial['cache_accepted'] and max(v['fs'] for v in distance.values())<=.01*(1+1e-8))
            atomic_json(path,row)


def spectral(spec: dict,reg: dict,output: Path) -> None:
    """Exact imaginary-time carrier witness; not a neural or architecture win."""
    if (output/'spectral.json').exists():raise FileExistsError('spectral control sealed')
    start=perf_counter();blocks=known_blocks(spec);data=loaded(output/'calibration.npz')
    _,_,_,_,carrier,bonds,_,_=setup(spec);units=json.loads((ROOT/spec['units_source']/'prepared.json').read_text())
    new_products,work=carrier_h2(spec,data['states'],carrier,bonds,units['cscale'],start)
    expected=np.column_stack((data['carrier'],36*data['hcarrier']))
    if np.max(abs(new_products[:,:2]-expected))>2e-9*max(1.,float(np.max(abs(expected)))):
        raise FloatingPointError('new calibration carrier C/HC replay fails')
    known=loaded(output/'known_products.npz')['products'];c,hc,h2=known[:64].T
    e=float((np.sum(c.conj()*hc/blocks[0]['q'])/np.sum(abs(c)**2/blocks[0]['q'])).real)
    raw=-(hc-e*c);hraw=-(h2-e*hc)
    speed=coherent_tangent_diagnostics(c,hc,raw,hraw,blocks[0]['q'],36)['tangent_fs_speed']
    np.savez_compressed(output/'spectral_products.npz',products=new_products)
    dc=-(new_products[:,1]-e*new_products[:,0])/speed;dh=-(new_products[:,2]-e*new_products[:,1])/speed
    guard=independent_update_calibration(data['carrier'],36*data['hcarrier'],dc,dh,data['q'],data['groups'],36,maximum_alpha=.1)
    energy_eligible=guard['checks']['nonzero_tangent_all_folds'] and all(v['one_sided_energy_slope_per_site']<0 for v in guard['regions'].values())
    certificate=imaginary_time_energy_certificate(np.array([b[2] for b in bonds]),e)
    policy=CarrierProtection(**{**spec['protection'],'maximum_variance_ratio':1.05});candidate=None
    if energy_eligible:
        all_blocks=[*blocks,data];parent=np.concatenate([b['carrier'] for b in all_blocks]);q=np.concatenate([b['q'] for b in all_blocks])
        delta=np.r_[-(known[:,1]-e*known[:,0])/speed,dc];regions={'full':np.ones(len(q),bool)};offset=0
        for i,b in enumerate(all_blocks):
            for k,mask in chain_regions(b['groups']).items():
                whole=np.zeros(len(q),bool);whole[offset:offset+len(mask)]=mask;regions[f'{i}_{k}']=whole
            offset+=len(b['q'])
        max_speed=max(guard['regions'][k]['tangent_fs_speed'] for k in guard['regions'])
        match=matched_chain_state_step(lambda alpha:parent+alpha*delta,parent,q,regions,.01,
            min(.1,.01/max_speed,speed*certificate['maximum_beta']),12,relative_tolerance=.01)
        cp=data['carrier'];ch=36*data['hcarrier'];p=cp+match.alpha*dc;hp=ch+match.alpha*dh
        gates=[spectral_quality_gate(cp[m],ch[m],p[m],hp[m],data['q'][m],data['groups'][m],policy)
            for key,m in chain_regions(data['groups']).items() if key in ('full','half_0','half_1')]
        candidate=dict(alpha=match.alpha,coefficients=[1+match.alpha*e/speed,-match.alpha/speed],matching=match.as_dict(),
            gates=gates,cache_accepted=bool(match.alpha>0 and all(g['accepted'] for g in gates)),
            variance_risk_cap=1.05,neural_update=False)
    row=dict(registration_sha256=reg['registration_sha256'],anchor_energy_total=e,fit_FS_speed=speed,
        calibration_guard=guard,energy_primary_eligible=energy_eligible,energy_certificate=certificate,candidate=candidate,H2_work=work,
        product_sha256=file_sha256(output/'spectral_products.npz'),accepted=False,neural_win=False,
        spectral_parent='same analytic singlet/A1 carrier',population_descent_identity='dE=-2Var(H)/(N*fit_FS_speed)',peak_rss_bytes=peak_rss_bytes())
    atomic_json(output/'spectral.json',row);print('spectral control energy eligible',energy_eligible,'finite',candidate is not None,flush=True)


def spectral_quality_gate(c: np.ndarray,hc: np.ndarray,p: np.ndarray,hp: np.ndarray,
                          q: np.ndarray,groups: np.ndarray,policy: CarrierProtection) -> dict:
    """Predeclared spectral-only two-SE compatibility; retain original point gate."""
    original=carrier_quality_gate(c,hc,c,hc,p,hp,q,groups,36,policy)
    tolerance=2*original['paired_chain_jackknife_se']['energy_per_site']
    adjusted=carrier_quality_gate(c,hc,c,hc,p,hp,q,groups,36,
        CarrierProtection(**{**policy.__dict__,'energy_tolerance_per_site':tolerance}))
    adjusted.update(original_zero_tolerance_point_gate=original,
        tolerance_rule='two paired delete-chain SE, spectral control ONLY',
        statistical_compatibility_not_resolved_variational_win=True)
    return adjusted


def finish(spec: dict,reg: dict,output: Path) -> None:
    """Neural untouched audit as before; separate spectral validation if needed."""
    finalize(spec,reg,output)
    spectral_row=json.loads((output/'spectral.json').read_text());candidate=spectral_row['candidate']
    # A spectral-only confirmation uses the SAME reserved seed/config. It is
    # not a second audit or a baseline chosen after looking at the audit.
    if candidate and candidate['cache_accepted']:
        start=perf_counter();_,_,_,_,carrier,bonds,_,_=setup(spec);units=json.loads((ROOT/spec['units_source']/'prepared.json').read_text())
        audit_path=output/'audit.npz'
        if audit_path.exists():
            a=loaded(audit_path);data={k:a[k] for k in ('states','q','carrier','hcarrier','groups')}
        else:data,_=draw(proposal_spec(spec,'audit'),36,carrier,bonds,units['cscale'],spec['seeds']['audit'],16)
        products,work=carrier_h2(spec,data['states'],carrier,bonds,units['cscale'],start)
        c,hc=products[:,0],products[:,1];a,b=candidate['coefficients'];p=a*c+b*hc;hp=a*hc+b*products[:,2]
        policy=CarrierProtection(**{**spec['protection'],'maximum_variance_ratio':1.05})
        gate=spectral_quality_gate(c,hc,p,hp,data['q'],data['groups'],policy)
        distances={k:physical_distance(c[m],p[m],data['q'][m]) for k,m in chain_regions(data['groups']).items()}
        metrics=paired_chain_audit(np.column_stack((c,p)),np.column_stack((hc,hp)),data['q'],data['groups'],36,1024,spec['seeds']['bootstrap'],['carrier','imaginary_time'])
        accepted=bool(gate['accepted'] and max(v['fs'] for v in distances.values())<=.01*(1+1e-8))
        np.savez_compressed(output/'spectral_audit.npz',**data,products=products,psi=p,hpsi=hp)
        spectral_row.update(accepted=accepted,audit=dict(gate=gate,distances=distances,metrics=metrics,H2_work=work,
            used_for_gradients=False,no_post_audit_refit=True,product_sha256=file_sha256(output/'spectral_audit.npz')))
        atomic_json(output/'spectral.json',spectral_row)
    final=json.loads((output/'results.json').read_text());final.update(spectral_accepted=spectral_row['accepted'],
        spectral_audit_records=128 if 'audit' in spectral_row else 0,spectral_is_not_neural_win=True,
        untouched_audit_records_total=128 if (output/'audit.npz').exists() or 'audit' in spectral_row else 0)
    coefficients=candidate['coefficients'] if spectral_row['accepted'] else [1.,0.]
    decision=dict(raw_candidate_constructed=candidate is not None,accepted=spectral_row['accepted'],
        final_coefficients=coefficients,neural_update=False)
    save_gated_checkpoint(output/'spectral_checkpoint.pkl',{'coefficients':jnp.asarray(coefficients)},
        optimizer_state=None,run_identity=dict(registration_sha256=reg['registration_sha256'],stage='spectral'),decision=decision)
    atomic_json(output/'results.json',final)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=('register','prepare','geometry','calibrate','propose','spectral','finalize'),required=True)
    parser.add_argument('--arm',choices=('network','free'));args=parser.parse_args();spec=json.loads(args.config.read_text());reg=registration(spec)
    output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists():raise FileExistsError('sealed registration')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if json.loads((output/'registration.json').read_text())!=reg:raise ValueError('numerical registration mismatch')
    if args.stage in ('geometry','propose') and not args.arm:parser.error('--arm required')
    start=perf_counter()
    def alarm(_sig,_frame):raise TimeoutError('bounded process deadline')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='prepare':prepare(spec,reg,output)
        elif args.stage=='geometry':geometry(spec,reg,output,args.arm)
        elif args.stage=='calibrate':calibrate(spec,reg,output)
        elif args.stage=='propose':proposals(spec,reg,output,args.arm)
        elif args.stage=='spectral':spectral(spec,reg,output)
        else:finish(spec,reg,output)
        check_resources(spec,start);atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',dict(wall_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as e:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',dict(type=type(e).__name__,message=str(e),wall_seconds=perf_counter()-start));raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
