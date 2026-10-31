"""One retained-carrier neural/free proposal and a continued-filter control.

Reuse fixed Schur/A1/H kernels, pullbacks, JVP, FS search, node-safe moments,
checkpoint and paired audit. No new head, sampling ladder or old-state refit.
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

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact,save_parameter_artifact
from su2_nqs.models.finite_schur_decoder import FiniteSchurDecoderConfig,make_finite_schur_amplitude
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products,carrier_span_diagnostics
from su2_nqs.vmc.cross_sample_geometry import equal_mass_geometry,observable_common_direction,hermitian_energy_force
from su2_nqs.vmc.full_rayleigh import reverse_rayleigh_weights,full_rayleigh_gradient
from su2_nqs.vmc.joint_moments import residual_moment_cotangents
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.coherent_tangent import coherent_tangent_diagnostics
from su2_nqs.vmc.importance_tangent import tangent_system
from su2_nqs.vmc.update_calibration import independent_update_calibration
from su2_nqs.vmc.carrier_protection import CarrierProtection,carrier_quality_gate,physical_distance
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from .spin_decoder_cost import ROOT,physical_config,peak_rss_bytes
from .spin_finite_schur_optimize import setup,registration as base_registration,check_resources,proposal_spec,full_products
from .spin_finite_schur_direction import candidate_products
from .spin_finite_schur_generalization import trust_regions
from .spin_finite_schur_population import loaded,carrier_h2
from .spin_finite_schur_rayleigh import chain_regions
from .spin_schur_carrier_control import batched,draw

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_reanchor.json'
ARMS=('network','free')
OBJECTIVES=('E_full','E_half_0','E_half_1','R_full','R_half_0','R_half_1')


def configuration(path: Path) -> dict:
    """Keep the existing immutable physical configuration, override experiment roles."""
    new=json.loads(path.read_text());spec={**json.loads((ROOT/new['base_config']).read_text()),**new}
    # Old four-method protocol fields are deliberately not inherited.
    for key in ('methods','primary_method','method_meanings','reuse_h2_source',
                'mechanism','validation','spectral_control','cost'):
        spec.pop(key,None)
    return spec


def retained_coefficients(spec: dict) -> np.ndarray:
    """Require the retained checkpoint and its exact registered physical identity."""
    source=ROOT/spec['carrier_source'];old=json.loads((source/'registration.json').read_text())
    row=json.loads((source/'spectral.json').read_text())
    if not row['accepted'] or row['neural_win']:raise ValueError('retained carrier-only checkpoint required')
    state=load_resumable_checkpoint(source/'spectral_checkpoint.pkl',expected_run_identity=
        dict(registration_sha256=old['registration_sha256'],stage='spectral'))
    coef=np.asarray(state['params']['coefficients'],float)
    np.testing.assert_array_equal(coef,row['candidate']['coefficients'])
    if state['completed_steps']!=1 or coef.shape!=(2,):raise ValueError('one retained degree-one carrier required')
    return coef


def registration(spec: dict) -> dict:
    """Seal one direction per arm and fixed data/budget/retention rules."""
    reg=base_registration(spec);reg.pop('registration_sha256');retained_coefficients(spec)
    old=json.loads((ROOT/spec['carrier_source']/'registration.json').read_text())['config']
    keys=('parent','geometry','rows','columns','j2_over_j1','multiplicity_width','maximum_q',
          'embedding_dim','depth','projection_chunk','H_chunk','H_microbatch','protection')
    if any(spec[k]!=old[k] for k in keys) or spec['steps']!=1:raise ValueError('only physical carrier changes')
    if (spec['direction_damping']!=.1 or spec['relative_cutoff']!=1e-6
            or spec['response_safety']!=.5 or spec['protection']['maximum_variance_ratio']!=1.
            or any(v in old['seeds'].values() for v in spec['seeds'].values())):
        raise ValueError('same fixed direction geometry/protection and fresh seeds required')
    paths=[ROOT/spec['base_config'],ROOT/spec['units_source']/'prepared.json',ROOT/spec['fit_source']/'control.npz',
        ROOT/spec['geometry_source']/'calibration.npz']
    paths += [ROOT/spec['carrier_source']/name for name in ('registration.json','spectral.json','spectral_checkpoint.pkl','known_products.npz')]
    for arm in ARMS:
        paths += [ROOT/spec['force_source']/arm/'products.npz',ROOT/spec['carrier_source']/arm/'geometry.npz']
    reg['inputs'].update({str(p.relative_to(ROOT)):file_sha256(p) for p in paths})
    reg['sources']=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    reg['registration_sha256']=canonical_sha256(reg);return reg


def reanchor(data: dict,powers: np.ndarray,coefficients: np.ndarray,n: int) -> dict:
    """Replace BOTH C and HC; preserve exact fixed Q, configurations and references."""
    p,hp=polynomial_carrier_products(powers,coefficients)
    return {**data,'carrier':p,'hcarrier':hp/n,'base_products':powers}


def prepare(spec: dict,reg: dict,output: Path) -> None:
    """Reuse known exact powers; fix filter direction before any new sample."""
    if (output/'prepared.json').exists():raise FileExistsError('prepared parent sealed')
    source=ROOT/spec['carrier_source'];coef=retained_coefficients(spec)
    fit=loaded(ROOT/spec['fit_source']/'control.npz');geo=loaded(ROOT/spec['geometry_source']/'calibration.npz')
    powers=loaded(source/'known_products.npz')['products'];n=36
    for name,data,subset in (('fit',fit,powers[:64]),('geometry',geo,powers[64:192])):
        np.testing.assert_allclose(subset[:,0],data['carrier'],atol=2e-9,rtol=2e-9)
        np.testing.assert_allclose(subset[:,1],n*data['hcarrier'],atol=2e-9,rtol=2e-9)
        np.savez_compressed(output/(name+'.npz'),**reanchor(data,subset,coef,n))
    parent=reanchor(fit,powers[:64],coef,n);old=json.loads((source/'spectral.json').read_text())
    anchor=old['anchor_energy_total'];raw=anchor*powers[:64,0]-powers[:64,1]
    hraw=anchor*powers[:64,1]-powers[:64,2]
    speed=coherent_tangent_diagnostics(parent['carrier'],n*parent['hcarrier'],raw,hraw,fit['q'],n)['tangent_fs_speed']
    bound=old['energy_certificate']['spectral_radius_bound'] if 'spectral_radius_bound' in old['energy_certificate'] else .75*sum(abs(b[2]) for b in setup(spec)[5])
    beta=-coef[1];maximum_beta=.75/(bound-anchor)
    if speed<=1e-12 or not (0<beta<maximum_beta) or abs(coef[0]-(1+beta*anchor))>1e-12:
        raise ValueError('positive retained filter and nonzero continued direction required')
    units=json.loads((ROOT/spec['units_source']/'prepared.json').read_text())
    atomic_json(output/'prepared.json',dict(registration_sha256=reg['registration_sha256'],
        cscale=units['cscale'],residual_scale=units['residual_scale'],
        coefficients=coef.tolist(),parent_definition='C1=aC0+bHC0; HC1=aHC0+bH2C0, TOTAL H',
        filter_direction_coefficients=(np.array([anchor,-1.])/speed).tolist(),
        filter_fit_FS_speed=speed,filter_maximum_alpha=(maximum_beta-beta)*speed,
        filter_certificate=dict(radius_bound=bound,anchor_energy_total=anchor,
            current_beta=beta,maximum_beta=maximum_beta,minimum_filter_amplitude=.25,
            positive_filter_ratio_monotone_in_energy=True,true_energy_nonincrease=True,
            variance_nonincrease_proved=False,same_degree_one_space_not_second_Lanczos=True),
        new_samples=0,new_carrier_H_centers=0,parent_source_checkpoint_sha256=file_sha256(source/'spectral_checkpoint.pkl'),
        fit_sha256=file_sha256(output/'fit.npz'),geometry_sha256=file_sha256(output/'geometry.npz')))


def directions(spec: dict,reg: dict,output: Path,arm: str) -> None:
    """Recompute forces at C1; reuse ALL original-parameter amplitude Jacobians."""
    folder=output/arm;folder.mkdir();start=perf_counter();fit=loaded(output/'fit.npz');geo=loaded(output/'geometry.npz')
    units=json.loads((output/'prepared.json').read_text());scale=units['residual_scale']
    cached=loaded(ROOT/spec['force_source']/arm/'products.npz');old=loaded(ROOT/spec['carrier_source']/arm/'geometry.npz')
    jac=cached['control_jacobian'];gj=old['jacobian'];p,hp,q=fit['carrier'],36*fit['hcarrier'],fit['q']
    _,initial,generators,_,_,_,plan,_=setup(spec);params=initial[arm];bank,vjp=jax.vjp(generators[arm],params)
    cfg=FiniteSchurDecoderConfig(replace(physical_config(6,6,2),maximum_intermediate_q=4),3,3,16,2)
    pullback=jax.jit(jax.vmap(make_projected_product_pullback(make_finite_schur_amplitude(cfg),36,chunk_size=spec['H_chunk']),in_axes=(None,None,0,0)))
    wp,wh,_=residual_moment_cotangents(p,hp,q,36);reverse=reverse_rayleigh_weights(p,q)
    halves=[];full=[];work=[];regions=chain_regions(fit['groups'])
    for parity in (0,1):
        mask=regions[f'half_{parity}'];sp,sh,_=residual_moment_cotangents(p[mask],hp[mask],q[mask],36)
        weights_p=scale*np.stack((sp,wp[mask],np.zeros(mask.sum()),np.zeros(mask.sum())))
        weights_h=scale*np.stack((sh,wh[mask],reverse_rayleigh_weights(p[mask],q[mask]),reverse[mask]))
        gradients=jax.tree.map(lambda b:jnp.zeros((4,*b.shape),b.dtype),bank);values=np.zeros(4);tick=perf_counter();x=fit['states'][mask]
        for off in range(0,len(x),spec['H_microbatch']):
            sl=slice(off,off+spec['H_microbatch']);prepared=plan(jnp.asarray(x[sl]))
            value,g,*_=pullback(bank,prepared,jnp.asarray(weights_p[:,sl]),jnp.asarray(weights_h[:,sl]))
            gradients=jax.tree.map(lambda a,b:a+b,gradients,g);values+=np.asarray(value)
        rows=[np.asarray(ravel_pytree(vjp(jax.tree.map(lambda b:b[i],gradients))[0])[0]) for i in range(4)]
        wanted=np.real(weights_p@fit['reference_values'][mask]+weights_h@fit['reference_H_total'][mask])
        error=float(np.max(abs(values-wanted)/np.maximum(1.,abs(wanted))))
        if error>2e-10:raise FloatingPointError('four-RHS physical transpose fails')
        halves.append((rows[0],rows[2]));full.append((rows[1],rows[3]))
        work.append(dict(centers=len(x),RHS=4,seconds=perf_counter()-tick,objective_replay_relative_error=error));check_resources(spec,start)
    rforces=np.stack((sum(v[0] for v in full),*[v[0] for v in halves]))
    reverses=[sum(v[1] for v in full),*[v[1] for v in halves]]
    masks=[regions[k] for k in ('full','half_0','half_1')]
    epop=np.stack([hermitian_energy_force(p[m],hp[m],jac[m],q[m],36) for m in masks])
    efinite=np.stack([full_rayleigh_gradient(p[m],hp[m],jac[m],q[m],g)['gradient']/36 for m,g in zip(masks,reverses,strict=True)])
    forces=np.r_[epop,rforces];gp,gjs,gq=equal_mass_geometry([p,geo['carrier']],[jac,gj],[q,geo['q']])
    direction,solve=observable_common_direction(gp,gjs,gq,forces,OBJECTIVES,damping=.1,relative_cutoff=1e-6)
    a,_,_,_=tangent_system(p,hp,jac,q);speed=float(np.linalg.norm(a@direction)/np.sqrt(2))
    direction=direction/speed if solve['common_descent'] and speed>1e-12 else np.zeros_like(direction)
    np.savez_compressed(folder/'direction.npz',direction=direction,gradients=forces,finite_gradients=np.r_[efinite,rforces],
        jacobian=jac,geometry_jacobian=gj,geometry_reference_values=old['reference_values'])
    atomic_json(folder/'direction.json',dict(registration_sha256=reg['registration_sha256'],arm=arm,solve=solve,
        direction_sha256=file_sha256(folder/'direction.npz'),pullback_work=work,full_parameter_count=len(direction),
        fit_unit_FS_speed_before_normalization=speed,force_kind='FIT64 population E plus complete finite-Q R',
        parent_parameter_sha256=pytree_sha256(params),new_fitting_records=0,
        new_population_R_not_claimed=True,architecture_unchanged=True,peak_rss_bytes=peak_rss_bytes()))
    print(arm,'C1 common direction',solve['common_descent'],flush=True)


def calibration(spec: dict,reg: dict,output: Path) -> None:
    """Fresh shared C0-Q records; reconstruct C1 and HC1 without approximations."""
    if (output/'calibration.npz').exists():raise FileExistsError('calibration already consumed')
    seals={a:file_sha256(output/a/'direction.npz') for a in ARMS}
    start=perf_counter();_,_,_,_,carrier,bonds,_,_=setup(spec);units=json.loads((output/'prepared.json').read_text())
    base,info=draw(proposal_spec(spec,'calibration'),36,carrier,bonds,units['cscale'],spec['seeds']['calibration'],16)
    products,work=carrier_h2(spec,base['states'],carrier,bonds,units['cscale'],start)
    np.testing.assert_allclose(products[:,:2],np.c_[base['carrier'],36*base['hcarrier']],atol=2e-9,rtol=2e-9)
    data=reanchor(base,products,np.asarray(units['coefficients']),36)
    np.savez_compressed(output/'calibration.npz',**data)
    atomic_json(output/'calibration.json',dict(registration_sha256=reg['registration_sha256'],direction_seals=seals,
        product_sha256=file_sha256(output/'calibration.npz'),sampling=info,H2_work=work,
        positive_Q_is_fixed_C0_bridge_not_C1_Born=True,used_for_gradients=False))


def quality(data: dict,p: np.ndarray,hp: np.ndarray,spec: dict) -> list[dict]:
    """Unchanged strict full/even/odd C1 protection, not a population theorem."""
    c,hc=data['carrier'],36*data['hcarrier'];regions=chain_regions(data['groups']);policy=CarrierProtection(**spec['protection'])
    return [carrier_quality_gate(c[m],hc[m],c[m],hc[m],p[m],hp[m],data['q'][m],data['groups'][m],36,policy)
        for key,m in regions.items() if key in ('full','half_0','half_1')]


def propose(spec: dict,reg: dict,output: Path,arm: str) -> None:
    """At most one actual neural/free parameter update, no refit from fresh data."""
    folder=output/arm
    if (folder/'proposal.json').exists():raise FileExistsError('proposal sealed')
    fit,geo,new=(loaded(output/(k+'.npz')) for k in ('fit','geometry','calibration'));units=json.loads((output/'prepared.json').read_text())
    saved=loaded(folder/'direction.npz');meta=json.loads((folder/'direction.json').read_text());d=saved['direction'];scale=units['residual_scale']
    seals=json.loads((output/'calibration.json').read_text())
    if seals['direction_seals'][arm]!=file_sha256(folder/'direction.npz') or seals['product_sha256']!=file_sha256(output/'calibration.npz'):
        raise ValueError('directions/calibration changed after sealing')
    n,initial,generators,projected,_,_,plan,evaluate=setup(spec);params=initial[arm];generator=generators[arm];flat,unravel=ravel_pytree(params)
    bank=generator(params);dbank=jax.jvp(generator,(params,),(unravel(-jnp.asarray(d)),))[1];jvp=make_projected_product_jvp(evaluate)
    centers=np.r_[fit['states'],new['states']];arrays=[[] for _ in range(4)];plans=[];start=perf_counter()
    for off in range(0,len(centers),spec['H_microbatch']):
        prepared=plan(jnp.asarray(centers[off:off+spec['H_microbatch']]));hp,p,*_,hd,delta=jvp(bank,dbank,prepared);plans.append(prepared)
        for rows,v in zip(arrays,(p,hp,delta,hd),strict=True):rows.append(np.asarray(v))
        check_resources(spec,start)
    phi,hphi,delta,hdelta=[np.concatenate(v) for v in arrays];delta*=scale;hdelta*=scale
    new.update(reference_values=phi[64:],reference_H_total=hphi[64:])
    replay=max(np.linalg.norm(phi[:64]-fit['reference_values'])/np.linalg.norm(fit['reference_values']),
        np.linalg.norm(hphi[:64]-fit['reference_H_total'])/np.linalg.norm(fit['reference_H_total']))
    record=[coherent_tangent_diagnostics(fit['carrier'][m],36*fit['hcarrier'][m],delta[:64][m],hdelta[:64][m],fit['q'][m],36)
        for m in [chain_regions(fit['groups'])[k] for k in ('full','half_0','half_1')]]
    observed=np.array([v['energy_slope_per_site'] for v in record]+[v['variance_slope_per_site'] for v in record])
    graderror=float(np.max(abs(observed+saved['finite_gradients']@d)))
    jerror=float(np.linalg.norm(delta[:64]+saved['jacobian']@d)/max(np.linalg.norm(delta[:64]),1e-30))
    if replay>2e-9 or graderror>2e-8 or jerror>2e-9:raise FloatingPointError('C1 primal/complete gradient/JVP replay fails')
    guard=independent_update_calibration(new['carrier'],36*new['hcarrier'],delta[64:],hdelta[64:],new['q'],new['groups'],36,maximum_alpha=.1,response_safety=.5)
    fd={}
    if np.linalg.norm(d)>0:
        eps=spec['finite_difference_step']/max(1.,np.linalg.norm(d));first=plans[64//spec['H_microbatch']]
        plus=evaluate(generator(unravel(flat-eps*jnp.asarray(d))),first);minus=evaluate(generator(unravel(flat+eps*jnp.asarray(d))),first)
        m=spec['H_microbatch'];fd=dict(amplitude=float(np.linalg.norm(scale*np.asarray(plus[1]-minus[1])/(2*eps)-delta[64:64+m])/max(np.linalg.norm(delta[64:64+m]),1e-30)),
            H_amplitude=float(np.linalg.norm(scale*np.asarray(plus[0]-minus[0])/(2*eps)-hdelta[64:64+m])/max(np.linalg.norm(hdelta[64:64+m]),1e-30)))
        if max(fd.values())>2e-6:raise FloatingPointError('C1 actual-parameter FD fails')
    span=carrier_span_diagnostics(new['carrier'],36*new['hcarrier'],delta[64:],hdelta[64:],new['base_products'][:,:2],new['base_products'][:,1:3],new['q'],36)
    trial=dict(raw_candidate_constructed=False,accepted=False,cache_accepted=False,eligibility=guard)
    if guard['eligible'] and meta['solve']['common_descent']:
        distance_states=np.r_[fit['states'],geo['states'],new['states']];q=np.r_[fit['q'],geo['q'],new['q']];parent=np.r_[fit['carrier'],geo['carrier'],new['carrier']]
        ref=np.r_[fit['reference_values'],saved['geometry_reference_values'],new['reference_values']]
        def amplitude(alpha):return parent+scale*(batched(lambda x:projected(generator(unravel(flat-alpha*jnp.asarray(d))),x),distance_states,8)-ref)
        match=matched_chain_state_step(amplitude,parent,q,trust_regions([fit,geo,new]),.01,guard['maximum_alpha'],12,relative_tolerance=.01)
        candidate=unravel(flat-match.alpha*jnp.asarray(d));data={k:np.r_[fit[k],new[k]] for k in ('carrier','hcarrier','reference_values','reference_H_total')}
        p,hp,work=candidate_products(generator(candidate),data,plans,evaluate,scale,36)
        expected=np.r_[match.amplitudes[:64],match.amplitudes[192:]]
        error=float(np.max(abs(p-expected)))
        if error>2e-10*max(1.,float(np.max(abs(p)))):raise FloatingPointError('C1 nonlinear amplitude/H replay fails')
        gates=quality(new,p[64:],hp[64:],spec)
        passing=match.alpha>0 and all(g['accepted'] for g in gates) and max(v['fs'] for v in match.distances.values())<=.01*(1+1e-8)
        save_parameter_artifact(folder/'candidate.json',candidate,identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,kind='C1_neural_candidate'))
        np.savez_compressed(folder/'candidate_control.npz',psi=p,hpsi=hp)
        trial.update(raw_candidate_constructed=True,cache_accepted=bool(passing),matching=match.as_dict(),quality_gates=gates,
            candidate_work=work,amplitude_H_replay_error=error,underfilled_FS_not_equal_distance=not match.search['matched'])
    np.savez_compressed(folder/'products.npz',delta=delta,hdelta=hdelta,reference_values=new['reference_values'],reference_H_total=new['reference_H_total'])
    save_gated_checkpoint(folder/'proposal_checkpoint.pkl',params,optimizer_state=dict(direction=d),run_identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='proposal'),decision=trial)
    atomic_json(folder/'proposal.json',dict(registration_sha256=reg['registration_sha256'],trial=trial,fit_tangent=record,
        known_reference_replay_relative_error=float(replay),complete_gradient_slope_error=graderror,cached_Jacobian_error=jerror,
        finite_difference_relative_errors=fd,outside_filter_span=span,JVP_H_centers=192,FD_H_centers=8 if fd else 0,
        JVP_and_proposal_seconds=perf_counter()-start,product_sha256=file_sha256(folder/'products.npz')))
    print(arm,'C1 new eligible',guard['eligible'],'constructed',trial['raw_candidate_constructed'],'calibrated',trial['cache_accepted'],flush=True)


def filter_proposal(spec: dict,reg: dict,output: Path) -> None:
    """Frozen continuation in span(C0,HC0), same actual-chain radius and strict R."""
    if (output/'filter.json').exists():raise FileExistsError('filter candidate sealed')
    fit,geo,new=(loaded(output/(k+'.npz')) for k in ('fit','geometry','calibration'));prep=json.loads((output/'prepared.json').read_text())
    coef=np.asarray(prep['coefficients']);d=np.asarray(prep['filter_direction_coefficients']);data=[fit,geo,new]
    parent=np.r_[tuple(v['carrier'] for v in data)];q=np.r_[tuple(v['q'] for v in data)];powers=np.concatenate([v['base_products'] for v in data])
    tangent,_=polynomial_carrier_products(powers,d);guard=coherent_tangent_diagnostics(new['carrier'],36*new['hcarrier'],*polynomial_carrier_products(new['base_products'],d),new['q'],36)
    maxalpha=min(.1,prep['filter_maximum_alpha'],.01/max(guard['tangent_fs_speed'],1e-30))
    match=matched_chain_state_step(lambda alpha:parent+alpha*tangent,parent,q,trust_regions(data),.01,maxalpha,12,relative_tolerance=.01)
    candidate=coef+match.alpha*d;p,hp=polynomial_carrier_products(new['base_products'],candidate);gates=quality(new,p,hp,spec)
    passing=match.alpha>0 and all(g['accepted'] for g in gates) and max(v['fs'] for v in match.distances.values())<=.01*(1+1e-8)
    atomic_json(output/'filter.json',dict(registration_sha256=reg['registration_sha256'],coefficients=candidate.tolist(),
        trial=dict(raw_candidate_constructed=bool(match.alpha>0),accepted=False,cache_accepted=bool(passing)),
        matching=match.as_dict(),quality_gates=gates,certificate=prep['filter_certificate'],
        underfilled_FS_not_equal_distance=not match.search['matched'],ordinary_carrier_control_not_neural_win=True))
    print('continued filter calibrated',passing,flush=True)


def finalize(spec: dict,reg: dict,output: Path) -> None:
    """One untouched audit for every constructed state, no post-audit tuning."""
    if (output/'results.json').exists():raise FileExistsError('endpoints sealed')
    rows={a:json.loads((output/a/'proposal.json').read_text()) for a in ARMS};fr=json.loads((output/'filter.json').read_text())
    prep=json.loads((output/'prepared.json').read_text());coef=np.asarray(prep['coefficients']);n,initial,generators,_,carrier,bonds,plan,evaluate=setup(spec)
    start=perf_counter();base,sampling=draw(proposal_spec(spec,'audit'),n,carrier,bonds,prep['cscale'],spec['seeds']['audit'],16)
    powers,h2work=carrier_h2(spec,base['states'],carrier,bonds,prep['cscale'],start)
    np.testing.assert_allclose(powers[:,:2],np.c_[base['carrier'],n*base['hcarrier']],atol=2e-9,rtol=2e-9)
    data=reanchor(base,powers,coef,n);p,hp=data['carrier'],n*data['hcarrier'];values=[p];products=[hp];names=['C1'];gates={};distances={};accepted={};works={}
    fc,fh=polynomial_carrier_products(powers,np.asarray(fr['coefficients']));names.append('filter');values.append(fc);products.append(fh)
    positions={tuple(map(int,x)):i for i,x in enumerate(data['states'])}
    def action(x):
        ids=[positions[tuple(map(int,y))] for y in np.asarray(x)];return p[ids],hp[ids]
    reference=generators['network'](initial['network']);regions=chain_regions(data['groups']);policy=CarrierProtection(**spec['protection'])
    for arm,row in rows.items():
        if row['trial']['raw_candidate_constructed']:
            candidate=load_parameter_artifact(output/arm/'candidate.json',initial[arm])
            cp,ch,work=full_products(generators[arm](candidate),reference,data['states'],action,plan,evaluate,scale=prep['residual_scale'],microbatch=spec['H_microbatch'])
            names.append(arm);values.append(cp);products.append(ch);works[arm]=work
    for i,name in enumerate(names[1:],1):
        gates[name]=carrier_quality_gate(p,hp,p,hp,values[i],products[i],data['q'],data['groups'],n,policy);gates[name]['control_used_for_training']=False
        distances[name]={k:physical_distance(p[m],values[i][m],data['q'][m]) for k,m in regions.items()}
        calibration_pass=fr['trial']['cache_accepted'] if name=='filter' else rows[name]['trial']['cache_accepted']
        accepted[name]=bool(calibration_pass and gates[name]['accepted'] and max(v['fs'] for v in distances[name].values())<=.01*(1+1e-8))
    comparisons={}
    for arm in ARMS:
        if arm not in names:continue
        i=names.index(arm);g=carrier_quality_gate(fc,fh,p,hp,values[i],products[i],data['q'],data['groups'],n,policy)
        comparisons[arm]=dict(delta_energy=g['delta_energy_vs_carrier'],delta_variance=g['delta_variance_vs_carrier'],paired_se=g['paired_chain_jackknife_se'],
            two_SE_energy_and_nonworse_point_variance=bool(g['delta_energy_vs_carrier']+2*g['paired_chain_jackknife_se']['energy_per_site']<0 and g['delta_variance_vs_carrier']<=0),
            same_evaluation_records_not_equal_method_cost=True)
    psi,hpsi=np.stack(values,axis=1),np.stack(products,axis=1)
    metrics=paired_chain_audit(psi,hpsi,data['q'],data['groups'],n,1024,spec['seeds']['bootstrap'],names)
    np.savez_compressed(output/'audit.npz',**data,psi=psi,hpsi=hpsi)
    atomic_json(output/'audit.json',dict(registration_sha256=reg['registration_sha256'],names=names,metrics=metrics,gates=gates,
        distances=distances,comparisons_with_filter=comparisons,sampling=sampling,H2_work=h2work,neural_full_H_work=works,
        product_sha256=file_sha256(output/'audit.npz'),used_for_gradients=False,no_post_audit_refit=True))
    for arm in ARMS:
        params=initial[arm];trial=rows[arm]['trial'];keep=accepted.get(arm,False)
        if keep:params=load_parameter_artifact(output/arm/'candidate.json',params)
        decision={**trial,'accepted':keep,'initial_parameter_sha256':pytree_sha256(initial[arm]),'final_parameter_sha256':pytree_sha256(params)}
        identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='final')
        save_parameter_artifact(output/arm/'endpoint.json',params,identity=identity)
        save_gated_checkpoint(output/arm/'checkpoint.pkl',params,optimizer_state=dict(direction=loaded(output/arm/'direction.npz')['direction']),run_identity=identity,decision=decision)
        atomic_json(output/arm/'decision.json',decision)
    filter_keep=accepted.get('filter',False);filter_decision={**fr['trial'],'accepted':filter_keep}
    save_gated_checkpoint(output/'filter_checkpoint.pkl',{'coefficients':np.asarray(fr['coefficients'] if filter_keep else coef)},optimizer_state={},
        run_identity=dict(registration_sha256=reg['registration_sha256'],stage='filter'),decision=filter_decision)
    result=dict(registration_sha256=reg['registration_sha256'],status='complete',physical_parent='retained C1',accepted=accepted,
        comparisons_with_filter=comparisons,constructed_neural_arms=[a for a in ARMS if rows[a]['trial']['raw_candidate_constructed']],
        new_fitting_records=0,new_calibration_records=128,new_untouched_audit_records=128,
        new_carrier_H2_centers=256,new_neural_audit_H_centers=sum(v['centers'] for v in works.values()),
        architecture_unchanged=True,old_trajectory_extended=False,new_ED=False,remote_submitted=False,
        neural_gain_not_implied_by_carrier_filter_improvement=True)
    atomic_json(output/'results.json',result);print(json.dumps(result,indent=2),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',required=True,choices=('register','prepare','directions','calibrate','propose','filter','finalize'));parser.add_argument('--arm',choices=ARMS);args=parser.parse_args()
    spec=configuration(args.config);reg=registration(spec);output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists():raise FileExistsError('registration sealed')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg!=json.loads((output/'registration.json').read_text()):raise ValueError('numeric registration mismatch')
    if args.stage in ('directions','propose') and args.arm is None:parser.error('--arm required')
    start=perf_counter();previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('registered process limit exceeded')));signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='prepare':prepare(spec,reg,output)
        elif args.stage=='directions':directions(spec,reg,output,args.arm)
        elif args.stage=='calibrate':calibration(spec,reg,output)
        elif args.stage=='propose':propose(spec,reg,output,args.arm)
        elif args.stage=='filter':filter_proposal(spec,reg,output)
        else:finalize(spec,reg,output)
        check_resources(spec,start);atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',dict(wall_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start));raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
