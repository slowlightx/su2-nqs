"""Bounded full-parameter force/geometry ladder at the SAME physical parent.

Fit, pool, resolve, Hermitian: change one optimizer ingredient at a time.
Known independent records enter GEOMETRY ONLY. New calibration can bound a
fixed direction, then a separate conditional untouched audit decides commits.
No architecture, head, carrier, symmetry, H kernel or trajectory extension.
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
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact,load_parameter_artifact
from su2_nqs.vmc.cross_sample_geometry import hermitian_energy_force,equal_mass_geometry,observable_common_direction
from su2_nqs.vmc.importance_tangent import make_amplitude_jacobian,tangent_system
from su2_nqs.vmc.joint_moments import normalized_common_moment_direction
from su2_nqs.vmc.update_calibration import independent_update_calibration
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp
from su2_nqs.vmc.carrier_protection import CarrierProtection,carrier_quality_gate,physical_distance,physical_moments
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from .spin_decoder_cost import ROOT,peak_rss_bytes
from .spin_finite_schur_optimize import setup,registration as base_registration,check_resources,proposal_spec,full_products
from .spin_finite_schur_direction import candidate_products
from .spin_finite_schur_jvp import tangent_records
from .spin_finite_schur_rayleigh import chain_regions
from .spin_finite_schur_moments import OBJECTIVES
from .spin_schur_carrier_control import batched,draw

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_generalization.json'
METHODS=('fit','pool','resolve','hermitian')


def registration(spec: dict) -> dict:
    """Seal the ordered comparison, all inputs and data roles before new work."""
    result=base_registration(spec);result.pop('registration_sha256');CarrierProtection(**spec['protection'])
    old=json.loads((ROOT/spec['geometry_source']/'registration.json').read_text())['config']
    keys=('parent','geometry','rows','columns','j2_over_j1','multiplicity_width','maximum_q',
        'embedding_dim','depth','projection_chunk','H_chunk','H_microbatch','protection')
    if any(spec[k]!=old[k] for k in keys):raise ValueError('same physical state, kernels, architecture and protection required')
    if (spec['methods']!=list(METHODS) or spec['primary_method']!='hermitian' or spec['steps']!=1
            or spec['direction_damping']!=.1 or spec['relative_cutoff']!=1e-6
            or spec['JVP_direction_batch']!=2 or spec['response_safety']!=.5):
        raise ValueError('fixed four-method ladder, primary method and geometry rules required')
    if (len(set(spec['seeds'].values()))!=3 or
            any(v in old['seeds'].values() for v in spec['seeds'].values())):
        raise ValueError('fresh calibration and untouched audit seeds required')
    for stage in ('calibration','audit'):
        if (spec[stage]['chains'],spec[stage]['records'])!=(8,16):raise ValueError('eight chains times16 records required')
    paths=[ROOT/spec['fit_source']/'control.npz',ROOT/spec['units_source']/'prepared.json']
    paths+=[ROOT/spec['geometry_source']/p for p in ('registration.json','calibration.npz')]
    paths+=[ROOT/spec['force_source']/'registration.json']
    paths+=[ROOT/spec['force_source']/arm/'products.npz' for arm in ('network','free')]
    result['inputs'].update({str(p.relative_to(ROOT)):file_sha256(p) for p in paths})
    result['sources']=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    result['registration_sha256']=canonical_sha256(result)
    return result


def inputs(spec: dict, arm: str) -> tuple[dict,dict,dict,dict]:
    """Only states, C,Q,groups are read from seen GEOMETRY data; no HC/force."""
    with np.load(ROOT/spec['fit_source']/'control.npz') as p:fit={k:p[k] for k in p.files}
    with np.load(ROOT/spec['geometry_source']/'calibration.npz') as p:
        geo={k:p[k] for k in ('states','carrier','q','groups')}
    with np.load(ROOT/spec['force_source']/arm/'products.npz') as p:
        force={k:p[k] for k in ('control_jacobian','energy_gradients','residual_gradients','direction')}
    units=json.loads((ROOT/spec['units_source']/'prepared.json').read_text())
    return fit,geo,force,units


def directions(spec: dict, fit: dict, geo: dict, cache: dict, geo_jacobian: np.ndarray
               ) -> tuple[dict,dict,np.ndarray,np.ndarray]:
    """Full-model directions; GEOMETRY values do not enter force objectives."""
    n=spec['rows']*spec['columns'];jac=cache['control_jacobian'];c,hc,q=fit['carrier'],n*fit['hcarrier'],fit['q']
    masks=[chain_regions(fit['groups'])[k] for k in ('full','half_0','half_1')]
    populations=np.stack([hermitian_energy_force(c[m],hc[m],jac[m],q[m],n) for m in masks])
    finite=np.r_[cache['energy_gradients'],cache['residual_gradients']]
    hermitian=np.r_[populations,cache['residual_gradients']]
    gp,gj,gq=equal_mass_geometry([c,geo['carrier']],[jac,geo_jacobian],[q,geo['q']])
    variants={};records={};fit_a,_,_,_=tangent_system(c,hc,jac,q)
    for method in METHODS:
        p,j,density=(c,jac,q) if method=='fit' else (gp,gj,gq)
        objectives=hermitian if method=='hermitian' else finite
        if method in ('resolve','hermitian'):
            d,info=observable_common_direction(p,j,density,objectives,OBJECTIVES,
                damping=spec['direction_damping'],relative_cutoff=spec['relative_cutoff'])
        else:
            d,info=normalized_common_moment_direction(p,np.zeros_like(p),j,density,objectives,OBJECTIVES,
                damping=spec['direction_damping'])
        speed=float(np.linalg.norm(fit_a@d)/np.sqrt(2))
        if info['common_descent'] and speed>1e-12:d=d/speed
        else:d=np.zeros_like(d);info['common_descent']=False
        records[method]=dict(solve=info,raw_fit_FS_speed=speed,normalization='unit SAME fit-full physical FS',
            fit_objective_slopes=(-objectives@d).tolist(),finite_fit_slopes=(-finite@d).tolist(),
            force_kind='Hermitian population energy estimator; finite-Q residual derivative' if method=='hermitian' else
                'complete two-sided finite-Q energy and residual derivatives',
            geometry_kind='fit-only ridge' if method=='fit' else 'equal-parent-mass fit + independent geometry',
            resolved_range=method in ('resolve','hermitian'))
        variants[method]=d
    replay=float(np.linalg.norm(variants['fit']-cache['direction'])/max(np.linalg.norm(cache['direction']),1e-30))
    if replay>2e-9:raise FloatingPointError('fit control does not replay the sealed full-model joint direction')
    records['fit']['sealed_direction_relative_error']=replay
    return variants,records,finite,hermitian


def geometry(spec: dict, reg: dict, output: Path, arm: str) -> None:
    """Use seen independent configurations for Jacobian geometry ONLY."""
    folder=output/arm;folder.mkdir(exist_ok=True)
    if any(folder.iterdir()):raise FileExistsError('completed/partial geometry preserved')
    start=perf_counter();fit,geo,cache,units=inputs(spec,arm)
    _,initial,generators,projected,_,_,_,_=setup(spec);params,generator=initial[arm],generators[arm]
    fn=lambda p,x:units['residual_scale']*projected(generator(p),x[None])[0]
    jacobian=make_amplitude_jacobian(fn,params,chunk_size=4);tick=perf_counter()
    jac=np.asarray(jacobian(params,jnp.asarray(geo['states'])));jac_seconds=perf_counter()-tick
    variants,records,finite,hermitian=directions(spec,fit,geo,cache,jac)
    reference=batched(lambda x:projected(generator(params),x),geo['states'],8)
    np.savez_compressed(folder/'geometry.npz',jacobian=jac,reference_values=reference,
        finite_gradients=finite,hermitian_gradients=hermitian,**{k+'_direction':v for k,v in variants.items()})
    atomic_json(folder/'geometry.json',dict(registration_sha256=reg['registration_sha256'],arm=arm,
        methods=records,new_Jacobian_records=len(jac),Jacobian_seconds=jac_seconds,
        geometry_force_HC_not_read=True,geometry_forces_not_fitted=True,old_audits_not_read=True,
        full_parameter_count=jac.shape[1],parameter_sha256=pytree_sha256(params),
        geometry_sha256=file_sha256(folder/'geometry.npz'),peak_rss_bytes=peak_rss_bytes()))
    check_resources(spec,start);print(arm,'geometry ranks',[records[k]['solve'].get('resolved_rank') for k in METHODS],flush=True)


def calibrate(spec: dict, reg: dict, output: Path) -> None:
    """One new shared calibration, AFTER all geometry-only directions are sealed."""
    if (output/'calibration.npz').exists():raise FileExistsError('calibration already consumed')
    seals={a:file_sha256(output/a/'geometry.npz') for a in ('network','free')}
    n,_,_,_,carrier,bonds,_,_=setup(spec)
    units=json.loads((ROOT/spec['units_source']/'prepared.json').read_text())
    data,info=draw(proposal_spec(spec,'calibration'),n,carrier,bonds,units['cscale'],spec['seeds']['calibration'],spec['calibration']['records'])
    np.savez_compressed(output/'calibration.npz',**data)
    atomic_json(output/'calibration.json',dict(registration_sha256=reg['registration_sha256'],direction_seals=seals,
        calibration_sha256=file_sha256(output/'calibration.npz'),sampling=info,used_for_gradients=False,
        used_for_step_selection=True,is_untouched_audit=False))
    print('shared fresh calibration',len(data['q']),flush=True)


def trust_regions(data: list[dict]) -> dict[str,np.ndarray]:
    """Whole union and each dataset/full/folds/chain; one global phase retained."""
    size=sum(len(d['q']) for d in data);regions=dict(full=np.ones(size,bool));offset=0
    for name,d in zip(('fit','geometry','calibration'),data,strict=True):
        for key,mask in chain_regions(d['groups']).items():
            full=np.zeros(size,bool);full[offset:offset+len(mask)]=mask;regions[name+'_'+key]=full
        offset+=len(d['q'])
    return regions


def propose(spec: dict, reg: dict, output: Path, arm: str) -> dict:
    """One direction/proposal per method; no refit from calibration or audit."""
    folder=output/arm;start=perf_counter();fit,geo,cache,units=inputs(spec,arm);scale=units['residual_scale']
    with np.load(output/'calibration.npz') as p:new={k:p[k] for k in p.files}
    with np.load(folder/'geometry.npz') as p:saved={k:p[k] for k in p.files}
    metadata=json.loads((folder/'geometry.json').read_text())
    sealed=json.loads((output/'calibration.json').read_text())
    if (metadata['geometry_sha256']!=file_sha256(folder/'geometry.npz') or
            sealed['direction_seals'][arm]!=metadata['geometry_sha256'] or
            sealed['calibration_sha256']!=file_sha256(output/'calibration.npz')):
        raise ValueError('directions or fresh calibration changed after sealing')
    n,initial,generators,projected,_,_,plan,evaluate=setup(spec)
    params,generator=initial[arm],generators[arm];flat,unravel=ravel_pytree(params);bank=generator(params)
    if any((folder/m).exists() for m in METHODS):raise FileExistsError('completed/partial proposals preserved')
    all_states=np.r_[fit['states'],new['states']];count_fit=len(fit['q']);m=spec['H_microbatch']
    plans=[plan(jnp.asarray(all_states[i:i+m])) for i in range(0,len(all_states),m)]
    scalar=make_projected_product_jvp(evaluate)
    batch=jax.jit(jax.vmap(scalar,in_axes=(None,0,None)))
    products={};work=[]
    for begin in range(0,len(METHODS),spec['JVP_direction_batch']):
        names=METHODS[begin:begin+spec['JVP_direction_batch']]
        dbanks=[jax.jvp(generator,(params,),(unravel(-jnp.asarray(saved[name+'_direction'])),))[1] for name in names]
        dbanks=jax.tree.map(lambda *a:jnp.stack(a),*dbanks)
        arrays=[[] for _ in range(4)];tick=perf_counter();counts=[]
        for prepared in plans:
            hp,p,count,padded,hd,d=batch(bank,dbanks,prepared)
            for rows,v in zip(arrays,(p,hp,d,hd),strict=True):rows.append(np.asarray(v))
            counts.append((int(count[0]),int(padded[0])))
        phi,hphi,delta,hdelta=[np.concatenate(a,axis=1) for a in arrays]
        for i,name in enumerate(names):products[name]=(scale*delta[i],scale*hdelta[i])
        work.append(dict(methods=list(names),seconds=perf_counter()-tick,
            primal_plus_tangent_calls=len(plans),directions_per_call=len(names),
            derivative_scalar_center_equivalents=len(all_states)*len(names),
            unique_raw_queries_per_direction=sum(v[0] for v in counts),padded_raw_queries_per_direction=sum(v[1] for v in counts)))
        if begin==0:
            new['reference_values'],new['reference_H_total']=phi[0,count_fit:],hphi[0,count_fit:]
            replay=max(np.linalg.norm(phi[0,:count_fit]-fit['reference_values'])/max(np.linalg.norm(fit['reference_values']),1e-30),
                np.linalg.norm(hphi[0,:count_fit]-fit['reference_H_total'])/max(np.linalg.norm(fit['reference_H_total']),1e-30))
            if replay>2e-9:raise FloatingPointError('fit reference primal replay fails')
        check_resources(spec,start)
    np.savez_compressed(folder/'reference.npz',reference_values=new['reference_values'],reference_H_total=new['reference_H_total'])
    moment_data={k:np.r_[fit[k],new[k]] for k in ('states','q','carrier','hcarrier','reference_values','reference_H_total')}
    moment_data['groups']=np.r_[fit['groups'],new['groups']+8]
    distance_states=np.r_[fit['states'],geo['states'],new['states']]
    distance_q=np.r_[fit['q'],geo['q'],new['q']]
    parent=np.r_[fit['carrier'],geo['carrier'],new['carrier']]
    references=np.r_[fit['reference_values'],saved['reference_values'],new['reference_values']]
    regions=trust_regions([fit,geo,new]);policy=CarrierProtection(**spec['protection']);rows={}
    for method in METHODS:
        destination=folder/method;destination.mkdir();direction=saved[method+'_direction'];delta,hdelta=products[method]
        fd,hd=delta[:count_fit],hdelta[:count_fit];nd,nh=delta[count_fit:],hdelta[count_fit:]
        record=tangent_records(fit,fd,hd,n)
        observed=np.array([record['full']['energy_slope_per_site'],*[r['energy_slope_per_site'] for r in record['halves']],
            record['full']['variance_slope_per_site'],*[r['variance_slope_per_site'] for r in record['halves']]])
        error=float(np.max(abs(observed+saved['finite_gradients']@direction)))
        jerr=float(np.linalg.norm(fd+cache['control_jacobian']@direction)/max(np.linalg.norm(fd),1e-30))
        if error>2e-8 or jerr>2e-9:raise FloatingPointError('full gradient/Jacobian vs actual H JVP fails')
        check=independent_update_calibration(new['carrier'],n*new['hcarrier'],nd,nh,new['q'],new['groups'],n,
            maximum_alpha=spec['direction_step_size'],response_safety=spec['response_safety'])
        errors={};fd_centers=0
        if np.linalg.norm(direction)>0:
            eps=spec['finite_difference_step']/max(1.,np.linalg.norm(direction));prepared=plans[count_fit//m]
            plus=evaluate(generator(unravel(flat-eps*jnp.asarray(direction))),prepared)
            minus=evaluate(generator(unravel(flat+eps*jnp.asarray(direction))),prepared)
            errors=dict(amplitude=float(np.linalg.norm(scale*np.asarray(plus[1]-minus[1])/(2*eps)-nd[:m])/max(np.linalg.norm(nd[:m]),1e-30)),
                H_amplitude=float(np.linalg.norm(scale*np.asarray(plus[0]-minus[0])/(2*eps)-nh[:m])/max(np.linalg.norm(nh[:m]),1e-30)))
            if max(errors.values())>2e-6:raise FloatingPointError('new actual-parameter full-H FD fails')
            fd_centers=2*m
        trial=dict(raw_candidate_constructed=False,accepted=False,cache_accepted=False,
            candidate_H_centers=0,eligibility=check,final_parameter_sha256=pytree_sha256(params))
        candidate_work=None
        if check['eligible'] and metadata['methods'][method]['solve']['common_descent']:
            def candidate(alpha):
                b=generator(unravel(flat-alpha*jnp.asarray(direction)))
                return parent+scale*(batched(lambda x:projected(b,x),distance_states,8)-references)
            match=matched_chain_state_step(candidate,parent,distance_q,regions,policy.maximum_step_fs,check['maximum_alpha'],12,
                relative_tolerance=spec['matching_relative_tolerance'])
            proposed=unravel(flat-match.alpha*jnp.asarray(direction))
            psi,hpsi,candidate_work=candidate_products(generator(proposed),moment_data,plans,evaluate,scale,n)
            expected=np.r_[match.amplitudes[:count_fit],match.amplitudes[count_fit+len(geo['q']):]]
            discrepancy=float(np.max(abs(psi-expected)))
            if discrepancy>2e-10*max(1.,float(np.max(abs(psi)))):raise FloatingPointError('candidate amplitude/H replay fails')
            cp,ch=new['carrier'],n*new['hcarrier'];npsi,nhpsi=psi[count_fit:],hpsi[count_fit:]
            gates=[];ray=[]
            for mask in [chain_regions(new['groups'])[key] for key in ('full','half_0','half_1')]:
                gates.append(carrier_quality_gate(cp[mask],ch[mask],cp[mask],ch[mask],npsi[mask],nhpsi[mask],new['q'][mask],new['groups'][mask],n,policy))
                before=physical_moments(cp[mask],ch[mask],new['q'][mask],n)
                after=physical_moments(cp[mask]+match.alpha*nd[mask],ch[mask]+match.alpha*nh[mask],new['q'][mask],n)
                ray.append(dict(delta_energy=after['energy_per_site']-before['energy_per_site'],delta_residual=after['variance_per_site']-before['variance_per_site']))
            ray_pass=all(r['delta_energy']<=0 and r['delta_residual']<=0 for r in ray)
            passing=bool(match.alpha>0 and ray_pass and all(g['accepted'] for g in gates) and
                max(v['fs'] for v in match.distances.values())<=policy.maximum_step_fs*(1+1e-8))
            identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,method=method,kind='single_raw_proposal')
            save_parameter_artifact(destination/'candidate.json',proposed,identity=identity)
            np.savez_compressed(destination/'candidate_control.npz',psi=psi,hpsi=hpsi)
            trial.update(raw_candidate_constructed=True,cache_accepted=passing,matching=match.as_dict(),quality_gates=gates,
                tangent_ray_moments=ray,tangent_ray_moment_gate=ray_pass,candidate_H_centers=len(psi),
                candidate_parameter_sha256=pytree_sha256(proposed),candidate_sha256=file_sha256(destination/'candidate.json'),
                amplitude_H_replay_error=discrepancy,underfilled_FS_not_equal_distance=not match.search['matched'])
        np.savez_compressed(destination/'products.npz',direction=direction,fit_delta=fd,fit_hdelta=hd,delta=nd,hdelta=nh)
        row=dict(registration_sha256=reg['registration_sha256'],arm=arm,method=method,trial=trial,
            optimizer=metadata['methods'][method],fit_tangent=record,calibration_guard=check,
            fit_complete_gradient_slope_error=error,fit_amplitude_Jacobian_relative_error=jerr,
            finite_difference_relative_errors=errors,finite_difference_H_centers=fd_centers,candidate_work=candidate_work,
            JVP_scalar_H_centers=len(all_states),product_sha256=file_sha256(destination/'products.npz'),
            fixed_direction_not_refitted=True,calibration_used_for_step_not_gradients=True)
        save_gated_checkpoint(destination/'proposal_checkpoint.pkl',params,optimizer_state=dict(direction=direction),
            run_identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,method=method,stage='proposal'),decision=trial)
        atomic_json(destination/'results.json',row);rows[method]=row
        print(arm,method,'new slopes',[(r['energy_slope_per_site'],r['variance_slope_per_site']) for r in check['regions'].values()],
            'eligible',check['eligible'],'finite',trial['raw_candidate_constructed'],'calibrated',trial['cache_accepted'],flush=True)
        check_resources(spec,start)
    result=dict(registration_sha256=reg['registration_sha256'],arm=arm,methods=list(rows),JVP_work=work,
        reference_fit_replay_relative_error=float(replay),reference_sha256=file_sha256(folder/'reference.npz'),
        new_fit_records=0,new_geometry_records=0,new_calibration_records=len(new['q']),new_pullback_centers=0,
        complete_full_parameter_updates=True,architecture_unchanged=True,old_audits_not_read=True,peak_rss_bytes=peak_rss_bytes())
    atomic_json(folder/'results.json',result);return result


def finalize(spec: dict, reg: dict, output: Path) -> dict:
    """One common untouched audit; report every constructed proposal, no selection."""
    if (output/'results.json').exists():raise FileExistsError('final endpoints sealed')
    rows={a+'_'+m:json.loads((output/a/m/'results.json').read_text()) for a in ('network','free') for m in METHODS}
    constructed=[name for name,r in rows.items() if r['trial']['raw_candidate_constructed']]
    passing=[name for name,r in rows.items() if r['trial']['cache_accepted']];accepted={name:False for name in rows}
    n,initial,generators,_,carrier,bonds,plan,evaluate=setup(spec)
    units=json.loads((ROOT/spec['units_source']/'prepared.json').read_text());works={};audit=None
    if passing:
        data,info=draw(proposal_spec(spec,'audit'),n,carrier,bonds,units['cscale'],spec['seeds']['audit'],spec['audit']['records'])
        c,hc=data['carrier'],n*data['hcarrier'];positions={tuple(map(int,x)):i for i,x in enumerate(data['states'])}
        def action(x):
            ids=[positions[tuple(map(int,y))] for y in np.asarray(x)];return c[ids],hc[ids]
        values,products=[c],[hc];gates={};distances={};regions=chain_regions(data['groups']);policy=CarrierProtection(**spec['protection'])
        reference=generators['network'](initial['network'])
        for name in constructed:
            arm,method=name.split('_');folder=output/arm/method
            candidate=load_parameter_artifact(folder/'candidate.json',initial[arm])
            p,hp,work=full_products(generators[arm](candidate),reference,data['states'],action,plan,evaluate,scale=units['residual_scale'],microbatch=spec['H_microbatch'])
            values.append(p);products.append(hp);works[name]=work
            gates[name]=carrier_quality_gate(c,hc,c,hc,p,hp,data['q'],data['groups'],n,policy)
            gates[name]['control_used_for_training']=False
            distances[name]={key:physical_distance(c[mask],p[mask],data['q'][mask]) for key,mask in regions.items()}
            accepted[name]=bool(name in passing and gates[name]['accepted'] and max(v['fs'] for v in distances[name].values())<=policy.maximum_step_fs*(1+1e-8))
        values,products=np.stack(values,axis=1),np.stack(products,axis=1)
        metrics=paired_chain_audit(values,products,data['q'],data['groups'],n,spec['audit']['bootstrap_replicates'],spec['seeds']['bootstrap'],['carrier',*constructed])
        np.savez_compressed(output/'audit.npz',**data,psi=values,hpsi=products)
        audit=dict(metrics=metrics,gates=gates,distances=distances,sampling=info,full_H_work=works,
            audit_sha256=file_sha256(output/'audit.npz'),used_for_gradients=False,no_post_audit_refit=True,
            primary_method=spec['primary_method'],multiple_comparisons_not_best_method_selection=True)
        atomic_json(output/'audit.json',audit)
    for name,row in rows.items():
        arm,method=name.split('_');folder=output/arm/method;params=initial[arm]
        if accepted[name]:params=load_parameter_artifact(folder/'candidate.json',initial[arm])
        decision={**row['trial'],'accepted':accepted[name],'initial_parameter_sha256':pytree_sha256(initial[arm]),
            'final_parameter_sha256':pytree_sha256(params),'untouched_audit_performed':audit is not None}
        identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,method=method,stage='final')
        save_parameter_artifact(folder/'endpoint.json',params,identity=identity)
        with np.load(folder/'products.npz') as p:direction=p['direction']
        save_gated_checkpoint(folder/'checkpoint.pkl',params,optimizer_state=dict(direction=direction),run_identity=identity,decision=decision)
        atomic_json(folder/'decision.json',decision)
    result=dict(registration_sha256=reg['registration_sha256'],primary_method=spec['primary_method'],accepted=accepted,
        accepted_proposals=sum(accepted.values()),constructed=constructed,calibration_passing=passing,
        new_audit_records=128 if audit else 0,new_audit_H_centers=sum(w['centers'] for w in works.values()),
        new_network_training_records=0,trajectory_extended=False,new_ED=False,remote_submitted=False,
        architecture_unchanged=True,independent_point_acceptance_not_PRL_win=True)
    atomic_json(output/'results.json',result);print(json.dumps(result,indent=2),flush=True);return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--stage',choices=('register','geometry','calibrate','propose','finalize'),required=True)
    parser.add_argument('--arm',choices=('network','free'));args=parser.parse_args()
    spec=json.loads(args.config.read_text());reg=registration(spec);output=ROOT/spec['output'];output.mkdir(parents=True,exist_ok=True)
    if args.stage=='register':
        if (output/'registration.json').exists():raise FileExistsError('registration sealed')
        atomic_json(output/'registration.json',reg);print(reg['registration_sha256']);return
    if reg!=json.loads((output/'registration.json').read_text()):raise ValueError('numeric registration mismatch')
    if args.stage in ('geometry','propose') and args.arm is None:parser.error('--arm required')
    start=perf_counter()
    def alarm(_signal,_frame):raise TimeoutError('registered process limit exceeded')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        if args.stage=='geometry':geometry(spec,reg,output,args.arm)
        elif args.stage=='calibrate':calibrate(spec,reg,output)
        elif args.stage=='propose':propose(spec,reg,output,args.arm)
        else:finalize(spec,reg,output)
        check_resources(spec,start)
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_execution.json',dict(wall_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),complete=True))
    except Exception as error:
        atomic_json(output/f'{args.stage}_{args.arm or "shared"}_failure.json',dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start));raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
