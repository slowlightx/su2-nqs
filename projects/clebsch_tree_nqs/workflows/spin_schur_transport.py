"""Three-edge route-1 canary: form joint directions AFTER physical retraction.

Reuse the saved chi4 parent and known quadrature, with the SAME physical
carrier/decoder/projector/H. No new head, capacity, old-audit fitting or ED.
The transport probes and actual full-H candidates have hard finite budgets.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact,save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.models.schur_metric_retraction import split_physical_reduced_pair
from su2_nqs.models.schur_two_site import embed_reduced_bank,two_site_layout,join_reduced_pair,make_two_site_features
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products
from su2_nqs.vmc.cutset_training import paired_chain_audit,weighted_state_distance
from su2_nqs.vmc.joint_linear_step import joint_linear_coefficients,linear_moment_slopes
from su2_nqs.vmc.projected_spin_linear import make_projected_linear_products
from su2_nqs.vmc.retraction_transport import retracted_real_columns,joint_real_coefficients
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint,save_resumable_checkpoint
from . import spin_schur_metric_capacity as previous

base=previous.base
CONFIG=base.ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_transport.json'


def setup(spec: dict):
    small,initial,carrier,bonds,projector,coef,units=base.parent_setup(spec)
    cfg=replace(small,tree=replace(small.tree,feature_dim=4))
    reference=embed_reduced_bank(initial,small,cfg)
    folder=base.ROOT/spec['parent_source']
    parent=load_parameter_artifact(folder/'chi4/endpoint.json',reference)
    return cfg,reference,parent,carrier,bonds,projector,coef,units


def prepare(spec: dict, output: Path) -> None:
    """Seal sources and fixed budget BEFORE transporting or evaluating candidates."""
    if (spec['edges']!=[[2,0],[6,3],[6,5]] or spec['active_width']!=4
            or spec['probe_fs']!=1e-4 or spec['max_actual_candidates']!=2):
        raise ValueError('fixed three-edge transport budget required')
    reg=previous.registration(spec,CONFIG)
    old=base.ROOT/spec['parent_source']
    for name in ('registration.json','prepared.json','known.npz','chi4/endpoint.json',
                 'chi4/endpoint.npz','chi4/endpoint_known.npz','chi4/checkpoint.pkl','chi4/results.json'):
        reg['inputs'][str((old/name).relative_to(base.ROOT))]=file_sha256(old/name)
    reg['sources']=transitive_local_source_manifest((Path(__file__),),root=base.ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    reg.pop('registration_sha256');reg['registration_sha256']=canonical_sha256(reg)
    output.mkdir(exist_ok=False);atomic_json(output/'registration.json',reg)
    known=base.loaded(old/'known.npz');saved=base.loaded(old/'chi4/endpoint_known.npz')
    np.savez_compressed(output/'known.npz',**known)
    np.savez_compressed(output/'parent_known.npz',**saved)
    atomic_json(output/'prepared.json',dict(registration_sha256=reg['registration_sha256'],
        known_sha256=file_sha256(output/'known.npz'),parent_known_sha256=file_sha256(output/'parent_known.npz'),
        new_training_samples=0,new_reference_H_calls=0,old_audits_used_to_fit=False))
    print('sealed transport: three internal edges, chi4; reused192 amplitudes/32 H centers',flush=True)


def carrier_complement(tangent: np.ndarray, powers: np.ndarray, q: np.ndarray) -> list[float]:
    """Diagnostic outside C0/HC0 AMPLITUDE span; not energy beyond filtering."""
    a=powers[:,:2]/np.sqrt(q[:,None]);v=tangent/np.sqrt(q[:,None])
    coef=np.linalg.lstsq(a,v,rcond=1e-10)[0]
    return (np.sum(abs(v-a@coef)**2,axis=0)/np.maximum(np.sum(abs(v)**2,axis=0),1e-300)).tolist()


def train(spec: dict, reg: dict, output: Path) -> None:
    """Three finite attempts, actual retracted H gate; never a long trajectory."""
    start=perf_counter();cfg,reference,parent,_,bonds,projector,_,units=setup(spec)
    known=base.loaded(output/'known.npz');saved=base.loaded(output/'parent_known.npz')
    x,q,hi=known['states'],known['q'],known['H_indices'];nf=spec['H_centers_per_block'];scale=units['residual_scale']
    bank=parent;phi,p,hp=saved['phi'],saved['psi'],saved['hpsi'];history=[];completed=0
    identity=dict(registration_sha256=reg['registration_sha256'],width=4)
    if (output/'checkpoint.pkl').exists():
        check=load_resumable_checkpoint(output/'checkpoint.pkl',expected_run_identity=identity)
        bank,history,completed=check['params'],check['history'],check['completed_steps']
        saved=check['optimizer_state']['known'];phi,p,hp=saved['phi'],saved['psi'],saved['hpsi']
    c,hc=known['carrier'],known['hcarrier'];carrier_m=previous.block_moments(c[hi],hc[hi],q[hi],nf)
    blocks=[np.arange(len(hi))<nf,np.arange(len(hi))>=nf]
    _,plan,evaluate=base.bank_functions(cfg,projector,bonds,spec)
    source=base.ROOT/spec['carrier_source']
    powers=np.concatenate([base.loaded(source/(k+'.npz'))['base_products'] for k in ('fit','geometry')])
    for attempt,pair in enumerate(spec['edges']):
        if attempt<completed:continue
        tick=perf_counter();layout=two_site_layout(cfg,*pair);joined=join_reduced_pair(bank,layout)
        lp,la,fp=make_projected_linear_products(make_two_site_features(layout),projector,bonds,chunk_size=spec['linear_H_chunk'])
        lp,la,fp=map(jax.jit,(lp,la,fp));f=base.batched(lambda z:fp(bank,z),x,2)
        replay=float(np.max(abs(scale*(f@joined-phi))))
        if replay>2e-8:raise FloatingPointError('parent feature replay failed')
        raw,metric=base.search_directions(f[hi],p[hi],hp,q[hi],scale,spec)
        def retract(z):
            candidate,_=split_physical_reduced_pair(z,bank,layout,metric_cutoff=spec['retraction_metric_cutoff'])
            return join_reduced_pair(candidate,layout)
        transported,transport=retracted_real_columns(joined,raw,retract,f,p,q,amplitude_scale=scale,
            probe_fs=spec['probe_fs'],maximum_relative_defect=spec['maximum_transport_defect'])
        row=dict(attempt=attempt,pair=pair,accepted=False,feature_replay_error=replay,metric=metric,
            transport=transport,before=previous.block_moments(p[hi],hp,q[hi],nf),actual_candidates=[],probe_count=0)
        if transported.shape[1]:
            columns=np.c_[raw,transported];actions=[];values=[];query_work=[]
            for i in range(0,len(hi),spec['H_microbatch']):
                prepared=lp(jnp.asarray(x[hi[i:i+spec['H_microbatch']]]))
                h,a,count,padded=la(bank,jnp.asarray(columns),prepared)
                values.append(np.asarray(a));actions.append(np.asarray(h))
                query_work.append(dict(unique=int(count),padded=int(padded)));base.resources(spec,start)
            t,ht=scale*np.concatenate(values),scale*np.concatenate(actions);k=raw.shape[1]
            linear_replay=float(np.max(abs(t-scale*f[hi]@columns)))
            if linear_replay>2e-8:raise FloatingPointError('linear H amplitude replay failed')
            rawcoef,rawinfo=joint_linear_coefficients(p[hi],hp,t[:,:k],ht[:,:k],q[hi],blocks,n_sites=36,damping=spec['metric_damping'])
            coef,info=joint_real_coefficients(p[hi],hp,t[:,k:],ht[:,k:],q[hi],blocks,n_sites=36,damping=spec['metric_damping'])
            rawreal=np.r_[rawcoef.real,rawcoef.imag]
            mapped=np.asarray([rawreal[v['input_column']]*v['retracted_fs_speed'] for v in transport['columns'] if v['retained']])
            diagnostic={}
            for name,dirs,hd,weights in (('unretracted',t[:,:k],ht[:,:k],rawcoef),
                                       ('same_ray_after_transport',t[:,k:],ht[:,k:],mapped)):
                diagnostic[name]=[{key:float(value[0]) for key,value in linear_moment_slopes(p[hi][m],hp[m],
                    (dirs[m]@weights)[:,None],(hd[m]@weights)[:,None],q[hi][m],36).items()} for m in blocks]
            row.update(raw_joint_geometry=rawinfo,joint_geometry=info,slopes=diagnostic,
                carrier_amplitude_complement=carrier_complement(scale*f@transported,powers,q),
                linear_H_columns=columns.shape[1],linear_H_centers=len(hi),linear_work=query_work,
                linear_amplitude_replay_error=linear_replay,full_H_equivalent_columns=len(hi)*columns.shape[1])
            options=[]
            if info['common_descent']:
                delta=transported@coef*spec['maximum_step_fs']
                for alpha in spec['step_fractions']:
                    row['probe_count']+=1;desired=joined+alpha*delta
                    linear_p=p+scale*f@(alpha*delta);linear_h=hp+alpha*spec['maximum_step_fs']*ht[:,k:]@coef
                    prediction=previous.block_moments(linear_p[hi],linear_h,q[hi],nf)
                    if not previous.quality(row['before'],prediction,carrier_m,spec)['accepted']:continue
                    candidate,trunc=split_physical_reduced_pair(desired,bank,layout,metric_cutoff=spec['retraction_metric_cutoff'])
                    cz=join_reduced_pair(candidate,layout);candidate_phi=f@cz
                    cp=c+scale*(candidate_phi-known['reference'])
                    distances=[weighted_state_distance(p[m],cp[m],q[m]) for m in (slice(0,64),slice(64,None))]
                    parent_dist=[weighted_state_distance(c[m],cp[m],q[m]) for m in (slice(0,64),slice(64,None))]
                    td=weighted_state_distance(linear_p,cp,q);fs=max(d['fs'] for d in distances)
                    if (fs>spec['maximum_step_fs'] or max(d['fs'] for d in parent_dist)>spec['maximum_parent_fs']
                            or min(d['ess_fraction'] for d in distances)<spec['minimum_ess_fraction']
                            or td['fs']>spec['maximum_truncation_fs_fraction']*max(fs,1e-6)):continue
                    gain=np.mean([b['energy_per_site']-a['energy_per_site'] for a,b in zip(prediction,row['before'])])
                    options.append((gain,alpha,candidate,candidate_phi,cp,trunc,distances,parent_dist,td,prediction))
            options.sort(key=lambda v:-v[0]);row['eligible_amplitude_options']=len(options);accepted=[]
            for option in options[:spec['max_actual_candidates']]:
                _,alpha,candidate,cz,cp,trunc,ds,pds,td,prediction=option
                ca,ch,work=base.products(candidate,x[hi],plan,evaluate)
                chp=hc[hi]+scale*(ch-known['reference_H'])
                error=float(np.max(abs(scale*(ca-cz[hi]))))
                if error>2e-8:raise FloatingPointError('candidate projected H amplitude replay failed')
                after=previous.block_moments(cp[hi],chp,q[hi],nf);gate=previous.quality(row['before'],after,carrier_m,spec)
                record=dict(alpha=alpha,truncation=trunc,distances=ds,parent_distances=pds,
                    physical_truncation_distance=td,predicted_uncompressed_moments=prediction,after=after,
                    gate=gate,candidate_work=work,retraction_H_amplitude_error=error)
                row['actual_candidates'].append(record)
                if gate['accepted']:
                    gain=np.mean([b['energy_per_site']-a['energy_per_site'] for a,b in zip(after,row['before'])])
                    accepted.append((gain,candidate,cz,cp,chp,len(row['actual_candidates'])-1))
            if accepted:
                _,bank,phi,p,hp,selected=max(accepted,key=lambda v:v[0]);row.update(accepted=True,selected_candidate=selected)
        row.update(seconds=perf_counter()-tick,peak_rss_bytes=base.peak_rss_bytes());history.append(row)
        np.savez_compressed(output/'current_known.npz',phi=phi,psi=p,hpsi=hp)
        save_resumable_checkpoint(output/'checkpoint.pkl',bank,optimizer_state={
            'reference':reference,'known':{'phi':phi,'psi':p,'hpsi':hp}},
            completed_steps=attempt+1,next_sample_index=0,run_identity=identity,history=history,
            metadata={'fixed_three_edge_budget':True,'old_audits_used_to_fit':False,'stationarity_certified':False})
        print('transport edge',pair,'stable columns',transport['retained_real_columns'],'accepted',row['accepted'],
              'actual candidates',len(row['actual_candidates']),'sec',round(row['seconds'],2),flush=True)
        jax.clear_caches();base.resources(spec,start)
    save_parameter_artifact(output/'endpoint.json',bank,identity=identity)
    result=dict(registration_sha256=reg['registration_sha256'],history=history,attempts=len(history),
        accepted=sum(v['accepted'] for v in history),endpoint_sha256=pytree_sha256(bank),
        seconds=perf_counter()-start,peak_rss_bytes=base.peak_rss_bytes(),stationarity_certified=False,
        no_post_audit_refit=True,gpu_hours=0,large_ED_calls=0,route2_implemented=False,route3_implemented=False)
    atomic_json(output/'training.json',result)
    if not result['accepted']:
        atomic_json(output/'results.json',dict(result,route_decision='no_new_endpoint_no_audit_no_more_same_directions',
            fresh_audit_performed=False))


def audit(spec: dict, reg: dict, output: Path) -> None:
    """Only a changed, actual-gate-safe endpoint gets the registered fresh audit."""
    training=base.configuration(output/'training.json')
    if not training['accepted']:raise ValueError('no candidate survived; do not add audit records')
    start=perf_counter();cfg,reference,parent,carrier,bonds,projector,coef,units=setup(spec)
    endpoint=load_parameter_artifact(output/'endpoint.json',parent)
    names=('carrier','previous4','transport4');seals=dict(parent=file_sha256(base.ROOT/spec['parent_source']/'chi4/endpoint.npz'),
                                                         endpoint=file_sha256(output/'endpoint.npz'))
    atomic_json(output/'audit_registration.json',dict(registration_sha256=reg['registration_sha256'],
        endpoint_seals=seals,seeds=spec['audit_seeds'],no_post_audit_refit=True))
    _,plan,evaluate=base.bank_functions(cfg,projector,bonds,spec)
    h2_spec=base.configuration(base.ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_population.json')
    h2_spec['wall_limit_seconds_per_process']=spec['wall_limit_seconds_per_process'];records=[]
    for i,seed in enumerate(spec['audit_seeds']):
        sample,info=base.draw({**spec['audit'],'uniform_floor_relative_to_pilot_mean':.2,'bridge_fraction':.5},
                             36,carrier,bonds,units['cscale'],seed,spec['audit']['records'])
        powers,work=base.carrier_h2(h2_spec,sample['states'],carrier,bonds,units['cscale'],start)
        p,hp=polynomial_carrier_products(powers,coef);r,hr,rw=base.products(reference,sample['states'],plan,evaluate)
        amps=[p];actions=[hp];queries={'reference':rw}
        for name,bank in (('previous4',parent),('transport4',endpoint)):
            a,ha,w=base.products(bank,sample['states'],plan,evaluate)
            amps.append(p+units['residual_scale']*(a-r));actions.append(hp+units['residual_scale']*(ha-hr));queries[name]=w
        sample.update(psi=np.column_stack(amps),hpsi=np.column_stack(actions),base_products=powers)
        np.savez_compressed(output/f'audit{i}.npz',**sample)
        records.append(dict(seed=seed,sampling=info,carrier_H2_work=work,residual_work=queries));base.resources(spec,start)
        print('transport fresh stream',i,'complete',flush=True)
    data={k:np.concatenate([base.loaded(output/f'audit{i}.npz')[k] for i in range(2)]) for k in ('psi','hpsi','q','groups')}
    data['groups'][128:]+=32
    report=paired_chain_audit(**data,n_sites=36,bootstrap_replicates=spec['audit']['bootstrap_replicates'],
        seed=spec['bootstrap_seed'],arm_names=names)
    gates={}
    for name in ('carrier','previous4'):
        d=report['paired'][f'transport4-minus-{name}'];e,v=d['energy_per_site'],d['variance_per_site']
        gates[name]=dict(resolved_energy_gain=e['value']<-2*e['chain_jackknife_se'],variance_point_nonworse=v['value']<=0)
    passed=all(all(g.values()) for g in gates.values())
    result=dict(training=training,registration_sha256=reg['registration_sha256'],endpoint_seals=seals,names=list(names),
        streams=records,combined_audit=report,gates=gates,fresh_audit_performed=True,
        route_decision='bounded_transport_gain_not_neural_or_literature_win' if passed else 'no_resolved_transport_gain_keep_C2',
        no_post_audit_refit=True,audit_seconds=perf_counter()-start,peak_rss_bytes=base.peak_rss_bytes(),
        gpu_hours=0,large_ED_calls=0,route2_implemented=False,route3_implemented=False)
    result['result_sha256']=canonical_sha256(result);atomic_json(output/'results.json',result)
    print('transport fresh gates',gates,flush=True)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train','audit'))
    args=parser.parse_args();spec=base.configuration(CONFIG);output=base.ROOT/spec['output']
    if args.stage=='prepare':prepare(spec,output);return
    reg=base.configuration(output/'registration.json')
    if reg['config']!=spec:raise ValueError('sealed configuration changed')
    for path,digest in {**reg['sources'],**reg['inputs']}.items():
        if file_sha256(base.ROOT/path)!=digest:raise ValueError('sealed source/input changed: '+path)
    (train if args.stage=='train' else audit)(spec,reg,output)


if __name__=='__main__':main()
