"""Route-1 correction: physical tree truncation and joint actual-step choice.

Reuse the sealed capacity trial's parent, quadrature, features, directions and
full-H executor. No old source/result is edited. Only reduced-copy widths2/4
are tested; no new neural architecture, ED, GPU job or routes2/3.
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
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact,load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import reduced_entry_metadata
from su2_nqs.models.schur_two_site import (
    embed_reduced_bank,internal_edges,two_site_layout,join_reduced_pair,
    split_reduced_pair,make_two_site_features,
)
from su2_nqs.models.schur_metric_retraction import split_physical_reduced_pair
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products
from su2_nqs.vmc.carrier_subspace import fit_carrier_subspace
from su2_nqs.vmc.cutset_training import paired_chain_audit,weighted_state_distance
from su2_nqs.vmc.joint_linear_step import joint_linear_coefficients,joint_step_gate
from su2_nqs.vmc.projected_spin_linear import make_projected_linear_products
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint,load_resumable_checkpoint
from . import spin_schur_capacity as base

CONFIG=base.ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_metric_capacity.json'


def registration(spec: dict, config_path: Path) -> dict:
    """Seal new numerical sources/config and old known inputs, not old audits."""
    if spec['max_actual_candidates']!=2 or spec['step_fractions']!=[1,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625]:
        raise ValueError('fixed two-candidate and finite-step budget required')
    reg=base.registration(spec)
    old=base.ROOT/spec['known_source']
    paths=[config_path,old/'registration.json',old/'known.npz',old/'prepared.json']
    for width in spec['widths']:
        paths.extend(old/f'chi{width}'/p for p in ('endpoint.json','endpoint.npz','results.json','checkpoint.pkl'))
    reg['inputs'].update({str(p.relative_to(base.ROOT)):file_sha256(p) for p in paths})
    reg['sources']=transitive_local_source_manifest((Path(__file__),),root=base.ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    reg.pop('registration_sha256')
    reg['registration_sha256']=canonical_sha256(reg)
    return reg


def prepare(spec: dict, reg: dict, output: Path) -> None:
    """Reuse sealed known products; verify physical-metric zero-step replay."""
    small,initial,_,bonds,projector,_,units=base.parent_setup(spec)
    old=base.ROOT/spec['known_source'];known=base.loaded(old/'known.npz')
    np.savez_compressed(output/'known.npz',**known)
    prep=base.configuration(old/'prepared.json')
    replay={}
    for width in spec['widths']:
        cfg=replace(small,tree=replace(small.tree,feature_dim=width))
        bank=embed_reduced_bank(initial,small,cfg)
        layout=two_site_layout(cfg,*internal_edges(cfg)[0])
        candidate,info=split_physical_reduced_pair(join_reduced_pair(bank,layout),bank,layout,
                                                  metric_cutoff=spec['retraction_metric_cutoff'])
        projected,_,_=base.bank_functions(cfg,projector,bonds,spec)
        phi=base.batched(lambda z:projected(candidate,z),known['states'],4)
        error=float(np.max(abs(units['residual_scale']*(phi-known['reference']))))
        if error>2e-8:raise FloatingPointError('zero-step physical retraction changes projected parent')
        replay[str(width)]=dict(projected_physical_replay_error=error,truncation=info)
    save_parameter_artifact(output/'reference.json',initial,
                            identity={'registration_sha256':reg['registration_sha256']})
    atomic_json(output/'prepared.json',dict(prep,registration_sha256=reg['registration_sha256'],
        known_source_sha256=file_sha256(old/'known.npz'),new_training_samples=0,
        reference_H_recomputed=False,known_product_sha256=file_sha256(output/'known.npz'),zero_step_replay=replay))
    print('prepared: reused192 known records and32 full-H products; zero-step replay',
          {k:v['projected_physical_replay_error'] for k,v in replay.items()},flush=True)


def quality(before: list[dict], after: list[dict], carrier: list[dict], spec: dict) -> dict:
    """Prospective stricter joint controls, never a relaxation of old gates."""
    return joint_step_gate(before,after,carrier,energy_tolerance=spec['control_energy_tolerance'],
                           variance_ratio=spec['maximum_variance_ratio'])


def block_moments(p: np.ndarray, hp: np.ndarray, q: np.ndarray, nf: int) -> list[dict]:
    return [base.moment(p[m],hp[m],q[m]) for m in (slice(0,nf),slice(nf,None))]


def train(spec: dict, reg: dict, output: Path, width: int) -> None:
    """Fixed14 local attempts, at most TWO actual candidate full-H evaluations."""
    start=perf_counter();small,initial,_,bonds,projector,_,units=base.parent_setup(spec)
    cfg=replace(small,tree=replace(small.tree,feature_dim=width))
    reference=embed_reduced_bank(initial,small,cfg);bank=reference
    folder=output/f'chi{width}';folder.mkdir(exist_ok=True)
    identity=dict(registration_sha256=reg['registration_sha256'],width=width)
    known=base.loaded(output/'known.npz');x,q=known['states'],known['q']
    hi,nf=known['H_indices'],spec['H_centers_per_block'];scale=units['residual_scale']
    projected,plan,evaluate=base.bank_functions(cfg,projector,bonds,spec)
    history=[];completed=0
    if (folder/'checkpoint.pkl').exists():
        saved=load_resumable_checkpoint(folder/'checkpoint.pkl',expected_run_identity=identity)
        bank,history,completed=saved['params'],saved['history'],saved['completed_steps']
    phi=base.batched(lambda z:projected(bank,z),x,4)
    _,hphi,initial_work=base.products(bank,x[hi],plan,evaluate)
    c,hc=known['carrier'],known['hcarrier']
    p=c+scale*(phi-known['reference']);hp=hc[hi]+scale*(hphi-known['reference_H'])
    carrier_m=block_moments(c[hi],hc[hi],q[hi],nf)
    edges=internal_edges(cfg);schedule=list(edges)+list(reversed(edges))
    for attempt,pair in enumerate(schedule):
        if attempt<completed:continue
        tick=perf_counter();layout=two_site_layout(cfg,*pair);joined=join_reduced_pair(bank,layout)
        linear_plan,linear_action,feature_projector=make_projected_linear_products(
            make_two_site_features(layout),projector,bonds,chunk_size=spec['linear_H_chunk'])
        feature_projector,linear_plan,linear_action=map(jax.jit,(feature_projector,linear_plan,linear_action))
        f=base.batched(lambda z:feature_projector(bank,z),x,2)
        replay=float(np.max(abs(scale*(f@joined-phi))))
        if replay>2e-8:raise FloatingPointError('joined feature replay failed')
        directions,metric=base.search_directions(f[hi],p[hi],hp,q[hi],scale,spec)
        before=block_moments(p[hi],hp,q[hi],nf)
        row=dict(attempt=attempt,pair=list(pair),feature_replay_error=replay,metric=metric,
                 accepted=False,before=before,actual_candidates=[],probe_count=0)
        if directions.shape[1]:
            values,actions,work=[],[],[]
            for i in range(0,len(hi),spec['H_microbatch']):
                prepared=linear_plan(jnp.asarray(x[hi[i:i+spec['H_microbatch']]]))
                h,a,count,padded=linear_action(bank,jnp.asarray(directions),prepared)
                values.append(np.asarray(a));actions.append(np.asarray(h))
                work.append(dict(unique=int(count),padded=int(padded)))
                base.resources(spec,start)
            t,ht=scale*np.concatenate(values),scale*np.concatenate(actions)
            blocks=[np.arange(len(hi))<nf,np.arange(len(hi))>=nf]
            common,common_info=joint_linear_coefficients(p[hi],hp,t,ht,q[hi],blocks,
                n_sites=36,damping=spec['metric_damping'])
            fit=fit_carrier_subspace(np.c_[p[hi[:nf]],t[:nf]],np.c_[hp[:nf],ht[:nf]],q[hi[:nf]],
                                    n_sites=36,relative_rank_tolerance=1e-8,maximum_condition_number=1e8)
            beta=fit.coefficients
            rays=[]
            if common_info['common_descent']:
                rays.append(('joint',common*spec['maximum_step_fs']))
            if abs(beta[0])>.1:
                rays.append(('energy_ritz',beta[1:]/beta[0]))
            row.update(joint_geometry=common_info,ritz=fit.as_dict(),
                linear_H_centers=len(hi),linear_H_columns=directions.shape[1],
                linear_raw_queries=sum(w['unique'] for w in work),
                full_H_equivalent_columns=len(hi)*directions.shape[1])
            options=[]
            for ray,coefficient in rays:
                for alpha in spec['step_fractions']:
                    step=alpha*coefficient;desired=joined+directions@step
                    uncompressed=p+scale*f@(directions@step)
                    uncompressed_H=hp+ht@step
                    prediction=block_moments(uncompressed[hi],uncompressed_H,q[hi],nf)
                    predicted_gate=quality(before,prediction,carrier_m,spec)
                    row['probe_count']+=1
                    if not predicted_gate['accepted']:continue
                    coordinate,_=split_reduced_pair(desired,bank,layout)
                    candidate,trunc=split_physical_reduced_pair(desired,bank,layout,
                        metric_cutoff=spec['retraction_metric_cutoff'],
                        comparison_entries=join_reduced_pair(coordinate,layout))
                    cz=join_reduced_pair(candidate,layout);candidate_phi=f@cz
                    candidate_p=c+scale*(candidate_phi-known['reference'])
                    distances=[weighted_state_distance(p[m],candidate_p[m],q[m]) for m in (slice(0,64),slice(64,None))]
                    parent_dist=[weighted_state_distance(c[m],candidate_p[m],q[m]) for m in (slice(0,64),slice(64,None))]
                    trunc_distance=weighted_state_distance(uncompressed,candidate_p,q)
                    step_fs=max(v['fs'] for v in distances)
                    if (step_fs>spec['maximum_step_fs'] or max(v['fs'] for v in parent_dist)>spec['maximum_parent_fs']
                            or min(v['ess_fraction'] for v in distances)<spec['minimum_ess_fraction']
                            or trunc_distance['fs']>spec['maximum_truncation_fs_fraction']*max(step_fs,1e-6)):
                        continue
                    # Ranking only uses the UNCOMPRESSED known-H prediction;
                    # it is explicitly not H of the retracted state. Verify it
                    # below using the original complete projected H kernel.
                    gain=sum(b['energy_per_site']-a['energy_per_site'] for a,b in zip(prediction,before))/2
                    options.append((gain,ray,alpha,candidate,candidate_phi,candidate_p,trunc,
                                    distances,parent_dist,trunc_distance,prediction))
            options.sort(key=lambda v:(-v[0],v[1],-v[2]))
            # Same ray gets smaller alternatives only after its best option;
            # actual TWO-candidate cap is fixed regardless of gate failures.
            accepted=[]
            for option in options[:spec['max_actual_candidates']]:
                _,ray,alpha,candidate,candidate_phi,candidate_p,trunc,distances,parent_dist,td,prediction=option
                cp,ch,cwork=base.products(candidate,x[hi],plan,evaluate)
                candidate_hp=hc[hi]+scale*(ch-known['reference_H'])
                actual_replay=float(np.max(abs(scale*(cp-candidate_phi[hi]))))
                if actual_replay>2e-8:raise FloatingPointError('retracted full-H amplitude replay failed')
                after=block_moments(candidate_p[hi],candidate_hp,q[hi],nf)
                gate=quality(before,after,carrier_m,spec)
                record=dict(ray=ray,alpha=alpha,truncation=trunc,distances=distances,parent_distances=parent_dist,
                    physical_truncation_distance=td,predicted_uncompressed_moments=prediction,
                    after=after,gate=gate,candidate_work=cwork,retraction_H_amplitude_error=actual_replay)
                row['actual_candidates'].append(record)
                if gate['accepted']:
                    gain=sum(b['energy_per_site']-a['energy_per_site'] for a,b in zip(after,before))/2
                    accepted.append((gain,candidate,candidate_phi,candidate_p,candidate_hp,len(row['actual_candidates'])-1))
            if accepted:
                _,bank,phi,p,hp,selected=max(accepted,key=lambda v:v[0])
                row.update(accepted=True,selected_candidate=selected)
            row['eligible_amplitude_options']=len(options)
        row.update(seconds=perf_counter()-tick,peak_rss_bytes=base.peak_rss_bytes())
        history.append(row)
        save_resumable_checkpoint(folder/'checkpoint.pkl',bank,optimizer_state={'reference':reference},
            completed_steps=attempt+1,next_sample_index=0,run_identity=identity,history=history,
            metadata={'no_new_fit_samples':True,'stationarity_certified':False,'joint_actual_step_gate':True})
        print('chi',width,'attempt',attempt+1,'/14 accepted',row['accepted'],
              'actual H candidates',len(row['actual_candidates']),
              'FIT',base.moment(p[hi[:nf]],hp[:nf],q[hi[:nf]]),'sec',round(row['seconds'],2),flush=True)
        jax.clear_caches();base.resources(spec,start)
    save_parameter_artifact(folder/'endpoint.json',bank,identity=identity)
    np.savez_compressed(folder/'endpoint_known.npz',psi=p,hpsi=hp,phi=phi)
    atomic_json(folder/'results.json',dict(registration_sha256=reg['registration_sha256'],width=width,
        full_free_parameters=reduced_entry_metadata(cfg)['free_real_parameters'],attempts=len(history),
        accepted=sum(v['accepted'] for v in history),history=history,initial_work=initial_work,
        seconds=perf_counter()-start,peak_rss_bytes=base.peak_rss_bytes(),endpoint_sha256=pytree_sha256(bank),
        stationarity_certified=False,no_post_audit_refit=True,endpoint_sealed_before_fresh_audit=True))


def audit(spec: dict, reg: dict, output: Path) -> None:
    """New records compare new/OLD frozen endpoints on one common quadrature."""
    small,initial,carrier,bonds,projector,coef,units=base.parent_setup(spec)
    old=base.ROOT/spec['known_source'];names=('carrier','old2','new2','old4','new4')
    seals={f'{kind}{w}':file_sha256(folder/f'chi{w}/endpoint.npz')
           for kind,folder in (('old',old),('new',output)) for w in (2,4)}
    atomic_json(output/'audit_registration.json',dict(registration_sha256=reg['registration_sha256'],
        endpoint_seals=seals,seeds=spec['audit_seeds'],no_post_audit_refit=True))
    records=[];start=perf_counter()
    h2_spec=base.configuration(base.ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_population.json')
    h2_spec.update(wall_limit_seconds_per_process=spec['wall_limit_seconds_per_process'])
    for index,seed in enumerate(spec['audit_seeds']):
        sample,info=base.draw({**spec['audit'],'uniform_floor_relative_to_pilot_mean':.2,'bridge_fraction':.5},
                             36,carrier,bonds,units['cscale'],seed,spec['audit']['records'])
        powers,pwork=base.carrier_h2(h2_spec,sample['states'],carrier,bonds,units['cscale'],start)
        p,hp=polynomial_carrier_products(powers,coef);amps=[p];actions=[hp];work={}
        for width in (2,4):
            cfg=replace(small,tree=replace(small.tree,feature_dim=width));reference=embed_reduced_bank(initial,small,cfg)
            _,plan,evaluate=base.bank_functions(cfg,projector,bonds,spec)
            r,hr,rw=base.products(reference,sample['states'],plan,evaluate)
            work[f'reference{width}']=rw
            for kind,folder in (('old',old),('new',output)):
                bank=load_parameter_artifact(folder/f'chi{width}/endpoint.json',reference)
                a,ha,w=base.products(bank,sample['states'],plan,evaluate)
                amps.append(p+units['residual_scale']*(a-r));actions.append(hp+units['residual_scale']*(ha-hr))
                work[f'{kind}{width}']=w;base.resources(spec,start)
        sample.update(psi=np.column_stack(amps),hpsi=np.column_stack(actions),base_products=powers)
        np.savez_compressed(output/f'audit{index}.npz',**sample)
        row=paired_chain_audit(sample['psi'],sample['hpsi'],sample['q'],sample['groups'],36,
                               spec['audit']['bootstrap_replicates'],spec['bootstrap_seed']+index,names)
        records.append(dict(seed=seed,sampling=info,carrier_H2_work=pwork,residual_work=work,audit=row))
        print('fresh audit',index,'sealed',flush=True);jax.clear_caches()
    combined={k:np.concatenate([base.loaded(output/f'audit{i}.npz')[k] for i in range(2)]) for k in ('psi','hpsi','q','groups')}
    combined['groups'][128:]+=32
    row=paired_chain_audit(**combined,n_sites=36,bootstrap_replicates=spec['audit']['bootstrap_replicates'],
                           seed=spec['bootstrap_seed']+2,arm_names=names)
    gates={}
    for width in (2,4):
        d=row['paired'][f'new{width}-minus-carrier']['energy_per_site']
        r=row['paired'][f'new{width}-minus-carrier']['variance_per_site']
        gates[str(width)]=dict(resolved_energy_gain=d['value']<-2*d['chain_jackknife_se'],
                               variance_point_nonworse=r['value']<=0)
    passed=any(all(v.values()) for v in gates.values())
    final=dict(registration_sha256=reg['registration_sha256'],endpoint_seals=seals,names=list(names),
        streams=records,combined_audit=row,carrier_relative_gates=gates,
        resolved_carrier_relative_gain=passed,stationarity_certified=False,capacity_exhaustion_proved=False,
        route2_implemented=False,route3_implemented=False,no_post_audit_refit=True,
        new_H2_centers=256,audit_seconds=perf_counter()-start,peak_rss_bytes=base.peak_rss_bytes(),
        gpu_hours=0,large_ED_calls=0,route_decision='route1_positive_bounded_signal' if passed else 'route1_no_resolved_gain_do_not_extend_blindly')
    final['result_sha256']=canonical_sha256(final);atomic_json(output/'results.json',final)
    print('carrier-relative gates',gates,flush=True)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train','audit'))
    parser.add_argument('--width',type=int,choices=(2,4));parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();spec=base.configuration(args.config);output=base.ROOT/spec['output']
    if args.stage=='prepare':
        output.mkdir(exist_ok=False);reg=registration(spec,args.config);atomic_json(output/'registration.json',reg);prepare(spec,reg,output)
    else:
        reg=base.configuration(output/'registration.json')
        for path,digest in {**reg['inputs'],**reg['sources']}.items():
            if file_sha256(base.ROOT/path)!=digest:raise ValueError('registered source/input changed: '+path)
        if reg['config']!=spec:raise ValueError('registered configuration changed')
        if args.stage=='train':
            if args.width is None:parser.error('--width required')
            train(spec,reg,output,args.width)
        else:audit(spec,reg,output)


if __name__=='__main__':main()
