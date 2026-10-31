"""Registered same-checkpoint, same-backbone bounded optimizer comparison.

Reuses the full-ViT pilot's model, Hamiltonian, sampler, Adam and audit
statistics. This driver owns only arm orchestration/provenance; the generic
sample-space SR implementation lives in su2_nqs.vmc. No ED or remote launch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.vmc.dual_sampled_sr import make_dual_sampled_sr_update, sampled_overlap_diagnostics
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig,fixed_sz_spatial_vit_parameter_count
from su2_nqs.benchmarks.spatial_vit_sampling import projected_log_function,make_log_exchange_sampler,make_full_heisenberg_ratios
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256,array_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest,numerical_runtime_identity
from .spin import _groups,_bonds
from .spin_full_vit_pilot import ROOT,_file_sha256,_sealed_json,make_sampled_adam_update,audit_statistics

CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_full_vit_optimization.json'
OUTPUT=ROOT/'projects/clebsch_tree_nqs/results/spin_full_vit_optimization_v1'


def seal(path: Path, data: dict) -> dict:
    """Write a newly computed record with its canonical content seal."""
    data=dict(data)
    data['result_sha256']=canonical_sha256(data)
    atomic_json(path,data)
    return data


def load_parent(spec: dict, geometry: str) -> tuple[dict,dict]:
    """Fail closed if the frozen 1064-step model/optimizer/source bytes change."""
    folder=ROOT/spec['parent_directory']/geometry
    r=_sealed_json(folder/'results.json')
    expected=spec['parents'][geometry]
    if r['result_sha256']!=expected['result_sha256'] or _file_sha256(folder/'checkpoint.pkl')!=expected['checkpoint_sha256']:
        raise ValueError('parent artifacts differ from registration')
    p=load_checkpoint(folder/'checkpoint.pkl')
    if (p['metadata']['step']!=spec['parent_step'] or int(p['optimizer_state'][0])!=spec['parent_step'] or
        pytree_sha256(p['params']['model'])!=r['final_params_sha256'] or r['parameter_count']!=4032):
        raise ValueError('parent model/step/optimizer mismatch')
    return p,r


def build_update(arm: dict, log_fn, params):
    """Common update interface; SR does not secretly retain/reset Adam moments."""
    log_batch=jax.vmap(log_fn,in_axes=(None,0))
    if arm['optimizer']=='adam':
        adam=make_sampled_adam_update(log_batch,arm)
        @jax.jit
        def update(p,opt,x,e):
            changed,opt,norm=adam(p,opt,x,e)
            diag=sampled_overlap_diagnostics(log_batch(changed,x)-log_batch(p,x))
            diag.update(gradient_norm=norm,accepted=jnp.asarray(True))
            return changed,opt,diag
        return update
    if arm['optimizer']!='ridge_sr':
        raise ValueError('unknown registered optimizer')
    kernel=make_dual_sampled_sr_update(log_fn,params,**{k:v for k,v in arm.items() if k not in ('id','optimizer')})
    @jax.jit
    def update(p,opt,x,e):
        changed,diag=kernel(p,x,e)
        return changed,dict(step=opt['step']+1),diag
    return update


def compare_audits(left: np.ndarray, right: np.ndarray, *, n_sites: int,
                   seed: int, replicates: int) -> dict:
    """Unpaired chain-bootstrap left-minus-right, with fixed primary-family CI."""
    rng=np.random.default_rng(seed)
    boots=[]; points=[]
    for values in (left,right):
        paths=values.transpose(0,2,1).reshape(-1,values.shape[1])
        means=paths.mean(axis=1); second=np.mean(abs(paths)**2,axis=1)
        indices=rng.integers(len(means),size=(replicates,len(means)))
        m=means[indices].mean(axis=1)
        boots.append((m.real/n_sites,(second[indices].mean(axis=1)-abs(m)**2)/n_sites))
        points.append((means.mean().real/n_sites,np.mean(abs(paths-means.mean())**2)/n_sites))
    result={}
    for i,key in enumerate(('energy_per_site','variance_per_site')):
        delta=boots[0][i]-boots[1][i]
        result[key]=dict(difference=float(points[0][i]-points[1][i]),
            bootstrap_se=float(delta.std(ddof=1)),chain_bootstrap_95=np.quantile(delta,[.025,.975]).tolist(),
            primary_family_bonferroni_98_75=np.quantile(delta,[.00625,.99375]).tolist())
    return result


def prepare_initial_state(path: Path, params: object, log_fn, *, seed: int,
                          chains: int, n_sites: int, burn: int,
                          registration_sha256: str) -> dict:
    """Create or reload one common, thermally prepared fixed-M2=0 fork."""
    if path.exists():
        base=load_checkpoint(path)
        if base['metadata']['registration_sha256']!=registration_sha256:
            raise ValueError('initial chains belong to another protocol')
    else:
        init,key=jax.random.split(jax.random.PRNGKey(seed))
        x=initialize_spin_exchange_chains(init,n_chains=chains,n_sites=n_sites,n_up=n_sites//2)
        thermal=make_log_exchange_sampler(log_fn,burn)
        x,key,_,_=thermal(params,key,x)
        jax.block_until_ready(x)
        save_checkpoint(path,dict(model=params,chains=x,key=key),
            metadata=dict(registration_sha256=registration_sha256))
        base=load_checkpoint(path)
    bits=np.asarray(base['params']['chains'])
    if (pytree_sha256(base['params']['model'])!=pytree_sha256(params) or
        bits.shape!=(chains,n_sites) or not np.isin(bits,(0,1)).all() or
        not np.all(bits.sum(axis=1)==n_sites//2)):
        raise ValueError('initial model/chain sector changed')
    return base


def frozen_audit(params: object, label: str, geometry: str, destination: Path,
                 spec: dict, registration: dict, log_fn, local) -> tuple[dict,np.ndarray]:
    """Persist fresh audit streams; never return a training or selection signal."""
    a=spec['audit']
    thermal=make_log_exchange_sampler(log_fn,a['burn_proposals'])
    sampler=make_log_exchange_sampler(log_fn,a['proposals_between_snapshots'])
    values_list=[]; records=[]
    for seed in a['seeds'][geometry][label]:
        path=destination/f'audit_{label}_{seed}.json'
        raw_path=path.with_suffix('.npz')
        if path.exists():
            record=_sealed_json(path)
            if (record['registration_sha256']!=registration['result_sha256'] or
                record['params_sha256']!=pytree_sha256(params) or record['raw_sha256']!=_file_sha256(raw_path)):
                raise ValueError('audit resume identity changed')
            with np.load(raw_path) as raw:
                values=raw['local_energies']
        else:
            started=perf_counter()
            init,rng=jax.random.split(jax.random.PRNGKey(seed))
            x=initialize_spin_exchange_chains(init,n_chains=a['chains'],n_sites=36,n_up=18)
            x,rng,_,_=thermal(params,rng,x)
            samples=[]; values=[]; moves=[]
            for _ in range(a['snapshots']):
                x,rng,_,moved=sampler(params,rng,x)
                values.append(np.asarray(local(params,x))); samples.append(np.asarray(x)); moves.append(float(moved))
            values=np.asarray(values)
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite independent audit; no re-tuning allowed')
            np.savez_compressed(raw_path,states=np.asarray(samples),local_energies=values,
                final_key=np.asarray(rng),nontrivial_acceptance=np.asarray(moves))
            record=seal(path,dict(seed=seed,label=label,params_sha256=pytree_sha256(params),
                registration_sha256=registration['result_sha256'],raw_sha256=_file_sha256(raw_path),
                samples=int(values.size),seconds=perf_counter()-started,
                mean_nontrivial_acceptance=float(np.mean(moves)),
                unique_configurations=int(len(np.unique(np.asarray(samples).reshape(-1,36),axis=0)))))
        values_list.append(values); records.append(record)
        print(geometry,label,'audit',seed,'E/N',float(values.mean().real/36),flush=True)
    values=np.asarray(values_list)
    summary=audit_statistics(values,36,bootstrap_seed=a['bootstrap_seed'],replicates=a['bootstrap_replicates'])
    summary['streams']=records
    return summary,values


def run(geometry: str, output: Path=OUTPUT, *, resume: bool=False) -> dict:
    """Run every registered arm, then frozen audits; no online hyperparameter choice."""
    started=perf_counter()
    spec=json.loads(CONFIG.read_text())
    seeds=[v for pools in spec['audit']['seeds'].values() for stream in pools.values() for v in stream]
    if len(seeds)!=len(set(seeds)) or set(seeds)&set(spec['training_seeds'].values()):
        raise ValueError('audit and training seeds must be disjoint')
    for g in spec['parents']:
        load_parent(spec,g)
    parent,reference=load_parent(spec,geometry)
    output.mkdir(parents=True,exist_ok=True)
    reg_data=dict(config=spec,config_sha256=canonical_sha256(spec),
        source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
        runtime=numerical_runtime_identity())
    reg_path=output/'registration.json'
    if reg_path.exists():
        registration=_sealed_json(reg_path)
        if any(registration[k]!=v for k,v in reg_data.items()):
            raise ValueError('registered protocol/source/runtime changed')
    else:
        registration=seal(reg_path,{**reg_data,'registered_at_utc':datetime.now(timezone.utc).isoformat()})
    destination=output/geometry
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'results.json').exists():
        raise FileExistsError('completed optimizer comparison is immutable')
    initial_path=destination/'initial_state.pkl'
    if initial_path.exists() and not resume:
        raise FileExistsError('use --resume for an interrupted comparison')
    config=FixedSzSpatialViTConfig(**reference['model_config'])
    groups=_groups(geometry,6,6); bonds=_bonds(geometry,6,6,reference['j2_over_j1'])
    log_fn,images=projected_log_function(config,groups['gamma_a1'])
    log_batch=jax.jit(jax.vmap(log_fn,in_axes=(None,0)))
    local=make_full_heisenberg_ratios(log_fn,bonds)
    sample=make_log_exchange_sampler(log_fn,spec['proposals_per_update'])
    p0=parent['params']['model']
    preparation_started=perf_counter()
    base=prepare_initial_state(initial_path,p0,log_fn,seed=spec['training_seeds'][geometry],
        chains=spec['chains'],n_sites=36,burn=spec['initial_burn_proposals'],
        registration_sha256=registration['result_sha256'])
    # One shared full-H compilation batch. Warm candidates are discarded.
    work_path=destination/'execution.json'
    work=json.loads(work_path.read_text()) if work_path.exists() else dict(warmup_full_H_calls=0,compile_invocations=0)
    warm=local(p0,base['params']['chains']); jax.block_until_ready(warm)
    jax.block_until_ready(sample(p0,base['params']['key'],base['params']['chains']))
    work['warmup_full_H_calls']+=spec['chains'];work['compile_invocations']+=1
    atomic_json(work_path,work)
    preparation_seconds=perf_counter()-preparation_started
    trained={}; training={}
    print(geometry,'registered',registration['result_sha256'],'parent step',spec['parent_step'],flush=True)
    for arm in spec['arms']:
        label=arm['id']; folder=destination/label
        folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'checkpoint.pkl'; train_path=folder/'training.json'
        if train_path.exists():
            record=_sealed_json(train_path)
            if (record['checkpoint_sha256']!=_file_sha256(checkpoint) or
                record['registration_sha256']!=registration['result_sha256'] or
                record['initial_model_sha256']!=reference['final_params_sha256'] or record['arm']!=arm):
                raise ValueError('completed training seal mismatch')
            training[label]=record;trained[label]=load_checkpoint(checkpoint)['params']['model']
            continue
        if checkpoint.exists():
            saved=load_checkpoint(checkpoint)
            state,opt,meta=saved['params'],saved['optimizer_state'],saved['metadata']
            if meta['registration_sha256']!=registration['result_sha256'] or len(meta['history'])!=meta['completed_steps']:
                raise ValueError('optimizer checkpoint identity/count mismatch')
            expected_count=meta['completed_steps']+(spec['parent_step'] if arm['optimizer']=='adam' else 0)
            actual_count=int(opt[0] if arm['optimizer']=='adam' else opt['step'])
            if actual_count!=expected_count:
                raise ValueError('restored optimizer count mismatch')
        else:
            state=base['params']
            opt=parent['optimizer_state'] if arm['optimizer']=='adam' else dict(step=jnp.asarray(0))
            meta=dict(registration_sha256=registration['result_sha256'],completed_steps=0,history=[],
                active_seconds=0.,optimizer_compile_seconds=0.,status='running',
                initial_model_sha256=pytree_sha256(p0),initial_state_sha256=pytree_sha256(state),
                initial_optimizer_sha256=pytree_sha256(opt))
        update=build_update(arm,log_fn,p0)
        tick=perf_counter()
        jax.block_until_ready(update(p0,opt,base['params']['chains'],warm))
        meta['optimizer_compile_seconds']+=perf_counter()-tick
        save_checkpoint(checkpoint,state,optimizer_state=opt,metadata=meta)
        for step in range(meta['completed_steps']+1,spec['steps']+1):
            if meta['active_seconds']>=spec['maximum_training_seconds_per_arm']:
                meta['status']='time_cap';break
            tick=perf_counter()
            x,key,_,moved=sample(state['model'],state['key'],state['chains'])
            energy=local(state['model'],x)
            candidate,next_opt,diag=update(state['model'],opt,x,energy)
            jax.block_until_ready((candidate,next_opt,diag,energy))
            if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves((candidate,next_opt,diag,energy))):
                raise FloatingPointError('nonfinite optimizer step; preserve last valid checkpoint')
            if float(diag.get('relative_linear_residual',0.))>1e-7:
                raise FloatingPointError('SR true residual failed; do not conceal by trust scaling')
            state=dict(model=candidate,chains=x,key=key);opt=next_opt
            elapsed=perf_counter()-tick;meta['active_seconds']+=elapsed;meta['completed_steps']=step
            values=np.asarray(energy)
            row=dict(step=step,energy_before_update_per_site=float(values.mean().real/36),
                variance_before_update_per_site=float(np.mean(abs(values-values.mean())**2)/36),
                nontrivial_acceptance=float(moved),seconds=elapsed,active_seconds=meta['active_seconds'],
                **{k:np.asarray(v).item() for k,v in diag.items()})
            if step==1:
                row['sample_batch_sha256']=array_sha256(np.asarray(x))
                row['local_energy_batch_sha256']=array_sha256(values)
            meta['history'].append(row)
            save_checkpoint(checkpoint,state,optimizer_state=opt,metadata=meta)
            if step==1 or step%16==0:
                print(geometry,label,row,flush=True)
        if meta['completed_steps']==spec['steps']:
            meta['status']='fixed_budget_complete'
        save_checkpoint(checkpoint,state,optimizer_state=opt,metadata=meta)
        frozen=load_checkpoint(checkpoint)
        replay=float(np.max(abs(np.asarray(log_batch(state['model'],state['chains']))-
            np.asarray(log_batch(frozen['params']['model'],state['chains'])))))
        ref_logs=log_batch(state['model'],state['chains']);error=0.
        for perm in groups['gamma_a1'][::max(1,len(groups['gamma_a1'])//8)]:
            changed=log_batch(state['model'],state['chains'][:,jnp.asarray(perm)])
            error=max(error,float(jnp.max(abs(jnp.exp(changed-ref_logs)-1))))
        training[label]=seal(train_path,dict(**meta,arm=arm,checkpoint_sha256=_file_sha256(checkpoint),
            final_params_sha256=pytree_sha256(state['model']),training_state_sha256=pytree_sha256((state,opt)),
            parameter_count=fixed_sz_spatial_vit_parameter_count(state['model']),
            full_H_configuration_calls=meta['completed_steps']*spec['chains'],
            checkpoint_log_amplitude_replay_max=replay,sampled_A1_relative_error=error))
        trained[label]=state['model']
    if (len({t['initial_state_sha256'] for t in training.values()})!=1 or
        len({t['history'][0]['sample_batch_sha256'] for t in training.values()})!=1 or
        len({t['history'][0]['local_energy_batch_sha256'] for t in training.values()})!=1):
        raise ValueError('optimizer arms did not share their initial physical batch')
    initial_norms=[t['history'][0]['gradient_norm'] for t in training.values()]
    # No audit information is available to any optimizer on this lattice.
    audits={};raw={}
    for label,params in dict(parent=p0,**trained).items():
        audits[label],raw[label]=frozen_audit(params,label,geometry,destination,spec,registration,log_fn,local)
    comparisons={}
    pairs=[('ridge_sr','adam'),('adam_low_lr','adam')]+[(a['id'],'parent') for a in spec['arms']]
    for i,(left,right) in enumerate(pairs):
        comparisons[f'{left}_minus_{right}']=compare_audits(raw[left],raw[right],n_sites=36,
            seed=spec['audit']['bootstrap_seed']+i+1,replicates=spec['audit']['bootstrap_replicates'])
    matched=all(r['completed_steps']==spec['steps'] for r in training.values())
    primary=comparisons['ridge_sr_minus_adam']
    result=dict(task=spec['task'],geometry=geometry,j2_over_j1=reference['j2_over_j1'],
        registration_sha256=registration['result_sha256'],parent_result_sha256=reference['result_sha256'],
        model_config=reference['model_config'],parameter_count=4032,training=training,audits=audits,
        comparisons=comparisons,matched_training_work=matched,
        paired_initialization_verified=True,first_training_batch_identical=True,
        initial_gradient_norm_relative_discrepancy=float(np.ptp(initial_norms)/max(max(initial_norms),1e-30)),
        primary_conditional_joint_improvement=bool(matched and all(
            v['primary_family_bonferroni_98_75'][1]<0 for v in primary.values())),
        full_H_configuration_calls=work['warmup_full_H_calls']+
            sum(v['full_H_configuration_calls'] for v in training.values())+sum(v['samples'] for v in audits.values()),
        warmup_full_H_calls=work['warmup_full_H_calls'],preparation_seconds_this_invocation=preparation_seconds,
        wall_seconds_this_invocation=perf_counter()-started,bond_count=len(bonds),amplitude_images=images,
        remote_submitted=False,production_released=False,architecture_win_claimed=False,
        spin_symmetry='fixed Sz=0; NOT exact S=0',spatial_symmetry='exact Gamma-A1',
        runtime=numerical_runtime_identity())
    for label,record in training.items():
        saved=load_checkpoint(destination/label/'checkpoint.pkl')
        if record['training_state_sha256']!=pytree_sha256((saved['params'],saved['optimizer_state'])):
            raise ValueError('audit altered training checkpoint')
    result=seal(destination/'results.json',result)
    print(geometry,'COMPLETE',primary,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',choices=('triangular','square'),required=True)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.geometry,args.output,resume=args.resume)
