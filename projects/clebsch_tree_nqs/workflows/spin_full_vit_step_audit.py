"""Single-step generalization audit at a frozen parent; no training trajectory.

Reuses the sealed ViT/Hamiltonian/optimizer and sampler implementations.
All candidate states are diagnostic snapshots, not adopted checkpoints.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from time import perf_counter
from collections.abc import Callable
import jax
import jax.numpy as jnp
import numpy as np
from .spin_full_vit_recheck import registration as prior_registration,components
from .spin_full_vit_optimization import load_parent,build_update,prepare_initial_state,seal
from .spin_full_vit_pilot import ROOT,_sealed_json,_file_sha256,audit_statistics
from su2_nqs.benchmarks.spatial_vit_sampling import make_log_exchange_sampler
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.tempered_spin_exchange import make_tempered_spin_exchange
from su2_nqs.vmc.correlated_audit import correlated_summary,paired_chain_intervals
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.frontier.artifacts import canonical_sha256,pytree_sha256,array_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest,numerical_runtime_identity

CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_full_vit_step_audit.json'
OUTPUT=ROOT/'projects/clebsch_tree_nqs/results/spin_full_vit_step_audit_v1'


def register(output: Path) -> tuple[dict,dict,dict,dict]:
    """Seal fixed candidates, independent seeds and prior inputs before sampling."""
    spec=json.loads(CONFIG.read_text());old_spec,prior,old=prior_registration(ROOT/spec['recheck_directory'])
    if old['result_sha256']!=spec['recheck_registration_sha256']:raise ValueError('prior protocol mismatch')
    reference_path=ROOT/spec['recheck_directory']/'square/reference_check.json';reference=_sealed_json(reference_path)
    if reference['registration_sha256']!=old['result_sha256']:raise ValueError('prior reference mismatch')
    seeds=spec['training_seeds']+[s['seed'] for s in spec['validation']['streams']]
    oldseeds=old_spec['triangular']['training_seeds']+old_spec['square']['seeds']+list(prior['training_seeds'].values())
    oldseeds += [s for pool in old_spec['triangular']['audit']['seeds'].values() for v in pool.values() for s in v]
    oldseeds += [s for pool in prior['audit']['seeds'].values() for v in pool.values() for s in v]
    if len(seeds)!=len(set(seeds)) or set(seeds)&set(oldseeds):raise ValueError('seed reuse')
    output.mkdir(parents=True,exist_ok=True)
    data=dict(config=spec,config_sha256=canonical_sha256(spec),reference_sha256=reference['result_sha256'],
        source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),runtime=numerical_runtime_identity())
    path=output/'registration.json'
    if path.exists():
        reg=_sealed_json(path)
        if any(reg[k]!=v for k,v in data.items()):raise ValueError('step audit registration changed')
    else:reg=seal(path,{**data,'registered_at_utc':datetime.now(timezone.utc).isoformat()})
    return spec,prior,reg,reference


def evaluate_chunks(fn: Callable,params: object,states: np.ndarray,chunk: int) -> np.ndarray:
    """Evaluate every configuration exactly once, without padding H calls."""
    return np.concatenate([np.asarray(fn(params,jnp.asarray(states[i:i+chunk]))) for i in range(0,len(states),chunk)])


def probe_neighborhood(centers: np.ndarray,bonds: list) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Deterministic unique M2-preserving one-bond neighborhoods, not Born draws."""
    centers=np.asarray(centers,dtype=np.int32)
    if centers.shape!=(24,36) or not np.isin(centers,[0,1]).all() or not np.all(centers.sum(axis=1)==18):
        raise ValueError('expected 24 fixed M2=0 reference configurations')
    lookup={};states=[];groups=[set() for _ in range(3)];central=[]
    def add(x: np.ndarray) -> int:
        key=tuple(x)
        if key not in lookup:lookup[key]=len(states);states.append(x.copy())
        return lookup[key]
    for k,x in enumerate(centers):
        index=add(x);central.append(index);groups[k//8].add(index)
        for i,j,c in bonds:
            if c and x[i]!=x[j]:
                y=x.copy();y[i],y[j]=x[j],x[i];groups[k//8].add(add(y))
    membership=np.zeros((3,len(states)),dtype=bool)
    for g,indices in enumerate(groups):membership[g,list(indices)]=True
    return np.asarray(states),np.asarray(central),membership


def probe_diagnostics(log_ratios: np.ndarray,central: np.ndarray,states: np.ndarray,
                      train_states: np.ndarray) -> dict:
    """Gauge-invariant relative-weight changes; never normalize into basin mass."""
    delta=2*np.asarray(log_ratios).real;means=delta[central].reshape(3,8).mean(axis=1)
    if not np.isfinite(log_ratios).all():raise ValueError('nonfinite diagnostic ratios')
    seen={tuple(x) for x in train_states};outside=np.array([tuple(x) not in seen for x in states])
    centered=delta-means[1];x=centered[outside]
    return dict(central_high_minus_low_log_weight_change=float(means[2]-means[1]),
        central_uniform_minus_low_log_weight_change=float(means[0]-means[1]),
        outside_training_count=int(outside.sum()),training_overlap_count=int((~outside).sum()),
        outside_training_centered_quantiles=np.quantile(x,[0,.01,.5,.99,1]).tolist(),
        outside_training_max_abs_centered_log_weight_change=float(np.max(abs(x))))


def parent_validation(params: object,log_fn: Callable,local: Callable,spec: dict,
                      reg: dict,output: Path) -> tuple[np.ndarray,np.ndarray,list]:
    """Two fresh parent-target streams, snapshot-atomic resume; no candidate feedback."""
    cfg=spec['validation'];R=len(cfg['betas']);C=cfg['chains'];records=[];xs=[];es=[]
    for stream in cfg['streams']:
        kernel,seed=stream['kernel'],stream['seed'];folder=output/f'validation_{kernel}_{seed}';folder.mkdir(exist_ok=True)
        record_path=folder/'pool.json';raw_path=folder/'pool.npz';cp=folder/'chains.pkl'
        if record_path.exists():
            record=_sealed_json(record_path)
            if record['registration_sha256']!=reg['result_sha256'] or record['raw_sha256']!=_file_sha256(raw_path):raise ValueError('validation changed')
            with np.load(raw_path) as data:x,e=data['states'],data['parent_energies']
        else:
            started=perf_counter()
            if cp.exists():
                saved=load_checkpoint(cp);state,meta=saved['params'],saved['metadata']
                if meta['registration_sha256']!=reg['result_sha256'] or meta['params_sha256']!=pytree_sha256(params):raise ValueError('validation checkpoint changed')
            else:
                init,key=jax.random.split(jax.random.PRNGKey(seed));n=C if kernel=='ordinary' else R*C
                x=initialize_spin_exchange_chains(init,n_chains=n,n_sites=36,n_up=18)
                if kernel=='ordinary':
                    x,key,_,_=make_log_exchange_sampler(log_fn,cfg['ordinary_burn_proposals'])(params,key,x)
                    tags=jnp.zeros((1,C),dtype=jnp.int32)
                else:
                    tags=jnp.broadcast_to(jnp.arange(R)[:,None],(R,C));x=x.reshape(R,C,36)
                    x,key,tags,_=make_tempered_spin_exchange(log_fn,cfg['betas'],
                        proposals=cfg['ordinary_burn_proposals']//R,swap_every=cfg['swap_every'])(params,key,x,tags)
                jax.block_until_ready(x);state=dict(states=x,key=key,lineages=tags)
                meta=dict(registration_sha256=reg['result_sha256'],params_sha256=pytree_sha256(params),states=[],energies=[],history=[],seconds=perf_counter()-started)
                save_checkpoint(cp,state,metadata=meta)
            if len(meta['states'])!=len(meta['energies']) or len(meta['history'])!=len(meta['states']):raise ValueError('snapshot count mismatch')
            sampler=(make_log_exchange_sampler(log_fn,cfg['ordinary_thin_proposals']) if kernel=='ordinary' else
                make_tempered_spin_exchange(log_fn,cfg['betas'],proposals=cfg['ordinary_thin_proposals']//R,swap_every=cfg['swap_every']))
            for step in range(len(meta['states']),cfg['snapshots']):
                tick=perf_counter()
                if kernel=='ordinary':
                    x,key,_,move=sampler(params,state['key'],state['states']);cold=x;tags=state['lineages'];diag=dict(move=float(move))
                else:
                    x,key,tags,d=sampler(params,state['key'],state['states'],state['lineages']);cold=x[-1]
                    diag={k:np.asarray(v).tolist() for k,v in d.items()}
                e=np.asarray(local(params,cold))
                if not np.isfinite(e).all():raise FloatingPointError('parent audit nonfinite')
                state=dict(states=x,key=key,lineages=tags);meta['states'].append(np.asarray(cold));meta['energies'].append(e);meta['history'].append(diag)
                meta['seconds']+=perf_counter()-tick
                save_checkpoint(cp,state,metadata=meta)
            save_checkpoint(cp,state,metadata=meta)
            x,e=np.asarray(meta['states']),np.asarray(meta['energies']);np.savez_compressed(raw_path,states=x,parent_energies=e)
            record=seal(record_path,dict(registration_sha256=reg['result_sha256'],raw_sha256=_file_sha256(raw_path),
                checkpoint_sha256=_file_sha256(cp),params_sha256=pytree_sha256(params),stream=stream,seconds=meta['seconds'],full_H_calls=e.size,history=meta['history']))
        print('PARENT VALIDATION',kernel,seed,'E/N',e.real.mean()/36,flush=True)
        xs.append(x);es.append(e);records.append(record)
    return np.asarray(xs),np.asarray(es),records


def run(output: Path=OUTPUT) -> dict:
    """Run the complete fixed candidate list without adopting or tuning a state."""
    started=perf_counter();spec,prior,reg,reference=register(output)
    if (output/'results.json').exists():raise FileExistsError('completed audit is immutable; use a fresh --output')
    parent,ref=load_parent(prior,'square');p0=parent['params']['model'];p0hash=pytree_sha256(p0)
    logs,local,group,bonds=components(ref);batch_logs=jax.jit(jax.vmap(logs,in_axes=(None,0)));chunk=spec['evaluation_chunk_size']
    val_x,val_e,pools=parent_validation(p0,logs,local,spec,reg,output)
    flat_val=val_x.reshape(-1,36);val_logs=evaluate_chunks(batch_logs,p0,flat_val,chunk).reshape(val_e.shape)
    probes,central,membership=probe_neighborhood(np.asarray(reference['states']),bonds)
    old_probe_logs=evaluate_chunks(batch_logs,p0,probes,chunk)
    probe_path=output/'probes.npz';probe_arrays=dict(states=probes,central_indices=central,membership=membership,parent_logs=old_probe_logs)
    if probe_path.exists():
        with np.load(probe_path) as data:
            if any(not np.array_equal(data[k],v) for k,v in probe_arrays.items()):raise ValueError('probe replay changed')
    else:np.savez_compressed(probe_path,**probe_arrays)
    records=[]
    for seed in spec['training_seeds']:
        folder=output/f'batch_{seed}';folder.mkdir(exist_ok=True)
        base=prepare_initial_state(folder/'initial.pkl',p0,logs,seed=seed,chains=spec['training_chains'],n_sites=36,
            burn=spec['training_burn_proposals'],registration_sha256=reg['result_sha256'])
        train=np.asarray(base['params']['chains']);train_path=folder/'parent.npz';train_record=folder/'parent.json'
        if train_record.exists():
            meta=_sealed_json(train_record)
            if meta['registration_sha256']!=reg['result_sha256'] or meta['raw_sha256']!=_file_sha256(train_path) or meta['states_sha256']!=array_sha256(train):raise ValueError('training batch changed')
            with np.load(train_path) as data:train_e,train_logs=data['energies'],data['logs']
        else:
            train_e=evaluate_chunks(local,p0,train,chunk);train_logs=evaluate_chunks(batch_logs,p0,train,chunk)
            np.savez_compressed(train_path,energies=train_e,logs=train_logs)
            seal(train_record,dict(registration_sha256=reg['result_sha256'],raw_sha256=_file_sha256(train_path),states_sha256=array_sha256(train)))
        for plan in spec['arms']:
            arm={**next(a for a in prior['arms'] if a['id']==plan['base']),**plan['overrides'],'id':plan['id']}
            destination=folder/arm['id'];destination.mkdir(exist_ok=True);record_path=destination/'results.json';raw_path=destination/'audit.npz';cp=destination/'candidate.pkl'
            if record_path.exists():
                r=_sealed_json(record_path)
                if r['registration_sha256']!=reg['result_sha256'] or r['checkpoint_sha256']!=_file_sha256(cp) or r['raw_sha256']!=_file_sha256(raw_path):raise ValueError('candidate result changed')
                records.append(r);continue
            tick=perf_counter()
            if cp.exists():
                saved=load_checkpoint(cp);p=saved['params']['model'];meta=saved['metadata']
                if meta['registration_sha256']!=reg['result_sha256'] or meta['parent_sha256']!=p0hash or meta['arm']!=arm or meta['batch_seed']!=seed:raise ValueError('candidate checkpoint changed')
                diag=meta['diagnostics']
            else:
                opt=parent['optimizer_state'] if arm['optimizer']=='adam' else dict(step=jnp.asarray(0))
                p,opt,d=build_update(arm,logs,p0)(p0,opt,jnp.asarray(train),jnp.asarray(train_e));jax.block_until_ready(p)
                if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves((p,opt,d))):raise FloatingPointError('candidate nonfinite')
                diag={k:np.asarray(v).item() for k,v in d.items()}
                if diag.get('relative_linear_residual',0)>1e-7:raise ValueError('candidate SR residual')
                save_checkpoint(cp,dict(model=p),optimizer_state=opt,metadata=dict(registration_sha256=reg['result_sha256'],
                    parent_sha256=p0hash,batch_seed=seed,arm=arm,diagnostic_only=True,optimizer_steps_from_parent=1,diagnostics=diag))
            if pytree_sha256(p)!=pytree_sha256(load_checkpoint(cp)['params']['model']):raise ValueError('candidate replay')
            new_train_e=evaluate_chunks(local,p,train,chunk);new_val_e=evaluate_chunks(local,p,flat_val,chunk).reshape(val_e.shape)
            train_ratio=evaluate_chunks(batch_logs,p,train,chunk)-train_logs
            val_ratio=evaluate_chunks(batch_logs,p,flat_val,chunk).reshape(val_e.shape)-val_logs
            probe_ratio=evaluate_chunks(batch_logs,p,probes,chunk)-old_probe_logs
            training=correlated_summary(train_ratio,train_e,new_train_e,n_sites=36)
            validation=correlated_summary(val_ratio,val_e,new_val_e,n_sites=36)
            by_stream=[correlated_summary(val_ratio[s],val_e[s],new_val_e[s],n_sites=36) for s in range(len(pools))]
            probing=probe_diagnostics(probe_ratio,central,probes,train)
            uncertainty=paired_chain_intervals(val_ratio,val_e,new_val_e,n_sites=36,**spec['bootstrap'])
            # Compare physical amplitudes, insensitive to log branch choices.
            sample=train[:4];before=batch_logs(p,jnp.asarray(sample))
            a1=max(float(jnp.max(abs(jnp.exp(batch_logs(p,jnp.asarray(sample[:,g]))-before)-1))) for g in group[::max(1,len(group)//8)])
            limits=spec['reference_flags']
            flags=dict(training_trust_pass=training['sampled_fs_sine']<=limits['held_out_fs'] and training['reweighting_ess_fraction']>=limits['held_out_ess'],
                any_validation_limit_exceeded=any(s['sampled_fs_sine']>limits['held_out_fs'] or s['reweighting_ess_fraction']<limits['held_out_ess'] for s in by_stream),
                central_high_weight_growth_warning=probing['central_high_minus_low_log_weight_change']>limits['probe_log_weight_shift'],
                outside_neighborhood_change_warning=probing['outside_training_max_abs_centered_log_weight_change']>limits['probe_log_weight_shift'])
            np.savez_compressed(raw_path,train_log_ratios=train_ratio,train_candidate_energies=new_train_e,
                validation_log_ratios=val_ratio,validation_candidate_energies=new_val_e,probe_log_ratios=probe_ratio)
            r=seal(record_path,dict(registration_sha256=reg['result_sha256'],seed=seed,arm=arm,diagnostics=diag,
                training=training,validation=validation,validation_by_stream=by_stream,probes=probing,uncertainty=uncertainty,
                reference_flags=flags,parent_sha256=p0hash,candidate_sha256=pytree_sha256(p),checkpoint_sha256=_file_sha256(cp),
                raw_sha256=_file_sha256(raw_path),full_H_calls=len(train)+len(flat_val),seconds=perf_counter()-tick,
                optimizer_steps_from_parent=1,sampled_A1_error=a1,adopted=False))
            print('CANDIDATE',seed,arm['id'],'train/validation FS',training['sampled_fs_sine'],validation['sampled_fs_sine'],
                'dE',validation['delta_energy_per_site'],'high-low',probing['central_high_minus_low_log_weight_change'],flags,flush=True)
            records.append(r)
    parent_after,_=load_parent(prior,'square')
    if pytree_sha256(parent_after['params']['model'])!=p0hash:raise ValueError('parent modified')
    result=seal(output/'results.json',dict(registration_sha256=reg['result_sha256'],parent_sha256=p0hash,
        parent_validation=audit_statistics(val_e,36,bootstrap_seed=spec['bootstrap']['seed'],replicates=4000),
        validation_pools=pools,probe_count=len(probes),probe_sha256=_file_sha256(probe_path),candidates=records,
        full_H_calls=val_e.size+len(spec['training_seeds'])*spec['training_chains']+sum(r['full_H_calls'] for r in records),
        wall_seconds=perf_counter()-started,candidate_equilibrium_claimed=False,remote_submitted=False,adopted_candidates=0))
    print('COMPLETE',result['full_H_calls'],result['wall_seconds'],flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();run(args.output)
