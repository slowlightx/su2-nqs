"""Bounded seed replication and frozen-state sampler diagnosis; no new ansatz.

The previous optimizer protocol/source/artifacts stay immutable. Reuse its
optimizers, sampler, full-H evaluator, audit inference and checkpoint helpers.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np

from .spin_full_vit_optimization import (
    CONFIG as PRIOR_CONFIG,build_update,compare_audits,load_parent,prepare_initial_state,seal,frozen_audit,
)
from .spin_full_vit_pilot import ROOT,_sealed_json,_file_sha256
from .spin import _groups,_bonds
from su2_nqs.benchmarks.spatial_vit_sampling import projected_log_function,make_full_heisenberg_ratios,make_log_exchange_sampler
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig
from su2_nqs.vmc.tempered_spin_exchange import make_tempered_spin_exchange
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256,array_sha256,canonical_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest,numerical_runtime_identity

CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_full_vit_recheck.json'
OUTPUT=ROOT/'projects/clebsch_tree_nqs/results/spin_full_vit_recheck_v1'


def registration(output: Path) -> tuple[dict,dict,dict]:
    """Seal both experiments before training, with prior data/source identities."""
    spec=json.loads(CONFIG.read_text());prior=json.loads(PRIOR_CONFIG.read_text())
    old=ROOT/spec['prior_directory'];oldreg=_sealed_json(old/'registration.json')
    if oldreg['result_sha256']!=spec['prior_registration_sha256'] or oldreg['config']!=prior:
        raise ValueError('prior registered protocol changed')
    for path,digest in oldreg['source_manifest'].items():
        if _file_sha256(ROOT/path)!=digest:raise ValueError('prior numerical source changed')
    source_results={}
    for g in ('triangular','square'):
        load_parent(prior,g)
        r=_sealed_json(old/g/'results.json');source_results[g]=r['result_sha256']
        for label,t in r['training'].items():
            if _file_sha256(old/g/label/'checkpoint.pkl')!=t['checkpoint_sha256']:
                raise ValueError('prior checkpoint changed')
        for label,a in r['audits'].items():
            for s in a['streams']:
                if _file_sha256(old/g/f"audit_{label}_{s['seed']}.npz")!=s['raw_sha256']:
                    raise ValueError('prior audit changed')
    seeds=list(spec['triangular']['training_seeds'])+spec['square']['seeds']
    seeds += [s for pool in spec['triangular']['audit']['seeds'].values() for values in pool.values() for s in values]
    oldseeds=list(prior['training_seeds'].values())+[s for pool in prior['audit']['seeds'].values() for v in pool.values() for s in v]
    if len(seeds)!=len(set(seeds)) or set(seeds)&set(oldseeds):raise ValueError('seed reuse')
    output.mkdir(parents=True,exist_ok=True)
    data=dict(config=spec,config_sha256=canonical_sha256(spec),prior_results=source_results,
        source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),runtime=numerical_runtime_identity())
    path=output/'registration.json'
    if path.exists():
        reg=_sealed_json(path)
        if any(reg[k]!=v for k,v in data.items()):raise ValueError('registered recheck changed')
    else:reg=seal(path,{**data,'registered_at_utc':datetime.now(timezone.utc).isoformat()})
    return spec,prior,reg


def components(reference: dict):
    """Build unchanged physical-input model, projector and complete Hamiltonian."""
    config=FixedSzSpatialViTConfig(**reference['model_config'])
    bonds=_bonds(config.geometry,6,6,reference['j2_over_j1'])
    group=_groups(config.geometry,6,6)['gamma_a1']
    log_fn,_=projected_log_function(config,group)
    return log_fn,make_full_heisenberg_ratios(log_fn,bonds),group,bonds


def train_fork(folder: Path,base: dict,parent: dict,arm: dict,spec: dict,reg: dict,
               log_fn,local,warm: jax.Array) -> tuple[dict,object]:
    """One fixed-budget arm; shared optimizer kernels, atomic resume each step."""
    folder.mkdir(parents=True,exist_ok=True);cp=folder/'checkpoint.pkl';record=folder/'training.json'
    if record.exists():
        t=_sealed_json(record)
        if t['registration_sha256']!=reg['result_sha256'] or t['checkpoint_sha256']!=_file_sha256(cp):
            raise ValueError('training artifact changed')
        return t,load_checkpoint(cp)['params']['model']
    if cp.exists():
        saved=load_checkpoint(cp);state,opt,meta=saved['params'],saved['optimizer_state'],saved['metadata']
        if meta['registration_sha256']!=reg['result_sha256'] or len(meta['history'])!=meta['completed_steps']:
            raise ValueError('checkpoint identity/count mismatch')
        count=int(opt[0] if arm['id']=='adam' else opt['step'])
        if count!=meta['completed_steps']+(1064 if arm['id']=='adam' else 0):raise ValueError('optimizer counter mismatch')
    else:
        state=base['params'];opt=parent['optimizer_state'] if arm['id']=='adam' else dict(step=jnp.asarray(0))
        meta=dict(registration_sha256=reg['result_sha256'],completed_steps=0,history=[],active_seconds=0.,
            compile_seconds=0.,initial_model_sha256=pytree_sha256(state['model']),
            initial_state_sha256=pytree_sha256(state),initial_optimizer_sha256=pytree_sha256(opt))
    update=build_update(arm,log_fn,state['model']);sampler=make_log_exchange_sampler(log_fn,spec['proposals_per_update'])
    tick=perf_counter();jax.block_until_ready(update(base['params']['model'],opt,base['params']['chains'],warm))
    meta['compile_seconds']+=perf_counter()-tick
    save_checkpoint(cp,state,optimizer_state=opt,metadata=meta)
    for step in range(meta['completed_steps']+1,spec['steps']+1):
        if meta['active_seconds']>=spec['maximum_training_seconds_per_arm']:break
        tick=perf_counter();x,key,_,move=sampler(state['model'],state['key'],state['chains'])
        e=local(state['model'],x);p,opt,diag=update(state['model'],opt,x,e)
        jax.block_until_ready((p,opt,diag,e))
        if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves((p,opt,diag,e))):raise FloatingPointError('nonfinite update')
        if float(diag.get('relative_linear_residual',0))>1e-7:raise FloatingPointError('SR residual failed')
        elapsed=perf_counter()-tick;meta['active_seconds']+=elapsed;meta['completed_steps']=step
        values=np.asarray(e);state=dict(model=p,chains=x,key=key)
        row=dict(step=step,energy_before_update_per_site=float(values.mean().real/36),
            variance_before_update_per_site=float(np.mean(abs(values-values.mean())**2)/36),
            seconds=elapsed,active_seconds=meta['active_seconds'],nontrivial_acceptance=float(move),
            **{k:np.asarray(v).item() for k,v in diag.items()})
        if step==1:row.update(sample_batch_sha256=array_sha256(np.asarray(x)),local_energy_batch_sha256=array_sha256(values))
        meta['history'].append(row);save_checkpoint(cp,state,optimizer_state=opt,metadata=meta)
        if step==1 or step%32==0:print(folder.parent.name,arm['id'],step,row['energy_before_update_per_site'],flush=True)
    replay=load_checkpoint(cp)
    if pytree_sha256((state,opt))!=pytree_sha256((replay['params'],replay['optimizer_state'])):raise ValueError('checkpoint replay')
    t=seal(record,dict(**meta,arm=arm,checkpoint_sha256=_file_sha256(cp),
        final_params_sha256=pytree_sha256(state['model']),training_state_sha256=pytree_sha256((state,opt)),
        full_H_configuration_calls=meta['completed_steps']*spec['chains'],parameter_count=4032,
        status='fixed_budget_complete' if meta['completed_steps']==spec['steps'] else 'time_cap'))
    return t,state['model']


def triangular(seed: int,output: Path,*,resume: bool=False) -> dict:
    """Replicate training randomness, not independent neural initialization."""
    started=perf_counter();spec,prior,reg=registration(output);cfg=spec['triangular']
    if seed not in cfg['training_seeds']:raise ValueError('unregistered seed')
    folder=output/'triangular'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():raise FileExistsError('completed replication is immutable')
    if (folder/'initial.pkl').exists() and not resume:raise FileExistsError('use --resume')
    parent,ref=load_parent(prior,'triangular');p0=parent['params']['model'];log_fn,local,group,_=components(ref)
    base=prepare_initial_state(folder/'initial.pkl',p0,log_fn,seed=seed,chains=cfg['chains'],n_sites=36,
        burn=cfg['initial_burn_proposals'],registration_sha256=reg['result_sha256'])
    warm=local(p0,base['params']['chains']);jax.block_until_ready(warm)
    work_path=folder/'execution.json';work=json.loads(work_path.read_text()) if work_path.exists() else dict(warmup_full_H_calls=0)
    work['warmup_full_H_calls']+=64;atomic_json(work_path,work)
    arms=[next(a for a in prior['arms'] if a['id']==name) for name in cfg['arms']]
    trained={};records={}
    for arm in arms:records[arm['id']],trained[arm['id']]=train_fork(folder/arm['id'],base,parent,arm,cfg,reg,log_fn,local,warm)
    if len({r['initial_state_sha256'] for r in records.values()})!=1 or len({r['history'][0]['sample_batch_sha256'] for r in records.values()})!=1:
        raise ValueError('unmatched initialization')
    audit_cfg=dict(cfg['audit']);audit_cfg['seeds']={'triangular':cfg['audit']['seeds'][str(seed)]}
    audits={};raw={};symmetry={}
    logs=jax.jit(jax.vmap(log_fn,in_axes=(None,0)))
    for label,p in trained.items():
        audits[label],raw[label]=frozen_audit(p,label,'triangular',folder,{'audit':audit_cfg},reg,log_fn,local)
        x=base['params']['chains'][:8];before=logs(p,x)
        symmetry[label]=max(float(jnp.max(abs(jnp.exp(logs(p,x[:,jnp.asarray(g)])-before)-1))) for g in group[::max(1,len(group)//8)])
    comparison=compare_audits(raw['ridge_sr'],raw['adam'],n_sites=36,seed=audit_cfg['bootstrap_seed']+seed%100,replicates=4000)
    matched=all(r['completed_steps']==128 for r in records.values())
    result=dict(seed=seed,registration_sha256=reg['result_sha256'],training=records,audits=audits,comparison=comparison,
        sampled_A1_relative_errors=symmetry,matched_training_work=matched,
        conditional_joint_pass=bool(matched and all(v['primary_family_bonferroni_98_75'][1]<0 for v in comparison.values())),
        full_H_configuration_calls=work['warmup_full_H_calls']+sum(r['full_H_configuration_calls'] for r in records.values())+sum(a['samples'] for a in audits.values()),
        wall_seconds=perf_counter()-started,independent_weight_initialization=False,remote_submitted=False)
    for label,t in records.items():
        if _file_sha256(folder/label/'checkpoint.pkl')!=t['checkpoint_sha256']:raise ValueError('audit altered checkpoint')
    seal(folder/'results.json',result);print('TRIANGULAR COMPLETE',seed,comparison,flush=True)
    return result


def square_start_pool(spec: dict,seed: int) -> tuple[np.ndarray,np.ndarray,jax.Array]:
    """Prespecified biased low/high cohorts plus fresh uniform fixed-M2=0 starts."""
    cfg=spec['square'];old=ROOT/spec['prior_directory']/'square';r=_sealed_json(old/'results.json')
    paths=[];states=[]
    for rec in r['audits']['ridge_sr']['streams']:
        with np.load(old/f"audit_ridge_sr_{rec['seed']}.npz") as f:
            paths.extend(f['local_energies'].real.mean(axis=0)/36);states.extend(f['states'][-1])
    paths=np.asarray(paths);states=np.asarray(states);n=cfg['chains_per_cohort']
    low=states[np.flatnonzero(paths<-.4)[:n]];high=states[np.flatnonzero(paths>0)[:n]]
    if len(low)!=n or len(high)!=n:raise ValueError('registered diagnostic starts unavailable')
    init,hot,key=jax.random.split(jax.random.PRNGKey(seed),3)
    uniform=initialize_spin_exchange_chains(init,n_chains=n,n_sites=36,n_up=18)
    cold=np.concatenate((np.asarray(uniform),low,high))
    replicas=len(cfg['betas']);warm=initialize_spin_exchange_chains(hot,n_chains=(replicas-1)*len(cold),n_sites=36,n_up=18)
    all_states=np.concatenate((np.asarray(warm).reshape(replicas-1,len(cold),36),cold[None]),axis=0)
    return cold,all_states,key


def frozen_job_summary(energies: np.ndarray,cohort_size: int,last_blocks: int) -> dict:
    """Separate deliberately biased start cohorts; no equilibrium energy claim."""
    tail=energies[-last_blocks:];groups=[]
    for index in range(3):
        x=tail[:,index*cohort_size:(index+1)*cohort_size];m=x.mean()
        groups.append(dict(energy_per_site=float(m.real/36),variance_per_site=float(np.mean(abs(x-m)**2)/36),
            chain_se=float(x.real.mean(axis=0).std(ddof=1)/np.sqrt(cohort_size)/36)))
    paths=tail.real.T;half=last_blocks//2
    split=np.concatenate((paths[:,:half],paths[:,-half:]),axis=0)
    within=np.var(split,axis=1,ddof=1).mean();between=half*np.var(split.mean(axis=1),ddof=1)
    rhat=float(np.sqrt(((half-1)*within+between)/(half*within))) if within>0 else None
    return dict(cohorts=groups,energy_start_cohort_range=float(np.ptp([g['energy_per_site'] for g in groups])),
        descriptive_split_rhat=rhat,equilibrium_certified=False)


def explicit_full_H(log_fn,params: object,bits: np.ndarray,bonds: list) -> np.ndarray:
    """Independent unchunked physical-configuration H replay; no Hilbert enumeration."""
    batch=jax.jit(jax.vmap(log_fn,in_axes=(None,0)));result=[]
    for x in bits:
        images=np.tile(x,(len(bonds)+1,1));diagonal=0.;weights=[]
        for index,(i,j,c) in enumerate(bonds):
            opposite=x[i]!=x[j];diagonal+=c*(-.25 if opposite else .25)
            images[index+1,i],images[index+1,j]=x[j],x[i]
            weights.append(.5*c if opposite else 0.)
        logs=np.asarray(batch(params,jnp.asarray(images)))
        result.append(diagonal+np.dot(weights,np.exp(logs[1:]-logs[0])))
    return np.asarray(result)


def square_reference_check(spec: dict,prior: dict,reg: dict,output: Path) -> dict:
    """Compare two unchanged checkpoints on identical starts and independent H action."""
    folder=output/'square';folder.mkdir(parents=True,exist_ok=True);path=folder/'reference_check.json'
    if path.exists():
        r=_sealed_json(path)
        if r['registration_sha256']!=reg['result_sha256']:raise ValueError('reference check registration changed')
        return r
    parent,ref=load_parent(prior,'square');log_fn,local,_,bonds=components(ref)
    frozen=load_checkpoint(ROOT/spec['prior_directory']/'square/ridge_sr/checkpoint.pkl')['params']['model']
    cold,_,_=square_start_pool(spec,spec['square']['seeds'][0]);n=spec['square']['chains_per_cohort']
    selected=[0,n,2*n,1,n+1,2*n+1];diagnostics={}
    for label,p in (('parent',parent['params']['model']),('ridge_sr',frozen)):
        e=np.asarray(local(p,jnp.asarray(cold)));independent=explicit_full_H(log_fn,p,cold[selected],bonds)
        error=float(np.max(abs(e[selected]-independent)));relative=error/max(1.,float(np.max(abs(independent))))
        if relative>1e-10 or not np.isfinite(e).all():raise ValueError('independent frozen H replay failed')
        logw=2*np.asarray(jax.jit(jax.vmap(log_fn,in_axes=(None,0)))(p,jnp.asarray(cold))).real
        diagnostics[label]=dict(parameter_sha256=pytree_sha256(p),H_replay_max_absolute=error,
            H_replay_max_relative=relative,local_energies_per_site=(e.real/36).tolist(),
            log_born_weights=logw.tolist(),cohort_mean_E_per_site=(e.real.reshape(3,n).mean(axis=1)/36).tolist(),
            high_minus_low_mean_log_weight=float(logw[2*n:].mean()-logw[n:2*n].mean()))
    return seal(path,dict(registration_sha256=reg['result_sha256'],states=cold.tolist(),
        diagnostics=diagnostics,full_H_configuration_calls=60,
        interpretation='Selected-configuration relative weights are gauge-invariant probes, NOT basin probabilities or energies of a Born ensemble.'))


def square(state_label: str,kernel: str,seed: int,output: Path,*,resume: bool=False) -> dict:
    """Frozen-parameter intervention: no optimizer or training invocation."""
    started=perf_counter();spec,prior,reg=registration(output);cfg=spec['square']
    if state_label not in cfg['parameter_states'] or kernel not in cfg['kernels'] or seed not in cfg['seeds']:raise ValueError('unregistered job')
    square_reference_check(spec,prior,reg,output)
    folder=output/'square'/f'{state_label}_{kernel}_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():raise FileExistsError('completed frozen job is immutable')
    cp=folder/'chains.pkl'
    if cp.exists() and not resume:raise FileExistsError('use --resume')
    parent,ref=load_parent(prior,'square');log_fn,local,_,_=components(ref)
    source=ROOT/prior['parent_directory']/'square'/'checkpoint.pkl' if state_label=='parent' else ROOT/spec['prior_directory']/'square'/'ridge_sr'/'checkpoint.pkl'
    source_hash=_file_sha256(source);p=load_checkpoint(source)['params']['model'];param_hash=pytree_sha256(p)
    cold,hot,key=square_start_pool(spec,seed);R=len(cfg['betas']);C=len(cold)
    if cp.exists():
        saved=load_checkpoint(cp);state,meta=saved['params'],saved['metadata']
        if meta['registration_sha256']!=reg['result_sha256'] or meta['param_hash']!=param_hash:raise ValueError('frozen checkpoint mismatch')
        count=meta['completed_blocks']
        if len(meta['history'])!=count or len(meta['energies'])!=count+1 or len(meta['cold_states'])!=count+1:
            raise ValueError('frozen checkpoint count mismatch')
        x=np.asarray(state['states'])
        if x.shape!=((C,36) if kernel=='ordinary' else (R,C,36)) or not np.isin(x,[0,1]).all() or not np.all(x.sum(axis=-1)==18):
            raise ValueError('frozen checkpoint spin sector mismatch')
        if kernel=='tempered' and (len(meta['cold_lineages'])!=count or not np.array_equal(np.sort(state['lineages'],axis=0),np.broadcast_to(np.arange(R)[:,None],(R,C)))):
            raise ValueError('frozen checkpoint lineage mismatch')
    else:
        state=dict(states=jnp.asarray(cold if kernel=='ordinary' else hot),key=key,
            lineages=jnp.broadcast_to(jnp.arange(R)[:,None],(R,C)))
        e=np.asarray(local(p,jnp.asarray(cold)))
        meta=dict(registration_sha256=reg['result_sha256'],param_hash=param_hash,completed_blocks=0,
            active_seconds=0.,energies=[e],cold_states=[cold],history=[],cold_lineages=[])
        save_checkpoint(cp,state,metadata=meta)
    sampler=(make_log_exchange_sampler(log_fn,cfg['ordinary_proposals_per_block']) if kernel=='ordinary' else
        make_tempered_spin_exchange(log_fn,cfg['betas'],proposals=cfg['tempered_proposals_per_block'],swap_every=cfg['swap_every']))
    # No warm H evaluation; first call compilation is included in this bounded job.
    for block in range(meta['completed_blocks']+1,cfg['blocks']+1):
        if meta['active_seconds']>=cfg['maximum_seconds_per_job']:break
        tick=perf_counter()
        if kernel=='ordinary':
            x,key,_,move=sampler(p,state['key'],state['states']);cold=x
            state=dict(states=x,key=key,lineages=state['lineages']);diag=dict(local_move_fraction=[float(move)])
        else:
            x,key,tags,d=sampler(p,state['key'],state['states'],state['lineages']);cold=x[-1]
            state=dict(states=x,key=key,lineages=tags);diag={k:np.asarray(v).tolist() for k,v in d.items()}
            meta['cold_lineages'].append(np.asarray(tags[-1]))
        e=np.asarray(local(p,cold))
        if not np.isfinite(e).all():raise FloatingPointError('frozen energy nonfinite; no retuning')
        meta['active_seconds']+=perf_counter()-tick;meta['completed_blocks']=block
        meta['energies'].append(e);meta['cold_states'].append(np.asarray(cold));meta['history'].append(diag)
        save_checkpoint(cp,state,metadata=meta)
        if block%8==0:print(state_label,kernel,seed,'block',block,'cohort E/N',e.real.reshape(3,-1).mean(axis=1)/36,flush=True)
    e=np.asarray(meta['energies']);raw=folder/'trajectories.npz'
    np.savez_compressed(raw,local_energies=e,states=np.asarray(meta['cold_states']),cold_lineages=np.asarray(meta['cold_lineages']))
    if _file_sha256(source)!=source_hash or pytree_sha256(p)!=param_hash:raise ValueError('frozen parameters altered')
    if meta['completed_blocks']<cfg['last_blocks_for_comparison']:raise RuntimeError('insufficient bounded diagnostic trajectory')
    result=dict(state=state_label,kernel=kernel,seed=seed,registration_sha256=reg['result_sha256'],
        parameter_sha256=param_hash,source_checkpoint_sha256=source_hash,raw_sha256=_file_sha256(raw),
        completed_blocks=meta['completed_blocks'],status='fixed_budget_complete' if meta['completed_blocks']==cfg['blocks'] else 'time_cap',
        summary=frozen_job_summary(e,cfg['chains_per_cohort'],cfg['last_blocks_for_comparison']),
        history=meta['history'],active_seconds=meta['active_seconds'],wall_seconds=perf_counter()-started,
        full_H_configuration_calls=e.size,proposal_amplitude_evaluations=meta['completed_blocks']*C*cfg['ordinary_proposals_per_block'],
        refresh_amplitude_evaluations=meta['completed_blocks']*C*(R if kernel=='tempered' else 1),
        optimizer_updates=0,equilibrium_energy_claimed=False)
    seal(folder/'results.json',result);print('SQUARE FROZEN COMPLETE',state_label,kernel,seed,result['summary'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('triangular','square'))
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--state',choices=('parent','ridge_sr'))
    parser.add_argument('--kernel',choices=('ordinary','tempered'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.task=='triangular':triangular(args.seed,args.output,resume=args.resume)
    else:square(args.state,args.kernel,args.seed,args.output,resume=args.resume)
