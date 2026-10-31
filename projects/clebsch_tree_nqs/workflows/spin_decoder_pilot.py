"""Bounded same-architecture N36 health, joint energy pilot and fresh audits."""
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

from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios, make_log_exchange_sampler
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.vmc.bounded_physical_pilot import (
    make_dynamic_force, initialize_pilot_adam, bounded_adam_step, energy_summary, independent_audit_comparison,
)
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.projected_spin_work import make_chunked_a1_amplitudes
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint
from .spin import _groups, _bonds
from .spin_decoder_cost import ROOT, physical_config, amplitude_functions, peak_rss_bytes
from .spin_vap_decoder import init_decoder
from .spin_sampled_runtime import _autocorrelation_summary

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_decoder_pilot.json'


def registration(spec: dict) -> dict:
    """Seal protocol, inherited parameters, executable source and numerical runtime."""
    if (spec['chains'],spec['microbatch'],spec['train_snapshots_per_step'])!=(16,8,2):
        raise ValueError('this bounded protocol requires16 chains,8-center H and2 training snapshots')
    for prefix in ('health','audit'):
        if spec[prefix+'_snapshots']%spec[prefix+'_energy_every'] or spec[prefix+'_snapshots']//spec[prefix+'_energy_every']<4:
            raise ValueError('need at least4 equally spaced energy snapshots per frozen phase')
    parent=ROOT/spec['parent'];paths=[]
    for case in spec['cases']:
        paths += [parent/case['geometry']/name for name in ('parameters.json','parameters.npz','values.npz')]
    r=dict(config=spec,inputs={str(p.relative_to(ROOT)):file_sha256(p)for p in paths},
        runtime=numerical_runtime_identity(),sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    r['registration_sha256']=canonical_sha256(r);return r


def starting_chains(seed: int, tag: int) -> tuple[np.ndarray,np.ndarray]:
    """Eight uniform and eight ordered M=0 starts; no amplitude-based selection."""
    key=jax.random.fold_in(jax.random.PRNGKey(seed),tag)
    initial,k0,k1=jax.random.split(key,3)
    uniform=np.asarray(initialize_spin_exchange_chains(initial,n_chains=8,n_sites=36,n_up=18))
    r,c=np.indices((6,6));patterns=[(r+c)%2,r%2,c%2,1-(r+c)%2]
    ordered=np.asarray([patterns[i%4].ravel() for i in range(8)],dtype=np.int32)
    return np.concatenate((uniform,ordered)),np.asarray(jnp.stack((k0,k1)))


def summarize_phase(record: dict) -> dict:
    """Short-chain diagnostics are descriptive, never an equilibrium certificate."""
    energies=np.asarray(record['energies']);logs=np.asarray(record['log_weights'])
    nn=np.asarray(record['nn']);states=np.asarray(record['states'])
    r=energy_summary(energies,36)
    r['diagnostics']={name:_autocorrelation_summary(values) for name,values in
        [('energy',energies.real/36),('log_weight',logs),('nn_correlation',nn)]}
    r['cohorts']=[energy_summary(energies[:,i:i+8],36)for i in (0,8)]
    a,b=r['cohorts'];r['cohort_energy_difference']=b['energy_per_site']-a['energy_per_site']
    r['cohort_energy_difference_se']=float(np.hypot(a['energy_chain_se'],b['energy_chain_se']))
    r['unique_states_per_chain']=[len(np.unique(states[:,i],axis=0))for i in range(16)]
    r['mean_acceptance_including_null']=float(np.mean(np.asarray(record['accept_moves'])[:,:,0]))
    r['mean_nontrivial_move_fraction']=float(np.mean(np.asarray(record['accept_moves'])[:,:,1]))
    r['screening_ok']=bool(min(r['unique_states_per_chain'])>1 and all(
        v['split_rhat'] is not None and v['split_rhat']<=1.2 for v in r['diagnostics'].values())
        and abs(r['cohort_energy_difference'])<=3*max(r['cohort_energy_difference_se'],1e-12))
    return r


def run(path: Path,geometry: str,*,resume: bool=False) -> dict:
    """One bounded process; checkpoints include every phase and all RNG states."""
    spec=json.loads(path.read_text());folder=ROOT/spec['output'];d=folder/geometry
    reg=json.loads((folder/'registration.json').read_text())
    if reg!=registration(spec):raise ValueError('registration mismatch')
    if (d/'results.json').exists():raise FileExistsError('completed pilot')
    case=next(c for c in spec['cases']if c['geometry']==geometry)
    identity=dict(registration_sha256=reg['registration_sha256'],geometry=geometry)
    checkpoint=d/'checkpoint.pkl';parent=ROOT/spec['parent']/geometry
    config=physical_config(6,6,2);params0=load_parameter_artifact(parent/'parameters.json',init_decoder(2026093003,config))
    theta0,unravel=ravel_pytree(params0);theta=theta0;optimizer=initialize_pilot_adam(theta);history=[]
    start=perf_counter()
    if resume:
        saved=load_resumable_checkpoint(checkpoint,expected_run_identity=identity)
        theta=jnp.asarray(saved['params']);optimizer=jax.tree_util.tree_map(jnp.asarray,saved['optimizer_state'])
        history=saved['history'];state=saved['metadata'];previous_seconds=state['active_seconds']
    else:
        if checkpoint.exists()or(d/'progress.json').exists():raise FileExistsError('use --resume for this pilot')
        state=dict(phase='health',burn_done=0,snapshot_done=0,chains=None,keys=None,records={},
            summaries={},active_seconds=0.,full_H_centers=0,proposal_attempts=0,compiled={},
            source_parameters_sha256=pytree_sha256(params0),training_blocked=False)
        previous_seconds=0.
    def persist(event: str) -> None:
        state['active_seconds']=previous_seconds+perf_counter()-start
        save_resumable_checkpoint(checkpoint,np.asarray(theta),optimizer_state=jax.device_get(optimizer),
            completed_steps=len(history),next_sample_index=state['proposal_attempts'],
            run_identity=identity,history=history,metadata=state)
        atomic_json(d/'progress.json',dict(event=event,phase=state['phase'],burn_done=state['burn_done'],
            snapshot_done=state['snapshot_done'],completed_steps=len(history),active_seconds=state['active_seconds'],
            full_H_centers=state['full_H_centers'],peak_rss_bytes=peak_rss_bytes(),history=history,
            summaries=state['summaries']))
        print(geometry,event,'step',len(history),round(state['active_seconds'],2),'s',flush=True)
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30:raise MemoryError('registered RSS cap')
        if state['active_seconds']>spec['wall_limit_seconds_per_geometry']:raise TimeoutError('registered cumulative wall cap')
    def alarm(_s,_f):raise TimeoutError('registered cumulative wall cap')
    old_signal=signal.signal(signal.SIGALRM,alarm)
    signal.alarm(max(1,int(spec['wall_limit_seconds_per_geometry']-previous_seconds)))
    try:
        persist('setup')
        group=_groups(geometry,6,6)['gamma_a1'];old=amplitude_functions(config,group)
        bonds=_bonds(geometry,6,6,case['j2_over_j1'])
        log_fn=lambda t,x:jnp.log(old['a1'](unravel(t),x))
        H=make_full_heisenberg_ratios(log_fn,bonds)
        logs=jax.jit(jax.vmap(log_fn,in_axes=(None,0)))
        sampler=make_log_exchange_sampler(log_fn,spec['proposal_block'])
        chunked=make_chunked_a1_amplitudes(old['raw'],old['projector'],chunk_size=spec['gradient_chunk_size'])
        centers=jax.jit(lambda t,x:chunked(unravel(t),x))
        force=make_dynamic_force(centers,36)
        nn_edges=np.asarray([(a,b)for a,b,J in bonds if J==1.],dtype=np.int32)
        def sample_block(parameters):
            rows=[];keys=[];diagnostics=[]
            for i in (0,1):
                result=sampler(parameters,jnp.asarray(state['keys'][i]),jnp.asarray(state['chains'][8*i:8*i+8]))
                x,k,accept,moves=jax.device_get(result);rows.append(x);keys.append(k);diagnostics.append([float(accept),float(moves)])
            state['chains']=np.concatenate(rows);state['keys']=np.asarray(keys)
            state['proposal_attempts']+=spec['chains']*spec['proposal_block']
            if not np.all(state['chains'].sum(axis=1)==18):raise ValueError('sector violation')
            return diagnostics
        def local_energies(parameters,states):
            outputs=[]
            for i in range(0,len(states),8):
                x=np.asarray(H(parameters,jnp.asarray(states[i:i+8])))
                state['full_H_centers']+=len(x)
                if not np.isfinite(x).all():raise FloatingPointError('nonfinite H ratio, no filtering permitted')
                outputs.append(x)
            return np.concatenate(outputs)
        def begin(phase):
            state['phase']=phase;state['burn_done']=0;state['snapshot_done']=0
            if phase!='done':
                tag={'health':1,'train':2,'audit_parent':3,'audit_terminal':4}[phase]
                state['chains'],state['keys']=starting_chains(case['seed'],tag)
                if phase!='train':state['records'][phase]=dict(states=[],log_weights=[],nn=[],energies=[],energy_indices=[],accept_moves=[])
            persist('begin_'+phase)
        # One fixed parent H replay is calibration work, never a training sample.
        with np.load(parent/'values.npz')as a:probes=a['states'];expected=a['a1_H_b8']
        before=perf_counter();observed=local_energies(theta0,probes)
        state['compiled']['H_warmup_seconds']=perf_counter()-before
        state['compiled']['H_parent_scaled_error']=float(np.max(abs(observed-expected)/(1+abs(expected))))
        if state['compiled']['H_parent_scaled_error']>1e-9:raise ValueError('old H replay failed')
        persist('H_calibrated')
        if state['chains'] is None:begin('health')
        while state['phase']!='done':
            phase=state['phase'];parameters=theta0 if phase in ('health','audit_parent')else theta
            burn=spec['train_burn_blocks']if phase=='train'else spec['health_burn_blocks']if phase=='health'else spec['audit_burn_blocks']
            while state['burn_done']<burn:
                sample_block(parameters);state['burn_done']+=1;persist(phase+'_burn')
            if phase=='train':
                while len(history)<spec['train_steps']:
                    before=perf_counter();batches=[];moves=[]
                    for _ in range(spec['train_snapshots_per_step']):
                        moves.append(sample_block(theta));batches.append(state['chains'].copy())
                    samples=np.concatenate(batches);energies=local_energies(theta,samples)
                    gradient=force(theta,jnp.asarray(samples),jnp.asarray(energies));jax.block_until_ready(gradient)
                    theta,optimizer,info=bounded_adam_step(theta,gradient,optimizer,jnp.asarray(samples),amplitudes=centers,
                        learning_rate=spec['learning_rate'],clip_norm=spec['gradient_clip_norm'],
                        maximum_fs=spec['maximum_fs'],bisections=spec['fs_bisections'])
                    mean=np.mean(energies)
                    history.append(dict(step=len(history)+1,energy_before_update_per_site=float(mean.real/36),
                        variance_before_update_per_site=float(np.mean(abs(energies-mean)**2)/36),
                        imaginary_energy_per_site=float(mean.imag/36),samples=len(samples),
                        mean_move_fraction=float(np.mean(np.asarray(moves)[:,:,1])),seconds=perf_counter()-before,**info))
                    persist('train_update')
                begin('audit_parent');continue
            snapshots=spec['health_snapshots']if phase=='health'else spec['audit_snapshots']
            every=spec['health_energy_every']if phase=='health'else spec['audit_energy_every']
            record=state['records'][phase]
            while state['snapshot_done']<snapshots:
                moves=sample_block(parameters);bits=state['chains'].copy()
                amplitude_logs=np.concatenate([np.asarray(logs(parameters,jnp.asarray(bits[i:i+8])))for i in (0,8)])
                if not np.isfinite(amplitude_logs).all():raise FloatingPointError('post-burn zero/nonfinite amplitudes')
                z=2*bits-1;record['states'].append(bits);record['log_weights'].append(2*amplitude_logs.real)
                record['nn'].append(.75*np.mean(z[:,nn_edges[:,0]]*z[:,nn_edges[:,1]],axis=1))
                record['accept_moves'].append(moves);state['snapshot_done']+=1
                if state['snapshot_done']%every==0:
                    record['energies'].append(local_energies(parameters,bits));record['energy_indices'].append(state['snapshot_done'])
                persist(phase+'_observe')
            state['summaries'][phase]=summarize_phase(record)
            if phase=='health':
                if min(state['summaries'][phase]['unique_states_per_chain'])<=1:
                    state['training_blocked']=True;begin('done')
                else:begin('train')
            elif phase=='audit_parent':begin('audit_terminal')
            else:begin('done')
        # Export parameter/checkpoint replay and raw chain time series, never best-step selection.
        parameter_file=d/'terminal_parameters.json'
        if not parameter_file.exists():save_parameter_artifact(parameter_file,unravel(theta),identity=identity)
        replay=load_parameter_artifact(parameter_file,params0,expected_identity=identity)
        actual=np.asarray(centers(theta,jnp.asarray(probes)))
        restored=ravel_pytree(replay)[0]
        replay_error=float(np.max(abs(np.asarray(centers(restored,jnp.asarray(probes)))-actual)))
        symmetry=max(float(np.linalg.norm(np.asarray(centers(theta,jnp.asarray(probes[:,g])))-actual)/np.linalg.norm(actual))
            for g in np.asarray(group)[[0,len(group)//3,len(group)//2+1,-1]])
        raw={f'{phase}_{name}':np.asarray(value)for phase,r in state['records'].items()for name,value in r.items()}
        np.savez_compressed(d/'chains.npz',**raw)
        comparison=None
        if not state['training_blocked']:
            comparison=independent_audit_comparison(raw['audit_parent_energies'],raw['audit_terminal_energies'],n_sites=36,
                seed=case['seed']+100,replicates=spec['bootstrap_replicates'])
        result=dict(geometry=geometry,case=case,registration_sha256=reg['registration_sha256'],history=history,
            summaries=state['summaries'],comparison=comparison,training_blocked=state['training_blocked'],
            completed_steps=len(history),initial_parameters_sha256=pytree_sha256(params0),
            terminal_parameters_sha256=pytree_sha256(replay),parameter_count=len(theta),spatial_images=len(group),
            checkpoint_replay_max=replay_error,A1_relative_error=symmetry,compiled=state['compiled'],
            full_H_centers=state['full_H_centers'],proposal_attempts=state['proposal_attempts'],
            active_seconds=previous_seconds+perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),
            raw_sha256=file_sha256(d/'chains.npz'),checkpoint_sha256=file_sha256(checkpoint),
            backend=jax.default_backend(),new_ED=False,remote_submitted=False,equilibrium_certified=False,
            checkpoint_selection='fixed_terminal_not_best_energy')
        result['result_sha256']=canonical_sha256(result);atomic_json(d/'results.json',result)
        print(geometry,'COMPLETE',json.dumps(comparison),flush=True);return result
    except Exception as error:
        atomic_json(d/'failure.json',dict(type=type(error).__name__,message=str(error),phase=state['phase'],
            completed_steps=len(history),active_seconds=previous_seconds+perf_counter()-start))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_signal)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--register',action='store_true');p.add_argument('--geometry',choices=('triangular','square'))
    p.add_argument('--resume',action='store_true');a=p.parse_args();spec=json.loads(a.config.read_text())
    if a.register:
        dest=ROOT/spec['output']/'registration.json'
        if dest.exists():raise FileExistsError(dest)
        r=registration(spec);atomic_json(dest,r);print(r['registration_sha256'])
    else:run(a.config,a.geometry,resume=a.resume)


if __name__=='__main__':main()
