"""Preregistered fixed-parent, equal-H nearest/global/frame-proposal comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.analytic_plaquette_carrier import aligned_plaquette_a1_local_data
from su2_nqs.benchmarks.coherent_plaquette_sampling import ordered_frame_orbit_with_weights
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.cell_singlet_initialization import singlet_bond_diagnostics
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.vmc.frame_metropolis import frame_proposal, make_frame_metropolis
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint
from .spin import _groups, _bonds
from .spin_decoder_cost import ROOT, physical_config, amplitude_functions, peak_rss_bytes
from .spin_decoder_pilot import starting_chains, summarize_phase
from .spin_vap_decoder import init_decoder

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_decoder_sampler_comparison.json'


def registration(spec: dict) -> dict:
    """Seal fixed budgets, parameters, runtime and transitive executable sources."""
    if spec['chains']!=16 or spec['microbatch']!=8 or spec['training_steps']!=0:
        raise ValueError('fixed-parent protocol requires 16 chains, microbatch 8, no training')
    if spec['snapshots']%spec['energy_every'] or spec['snapshots']//spec['energy_every']<4:
        raise ValueError('invalid energy snapshot budget')
    paths=[ROOT/spec['parent']/case['geometry']/name for case in spec['cases']
        for name in ('parameters.json','parameters.npz','values.npz','embedding.npz')]
    result=dict(config=spec,inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(),sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    result['registration_sha256']=canonical_sha256(result)
    return result


def diagnostics(record: dict, bonds: list, cells: tuple, seconds: float) -> dict:
    """Retain tails and distinguish sampler disagreement from a change of state."""
    result=summarize_phase(record)
    energy=np.asarray(record['energies']);indices=np.asarray(record['energy_indices'])-1
    states=np.asarray(record['states'],dtype=np.int32)[indices]
    bond=singlet_bond_diagnostics(states.reshape(-1,36),bonds,cells)
    diagonal=np.asarray(bond['total']).reshape(energy.shape)
    difference=(energy.real/36-diagonal).mean(axis=0)
    centered=abs(energy-energy.mean())**2
    chain_weights=centered.sum(axis=0)
    loo=[float(np.mean(abs(np.delete(energy,i,axis=1)-np.delete(energy,i,axis=1).mean())**2)/36)
        for i in range(16)]
    result.update(H_minus_3Hz_per_site=float(difference.mean()),
        H_minus_3Hz_chain_se=float(difference.std(ddof=1)/4),
        maximum_abs_local_energy_per_site=float(abs(energy/36).max()),
        largest_chain_variance_fraction=float(chain_weights.max()/max(chain_weights.sum(),1e-300)),
        leave_one_chain_out_variance_range=[min(loo),max(loo)],
        top_one_percent_variance_fraction=float(np.sort(centered.ravel())[-max(1,int(np.ceil(energy.size*.01))):].sum()/max(centered.sum(),1e-300)),
        log_target_over_Q_range=[float(np.min(np.asarray(record['log_weights'])-record['log_q'])),
            float(np.max(np.asarray(record['log_weights'])-record['log_q']))],
        energy_ess_per_second=result['diagnostics']['energy']['ess']/seconds,
        equilibrium_certified=False,variational_change=False)
    return result


def run(path: Path, geometry: str, arm: str, *, resume: bool=False) -> dict:
    """One isolated bounded process with checkpointed chain/RNG/work state."""
    spec=json.loads(path.read_text());root=ROOT/spec['output'];dest=root/geometry/arm
    reg=json.loads((root/'registration.json').read_text())
    if reg!=registration(spec):raise ValueError('registration mismatch')
    if arm not in spec['arms']:raise ValueError('unregistered arm')
    if (dest/'results.json').exists():raise FileExistsError('completed comparison arm')
    case=next(c for c in spec['cases'] if c['geometry']==geometry)
    identity=dict(registration_sha256=reg['registration_sha256'],geometry=geometry,arm=arm)
    checkpoint=dest/'checkpoint.pkl';parent=ROOT/spec['parent']/geometry
    config=physical_config(6,6,2)
    params=load_parameter_artifact(parent/'parameters.json',init_decoder(2026093003,config))
    initial_hash=pytree_sha256(params);started=perf_counter();previous=0.
    if resume:
        saved=load_resumable_checkpoint(checkpoint,expected_run_identity=identity)
        if pytree_sha256(saved['params'])!=initial_hash:raise ValueError('parent changed')
        state=saved['metadata'];previous=state['active_seconds']
    else:
        if checkpoint.exists():raise FileExistsError('use --resume')
        chains,keys=starting_chains(case['seed'],10)
        keys=np.asarray([jax.random.fold_in(jnp.asarray(k),spec['arms'].index(arm)+1) for k in keys])
        state=dict(chains=chains,keys=keys,initial_states=chains.copy(),burn_done=0,snapshot_done=0,
            full_H_centers=0,proposal_attempts=0,active_seconds=0.,kernel_counts=np.zeros((3,3),dtype=np.int64),
            record={key:[] for key in ('states','log_weights','log_q','nn','energies','energy_indices','accept_moves')},
            replay_errors=[],sampling_seconds=0.,H_seconds=0.)
    def persist(event):
        state['active_seconds']=previous+perf_counter()-started
        save_resumable_checkpoint(checkpoint,jax.device_get(params),optimizer_state=None,completed_steps=0,
            next_sample_index=state['proposal_attempts'],run_identity=identity,history=[],metadata=state)
        atomic_json(dest/'progress.json',dict(event=event,burn_done=state['burn_done'],snapshot_done=state['snapshot_done'],
            full_H_centers=state['full_H_centers'],active_seconds=state['active_seconds'],peak_rss_bytes=peak_rss_bytes()))
        if event not in ('burn','observe') or (state['burn_done']+state['snapshot_done'])%8==0:
            print(geometry,arm,event,state['burn_done'],state['snapshot_done'],round(state['active_seconds'],2),flush=True)
        if state['active_seconds']>spec['wall_limit_seconds_per_arm']:raise TimeoutError('registered time cap')
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30:raise MemoryError('registered RSS cap')
    def alarm(_s,_f):raise TimeoutError('registered time cap')
    old_signal=signal.signal(signal.SIGALRM,alarm)
    signal.alarm(max(1,int(spec['wall_limit_seconds_per_arm']-previous)))
    try:
        persist('setup')
        group=_groups(geometry,6,6)['gamma_a1'];functions=amplitude_functions(config,group)
        log_fn=lambda p,x:jnp.log(functions['a1'](p,x))
        logs=jax.jit(jax.vmap(log_fn,in_axes=(None,0)))
        bonds=_bonds(geometry,6,6,case['j2_over_j1'])
        edges=[(a,b) for a,b,J in bonds if J==1.]
        nn_edges=np.asarray(edges,dtype=np.int32)
        frames,weights=ordered_frame_orbit_with_weights(config.cells,group)
        local,_=aligned_plaquette_a1_local_data()
        with np.load(parent/'embedding.npz') as artifact:
            saved_local=artifact['local_physical']
        overlap=np.vdot(local,saved_local)
        local_error=float(np.linalg.norm(saved_local-local*overlap))
        if local_error>1e-12:raise ValueError('proposal core not aligned with frozen parent')
        log_q,draw=frame_proposal(frames,weights,abs(local)**2,uniform_fraction=spec['uniform_fraction'])
        log_q=jax.jit(log_q)
        sampler=make_frame_metropolis(log_fn,mode=arm,proposals=spec['proposal_block'],edges=edges,n_sites=36,
            log_proposal=log_q,draw_proposal=draw,frame_probability=spec['frame_probability'])
        H=make_full_heisenberg_ratios(log_fn,bonds)
        def energies(bits):
            before=perf_counter();out=[]
            for i in range(0,len(bits),8):
                value=np.asarray(H(params,jnp.asarray(bits[i:i+8])))
                state['full_H_centers']+=len(value)
                if not np.isfinite(value).all():raise FloatingPointError('nonfinite H ratio; no filtering allowed')
                out.append(value)
            state['H_seconds']+=perf_counter()-before
            return np.concatenate(out)
        def sample():
            before=perf_counter();states=[];keys=[];moves=[]
            for i in (0,1):
                x,k,counts=jax.device_get(sampler(params,jnp.asarray(state['keys'][i]),jnp.asarray(state['chains'][8*i:8*i+8])))
                states.append(x);keys.append(k);state['kernel_counts']+=counts
                moves.append([float(counts[:,j].sum()/counts[:,0].sum()) for j in (1,2)])
            state['chains']=np.concatenate(states);state['keys']=np.asarray(keys)
            state['proposal_attempts']+=16*spec['proposal_block'];state['sampling_seconds']+=perf_counter()-before
            if not np.all(state['chains'].sum(axis=1)==18):raise ValueError('m=0 violation')
            return moves
        with np.load(parent/'values.npz') as a:probes=a['states'];expected=a['a1_H_b8']
        # A resume recompilation is separately counted (never hidden in matched work).
        actual=energies(probes);error=float(np.max(abs(actual-expected)/(1+abs(expected))))
        if error>1e-9:raise ValueError('parent H replay failed')
        state['replay_errors'].append(error);persist('H_replay')
        while state['burn_done']<spec['burn_blocks']:
            sample();state['burn_done']+=1;persist('burn')
        record=state['record']
        while state['snapshot_done']<spec['snapshots']:
            moves=sample();bits=state['chains'].copy()
            amp_logs=np.concatenate([np.asarray(logs(params,jnp.asarray(bits[i:i+8]))) for i in (0,8)])
            if not np.isfinite(amp_logs).all():raise FloatingPointError('zero/nonfinite sampled amplitude')
            z=2*bits-1
            record['states'].append(bits);record['log_weights'].append(2*amp_logs.real)
            record['log_q'].append(np.asarray(log_q(jnp.asarray(bits))))
            record['nn'].append(.75*np.mean(z[:,nn_edges[:,0]]*z[:,nn_edges[:,1]],axis=1))
            record['accept_moves'].append(moves);state['snapshot_done']+=1
            if state['snapshot_done']%spec['energy_every']==0:
                record['energies'].append(energies(bits));record['energy_indices'].append(state['snapshot_done'])
            persist('observe')
        summary=diagnostics(record,bonds,config.cells,previous+perf_counter()-started)
        persist('done')
        loaded=load_resumable_checkpoint(checkpoint,expected_run_identity=identity)
        restored=loaded['metadata']
        checkpoint_exact=bool(pytree_sha256(loaded['params'])==initial_hash and
            np.array_equal(restored['chains'],state['chains']) and np.array_equal(restored['keys'],state['keys']))
        if not checkpoint_exact or pytree_sha256(params)!=initial_hash:raise ValueError('frozen-state checkpoint mismatch')
        np.savez_compressed(dest/'chains.npz',**{k:np.asarray(v) for k,v in record.items()},initial_states=state['initial_states'],
            terminal_states=state['chains'],terminal_keys=state['keys'])
        result=dict(geometry=geometry,arm=arm,registration_sha256=reg['registration_sha256'],summary=summary,
            initial_parameters_sha256=initial_hash,final_parameters_sha256=pytree_sha256(params),
            initial_states_sha256=canonical_sha256(state['initial_states'].tolist()),parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(params)),
            spatial_images=len(group),ordered_proposal_frames=len(frames),frame_weights=weights.tolist(),local_core_error=local_error,
            full_H_centers=state['full_H_centers'],measurement_H_centers=np.asarray(record['energies']).size,
            replay_errors=state['replay_errors'],proposal_attempts=state['proposal_attempts'],kernel_counts=state['kernel_counts'].tolist(),
            sampler_target_amplitude_centers=state['proposal_attempts']+16*(state['burn_done']+state['snapshot_done']),
            snapshot_target_amplitude_centers=16*state['snapshot_done'],sampling_seconds=state['sampling_seconds'],H_seconds=state['H_seconds'],
            active_seconds=previous+perf_counter()-started,peak_rss_bytes=peak_rss_bytes(),checkpoint_exact=checkpoint_exact,
            raw_sha256=file_sha256(dest/'chains.npz'),checkpoint_sha256=file_sha256(checkpoint),
            training_steps=0,new_ED=False,remote_submitted=False,equilibrium_certified=False)
        result['result_sha256']=canonical_sha256(result);atomic_json(dest/'results.json',result)
        print('COMPLETE',geometry,arm,json.dumps(summary),flush=True)
        return result
    except Exception as error:
        atomic_json(dest/'failure.json',dict(type=type(error).__name__,message=str(error),active_seconds=previous+perf_counter()-started))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_signal)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--register',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--geometry',choices=('triangular','square'));parser.add_argument('--arm',choices=('nearest','global','frame_mix'))
    args=parser.parse_args();spec=json.loads(args.config.read_text())
    if args.register:
        dest=ROOT/spec['output']/'registration.json'
        if dest.exists():raise FileExistsError(dest)
        value=registration(spec);atomic_json(dest,value);print(value['registration_sha256'])
    else:run(args.config,args.geometry,args.arm,resume=args.resume)


if __name__=='__main__':main()
