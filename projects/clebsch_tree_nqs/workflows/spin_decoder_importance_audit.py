"""Bounded IID core/H-connected moment audit, with the physical parent fixed."""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.analytic_plaquette_carrier import aligned_plaquette_a1_local_data
from su2_nqs.benchmarks.coherent_plaquette_sampling import ordered_frame_orbit_with_weights
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest,numerical_runtime_identity
from su2_nqs.vmc.frame_metropolis import frame_proposal
from su2_nqs.vmc.exchange_bridge_importance import make_exchange_bridge,make_amplitude_heisenberg_action,importance_audit
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint,load_resumable_checkpoint
from .spin import _groups,_bonds
from .spin_decoder_cost import ROOT,physical_config,amplitude_functions,peak_rss_bytes
from .spin_vap_decoder import init_decoder

jax.config.update('jax_enable_x64',True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_decoder_importance_audit.json'


def registration(spec: dict) -> dict:
    """Seal the finite protocol, inherited state, numeric sources and runtime."""
    if (spec['independent_groups'],spec['samples_per_group'],spec['microbatch'],spec['training_steps'])!=(16,32,8,0):
        raise ValueError('this bounded experiment requires16x32 samples, microbatch8 and no training')
    paths=[ROOT/spec['parent']/case['geometry']/name for case in spec['cases']
        for name in ('parameters.json','parameters.npz','values.npz','embedding.npz')]
    r=dict(config=spec,inputs={str(p.relative_to(ROOT)):file_sha256(p)for p in paths},
        runtime=numerical_runtime_identity(),sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    r['registration_sha256']=canonical_sha256(r);return r


def support_decomposition(data: dict, spec: dict) -> dict:
    """Partition weighted moments by bit support, NOT by a spin-q sector."""
    floor=np.log(spec['base_uniform_fraction']/comb(36,18));tol=spec['support_log_floor_tolerance']
    core=np.asarray(data['log_base'])>floor+tol
    pushed=np.asarray(data['log_pushed'])>floor+tol
    psi=np.asarray(data['psi']);hp=np.asarray(data['hpsi']);q=np.exp(data['log_proposal'])
    z=abs(psi)**2/q;e=psi.conj()*hp/q;m=abs(hp)**2/q;E=e.sum()/z.sum()
    residual=abs(hp-E*psi)**2/q
    out={}
    for name,mask in [('core',core),('H_only',~core&pushed),('outside_both',~core&~pushed)]:
        out[name]=dict(sample_count=int(mask.sum()),normalization_fraction=float(z[mask].sum()/z.sum()),
            energy_per_site_contribution=float(e[mask].real.sum()/z.sum()/36),
            imaginary_energy_per_site_contribution=float(e[mask].imag.sum()/z.sum()/36),
            second_moment_fraction=float(m[mask].sum()/m.sum()),
            residual_variance_per_site_contribution=float(residual[mask].sum()/z.sum()/36),
            residual_fraction=float(residual[mask].sum()/residual.sum()))
    return out


def run(path: Path, geometry: str, *, resume: bool=False) -> dict:
    """Serial CPU audit with indexed independent groups and per-microbatch resume."""
    spec=json.loads(path.read_text());root=ROOT/spec['output'];dest=root/geometry
    reg=json.loads((root/'registration.json').read_text())
    if reg!=registration(spec):raise ValueError('registration mismatch')
    if (dest/'results.json').exists():raise FileExistsError('completed audit')
    case=next(c for c in spec['cases']if c['geometry']==geometry)
    identity=dict(registration_sha256=reg['registration_sha256'],geometry=geometry)
    config=physical_config(6,6,2);parent=ROOT/spec['parent']/geometry
    params=load_parameter_artifact(parent/'parameters.json',init_decoder(2026093003,config))
    parent_hash=pytree_sha256(params);checkpoint=dest/'checkpoint.pkl';start=perf_counter();previous=0.
    names=('states','used_bridge','groups','log_proposal','log_base','log_pushed','psi','hpsi')
    if resume:
        saved=load_resumable_checkpoint(checkpoint,expected_run_identity=identity)
        if pytree_sha256(saved['params'])!=parent_hash:raise ValueError('parent changed')
        state=saved['metadata'];previous=state['active_seconds']
    else:
        if checkpoint.exists():raise FileExistsError('use --resume')
        state=dict(group_index=0,offset=0,pending=None,record={k:[]for k in names},full_H_centers=0,
            measurement_H_centers=0,proposal_draws=0,proposal_density_centers=0,replay_errors=[],
            active_seconds=0.,H_seconds=0.,proposal_seconds=0.)
    def persist(event):
        state['active_seconds']=previous+perf_counter()-start
        save_resumable_checkpoint(checkpoint,jax.device_get(params),optimizer_state=None,completed_steps=0,
            next_sample_index=state['measurement_H_centers'],run_identity=identity,history=[],metadata=state)
        atomic_json(dest/'progress.json',dict(event=event,group=state['group_index'],offset=state['offset'],
            full_H_centers=state['full_H_centers'],active_seconds=state['active_seconds'],peak_rss_bytes=peak_rss_bytes()))
        if event!='microbatch' or state['offset']==0:
            print(geometry,event,'group',state['group_index'],round(state['active_seconds'],2),'s',flush=True)
        if state['active_seconds']>spec['wall_limit_seconds_per_geometry']:raise TimeoutError('registered wall cap')
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30:raise MemoryError('registered RSS cap')
    def alarm(_s,_f):raise TimeoutError('registered wall cap')
    old_signal=signal.signal(signal.SIGALRM,alarm)
    signal.alarm(max(1,int(spec['wall_limit_seconds_per_geometry']-previous)))
    try:
        persist('setup')
        group=_groups(geometry,6,6)['gamma_a1'];fn=amplitude_functions(config,group)
        bonds=_bonds(geometry,6,6,case['j2_over_j1']);edges=[(a,b)for a,b,_ in bonds]
        p=np.abs([J for _,_,J in bonds]);p=p/p.sum()
        frames,weights=ordered_frame_orbit_with_weights(config.cells,group);local,_=aligned_plaquette_a1_local_data()
        with np.load(parent/'embedding.npz')as a:saved_local=a['local_physical']
        local_error=float(np.linalg.norm(saved_local-local*np.vdot(local,saved_local)))
        if local_error>1e-12:raise ValueError('proposal core not aligned with fixed parent')
        local_p=abs(local)**2;minimum_core=float(local_p[local_p>0].min()**len(config.cells)*weights[weights>0].min())
        floor=spec['base_uniform_fraction']/comb(36,18)
        support_gap=float(np.log1p((1-spec['base_uniform_fraction'])*minimum_core*p[p>0].min()/floor))
        if support_gap<100*spec['support_log_floor_tolerance']:raise ValueError('support threshold cannot resolve smallest positive component')
        log_base,draw_base=frame_proposal(frames,weights,abs(local)**2,uniform_fraction=spec['base_uniform_fraction'])
        density,draw=make_exchange_bridge(log_base,draw_base,edges,p,n_sites=36,bridge_fraction=spec['bridge_fraction'])
        draw_group=jax.jit(lambda key:draw(key,spec['samples_per_group']))
        H=make_amplitude_heisenberg_action(fn['a1'],bonds,n_sites=36)
        def action(bits):
            before=perf_counter();psi,hp=jax.device_get(H(params,jnp.asarray(bits)))
            state['H_seconds']+=perf_counter()-before;state['full_H_centers']+=len(bits)
            if not(np.isfinite(psi).all()and np.isfinite(hp).all()):raise FloatingPointError('nonfinite action, no filtering permitted')
            return psi,hp
        with np.load(parent/'values.npz')as a:probes=a['states'];expected=a['a1_H_b8']
        psi,hp=action(probes)
        if np.any(psi==0):raise ValueError('saved replay probes unexpectedly nodal')
        error=float(np.max(abs(hp/psi-expected)/(1+abs(expected))))
        if error>1e-9:raise ValueError('direct amplitude action failed original-H replay')
        state['replay_errors'].append(error);persist('H_replay')
        while state['group_index']<spec['independent_groups']:
            if state['pending'] is None:
                before=perf_counter();key=jax.random.fold_in(jax.random.PRNGKey(case['seed']),state['group_index'])
                bits,used=jax.device_get(draw_group(key));state['proposal_seconds']+=perf_counter()-before
                if not np.all(bits.sum(axis=1)==18):raise ValueError('m=0 violation')
                state['pending']=dict(states=bits,used_bridge=used,key=np.asarray(key))
                state['proposal_draws']+=len(bits);persist('draw_group')
            lo=state['offset'];hi=lo+spec['microbatch'];bits=state['pending']['states'][lo:hi]
            before=perf_counter();lq,lb,lp=jax.device_get(density(jnp.asarray(bits)))
            state['proposal_seconds']+=perf_counter()-before;state['proposal_density_centers']+=len(bits)
            if not np.isfinite(lq).all():raise FloatingPointError('nonfinite proposal density')
            psi,hp=action(bits);state['measurement_H_centers']+=len(bits)
            values=(bits,state['pending']['used_bridge'][lo:hi],np.full(len(bits),state['group_index'],dtype=np.int32),lq,lb,lp,psi,hp)
            for name,value in zip(names,values):state['record'][name].append(np.asarray(value))
            state['offset']=hi
            if hi==spec['samples_per_group']:
                state['pending']=None;state['offset']=0;state['group_index']+=1
            persist('microbatch')
        data={k:np.concatenate(v)for k,v in state['record'].items()}
        audit=importance_audit(data['psi'],data['hpsi'],np.exp(data['log_proposal']),data['groups'],n_sites=36,
            bootstrap_replicates=spec['bootstrap_replicates'],seed=case['seed']+100)
        support=support_decomposition(data,spec)
        persist('done');restored=load_resumable_checkpoint(checkpoint,expected_run_identity=identity)
        metadata=restored['metadata']
        replay=bool(pytree_sha256(restored['params'])==parent_hash and metadata['measurement_H_centers']==512
            and metadata['group_index']==16 and metadata['offset']==0 and metadata['pending'] is None
            and all(np.array_equal(np.concatenate(metadata['record'][k]),v)for k,v in data.items()))
        if not replay or pytree_sha256(params)!=parent_hash:raise ValueError('frozen checkpoint mismatch')
        np.savez_compressed(dest/'samples.npz',**data)
        result=dict(geometry=geometry,case=case,registration_sha256=reg['registration_sha256'],audit=audit,support=support,
            initial_parameters_sha256=parent_hash,final_parameters_sha256=pytree_sha256(params),checkpoint_exact=replay,
            parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(params)),spatial_images=len(group),bond_count=len(bonds),
            ordered_frames=len(frames),full_H_centers=state['full_H_centers'],measurement_H_centers=state['measurement_H_centers'],
            local_core_error=local_error,minimum_positive_support_log_gap=support_gap,
            target_amplitude_centers_in_H=state['full_H_centers']*(len(bonds)+1),
            cheap_base_density_centers=state['proposal_density_centers']*(len(bonds)+1),proposal_draws=state['proposal_draws'],
            H_seconds=state['H_seconds'],proposal_seconds=state['proposal_seconds'],replay_errors=state['replay_errors'],
            active_seconds=previous+perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),
            raw_sha256=file_sha256(dest/'samples.npz'),checkpoint_sha256=file_sha256(checkpoint),
            training_steps=0,new_ED=False,remote_submitted=False,parent_is_ViT=False,reliability_certified=False)
        result['result_sha256']=canonical_sha256(result);atomic_json(dest/'results.json',result)
        print('COMPLETE',geometry,json.dumps(audit),json.dumps(support),flush=True);return result
    except Exception as error:
        atomic_json(dest/'failure.json',dict(type=type(error).__name__,message=str(error),active_seconds=previous+perf_counter()-start))
        raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old_signal)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--register',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--geometry',choices=('triangular','square'));a=parser.parse_args();spec=json.loads(a.config.read_text())
    if a.register:
        dest=ROOT/spec['output']/'registration.json'
        if dest.exists():raise FileExistsError(dest)
        r=registration(spec);atomic_json(dest,r);print(r['registration_sha256'])
    else:run(a.config,a.geometry,resume=a.resume)


if __name__=='__main__':main()
