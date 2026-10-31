"""Untouched-chain energy and variance audit of a frozen physical ClebschTree.

Use independent exchange-chain keys and the complete coherent A1 Hamiltonian.
No parameter update, endpoint selection, or exact diagonalization occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,numerical_runtime_identity,transitive_local_source_manifest
from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
from su2_nqs.models.clebsch_tree_orbit_dag import make_clebsch_execution
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains,run_exchange_sweeps
from .spin import _bonds
from .spin_clebsch_tree_sampled import ROOT,components

jax.config.update('jax_enable_x64',True)


def summarize_chains(values: np.ndarray,n_sites: int,*,bootstrap_seed: int) -> dict:
    """Estimate E and Var(H) from independent streams of correlated chains."""
    v=np.asarray(values,dtype=np.float64)
    if v.ndim != 3 or min(v.shape) < 2 or not np.isfinite(v).all() or n_sites < 1:
        raise ValueError('expected finite (streams,snapshots,chains) and positive sites')
    paths=v.transpose(0,2,1).reshape(-1,v.shape[1]); means=paths.mean(axis=1)
    energy=float(means.mean()); centered=paths-energy
    second=np.mean(paths**2,axis=1)
    influence=np.mean(centered**2,axis=1)
    draws=np.random.default_rng(bootstrap_seed).integers(len(paths),size=(2000,len(paths)))
    boot_mean=means[draws].mean(axis=1)
    boot_variance=(second[draws].mean(axis=1)-boot_mean**2)/n_sites
    half=paths.shape[1]//2
    if half >= 2:
        split=np.concatenate((paths[:,:half],paths[:,-half:]),axis=0)
        within=np.var(split,axis=1,ddof=1).mean()
        between=half*np.var(split.mean(axis=1),ddof=1)
        rhat=float(np.sqrt(((half-1)*within+between)/(half*within))) if within>0 else None
    else:
        rhat=None
    tails=np.sort(centered.ravel()**2); count=max(1,int(np.ceil(.01*len(tails))))
    return dict(energy_per_site=energy/n_sites,
        energy_chain_SE=float(means.std(ddof=1)/np.sqrt(len(paths))/n_sites),
        energy_chain_bootstrap_95=np.quantile(boot_mean/n_sites,[.025,.975]).tolist(),
        variance_per_site=float(np.mean(centered**2)/n_sites),
        variance_chain_SE=float(influence.std(ddof=1)/np.sqrt(len(paths))/n_sites),
        variance_chain_bootstrap_95=np.quantile(boot_variance,[.025,.975]).tolist(),
        stream_energies_per_site=(v.mean(axis=(1,2))/n_sites).tolist(),
        stream_variances_per_site=[float(np.var(stream)/n_sites) for stream in v],
        split_chain_energy_rhat=rhat,
        top_one_percent_variance_share=float(tails[-count:].sum()/tails.sum()) if tails.sum()>0 else 0.,
        records=int(v.size),independent_chains=int(len(paths)),mixing_certified=False)


def audit(spec: dict,output: Path) -> dict:
    """Evaluate a fixed endpoint on entirely new native physical chains."""
    if spec['require_gpu'] and jax.default_backend() != 'gpu': raise RuntimeError('GPU required')
    if output.exists(): raise FileExistsError('audit output already exists')
    train_dir=ROOT/Path(spec['training_directory'])
    registration=json.loads((train_dir/'registration.json').read_text())
    quality=spec.get('quality_checkpoint',False)
    if not isinstance(quality,bool): raise ValueError('quality checkpoint selection must be boolean')
    training=json.loads((train_dir/'training.json').read_text())
    checkpoint_path=(train_dir/'quality'/f'step{spec["endpoint_step"]:05d}.pkl'
                     if quality else train_dir/'checkpoint.pkl')
    checkpoint=load_checkpoint(checkpoint_path); meta=checkpoint['metadata']
    if (registration['sha256'] != canonical_sha256({k:v for k,v in registration.items() if k!='sha256'})
            or training['registration_sha256'] != registration['sha256']
            or (training['completed_steps'] < spec['endpoint_step'] if quality
                else training['completed_steps'] != spec['endpoint_step'])
            or meta['completed_steps'] != spec['endpoint_step']
            or meta['registration_sha256'] != registration['sha256']
            or len(meta['history']) != spec['endpoint_step']
            or (not quality and training['checkpoint_sha256'] != file_sha256(checkpoint_path))
            or meta['params_sha256'] != pytree_sha256(checkpoint['params'])
            or training['status'] not in ('safe_pause','fixed_steps_complete')):
        raise ValueError('frozen training endpoint not sealed at requested step')
    for name,digest in registration['source_manifest'].items():
        if file_sha256(ROOT/name) != digest: raise ValueError('trained numerical source changed: '+name)
    physical=registration['config']
    cfg,_,_,bonds,images,order=components(physical)
    execution=physical.get('execution',{})
    kernel=execution.get('kernel','legacy')
    orbit_chunk=execution.get('projection_chunk',physical['projection_chunk'])
    if kernel not in ('legacy','orbit_dag'):
        raise ValueError('unsupported frozen training execution kernel')
    if kernel == 'orbit_dag':
        projected,projection=make_clebsch_execution(cfg,order,images,
            orbit_chunk=orbit_chunk,kernel=kernel)
    else:
        projected,projection=make_projected_clebsch(cfg,order,images,orbit_chunk=orbit_chunk,
            fold_root=physical.get('fold_root_swap',False))
    if physical.get('fold_root_swap',False) and not projection['root_fold']:
        raise ValueError('registered root fold is not a verified physical symmetry')
    amplitude=lambda p,row:projected(p,row[None])[0]
    nearest=[(a,b) for a,b,_ in _bonds(physical['geometry'],physical['rows'],physical['columns'],0.)]
    action=make_amplitude_heisenberg_action(amplitude,bonds,n_sites=cfg.tree.n_leaves)
    chains=spec['chains_per_stream']; snapshots=spec['snapshots_per_stream']; chunk=spec['H_sample_chunk']
    if chains<2 or snapshots<2 or chains*snapshots%chunk:
        raise ValueError('audit needs >=2 chains/snapshots and divisible H chunks')
    def sweep(p,key,states,proposals):
        return run_exchange_sweeps(p,key,states,amplitude_fn=amplitude,particle_kind='spin',
                                   sweeps=proposals,exchange_edges=nearest)
    sweep=jax.jit(sweep,static_argnames=('proposals',))
    @jax.jit
    def batch_H(params,rows):
        psi,hpsi=jax.lax.map(lambda block:action(params,block),rows.reshape(-1,chunk,cfg.tree.n_leaves))
        return psi.reshape(-1),hpsi.reshape(-1)
    params=checkpoint['params']; streams=[]; accepts=[]
    replay=None
    if kernel == 'orbit_dag':
        # Independent legacy amplitude/H calculation on the actual frozen
        # endpoint. All old source hashes were checked above; choosing the
        # faster execution kernel cannot bypass that source integrity gate.
        legacy,_=make_projected_clebsch(cfg,order,images,orbit_chunk=orbit_chunk)
        reference=make_amplitude_heisenberg_action(
            lambda p,row:legacy(p,row[None])[0],bonds,n_sites=cfg.tree.n_leaves)
        centers=jnp.asarray(meta['sampler_states'])[:2]
        old_psi,old_h=jax.block_until_ready(reference(params,centers))
        new_psi,new_h=jax.block_until_ready(action(params,centers))
        def relative(a,b):
            return float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(a),1e-25))
        replay=dict(amplitude_relative_error=relative(old_psi,new_psi),
                    H_product_relative_error=relative(old_h,new_h),
                    H_ratio_max_absolute_error=float(jnp.max(jnp.abs(old_h/old_psi-new_h/new_psi))),
                    independent_legacy_backend=True,actual_frozen_endpoint=True)
        if max(replay[name] for name in ('amplitude_relative_error','H_product_relative_error')) > 2e-11:
            raise ArithmeticError('frozen audit execution changed amplitude/H action')
    for stream in range(spec['streams']):
        key=jax.random.PRNGKey(spec['audit_seeds'][stream])
        states=initialize_spin_exchange_chains(key,n_chains=chains,n_sites=cfg.tree.n_leaves,
                                                n_up=cfg.tree.n_leaves//2)
        states,_,key,_=sweep(params,key,states,spec['burn_proposals'])
        rows=[]; acceptance=[]
        for _ in range(snapshots):
            states,amps,key,diag=sweep(params,key,states,spec['proposals_between_snapshots'])
            if np.any(np.asarray(amps)==0): raise FloatingPointError('audit chain has an exact node')
            rows.append(states); acceptance.append(float(diag['conditional_move_acceptance']))
        batch=jnp.stack(rows).reshape(-1,cfg.tree.n_leaves)
        psi,hpsi=batch_H(params,batch)
        ratios=np.asarray(hpsi/psi).reshape(snapshots,chains)
        if not np.isfinite(ratios).all(): raise FloatingPointError('nonfinite audit full-H ratio')
        streams.append(ratios); accepts.append(float(np.mean(acceptance)))
    data=np.asarray(streams)
    stats=summarize_chains(data,cfg.tree.n_leaves,bootstrap_seed=spec['bootstrap_seed'])
    target=(None if physical.get('literature_target_energy_total') is None else
            float(physical['literature_target_energy_total'])/cfg.tree.n_leaves)
    target_se=(None if physical.get('literature_target_total_standard_error') is None else
               float(physical['literature_target_total_standard_error'])/cfg.tree.n_leaves)
    stats['literature_energy_per_site']=target
    stats['literature_energy_SE_per_site']=target_se
    stats['energy_minus_literature_per_site']=(None if target is None else stats['energy_per_site']-target)
    result=dict(status='complete',method='physical ClebschTree exact S0/coherent A1',
                checkpoint_sha256=file_sha256(checkpoint_path),training_registration_sha256=registration['sha256'],
                training_steps=spec['endpoint_step'],statistics=stats,conditional_move_acceptance=accepts,
                audit_config=spec,projected_images=projection['evaluation_images'],
                source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                    package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
                runtime=numerical_runtime_identity(),independent_seeds_from_training=True,
                fixed_endpoint_no_selection=True,execution_kernel=kernel,
                independent_execution_replay=replay)
    output.mkdir(parents=True); tmp=output/'local_energies.tmp.npz'
    np.savez_compressed(tmp,local_energies=data)
    os.replace(tmp,output/'local_energies.npz')
    atomic_json(output/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(args.config.read_text())
    print(json.dumps(audit(spec,ROOT/spec['output'])['statistics'],indent=2),flush=True)
