"""Original-style physical-input ClebschTree large-size cost/sampling gate.

Reuse reachable layer-shared reduced matrices, coherent full A1, physical
exchange MCMC and division-free full-H products. There is no 64-bit support
encoding, global reduced-basis conversion, AGP carrier or new response head.
This gate measures whether the subsequent square8 stochastic reproduction is
affordable; random initialization is NOT an optimized energy benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.literature_clebsch_tree import (
    LiteratureClebschTreeConfig, init_literature_clebsch_tree, make_clebsch_amplitudes,
)
from su2_nqs.models.clebsch_tree_transfer import (
    transfer_clebsch_parameters, calibrate_clebsch_parameters,
)
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains, run_exchange_sweeps
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest,numerical_runtime_identity,file_sha256
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from .spin import _groups,_bonds

ROOT=Path(__file__).resolve().parents[3]
jax.config.update('jax_enable_x64',True)


def validate_initialization_coupling(spec: dict, physical: dict, transfer: dict) -> None:
    """Allow a declared cross-H warm start, never reuse the old H certificate.

    Only the reduced parameters are transferred. This cost gate recomputes
    the target Hamiltonian action, gradients and sampler; the subsequent
    trainer starts fresh optimizer/RNG/chains from this new gate.
    """
    if physical['geometry'] != spec['geometry']:
        raise ValueError('initialization must retain lattice geometry')
    if physical['j2_over_j1'] == spec['j2_over_j1']:
        if transfer.get('coupling_warm_start') is not None:
            raise ValueError('cross-coupling declaration requires different Hamiltonians')
        return
    declaration = transfer.get('coupling_warm_start')
    expected = dict(parent_j2_over_j1=physical['j2_over_j1'],
                    target_j2_over_j1=spec['j2_over_j1'], reset_optimizer_sampler_rng=True)
    if declaration != expected:
        raise ValueError('different coupling requires an explicit reset-only warm start')


def components(spec: dict) -> tuple:
    """Compile the SAME exact-spin physical model and full coherent projector."""
    tree,order=square_geometry_tree(spec['rows'],spec['columns'])
    cfg=LiteratureClebschTreeConfig(tree,multiplicity_cap=spec['multiplicity_cap'],maximum_q=spec['maximum_q'],
                                  high_spin_cap=spec.get('high_spin_cap'))
    raw=make_clebsch_amplitudes(cfg)
    images=np.asarray(_groups(spec['geometry'],spec['rows'],spec['columns'])['gamma_a1'],dtype=np.int32)
    chunk=spec['projection_chunk']
    if len(images)%chunk:
        raise ValueError('exact projection chunk must divide complete group order')
    images_jax=jnp.asarray(images).reshape(-1,chunk,tree.n_leaves)
    ordered=jnp.asarray(order)
    def amplitude(params,bits):
        return jax.lax.map(lambda maps:jnp.sum(raw(params,bits[maps][:,ordered])),images_jax).sum()/len(images)
    bonds=_bonds(spec['geometry'],spec['rows'],spec['columns'],spec['j2_over_j1'])
    return cfg,raw,amplitude,bonds,images,order


def cost(spec: dict,output: Path) -> dict:
    """Time actual projected amplitude/H/gradient/MCMC and verify exact checks."""
    if spec['geometry'] != 'square' or spec['rows'] != spec['columns']:
        raise ValueError('first literature replication gate is square, not triangular')
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('registered A100 cost test requires GPU')
    if (output/'results.json').exists(): raise FileExistsError('do not overwrite completed gate')
    cfg,raw,amp,bonds,images,order=components(spec)
    output.mkdir(parents=True,exist_ok=True)
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                   package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    reg=dict(config=spec,source_manifest=sources,runtime=numerical_runtime_identity())
    reg['sha256']=canonical_sha256(reg); atomic_json(output/'registration.json',reg)
    rng=np.random.default_rng(spec['calibration_seed'])
    n=cfg.tree.n_leaves
    calibration=np.asarray([rng.permutation([0]*(n//2)+[1]*(n//2)) for _ in range(1000)],dtype=np.int32)
    p,initialization=init_literature_clebsch_tree(jax.random.PRNGKey(spec['seed']),cfg,calibration[:,order])
    transfer=spec.get('initialization_parent')
    if transfer is not None:
        source_dir=ROOT/Path(transfer['directory'])
        source_reg=json.loads((source_dir/'registration.json').read_text())
        source_path=source_dir/transfer.get('checkpoint_path','checkpoint.pkl')
        source=load_checkpoint(source_path)
        physical=source_reg['config']
        if (source_reg['sha256'] != canonical_sha256({k:v for k,v in source_reg.items() if k!='sha256'})
                or source_reg['sha256'] != transfer['registration_sha256']
                or file_sha256(source_path) != transfer['checkpoint_sha256']
                or source['metadata']['registration_sha256'] != source_reg['sha256']
                or pytree_sha256(source['params']) != transfer['params_sha256']):
            raise ValueError('immutable small-system initialization parent changed')
        validate_initialization_coupling(spec, physical, transfer)
        source_tree,_=square_geometry_tree(physical['rows'],physical['columns'])
        source_cfg=LiteratureClebschTreeConfig(source_tree,
            multiplicity_cap=physical['multiplicity_cap'], maximum_q=physical['maximum_q'],
            high_spin_cap=physical.get('high_spin_cap'))
        p,report=transfer_clebsch_parameters(source_cfg,source['params'],cfg,p,mode='initialization')
        p,recalibration=calibrate_clebsch_parameters(cfg,p,calibration[:,order])
        initialization.update(initialization_parent=transfer,reduced_bank_transfer=report,
                              transferred_ray_recalibration=recalibration,
                              small_projected_physical_state_is_not_embedded=True,
                              parent_H_certificate_not_reused=True,
                              optimizer_sampler_rng_reset=True)
    atomic_json(output/'initialization.json',initialization)
    states=calibration[:2]
    times={}; validation={}
    def measure(name,function,args):
        t=perf_counter(); f=jax.jit(function); first=jax.block_until_ready(f(*args)); cold=perf_counter()-t
        runs=[]
        for _ in range(2):
            t=perf_counter(); value=jax.block_until_ready(f(*args)); runs.append(perf_counter()-t)
        times[name]=dict(cold_seconds=cold,warm_seconds=runs)
        atomic_json(output/'progress.json',dict(times=times,validation=validation))
        print(name,times[name],flush=True)
        return first
    physical=jax.vmap(amp,in_axes=(None,0))
    a=np.asarray(measure('complete_A1_amplitudes',physical,(p,jnp.asarray(states))))
    if np.any(a == 0) or not np.isfinite(a).all(): raise FloatingPointError('initial gate centers have nodes')
    # Ward identity for S+ on random M=1 bras, no Hilbert enumeration.
    ward=[]
    for _ in range(4):
        bra=rng.permutation([0]*(n//2-1)+[1]*(n//2+1))
        lowering=[]
        for i in np.flatnonzero(bra):
            row=bra.copy(); row[i]=0; lowering.append(row)
        values=np.asarray(jax.jit(raw)(p,jnp.asarray(np.asarray(lowering)[:,order])))
        ward.append(abs(values.sum())/max(np.linalg.norm(values),1e-25))
    validation['raw_S_plus_relative_max']=max(ward)
    transformed=states[0][images[7]]
    other=float(jax.jit(amp)(p,jnp.asarray(transformed)))
    validation['complete_A1_relative_error']=abs(other-a[0])/max(abs(a[0]),1e-25)
    hfunction=make_amplitude_heisenberg_action(amp,bonds,n_sites=n)
    psi,hpsi=measure('complete_H_products',hfunction,(p,jnp.asarray(states)))
    graph=SzHeisenbergGraph(n,tuple(bonds),n_up=n//2)
    connected,elements=graph.get_conn(tuple(states[0]))
    explicit=np.dot(np.asarray(elements),np.asarray(jax.jit(physical)(p,jnp.asarray(connected))))
    validation['H_relative_error']=float(abs(explicit-hpsi[0])/max(abs(explicit),abs(hpsi[0]),1e-25))
    def loss(params,x): return jnp.mean(jnp.log(jnp.abs(physical(params,x))))
    gradients=measure('full_projection_gradient',jax.grad(loss),(p,jnp.asarray(states)))
    validation['gradient_finite']=all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))
    direction=jax.tree.map(lambda v:jnp.ones_like(v)/np.sqrt(initialization['parameter_count']),p)
    _,derivative=jax.jvp(lambda z:loss(z,jnp.asarray(states)),(p,),(direction,))
    plus=jax.tree.map(lambda v,d:v+1e-5*d,p,direction)
    minus=jax.tree.map(lambda v,d:v-1e-5*d,p,direction)
    fd=(loss(plus,jnp.asarray(states))-loss(minus,jnp.asarray(states)))/2e-5
    validation['gradient_finite_difference_error']=float(abs(fd-derivative)/max(1.,abs(float(derivative))))
    key=jax.random.PRNGKey(spec['sampling_seed'])
    chains=initialize_spin_exchange_chains(key,n_chains=8,n_sites=n,n_up=n//2)
    nearest=[(a,b) for a,b,_ in _bonds(spec['geometry'],spec['rows'],spec['columns'],0.)]
    def sampler(params,key,states):
        return run_exchange_sweeps(params,key,states,amplitude_fn=amp,particle_kind='spin',
                                      sweeps=16,exchange_edges=nearest)
    sampled=measure('nearest_exchange_16_proposals',sampler,(p,key,chains))
    frozen=output/'checkpoint.pkl'
    save_checkpoint(frozen,p,optimizer_state=dict(step=0),metadata=dict(rng=np.asarray(sampled[2]),
                    sampler_states=np.asarray(sampled[0]),registration_sha256=reg['sha256']))
    saved=load_checkpoint(frozen)
    validation['checkpoint_params_exact']=pytree_sha256(saved['params']) == pytree_sha256(p)
    continued=jax.jit(sampler)(p,sampled[2],sampled[0])
    replay=jax.jit(sampler)(saved['params'],jnp.asarray(saved['metadata']['rng']),jnp.asarray(saved['metadata']['sampler_states']))
    validation['sampler_state_replay_exact']=np.array_equal(np.asarray(replay[0]),np.asarray(continued[0]))
    validation['sampler_rng_replay_exact']=np.array_equal(np.asarray(replay[2]),np.asarray(continued[2]))
    passed=(validation['raw_S_plus_relative_max'] < 2e-11 and validation['complete_A1_relative_error'] < 2e-11
            and validation['H_relative_error'] < 2e-11 and validation['gradient_finite_difference_error'] < 2e-6
            and all(validation[k] for k in
                ('gradient_finite','checkpoint_params_exact','sampler_state_replay_exact','sampler_rng_replay_exact')))
    result=dict(status='passed' if passed else 'failed',config=spec,parameter_count=initialization['parameter_count'],
        group_order=len(images),validation=validation,times=times,training_steps=0,energy_accuracy_claim=False,
        checkpoint_sha256=file_sha256(frozen),optimizer_is_not_trained=True,registration_sha256=reg['sha256'],
        subsequent_stochastic_reproduction_requires_measured_training_budget=True)
    atomic_json(output/'results.json',result)
    if not passed: raise ArithmeticError('exact physical gate failed')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); spec=json.loads(args.config.read_text())
    print(json.dumps(cost(spec,args.output or ROOT/spec['output']),indent=2))
