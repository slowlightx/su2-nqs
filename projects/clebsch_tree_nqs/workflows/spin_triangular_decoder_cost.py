"""Bounded same-cover triangular ClebschTree binary/free/C3 cost comparison.

No optimization or random-state energy claim. All arms begin at one common
physical state and use the same effective-bank complete-A1/H kernel.
This is a separate tensor-tree workflow, not a Spatial-ViT decoder or hybrid.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest,numerical_runtime_identity
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank,
    make_mixed_clebsch_amplitude,make_mixed_projected_amplitude,mixed_parameter_count,
)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains,run_exchange_sweeps
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint

ROOT=Path(__file__).resolve().parents[3]
jax.config.update('jax_enable_x64',True)


def run(spec: dict, output: Path) -> dict:
    """Time complete operations and verify the actual physical state/ratios."""
    if spec.get('require_gpu') and jax.default_backend()!='gpu':
        raise RuntimeError('registered A100 cost test requires an allocated GPU')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('preserve completed/partial cost evidence in a fresh output')
    cfg=MixedClebschConfig(**spec['model'])
    tree=triangular_schur_hierarchy(cfg.length)
    output.mkdir(parents=True)
    # vmc.__init__ eagerly binds this historic function/submodule collision
    # through import_module, which an AST-only import scan cannot discover.
    sources=transitive_local_source_manifest((Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'),root=ROOT,
                     package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    reg=dict(config=spec,sources=sources,runtime=numerical_runtime_identity(),
             physical_state_matched=True,new_training_steps=0,
             compile_cache='Shared effective-bank kernels; compare warm times, not first-call order.')
    reg['sha256']=canonical_sha256(reg);atomic_json(output/'registration.json',reg)
    for name in sources:
        path=output/'source_snapshots'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    family=init_mixed_clebsch_family(jax.random.PRNGKey(spec['seed']),cfg)
    images=np.asarray(triangular_hnf_gamma_a1_group(((cfg.length,0),(0,cfg.length))),np.int32)
    raw=make_mixed_clebsch_amplitude(cfg)
    projected=make_mixed_projected_amplitude(cfg,images,chunk=spec['projection_chunk'])
    evaluate=jax.jit(jax.vmap(projected,in_axes=(None,0)))
    raw_evaluate=jax.jit(jax.vmap(raw,in_axes=(None,0)))
    bonds=triangular_j1j2_bonds(cfg.length,cfg.length,j2=spec['j2_over_j1'],periodic=True)
    full_h=make_amplitude_heisenberg_action(projected,bonds,n_sites=tree.n_sites)
    rng=np.random.default_rng(spec['calibration_seed'])
    candidates=np.asarray([rng.permutation([0]*(tree.n_sites//2)+[1]*(tree.n_sites//2))
                           for _ in range(8)],np.int32)
    common=mixed_clebsch_bank(family['ternary_c3'],cfg,'ternary_c3')
    initial=np.asarray(evaluate(common,jnp.asarray(candidates)))
    order=np.argsort(abs(initial))[-2:]
    states=jnp.asarray(candidates[order])
    times={};validation={};started=perf_counter()
    def measure(name,fn,args):
        if perf_counter()-started>spec['maximum_seconds']:
            raise TimeoutError('bounded cost protocol exhausted; partial timing is retained')
        tick=perf_counter();value=jax.block_until_ready(fn(*args));cold=perf_counter()-tick
        warm=[]
        for _ in range(spec['warm_repeats']):
            tick=perf_counter();jax.block_until_ready(fn(*args));warm.append(perf_counter()-tick)
        memory=[d.memory_stats() or {} for d in jax.local_devices()]
        times[name]=dict(first_call_seconds=cold,warm_seconds=warm,
            gpu_peak_bytes=max(int(m.get('peak_bytes_in_use',0)) for m in memory))
        atomic_json(output/'progress.json',dict(times=times,validation=validation))
        print(name,json.dumps(times[name]),flush=True)
        return value
    reference_amplitudes=None;reference_h=None;arms={}
    for mode,p in family.items():
        generate=jax.jit(lambda z:mixed_clebsch_bank(z,cfg,mode))
        bank=measure(mode+'/bank_generation',generate,(p,))
        amp=np.asarray(measure(mode+'/complete_A1_amplitudes',evaluate,(bank,states)))
        psi,hpsi=measure(mode+'/complete_H_products',full_h,(bank,states))
        psi,hpsi=np.asarray(psi),np.asarray(hpsi)
        if reference_amplitudes is None:
            reference_amplitudes=amp;reference_h=hpsi
        validation[mode+'/matched_amplitude_relative']=float(np.max(abs(amp-reference_amplitudes))/max(np.max(abs(reference_amplitudes)),1e-100))
        validation[mode+'/matched_H_relative']=float(np.max(abs(hpsi-reference_h))/max(np.max(abs(reference_h)),1e-100))
        transformed=np.asarray(evaluate(bank,states[:,images[37]]))
        validation[mode+'/A1_relative']=float(np.max(abs(transformed-amp))/max(np.max(abs(amp)),1e-100))
        bra=rng.permutation([0]*(tree.n_sites//2-1)+[1]*(tree.n_sites//2+1))
        lowering=[]
        for i in np.flatnonzero(bra):
            row=bra.copy();row[i]=0;lowering.append(row)
        ward=np.asarray(raw_evaluate(bank,jnp.asarray(np.asarray(lowering))))
        validation[mode+'/S_plus_relative']=float(abs(ward.sum())/max(np.linalg.norm(ward),1e-100))
        def loss(parameters):
            b=mixed_clebsch_bank(parameters,cfg,mode)
            return jnp.mean(jnp.log(jnp.abs(jax.vmap(projected,in_axes=(None,0))(b,states))))
        gradient=measure(mode+'/complete_parameter_gradient',jax.jit(jax.grad(loss)),(p,))
        validation[mode+'/gradient_finite']=all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
        direction=jax.tree.map(lambda z:jnp.ones_like(z)/np.sqrt(mixed_parameter_count(p)),p)
        derivative=jax.jit(lambda z,v:jax.jvp(loss,(z,),(v,))[1])(p,direction)
        plus=jax.tree.map(lambda z,v:z+1e-5*v,p,direction)
        minus=jax.tree.map(lambda z,v:z-1e-5*v,p,direction)
        finite=(jax.jit(loss)(plus)-jax.jit(loss)(minus))/(2e-5)
        validation[mode+'/gradient_fd_error']=float(abs(finite-derivative)/max(1.,abs(float(derivative))))
        key=jax.random.PRNGKey(spec['sampling_seed'])
        chains=initialize_spin_exchange_chains(key,n_chains=8,n_sites=tree.n_sites,n_up=tree.n_sites//2)
        def sample(b,key,x):
            return run_exchange_sweeps(b,key,x,amplitude_fn=projected,particle_kind='spin',
                                      sweeps=16,exchange_edges=[(a,b) for a,b,_ in bonds])
        sampler=jax.jit(sample)
        sampled=measure(mode+'/eight_chain_16_proposals',sampler,(bank,key,chains))
        checkpoint=output/(mode+'.pkl')
        save_checkpoint(checkpoint,p,optimizer_state=dict(step=0),metadata=dict(
            rng=np.asarray(sampled[2]),sampler_states=np.asarray(sampled[0]),
            model=asdict(cfg),mode=mode,registration_sha256=reg['sha256']))
        saved=load_checkpoint(checkpoint)
        replay_bank=generate(saved['params'])
        validation[mode+'/checkpoint_parameters_exact']=pytree_sha256(p)==pytree_sha256(saved['params'])
        continued=sampler(bank,sampled[2],sampled[0])
        replay=sampler(replay_bank,jnp.asarray(saved['metadata']['rng']),jnp.asarray(saved['metadata']['sampler_states']))
        validation[mode+'/sampler_replay_exact']=bool(np.array_equal(np.asarray(continued[0]),np.asarray(replay[0]))
            and np.array_equal(np.asarray(continued[2]),np.asarray(replay[2])))
        arms[mode]=dict(real_parameter_count=mixed_parameter_count(p),
            amplitude_at_calibration_centers=amp.tolist(),full_H_at_centers=hpsi.tolist(),
            local_energy_ratios=(hpsi/psi).tolist(),calibration_centers_are_not_Born_samples=True)
    # Independent aggregated physical connectivity oracle on one center.
    graph=SzHeisenbergGraph(tree.n_sites,tuple(bonds),n_up=tree.n_sites//2)
    connected,elements=graph.get_conn(tuple(np.asarray(states[0])))
    direct=jax.jit(lambda b,x:jax.lax.map(lambda row:projected(b,row),x,batch_size=2))
    expected=np.dot(elements,np.asarray(direct(common,jnp.asarray(np.asarray(connected)))))
    validation['aggregated_physical_H_relative']=float(abs(expected-reference_h[0])/max(abs(expected),abs(reference_h[0]),1e-100))
    passed=all(v if isinstance(v,bool) else v < (2e-5 if 'gradient_fd' in k else 2e-11)
               for k,v in validation.items())
    result=dict(status='passed' if passed else 'failed',config=spec,arms=arms,times=times,
        validation=validation,group_order=len(images),layer_counts=tree.layer_counts,
        bank_kernel_identical_between_arms=True,binary_six_site_is_complete=True,
        full_H_centers_per_call=2,physical_bonds=len(bonds),
        raw_amplitude_evaluations_per_full_H_call=2*len(images)*(1+len(bonds)),
        optimizer_steps=0,optimized_energy_claim=False,
        wall_seconds=perf_counter()-started,registration_sha256=reg['sha256'])
    atomic_json(output/'results.json',result)
    if not passed:raise ArithmeticError('physical decoder validation failed')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();spec=json.loads(args.config.read_text())
    print(json.dumps(run(spec,args.output or Path(spec['output'])),indent=2),flush=True)
