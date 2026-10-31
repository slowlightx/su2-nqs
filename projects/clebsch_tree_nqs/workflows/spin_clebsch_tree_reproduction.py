"""Bounded exact-small capacity/optimization calibration of original CT.

No eigensolve: reuse the stored scalar ED anchor. All 12870 M0 amplitudes
are evaluated once before gather-based full A1 projection. Exact expectations
remove MCMC equilibration as a confounder; this is explicitly NOT a stochastic
8x8 accuracy reproduction. Parameters/Adam moments/RNG are resumable.
"""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    graph_hamiltonian_coo, coo_matvec, physical_permutation_index_table,
    reynolds_project_state,
)
from su2_nqs.models.literature_clebsch_tree import (
    LiteratureClebschTreeConfig, init_literature_clebsch_tree, make_clebsch_amplitudes,
)
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.frontier.artifacts import canonical_sha256, pytree_sha256, atomic_json
from su2_nqs.frontier.provenance import transitive_local_source_manifest, file_sha256
from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint
from .spin import _groups, _bonds

ROOT=Path(__file__).resolve().parents[3]
jax.config.update('jax_enable_x64',True)


def run(spec: dict, output: Path, maximum_seconds: float, until_step: int | None = None) -> dict:
    """Train ONLY a frozen exact-small model, retaining all optimizer state."""
    if spec['rows']*spec['columns'] > 16 or spec.get('expectation') != 'exact_full_support':
        raise ValueError('this calibration cannot be mislabeled sampled/large-size reproduction')
    tree,order=square_geometry_tree(spec['rows'],spec['columns'])
    cfg=LiteratureClebschTreeConfig(tree,multiplicity_cap=spec['multiplicity_cap'],maximum_q=spec['maximum_q'])
    edges=_bonds(spec['geometry'],spec['rows'],spec['columns'],spec['j2_over_j1'])
    r,c,v,basis=graph_hamiltonian_coo(tree.n_leaves,0,edges)
    x=jnp.asarray(basis,dtype=jnp.int32)
    indices=jnp.asarray(physical_permutation_index_table(basis,_groups(spec['geometry'],spec['rows'],spec['columns'])['gamma_a1']))
    raw=make_clebsch_amplitudes(cfg)
    def state(params):
        return reynolds_project_state(raw(params,x[:,jnp.asarray(order)]),indices)
    state=jax.jit(state)
    def objective(params):
        psi=state(params); hp=coo_matvec(r,c,v,psi)
        norm=jnp.vdot(psi,psi).real
        e=jnp.vdot(psi,hp).real/norm
        var=jnp.vdot(hp,hp).real/norm-e**2
        return e,(var,psi,norm)
    value_grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
    output.mkdir(parents=True,exist_ok=True)
    manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                   package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    identity=dict(config=spec,source_manifest=manifest)
    seal=canonical_sha256(identity)
    path=output/'registration.json'
    if path.exists():
        if json.loads(path.read_text()) != dict(identity,sha256=seal):
            raise ValueError('registered model/config/source changed')
    else:
        atomic_json(path,dict(identity,sha256=seal))
    key=jax.random.PRNGKey(spec['seed'])
    # Host deterministic balanced rows, including a separate initialization key.
    rng=np.random.default_rng(spec['calibration_seed'])
    calibration=np.asarray([rng.permutation([0]*(tree.n_leaves//2)+[1]*(tree.n_leaves//2)) for _ in range(1000)],dtype=np.int32)
    params,initialization=init_literature_clebsch_tree(key,cfg,calibration[:,order])
    warm_parent=spec.get('parameters_from')
    if warm_parent is not None:
        parent=ROOT/Path(warm_parent['directory'])
        checkpoint=parent/warm_parent['checkpoint_path']
        parent_reg=json.loads((parent/'registration.json').read_text())
        if (file_sha256(checkpoint) != warm_parent['checkpoint_sha256']
                or parent_reg['sha256'] != warm_parent['registration_sha256']):
            raise ValueError('immutable optimization parent changed')
        for name in ('geometry','rows','columns','j2_over_j1','multiplicity_cap','maximum_q'):
            if parent_reg['config'][name] != spec[name]:
                raise ValueError('optimization condition changed model '+name)
        previous_state=load_checkpoint(checkpoint)
        if (previous_state['metadata']['registration_sha256'] != parent_reg['sha256']
                or pytree_sha256(previous_state['params']) != warm_parent['params_sha256']):
            raise ValueError('optimization parent parameters changed')
        params=previous_state['params']
        initialization.update(parameters_from=warm_parent,adam_moments_and_rng_reset=True)
    optimizer=_manual_adam_init(params); history=[]; completed=0
    if (output/'checkpoint.pkl').exists():
        saved=load_checkpoint(output/'checkpoint.pkl')
        if saved['metadata']['registration_sha256'] != seal:
            raise ValueError('checkpoint does not belong to this frozen protocol')
        params=saved['params']; optimizer=saved['optimizer_state']
        key=jnp.asarray(saved['metadata']['rng']); history=saved['metadata']['history']
        completed=saved['metadata']['completed_steps']
        if optimizer[0] != completed or len(history) != completed:
            raise ValueError('optimizer/step/history mismatch')
    else:
        atomic_json(output/'initialization.json',initialization)
        # Retain exact source snapshots for future independent replay even if
        # subsequent opt-in workflow features are added to the working copy.
        for name,digest in manifest.items():
            target=output/'source_snapshots'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    flat,unravel=ravel_pytree(params)
    def score(theta,row):
        images=x[indices[:,row]][:,jnp.asarray(order)]
        amplitude=jnp.mean(raw(unravel(theta),images))
        # The real log-derivative is valid on the sampled nonzero support.
        return jnp.log(jnp.abs(amplitude))
    score_grad=jax.grad(score)
    @jax.jit
    def sr_direction(params,gradient,probabilities,rng):
        theta,_=ravel_pytree(params); g,_=ravel_pytree(gradient)
        rows=jax.random.choice(rng,len(x),(spec['sr_samples'],),p=probabilities)
        scores=jax.lax.map(lambda row:score_grad(theta,row),rows,batch_size=2)
        a=(scores-scores.mean(axis=0))/jnp.sqrt(len(rows))
        shift=spec['sr_damping']
        dual=jax.scipy.linalg.solve(shift*jnp.eye(len(rows))+a@a.T,a@g,assume_a='pos')
        return unravel((g-a.T@dual)/shift)
    stop=[False]
    def request_stop(signum,frame): stop[0]=True
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    for s in previous: signal.signal(s,request_stop)
    tick=perf_counter(); endpoint=spec['steps'] if until_step is None else min(until_step,spec['steps'])
    try:
        while completed < endpoint and perf_counter()-tick < maximum_seconds and not stop[0]:
            start=perf_counter()
            (e,(var,psi,norm)),gradient=value_grad(params)
            if not np.isfinite(float(e)) or not np.isfinite(float(var)) or float(norm) <= 1e-25:
                raise FloatingPointError('invalid exact physical state')
            key,next_key=jax.random.split(key)
            warm=completed < spec['sr_warm_steps']
            direction=sr_direction(params,gradient,abs(psi)**2/norm,next_key) if warm else gradient
            norm_g=jnp.sqrt(sum(jnp.sum(a*a) for a in jax.tree.leaves(direction)))
            if not warm:
                direction=jax.tree.map(lambda a:a*jnp.minimum(1.,spec['gradient_clip']/norm_g),direction)
            lr=spec['sr_adam_learning_rate'] if warm else spec['adam_learning_rate']
            params,optimizer=_manual_adam_update(params,direction,optimizer,lr)
            if not all(np.isfinite(a).all() for a in jax.tree.leaves(params)):
                raise FloatingPointError('nonfinite parameters')
            completed+=1
            history.append(dict(step=completed,energy_before_update_per_site=float(e)/tree.n_leaves,
                variance_before_update_per_site=float(var)/tree.n_leaves,optimizer='SR_then_Adam' if warm else 'Adam',
                direction_norm=float(norm_g),seconds=perf_counter()-start))
            metadata=dict(registration_sha256=seal,completed_steps=completed,history=history,rng=np.asarray(key))
            save_checkpoint(output/'checkpoint.pkl',params,optimizer_state=optimizer,metadata=metadata)
            if completed in spec['milestones']:
                save_checkpoint(output/'quality'/f'step{completed:05d}.pkl',params,optimizer_state=optimizer,metadata=metadata)
            atomic_json(output/'progress.json',dict(completed_steps=completed,last=history[-1],parameter_count=len(flat),registration_sha256=seal))
            if completed % 25 == 0: print(history[-1],flush=True)
        e,(var,psi,norm)=jax.jit(objective)(params)
        result=dict(completed_steps=completed,status='fixed_steps_complete' if completed == spec['steps'] else 'safe_pause',
            energy_per_site=float(e)/tree.n_leaves,variance_per_site=float(var)/tree.n_leaves,
            exact_energy_anchor=spec['exact_energy_anchor'],relative_energy_error=(float(e)-spec['exact_energy_anchor'])/abs(spec['exact_energy_anchor']),
            parameter_count=len(flat),params_sha256=pytree_sha256(params),checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),
            samples_are_iid_exact_Born_for_SR_only=True,full_expectation=True,stochastic_literature_reproduction=False,
            new_eigensolve=False,seconds=perf_counter()-tick,history=history,registration_sha256=seal)
        atomic_json(output/'results.json',result)
        return result
    finally:
        for s,handler in previous.items(): signal.signal(s,handler)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--maximum-seconds',type=float,default=3600.)
    parser.add_argument('--until-step',type=int)
    args=parser.parse_args(); spec=json.loads(args.config.read_text())
    result=run(spec,args.output or ROOT/spec['output'],args.maximum_seconds,args.until_step)
    print(json.dumps({k:v for k,v in result.items() if k != 'history'},indent=2))
