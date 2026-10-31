"""Energy-only ensemble SR in equivalent affine GRU parameter coordinates.

Output and callback parameters are converted to the original physical GRU
API. No physical observable or exact-reference label enters optimization.
"""
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from fnqs.diagnostics.response import response_records
from fnqs.pilots.engine import Options,response_loss,solve
from fnqs.training import init_optimizer
from fnqs.variants.size_features import SizeFeatureConfig
from fnqs.variants.coupling_units import CouplingUnits,to_units,to_raw,physical_log_psi,physical_sample


@partial(jax.jit,static_argnames=('config','units','options'))
def sampled_block(params: dict,bits: jax.Array,g: jax.Array,config: SizeFeatureConfig,units: CouplingUnits,options: Options) -> tuple:
    """Center scores per physical H, with original energy/site force."""
    if options.response_weight!=0:raise ValueError('coordinate SR is energy-only')
    flat,unravel=ravel_pytree(params)
    amplitude=lambda p,b,x:physical_log_psi(p,b,x,config,units)
    scores=jax.vmap(jax.grad(lambda v,b:amplitude(unravel(v),b,g)),(None,0))(flat,bits)
    rows=response_records(params,bits,g,amplitude)
    centered=scores-scores.mean(axis=0);energy=rows['energy']
    force=centered.T@(energy-energy.mean())/(len(bits)*bits.shape[1])
    return centered/jnp.sqrt(len(bits)),force,energy.mean()/bits.shape[1],response_loss(rows,jnp.ones(len(bits)))


@partial(jax.jit,static_argnames=('count','n','config','units'))
def draw(params: dict,key: jax.Array,count: int,n: int,g: jax.Array,config: SizeFeatureConfig,units: CouplingUnits) -> jax.Array:
    """Independent physical Born draws in the optimizer coordinate chart."""
    return physical_sample(params,key,count,n,g,config,units)


def train(raw_params: dict,key: jax.Array,n: int,schedule: list,config: SizeFeatureConfig,units: CouplingUnits,options: Options,callback=None) -> tuple:
    """Optimize equivalent coordinates and return raw physical parameters.

    Original SR damping, tolerance and clipping values are kept. Their
    effective action changes under the coordinate map and is the tested
    intervention. SR has no momentum; raw checkpoint moments remain zero.
    """
    if options.response_weight!=0:raise ValueError('coordinate SR is energy-only')
    params=to_units(raw_params,units);state=init_optimizer(raw_params)
    work=dict(updates=0,task_evaluations=0,hamiltonian_actions=0,source_actions=0,born_samples=0,basis_rows=0,sr_correction_solves=0)
    history=[]
    for step,tasks in enumerate(schedule,1):
        if not tasks:raise ValueError('empty update')
        blocks=[];forces=[];energies=[];losses=[]
        for task in tasks:
            g=np.asarray(task,dtype=np.float64)
            if g.shape!=(4,) or not np.isfinite(g).all() or g[3]<=0 or g[2]<0:
                raise ValueError('invalid physical stoquastic task')
            key,subkey=jax.random.split(key)
            bits=draw(params,subkey,options.samples,n,jnp.asarray(g),config,units)
            a,f,e,r=sampled_block(params,bits,jnp.asarray(g),config,units,options)
            blocks.append(a);forces.append(f);energies.append(float(e));losses.append(float(r))
        count=len(tasks);a=jnp.concatenate(blocks)/jnp.sqrt(count);rhs=jnp.stack(forces).mean(axis=0)
        direction,residual,corrections=solve(a,rhs,options)
        flat,unravel=ravel_pytree(params)
        scale=min(options.learning_rate,options.max_step/max(float(jnp.linalg.norm(direction)),1e-30))
        params=unravel(flat-scale*direction)
        work['updates']+=1;work['task_evaluations']+=count;work['hamiltonian_actions']+=2*count
        work['source_actions']+=count;work['born_samples']+=options.samples*count;work['sr_correction_solves']+=corrections
        history.append(dict(step=step,tasks=tasks,energy_per_site=energies,response_loss=losses,true_sr_residual=residual,
                            step_norm=float(scale*jnp.linalg.norm(direction)),sr_corrections=corrections,step_norm_coordinate='affine_optimizer_chart'))
        state['step']=step;state['work']=dict(work)
        if callback is not None:callback(to_raw(params,units),state,key,history)
    return to_raw(params,units),state,key,history
