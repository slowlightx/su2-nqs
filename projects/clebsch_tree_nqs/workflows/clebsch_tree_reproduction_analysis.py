"""Replay fixed CT optimization endpoints and produce one non-bar comparison.

No training or eigensolve. Stored scalar ED and independently accumulated S+
and full-A1 invariance are used to separate accuracy from strict symmetry.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.literature_clebsch_tree import LiteratureClebschTreeConfig,make_clebsch_amplitudes
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis,physical_permutation_index_table,reynolds_project_state,graph_hamiltonian_coo,coo_matvec
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.adam import _manual_adam_init,_manual_adam_update
from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from .spin import _groups,_bonds

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'projects/clebsch_tree_nqs/results/vieijra_nys_reproduction_20261014'


def analyze() -> dict:
    """Compare the two preregistered fixed endpoints, not a best-checkpoint pick."""
    import matplotlib.pyplot as plt
    names=['square4_j05','square4_j05_adam_reset_x64']
    results=[json.loads((BASE/n/'results.json').read_text()) for n in names]
    tree,order=square_geometry_tree(4,4)
    cfg=LiteratureClebschTreeConfig(tree,multiplicity_cap=8,maximum_q=8)
    raw=make_clebsch_amplitudes(cfg)
    basis=fixed_m_spin_basis(16,0); indices=physical_permutation_index_table(basis,_groups('square',4,4)['gamma_a1'])
    triplet_basis=fixed_m_spin_basis(16,2)
    positions={row:i for i,row in enumerate(triplet_basis)}
    records=[]
    state=jax.jit(lambda p:reynolds_project_state(raw(p,jnp.asarray(basis)[:,jnp.asarray(order)]),indices))
    for name,r in zip(names,results,strict=True):
        p=load_checkpoint(BASE/name/'checkpoint.pkl')['params']
        psi=np.asarray(reynolds_project_state(jax.jit(raw)(p,jnp.asarray(basis)[:,jnp.asarray(order)]),indices))
        psi=psi/np.linalg.norm(psi)
        raised=np.zeros(len(triplet_basis))
        for value,bits in zip(psi,basis,strict=True):
            for site,bit in enumerate(bits):
                if bit == 0:
                    row=list(bits); row[site]=1; raised[positions[tuple(row)]]+=value
        ward=float(np.linalg.norm(raised))
        a1=float(np.max(abs(psi[indices]-psi[None,:])))
        if ward > 2e-11 or a1 > 2e-12: raise ArithmeticError('independent endpoint symmetry failed')
        records.append(dict(name=name,S_plus_norm=ward,A1_max_absolute=a1,
                            energy_per_site=r['energy_per_site'],variance_per_site=r['variance_per_site'],
                            relative_energy_error=r['relative_energy_error'],parameters=r['parameter_count']))
    # Fixed-parent, identical-gradient proposals only: NOT new trained states
    # or another search for small-system energy digits.
    parent=load_checkpoint(BASE/names[0]/'quality/step00100.pkl')
    rows,cols,vals,_=graph_hamiltonian_coo(16,0,_bonds('square',4,4,0.5))
    def energy(p):
        psi=state(p); hp=coo_matvec(rows,cols,vals,psi)
        return jnp.vdot(psi,hp).real/jnp.vdot(psi,psi).real
    gradient=jax.jit(jax.grad(energy))(parent['params'])
    norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(gradient)))
    clipped=jax.tree.map(lambda g:g*jnp.minimum(1.,.001/norm),gradient)
    before=np.asarray(state(parent['params'])); before=before/np.linalg.norm(before)
    proposals={}
    for label,opt in [('inherited',parent['optimizer_state']),('reset',_manual_adam_init(parent['params']))]:
        candidate,_=_manual_adam_update(parent['params'],clipped,opt,.001)
        after=np.asarray(state(candidate)); after=after/np.linalg.norm(after)
        fs=float(np.sqrt(max(0.,1-abs(np.vdot(before,after))**2)))
        proposals[label]=dict(exact_FS_sine=fs,energy_change_per_site=float(energy(candidate)-energy(parent['params']))/16,
                             parameter_update_norm=float(jnp.sqrt(sum(jnp.sum((b-a)**2) for a,b in zip(jax.tree.leaves(parent['params']),jax.tree.leaves(candidate),strict=True)))))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,r,offset,label,color in zip(names,results,[0,100],['Inherited Adam moments','Reset moments; same parent'],['0.5','tab:blue']):
        history=r['history']; steps=np.array([x['step']-1+offset for x in history])
        gap=np.array([x['energy_before_update_per_site'] for x in history])-r['exact_energy_anchor']/16
        var=np.array([x['variance_before_update_per_site'] for x in history])
        axes[0].semilogy(steps,np.maximum(gap,1e-14),color=color,label=label,lw=1.8)
        axes[1].semilogy(steps,np.maximum(var,1e-14),color=color,label=label,lw=1.8)
        axes[0].scatter(offset+r['completed_steps'],r['energy_per_site']-r['exact_energy_anchor']/16,color=color,s=25)
        axes[1].scatter(offset+r['completed_steps'],r['variance_per_site'],color=color,s=25)
    for ax in axes:
        ax.axvline(100,color='k',ls=':',lw=1); ax.set_xlabel('Total physical updates'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Energy error per site'); axes[1].set_ylabel('Variance per site')
    axes[0].legend(fontsize=8); fig.suptitle('Original-style CT, square 4×4 J₂/J₁=0.5; 2506 parameters, exact expectations')
    folder=ROOT/'projects/clebsch_tree_nqs/figures/exploration'; folder.mkdir(parents=True,exist_ok=True)
    figure=folder/'clebsch_tree_optimizer_calibration_20261014.png'; fig.savefig(figure,dpi=180); plt.close(fig)
    report=dict(endpoints=records,figure=str(figure.relative_to(ROOT)),sampling_reproduction=False,
                capacity_changed=False,reset_condition_from_same_physical_parent=True,new_ED=False,
                fixed_parent_identical_clipped_gradient_proposals=proposals,proposal_is_not_added_training=True)
    atomic_json(BASE/'comparison.json',report)
    print(json.dumps(report,indent=2)); return report


def projection_replay(parent: Path,output: Path) -> dict:
    """Same trained parameters, exact amplitudes/H/gradients, warmed GPU times."""
    from .spin_clebsch_tree_sampled import components
    from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
    from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
    if jax.default_backend() != 'gpu': raise RuntimeError('registered projection timing requires GPU')
    if (output/'results.json').exists(): raise FileExistsError(output)
    reg=json.loads((parent/'registration.json').read_text()); spec=reg['config']
    saved=load_checkpoint(parent/'checkpoint.pkl'); p=saved['params']
    endpoint=json.loads((parent/'training.json').read_text())
    if (saved['metadata']['params_sha256'] != pytree_sha256(p)
            or endpoint['params_sha256'] != pytree_sha256(p)
            or endpoint['checkpoint_sha256'] != file_sha256(parent/'checkpoint.pkl')
            or saved['metadata']['registration_sha256'] != reg['sha256']):
        raise ValueError('trained projection parent changed')
    cfg,raw,amp,bonds,images,order=components(spec)
    folded,report=make_projected_clebsch(cfg,order,images,orbit_chunk=8)
    old=jax.vmap(amp,in_axes=(None,0)); x=jnp.asarray(saved['metadata']['sampler_states'][:2])
    times={}; values={}
    for label,fn in [('full',old),('folded',folded)]:
        for kind,function in [('amplitude',fn),('H',make_amplitude_heisenberg_action(lambda z,row:fn(z,row[None])[0],bonds,n_sites=cfg.tree.n_leaves)),
                              ('gradient',jax.grad(lambda z,rows:jnp.mean(jnp.log(jnp.abs(fn(z,rows))))))]:
            executable=jax.jit(function); t=perf_counter(); value=jax.block_until_ready(executable(p,x)); cold=perf_counter()-t
            timings=[]
            for _ in range(2):
                t=perf_counter(); value=jax.block_until_ready(executable(p,x)); timings.append(perf_counter()-t)
            key=label+'_'+kind; times[key]=dict(cold_seconds=cold,warm_seconds=timings); values[key]=value
            print(key,times[key],flush=True)
    errors={}
    for kind in ('amplitude','H','gradient'):
        discrepancies=[]
        for a,b in zip(jax.tree.leaves(values['full_'+kind]),jax.tree.leaves(values['folded_'+kind]),strict=True):
            a,b=np.asarray(a),np.asarray(b)
            discrepancies.append(float(np.max(abs(a-b)))/max(float(np.max(abs(a))),float(np.max(abs(b))),1e-25))
        errors[kind]=max(discrepancies)
    if max(errors.values()) > 3e-11: raise ArithmeticError('folded physical state/action/tangent changed')
    result=dict(parent_checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'),parent_training_steps=saved['metadata']['completed_steps'],
                exact_projection=report,maximum_relative_errors=errors,times=times,parameters_unchanged=True,training_steps_added=0)
    result['source_manifest']=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    atomic_json(output/'results.json',result); print(json.dumps(result,indent=2)); return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection-parent',type=Path); parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.projection_parent:
        if args.output is None: raise ValueError('new projection replay output required')
        projection_replay(args.projection_parent,args.output)
    else: analyze()
