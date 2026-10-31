"""Prepare physical parents for the existing bounded pilot, then analyze it.

No copy of its H, sampler, optimizer, checkpoint or training loop lives here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.cell_singlet_initialization import (
    local_singlet_reference, cell_product_parameters, perturb_cell_product,
    product_amplitudes, singlet_bond_diagnostics,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.models.cell_multiplicity_clebsch_tree import spin_cell_multiplicity_parameter_count
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from .spin import _bonds, _groups
from .spin_decoder_cost import ROOT, physical_config, amplitude_functions, peak_rss_bytes, batched_host_amplitudes
from .spin_decoder_pilot import starting_chains

jax.config.update('jax_enable_x64', True)
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_decoder_physical_init.json'


def registration(spec: dict) -> dict:
    """Seal initialization policy, pilot protocol and fixed pre-existing probes."""
    paths=[ROOT/spec['pilot_config']]
    for case in spec['cases']:
        paths.append(ROOT/spec['cost_parent']/case['geometry']/'values.npz')
        paths.append(ROOT/spec['previous_pilot']/case['geometry']/'results.json')
    result=dict(config=spec,inputs={str(p.relative_to(ROOT)):file_sha256(p)for p in paths},
        runtime=numerical_runtime_identity(),sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    result['registration_sha256']=canonical_sha256(result)
    return result


def prepare(path: Path, geometry: str) -> dict:
    """Exact-product embedding and bounded frozen N36 physical-parent replay."""
    spec=json.loads(path.read_text());folder=ROOT/spec['output'];d=folder/geometry
    reg=json.loads((folder/'registration.json').read_text())
    if reg!=registration(spec):raise ValueError('registration changed')
    if (d/'progress.json').exists():raise FileExistsError('preparation refuses overwrite')
    case=next(c for c in spec['cases']if c['geometry']==geometry)
    start=perf_counter()
    def progress(stage):
        atomic_json(d/'progress.json',dict(stage=stage,seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes()))
        print(geometry,stage,round(perf_counter()-start,2),'s',flush=True)
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30:raise MemoryError('preparation RSS cap')
    def alarm(_s,_f):raise TimeoutError('preparation wall cap')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(spec['wall_limit_seconds'])
    try:
        progress('begin');config=physical_config(6,6,2)
        bonds=_bonds(geometry,6,6,case['j2_over_j1']);group=_groups(geometry,6,6)['gamma_a1']
        local_bonds=[]
        for cell in config.cells:
            inverse={site:i for i,site in enumerate(cell)}
            local_bonds.append(tuple((inverse[a],inverse[b],J)for a,b,J in bonds if a in inverse and b in inverse))
        if any(x!=local_bonds[0]for x in local_bonds):raise ValueError('cells have different open Hamiltonians')
        ref=local_singlet_reference(local_bonds[0]);core=cell_product_parameters(config,ref['coefficients'])
        params=perturb_cell_product(core,seed=spec['perturbation_seed'],scale=spec['perturbation_scale'])
        identity=dict(registration_sha256=reg['registration_sha256'],geometry=geometry)
        save_parameter_artifact(d/'parameters.json',params,identity=identity)
        replay=load_parameter_artifact(d/'parameters.json',params,expected_identity=identity)
        if pytree_sha256(params)!=pytree_sha256(replay):raise ValueError('parameter replay')
        count=spin_cell_multiplicity_parameter_count(params)
        if count!=4492:raise ValueError('changed model size')
        rng=np.random.default_rng(spec['probe_seed'])
        uniform=np.zeros((spec['uniform_probe_count'],36),dtype=np.int32)
        for row in uniform:row[rng.choice(36,18,replace=False)]=1
        choices=rng.choice(16,(spec['product_probe_count'],9),p=ref['physical']**2)
        born=np.zeros((len(choices),36),dtype=np.int32)
        for k,cell in enumerate(config.cells):born[:,cell]=(choices[:,k,None]>>np.arange(3,-1,-1))&1
        pilot=json.loads((ROOT/spec['pilot_config']).read_text())
        pilot_case=next(c for c in pilot['cases']if c['geometry']==geometry)
        starts=np.concatenate([starting_chains(pilot_case['seed'],tag)[0]for tag in (1,2,3,4)])
        probes=np.concatenate((uniform,born,starts))
        funcs=amplitude_functions(config,group)
        raw=jax.jit(jax.vmap(funcs['raw'],in_axes=(None,0)))
        projected=jax.jit(jax.vmap(funcs['a1'],in_axes=(None,0)))
        core_raw=batched_host_amplitudes(raw,core,probes)
        core_a1=batched_host_amplitudes(projected,core,probes)
        oracle=product_amplitudes(probes,config.cells,ref['physical'])
        image_bits=probes[:,np.asarray(group)].reshape(-1,36)
        oracle_a1=product_amplitudes(image_bits,config.cells,ref['physical']).reshape(len(probes),len(group)).mean(axis=1)
        initialized=batched_host_amplitudes(projected,params,probes)
        def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
        checks=dict(core_raw_relative=relative(core_raw,oracle),core_A1_relative=relative(core_a1,oracle_a1),
                    core_A1_zero_probes=int(np.count_nonzero(core_a1==0)),
                    initialized_zero_probes=int(np.count_nonzero(initialized==0)),
                    minimum_initialized_magnitude=float(np.min(abs(initialized))),
                    maximum_initialized_magnitude=float(np.max(abs(initialized))),
                    probe_count=len(probes),local_eigen_residual=ref['eigen_residual'])
        if max(checks['core_raw_relative'],checks['core_A1_relative'])>1e-11:raise ValueError('product embedding failed')
        if not np.isfinite(initialized).all()or checks['initialized_zero_probes']:raise ValueError('nonfinite/zero initial probes, no replacement allowed')
        progress('embedding_and_support_checked')
        with np.load(ROOT/spec['cost_parent']/geometry/'values.npz')as a:states=a['states'].copy()
        H=make_full_heisenberg_ratios(lambda p,x:jnp.log(funcs['a1'](p,x)),bonds)
        values=np.asarray(H(params,jnp.asarray(states)))
        if not np.isfinite(values).all():raise ValueError('nonfinite inherited-probe H ratios')
        np.savez_compressed(d/'values.npz',states=states,a1_H_b8=values)
        np.savez_compressed(d/'embedding.npz',probes=probes,core_raw=core_raw,core_A1=core_a1,
                            oracle_raw=oracle,oracle_A1=oracle_a1,initialized=initialized,
                            local_physical=ref['physical'],coefficients=ref['coefficients'])
        progress('parent_ready')
        result=dict(geometry=geometry,case=case,registration_sha256=reg['registration_sha256'],
            coefficients=ref['coefficients'].tolist(),local_bonds=ref['bonds'],local_energy=ref['energy'],
            raw_unperturbed_product_energy_per_site=ref['energy']/4,local_singlet_gap=ref['singlet_gap'],
            parameter_count=count,parameters_sha256=pytree_sha256(params),checks=checks,
            values_sha256=file_sha256(d/'values.npz'),embedding_sha256=file_sha256(d/'embedding.npz'),
            full_H_centers=8,training_steps=0,ED_max_sites=4,
            seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),remote_submitted=False)
        result['result_sha256']=canonical_sha256(result);atomic_json(d/'results.json',result)
        print(geometry,'READY',checks,flush=True);return result
    except Exception as error:
        atomic_json(d/'failure.json',dict(error=str(error),seconds=perf_counter()-start));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


def analyze(path: Path) -> dict:
    """Read saved pilot chains only; no new amplitudes, samples, or H calls."""
    spec=json.loads(path.read_text());pilot=json.loads((ROOT/spec['pilot_config']).read_text())
    destination=ROOT/pilot['output'];output={}
    def summarize(x):
        means=np.asarray(x).mean(axis=0)
        return dict(mean=float(np.mean(x)),chain_se=float(np.std(means,ddof=1)/np.sqrt(len(means))))
    inputs={}
    for case in pilot['cases']:
        g=case['geometry'];d=destination/g;r=json.loads((d/'results.json').read_text())
        if r['result_sha256']!=canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'}):raise ValueError('pilot seal')
        if r['raw_sha256']!=file_sha256(d/'chains.npz'):raise ValueError('raw data changed')
        inputs[g]=dict(result_sha256=r['result_sha256'],raw_sha256=r['raw_sha256'])
        bonds=_bonds(g,6,6,case['j2_over_j1']);cells=physical_config(6,6,2).cells
        output[g]={}
        with np.load(d/'chains.npz')as a:
            for phase in ('health','audit_parent','audit_terminal'):
                if phase+'_states'not in a:continue
                states=a[phase+'_states'][a[phase+'_energy_indices']-1]
                energies=a[phase+'_energies'].real/36
                parts=singlet_bond_diagnostics(states,bonds,cells)
                row={name:summarize(x)for name,x in parts.items()}
                row['H_minus_3Hz']=summarize(energies-parts['total'])
                for name,value in (('J1',1.),('J2',case['j2_over_j1'])):
                    shell=[b for b in bonds if b[2]==value]
                    row[name]=summarize(singlet_bond_diagnostics(states,shell,cells)['total'])
                output[g][phase]=row
    result=dict(inputs=inputs,diagnostics=output,additional_samples=0,additional_H_calls=0,
        caveat='3Hz is unbiased only with equilibrium sampling of the exact singlet; its variance is not Var(H). A1 ties equivalent bonds; partition terms alone do not identify a cluster phase.')
    result['sha256']=canonical_sha256(result);atomic_json(destination/'bond_diagnostics.json',result)
    print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--register',action='store_true');p.add_argument('--geometry',choices=('triangular','square'))
    p.add_argument('--analyze',action='store_true');a=p.parse_args();spec=json.loads(a.config.read_text())
    if a.register:
        dest=ROOT/spec['output']/'registration.json'
        if dest.exists():raise FileExistsError(dest)
        r=registration(spec);atomic_json(dest,r);print(r['registration_sha256'])
    elif a.analyze:analyze(a.config)
    else:prepare(a.config,a.geometry)


if __name__=='__main__':main()
