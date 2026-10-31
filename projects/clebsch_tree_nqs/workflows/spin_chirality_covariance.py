"""Frozen same-operator ordinary correlation control for two-copy readouts.

Use existing independent Born streams and exactly their archived amplitude.
No optimization, new chains, H calls, SU2 repair, or dense regional RDM.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.chirality_covariance import (
    PAIRS, chirality_product_connections, compare_chirality_matrices,
    cluster_jackknife, jackknife_error,
)
from su2_nqs.benchmarks.replica_spin_response import (
    vector_chirality_z_connections, operator_action_over_sample,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from . import spin_spatial_vit_quality as quality
from .spin import _groups

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]


def frozen_energy_audit_path(settings: dict) -> Path:
    """Choose a real sealed audit, never rename an energy-only record as an RDM.

    Historical requests retain ``results.json``. New energy-only protocols may
    explicitly select ``energy.json``; all state/source/stream checks below are
    identical, and this selector cannot escape the registered audit directory.
    """
    filename = settings.get('energy_audit_filename', 'results.json')
    if filename not in ('results.json', 'energy.json'):
        raise ValueError('select the actual sealed energy.json or results.json audit')
    return Path(settings['parent'])/'audits'/f"step{settings['step']:05d}"/filename


def validate_frozen_energy_audit(settings: dict, registration: dict, audit: dict) -> None:
    """Require a same-state sealed full-H audit and exactly its two Born streams."""
    reports = audit['streams']
    declared = {f"stream{s['seed']}.npz": s['raw_sha256'] for s in reports}
    if (audit['registration_sha256'] != registration['result_sha256']
            or audit['step'] != settings['step']
            or audit['audit_projection'] != 'gamma_a1'
            or audit['checkpoint_sha256'] != settings['checkpoint_sha256']
            or declared != settings['streams'] or len(reports) != 2
            or audit['full_H_calls'] != sum(s['full_H_calls'] for s in reports)
            or any(s['registration_sha256'] != registration['result_sha256']
                   or s['params_sha256'] != audit['params_sha256']
                   or s['records'] != s['full_H_calls'] for s in reports)):
        raise ValueError('audit registration/state/projection/full-H/Born-stream identity changed')


def load_frozen_inputs(settings: dict) -> tuple[dict, object, dict, list[np.ndarray], dict]:
    """Replay the old reader's physical-state guards without changing it."""
    parent = Path(settings['parent']); fp = settings['frozen_input_fingerprints']
    for path, digest in ((parent/'registration.json', fp['parent_registration_file_sha256']),
        (frozen_energy_audit_path(settings), fp['energy_audit_file_sha256']),
        (ROOT/'projects/clebsch_tree_nqs/workflows/spin_replica_response.py', fp['old_reader_source_sha256'])):
        if file_sha256(path) != digest:raise ValueError('frozen input changed: '+str(path))
    reg = quality.read_sealed(parent/'registration.json')
    audit = quality.read_sealed(frozen_energy_audit_path(settings))
    validate_frozen_energy_audit(settings, reg, audit)
    checkpoint = parent/'quality'/f"step{settings['step']:05d}.pkl"
    if (reg['result_sha256'] != fp['parent_registration']
        or audit['result_sha256'] != fp['energy_audit_result_sha256']
        or file_sha256(checkpoint) != settings['checkpoint_sha256']
        or audit['checkpoint_sha256'] != settings['checkpoint_sha256']):
        raise ValueError('frozen parent/energy audit mismatch')
    for relative,digest in reg['source_manifest'].items():
        if file_sha256(parent/'source_snapshots'/relative) != digest:
            raise ValueError('parent source snapshot changed: '+relative)
    for relative in ('su2_nqs/models/fixed_sz_spatial_vit.py','su2_nqs/models/literature_spatial_vit.py',
        'su2_nqs/benchmarks/spatial_vit_sampling.py','projects/clebsch_tree_nqs/workflows/spin.py'):
        if file_sha256(ROOT/relative) != reg['source_manifest'][relative]:
            raise ValueError('physical amplitude dependency changed: '+relative)
    relative='projects/clebsch_tree_nqs/workflows/spin_spatial_vit_quality.py'
    old=ast.parse((parent/'source_snapshots'/relative).read_text())
    current=ast.parse((ROOT/relative).read_text())
    def definition(tree, name):
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
    if ast.dump(definition(old,'validate_spec'),include_attributes=False) != ast.dump(definition(current,'validate_spec'),include_attributes=False):
        raise ValueError('frozen model validation changed')
    namespace=dict(quality.__dict__)
    exec(compile(ast.Module(body=[definition(old,'model_log')],type_ignores=[]),
                 '<trusted-archived-model-log>','exec'),namespace)
    spec=reg['config'];cfg=quality.validate_spec(spec);saved=load_checkpoint(checkpoint)
    params=saved['params']['model']
    if (saved['metadata']['registration_sha256'] != reg['result_sha256']
        or saved['metadata']['completed_steps'] != settings['step']
        or quality.checkpoint_projection(saved['metadata']) != 'gamma_a1'
        or pytree_sha256(params) != audit['params_sha256']
        or spec.get('spin_inversion_even') is not True):
        raise ValueError('exact A1 / spin-flip / frozen parameter identity required')
    streams=[];seeds=[]
    for name,digest in settings['streams'].items():
        path=parent/'audits'/f"step{settings['step']:05d}"/name
        manifest=quality.read_sealed(path.with_suffix('.json'))
        if (file_sha256(path) != digest or manifest['raw_sha256'] != digest
            or manifest['params_sha256'] != pytree_sha256(params)
            or manifest['registration_sha256'] != reg['result_sha256']):
            raise ValueError('saved Born cache changed')
        with np.load(path) as z:streams.append(z['states'])
        seeds.append(manifest['seed'])
    if (len(streams)!=2 or len(set(seeds))!=2 or streams[0].shape!=streams[1].shape
        or streams[0].ndim!=3 or not np.isin(streams,(0,1)).all()
        or not np.all(np.asarray(streams).sum(-1)==cfg.target_n_up)):
        raise ValueError('two complete independent fixed-Sz streams required')
    group=_groups(cfg.geometry,cfg.rows,cfg.columns)['gamma_a1']
    log,images=namespace['model_log'](spec,cfg,group)
    return params,jax.jit(jax.vmap(log,in_axes=(None,0))),cfg,streams,dict(
        parent_registration=reg['result_sha256'],params_sha256=pytree_sha256(params),
        projection_images=images,energy_audit=audit['energy'],seeds=seeds,
        stream_shapes=[list(x.shape) for x in streams])


def run(config: Path, *, preflight_only: bool=False) -> dict:
    """Evaluate the same three row operators and compare paired uncertainties."""
    settings=json.loads(config.read_text())
    if not preflight_only and jax.default_backend()!='gpu':raise RuntimeError('GPU allocation required')
    params,batch_fn,cfg,streams,identity=load_frozen_inputs(settings)
    if preflight_only:return identity
    output=Path(settings['output'])
    if output.exists():raise FileExistsError('readout output immutable; register a new path')
    reference=Path(settings['replica_reference'])
    if file_sha256(reference/'registration.json') != settings['replica_registration_file_sha256']:
        raise ValueError('paired replica readout changed')
    old_reg=json.loads((reference/'registration.json').read_text())
    if (old_reg['settings']['checkpoint_sha256']!=settings['checkpoint_sha256']
        or old_reg['settings']['streams']!=settings['streams']):
        raise ValueError('ordinary and replica estimates require the same state/streams')
    output.mkdir(parents=True)
    sources={str(p.relative_to(ROOT)):file_sha256(p) for p in (Path(__file__),
        ROOT/'su2_nqs/benchmarks/chirality_covariance.py',
        ROOT/'su2_nqs/benchmarks/replica_spin_response.py')}
    registration=dict(settings=settings,identity=identity,sources=sources,
        config_file_sha256=file_sha256(config),new_Born_samples=0,new_H_calls=0,new_training=0)
    registration['sha256']=canonical_sha256(registration)
    atomic_json(output/'registration.json',registration)
    start=perf_counter();queries=0
    def query(bits):
        nonlocal queries
        if len(bits)==0:return np.empty(0,complex)
        rows,inverse=np.unique(bits,axis=0,return_inverse=True);values=[]
        size=settings['query_batch']
        for offset in range(0,len(rows),size):
            if perf_counter()-start>settings['wall_seconds']:raise TimeoutError('fixed readout cap exceeded')
            chunk=rows[offset:offset+size]
            padded=np.pad(chunk,((0,size-len(chunk)),(0,0)),mode='edge')
            values.append(np.asarray(batch_fn(params,jnp.asarray(padded)))[:len(chunk)])
            queries+=len(chunk)
        return np.concatenate(values)[inverse]
    bonds=[tuple(map(tuple,settings['chirality_channels']['prefix18'][f'row{i}_kappa_z'])) for i in range(3)]
    records=[];actions=[]
    for x in streams:
        x=x.reshape(-1,cfg.n_sites);pieces=[];first_pieces=[]
        for offset in range(0,len(x),settings['center_chunk']):
            bits=x[offset:offset+settings['center_chunk']];denominator=query(bits)
            connections=[chirality_product_connections(bits,bonds[i],bonds[j]) for i,j in PAIRS]
            first=[vector_chirality_z_connections(bits,edges) for edges in bonds]
            all_connections=connections+first
            lengths=[len(t) for t,_,_ in all_connections]
            logs=query(np.concatenate([t for t,_,_ in all_connections]))
            cuts=np.cumsum([0]+lengths)
            local=np.column_stack([operator_action_over_sample(logs[cuts[k]:cuts[k+1]],s,c,denominator)
                                   for k,(_,s,c) in enumerate(all_connections)])
            pieces.append(local[:,:6]);first_pieces.append(local[:,6:])
        records.append(np.concatenate(pieces));actions.append(np.concatenate(first_pieces))
    products=np.stack(records,axis=1);one_point=np.stack(actions,axis=1)
    chains=streams[0].shape[1];labels=np.tile(np.arange(chains),streams[0].shape[0])
    with np.load(reference/'prefix18.npz') as z:
        if not np.array_equal(labels,z['chain_pair_labels']):raise ValueError('chain pairing changed')
        order=list(old_reg['settings']['momenta'])+list(old_reg['settings']['chirality_channels']['prefix18'])
        names=['row0_kappa_z','row1_kappa_z','row2_kappa_z','row0_plus_row1_kappa_z',
               'row0_plus_row2_kappa_z','row1_plus_row2_kappa_z']
        numerator=z['numerator'][:,[order.index(n) for n in names]];swap=z['swap']
    result=compare_chirality_matrices(products.mean(axis=1),numerator,swap,labels)
    mean,leave=cluster_jackknife(one_point.mean(axis=1),labels)
    result.update(one_point_means_real=mean.real.tolist(),one_point_means_imaginary=mean.imag.tolist(),
        one_point_means_real_SE=jackknife_error(leave.real).tolist(),
        one_point_means_imaginary_SE=jackknife_error(leave.imag).tolist(),
        exact_connected_subtraction='parent spin-flip-even => <kappa_z>=0; diagonal fluctuations retained',
        energy_audit=identity['energy_audit'],wall_seconds=perf_counter()-start,
        projected_amplitude_queries=queries,projection_images=identity['projection_images'],
        source_registration_sha256=registration['sha256'],new_Born_samples=0,new_H_calls=0,new_training=0,
        state_mixing_certified=False,phase_classification=False,
        nodal_intermediate_contributions_retained=True)
    np.savez_compressed(output/'records.npz',ordinary_products=products,one_point_actions=one_point,
                        replica_numerators=numerator,swap=swap,chain_pair_labels=labels)
    atomic_json(output/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args.config,preflight_only=args.preflight_only),indent=2))
