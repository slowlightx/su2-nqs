"""Frozen ordinary-ViT SWAP response from two existing independent audit streams.

No optimization, new Born sampling, Hamiltonian calls, SU(2) repair or dense
RDM. Each persistent chain is used in exactly one independent replica pair.
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

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.benchmarks.replica_spin_response import (
    swap_response_records, summarize_swap_response, explicit_half_rdm_cost,
    vector_chirality_z_connections, operator_action_over_sample,
    hermitian_replica_numerator,
    double_spin_connections, row_pi_bonds,
)
from .spin_spatial_vit_quality import ROOT, read_sealed, validate_spec, model_log, checkpoint_projection
from . import spin_spatial_vit_quality as quality_module
from .spin import _groups
from .spin_chirality_covariance import frozen_energy_audit_path, validate_frozen_energy_audit

jax.config.update('jax_enable_x64', True)


def run(config_path: Path) -> dict:
    """Evaluate a preregistered geometry and all its fixed cache records."""
    settings = json.loads(config_path.read_text())
    if jax.default_backend() != 'gpu':
        raise RuntimeError('bounded GPU readout, never run on login CPU')
    parent = Path(settings['parent'])
    destination = Path(settings['output'])
    if destination.exists():
        raise FileExistsError('immutable completed/partial readout; use a new registered output')
    reg = read_sealed(parent/'registration.json')
    spec = reg['config']
    cfg = validate_spec(spec)
    step = settings['step']
    checkpoint = parent/'quality'/f'step{step:05d}.pkl'
    if file_sha256(checkpoint) != settings['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    audit = read_sealed(frozen_energy_audit_path(settings))
    validate_frozen_energy_audit(settings, reg, audit)
    if audit['checkpoint_sha256'] != settings['checkpoint_sha256']:
        raise ValueError('independent energy audit belongs to another state')
    for relative, digest in reg['source_manifest'].items():
        if file_sha256(parent/'source_snapshots'/relative) != digest:
            raise ValueError('frozen parent source snapshot changed')
    relative = 'projects/clebsch_tree_nqs/workflows/spin_spatial_vit_quality.py'
    old = ast.parse((parent/'source_snapshots'/relative).read_text())
    new = ast.parse((ROOT/relative).read_text())
    for name in ('validate_spec',):
        pick = lambda tree: ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name),include_attributes=False)
        if pick(old) != pick(new): raise ValueError('frozen amplitude definition changed')
    for relative in ('su2_nqs/models/fixed_sz_spatial_vit.py','su2_nqs/models/literature_spatial_vit.py',
                     'su2_nqs/benchmarks/spatial_vit_sampling.py','projects/clebsch_tree_nqs/workflows/spin.py'):
        if file_sha256(ROOT/relative) != reg['source_manifest'][relative]:
            raise ValueError('frozen physical amplitude dependency changed: '+relative)
    # Use the ACTUAL archived function, not a newly modified implementation.
    # Only its trusted FunctionDef is compiled; numerical model dependencies
    # above must still match the archived hashes exactly. In particular an
    # all-forbidden-input NaN guard was later added; it must not redefine this
    # parent. Forbidden swaps below are zero by the parent's fixed-Sz law.
    frozen_definition = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name == 'model_log')
    namespace = dict(quality_module.__dict__)
    exec(compile(ast.Module(body=[frozen_definition],type_ignores=[]),
                 '<trusted-frozen-model-log>', 'exec'),namespace)
    frozen_model_log = namespace['model_log']
    saved = load_checkpoint(checkpoint)
    params = saved['params']['model']
    if (saved['metadata']['registration_sha256'] != reg['result_sha256']
            or saved['metadata']['completed_steps'] != step
            or checkpoint_projection(saved['metadata']) != 'gamma_a1'
            or pytree_sha256(params) != audit['params_sha256']):
        raise ValueError('physical parameter/projection identity mismatch')
    records = []
    inputs = {str(checkpoint): settings['checkpoint_sha256']}
    if len(settings['streams']) != 2 or len(set(settings['streams'])) != 2:
        raise ValueError('exactly two independent source streams required')
    seeds = []
    for name, digest in settings['streams'].items():
        path = parent/'audits'/f'step{step:05d}'/name
        manifest = read_sealed(path.with_suffix('.json'))
        if (file_sha256(path) != digest or manifest['raw_sha256'] != digest
                or manifest['params_sha256'] != pytree_sha256(params)
                or manifest['registration_sha256'] != reg['result_sha256']):
            raise ValueError('independent Born cache changed')
        seeds.append(manifest['seed'])
        with np.load(path) as z: records.append(z['states'])
        inputs[str(path)] = digest
    x, y = records
    if (len(set(seeds)) != 2 or x.shape != y.shape or x.ndim != 3
            or x.shape[2] != cfg.n_sites or x.shape[1] < 4
            or not np.isin(records,(0,1)).all() or not np.all(np.asarray(records).sum(-1) == cfg.target_n_up)):
        raise ValueError('all complete independent chains must share the registered fixed-Sz support')
    destination.mkdir(parents=True)
    sources = transitive_local_source_manifest((Path(__file__),),root=ROOT,
                package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration = dict(settings=settings, parent_registration=reg['result_sha256'],
        inputs=inputs, source_manifest=sources, independent_seeds=seeds,
        new_samples=0, new_training=0, new_H_calls=0,
        pairing='stream0 chain c with stream1 chain c, all snapshots; each chain used once',
        fixed_before_queries=True)
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(destination/'registration.json',registration)
    for name in sources:
        path=destination/'source_snapshots'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
        if file_sha256(path) != sources[name]: raise RuntimeError('readout source changed during registration')
    group = _groups(cfg.geometry,cfg.rows,cfg.columns)['gamma_a1']
    log, image_count = frozen_model_log(spec,cfg,group)
    batch_fn = jax.jit(jax.vmap(log,in_axes=(None,0)))
    start = perf_counter()
    queries = 0
    def query(bits):
        nonlocal queries
        if not len(bits):
            return np.empty(0,complex)
        # Exact repeated physical configurations, not a stochastic truncation.
        bits,inverse=np.unique(bits,axis=0,return_inverse=True)
        values = []
        batch = settings['query_batch']
        for offset in range(0,len(bits),batch):
            if perf_counter()-start > settings['wall_seconds']:
                raise TimeoutError('fixed readout cap, no partial response claim')
            a = bits[offset:offset+batch]
            padded = np.pad(a,((0,batch-len(a)),(0,0)),mode='edge')
            values.append(np.asarray(batch_fn(params,jnp.asarray(padded)))[:len(a)])
            queries += len(a)
        return np.concatenate(values)[inverse]
    def chirality_action(bits, denominators, bonds):
        targets,source,coefficients=vector_chirality_z_connections(bits,tuple(map(tuple,bonds)))
        return operator_action_over_sample(query(targets),source,coefficients,denominators)
    def quadrupole_action(bits, denominators, bonds, delta_m):
        targets,source,coefficients=double_spin_connections(bits,bonds,delta_m=delta_m)
        return operator_action_over_sample(query(targets),source,coefficients,denominators)
    chains = x.shape[1]
    x,y = x.reshape(-1,cfg.n_sites),y.reshape(-1,cfg.n_sites)
    labels = np.tile(np.arange(chains),len(x)//chains)
    original = np.column_stack((query(x),query(y)))
    momenta = settings['momenta']
    summaries = {}
    orbit = [(np.arange(cfg.n_sites),False)]
    if settings.get('relative_replica_orbit') == 'complete_x_translations_times_spin_flip':
        if spec.get('spin_inversion_even') is not True:
            raise ValueError('relative spin flip requires the actual even parent symmetry')
        grid=np.arange(cfg.n_sites).reshape(cfg.rows,cfg.columns)
        orbit=[(np.roll(grid,shift,axis=1).ravel(),flip)
               for shift in range(cfg.columns) for flip in (False,True)]
    elif settings.get('relative_replica_orbit') is not None:
        raise ValueError('unregistered relative-replica group')
    # Check the symmetry actually used to reuse denominator queries.
    for gather,flip in orbit:
        transformed=y[:4,gather]
        if flip: transformed=1-transformed
        replay=query(transformed)
        if np.max(abs(np.exp(replay-original[:4,1])-1)) > 1e-10:
            raise ValueError('relative replica symmetry does not preserve the frozen physical amplitude')
    for name, sites in settings['regions'].items():
        if len(set(sites)) != len(sites) or min(sites) < 0 or max(sites) >= cfg.n_sites:
            raise ValueError('unique legal physical regional sites required')
        coordinates = np.array([[i%cfg.columns,i//cfg.columns] for i in sites])
        phases = np.exp(2j*np.pi*coordinates @ np.array(list(momenta.values())).reshape(-1,2).T)
        values = np.stack(((x[:,sites]-.5)@phases,(y[:,sites]-.5)@phases),axis=1)
        chirality=settings.get('chirality_channels',{}).get(name,{})
        if any(not edges or any(a not in sites or b not in sites for a,b in edges)
               for edges in chirality.values()):
            raise ValueError('frozen regional chirality support must lie wholly inside the cut')
        quad_rows=settings.get('bond_quadrupole_rows',{}).get(name,[])
        quad={str(r):row_pi_bonds(r,cfg.columns) for r in quad_rows}
        if len(quad)!=len(quad_rows) or any(a not in sites or b not in sites
                for edges in quad.values() for a,b,_ in edges):
            raise ValueError('unique complete quadrupole rows inside region required')
        quad_order=[f'Qminus_row{a}_Qplus_row{b}' for a in quad for b in quad]
        w=np.zeros(len(x),complex);n=np.zeros((len(x),len(momenta)+len(chirality)+len(quad_order)),complex)
        orbit_legal=[]
        for gather,flip in orbit:
            partner=y[:,gather]
            if flip: partner=1-partner
            swapped_x, swapped_y = x.copy(),partner.copy()
            swapped_x[:,sites],swapped_y[:,sites] = partner[:,sites],x[:,sites]
            legal=swapped_x.sum(-1) == cfg.target_n_up
            swapped=np.full(original.shape,-np.inf+0j)
            if legal.any():
                swapped[legal]=np.column_stack((query(swapped_x[legal]),query(swapped_y[legal])))
            current_values=np.stack((values[:,0],(partner[:,sites]-.5)@phases),axis=1)
            wi,ni=swap_response_records(original,swapped,current_values)
            n[:,:len(momenta)] += ni/len(orbit)
            for j,edges in enumerate(chirality.values(),start=len(momenta)):
                actions=np.zeros(original.shape,complex)
                if legal.any():
                    actions[legal,0]=chirality_action(swapped_x[legal],original[legal,0],edges)
                    actions[legal,1]=chirality_action(swapped_y[legal],original[legal,1],edges)
                n[:,j] += hermitian_replica_numerator(actions)/len(orbit)
            if quad:
                # Q-- on the first swapped row, Q++ on the second. These
                # have ZERO ordinary SWAP amplitude at fixed Sz; filtering
                # by the plain-SWAP legal mask would discard the response.
                charged_legal=(swapped_x.sum(-1)==cfg.target_n_up-2)
                if not np.array_equal(charged_legal,swapped_y.sum(-1)==cfg.target_n_up+2):
                    raise ValueError('independent replicas must conserve total doubled magnetization')
                actions_minus=np.zeros((len(x),len(quad)),complex)
                actions_plus=np.zeros_like(actions_minus)
                if charged_legal.any():
                    for j,edges in enumerate(quad.values()):
                        actions_minus[charged_legal,j]=quadrupole_action(
                            swapped_x[charged_legal],original[charged_legal,0],edges,-4)
                        actions_plus[charged_legal,j]=quadrupole_action(
                            swapped_y[charged_legal],original[charged_legal,1],edges,4)
                offset=len(momenta)+len(chirality)
                quad_record=np.einsum('ia,ib->iab',actions_minus,actions_plus)
                if settings.get('replica_exchange_average',False):
                    # The replica labels are exchangeable. Use BOTH magnetic
                    # imbalances, without treating them as independent chains.
                    reverse=swapped_y.sum(-1)==cfg.target_n_up-2
                    reverse_minus=np.zeros_like(actions_minus);reverse_plus=np.zeros_like(actions_plus)
                    if reverse.any():
                        for j,edges in enumerate(quad.values()):
                            reverse_minus[reverse,j]=quadrupole_action(
                                swapped_y[reverse],original[reverse,1],edges,-4)
                            reverse_plus[reverse,j]=quadrupole_action(
                                swapped_x[reverse],original[reverse,0],edges,4)
                    quad_record=(quad_record+np.einsum('ia,ib->iab',reverse_minus,reverse_plus))/2
                n[:,offset:]+=quad_record.reshape(len(x),-1)/len(orbit)
            w += wi/len(orbit)
            orbit_legal.append(float(legal.mean()))
        np.savez_compressed(destination/(name+'.npz'),swap=w,numerator=n,
            chain_pair_labels=labels)
        try:
            summary = summarize_swap_response(w,n,labels)
            summary['status'] = 'estimated_not_equilibrium_certified'
        except ValueError as error:
            summary = dict(status='unresolved', reason=str(error),
                purity_real_mean=float(w.real.mean()),purity_imaginary_mean=float(w.imag.mean()),
                phase_classification=False)
        ordinary = .5*(abs(values[:,0])**2+abs(values[:,1])**2)
        per_chain = np.array([ordinary[labels == g].mean(axis=0) for g in range(chains)])
        summary.update(momentum_order=list(momenta),
            channel_order=list(momenta)+list(chirality)+quad_order,region_sites=sites,
            quadrupole_rows=quad_rows,
            replica_exchange_average=settings.get('replica_exchange_average',False),
            quadrupole_definition='Tr(rho Q_a-- rho Q_b++)/Tr(rho^2); doubled tensor q=4; no SU2 equality between magnetic components assumed',
            chirality_bonds=chirality,
            definition='Tr(rho O_dagger rho O)/Tr(rho^2); spin Fourier sums and Hermitian kappa_z sums',
            vector_not_scalar_chirality=True, entanglement_not_Hamiltonian_response=True,
            exact_relative_replica_orbit_size=len(orbit),
            number_of_independent_chain_pairs_unchanged=chains,
            per_orbit_legal_swap_fractions=orbit_legal,
            ordinary_SzSz=ordinary.mean(axis=0).tolist(),
            ordinary_SzSz_pair_SE=(per_chain.std(axis=0,ddof=1)/np.sqrt(chains)).tolist(),
            connected_one_point='exact spin-flip-even parent => <Sz_i>=0; onsite fluctuations retained',
            response_does_not_assume_SU2=True)
        np.savez_compressed(destination/(name+'.npz'),swap=w,numerator=n,
            ordinary=ordinary,chain_pair_labels=labels)
        atomic_json(destination/(name+'.json'),summary)
        summaries[name] = summary
    result = dict(registration_sha256=registration['sha256'],summaries=summaries,
        wall_seconds=perf_counter()-start, projected_amplitude_queries=queries,
        projection_images=image_count, new_samples=0,new_H_calls=0,new_training=0,
        explicit18_cost=explicit_half_rdm_cost(18),
        all_inputs_unchanged=all(file_sha256(Path(p)) == h for p,h in inputs.items()),
        energy_audit= audit['energy'],
        strict_SU2=False, phase_classification=False)
    atomic_json(destination/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    run(parser.parse_args().config)
