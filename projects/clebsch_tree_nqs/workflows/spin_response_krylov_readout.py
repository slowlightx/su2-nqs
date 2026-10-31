"""Bounded two-vector K/M response Ritz readout on existing Q1 proposals.

This reuses frozen full-A1 amplitudes and fixed records from previous sealed
readouts. No new samples, source training, or optimization occurs. m4 and
physical Ritz residuals are deliberately absent, requiring a future Q2 law.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from shutil import copy2
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.ground_state_response_krylov import one_step_response_from_moments
from su2_nqs.benchmarks.spin_response_krylov_moments import two_action_plan, third_source_moments
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint
from .spin_spatial_vit_quality import (
    ROOT, read_sealed, sealed_json, validate_spec, model_log, cached_quality_identity,
    checkpoint_projection, _groups, _bonds,
)


def summarize(values: dict[str, np.ndarray], chain_ids: np.ndarray) -> dict:
    """Whole-chain paired jackknife of both Ritz poles and source weights."""
    ids = np.unique(chain_ids); n = len(ids)
    if n < 2:
        raise ValueError('at least two independent chains required')
    means = {key: np.array([value[chain_ids == group].mean() for group in ids])
             for key, value in values.items()}
    out = {}
    for star, names in (('Kstar', ('K', 'minusK')), ('Mstar', ('M1', 'M2', 'M3'))):
        rows = np.array([[np.mean([means[name+'__m'+str(r)][c] for name in names])
                          for c in range(n)] for r in range(4)])
        parent = np.array([means['parent__m0'], means['parent__m1']])

        def metrics(m, p):
            ritz = one_step_response_from_moments(m)
            centroid = m[1]/m[0]
            parent_energy = p[1]/p[0]
            return np.array([centroid-parent_energy,
                m[2]/m[0]-centroid**2,
                *(ritz['energies']-parent_energy), *ritz['source_weights']])

        average = rows.mean(1); pa = parent.mean(1)
        try:
            center = metrics(average, pa)
            replicas = np.array([metrics((n*average-rows[:, c])/(n-1),
                                        (n*pa-parent[:, c])/(n-1)) for c in range(n)])
            se = np.sqrt((n-1)*np.mean((replicas-replicas.mean(0))**2, axis=0))
            fields = ('centroid_from_parent', 'source_width_squared', 'lowest_ritz_from_parent',
                      'upper_ritz_from_parent', 'lowest_source_weight', 'upper_source_weight')
            out[star] = dict(status='positive_source_Gram', metrics={key:dict(value=float(v),
                chain_SE=float(e), jackknife_bias_estimate=float((n-1)*(replicas[:, i].mean()-v)))
                for i,(key,v,e) in enumerate(zip(fields,center,se))},
                moments_unnormalized=average.tolist(),
                m3_imaginary=float(np.mean([means[name+'__m3_imag'].mean() for name in names])))
        except ValueError as exc:
            out[star] = dict(status='unresolved_moment_matrix', reason=str(exc),
                             moments_unnormalized=average.tolist())
    return dict(independent_chains=n, response_stars=out,
        interpretation='two-dimensional source-accessible Ritz quadrature; no m4 or physical residual certification; approximate-parent relative energies are not gaps')


def run(path: Path) -> dict:
    cfg = json.loads(path.read_text()); out = ROOT/cfg['output']
    source = Path(cfg['source_readout'])
    previous = read_sealed(source/'registration.json')
    if file_sha256(source/'registration.json') != cfg['source_registration_sha256']:
        raise ValueError('sealed original readout registration changed')
    proposal = source/'proposal.npz'
    if file_sha256(proposal) != previous['proposal_sha256']:
        raise ValueError('previous Q1 proposal changed')
    parent = Path(previous['config']['parent']); step = previous['config']['step']
    frozen, training, identity_path = cached_quality_identity(parent, step, allow_registered_milestone=True)
    spec = frozen['config']; model = validate_spec(spec)
    if (model.geometry != 'triangular' or model.rows != 6 or model.columns != 6
            or spec.get('coordinate_convention', 'physical120') != 'physical120'
            or not spec.get('spin_inversion_even', False) or model.target_m2 != 0):
        raise ValueError('registered physical120 N36 Sz0 flip-even momentum bank required')
    checkpoint = parent/'quality'/f'step{step:05d}.pkl'
    if file_sha256(checkpoint) != previous['config']['checkpoint_sha256']:
        raise ValueError('original physical parent changed')
    saved = load_checkpoint(checkpoint); params = saved['params']['model']; identity = pytree_sha256(params)
    if (identity != training['final_params_sha256'] or identity != previous['params_sha256']
            or checkpoint_projection(saved['metadata']) != 'gamma_a1'):
        raise ValueError('physical full-A1 parameter identity changed')
    workflow = Path(__file__).with_name('spin_spatial_vit_quality.py')
    old = ast.parse((parent/'source_snapshots'/workflow.relative_to(ROOT)).read_text()); new = ast.parse(workflow.read_text())
    for name in ('validate_spec', 'model_log'):
        pick = lambda tree: next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        if ast.dump(pick(old), include_attributes=False) != ast.dump(pick(new), include_attributes=False):
            raise ValueError('physical amplitude AST changed')
    for name in ('su2_nqs/models/fixed_sz_spatial_vit.py', 'su2_nqs/models/literature_spatial_vit.py',
                 'projects/clebsch_tree_nqs/workflows/spin.py'):
        if file_sha256(ROOT/name) != frozen['source_manifest'][name]:
            raise ValueError('physical amplitude dependency changed')
    with np.load(proposal) as archive:
        all_states, all_ids = archive['states'], archive['chain_ids']
    selected = np.sort(np.concatenate([np.flatnonzero(all_ids == c)[
        np.arange(cfg['records_per_chain'])*cfg['record_stride']] for c in np.unique(all_ids)]))
    states, chains = all_states[selected], all_ids[selected]
    if len(states) != cfg['records'] or jax.default_backend() != cfg['backend']:
        raise ValueError('explicit allocated backend and registered records required')
    inputs = {str(p):file_sha256(p) for p in (path, proposal, source/'registration.json', checkpoint,
                                           parent/'registration.json', identity_path)}
    sources_manifest = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    registration = dict(config=cfg, inputs=inputs, sources=sources_manifest, params_sha256=identity,
        full_A1=True, strict_SU2=False, connected_spin_source_mean=0.,
        mean_zero_reason='Sz0 plus exact even global spin inversion; all local vector means vanish',
        records=len(states), independent_chains=len(np.unique(chains)), new_training=0, new_Born_samples=0,
        reused_auxiliary_Q1_records=True, m4_not_measured=True)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'registration.json').exists():
        prior = read_sealed(out/'registration.json')
        if any(prior[key] != value for key,value in registration.items()):
            raise ValueError('immutable response-Krylov registration changed')
    else:
        for name in sources_manifest:
            target = out/'source_snapshots'/name; target.parent.mkdir(parents=True, exist_ok=True)
            copy2(ROOT/name, target)
        sealed_json(out/'registration.json', registration)
    phase = {name: np.exp(2j*np.pi*(np.arange(36)//6*k[0]+np.arange(36)%6*k[1]))
             for name,k in dict(K=(1/3,1/3), minusK=(-1/3,-1/3), M1=(.5,0), M2=(0,.5), M3=(.5,.5)).items()}
    log_fn, images = model_log(spec, model, _groups(model.geometry, 6, 6)['gamma_a1'])
    kernel = jax.jit(jax.vmap(log_fn, in_axes=(None,0)))
    bonds = _bonds(model.geometry, 6, 6, spec['j2_over_j1'])
    start = perf_counter(); calls = 0; batch = cfg['query_batch']; chunks = {}
    h_rows = 0; total_queries = 0

    def check():
        if perf_counter()-start > cfg['seconds_cap']:
            raise TimeoutError('registered response budget exhausted; completed chunks retained')

    def query(x):
        nonlocal calls
        values = []
        for i in range(0, len(x), batch):
            check(); block = x[i:i+batch]
            padded = np.pad(block, ((0,batch-len(block)),(0,0)), mode='edge')
            values.append(np.asarray(kernel(params,jnp.asarray(padded)))[:len(block)])
            calls += len(block)
        return np.concatenate(values)

    try:
        size = cfg['centers_per_chunk']
        for offset in range(0,len(states),size):
            check(); chunk = out/f'chunk{offset:06d}.npz'; meta = chunk.with_suffix('.json')
            if chunk.exists():
                record = read_sealed(meta)
                if file_sha256(chunk) != record['raw_sha256']:
                    raise ValueError('saved response moment chunk changed')
                with np.load(chunk) as archive: values = {key:archive[key] for key in archive.files}
            else:
                tick = perf_counter(); plan = two_action_plan(states[offset:offset+size], bonds)
                values = third_source_moments(plan,query(plan.configurations),phase,
                                              energy_shift=previous['config']['energy_shift'])
                np.savez_compressed(chunk,**values)
                record=sealed_json(meta,dict(raw_sha256=file_sha256(chunk),unique_queries=len(plan.configurations),
                    first_H_rows=len(plan.first_h_centers),
                    full_H_row_applications=6*(len(plan.first_h_centers)+2*len(plan.center_indices)),
                    seconds=perf_counter()-tick,records=len(plan.center_indices)))
            h_rows += record['full_H_row_applications'];total_queries += record['unique_queries']
            for key,value in values.items(): chunks.setdefault(key,[]).append(value)
            atomic_json(out/'progress.json',dict(completed_records=offset+len(next(iter(values.values()))),
                seconds=perf_counter()-start, projected_queries_this_run=calls))
        values = {key:np.concatenate(value) for key,value in chunks.items()}
        np.savez_compressed(out/'moments.npz',**values,chain_ids=chains,selected_record_indices=selected)
        for name,digest in inputs.items():
            if file_sha256(Path(name)) != digest: raise ValueError('input changed during readout')
        return sealed_json(out/'results.json',dict(status='completed',**summarize(values,chains),
            raw_sha256=file_sha256(out/'moments.npz'),image_count=images,
            seconds_this_run=perf_counter()-start,projected_queries_this_run=calls,
            full_H_actions=h_rows,total_projected_queries=total_queries,
            two_action_center_evaluations=12*len(states),
            work_definition='full_H_actions counts every offcenter Hphi row as well as both center H rows for parent and five sources',
            m4_not_measured=True,new_training=0,new_Born_samples=0))
    except TimeoutError as exc:
        return sealed_json(out/'incomplete.json',dict(status='budget_limited',reason=str(exc),
            seconds_this_run=perf_counter()-start,projected_queries_this_run=calls))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().config),indent=2))
