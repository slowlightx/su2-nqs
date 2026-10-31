"""Frozen scalar witnesses and disjoint spatial correlations on saved ViT states.

Reuses complete Born audit chains and the original full-A1 amplitude. No
optimization or operator fitting occurs here. Work is checkpointed in small
immutable chunks. A completed local readout precedes the bounded distance scan.
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

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.regional_density import disjoint_regional_query_plan, disjoint_born_operator_moments
from .spin_spatial_vit_quality import (
    ROOT, read_sealed, sealed_json, validate_spec, model_log, cached_quality_identity,
    checkpoint_projection, _groups,
)


def summarize(values: dict[str, np.ndarray], groups: np.ndarray) -> dict:
    """Whole-chain uncertainty and paired nonlinear connected-correlation error."""
    ids = np.unique(groups)
    chains = {k: np.array([v[groups == c].mean() for c in ids]) for k, v in values.items()}
    out = {k: dict(value=float(v.mean()), chain_SE=float(v.std(ddof=1)/np.sqrt(len(v))))
           for k, v in chains.items()}
    for key in values:
        if not key.endswith('__AB'):
            continue
        name = key[:-4]
        a, b, ab = (chains[name+'__'+s] for s in ('A', 'B', 'AB'))
        connected = ab.mean()-a.mean()*b.mean()
        n = len(ids)
        ja, jb, jab = ((n*v.mean()-v)/(n-1) for v in (a, b, ab))
        jack = jab-ja*jb
        # Delete-chain correction removes leading product-of-means bias.
        corrected = n*connected-(n-1)*jack.mean()
        out[name+'__connected'] = dict(value=float(corrected), plugin_value=float(connected),
            chain_SE=float(np.sqrt((n-1)*np.mean((jack-jack.mean())**2))),
            method='whole-chain jackknife bias-corrected covariance', independent_chains=n)
    return out


def run(path: Path) -> dict:
    cfg = json.loads(path.read_text()); out = ROOT/cfg['output']
    parent = Path(cfg['parent']); step = cfg['step']
    frozen, training, identity_path = cached_quality_identity(parent, step, allow_registered_milestone=True)
    spec = frozen['config']; model = validate_spec(spec)
    checkpoint = parent/'quality'/f'step{step:05d}.pkl'
    if file_sha256(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('frozen endpoint changed')
    saved = load_checkpoint(checkpoint); params = saved['params']['model']
    identity = pytree_sha256(params)
    if identity != training['final_params_sha256'] or checkpoint_projection(saved['metadata']) != 'gamma_a1':
        raise ValueError('complete A1 endpoint identity differs')
    # Do not silently change the state definition while introducing a readout.
    workflow = Path(__file__).with_name('spin_spatial_vit_quality.py')
    old = ast.parse((parent/'source_snapshots'/workflow.relative_to(ROOT)).read_text())
    new = ast.parse(workflow.read_text())
    for name in ('validate_spec', 'model_log'):
        pick = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(pick(old), include_attributes=False) != ast.dump(pick(new), include_attributes=False):
            raise ValueError('original amplitude AST changed')
    for name in ('su2_nqs/models/fixed_sz_spatial_vit.py', 'su2_nqs/models/literature_spatial_vit.py',
                 'projects/clebsch_tree_nqs/workflows/spin.py'):
        if file_sha256(ROOT/name) != frozen['source_manifest'][name]:
            raise ValueError('original amplitude dependency changed')
    artifact = ROOT/cfg['operators']
    if file_sha256(artifact) != cfg['operators_sha256']:
        raise ValueError('frozen operators changed')
    with np.load(artifact) as z:
        ops = {name: z[key] for name, key in cfg['operator_keys'].items()}
    inputs = {str(p): file_sha256(p) for p in (path, checkpoint, artifact, parent/'registration.json', identity_path)}
    audit_dir = parent/'audits'/f'step{step:05d}'
    streams = []
    for original_seed in spec['audit']['seeds']:
        p = audit_dir/f'stream{original_seed+1009*step}.npz'
        meta = read_sealed(p.with_suffix('.json'))
        if (file_sha256(p) != meta['raw_sha256'] or meta['params_sha256'] != identity or
                meta['registration_sha256'] != frozen['result_sha256']):
            raise ValueError('Born audit cache changed')
        with np.load(p) as z:
            streams.append(np.array(z['states']))
        for item in (p, p.with_suffix('.json')):
            inputs[str(item)] = file_sha256(item)
    if jax.default_backend() != cfg['backend']:
        raise RuntimeError('explicit allocated backend required')
    out.mkdir(parents=True, exist_ok=True)
    registration = dict(config=cfg, inputs=inputs, params_sha256=identity, full_A1=True,
        strict_SU2=False, reused_Born_chains=True, new_training=0, new_samples=0, new_H_calls=0)
    if (out/'registration.json').exists():
        previous = read_sealed(out/'registration.json')
        for k, v in registration.items():
            if previous[k] != v:
                raise ValueError('immutable registration changed')
    else:
        sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
        for name in sources:
            target = out/'source_snapshots'/name; target.parent.mkdir(parents=True, exist_ok=True)
            copy2(ROOT/name, target)
        sealed_json(out/'registration.json', dict(**registration, sources=sources))
    log_fn, images = model_log(spec, model, _groups(model.geometry, model.rows, model.columns)['gamma_a1'])
    kernel = jax.jit(jax.vmap(log_fn, in_axes=(None, 0)))
    start = perf_counter(); batch = cfg['query_batch']; calls = 0
    def check():
        if perf_counter()-start > cfg['seconds_cap']:
            raise TimeoutError('fixed readout budget exhausted; completed chunks preserved')
    def query(bits):
        nonlocal calls
        values = []
        for i in range(0, len(bits), batch):
            check(); block = bits[i:i+batch]
            padded = np.pad(block, ((0, batch-len(block)), (0, 0)), mode='edge')
            values.append(np.asarray(kernel(params, jnp.asarray(padded)))[:len(block)])
            calls += len(block)
        return np.concatenate(values)
    def measure(name, regions, snapshots):
        destination = out/name; destination.mkdir(exist_ok=True)
        states = np.concatenate([s[snapshots].reshape(-1, model.n_sites) for s in streams])
        groups = np.concatenate([np.tile(np.arange(s.shape[1])+i*s.shape[1], len(snapshots))
                                 for i, s in enumerate(streams)])
        collected = {}; size = cfg['centers_per_chunk']
        for offset in range(0, len(states), size):
            check(); chunk = destination/f'chunk{offset:06d}.npz'; meta = chunk.with_suffix('.json')
            if chunk.exists():
                if file_sha256(chunk) != read_sealed(meta)['raw_sha256']:
                    raise ValueError('saved readout chunk changed')
                with np.load(chunk) as z:
                    values = {k: np.array(z[k]) for k in z.files}
            else:
                tick = perf_counter()
                plan = disjoint_regional_query_plan(states[offset:offset+size], regions)
                values = disjoint_born_operator_moments(plan, query(plan.configurations), ops)
                np.savez_compressed(chunk, **values)
                sealed_json(meta, dict(raw_sha256=file_sha256(chunk), offset=offset,
                    records=plan.records, unique_queries=len(plan.configurations), seconds=perf_counter()-tick))
            for k, v in values.items():
                collected.setdefault(k, []).append(v)
            atomic_json(out/'progress.json', dict(phase=name, completed_records=offset+len(next(iter(values.values()))),
                total_records=len(states), seconds=perf_counter()-start, projected_queries=calls))
        values = {k: np.concatenate(v) for k, v in collected.items()}
        np.savez_compressed(destination/'moments.npz', **values, chain_ids=groups)
        result = dict(regions=regions, snapshots=snapshots, records=len(states), independent_chains=len(np.unique(groups)),
            moments=summarize(values, groups), raw_sha256=file_sha256(destination/'moments.npz'),
            interpretation='frozen linear scalar-operator moments; no entanglement-spectrum or QSL certification')
        sealed_json(destination/'results.json', result)
        return result
    base = cfg['region']
    try:
        local = measure('local', [base], cfg['local_snapshots'])
        gate = local['moments'][cfg['distance_gate_operator']+'__A']
        passed = gate['value'] > 2*gate['chain_SE']
        distances = {}
        if passed:
            for dr, dc in cfg['displacements']:
                translated = [((s//model.columns+dr)%model.rows)*model.columns+(s%model.columns+dc)%model.columns for s in base]
                distances[f'dr{dr}_dc{dc}'] = measure(f'dr{dr}_dc{dc}', [base, translated], cfg['distance_snapshots'])
        for p, digest in inputs.items():
            if file_sha256(Path(p)) != digest:
                raise ValueError('parent/input changed during readout')
        return sealed_json(out/'results.json', dict(status='completed', local=local, distances=distances,
            distance_gate=gate, distance_gate_passed=passed,
            gate_scope='resolved frozen SU2 transfer channel enables exploratory spatial scan, not a QSL uniqueness gate',
            image_count=images, seconds=perf_counter()-start, projected_amplitude_queries=calls,
            finite_torus=True, correlation_length_fit=False, new_training=0, new_samples=0, new_H_calls=0))
    except TimeoutError as exc:
        return sealed_json(out/'incomplete.json', dict(status='budget_limited', reason=str(exc),
            seconds=perf_counter()-start, projected_amplitude_queries=calls, partial_chunks_preserved=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run(args.config), indent=2))
