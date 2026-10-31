"""One fixed-parent cross-chain calibration; no training or new sampling.

Reuse the sealed physical decoder, projected full-H kernels and cached chains.
Only new tangent H products are queried. No previous numerical source is edited.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.schur_metric_retraction import split_physical_reduced_pair
from su2_nqs.models.schur_two_site import two_site_layout, join_reduced_pair, make_two_site_features
from su2_nqs.vmc.cross_chain_force import cross_chain_calibration
from su2_nqs.vmc.projected_spin_linear import make_projected_linear_products
from su2_nqs.vmc.retraction_transport import retracted_real_columns
from . import spin_schur_transport as frozen

base = frozen.base
CONFIG = base.ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_force_calibration.json'


def verify(reg: dict) -> None:
    for path, digest in {**reg['sources'], **reg['inputs']}.items():
        if file_sha256(base.ROOT/path) != digest:
            raise ValueError('sealed source/input changed: '+path)


def prepare(spec: dict, output: Path) -> None:
    """Seal a single predeclared diagnostic BEFORE querying its tangent H."""
    if (spec['edges'] != [[2,0],[6,3],[6,5]] or spec['records_per_chain'] != 1
            or spec['fold_rule'] != 'chain_id_parity_balanced_in_each_saved_stream'
            or spec['metric_damping'] != .1 or spec['maximum_new_H_equivalent_columns'] != 768):
        raise ValueError('single fixed-parent calibration budget required')
    cache = base.ROOT/spec['cache_source']
    old_reg = base.configuration(cache/'registration.json')
    verify(old_reg)
    setup_spec = base.configuration(base.ROOT/spec['setup_config'])
    if setup_spec != old_reg['config']:
        raise ValueError('must reuse precisely the sealed setup')
    _, _, parent, _, _, _, _, _ = frozen.setup(setup_spec)
    paths = [CONFIG, base.ROOT/spec['setup_config']]
    paths += [cache/name for name in ('registration.json','known.npz','parent_known.npz',
                                      'audit_registration.json','audit0.npz','audit1.npz','results.json')]
    inputs = dict(old_reg['inputs'])
    inputs.update({str(p.relative_to(base.ROOT)): file_sha256(p) for p in paths})
    reg = dict(config=spec, runtime=numerical_runtime_identity(), parent_sha256=pytree_sha256(parent),
        inputs=inputs, sources=transitive_local_source_manifest((Path(__file__),), root=base.ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    reg['registration_sha256'] = canonical_sha256(reg)
    output.mkdir(exist_ok=False)
    atomic_json(output/'registration.json', reg)
    samples = []
    for stream in range(2):
        data = base.loaded(cache/f'audit{stream}.npz')
        groups, first = np.unique(data['groups'], return_index=True)
        if len(groups) != 32 or not np.array_equal(groups, np.arange(32)):
            raise ValueError('expected32 saved chains per stream')
        # First occurrence per ID: records are interleaved, not groups of four.
        samples.append(dict(states=data['states'][first], q=data['q'][first],
            psi=data['psi'][first,1], hpsi=data['hpsi'][first,1], carrier=data['psi'][first,0],
            groups=groups+32*stream, source_stream=np.full(32,stream), source_row=first))
    selected = {k: np.concatenate([v[k] for v in samples]) for k in samples[0]}
    selected['folds'] = selected['groups'] % 2
    np.savez_compressed(output/'diagnostic.npz', **selected)
    atomic_json(output/'prepared.json', dict(registration_sha256=reg['registration_sha256'],
        diagnostic_sha256=file_sha256(output/'diagnostic.npz'), chains=64, records=64,
        known_cache_not_untouched=True, first_record_per_chain=True, parent_sha256=reg['parent_sha256'],
        new_samples=0, parameter_updates=0, new_H_calls=0))
    print('sealed ONE calibration: fixed previous4,64 cached chains,3 edges; no state updates', flush=True)


def calibrate(spec: dict, reg: dict, output: Path) -> None:
    """At most768 tangent-H column-centres; checkpoint diagnostics, not states."""
    if (output/'results.json').exists():
        raise ValueError('once-only calibration already complete; do not repeat or train')
    start = perf_counter()
    setup_spec = base.configuration(base.ROOT/spec['setup_config'])
    cfg, reference, parent, _, bonds, projector, _, units = frozen.setup(setup_spec)
    if pytree_sha256(parent) != reg['parent_sha256']:
        raise ValueError('parent changed')
    cache = base.ROOT/spec['cache_source']
    known, saved = base.loaded(cache/'known.npz'), base.loaded(cache/'parent_known.npz')
    selected = base.loaded(output/'diagnostic.npz')
    if file_sha256(output/'diagnostic.npz') != base.configuration(output/'prepared.json')['diagnostic_sha256']:
        raise ValueError('selected diagnostic records changed')
    x, q, hi = known['states'], known['q'], known['H_indices']
    scale = units['residual_scale']
    projected, _, _ = base.bank_functions(cfg, projector, bonds, setup_spec)
    reference_amplitudes = base.batched(lambda z: projected(reference,z), selected['states'], 4)
    history, total_H = [], 0
    for index, pair in enumerate(spec['edges']):
        json_path, npz_path = output/f'edge{index}.json', output/f'edge{index}.npz'
        if json_path.exists():
            row = base.configuration(json_path)
            if row['parent_sha256'] != reg['parent_sha256'] or file_sha256(npz_path) != row['columns_sha256']:
                raise ValueError('diagnostic checkpoint changed')
            history.append(row); total_H += row['new_H_equivalent_columns']; continue
        tick = perf_counter(); layout = two_site_layout(cfg,*pair)
        joined = join_reduced_pair(parent,layout)
        lp, la, fp = make_projected_linear_products(make_two_site_features(layout), projector, bonds,
                                                   chunk_size=setup_spec['linear_H_chunk'])
        lp, la, fp = map(jax.jit, (lp, la, fp))
        f = base.batched(lambda z: fp(parent,z), x, 2)
        replay = float(np.max(abs(scale*(f@joined-saved['phi']))))
        if replay > 2e-8: raise FloatingPointError('fixed-parent known feature replay failed')
        raw, metric = base.search_directions(f[hi], saved['psi'][hi], saved['hpsi'], q[hi], scale, setup_spec)
        def retract(z):
            candidate, _ = split_physical_reduced_pair(z,parent,layout,
                                                       metric_cutoff=setup_spec['retraction_metric_cutoff'])
            return join_reduced_pair(candidate,layout)
        columns, transport = retracted_real_columns(joined,raw,retract,f,saved['psi'],q,
            amplitude_scale=scale, probe_fs=setup_spec['probe_fs'],
            maximum_relative_defect=setup_spec['maximum_transport_defect'])
        if not 1 <= columns.shape[1] <= 4:
            raise FloatingPointError('no stable physical tangent or column budget exceeded')
        if total_H+len(selected['q'])*columns.shape[1] > spec['maximum_new_H_equivalent_columns']:
            raise RuntimeError('fixed full-H budget exceeded')
        fc = base.batched(lambda z: fp(parent,z), selected['states'], 2)
        parent_replay = float(np.max(abs(selected['carrier']+scale*(fc@joined-reference_amplitudes)-selected['psi'])))
        if parent_replay > 2e-8: raise FloatingPointError('fixed-parent diagnostic amplitude replay failed')
        values, actions, work = [], [], []
        for i in range(0, len(selected['q']), setup_spec['H_microbatch']):
            prepared = lp(jnp.asarray(selected['states'][i:i+setup_spec['H_microbatch']]))
            h, a, count, padded = la(parent,jnp.asarray(columns),prepared)
            values.append(np.asarray(a)); actions.append(np.asarray(h))
            work.append(dict(unique=int(count), padded=int(padded))); base.resources(spec,start)
        t, ht = scale*np.concatenate(values), scale*np.concatenate(actions)
        linear_replay = float(np.max(abs(t-scale*fc@columns)))
        if linear_replay > 2e-8: raise FloatingPointError('tangent full-H amplitude replay failed')
        report = cross_chain_calibration(selected['psi'],selected['hpsi'],t,ht,selected['q'],
            selected['groups'],selected['folds'],n_sites=36,damping=spec['metric_damping'],
            finite_difference_step=spec['finite_difference_step'])
        if max(max(v['linear_state_finite_difference_error'].values()) for v in report['folds']) > 2e-8:
            raise FloatingPointError('complete two-sided finite-difference check failed')
        np.savez_compressed(npz_path,columns=columns,tangent=t,h_tangent=ht)
        total_H += len(selected['q'])*columns.shape[1]
        row = dict(index=index,pair=pair,parent_sha256=reg['parent_sha256'],metric=metric,transport=transport,
            known_feature_replay_error=replay,diagnostic_parent_replay_error=parent_replay,
            linear_H_amplitude_replay_error=linear_replay,calibration=report,
            columns_sha256=file_sha256(npz_path),new_H_equivalent_columns=len(selected['q'])*columns.shape[1],
            linear_query_work=work,seconds=perf_counter()-tick,peak_rss_bytes=base.peak_rss_bytes())
        atomic_json(json_path,row); history.append(row)
        atomic_json(output/'checkpoint.json',dict(registration_sha256=reg['registration_sha256'],
            parent_sha256=reg['parent_sha256'],completed_edges=index+1,new_H_equivalent_columns=total_H,
            parameter_updates=0,new_samples=0,diagnostic_only=True))
        print('fixed-parent edge',pair,'held E/R',[(r['held']['energy'],r['held']['variance']) for r in report['folds']],
              'FS transfer',[r['fs_speed_transfer'] for r in report['folds']],flush=True)
        jax.clear_caches(); base.resources(spec,start)
    verify(reg)
    if pytree_sha256(parent) != reg['parent_sha256']: raise ValueError('parent mutated')
    result = dict(registration_sha256=reg['registration_sha256'],parent_sha256=reg['parent_sha256'],
        history=history,new_H_equivalent_columns=total_H,new_samples=0,parameter_updates=0,
        gpu_hours=0,ED_calls=0,seconds=perf_counter()-start,peak_rss_bytes=base.peak_rss_bytes(),
        useful_point_edges=[v['pair'] for v in history if v['calibration']['held_joint_point_descent_both']],
        conditional_two_SE_edges=[v['pair'] for v in history if v['calibration']['held_energy_two_SE_nonworse_variance_both']],
        not_an_untouched_audit=True,stationarity_certified=False,
        route_decision='stop_after_one_calibration_no_parameter_update_no_training_permission')
    result['result_sha256'] = canonical_sha256(result)
    atomic_json(output/'results.json',result)
    print('one calibration COMPLETE; point edges',result['useful_point_edges'],
          'conditional2SE edges',result['conditional_two_SE_edges'],'NO UPDATE',flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('stage',choices=('prepare','calibrate'))
    args = parser.parse_args(); spec = base.configuration(CONFIG); output = base.ROOT/spec['output']
    if args.stage == 'prepare': prepare(spec,output); return
    reg = base.configuration(output/'registration.json')
    if reg['config'] != spec: raise ValueError('sealed configuration changed')
    verify(reg); calibrate(spec,reg,output)


if __name__ == '__main__': main()
