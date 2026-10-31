"""Bounded same-parent spatial spin/dimer/chirality correlations on cached chains.

Few predeclared distances, all translations, full physical A1 amplitudes.
No new chains, training, H calls, SU2 factor-three substitution or dense RDM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.spin_correlations import (
    spin_dot_connections, dimer_product_connections, chirality_dot_connections,
)
from su2_nqs.benchmarks.spatial_vit_physics import scalar_structure_query_plan, scalar_structure_records
from su2_nqs.benchmarks.chirality_covariance import (
    chirality_product_connections, cluster_jackknife, jackknife_error,
)
from su2_nqs.benchmarks.replica_spin_response import operator_action_over_sample
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from .spin_chirality_covariance import ROOT, load_frozen_inputs

jax.config.update('jax_enable_x64', True)


def run(config: Path, *, preflight_only: bool = False) -> dict:
    """Measure fixed bond orientation and row separations, retaining chain errors."""
    request = json.loads(config.read_text())
    settings = json.loads((ROOT/request['base_config']).read_text())
    settings.update({k: v for k, v in request.items() if k != 'base_config'})
    params, batch_fn, cfg, streams, identity = load_frozen_inputs(settings)
    snapshots = np.asarray(settings['snapshot_indices'])
    distances = tuple(settings['row_distances'])
    scalar_chirality = bool(settings.get('include_rotational_scalar_chirality', False))
    structure = bool(settings.get('include_structure_factors', False))
    momenta = settings.get('structure_momenta', settings.get('momenta', {}))
    if structure and set(momenta) != {'Gamma','K','minus_K','M1','M2','M3'}:
        raise ValueError('structure readout requires Gamma, K/-K and all three M points')
    if (cfg.geometry != 'triangular' or cfg.rows != 6 or cfg.columns != 6 or snapshots.ndim != 1 or snapshots.dtype.kind not in 'iu'
            or not len(snapshots) or np.any(np.diff(snapshots) <= 0) or snapshots[0] < 0
            or snapshots[-1] >= streams[0].shape[0] or distances != (1, 2, 3)):
        raise ValueError('registered triangular6x6, increasing cached indices, distances1,2,3 required')
    if preflight_only:
        return identity
    if jax.default_backend() != 'gpu':
        raise RuntimeError('registered A100 readout requires GPU')
    output = Path(settings['output'])
    if output.exists():
        raise FileExistsError('choose an unused immutable result directory')
    output.mkdir(parents=True)
    manifest = transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration = dict(settings=settings, identity=identity, sources=manifest,
        config_sha256=file_sha256(config), new_Born_samples=0, new_training=0, new_H_calls=0,
        runtime=dict(jax_version=jax.__version__, numpy_version=np.__version__,
                     x64_enabled=bool(jax.config.x64_enabled), device_kinds=[d.device_kind for d in jax.devices()]),
        ordinary_operators=['S_i dot S_j', 'connected (S_i dot S_i+x)(S_j dot S_j+x)',
                            'connected kappa_z(i,i+x) kappa_z(j,j+x)'],
        translation_average='all36 physical origins; equal weight per original cached chain observation',
        boundary='triangular6x6 PBC; no thermodynamic correlation-length fit',
        chirality_scalar_definition='Disjoint bonds: (Si.Sk)(Sj.Sl)-(Si.Sl)(Sj.Sk); no factor3 assumption'if scalar_chirality else None,
        structure_definition='S(k)=sum_ij exp[2pi*i*k.(ri-rj)]<Si.Sj>/N; self term3/4 retained; physical coordinates(row,column)',
        exact_zero_one_points='Fixed Sz and actual even spin inversion imply <S_i>=0 and <bold_kappa>=0; dimer means measured.')
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(output/'registration.json', registration)
    start = perf_counter(); queries = 0

    def query(bits):
        nonlocal queries
        unique, inverse = np.unique(bits, axis=0, return_inverse=True)
        values = []; size = settings['query_batch']
        for offset in range(0, len(unique), size):
            if perf_counter()-start > settings['wall_seconds']:
                raise TimeoutError('fixed distance-readout cap; raw completed chunks retained')
            chunk = unique[offset:offset+size]
            padded = np.pad(chunk, ((0, size-len(chunk)), (0, 0)), mode='edge')
            values.append(np.asarray(batch_fn(params, jnp.asarray(padded)))[:len(chunk)])
            queries += len(chunk)
        return np.concatenate(values)[inverse]

    x = np.concatenate([s[snapshots].reshape(-1, cfg.n_sites) for s in streams])
    nchain = streams[0].shape[1]
    labels = np.concatenate([np.tile(np.arange(nchain)+i*nchain, len(snapshots)) for i in range(2)])
    # For each distance: spin pair, dimer product, left/right dimer means, kappa_z product.
    columns = 6 if scalar_chirality else 5
    records = np.zeros((len(x), len(distances), columns), complex)
    structure_scalar = np.zeros((len(x),len(momenta)),complex) if structure else None
    structure_zz = np.zeros((len(x),len(momenta))) if structure else None
    centers = settings['center_chunk']
    for offset in range(0, len(x), centers):
        bits = x[offset:offset+centers]; denominator = query(bits)
        connectors = []
        for r in range(6):
            for c in range(6):
                left = (r*6+c, r*6+(c+1)%6)
                for distance in distances:
                    right = (((r+distance)%6)*6+c, ((r+distance)%6)*6+(c+1)%6)
                    connectors.extend([spin_dot_connections(bits, (left[0], right[0])),
                        dimer_product_connections(bits, left, right), spin_dot_connections(bits, left),
                        spin_dot_connections(bits, right), chirality_product_connections(bits, (left,), (right,))])
                    if scalar_chirality:connectors.append(chirality_dot_connections(bits,left,right))
        lengths = [len(t) for t, _, _ in connectors]
        ends = np.cumsum([0]+lengths)
        plan=scalar_structure_query_plan(bits)if structure else None
        targets=[t for t, _, _ in connectors]
        if structure:targets.append(plan.configurations)
        logs = query(np.concatenate(targets))
        local = np.column_stack([operator_action_over_sample(logs[ends[k]:ends[k+1]], s, a, denominator)
                                 for k, (_, s, a) in enumerate(connectors)])
        records[offset:offset+len(bits)] = local.reshape(len(bits), 36, len(distances), columns).mean(axis=1)
        if structure:
            coords=np.array([(r,c)for r in range(cfg.rows)for c in range(cfg.columns)])
            ss,zz=scalar_structure_records(plan,logs[ends[-1]:],coords,momenta)
            structure_scalar[offset:offset+len(bits)]=ss;structure_zz[offset:offset+len(bits)]=zz
        np.savez_compressed(output/'partial_records.npz', records=records[:offset+len(bits)],
            chain_labels=labels[:offset+len(bits)], completed_centers=offset+len(bits),
            source_registration_sha256=registration['sha256'])
        atomic_json(output/'progress.json', dict(completed_centers=offset+len(bits), total_centers=len(x),
            projected_amplitude_queries=queries, wall_seconds=perf_counter()-start))
    mean, leave = cluster_jackknife(records, labels)

    def connected(values):
        entries=[values[..., 0].real,
                         (values[..., 1]-values[..., 2]*values[..., 3]).real,
                         values[..., 4].real]
        if scalar_chirality:entries.append(values[...,5].real)
        return np.stack(entries,axis=-1)

    result = dict(source_registration_sha256=registration['sha256'],
        row_distances=distances, connected=connected(mean).tolist(),
        connected_SE=jackknife_error(connected(leave)).tolist(),
        raw_moments_real=mean.real.tolist(), raw_moments_imaginary=mean.imag.tolist(),
        raw_imaginary_SE=jackknife_error(leave.imag).tolist(),
        observable_order=['scalar_spin', 'scalar_dimer_connected', 'vector_chirality_z_connected'],
        energy_audit=identity['energy_audit'], independent_chains=len(leave), samples=len(x),
        wall_seconds=perf_counter()-start, projected_amplitude_queries=queries,
        strict_SU2_assumed=False, isotropic_factor3=False, phase_classification=False,
        new_Born_samples=0, new_H_calls=0, new_training=0, equilibrium_certified=False)
    if scalar_chirality:result['observable_order'].append('rotational_scalar_chirality_connected')
    extra={}
    if structure:
        sm,sl=cluster_jackknife(structure_scalar,labels);zm,zl=cluster_jackknife(structure_zz,labels)
        result['structure_factors']={name:dict(scalar=dict(value=float(sm[i].real),imaginary=float(sm[i].imag),
            chain_SE=float(jackknife_error(sl.real)[i]),imaginary_SE=float(jackknife_error(sl.imag)[i])),
            zz=dict(value=float(zm[i]),chain_SE=float(jackknife_error(zl)[i])))for i,name in enumerate(momenta)}
        result['total_spin_squared']=dict(value=float(sm[list(momenta).index('Gamma')].real*cfg.n_sites),
            chain_SE=float(jackknife_error(sl.real)[list(momenta).index('Gamma')]*cfg.n_sites))
        extra=dict(structure_scalar=structure_scalar,structure_zz=structure_zz,momentum_order=np.array(list(momenta)))
    np.savez_compressed(output/'records.npz', records=records, chain_labels=labels,**extra)
    result['raw_sha256'] = file_sha256(output/'records.npz')
    if settings.get('legacy_distance_reference'):
        reference=Path(settings['legacy_distance_reference'])
        if file_sha256(reference/'records.npz')!=settings['legacy_raw_sha256']:
            raise ValueError('frozen legacy distance records changed')
        oldreg=json.loads((reference/'registration.json').read_text())
        if (oldreg['settings']['checkpoint_sha256']!=settings['checkpoint_sha256']
                or oldreg['settings']['streams']!=settings['streams']):
            raise ValueError('legacy reference is not this exact parent and chains')
        with np.load(reference/'records.npz')as z:
            if not np.array_equal(z['chain_labels'],labels):raise ValueError('legacy chain labels changed')
            replay=float(np.max(abs(records[...,:5]-z['records'])))
        atomic_json(output/'legacy_replay.json',dict(max_error=replay,tolerance=2e-11,passed=replay<=2e-11))
        if replay>2e-11:raise ValueError('unchanged physical distance records failed replay')
        result['legacy_distance_max_replay_error']=replay
    atomic_json(output/'results.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, preflight_only=args.preflight_only), indent=2))
