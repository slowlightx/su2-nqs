"""Frozen NQS C2/M comparison, using archived Born chains and the SAME operators.

No new training, chains, H calls, or dense regional RDM. C2 controls the
Renyi reweighting of C before attributing an M/C difference to transitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.chirality_covariance import PAIRS, chirality_product_connections
from su2_nqs.benchmarks.replica_spin_response import operator_action_over_sample
from su2_nqs.benchmarks.schmidt_covariance import (
    renyi2_product_numerator, compare_renyi_matched_covariances,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from .spin_chirality_covariance import ROOT, load_frozen_inputs

jax.config.update('jax_enable_x64', True)


def run(config: Path, *, preflight_only: bool = False, swap_probe_only: bool = False) -> dict:
    """Measure six symmetrized products at fixed prefix18, with full seals."""
    request = json.loads(config.read_text())
    base = ROOT/request['base_config']
    settings = json.loads(base.read_text())
    settings.update({key: value for key, value in request.items() if key != 'base_config'})
    params, batch_fn, cfg, streams, identity = load_frozen_inputs(settings)
    if preflight_only:
        return identity
    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU allocation required')
    if (set(settings['regions']) != {'prefix18'}
            or settings['regions']['prefix18'] != list(range(18))
            or settings['relative_replica_orbit'] != 'complete_x_translations_times_spin_flip'):
        raise ValueError('this registration fixes prefix18 and the archived relative orbit')
    reference = Path(settings['ordinary_reference'])
    if file_sha256(reference/'registration.json') != settings['ordinary_registration_file_sha256']:
        raise ValueError('ordinary control registration changed')
    reg = json.loads((reference/'registration.json').read_text())
    if (reg['settings']['checkpoint_sha256'] != settings['checkpoint_sha256']
            or reg['settings']['streams'] != settings['streams']):
        raise ValueError('C2 and C/M must use the identical frozen state and chains')
    output = Path(settings['output'])
    if output.exists():
        raise FileExistsError('frozen readout output immutable; choose a new directory')
    output.mkdir(parents=True)
    paths = [Path(__file__), ROOT/'su2_nqs/benchmarks/schmidt_covariance.py',
             ROOT/'su2_nqs/benchmarks/chirality_covariance.py',
             ROOT/'su2_nqs/benchmarks/replica_spin_response.py',
             ROOT/'projects/clebsch_tree_nqs/workflows/spin_chirality_covariance.py']
    sources = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
    registration = dict(settings=settings, identity=identity, sources=sources,
                        config_file_sha256=file_sha256(config),
                        base_config_file_sha256=file_sha256(base),
                        runtime=dict(jax_version=jax.__version__, numpy_version=np.__version__,
                            x64_enabled=bool(jax.config.x64_enabled),
                            device_kinds=[d.device_kind for d in jax.devices()]),
                        new_Born_samples=0, new_H_calls=0, new_training=0)
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(output/'registration.json', registration)
    for source in paths:
        snapshot = output/'source_snapshot'/source.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        if file_sha256(snapshot) != sources[str(source.relative_to(ROOT))]:
            raise RuntimeError('source changed during readout registration')
    start = perf_counter(); queries = 0

    def query(bits: np.ndarray) -> np.ndarray:
        nonlocal queries
        if not len(bits):
            return np.empty(0, complex)
        unique, inverse = np.unique(bits, axis=0, return_inverse=True)
        size = settings['query_batch']; values = []
        for offset in range(0, len(unique), size):
            if perf_counter()-start > settings['wall_seconds']:
                raise TimeoutError('fixed readout cap; no partial physics claim')
            chunk = unique[offset:offset+size]
            padded = np.pad(chunk, ((0, size-len(chunk)), (0, 0)), mode='edge')
            values.append(np.asarray(batch_fn(params, jnp.asarray(padded)))[:len(chunk)])
            queries += len(chunk)
        return np.concatenate(values)[inverse]

    x, y = [s.reshape(-1, cfg.n_sites) for s in streams]
    old = np.column_stack((query(x), query(y)))
    grid = np.arange(cfg.n_sites).reshape(cfg.rows, cfg.columns)
    orbit = [(np.roll(grid, shift, axis=1).ravel(), flip)
             for shift in range(cfg.columns) for flip in (False, True)]
    # Reusing original denominators requires actual amplitude, not geometry,
    # invariance. The loaded parent already certifies A1 and spin-flip even.
    for gather, flip in orbit:
        bits = y[:4, gather]
        if flip: bits = 1-bits
        if np.max(abs(np.exp(query(bits)-old[:4, 1])-1)) > 1e-10:
            raise ValueError('relative symmetry does not preserve the physical amplitude')
    edges = [tuple(map(tuple, settings['chirality_channels']['prefix18'][f'row{i}_kappa_z']))
             for i in range(3)]
    sites = np.asarray(settings['regions']['prefix18'])
    if swap_probe_only:
        # A cheap runtime oracle BEFORE the expensive product actions. Keep
        # unaveraged log amplitudes even on failure, so device/batch placement
        # errors can be investigated without rerunning all operator queries.
        global_weights = np.zeros(len(x), complex)
        chunk_weights = np.zeros(len(x), complex)
        log_records = np.full((len(orbit), len(x), 2), -np.inf+0j)
        exp_identity_error = 0.
        for oi, (gather, flip) in enumerate(orbit):
            partner = y[:, gather]
            if flip: partner = 1-partner
            u, v = x.copy(), partner.copy()
            u[:, sites], v[:, sites] = partner[:, sites], x[:, sites]
            ids = np.flatnonzero(u.sum(-1) == cfg.target_n_up)
            logs = np.column_stack((query(u[ids]), query(v[ids])))
            log_records[oi, ids] = logs
            ratio_logs = logs-old[ids]
            weights = np.exp(ratio_logs.sum(axis=1))
            exp_identity_error = max(exp_identity_error, float(np.max(
                abs(weights-np.prod(np.exp(ratio_logs), axis=1)))))
            global_weights[ids] += weights/len(orbit)
            for offset in range(0, len(ids), settings['center_chunk']):
                select = ids[offset:offset+settings['center_chunk']]
                logs_chunk = query(np.concatenate((u[select], v[select]))).reshape(2, -1).T
                chunk_weights[select] += np.exp((logs_chunk-old[select]).sum(axis=1))/len(orbit)
        with np.load(reference/'records.npz') as z:
            archived = z['swap'].copy()
        difference = global_weights-archived
        worst = np.argsort(abs(difference))[-16:][::-1]
        probe = dict(registration_sha256=registration['sha256'],
            backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
            parameter_dtypes=sorted({str(a.dtype) for a in jax.tree_util.tree_leaves(params)}),
            global_layout_old_swap_max_error=float(np.max(abs(difference))),
            chunk_layout_old_swap_max_error=float(np.max(abs(chunk_weights-archived))),
            global_vs_chunk_max_error=float(np.max(abs(global_weights-chunk_weights))),
            exp_sum_vs_product_max_error=exp_identity_error,
            archived_max_absolute_weight=float(np.max(abs(archived))),
            original_guard=2e-11, guard_passed=bool(np.max(abs(difference)) <= 2e-11),
            worst_record_ids=worst.tolist(), wall_seconds=perf_counter()-start,
            projected_amplitude_queries=queries, physical_result=False)
        np.savez_compressed(output/'swap_probe.npz', original_logs=old,
            swapped_logs=log_records, global_weights=global_weights,
            chunk_weights=chunk_weights, archived_weights=archived)
        atomic_json(output/'swap_probe.json', probe)
        print(json.dumps(probe, indent=2), flush=True)
        return probe
    # Certify the entire archived SWAP BEFORE costly O_i O_j queries. A
    # different accelerator may not reproduce an ill-conditioned amplitude
    # closely enough; exact small-system math tests do not waive this guard.
    runtime_swap = np.zeros(len(x), complex)
    for gather, flip in orbit:
        partner = y[:, gather]
        if flip: partner = 1-partner
        u, v = x.copy(), partner.copy()
        u[:, sites], v[:, sites] = partner[:, sites], x[:, sites]
        ids = np.flatnonzero(u.sum(-1) == cfg.target_n_up)
        logs = np.column_stack((query(u[ids]), query(v[ids])))
        runtime_swap[ids] += np.exp((logs-old[ids]).sum(axis=1))/len(orbit)
    with np.load(reference/'records.npz') as z:
        runtime_error = float(np.max(abs(runtime_swap-z['swap'])))
    atomic_json(output/'runtime_replay.json', dict(max_error=runtime_error,
        tolerance=2e-11, passed=bool(runtime_error <= 2e-11),
        before_expensive_operator_actions=True, wall_seconds=perf_counter()-start))
    if runtime_error > 2e-11:
        np.savez_compressed(output/'failed_runtime_swap.npz', swap=runtime_swap,
                            original_logs=old)
        raise ValueError(f'pre-action old purity replay failed: {runtime_error}')
    numerators = np.zeros((len(x), 6), complex)
    swap = np.zeros(len(x), complex)
    legal_fractions = []
    for orbit_index, (gather, flip) in enumerate(orbit):
        partner = y[:, gather]
        if flip: partner = 1-partner
        u, v = x.copy(), partner.copy()
        u[:, sites], v[:, sites] = partner[:, sites], x[:, sites]
        legal = u.sum(-1) == cfg.target_n_up
        legal_fractions.append(float(legal.mean()))
        ids = np.flatnonzero(legal)
        for offset in range(0, len(ids), settings['center_chunk']):
            select = ids[offset:offset+settings['center_chunk']]
            bits = np.concatenate((u[select], v[select]))
            denominator = np.concatenate((old[select, 0], old[select, 1]))
            swapped = np.exp(query(bits)-denominator).reshape(2, -1).T
            swap[select] += np.prod(swapped, axis=1)/len(orbit)
            connections = [chirality_product_connections(bits, edges[i], edges[j]) for i, j in PAIRS]
            lengths = [len(t) for t, _, _ in connections]
            boundaries = np.cumsum([0]+lengths)
            logs = query(np.concatenate([t for t, _, _ in connections]))
            for k, (_, source, coefficients) in enumerate(connections):
                action = operator_action_over_sample(logs[boundaries[k]:boundaries[k+1]],
                    source, coefficients, denominator).reshape(2, -1).T
                numerators[select, k] += renyi2_product_numerator(action, swapped)/len(orbit)
        atomic_json(output/'progress.json', dict(completed_orbits=orbit_index+1,
            total_orbits=len(orbit), wall_seconds=perf_counter()-start,
            projected_amplitude_queries=queries))
        np.savez_compressed(output/'orbit_checkpoint.npz',
            completed_orbits=orbit_index+1, product_numerators=numerators,
            swap=swap, source_registration_sha256=registration['sha256'])
    labels = np.tile(np.arange(streams[0].shape[1]), streams[0].shape[0])
    # Preserve raw work before verification; a failed guard is NOT a result,
    # but must not erase the evidence needed to diagnose the failure.
    np.savez_compressed(output/'unverified_records.npz', product_numerators=numerators,
        swap=swap, chain_pair_labels=labels)
    with np.load(reference/'records.npz') as z:
        if not np.array_equal(labels, z['chain_pair_labels']):
            raise ValueError('independent chain pairing changed')
        replay = float(np.max(abs(swap-z['swap'])))
        if replay > 2e-11:
            atomic_json(output/'verification_failure.json', dict(
                old_swap_record_max_error=replay, tolerance=2e-11,
                physical_result=False, wall_seconds=perf_counter()-start))
            raise ValueError(f'old purity record replay failed: {replay}')
        m = z['replica_numerators'].copy()
        m[:, 3:] = (m[:, 3:]-m[:, [0, 0, 1]]-m[:, [1, 2, 2]])/2
        ordinary = z['ordinary_products']
    result = compare_renyi_matched_covariances(numerators, m, swap, labels)
    result.update(source_registration_sha256=registration['sha256'],
        energy_audit=identity['energy_audit'], wall_seconds=perf_counter()-start,
        projected_amplitude_queries=queries, old_swap_record_max_error=replay,
        legal_swap_fractions=legal_fractions, new_Born_samples=0, new_H_calls=0,
        new_training=0, operator_order=['row0', 'row1', 'row2'],
        operator_change=False, cut_change=False, equilibrium_certified=False)
    np.savez_compressed(output/'records.npz', product_numerators=numerators,
        inter_schmidt_numerators=m, swap=swap, chain_pair_labels=labels,
        ordinary_products=ordinary)
    atomic_json(output/'results.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_pipeline(plan_path: Path) -> dict:
    """Bind M, C and C2 serially to one preregistered parent and Born cache.

    Only upstream artifact hashes are filled at runtime; states, geometry,
    operators, chain pairing and per-stage budgets are fixed by the plan.
    No partial stage is reused or presented as a physical result.
    """
    from .spin_replica_response import run as replica_run
    from .spin_chirality_covariance import run as ordinary_run

    plan = json.loads(plan_path.read_text())
    base_path = ROOT/plan['base_config']
    if file_sha256(base_path) != plan['base_config_sha256']:
        raise ValueError('preregistered base configuration changed')
    base = json.loads(base_path.read_text())
    # The archived replica reader forms a two-column Fourier coordinate
    # matrix even when only its chirality probes will later be extracted.
    # Preserve its registered momentum controls rather than changing that
    # immutable reader or allowing an empty array to fail after queries.
    wavevectors = np.asarray(list(base.get('momenta', {}).values()))
    if wavevectors.ndim != 2 or wavevectors.shape[1] != 2 or not len(wavevectors):
        raise ValueError('archived replica reader requires nonempty two-component momenta')
    # Validate every sealed parent input before creating operational artifacts.
    _, _, _, _, identity = load_frozen_inputs(base)
    output = Path(plan['output'])
    if output.exists():
        raise FileExistsError('pipeline artifacts are immutable; choose an unused output')
    if set(plan['stage_wall_seconds']) != {'replica', 'ordinary', 'escort'}:
        raise ValueError('all three stage budgets must be fixed before queries')
    output.mkdir(parents=True)
    registration = dict(plan=plan, plan_file_sha256=file_sha256(plan_path),
        identity=identity, pipeline_source_sha256=file_sha256(Path(__file__)),
        new_Born_samples=0, new_H_calls=0, new_training=0,
        binding='same parent, complete streams, prefix18, six frozen row probes')
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(output/'registration.json', registration)
    start = perf_counter()
    replica = dict(base, output=str(output/'replica'),
                   wall_seconds=plan['stage_wall_seconds']['replica'])
    atomic_json(output/'replica_config.json', replica)
    replica_run(output/'replica_config.json')
    ordinary = dict(base, output=str(output/'ordinary'),
        replica_reference=str(output/'replica'),
        replica_registration_file_sha256=file_sha256(output/'replica/registration.json'),
        wall_seconds=plan['stage_wall_seconds']['ordinary'])
    atomic_json(output/'ordinary_config.json', ordinary)
    ordinary_run(output/'ordinary_config.json')
    escort = dict(base_config=str((output/'ordinary_config.json').resolve()),
        output=str(output/'escort'), ordinary_reference=str(output/'ordinary'),
        ordinary_registration_file_sha256=file_sha256(output/'ordinary/registration.json'),
        wall_seconds=plan['stage_wall_seconds']['escort'])
    atomic_json(output/'escort_config.json', escort)
    run(output/'escort_config.json')
    result = dict(registration_sha256=registration['sha256'],
        stages={name:dict(path=str(output/name),
            registration_file_sha256=file_sha256(output/name/'registration.json'),
            results_file_sha256=file_sha256(output/name/'results.json'))
            for name in ('replica','ordinary','escort')},
        wall_seconds=perf_counter()-start, new_Born_samples=0, new_H_calls=0, new_training=0)
    atomic_json(output/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--config', type=Path)
    choice.add_argument('--pipeline-config', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--swap-probe-only', action='store_true')
    args = parser.parse_args()
    if args.pipeline_config:
        if args.preflight_only or args.swap_probe_only:
            parser.error('pipeline does not reinterpret single-stage probe flags')
        result = run_pipeline(args.pipeline_config)
    else:
        result = run(args.config, preflight_only=args.preflight_only,
                     swap_probe_only=args.swap_probe_only)
    print(json.dumps(result, indent=2))
