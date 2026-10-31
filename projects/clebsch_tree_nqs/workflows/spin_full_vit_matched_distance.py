"""Frozen-parent damping rays at equal calibration-only physical distances.

No optimizer, Hamiltonian, sampler or ansatz is reimplemented here. All eight
candidates freeze before the new validation chains are initialized.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from .spin_full_vit_step_audit import register as direction_register, evaluate_chunks, parent_validation, probe_diagnostics
from .spin_full_vit_recheck import components
from .spin_full_vit_optimization import load_parent, seal
from .spin_full_vit_pilot import ROOT, _sealed_json, _file_sha256, audit_statistics
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.correlated_audit import correlated_summary, paired_chain_intervals
from su2_nqs.vmc.matched_state_distance import ratio_distance, match_ray_distance, paired_candidate_intervals
from su2_nqs.frontier.artifacts import canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest, numerical_runtime_identity

CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_full_vit_matched_distance.json'
OUTPUT = ROOT/'projects/clebsch_tree_nqs/results/spin_full_vit_matched_distance_v1'


def candidate_name(seed: int, arm: str, target: float) -> str:
    """Stable identifier for the fixed Cartesian candidate list."""
    return f'batch_{seed}_{arm}_fs{target:.3f}'


def ray_params(parent: object, direction_endpoint: object, scale: float) -> object:
    """Rescale a sealed single-step ray without changing its frozen origin."""
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('finite nonnegative ray scale required')
    return jax.tree_util.tree_map(lambda p, c: p+scale*(c-p), parent, direction_endpoint)


def register(output: Path) -> tuple[dict, dict, dict]:
    """Pin all old directions and new source/seed/budget choices before sampling."""
    spec = json.loads(CONFIG.read_text()); source = ROOT/spec['direction_directory']
    old_spec, prior, oldreg, _ = direction_register(source)
    if oldreg['result_sha256'] != spec['direction_registration_sha256']:
        raise ValueError('direction registration mismatch')
    for path, sha in oldreg['source_manifest'].items():
        if _file_sha256(ROOT/path) != sha:
            raise ValueError('sealed source changed')
    old_result = _sealed_json(source/'results.json')
    inputs = {}
    for seed in spec['training_seeds']:
        for arm in spec['directions']:
            folder = source/f'batch_{seed}'/arm
            r = _sealed_json(folder/'results.json')
            if (r['registration_sha256'] != oldreg['result_sha256']
                    or r['checkpoint_sha256'] != _file_sha256(folder/'candidate.pkl')
                    or r['parent_sha256'] != old_result['parent_sha256']):
                raise ValueError('old direction changed')
            inputs[str((folder/'candidate.pkl').relative_to(ROOT))] = r['checkpoint_sha256']
    inputs[str((source/'probes.npz').relative_to(ROOT))] = _file_sha256(source/'probes.npz')
    if inputs[str((source/'probes.npz').relative_to(ROOT))] != old_result['probe_sha256']:
        raise ValueError('probe input changed')
    fresh = [s['seed'] for key in ('calibration', 'validation') for s in spec[key]['streams']]
    prior_seeds = old_spec['training_seeds']+[s['seed'] for s in old_spec['validation']['streams']]
    if len(set(fresh)) != len(fresh) or set(fresh) & set(prior_seeds):
        raise ValueError('sampling seed reuse')
    output.mkdir(parents=True, exist_ok=True)
    data = dict(config=spec, config_sha256=canonical_sha256(spec), input_files=inputs,
                direction_result_sha256=old_result['result_sha256'],
                source_manifest=transitive_local_source_manifest((Path(__file__),), root=ROOT,
                    package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')),
                runtime=numerical_runtime_identity())
    path = output/'registration.json'
    if path.exists():
        reg = _sealed_json(path)
        if any(reg[k] != v for k, v in data.items()):
            raise ValueError('matched-distance protocol changed')
    else:
        reg = seal(path, {**data, 'registered_at_utc': datetime.now(timezone.utc).isoformat()})
    return spec, prior, reg


def run(output: Path = OUTPUT) -> dict:
    """Run a fixed calibration and independent common-parent audit, without adoption."""
    started = perf_counter(); spec, prior, reg = register(output)
    if (output/'results.json').exists():
        raise FileExistsError('completed experiment is immutable')
    source = ROOT/spec['direction_directory']
    saved_parent, ref = load_parent(prior, 'square'); p0 = saved_parent['params']['model']
    ph = pytree_sha256(p0); logs, local, group, _ = components(ref)
    batch_logs = jax.jit(jax.vmap(logs, in_axes=(None, 0))); chunk = spec['evaluation_chunk_size']
    cal_folder = output/'calibration'; cal_folder.mkdir(exist_ok=True)
    cal_x, cal_e, cal_pools = parent_validation(p0, logs, local, {'validation': spec['calibration']}, reg, cal_folder)
    flat_cal = cal_x.reshape(-1, 36)
    cal_logs = evaluate_chunks(batch_logs, p0, flat_cal, chunk)
    frozen = []
    for seed in spec['training_seeds']:
        for arm in spec['directions']:
            endpoint = load_checkpoint(source/f'batch_{seed}'/arm/'candidate.pkl')['params']['model']
            for target in spec['target_fs_sines']:
                name = candidate_name(seed, arm, target); folder = output/name; folder.mkdir(exist_ok=True)
                path = folder/'matched.json'; cp = folder/'candidate.pkl'
                if path.exists():
                    match = _sealed_json(path)
                    if (match['registration_sha256'] != reg['result_sha256']
                            or match['checkpoint_sha256'] != _file_sha256(cp)):
                        raise ValueError('matched candidate changed')
                else:
                    tick = perf_counter()
                    def calibration_ratios(scale: float) -> np.ndarray:
                        return evaluate_chunks(batch_logs, ray_params(p0, endpoint, scale), flat_cal, chunk)-cal_logs
                    fit = match_ray_distance(calibration_ratios, target=target, **spec['matching'])
                    if not fit['matched']:
                        seal(folder/'matching_failure.json', dict(registration_sha256=reg['result_sha256'], fit=fit))
                        raise RuntimeError('bounded radius calibration failed; do not extend budget')
                    p = ray_params(p0, endpoint, fit['scale'])
                    save_checkpoint(cp, dict(model=p), metadata=dict(registration_sha256=reg['result_sha256'],
                        parent_sha256=ph, seed=seed, direction=arm, target=target, ray_scale=fit['scale'], diagnostic_only=True))
                    if pytree_sha256(load_checkpoint(cp)['params']['model']) != pytree_sha256(p):
                        raise ValueError('checkpoint replay failed')
                    match = seal(path, dict(registration_sha256=reg['result_sha256'], name=name,
                        seed=seed, direction=arm, target=target, fit=fit, parent_sha256=ph,
                        candidate_sha256=pytree_sha256(p), checkpoint_sha256=_file_sha256(cp),
                        seconds=perf_counter()-tick))
                print('MATCH', name, match['fit']['distance'], 'scale', match['fit']['scale'], flush=True)
                frozen.append(match)
    frozen_data = dict(registration_sha256=reg['result_sha256'], candidates=frozen,
                       calibration_pool_hashes=[r['result_sha256'] for r in cal_pools])
    frozen_path = output/'frozen_candidates.json'
    if frozen_path.exists():
        freeze = _sealed_json(frozen_path)
        if any(freeze[k] != v for k, v in frozen_data.items()):
            raise ValueError('frozen list changed')
    else:
        # All eight candidates are immutable before accessing any validation samples.
        freeze = seal(frozen_path, {**frozen_data, 'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    val_folder = output/'validation'; val_folder.mkdir(exist_ok=True)
    val_x, val_e, pools = parent_validation(p0, logs, local, spec, reg, val_folder)
    flat_val = val_x.reshape(-1, 36)
    old_logs = evaluate_chunks(batch_logs, p0, flat_val, chunk).reshape(val_e.shape)
    with np.load(source/'probes.npz') as d:
        probes, central, probe_logs = d['states'], d['central_indices'], d['parent_logs']
    records = []
    for match in frozen:
        folder = output/match['name']; path = folder/'results.json'; raw_path = folder/'audit.npz'
        p = load_checkpoint(folder/'candidate.pkl')['params']['model']
        if path.exists():
            r = _sealed_json(path)
            if (r['registration_sha256'] != reg['result_sha256']
                    or r['raw_sha256'] != _file_sha256(raw_path)
                    or r['frozen_list_sha256'] != freeze['result_sha256']):
                raise ValueError('candidate audit changed')
        else:
            tick = perf_counter()
            lr = evaluate_chunks(batch_logs, p, flat_val, chunk).reshape(val_e.shape)-old_logs
            e = evaluate_chunks(local, p, flat_val, chunk).reshape(val_e.shape)
            plr = evaluate_chunks(batch_logs, p, probes, chunk)-probe_logs
            base = load_checkpoint(source/f"batch_{match['seed']}"/'initial.pkl')
            train = np.asarray(base['params']['chains'])
            tr = evaluate_chunks(batch_logs, p, train, chunk)-evaluate_chunks(batch_logs, p0, train, chunk)
            before = batch_logs(p, jnp.asarray(train[:4]))
            a1 = max(float(jnp.max(abs(jnp.exp(batch_logs(p, jnp.asarray(train[:4, g]))-before)-1)))
                     for g in group[::max(1, len(group)//8)])
            val = correlated_summary(lr, val_e, e, n_sites=36)
            np.savez_compressed(raw_path, log_ratios=lr, candidate_energies=e, probe_log_ratios=plr,
                                training_log_ratios=tr)
            r = seal(path, dict(registration_sha256=reg['result_sha256'], match=match,
                frozen_list_sha256=freeze['result_sha256'], audit=val,
                audit_by_stream=[correlated_summary(lr[s], val_e[s], e[s], n_sites=36) for s in range(len(pools))],
                uncertainty=paired_chain_intervals(lr, val_e, e, n_sites=36, **spec['bootstrap']),
                training_fs_sine=ratio_distance(tr),
                probes=probe_diagnostics(plr, central, probes, train), sampled_A1_error=a1,
                raw_sha256=_file_sha256(raw_path), full_H_calls=e.size, seconds=perf_counter()-tick,
                adopted=False))
        print('AUDIT', match['name'], 'FS', r['audit']['sampled_fs_sine'],
              'dE', r['audit']['delta_energy_per_site'], 'dVar', r['audit']['delta_variance_per_site'], flush=True)
        records.append(r)
    comparisons = []
    for seed in spec['training_seeds']:
        for target in spec['target_fs_sines']:
            pair = [next(r for r in records if r['match']['name'] == candidate_name(seed, arm, target))
                    for arm in spec['directions']]
            arrays = []
            for r in pair:
                with np.load(output/r['match']['name']/'audit.npz') as d:
                    arrays.extend((d['log_ratios'], d['candidate_energies']))
            fs = [r['audit']['sampled_fs_sine'] for r in pair]
            mismatch = abs(fs[1]-fs[0])/np.mean(fs)
            comparisons.append(dict(seed=seed, target=target, sign='damping_1_minus_damping_0.1',
                audit_relative_distance_mismatch=float(mismatch),
                independent_matching_warning=bool(mismatch > spec['audit_distance_relative_tolerance']),
                uncertainty=paired_candidate_intervals(*arrays, n_sites=36, **spec['bootstrap'])))
    parent_after, _ = load_parent(prior, 'square')
    if pytree_sha256(parent_after['params']['model']) != ph:
        raise ValueError('frozen parent changed')
    result = seal(output/'results.json', dict(registration_sha256=reg['result_sha256'], parent_sha256=ph,
        frozen_list_sha256=freeze['result_sha256'], calibration_pools=cal_pools, validation_pools=pools,
        calibration_parent_statistics=audit_statistics(cal_e, 36, bootstrap_seed=spec['bootstrap']['seed'], replicates=4000),
        audit_parent_statistics=audit_statistics(val_e, 36, bootstrap_seed=spec['bootstrap']['seed'], replicates=4000),
        candidates=records, comparisons=comparisons, full_H_calls=cal_e.size+val_e.size+sum(r['full_H_calls'] for r in records),
        wall_seconds=perf_counter()-started, adopted_candidates=0, candidate_equilibrium_claimed=False,
        remote_submitted=False, exact_SU2=False))
    print('COMPLETE', result['full_H_calls'], result['wall_seconds'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); run(args.output)
